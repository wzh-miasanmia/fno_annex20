"""Train pure Geo-FNO directly on STAR-CCM+ CSV nodes.

Unlike ``train_concentration_fno_cnn_extended_s.py``, this pipeline never calls
``scipy.interpolate.griddata``.  The CSV coordinates, velocity, and passive
scalar values stay on their native nodes throughout loading, training, and
evaluation.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.geofno import GeoFNO2D  # noqa: E402
from model.complex_adamw import ComplexAdamW  # noqa: E402
from model.train_fixed_u_srcpos_to_c import (  # noqa: E402
    TargetTransform,
    build_source_mask_gaussian,
    build_source_mask_rect,
    build_source_mask_soft_rect_sigmoid,
    safe_float_tag,
    set_seed,
    split_indices,
    split_indices_from_csv,
)
from model.train_concentration_fno_cnn_extended_s import case_id_to_str, load_case_csv  # noqa: E402


@dataclass
class PointNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / (self.std + 1e-12)

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return value * (self.std + 1e-12) + self.mean


class RawPointCloudDataset(Dataset):
    """One CFD export per sample, with no grid conversion."""

    def __init__(
        self,
        workdir: Path,
        exports_dirname: str = "exports",
        cases_csv: str = "cases.csv",
        Lx: float = 9.0,
        Ly: float = 3.0,
        mask_kind: str = "soft_rect",
        src_size: float = 0.20,
        sigma: float = 0.12,
        soft_k: float = 80.0,
        use_velocity_input: bool = True,
        use_inlet_scalar: bool = True,
        fixed_inlet_u: Optional[float] = None,
        inlet_u_tol: float = 1e-8,
        limit_samples: Optional[int] = None,
        sample_seed: int = 42,
        cache_in_memory: bool = False,
        quadrature_mode: str = "auto",
        quadrature_column: Optional[str] = None,
        quadrature_cache_dir: Optional[Path] = None,
        density_neighbors: int = 8,
        joint_speed_target: bool = False,
        joint_velocity_target: bool = False,
        speed_target_mode: str = "inlet_normalized",
    ):
        self.workdir = Path(workdir)
        self.exports = self.workdir / exports_dirname
        self.cases_path = self.workdir / cases_csv
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.src_size = float(src_size)
        self.sigma = float(sigma)
        self.soft_k = float(soft_k)
        self.use_velocity_input = bool(use_velocity_input)
        self.use_inlet_scalar = bool(use_inlet_scalar)
        self.cache_in_memory = bool(cache_in_memory)
        self.quadrature_mode = str(quadrature_mode).lower()
        if self.quadrature_mode not in {"auto", "column", "knn"}:
            raise ValueError("quadrature_mode must be auto, column, or knn.")
        self.quadrature_column = quadrature_column
        self.quadrature_cache_dir = Path(quadrature_cache_dir) if quadrature_cache_dir else None
        if self.quadrature_cache_dir is not None:
            self.quadrature_cache_dir.mkdir(parents=True, exist_ok=True)
        self.density_neighbors = max(2, int(density_neighbors))
        self.joint_speed_target = bool(joint_speed_target)
        self.joint_velocity_target = bool(joint_velocity_target)
        if self.joint_speed_target and self.joint_velocity_target:
            raise ValueError("joint_speed_target and joint_velocity_target are mutually exclusive.")
        self.speed_target_mode = str(speed_target_mode)
        if self.speed_target_mode not in {"raw", "inlet_normalized"}:
            raise ValueError("speed_target_mode must be raw or inlet_normalized.")
        self.quadrature_source = "not_loaded"
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

        kind = str(mask_kind).lower().strip().replace("-", "_")
        if kind == "softrect":
            kind = "soft_rect"
        if kind not in {"rect", "gauss", "soft_rect"}:
            raise ValueError("mask_kind must be rect, gauss, or soft_rect.")
        self.mask_kind = kind
        if not self.exports.exists():
            raise FileNotFoundError(f"exports folder not found: {self.exports}")
        if not self.cases_path.exists():
            raise FileNotFoundError(f"cases.csv not found: {self.cases_path}")

        rows = pd.read_csv(self.cases_path)
        required = ["case_id", "inlet_u", "src_x", "src_y"]
        missing = [name for name in required if name not in rows.columns]
        if missing:
            raise ValueError(f"cases.csv missing columns {missing}.")
        rows = rows[required].copy()
        if fixed_inlet_u is not None:
            rows = rows[np.abs(rows["inlet_u"].astype(float) - float(fixed_inlet_u)) <= inlet_u_tol]
        if limit_samples is not None and int(limit_samples) < len(rows):
            rows = rows.sample(n=int(limit_samples), random_state=int(sample_seed))

        self.files: List[Path] = []
        kept = []
        missing_files: List[Path] = []
        seen_files: Dict[Path, int] = {}
        for _, row in rows.reset_index(drop=True).iterrows():
            case_id = case_id_to_str(row["case_id"])
            exact = self.exports / (
                f"case_{case_id}_u{safe_float_tag(float(row['inlet_u']))}"
                f"_sx{safe_float_tag(float(row['src_x']))}_sy{safe_float_tag(float(row['src_y']))}.csv"
            )
            if not exact.is_file():
                missing_files.append(exact)
                continue
            if exact in seen_files:
                first_row = seen_files[exact]
                raise ValueError(
                    f"Duplicate cases.csv rows resolve to the same export {exact} "
                    f"(first row {first_row})."
                )
            seen_files[exact] = len(kept)
            self.files.append(exact)
            kept.append(row)
        if missing_files:
            examples = "\n".join(f"  - {path}" for path in missing_files[:5])
            suffix = "" if len(missing_files) <= 5 else f"\n  ... and {len(missing_files) - 5} more"
            raise FileNotFoundError(
                f"{len(missing_files)} cases.csv rows have no exact export match. "
                "Fallback matching is disabled to prevent loading the wrong inlet/source case:\n"
                f"{examples}{suffix}"
            )
        self.rows = pd.DataFrame(kept).reset_index(drop=True)
        self.rows["case_id"] = self.rows["case_id"].map(case_id_to_str)
        if self.rows.empty:
            raise RuntimeError("No valid exported CSV files matched cases.csv.")

    @property
    def in_channels(self) -> int:
        return 5 + (2 if self.use_velocity_input else 0) + (1 if self.use_inlet_scalar else 0)

    @property
    def input_channels(self) -> List[str]:
        names = ["source_mask", "X", "Y", "src_x", "src_y"]
        if self.use_velocity_input:
            names.extend(["Ux", "Uy"])
        if self.use_inlet_scalar:
            names.append("inlet_u")
        return names

    def __len__(self) -> int:
        return len(self.rows)

    def _source_mask(self, x: np.ndarray, y: np.ndarray, sx: float, sy: float) -> np.ndarray:
        if self.mask_kind == "rect":
            return build_source_mask_rect(x, y, sx, sy, self.src_size)
        if self.mask_kind == "gauss":
            return build_source_mask_gaussian(x, y, sx, sy, self.sigma)
        return build_source_mask_soft_rect_sigmoid(x, y, sx, sy, self.src_size, self.soft_k)

    @staticmethod
    def _normalize_quadrature(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("Quadrature values must be a finite positive 1D array.")
        return (values / max(values.mean(), 1e-12)).astype(np.float32)

    def _area_column(self, frame: pd.DataFrame) -> Optional[str]:
        if self.quadrature_column:
            if self.quadrature_column not in frame.columns:
                if self.quadrature_mode == "column":
                    raise ValueError(
                        f"Requested quadrature column {self.quadrature_column!r} is absent. "
                        f"Available columns: {list(frame.columns)}"
                    )
                return None
            return self.quadrature_column
        normalized = {
            "".join(character for character in name.lower() if character.isalnum()): name
            for name in frame.columns
        }
        for candidate in (
            "cellarea",
            "cellaream2",
            "cellvolume",
            "cellvolumem3",
            "volume",
        ):
            if candidate in normalized:
                return normalized[candidate]
        return None

    def _knn_quadrature(self, coords: np.ndarray, index: int) -> np.ndarray:
        cache_path = None
        if self.quadrature_cache_dir is not None:
            cache_path = self.quadrature_cache_dir / (
                f"{self.files[index].stem}.knn{self.density_neighbors}.npy"
            )
            if cache_path.is_file():
                cached = np.load(cache_path)
                if cached.shape == (coords.shape[0],):
                    return self._normalize_quadrature(cached)

        count = coords.shape[0]
        if count < 3:
            weights = np.ones(count, dtype=np.float32)
        else:
            neighbors = min(self.density_neighbors + 1, count)
            distances, _ = cKDTree(coords.astype(np.float64)).query(coords, k=neighbors, workers=1)
            radius = distances[:, -1]
            positive = radius[radius > 0.0]
            floor = np.quantile(positive, 0.01) if positive.size else 1.0
            radius = np.maximum(radius, floor)
            # 1/rho is proportional to the disk area enclosing k neighbours.
            weights = self._normalize_quadrature(radius**2)

        if cache_path is not None:
            temporary = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp.npy")
            np.save(temporary, weights)
            os.replace(temporary, cache_path)
        return weights

    def quadrature_for_frame(
        self, frame: pd.DataFrame, coords: np.ndarray, index: int
    ) -> np.ndarray:
        column = None if self.quadrature_mode == "knn" else self._area_column(frame)
        if column is not None:
            self.quadrature_source = f"csv:{column}"
            return self._normalize_quadrature(frame[column].to_numpy(dtype=np.float64))
        if self.quadrature_mode == "column":
            raise ValueError("No CFD cell-area/cell-volume column was found for quadrature_mode=column.")
        self.quadrature_source = f"knn:k={self.density_neighbors}"
        return self._knn_quadrature(coords, index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        index = int(index)
        if self.cache_in_memory and index in self._cache:
            return self._cache[index]
        row = self.rows.iloc[index]
        inlet_u = float(row["inlet_u"])
        sx, sy = float(row["src_x"]), float(row["src_y"])
        frame = load_case_csv(self.files[index])
        x_phys = frame["X (m)"].to_numpy(dtype=np.float32)
        y_phys = frame["Y (m)"].to_numpy(dtype=np.float32)
        coords = np.stack([x_phys / self.Lx, y_phys / self.Ly], axis=-1).astype(np.float32)
        coords = np.clip(coords, 0.0, 1.0)
        quadrature = self.quadrature_for_frame(frame, coords, index)
        channels: List[np.ndarray] = [
            self._source_mask(x_phys, y_phys, sx, sy),
            coords[:, 0],
            coords[:, 1],
            np.full(coords.shape[0], sx / self.Lx, dtype=np.float32),
            np.full(coords.shape[0], sy / self.Ly, dtype=np.float32),
        ]
        if self.use_velocity_input:
            channels.extend(
                [
                    frame["Velocity[i] (m/s)"].to_numpy(dtype=np.float32),
                    frame["Velocity[j] (m/s)"].to_numpy(dtype=np.float32),
                ]
            )
        if self.use_inlet_scalar:
            channels.append(np.full(coords.shape[0], inlet_u, dtype=np.float32))
        concentration = frame["Passive Scalar"].to_numpy(dtype=np.float32)[:, None]
        if self.joint_velocity_target:
            ux = frame["Velocity[i] (m/s)"].to_numpy(dtype=np.float32)[:, None]
            uy = frame["Velocity[j] (m/s)"].to_numpy(dtype=np.float32)[:, None]
            if self.speed_target_mode == "inlet_normalized":
                scale = max(abs(inlet_u), 1e-8)
                ux, uy = ux / scale, uy / scale
            target = np.concatenate((ux, uy, concentration), axis=-1)
        elif self.joint_speed_target:
            ux = frame["Velocity[i] (m/s)"].to_numpy(dtype=np.float32)
            uy = frame["Velocity[j] (m/s)"].to_numpy(dtype=np.float32)
            speed = np.sqrt(ux * ux + uy * uy).astype(np.float32)[:, None]
            if self.speed_target_mode == "inlet_normalized":
                speed = speed / max(abs(inlet_u), 1e-8)
            target = np.concatenate((speed, concentration), axis=-1)
        else:
            target = concentration
        sample = {
            "features": torch.from_numpy(np.stack(channels, axis=-1)),
            "coords": torch.from_numpy(coords),
            "target": torch.from_numpy(target),
            "code": torch.tensor([sx / self.Lx, sy / self.Ly, inlet_u], dtype=torch.float32),
            "quadrature": torch.from_numpy(quadrature),
            "case_index": torch.tensor(index, dtype=torch.long),
        }
        if self.cache_in_memory:
            self._cache[index] = sample
        return sample


def pad_point_cloud_batch(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch = len(samples)
    max_points = max(sample["coords"].shape[0] for sample in samples)
    channels = samples[0]["features"].shape[1]
    target_channels = samples[0]["target"].shape[1]
    result = {
        "features": torch.zeros(batch, max_points, channels, dtype=torch.float32),
        "coords": torch.zeros(batch, max_points, 2, dtype=torch.float32),
        "target": torch.zeros(batch, max_points, target_channels, dtype=torch.float32),
        "valid": torch.zeros(batch, max_points, dtype=torch.bool),
        "code": torch.stack([sample["code"] for sample in samples]),
        "quadrature": torch.zeros(batch, max_points, dtype=torch.float32),
        "case_index": torch.stack([sample["case_index"] for sample in samples]),
    }
    for batch_index, sample in enumerate(samples):
        n_points = sample["coords"].shape[0]
        for name in ("features", "coords", "target"):
            result[name][batch_index, :n_points] = sample[name]
        result["quadrature"][batch_index, :n_points] = sample["quadrature"]
        result["valid"][batch_index, :n_points] = True
    return result


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def make_normalizers(
    loader: DataLoader, device: torch.device, target_transform: TargetTransform
) -> Tuple[PointNormalizer, PointNormalizer]:
    x_sum = x_square = y_sum = y_square = None
    count = 0
    for batch in loader:
        batch = _to_device(batch, device)
        valid = batch["valid"].unsqueeze(-1)
        features = batch["features"]
        target = target_transform.forward(batch["target"])
        x_batch_sum = (features * valid).sum(dim=(0, 1), keepdim=True)
        x_batch_square = (features.square() * valid).sum(dim=(0, 1), keepdim=True)
        y_batch_sum = (target * valid).sum(dim=(0, 1), keepdim=True)
        y_batch_square = (target.square() * valid).sum(dim=(0, 1), keepdim=True)
        x_sum = x_batch_sum if x_sum is None else x_sum + x_batch_sum
        x_square = x_batch_square if x_square is None else x_square + x_batch_square
        y_sum = y_batch_sum if y_sum is None else y_sum + y_batch_sum
        y_square = y_batch_square if y_square is None else y_square + y_batch_square
        count += int(valid.sum())
    x_mean, y_mean = x_sum / count, y_sum / count
    x_std = torch.sqrt(torch.clamp(x_square / count - x_mean.square(), min=1e-12))
    y_std = torch.sqrt(torch.clamp(y_square / count - y_mean.square(), min=1e-12))
    return PointNormalizer(x_mean, x_std), PointNormalizer(y_mean, y_std)


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.unsqueeze(-1).to(dtype=value.dtype)
    return (value * mask).sum() / torch.clamp(mask.sum() * value.shape[-1], min=1.0)


def concentration_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_raw: torch.Tensor,
    valid: torch.Tensor,
    alpha: float,
    gamma: float,
    cap: float,
) -> torch.Tensor:
    positive = torch.clamp(target_raw.detach(), min=0.0)
    scale = torch.clamp(positive.amax(dim=1, keepdim=True), min=1e-12)
    weights = torch.clamp(1.0 + alpha * torch.pow(positive / scale, gamma), max=cap)
    weights = weights * valid.unsqueeze(-1).to(dtype=weights.dtype)
    return (weights * (prediction - target).square()).sum() / torch.clamp(weights.sum(), min=1.0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    x_norm: PointNormalizer,
    y_norm: PointNormalizer,
    transform: TargetTransform,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    squared_error = absolute_error = target_square = 0.0
    jacobian_sum = jacobian_nonpositive = 0.0
    jacobian_min = float("inf")
    jacobian_count = 0
    count = 0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        features = x_norm.encode(batch["features"])
        prediction_n, aux = model(
            features,
            batch["coords"],
            batch["valid"],
            code=batch["code"][..., :2],
            quadrature=batch["quadrature"],
            return_aux=True,
        )
        prediction = transform.inverse(y_norm.decode(prediction_n))
        mask = batch["valid"].unsqueeze(-1)
        error = (prediction - batch["target"])[mask]
        target = batch["target"][mask]
        squared_error += float(error.square().sum())
        absolute_error += float(error.abs().sum())
        target_square += float(target.square().sum())
        count += int(error.numel())
        determinant = aux["jacobian_det"][batch["valid"]]
        jacobian_sum += float(determinant.sum())
        jacobian_nonpositive += float((determinant <= 0.0).sum())
        jacobian_min = min(jacobian_min, float(determinant.min()))
        jacobian_count += int(determinant.numel())
    return {
        "mse": squared_error / max(1, count),
        "mae": absolute_error / max(1, count),
        "relative_l2": math.sqrt(squared_error / max(target_square, 1e-12)),
        "jacobian_min": jacobian_min,
        "jacobian_mean": jacobian_sum / max(1, jacobian_count),
        "jacobian_nonpositive_fraction": jacobian_nonpositive / max(1, jacobian_count),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    x_norm: PointNormalizer,
    y_norm: PointNormalizer,
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "x_mean": x_norm.mean.detach().cpu(),
            "x_std": x_norm.std.detach().cpu(),
            "y_mean": y_norm.mean.detach().cpu(),
            "y_std": y_norm.std.detach().cpu(),
            "meta": meta,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train pure Geo-FNO directly on unstructured CFD nodes (no interpolation)."
    )
    parser.add_argument("--workdir", default=str(PROJECT_ROOT / "data" / "data_v_S_new"))
    parser.add_argument("--exports", default="exports")
    parser.add_argument("--cases", default="cases.csv")
    parser.add_argument(
        "--save",
        default=str(PROJECT_ROOT / "checkpoints" / "geofno" / "geofno_raw_nodes.pt"),
    )
    parser.add_argument("--Lx", type=float, default=9.0)
    parser.add_argument("--Ly", type=float, default=3.0)
    parser.add_argument("--mask_kind", choices=["rect", "gauss", "soft_rect"], default="soft_rect")
    parser.add_argument("--src_size", type=float, default=0.20)
    parser.add_argument("--sigma", type=float, default=0.12)
    parser.add_argument("--soft_k", type=float, default=80.0)
    parser.add_argument("--no_velocity_input", dest="use_velocity_input", action="store_false", default=True)
    parser.add_argument("--no_inlet_scalar", dest="use_inlet_scalar", action="store_false", default=True)
    parser.add_argument("--fixed_inlet_u", type=float, default=None)
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--quadrature_mode", choices=["auto", "column", "knn"], default="auto")
    parser.add_argument("--quadrature_column", default=None)
    parser.add_argument("--quadrature_cache_dir", default=None)
    parser.add_argument("--density_neighbors", type=int, default=8)

    parser.add_argument("--split_mode", choices=["random", "spatial_checkerboard", "spatial_uniform", "spatial_left_right"], default="random")
    parser.add_argument("--split_csv", default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    parser.add_argument("--latent_ny", type=int, default=48)
    parser.add_argument("--latent_nx", type=int, default=144)
    parser.add_argument("--modes1", type=int, default=12)
    parser.add_argument("--modes2", type=int, default=12)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--deformation_width", type=int, default=32)
    parser.add_argument("--max_displacement", type=float, default=0.15)
    parser.add_argument("--point_chunk_size", type=int, default=2048)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--iphi_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--field_weight", type=float, default=1.0)
    parser.add_argument("--nonnegative_weight", type=float, default=0.001)
    parser.add_argument("--jacobian_weight", type=float, default=0.01)
    parser.add_argument("--jacobian_floor", type=float, default=0.05)
    parser.add_argument("--jacobian_epsilon", type=float, default=1e-3)
    parser.add_argument("--concentration_weight_alpha", type=float, default=2.0)
    parser.add_argument("--concentration_weight_gamma", type=float, default=0.5)
    parser.add_argument("--concentration_weight_cap", type=float, default=5.0)
    parser.add_argument("--target_transform", choices=["raw", "log10"], default="raw")
    parser.add_argument("--target_transform_eps", type=float, default=1e-8)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.amp:
        print(
            "[WARN] AMP disabled: PyTorch GradScaler does not support the "
            "ComplexFloat Fourier parameters used by Geo-FNO."
        )
        args.amp = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    dataset = RawPointCloudDataset(
        workdir=Path(args.workdir),
        exports_dirname=args.exports,
        cases_csv=args.cases,
        Lx=args.Lx,
        Ly=args.Ly,
        mask_kind=args.mask_kind,
        src_size=args.src_size,
        sigma=args.sigma,
        soft_k=args.soft_k,
        use_velocity_input=args.use_velocity_input,
        use_inlet_scalar=args.use_inlet_scalar,
        fixed_inlet_u=args.fixed_inlet_u,
        limit_samples=args.limit_samples,
        sample_seed=args.sample_seed,
        cache_in_memory=args.cache,
        quadrature_mode=args.quadrature_mode,
        quadrature_column=args.quadrature_column,
        quadrature_cache_dir=(
            Path(args.quadrature_cache_dir)
            if args.quadrature_cache_dir
            else Path(args.save).parent / "quadrature_cache"
        ),
        density_neighbors=args.density_neighbors,
    )
    if args.split_csv:
        train_indices, val_indices, test_indices = split_indices_from_csv(dataset.rows, args.split_csv)
    else:
        train_indices, val_indices, test_indices = split_indices(
            dataset.rows,
            split_mode=args.split_mode,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
        )
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    split_labels = np.full(len(dataset), "", dtype=object)
    split_labels[np.asarray(train_indices, dtype=int)] = "train"
    split_labels[np.asarray(val_indices, dtype=int)] = "val"
    split_labels[np.asarray(test_indices, dtype=int)] = "test"
    split_table = dataset.rows.copy()
    split_table["split"] = split_labels
    split_path = save_path.parent / "split_assignments_geofno.csv"
    test_path = save_path.parent / "test_set_list_geofno.csv"
    split_table.to_csv(split_path, index=False)
    split_table.iloc[test_indices].to_csv(test_path, index=False)
    loader_options = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": pad_point_cloud_batch,
    }
    train_loader = DataLoader(Subset(dataset, train_indices), shuffle=True, **loader_options)
    train_stats_loader = DataLoader(Subset(dataset, train_indices), shuffle=False, **loader_options)
    val_loader = DataLoader(Subset(dataset, val_indices), shuffle=False, **loader_options)
    test_loader = DataLoader(Subset(dataset, test_indices), shuffle=False, **loader_options)

    transform = TargetTransform(args.target_transform, args.target_transform_eps)
    x_norm, y_norm = make_normalizers(train_stats_loader, device, transform)
    model = GeoFNO2D(
        in_channels=dataset.in_channels,
        width=args.width,
        modes1=args.modes1,
        modes2=args.modes2,
        depth=args.depth,
        latent_size=(args.latent_ny, args.latent_nx),
        code_dim=2,
        deformation_width=args.deformation_width,
        max_displacement=args.max_displacement,
        point_chunk_size=args.point_chunk_size,
    ).to(device)
    deformation_parameters = list(model.geofno.deformation.parameters())
    deformation_ids = {id(parameter) for parameter in deformation_parameters}
    operator_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in deformation_ids
    ]
    optimizer = ComplexAdamW(
        [
            {"params": operator_parameters, "lr": args.lr},
            {"params": deformation_parameters, "lr": args.iphi_lr},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_val, best_epoch, stale = float("inf"), -1, 0
    history: List[dict] = []
    history_path = save_path.parent / "history_geofno.csv"

    print(
        f"[INFO] device={device} cases={len(dataset)} train/val/test="
        f"{len(train_indices)}/{len(val_indices)}/{len(test_indices)} in_ch={dataset.in_channels}"
    )
    print(f"[INFO] input_channels={dataset.input_channels}")
    first_sample = dataset[0]
    print(
        f"[INFO] quadrature={dataset.quadrature_source} "
        f"range=({float(first_sample['quadrature'].min()):.3e},"
        f"{float(first_sample['quadrature'].max()):.3e}); "
        "IPHI code=[src_x/Lx,src_y/Ly]"
    )
    print(
        f"[INFO] pure Geo-FNO: raw nodes -> latent={args.latent_ny}x{args.latent_nx} "
        f"modes=({args.modes1},{args.modes2}) -> raw nodes; interpolation=disabled"
    )
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()
        train_total = 0.0
        train_batches = 0
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            features = x_norm.encode(batch["features"])
            target_raw = batch["target"]
            target_n = y_norm.encode(transform.forward(target_raw))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                prediction_n, aux = model(
                    features,
                    batch["coords"],
                    batch["valid"],
                    code=batch["code"][..., :2],
                    quadrature=batch["quadrature"],
                    jacobian_epsilon=args.jacobian_epsilon,
                    return_aux=True,
                )
                field_loss = concentration_weighted_mse(
                    prediction_n,
                    target_n,
                    target_raw,
                    batch["valid"],
                    args.concentration_weight_alpha,
                    args.concentration_weight_gamma,
                    args.concentration_weight_cap,
                )
                prediction_raw = transform.inverse(y_norm.decode(prediction_n))
                nonnegative = masked_mean(torch.relu(-prediction_raw).square(), batch["valid"])
                jacobian_penalty = (
                    torch.relu(args.jacobian_floor - aux["jacobian_det"]).square()
                    * batch["valid"].to(dtype=prediction_n.dtype)
                ).sum() / torch.clamp(batch["valid"].sum(), min=1)
                loss = (
                    args.field_weight * field_loss
                    + args.nonnegative_weight * nonnegative
                    + args.jacobian_weight * jacobian_penalty
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.detach())
            train_batches += 1
        scheduler.step()

        val_metrics = evaluate(model, val_loader, x_norm, y_norm, transform, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_total / max(1, train_batches),
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "val_relative_l2": val_metrics["relative_l2"],
                "val_jacobian_min": val_metrics["jacobian_min"],
                "val_jacobian_mean": val_metrics["jacobian_mean"],
                "val_jacobian_nonpositive_fraction": val_metrics[
                    "jacobian_nonpositive_fraction"
                ],
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - start_time,
            }
        )
        pd.DataFrame(history).to_csv(history_path, index=False)
        if val_metrics["mse"] < best_val:
            best_val, best_epoch, stale = val_metrics["mse"], epoch, 0
            meta = {
                "architecture": "GeoFNO2D",
                "architecture_version": "paper_weighted_v2",
                "input_representation": "raw_point_cloud",
                "interpolation": False,
                "cnn_enabled": False,
                "in_channels": dataset.in_channels,
                "input_channels": dataset.input_channels,
                "source_mask_kind": dataset.mask_kind,
                "source_size": dataset.src_size,
                "latent_size": [args.latent_ny, args.latent_nx],
                "modes1": args.modes1,
                "modes2": args.modes2,
                "width": args.width,
                "depth": args.depth,
                "deformation_width": args.deformation_width,
                "max_displacement": args.max_displacement,
                "point_chunk_size": args.point_chunk_size,
                "code_dim": 2,
                "iphi_code": ["src_x_over_Lx", "src_y_over_Ly"],
                "quadrature_mode": args.quadrature_mode,
                "quadrature_column": args.quadrature_column,
                "quadrature_source": dataset.quadrature_source,
                "density_neighbors": args.density_neighbors,
                "jacobian_weight": args.jacobian_weight,
                "jacobian_floor": args.jacobian_floor,
                "jacobian_epsilon": args.jacobian_epsilon,
                "complex_optimizer": "ComplexAdamW",
                "iphi_lr": args.iphi_lr,
                "Lx": args.Lx,
                "Ly": args.Ly,
                "target_transform": args.target_transform,
                "target_transform_eps": args.target_transform_eps,
                "best_epoch": best_epoch,
                "best_val_mse": best_val,
                "split_assignments": str(split_path),
            }
            save_checkpoint(save_path, model, x_norm, y_norm, meta)
        else:
            stale += 1
        print(
            f"[EPOCH {epoch:04d}] train={train_total / max(1, train_batches):.6e} "
            f"val_mse={val_metrics['mse']:.6e} val_rel_l2={val_metrics['relative_l2']:.6e} "
            f"jac_min={val_metrics['jacobian_min']:.3e} "
            f"best={best_val:.6e}@{best_epoch} time={time.time() - start_time:.1f}s"
        )
        if stale >= args.patience:
            print(f"[INFO] early stopping after {stale} epochs without improvement.")
            break

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, x_norm, y_norm, transform, device)
    checkpoint["meta"].update({f"test_{name}": value for name, value in test_metrics.items()})
    torch.save(checkpoint, save_path)
    print(
        f"[RESULT] best_val_mse={best_val:.6e} best_epoch={best_epoch} "
        f"test_mse={test_metrics['mse']:.6e} test_rel_l2={test_metrics['relative_l2']:.6e}"
    )
    print(f"[SAVED] {save_path.resolve()}")
    print(f"[SAVED] {history_path.resolve()}")
    print(f"[SAVED] {split_path.resolve()}")


if __name__ == "__main__":
    main()
