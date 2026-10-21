"""Train a two-output Geo-FNO velocity surrogate on raw CFD nodes.

The model maps ``[X, Y, inlet_u]`` to ``[Ux, Uy]`` at the same unstructured
STAR-CCM+ nodes.  No grid interpolation or image-grid conversion is used.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.geofno import GeoFNO2D  # noqa: E402
from model.complex_adamw import ComplexAdamW  # noqa: E402
from model.train_concentration_fno_cnn_extended_s import load_case_csv  # noqa: E402
from model.train_concentration_geofno_extended_s import (  # noqa: E402
    PointNormalizer,
    RawPointCloudDataset,
    _to_device,
    make_normalizers,
    masked_mean,
    pad_point_cloud_batch,
)
from model.train_fixed_u_srcpos_to_c import (  # noqa: E402
    TargetTransform,
    set_seed,
    split_indices,
    split_indices_from_csv,
)
from model.train_velocity_fno_extended_s import split_indices_by_inlet  # noqa: E402


class RawVelocityPointCloudDataset(Dataset):
    """Velocity fields loaded on the native nodes of each exported CSV."""

    def __init__(self, base: RawPointCloudDataset, target_mode: str = "inlet_normalized"):
        self.base = base
        self.rows = base.rows
        self.target_mode = str(target_mode)
        if self.target_mode not in {"raw", "inlet_normalized"}:
            raise ValueError("target_mode must be raw or inlet_normalized.")

    @property
    def in_channels(self) -> int:
        return 3

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        index = int(index)
        row = self.rows.iloc[index]
        inlet_u = float(row["inlet_u"])
        frame = load_case_csv(self.base.files[index])
        x = frame["X (m)"].to_numpy(dtype=np.float32) / self.base.Lx
        y = frame["Y (m)"].to_numpy(dtype=np.float32) / self.base.Ly
        coords = np.clip(np.stack([x, y], axis=-1), 0.0, 1.0).astype(np.float32)
        quadrature = self.base.quadrature_for_frame(frame, coords, index)
        inlet = np.full((coords.shape[0], 1), inlet_u, dtype=np.float32)
        features = np.concatenate([coords, inlet], axis=-1)
        target = frame[["Velocity[i] (m/s)", "Velocity[j] (m/s)"]].to_numpy(dtype=np.float32)
        if self.target_mode == "inlet_normalized":
            target = target / max(abs(inlet_u), 1e-8)
        return {
            "features": torch.from_numpy(features),
            "coords": torch.from_numpy(coords),
            "target": torch.from_numpy(target),
            # Source position is a mesh-refinement descriptor here, not a
            # physical input to the velocity PDE.
            "code": torch.tensor(
                [float(row["src_x"]) / self.base.Lx, float(row["src_y"]) / self.base.Ly, inlet_u],
                dtype=torch.float32,
            ),
            "quadrature": torch.from_numpy(quadrature),
            "case_index": torch.tensor(index, dtype=torch.long),
        }


def velocity_to_physical(
    velocity: torch.Tensor, raw_features: torch.Tensor, target_mode: str
) -> torch.Tensor:
    if target_mode == "inlet_normalized":
        return velocity * raw_features[..., 2:3]
    return velocity


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    x_norm: PointNormalizer,
    y_norm: PointNormalizer,
    device: torch.device,
    target_mode: str,
) -> Dict[str, float]:
    model.eval()
    norm_square = raw_square = raw_absolute = target_square = 0.0
    jacobian_sum = jacobian_nonpositive = 0.0
    jacobian_min = float("inf")
    jacobian_count = 0
    count = 0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        target_n = y_norm.encode(batch["target"])
        prediction_n, aux = model(
            x_norm.encode(batch["features"]),
            batch["coords"],
            batch["valid"],
            code=batch["code"][..., :2],
            quadrature=batch["quadrature"],
            return_aux=True,
        )
        prediction = y_norm.decode(prediction_n)
        prediction_raw = velocity_to_physical(prediction, batch["features"], target_mode)
        target_raw = velocity_to_physical(batch["target"], batch["features"], target_mode)
        mask = batch["valid"].unsqueeze(-1).expand_as(target_n)
        norm_error = (prediction_n - target_n)[mask]
        raw_error = (prediction_raw - target_raw)[mask]
        raw_target = target_raw[mask]
        norm_square += float(norm_error.square().sum())
        raw_square += float(raw_error.square().sum())
        raw_absolute += float(raw_error.abs().sum())
        target_square += float(raw_target.square().sum())
        count += int(raw_error.numel())
        determinant = aux["jacobian_det"][batch["valid"]]
        jacobian_sum += float(determinant.sum())
        jacobian_nonpositive += float((determinant <= 0.0).sum())
        jacobian_min = min(jacobian_min, float(determinant.min()))
        jacobian_count += int(determinant.numel())
    return {
        "mse_norm": norm_square / max(1, count),
        "mse_raw": raw_square / max(1, count),
        "mae_raw": raw_absolute / max(1, count),
        "relative_l2_raw": math.sqrt(raw_square / max(target_square, 1e-12)),
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
        description="Train raw-node Velocity Geo-FNO: [X,Y,inlet_u] -> [Ux,Uy]."
    )
    parser.add_argument("--workdir", default=str(PROJECT_ROOT / "data" / "data_v_S_new"))
    parser.add_argument("--exports", default="exports")
    parser.add_argument("--cases", default="cases.csv")
    parser.add_argument(
        "--save",
        default=str(PROJECT_ROOT / "checkpoints" / "velocity_geofno" / "velocity_geofno_raw_nodes.pt"),
    )
    parser.add_argument("--Lx", type=float, default=9.0)
    parser.add_argument("--Ly", type=float, default=3.0)
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--quadrature_mode", choices=["auto", "column", "knn"], default="auto")
    parser.add_argument("--quadrature_column", default=None)
    parser.add_argument("--quadrature_cache_dir", default=None)
    parser.add_argument("--density_neighbors", type=int, default=8)
    parser.add_argument(
        "--split_mode",
        choices=["inlet_group", "random", "spatial_checkerboard", "spatial_uniform", "spatial_left_right"],
        default="inlet_group",
    )
    parser.add_argument("--split_csv", default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--target_mode", choices=["raw", "inlet_normalized"], default="inlet_normalized")

    parser.add_argument("--latent_ny", type=int, default=48)
    parser.add_argument("--latent_nx", type=int, default=144)
    parser.add_argument("--modes1", type=int, default=12)
    parser.add_argument("--modes2", type=int, default=12)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--deformation_width", type=int, default=32)
    parser.add_argument("--max_displacement", type=float, default=0.15)
    parser.add_argument("--point_chunk_size", type=int, default=2048)

    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--iphi_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--jacobian_weight", type=float, default=0.01)
    parser.add_argument("--jacobian_floor", type=float, default=0.05)
    parser.add_argument("--jacobian_epsilon", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--min_delta", type=float, default=0.0)
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
    # The base dataset is used for validated case/file discovery only.  Its
    # gridding path is never called by RawVelocityPointCloudDataset.
    base = RawPointCloudDataset(
        workdir=Path(args.workdir),
        exports_dirname=args.exports,
        cases_csv=args.cases,
        Lx=args.Lx,
        Ly=args.Ly,
        use_velocity_input=False,
        use_inlet_scalar=False,
        limit_samples=args.limit_samples,
        sample_seed=args.sample_seed,
        cache_in_memory=False,
        quadrature_mode=args.quadrature_mode,
        quadrature_column=args.quadrature_column,
        quadrature_cache_dir=(
            Path(args.quadrature_cache_dir)
            if args.quadrature_cache_dir
            else Path(args.save).parent / "quadrature_cache"
        ),
        density_neighbors=args.density_neighbors,
    )
    dataset = RawVelocityPointCloudDataset(base, target_mode=args.target_mode)
    if args.split_csv:
        train_indices, val_indices, test_indices = split_indices_from_csv(dataset.rows, args.split_csv)
    elif args.split_mode == "inlet_group":
        train_indices, val_indices, test_indices, _, _, _ = split_indices_by_inlet(
            dataset.rows, args.val_ratio, args.test_ratio, args.split_seed
        )
    else:
        train_indices, val_indices, test_indices = split_indices(
            dataset.rows, args.split_mode, args.val_ratio, args.test_ratio, args.split_seed
        )

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    split_labels = np.full(len(dataset), "", dtype=object)
    split_labels[np.asarray(train_indices)] = "train"
    split_labels[np.asarray(val_indices)] = "val"
    split_labels[np.asarray(test_indices)] = "test"
    split_table = dataset.rows.copy()
    split_table["split"] = split_labels
    split_path = save_path.parent / "split_assignments_velocity_geofno.csv"
    split_table.to_csv(split_path, index=False)

    loader_options = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": pad_point_cloud_batch,
    }
    train_loader = DataLoader(Subset(dataset, train_indices), shuffle=True, **loader_options)
    stats_loader = DataLoader(Subset(dataset, train_indices), shuffle=False, **loader_options)
    val_loader = DataLoader(Subset(dataset, val_indices), shuffle=False, **loader_options)
    test_loader = DataLoader(Subset(dataset, test_indices), shuffle=False, **loader_options)
    identity_transform = TargetTransform("raw")
    x_norm, y_norm = make_normalizers(stats_loader, device, identity_transform)

    model = GeoFNO2D(
        in_channels=dataset.in_channels,
        out_channels=2,
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
    history_path = save_path.parent / "history_velocity_geofno.csv"
    started = time.perf_counter()

    print(
        f"[INFO] device={device} raw-node Velocity Geo-FNO cases={len(dataset)} "
        f"train/val/test={len(train_indices)}/{len(val_indices)}/{len(test_indices)}"
    )
    print(
        f"[INFO] [X,Y,inlet_u] -> latent={args.latent_ny}x{args.latent_nx} -> [Ux,Uy]; "
        "interpolation=disabled"
    )
    first_sample = dataset[0]
    print(
        f"[INFO] quadrature={base.quadrature_source} "
        f"range=({float(first_sample['quadrature'].min()):.3e},"
        f"{float(first_sample['quadrature'].max()):.3e}); "
        "IPHI code=[src_x/Lx,src_y/Ly]"
    )
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_total = 0.0
        batches = 0
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            target_n = y_norm.encode(batch["target"])
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                prediction_n, aux = model(
                    x_norm.encode(batch["features"]),
                    batch["coords"],
                    batch["valid"],
                    code=batch["code"][..., :2],
                    quadrature=batch["quadrature"],
                    jacobian_epsilon=args.jacobian_epsilon,
                    return_aux=True,
                )
                field_loss = masked_mean((prediction_n - target_n).square(), batch["valid"])
                jacobian_penalty = (
                    torch.relu(args.jacobian_floor - aux["jacobian_det"]).square()
                    * batch["valid"].to(dtype=prediction_n.dtype)
                ).sum() / torch.clamp(batch["valid"].sum(), min=1)
                loss = field_loss + args.jacobian_weight * jacobian_penalty
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.detach())
            batches += 1
        scheduler.step()
        val = evaluate(model, val_loader, x_norm, y_norm, device, args.target_mode)
        train_loss = loss_total / max(1, batches)
        history.append(
            {
                "epoch": epoch,
                "train_mse_norm": train_loss,
                **{f"val_{name}": value for name, value in val.items()},
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - epoch_start,
            }
        )
        pd.DataFrame(history).to_csv(history_path, index=False)
        if (best_val - val["mse_norm"]) > args.min_delta:
            best_val, best_epoch, stale = val["mse_norm"], epoch, 0
            save_checkpoint(
                save_path,
                model,
                x_norm,
                y_norm,
                {
                    "architecture": "VelocityGeoFNO2D",
                    "architecture_version": "paper_weighted_v2",
                    "input_representation": "raw_point_cloud",
                    "input_channels": ["X", "Y", "inlet_u"],
                    "output_channels": ["Ux", "Uy"],
                    "interpolation": False,
                    "in_channels": dataset.in_channels,
                    "out_channels": 2,
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
                    "quadrature_source": base.quadrature_source,
                    "density_neighbors": args.density_neighbors,
                    "jacobian_weight": args.jacobian_weight,
                    "jacobian_floor": args.jacobian_floor,
                    "jacobian_epsilon": args.jacobian_epsilon,
                    "complex_optimizer": "ComplexAdamW",
                    "iphi_lr": args.iphi_lr,
                    "target_mode": args.target_mode,
                    "Lx": args.Lx,
                    "Ly": args.Ly,
                    "best_epoch": best_epoch,
                    "best_val_mse_norm": best_val,
                    "split_assignments": str(split_path),
                },
            )
        else:
            stale += 1
        print(
            f"[EPOCH {epoch:04d}] train_norm={train_loss:.6e} "
            f"val_norm={val['mse_norm']:.6e} val_raw={val['mse_raw']:.6e} "
            f"jac_min={val['jacobian_min']:.3e} "
            f"best={best_val:.6e}@{best_epoch} time={time.perf_counter() - epoch_start:.1f}s"
        )
        if stale >= args.patience:
            print(f"[INFO] early stopping after {stale} epochs without improvement.")
            break

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test = evaluate(model, test_loader, x_norm, y_norm, device, args.target_mode)
    checkpoint["meta"].update({f"test_{name}": value for name, value in test.items()})
    checkpoint["meta"]["total_training_time_sec"] = time.perf_counter() - started
    torch.save(checkpoint, save_path)
    print(
        f"[RESULT] best_epoch={best_epoch} best_val_norm={best_val:.6e} "
        f"test_norm={test['mse_norm']:.6e} test_raw={test['mse_raw']:.6e}"
    )
    print(f"[SAVED] {save_path.resolve()}")
    print(f"[SAVED] {history_path.resolve()}")
    print(f"[SAVED] {split_path.resolve()}")


if __name__ == "__main__":
    main()
