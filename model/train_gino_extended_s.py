"""Train a basic GINO directly on the native Annex-20 CFD point clouds.

Both velocity and concentration tasks are supported.  The script never calls
``griddata``: learned local GNO transforms move features between native points
and a regular latent grid, where standard FNO layers are applied.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.complex_adamw import ComplexAdamW  # noqa: E402
from model.gino import BasicGINO2D  # noqa: E402
from model.train_concentration_geofno_extended_s import (  # noqa: E402
    PointNormalizer,
    RawPointCloudDataset,
    _to_device,
    pad_point_cloud_batch,
    save_checkpoint,
)
from model.train_fixed_u_srcpos_to_c import (  # noqa: E402
    TargetTransform,
    set_seed,
    split_indices,
    split_indices_from_csv,
)
from model.train_velocity_geofno_extended_s import RawVelocityPointCloudDataset  # noqa: E402


DEFAULT_WORKDIR = PROJECT_ROOT / "data" / "variable_inlet"


class TaskDataset(Dataset):
    """Expose one consistent interface for the two GINO tasks."""

    def __init__(
        self,
        base: RawPointCloudDataset,
        task: str,
        velocity_target_mode: str,
        velocity_aux_speed: bool = False,
    ):
        self.base = base
        self.task = str(task)
        self.velocity_aux_speed = bool(velocity_aux_speed)
        self.dataset = (
            RawVelocityPointCloudDataset(base, target_mode=velocity_target_mode)
            if self.task == "velocity"
            else base
        )
        self.rows = base.rows
        self.files = base.files

    @property
    def in_channels(self) -> int:
        return int(self.dataset.in_channels)

    @property
    def out_channels(self) -> int:
        if self.task == "velocity":
            return 3 if self.velocity_aux_speed else 2
        if self.task == "joint":
            return 2
        if self.task == "joint_uvc":
            return 3
        return 1

    @property
    def output_channels(self) -> List[str]:
        if self.task == "velocity":
            return ["Ux", "Uy", "speed"] if self.velocity_aux_speed else ["Ux", "Uy"]
        if self.task == "joint":
            return ["speed", "concentration"]
        if self.task == "joint_uvc":
            return ["Ux", "Uy", "concentration"]
        return ["concentration"]

    @property
    def input_channels(self) -> List[str]:
        return ["X", "Y", "inlet_u"] if self.task == "velocity" else self.base.input_channels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[int(index)]
        if self.task == "velocity" and self.velocity_aux_speed:
            sample = dict(sample)
            speed = torch.linalg.vector_norm(sample["target"][..., :2], dim=-1, keepdim=True)
            sample["target"] = torch.cat((sample["target"], speed), dim=-1)
        return sample


def point_weights(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    weights = batch["quadrature"] * batch["valid"].to(batch["quadrature"].dtype)
    return weights.unsqueeze(-1)


@torch.no_grad()
def make_weighted_normalizers(
    loader: DataLoader, device: torch.device, transform: TargetTransform
) -> Tuple[PointNormalizer, PointNormalizer]:
    x_sum = x_square = y_sum = y_square = None
    total_weight = 0.0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        weights = point_weights(batch)
        features = batch["features"]
        target = transform.forward(batch["target"])
        batch_weight = weights.sum()
        x_part = (features * weights).sum(dim=(0, 1), keepdim=True)
        x_square_part = (features.square() * weights).sum(dim=(0, 1), keepdim=True)
        y_part = (target * weights).sum(dim=(0, 1), keepdim=True)
        y_square_part = (target.square() * weights).sum(dim=(0, 1), keepdim=True)
        x_sum = x_part if x_sum is None else x_sum + x_part
        x_square = x_square_part if x_square is None else x_square + x_square_part
        y_sum = y_part if y_sum is None else y_sum + y_part
        y_square = y_square_part if y_square is None else y_square + y_square_part
        total_weight += float(batch_weight)
    if total_weight <= 0.0:
        raise RuntimeError("Training split has no positive quadrature weight.")
    x_mean, y_mean = x_sum / total_weight, y_sum / total_weight
    x_std = torch.sqrt(torch.clamp(x_square / total_weight - x_mean.square(), min=1e-12))
    y_std = torch.sqrt(torch.clamp(y_square / total_weight - y_mean.square(), min=1e-12))
    return PointNormalizer(x_mean, x_std), PointNormalizer(y_mean, y_std)


def weighted_training_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_raw: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    task: str,
    concentration_alpha: float,
    concentration_gamma: float,
    concentration_cap: float,
) -> torch.Tensor:
    weights = point_weights(batch)
    if task == "concentration":
        positive = target_raw.detach().clamp_min(0.0)
        scale = positive.amax(dim=1, keepdim=True).clamp_min(1e-12)
        plume = torch.clamp(
            1.0 + concentration_alpha * torch.pow(positive / scale, concentration_gamma),
            max=concentration_cap,
        )
        weights = weights * plume
    squared = (prediction - target).square()
    denominator = (weights.sum() * squared.shape[-1]).clamp_min(1.0)
    return (weights * squared).sum() / denominator


def velocity_multitask_loss(
    prediction_n: torch.Tensor,
    target_n: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    y_norm: PointNormalizer,
    speed_loss_weight: float,
    consistency_weight: float,
) -> torch.Tensor:
    """UV supervision + direct speed supervision + UV/speed consistency."""
    weights = point_weights(batch)
    weight_sum = weights.sum().clamp_min(1.0)
    uv_loss = (weights * (prediction_n[..., :2] - target_n[..., :2]).square()).sum()
    uv_loss = uv_loss / (2.0 * weight_sum)
    speed_loss = (weights * (prediction_n[..., 2:3] - target_n[..., 2:3]).square()).sum()
    speed_loss = speed_loss / weight_sum

    # Decode normalized network outputs to inlet-normalized physical velocity
    # units before enforcing s = sqrt(Ux^2 + Uy^2).
    prediction = y_norm.decode(prediction_n)
    speed_from_uv = torch.linalg.vector_norm(prediction[..., :2], dim=-1, keepdim=True)
    consistency = (weights * (prediction[..., 2:3] - speed_from_uv).square()).sum()
    consistency = consistency / weight_sum
    return uv_loss + speed_loss_weight * speed_loss + consistency_weight * consistency


def joint_speed_concentration_loss(
    prediction_n: torch.Tensor,
    target_n: torch.Tensor,
    target_raw: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    speed_weight: float,
    concentration_weight: float,
    concentration_alpha: float,
    concentration_gamma: float,
    concentration_cap: float,
) -> torch.Tensor:
    """Scale-balanced joint supervision for speed and concentration."""
    weights = point_weights(batch)
    denominator = weights.sum().clamp_min(1.0)
    speed_loss = (weights * (prediction_n[..., :1] - target_n[..., :1]).square()).sum()
    speed_loss = speed_loss / denominator
    concentration = target_raw[..., 1:2].detach().clamp_min(0.0)
    scale = concentration.amax(dim=1, keepdim=True).clamp_min(1e-12)
    plume = torch.clamp(
        1.0 + concentration_alpha * torch.pow(concentration / scale, concentration_gamma),
        max=concentration_cap,
    )
    concentration_weights = weights * plume
    concentration_loss = (
        concentration_weights * (prediction_n[..., 1:2] - target_n[..., 1:2]).square()
    ).sum() / concentration_weights.sum().clamp_min(1.0)
    return speed_weight * speed_loss + concentration_weight * concentration_loss


def joint_velocity_concentration_loss(
    prediction_n: torch.Tensor,
    target_n: torch.Tensor,
    target_raw: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    velocity_weight: float,
    concentration_weight: float,
    concentration_alpha: float,
    concentration_gamma: float,
    concentration_cap: float,
) -> torch.Tensor:
    """Scale-balanced supervision for vector velocity and concentration."""
    weights = point_weights(batch)
    denominator = weights.sum().clamp_min(1.0)
    velocity_loss = (weights * (prediction_n[..., :2] - target_n[..., :2]).square()).sum()
    velocity_loss = velocity_loss / (2.0 * denominator)
    concentration = target_raw[..., 2:3].detach().clamp_min(0.0)
    scale = concentration.amax(dim=1, keepdim=True).clamp_min(1e-12)
    plume = torch.clamp(
        1.0 + concentration_alpha * torch.pow(concentration / scale, concentration_gamma),
        max=concentration_cap,
    )
    concentration_weights = weights * plume
    concentration_loss = (
        concentration_weights * (prediction_n[..., 2:3] - target_n[..., 2:3]).square()
    ).sum() / concentration_weights.sum().clamp_min(1.0)
    return velocity_weight * velocity_loss + concentration_weight * concentration_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    x_norm: PointNormalizer,
    y_norm: PointNormalizer,
    transform: TargetTransform,
    device: torch.device,
    task: str,
    velocity_target_mode: str,
) -> Dict[str, float]:
    model.eval()
    weighted_square = weighted_absolute = weighted_target_square = weighted_count = 0.0
    node_square = node_count = 0.0
    speed_square = speed_target_square = speed_weighted_count = 0.0
    speed_node_square = speed_node_target_square = speed_node_count = 0.0
    consistency_square = 0.0
    joint_parts = {
        "velocity": {"square": 0.0, "target_square": 0.0, "absolute": 0.0, "count": 0.0,
                     "node_square": 0.0, "node_target_square": 0.0, "node_count": 0},
        "speed": {"square": 0.0, "target_square": 0.0, "absolute": 0.0, "count": 0.0,
                  "node_square": 0.0, "node_target_square": 0.0, "node_count": 0},
        "concentration": {"square": 0.0, "target_square": 0.0, "absolute": 0.0, "count": 0.0,
                          "node_square": 0.0, "node_target_square": 0.0, "node_count": 0},
    }
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        prediction_n = model(
            x_norm.encode(batch["features"]),
            batch["coords"],
            batch["valid"],
            batch["quadrature"],
        )
        prediction = transform.inverse(y_norm.decode(prediction_n))
        target = batch["target"]
        if task == "velocity" and velocity_target_mode == "inlet_normalized":
            inlet = batch["code"][:, 2].reshape(-1, 1, 1)
            prediction = prediction * inlet
            target = target * inlet
        elif task == "joint":
            prediction = prediction.clone()
            target = target.clone()
            if velocity_target_mode == "inlet_normalized":
                inlet = batch["code"][:, 2].reshape(-1, 1, 1)
                prediction[..., :1] = prediction[..., :1] * inlet
                target[..., :1] = target[..., :1] * inlet
        elif task == "joint_uvc":
            prediction = prediction.clone()
            target = target.clone()
            if velocity_target_mode == "inlet_normalized":
                inlet = batch["code"][:, 2].reshape(-1, 1, 1)
                prediction[..., :2] = prediction[..., :2] * inlet
                target[..., :2] = target[..., :2] * inlet
        valid = batch["valid"].unsqueeze(-1)
        weights = point_weights(batch)
        error = prediction - target
        weighted_square += float((weights * error.square()).sum())
        weighted_absolute += float((weights * error.abs()).sum())
        weighted_target_square += float((weights * target.square()).sum())
        weighted_count += float(weights.sum()) * error.shape[-1]
        node_square += float(error.square()[valid.expand_as(error)].sum())
        node_count += int(valid.sum()) * error.shape[-1]
        if task == "velocity" and prediction.shape[-1] >= 3:
            speed_error = prediction[..., 2:3] - target[..., 2:3]
            speed_square += float((weights * speed_error.square()).sum())
            speed_target_square += float((weights * target[..., 2:3].square()).sum())
            speed_weighted_count += float(weights.sum())
            speed_valid = valid.expand_as(speed_error)
            speed_node_square += float(speed_error.square()[speed_valid].sum())
            speed_node_target_square += float(target[..., 2:3].square()[speed_valid].sum())
            speed_node_count += int(speed_valid.sum())
            speed_from_uv = torch.linalg.vector_norm(prediction[..., :2], dim=-1, keepdim=True)
            consistency_square += float(
                (weights * (prediction[..., 2:3] - speed_from_uv).square()).sum()
            )
        if task == "joint":
            for channel, name in enumerate(("speed", "concentration")):
                part = joint_parts[name]
                part_error = prediction[..., channel : channel + 1] - target[..., channel : channel + 1]
                part_target = target[..., channel : channel + 1]
                part_valid = valid.expand_as(part_error)
                part["square"] += float((weights * part_error.square()).sum())
                part["absolute"] += float((weights * part_error.abs()).sum())
                part["target_square"] += float((weights * part_target.square()).sum())
                part["count"] += float(weights.sum())
                part["node_square"] += float(part_error.square()[part_valid].sum())
                part["node_target_square"] += float(part_target.square()[part_valid].sum())
                part["node_count"] += int(part_valid.sum())
        elif task == "joint_uvc":
            derived_speed = torch.linalg.vector_norm(prediction[..., :2], dim=-1, keepdim=True)
            target_speed = torch.linalg.vector_norm(target[..., :2], dim=-1, keepdim=True)
            specifications = (
                ("velocity", prediction[..., :2], target[..., :2]),
                ("speed", derived_speed, target_speed),
                ("concentration", prediction[..., 2:3], target[..., 2:3]),
            )
            for name, part_prediction, part_target in specifications:
                part = joint_parts[name]
                part_error = part_prediction - part_target
                part_valid = valid.expand_as(part_error)
                channels = part_error.shape[-1]
                part["square"] += float((weights * part_error.square()).sum())
                part["absolute"] += float((weights * part_error.abs()).sum())
                part["target_square"] += float((weights * part_target.square()).sum())
                part["count"] += float(weights.sum()) * channels
                part["node_square"] += float(part_error.square()[part_valid].sum())
                part["node_target_square"] += float(part_target.square()[part_valid].sum())
                part["node_count"] += int(part_valid.sum())
    result = {
        "weighted_mse": weighted_square / max(weighted_count, 1e-12),
        "weighted_mae": weighted_absolute / max(weighted_count, 1e-12),
        "weighted_relative_l2": math.sqrt(
            weighted_square / max(weighted_target_square, 1e-12)
        ),
        "node_mse": node_square / max(node_count, 1.0),
    }
    if task == "velocity" and speed_weighted_count > 0.0:
        result.update(
            {
                "speed_weighted_mse": speed_square / speed_weighted_count,
                "speed_weighted_relative_l2": math.sqrt(
                    speed_square / max(speed_target_square, 1e-12)
                ),
                "speed_node_mse": speed_node_square / max(speed_node_count, 1.0),
                "speed_node_relative_l2": math.sqrt(
                    speed_node_square / max(speed_node_target_square, 1e-12)
                ),
                "speed_consistency_weighted_mse": consistency_square / speed_weighted_count,
            }
        )
    if task in {"joint", "joint_uvc"}:
        names = ("speed", "concentration") if task == "joint" else ("velocity", "speed", "concentration")
        for name in names:
            part = joint_parts[name]
            result.update(
                {
                    f"{name}_weighted_mse": part["square"] / max(part["count"], 1e-12),
                    f"{name}_weighted_mae": part["absolute"] / max(part["count"], 1e-12),
                    f"{name}_weighted_relative_l2": math.sqrt(
                        part["square"] / max(part["target_square"], 1e-12)
                    ),
                    f"{name}_node_mse": part["node_square"] / max(part["node_count"], 1),
                    f"{name}_node_relative_l2": math.sqrt(
                        part["node_square"] / max(part["node_target_square"], 1e-12)
                    ),
                }
            )
    return result


def build_dataset(args: argparse.Namespace) -> TaskDataset:
    concentration = args.task == "concentration"
    joint = args.task == "joint"
    joint_uvc = args.task == "joint_uvc"
    base = RawPointCloudDataset(
        workdir=Path(args.workdir),
        exports_dirname=args.exports,
        cases_csv=args.cases,
        Lx=args.Lx,
        Ly=args.Ly,
        mask_kind=args.mask_kind,
        src_size=args.src_size,
        sigma=args.sigma,
        soft_k=args.soft_k,
        use_velocity_input=concentration,
        use_inlet_scalar=concentration or joint or joint_uvc,
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
        joint_speed_target=joint,
        joint_velocity_target=joint_uvc,
        speed_target_mode=args.velocity_target_mode,
    )
    return TaskDataset(base, args.task, args.velocity_target_mode, args.velocity_aux_speed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train basic GINO on native CFD nodes without preprocessing interpolation."
    )
    parser.add_argument(
        "--task", choices=["velocity", "concentration", "joint", "joint_uvc"], required=True
    )
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--exports", default="exports")
    parser.add_argument("--cases", default="cases.csv")
    parser.add_argument("--save", default=None)
    parser.add_argument("--Lx", type=float, default=9.0)
    parser.add_argument("--Ly", type=float, default=3.0)
    parser.add_argument("--mask_kind", choices=["rect", "gauss", "soft_rect"], default="soft_rect")
    parser.add_argument("--src_size", type=float, default=0.20)
    parser.add_argument("--sigma", type=float, default=0.12)
    parser.add_argument("--soft_k", type=float, default=80.0)
    parser.add_argument("--velocity_target_mode", choices=["raw", "inlet_normalized"], default="inlet_normalized")
    parser.add_argument("--velocity_aux_speed", action="store_true")
    parser.add_argument("--speed_loss_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=0.1)
    parser.add_argument("--joint_speed_weight", type=float, default=1.0)
    parser.add_argument("--joint_concentration_weight", type=float, default=1.0)
    parser.add_argument("--joint_velocity_weight", type=float, default=1.0)
    parser.add_argument("--target_transform", choices=["raw", "log10"], default="raw")
    parser.add_argument("--target_transform_eps", type=float, default=1e-8)
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--quadrature_mode", choices=["auto", "column", "knn"], default="auto")
    parser.add_argument("--quadrature_column", default=None)
    parser.add_argument("--quadrature_cache_dir", default=None)
    parser.add_argument("--density_neighbors", type=int, default=8)

    parser.add_argument(
        "--split_mode",
        choices=["random", "spatial_checkerboard", "spatial_uniform", "spatial_left_right"],
        default="random",
    )
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
    parser.add_argument("--kernel_width", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--nonnegative_weight", type=float, default=1e-3)
    parser.add_argument("--concentration_weight_alpha", type=float, default=2.0)
    parser.add_argument("--concentration_weight_gamma", type=float, default=0.5)
    parser.add_argument("--concentration_weight_cap", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.save is None:
        args.save = str(
            PROJECT_ROOT / "checkpoints" / f"gino_{args.task}" / f"gino_{args.task}_raw_nodes.pt"
        )
    if args.task == "velocity" and args.target_transform != "raw":
        parser.error("Velocity training requires --target_transform raw.")
    if args.task in {"joint", "joint_uvc"} and args.target_transform != "raw":
        parser.error("Joint velocity/concentration training requires --target_transform raw.")
    if args.task != "velocity" and args.velocity_aux_speed:
        parser.error("--velocity_aux_speed is only valid for --task velocity.")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    dataset = build_dataset(args)
    if args.split_csv:
        train_indices, val_indices, test_indices = split_indices_from_csv(
            dataset.rows, args.split_csv
        )
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
    split_path = save_path.parent / f"split_assignments_gino_{args.task}.csv"
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

    transform = TargetTransform(args.target_transform, args.target_transform_eps)
    x_norm, y_norm = make_weighted_normalizers(stats_loader, device, transform)
    model = BasicGINO2D(
        in_channels=dataset.in_channels,
        out_channels=dataset.out_channels,
        width=args.width,
        modes1=args.modes1,
        modes2=args.modes2,
        depth=args.depth,
        latent_size=(args.latent_ny, args.latent_nx),
        kernel_width=args.kernel_width,
    ).to(device)
    optimizer = ComplexAdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    first_sample = dataset[0]
    print(
        f"[INFO] device={device} task={args.task} cases={len(dataset)} train/val/test="
        f"{len(train_indices)}/{len(val_indices)}/{len(test_indices)}"
    )
    print(f"[INFO] input_channels={dataset.input_channels}")
    print(
        f"[INFO] quadrature={dataset.base.quadrature_source} "
        f"range=({float(first_sample['quadrature'].min()):.3e},"
        f"{float(first_sample['quadrature'].max()):.3e})"
    )
    print(
        f"[INFO] basic GINO: raw nodes -> learned local GNO -> "
        f"latent={args.latent_ny}x{args.latent_nx} -> FNO -> learned local GNO -> raw nodes; "
        "interpolation=disabled IPHI=disabled"
    )

    best_val, best_epoch, stale = float("inf"), -1, 0
    history: List[dict] = []
    history_path = save_path.parent / f"history_gino_{args.task}.csv"
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        train_total = 0.0
        train_batches = 0
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            features = x_norm.encode(batch["features"])
            target_raw = batch["target"]
            target_n = y_norm.encode(transform.forward(target_raw))
            optimizer.zero_grad(set_to_none=True)
            prediction_n = model(
                features, batch["coords"], batch["valid"], batch["quadrature"]
            )
            if args.task == "velocity" and args.velocity_aux_speed:
                loss = velocity_multitask_loss(
                    prediction_n,
                    target_n,
                    batch,
                    y_norm,
                    args.speed_loss_weight,
                    args.consistency_weight,
                )
            elif args.task == "joint":
                loss = joint_speed_concentration_loss(
                    prediction_n,
                    target_n,
                    target_raw,
                    batch,
                    args.joint_speed_weight,
                    args.joint_concentration_weight,
                    args.concentration_weight_alpha,
                    args.concentration_weight_gamma,
                    args.concentration_weight_cap,
                )
            elif args.task == "joint_uvc":
                loss = joint_velocity_concentration_loss(
                    prediction_n,
                    target_n,
                    target_raw,
                    batch,
                    args.joint_velocity_weight,
                    args.joint_concentration_weight,
                    args.concentration_weight_alpha,
                    args.concentration_weight_gamma,
                    args.concentration_weight_cap,
                )
            else:
                loss = weighted_training_loss(
                    prediction_n,
                    target_n,
                    target_raw,
                    batch,
                    args.task,
                    args.concentration_weight_alpha,
                    args.concentration_weight_gamma,
                    args.concentration_weight_cap,
                )
            if args.task in {"concentration", "joint", "joint_uvc"} and args.nonnegative_weight > 0.0:
                prediction_raw = transform.inverse(y_norm.decode(prediction_n))
                if args.task == "joint_uvc":
                    negative = torch.relu(-prediction_raw[..., 2:3]).square()
                else:
                    negative = torch.relu(-prediction_raw).square()
                negative_loss = (negative * point_weights(batch)).sum() / point_weights(
                    batch
                ).sum().clamp_min(1.0)
                loss = loss + args.nonnegative_weight * negative_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_total += float(loss.detach())
            train_batches += 1
        scheduler.step()

        val = evaluate(
            model,
            val_loader,
            x_norm,
            y_norm,
            transform,
            device,
            args.task,
            args.velocity_target_mode,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_total / max(train_batches, 1),
            "val_weighted_mse": val["weighted_mse"],
            "val_weighted_mae": val["weighted_mae"],
            "val_weighted_relative_l2": val["weighted_relative_l2"],
            "val_node_mse": val["node_mse"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - start,
        }
        if "speed_weighted_mse" in val:
            row.update({f"val_{name}": value for name, value in val.items() if name.startswith("speed_")})
        if args.task in {"joint", "joint_uvc"}:
            row.update(
                {
                    f"val_{name}": value
                    for name, value in val.items()
                    if name.startswith("concentration_")
                }
            )
        if args.task == "joint_uvc":
            row.update(
                {
                    f"val_{name}": value
                    for name, value in val.items()
                    if name.startswith("velocity_")
                }
            )
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        validation_score = (
            0.5
            * (
                val[
                    "velocity_weighted_relative_l2"
                    if args.task == "joint_uvc"
                    else "speed_weighted_relative_l2"
                ]
                + val["concentration_weighted_relative_l2"]
            )
            if args.task in {"joint", "joint_uvc"}
            else val["weighted_mse"]
        )
        if validation_score < best_val:
            best_val, best_epoch, stale = validation_score, epoch, 0
            meta = {
                "architecture": "BasicGINO2D",
                "architecture_version": "local_stencil_quadrature_v1",
                "task": args.task,
                "input_representation": "raw_point_cloud",
                "interpolation": False,
                "iphi_enabled": False,
                "in_channels": dataset.in_channels,
                "out_channels": dataset.out_channels,
                "input_channels": dataset.input_channels,
                "output_channels": dataset.output_channels,
                "latent_size": [args.latent_ny, args.latent_nx],
                "modes1": args.modes1,
                "modes2": args.modes2,
                "width": args.width,
                "depth": args.depth,
                "kernel_width": args.kernel_width,
                "local_neighbourhood": "four_adjacent_latent_cells",
                "quadrature_source": dataset.base.quadrature_source,
                "density_neighbors": args.density_neighbors,
                "velocity_target_mode": args.velocity_target_mode,
                "velocity_aux_speed": args.velocity_aux_speed,
                "speed_loss_weight": args.speed_loss_weight,
                "consistency_weight": args.consistency_weight,
                "joint_speed_weight": args.joint_speed_weight,
                "joint_concentration_weight": args.joint_concentration_weight,
                "joint_velocity_weight": args.joint_velocity_weight,
                "training_strategy": (
                    "joint_end_to_end" if args.task in {"joint", "joint_uvc"} else "single_task"
                ),
                "velocity_teacher_input": False if args.task in {"joint", "joint_uvc"} else None,
                "target_transform": args.target_transform,
                "target_transform_eps": args.target_transform_eps,
                "Lx": args.Lx,
                "Ly": args.Ly,
                "source_mask_kind": args.mask_kind,
                "source_size": args.src_size,
                "sigma": args.sigma,
                "soft_k": args.soft_k,
                "best_epoch": best_epoch,
                "best_val_score": best_val,
                "split_assignments": str(split_path),
            }
            save_checkpoint(save_path, model, x_norm, y_norm, meta)
        else:
            stale += 1
        print(
            f"[EPOCH {epoch:04d}] train={row['train_loss']:.6e} "
            f"val_wmse={val['weighted_mse']:.6e} "
            f"val_rel_l2={val['weighted_relative_l2']:.6e} "
            f"best={best_val:.6e}@{best_epoch} time={row['seconds']:.1f}s"
            + (
                f" speed_rel={val['speed_weighted_relative_l2']:.6e}"
                if "speed_weighted_relative_l2" in val
                else ""
            )
            + (
                f" concentration_rel={val['concentration_weighted_relative_l2']:.6e}"
                if args.task in {"joint", "joint_uvc"}
                else ""
            )
            + (
                f" velocity_rel={val['velocity_weighted_relative_l2']:.6e}"
                if args.task == "joint_uvc"
                else ""
            )
        )
        if stale >= args.patience:
            print(f"[INFO] early stopping after {stale} stale epochs.")
            break

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test = evaluate(
        model,
        test_loader,
        x_norm,
        y_norm,
        transform,
        device,
        args.task,
        args.velocity_target_mode,
    )
    print(
        f"[TEST] weighted_mse={test['weighted_mse']:.6e} "
        f"weighted_mae={test['weighted_mae']:.6e} "
        f"weighted_rel_l2={test['weighted_relative_l2']:.6e} "
        f"node_mse={test['node_mse']:.6e}"
        + (
            f" speed_wmse={test['speed_weighted_mse']:.6e} "
            f"speed_wrel_l2={test['speed_weighted_relative_l2']:.6e} "
            f"speed_node_rel_l2={test['speed_node_relative_l2']:.6e} "
            f"consistency_wmse={test['speed_consistency_weighted_mse']:.6e}"
            if "speed_consistency_weighted_mse" in test
            else ""
        )
        + (
            f" speed_wrel_l2={test['speed_weighted_relative_l2']:.6e} "
            f"speed_node_rel_l2={test['speed_node_relative_l2']:.6e} "
            f"concentration_wrel_l2={test['concentration_weighted_relative_l2']:.6e} "
            f"concentration_node_rel_l2={test['concentration_node_relative_l2']:.6e}"
            if args.task == "joint"
            else ""
        )
        + (
            f" velocity_wrel_l2={test['velocity_weighted_relative_l2']:.6e} "
            f"velocity_node_rel_l2={test['velocity_node_relative_l2']:.6e} "
            f"speed_wrel_l2={test['speed_weighted_relative_l2']:.6e} "
            f"speed_node_rel_l2={test['speed_node_relative_l2']:.6e} "
            f"concentration_wrel_l2={test['concentration_weighted_relative_l2']:.6e} "
            f"concentration_node_rel_l2={test['concentration_node_relative_l2']:.6e}"
            if args.task == "joint_uvc"
            else ""
        )
    )
    print(f"[DONE] checkpoint={save_path} history={history_path} split={split_path}")


if __name__ == "__main__":
    main()
