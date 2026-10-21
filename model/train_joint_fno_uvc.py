"""Train a standard FNO from [source mask, X, Y, inlet_u] to UVC or C only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.train_concentration_fno_cnn_extended_s import ExtendedSFieldDataset  # noqa: E402
from model.train_fixed_u_srcpos_to_c import Normalizer, is_power_of_two, set_seed  # noqa: E402
from model.train_velocity_fno_extended_s import (  # noqa: E402
    VelocityFNO2D,
    make_normalizers,
    split_loaders,
)


class JointUVCDataset(Dataset):
    """Return four conditioning fields and the three physical target fields."""

    def __init__(self, base: ExtendedSFieldDataset):
        self.base = base
        self.rows = base.rows

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index: int):
        row = self.rows.iloc[int(index)]
        case_id = str(row["case_id"])
        inlet_u = float(row["inlet_u"])
        source_x = float(row["src_x"])
        source_y = float(row["src_y"])
        ux, uy, concentration = self.base._load_grids(case_id)
        inlet_field = np.full((1, self.base.ny, self.base.nx), inlet_u, dtype=np.float32)
        inputs = np.concatenate(
            [
                self.base._make_mask(source_x, source_y),
                self.base.Xc.astype(np.float32),
                self.base.Yc.astype(np.float32),
                inlet_field,
            ],
            axis=0,
        )
        targets = np.stack([ux, uy, concentration], axis=0).astype(np.float32)
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class ConcentrationOnlyDataset(JointUVCDataset):
    """Use identical inputs and cases as JointUVCDataset but return C only."""

    def __getitem__(self, index: int):
        inputs, targets = super().__getitem__(index)
        return inputs, targets[2:3]


def evaluate_loss(model, loader, x_norm, y_norm, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_n = y_norm.encode(targets)
            prediction_n = model(x_norm.encode(inputs))
            total += torch.mean((prediction_n - target_n) ** 2).item() * inputs.shape[0]
            count += inputs.shape[0]
    return total / max(1, count)


def save_checkpoint(path, model, x_norm, y_norm, best_val, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "x_mean": x_norm.mean.detach().cpu(),
            "x_std": x_norm.std.detach().cpu(),
            "y_mean": y_norm.mean.detach().cpu(),
            "y_std": y_norm.std.detach().cpu(),
            "best_val_mse": float(best_val),
            "meta": meta,
        },
        path,
    )


@torch.no_grad()
def evaluate_test(model, loader, dataset, test_indices, x_norm, y_norm, device, output_dir, target_mode):
    model.eval()
    rows = []
    offset = 0
    n_outputs = 3 if target_mode == "joint_uvc" else 1
    squared_sum = np.zeros(n_outputs, dtype=np.float64)
    value_count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        prediction = y_norm.decode(model(x_norm.encode(inputs)))
        difference = prediction - targets
        squared_sum += difference.square().sum(dim=(0, 2, 3)).cpu().numpy()
        value_count += targets.shape[0] * targets.shape[2] * targets.shape[3]

        for local_index in range(targets.shape[0]):
            dataset_index = int(test_indices[offset + local_index])
            metadata = dataset.rows.iloc[dataset_index]
            truth = targets[local_index]
            pred = prediction[local_index]
            c_index = 2 if target_mode == "joint_uvc" else 0
            concentration_error = torch.linalg.vector_norm((pred[c_index] - truth[c_index]).reshape(-1))
            concentration_scale = torch.linalg.vector_norm(truth[c_index].reshape(-1)).clamp_min(1e-12)
            row = {
                "case_id": str(metadata["case_id"]),
                "inlet_u": float(metadata["inlet_u"]),
                "src_x": float(metadata["src_x"]),
                "src_y": float(metadata["src_y"]),
                "concentration_rel_l2": float((concentration_error / concentration_scale).cpu()),
                "concentration_mse": float(torch.mean((pred[c_index] - truth[c_index]) ** 2).cpu()),
            }
            if target_mode == "joint_uvc":
                velocity_error = torch.linalg.vector_norm((pred[:2] - truth[:2]).reshape(-1))
                velocity_scale = torch.linalg.vector_norm(truth[:2].reshape(-1)).clamp_min(1e-12)
                row.update(
                    {
                        "velocity_rel_l2": float((velocity_error / velocity_scale).cpu()),
                        "ux_mse": float(torch.mean((pred[0] - truth[0]) ** 2).cpu()),
                        "uy_mse": float(torch.mean((pred[1] - truth[1]) ** 2).cpu()),
                    }
                )
            rows.append(row)
        offset += targets.shape[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.DataFrame(rows)
    prefix = "joint_fno_uvc" if target_mode == "joint_uvc" else "fno_concentration_variable_u"
    cases_path = output_dir / f"{prefix}_test_cases.csv"
    inlet_path = output_dir / f"{prefix}_metrics_by_inlet.csv"
    summary_path = output_dir / f"{prefix}_test_summary.json"
    cases.to_csv(cases_path, index=False)
    metrics = ["concentration_rel_l2", "concentration_mse"]
    if target_mode == "joint_uvc":
        metrics = ["velocity_rel_l2", *metrics, "ux_mse", "uy_mse"]
    by_inlet = cases.groupby("inlet_u", as_index=False)[metrics].agg(["mean", "std", "median"])
    by_inlet.columns = ["_".join(filter(None, map(str, column))) for column in by_inlet.columns]
    by_inlet.to_csv(inlet_path, index=False)
    concentration_sum_index = 2 if target_mode == "joint_uvc" else 0
    summary = {
        "target_mode": target_mode,
        "n_cases": int(len(cases)),
        "test_inlet_velocities": sorted(cases["inlet_u"].unique().tolist()),
        "mean_concentration_rel_l2": float(cases["concentration_rel_l2"].mean()),
        "median_concentration_rel_l2": float(cases["concentration_rel_l2"].median()),
        "pooled_concentration_mse": float(squared_sum[concentration_sum_index] / max(1, value_count)),
        "cases_csv": str(cases_path.resolve()),
        "per_inlet_csv": str(inlet_path.resolve()),
    }
    if target_mode == "joint_uvc":
        summary.update(
            {
                "mean_velocity_rel_l2": float(cases["velocity_rel_l2"].mean()),
                "median_velocity_rel_l2": float(cases["velocity_rel_l2"].median()),
                "pooled_ux_mse": float(squared_sum[0] / max(1, value_count)),
                "pooled_uy_mse": float(squared_sum[1] / max(1, value_count)),
            }
        )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--exports", default="exports")
    parser.add_argument("--cases", default="cases.csv")
    parser.add_argument("--grid_cache_dir", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_mode", default="joint_uvc", choices=["joint_uvc", "concentration"])
    parser.add_argument("--split_mode", default="inlet_group", choices=["inlet_group", "random"])
    parser.add_argument("--split_csv", default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--mask_kind", default="soft_rect", choices=["rect", "gauss", "soft_rect"])
    parser.add_argument("--src_size", type=float, default=0.20)
    parser.add_argument("--sigma", type=float, default=0.12)
    parser.add_argument("--soft_k", type=float, default=30.0)
    parser.add_argument("--nx", type=int, default=384)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--modes1", type=int, default=16)
    parser.add_argument("--modes2", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--max_u_ratio", type=float, default=2.0)
    parser.add_argument("--min_c_max", type=float, default=1e-8)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    if args.amp and device.type == "cuda" and not (is_power_of_two(args.nx) and is_power_of_two(args.ny)):
        print("[WARN] Disabling AMP because the FFT grid is not power-of-two compatible.")
        args.amp = False

    base = ExtendedSFieldDataset(
        workdir=Path(args.workdir),
        exports_dirname=args.exports,
        cases_csv=args.cases,
        nx=args.nx,
        ny=args.ny,
        mask_kind=args.mask_kind,
        src_size=args.src_size,
        sigma=args.sigma,
        soft_k=args.soft_k,
        use_velocity_input=False,
        use_inlet_scalar=True,
        use_grid_cache=True,
        grid_cache_dir=Path(args.grid_cache_dir),
        limit_samples=args.limit_samples,
        sample_seed=args.sample_seed,
        max_u_ratio=args.max_u_ratio,
        min_c_max=args.min_c_max,
    )
    dataset = JointUVCDataset(base) if args.target_mode == "joint_uvc" else ConcentrationOnlyDataset(base)
    (train_loader, val_loader, test_loader), (_, _, test_indices) = split_loaders(dataset, args, device)
    x_norm, y_norm = make_normalizers(train_loader, device)
    model = VelocityFNO2D(
        in_ch=4,
        out_ch=3 if args.target_mode == "joint_uvc" else 1,
        width=args.width,
        modes1=args.modes1,
        modes2=args.modes2,
        depth=args.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    save_path = Path(args.save)
    history_path = save_path.with_name(f"history_{save_path.stem}.csv")
    meta = {
        "architecture": "standard_FNO2D_joint_UVC" if args.target_mode == "joint_uvc" else "standard_FNO2D_concentration_variable_u",
        "input_channels": ["S_mask", "X", "Y", "inlet_u"],
        "output_channels": ["Ux", "Uy", "C"] if args.target_mode == "joint_uvc" else ["C"],
        "target_mode": args.target_mode,
        "dataset_cases": len(dataset),
        "dataset_workdir": str(base.workdir.resolve()),
        "grid_cache_dir": str(base.grid_cache_dir.resolve()),
        "nx": args.nx,
        "ny": args.ny,
        "width": args.width,
        "modes1": args.modes1,
        "modes2": args.modes2,
        "depth": args.depth,
        "mask_kind": base.mask_kind,
        "src_size": base.src_size,
        "soft_k": base.soft_k,
        "split": dict(base.split_meta),
        "target_scaling": "channelwise standardization of physical Ux, Uy, C",
    }
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    started = time.perf_counter()

    target_label = "[Ux,Uy,C]" if args.target_mode == "joint_uvc" else "[C]"
    print(f"[INFO] device={device} cases={len(dataset)} model=standard FNO [S,X,Y,u_in]->{target_label}")
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total = 0.0
        count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                target_n = y_norm.encode(targets)
                prediction_n = model(x_norm.encode(inputs))
                loss = torch.mean((prediction_n - target_n) ** 2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * inputs.shape[0]
            count += inputs.shape[0]
        scheduler.step()
        train_loss = total / max(1, count)
        val_loss = evaluate_loss(model, val_loader, x_norm, y_norm, device)
        improved = best_val - val_loss > args.min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            meta.update({"best_epoch": epoch, "epochs_completed": epoch, "total_training_time_sec": time.perf_counter() - started})
            save_checkpoint(save_path, model, x_norm, y_norm, best_val, meta)
        else:
            bad_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_mse_normalized": train_loss,
                "val_mse_normalized": val_loss,
                "best_val_mse_normalized": best_val,
                "lr": optimizer.param_groups[0]["lr"],
                "bad_epochs": bad_epochs,
                "epoch_time_sec": time.perf_counter() - epoch_started,
            }
        )
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"[{epoch:03d}/{args.epochs}] train={train_loss:.6e} val={val_loss:.6e} best={best_val:.6e} best_ep={best_epoch} bad={bad_epochs}/{args.patience}", flush=True)
        if bad_epochs >= args.patience:
            print(f"[EARLY STOP] best_epoch={best_epoch} best_val={best_val:.6e}")
            break

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    summary = evaluate_test(
        model, test_loader, dataset, test_indices, x_norm, y_norm, device, Path(args.output_dir), args.target_mode
    )
    checkpoint["meta"]["test_summary"] = summary
    torch.save(checkpoint, save_path)
    print("[RESULT] " + json.dumps(summary, sort_keys=True), flush=True)
    print(f"[SAVED] checkpoint={save_path.resolve()} history={history_path.resolve()}")


if __name__ == "__main__":
    main()
