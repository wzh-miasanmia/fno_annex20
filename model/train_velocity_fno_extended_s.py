import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.train_concentration_fno_cnn_extended_s import (  # noqa: E402
    ExtendedSFieldDataset,
    Normalizer,
    SpectralConv2d,
    is_power_of_two,
    set_seed,
    split_indices,
    split_indices_from_csv,
)


class VelocityDataset(Dataset):
    def __init__(self, base: ExtendedSFieldDataset, target_mode: str = "inlet_normalized"):
        self.base = base
        self.rows = base.rows
        self.split_meta = base.split_meta
        self.target_mode = str(target_mode)
        if self.target_mode not in {"raw", "inlet_normalized"}:
            raise ValueError("target_mode must be one of: raw, inlet_normalized")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        idx = int(idx)
        row = self.base.rows.iloc[idx]
        inlet_u = float(row["inlet_u"])
        cid = str(row["case_id"])
        ux, uy, _ = self.base._load_grids(cid)
        sx = float(row["src_x"])
        sy = float(row["src_y"])
        geometry_mask = torch.from_numpy(
            self.base._make_mask(sx, sy).astype("float32")
        )
        u_field = torch.full((1, self.base.ny, self.base.nx), inlet_u, dtype=torch.float32)
        x = torch.cat(
            [
                geometry_mask,
                u_field,
                torch.from_numpy(self.base.Xc.astype("float32")),
                torch.from_numpy(self.base.Yc.astype("float32")),
            ],
            dim=0,
        )
        y_array = np.stack([ux, uy], axis=0).astype("float32")
        if self.target_mode == "inlet_normalized":
            y_array = y_array / max(abs(inlet_u), 1e-8)
        y = torch.from_numpy(y_array)
        return x, y


class VelocityFNO2D(nn.Module):
    def __init__(self, in_ch: int = 4, out_ch: int = 2, width: int = 64, modes1: int = 24, modes2: int = 16, depth: int = 4):
        super().__init__()
        self.fc0 = nn.Conv2d(in_ch, width, kernel_size=1)
        self.spec_layers = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)])
        self.w_layers = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)])
        self.act = nn.GELU()
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.fc0(x)
        for spec, w in zip(self.spec_layers, self.w_layers):
            x = self.act(spec(x) + w(x))
        x = self.act(self.fc1(x))
        return self.fc2(x)


@torch.no_grad()
def make_normalizers(loader: DataLoader, device: torch.device) -> Tuple[Normalizer, Normalizer]:
    x_sum = x_sq_sum = y_sum = y_sq_sum = None
    x_count = y_count = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        xs = xb.sum(dim=(0, 2, 3), keepdim=True)
        x2s = (xb * xb).sum(dim=(0, 2, 3), keepdim=True)
        ys = yb.sum(dim=(0, 2, 3), keepdim=True)
        y2s = (yb * yb).sum(dim=(0, 2, 3), keepdim=True)
        x_sum = xs if x_sum is None else x_sum + xs
        x_sq_sum = x2s if x_sq_sum is None else x_sq_sum + x2s
        y_sum = ys if y_sum is None else y_sum + ys
        y_sq_sum = y2s if y_sq_sum is None else y_sq_sum + y2s
        x_count += xb.shape[0] * xb.shape[2] * xb.shape[3]
        y_count += yb.shape[0] * yb.shape[2] * yb.shape[3]
    x_mean = x_sum / max(1, x_count)
    y_mean = y_sum / max(1, y_count)
    x_var = torch.clamp(x_sq_sum / max(1, x_count) - x_mean * x_mean, min=1e-12)
    y_var = torch.clamp(y_sq_sum / max(1, y_count) - y_mean * y_mean, min=1e-12)
    return Normalizer(x_mean, torch.sqrt(x_var)), Normalizer(y_mean, torch.sqrt(y_var))


def velocity_to_physical(y: torch.Tensor, x_raw: torch.Tensor, target_mode: str) -> torch.Tensor:
    if target_mode == "inlet_normalized":
        return y * x_raw[:, 1:2]
    return y


@torch.no_grad()
def eval_mse(model, loader, x_norm, y_norm, device, target_mode: str):
    model.eval()
    norm_total = raw_total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        target_n = y_norm.encode(yb)
        pred_n = model(x_norm.encode(xb))
        pred = y_norm.decode(pred_n)
        pred_physical = velocity_to_physical(pred, xb, target_mode)
        target_physical = velocity_to_physical(yb, xb, target_mode)
        norm_total += torch.mean((pred_n - target_n) ** 2).item() * xb.shape[0]
        raw_total += torch.mean((pred_physical - target_physical) ** 2).item() * xb.shape[0]
        n += xb.shape[0]
    return norm_total / max(1, n), raw_total / max(1, n)


def save_checkpoint(path: Path, model, x_norm, y_norm, best_val, meta):
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


def split_indices_by_inlet(
    rows: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    inlet_values = np.sort(rows["inlet_u"].astype(float).unique())
    if len(inlet_values) < 3:
        raise ValueError("inlet_group split requires at least three unique inlet velocities.")
    rng = np.random.default_rng(int(seed))
    shuffled = inlet_values.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * float(test_ratio))))
    n_val = max(1, int(round(len(shuffled) * float(val_ratio))))
    if n_test + n_val >= len(shuffled):
        n_val = 1
        n_test = 1
    test_u = shuffled[:n_test]
    val_u = shuffled[n_test : n_test + n_val]
    train_u = shuffled[n_test + n_val :]
    inlet = rows["inlet_u"].astype(float).to_numpy()
    train_idx = np.flatnonzero(np.isin(inlet, train_u)).tolist()
    val_idx = np.flatnonzero(np.isin(inlet, val_u)).tolist()
    test_idx = np.flatnonzero(np.isin(inlet, test_u)).tolist()
    return train_idx, val_idx, test_idx, train_u, val_u, test_u


def split_loaders(dataset: VelocityDataset, args, device):
    base = dataset.base
    base.split_meta = {
        "split_mode": "external_csv" if args.split_csv else str(args.split_mode),
        "split_csv": None if args.split_csv is None else str(Path(args.split_csv).resolve()),
        "split_seed": int(args.split_seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
    }
    if args.split_csv:
        tr_idx, va_idx, te_idx = split_indices_from_csv(base.rows, args.split_csv)
        train_u = np.sort(base.rows.iloc[tr_idx]["inlet_u"].astype(float).unique())
        val_u = np.sort(base.rows.iloc[va_idx]["inlet_u"].astype(float).unique())
        test_u = np.sort(base.rows.iloc[te_idx]["inlet_u"].astype(float).unique())
    elif args.split_mode == "inlet_group":
        tr_idx, va_idx, te_idx, train_u, val_u, test_u = split_indices_by_inlet(
            base.rows,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
        )
    else:
        tr_idx, va_idx, te_idx = split_indices(base.rows, args.split_mode, args.val_ratio, args.test_ratio, args.split_seed)
        train_u = np.sort(base.rows.iloc[tr_idx]["inlet_u"].astype(float).unique())
        val_u = np.sort(base.rows.iloc[va_idx]["inlet_u"].astype(float).unique())
        test_u = np.sort(base.rows.iloc[te_idx]["inlet_u"].astype(float).unique())

    split_tag = "external_csv" if args.split_csv else str(args.split_mode)
    assignment_path = Path(args.save).parent / f"split_assignments_{split_tag}.csv"
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    assignments = pd.DataFrame(
        {
            "case_id": base.rows["case_id"].values,
            "split": np.full(len(base.rows), "train", dtype=object),
        }
    )
    assignments.loc[va_idx, "split"] = "val"
    assignments.loc[te_idx, "split"] = "test"
    assignments.to_csv(assignment_path, index=False)
    inlet_split_path = Path(args.save).parent / f"inlet_split_{split_tag}.csv"
    pd.DataFrame(
        {
            "inlet_u": np.concatenate([train_u, val_u, test_u]),
            "split": (
                ["train"] * len(train_u)
                + ["val"] * len(val_u)
                + ["test"] * len(test_u)
            ),
        }
    ).to_csv(inlet_split_path, index=False)
    pin = device.type == "cuda"
    kwargs = {"num_workers": args.workers, "pin_memory": pin, "persistent_workers": args.workers > 0}
    loaders = [
        DataLoader(torch.utils.data.Subset(dataset, tr_idx), batch_size=args.batch, shuffle=True, **kwargs),
        DataLoader(torch.utils.data.Subset(dataset, va_idx), batch_size=args.batch, shuffle=False, **kwargs),
        DataLoader(torch.utils.data.Subset(dataset, te_idx), batch_size=args.batch, shuffle=False, **kwargs),
    ]
    print(f"[INFO] cases: total={len(dataset)} train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")
    print(
        f"[INFO] inlet groups: train={len(train_u)} val={len(val_u)} test={len(test_u)} "
        f"| val_u={np.sort(val_u).tolist()} | test_u={np.sort(test_u).tolist()}"
    )
    print(f"[INFO] split assignments={assignment_path}")
    print(f"[INFO] inlet split={inlet_split_path}")
    return loaders, (tr_idx, va_idx, te_idx)


def main():
    ap = argparse.ArgumentParser(description="Train stage-1 Velocity-FNO: [S_geometry_mask, inlet_u, X, Y] -> [Ux, Uy].")
    ap.add_argument("--workdir", type=str, default=str(PROJECT_ROOT / "data" / "data_v_S_new"))
    ap.add_argument("--exports", type=str, default="exports")
    ap.add_argument("--cases", type=str, default="cases.csv")
    ap.add_argument(
        "--save",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "velocity_fno" / "velocity_fno_geometry_inlet_group_norm_9900.pt"),
    )
    ap.add_argument(
        "--grid_cache_dir",
        type=str,
        default=str(PROJECT_ROOT / "grid_cache" / "data_v_S_new_9900_nx384_ny128"),
    )
    ap.add_argument("--limit_samples", type=int, default=None)
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--max_u_ratio", type=float, default=2.0)
    ap.add_argument("--min_c_max", type=float, default=1e-8)
    ap.add_argument(
        "--split_mode",
        type=str,
        default="inlet_group",
        choices=["inlet_group", "random", "spatial_checkerboard", "spatial_uniform", "spatial_left_right"],
    )
    ap.add_argument("--split_csv", type=str, default=None)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--Lx", type=float, default=9.0)
    ap.add_argument("--Ly", type=float, default=3.0)
    ap.add_argument("--nx", type=int, default=384)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--mask_kind", type=str, default="rect", choices=["rect", "gauss", "soft_rect", "soft-rect"])
    ap.add_argument("--src_size", type=float, default=0.20)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--soft_k", type=float, default=80.0)
    ap.add_argument("--modes1", type=int, default=24)
    ap.add_argument("--modes2", type=int, default=16)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=35)
    ap.add_argument("--min_delta", type=float, default=0.0)
    ap.add_argument("--target_mode", type=str, default="inlet_normalized", choices=["raw", "inlet_normalized"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--log_every", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp and device.type == "cuda" and not (is_power_of_two(args.nx) and is_power_of_two(args.ny)):
        print("[WARN] Disabling AMP because FFT grid is not power-of-two compatible for half precision.")
        args.amp = False

    base = ExtendedSFieldDataset(
        workdir=Path(args.workdir),
        exports_dirname=args.exports,
        cases_csv=args.cases,
        Lx=args.Lx,
        Ly=args.Ly,
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
    dataset = VelocityDataset(base, target_mode=args.target_mode)
    (train_loader, val_loader, test_loader), _ = split_loaders(dataset, args, device)
    x_norm, y_norm = make_normalizers(train_loader, device)

    model = VelocityFNO2D(in_ch=4, width=args.width, modes1=args.modes1, modes2=args.modes2, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    meta = {
        "architecture": "VelocityFNO2D",
        "architecture_note": "Stage-1 model: predicts Ux,Uy from [S_geometry_mask,inlet_u,X,Y].",
        "in_ch": 4,
        "input_channels": ["S_geometry_mask", "inlet_u", "X", "Y"],
        "geometry_mask_kind": str(base.mask_kind),
        "src_size": float(base.src_size),
        "out_ch": 2,
        "width": int(args.width),
        "modes1": int(args.modes1),
        "modes2": int(args.modes2),
        "depth": int(args.depth),
        "target_mode": str(args.target_mode),
        "dataset_workdir": str(base.workdir.resolve()),
        "dataset_exports_dir": str(base.exports.resolve()),
        "dataset_cases_path": str(base.cases_path.resolve()),
        "dataset_cases": int(len(base)),
        "grid_cache_dir": str(base.grid_cache_dir.resolve()),
        "quality_filter_max_u_ratio": base.max_u_ratio,
        "quality_filter_min_c_max": base.min_c_max,
        "quality_filter_removed": list(base.quality_filter_removed),
        "Lx": float(base.Lx),
        "Ly": float(base.Ly),
        "nx": int(base.nx),
        "ny": int(base.ny),
        "split": dict(base.split_meta),
    }

    best_val = float("inf")
    best_epoch = -1
    bad = 0
    history: List[dict] = []
    started = time.perf_counter()
    print(
        f"[INFO] device={device} model=VelocityFNO2D width={args.width} "
        f"modes=({args.modes1},{args.modes2}) depth={args.depth} target_mode={args.target_mode}"
    )

    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total = raw_total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                ybn = y_norm.encode(yb)
                pred_n = model(x_norm.encode(xb))
                loss = torch.mean((pred_n - ybn) ** 2)
                pred = y_norm.decode(pred_n)
            scaler.scale(loss).backward()
            if args.grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(opt)
            scaler.update()
            total += loss.item() * xb.shape[0]
            pred_physical = velocity_to_physical(pred, xb, args.target_mode)
            target_physical = velocity_to_physical(yb, xb, args.target_mode)
            raw_total += torch.mean((pred_physical - target_physical) ** 2).item() * xb.shape[0]
            n += xb.shape[0]
        sched.step()
        train_norm = total / max(1, n)
        train_raw = raw_total / max(1, n)
        val_norm, val_raw = eval_mse(model, val_loader, x_norm, y_norm, device, args.target_mode)
        improved = (best_val - val_norm) > float(args.min_delta)
        if improved:
            best_val = float(val_norm)
            best_epoch = int(ep)
            bad = 0
            meta.update({"best_epoch": best_epoch, "epochs_completed": int(ep), "total_training_time_sec": float(time.perf_counter() - started)})
            save_checkpoint(Path(args.save), model, x_norm, y_norm, best_val, meta)
        else:
            bad += 1
        row = {
            "epoch": int(ep),
            "train_mse_norm": float(train_norm),
            "train_mse_raw": float(train_raw),
            "val_mse_norm": float(val_norm),
            "val_mse_raw": float(val_raw),
            "best_val_mse": float(best_val),
            "lr": float(opt.param_groups[0]["lr"]),
            "bad_count": int(bad),
            "epoch_time_sec": float(time.perf_counter() - t0),
            "cumulative_train_time_sec": float(time.perf_counter() - started),
        }
        history.append(row)
        if ep == 1 or ep % max(1, args.log_every) == 0:
            print(f"[{ep:04d}/{args.epochs}] train_norm={train_norm:.6e} train_raw={train_raw:.6e} val_norm={val_norm:.6e} val_raw={val_raw:.6e} best={best_val:.6e} best_ep={best_epoch} bad={bad}/{args.patience}")
        if bad >= args.patience:
            print(f"[EARLY STOP] Best epoch={best_epoch} val_norm={best_val:.6e}")
            break

    hist_path = Path(args.save).with_name(f"history_{Path(args.save).stem}.csv")
    pd.DataFrame(history).to_csv(hist_path, index=False)
    ckpt = torch.load(args.save, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_norm, test_raw = eval_mse(model, test_loader, x_norm, y_norm, device, args.target_mode)
    ckpt["meta"].update({"test_mse_norm": float(test_norm), "test_mse_raw": float(test_raw)})
    torch.save(ckpt, args.save)
    print(f"[RESULT] best_epoch={best_epoch} best_val_norm={best_val:.6e} test_norm={test_norm:.6e} test_raw={test_raw:.6e}")
    print(f"[INFO] history={hist_path}")
    print(f"[INFO] checkpoint={Path(args.save)}")


if __name__ == "__main__":
    main()
