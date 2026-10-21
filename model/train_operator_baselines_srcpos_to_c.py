import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_fixed_u_srcpos_to_c import (
    CasesFNOFieldDataset,
    Normalizer,
    TargetTransform,
    evaluate_metrics,
    gradient_matching_loss,
    is_power_of_two,
    laplacian_smoothness_loss,
    make_normalizers,
    save_history_csv,
    save_loss_curve,
    set_seed,
    split_indices,
    split_indices_from_csv,
)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FCN2D(nn.Module):
    """A plain fully convolutional baseline with only local 3x3 kernels."""

    def __init__(self, in_ch: int = 3, out_ch: int = 1, width: int = 64, depth: int = 8):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, width, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(max(0, int(depth) - 2)):
            layers.extend(
                [
                    nn.Conv2d(width, width, kernel_size=3, padding=1),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Conv2d(width, out_ch, kernel_size=1))
        self.net = nn.Sequential(*layers)
        self.width = int(width)
        self.depth = int(depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2D(nn.Module):
    """A compact U-Net baseline for image-to-image field regression."""

    def __init__(self, in_ch: int = 3, out_ch: int = 1, width: int = 32, depth: int = 4):
        super().__init__()
        if int(depth) != 4:
            raise ValueError("UNet2D currently uses depth=4.")
        w = int(width)
        self.enc1 = ConvBlock(in_ch, w)
        self.enc2 = ConvBlock(w, 2 * w)
        self.enc3 = ConvBlock(2 * w, 4 * w)
        self.enc4 = ConvBlock(4 * w, 8 * w)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(8 * w, 16 * w)

        self.up4 = nn.ConvTranspose2d(16 * w, 8 * w, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(16 * w, 8 * w)
        self.up3 = nn.ConvTranspose2d(8 * w, 4 * w, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(8 * w, 4 * w)
        self.up2 = nn.ConvTranspose2d(4 * w, 2 * w, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(4 * w, 2 * w)
        self.up1 = nn.ConvTranspose2d(2 * w, w, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(2 * w, w)
        self.out = nn.Conv2d(w, out_ch, kernel_size=1)
        self.width = w
        self.depth = int(depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


class DeepONet2D(nn.Module):
    """Branch-trunk DeepONet baseline on the same structured grid."""

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        width: int = 64,
        rank: int = 128,
        trunk_width: int = 128,
    ):
        super().__init__()
        if out_ch != 1:
            raise ValueError("DeepONet2D currently supports out_ch=1.")
        w = int(width)
        r = int(rank)
        self.branch = nn.Sequential(
            nn.Conv2d(in_ch, w, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(w, w, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(w, 2 * w, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * w, 4 * w, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4 * w, r),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, int(trunk_width)),
            nn.GELU(),
            nn.Linear(int(trunk_width), int(trunk_width)),
            nn.GELU(),
            nn.Linear(int(trunk_width), r),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.width = w
        self.rank = r
        self.trunk_width = int(trunk_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, ny, nx = x.shape
        branch_code = self.branch(x)
        coords = x[:, 1:3].permute(0, 2, 3, 1).reshape(bsz, ny * nx, 2)
        trunk_code = self.trunk(coords)
        out = torch.einsum("br,bnr->bn", branch_code, trunk_code) / np.sqrt(float(self.rank))
        return (out + self.bias).reshape(bsz, 1, ny, nx)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def build_model(args: argparse.Namespace, in_ch: int) -> nn.Module:
    kind = str(args.model_kind).lower().strip()
    if kind == "fcn":
        return FCN2D(in_ch=in_ch, out_ch=1, width=args.width, depth=args.depth)
    if kind == "unet":
        return UNet2D(in_ch=in_ch, out_ch=1, width=args.width, depth=args.depth)
    if kind == "deeponet":
        return DeepONet2D(
            in_ch=in_ch,
            out_ch=1,
            width=args.width,
            rank=args.rank,
            trunk_width=args.trunk_width,
        )
    raise ValueError("model_kind must be one of: fcn, unet, deeponet")


def save_checkpoint(save_path: Path, model: nn.Module, x_norm: Normalizer, y_norm: Normalizer, best_val_mse: float, meta: dict):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "x_mean": x_norm.mean.detach().cpu(),
            "x_std": x_norm.std.detach().cpu(),
            "y_mean": y_norm.mean.detach().cpu(),
            "y_std": y_norm.std.detach().cpu(),
            "best_val_mse": float(best_val_mse),
            "meta": meta,
        },
        save_path,
    )


@torch.no_grad()
def evaluate_case_metrics(model, subset, x_norm, y_norm, target_transform, device, experiment_name: str, ckpt_path: Path) -> pd.DataFrame:
    model.eval()
    base_ds = subset.dataset if hasattr(subset, "dataset") else subset
    indices = list(subset.indices) if hasattr(subset, "indices") else list(range(len(subset)))
    records = []
    for ds_idx in indices:
        xb, yb = base_ds[int(ds_idx)]
        xb = xb.unsqueeze(0).to(device)
        yb = yb.unsqueeze(0).to(device)
        pred_n = model(x_norm.encode(xb))
        pred = target_transform.inverse(y_norm.decode(pred_n))
        c_gt = yb.detach().cpu().numpy()[0, 0]
        c_pr = pred.detach().cpu().numpy()[0, 0]
        err = c_pr - c_gt
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        rel_l2 = float(np.linalg.norm(err.reshape(-1)) / (np.linalg.norm(c_gt.reshape(-1)) + 1e-12))
        high_mask = np.abs(c_gt) >= float(np.percentile(np.abs(c_gt), 90.0))
        high_rmse = float(np.sqrt(np.mean(err[high_mask] ** 2))) if np.any(high_mask) else float("nan")
        high_mae = float(np.mean(np.abs(err[high_mask]))) if np.any(high_mask) else float("nan")
        row = base_ds.rows.iloc[int(ds_idx)]
        records.append(
            {
                "experiment": experiment_name,
                "checkpoint": str(ckpt_path),
                "case_id": str(row["case_id"]),
                "inlet_u": float(row["inlet_u"]),
                "src_x": float(row["src_x"]),
                "src_y": float(row["src_y"]),
                "rmse_c": rmse,
                "mae_c": mae,
                "rel_l2_c": rel_l2,
                "high_rmse_c": high_rmse,
                "high_mae_c": high_mae,
                "mean_gt_c": float(np.mean(c_gt)),
                "max_gt_c": float(np.max(c_gt)),
            }
        )
    return pd.DataFrame(records)


def train(args, model, train_loader, val_loader, x_norm, y_norm, target_transform, device, meta: dict):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    save_path = Path(args.save)
    best_val_norm = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    start = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        ep_start = time.perf_counter()
        model.train()
        total_norm = 0.0
        total_raw = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                target_t = target_transform.forward(yb)
                target_n = y_norm.encode(target_t)
                pred_n = model(x_norm.encode(xb))
                loss_mse = torch.mean((pred_n - target_n) ** 2)
                pred_raw = target_transform.inverse(y_norm.decode(pred_n))
                loss_grad = (
                    args.gradient_weight * gradient_matching_loss(pred_raw, yb)
                    if args.gradient_weight > 0
                    else torch.zeros((), device=device)
                )
                loss_smooth = (
                    args.smoothness_weight * laplacian_smoothness_loss(pred_raw)
                    if args.smoothness_weight > 0
                    else torch.zeros((), device=device)
                )
                loss = loss_mse + loss_grad + loss_smooth

            scaler.scale(loss).backward()
            if args.grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                raw_mse = torch.mean((pred_raw - yb) ** 2)
            total_norm += float(loss.item()) * xb.shape[0]
            total_raw += float(raw_mse.item()) * xb.shape[0]
            n += xb.shape[0]

        sched.step()
        train_step_norm = total_norm / max(1, n)
        train_step_raw = total_raw / max(1, n)
        train_eval_norm, train_eval_raw = evaluate_metrics(model, train_loader, x_norm, y_norm, target_transform, device)
        val_norm, val_raw = evaluate_metrics(model, val_loader, x_norm, y_norm, target_transform, device)

        elapsed = float(time.perf_counter() - start)
        epoch_time = float(time.perf_counter() - ep_start)
        improved = (best_val_norm - val_norm) > float(args.min_delta)
        if improved:
            best_val_norm = float(val_norm)
            best_epoch = int(ep)
            bad_epochs = 0
            meta.update(
                {
                    "epochs_completed": int(ep),
                    "best_epoch": int(best_epoch),
                    "total_training_time_sec": elapsed,
                    "avg_epoch_time_sec": elapsed / float(ep),
                }
            )
            save_checkpoint(save_path, model, x_norm, y_norm, best_val_norm, meta)
        else:
            bad_epochs += 1

        history.append(
            {
                "epoch": int(ep),
                "train_loss": float(train_step_norm),
                "val_mse": float(val_norm),
                "best_val_mse": float(best_val_norm),
                "train_loss_norm": float(train_step_norm),
                "val_mse_norm": float(val_norm),
                "train_step_mse_norm": float(train_step_norm),
                "train_eval_mse_norm": float(train_eval_norm),
                "train_mse_raw": float(train_step_raw),
                "train_step_mse_raw": float(train_step_raw),
                "train_eval_mse_raw": float(train_eval_raw),
                "val_mse_raw": float(val_raw),
                "lr": float(opt.param_groups[0]["lr"]),
                "bad_count": int(bad_epochs),
                "epoch_time_sec": epoch_time,
                "cumulative_train_time_sec": elapsed,
                "avg_epoch_time_sec": elapsed / float(ep),
            }
        )

        if args.save_last:
            save_checkpoint(save_path.with_name(save_path.stem + "_last.pt"), model, x_norm, y_norm, best_val_norm, meta)

        if ep % max(1, args.log_every) == 0 or ep == 1:
            print(
                f"[{ep:04d}/{args.epochs}] "
                f"train_step_norm={train_step_norm:.6e} train_eval_norm={train_eval_norm:.6e} "
                f"val_norm={val_norm:.6e} train_step_raw={train_step_raw:.6e} val_raw={val_raw:.6e} "
                f"best_val_norm={best_val_norm:.6e} best_ep={best_epoch} bad={bad_epochs}/{args.patience} "
                f"epoch_time={epoch_time:.2f}s total_time={elapsed/60.0:.2f}m"
            )

        if bad_epochs >= args.patience:
            print(f"[EARLY STOP] no improvement for {args.patience} epochs. Best at epoch {best_epoch}.")
            break

    elapsed = float(time.perf_counter() - start)
    meta.update(
        {
            "epochs_completed": int(len(history)),
            "best_epoch": int(best_epoch),
            "total_training_time_sec": elapsed,
            "avg_epoch_time_sec": elapsed / float(max(1, len(history))),
        }
    )
    return best_val_norm, best_epoch, history, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_kind", type=str, required=True, choices=["fcn", "unet", "deeponet"])
    ap.add_argument("--workdir", type=str, default="data/dense969")
    ap.add_argument("--exports", type=str, default="exports")
    ap.add_argument("--cases", type=str, default="cases.csv")
    ap.add_argument("--limit_samples", type=int, default=None)
    ap.add_argument("--limit_strategy", type=str, default="uniform_source", choices=["uniform_source", "head", "random"])
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--split_mode", type=str, default="random", choices=["random", "spatial_checkerboard", "spatial_uniform", "spatial_left_right"])
    ap.add_argument("--split_csv", type=str, default=None)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--fixed_inlet_u", type=float, default=0.2)
    ap.add_argument("--inlet_u_tol", type=float, default=1e-8)
    ap.add_argument("--source_aspect_ratio", type=float, default=3.0)

    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--smoothness_weight", type=float, default=0.0)
    ap.add_argument("--gradient_weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--save", type=str, required=True)
    ap.add_argument("--target_transform", type=str, default="raw", choices=["raw", "log10"])
    ap.add_argument("--target_transform_eps", type=float, default=1e-8)
    ap.add_argument("--use_velocity_input", dest="use_velocity_input", action="store_true", default=False)
    ap.add_argument("--no_velocity_input", dest="use_velocity_input", action="store_false")

    ap.add_argument("--Lx", type=float, default=9.0)
    ap.add_argument("--Ly", type=float, default=3.0)
    ap.add_argument("--nx", type=int, default=128 * 3)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--trunk_width", type=int, default=128)

    ap.add_argument("--mask_kind", type=str, default="soft_rect", choices=["rect", "gauss", "soft_rect", "soft-rect"])
    ap.add_argument("--src_size", type=float, default=0.20)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--soft_k", type=float, default=30.0)

    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min_delta", type=float, default=0.0)
    ap.add_argument("--save_last", action="store_true")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--save_curves", action="store_true")
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--summary_csv", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        print(f"[INFO] device=cuda | GPU={torch.cuda.get_device_name(0)} | VRAM={prop.total_memory/(1024**3):.1f} GB")
    else:
        print("[INFO] device=cpu")

    dataset = CasesFNOFieldDataset(
        workdir=Path(args.workdir),
        exports_dirname=args.exports,
        cases_csv=args.cases,
        Lx=args.Lx,
        Ly=args.Ly,
        nx=args.nx,
        ny=args.ny,
        cache_in_memory=args.cache,
        mask_kind=args.mask_kind,
        src_size=args.src_size,
        sigma=args.sigma,
        soft_k=args.soft_k,
        limit_samples=args.limit_samples,
        limit_strategy=args.limit_strategy,
        sample_seed=args.sample_seed,
        source_aspect_ratio=args.source_aspect_ratio,
        fixed_inlet_u=args.fixed_inlet_u,
        inlet_u_tol=args.inlet_u_tol,
        use_velocity_input=args.use_velocity_input,
    )
    dataset.split_meta = {
        "split_mode": "external_csv" if args.split_csv else str(args.split_mode),
        "split_csv": None if args.split_csv is None else str(Path(args.split_csv).resolve()),
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
    }
    if args.split_csv:
        tr_idx, va_idx, te_idx = split_indices_from_csv(dataset.rows, args.split_csv)
    else:
        tr_idx, va_idx, te_idx = split_indices(
            dataset.rows,
            split_mode=args.split_mode,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
        )

    train_ds = torch.utils.data.Subset(dataset, tr_idx)
    val_ds = torch.utils.data.Subset(dataset, va_idx)
    test_ds = torch.utils.data.Subset(dataset, te_idx)
    pin = device.type == "cuda"
    workers = int(args.workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=workers, pin_memory=pin, persistent_workers=(workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=workers, pin_memory=pin, persistent_workers=(workers > 0))
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=workers, pin_memory=pin, persistent_workers=(workers > 0))

    in_ch = 5 if args.use_velocity_input else 3
    model = build_model(args, in_ch=in_ch).to(device)
    n_params = count_parameters(model)
    print(f"[INFO] model={args.model_kind} | params={n_params:,}")
    print(f"[INFO] cases: n_total={len(dataset)}, n_train={len(train_ds)}, n_val={len(val_ds)}, n_test={len(test_ds)}")
    print(f"[INFO] split: mode={dataset.split_meta['split_mode']}, split_seed={args.split_seed}, train_seed={args.seed}")
    print(f"[INFO] input channels={in_ch} | mask={args.mask_kind}, soft_k={args.soft_k}, fixed_u={args.fixed_inlet_u}")

    if args.amp and device.type == "cuda" and not (is_power_of_two(args.nx) and is_power_of_two(args.ny)):
        print(f"[WARN] Disabling AMP because grid (ny={args.ny}, nx={args.nx}) is not power-of-two for this setup.")
        args.amp = False

    target_transform = TargetTransform(kind=args.target_transform, eps=args.target_transform_eps)
    x_norm, y_norm = make_normalizers(train_loader, device, target_transform)
    save_path = Path(args.save)
    meta = {
        "model_kind": str(args.model_kind),
        "in_ch": int(in_ch),
        "out_ch": 1,
        "width": int(args.width),
        "depth": int(args.depth),
        "rank": int(args.rank),
        "trunk_width": int(args.trunk_width),
        "n_params": int(n_params),
        "mask_kind": str(args.mask_kind).replace("-", "_"),
        "soft_k": float(args.soft_k),
        "amp": bool(args.amp),
        "fixed_inlet_u": float(args.fixed_inlet_u),
        "target_transform": str(args.target_transform),
        "target_transform_eps": float(args.target_transform_eps),
        "smoothness_weight": float(args.smoothness_weight),
        "gradient_weight": float(args.gradient_weight),
        "use_velocity_input": bool(args.use_velocity_input),
        "dataset_workdir": str(Path(args.workdir).resolve()),
        "dataset_exports_dir": str((Path(args.workdir) / args.exports).resolve()),
        "dataset_cases_path": str((Path(args.workdir) / args.cases).resolve()),
        "Lx": float(args.Lx),
        "Ly": float(args.Ly),
        "nx": int(args.nx),
        "ny": int(args.ny),
        "src_size": float(args.src_size),
        "sigma": float(args.sigma),
        "split_mode": dataset.split_meta["split_mode"],
        "split_csv": dataset.split_meta["split_csv"],
        "split_seed": int(args.split_seed),
        "split_val_ratio": float(args.val_ratio),
        "split_test_ratio": float(args.test_ratio),
    }

    best_val, best_epoch, history, meta = train(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        x_norm=x_norm,
        y_norm=y_norm,
        target_transform=target_transform,
        device=device,
        meta=meta,
    )
    ckpt_cpu = torch.load(save_path, map_location="cpu")
    ckpt_cpu.setdefault("meta", {}).update(meta)
    torch.save(ckpt_cpu, save_path)

    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    x_norm = Normalizer(ckpt["x_mean"].to(device), ckpt["x_std"].to(device))
    y_norm = Normalizer(ckpt["y_mean"].to(device), ckpt["y_std"].to(device))
    test_norm, test_raw = evaluate_metrics(model, test_loader, x_norm, y_norm, target_transform, device)

    hist_csv = save_path.with_name(f"history_{save_path.stem}.csv")
    save_history_csv(hist_csv, history)
    if args.save_curves:
        save_loss_curve(save_path.with_name(f"curve_{save_path.stem}.png"), history)

    out_csv = Path(args.out_csv) if args.out_csv else save_path.with_name(f"metrics_{save_path.stem}_test.csv")
    summary_csv = Path(args.summary_csv) if args.summary_csv else save_path.with_name(f"summary_{save_path.stem}_test.csv")
    df_metrics = evaluate_case_metrics(
        model=model,
        subset=test_ds,
        x_norm=x_norm,
        y_norm=y_norm,
        target_transform=target_transform,
        device=device,
        experiment_name=save_path.stem,
        ckpt_path=save_path,
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(out_csv, index=False)
    summary = pd.DataFrame(
        [
            {
                "experiment": save_path.stem,
                "model_kind": args.model_kind,
                "checkpoint": str(save_path),
                "n_params": int(n_params),
                "n_cases": int(len(df_metrics)),
                "best_epoch": int(best_epoch),
                "epochs_completed": int(meta["epochs_completed"]),
                "total_training_time_sec": float(meta["total_training_time_sec"]),
                "avg_epoch_time_sec": float(meta["avg_epoch_time_sec"]),
                "test_mse_norm": float(test_norm),
                "test_mse_raw": float(test_raw),
                "mean_rel_l2_c": float(df_metrics["rel_l2_c"].mean()),
                "std_rel_l2_c": float(df_metrics["rel_l2_c"].std(ddof=0)),
                "mean_rmse_c": float(df_metrics["rmse_c"].mean()),
                "mean_mae_c": float(df_metrics["mae_c"].mean()),
                "mean_high_rmse_c": float(df_metrics["high_rmse_c"].mean()),
                "mean_high_mae_c": float(df_metrics["high_mae_c"].mean()),
            }
        ]
    )
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)

    print(f"\n[RESULT] model={args.model_kind} best_val_norm={best_val:.6e} (epoch {best_epoch}) test_norm={test_norm:.6e} test_raw={test_raw:.6e}")
    print(f"[SAVED] checkpoint -> {save_path.resolve()}")
    print(f"[SAVED] history -> {hist_csv.resolve()}")
    print(f"[SAVED] metrics -> {out_csv.resolve()}")
    print(f"[SAVED] summary -> {summary_csv.resolve()}")


if __name__ == "__main__":
    main()
