import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import griddata
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.train_fixed_u_srcpos_to_c import (  # noqa: E402
    Normalizer,
    TargetTransform,
    build_source_mask_gaussian,
    build_source_mask_rect,
    build_source_mask_soft_rect_sigmoid,
    is_power_of_two,
    safe_float_tag,
    set_seed,
    split_indices,
    split_indices_from_csv,
)


def load_case_csv(case_path: Path) -> pd.DataFrame:
    df = pd.read_csv(case_path)
    required = ["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Passive Scalar", "X (m)", "Y (m)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {case_path}. Got: {list(df.columns)}")
    return df[required].copy()


def df_to_grid(
    df: pd.DataFrame,
    Lx: float = 9.0,
    Ly: float = 3.0,
    nx: int = 384,
    ny: int = 128,
):
    x = np.linspace(0.0, Lx, nx, dtype=np.float32)
    y = np.linspace(0.0, Ly, ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(x, y)

    pts = df[["X (m)", "Y (m)"]].to_numpy(dtype=np.float64)
    ux = df["Velocity[i] (m/s)"].to_numpy(dtype=np.float64)
    uy = df["Velocity[j] (m/s)"].to_numpy(dtype=np.float64)
    cc = df["Passive Scalar"].to_numpy(dtype=np.float64)

    def interp(vals: np.ndarray):
        lin = griddata(pts, vals, (Xg, Yg), method="linear")
        near = griddata(pts, vals, (Xg, Yg), method="nearest")
        return np.where(np.isnan(lin), near, lin).astype(np.float32)

    return interp(ux), interp(uy), interp(cc)


def finite_difference_grad(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    gx = torch.zeros_like(y)
    gy = torch.zeros_like(y)
    gx[..., :, 1:-1] = 0.5 * (y[..., :, 2:] - y[..., :, :-2])
    gy[..., 1:-1, :] = 0.5 * (y[..., 2:, :] - y[..., :-2, :])
    gx[..., :, 0] = y[..., :, 1] - y[..., :, 0]
    gx[..., :, -1] = y[..., :, -1] - y[..., :, -2]
    gy[..., 0, :] = y[..., 1, :] - y[..., 0, :]
    gy[..., -1, :] = y[..., -1, :] - y[..., -2, :]
    return gx, gy


def gradient_matching_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_gx, pred_gy = finite_difference_grad(pred)
    tgt_gx, tgt_gy = finite_difference_grad(target)
    return 0.5 * (torch.mean((pred_gx - tgt_gx) ** 2) + torch.mean((pred_gy - tgt_gy) ** 2))


def gradient_head_loss(pred_grad: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tgt_gx, tgt_gy = finite_difference_grad(target)
    target_grad = torch.cat([tgt_gx, tgt_gy], dim=1)
    return torch.mean((pred_grad - target_grad) ** 2)


def concentration_weight_map(
    target: torch.Tensor,
    alpha: float = 2.0,
    gamma: float = 0.5,
    cap: float = 5.0,
    quantile: float = 0.99,
) -> torch.Tensor:
    """Emphasize plume cells without allowing a few peaks to dominate the loss."""
    if float(alpha) <= 0.0:
        return torch.ones_like(target)
    flat = torch.clamp(target.detach(), min=0.0).flatten(start_dim=1)
    scale = torch.quantile(flat, q=float(quantile), dim=1, keepdim=True)
    scale = torch.clamp(scale, min=1e-12).view(-1, 1, 1, 1)
    relative = torch.clamp(torch.clamp(target.detach(), min=0.0) / scale, min=0.0, max=1.0)
    weights = 1.0 + float(alpha) * torch.pow(relative, float(gamma))
    return torch.clamp(weights, max=float(cap))


def weighted_field_mse(
    pred_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    target_raw: torch.Tensor,
    alpha: float,
    gamma: float,
    cap: float,
    quantile: float,
) -> torch.Tensor:
    weights = concentration_weight_map(target_raw, alpha=alpha, gamma=gamma, cap=cap, quantile=quantile)
    squared_error = (pred_normalized - target_normalized) ** 2
    return torch.sum(weights * squared_error) / torch.clamp(torch.sum(weights), min=1.0)


def integral_matching_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match the domain-integrated concentration; this is not a PDE flux balance."""
    pred_mean = pred.mean(dim=(-2, -1))
    target_mean = target.mean(dim=(-2, -1))
    scale = torch.clamp(target.abs().mean(dim=(-2, -1)), min=1e-8)
    return torch.mean(((pred_mean - target_mean) / scale) ** 2)


def nonnegative_concentration_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    negative = torch.relu(-pred)
    scale = torch.clamp(torch.mean(target**2, dim=(-2, -1), keepdim=True), min=1e-12)
    return torch.mean((negative**2) / scale)


def case_id_to_str(case_id) -> str:
    if pd.isna(case_id):
        raise ValueError("case_id cannot be NaN")
    try:
        value = float(case_id)
        if value.is_integer():
            return str(int(value))
    except (TypeError, ValueError):
        pass
    return str(case_id).strip()


def make_normalizers_streaming(
    train_loader: DataLoader,
    device: torch.device,
    target_transform: TargetTransform,
) -> Tuple[Normalizer, Normalizer]:
    x_sum = x_sq_sum = y_sum = y_sq_sum = None
    x_count = y_count = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = target_transform.forward(yb.to(device, non_blocking=True))

        x_dims = (0, 2, 3)
        y_dims = (0, 2, 3)
        xs = xb.sum(dim=x_dims, keepdim=True)
        x2s = (xb * xb).sum(dim=x_dims, keepdim=True)
        ys = yb.sum(dim=y_dims, keepdim=True)
        y2s = (yb * yb).sum(dim=y_dims, keepdim=True)

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


class ExtendedSFieldDataset(Dataset):
    def __init__(
        self,
        workdir: Path,
        exports_dirname: str = "exports",
        cases_csv: str = "cases.csv",
        Lx: float = 9.0,
        Ly: float = 3.0,
        nx: int = 384,
        ny: int = 128,
        cache_in_memory: bool = False,
        mask_kind: str = "soft_rect",
        src_size: float = 0.20,
        sigma: float = 0.12,
        soft_k: float = 80.0,
        fixed_inlet_u: Optional[float] = None,
        inlet_u_tol: float = 1e-8,
        limit_samples: Optional[int] = None,
        sample_seed: int = 42,
        use_velocity_input: bool = True,
        use_inlet_scalar: bool = True,
        use_grid_cache: bool = True,
        grid_cache_dir: Optional[Path] = None,
        max_u_ratio: Optional[float] = None,
        min_c_max: Optional[float] = None,
    ):
        self.workdir = Path(workdir)
        self.exports = self.workdir / exports_dirname
        self.cases_path = self.workdir / cases_csv
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.nx = int(nx)
        self.ny = int(ny)
        self.cache_in_memory = bool(cache_in_memory)
        self.use_velocity_input = bool(use_velocity_input)
        self.use_inlet_scalar = bool(use_inlet_scalar)
        self.use_grid_cache = bool(use_grid_cache)
        self.max_u_ratio = None if max_u_ratio is None else float(max_u_ratio)
        self.min_c_max = None if min_c_max is None else float(min_c_max)
        self.grid_cache_dir = None if grid_cache_dir is None else Path(grid_cache_dir)
        if self.use_grid_cache:
            if self.grid_cache_dir is None:
                self.grid_cache_dir = PROJECT_ROOT / "grid_cache" / f"nx{self.nx}_ny{self.ny}"
            self.grid_cache_dir.mkdir(parents=True, exist_ok=True)
        self.src_size = float(src_size)
        self.sigma = float(sigma)
        self.soft_k = float(soft_k)
        self.fixed_inlet_u = None if fixed_inlet_u is None else float(fixed_inlet_u)
        self.inlet_u_tol = float(inlet_u_tol)

        mk = str(mask_kind).lower().strip().replace("-", "_").replace(" ", "")
        if mk in {"softrect", "soft_rect"}:
            mk = "soft_rect"
        if mk == "gaussian":
            mk = "gauss"
        if mk not in {"rect", "gauss", "soft_rect"}:
            raise ValueError("mask_kind must be one of: rect, gauss, soft_rect")
        self.mask_kind = mk

        if not self.exports.exists():
            raise FileNotFoundError(f"exports folder not found: {self.exports}")
        if not self.cases_path.exists():
            raise FileNotFoundError(f"cases.csv not found: {self.cases_path}")

        rows = pd.read_csv(self.cases_path)
        required = ["case_id", "inlet_u", "src_x", "src_y"]
        missing = [c for c in required if c not in rows.columns]
        if missing:
            raise ValueError(f"cases.csv missing columns {missing}. Got: {list(rows.columns)}")

        rows = rows[required].copy()
        if self.fixed_inlet_u is not None:
            rows = rows[np.abs(rows["inlet_u"].astype(float) - self.fixed_inlet_u) <= self.inlet_u_tol].copy()
        if limit_samples is not None and int(limit_samples) < len(rows):
            rows = rows.sample(n=int(limit_samples), random_state=int(sample_seed)).copy()

        rows = rows.reset_index(drop=True)
        self.file_map: Dict[str, Path] = {}
        keep_rows = []
        missing_files = 0
        for _, row in rows.iterrows():
            cid = case_id_to_str(row["case_id"])
            inlet_u = float(row["inlet_u"])
            sx = float(row["src_x"])
            sy = float(row["src_y"])
            exact = self.exports / f"case_{cid}_u{safe_float_tag(inlet_u)}_sx{safe_float_tag(sx)}_sy{safe_float_tag(sy)}.csv"
            if exact.exists():
                self.file_map[cid] = exact
                keep_rows.append(row)
                continue
            hits = sorted(self.exports.glob(f"case_{cid}_*.csv"))
            if hits:
                self.file_map[cid] = hits[0]
                keep_rows.append(row)
            else:
                missing_files += 1

        self.rows = pd.DataFrame(keep_rows).reset_index(drop=True)
        self.rows["case_id"] = self.rows["case_id"].map(case_id_to_str)
        if len(self.rows) == 0:
            raise RuntimeError("No valid exported CSVs found for cases.csv rows.")
        if missing_files:
            print(f"[WARN] Missing {missing_files} exported CSVs. Using {len(self.rows)} cases.")

        xx = np.linspace(0.0, 1.0, self.nx, dtype=np.float32)
        yy = np.linspace(0.0, 1.0, self.ny, dtype=np.float32)
        Xc, Yc = np.meshgrid(xx, yy)
        self.Xc = Xc[None, ...]
        self.Yc = Yc[None, ...]

        x_phys = np.linspace(0.0, self.Lx, self.nx, dtype=np.float32)
        y_phys = np.linspace(0.0, self.Ly, self.ny, dtype=np.float32)
        self.Xg_phys, self.Yg_phys = np.meshgrid(x_phys, y_phys)
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.split_meta: Dict[str, object] = {}
        self.quality_filter_removed: List[dict] = []
        self._apply_quality_filter()

    @property
    def in_channels(self) -> int:
        return 3 + (2 if self.use_velocity_input else 0) + (1 if self.use_inlet_scalar else 0)

    def __len__(self):
        return len(self.rows)

    def _make_mask(self, sx: float, sy: float) -> np.ndarray:
        if self.mask_kind == "rect":
            mask = build_source_mask_rect(self.Xg_phys, self.Yg_phys, sx, sy, self.src_size)
        elif self.mask_kind == "gauss":
            mask = build_source_mask_gaussian(self.Xg_phys, self.Yg_phys, sx, sy, self.sigma)
        else:
            mask = build_source_mask_soft_rect_sigmoid(self.Xg_phys, self.Yg_phys, sx, sy, self.src_size, self.soft_k)
        return mask[None, ...].astype(np.float32)

    def _grid_cache_path(self, cid: str) -> Path:
        if self.grid_cache_dir is None:
            raise RuntimeError("grid_cache_dir is not configured.")
        return self.grid_cache_dir / f"case_{cid}_nx{self.nx}_ny{self.ny}.npz"

    def _load_grids(self, cid: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.use_grid_cache:
            cache_path = self._grid_cache_path(cid)
            if cache_path.exists():
                data = np.load(cache_path)
                return data["Ux"].astype(np.float32), data["Uy"].astype(np.float32), data["C"].astype(np.float32)

        Ux, Uy, C = df_to_grid(load_case_csv(self.file_map[cid]), Lx=self.Lx, Ly=self.Ly, nx=self.nx, ny=self.ny)
        if self.use_grid_cache:
            cache_path = self._grid_cache_path(cid)
            tmp_path = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp_path, Ux=Ux, Uy=Uy, C=C)
            tmp_path.replace(cache_path)
        return Ux, Uy, C

    def _apply_quality_filter(self):
        if self.max_u_ratio is None and self.min_c_max is None:
            return

        keep_rows = []
        removed = []
        for _, row in self.rows.iterrows():
            cid = case_id_to_str(row["case_id"])
            inlet_u = float(row["inlet_u"])
            Ux, Uy, C = self._load_grids(cid)
            max_abs_u = float(max(np.max(np.abs(Ux)), np.max(np.abs(Uy))))
            c_max = float(np.max(C))
            u_ratio = max_abs_u / max(abs(inlet_u), 1e-12)

            reasons = []
            if self.max_u_ratio is not None and u_ratio > self.max_u_ratio:
                reasons.append(f"u_ratio={u_ratio:.6g}>{self.max_u_ratio:.6g}")
            if self.min_c_max is not None and c_max < self.min_c_max:
                reasons.append(f"c_max={c_max:.6g}<{self.min_c_max:.6g}")

            if reasons:
                removed.append(
                    {
                        "case_id": cid,
                        "inlet_u": inlet_u,
                        "src_x": float(row["src_x"]),
                        "src_y": float(row["src_y"]),
                        "max_abs_u": max_abs_u,
                        "u_ratio": float(u_ratio),
                        "c_max": c_max,
                        "reason": ";".join(reasons),
                    }
                )
            else:
                keep_rows.append(row)

        self.quality_filter_removed = removed
        self.rows = pd.DataFrame(keep_rows).reset_index(drop=True)
        self.rows["case_id"] = self.rows["case_id"].map(case_id_to_str)
        print(
            f"[INFO] quality_filter: kept={len(self.rows)} removed={len(removed)} "
            f"max_u_ratio={self.max_u_ratio} min_c_max={self.min_c_max}"
        )
        if removed:
            print("[INFO] removed case_ids:", ",".join(str(r["case_id"]) for r in removed))

    def __getitem__(self, idx: int):
        idx = int(idx)
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        row = self.rows.iloc[idx]
        cid = case_id_to_str(row["case_id"])
        inlet_u = float(row["inlet_u"])
        sx = float(row["src_x"])
        sy = float(row["src_y"])

        Ux, Uy, C = self._load_grids(cid)
        channels = [self._make_mask(sx, sy), self.Xc, self.Yc]
        if self.use_velocity_input:
            channels.extend([Ux[None, ...], Uy[None, ...]])
        if self.use_inlet_scalar:
            channels.append(np.full((1, self.ny, self.nx), inlet_u, dtype=np.float32))

        x_t = torch.from_numpy(np.concatenate(channels, axis=0).astype(np.float32)).float()
        y_t = torch.from_numpy(C[None, ...].astype(np.float32)).float()
        out = (x_t, y_t)
        if self.cache_in_memory:
            self._cache[idx] = out
        return out


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = 1.0 / (self.in_channels * self.out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(self.in_channels, self.out_channels, self.modes1, self.modes2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, ny, nx = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        m1 = min(self.modes1, ny)
        m2 = min(self.modes2, x_ft.shape[-1])
        out_ft = torch.zeros(bsz, self.out_channels, ny, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        weight = torch.view_as_complex(self.weight)
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], weight[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(ny, nx), norm="ortho")


class FNOBackbone2D(nn.Module):
    def __init__(self, in_ch: int, width: int = 64, modes1: int = 24, modes2: int = 16, depth: int = 4):
        super().__init__()
        self.fc0 = nn.Conv2d(in_ch, width, kernel_size=1)
        self.spec_layers = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)])
        self.w_layers = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)])
        self.act = nn.GELU()
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.fc0(x)
        for spec, w in zip(self.spec_layers, self.w_layers):
            feat = self.act(spec(feat) + w(feat))
        hidden = self.act(self.fc1(feat))
        coarse = self.fc2(hidden)
        return coarse, hidden


class CNNGradientRefiner(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int = 64, layers: int = 4, residual_scale: float = 0.25):
        super().__init__()
        blocks: List[nn.Module] = []
        ch = in_ch
        for _ in range(max(1, int(layers))):
            blocks.extend([nn.Conv2d(ch, hidden_ch, kernel_size=3, padding=1), nn.GELU()])
            ch = hidden_ch
        self.trunk = nn.Sequential(*blocks)
        self.residual = nn.Conv2d(hidden_ch, 1, kernel_size=3, padding=1)
        self.grad_head = nn.Conv2d(hidden_ch, 2, kernel_size=3, padding=1)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(x)
        return self.residual_scale * self.residual(feat), self.grad_head(feat)


class FNOCNN2D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        width: int = 64,
        modes1: int = 24,
        modes2: int = 16,
        depth: int = 4,
        cnn_width: int = 64,
        cnn_layers: int = 4,
        residual_scale: float = 0.25,
    ):
        super().__init__()
        self.fno = FNOBackbone2D(in_ch=in_ch, width=width, modes1=modes1, modes2=modes2, depth=depth)
        self.cnn = CNNGradientRefiner(
            in_ch=in_ch + 1,
            hidden_ch=cnn_width,
            layers=cnn_layers,
            residual_scale=residual_scale,
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        coarse, _ = self.fno(x)
        residual, grad = self.cnn(torch.cat([x, coarse], dim=1))
        pred = coarse + residual
        if return_aux:
            return pred, {"coarse": coarse, "residual": residual, "grad": grad}
        return pred


@torch.no_grad()
def evaluate_metrics(model, loader, x_norm, y_norm, target_transform, device):
    model.eval()
    mse_norm = 0.0
    mse_raw = 0.0
    grad_raw = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        yb_n = y_norm.encode(target_transform.forward(yb))
        pred_n = model(x_norm.encode(xb))
        pred = target_transform.inverse(y_norm.decode(pred_n))
        mse_norm += torch.mean((pred_n - yb_n) ** 2).item() * xb.shape[0]
        mse_raw += torch.mean((pred - yb) ** 2).item() * xb.shape[0]
        grad_raw += gradient_matching_loss(pred, yb).item() * xb.shape[0]
        n += xb.shape[0]
    denom = max(1, n)
    return mse_norm / denom, mse_raw / denom, grad_raw / denom


def save_checkpoint(path: Path, model: nn.Module, x_norm: Normalizer, y_norm: Normalizer, best_val: float, meta: dict):
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


def train(
    model: FNOCNN2D,
    train_loader: DataLoader,
    val_loader: DataLoader,
    x_norm: Normalizer,
    y_norm: Normalizer,
    target_transform: TargetTransform,
    device: torch.device,
    args,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    base_ds = train_loader.dataset.dataset if hasattr(train_loader.dataset, "dataset") else train_loader.dataset
    meta = {
        "architecture": "FNOCNN2D",
        "architecture_note": "FNO predicts the global scalar field; CNN locally refines residuals and has a gradient head.",
        "in_ch": int(base_ds.in_channels),
        "out_ch": 1,
        "width": int(args.width),
        "modes1": int(args.modes1),
        "modes2": int(args.modes2),
        "depth": int(args.depth),
        "cnn_width": int(args.cnn_width),
        "cnn_layers": int(args.cnn_layers),
        "residual_scale": float(args.residual_scale),
        "dataset_workdir": str(base_ds.workdir.resolve()),
        "dataset_exports_dir": str(base_ds.exports.resolve()),
        "dataset_cases_path": str(base_ds.cases_path.resolve()),
        "dataset_cases": int(len(base_ds)),
        "fixed_inlet_u": base_ds.fixed_inlet_u,
        "use_velocity_input": bool(base_ds.use_velocity_input),
        "use_inlet_scalar": bool(base_ds.use_inlet_scalar),
        "input_channels": ["S_geometry_mask", "X", "Y"]
        + (["Ux", "Uy"] if base_ds.use_velocity_input else [])
        + (["inlet_u"] if base_ds.use_inlet_scalar else []),
        "use_grid_cache": bool(base_ds.use_grid_cache),
        "grid_cache_dir": None if base_ds.grid_cache_dir is None else str(base_ds.grid_cache_dir.resolve()),
        "quality_filter_max_u_ratio": base_ds.max_u_ratio,
        "quality_filter_min_c_max": base_ds.min_c_max,
        "quality_filter_removed": list(getattr(base_ds, "quality_filter_removed", [])),
        "target_transform": str(target_transform.kind),
        "target_transform_eps": float(target_transform.eps),
        "mask_kind": str(base_ds.mask_kind),
        "src_size": float(base_ds.src_size),
        "sigma": float(base_ds.sigma),
        "soft_k": float(base_ds.soft_k),
        "Lx": float(base_ds.Lx),
        "Ly": float(base_ds.Ly),
        "nx": int(base_ds.nx),
        "ny": int(base_ds.ny),
        "split": dict(base_ds.split_meta),
        "field_weight": float(args.field_weight),
        "concentration_weight_alpha": float(args.concentration_weight_alpha),
        "concentration_weight_gamma": float(args.concentration_weight_gamma),
        "concentration_weight_cap": float(args.concentration_weight_cap),
        "concentration_weight_quantile": float(args.concentration_weight_quantile),
        "gradient_weight": float(args.gradient_weight),
        "gradient_head_weight": float(args.gradient_head_weight),
        "coarse_aux_weight": float(args.coarse_aux_weight),
        "integral_weight": float(args.integral_weight),
        "nonnegative_weight": float(args.nonnegative_weight),
    }

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    started = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total = total_raw = total_grad = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                xbn = x_norm.encode(xb)
                target_t = target_transform.forward(yb)
                target_n = y_norm.encode(target_t)
                pred_n, aux = model(xbn, return_aux=True)
                loss_field = weighted_field_mse(
                    pred_n,
                    target_n,
                    yb,
                    alpha=args.concentration_weight_alpha,
                    gamma=args.concentration_weight_gamma,
                    cap=args.concentration_weight_cap,
                    quantile=args.concentration_weight_quantile,
                )
                loss_coarse = torch.mean((aux["coarse"] - target_n) ** 2)

                pred_raw = target_transform.inverse(y_norm.decode(pred_n))
                loss_grad = gradient_matching_loss(pred_raw, yb)
                loss_grad_head = gradient_head_loss(aux["grad"], yb)
                loss_integral = integral_matching_loss(pred_raw, yb)
                loss_nonnegative = nonnegative_concentration_loss(pred_raw, yb)
                loss = (
                    float(args.field_weight) * loss_field
                    + float(args.gradient_weight) * loss_grad
                    + float(args.gradient_head_weight) * loss_grad_head
                    + float(args.coarse_aux_weight) * loss_coarse
                    + float(args.integral_weight) * loss_integral
                    + float(args.nonnegative_weight) * loss_nonnegative
                )

            scaler.scale(loss).backward()
            if args.grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                total_raw += torch.mean((pred_raw - yb) ** 2).item() * xb.shape[0]
                total_grad += loss_grad.item() * xb.shape[0]
            total += loss.item() * xb.shape[0]
            n += xb.shape[0]

        sched.step()
        train_loss = total / max(1, n)
        train_raw = total_raw / max(1, n)
        train_grad = total_grad / max(1, n)
        val_norm, val_raw, val_grad = evaluate_metrics(model, val_loader, x_norm, y_norm, target_transform, device)

        improved = (best_val - val_norm) > float(args.min_delta)
        if improved:
            best_val = float(val_norm)
            best_epoch = int(ep)
            bad_epochs = 0
            meta.update(
                {
                    "epochs_completed": int(ep),
                    "best_epoch": int(best_epoch),
                    "total_training_time_sec": float(time.perf_counter() - started),
                }
            )
            save_checkpoint(Path(args.save), model, x_norm, y_norm, best_val, meta)
        else:
            bad_epochs += 1

        row = {
            "epoch": int(ep),
            "train_loss": float(train_loss),
            "train_mse_raw": float(train_raw),
            "train_gradient_raw": float(train_grad),
            "val_mse_norm": float(val_norm),
            "val_mse_raw": float(val_raw),
            "val_gradient_raw": float(val_grad),
            "best_val_mse": float(best_val),
            "lr": float(opt.param_groups[0]["lr"]),
            "bad_count": int(bad_epochs),
            "epoch_time_sec": float(time.perf_counter() - epoch_started),
            "cumulative_train_time_sec": float(time.perf_counter() - started),
        }
        history.append(row)

        if args.save_last:
            save_checkpoint(Path(args.save).with_name(Path(args.save).stem + "_last.pt"), model, x_norm, y_norm, best_val, meta)

        if ep == 1 or ep == args.epochs or ep % max(1, int(args.log_every)) == 0:
            print(
                f"[{ep:04d}/{args.epochs}] train={train_loss:.6e} train_raw={train_raw:.6e} "
                f"train_grad={train_grad:.6e} val_norm={val_norm:.6e} val_raw={val_raw:.6e} "
                f"val_grad={val_grad:.6e} best={best_val:.6e} best_ep={best_epoch} bad={bad_epochs}/{args.patience}"
            )

        if bad_epochs >= int(args.patience):
            print(f"[EARLY STOP] no improvement for {args.patience} epochs. Best epoch={best_epoch}, val_norm={best_val:.6e}")
            break

    hist_path = Path(args.save).with_name(f"history_{Path(args.save).stem}.csv")
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hist_path, index=False)
    return best_val, best_epoch, hist_path


def build_loaders(dataset: ExtendedSFieldDataset, args, device: torch.device):
    dataset.split_meta = {
        "split_mode": "external_csv" if args.split_csv else str(args.split_mode),
        "split_csv": None if args.split_csv is None else str(Path(args.split_csv).resolve()),
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

    split_tag = "external_csv" if args.split_csv else str(args.split_mode).lower().strip()
    split_path = Path(args.save).parent / f"test_set_list_{split_tag}.csv"
    assignment_path = Path(args.save).parent / f"split_assignments_{split_tag}.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"case_id": dataset.rows.iloc[te_idx]["case_id"].values}).to_csv(split_path, index=False)
    assignments = pd.DataFrame(
        {
            "case_id": dataset.rows["case_id"].values,
            "split": np.full(len(dataset.rows), "train", dtype=object),
        }
    )
    assignments.loc[va_idx, "split"] = "val"
    assignments.loc[te_idx, "split"] = "test"
    assignments.to_csv(assignment_path, index=False)

    pin = device.type == "cuda"
    kwargs = {
        "num_workers": int(args.workers),
        "pin_memory": pin,
        "persistent_workers": int(args.workers) > 0,
    }
    train_loader = DataLoader(torch.utils.data.Subset(dataset, tr_idx), batch_size=args.batch, shuffle=True, **kwargs)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, va_idx), batch_size=args.batch, shuffle=False, **kwargs)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, te_idx), batch_size=args.batch, shuffle=False, **kwargs)
    print(f"[INFO] cases: total={len(dataset)}, train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}")
    print(f"[INFO] saved test case list: {split_path.resolve()}")
    print(f"[INFO] saved split assignments: {assignment_path.resolve()}")
    return train_loader, val_loader, test_loader


def main():
    ap = argparse.ArgumentParser(description="Train FNO+CNN on data_v_S_new. FNO captures fields; CNN captures gradients.")
    ap.add_argument("--workdir", type=str, default=str(PROJECT_ROOT / "data" / "data_v_S_new"))
    ap.add_argument("--exports", type=str, default="exports")
    ap.add_argument("--cases", type=str, default="cases.csv")
    ap.add_argument(
        "--save",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "fno_cnn" / "fno_cnn_geometry_composite_loss_9900.pt"),
    )
    ap.add_argument("--fixed_inlet_u", type=float, default=None, help="Optional ablation: train one inlet velocity only. Default uses all available cases.")
    ap.add_argument("--inlet_u_tol", type=float, default=1e-8)
    ap.add_argument("--limit_samples", type=int, default=None)
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--split_mode", type=str, default="random", choices=["random", "spatial_checkerboard", "spatial_uniform", "spatial_left_right"])
    ap.add_argument("--split_csv", type=str, default=None)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)

    ap.add_argument("--Lx", type=float, default=9.0)
    ap.add_argument("--Ly", type=float, default=3.0)
    ap.add_argument("--nx", type=int, default=384)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--mask_kind", type=str, default="soft_rect", choices=["rect", "gauss", "soft_rect", "soft-rect"])
    ap.add_argument("--src_size", type=float, default=0.20)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--soft_k", type=float, default=80.0)
    ap.add_argument("--no_velocity_input", dest="use_velocity_input", action="store_false", default=True)
    ap.add_argument("--no_inlet_scalar", dest="use_inlet_scalar", action="store_false", default=True)
    ap.add_argument("--no_grid_cache", dest="use_grid_cache", action="store_false", default=True)
    ap.add_argument(
        "--grid_cache_dir",
        type=str,
        default=str(PROJECT_ROOT / "grid_cache" / "data_v_S_new_9900_nx384_ny128"),
    )
    ap.add_argument("--max_u_ratio", type=float, default=None, help="Drop cases with max_abs_velocity / inlet_u above this value.")
    ap.add_argument("--min_c_max", type=float, default=None, help="Drop cases with max scalar concentration below this value.")

    ap.add_argument("--modes1", type=int, default=24)
    ap.add_argument("--modes2", type=int, default=16)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--cnn_width", type=int, default=64)
    ap.add_argument("--cnn_layers", type=int, default=4)
    ap.add_argument("--residual_scale", type=float, default=0.25)

    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--field_weight", type=float, default=1.0)
    ap.add_argument("--concentration_weight_alpha", type=float, default=2.0)
    ap.add_argument("--concentration_weight_gamma", type=float, default=0.5)
    ap.add_argument("--concentration_weight_cap", type=float, default=5.0)
    ap.add_argument("--concentration_weight_quantile", type=float, default=0.99)
    ap.add_argument("--gradient_weight", type=float, default=0.05)
    ap.add_argument("--gradient_head_weight", type=float, default=0.01)
    ap.add_argument("--coarse_aux_weight", type=float, default=0.10)
    ap.add_argument("--integral_weight", type=float, default=0.01)
    ap.add_argument("--nonnegative_weight", type=float, default=0.001)
    ap.add_argument("--target_transform", type=str, default="raw", choices=["raw", "log10"])
    ap.add_argument("--target_transform_eps", type=float, default=1e-8)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--min_delta", type=float, default=0.0)
    ap.add_argument("--save_last", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", action="store_true", help="Cache gridded samples in RAM. The full 9900-case dataset can use a lot of memory.")
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
            raise RuntimeError("--device cuda requested, but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[INFO] device=cuda | GPU={props.name} | VRAM={props.total_memory / (1024 ** 3):.1f} GB")
    else:
        print("[INFO] device=cpu")

    if args.amp and device.type == "cuda" and not (is_power_of_two(args.nx) and is_power_of_two(args.ny)):
        print(f"[WARN] Disabling AMP because FFT grid ny={args.ny}, nx={args.nx} is not power-of-two compatible for half precision.")
        args.amp = False

    dataset = ExtendedSFieldDataset(
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
        fixed_inlet_u=args.fixed_inlet_u,
        inlet_u_tol=args.inlet_u_tol,
        limit_samples=args.limit_samples,
        sample_seed=args.sample_seed,
        use_velocity_input=args.use_velocity_input,
        use_inlet_scalar=args.use_inlet_scalar,
        use_grid_cache=args.use_grid_cache,
        grid_cache_dir=None if args.grid_cache_dir is None else Path(args.grid_cache_dir),
        max_u_ratio=args.max_u_ratio,
        min_c_max=args.min_c_max,
    )
    train_loader, val_loader, test_loader = build_loaders(dataset, args, device)

    print(
        f"[INFO] grid: ny={args.ny}, nx={args.nx} | in_ch={dataset.in_channels} "
        f"| channels=[S_geometry_mask,X,Y"
        f"{',Ux,Uy' if args.use_velocity_input else ''}{',inlet_u' if args.use_inlet_scalar else ''}]"
    )
    print(f"[INFO] grid_cache: enabled={dataset.use_grid_cache} | dir={dataset.grid_cache_dir}")
    print(
        f"[INFO] model: FNO width={args.width}, depth={args.depth}, modes=({args.modes1},{args.modes2}) "
        f"| CNN width={args.cnn_width}, layers={args.cnn_layers}, residual_scale={args.residual_scale}"
    )
    print(
        f"[INFO] losses: weighted_field={args.field_weight} "
        f"(alpha={args.concentration_weight_alpha}, gamma={args.concentration_weight_gamma}, "
        f"cap={args.concentration_weight_cap}, q={args.concentration_weight_quantile}), "
        f"gradient={args.gradient_weight}, gradient_head={args.gradient_head_weight}, "
        f"coarse_aux={args.coarse_aux_weight}, integral={args.integral_weight}, "
        f"nonnegative={args.nonnegative_weight}"
    )

    target_transform = TargetTransform(kind=args.target_transform, eps=float(args.target_transform_eps))
    x_norm, y_norm = make_normalizers_streaming(train_loader, device, target_transform)

    model = FNOCNN2D(
        in_ch=dataset.in_channels,
        width=args.width,
        modes1=args.modes1,
        modes2=args.modes2,
        depth=args.depth,
        cnn_width=args.cnn_width,
        cnn_layers=args.cnn_layers,
        residual_scale=args.residual_scale,
    )
    best_val, best_epoch, hist_path = train(model, train_loader, val_loader, x_norm, y_norm, target_transform, device, args)

    ckpt = torch.load(args.save, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    x_norm = Normalizer(ckpt["x_mean"].to(device), ckpt["x_std"].to(device))
    y_norm = Normalizer(ckpt["y_mean"].to(device), ckpt["y_std"].to(device))
    test_norm, test_raw, test_grad = evaluate_metrics(model, test_loader, x_norm, y_norm, target_transform, device)

    ckpt.setdefault("meta", {})
    ckpt["meta"].update({"test_mse_norm": float(test_norm), "test_mse_raw": float(test_raw), "test_gradient_raw": float(test_grad)})
    torch.save(ckpt, args.save)

    print(f"\n[RESULT] best_val_norm={best_val:.6e} (epoch {best_epoch}), test_norm={test_norm:.6e}, test_raw={test_raw:.6e}, test_grad={test_grad:.6e}")
    print(f"[SAVED] {Path(args.save).resolve()}")
    print(f"[SAVED] {hist_path.resolve()}")


if __name__ == "__main__":
    main()
