# train_fixed_u_srcpos_to_c.py
# Train FNO on a single inlet_u (fixed speed) across different source positions to predict scalar field C from:
#   - inlet velocity inlet_u
#   - source mask field S_mask(x,y)
#
# Input to FNO is 3 channels:
#   [S_mask_field, Xcoord(0..1), Ycoord(0..1)]
#
# Output is 1 channel:
#   [C]
#
# Example:
#   python train_fixed_u_srcpos_to_c.py --workdir .\data --exports exports --cases cases.csv --save checkpoints\fixed_u_srcpos_to_c.pt --device cuda --amp

import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from scipy.interpolate import griddata


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_power_of_two(n: int) -> bool:
    n = int(n)
    return n > 0 and (n & (n - 1)) == 0


def laplacian_smoothness_loss(y: torch.Tensor) -> torch.Tensor:
    """
    Penalize local oscillations via a discrete 5-point Laplacian energy.
    y: (B, C, H, W)
    """
    if y.shape[-2] < 3 or y.shape[-1] < 3:
        return torch.zeros((), device=y.device, dtype=y.dtype)
    center = y[..., 1:-1, 1:-1]
    lap = (
        -4.0 * center
        + y[..., :-2, 1:-1]
        + y[..., 2:, 1:-1]
        + y[..., 1:-1, :-2]
        + y[..., 1:-1, 2:]
    )
    return torch.mean(lap ** 2)


def gradient_matching_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Match first-order spatial gradients between prediction and target.
    pred, target: (B, C, H, W)
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")
    if pred.shape[-2] < 2 or pred.shape[-1] < 2:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    dx_pred = pred[..., :, 1:] - pred[..., :, :-1]
    dx_tgt = target[..., :, 1:] - target[..., :, :-1]
    dy_pred = pred[..., 1:, :] - pred[..., :-1, :]
    dy_tgt = target[..., 1:, :] - target[..., :-1, :]
    return 0.5 * (torch.mean((dx_pred - dx_tgt) ** 2) + torch.mean((dy_pred - dy_tgt) ** 2))


# -----------------------------
# Utils
# -----------------------------
def safe_float_tag(x: float) -> str:
    return f"{x:.4f}"


def split_indices_random(n: int, val_ratio=0.15, test_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test = idx[:n_test]
    val = idx[n_test:n_test + n_val]
    train = idx[n_test + n_val:]
    return train.tolist(), val.tolist(), test.tolist()


def split_indices_spatial_checkerboard(rows: pd.DataFrame):
    if not {"src_x", "src_y"}.issubset(rows.columns):
        raise ValueError("rows must contain src_x and src_y for spatial split.")

    x_vals = np.array(sorted(pd.unique(rows["src_x"].astype(float))), dtype=np.float64)
    y_vals = np.array(sorted(pd.unique(rows["src_y"].astype(float))), dtype=np.float64)
    x_lookup = {float(v): i for i, v in enumerate(x_vals.tolist())}
    y_lookup = {float(v): i for i, v in enumerate(y_vals.tolist())}

    train, val, test = [], [], []
    for idx, row in rows.reset_index(drop=True).iterrows():
        ix = x_lookup[float(row["src_x"])]
        iy = y_lookup[float(row["src_y"])]
        parity = (ix + iy) % 2
        y_band = iy % 4
        if parity == 0:
            train.append(int(idx))
        elif y_band in (0, 1):
            test.append(int(idx))
        else:
            val.append(int(idx))

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError(
            f"Spatial checkerboard split produced empty subset(s): "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )
    return train, val, test


def _uniform_indices(n_total: int, n_pick: int) -> np.ndarray:
    if n_pick >= n_total:
        return np.arange(n_total, dtype=int)
    idx = np.round(np.linspace(0, n_total - 1, n_pick)).astype(int)
    idx = np.unique(idx)
    if len(idx) == n_pick:
        return idx.astype(int)
    missing = [i for i in range(n_total) if i not in set(idx.tolist())]
    idx = np.concatenate([idx, np.array(missing[: max(0, n_pick - len(idx))], dtype=int)])
    idx.sort()
    return idx.astype(int)


def split_indices_spatial_uniform(
    rows: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    if not {"src_x", "src_y"}.issubset(rows.columns):
        raise ValueError("rows must contain src_x and src_y for spatial split.")

    rows = rows.reset_index(drop=True)
    x_vals = np.array(sorted(pd.unique(rows["src_x"].astype(float))), dtype=np.float64)
    y_vals = np.array(sorted(pd.unique(rows["src_y"].astype(float))), dtype=np.float64)
    nx_all, ny_all = len(x_vals), len(y_vals)
    n_total = len(rows)
    n_test_target = int(round(n_total * float(test_ratio)))
    n_val_target = int(round(n_total * float(val_ratio)))
    aspect = float(nx_all) / max(float(ny_all), 1.0)

    best = None
    for nx_pick in range(1, nx_all + 1):
        for ny_pick in range(1, ny_all + 1):
            n_pick = nx_pick * ny_pick
            score = (
                abs(n_pick - n_test_target),
                abs((float(nx_pick) / max(float(ny_pick), 1.0)) - aspect),
                abs(nx_pick - ny_pick),
            )
            if best is None or score < best[0]:
                best = (score, nx_pick, ny_pick)

    _, nx_pick, ny_pick = best
    x_sel = set(_uniform_indices(nx_all, nx_pick).tolist())
    y_sel = set(_uniform_indices(ny_all, ny_pick).tolist())
    x_lookup = {float(v): i for i, v in enumerate(x_vals.tolist())}
    y_lookup = {float(v): i for i, v in enumerate(y_vals.tolist())}

    test = []
    remaining = []
    for idx, row in rows.iterrows():
        ix = x_lookup[float(row["src_x"])]
        iy = y_lookup[float(row["src_y"])]
        if ix in x_sel and iy in y_sel:
            test.append(int(idx))
        else:
            remaining.append(int(idx))

    if len(test) == 0 or len(remaining) == 0:
        raise RuntimeError(
            f"Spatial uniform split produced empty subset(s): train+val={len(remaining)}, test={len(test)}"
        )

    rng = np.random.default_rng(seed)
    remaining = np.array(remaining, dtype=int)
    rng.shuffle(remaining)
    n_val = min(n_val_target, max(1, len(remaining) - 1))
    val = remaining[:n_val].tolist()
    train = remaining[n_val:].tolist()

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError(
            f"Spatial uniform split produced empty subset(s): "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )
    return train, val, test


def split_indices_spatial_left_right(
    rows: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    if not {"src_x", "src_y"}.issubset(rows.columns):
        raise ValueError("rows must contain src_x and src_y for spatial split.")

    rows = rows.reset_index(drop=True)
    x_vals = np.array(sorted(pd.unique(rows["src_x"].astype(float))), dtype=np.float64)
    n_x = len(x_vals)
    n_total = len(rows)
    n_test_target = int(round(n_total * float(test_ratio)))
    n_val_target = int(round(n_total * float(val_ratio)))

    best = None
    for n_test_x in range(1, n_x):
        n_test = n_test_x * rows["src_y"].nunique()
        score = abs(n_test - n_test_target)
        if best is None or score < best[0]:
            best = (score, n_test_x)

    _, n_test_x = best
    test_x_vals = set(x_vals[-n_test_x:].tolist())

    test = []
    remaining = []
    for idx, row in rows.iterrows():
        if float(row["src_x"]) in test_x_vals:
            test.append(int(idx))
        else:
            remaining.append(int(idx))

    if len(test) == 0 or len(remaining) == 0:
        raise RuntimeError(
            f"Spatial left-right split produced empty subset(s): train+val={len(remaining)}, test={len(test)}"
        )

    rng = np.random.default_rng(seed)
    remaining = np.array(remaining, dtype=int)
    rng.shuffle(remaining)
    n_val = min(n_val_target, max(1, len(remaining) - 1))
    val = remaining[:n_val].tolist()
    train = remaining[n_val:].tolist()

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError(
            f"Spatial left-right split produced empty subset(s): "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )
    return train, val, test


def split_indices(
    rows: pd.DataFrame,
    split_mode: str = "random",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    mode = str(split_mode).lower().strip()
    if mode == "random":
        return split_indices_random(len(rows), val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    if mode == "spatial_checkerboard":
        return split_indices_spatial_checkerboard(rows)
    if mode == "spatial_uniform":
        return split_indices_spatial_uniform(rows, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    if mode == "spatial_left_right":
        return split_indices_spatial_left_right(rows, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    raise ValueError("split_mode must be one of: random, spatial_checkerboard, spatial_uniform, spatial_left_right")


def _even_indices(n_total: int, n_pick: int) -> np.ndarray:
    if n_pick >= n_total:
        return np.arange(n_total, dtype=int)
    return np.floor(np.linspace(0, n_total, num=n_pick, endpoint=False)).astype(int)


def sample_cases_uniform_source(
    rows: pd.DataFrame,
    n_cases: int,
    seed: int = 42,
    target_aspect_ratio: float = 3.0,
) -> pd.DataFrame:
    """
    Select n_cases with approximately uniform source-position coverage while
    intentionally reducing the number of source positions.
    """
    if n_cases >= len(rows):
        return rows.copy()
    if n_cases <= 0:
        raise ValueError("limit_samples must be > 0.")

    rng = np.random.default_rng(seed)
    group_sizes = rows.groupby(["src_x", "src_y"]).size()
    max_per_src = int(group_sizes.max())
    n_src_target = int(np.ceil(n_cases / max_per_src))

    x_vals = np.array(sorted(rows["src_x"].unique().tolist()), dtype=np.float64)
    y_vals = np.array(sorted(rows["src_y"].unique().tolist()), dtype=np.float64)
    nx_all, ny_all = len(x_vals), len(y_vals)

    ratio = float(target_aspect_ratio)
    if ratio <= 0:
        raise ValueError("target_aspect_ratio must be > 0.")
    nx_pick = int(round(np.sqrt(n_src_target * ratio)))
    nx_pick = max(1, min(nx_pick, nx_all))
    ny_pick = int(np.ceil(n_src_target / nx_pick))
    ny_pick = max(1, min(ny_pick, ny_all))

    while (nx_pick * ny_pick) < n_src_target and (nx_pick < nx_all or ny_pick < ny_all):
        if nx_pick < nx_all and (nx_pick / nx_all) <= (ny_pick / ny_all):
            nx_pick += 1
        elif ny_pick < ny_all:
            ny_pick += 1
        else:
            nx_pick += 1

    x_sel = x_vals[_even_indices(nx_all, nx_pick)]
    y_sel = y_vals[_even_indices(ny_all, ny_pick)]
    grid_pos = [(float(x), float(y)) for x in x_sel for y in y_sel]

    if len(grid_pos) > n_src_target:
        pick_idx = _even_indices(len(grid_pos), n_src_target)
        grid_pos = [grid_pos[i] for i in pick_idx]

    pos_df = pd.DataFrame(grid_pos, columns=["src_x", "src_y"])
    pool = rows.merge(pos_df, on=["src_x", "src_y"], how="inner")

    if len(pool) < n_cases:
        remain = rows.merge(pool[["case_id"]], on="case_id", how="left", indicator=True)
        remain = remain[remain["_merge"] == "left_only"].drop(columns=["_merge"])
        need = n_cases - len(pool)
        extra = remain.sample(n=need, random_state=seed) if need > 0 else remain.iloc[:0].copy()
        pool = pd.concat([pool, extra], axis=0, ignore_index=True)

    groups = pool.groupby(["src_x", "src_y"]).indices
    keys = list(groups.keys())
    rng.shuffle(keys)
    shuffled = {k: rng.permutation(groups[k]).tolist() for k in keys}

    chosen = []
    while len(chosen) < n_cases:
        progressed = False
        for k in keys:
            if shuffled[k]:
                chosen.append(shuffled[k].pop())
                progressed = True
                if len(chosen) >= n_cases:
                    break
        if not progressed:
            break

    out = pool.iloc[chosen].copy().reset_index(drop=True)
    return out


# -----------------------------
# Grid interpolation
# -----------------------------
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
    nx: int = 128 * 3,
    ny: int = 128,
    x_col: str = "X (m)",
    y_col: str = "Y (m)",
    ux_col: str = "Velocity[i] (m/s)",
    uy_col: str = "Velocity[j] (m/s)",
    c_col: str = "Passive Scalar",
):
    x = np.linspace(0.0, Lx, nx, dtype=np.float32)
    y = np.linspace(0.0, Ly, ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(x, y)

    pts = df[[x_col, y_col]].to_numpy(dtype=np.float64)
    ux = df[ux_col].to_numpy(dtype=np.float64)
    uy = df[uy_col].to_numpy(dtype=np.float64)
    cc = df[c_col].to_numpy(dtype=np.float64)

    def interp(vals: np.ndarray):
        lin = griddata(pts, vals, (Xg, Yg), method="linear")
        near = griddata(pts, vals, (Xg, Yg), method="nearest")
        out = np.where(np.isnan(lin), near, lin).astype(np.float32)
        return out

    Ux = interp(ux)
    Uy = interp(uy)
    C = interp(cc)
    return x, y, Ux, Uy, C


# -----------------------------
# Source mask builder
# -----------------------------
def build_source_mask_rect(Xg: np.ndarray, Yg: np.ndarray, src_x: float, src_y: float, src_size: float) -> np.ndarray:
    hx = 0.5 * src_size
    hy = 0.5 * src_size
    mask = (
        (Xg >= (src_x - hx)) & (Xg <= (src_x + hx)) &
        (Yg >= (src_y - hy)) & (Yg <= (src_y + hy))
    ).astype(np.float32)
    return mask


def build_source_mask_gaussian(Xg: np.ndarray, Yg: np.ndarray, src_x: float, src_y: float, sigma: float) -> np.ndarray:
    r2 = (Xg - src_x) ** 2 + (Yg - src_y) ** 2
    mask = np.exp(-0.5 * r2 / (sigma ** 2)).astype(np.float32)
    return mask


def build_source_mask_soft_rect_sigmoid(
    Xg: np.ndarray,
    Yg: np.ndarray,
    src_x: float,
    src_y: float,
    src_size: float,
    soft_k: float = 80.0,
) -> np.ndarray:
    """
    Smooth rectangular mask using a sigmoid on the rectangle signed distance.
    Larger soft_k -> sharper boundary.
    """
    hx = 0.5 * src_size
    hy = 0.5 * src_size

    dx = np.abs(Xg - src_x) - hx
    dy = np.abs(Yg - src_y) - hy
    sdf = np.maximum(dx, dy)  # <=0 inside rectangle

    z = np.clip(soft_k * sdf, -60.0, 60.0)
    mask = (1.0 / (1.0 + np.exp(z))).astype(np.float32)
    return mask


# -----------------------------
# Normalizer
# -----------------------------
@dataclass
class Normalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-12)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + 1e-12) + self.mean


@dataclass
class TargetTransform:
    kind: str = "raw"
    eps: float = 1e-8

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.kind == "raw":
            return y
        if self.kind == "log10":
            return torch.log10(torch.clamp(y, min=0.0) + self.eps)
        raise ValueError(f"Unsupported target transform: {self.kind}")

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self.kind == "raw":
            return y
        if self.kind == "log10":
            return torch.clamp(torch.pow(10.0, y) - self.eps, min=0.0)
        raise ValueError(f"Unsupported target transform: {self.kind}")


def make_normalizers(train_loader: DataLoader, device: torch.device, target_transform: TargetTransform):
    # NOTE: This scans the whole training set once. With only ~300 cases it's OK.
    xs, ys = [], []
    for xb, yb in train_loader:
        xs.append(xb)
        ys.append(yb)

    x = torch.cat(xs, dim=0).to(device)
    y = torch.cat(ys, dim=0).to(device)
    y_t = target_transform.forward(y)

    x_mean = x.mean(dim=(0, 2, 3), keepdim=True)
    x_std  = x.std(dim=(0, 2, 3), keepdim=True)
    y_mean = y_t.mean(dim=(0, 2, 3), keepdim=True)
    y_std  = y_t.std(dim=(0, 2, 3), keepdim=True)

    x_std = torch.clamp(x_std, min=1e-6)
    y_std = torch.clamp(y_std, min=1e-6)

    return Normalizer(x_mean, x_std), Normalizer(y_mean, y_std)


@torch.no_grad()
def evaluate_metrics(
    model,
    loader,
    x_norm: Normalizer,
    y_norm: Normalizer,
    target_transform: TargetTransform,
    device: torch.device,
):
    model.eval()
    mse_norm = 0.0
    mse_raw = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        yb_t = target_transform.forward(yb)
        yb_n = y_norm.encode(yb_t)
        pred_n = model(x_norm.encode(xb))
        pred = target_transform.inverse(y_norm.decode(pred_n))

        mse_norm += torch.mean((pred_n - yb_n) ** 2).item() * xb.shape[0]
        mse_raw += torch.mean((pred - yb) ** 2).item() * xb.shape[0]
        n += xb.shape[0]

    denom = max(1, n)
    return mse_norm / denom, mse_raw / denom


# -----------------------------
# Dataset
# -----------------------------
class CasesFNOFieldDataset(Dataset):
    """
    Each sample:
      x_field: (3,ny,nx) = [S_mask_field, Xcoord(0..1), Ycoord(0..1)]
      y_field: (1,ny,nx) = [C]
    """

    def __init__(
        self,
        workdir: Path,
        exports_dirname: str,
        cases_csv: str,
        Lx: float = 9.0,
        Ly: float = 3.0,
        nx: int = 128 * 3,
        ny: int = 128,
        cache_in_memory: bool = True,
        mask_kind: str = "rect",
        src_size: float = 0.20,
        sigma: float = 0.12,
        soft_k: float = 80.0,
        limit_samples: Optional[int] = None,
        limit_strategy: str = "uniform_source",
        sample_seed: int = 42,
        source_aspect_ratio: float = 3.0,
        fixed_inlet_u: Optional[float] = None,
        inlet_u_tol: float = 1e-8,
    ):
        self.workdir = Path(workdir)
        self.exports = self.workdir / exports_dirname
        self.cases_path = (self.workdir / cases_csv) if not Path(cases_csv).is_absolute() else Path(cases_csv)

        if not self.exports.exists():
            raise FileNotFoundError(f"exports folder not found: {self.exports}")
        if not self.cases_path.exists():
            raise FileNotFoundError(f"cases.csv not found: {self.cases_path}")

        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.nx = int(nx)
        self.ny = int(ny)
        self.cache_in_memory = bool(cache_in_memory)

        mk = str(mask_kind).lower().strip().replace("-", "_").replace(" ", "")
        if mk in ["softrect", "soft_rect(sigmoid)", "softrect(sigmoid)", "soft_rect"]:
            mk = "soft_rect"
        if mk == "gaussian":
            mk = "gauss"

        self.mask_kind = mk
        if self.mask_kind not in ["rect", "gauss", "soft_rect"]:
            raise ValueError("mask_kind must be one of: 'rect', 'gauss', 'soft_rect'")
        self.src_size = float(src_size)
        self.sigma = float(sigma)
        self.soft_k = float(soft_k)

        df = pd.read_csv(self.cases_path)
        required = ["case_id", "inlet_u", "src_x", "src_y"]
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f"cases.csv missing columns {miss}. Got: {list(df.columns)}")

        rows = df[required].copy()

        self.fixed_inlet_u = None if fixed_inlet_u is None else float(fixed_inlet_u)
        self.inlet_u_tol = float(inlet_u_tol)
        if self.fixed_inlet_u is None:
            raise ValueError("fixed_inlet_u must be provided. This script trains on one inlet velocity only.")

        rows = rows[np.abs(rows["inlet_u"].astype(float) - self.fixed_inlet_u) <= self.inlet_u_tol].copy()
        rows = rows.reset_index(drop=True)
        if len(rows) == 0:
            raise RuntimeError(
                f"No cases found for fixed_inlet_u={self.fixed_inlet_u:.6f} within tolerance={self.inlet_u_tol:.3e}."
            )
        uniq_src_all = rows[["src_x", "src_y"]].drop_duplicates().shape[0]
        print(
            f"[INFO] fixed inlet_u={self.fixed_inlet_u:.6f} (tol={self.inlet_u_tol:.3e}) "
            f"=> {len(rows)} cases, unique sources={uniq_src_all}"
        )

        self.limit_strategy = str(limit_strategy).lower()
        if self.limit_strategy not in ["uniform_source", "head", "random"]:
            raise ValueError("limit_strategy must be one of: uniform_source, head, random")
        self.sample_seed = int(sample_seed)
        self.source_aspect_ratio = float(source_aspect_ratio)

        if limit_samples is not None:
            limit_n = int(limit_samples)
            if limit_n <= 0:
                raise ValueError("limit_samples must be > 0.")
            if limit_n < len(rows):
                if self.limit_strategy == "uniform_source":
                    rows = sample_cases_uniform_source(
                        rows,
                        n_cases=limit_n,
                        seed=self.sample_seed,
                        target_aspect_ratio=self.source_aspect_ratio,
                    )
                elif self.limit_strategy == "random":
                    rows = rows.sample(n=limit_n, random_state=self.sample_seed).copy()
                else:
                    rows = rows.head(limit_n).copy()
                uniq_src = rows[["src_x", "src_y"]].drop_duplicates().shape[0]
                print(
                    f"[INFO] limited to {len(rows)} cases using {self.limit_strategy} "
                    f"(unique sources={uniq_src}, source_aspect_ratio={self.source_aspect_ratio:.3f})."
                )

        self.rows = rows
        self.rows["case_id"] = self.rows["case_id"].astype(str)

        # Coord channels in [0,1]
        xx = np.linspace(0.0, 1.0, self.nx, dtype=np.float32)
        yy = np.linspace(0.0, 1.0, self.ny, dtype=np.float32)
        Xc, Yc = np.meshgrid(xx, yy)
        self.Xc = Xc[None, ...]
        self.Yc = Yc[None, ...]

        # Physical grid for masks
        x_phys = np.linspace(0.0, self.Lx, self.nx, dtype=np.float32)
        y_phys = np.linspace(0.0, self.Ly, self.ny, dtype=np.float32)
        self.Xg_phys, self.Yg_phys = np.meshgrid(x_phys, y_phys)

        # Build file map
        self.file_map: Dict[str, Path] = {}
        keep_rows = []
        missing = 0

        for _, r in self.rows.iterrows():
            cid = str(r["case_id"]).strip()
            inlet_u = float(r["inlet_u"])
            sx = float(r["src_x"])
            sy = float(r["src_y"])

            exact = self.exports / (
                f"case_{cid}_u{safe_float_tag(inlet_u)}_sx{safe_float_tag(sx)}_sy{safe_float_tag(sy)}.csv"
            )
            if exact.exists():
                self.file_map[cid] = exact
                keep_rows.append(r)
                continue

            hits = sorted(self.exports.glob(f"case_{cid}_*.csv"))
            if hits:
                self.file_map[cid] = hits[0]
                keep_rows.append(r)
            else:
                missing += 1

        self.rows = pd.DataFrame(keep_rows)
        if len(self.rows) == 0:
            raise RuntimeError("No valid cases found (no matching exported csv). Check exports + cases.csv.")
        if missing > 0:
            print(f"[WARN] Missing {missing} cases (no CSV found). Using {len(self.rows)} cases.")

        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.split_meta: Dict[str, object] = {}

    def __len__(self):
        return len(self.rows)

    def _make_mask(self, sx: float, sy: float) -> np.ndarray:
        if self.mask_kind == "rect":
            m = build_source_mask_rect(self.Xg_phys, self.Yg_phys, sx, sy, self.src_size)
        elif self.mask_kind == "gauss":
            m = build_source_mask_gaussian(self.Xg_phys, self.Yg_phys, sx, sy, self.sigma)
        else:
            m = build_source_mask_soft_rect_sigmoid(
                self.Xg_phys,
                self.Yg_phys,
                sx,
                sy,
                self.src_size,
                self.soft_k,
            )
        return m[None, ...]

    def _load_one(self, idx: int):
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        r = self.rows.iloc[idx]
        cid = str(r["case_id"]).strip()
        inlet_u = float(r["inlet_u"])
        sx = float(r["src_x"])
        sy = float(r["src_y"])

        case_path = self.file_map[cid]
        df_case = load_case_csv(case_path)
        _, _, Ux, Uy, C = df_to_grid(df_case, Lx=self.Lx, Ly=self.Ly, nx=self.nx, ny=self.ny)

        y_field = C[None, ...].astype(np.float32)

        S_mask = self._make_mask(sx, sy).astype(np.float32)
        x_field = np.concatenate([S_mask, self.Xc, self.Yc], axis=0).astype(np.float32)

        x_t = torch.from_numpy(x_field).float()
        y_t = torch.from_numpy(y_field).float()

        out = (x_t, y_t)
        if self.cache_in_memory:
            self._cache[idx] = out
        return out

    def __getitem__(self, idx: int):
        return self._load_one(int(idx))


# -----------------------------
# Minimal FNO2D
# -----------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, 2, dtype=torch.float32)
        )

    def compl_mul2d(self, input_fft, weight):
        return torch.einsum("bixy,ioxy->boxy", input_fft, weight)

    def forward(self, x):
        B, C, ny, nx = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        m1 = min(self.modes1, ny)
        m2 = min(self.modes2, x_ft.shape[-1])

        out_ft = torch.zeros(B, self.out_channels, ny, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        weight = torch.view_as_complex(self.weight)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            weight[:, :, :m1, :m2],
        )

        x = torch.fft.irfft2(out_ft, s=(ny, nx), norm="ortho")
        return x


class FNO2D(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, width=48, modes1=16, modes2=16, depth=4):
        super().__init__()
        self.fc0 = nn.Conv2d(in_ch, width, kernel_size=1)

        self.spec_layers = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)])
        self.w_layers = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)])

        self.act = nn.GELU()
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.fc0(x)
        for spec, w in zip(self.spec_layers, self.w_layers):
            x = self.act(spec(x) + w(x))
        x = self.act(self.fc1(x))
        return self.fc2(x)


# -----------------------------
# Save helpers
# -----------------------------
def save_checkpoint(save_path: Path, model: nn.Module, x_norm: Normalizer, y_norm: Normalizer,
                    best_val_mse: float, meta: dict):
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


def save_history_csv(path: Path, history: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)


def save_loss_curve(path: Path, history: List[dict]):
    if len(history) == 0:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print(f"[WARN] matplotlib not available, skip curve plot: {path}")
        return

    ep = [int(r["epoch"]) for r in history]
    tr = [float(r["train_loss"]) for r in history]
    va = [float(r["val_mse"]) for r in history]
    plt.figure(figsize=(8, 5))
    plt.plot(ep, tr, label="Training loss", linewidth=1.8)
    plt.plot(ep, va, label="Validation loss", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# -----------------------------
# Training loop (GPU + AMP + early stopping + save best)
# -----------------------------
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    x_norm: Normalizer,
    y_norm: Normalizer,
    target_transform: TargetTransform,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    smoothness_weight: float,
    gradient_weight: float,
    save_path: Path,
    patience: int = 50,
    min_delta: float = 0.0,
    save_last: bool = False,
    use_amp: bool = False,
    log_every: int = 1,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == "cuda"))

    best_val_norm = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    log_every = max(1, int(log_every))
    train_start_time = time.perf_counter()

    base_ds = train_loader.dataset.dataset if hasattr(train_loader.dataset, "dataset") else train_loader.dataset
    mask_kind = getattr(base_ds, "mask_kind", "rect")
    fixed_u = getattr(base_ds, "fixed_inlet_u", None)
    soft_k = getattr(base_ds, "soft_k", None)
    split_meta = getattr(base_ds, "split_meta", {})
    meta = {
        "in_ch": 3,
        "out_ch": 1,
        "width": int(model.fc0.out_channels),
        "modes1": int(model.spec_layers[0].modes1),
        "modes2": int(model.spec_layers[0].modes2),
        "depth": int(len(model.spec_layers)),
        "mask_kind": str(mask_kind),
        "soft_k": (None if soft_k is None else float(soft_k)),
        "amp": bool(use_amp),
        "fixed_inlet_u": (None if fixed_u is None else float(fixed_u)),
        "target_transform": str(target_transform.kind),
        "target_transform_eps": float(target_transform.eps),
        "smoothness_weight": float(smoothness_weight),
        "gradient_weight": float(gradient_weight),
        "dataset_workdir": str(Path(getattr(base_ds, "workdir", "")).resolve()) if getattr(base_ds, "workdir", None) is not None else None,
        "dataset_exports_dir": str(Path(getattr(base_ds, "exports", "")).resolve()) if getattr(base_ds, "exports", None) is not None else None,
        "dataset_cases_path": str(Path(getattr(base_ds, "cases_path", "")).resolve()) if getattr(base_ds, "cases_path", None) is not None else None,
        "Lx": float(getattr(base_ds, "Lx", 9.0)),
        "Ly": float(getattr(base_ds, "Ly", 3.0)),
        "nx": int(getattr(base_ds, "nx", 128 * 3)),
        "ny": int(getattr(base_ds, "ny", 128)),
        "src_size": float(getattr(base_ds, "src_size", 0.20)),
        "sigma": float(getattr(base_ds, "sigma", 0.12)),
        "split_mode": split_meta.get("split_mode", "random"),
        "split_seed": split_meta.get("split_seed", split_meta.get("seed", None)),
        "split_val_ratio": split_meta.get("val_ratio", None),
        "split_test_ratio": split_meta.get("test_ratio", None),
    }

    for ep in range(1, epochs + 1):
        epoch_start_time = time.perf_counter()
        model.train()
        total_norm = 0.0
        total_raw = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # AMP autocast only on CUDA
            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == "cuda")):
                target_t = target_transform.forward(yb)
                target_n = y_norm.encode(target_t)
                pred_n = model(x_norm.encode(xb))
                loss_mse = torch.mean((pred_n - target_n) ** 2)
                pred_raw_for_loss = target_transform.inverse(y_norm.decode(pred_n))
                loss_grad = (
                    float(gradient_weight) * gradient_matching_loss(pred_raw_for_loss, yb)
                    if float(gradient_weight) > 0.0
                    else torch.zeros((), device=pred_n.device, dtype=pred_n.dtype)
                )
                loss_smooth = (
                    float(smoothness_weight) * laplacian_smoothness_loss(pred_raw_for_loss)
                    if float(smoothness_weight) > 0.0
                    else torch.zeros((), device=pred_n.device, dtype=pred_n.dtype)
                )
                loss = loss_mse + loss_smooth + loss_grad

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                pred_raw = target_transform.inverse(y_norm.decode(pred_n.detach()))
                loss_raw = torch.mean((pred_raw - yb) ** 2)

            total_norm += loss.item() * xb.shape[0]
            total_raw += loss_raw.item() * xb.shape[0]
            n += xb.shape[0]

        sched.step()

        train_step_mse_norm = total_norm / max(1, n)
        train_step_mse_raw = total_raw / max(1, n)
        train_eval_mse_norm, train_eval_mse_raw = evaluate_metrics(
            model, train_loader, x_norm, y_norm, target_transform, device
        )
        val_mse_norm, val_mse_raw = evaluate_metrics(
            model, val_loader, x_norm, y_norm, target_transform, device
        )

        epoch_time_sec = float(time.perf_counter() - epoch_start_time)
        cumulative_train_time_sec = float(time.perf_counter() - train_start_time)
        avg_epoch_time_sec = cumulative_train_time_sec / float(ep)

        improved = (best_val_norm - val_mse_norm) > float(min_delta)
        if improved:
            best_val_norm = val_mse_norm
            best_epoch = ep
            bad_epochs = 0
            meta.update({
                "epochs_completed": int(ep),
                "best_epoch": int(best_epoch),
                "total_training_time_sec": float(cumulative_train_time_sec),
                "avg_epoch_time_sec": float(avg_epoch_time_sec),
            })
            save_checkpoint(save_path, model, x_norm, y_norm, best_val_norm, meta)
        else:
            bad_epochs += 1

        lr_now = float(opt.param_groups[0]["lr"])
        history.append({
            "epoch": int(ep),
            "train_loss": float(train_step_mse_norm),
            "val_mse": float(val_mse_norm),
            "best_val_mse": float(best_val_norm),
            "train_loss_norm": float(train_step_mse_norm),
            "val_mse_norm": float(val_mse_norm),
            "train_step_mse_norm": float(train_step_mse_norm),
            "train_eval_mse_norm": float(train_eval_mse_norm),
            "train_mse_raw": float(train_step_mse_raw),
            "train_step_mse_raw": float(train_step_mse_raw),
            "train_eval_mse_raw": float(train_eval_mse_raw),
            "val_mse_raw": float(val_mse_raw),
            "lr": lr_now,
            "bad_count": int(bad_epochs),
            "epoch_time_sec": float(epoch_time_sec),
            "cumulative_train_time_sec": float(cumulative_train_time_sec),
            "avg_epoch_time_sec": float(avg_epoch_time_sec),
        })

        if save_last:
            last_path = save_path.with_name(save_path.stem + "_last.pt")
            meta.update({
                "epochs_completed": int(ep),
                "best_epoch": int(best_epoch),
                "total_training_time_sec": float(cumulative_train_time_sec),
                "avg_epoch_time_sec": float(avg_epoch_time_sec),
            })
            save_checkpoint(last_path, model, x_norm, y_norm, best_val_norm, meta)

        if ep % log_every == 0 or ep == 1 or ep == epochs:
            print(
                f"[{ep:04d}/{epochs}] "
                f"train_step_norm={train_step_mse_norm:.6e} train_eval_norm={train_eval_mse_norm:.6e} val_norm={val_mse_norm:.6e} "
                f"train_step_raw={train_step_mse_raw:.6e} train_eval_raw={train_eval_mse_raw:.6e} val_raw={val_mse_raw:.6e} "
                f"best_val_norm={best_val_norm:.6e} best_ep={best_epoch} bad={bad_epochs}/{patience} "
                f"epoch_time={epoch_time_sec:.2f}s total_time={cumulative_train_time_sec/60.0:.2f}m"
            )

        if bad_epochs >= patience:
            print(f"[EARLY STOP] no improvement for {patience} epochs. Best at epoch {best_epoch} (val_norm={best_val_norm:.6e}).")
            break

    total_training_time_sec = float(time.perf_counter() - train_start_time)
    avg_epoch_time_sec = total_training_time_sec / float(len(history)) if len(history) > 0 else float("nan")
    meta.update({
        "epochs_completed": int(len(history)),
        "best_epoch": int(best_epoch),
        "total_training_time_sec": float(total_training_time_sec),
        "avg_epoch_time_sec": float(avg_epoch_time_sec),
    })
    return float(best_val_norm), int(best_epoch), history, {
        "total_training_time_sec": float(total_training_time_sec),
        "avg_epoch_time_sec": float(avg_epoch_time_sec),
        "epochs_completed": int(len(history)),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workdir",
        type=str,
        default=r"D:\Study\DL_Models\FNO\data_extended_S",
        help="Folder containing cases.csv and exports/",
    )
    ap.add_argument("--exports", type=str, default="exports", help="Exports subfolder name under workdir")
    ap.add_argument("--cases", type=str, default="cases.csv", help="cases.csv filename (relative to workdir)")
    ap.add_argument("--limit_samples", type=int, default=None)
    ap.add_argument(
        "--limit_strategy",
        type=str,
        default="uniform_source",
        choices=["uniform_source", "head", "random"],
        help="How to pick limit_samples cases. 'uniform_source' favors uniform source-position coverage.",
    )
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--split_seed", type=int, default=42, help="Seed for train/val/test split. Keep fixed across multi-seed runs.")
    ap.add_argument(
        "--split_mode",
        type=str,
        default="random",
        choices=["random", "spatial_checkerboard", "spatial_uniform", "spatial_left_right"],
        help="Dataset split strategy. spatial_left_right holds out the right-side source-position region for test.",
    )
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--fixed_inlet_u", type=float, required=True, help="Train only samples with this inlet_u")
    ap.add_argument("--inlet_u_tol", type=float, default=1e-8, help="Tolerance for matching inlet_u")
    ap.add_argument(
        "--source_aspect_ratio",
        type=float,
        default=3.0,
        help="Target x:y ratio for source-position coverage when limit_strategy=uniform_source.",
    )

    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--smoothness_weight", type=float, default=0.0)
    ap.add_argument("--gradient_weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", action="store_true", help="Cache samples in RAM (faster, more memory)")
    ap.add_argument("--save", type=str, default="checkpoints/fixed_u_srcpos_to_c.pt")
    ap.add_argument("--target_transform", type=str, default="raw", choices=["raw", "log10"])
    ap.add_argument("--target_transform_eps", type=float, default=1e-8)

    ap.add_argument("--Lx", type=float, default=9.0)
    ap.add_argument("--Ly", type=float, default=3.0)
    ap.add_argument("--nx", type=int, default=128 * 3)
    ap.add_argument("--ny", type=int, default=128)

    ap.add_argument("--modes1", type=int, default=16)
    ap.add_argument("--modes2", type=int, default=16)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--depth", type=int, default=4)

    # mask options
    ap.add_argument("--mask_kind", type=str, default="rect", choices=["rect", "gauss", "soft_rect", "soft-rect"])
    ap.add_argument("--src_size", type=float, default=0.20, help="rect/soft-rect mask size [m] (top-hat square)")
    ap.add_argument("--sigma", type=float, default=0.12, help="gaussian sigma [m] (if mask_kind=gauss)")
    ap.add_argument("--soft_k", type=float, default=80.0, help="soft-rect sigmoid sharpness (larger = sharper)")

    # early stopping
    ap.add_argument("--patience", type=int, default=30, help="early stop patience (epochs without improvement)")
    ap.add_argument("--min_delta", type=float, default=0.0, help="minimum val improvement to reset patience")
    ap.add_argument("--save_last", action="store_true", help="also save *_last.pt each epoch")

    # 閴?GPU controls
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="force device")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader num_workers (0 on Windows sometimes safer)")
    ap.add_argument("--amp", action="store_true", help="Use AMP mixed precision (CUDA only)")
    ap.add_argument("--log_every", type=int, default=1, help="Print training metrics every N epochs.")
    ap.add_argument("--save_curves", action="store_true", help="Save training curve as PNG.")

    args = ap.parse_args()

    set_seed(args.seed)

    # CUDA perf knobs
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("You asked for --device cuda, but torch.cuda.is_available() is False.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[INFO] device=cuda | GPU={name} | capability={cap} | VRAM={mem_gb:.1f} GB")
    else:
        print("[INFO] device=cpu")

    workdir = Path(args.workdir)
    dataset = CasesFNOFieldDataset(
        workdir=workdir,
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
    )

    dataset.split_meta = {
        "split_mode": str(args.split_mode),
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
    }
    tr_idx, va_idx, te_idx = split_indices(
        dataset.rows,
        split_mode=args.split_mode,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    )

    # View and save dataset information
    train_ids = dataset.rows.iloc[tr_idx]["case_id"].values
    val_ids = dataset.rows.iloc[va_idx]["case_id"].values
    test_ids = dataset.rows.iloc[te_idx]["case_id"].values
    print(f"Test Set Case IDs: {test_ids}")
    split_tag = str(args.split_mode).lower().strip()
    pd.DataFrame({"case_id": test_ids}).to_csv(f"test_set_list_{split_tag}.csv", index=False)

    train_ds = torch.utils.data.Subset(dataset, tr_idx)
    val_ds = torch.utils.data.Subset(dataset, va_idx)
    test_ds = torch.utils.data.Subset(dataset, te_idx)

    # DataLoader GPU-friendly settings
    pin = (device.type == "cuda")
    workers = int(args.workers)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=workers, pin_memory=pin,
        persistent_workers=(workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=workers, pin_memory=pin,
        persistent_workers=(workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False,
        num_workers=workers, pin_memory=pin,
        persistent_workers=(workers > 0),
    )

    print(f"[INFO] cases: n_total={len(dataset)}, n_train={len(train_ds)}, n_val={len(val_ds)}, n_test={len(test_ds)}")
    print(
        f"[INFO] split: mode={args.split_mode}, split_seed={args.split_seed}, train_seed={args.seed}, "
        f"val_ratio={args.val_ratio}, test_ratio={args.test_ratio}"
    )
    print(f"[INFO] grid: ny={args.ny}, nx={args.nx}")
    print("[INFO] input channels = 3: [S_mask, X, Y] | output channels = 1: [C]")
    print(f"[INFO] mask_kind={args.mask_kind} | src_size={args.src_size} m | sigma={args.sigma} m | soft_k={args.soft_k}")
    print(f"[INFO] target_transform={args.target_transform} | eps={args.target_transform_eps}")
    print(f"[INFO] smoothness_weight={args.smoothness_weight}")
    print(f"[INFO] gradient_weight={args.gradient_weight}")
    print(f"[INFO] early_stop: patience={args.patience}, min_delta={args.min_delta}")
    print(f"[INFO] fixed_inlet_u={args.fixed_inlet_u:.6f} | inlet_u_tol={args.inlet_u_tol:.3e}")

    use_amp = bool(args.amp and device.type == "cuda")
    if use_amp and not (is_power_of_two(args.nx) and is_power_of_two(args.ny)):
        print(
            f"[WARN] Disabling AMP because FFT grid size (ny={args.ny}, nx={args.nx}) "
            "is not power-of-two compatible for half precision cuFFT on this setup."
        )
        use_amp = False
    print(f"[INFO] dataloader: workers={workers}, pin_memory={pin}, amp={use_amp}")

    model = FNO2D(
        in_ch=3,
        out_ch=1,
        width=args.width,
        modes1=args.modes1,
        modes2=args.modes2,
        depth=args.depth,
    )

    target_transform = TargetTransform(kind=str(args.target_transform), eps=float(args.target_transform_eps))

    # Normalizers (computed on chosen device)
    x_norm, y_norm = make_normalizers(train_loader, device, target_transform)

    save_path = Path(args.save)
    best_val_norm, best_epoch, history, timing_info = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        x_norm=x_norm,
        y_norm=y_norm,
        target_transform=target_transform,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        smoothness_weight=args.smoothness_weight,
        gradient_weight=args.gradient_weight,
        save_path=save_path,
        patience=args.patience,
        min_delta=args.min_delta,
        save_last=args.save_last,
        use_amp=use_amp,
        log_every=args.log_every,
    )

    # Update best checkpoint meta with final timing summary
    best_ckpt_cpu = torch.load(save_path, map_location="cpu")
    best_ckpt_cpu.setdefault("meta", {})
    best_ckpt_cpu["meta"].update({
        "epochs_completed": int(timing_info["epochs_completed"]),
        "best_epoch": int(best_epoch),
        "total_training_time_sec": float(timing_info["total_training_time_sec"]),
        "avg_epoch_time_sec": float(timing_info["avg_epoch_time_sec"]),
    })
    torch.save(best_ckpt_cpu, save_path)

    # Load best + test
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    x_norm = Normalizer(ckpt["x_mean"].to(device), ckpt["x_std"].to(device))
    y_norm = Normalizer(ckpt["y_mean"].to(device), ckpt["y_std"].to(device))
    target_transform = TargetTransform(
        kind=str(ckpt.get("meta", {}).get("target_transform", "raw")),
        eps=float(ckpt.get("meta", {}).get("target_transform_eps", args.target_transform_eps)),
    )

    test_mse_norm, test_mse_raw = evaluate_metrics(model, test_loader, x_norm, y_norm, target_transform, device)
    hist_csv = save_path.with_name(f"history_{save_path.stem}.csv")
    save_history_csv(hist_csv, history)
    if args.save_curves:
        curve_png = save_path.with_name(f"curve_{save_path.stem}.png")
        save_loss_curve(curve_png, history)

    print(f"\\n[RESULT] best_val_norm={best_val_norm:.6e} (epoch {best_epoch}), test_norm={test_mse_norm:.6e}, test_raw={test_mse_raw:.6e}")
    print(f"[SAVED] {save_path.resolve()}")
    print(f"[SAVED] {hist_csv.resolve()}")
    if args.save_curves:
        print(f"[SAVED] {curve_png.resolve()}")


if __name__ == "__main__":
    main()

