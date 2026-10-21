import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import griddata


# =========================================================
# Utilities
# =========================================================
def safe_float_tag(x: float) -> str:
    return f"{x:.4f}"


def resolve_case_file(exports_dir: Path, case_id: str, inlet_u: float, sx: float, sy: float) -> Path:
    exact = exports_dir / (
        f"case_{case_id}_u{safe_float_tag(inlet_u)}_sx{safe_float_tag(sx)}_sy{safe_float_tag(sy)}.csv"
    )
    if exact.exists():
        return exact

    hits = sorted(exports_dir.glob(f"case_{case_id}_*.csv"))
    if hits:
        return hits[0]

    raise FileNotFoundError(f"No exported CSV found for case_id={case_id} under {exports_dir}")


def load_case_csv(case_path: Path) -> pd.DataFrame:
    df = pd.read_csv(case_path)
    required = ["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Passive Scalar", "X (m)", "Y (m)"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns {miss} in {case_path}. Got: {list(df.columns)}")
    return df[required].copy()


def df_to_grid(df: pd.DataFrame, Lx=9.0, Ly=3.0, nx=128 * 3, ny=128):
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

    Ux = interp(ux)
    Uy = interp(uy)
    C = interp(cc)
    return x, y, Ux, Uy, C


def build_source_mask_rect(Xg: np.ndarray, Yg: np.ndarray, src_x: float, src_y: float, src_size: float) -> np.ndarray:
    hx = 0.5 * src_size
    hy = 0.5 * src_size
    return (
        (Xg >= (src_x - hx)) & (Xg <= (src_x + hx)) &
        (Yg >= (src_y - hy)) & (Yg <= (src_y + hy))
    ).astype(np.float32)


def build_source_mask_gaussian(Xg: np.ndarray, Yg: np.ndarray, src_x: float, src_y: float, sigma: float) -> np.ndarray:
    r2 = (Xg - src_x) ** 2 + (Yg - src_y) ** 2
    return np.exp(-0.5 * r2 / (sigma ** 2)).astype(np.float32)


def build_source_mask_soft_rect_sigmoid(
    Xg: np.ndarray,
    Yg: np.ndarray,
    src_x: float,
    src_y: float,
    src_size: float,
    soft_k: float = 80.0,
) -> np.ndarray:
    hx = 0.5 * src_size
    hy = 0.5 * src_size
    dx = np.abs(Xg - src_x) - hx
    dy = np.abs(Yg - src_y) - hy
    sdf = np.maximum(dx, dy)
    z = np.clip(soft_k * sdf, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(z))).astype(np.float32)


def build_input_field(
    sx: float,
    sy: float,
    ny: int,
    nx: int,
    Lx: float,
    Ly: float,
    mask_kind: str,
    src_size: float,
    sigma: float,
    soft_k: float,
    ux: np.ndarray = None,
    uy: np.ndarray = None,
) -> np.ndarray:
    xx01 = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    yy01 = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    Xc, Yc = np.meshgrid(xx01, yy01)
    Xc = Xc[None, ...]
    Yc = Yc[None, ...]

    x_phys = np.linspace(0.0, Lx, nx, dtype=np.float32)
    y_phys = np.linspace(0.0, Ly, ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(x_phys, y_phys)

    mk = str(mask_kind).lower().strip().replace("-", "_")
    if mk == "gaussian":
        mk = "gauss"
    if mk in ["softrect", "soft_rect(sigmoid)", "softrect(sigmoid)"]:
        mk = "soft_rect"

    if mk == "rect":
        s = build_source_mask_rect(Xg, Yg, sx, sy, src_size)
    elif mk == "gauss":
        s = build_source_mask_gaussian(Xg, Yg, sx, sy, sigma)
    elif mk == "soft_rect":
        s = build_source_mask_soft_rect_sigmoid(Xg, Yg, sx, sy, src_size, soft_k)
    else:
        raise ValueError(f"Unsupported mask_kind: {mask_kind}")
    s = s[None, ...]

    channels = [s, Xc, Yc]
    if ux is not None or uy is not None:
        if ux is None or uy is None:
            raise ValueError("ux and uy must be provided together.")
        channels.extend([ux[None, ...].astype(np.float32), uy[None, ...].astype(np.float32)])
    x_field = np.concatenate(channels, axis=0)
    return x_field[None, ...]


# =========================================================
# Model (must match training)
# =========================================================
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
        bsz, _, ny, nx = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        m1 = min(self.modes1, ny)
        m2 = min(self.modes2, x_ft.shape[-1])

        out_ft = torch.zeros(bsz, self.out_channels, ny, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        weight = torch.view_as_complex(self.weight)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], weight[:, :, :m1, :m2])
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


# =========================================================
# Sensor sampling / inverse search
# =========================================================
def make_sensor_indices(
    ny: int,
    nx: int,
    n_sensors: int,
    strategy: str,
    seed: int,
    margin_x: int = 8,
    margin_y: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    mx = max(0, min(int(margin_x), max(0, (nx - 1) // 2 - 1)))
    my = max(0, min(int(margin_y), max(0, (ny - 1) // 2 - 1)))
    x0, x1 = mx, max(mx, nx - 1 - mx)
    y0, y1 = my, max(my, ny - 1 - my)

    xs_all = np.arange(x0, x1 + 1, dtype=int)
    ys_all = np.arange(y0, y1 + 1, dtype=int)

    if strategy == "downstream":
        x_mid = int(np.ceil((x0 + x1) / 2.0))
        xs_all = np.arange(x_mid, x1 + 1, dtype=int)

    coords_all = np.array([(yy, xx) for yy in ys_all for xx in xs_all], dtype=int)
    n_all = len(coords_all)
    n_sensors = max(1, min(int(n_sensors), n_all))

    if strategy == "random":
        pick = rng.choice(n_all, size=n_sensors, replace=False)
        coords = coords_all[pick]
        iy = coords[:, 0]
        ix = coords[:, 1]
        return iy.astype(int), ix.astype(int)

    # grid-like strategies over the candidate interior window
    gy = max(1, int(np.sqrt(n_sensors * len(ys_all) / max(1, len(xs_all)))))
    gx = max(1, int(np.ceil(n_sensors / gy)))
    y_lo, y_hi = int(ys_all[0]), int(ys_all[-1])
    x_lo, x_hi = int(xs_all[0]), int(xs_all[-1])

    if strategy == "grid_centered" and int(np.sqrt(n_sensors)) ** 2 == n_sensors:
        gy = int(np.sqrt(n_sensors))
        gx = int(np.sqrt(n_sensors))

    if gy == 1:
        ys = np.array([(y_lo + y_hi) // 2], dtype=int)
    else:
        if strategy == "grid_centered":
            ys = np.linspace(y_lo, y_hi, gy + 2, dtype=int)[1:-1]
        else:
            ys = np.linspace(y_lo, y_hi, gy, dtype=int)
    if gx == 1:
        xs = np.array([(x_lo + x_hi) // 2], dtype=int)
    else:
        if strategy == "grid_centered":
            xs = np.linspace(x_lo, x_hi, gx + 2, dtype=int)[1:-1]
        else:
            xs = np.linspace(x_lo, x_hi, gx, dtype=int)

    coords = [(yy, xx) for yy in ys for xx in xs]
    coords = coords[:n_sensors]
    iy = np.array([c[0] for c in coords], dtype=int)
    ix = np.array([c[1] for c in coords], dtype=int)
    return iy, ix


def predict_c_field(model, x_stats, y_stats, device, x_field: np.ndarray) -> np.ndarray:
    x_mean, x_std = x_stats
    y_mean, y_std = y_stats

    x_t = torch.from_numpy(x_field).to(device)
    x_t_n = (x_t - x_mean) / (x_std + 1e-12)

    with torch.no_grad():
        y_pred_n = model(x_t_n)
    y_pred = y_pred_n * (y_std + 1e-12) + y_mean

    c = y_pred.cpu().numpy()[0, 0]
    return c




def _build_input_field_torch(
    sx: torch.Tensor,
    sy: torch.Tensor,
    ny: int,
    nx: int,
    Lx: float,
    Ly: float,
    mask_kind: str,
    src_size: float,
    sigma: float,
    soft_k: float,
    device: torch.device,
    dtype: torch.dtype,
    ux: torch.Tensor = None,
    uy: torch.Tensor = None,
) -> torch.Tensor:
    xx01 = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)
    yy01 = torch.linspace(0.0, 1.0, ny, device=device, dtype=dtype)
    Yc, Xc = torch.meshgrid(yy01, xx01, indexing='ij')
    Xc = Xc.unsqueeze(0).unsqueeze(0)
    Yc = Yc.unsqueeze(0).unsqueeze(0)

    x_phys = torch.linspace(0.0, float(Lx), nx, device=device, dtype=dtype)
    y_phys = torch.linspace(0.0, float(Ly), ny, device=device, dtype=dtype)
    Yg, Xg = torch.meshgrid(y_phys, x_phys, indexing='ij')

    mk = str(mask_kind).lower()
    if mk == 'gauss':
        r2 = (Xg - sx) ** 2 + (Yg - sy) ** 2
        S = torch.exp(-0.5 * r2 / (float(sigma) ** 2))
    else:
        # Differentiable soft-rect (also used when mask_kind='rect')
        hx = 0.5 * float(src_size)
        hy = 0.5 * float(src_size)
        dx = torch.abs(Xg - sx) - hx
        dy = torch.abs(Yg - sy) - hy
        sdf = torch.maximum(dx, dy)
        S = torch.sigmoid(-float(soft_k) * sdf)

    S = S.unsqueeze(0).unsqueeze(0)
    channels = [S, Xc, Yc]
    if ux is not None or uy is not None:
        if ux is None or uy is None:
            raise ValueError("ux and uy must be provided together.")
        channels.extend([ux, uy])
    return torch.cat(channels, dim=1)


def _logit(p: torch.Tensor) -> torch.Tensor:
    p = torch.clamp(p, 1e-5, 1.0 - 1e-5)
    return torch.log(p) - torch.log(1.0 - p)


def _optimize_source_autodiff(
    model,
    x_mean,
    x_std,
    y_mean,
    y_std,
    device,
    iy: np.ndarray,
    ix: np.ndarray,
    obs: np.ndarray,
    ny: int,
    nx: int,
    Lx: float,
    Ly: float,
    mask_kind: str,
    src_size: float,
    sigma: float,
    soft_k: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    steps: int,
    lr: float,
    n_restarts: int,
    restart_seed: int = 42,
    ux_field: np.ndarray = None,
    uy_field: np.ndarray = None,
):
    iy_t = torch.as_tensor(iy, device=device, dtype=torch.long)
    ix_t = torch.as_tensor(ix, device=device, dtype=torch.long)
    obs_t = torch.as_tensor(obs, device=device, dtype=x_mean.dtype)
    ux_t = None if ux_field is None else torch.as_tensor(ux_field, device=device, dtype=x_mean.dtype).unsqueeze(0).unsqueeze(0)
    uy_t = None if uy_field is None else torch.as_tensor(uy_field, device=device, dtype=x_mean.dtype).unsqueeze(0).unsqueeze(0)

    best = {'loss': float('inf'), 'sx': None, 'sy': None, 'restart': -1, 'trace': []}
    x_rng = float(x_max - x_min)
    y_rng = float(y_max - y_min)
    rng = np.random.default_rng(int(restart_seed))

    for r in range(max(1, int(n_restarts))):
        if r == 0:
            sx0 = 0.5 * (x_min + x_max)
            sy0 = 0.5 * (y_min + y_max)
        else:
            sx0 = rng.uniform(x_min, x_max)
            sy0 = rng.uniform(y_min, y_max)

        px0 = torch.tensor((sx0 - x_min) / x_rng, device=device, dtype=x_mean.dtype)
        py0 = torch.tensor((sy0 - y_min) / y_rng, device=device, dtype=x_mean.dtype)
        zx = nn.Parameter(_logit(px0))
        zy = nn.Parameter(_logit(py0))

        opt = torch.optim.Adam([zx, zy], lr=float(lr))

        local_trace = []
        local_best_loss = float('inf')
        local_best_xy = (sx0, sy0)

        for t in range(max(1, int(steps))):
            opt.zero_grad(set_to_none=True)

            sx = x_min + x_rng * torch.sigmoid(zx)
            sy = y_min + y_rng * torch.sigmoid(zy)

            x_in = _build_input_field_torch(
                sx=sx,
                sy=sy,
                ny=ny,
                nx=nx,
                Lx=Lx,
                Ly=Ly,
                mask_kind=mask_kind,
                src_size=src_size,
                sigma=sigma,
                soft_k=soft_k,
                device=device,
                dtype=x_mean.dtype,
                ux=ux_t,
                uy=uy_t,
            )

            x_n = (x_in - x_mean) / (x_std + 1e-12)
            y_n = model(x_n)
            y = y_n * (y_std + 1e-12) + y_mean
            pred_obs = y[0, 0, iy_t, ix_t]
            loss = torch.mean((pred_obs - obs_t) ** 2)
            loss.backward()
            opt.step()

            l = float(loss.detach().cpu())
            sx_f = float(sx.detach().cpu())
            sy_f = float(sy.detach().cpu())
            local_trace.append((r, t, sx_f, sy_f, l))

            if l < local_best_loss:
                local_best_loss = l
                local_best_xy = (sx_f, sy_f)

        if local_best_loss < best['loss']:
            best['loss'] = local_best_loss
            best['sx'], best['sy'] = local_best_xy
            best['restart'] = r
            best['trace'] = local_trace

    return best


def _optimize_source_autodiff_from_initializations(
    model,
    x_mean,
    x_std,
    y_mean,
    y_std,
    device,
    iy: np.ndarray,
    ix: np.ndarray,
    obs: np.ndarray,
    initial_xy: np.ndarray,
    ny: int,
    nx: int,
    Lx: float,
    Ly: float,
    mask_kind: str,
    src_size: float,
    sigma: float,
    soft_k: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    steps: int,
    lr: float,
    ux_field: np.ndarray = None,
    uy_field: np.ndarray = None,
):
    """Optimize continuous source coordinates from prescribed coarse-search initializations."""
    initial_xy = np.asarray(initial_xy, dtype=np.float64)
    if initial_xy.ndim != 2 or initial_xy.shape[1] != 2 or len(initial_xy) == 0:
        raise ValueError("initial_xy must have shape (n_initializations, 2) with at least one row.")

    iy_t = torch.as_tensor(iy, device=device, dtype=torch.long)
    ix_t = torch.as_tensor(ix, device=device, dtype=torch.long)
    obs_t = torch.as_tensor(obs, device=device, dtype=x_mean.dtype)
    ux_t = None if ux_field is None else torch.as_tensor(ux_field, device=device, dtype=x_mean.dtype).unsqueeze(0).unsqueeze(0)
    uy_t = None if uy_field is None else torch.as_tensor(uy_field, device=device, dtype=x_mean.dtype).unsqueeze(0).unsqueeze(0)

    best = {'loss': float('inf'), 'sx': None, 'sy': None, 'restart': -1, 'trace': []}
    x_rng = float(x_max - x_min)
    y_rng = float(y_max - y_min)

    for r, (sx0, sy0) in enumerate(initial_xy):
        sx0 = float(np.clip(sx0, x_min, x_max))
        sy0 = float(np.clip(sy0, y_min, y_max))
        px0 = torch.tensor((sx0 - x_min) / x_rng, device=device, dtype=x_mean.dtype)
        py0 = torch.tensor((sy0 - y_min) / y_rng, device=device, dtype=x_mean.dtype)
        zx = nn.Parameter(_logit(px0))
        zy = nn.Parameter(_logit(py0))
        opt = torch.optim.Adam([zx, zy], lr=float(lr))

        local_trace = []
        local_best_loss = float('inf')
        local_best_xy = (sx0, sy0)

        for t in range(max(1, int(steps))):
            opt.zero_grad(set_to_none=True)
            sx = x_min + x_rng * torch.sigmoid(zx)
            sy = y_min + y_rng * torch.sigmoid(zy)
            x_in = _build_input_field_torch(
                sx=sx,
                sy=sy,
                ny=ny,
                nx=nx,
                Lx=Lx,
                Ly=Ly,
                mask_kind=mask_kind,
                src_size=src_size,
                sigma=sigma,
                soft_k=soft_k,
                device=device,
                dtype=x_mean.dtype,
                ux=ux_t,
                uy=uy_t,
            )
            x_n = (x_in - x_mean) / (x_std + 1e-12)
            y_n = model(x_n)
            y = y_n * (y_std + 1e-12) + y_mean
            pred_obs = y[0, 0, iy_t, ix_t]
            loss = torch.mean((pred_obs - obs_t) ** 2)
            loss.backward()
            opt.step()

            l = float(loss.detach().cpu())
            sx_f = float(sx.detach().cpu())
            sy_f = float(sy.detach().cpu())
            local_trace.append((r, t, sx_f, sy_f, l))
            if l < local_best_loss:
                local_best_loss = l
                local_best_xy = (sx_f, sy_f)

        if local_best_loss < best['loss']:
            best['loss'] = local_best_loss
            best['sx'], best['sy'] = local_best_xy
            best['restart'] = r
            best['trace'] = local_trace

    return best


def unique_candidate_sources(rows: pd.DataFrame) -> List[Tuple[float, float]]:
    uniq = rows[["src_x", "src_y"]].drop_duplicates().copy()
    return [(float(r.src_x), float(r.src_y)) for r in uniq.itertuples(index=False)]


def main():
    ap = argparse.ArgumentParser(description="Inverse source localization from sparse C observations using trained fixed-u FNO.")
    ap.add_argument("--workdir", type=str, default="data/dense969")
    ap.add_argument("--exports", type=str, default="exports")
    ap.add_argument("--cases", type=str, default="cases.csv")
    ap.add_argument("--ckpt", type=str, default="runs/baseline/model.pt")

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--case_id", type=str)
    grp.add_argument("--idx", type=int)

    ap.add_argument("--Lx", type=float, default=9.0)
    ap.add_argument("--Ly", type=float, default=3.0)
    ap.add_argument("--nx", type=int, default=128 * 3)
    ap.add_argument("--ny", type=int, default=128)

    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--modes1", type=int, default=16)
    ap.add_argument("--modes2", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)

    ap.add_argument("--mask_kind", type=str, default="auto", choices=["auto", "rect", "gauss", "soft_rect", "soft-rect"])
    ap.add_argument("--src_size", type=float, default=0.20)
    ap.add_argument("--sigma", type=float, default=0.12)

    ap.add_argument("--n_sensors", type=int, default=32)
    ap.add_argument("--sensor_strategy", type=str, default="random", choices=["random", "grid", "grid_centered", "downstream"])
    ap.add_argument("--sensor_seed", type=int, default=42)
    ap.add_argument("--sensor_margin_x", type=int, default=8, help="Excluded grid cells near left/right boundaries when placing sensors")
    ap.add_argument("--sensor_margin_y", type=int, default=4, help="Excluded grid cells near bottom/top boundaries when placing sensors")
    ap.add_argument("--noise_std", type=float, default=0.0, help="Gaussian noise std added to sparse observations")
    # Autodiff inverse optimization (no discrete enumeration)
    ap.add_argument("--inv_steps", type=int, default=300, help="Gradient steps per restart")
    ap.add_argument("--inv_lr", type=float, default=0.05, help="Learning rate for source optimization")
    ap.add_argument("--inv_restarts", type=int, default=8, help="Random restarts for source optimization")
    ap.add_argument("--inv_seed", type=int, default=42, help="Random seed for inverse restarts")
    ap.add_argument("--inv_xmin", type=float, default=None)
    ap.add_argument("--inv_xmax", type=float, default=None)
    ap.add_argument("--inv_ymin", type=float, default=None)
    ap.add_argument("--inv_ymax", type=float, default=None)
    ap.add_argument("--soft_k", type=float, default=80.0, help="Soft-rect sharpness for differentiable mask")
    ap.add_argument("--inlet_u_tol", type=float, default=1e-8)
    ap.add_argument("--save_trace_csv", type=str, default="", help="Optional path to save optimization trace")
    ap.add_argument("--show_plot", action="store_true", help="Show inverse visualization figure")
    ap.add_argument("--save_plot", type=str, default="", help="Optional path to save inverse visualization png")
    ap.add_argument("--plot_log_c", action="store_true", help="Plot C and error in log10(C+eps)")
    ap.add_argument("--plot_eps", type=float, default=1e-12)

    args = ap.parse_args()

    workdir = Path(args.workdir)
    exports_dir = workdir / args.exports
    cases_path = (workdir / args.cases) if not Path(args.cases).is_absolute() else Path(args.cases)

    if not exports_dir.exists():
        raise FileNotFoundError(f"exports folder not found: {exports_dir}")
    if not cases_path.exists():
        raise FileNotFoundError(f"cases.csv not found: {cases_path}")

    # Load checkpoint/model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)

    meta = ckpt.get("meta", {})
    in_ch = int(meta.get("in_ch", int(ckpt["x_mean"].shape[1])))
    out_ch = int(meta.get("out_ch", 1))
    if in_ch not in (3, 5):
        raise ValueError(
            f"Checkpoint in_ch={in_ch}, but this inverse script expects in_ch=3 or in_ch=5. "
            "Please use a checkpoint retrained by train_fixed_u_srcpos_to_c.py."
        )
    if out_ch != 1:
        raise ValueError(f"This inverse script expects C-only checkpoint (out_ch=1), got out_ch={out_ch}")

    fixed_u = meta.get("fixed_inlet_u", None)
    if fixed_u is None:
        raise ValueError("Checkpoint meta missing fixed_inlet_u.")
    fixed_u = float(fixed_u)

    ckpt_mask_kind = str(meta.get("mask_kind", "rect")).lower().strip().replace("-", "_")
    if ckpt_mask_kind == "gaussian":
        ckpt_mask_kind = "gauss"
    if ckpt_mask_kind in ["softrect", "soft_rect(sigmoid)", "softrect(sigmoid)"]:
        ckpt_mask_kind = "soft_rect"
    mask_kind = ckpt_mask_kind if args.mask_kind == "auto" else str(args.mask_kind).lower().strip().replace("-", "_")
    if mask_kind == "gaussian":
        mask_kind = "gauss"
    if mask_kind in ["softrect", "soft_rect(sigmoid)", "softrect(sigmoid)"]:
        mask_kind = "soft_rect"
    if mask_kind not in ["rect", "gauss", "soft_rect"]:
        mask_kind = "rect"

    model = FNO2D(in_ch=in_ch, out_ch=out_ch, width=args.width, modes1=args.modes1, modes2=args.modes2, depth=args.depth).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_mean = ckpt["x_mean"].to(device)
    x_std = ckpt["x_std"].to(device)
    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)

    # Load cases and choose target
    df_cases = pd.read_csv(cases_path)
    req = ["case_id", "inlet_u", "src_x", "src_y"]
    miss = [c for c in req if c not in df_cases.columns]
    if miss:
        raise ValueError(f"cases.csv missing columns {miss}. Got: {list(df_cases.columns)}")

    df_cases["case_id"] = df_cases["case_id"].astype(str)
    df_fixed = df_cases[np.abs(df_cases["inlet_u"].astype(float) - fixed_u) <= float(args.inlet_u_tol)].copy()
    if len(df_fixed) == 0:
        raise RuntimeError(
            f"No case found in cases.csv matching fixed_inlet_u={fixed_u:.6f} (tol={args.inlet_u_tol:.3e})."
        )

    if args.case_id is not None:
        hit = df_fixed[df_fixed["case_id"] == str(args.case_id)]
        if len(hit) == 0:
            raise ValueError(f"case_id={args.case_id} not found in fixed-u subset.")
        row = hit.iloc[0]
    else:
        if args.idx < 0 or args.idx >= len(df_fixed):
            raise ValueError(f"idx out of range for fixed-u subset: idx={args.idx}, n={len(df_fixed)}")
        row = df_fixed.iloc[int(args.idx)]

    case_id = str(row["case_id"]).strip()
    true_u = float(row["inlet_u"])
    true_sx = float(row["src_x"])
    true_sy = float(row["src_y"])

    # Build observation from GT C of chosen case
    gt_file = resolve_case_file(exports_dir, case_id, true_u, true_sx, true_sy)
    df_gt = load_case_csv(gt_file)
    _, _, Ux_gt, Uy_gt, C_gt = df_to_grid(df_gt, Lx=args.Lx, Ly=args.Ly, nx=args.nx, ny=args.ny)

    iy, ix = make_sensor_indices(args.ny, args.nx, args.n_sensors, args.sensor_strategy, args.sensor_seed, args.sensor_margin_x, args.sensor_margin_y)
    obs = C_gt[iy, ix].astype(np.float32)
    if args.noise_std > 0.0:
        rng = np.random.default_rng(args.sensor_seed + 7)
        obs = obs + rng.normal(0.0, float(args.noise_std), size=obs.shape).astype(np.float32)
    # Autodiff inverse: optimize source location directly
    x_min_data = float(df_fixed["src_x"].min())
    x_max_data = float(df_fixed["src_x"].max())
    y_min_data = float(df_fixed["src_y"].min())
    y_max_data = float(df_fixed["src_y"].max())

    x_min = x_min_data if args.inv_xmin is None else float(args.inv_xmin)
    x_max = x_max_data if args.inv_xmax is None else float(args.inv_xmax)
    y_min = y_min_data if args.inv_ymin is None else float(args.inv_ymin)
    y_max = y_max_data if args.inv_ymax is None else float(args.inv_ymax)

    if not (x_min < x_max and y_min < y_max):
        raise ValueError("Invalid inverse bounds. Need xmin<xmax and ymin<ymax.")

    best = _optimize_source_autodiff(
        model=model,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        iy=iy,
        ix=ix,
        obs=obs,
        ny=args.ny,
        nx=args.nx,
        Lx=args.Lx,
        Ly=args.Ly,
        mask_kind=mask_kind,
        src_size=float(args.src_size),
        sigma=float(args.sigma),
        soft_k=float(args.soft_k),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        steps=int(args.inv_steps),
        lr=float(args.inv_lr),
        n_restarts=int(args.inv_restarts),
        restart_seed=int(args.inv_seed),
        ux_field=Ux_gt if in_ch == 5 else None,
        uy_field=Uy_gt if in_ch == 5 else None,
    )

    est_sx = float(best["sx"])
    est_sy = float(best["sy"])
    est_obj = float(best["loss"])
    est_mode = "autodiff"

    dist = float(np.sqrt((est_sx - true_sx) ** 2 + (est_sy - true_sy) ** 2))

    print("[INFO] device:", device)
    print(f"[INFO] checkpoint fixed_inlet_u={fixed_u:.6f} | target case_id={case_id}")
    print(f"[INFO] true source=({true_sx:.6f}, {true_sy:.6f})")
    print(f"[INFO] sensors: n={len(obs)}, strategy={args.sensor_strategy}, noise_std={args.noise_std}")
    print(f"[INFO] inverse method=autodiff | steps={args.inv_steps} lr={args.inv_lr} restarts={args.inv_restarts}")
    print(f"[INFO] bounds x:[{x_min:.3f},{x_max:.3f}] y:[{y_min:.3f},{y_max:.3f}] | soft_k={args.soft_k}")
    print(f"[INFO] best restart={best['restart']} | obj={est_obj:.6e}")
    print()
    print(f"[RESULT] mode={est_mode}")
    print(f"[RESULT] estimated source=({est_sx:.6f}, {est_sy:.6f})")
    print(f"[RESULT] localization error distance={dist:.6f} m")

    trace_df = pd.DataFrame(best["trace"], columns=["restart", "step", "sx", "sy", "obs_mse"]) if best.get("trace") else pd.DataFrame(columns=["restart", "step", "sx", "sy", "obs_mse"])

    if args.show_plot or args.save_plot:
        # Predict full field at estimated source for visualization
        x_in_est = build_input_field(
            sx=est_sx,
            sy=est_sy,
            ny=args.ny,
            nx=args.nx,
            Lx=args.Lx,
            Ly=args.Ly,
            mask_kind=mask_kind,
            src_size=float(args.src_size),
            sigma=float(args.sigma),
            soft_k=float(args.soft_k),
            ux=Ux_gt if in_ch == 5 else None,
            uy=Uy_gt if in_ch == 5 else None,
        )
        C_est = predict_c_field(model, (x_mean, x_std), (y_mean, y_std), device, x_in_est)

        xs = np.linspace(0.0, args.Lx, args.nx, dtype=np.float32)
        ys = np.linspace(0.0, args.Ly, args.ny, dtype=np.float32)
        sensor_x = xs[ix]
        sensor_y = ys[iy]

        if args.plot_log_c:
            eps = float(args.plot_eps)
            C_gt_show = np.log10(np.maximum(C_gt, 0.0) + eps)
            C_est_show = np.log10(np.maximum(C_est, 0.0) + eps)
            title_l = f"log10(C+{eps:g})"
        else:
            C_gt_show = C_gt
            C_est_show = C_est
            title_l = "C"

        vmin_c = float(min(np.min(C_gt_show), np.min(C_est_show)))
        vmax_c = float(max(np.max(C_gt_show), np.max(C_est_show)))

        # Stable layout: 2 panels + 1 dedicated colorbar column
        fig = plt.figure(figsize=(11.8, 4.4))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.05], wspace=0.28)

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        cax0 = fig.add_subplot(gs[0, 2])
        axes = [ax0, ax1]

        im0 = ax0.imshow(C_gt_show, origin="lower", extent=[0, args.Lx, 0, args.Ly], aspect="equal", vmin=vmin_c, vmax=vmax_c)
        ax0.set_title(f"GT {title_l}")
        # Color sensors by observed concentration for interpretability
        ax0.scatter(sensor_x, sensor_y, s=15, c="white", edgecolors="black", linewidths=0.25, alpha=0.95, label="sensors")
        ax0.scatter([true_sx], [true_sy], s=60, marker="x", c="red", linewidths=2.1, label="true src")
        ax0.legend(loc="upper right", fontsize=8)

        im1 = ax1.imshow(C_est_show, origin="lower", extent=[0, args.Lx, 0, args.Ly], aspect="equal", vmin=vmin_c, vmax=vmax_c)
        ax1.set_title(f"Pred @ est src {title_l}")
        ax1.scatter(sensor_x, sensor_y, s=15, c="white", edgecolors="black", linewidths=0.25, alpha=0.95, label="sensors")
        ax1.scatter([est_sx], [est_sy], s=62, marker="x", c="black", linewidths=2.1, label="estimated src")
        ax1.legend(loc="upper right", fontsize=8)

        for ax in axes:
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

        cbar0 = fig.colorbar(im1, cax=cax0)
        cbar0.set_label(title_l)

        fig.suptitle(
            f"Inverse localization ({est_mode}) | case_id={case_id} | true=({true_sx:.3f},{true_sy:.3f}) | "
            f"est=({est_sx:.3f},{est_sy:.3f}) | err={dist:.3f} m",
            fontsize=11,
        )
        fig.subplots_adjust(top=0.84)

        if args.save_plot:
            out_plot = Path(args.save_plot)
            if not out_plot.is_absolute():
                out_plot = Path.cwd() / out_plot
            out_plot.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_plot, dpi=180)
            print(f"[SAVED] inverse plot: {out_plot}")

        if args.show_plot:
            plt.show()
        else:
            plt.close(fig)
    if args.save_trace_csv:
        out_path = Path(args.save_trace_csv)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trace_df.to_csv(out_path, index=False)
        print(f"[SAVED] optimization trace csv: {out_path}")


if __name__ == "__main__":
    main()
