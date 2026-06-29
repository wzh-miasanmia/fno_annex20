# infer_fixed_u_srcpos_to_c.py
# Infer + visualize scalar field C using your trained fixed-u FNO:
#   inputs: source mask S_mask(x,y) + coordinates
#   outputs: C
#
# It ALSO loads the corresponding GT (STAR) field and compares:
#   - Rel L2 for C
#   - 1x3 plot: (GT | Pred | Error) for C
#   - Centerline profile (y = Ly/2) for C
#
# Dataset:
#   cases.csv columns: case_id,inlet_u,src_x,src_y
#   exports folder contains: case_{case_id}_u{:.4f}_sx{:.4f}_sy{:.4f}.csv
#   STAR export CSV columns include at least:
#     "Velocity[i] (m/s)","Velocity[j] (m/s)","Passive Scalar","X (m)","Y (m)"
#
# Usage:
#   # by case_id
#   python .\visualization\infer_fixed_u_srcpos_to_c.py --workdir .\data --exports exports --cases cases.csv --ckpt .\checkpoints\fixed_u_srcpos_to_c.pt --case_id 000001
#
#   # by index in cases.csv (0-based)
#   python .\visualization\infer_fixed_u_srcpos_to_c.py --workdir .\data --exports exports --cases cases.csv --ckpt .\checkpoints\fixed_u_srcpos_to_c.pt --idx 0
#
# Notes:
# - Model input channels = 3: [S_mask, X, Y]
# - S_mask is built from (src_x, src_y) using either:
#     rect: top-hat square mask of size src_size (meters)
#     gauss: gaussian mask with sigma (meters)
# - mask_kind should match training. If checkpoint has meta["mask_kind"], it is used by default.
# - We use x_mean/x_std/y_mean/y_std saved in checkpoint for normalization/denorm.
# - If you trained with AMP, inference here is still fp32 (safe). You can add autocast if you want.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# -----------------------------
# Utils
# -----------------------------
def safe_float_tag(x: float) -> str:
    return f"{x:.4f}"


def l2_rel(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(a)) + eps
    return num / den


def resolve_case_file(exports_dir: Path, case_id: str, inlet_u: float, sx: float, sy: float) -> Path:
    """
    Prefer exact macro naming:
      case_{case_id}_u{u:.4f}_sx{sx:.4f}_sy{sy:.4f}.csv
    If not found, fallback to first glob match: case_{case_id}_*.csv
    """
    exact = exports_dir / (
        f"case_{case_id}_u{safe_float_tag(inlet_u)}_sx{safe_float_tag(sx)}_sy{safe_float_tag(sy)}.csv"
    )
    if exact.exists():
        return exact

    hits = sorted(exports_dir.glob(f"case_{case_id}_*.csv"))
    if hits:
        return hits[0]

    raise FileNotFoundError(f"No exported CSV found for case_id={case_id} under {exports_dir}")


# -----------------------------
# STAR CSV -> uniform grid
# -----------------------------
def load_case_csv(case_path: Path) -> pd.DataFrame:
    df = pd.read_csv(case_path)

    required = ["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Passive Scalar", "X (m)", "Y (m)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {case_path}. Got: {list(df.columns)}")

    return df[required].copy()


def df_to_grid(df: pd.DataFrame, Lx=9.0, Ly=3.0, nx=128 * 3, ny=128):
    x = np.linspace(0.0, Lx, nx, dtype=np.float32)
    y = np.linspace(0.0, Ly, ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(x, y)

    pts = df[["X (m)", "Y (m)"]].to_numpy(dtype=np.float64)
    ux = df["Velocity[i] (m/s)"].to_numpy(dtype=np.float64)
    uy = df["Velocity[j] (m/s)"].to_numpy(dtype=np.float64)
    cc = df["Passive Scalar"].to_numpy(dtype=np.float64)

    def interp(vals):
        lin = griddata(pts, vals, (Xg, Yg), method="linear")
        near = griddata(pts, vals, (Xg, Yg), method="nearest")
        return np.where(np.isnan(lin), near, lin).astype(np.float32)

    Ux = interp(ux)
    Uy = interp(uy)
    C = interp(cc)
    return x, y, Ux, Uy, C


# -----------------------------
# Source mask builder (must match training)
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
    hx = 0.5 * src_size
    hy = 0.5 * src_size
    dx = np.abs(Xg - src_x) - hx
    dy = np.abs(Yg - src_y) - hy
    sdf = np.maximum(dx, dy)
    z = np.clip(soft_k * sdf, -60.0, 60.0)
    mask = (1.0 / (1.0 + np.exp(z))).astype(np.float32)
    return mask


# -----------------------------
# Input builder (must match training)
# -----------------------------
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
) -> np.ndarray:
    """
    Returns x_field with shape (1,3,ny,nx):
      [S_mask_field, Xcoord(0..1), Ycoord(0..1)]
    """
    # X,Y coord channels in [0,1]
    xx01 = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    yy01 = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    Xc, Yc = np.meshgrid(xx01, yy01)
    Xc = Xc[None, ...].astype(np.float32)
    Yc = Yc[None, ...].astype(np.float32)

    # Physical grid for mask (meters)
    x_phys = np.linspace(0.0, Lx, nx, dtype=np.float32)
    y_phys = np.linspace(0.0, Ly, ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(x_phys, y_phys)

    mk = str(mask_kind).lower().strip().replace("-", "_")
    if mk in ["softrect", "soft_rect(sigmoid)", "softrect(sigmoid)"]:
        mk = "soft_rect"
    if mk == "gaussian":
        mk = "gauss"

    if mk == "rect":
        S = build_source_mask_rect(Xg, Yg, sx, sy, src_size)
    elif mk == "gauss":
        S = build_source_mask_gaussian(Xg, Yg, sx, sy, sigma)
    elif mk == "soft_rect":
        S = build_source_mask_soft_rect_sigmoid(Xg, Yg, sx, sy, src_size, soft_k)
    else:
        raise ValueError("mask_kind must be one of: 'rect', 'gauss', 'soft_rect'")

    S = S[None, ...].astype(np.float32)

    x_field = np.concatenate([S, Xc, Yc], axis=0)  # (3,ny,nx)
    return x_field[None, ...]  # (1,3,ny,nx)


# -----------------------------
# Plot helper
# -----------------------------
def plot_field(ax, x, y, F, title, vmin=None, vmax=None):
    im = ax.imshow(
        F,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        # cmap="coolwarm",   # add if need coolwarm colormap, with 0 equals to white
    )
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return im


# -----------------------------
# Minimal FNO2D (must match training)
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


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=str, required=True, help="Folder containing cases.csv and exports/")
    ap.add_argument("--exports", type=str, default="exports", help="Exports subfolder under workdir")
    ap.add_argument("--cases", type=str, default="cases.csv", help="cases.csv path (relative to workdir or absolute)")
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint .pt from train_fixed_u_srcpos_to_c.py")

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--case_id", type=str, help="case_id to test (string)")
    grp.add_argument("--idx", type=int, help="row index in cases.csv (0-based)")

    ap.add_argument("--Lx", type=float, default=9.0)
    ap.add_argument("--Ly", type=float, default=3.0)
    ap.add_argument("--nx", type=int, default=128 * 3)
    ap.add_argument("--ny", type=int, default=128)

    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--modes1", type=int, default=16)
    ap.add_argument("--modes2", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--inlet_u_tol", type=float, default=1e-8, help="Tolerance when checking checkpoint fixed_inlet_u")

    # C visualization
    ap.add_argument("--c_log", action="store_true", help="plot C as log10(C+eps) (GT/Pred/Error)")
    ap.add_argument("--c_eps", type=float, default=1e-12)

    # mask config (if not set, try to read from ckpt meta)
    ap.add_argument("--mask_kind", type=str, default="auto", choices=["auto", "rect", "gauss", "soft_rect", "soft-rect"])
    ap.add_argument("--src_size", type=float, default=0.20, help="rect/soft-rect mask size [m] (top-hat square)")
    ap.add_argument("--sigma", type=float, default=0.12, help="gaussian sigma [m] (if mask_kind=gauss)")
    ap.add_argument("--soft_k", type=float, default=None, help="soft-rect sigmoid sharpness; if not set, read from ckpt meta")

    args = ap.parse_args()

    workdir = Path(args.workdir)
    exports_dir = workdir / args.exports
    cases_path = (workdir / args.cases) if not Path(args.cases).is_absolute() else Path(args.cases)

    if not exports_dir.exists():
        raise FileNotFoundError(f"exports folder not found: {exports_dir}")
    if not cases_path.exists():
        raise FileNotFoundError(f"cases.csv not found: {cases_path}")

    df_cases = pd.read_csv(cases_path)
    req = ["case_id", "inlet_u", "src_x", "src_y"]
    missing = [c for c in req if c not in df_cases.columns]
    if missing:
        raise ValueError(f"cases.csv missing columns {missing}. Got: {list(df_cases.columns)}")

    df_cases["case_id"] = df_cases["case_id"].astype(str)

    if args.case_id is not None:
        hit = df_cases[df_cases["case_id"] == str(args.case_id)]
        if len(hit) == 0:
            raise ValueError(f"case_id={args.case_id} not found in cases.csv")
        row = hit.iloc[0]
    else:
        row = df_cases.iloc[int(args.idx)]

    case_id = str(row["case_id"]).strip()
    inlet_u = float(row["inlet_u"])
    sx = float(row["src_x"])
    sy = float(row["src_y"])

    case_file = resolve_case_file(exports_dir, case_id, inlet_u, sx, sy)

    # ---- Load GT ----
    df_gt = load_case_csv(case_file)
    x, y, Ux_gt, Uy_gt, C_gt = df_to_grid(df_gt, Lx=args.Lx, Ly=args.Ly, nx=args.nx, ny=args.ny)

    # ---- Load model + norms ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)

    x_mean = ckpt["x_mean"].to(device)
    x_std  = ckpt["x_std"].to(device)
    y_mean = ckpt["y_mean"].to(device)
    y_std  = ckpt["y_std"].to(device)

    meta = ckpt.get("meta", {})
    ckpt_in_ch = int(meta.get("in_ch", int(x_mean.shape[1])))
    if ckpt_in_ch != 3:
        raise ValueError(
            f"Checkpoint in_ch={ckpt_in_ch}, but this fixed-u C-only script now expects in_ch=3 ([S_mask, X, Y]). "
            "Please use a new checkpoint trained by train_fixed_u_srcpos_to_c.py."
        )
    ckpt_mask_kind = str(meta.get("mask_kind", "rect")).lower().strip().replace("-", "_")
    if ckpt_mask_kind in ["softrect", "soft_rect(sigmoid)", "softrect(sigmoid)"]:
        ckpt_mask_kind = "soft_rect"
    if ckpt_mask_kind == "gaussian":
        ckpt_mask_kind = "gauss"
    ckpt_soft_k = meta.get("soft_k", None)
    ckpt_soft_k = None if ckpt_soft_k is None else float(ckpt_soft_k)
    ckpt_fixed_u = meta.get("fixed_inlet_u", None)
    if ckpt_fixed_u is not None:
        ckpt_fixed_u = float(ckpt_fixed_u)
        if abs(inlet_u - ckpt_fixed_u) > float(args.inlet_u_tol):
            raise ValueError(
                f"Input case inlet_u={inlet_u:.6f} mismatches checkpoint fixed_inlet_u={ckpt_fixed_u:.6f} "
                f"(tol={args.inlet_u_tol:.3e})."
            )

    if args.mask_kind == "auto":
        mask_kind = ckpt_mask_kind
    else:
        mask_kind = str(args.mask_kind).lower().strip().replace("-", "_")

    if mask_kind in ["softrect", "soft_rect(sigmoid)", "softrect(sigmoid)"]:
        mask_kind = "soft_rect"
    if mask_kind == "gaussian":
        mask_kind = "gauss"
    if mask_kind not in ["rect", "gauss", "soft_rect"]:
        mask_kind = "rect"

    soft_k = float(args.soft_k) if args.soft_k is not None else (80.0 if ckpt_soft_k is None else ckpt_soft_k)

    model = FNO2D(
        in_ch=ckpt_in_ch,
        out_ch=1,
        width=args.width,
        modes1=args.modes1,
        modes2=args.modes2,
        depth=args.depth,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- Predict ----
    x_in = build_input_field(
        sx=sx,
        sy=sy,
        ny=args.ny,
        nx=args.nx,
        Lx=args.Lx,
        Ly=args.Ly,
        mask_kind=mask_kind,
        src_size=float(args.src_size),
        sigma=float(args.sigma),
        soft_k=float(soft_k),
    )  # (1,3,ny,nx)

    x_t = torch.from_numpy(x_in).to(device)
    x_t_n = (x_t - x_mean) / (x_std + 1e-12)

    with torch.no_grad():
        y_pred_n = model(x_t_n)

    y_pred = y_pred_n * (y_std + 1e-12) + y_mean
    y_pred = y_pred.cpu().numpy()[0]  # (1,ny,nx)
    C_pr = y_pred[0]

    # ---- Metrics ----
    print(f"[INFO] device={device}")
    print(f"[INFO] case_id={case_id}")
    print(f"[INFO] inlet_u={inlet_u:.6f} m/s | src=({sx:.6f},{sy:.6f})")
    print(f"[INFO] mask_kind={mask_kind} | src_size={args.src_size} m | sigma={args.sigma} m | soft_k={soft_k}")
    if ckpt_fixed_u is not None:
        print(f"[INFO] checkpoint fixed_inlet_u={ckpt_fixed_u:.6f} | tol={args.inlet_u_tol:.3e}")
    print(f"[INFO] GT file: {case_file}")
    print("\n[METRICS] Relative L2 errors")
    print(f"  C  : {l2_rel(C_gt, C_pr):.4e}")

    # ---- Visualization: GT vs Pred vs Error for C ----
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 11.5))

    if args.c_log:
        eps = float(args.c_eps)
        C_gt_plot = np.log10(np.maximum(C_gt, 0.0) + eps)
        C_pr_plot = np.log10(np.maximum(C_pr, 0.0) + eps)
        C_er_plot = C_pr_plot - C_gt_plot

        vmin = float(min(np.min(C_gt_plot), np.min(C_pr_plot), np.min(C_er_plot)))
        vmax = float(max(np.max(C_gt_plot), np.max(C_pr_plot), np.max(C_er_plot)))

        im = plot_field(axes[0], x, y, C_gt_plot, f"GT log10(C+{eps:g})", vmin=vmin, vmax=vmax)
        plot_field(axes[1], x, y, C_pr_plot, f"Pred log10(C+{eps:g})", vmin=vmin, vmax=vmax)
        plot_field(axes[2], x, y, C_er_plot, "Error log10(C+eps)", vmin=vmin, vmax=vmax)
    else:
        C_er = C_pr - C_gt
        vmin = float(min(np.min(C_gt), np.min(C_pr), np.min(C_er)))
        vmax = float(max(np.max(C_gt), np.max(C_pr), np.max(C_er)))

        im = plot_field(axes[0], x, y, C_gt, "GT C", vmin=vmin, vmax=vmax)
        plot_field(axes[1], x, y, C_pr, "Pred C", vmin=vmin, vmax=vmax)
        plot_field(axes[2], x, y, C_er, "Error C", vmin=vmin, vmax=vmax)

    fig.subplots_adjust(left=0.10, right=0.90, top=0.97, bottom=0.06, hspace=0.03)
    pos_top = axes[0].get_position()
    pos_bot = axes[-1].get_position()
    cbar_pad = 0.030
    cbar_w = 0.022
    cbar_x = pos_top.x1 + cbar_pad
    cbar_y = pos_bot.y0
    cbar_h = pos_top.y1 - pos_bot.y0
    cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

    # ---- Centerline profile (y = Ly/2) ----
    j_mid = int(np.argmin(np.abs(y - (args.Ly / 2))))

    plt.figure(figsize=(8, 4))
    plt.plot(x, C_gt[j_mid, :], label="GT C centerline")
    plt.plot(x, C_pr[j_mid, :], label="Pred C centerline")
    plt.xlabel("x (m)")
    plt.ylabel("C (passive scalar)")
    plt.title("Centerline scalar profile (y=Ly/2)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
