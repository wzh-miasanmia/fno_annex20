import os
import sys
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle


# =========================================================
# Paths / imports
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import visualization.inverse_src_from_sparse_c_fixed_u_continuous as inv


# =========================================================
# Fixed room size / grid
# =========================================================
Lx = 9.0
Ly = 3.0
NY = 128
NX = 128 * 3


# =========================================================
# Paper baseline checkpoint
# =========================================================
CKPT_PATH = str(
    PROJECT_ROOT
    / "paper_runs"
    / "baseline_dense969_u0p2_w48_m16_d4_soft_rect"
    / "checkpoints"
    / "dense969_u0p2_w48_m16_d4_soft_rect_sk30_rawtarget.pt"
)


# =========================================================
# ACH / CRE settings
# =========================================================
ROOM_THICKNESS = 1.0
INLET_HEIGHT = 0.168
INLET_WIDTH = ROOM_THICKNESS
INLET_AREA = INLET_HEIGHT * INLET_WIDTH
ROOM_VOLUME = Lx * Ly * ROOM_THICKNESS

OUTLET_WIDTH_X = 0.05
OUTLET_Y0 = 0.01
OUTLET_Y1 = 0.25

BZ_X0, BZ_X1 = 0.6, 8.4
BZ_Y0, BZ_Y1 = 0.075, 1.8

SRC_SIZE_FIXED = 0.20
SIGMA_FIXED = 0.12

PLOT_W = 11.2
PLOT_H = 3.7
LOSS_W = 4.4
LOSS_H = 3.1
DEFAULT_MARGIN_X = 2
DEFAULT_MARGIN_Y = 2
HYBRID_COARSE_SPACING = 0.25
HYBRID_TOP_K_DEFAULT = 4
HYBRID_BATCH_SIZE = 4


# =========================================================
# Minimal FNO2D (match training)
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
        out_ft = torch.zeros(
            bsz, self.out_channels, ny, x_ft.shape[-1], dtype=torch.cfloat, device=x.device
        )
        weight = torch.view_as_complex(self.weight)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], weight[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(ny, nx), norm="ortho")


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
# Helpers
# =========================================================
def compute_ach(inlet_u: float) -> float:
    q = float(inlet_u) * INLET_AREA
    return q * 3600.0 / ROOM_VOLUME


def build_source_mask_rect(xg: np.ndarray, yg: np.ndarray, src_x: float, src_y: float, src_size: float) -> np.ndarray:
    hx = 0.5 * src_size
    hy = 0.5 * src_size
    return (
        (xg >= (src_x - hx)) & (xg <= (src_x + hx)) & (yg >= (src_y - hy)) & (yg <= (src_y + hy))
    ).astype(np.float32)


def build_source_mask_gaussian(xg: np.ndarray, yg: np.ndarray, src_x: float, src_y: float, sigma: float) -> np.ndarray:
    r2 = (xg - src_x) ** 2 + (yg - src_y) ** 2
    return np.exp(-0.5 * r2 / (sigma ** 2)).astype(np.float32)


def build_source_mask_soft_rect_sigmoid(
    xg: np.ndarray,
    yg: np.ndarray,
    src_x: float,
    src_y: float,
    src_size: float,
    soft_k: float = 80.0,
) -> np.ndarray:
    hx = 0.5 * src_size
    hy = 0.5 * src_size
    dx = np.abs(xg - src_x) - hx
    dy = np.abs(yg - src_y) - hy
    sdf = np.maximum(dx, dy)
    z = np.clip(soft_k * sdf, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(z))).astype(np.float32)


def build_input_field(
    src_x: float,
    src_y: float,
    ny: int,
    nx: int,
    mask_kind: str,
    src_size: float,
    sigma: float,
    soft_k: float,
    ux: np.ndarray = None,
    uy: np.ndarray = None,
) -> np.ndarray:
    xx01 = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    yy01 = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    xc, yc = np.meshgrid(xx01, yy01)
    xc = xc[None, ...]
    yc = yc[None, ...]

    x_phys = np.linspace(0.0, Lx, nx, dtype=np.float32)
    y_phys = np.linspace(0.0, Ly, ny, dtype=np.float32)
    xg, yg = np.meshgrid(x_phys, y_phys)

    mk = str(mask_kind).lower().strip().replace("-", "_")
    if mk in ["softrect", "soft_rect(sigmoid)", "softrect(sigmoid)"]:
        mk = "soft_rect"
    if mk == "gaussian":
        mk = "gauss"

    if mk == "rect":
        s2d = build_source_mask_rect(xg, yg, src_x, src_y, src_size)
    elif mk == "gauss":
        s2d = build_source_mask_gaussian(xg, yg, src_x, src_y, sigma)
    elif mk == "soft_rect":
        s2d = build_source_mask_soft_rect_sigmoid(xg, yg, src_x, src_y, src_size, soft_k)
    else:
        s2d = build_source_mask_rect(xg, yg, src_x, src_y, src_size)

    s = s2d[None, ...]
    channels = [s, xc, yc]
    if ux is not None or uy is not None:
        if ux is None or uy is None:
            raise ValueError("ux and uy must be provided together.")
        channels.extend([ux[None, ...].astype(np.float32), uy[None, ...].astype(np.float32)])
    return np.concatenate(channels, axis=0)[None, ...]


def load_reference_velocity_field(meta: dict, fixed_inlet_u: float, ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    workdir = Path(str(meta.get("dataset_workdir", PROJECT_ROOT / "data")))
    exports_dir = Path(str(meta.get("dataset_exports_dir", workdir / "exports")))
    cases_path = Path(str(meta.get("dataset_cases_path", workdir / "cases.csv")))
    if not cases_path.exists():
        raise FileNotFoundError(f"Cannot load Ux/Uy for 5-channel checkpoint; cases.csv not found: {cases_path}")
    if not exports_dir.exists():
        raise FileNotFoundError(f"Cannot load Ux/Uy for 5-channel checkpoint; exports dir not found: {exports_dir}")

    rows = pd.read_csv(cases_path)
    hit = rows[np.abs(rows["inlet_u"].astype(float) - float(fixed_inlet_u)) <= 1e-8]
    if len(hit) == 0:
        raise RuntimeError(f"No reference case found for fixed_inlet_u={fixed_inlet_u:.6f}.")
    row = hit.iloc[0]
    case_file = inv.resolve_case_file(
        exports_dir,
        str(row["case_id"]).strip(),
        float(row["inlet_u"]),
        float(row["src_x"]),
        float(row["src_y"]),
    )
    _, _, ux, uy, _ = inv.df_to_grid(inv.load_case_csv(case_file), Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    return ux, uy


def make_xy_grids(ny: int, nx: int):
    xs = np.linspace(0.0, Lx, nx, dtype=np.float32)
    ys = np.linspace(0.0, Ly, ny, dtype=np.float32)
    return np.meshgrid(xs, ys)


def rect_mask(xg: np.ndarray, yg: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
    ya, yb = (y0, y1) if y0 <= y1 else (y1, y0)
    return ((xg >= xa) & (xg <= xb) & (yg >= ya) & (yg <= yb)).astype(np.float32)


def masked_mean(field: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    w = float(np.sum(mask))
    if w <= 0.0:
        return float("nan")
    return float(np.sum(field * mask) / (w + eps))


def compute_cre(c_field: np.ndarray, outlet_mask: np.ndarray, bz_mask: np.ndarray):
    c_out = masked_mean(c_field, outlet_mask)
    c_bz = masked_mean(c_field, bz_mask)
    if not np.isfinite(c_out) or not np.isfinite(c_bz) or c_bz == 0.0:
        return float("nan"), c_out, c_bz
    return float(c_out / c_bz), c_out, c_bz


def draw_rect(ax, x0: float, x1: float, y0: float, y1: float, linestyle: str = "--", linewidth: float = 1.5):
    xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
    ya, yb = (y0, y1) if y0 <= y1 else (y1, y0)
    ax.add_patch(Rectangle((xa, ya), xb - xa, yb - ya, fill=False, linestyle=linestyle, linewidth=linewidth))


def nearest_grid_indices(sensor_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, Lx, NX, dtype=np.float32)
    ys = np.linspace(0.0, Ly, NY, dtype=np.float32)
    ix = np.array([int(np.argmin(np.abs(xs - float(v)))) for v in sensor_df["x_m"]], dtype=int)
    iy = np.array([int(np.argmin(np.abs(ys - float(v)))) for v in sensor_df["y_m"]], dtype=int)
    return iy, ix


def internal_bounds(margin_x: int, margin_y: int) -> tuple[float, float, float, float]:
    xs = np.linspace(0.0, Lx, NX, dtype=np.float32)
    ys = np.linspace(0.0, Ly, NY, dtype=np.float32)
    return float(xs[margin_x]), float(xs[-margin_x - 1]), float(ys[margin_y]), float(ys[-margin_y - 1])


def default_sensor_table(n_sensors: int, margin_x: int, margin_y: int, strategy: str) -> pd.DataFrame:
    if strategy == "uniform":
        iy, ix = inv.make_sensor_indices(NY, NX, n_sensors, "grid_centered", 42, margin_x, margin_y)
        xs = np.linspace(0.0, Lx, NX, dtype=np.float32)
        ys = np.linspace(0.0, Ly, NY, dtype=np.float32)
        return pd.DataFrame({"x_m": xs[ix].round(3), "y_m": ys[iy].round(3)})

    x0, x1, y0, y1 = internal_bounds(margin_x, margin_y)
    return pd.DataFrame(
        {
            "x_m": np.full(n_sensors, round((x0 + x1) / 2.0, 3)),
            "y_m": np.linspace(y0, y1, n_sensors).round(3),
        }
    )


def candidate_sensor_points(margin_x: int, margin_y: int, dx: float = 0.25, dy: float = 0.25) -> pd.DataFrame:
    x0, x1, y0, y1 = internal_bounds(margin_x, margin_y)
    xs = np.arange(x0, x1 + 0.5 * dx, dx, dtype=float)
    ys = np.arange(y0, y1 + 0.5 * dy, dy, dtype=float)
    pts = [{"x_m": round(float(x), 3), "y_m": round(float(y), 3)} for y in ys for x in xs]
    return pd.DataFrame(pts)


def parse_point_selection(selection_payload) -> pd.DataFrame:
    if selection_payload is None:
        return pd.DataFrame(columns=["x_m", "y_m"])

    # Common Vega-Lite / Streamlit point selection shapes.
    if isinstance(selection_payload, dict):
        if "values" in selection_payload and isinstance(selection_payload["values"], list):
            vals = selection_payload["values"]
            rows = [{"x_m": float(v["x_m"]), "y_m": float(v["y_m"])} for v in vals if "x_m" in v and "y_m" in v]
            return pd.DataFrame(rows)
        if "x_m" in selection_payload and "y_m" in selection_payload:
            xs = selection_payload["x_m"]
            ys = selection_payload["y_m"]
            if isinstance(xs, list) and isinstance(ys, list):
                return pd.DataFrame({"x_m": [float(v) for v in xs], "y_m": [float(v) for v in ys]})
            return pd.DataFrame([{"x_m": float(xs), "y_m": float(ys)}])
        for value in selection_payload.values():
            parsed = parse_point_selection(value)
            if not parsed.empty:
                return parsed
    if isinstance(selection_payload, list):
        rows = []
        for item in selection_payload:
            if isinstance(item, dict) and "x_m" in item and "y_m" in item:
                rows.append({"x_m": float(item["x_m"]), "y_m": float(item["y_m"])})
        if rows:
            return pd.DataFrame(rows)
    return pd.DataFrame(columns=["x_m", "y_m"])


def apply_app_style():
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"],
        footer,
        #MainMenu {
            visibility: hidden;
        }
        [data-testid="stSidebar"] {
            background: #f7f9fb;
            border-right: 1px solid #dfe6ee;
        }
        .block-container {
            padding-top: 1.0rem;
            padding-bottom: 1.2rem;
            max-width: 1450px;
        }
        h1 {
            font-size: 2.0rem !important;
            letter-spacing: 0 !important;
            margin-bottom: 0.15rem !important;
            color: #102a43;
        }
        h2, h3 {
            color: #102a43;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dfe6ee;
            border-radius: 8px;
            padding: 0.55rem 0.75rem;
            box-shadow: 0 1px 2px rgba(16, 42, 67, 0.05);
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.88rem;
            color: #486581;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.25rem;
            color: #102a43;
        }
        .app-hero {
            border: 1px solid #d9e2ec;
            background: linear-gradient(90deg, #f8fbfd 0%, #ffffff 100%);
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.9rem;
        }
        .app-kicker {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.74rem;
            font-weight: 700;
            color: #0b7285;
            margin-bottom: 0.15rem;
        }
        .panel-title {
            font-size: 1.0rem;
            font-weight: 700;
            color: #102a43;
            margin-bottom: 0.35rem;
        }
        .small-note {
            color: #52616b;
            font-size: 0.92rem;
        }
        .stButton button {
            border-radius: 6px;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_query_param(name: str, default: str = "") -> str:
    try:
        value = st.query_params.get(name, default)
    except Exception:
        value = default
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value)


def is_paper_mode() -> bool:
    return get_query_param("paper", "false").lower() in {"1", "true", "yes", "on"}


# =========================================================
# Model loading / prediction
# =========================================================
@st.cache_resource(show_spinner=True)
def load_model_and_norm(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    meta = ckpt.get("meta", {})
    in_ch = int(meta.get("in_ch", int(ckpt["x_mean"].shape[1])))
    out_ch = int(meta.get("out_ch", 1))
    if in_ch not in (3, 5):
        raise ValueError(
            f"Checkpoint in_ch={in_ch}, but this app expects in_ch=3 or in_ch=5. "
            "Please use a checkpoint retrained by train_fixed_u_srcpos_to_c.py."
        )
    fixed_inlet_u = meta.get("fixed_inlet_u", None)
    if fixed_inlet_u is None:
        raise ValueError("Checkpoint meta missing fixed_inlet_u. Please use a fixed-u C-only checkpoint.")

    width = int(meta.get("width", 48))
    modes1 = int(meta.get("modes1", 16))
    modes2 = int(meta.get("modes2", 16))
    depth = int(meta.get("depth", 4))

    model = FNO2D(in_ch=in_ch, out_ch=out_ch, width=width, modes1=modes1, modes2=modes2, depth=depth).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_mean = ckpt["x_mean"].to(device)
    x_std = ckpt["x_std"].to(device)
    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)

    ckpt_mask_kind = str(meta.get("mask_kind", "rect")).lower().strip().replace("-", "_")
    if ckpt_mask_kind in ["softrect", "soft_rect(sigmoid)", "softrect(sigmoid)"]:
        ckpt_mask_kind = "soft_rect"
    if ckpt_mask_kind == "gaussian":
        ckpt_mask_kind = "gauss"
    if ckpt_mask_kind not in ["rect", "gauss", "soft_rect"]:
        ckpt_mask_kind = "rect"

    ckpt_soft_k = meta.get("soft_k", None)
    ckpt_soft_k = 80.0 if ckpt_soft_k is None else float(ckpt_soft_k)
    velocity_field = load_reference_velocity_field(meta, float(fixed_inlet_u), NY, NX) if in_ch == 5 else None

    x_min = float(meta.get("src_x_min", 0.5))
    x_max = float(meta.get("src_x_max", 7.5))
    y_min = float(meta.get("src_y_min", 0.5))
    y_max = float(meta.get("src_y_max", 2.5))

    return {
        "model": model,
        "norms": (x_mean, x_std, y_mean, y_std),
        "device": device,
        "fixed_inlet_u": float(fixed_inlet_u),
        "mask_kind": ckpt_mask_kind,
        "soft_k": ckpt_soft_k,
        "src_bounds": (x_min, x_max, y_min, y_max),
        "in_ch": in_ch,
        "velocity_field": velocity_field,
    }


def predict_c(
    model,
    norms,
    device,
    src_x: float,
    src_y: float,
    ny: int,
    nx: int,
    mask_kind: str,
    src_size: float,
    sigma: float,
    soft_k: float,
    velocity_field: tuple[np.ndarray, np.ndarray] = None,
):
    x_mean, x_std, y_mean, y_std = norms
    ux, uy = (None, None) if velocity_field is None else velocity_field
    x_np = build_input_field(src_x, src_y, ny, nx, mask_kind, src_size, sigma, soft_k, ux=ux, uy=uy)
    x_t = torch.from_numpy(x_np).to(device)
    x_t_n = (x_t - x_mean) / (x_std + 1e-12)
    with torch.no_grad():
        y_pred_n = model(x_t_n)
    y_pred = y_pred_n * (y_std + 1e-12) + y_mean
    return y_pred.cpu().numpy()[0, 0]


def make_coarse_source_grid(
    src_bounds: tuple[float, float, float, float], spacing: float
) -> np.ndarray:
    x_min, x_max, y_min, y_max = src_bounds

    def axis_values(v_min: float, v_max: float) -> np.ndarray:
        values = np.arange(v_min, v_max + 0.5 * spacing, spacing, dtype=np.float32)
        values = values[values <= v_max + 1e-6]
        if len(values) == 0 or not np.isclose(values[-1], v_max):
            values = np.r_[values, np.float32(v_max)]
        return values

    xs = axis_values(float(x_min), float(x_max))
    ys = axis_values(float(y_min), float(y_max))
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)


def build_coarse_sensor_library(
    model,
    norms,
    device,
    iy: np.ndarray,
    ix: np.ndarray,
    mask_kind: str,
    soft_k: float,
    src_bounds: tuple[float, float, float, float],
    spacing: float,
    velocity_field: tuple[np.ndarray, np.ndarray] = None,
    batch_size: int = HYBRID_BATCH_SIZE,
):
    candidate_xy = make_coarse_source_grid(src_bounds, spacing)
    x_mean, x_std, y_mean, y_std = norms
    iy_t = torch.as_tensor(iy, device=device, dtype=torch.long)
    ix_t = torch.as_tensor(ix, device=device, dtype=torch.long)
    ux, uy = (None, None) if velocity_field is None else velocity_field
    candidate_obs = []

    with torch.no_grad():
        for start in range(0, len(candidate_xy), int(batch_size)):
            coords = candidate_xy[start : start + int(batch_size)]
            x_np = np.concatenate(
                [
                    build_input_field(
                        float(sx),
                        float(sy),
                        NY,
                        NX,
                        mask_kind,
                        float(SRC_SIZE_FIXED),
                        float(SIGMA_FIXED),
                        float(soft_k),
                        ux=ux,
                        uy=uy,
                    )
                    for sx, sy in coords
                ],
                axis=0,
            )
            x_t = torch.from_numpy(x_np).to(device)
            y_n = model((x_t - x_mean) / (x_std + 1e-12))
            y = y_n * (y_std + 1e-12) + y_mean
            candidate_obs.append(y[:, 0, iy_t, ix_t].cpu().numpy())

    return candidate_xy, np.concatenate(candidate_obs, axis=0).astype(np.float32)


def run_inverse(
    model,
    norms,
    device,
    c_obs: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    mask_kind: str,
    soft_k: float,
    src_bounds: tuple[float, float, float, float],
    inv_steps: int,
    inv_lr: float,
    inv_restarts: int,
    inv_seed: int,
    inverse_method: str = "hybrid",
    coarse_library: tuple[np.ndarray, np.ndarray] = None,
    velocity_field: tuple[np.ndarray, np.ndarray] = None,
):
    x_mean, x_std, y_mean, y_std = norms
    x_min, x_max, y_min, y_max = src_bounds
    obs = c_obs[iy, ix].astype(np.float32)
    ux, uy = (None, None) if velocity_field is None else velocity_field
    common_kwargs = dict(
        model=model,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        iy=iy,
        ix=ix,
        obs=obs,
        ny=NY,
        nx=NX,
        Lx=Lx,
        Ly=Ly,
        mask_kind="gauss" if mask_kind == "gauss" else "soft_rect",
        src_size=float(SRC_SIZE_FIXED),
        sigma=float(SIGMA_FIXED),
        soft_k=float(soft_k),
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        steps=int(inv_steps),
        lr=float(inv_lr),
        ux_field=ux,
        uy_field=uy,
    )
    method_info = {"method": inverse_method}
    if inverse_method == "hybrid":
        if coarse_library is None:
            raise ValueError("Hybrid inverse requires a coarse candidate sensor library.")
        candidate_xy, candidate_obs = coarse_library
        coarse_losses = np.mean((candidate_obs - obs[None, :]) ** 2, axis=1)
        top_k = min(max(1, int(inv_restarts)), len(candidate_xy))
        top_idx = np.argsort(coarse_losses)[:top_k]
        initial_xy = candidate_xy[top_idx]
        best = inv._optimize_source_autodiff_from_initializations(
            initial_xy=initial_xy,
            **common_kwargs,
        )
        method_info.update(
            {
                "n_coarse_candidates": int(len(candidate_xy)),
                "top_k": int(top_k),
                "coarse_best_sx": float(initial_xy[0, 0]),
                "coarse_best_sy": float(initial_xy[0, 1]),
                "coarse_best_loss": float(coarse_losses[top_idx[0]]),
            }
        )
    else:
        best = inv._optimize_source_autodiff(
            n_restarts=int(inv_restarts),
            restart_seed=int(inv_seed),
            **common_kwargs,
        )
    est_sx = float(best["sx"])
    est_sy = float(best["sy"])
    c_est = predict_c(
        model,
        norms,
        device,
        est_sx,
        est_sy,
        NY,
        NX,
        mask_kind,
        float(SRC_SIZE_FIXED),
        float(SIGMA_FIXED),
        float(soft_k),
        velocity_field=velocity_field,
    )
    trace_df = pd.DataFrame(best["trace"], columns=["restart", "step", "sx", "sy", "obs_mse"])
    method_info["selected_initialization"] = int(best["restart"])
    return est_sx, est_sy, float(best["loss"]), c_est, trace_df, method_info


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="FNO Viewer (Forward + Inverse)", layout="wide")
apply_app_style()
PAPER_MODE = is_paper_mode()
requested_module = get_query_param("module", "Forward").strip().lower()
default_module = "Inverse" if requested_module.startswith("inv") else "Forward"

st.markdown(
    """
    <div class="app-hero">
      <div class="app-kicker">FNO surrogate model for indoor contaminant transport</div>
      <h1>Interactive prediction and source localization</h1>
      <div class="small-note">
        Real-time field prediction and sparse-sensor inverse analysis for the ANNEX 20 benchmark room.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    assets = load_model_and_norm(CKPT_PATH)
except Exception as e:
    st.error("Error while loading baseline checkpoint.")
    st.exception(e)
    st.stop()

model = assets["model"]
norms = assets["norms"]
device = assets["device"]
fixed_inlet_u = assets["fixed_inlet_u"]
ckpt_mask_kind = assets["mask_kind"]
ckpt_soft_k = assets["soft_k"]
src_bounds = assets["src_bounds"]
velocity_field = assets["velocity_field"]

with st.sidebar:
    st.markdown("### Model setup")
    st.metric("Inlet velocity [m/s]", f"{fixed_inlet_u:.3f}")
    ach = compute_ach(float(fixed_inlet_u))
    st.metric("ACH [1/h]", f"{ach:.2f}")
    st.divider()
    st.markdown("### Source representation")
    st.metric("Mask", "soft-rect" if ckpt_mask_kind == "soft_rect" else ckpt_mask_kind)
    st.metric("Sharpness k", f"{ckpt_soft_k:.0f}")
    if not PAPER_MODE:
        st.divider()
        st.write("Baseline checkpoint")
        st.code(CKPT_PATH, language="text")

module = st.radio(
    "Module",
    ["Forward", "Inverse"],
    horizontal=True,
    label_visibility="collapsed",
    index=0 if default_module == "Forward" else 1,
)

if module == "Forward":
    with st.sidebar:
        st.header("Forward setup")
        src_x = st.slider("Source x [m]", 0.5, 7.5, 4.375, step=0.001)
        src_y = st.slider("Source y [m]", 0.5, 2.5, 1.125, step=0.001)

    C = predict_c(
        model,
        norms,
        device,
        float(src_x),
        float(src_y),
        NY,
        NX,
        ckpt_mask_kind,
        float(SRC_SIZE_FIXED),
        float(SIGMA_FIXED),
        float(ckpt_soft_k),
        velocity_field=velocity_field,
    )
    Xg, Yg = make_xy_grids(NY, NX)
    outlet_mask = rect_mask(Xg, Yg, Lx - OUTLET_WIDTH_X, Lx, OUTLET_Y0, OUTLET_Y1)
    bz_mask = rect_mask(Xg, Yg, BZ_X0, BZ_X1, BZ_Y0, BZ_Y1)
    cre, c_out, c_bz = compute_cre(C, outlet_mask, bz_mask)

    st.subheader("Forward prediction")
    st.markdown(
        '<div class="small-note">The trained FNO surrogate maps the selected source location to the full steady-state concentration field.</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1.7, 1.0], gap="large")
    with col1:
        fig_f, ax_f = plt.subplots(figsize=(PLOT_W, PLOT_H))
        im = ax_f.imshow(C, origin="lower", extent=[0, Lx, 0, Ly], aspect="auto", cmap="viridis")
        ax_f.set_title("C (Passive Scalar)")
        ax_f.set_xlabel("x [m]")
        ax_f.set_ylabel("y [m]")
        ax_f.set_box_aspect(Ly / Lx)
        plt.colorbar(im, ax=ax_f, fraction=0.040, pad=0.03)
        fig_f.subplots_adjust(left=0.06, right=0.985, bottom=0.14, top=0.90)
        st.pyplot(fig_f, clear_figure=True)

    with col2:
        st.markdown('<div class="panel-title">Field summary</div>', unsafe_allow_html=True)
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.metric("src_x [m]", f"{float(src_x):.3f}")
            st.metric("max C", f"{float(np.max(C)):.4f}")
            st.metric("CRE", f"{cre:.3f}" if np.isfinite(cre) else "nan")
        with mcol2:
            st.metric("src_y [m]", f"{float(src_y):.3f}")
            st.metric("mean C", f"{float(np.mean(C)):.4f}")
            st.metric("C_out (mean)", f"{c_out:.3f}" if np.isfinite(c_out) else "nan")

else:
    with st.sidebar:
        st.header("Inverse setup")
        inv_src_x = st.slider("GT source x [m]", 0.5, 7.5, 4.375, step=0.001)
        inv_src_y = st.slider("GT source y [m]", 0.5, 2.5, 1.125, step=0.001)
        sensor_mode = st.radio("Sensor layout", ["uniform", "self-define"], horizontal=True)
        if sensor_mode == "uniform":
            n_sensors = st.select_slider("Sensor number", options=[8, 12, 16, 24, 32, 48, 64], value=16)
        else:
            n_sensors = None
        inverse_method_label = st.selectbox(
            "Inverse method",
            ["Hybrid coarse-to-gradient", "Random-start gradient"],
            index=0,
        )
        inverse_method = "hybrid" if inverse_method_label.startswith("Hybrid") else "random_gradient"
        inv_steps = st.slider("Inverse steps", 50, 300, 150 if not PAPER_MODE else 100, 25)
        inv_lr = st.number_input("Inverse learning rate", min_value=0.001, max_value=0.2, value=0.05, step=0.005)
        if inverse_method == "hybrid":
            inv_restarts = st.slider("Coarse candidates refined (top-K)", 1, 8, HYBRID_TOP_K_DEFAULT, 1)
            inv_seed = 42
            st.caption(
                f"Global search: {HYBRID_COARSE_SPACING:.2f} m grid; "
                "the best candidates initialize continuous refinement."
            )
        else:
            inv_restarts = st.slider("Inverse restarts", 1, 8, 4 if not PAPER_MODE else 3, 1)
            inv_seed = st.number_input("Inverse random seed", min_value=0, max_value=99999, value=42, step=1)
        run_inverse_btn = st.button("Run inverse localization", type="primary")

    margin_x = DEFAULT_MARGIN_X
    margin_y = DEFAULT_MARGIN_Y

    C_inv_gt = predict_c(
        model,
        norms,
        device,
        float(inv_src_x),
        float(inv_src_y),
        NY,
        NX,
        ckpt_mask_kind,
        float(SRC_SIZE_FIXED),
        float(SIGMA_FIXED),
        float(ckpt_soft_k),
        velocity_field=velocity_field,
    )

    if "sensor_table" not in st.session_state:
        st.session_state.sensor_table = pd.DataFrame(columns=["x_m", "y_m"])
    if "picked_sensor_points" not in st.session_state:
        st.session_state.picked_sensor_points = pd.DataFrame(columns=["x_m", "y_m"])
    if "last_selection_signature" not in st.session_state:
        st.session_state.last_selection_signature = None

    sensor_table_key = f"{sensor_mode}_{n_sensors if n_sensors is not None else 'manual'}"
    if st.session_state.get("sensor_table_key") != sensor_table_key:
        st.session_state.sensor_table_key = sensor_table_key
        if sensor_mode == "uniform":
            st.session_state.sensor_table = default_sensor_table(int(n_sensors), int(margin_x), int(margin_y), "uniform")
        else:
            st.session_state.sensor_table = pd.DataFrame(columns=["x_m", "y_m"])
        st.session_state.picked_sensor_points = st.session_state.sensor_table.copy()
        st.session_state.last_selection_signature = None

    x0_int, x1_int, y0_int, y1_int = internal_bounds(int(margin_x), int(margin_y))

    st.subheader("Inverse source localization")
    st.markdown(
        '<div class="small-note">Sparse sensor observations are matched first by a coarse global search. The most promising source candidates then initialize continuous gradient-based refinement.</div>',
        unsafe_allow_html=True,
    )

    if sensor_mode == "self-define":
        st.markdown("**Sensor picking**")
        st.write("Click points one by one to add sensors. Click an already selected point again to remove it.")
        candidate_df = candidate_sensor_points(int(margin_x), int(margin_y))
        chart_df = candidate_df.copy()
        chart_df["selected"] = False
        if not st.session_state.picked_sensor_points.empty:
            selected_keys = {
                (round(float(r.x_m), 3), round(float(r.y_m), 3))
                for r in st.session_state.picked_sensor_points.itertuples(index=False)
            }
            chart_df["selected"] = [
                (round(float(x), 3), round(float(y), 3)) in selected_keys
                for x, y in zip(chart_df["x_m"], chart_df["y_m"])
            ]
        selector = alt.selection_point(name="picked", fields=["x_m", "y_m"], toggle=True, empty=True)
        chart = (
            alt.Chart(chart_df)
            .mark_circle(size=85)
            .encode(
                x=alt.X("x_m:Q", scale=alt.Scale(domain=[0, Lx]), title="x [m]"),
                y=alt.Y("y_m:Q", scale=alt.Scale(domain=[0, Ly]), title="y [m]"),
                color=alt.condition("datum.selected", alt.value("#d62728"), alt.value("#4c78a8")),
                opacity=alt.condition("datum.selected", alt.value(1.0), alt.value(0.28)),
                tooltip=["x_m:Q", "y_m:Q"],
            )
            .add_params(selector)
            .properties(height=260)
            .configure_axis(labelFontSize=10, titleFontSize=11)
            .configure_view(stroke=None)
        )
        event = st.altair_chart(chart, width="stretch", on_select="rerun", selection_mode="picked", key="sensor_picker")
        st.caption(f"GT source for reference: ({float(inv_src_x):.3f}, {float(inv_src_y):.3f})")
        picked_df = parse_point_selection(event.selection.picked if event and hasattr(event, "selection") else None)
        if not picked_df.empty:
            picked_df = picked_df.drop_duplicates().reset_index(drop=True)
            selection_signature = tuple(
                sorted((round(float(x), 3), round(float(y), 3)) for x, y in zip(picked_df["x_m"], picked_df["y_m"]))
            )
            if selection_signature != st.session_state.last_selection_signature:
                current = st.session_state.picked_sensor_points.copy()
                current_keys = [
                    (round(float(x), 3), round(float(y), 3))
                    for x, y in zip(current.get("x_m", []), current.get("y_m", []))
                ]
                for x, y in selection_signature:
                    if (x, y) in current_keys:
                        keep_mask = ~(
                            (current["x_m"].round(3) == round(x, 3))
                            & (current["y_m"].round(3) == round(y, 3))
                        )
                        current = current.loc[keep_mask].reset_index(drop=True)
                        current_keys = [
                            (round(float(cx), 3), round(float(cy), 3))
                            for cx, cy in zip(current.get("x_m", []), current.get("y_m", []))
                        ]
                    else:
                        current = pd.concat([current, pd.DataFrame([{"x_m": x, "y_m": y}])], ignore_index=True)
                        current_keys.append((x, y))
                st.session_state.picked_sensor_points = current.drop_duplicates().reset_index(drop=True)
                st.session_state.sensor_table = st.session_state.picked_sensor_points.copy()
                st.session_state.last_selection_signature = selection_signature
                st.rerun()
        elif not st.session_state.picked_sensor_points.empty:
            st.session_state.sensor_table = st.session_state.picked_sensor_points.copy()

        button_col1, button_col2 = st.columns(2)
        with button_col1:
            if st.button("Reset picked sensors"):
                st.session_state.picked_sensor_points = pd.DataFrame(columns=["x_m", "y_m"])
                st.session_state.sensor_table = pd.DataFrame(columns=["x_m", "y_m"])
                st.session_state.last_selection_signature = None
                st.rerun()
        with button_col2:
            st.caption(f"Selected {len(st.session_state.sensor_table)} points")

        edited_df = st.data_editor(
            st.session_state.sensor_table,
            num_rows="dynamic",
            width="stretch",
            height=220,
            key="sensor_editor",
        )
        sensor_df = edited_df.copy()
        sensor_df["x_m"] = sensor_df["x_m"].clip(x0_int, x1_int)
        sensor_df["y_m"] = sensor_df["y_m"].clip(y0_int, y1_int)
        st.session_state.sensor_table = sensor_df
    else:
        sensor_df = default_sensor_table(int(n_sensors), int(margin_x), int(margin_y), "uniform")
        with st.expander("Uniform sensor coordinates", expanded=False):
            st.dataframe(sensor_df, width="stretch", hide_index=True, height=220)

    if len(sensor_df) == 0:
        st.info("Pick at least one sensor point to run inverse localization.")
        st.stop()

    iy, ix = nearest_grid_indices(sensor_df)
    sensor_xs = np.linspace(0.0, Lx, NX, dtype=np.float32)[ix]
    sensor_ys = np.linspace(0.0, Ly, NY, dtype=np.float32)[iy]

    fig_s, ax_s = plt.subplots(figsize=(PLOT_W, PLOT_H))
    im_s = ax_s.imshow(C_inv_gt, origin="lower", extent=[0, Lx, 0, Ly], aspect="auto", cmap="viridis")
    ax_s.scatter(sensor_xs, sensor_ys, s=26, c="white", edgecolors="black", linewidths=0.7, zorder=3)
    ax_s.scatter([float(inv_src_x)], [float(inv_src_y)], marker="x", s=120, c="red", linewidths=2.0, zorder=4)
    draw_rect(ax_s, x0_int, x1_int, y0_int, y1_int, linestyle="--", linewidth=1.2)
    ax_s.set_title("GT field and sensor locations")
    ax_s.set_xlabel("x [m]")
    ax_s.set_ylabel("y [m]")
    ax_s.set_box_aspect(Ly / Lx)
    plt.colorbar(im_s, ax=ax_s, fraction=0.040, pad=0.03)
    fig_s.subplots_adjust(left=0.06, right=0.985, bottom=0.14, top=0.90)
    preview_expanded = not PAPER_MODE
    with st.expander("Preview GT field and sensor layout", expanded=preview_expanded):
        st.pyplot(fig_s, clear_figure=True)

    if run_inverse_btn or PAPER_MODE:
        with st.spinner("Running global search and continuous refinement..." if inverse_method == "hybrid" else "Running inverse optimization..."):
            coarse_library = None
            if inverse_method == "hybrid":
                if "inverse_coarse_library_cache" not in st.session_state:
                    st.session_state.inverse_coarse_library_cache = {}
                library_key = (
                    CKPT_PATH,
                    ckpt_mask_kind,
                    round(float(ckpt_soft_k), 6),
                    round(float(HYBRID_COARSE_SPACING), 6),
                    tuple(int(v) for v in iy),
                    tuple(int(v) for v in ix),
                )
                if library_key not in st.session_state.inverse_coarse_library_cache:
                    st.session_state.inverse_coarse_library_cache[library_key] = build_coarse_sensor_library(
                        model=model,
                        norms=norms,
                        device=device,
                        iy=iy,
                        ix=ix,
                        mask_kind=ckpt_mask_kind,
                        soft_k=ckpt_soft_k,
                        src_bounds=src_bounds,
                        spacing=HYBRID_COARSE_SPACING,
                        velocity_field=velocity_field,
                    )
                coarse_library = st.session_state.inverse_coarse_library_cache[library_key]

            est_sx, est_sy, obs_mse, C_inv, trace_df, method_info = run_inverse(
                model=model,
                norms=norms,
                device=device,
                c_obs=C_inv_gt,
                iy=iy,
                ix=ix,
                mask_kind=ckpt_mask_kind,
                soft_k=ckpt_soft_k,
                src_bounds=src_bounds,
                inv_steps=int(inv_steps),
                inv_lr=float(inv_lr),
                inv_restarts=int(inv_restarts),
                inv_seed=int(inv_seed),
                inverse_method=inverse_method,
                coarse_library=coarse_library,
                velocity_field=velocity_field,
            )
        loc_error = float(np.sqrt((est_sx - float(inv_src_x)) ** 2 + (est_sy - float(inv_src_y)) ** 2))
        err_field = C_inv - C_inv_gt

        cmin = 0.0
        cmax = max(float(np.max(C_inv_gt)), float(np.max(C_inv)))
        emax = float(np.max(np.abs(err_field)))

        st.markdown("#### Inverse result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated src_x [m]", f"{est_sx:.3f}")
        m2.metric("Estimated src_y [m]", f"{est_sy:.3f}")
        m3.metric("Localization error [m]", f"{loc_error:.3f}")
        m4, m5, m6 = st.columns(3)
        m4.metric("Observation MSE", f"{obs_mse:.5f}")
        m5.metric("Sensor count", f"{len(sensor_df)}")
        m6.metric("Inverse method", "Hybrid" if inverse_method == "hybrid" else "Random gradient")
        if inverse_method == "hybrid":
            st.caption(
                f"Coarse search evaluated {method_info['n_coarse_candidates']} candidates at "
                f"{HYBRID_COARSE_SPACING:.2f} m spacing and refined the best {method_info['top_k']}. "
                f"Best coarse initialization: ({method_info['coarse_best_sx']:.3f}, "
                f"{method_info['coarse_best_sy']:.3f}) m."
            )

        plot_col, loss_col = st.columns([1.75, 0.95], gap="large")

        fig_inv, axes = plt.subplots(1, 3, figsize=(PLOT_W, PLOT_H))
        im0 = axes[0].imshow(C_inv_gt, origin="lower", extent=[0, Lx, 0, Ly], aspect="auto", cmap="viridis", vmin=cmin, vmax=cmax)
        axes[0].scatter(sensor_xs, sensor_ys, s=22, c="white", edgecolors="black", linewidths=0.7)
        axes[0].scatter([float(inv_src_x)], [float(inv_src_y)], marker="x", s=120, c="red", linewidths=2.0)
        axes[0].set_title("GT")

        axes[1].imshow(C_inv, origin="lower", extent=[0, Lx, 0, Ly], aspect="auto", cmap="viridis", vmin=cmin, vmax=cmax)
        axes[1].scatter(sensor_xs, sensor_ys, s=22, c="white", edgecolors="black", linewidths=0.7)
        axes[1].scatter([est_sx], [est_sy], marker="x", s=120, c="black", linewidths=2.0)
        axes[1].set_title("Inverse prediction")

        im2 = axes[2].imshow(err_field, origin="lower", extent=[0, Lx, 0, Ly], aspect="auto", cmap="coolwarm", vmin=-emax, vmax=emax)
        axes[2].set_title("Prediction - GT")

        for ax in axes:
            ax.set_xlim(0, Lx)
            ax.set_ylim(0, Ly)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_box_aspect(Ly / Lx)

        cax1 = fig_inv.add_axes([0.10, 0.915, 0.34, 0.025])
        cb1 = fig_inv.colorbar(im0, cax=cax1, orientation="horizontal")
        cb1.set_label("C")
        cax2 = fig_inv.add_axes([0.57, 0.915, 0.34, 0.025])
        cb2 = fig_inv.colorbar(im2, cax=cax2, orientation="horizontal")
        cb2.set_label("Error")
        fig_inv.subplots_adjust(left=0.055, right=0.99, bottom=0.14, top=0.86, wspace=0.12)
        with plot_col:
            st.pyplot(fig_inv, clear_figure=True)

        if not trace_df.empty:
            best_restart = int(trace_df.loc[trace_df.groupby("restart")["obs_mse"].min().idxmin(), "restart"])
            best_df = trace_df[trace_df["restart"] == best_restart].sort_values("step")
            fig_loss, ax_loss = plt.subplots(figsize=(LOSS_W, LOSS_H))
            ax_loss.plot(best_df["step"], best_df["obs_mse"], color="#1f77b4", linewidth=2.0)
            ax_loss.set_xlabel("Optimization step")
            ax_loss.set_ylabel("Observation MSE")
            if inverse_method == "hybrid":
                ax_loss.set_title(f"Selected coarse candidate rank = {best_restart + 1}")
            else:
                ax_loss.set_title(f"Best inverse restart = {best_restart}")
            ax_loss.grid(True, linestyle="--", alpha=0.4)
            fig_loss.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.88)
            with loss_col:
                st.pyplot(fig_loss, clear_figure=True)
