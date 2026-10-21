import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import griddata


def safe_float_tag(x: float) -> str:
    return f"{x:.4f}"


def normalize_case_id(case_id) -> str:
    text = str(case_id).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


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
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {case_path}. Got: {list(df.columns)}")
    return df[required].copy()


def df_to_grid(df: pd.DataFrame, Lx: float = 9.0, Ly: float = 3.0, nx: int = 128 * 3, ny: int = 128):
    x = np.linspace(0.0, Lx, nx, dtype=np.float32)
    y = np.linspace(0.0, Ly, ny, dtype=np.float32)
    xg, yg = np.meshgrid(x, y)
    pts = df[["X (m)", "Y (m)"]].to_numpy(dtype=np.float64)
    cc = df["Passive Scalar"].to_numpy(dtype=np.float64)
    lin = griddata(pts, cc, (xg, yg), method="linear")
    near = griddata(pts, cc, (xg, yg), method="nearest")
    c = np.where(np.isnan(lin), near, lin).astype(np.float32)
    return x, y, c


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


def idw_weights(train_xy: np.ndarray, query_xy: np.ndarray, k: int, power: float, eps: float = 1e-12):
    d2 = np.sum((train_xy - query_xy[None, :]) ** 2, axis=1)
    order = np.argsort(d2)
    idx = order[: min(k, len(order))]
    d = np.sqrt(d2[idx])
    if np.any(d <= eps):
        exact = np.argmin(d)
        w = np.zeros_like(d, dtype=np.float32)
        w[exact] = 1.0
        return idx, w
    w = 1.0 / np.power(d + eps, power)
    w = (w / np.sum(w)).astype(np.float32)
    return idx, w


def load_infer_module(project_root: Path):
    infer_path = project_root / "visualization" / "infer_fixed_u_srcpos_to_c.py"
    spec = importlib.util.spec_from_file_location("infer_mod", infer_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalize_mask_kind(mask_kind: str) -> str:
    mk = str(mask_kind).lower().strip().replace("-", "_")
    if mk in ["softrect", "soft_rect(sigmoid)", "softrect(sigmoid)"]:
        mk = "soft_rect"
    if mk == "gaussian":
        mk = "gauss"
    return mk


def predict_c_field_fno(mod, ckpt_path: Path, src_x: float, src_y: float, Lx: float, Ly: float, nx: int, ny: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt.get("meta", {})
    model = mod.FNO2D(
        in_ch=int(meta.get("in_ch", 3)),
        out_ch=int(meta.get("out_ch", 1)),
        width=int(meta.get("width", 48)),
        modes1=int(meta.get("modes1", 16)),
        modes2=int(meta.get("modes2", 16)),
        depth=int(meta.get("depth", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_in = mod.build_input_field(
        sx=src_x,
        sy=src_y,
        ny=ny,
        nx=nx,
        Lx=Lx,
        Ly=Ly,
        mask_kind=normalize_mask_kind(meta.get("mask_kind", "rect")),
        src_size=float(meta.get("src_size", 0.20)),
        sigma=float(meta.get("sigma", 0.12)),
        soft_k=(80.0 if meta.get("soft_k", None) is None else float(meta["soft_k"])),
    )
    x_t = torch.from_numpy(x_in).to(device)
    x_mean = ckpt["x_mean"].to(device)
    x_std = ckpt["x_std"].to(device)
    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)

    with torch.no_grad():
        pred_n = model((x_t - x_mean) / (x_std + 1e-12))
        pred = pred_n * (y_std + 1e-12) + y_mean
    return pred.cpu().numpy()[0, 0]


def plot_field(ax, field, x, y, vmin=None, vmax=None, cmap=None):
    im = ax.imshow(
        field,
        origin="lower",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_ylabel("y (m)")
    return im


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--workdir", type=str, default="data_u0p2_dense_969")
    ap.add_argument("--inlet_u", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--src_x", type=float, default=4.875)
    ap.add_argument("--src_y", type=float, default=2.0)
    ap.add_argument("--Lx", type=float, default=9.0)
    ap.add_argument("--Ly", type=float, default=3.0)
    ap.add_argument("--nx", type=int, default=128 * 3)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument(
        "--ckpt",
        type=str,
        default="paper_runs/baseline_dense969_u0p2_w48_m16_d4_soft_rect/checkpoints/dense969_u0p2_w48_m16_d4_soft_rect_sk30_rawtarget.pt",
    )
    ap.add_argument("--out_path", type=str, required=True)
    args = ap.parse_args()

    workdir = root / args.workdir if not Path(args.workdir).is_absolute() else Path(args.workdir)
    ckpt_path = root / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    infer_mod = load_infer_module(root)

    rows = pd.read_csv(workdir / "cases.csv")
    rows = rows[np.abs(rows["inlet_u"].astype(float) - float(args.inlet_u)) <= 1e-9].copy().reset_index(drop=True)
    train_idx, _, _ = split_indices_random(len(rows), val_ratio=0.15, test_ratio=0.15, seed=args.split_seed)
    train_rows = rows.iloc[train_idx].reset_index(drop=True)
    train_xy = train_rows[["src_x", "src_y"]].to_numpy(dtype=np.float32)

    hit = rows[
        (np.abs(rows["src_x"].astype(float) - float(args.src_x)) <= 1e-9)
        & (np.abs(rows["src_y"].astype(float) - float(args.src_y)) <= 1e-9)
    ].copy()
    if len(hit) == 0:
        raise ValueError("Requested case not found in dataset.")
    row = hit.iloc[0]
    case_id = normalize_case_id(row["case_id"])
    exports_dir = workdir / "exports"
    csv_path = resolve_case_file(exports_dir, case_id, float(row["inlet_u"]), float(row["src_x"]), float(row["src_y"]))
    x, y, c_gt = df_to_grid(load_case_csv(csv_path), Lx=args.Lx, Ly=args.Ly, nx=args.nx, ny=args.ny)

    idx, w = idw_weights(train_xy, np.array([args.src_x, args.src_y], dtype=np.float32), k=args.k, power=args.power)
    neighbor_fields = []
    for neighbor_idx in idx:
        train_row = train_rows.iloc[int(neighbor_idx)]
        train_case_id = normalize_case_id(train_row["case_id"])
        train_csv = resolve_case_file(
            exports_dir,
            train_case_id,
            float(train_row["inlet_u"]),
            float(train_row["src_x"]),
            float(train_row["src_y"]),
        )
        _, _, c = df_to_grid(load_case_csv(train_csv), Lx=args.Lx, Ly=args.Ly, nx=args.nx, ny=args.ny)
        neighbor_fields.append(c.astype(np.float32))
    c_interp = np.tensordot(w, np.stack(neighbor_fields, axis=0), axes=(0, 0)).astype(np.float32)
    c_fno = predict_c_field_fno(infer_mod, ckpt_path, args.src_x, args.src_y, args.Lx, args.Ly, args.nx, args.ny)

    c_min = float(min(np.min(c_gt), np.min(c_interp), np.min(c_fno)))
    c_max = float(max(np.max(c_gt), np.max(c_interp), np.max(c_fno)))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, axes = plt.subplots(3, 1, figsize=(5.0, 4.7))
    row_labels = ["GT", "Interpolation", "FNO"]
    im = plot_field(axes[0], c_gt, x, y, vmin=c_min, vmax=c_max, cmap="viridis")
    plot_field(axes[1], c_interp, x, y, vmin=c_min, vmax=c_max, cmap="viridis")
    plot_field(axes[2], c_fno, x, y, vmin=c_min, vmax=c_max, cmap="viridis")

    for i, ax in enumerate(axes):
        ax.text(
            -0.20,
            0.5,
            row_labels[i],
            transform=ax.transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=12,
        )
        if i < len(axes) - 1:
            ax.tick_params(labelbottom=False)

    fig.subplots_adjust(left=0.16, right=0.79, top=0.98, bottom=0.12, hspace=0.02)
    fig.supxlabel("x (m)", y=0.035, fontsize=11)
    fig.canvas.draw()
    top = axes[0].get_position()
    bottom = axes[-1].get_position()
    cax = fig.add_axes([0.89, bottom.y0, 0.024, top.y1 - bottom.y0])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("C")

    out_path = root / args.out_path if not Path(args.out_path).is_absolute() else Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
