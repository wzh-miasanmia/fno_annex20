import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def load_infer_module(project_root: Path):
    infer_path = project_root / "visualization" / "infer_fixed_u_srcpos_to_c.py"
    spec = importlib.util.spec_from_file_location("infer_mod", infer_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_case_row(cases_path: Path, inlet_u: float, src_x: float, src_y: float, tol: float = 1e-9):
    df = pd.read_csv(cases_path)
    hit = df[
        (np.abs(df["inlet_u"].astype(float) - float(inlet_u)) <= tol)
        & (np.abs(df["src_x"].astype(float) - float(src_x)) <= tol)
        & (np.abs(df["src_y"].astype(float) - float(src_y)) <= tol)
    ].copy()
    if len(hit) == 0:
        raise ValueError(
            f"No case found in {cases_path} for inlet_u={inlet_u:.6f}, src_x={src_x:.6f}, src_y={src_y:.6f}"
        )
    row = hit.iloc[0].copy().astype(object)
    row["case_id"] = str(row["case_id"]).replace(".0", "")
    return row


def normalize_mask_kind(mask_kind: str) -> str:
    mk = str(mask_kind).lower().strip().replace("-", "_")
    if mk in ["softrect", "soft_rect(sigmoid)", "softrect(sigmoid)"]:
        mk = "soft_rect"
    if mk == "gaussian":
        mk = "gauss"
    return mk


def predict_c_field(mod, ckpt_path: Path, src_x: float, src_y: float, Lx: float, Ly: float, nx: int, ny: int, device: torch.device):
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
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--workdir", type=str, default="analysis/data_u0p2_dense_969")
    ap.add_argument("--inlet_u", type=float, default=0.2)
    ap.add_argument("--src_x", type=float, default=4.625)
    ap.add_argument("--src_y", type=float, default=1.625)
    ap.add_argument("--Lx", type=float, default=9.0)
    ap.add_argument("--Ly", type=float, default=3.0)
    ap.add_argument("--nx_eval", type=int, default=384)
    ap.add_argument("--ny_eval", type=int, default=128)
    ap.add_argument("--out_path", type=str, required=True)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    workdir = root / args.workdir if not Path(args.workdir).is_absolute() else Path(args.workdir)
    mod = load_infer_module(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    row = find_case_row(workdir / "cases.csv", args.inlet_u, args.src_x, args.src_y)
    case_file = mod.resolve_case_file(workdir / "exports", str(row["case_id"]), float(row["inlet_u"]), float(row["src_x"]), float(row["src_y"]))
    df_case = mod.load_case_csv(case_file)

    ckpt_dir = root / "checkpoints" / "resolution_soft_rect"
    configs = [
        ("(a) 48x16", 48, 16, ckpt_dir / "dense969_u0p2_w48_m16_d4_soft_rect_sk30_rawtarget_48x16.pt"),
        ("(b) 96x32", 96, 32, ckpt_dir / "dense969_u0p2_w48_m16_d4_soft_rect_sk30_rawtarget_96x32.pt"),
        ("(c) 192x64", 192, 64, ckpt_dir / "dense969_u0p2_w48_m16_d4_soft_rect_sk30_rawtarget_192x64.pt"),
        ("(d) 288x96", 288, 96, ckpt_dir / "dense969_u0p2_w48_m16_d4_soft_rect_sk30_rawtarget_288x96.pt"),
    ]

    loaded = []
    for title, nx_native, ny_native, ckpt in configs:
        x_gt, y_gt, _, _, c_gt_native = mod.df_to_grid(df_case, Lx=args.Lx, Ly=args.Ly, nx=nx_native, ny=ny_native)
        x_pred, y_pred, _, _, _ = mod.df_to_grid(df_case, Lx=args.Lx, Ly=args.Ly, nx=args.nx_eval, ny=args.ny_eval)
        c_pred_high = predict_c_field(mod, ckpt, args.src_x, args.src_y, args.Lx, args.Ly, args.nx_eval, args.ny_eval, device)
        loaded.append((title, x_gt, y_gt, c_gt_native, x_pred, y_pred, c_pred_high))

    c_min = min(
        min(float(np.min(gt)) for _, _, _, gt, _, _, _ in loaded),
        min(float(np.min(pred)) for _, _, _, _, _, _, pred in loaded),
    )
    c_max = max(
        max(float(np.max(gt)) for _, _, _, gt, _, _, _ in loaded),
        max(float(np.max(pred)) for _, _, _, _, _, _, pred in loaded),
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(4, 2, figsize=(8.2, 13.2))
    axes[0, 0].set_title("GT", fontsize=12)
    axes[0, 1].set_title("Prediction at 384x128", fontsize=12)

    im_c = None
    for i, (title, x_gt, y_gt, gt_native, x_pred, y_pred, pred_high) in enumerate(loaded):
        axes[i, 0].text(-0.18, 0.5, title, transform=axes[i, 0].transAxes, ha="right", va="center", rotation=90, fontsize=12)
        im_c = plot_field(axes[i, 0], gt_native, x_gt, y_gt, vmin=c_min, vmax=c_max, cmap="viridis")
        plot_field(axes[i, 1], pred_high, x_pred, y_pred, vmin=c_min, vmax=c_max, cmap="viridis")

    fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.06, wspace=0.18, hspace=0.16)
    cax_c = fig.add_axes([0.17, 0.94, 0.70, 0.02])
    cbar_c = fig.colorbar(im_c, cax=cax_c, orientation="horizontal")
    cbar_c.set_label("C")

    out_path = root / args.out_path if not Path(args.out_path).is_absolute() else Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
