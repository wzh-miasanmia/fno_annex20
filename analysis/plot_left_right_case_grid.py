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



def load_case_triplet(mod, workdir: Path, ckpt_path: Path, inlet_u: float, src_x: float, src_y: float, Lx: float, Ly: float, nx: int, ny: int, device: torch.device):
    row = find_case_row(workdir / "cases.csv", inlet_u, src_x, src_y)
    case_file = mod.resolve_case_file(
        workdir / "exports",
        str(row["case_id"]),
        float(row["inlet_u"]),
        float(row["src_x"]),
        float(row["src_y"]),
    )
    df_gt = mod.load_case_csv(case_file)
    x, y, _, _, c_gt = mod.df_to_grid(df_gt, Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    c_pred = predict_c_field(mod, ckpt_path, src_x, src_y, Lx, Ly, nx, ny, device)
    c_err = c_pred - c_gt
    rel_l2 = mod.l2_rel(c_gt, c_pred)
    return {
        "case_id": str(row["case_id"]),
        "src_x": float(src_x),
        "src_y": float(src_y),
        "x": x,
        "y": y,
        "gt": c_gt,
        "pred": c_pred,
        "err": c_err,
        "rel_l2": float(rel_l2),
    }



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
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--inlet_u", type=float, default=0.2)
    ap.add_argument("--Lx", type=float, default=9.0)
    ap.add_argument("--Ly", type=float, default=3.0)
    ap.add_argument("--nx", type=int, default=128 * 3)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--out_path", type=str, required=True)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    workdir = root / args.workdir if not Path(args.workdir).is_absolute() else Path(args.workdir)
    ckpt_path = root / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    mod = load_infer_module(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cases = [
        (6.500, 0.875),
        (7.125, 2.125),
        (7.500, 2.500),
    ]
    loaded = [
        load_case_triplet(mod, workdir, ckpt_path, args.inlet_u, sx, sy, args.Lx, args.Ly, args.nx, args.ny, device)
        for sx, sy in cases
    ]

    c_min = min(float(min(np.min(item["gt"]), np.min(item["pred"]))) for item in loaded)
    c_max = max(float(max(np.max(item["gt"]), np.max(item["pred"]))) for item in loaded)
    e_max = max(float(np.max(np.abs(item["err"]))) for item in loaded)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, axes = plt.subplots(3, 3, figsize=(12.6, 6.7))
    col_titles = ["(a)", "(b)", "(c)"]
    row_labels = ["GT", "Prediction", "Error"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=13)
    for i, row_label in enumerate(row_labels):
        axes[i, 0].text(
            -0.22,
            0.5,
            row_label,
            transform=axes[i, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=13,
        )

    im_c = None
    im_e = None
    for j, item in enumerate(loaded):
        im_c = plot_field(axes[0, j], item["gt"], item["x"], item["y"], vmin=c_min, vmax=c_max, cmap="viridis")
        plot_field(axes[1, j], item["pred"], item["x"], item["y"], vmin=c_min, vmax=c_max, cmap="viridis")
        im_e = plot_field(axes[2, j], item["err"], item["x"], item["y"], vmin=-e_max, vmax=e_max, cmap="coolwarm")

    for i in range(2):
        for j in range(3):
            axes[i, j].set_xlabel("")
            axes[i, j].tick_params(labelbottom=False)

    fig.subplots_adjust(left=0.10, right=0.96, top=0.84, bottom=0.09, wspace=0.18, hspace=0.03)
    cax_c = fig.add_axes([0.10, 0.88, 0.35, 0.022])
    cax_e = fig.add_axes([0.57, 0.88, 0.35, 0.022])
    cbar_c = fig.colorbar(im_c, cax=cax_c, orientation="horizontal")
    cbar_c.set_label("C")
    cbar_e = fig.colorbar(im_e, cax=cax_e, orientation="horizontal")
    cbar_e.set_label("Error")

    out_path = root / args.out_path if not Path(args.out_path).is_absolute() else Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
