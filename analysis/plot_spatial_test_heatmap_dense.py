import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_csv", type=str, required=True)
    ap.add_argument("--metric", type=str, default="rel_l2_c")
    ap.add_argument("--out_path", type=str, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.case_csv)
    if not {"src_x", "src_y", args.metric}.issubset(df.columns):
        raise ValueError(f"Missing required columns for metric={args.metric}")

    x_obs = df["src_x"].astype(float).to_numpy()
    y_obs = df["src_y"].astype(float).to_numpy()
    z_obs = df[args.metric].astype(float).to_numpy()

    x_full = np.array(sorted(pd.unique(df["src_x"].astype(float))), dtype=np.float64)
    y_step = 0.125
    y_full = np.arange(0.5, 2.5 + 0.5 * y_step, y_step, dtype=np.float64)

    xg, yg = np.meshgrid(x_full, y_full)
    pts = np.column_stack([x_obs, y_obs])
    z_lin = griddata(pts, z_obs, (xg, yg), method="linear")
    z_near = griddata(pts, z_obs, (xg, yg), method="nearest")
    z = np.where(np.isnan(z_lin), z_near, z_lin)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    im = ax.imshow(
        z,
        origin="lower",
        extent=[float(x_full.min()), float(x_full.max()), float(y_full.min()), float(y_full.max())],
        aspect="auto",
        cmap="viridis",
    )
    ax.set_xlabel(r"$x_s$ (m)")
    ax.set_ylabel(r"$y_s$ (m)")
    ax.set_xlim(float(x_full.min()), float(x_full.max()))
    ax.set_ylim(float(y_full.min()), float(y_full.max()))
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(args.metric)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
