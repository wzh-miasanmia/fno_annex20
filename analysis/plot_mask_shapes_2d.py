from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_rect(Xg: np.ndarray, Yg: np.ndarray, src_x: float, src_y: float, src_size: float) -> np.ndarray:
    hx = 0.5 * src_size
    hy = 0.5 * src_size
    return (
        (Xg >= (src_x - hx)) & (Xg <= (src_x + hx)) &
        (Yg >= (src_y - hy)) & (Yg <= (src_y + hy))
    ).astype(np.float32)


def build_gauss(Xg: np.ndarray, Yg: np.ndarray, src_x: float, src_y: float, sigma: float) -> np.ndarray:
    r2 = (Xg - src_x) ** 2 + (Yg - src_y) ** 2
    return np.exp(-0.5 * r2 / (sigma ** 2)).astype(np.float32)


def build_soft_rect(
    Xg: np.ndarray,
    Yg: np.ndarray,
    src_x: float,
    src_y: float,
    src_size: float,
    soft_k: float,
) -> np.ndarray:
    hx = 0.5 * src_size
    hy = 0.5 * src_size
    dx = np.abs(Xg - src_x) - hx
    dy = np.abs(Yg - src_y) - hy
    sdf = np.maximum(dx, dy)
    z = np.clip(soft_k * sdf, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(z))).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot 2D source-mask shape comparison for Rect / Gaussian / Soft-rect.")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--Lx", type=float, default=9.0)
    ap.add_argument("--Ly", type=float, default=3.0)
    ap.add_argument("--nx", type=int, default=384)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--src_x", type=float, default=4.5)
    ap.add_argument("--src_y", type=float, default=1.5)
    ap.add_argument("--src_size", type=float, default=0.20)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--soft_k", type=float, default=30.0)
    ap.add_argument("--zoom_half_width", type=float, default=0.5)
    ap.add_argument("--zoom_half_height", type=float, default=0.5)
    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--out_pdf", type=str, default="")
    ap.add_argument("--out_svg", type=str, default="")
    ap.add_argument("--out_tiff", type=str, default="")
    args = ap.parse_args()

    x = np.linspace(0.0, args.Lx, args.nx, dtype=np.float32)
    y = np.linspace(0.0, args.Ly, args.ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(x, y)

    fields = [
        ("Binary", build_rect(Xg, Yg, args.src_x, args.src_y, args.src_size)),
        ("Gaussian", build_gauss(Xg, Yg, args.src_x, args.src_y, args.sigma)),
        ("Soft rectangular", build_soft_rect(Xg, Yg, args.src_x, args.src_y, args.src_size, args.soft_k)),
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "axes.linewidth": 0.8,
        }
    )

    # Designed for a final single-column width of approximately 89 mm.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), sharey=True, constrained_layout=True)
    im = None
    for ax, (title, field) in zip(axes, fields):
        im = ax.imshow(
            field,
            origin="lower",
            extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlim(args.src_x - args.zoom_half_width, args.src_x + args.zoom_half_width)
        ax.set_ylim(args.src_y - args.zoom_half_height, args.src_y + args.zoom_half_height)
        ax.set_xlabel("x (m)")

    axes[0].set_ylabel("y (m)")
    for ax in axes[1:]:
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Mask value")

    out_png = Path(args.out_png)
    if not out_png.is_absolute():
        out_png = (root / out_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    if args.out_pdf:
        out_pdf = Path(args.out_pdf)
        if not out_pdf.is_absolute():
            out_pdf = (root / out_pdf).resolve()
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Saved: {out_pdf}")

    if args.out_svg:
        out_svg = Path(args.out_svg)
        if not out_svg.is_absolute():
            out_svg = (root / out_svg).resolve()
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg, bbox_inches="tight")
        print(f"Saved: {out_svg}")

    if args.out_tiff:
        out_tiff = Path(args.out_tiff)
        if not out_tiff.is_absolute():
            out_tiff = (root / out_tiff).resolve()
        out_tiff.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_tiff, dpi=600, bbox_inches="tight")
        print(f"Saved: {out_tiff}")

    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
