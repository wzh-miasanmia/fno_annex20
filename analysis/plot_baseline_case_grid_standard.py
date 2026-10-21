#!/usr/bin/env python3
"""Redraw the three-case baseline figure using the current manuscript style."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import plot_baseline_case_grid as baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=384)
    parser.add_argument("--ny", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = baseline.load_infer_module(args.project_root.resolve())
    specifications = [
        ("Near inlet", 0.875, 1.000),
        ("Central", 4.875, 2.000),
        ("Near outlet", 6.875, 1.250),
    ]
    fields = [
        baseline.load_case_triplet(
            module, args.workdir.resolve(), args.checkpoint.resolve(), 0.2,
            sx, sy, 9.0, 3.0, args.nx, args.ny, device,
        )
        for _, sx, sy in specifications
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "region": name,
            "case_id": item["case_id"],
            "src_x": sx,
            "src_y": sy,
            "relative_l2": item["rel_l2"],
        }
        for (name, sx, sy), item in zip(specifications, fields)
    ]).to_csv(args.output_dir / "baseline_selected_cases.csv", index=False)
    np.savez_compressed(
        args.output_dir / "baseline_gt_pred_grid_source_data.npz",
        x=fields[0]["x"], y=fields[0]["y"],
        labels=np.asarray([name for name, _, _ in specifications]),
        case_ids=np.asarray([item["case_id"] for item in fields]),
        src_x=np.asarray([sx for _, sx, _ in specifications]),
        src_y=np.asarray([sy for _, _, sy in specifications]),
        ground_truth=np.stack([item["gt"] for item in fields]),
        prediction=np.stack([item["pred"] for item in fields]),
        signed_error=np.stack([item["err"] for item in fields]),
        relative_l2=np.asarray([item["rel_l2"] for item in fields]),
    )

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })

    concentration_min = min(float(min(item["gt"].min(), item["pred"].min())) for item in fields)
    concentration_min = min(0.0, concentration_min)
    concentration_max = max(float(max(item["gt"].max(), item["pred"].max())) for item in fields)
    error_max = max(float(np.abs(item["err"]).max()) for item in fields)
    fig, axes = plt.subplots(4, 3, figsize=(12.6, 8.6))
    for j, ((name, _, _), item) in enumerate(zip(specifications, fields)):
        axes[0, j].set_title(f"({chr(97 + j)})  {name}", fontsize=13)
    for i, label in enumerate(["CFD", "FNO", "Error", "Source-height profile"]):
        axes[i, 0].text(-0.22, 0.5, label, transform=axes[i, 0].transAxes,
                        ha="right", va="center", rotation=90, fontsize=13)

    extent = [0.0, 9.0, 0.0, 3.0]
    image_c = image_e = None
    profile_handles = None
    for j, ((_, sx, sy), item) in enumerate(zip(specifications, fields)):
        image_c = axes[0, j].imshow(item["gt"], origin="lower", extent=extent, aspect="equal",
                                    cmap="viridis", vmin=concentration_min, vmax=concentration_max)
        axes[1, j].imshow(item["pred"], origin="lower", extent=extent, aspect="equal",
                          cmap="viridis", vmin=concentration_min, vmax=concentration_max)
        image_e = axes[2, j].imshow(item["err"], origin="lower", extent=extent, aspect="equal",
                                    cmap="coolwarm", vmin=-error_max, vmax=error_max)
        for i in range(3):
            axes[i, j].plot(sx, sy, marker="x", color="white", markersize=5.5,
                            markeredgewidth=1.1, zorder=4)
            axes[i, j].set_ylabel("$y$ (m)")
        profile_index = int(np.argmin(np.abs(item["y"] - sy)))
        line_gt, = axes[3, j].plot(item["x"], item["gt"][profile_index], color="#1f77b4",
                                   lw=1.8, label="CFD")
        line_pred, = axes[3, j].plot(item["x"], item["pred"][profile_index], color="#ff7f0e",
                                     lw=1.8, label="FNO")
        if profile_handles is None:
            profile_handles = (line_gt, line_pred)
        axes[3, j].axvline(sx, color="0.45", lw=0.8, ls=":")
        axes[3, j].text(0.03, 0.92, rf"$y={sy:.2f}$ m; relative $L_2={item['rel_l2']:.3f}$",
                        transform=axes[3, j].transAxes, ha="left", va="top", fontsize=9)
        axes[3, j].set_xlabel("$x$ (m)")
        axes[3, j].set_ylabel("$C$")
        axes[3, j].grid(True, linestyle="--", linewidth=0.6, alpha=0.30)

    for i in range(3):
        for j in range(3):
            axes[i, j].set_xlabel("")
            axes[i, j].tick_params(labelbottom=False)
    fig.legend(profile_handles, ["CFD", "FNO"], loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.53, 0.008))
    fig.subplots_adjust(left=0.105, right=0.96, top=0.82, bottom=0.10, wspace=0.18, hspace=0.12)
    cax_c = fig.add_axes([0.105, 0.90, 0.35, 0.022])
    cax_e = fig.add_axes([0.57, 0.90, 0.35, 0.022])
    cbar_c = fig.colorbar(image_c, cax=cax_c, orientation="horizontal")
    cbar_c.set_label("Concentration, $C$", labelpad=4)
    cbar_c.ax.xaxis.set_label_position("top")
    cbar_e = fig.colorbar(image_e, cax=cax_e, orientation="horizontal")
    cbar_e.set_label(r"Prediction error, $\widehat{C}-C_{\mathrm{CFD}}$", labelpad=4)
    cbar_e.ax.xaxis.set_label_position("top")

    stem = args.output_dir / "baseline_gt_pred_grid_shared_colorbars_redrawn"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.pdf/.png/.tiff")


if __name__ == "__main__":
    main()
