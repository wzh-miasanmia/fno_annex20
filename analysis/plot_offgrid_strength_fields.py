#!/usr/bin/env python3
"""Publication figure for off-grid FNO predictions at three release strengths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path(__file__).resolve().parent / "results/source_strength_validation/source_strength_fno_audit.json",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(__file__).resolve().parent / "results/source_strength_validation/source_strength_fno_case_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures/offgrid_strength_forward",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root / "analysis"))
    import evaluate_source_strength_scaled_fno as evaluation

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    metrics = pd.read_csv(args.metrics)
    location_error = metrics.groupby(["src_x", "src_y"])["rel_l2_c"].mean().sort_values()
    sx, sy = location_error.index[len(location_error) // 2]
    chosen = metrics[
        np.isclose(metrics["src_x"], sx) & np.isclose(metrics["src_y"], sy)
    ].sort_values("source_strength")
    if chosen["source_strength"].tolist() != [0.5, 1.0, 2.0]:
        raise RuntimeError("The representative location does not contain q=0.5, 1, and 2")

    validation_root = Path(audit["validation_root"])
    checkpoint_path = Path(audit["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state, meta = checkpoint["model_state"], checkpoint.get("meta", {})
    arch = evaluation.infer_arch_from_state_dict(state)
    model = evaluation.FNO2D(
        in_ch=int(meta.get("in_ch", arch["in_ch"])),
        out_ch=int(meta.get("out_ch", arch["out_ch"])),
        width=int(meta.get("width", arch["width"])),
        modes1=int(meta.get("modes1", arch["modes1"])),
        modes2=int(meta.get("modes2", arch["modes2"])),
        depth=int(meta.get("depth", arch["depth"])),
    )
    model.load_state_dict(state)
    model.eval()
    x_mean, x_std = checkpoint["x_mean"], checkpoint["x_std"]
    y_mean, y_std = checkpoint["y_mean"], checkpoint["y_std"]
    lx, ly = float(meta.get("Lx", 9.0)), float(meta.get("Ly", 3.0))
    nx, ny = int(meta.get("nx", 384)), int(meta.get("ny", 128))

    input_field = evaluation.build_input_field(
        sx=float(sx), sy=float(sy), ny=ny, nx=nx, Lx=lx, Ly=ly,
        mask_kind=str(meta.get("mask_kind", "soft_rect")),
        src_size=float(meta.get("src_size", 0.2)),
        sigma=float(meta.get("sigma", 0.12)),
        soft_k=float(meta.get("soft_k", 30.0)),
    )
    with torch.inference_mode():
        pred_n = model((torch.from_numpy(input_field) - x_mean) / (x_std + 1e-12))
        unit_prediction = (pred_n * (y_std + 1e-12) + y_mean).numpy()[0, 0]

    cases = pd.read_csv(validation_root / "cases.csv")
    fields = []
    for row in chosen.itertuples(index=False):
        q = float(row.source_strength)
        match = cases[cases["case_id"].astype(int) == int(row.case_id)]
        if len(match) != 1:
            raise RuntimeError(f"Could not uniquely identify case {row.case_id}")
        export = evaluation.case_file(validation_root / "exports", int(row.case_id))
        x, y, _, _, ground_truth = evaluation.df_to_grid(
            evaluation.load_case_csv(export), Lx=lx, Ly=ly, nx=nx, ny=ny
        )
        prediction = np.float32(q) * unit_prediction
        fields.append({
            "q": q,
            "case_id": int(row.case_id),
            "x": x,
            "y": y,
            "gt": ground_truth,
            "prediction": prediction,
            "error": prediction - ground_truth,
            "normalized_error": (prediction - ground_truth) / np.float32(q),
            "rel_l2": float(row.rel_l2_c),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "offgrid_strength_forward_source_data.npz",
        x=fields[0]["x"], y=fields[0]["y"], src_x=float(sx), src_y=float(sy),
        strengths=np.asarray([item["q"] for item in fields]),
        case_ids=np.asarray([item["case_id"] for item in fields]),
        ground_truth=np.stack([item["gt"] for item in fields]),
        prediction=np.stack([item["prediction"] for item in fields]),
        signed_error=np.stack([item["error"] for item in fields]),
        strength_normalized_error=np.stack([item["normalized_error"] for item in fields]),
        relative_l2=np.asarray([item["rel_l2"] for item in fields]),
    )
    chosen.assign(selection_rule="median of location-wise mean relative L2 error").to_csv(
        args.output_dir / "offgrid_strength_forward_selected_cases.csv", index=False
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

    concentration_min = min(float(min(item["gt"].min(), item["prediction"].min())) for item in fields)
    concentration_max = max(float(max(item["gt"].max(), item["prediction"].max())) for item in fields)
    error_max = max(float(np.abs(item["normalized_error"]).max()) for item in fields)
    fig, axes = plt.subplots(4, 3, figsize=(12.6, 8.6))
    profile_handles = None
    for j, item in enumerate(fields):
        axes[0, j].set_title(f"({chr(97 + j)})  $q={item['q']:g}$", fontsize=13)
    for i, label in enumerate(["CFD", "FNO", "Error / $q$", f"Profile at $y={sy:.2f}$ m"]):
        axes[i, 0].text(-0.22, 0.5, label, transform=axes[i, 0].transAxes,
                        ha="right", va="center", rotation=90, fontsize=13)

    image_c = image_e = None
    extent = [0.0, lx, 0.0, ly]
    for j, item in enumerate(fields):
        image_c = axes[0, j].imshow(item["gt"], origin="lower", extent=extent, aspect="equal",
                                    cmap="viridis", vmin=concentration_min, vmax=concentration_max)
        axes[1, j].imshow(item["prediction"], origin="lower", extent=extent, aspect="equal",
                          cmap="viridis", vmin=concentration_min, vmax=concentration_max)
        image_e = axes[2, j].imshow(item["normalized_error"], origin="lower", extent=extent, aspect="equal",
                                    cmap="coolwarm", vmin=-error_max, vmax=error_max)
        for i in range(3):
            axes[i, j].plot(sx, sy, marker="x", color="white", markersize=5.5,
                            markeredgewidth=1.1, zorder=4)
            axes[i, j].set_ylabel("$y$ (m)")
        profile_index = int(np.argmin(np.abs(item["y"] - sy)))
        line_gt, = axes[3, j].plot(item["x"], item["gt"][profile_index], color="#1f77b4", lw=1.8, label="CFD")
        line_pred, = axes[3, j].plot(item["x"], item["prediction"][profile_index], color="#ff7f0e", lw=1.8,
                                    label="FNO")
        if profile_handles is None:
            profile_handles = (line_gt, line_pred)
        axes[3, j].axvline(sx, color="0.45", lw=0.8, ls=":")
        axes[3, j].text(0.03, 0.92, rf"relative $L_2={item['rel_l2']:.3f}$",
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
    cbar_e.set_label(r"Strength-normalized error, $(\widehat{C}-C_{\mathrm{CFD}})/q$", labelpad=4)
    cbar_e.ax.xaxis.set_label_position("top")

    stem = args.output_dir / "offgrid_strength_gt_pred_grid_shared_colorbars"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Selected off-grid source: ({sx:.4f}, {sy:.4f}) m")
    print(f"Saved {stem}.pdf/.png/.tiff")


if __name__ == "__main__":
    main()
