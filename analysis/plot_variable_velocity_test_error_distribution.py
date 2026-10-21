#!/usr/bin/env python3
"""Plot concentration relative-L2 error distributions at held-out inlet velocities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=project_root
        / "analysis/results/fno_concentration_variable_u_hpc_1932478/"
        "fno_concentration_variable_u_test_cases.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "analysis/figures/variable_velocity_generalization",
    )
    parser.add_argument(
        "--stem",
        default="variable_velocity_test_error_distribution",
        help="Output filename stem; the original figure is preserved by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.cases)
    required = {"case_id", "inlet_u", "src_x", "src_y", "concentration_rel_l2"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    velocities = np.sort(data["inlet_u"].unique())
    if len(velocities) != 4:
        raise ValueError(f"Expected four held-out inlet velocities, found {len(velocities)}")

    groups = [
        data.loc[np.isclose(data["inlet_u"], velocity), "concentration_rel_l2"].to_numpy()
        for velocity in velocities
    ]
    if any(len(group) != 396 for group in groups):
        raise ValueError(f"Expected 396 cases per velocity, found {[len(group) for group in groups]}")
    if any(np.any(group <= 0) for group in groups):
        raise ValueError("All relative L2 errors must be positive for logarithmic plotting")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.stem
    source_columns = ["case_id", "inlet_u", "src_x", "src_y", "concentration_rel_l2"]
    data[source_columns].to_csv(stem.with_name(f"{args.stem}_source_data.csv"), index=False)
    summary = (
        data.groupby("inlet_u")["concentration_rel_l2"]
        .agg(n="size", mean="mean", std="std", median="median", q1=lambda x: x.quantile(0.25),
             q3=lambda x: x.quantile(0.75), minimum="min", maximum="max")
        .reset_index()
    )
    summary.to_csv(stem.with_name(f"{args.stem}_summary.csv"), index=False)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    positions = np.arange(1, len(velocities) + 1)
    colors = mpl.colormaps["Blues"](np.linspace(0.35, 0.65, len(velocities)))
    rng = np.random.default_rng(20260915)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    box = ax.boxplot(
        groups,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#D55E00",
            "markeredgecolor": "white",
            "markeredgewidth": 0.6,
            "markersize": 5.0,
            "zorder": 5,
        },
        medianprops={"color": "#202020", "linewidth": 1.2},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#3D4D4B")
        patch.set_linewidth(0.85)
        patch.set_alpha(0.78)

    for position, group, color in zip(positions, groups, colors):
        jitter = rng.uniform(-0.19, 0.19, size=len(group))
        ax.scatter(
            position + jitter,
            group,
            s=7.0,
            color=color,
            edgecolors="none",
            alpha=0.27,
            rasterized=True,
            zorder=1,
        )

    ax.set_yscale("log")
    ax.set_xlim(0.55, 4.45)
    ax.set_ylim(0.04, 4.0)
    ax.set_xticks(positions, [f"{velocity:.3f}" for velocity in velocities])
    ax.set_xlabel(r"Held-out inlet velocity, $u_{\mathrm{in}}$ (m/s)")
    ax.set_ylabel(r"Relative $L_2$ error")
    ax.grid(axis="y", which="major", color="#D6D6D6", linewidth=0.60, alpha=0.72)
    ax.tick_params(direction="out", length=3.0, width=0.8)

    legend_handles = [
        Line2D([0], [0], color="#202020", lw=1.2, label="Median"),
        Line2D(
            [0], [0], marker="D", linestyle="none", markerfacecolor="#D55E00",
            markeredgecolor="white", markeredgewidth=0.6, markersize=5.0, label="Mean"
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", ncol=2, columnspacing=1.2, handlelength=1.7)
    ax.text(
        0.99,
        0.98,
        r"$n=396$ source locations per velocity",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#555555",
        fontsize=8.5,
    )

    fig.tight_layout(pad=0.8)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    audit = {
        "input_cases": str(args.cases.resolve()),
        "n_cases": int(len(data)),
        "test_inlet_velocities": [float(value) for value in velocities],
        "cases_per_velocity": [int(len(group)) for group in groups],
        "y_scale": "logarithmic",
        "scatter_jitter_seed": 20260915,
        "figure_claim": "Prediction-error distributions remain bounded across four held-out inlet velocities, with modestly higher central errors at the two larger velocities.",
        "figure_archetype": "single-panel quantitative distribution",
        "style_reference": "Original single-blue variable-velocity distribution figure",
        "outputs": [str(stem.with_suffix(suffix).resolve()) for suffix in (".svg", ".pdf", ".png", ".tiff")],
    }
    stem.with_name(f"{args.stem}_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Saved {stem}.svg/.pdf/.png/.tiff")


if __name__ == "__main__":
    main()
