#!/usr/bin/env python3
"""Plot joint off-grid source-location and release-strength inversion results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


METHOD = "FNO_coarse_gradient_joint"
Q_VALUES = (0.5, 1.0, 2.0)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.75,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--sensors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manuscript-figures", type=Path, required=True)
    args = parser.parse_args()

    configure_style()
    cases = pd.read_csv(args.cases)
    sensors = pd.read_csv(args.sensors)
    selected = cases.loc[
        (cases["method"] == METHOD)
        & (cases["noise"] == 0.0)
        & (cases["seed"] == 42)
        & (cases["q_true"].isin(Q_VALUES))
    ].copy()
    if len(selected) != 45:
        raise ValueError(f"Expected 45 plotted cases, found {len(selected)}")
    counts = selected.groupby("q_true")["case_id"].size().to_dict()
    if any(counts.get(q, 0) != 15 for q in Q_VALUES):
        raise ValueError(f"Expected 15 cases per strength, found {counts}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manuscript_figures.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "inverse_joint_strength_figure_source_data.csv", index=False)

    shared_max = float(np.ceil(selected["loc_error_m"].max() * 10.0) / 10.0)
    norm = Normalize(vmin=0.0, vmax=shared_max)
    cmap = mpl.colormaps["viridis"]

    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 72 / 25.4), sharex=True, sharey=True)
    scatter = None
    for index, (ax, q) in enumerate(zip(axes, Q_VALUES)):
        frame = selected.loc[np.isclose(selected["q_true"], q)].sort_values("case_id")
        for row in frame.itertuples(index=False):
            ax.plot(
                [row.x_true, row.x_est],
                [row.y_true, row.y_est],
                color="#666666",
                lw=0.65,
                alpha=0.75,
                zorder=1,
            )
        scatter = ax.scatter(
            frame["x_true"],
            frame["y_true"],
            c=frame["loc_error_m"],
            cmap=cmap,
            norm=norm,
            s=22,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.scatter(
            frame["x_est"],
            frame["y_est"],
            s=17,
            marker="o",
            facecolor="none",
            edgecolor="#202020",
            linewidth=0.55,
            zorder=4,
        )
        ax.scatter(
            sensors["x_m"],
            sensors["y_m"],
            marker="*",
            s=19,
            color="#777777",
            alpha=0.55,
            linewidth=0.25,
            zorder=2,
        )
        mean_loc = frame["loc_error_m"].mean()
        mean_q = frame["q_relative_error"].mean()
        ax.text(
            0.02,
            0.96,
            rf"Mean location error = {mean_loc:.3f} m; mean $q$ error = {100*mean_q:.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#bdbdbd", "alpha": 0.92},
            zorder=5,
        )
        ax.set_title(rf"({chr(97 + index)})  $q={q:g}$")
        ax.set_xlim(0.0, 9.0)
        ax.set_ylim(0.0, 3.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 3, 6, 9])
        ax.set_yticks([0, 1.5, 3])
        ax.set_xlabel(r"Source coordinate, $x$ (m)")
        ax.grid(color="#dddddd", linewidth=0.4, zorder=0)
    axes[0].set_ylabel(r"Source coordinate, $y$ (m)")

    true_handle = plt.Line2D(
        [], [], marker="o", markersize=4.2, linestyle="None",
        markerfacecolor="#2A788E", markeredgecolor="white",
        label="True source (color = location error)",
    )
    estimated_handle = plt.Line2D(
        [], [], marker="o", markersize=4.2, linestyle="None",
        markerfacecolor="none", markeredgecolor="#202020", label="Estimated source",
    )
    displacement_handle = plt.Line2D(
        [], [], color="#666666", lw=0.75, label="Localization displacement"
    )
    sensor_handle = plt.Line2D(
        [], [], marker="*", markersize=5.0, linestyle="None",
        color="#777777", alpha=0.55, label="Sensor",
    )
    fig.legend(
        handles=[true_handle, estimated_handle, displacement_handle, sensor_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        columnspacing=1.25,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.30, wspace=0.14)
    cax = fig.add_axes([0.29, 0.19, 0.42, 0.035])
    cbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    cbar.set_ticks(np.linspace(0.0, shared_max, 6))
    cbar.ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    cbar.set_label("Localization error (m)", labelpad=2)

    stem = args.output_dir / "inverse_offgrid_joint_strength_spatial_error"
    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 400}),
        (".tiff", {"dpi": 600}),
    ):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    manuscript_pdf = args.manuscript_figures / "inverse_offgrid_joint_strength_spatial_error.pdf"
    fig.savefig(manuscript_pdf, bbox_inches="tight")
    plt.close(fig)

    summary = (
        selected.groupby("q_true")
        .agg(
            n_cases=("case_id", "size"),
            mean_location_error_m=("loc_error_m", "mean"),
            median_location_error_m=("loc_error_m", "median"),
            mean_relative_strength_error=("q_relative_error", "mean"),
            median_relative_strength_error=("q_relative_error", "median"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "inverse_joint_strength_figure_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved manuscript figure: {manuscript_pdf}")


if __name__ == "__main__":
    main()
