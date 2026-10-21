"""Plot the sensor-count and sensor-layout ablation on the 145 held-out cases.

The quantitative panels use the proposed FNO coarse-to-gradient inverse method.
Each point is the mean over 145 cases and three independently trained FNO
checkpoints; error bars are the sample standard deviation across checkpoints.
"""

from pathlib import Path
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from .sensors import layout_nodes
else:
    from sensors import layout_nodes


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "conditions.csv"
FIGURE_DIR = Path.cwd() / "figures"
STEM = FIGURE_DIR / "inverse_sensor_count_layout_ablation"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    keep = (
        (df["dataset"] == "dense")
        & (df["method"] == "FNO_coarse_gradient")
        & (df["n_sensors"].isin([4, 8, 16]))
    )
    df = df.loc[keep].copy()
    expected = 4 * 3 * 3
    if len(df) != expected:
        raise RuntimeError(f"Expected {expected} conditions, found {len(df)}")
    return df


def export_source_data(df: pd.DataFrame) -> None:
    cols = [
        "layout",
        "n_sensors",
        "noise",
        "fno_mean_error_m",
        "fno_seed_std",
    ]
    df[cols].sort_values(["noise", "layout", "n_sensors"]).to_csv(
        ROOT / "sensor_ablation_figure_source_data.csv", index=False
    )

    sensor_rows = []
    for layout in range(4):
        ix, iy = layout_nodes(layout)
        for rank, (i, j) in enumerate(zip(ix[:16], iy[:16]), start=1):
            sensor_rows.append(
                {
                    "layout": layout,
                    "rank": rank,
                    "x_m": i * 9.0 / 383.0,
                    "y_m": j * 3.0 / 127.0,
                }
            )
    pd.DataFrame(sensor_rows).to_csv(
        ROOT / "sensor_layout_coordinates.csv", index=False
    )


def plot(df: pd.DataFrame) -> None:
    set_style()
    colors = ["#3C6E91", "#D18F43", "#4C956C", "#8064A2"]
    markers = ["o", "s", "^", "D"]
    counts = np.array([4, 8, 16])
    noises = [0.0, 0.02, 0.05]

    fig = plt.figure(figsize=(7.25, 4.45), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.62, 1.40],
        hspace=0.30,
        left=0.075,
        right=0.985,
        top=0.91,
        bottom=0.16,
    )
    top = outer[0].subgridspec(1, 4, wspace=0.18)
    bottom = outer[1].subgridspec(1, 3, wspace=0.34)

    layout_axes = []
    for layout in range(4):
        ax = fig.add_subplot(top[0, layout])
        layout_axes.append(ax)
        ix, iy = layout_nodes(layout)
        x = ix[:8] * 9.0 / 383.0
        y = iy[:8] * 3.0 / 127.0
        ax.scatter(
            x,
            y,
            s=31,
            marker="*",
            color=colors[layout],
            edgecolors="none",
            zorder=3,
        )
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 3, 6, 9])
        ax.set_yticks([0, 1.5, 3])
        if layout > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"$y$ (m)", labelpad=1.5)
        ax.set_xlabel(r"$x$ (m)", labelpad=1.5)
        ax.set_title(f"Layout {layout}", pad=3.0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)

    layout_axes[0].text(
        -0.34,
        1.18,
        "a",
        transform=layout_axes[0].transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    fig.text(0.53, 0.975, "Eight-sensor configurations", ha="center", va="top", fontsize=9)

    y_limits = [(0.0, 0.52), (0.0, 0.68), (0.0, 1.00)]
    metric_axes = []
    for panel, (noise, ylim) in enumerate(zip(noises, y_limits)):
        ax = fig.add_subplot(bottom[0, panel])
        metric_axes.append(ax)
        for layout in range(4):
            block = df[(df["noise"] == noise) & (df["layout"] == layout)].sort_values(
                "n_sensors"
            )
            if list(block["n_sensors"].astype(int)) != list(counts):
                raise RuntimeError(f"Incomplete data for layout={layout}, noise={noise}")
            ax.errorbar(
                counts,
                block["fno_mean_error_m"],
                yerr=block["fno_seed_std"],
                color=colors[layout],
                marker=markers[layout],
                markersize=4.2,
                linewidth=1.2,
                elinewidth=0.8,
                capsize=2.0,
                label=f"Layout {layout}",
            )
        ax.set_xscale("log", base=2)
        ax.set_xlim(3.5, 18)
        ax.set_ylim(*ylim)
        ax.set_xticks(counts)
        ax.set_xticklabels([str(v) for v in counts])
        ax.set_xlabel("Number of sensors")
        if panel == 0:
            ax.set_ylabel("Mean localization error (m)")
        ax.set_title(f"Noise level = {noise:g}", pad=4.0)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.22,
            1.12,
            chr(ord("b") + panel),
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )

    handles, labels = metric_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.53, 0.035),
        frameon=False,
        handlelength=2.1,
        columnspacing=1.6,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='conditions.csv written by run_sensors.py aggregate.')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    INPUT = args.input
    ROOT = args.output_dir
    FIGURE_DIR = args.output_dir
    ROOT.mkdir(parents=True, exist_ok=True)
    STEM = FIGURE_DIR / 'inverse_sensor_count_layout_ablation'
    data = load_data()
    export_source_data(data)
    plot(data)
    print(STEM.with_suffix(".pdf"))
