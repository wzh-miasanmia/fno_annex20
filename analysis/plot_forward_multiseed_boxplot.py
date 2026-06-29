from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 200,
            "savefig.dpi": 300,
        }
    )


def parse_csv_list(text: str) -> list[str]:
    vals = [part.strip() for part in str(text).split(",") if part.strip()]
    if not vals:
        raise ValueError("Expected at least one comma-separated value.")
    return vals


def to_numeric_if_possible(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series)
    except Exception:
        return series


def sort_group_values(values: list) -> list:
    label_order = {
        "Rect": 0,
        "Gaussian": 1,
        "Soft-rect": 2,
    }
    if all(str(v) in label_order for v in values):
        return sorted(values, key=lambda x: label_order[str(x)])
    try:
        return sorted(values, key=lambda x: float(x))
    except Exception:
        return sorted(values, key=lambda x: str(x))


def make_group_arrays(df: pd.DataFrame, group_col: str, metric_col: str, ordered_groups: list) -> list[np.ndarray]:
    arrays = []
    for group in ordered_groups:
        vals = df.loc[df[group_col] == group, metric_col].astype(float).dropna().to_numpy()
        arrays.append(vals)
    return arrays


def plot_box_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    ordered_groups: list,
    x_tick_labels: list[str],
    ylabel: str,
    color: str,
    show_fliers: bool,
    whis,
) -> None:
    arrays = make_group_arrays(df, group_col, metric_col, ordered_groups)
    positions = np.arange(1, len(arrays) + 1)

    bp = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        whis=whis,
        showfliers=show_fliers,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"color": color, "linewidth": 1.2},
        capprops={"color": color, "linewidth": 1.2},
        boxprops={"facecolor": color, "alpha": 0.28, "edgecolor": color, "linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#444444",
            "markersize": 4.5,
            "linestyle": "none",
            "markeredgewidth": 0.9,
        },
    )
    _ = bp

    for pos, vals in zip(positions, arrays):
        if len(vals) == 0:
            continue
        mean_val = float(np.mean(vals))
        ax.scatter(
            [pos],
            [mean_val],
            s=22,
            marker="s",
            facecolors="white",
            edgecolors="black",
            linewidths=1.0,
            zorder=4,
        )

    ax.set_xticks(positions, x_tick_labels)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot box charts for forward multi-seed experiment results.")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--group_col", type=str, required=True)
    ap.add_argument("--group_label", type=str, default="")
    ap.add_argument("--error_metrics", type=str, default="mean_rel_l2_c")
    ap.add_argument("--error_labels", type=str, default="")
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--show_fliers", action="store_true")
    ap.add_argument("--whis_mode", type=str, default="range", choices=["range", "iqr"])
    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--out_pdf", type=str, default="")
    args = ap.parse_args()

    configure_style()
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = (root / csv_path).resolve()
    df = pd.read_csv(csv_path)

    if args.group_col not in df.columns:
        raise ValueError(f"group_col '{args.group_col}' not found in {csv_path}")

    error_metrics = parse_csv_list(args.error_metrics)
    error_labels = parse_csv_list(args.error_labels) if args.error_labels else []
    if error_labels and len(error_labels) != len(error_metrics):
        raise ValueError("error_labels must have the same number of entries as error_metrics")
    if not error_labels:
        default_labels = {
            "mean_rel_l2_c": r"MRL2",
            "mean_rmse_c": "RMSE",
            "mean_mae_c": "MAE",
        }
        error_labels = [default_labels.get(metric, metric) for metric in error_metrics]

    for metric in error_metrics:
        if metric not in df.columns:
            raise ValueError(f"error metric '{metric}' not found in {csv_path}")

    df = df.copy()
    df[args.group_col] = to_numeric_if_possible(df[args.group_col])
    for metric in error_metrics:
        df = df[df[metric].notna()].copy()

    ordered_groups = sort_group_values(df[args.group_col].dropna().unique().tolist())
    x_tick_labels = [str(int(v)) if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer()) else str(v) for v in ordered_groups]

    n_panels = len(error_metrics)
    fig, axes = plt.subplots(n_panels, 1, figsize=(5.0, 3.6), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    colors = ["#4c78a8"] * n_panels
    whis = (0, 100) if args.whis_mode == "range" else 1.5
    for idx, (metric, label) in enumerate(zip(error_metrics, error_labels)):
        plot_box_panel(
            axes[idx],
            df=df,
            group_col=args.group_col,
            metric_col=metric,
            ordered_groups=ordered_groups,
            x_tick_labels=x_tick_labels,
            ylabel=label,
            color=colors[idx % len(colors)],
            show_fliers=bool(args.show_fliers),
            whis=whis,
        )

    x_label = args.group_label if args.group_label else args.group_col
    for ax in axes:
        ax.set_xlabel("" if str(args.group_label).strip() == "" else x_label)

    out_png = Path(args.out_png)
    if not out_png.is_absolute():
        out_png = (root / out_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")

    if args.out_pdf:
        out_pdf = Path(args.out_pdf)
        if not out_pdf.is_absolute():
            out_pdf = (root / out_pdf).resolve()
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Saved: {out_pdf}")

    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
