import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history_csv", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    args = ap.parse_args()

    hist_path = Path(args.history_csv)
    out_path = Path(args.out_path)
    df = pd.read_csv(hist_path)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ax.plot(df["epoch"], df["val_mse"], label="Validation loss", linewidth=2.2, color="#ff7f0e")
    ax.plot(df["epoch"], df["train_loss"], label="Training loss", linewidth=2.2, color="#1f77b4")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_xlim(1, int(df["epoch"].max()))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.55)
    ax.legend(loc="upper right", frameon=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
