#!/usr/bin/env python3
"""Validate and aggregate the 7 sizes x 3 seeds strict data-size ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SIZES = (50, 100, 200, 300, 400, 600, 679)
SEEDS = (7, 42, 123)
NAME_RE = re.compile(r"summary_n(?P<size>\d+)_seed(?P<seed>\d+)\.csv$")


def configure_style() -> None:
    """Match the final-size typography used by the manuscript figures."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ids_hash(ids: list[int]) -> str:
    payload = "\n".join(str(value) for value in sorted(ids)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--test-case-ids", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    configure_style()

    expected_ids = pd.read_csv(args.test_case_ids)["case_id"].astype(int).tolist()
    expected_set = set(expected_ids)
    if len(expected_set) != 145:
        raise ValueError(f"Expected 145 unique fixed test IDs, got {len(expected_set)}")

    rows = []
    observed = set()
    for path in sorted(args.results_dir.glob("summary_n*_seed*.csv")):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        size, seed = int(match.group("size")), int(match.group("seed"))
        key = (size, seed)
        if key in observed:
            raise RuntimeError(f"Duplicate result for n={size}, seed={seed}")
        observed.add(key)
        case_path = args.results_dir / f"cases_n{size:03d}_seed{seed}.csv"
        if not case_path.exists():
            raise FileNotFoundError(case_path)
        case_df = pd.read_csv(case_path)
        case_ids = case_df["case_id"].astype(int).tolist()
        if len(case_ids) != 145 or set(case_ids) != expected_set:
            raise RuntimeError(f"Test-set mismatch for n={size}, seed={seed}")
        summary = pd.read_csv(path)
        if len(summary) != 1:
            raise RuntimeError(f"Expected one summary row in {path}")
        row = summary.iloc[0].to_dict()
        row.update({"n_train": size, "training_seed": seed, "test_ids_hash": ids_hash(case_ids)})
        rows.append(row)

    expected = {(size, seed) for size in SIZES for seed in SEEDS}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise RuntimeError(f"Incomplete task matrix; missing={missing}, extra={extra}")

    runs = pd.DataFrame(rows).sort_values(["n_train", "training_seed"]).reset_index(drop=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    runs_path = args.outdir / "strict_datasize_runs.csv"
    runs.to_csv(runs_path, index=False)

    metric_columns = [
        column
        for column in runs.select_dtypes(include=[np.number]).columns
        if column not in {"n_train", "training_seed", "n_cases", "split_seed"}
    ]
    grouped_rows = []
    for size, frame in runs.groupby("n_train", sort=True):
        row = {"n_train": int(size), "n_seeds": int(len(frame)), "n_validation": 145, "n_test": 145}
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        grouped_rows.append(row)
    grouped = pd.DataFrame(grouped_rows)
    summary_path = args.outdir / "strict_datasize_summary_multiseed.csv"
    grouped.to_csv(summary_path, index=False)

    y = grouped["mean_rel_l2_c_mean"].to_numpy()
    yerr = grouped["mean_rel_l2_c_std"].to_numpy()
    x = grouped["n_train"].to_numpy()
    # Final-size single-column figure (89 mm wide).
    fig, ax = plt.subplots(figsize=(89 / 25.4, 68 / 25.4), constrained_layout=True)
    ax.errorbar(x, y, yerr=yerr, marker="o", markersize=4, linewidth=1.4, capsize=2.5)
    ax.set_xlabel("Number of training cases")
    ax.set_ylabel("Mean relative $L_2$ error on fixed test set")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25, linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    stem = args.outdir / "strict_datasize_ablation"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    fig.savefig(stem.with_suffix(".tiff"), format="tiff", dpi=600)
    plt.close(fig)

    audit = {
        "task_matrix": {"sizes": list(SIZES), "seeds": list(SEEDS), "runs": len(runs)},
        "fixed_validation_cases": 145,
        "fixed_test_cases": 145,
        "test_ids_hash": ids_hash(expected_ids),
        "error_bars": "sample standard deviation across three training seeds (ddof=1)",
        "status": "complete",
    }
    (args.outdir / "strict_datasize_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(grouped[["n_train", "mean_rel_l2_c_mean", "mean_rel_l2_c_std"]].to_string(index=False))
    print(f"[OK] Aggregated outputs written to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
