#!/usr/bin/env python3
"""Aggregate seed-wise test means for the zero-shot super-resolution table."""

import argparse
from pathlib import Path

RESOLUTIONS = [(48, 16), (96, 32), (192, 64), (288, 96)]
SEEDS = [7, 42, 123]
METRICS = ["rel_l2_c", "mae_c", "high_rmse_c"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--highres-root", required=True)
    ap.add_argument("--test-ids", required=True)
    args = ap.parse_args()
    import pandas as pd

    root = Path(args.root)
    expected_ids = set(pd.read_csv(args.test_ids)["case_id"].astype(int))
    seed_rows = []

    for nx, ny in RESOLUTIONS:
        for seed in SEEDS:
            path = root / f"{nx}x{ny}" / f"seed{seed}" / "common_grid_test.csv"
            df = pd.read_csv(path)
            validate(df, path, expected_ids)
            seed_rows.append(summarize(df, nx, ny, seed))

    for seed in SEEDS:
        path = Path(args.highres_root) / f"seed{seed}" / "common_grid_test.csv"
        df = pd.read_csv(path)
        validate(df, path, expected_ids)
        seed_rows.append(summarize(df, 384, 128, seed))

    per_seed = pd.DataFrame(seed_rows).sort_values(["train_nx", "seed"])
    per_seed.to_csv(root / "per_seed.csv", index=False)
    summary_rows = []
    for (nx, ny), group in per_seed.groupby(["train_nx", "train_ny"], sort=True):
        row = {"training_resolution": f"{nx}x{ny}", "train_nx": nx, "train_ny": ny, "n_seeds": len(group)}
        for metric in METRICS:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_seed_{metric}"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if not (summary["n_seeds"] == 3).all():
        raise RuntimeError("Every resolution must contain exactly three seeds")
    summary.to_csv(root / "summary.csv", index=False)
    (root / "COMPLETE.txt").write_text(
        "All five resolutions aggregated from 3 seeds; SD is across seed-wise 145-case test means.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


def validate(df, path, expected_ids):
    import numpy as np

    required = {"case_id", *METRICS}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"{path} is missing columns {sorted(missing)}")
    ids = set(df["case_id"].astype(int))
    if len(df) != 145 or ids != expected_ids:
        raise RuntimeError(f"{path} does not contain the exact 145-case strict test set")
    if not np.isfinite(df[METRICS].to_numpy()).all():
        raise RuntimeError(f"{path} contains non-finite metrics")


def summarize(df, nx, ny, seed):
    return {
        "training_resolution": f"{nx}x{ny}",
        "train_nx": nx,
        "train_ny": ny,
        "seed": seed,
        "n_test": len(df),
        **{metric: float(df[metric].mean()) for metric in METRICS},
    }


if __name__ == "__main__":
    main()
