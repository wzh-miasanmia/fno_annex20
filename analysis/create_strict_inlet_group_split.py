"""Create the exact inlet-group split used by joint-FNO job 1929712."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VAL_VELOCITIES = np.asarray([0.146635, 0.199408, 0.681966, 2.0], dtype=float)
TEST_VELOCITIES = np.asarray([0.501484, 0.584804, 0.927401, 1.081484], dtype=float)


def matches(values: np.ndarray, candidates: np.ndarray, tolerance: float) -> np.ndarray:
    return np.any(np.abs(values[:, None] - candidates[None, :]) <= tolerance, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=5e-7)
    args = parser.parse_args()

    cases = pd.read_csv(args.cases)
    required = ["case_id", "inlet_u", "src_x", "src_y"]
    missing = [name for name in required if name not in cases.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    cases = cases[required].copy()
    inlet = cases["inlet_u"].astype(float).to_numpy()
    is_val = matches(inlet, VAL_VELOCITIES, args.tolerance)
    is_test = matches(inlet, TEST_VELOCITIES, args.tolerance)
    if np.any(is_val & is_test):
        raise RuntimeError("Validation and test inlet groups overlap.")
    cases["split"] = "train"
    cases.loc[is_val, "split"] = "val"
    cases.loc[is_test, "split"] = "test"

    counts = cases["split"].value_counts().to_dict()
    expected = {"train": 6732, "val": 1584, "test": 1584}
    if counts != expected:
        raise RuntimeError(f"Unexpected split counts: {counts}; expected {expected}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(args.output, index=False)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "counts": counts,
                "val_velocities": VAL_VELOCITIES.tolist(),
                "test_velocities": TEST_VELOCITIES.tolist(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
