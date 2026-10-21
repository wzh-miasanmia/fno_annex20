#!/usr/bin/env python3
"""Verify the new evaluator against the completed 384x128 seed-7 evaluation."""

import argparse
import json
from pathlib import Path


def validate_metrics(new, reference):
    """Reject incomplete/non-finite evaluations before comparing per-case metrics."""
    import numpy as np
    import pandas as pd

    cols = ["rel_l2_c", "mae_c", "high_rmse_c"]
    frames = []
    for label, frame in [("new", new), ("reference", reference)]:
        required = ["case_id", *cols]
        missing = [column for column in required if column not in frame]
        if missing:
            raise RuntimeError(f"{label} evaluation is missing columns: {missing}")
        ids = pd.to_numeric(frame["case_id"], errors="coerce")
        if (len(frame) != 145 or not np.isfinite(ids.to_numpy(dtype=float)).all()
                or not ids.is_unique or not (ids % 1 == 0).all()):
            raise RuntimeError(f"{label} evaluation must contain 145 unique, non-null finite integer case IDs")
        values = frame[cols].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise RuntimeError(f"{label} evaluation contains non-finite or invalid metrics")
        checked = frame.copy()
        checked["case_id"] = ids.astype(np.int64)
        checked[cols] = values
        frames.append(checked.sort_values("case_id").reset_index(drop=True))

    new, ref = frames
    if not new["case_id"].equals(ref["case_id"]):
        raise RuntimeError("New and reference evaluations do not contain identical 145-case test sets")
    max_abs = {col: float(np.max(np.abs(new[col] - ref[col]))) for col in cols}
    mean_abs = {col: float(np.mean(np.abs(new[col] - ref[col]))) for col in cols}
    if max(max_abs.values()) > 2e-5:
        raise RuntimeError(f"Evaluator mismatch exceeds tolerance: {max_abs}")
    return {"n_test": 145, "max_absolute_metric_difference": max_abs,
            "mean_absolute_metric_difference": mean_abs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    import pandas as pd

    result = validate_metrics(pd.read_csv(args.new), pd.read_csv(args.reference))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
