#!/usr/bin/env python3
"""Build an auditable fixed split and nested training subsets for data-size ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SIZES = (50, 100, 200, 300, 400, 600, 679)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reproduce_random_split(n: int, seed: int, val_ratio: float, test_ratio: float):
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    return indices[n_test + n_val :], indices[n_test : n_test + n_val], indices[:n_test]


def farthest_point_order(rows: pd.DataFrame) -> list[int]:
    """Return row positions ordered for progressively space-filling source coverage."""
    xy = rows[["src_x", "src_y"]].to_numpy(dtype=np.float64)
    span = np.ptp(xy, axis=0)
    if np.any(span <= 0):
        raise ValueError("src_x and src_y must both span a non-zero range")
    z = (xy - xy.min(axis=0)) / span
    case_ids = rows["case_id"].to_numpy(dtype=np.int64)
    centre_distance = np.sum((z - 0.5) ** 2, axis=1)
    first = int(np.lexsort((case_ids, centre_distance))[0])
    selected = [first]
    available = np.ones(len(rows), dtype=bool)
    available[first] = False
    min_distance = np.sum((z - z[first]) ** 2, axis=1)

    while len(selected) < len(rows):
        candidates = np.flatnonzero(available)
        distance = min_distance[candidates]
        best = int(candidates[np.lexsort((case_ids[candidates], -distance))[0]])
        selected.append(best)
        available[best] = False
        min_distance = np.minimum(min_distance, np.sum((z - z[best]) ** 2, axis=1))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--reference-test-ids", type=Path)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    cases = pd.read_csv(args.cases)
    required = {"case_id", "inlet_u", "src_x", "src_y"}
    missing = required.difference(cases.columns)
    if missing:
        raise ValueError(f"Missing required cases.csv columns: {sorted(missing)}")
    if len(cases) != 969 or cases["case_id"].duplicated().any():
        raise ValueError(f"Expected 969 unique cases, got {len(cases)} rows")

    train_idx, val_idx, test_idx = reproduce_random_split(
        len(cases), args.split_seed, args.val_ratio, args.test_ratio
    )
    if (len(train_idx), len(val_idx), len(test_idx)) != (679, 145, 145):
        raise RuntimeError("Fixed split did not produce 679/145/145")

    train = cases.iloc[train_idx].copy().reset_index(drop=True)
    val_ids = set(cases.iloc[val_idx]["case_id"].astype(int))
    test_ids = set(cases.iloc[test_idx]["case_id"].astype(int))
    train_ids = set(train["case_id"].astype(int))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Train/validation/test leakage detected")

    if args.reference_test_ids:
        reference = set(pd.read_csv(args.reference_test_ids)["case_id"].astype(int))
        if reference != test_ids:
            missing_ref = sorted(test_ids - reference)[:10]
            extra_ref = sorted(reference - test_ids)[:10]
            raise RuntimeError(
                f"Reproduced test set differs from reference; missing={missing_ref}, extra={extra_ref}"
            )

    args.outdir.mkdir(parents=True, exist_ok=True)
    labels = pd.Series(index=cases.index, dtype="object")
    labels.iloc[train_idx] = "train"
    labels.iloc[val_idx] = "validation"
    labels.iloc[test_idx] = "test"
    fixed_split = pd.DataFrame({"case_id": cases["case_id"].astype(int), "split": labels})
    fixed_split_path = args.outdir / f"fixed_split_seed{args.split_seed}.csv"
    fixed_split.to_csv(fixed_split_path, index=False)

    pd.DataFrame({"case_id": sorted(train_ids)}).to_csv(args.outdir / "training_pool_case_ids.csv", index=False)
    pd.DataFrame({"case_id": sorted(val_ids)}).to_csv(args.outdir / "validation_case_ids.csv", index=False)
    pd.DataFrame({"case_id": sorted(test_ids)}).to_csv(args.outdir / "test_case_ids.csv", index=False)

    order_pos = farthest_point_order(train)
    ordered = train.iloc[order_pos].copy().reset_index(drop=True)
    ordered.insert(0, "selection_rank", np.arange(1, len(ordered) + 1))
    ordered.to_csv(args.outdir / "train_nested_order_farthest.csv", index=False)

    index_rows = []
    previous: set[int] = set()
    for size in SIZES:
        chosen = set(ordered.iloc[:size]["case_id"].astype(int))
        if not previous.issubset(chosen):
            raise RuntimeError(f"Training subset n={size} is not nested")
        previous = chosen
        keep_ids = chosen | val_ids | test_ids
        manifest = cases[cases["case_id"].astype(int).isin(keep_ids)].copy()
        split = fixed_split[fixed_split["case_id"].astype(int).isin(keep_ids)].copy()
        tag = f"{size:03d}"
        manifest_path = args.outdir / f"cases_n{tag}.csv"
        split_path = args.outdir / f"split_n{tag}.csv"
        manifest.to_csv(manifest_path, index=False)
        split.to_csv(split_path, index=False)
        counts = split["split"].value_counts().to_dict()
        observed = (counts.get("train", 0), counts.get("validation", 0), counts.get("test", 0))
        if observed != (size, 145, 145):
            raise RuntimeError(f"Bad split counts for n={size}: {observed}")
        index_rows.append(
            {
                "n_train": size,
                "n_validation": 145,
                "n_test": 145,
                "n_manifest": len(manifest),
                "cases_csv": manifest_path.name,
                "split_csv": split_path.name,
                "cases_sha256": sha256(manifest_path),
                "split_sha256": sha256(split_path),
            }
        )

    pd.DataFrame(index_rows).to_csv(args.outdir / "manifest_index.csv", index=False)
    audit = {
        "source_cases": str(args.cases.resolve()),
        "source_cases_sha256": sha256(args.cases),
        "split_seed": args.split_seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "split_counts": {"train": 679, "validation": 145, "test": 145},
        "subset_sizes": list(SIZES),
        "subset_method": "deterministic farthest-point prefix on min-max-normalized (src_x, src_y)",
        "reference_test_ids": None if args.reference_test_ids is None else str(args.reference_test_ids.resolve()),
        "test_case_ids_sha256": sha256(args.outdir / "test_case_ids.csv"),
        "nested": True,
    }
    (args.outdir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(pd.DataFrame(index_rows).to_string(index=False))
    print(f"[OK] Strict nested manifests written to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
