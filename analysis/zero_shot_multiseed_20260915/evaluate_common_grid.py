#!/usr/bin/env python3
"""Evaluate a native-resolution FNO checkpoint zero-shot on the 384x128 CFD cache."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

def load_training_module(path: Path):
    spec = importlib.util.spec_from_file_location("strict_fno_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--trainer", default=str(Path(__file__).resolve().parents[2] /
                    "model/train_fixed_u_srcpos_to_c_strict_ablation.py"),
                    help="Frozen FNO module; defaults to the repository model snapshot")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--fields", required=True)
    ap.add_argument("--test-ids", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--train-nx", type=int, required=True)
    ap.add_argument("--train-ny", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    # The unified benchmark computed its published per-case metrics one sample
    # at a time. Match that evaluation path exactly in the consistency gate.
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trainer = load_training_module(Path(args.trainer))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt.get("meta", {})
    model = trainer.FNO2D(
        in_ch=int(meta.get("in_ch", 3)),
        out_ch=int(meta.get("out_ch", 1)),
        width=int(meta.get("width", 48)),
        modes1=int(meta.get("modes1", 16)),
        modes2=int(meta.get("modes2", 16)),
        depth=int(meta.get("depth", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_mean = ckpt["x_mean"].to(device)
    x_std = ckpt["x_std"].to(device)
    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)
    transform = str(meta.get("target_transform", "raw"))
    transform_eps = float(meta.get("target_transform_eps", 1e-8))
    if transform not in {"raw", "log10"}:
        raise ValueError(f"Unsupported target transform: {transform}")

    rows = pd.read_csv(args.rows)
    test_ids = pd.read_csv(args.test_ids)["case_id"].astype(int).tolist()
    test_set = set(test_ids)
    selected = rows[(rows["dataset"] == "dense") & rows["case_id"].astype(int).isin(test_set)].copy()
    selected["cache_index"] = selected.index
    selected["case_id"] = selected["case_id"].astype(int)
    selected = selected.set_index("case_id").loc[test_ids].reset_index()
    if len(selected) != 145 or selected["case_id"].nunique() != 145:
        raise RuntimeError(f"Expected 145 unique strict-test cases, got {len(selected)}")

    fields = np.load(args.fields, mmap_mode="r")
    if fields.shape[1:] != (128, 384):
        raise RuntimeError(f"Expected fields with shape (*,128,384), got {fields.shape}")

    nx, ny, lx, ly = 384, 128, 9.0, 3.0
    x_phys = np.linspace(0.0, lx, nx, dtype=np.float32)
    y_phys = np.linspace(0.0, ly, ny, dtype=np.float32)
    xg, yg = np.meshgrid(x_phys, y_phys)
    xc, yc = np.meshgrid(
        np.linspace(0.0, 1.0, nx, dtype=np.float32),
        np.linspace(0.0, 1.0, ny, dtype=np.float32),
    )
    src_size = float(meta.get("src_size", 0.20))
    soft_k = float(meta.get("soft_k", 30.0))

    records = []
    with torch.inference_mode():
        for start in range(0, len(selected), args.batch):
            part = selected.iloc[start : start + args.batch]
            batch_inputs = []
            truths = []
            for row in part.itertuples(index=False):
                mask = trainer.build_source_mask_soft_rect_sigmoid(
                    xg, yg, float(row.src_x), float(row.src_y), src_size, soft_k
                ).astype(np.float32)
                batch_inputs.append(np.stack([mask, xc, yc], axis=0))
                truths.append(np.asarray(fields[int(row.cache_index)], dtype=np.float32))
            xb = torch.from_numpy(np.stack(batch_inputs)).to(device)
            pred_t = model((xb - x_mean) / x_std) * y_std + y_mean
            if transform == "log10":
                pred_t = torch.clamp(torch.pow(10.0, pred_t) - transform_eps, min=0.0)
            preds = pred_t[:, 0].float().cpu().numpy()

            for row, pred, truth in zip(part.itertuples(index=False), preds, truths):
                error = pred - truth
                threshold = float(np.percentile(np.abs(truth), 90.0))
                high = np.abs(truth) >= threshold
                records.append({
                    "experiment": "zero_shot_multiseed",
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "case_id": int(row.case_id),
                    "src_x": float(row.src_x),
                    "src_y": float(row.src_y),
                    "train_nx": args.train_nx,
                    "train_ny": args.train_ny,
                    "seed": args.seed,
                    "rel_l2_c": float(np.linalg.norm(error.ravel()) / (np.linalg.norm(truth.ravel()) + 1e-12)),
                    "mae_c": float(np.mean(np.abs(error))),
                    "high_rmse_c": float(np.sqrt(np.mean(np.square(error[high])))),
                })

    result = pd.DataFrame.from_records(records)
    metric_cols = ["rel_l2_c", "mae_c", "high_rmse_c"]
    if len(result) != 145 or not np.isfinite(result[metric_cols].to_numpy()).all():
        raise RuntimeError("Evaluation output failed completeness or finite-value checks")
    result.to_csv(out_path, index=False)
    summary = {
        "train_resolution": f"{args.train_nx}x{args.train_ny}",
        "seed": args.seed,
        "n_test": len(result),
        **{name: float(result[name].mean()) for name in metric_cols},
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
