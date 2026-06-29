import argparse
import re
from pathlib import Path

import pandas as pd

try:
    import torch
except Exception:
    torch = None


SEED_RE = re.compile(r"_seed(\d+)$")


def seed_from_stem(stem: str) -> int | None:
    m = SEED_RE.search(stem)
    if not m:
        return None
    return int(m.group(1))


def resolve_checkpoint_path(root: Path, checkpoint_value: str) -> Path:
    path = Path(str(checkpoint_value))
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_timing_info(root: Path, checkpoint_value: str) -> dict:
    ckpt_path = resolve_checkpoint_path(root, checkpoint_value)
    history_path = ckpt_path.with_name(f"history_{ckpt_path.stem}.csv")
    out: dict = {}

    if history_path.exists():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty:
                last = hist.iloc[-1]
                if "epoch_time_sec" in hist.columns:
                    out["last_epoch_time_sec"] = float(last["epoch_time_sec"])
                    out["epoch_time_sec_mean"] = float(hist["epoch_time_sec"].astype(float).mean())
                    out["epoch_time_sec_std"] = float(hist["epoch_time_sec"].astype(float).std(ddof=1)) if len(hist) > 1 else float("nan")
                if "total_training_time_sec" in hist.columns:
                    out["total_training_time_sec"] = float(last["total_training_time_sec"])
                if "avg_epoch_time_sec" in hist.columns:
                    out["avg_epoch_time_sec"] = float(last["avg_epoch_time_sec"])
                out["history_path"] = str(history_path)
        except Exception:
            pass

    # Fallback for older/newer runs where timing may only live in checkpoint meta.
    if torch is not None and ("total_training_time_sec" not in out or "avg_epoch_time_sec" not in out) and ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            meta = ckpt.get("meta", {})
            if "total_training_time_sec" in meta and "total_training_time_sec" not in out:
                out["total_training_time_sec"] = float(meta["total_training_time_sec"])
            if "avg_epoch_time_sec" in meta and "avg_epoch_time_sec" not in out:
                out["avg_epoch_time_sec"] = float(meta["avg_epoch_time_sec"])
            if "epochs_completed" in meta:
                out["epochs_completed"] = int(meta["epochs_completed"])
        except Exception:
            pass

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate summary CSV files from multiple seeds into raw and summary tables."
    )
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--root", type=str, default=str(root))
    ap.add_argument("--base_stem", type=str, required=True)
    ap.add_argument("--results_dir", type=str, default=None)
    ap.add_argument("--out_raw_csv", type=str, default="")
    ap.add_argument("--out_summary_csv", type=str, default="")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    results_dir = Path(args.results_dir).resolve() if args.results_dir else (root / "analysis" / "results")
    pattern = f"summary_{args.base_stem}_seed*.csv"
    paths = sorted(results_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {results_dir / pattern}")

    rows = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["seed"] = seed_from_stem(path.stem.replace("summary_", "", 1))
        row["summary_path"] = str(path)
        if "checkpoint" in row and pd.notna(row["checkpoint"]):
            row.update(load_timing_info(root, str(row["checkpoint"])))
        rows.append(row)

    if not rows:
        raise RuntimeError(f"All matched files were empty: {results_dir / pattern}")

    raw_df = pd.DataFrame(rows).sort_values(["seed", "experiment"], na_position="last")
    metric_cols = [
        "mean_rel_l2_c",
        "mean_rmse_c",
        "mean_mae_c",
        "mean_abs_err_cre",
        "mean_rel_err_cre",
        "mean_high_rmse_c",
        "mean_high_mae_c",
        "total_training_time_sec",
        "avg_epoch_time_sec",
        "last_epoch_time_sec",
        "epoch_time_sec_mean",
    ]
    present_metrics = [c for c in metric_cols if c in raw_df.columns]

    summary = {
        "base_stem": args.base_stem,
        "n_seeds": int(raw_df["seed"].notna().sum()),
        "seeds": ",".join(str(int(v)) for v in raw_df["seed"].dropna().astype(int).tolist()),
    }
    for col in present_metrics:
        summary[f"{col}_mean"] = float(raw_df[col].mean())
        summary[f"{col}_std"] = float(raw_df[col].std(ddof=1)) if len(raw_df[col]) > 1 else float("nan")
        summary[f"{col}_min"] = float(raw_df[col].min())
        summary[f"{col}_max"] = float(raw_df[col].max())

    out_raw = Path(args.out_raw_csv) if args.out_raw_csv else (results_dir / f"compare_{args.base_stem}_multiseed.csv")
    out_summary = (
        Path(args.out_summary_csv)
        if args.out_summary_csv
        else (results_dir / f"compare_{args.base_stem}_multiseed_summary.csv")
    )
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    raw_df.to_csv(out_raw, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    print(f"[SAVED] raw -> {out_raw}")
    print(f"[SAVED] summary -> {out_summary}")


if __name__ == "__main__":
    main()
