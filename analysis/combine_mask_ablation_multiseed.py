from pathlib import Path
import argparse

import pandas as pd


LABELS = {
    "rect": "Rect",
    "gauss": "Gaussian",
    "softrect_sk30": "Soft-rect",
}


def parse_int_list(text: str) -> list[int]:
    vals = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not vals:
        raise ValueError("Expected at least one integer seed.")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description="Combine mask ablation multi-seed CSVs into one table.")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--results_dir", type=str, default=str(root / "analysis" / "results"))
    ap.add_argument("--include_seeds", type=str, default="")
    ap.add_argument("--exclude_seeds", type=str, default="")
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    include_seeds = set(parse_int_list(args.include_seeds)) if args.include_seeds else None
    exclude_seeds = set(parse_int_list(args.exclude_seeds)) if args.exclude_seeds else set()

    specs = [
        ("rect", results_dir / "compare_mask_rect_u0p2_multiseed.csv"),
        ("gauss", results_dir / "compare_mask_gauss_u0p2_multiseed.csv"),
        ("softrect_sk30", results_dir / "compare_mask_softrect_sk30_u0p2_multiseed.csv"),
    ]

    parts = []
    for key, path in specs:
        df = pd.read_csv(path)
        df["mask_label"] = LABELS[key]
        if include_seeds is not None:
            df = df[df["seed"].isin(include_seeds)].copy()
        if exclude_seeds:
            df = df[~df["seed"].isin(exclude_seeds)].copy()
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = (root / out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
