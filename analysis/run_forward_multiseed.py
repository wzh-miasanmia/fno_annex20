import argparse
import subprocess
import sys
from pathlib import Path


def parse_int_list(text: str) -> list[int]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("Expected at least one integer value.")
    return vals


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run one forward-model experiment over multiple random seeds "
        "and save seed-tagged checkpoints and evaluation CSVs."
    )
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--root", type=str, default=str(root))
    ap.add_argument("--base_stem", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="7,21,42,84,123")
    ap.add_argument("--workdir", type=str, default=None)
    ap.add_argument("--exports", type=str, default="exports")
    ap.add_argument("--cases", type=str, default="cases.csv")
    ap.add_argument("--fixed_inlet_u", type=float, default=0.2)
    ap.add_argument("--inlet_u_tol", type=float, default=1e-6)
    ap.add_argument("--limit_samples", type=int, default=0)
    ap.add_argument("--limit_strategy", type=str, default="uniform_source")
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--split_mode", type=str, default="random")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--nx", type=int, default=128 * 3)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--modes1", type=int, default=16)
    ap.add_argument("--modes2", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--mask_kind", type=str, default="soft_rect")
    ap.add_argument("--src_size", type=float, default=0.20)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--soft_k", type=float, default=30.0)
    ap.add_argument("--source_aspect_ratio", type=float, default=1.0)
    ap.add_argument("--target_transform", type=str, default="raw")
    ap.add_argument("--target_transform_eps", type=float, default=1e-8)
    ap.add_argument("--smoothness_weight", type=float, default=0.0)
    ap.add_argument("--gradient_weight", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min_delta", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--evaluate_subset", type=str, default="test")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save_curves", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    workdir = Path(args.workdir).resolve() if args.workdir else (root / "data")
    train_script = root / "model" / "train_fixed_u_srcpos_to_c.py"
    eval_script = root / "analysis" / "evaluate_fixed_u_srcpos_to_c.py"
    checkpoints_dir = root / "checkpoints"
    results_dir = root / "analysis" / "results"
    seeds = parse_int_list(args.seeds)

    for seed in seeds:
        stem = f"{args.base_stem}_seed{seed}"
        ckpt_path = checkpoints_dir / f"{stem}.pt"
        case_csv = results_dir / f"cases_{stem}.csv"
        summary_csv = results_dir / f"summary_{stem}.csv"

        if not ckpt_path.exists() or args.overwrite:
            cmd = [
                sys.executable,
                str(train_script),
                "--workdir", str(workdir),
                "--exports", args.exports,
                "--cases", args.cases,
                "--fixed_inlet_u", str(args.fixed_inlet_u),
                "--no_velocity_input",
                "--inlet_u_tol", str(args.inlet_u_tol),
                "--sample_seed", str(args.sample_seed),
                "--split_seed", str(args.split_seed),
                "--split_mode", args.split_mode,
                "--val_ratio", str(args.val_ratio),
                "--test_ratio", str(args.test_ratio),
                "--nx", str(args.nx),
                "--ny", str(args.ny),
                "--width", str(args.width),
                "--modes1", str(args.modes1),
                "--modes2", str(args.modes2),
                "--depth", str(args.depth),
                "--mask_kind", args.mask_kind,
                "--src_size", str(args.src_size),
                "--sigma", str(args.sigma),
                "--soft_k", str(args.soft_k),
                "--source_aspect_ratio", str(args.source_aspect_ratio),
                "--target_transform", args.target_transform,
                "--target_transform_eps", str(args.target_transform_eps),
                "--smoothness_weight", str(args.smoothness_weight),
                "--gradient_weight", str(args.gradient_weight),
                "--epochs", str(args.epochs),
                "--batch", str(args.batch),
                "--lr", str(args.lr),
                "--weight_decay", str(args.weight_decay),
                "--grad_clip", str(args.grad_clip),
                "--patience", str(args.patience),
                "--min_delta", str(args.min_delta),
                "--workers", str(args.workers),
                "--seed", str(seed),
                "--device", args.device,
                "--save", str(ckpt_path),
            ]
            if int(args.limit_samples) > 0:
                cmd.extend([
                    "--limit_samples", str(args.limit_samples),
                    "--limit_strategy", args.limit_strategy,
                ])
            if args.cache:
                cmd.append("--cache")
            if args.amp:
                cmd.append("--amp")
            if args.save_curves:
                cmd.append("--save_curves")
            run_cmd(cmd, root)
        else:
            print(f"[SKIP] checkpoint exists: {ckpt_path}")

        if not case_csv.exists() or not summary_csv.exists() or args.overwrite:
            eval_cmd = [
                sys.executable,
                str(eval_script),
                "--workdir", str(workdir),
                "--ckpt", str(ckpt_path),
                "--subset", args.evaluate_subset,
                "--out_csv", str(case_csv),
                "--summary_csv", str(summary_csv),
            ]
            run_cmd(eval_cmd, root)
        else:
            print(f"[SKIP] evaluation exists: {case_csv}")


if __name__ == "__main__":
    main()
