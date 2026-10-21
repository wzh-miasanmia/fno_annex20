import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def case_id_text(value):
    """Preserve integer case identifiers after pandas row coercion."""
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value)


def build_candidate_observations(
    grid_mod,
    inv,
    *,
    model,
    coords,
    x_mean,
    x_std,
    y_mean,
    y_std,
    iy,
    ix,
    ny,
    nx,
    lx,
    ly,
    mask_kind,
    src_size,
    sigma,
    soft_k,
    device,
    batch_size,
):
    dtype = x_mean.dtype
    iy_t = torch.as_tensor(iy, device=device, dtype=torch.long)
    ix_t = torch.as_tensor(ix, device=device, dtype=torch.long)
    values = []
    synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            batch = grid_mod.build_candidate_batch(
                inv,
                coords[start : start + batch_size],
                ny=ny,
                nx=nx,
                lx=lx,
                ly=ly,
                mask_kind=mask_kind,
                src_size=src_size,
                sigma=sigma,
                soft_k=soft_k,
                device=device,
                dtype=dtype,
            )
            pred_n = model((batch - x_mean) / (x_std + 1e-12))
            pred = pred_n * (y_std + 1e-12) + y_mean
            values.append(pred[:, 0, iy_t, ix_t].cpu().numpy())
    synchronize(device)
    elapsed = time.perf_counter() - t0
    return np.concatenate(values, axis=0), elapsed


def timed_grid_match(candidate_xy, candidate_obs, obs, repeats):
    repeats = max(1, int(repeats))
    t0 = time.perf_counter()
    best_idx = 0
    losses = None
    for _ in range(repeats):
        losses = np.mean((candidate_obs - obs[None, :]) ** 2, axis=1)
        best_idx = int(np.argmin(losses))
    elapsed = (time.perf_counter() - t0) / repeats
    return best_idx, float(losses[best_idx]), elapsed


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in details.groupby("method", sort=False, observed=True):
        loss = group["obs_mse"].to_numpy(dtype=float)
        runtime = group["online_runtime_s"].to_numpy(dtype=float)
        error = group["loc_error_m"].to_numpy(dtype=float)
        rows.append(
            {
                "method": method,
                "n_cases": len(group),
                "mean_obs_mse": np.mean(loss),
                "median_obs_mse": np.median(loss),
                "q25_obs_mse": np.quantile(loss, 0.25),
                "q75_obs_mse": np.quantile(loss, 0.75),
                "mean_online_runtime_s": np.mean(runtime),
                "median_online_runtime_s": np.median(runtime),
                "q25_online_runtime_s": np.quantile(runtime, 0.25),
                "q75_online_runtime_s": np.quantile(runtime, 0.75),
                "mean_loc_error_m": np.mean(error),
                "median_loc_error_m": np.median(error),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Uniform loss and runtime benchmark for gradient-only, fine-grid, and coarse-to-gradient inversion."
    )
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--project_root", type=str, default=str(root_default))
    parser.add_argument("--workdir", type=str, default="data_u0p2_dense_969")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="paper_runs/baseline_dense969_u0p2_w48_m16_d4_soft_rect/checkpoints/dense969_u0p2_w48_m16_d4_soft_rect_sk30_rawtarget.pt",
    )
    parser.add_argument(
        "--test_list",
        type=str,
        default="paper_runs/baseline_dense969_u0p2_w48_m16_d4_soft_rect/results/test_set_list_random.csv",
    )
    parser.add_argument("--cases_csv", type=str, default="data_u0p2_dense_969/cases.csv")
    parser.add_argument("--n_sensors", type=int, default=8)
    parser.add_argument("--sensor_strategy", type=str, default="grid")
    parser.add_argument("--sensor_seed", type=int, default=42)
    parser.add_argument("--sensor_margin_x", type=int, default=8)
    parser.add_argument("--sensor_margin_y", type=int, default=4)
    parser.add_argument("--coarse_stride", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--restart_seed", type=int, default=42)
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument(
        "--noise_mode",
        choices=["relative_max", "relative_rms", "absolute"],
        default="relative_max",
    )
    parser.add_argument("--noise_seed", type=int, default=42)
    parser.add_argument("--candidate_batch_size", type=int, default=4)
    parser.add_argument("--grid_timing_repeats", type=int, default=100)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--nx", type=int, default=384)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--Lx", type=float, default=9.0)
    parser.add_argument("--Ly", type=float, default=3.0)
    parser.add_argument("--out_dir", type=str, default="paper_runs/inverse_three_method_loss_time")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    inv = load_module(root / "visualization" / "inverse_src_from_sparse_c_fixed_u_continuous.py", "inverse_mod")
    grid_mod = load_module(root / "analysis" / "run_inverse_grid_search_baseline.py", "grid_mod")
    hybrid_mod = load_module(root / "analysis" / "run_inverse_hybrid_coarse_to_gradient.py", "hybrid_mod")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(root / args.ckpt, map_location=device, weights_only=False)
    meta = ckpt.get("meta", {})
    model = inv.FNO2D(
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
    fixed_u = float(meta["fixed_inlet_u"])
    mask_kind = str(meta.get("mask_kind", "soft_rect")).lower()
    src_size = float(meta.get("src_size", 0.2))
    sigma = float(meta.get("sigma", 0.12))
    soft_k = float(meta.get("soft_k", 30.0))

    cases = pd.read_csv(root / args.cases_csv)
    candidates = cases[np.isclose(cases["inlet_u"].astype(float), fixed_u)][["src_x", "src_y"]].drop_duplicates()
    candidates = candidates.sort_values(["src_x", "src_y"]).reset_index(drop=True)
    fine_xy = candidates[["src_x", "src_y"]].to_numpy(dtype=np.float32)
    dummy_obs = np.zeros((len(fine_xy), args.n_sensors), dtype=np.float32)
    coarse_xy, _ = hybrid_mod.select_coarse_library(fine_xy, dummy_obs, max(1, args.coarse_stride))
    if args.top_k > len(coarse_xy):
        raise ValueError("top_k exceeds coarse candidate count")

    iy, ix = inv.make_sensor_indices(
        args.ny,
        args.nx,
        args.n_sensors,
        args.sensor_strategy,
        args.sensor_seed,
        args.sensor_margin_x,
        args.sensor_margin_y,
    )

    # Warm up model and CUDA kernels before measuring candidate-library construction.
    _warm_obs, _ = build_candidate_observations(
        grid_mod,
        inv,
        model=model,
        coords=fine_xy[: args.candidate_batch_size],
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        iy=iy,
        ix=ix,
        ny=args.ny,
        nx=args.nx,
        lx=args.Lx,
        ly=args.Ly,
        mask_kind=mask_kind,
        src_size=src_size,
        sigma=sigma,
        soft_k=soft_k,
        device=device,
        batch_size=args.candidate_batch_size,
    )

    fine_obs, fine_setup_s = build_candidate_observations(
        grid_mod,
        inv,
        model=model,
        coords=fine_xy,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        iy=iy,
        ix=ix,
        ny=args.ny,
        nx=args.nx,
        lx=args.Lx,
        ly=args.Ly,
        mask_kind=mask_kind,
        src_size=src_size,
        sigma=sigma,
        soft_k=soft_k,
        device=device,
        batch_size=args.candidate_batch_size,
    )
    coarse_obs, coarse_setup_s = build_candidate_observations(
        grid_mod,
        inv,
        model=model,
        coords=coarse_xy,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        iy=iy,
        ix=ix,
        ny=args.ny,
        nx=args.nx,
        lx=args.Lx,
        ly=args.Ly,
        mask_kind=mask_kind,
        src_size=src_size,
        sigma=sigma,
        soft_k=soft_k,
        device=device,
        batch_size=args.candidate_batch_size,
    )

    test_ids = pd.read_csv(root / args.test_list)[["case_id"]]
    eval_cases = cases.merge(test_ids, on="case_id", how="inner")
    eval_cases = eval_cases[np.isclose(eval_cases["inlet_u"].astype(float), fixed_u)].reset_index(drop=True)
    if args.max_cases > 0:
        eval_cases = eval_cases.iloc[: args.max_cases].copy().reset_index(drop=True)
    if len(eval_cases) == 0:
        raise RuntimeError("No evaluation cases found")

    x_min, y_min = np.min(fine_xy, axis=0)
    x_max, y_max = np.max(fine_xy, axis=0)
    rng = np.random.default_rng(args.restart_seed)
    gradient_initial_xy = np.asarray(
        [[0.5 * (x_min + x_max), 0.5 * (y_min + y_max)]]
        + [[rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)] for _ in range(3)],
        dtype=np.float64,
    )

    common_opt = dict(
        inv=inv,
        model=model,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        iy=iy,
        ix=ix,
        ny=args.ny,
        nx=args.nx,
        lx=args.Lx,
        ly=args.Ly,
        mask_kind=mask_kind,
        src_size=src_size,
        sigma=sigma,
        soft_k=soft_k,
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        steps=args.steps,
        lr=args.lr,
    )

    # A short optimizer warm-up avoids charging first-use autograd kernels to the first case.
    first = eval_cases.iloc[0]
    first_file = inv.resolve_case_file(
        root / args.workdir / "exports",
        case_id_text(first.case_id),
        fixed_u,
        float(first.src_x),
        float(first.src_y),
    )
    first_df = inv.load_case_csv(first_file)
    _, _, _, _, first_c = inv.df_to_grid(first_df, Lx=args.Lx, Ly=args.Ly, nx=args.nx, ny=args.ny)
    first_sensor_obs = first_c[iy, ix].astype(np.float32)
    hybrid_mod.optimize_from_initializations(
        **{**common_opt, "obs": first_sensor_obs, "initial_xy": gradient_initial_xy, "steps": 3}
    )

    rows = []
    for case_index, case in enumerate(eval_cases.itertuples(index=False), start=1):
        gt_file = inv.resolve_case_file(
            root / args.workdir / "exports",
            case_id_text(case.case_id),
            fixed_u,
            float(case.src_x),
            float(case.src_y),
        )
        df_gt = inv.load_case_csv(gt_file)
        _, _, _, _, c_gt = inv.df_to_grid(df_gt, Lx=args.Lx, Ly=args.Ly, nx=args.nx, ny=args.ny)
        obs_clean = c_gt[iy, ix].astype(np.float32)
        if args.noise_mode == "relative_max":
            noise_sigma = float(args.noise_level) * max(float(np.max(np.abs(obs_clean))), 1e-12)
        elif args.noise_mode == "relative_rms":
            noise_sigma = float(args.noise_level) * max(float(np.sqrt(np.mean(obs_clean ** 2))), 1e-12)
        else:
            noise_sigma = float(args.noise_level)
        case_seed = np.random.SeedSequence([int(args.noise_seed), int(float(case.case_id))])
        noise_rng = np.random.default_rng(case_seed)
        noise = noise_rng.normal(0.0, noise_sigma, size=obs_clean.shape).astype(np.float32)
        obs = obs_clean + noise

        fine_idx, fine_loss, fine_runtime = timed_grid_match(
            fine_xy, fine_obs, obs, args.grid_timing_repeats
        )
        coarse_losses = np.mean((coarse_obs - obs[None, :]) ** 2, axis=1)
        top_idx = np.argsort(coarse_losses)[: args.top_k]
        hybrid_initial_xy = coarse_xy[top_idx]

        method_specs = [
            ("Gradient-only", gradient_initial_xy),
            ("Coarse-to-gradient", hybrid_initial_xy),
        ]
        if case_index % 2 == 0:
            method_specs.reverse()

        optimization_results = {}
        for method, initial_xy in method_specs:
            synchronize(device)
            t0 = time.perf_counter()
            result = hybrid_mod.optimize_from_initializations(
                **common_opt,
                obs=obs,
                initial_xy=initial_xy,
            )
            synchronize(device)
            optimization_results[method] = (result, time.perf_counter() - t0)

        true_x = float(case.src_x)
        true_y = float(case.src_y)
        fine_x, fine_y = fine_xy[fine_idx]
        rows.append(
            {
                "case_id": case.case_id,
                "method": "Fine-grid search",
                "true_sx": true_x,
                "true_sy": true_y,
                "est_sx": float(fine_x),
                "est_sy": float(fine_y),
                "obs_mse": fine_loss,
                "loc_error_m": float(np.hypot(fine_x - true_x, fine_y - true_y)),
                "online_runtime_s": fine_runtime,
                "noise_level": float(args.noise_level),
                "noise_mode": args.noise_mode,
                "noise_sigma": noise_sigma,
                "noise_seed": int(args.noise_seed),
            }
        )
        for method in ("Gradient-only", "Coarse-to-gradient"):
            result, runtime = optimization_results[method]
            rows.append(
                {
                    "case_id": case.case_id,
                    "method": method,
                    "true_sx": true_x,
                    "true_sy": true_y,
                    "est_sx": result["sx"],
                    "est_sy": result["sy"],
                    "obs_mse": result["loss"],
                    "loc_error_m": float(np.hypot(result["sx"] - true_x, result["sy"] - true_y)),
                    "online_runtime_s": runtime,
                    "noise_level": float(args.noise_level),
                    "noise_mode": args.noise_mode,
                    "noise_sigma": noise_sigma,
                    "noise_seed": int(args.noise_seed),
                }
            )

        if case_index == 1 or case_index % 10 == 0 or case_index == len(eval_cases):
            g = optimization_results["Gradient-only"]
            h = optimization_results["Coarse-to-gradient"]
            print(
                f"[{case_index:03d}/{len(eval_cases)}] case={case.case_id} "
                f"gradient={g[1]:.3f}s hybrid={h[1]:.3f}s grid={fine_runtime * 1e3:.4f}ms",
                flush=True,
            )

    details = pd.DataFrame(rows)
    method_order = ["Gradient-only", "Coarse-to-gradient", "Fine-grid search"]
    details["method"] = pd.Categorical(details["method"], categories=method_order, ordered=True)
    details = details.sort_values(["method", "case_id"]).reset_index(drop=True)
    summary = summarize(details)
    summary["noise_level"] = float(args.noise_level)
    summary["noise_mode"] = args.noise_mode
    summary["noise_seed"] = int(args.noise_seed)
    summary["mean_noise_sigma"] = details["noise_sigma"].mean()
    setup = pd.DataFrame(
        [
            {
                "method": "Coarse-to-gradient",
                "candidate_count": len(coarse_xy),
                "candidate_spacing_m": 0.125 * args.coarse_stride,
                "one_time_library_build_s": coarse_setup_s,
                "amortized_setup_s_per_case": coarse_setup_s / len(eval_cases),
            },
            {
                "method": "Fine-grid search",
                "candidate_count": len(fine_xy),
                "candidate_spacing_m": 0.125,
                "one_time_library_build_s": fine_setup_s,
                "amortized_setup_s_per_case": fine_setup_s / len(eval_cases),
            },
        ]
    )
    details.to_csv(out_dir / "inverse_three_method_loss_time_details.csv", index=False)
    summary.to_csv(out_dir / "inverse_three_method_loss_time_summary.csv", index=False)
    setup.to_csv(out_dir / "inverse_three_method_setup_time.csv", index=False)
    pd.DataFrame(gradient_initial_xy, columns=["src_x", "src_y"]).to_csv(
        out_dir / "gradient_initializations.csv", index=False
    )
    print("\nLOSS AND ONLINE RUNTIME\n" + summary.to_string(index=False))
    print("\nONE-TIME CANDIDATE LIBRARY SETUP\n" + setup.to_string(index=False))
    print(f"[SAVED] {out_dir}")


if __name__ == "__main__":
    main()
