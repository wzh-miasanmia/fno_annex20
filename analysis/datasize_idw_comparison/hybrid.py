import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def load_inverse_module(project_root: Path):
    script_path = project_root / "visualization" / "inverse_src_from_sparse_c_fixed_u_continuous.py"
    spec = importlib.util.spec_from_file_location("inverse_mod", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_coarse_library(candidate_xy: np.ndarray, candidate_obs: np.ndarray, stride: int):
    xs = np.unique(candidate_xy[:, 0])
    ys = np.unique(candidate_xy[:, 1])
    coarse_x = xs[::stride]
    coarse_y = ys[::stride]
    if not np.isclose(coarse_x[-1], xs[-1]):
        coarse_x = np.r_[coarse_x, xs[-1]]
    if not np.isclose(coarse_y[-1], ys[-1]):
        coarse_y = np.r_[coarse_y, ys[-1]]
    keep = np.isin(candidate_xy[:, 0], coarse_x) & np.isin(candidate_xy[:, 1], coarse_y)
    return candidate_xy[keep], candidate_obs[keep]


def optimize_from_initializations(
    inv,
    *,
    model,
    x_mean,
    x_std,
    y_mean,
    y_std,
    device,
    iy,
    ix,
    obs,
    initial_xy,
    ny,
    nx,
    lx,
    ly,
    mask_kind,
    src_size,
    sigma,
    soft_k,
    x_min,
    x_max,
    y_min,
    y_max,
    steps,
    lr,
):
    dtype = x_mean.dtype
    iy_t = torch.as_tensor(iy, device=device, dtype=torch.long)
    ix_t = torch.as_tensor(ix, device=device, dtype=torch.long)
    obs_t = torch.as_tensor(obs, device=device, dtype=dtype)

    initial_xy = np.asarray(initial_xy, dtype=np.float64)
    x_rng = float(x_max - x_min)
    y_rng = float(y_max - y_min)
    px = np.clip((initial_xy[:, 0] - x_min) / x_rng, 1e-5, 1.0 - 1e-5)
    py = np.clip((initial_xy[:, 1] - y_min) / y_rng, 1e-5, 1.0 - 1e-5)
    zx = nn.Parameter(torch.as_tensor(np.log(px) - np.log1p(-px), device=device, dtype=dtype))
    zy = nn.Parameter(torch.as_tensor(np.log(py) - np.log1p(-py), device=device, dtype=dtype))
    optimizer = torch.optim.Adam([zx, zy], lr=float(lr))

    n_init = len(initial_xy)
    best_loss = np.full(n_init, np.inf, dtype=np.float64)
    best_x = initial_xy[:, 0].copy()
    best_y = initial_xy[:, 1].copy()

    for _ in range(max(1, int(steps))):
        optimizer.zero_grad(set_to_none=True)
        sx = x_min + x_rng * torch.sigmoid(zx)
        sy = y_min + y_rng * torch.sigmoid(zy)
        batch = torch.cat(
            [
                inv._build_input_field_torch(
                    sx=sx[j],
                    sy=sy[j],
                    ny=ny,
                    nx=nx,
                    Lx=lx,
                    Ly=ly,
                    mask_kind=mask_kind,
                    src_size=src_size,
                    sigma=sigma,
                    soft_k=soft_k,
                    device=device,
                    dtype=dtype,
                )
                for j in range(n_init)
            ],
            dim=0,
        )
        pred_n = model((batch - x_mean) / (x_std + 1e-12))
        pred = pred_n * (y_std + 1e-12) + y_mean
        pred_obs = pred[:, 0, iy_t, ix_t]
        losses = torch.mean((pred_obs - obs_t.unsqueeze(0)) ** 2, dim=1)
        losses.sum().backward()

        loss_np = losses.detach().cpu().numpy()
        sx_np = sx.detach().cpu().numpy()
        sy_np = sy.detach().cpu().numpy()
        improved = loss_np < best_loss
        best_loss[improved] = loss_np[improved]
        best_x[improved] = sx_np[improved]
        best_y[improved] = sy_np[improved]
        optimizer.step()

    best_idx = int(np.argmin(best_loss))
    return {
        "sx": float(best_x[best_idx]),
        "sy": float(best_y[best_idx]),
        "loss": float(best_loss[best_idx]),
        "initialization_rank": best_idx + 1,
    }


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    err = details["loc_error_m"].to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "method": "Coarse grid plus gradient refinement",
                "n_cases": len(details),
                "mean_loc_error_m": np.mean(err),
                "median_loc_error_m": np.median(err),
                "std_loc_error_m": np.std(err, ddof=1),
                "q25_loc_error_m": np.quantile(err, 0.25),
                "q75_loc_error_m": np.quantile(err, 0.75),
                "q90_loc_error_m": np.quantile(err, 0.90),
                "success_le_0p125m": np.mean(err <= 0.125),
                "success_le_0p25m": np.mean(err <= 0.25),
                "mean_runtime_s_per_case": details["runtime_s"].mean(),
            }
        ]
    )


def main():
    ap = argparse.ArgumentParser(description="Coarse grid initialization followed by continuous gradient refinement.")
    root_default = Path(__file__).resolve().parents[1]
    ap.add_argument("--project_root", type=str, default=str(root_default))
    ap.add_argument("--workdir", type=str, default="data_u0p2_dense_969")
    ap.add_argument(
        "--ckpt",
        type=str,
        default="paper_runs/baseline_dense969_u0p2_w48_m16_d4_soft_rect/checkpoints/dense969_u0p2_w48_m16_d4_soft_rect_sk30_rawtarget.pt",
    )
    ap.add_argument(
        "--test_list",
        type=str,
        default="paper_runs/baseline_dense969_u0p2_w48_m16_d4_soft_rect/results/test_set_list_random.csv",
    )
    ap.add_argument("--cases_csv", type=str, default="data_u0p2_dense_969/cases.csv")
    ap.add_argument(
        "--candidate_library",
        type=str,
        default="paper_runs/inverse_grid_search_baseline_alltest_n16/candidate_sensor_library.npz",
    )
    ap.add_argument("--coarse_stride", type=int, default=2, help="Stride on the 0.125 m source grid; 2 gives 0.25 m.")
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max_cases", type=int, default=0, help="Optional smoke-test limit; 0 evaluates all cases.")
    ap.add_argument("--nx", type=int, default=384)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--Lx", type=float, default=9.0)
    ap.add_argument("--Ly", type=float, default=3.0)
    ap.add_argument("--out_dir", type=str, default="paper_runs/inverse_hybrid_coarse025_top4_alltest_n16")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    inv = load_inverse_module(root)
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

    library = np.load(root / args.candidate_library)
    candidate_xy = library["candidate_xy"].astype(np.float32)
    candidate_obs = library["candidate_obs"].astype(np.float32)
    iy = library["sensor_iy"].astype(int)
    ix = library["sensor_ix"].astype(int)
    coarse_xy, coarse_obs = select_coarse_library(candidate_xy, candidate_obs, max(1, args.coarse_stride))
    if args.top_k > len(coarse_xy):
        raise ValueError("top_k exceeds the number of coarse candidates")

    cases = pd.read_csv(root / args.cases_csv)
    test_ids = pd.read_csv(root / args.test_list)[["case_id"]]
    eval_cases = cases.merge(test_ids, on="case_id", how="inner")
    eval_cases = eval_cases[np.isclose(eval_cases["inlet_u"].astype(float), fixed_u)].reset_index(drop=True)
    if args.max_cases > 0:
        eval_cases = eval_cases.iloc[: args.max_cases].copy().reset_index(drop=True)
    x_min, y_min = np.min(candidate_xy, axis=0)
    x_max, y_max = np.max(candidate_xy, axis=0)

    rows = []
    for case_index, case in enumerate(eval_cases.itertuples(index=False), start=1):
        gt_file = inv.resolve_case_file(
            root / args.workdir / "exports",
            str(case.case_id),
            fixed_u,
            float(case.src_x),
            float(case.src_y),
        )
        df_gt = inv.load_case_csv(gt_file)
        _, _, _, _, c_gt = inv.df_to_grid(df_gt, Lx=args.Lx, Ly=args.Ly, nx=args.nx, ny=args.ny)
        obs = c_gt[iy, ix].astype(np.float32)
        coarse_losses = np.mean((coarse_obs - obs[None, :]) ** 2, axis=1)
        top_idx = np.argsort(coarse_losses)[: args.top_k]
        initial_xy = coarse_xy[top_idx]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = optimize_from_initializations(
            inv,
            model=model,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
            iy=iy,
            ix=ix,
            obs=obs,
            initial_xy=initial_xy,
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        runtime = time.perf_counter() - t0
        err = float(np.hypot(result["sx"] - float(case.src_x), result["sy"] - float(case.src_y)))
        rows.append(
            {
                "case_id": case.case_id,
                "true_sx": float(case.src_x),
                "true_sy": float(case.src_y),
                "coarse_best_sx": float(initial_xy[0, 0]),
                "coarse_best_sy": float(initial_xy[0, 1]),
                "coarse_best_obs_mse": float(coarse_losses[top_idx[0]]),
                "est_sx": result["sx"],
                "est_sy": result["sy"],
                "obs_mse": result["loss"],
                "loc_error_m": err,
                "selected_initialization_rank": result["initialization_rank"],
                "n_coarse_candidates": len(coarse_xy),
                "coarse_spacing_m": 0.125 * args.coarse_stride,
                "top_k": args.top_k,
                "steps": args.steps,
                "lr": args.lr,
                "runtime_s": runtime,
            }
        )
        if case_index == 1 or case_index % 10 == 0 or case_index == len(eval_cases):
            print(f"[{case_index:03d}/{len(eval_cases)}] case={case.case_id} error={err:.4f} m runtime={runtime:.2f} s")

    details = pd.DataFrame(rows)
    summary = summarize(details)
    summary["n_coarse_candidates"] = len(coarse_xy)
    summary["coarse_spacing_m"] = 0.125 * args.coarse_stride
    summary["top_k"] = args.top_k
    summary["steps_per_initialization"] = args.steps
    summary["learning_rate"] = args.lr
    details.to_csv(out_dir / "inverse_hybrid_details.csv", index=False)
    summary.to_csv(out_dir / "inverse_hybrid_summary.csv", index=False)
    pd.DataFrame(coarse_xy, columns=["src_x", "src_y"]).to_csv(out_dir / "coarse_candidate_grid.csv", index=False)
    print(summary.to_string(index=False))
    print(f"[SAVED] {out_dir}")


if __name__ == "__main__":
    main()
