import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_inverse_module(project_root: Path):
    script_path = project_root / "visualization" / "inverse_src_from_sparse_c_fixed_u_continuous.py"
    spec = importlib.util.spec_from_file_location("inverse_mod", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_candidate_batch(inv, coords, *, ny, nx, lx, ly, mask_kind, src_size, sigma, soft_k, device, dtype):
    fields = []
    for sx, sy in coords:
        fields.append(
            inv._build_input_field_torch(
                sx=torch.tensor(float(sx), device=device, dtype=dtype),
                sy=torch.tensor(float(sy), device=device, dtype=dtype),
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
        )
    return torch.cat(fields, dim=0)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    err = df["loc_error_m"].to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "method": "Discrete grid search",
                "n_cases": len(df),
                "mean_loc_error_m": np.mean(err),
                "median_loc_error_m": np.median(err),
                "std_loc_error_m": np.std(err, ddof=1),
                "q25_loc_error_m": np.quantile(err, 0.25),
                "q75_loc_error_m": np.quantile(err, 0.75),
                "q90_loc_error_m": np.quantile(err, 0.90),
                "success_le_0p125m": np.mean(err <= 0.125),
                "success_le_0p25m": np.mean(err <= 0.25),
            }
        ]
    )


def main():
    ap = argparse.ArgumentParser(
        description="Discrete inverse grid-search baseline using the same trained FNO and sensor observations as the gradient method."
    )
    ap.add_argument("--project_root", type=str, default=str(Path(__file__).resolve().parents[1]))
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
    ap.add_argument("--n_sensors", type=int, default=16)
    ap.add_argument("--sensor_strategy", choices=["random", "grid", "grid_centered", "downstream"], default="grid")
    ap.add_argument("--sensor_seed", type=int, default=42)
    ap.add_argument("--sensor_margin_x", type=int, default=8)
    ap.add_argument("--sensor_margin_y", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--nx", type=int, default=384)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--Lx", type=float, default=9.0)
    ap.add_argument("--Ly", type=float, default=3.0)
    ap.add_argument("--out_dir", type=str, default="paper_runs/inverse_grid_search_baseline_alltest_n16")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    workdir = root / args.workdir
    ckpt_path = root / args.ckpt
    test_list_path = root / args.test_list
    cases_path = root / args.cases_csv
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    inv = load_inverse_module(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    dtype = x_mean.dtype
    fixed_u = float(meta["fixed_inlet_u"])
    mask_kind = str(meta.get("mask_kind", "soft_rect")).lower()
    src_size = float(meta.get("src_size", 0.2))
    sigma = float(meta.get("sigma", 0.12))
    soft_k = float(meta.get("soft_k", 30.0))

    cases = pd.read_csv(cases_path)
    candidates = cases[np.isclose(cases["inlet_u"].astype(float), fixed_u)][["src_x", "src_y"]].drop_duplicates()
    candidates = candidates.sort_values(["src_x", "src_y"]).reset_index(drop=True)
    candidate_xy = candidates[["src_x", "src_y"]].to_numpy(dtype=np.float32)
    iy, ix = inv.make_sensor_indices(
        args.ny,
        args.nx,
        args.n_sensors,
        args.sensor_strategy,
        args.sensor_seed,
        args.sensor_margin_x,
        args.sensor_margin_y,
    )
    iy_t = torch.as_tensor(iy, device=device, dtype=torch.long)
    ix_t = torch.as_tensor(ix, device=device, dtype=torch.long)

    candidate_obs = []
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(candidate_xy), args.batch_size):
            coords = candidate_xy[start : start + args.batch_size]
            x = build_candidate_batch(
                inv,
                coords,
                ny=args.ny,
                nx=args.nx,
                lx=args.Lx,
                ly=args.Ly,
                mask_kind=mask_kind,
                src_size=src_size,
                sigma=sigma,
                soft_k=soft_k,
                device=device,
                dtype=dtype,
            )
            y = model((x - x_mean) / (x_std + 1e-12))
            y = y * (y_std + 1e-12) + y_mean
            candidate_obs.append(y[:, 0, iy_t, ix_t].cpu().numpy())
    torch.cuda.synchronize() if device.type == "cuda" else None
    library_time = time.perf_counter() - t0
    candidate_obs = np.concatenate(candidate_obs, axis=0)

    test_ids = pd.read_csv(test_list_path)[["case_id"]]
    eval_cases = cases.merge(test_ids, on="case_id", how="inner")
    eval_cases = eval_cases[np.isclose(eval_cases["inlet_u"].astype(float), fixed_u)].reset_index(drop=True)
    rows = []
    matching_time = 0.0
    for case in eval_cases.itertuples(index=False):
        gt_file = inv.resolve_case_file(
            workdir / "exports", str(case.case_id), fixed_u, float(case.src_x), float(case.src_y)
        )
        df_gt = inv.load_case_csv(gt_file)
        _, _, _, _, c_gt = inv.df_to_grid(df_gt, Lx=args.Lx, Ly=args.Ly, nx=args.nx, ny=args.ny)
        obs = c_gt[iy, ix].astype(np.float32)
        tm = time.perf_counter()
        losses = np.mean((candidate_obs - obs[None, :]) ** 2, axis=1)
        best_idx = int(np.argmin(losses))
        matching_time += time.perf_counter() - tm
        est_x, est_y = candidate_xy[best_idx]
        err = float(np.hypot(est_x - float(case.src_x), est_y - float(case.src_y)))
        rows.append(
            {
                "case_id": case.case_id,
                "true_sx": float(case.src_x),
                "true_sy": float(case.src_y),
                "est_sx": float(est_x),
                "est_sy": float(est_y),
                "loc_error_m": err,
                "obs_mse": float(losses[best_idx]),
                "n_sensors": args.n_sensors,
                "sensor_strategy": args.sensor_strategy,
                "sensor_seed": args.sensor_seed,
            }
        )

    details = pd.DataFrame(rows)
    summary = summarize(details)
    summary["n_candidates"] = len(candidate_xy)
    summary["candidate_spacing_m"] = 0.125
    summary["library_build_time_s"] = library_time
    summary["mean_matching_time_ms_per_case"] = 1000.0 * matching_time / len(details)
    details.to_csv(out_dir / "inverse_grid_search_details.csv", index=False)
    summary.to_csv(out_dir / "inverse_grid_search_summary.csv", index=False)
    np.savez_compressed(
        out_dir / "candidate_sensor_library.npz",
        candidate_xy=candidate_xy,
        candidate_obs=candidate_obs,
        sensor_iy=iy,
        sensor_ix=ix,
    )
    print(summary.to_string(index=False))
    print(f"[SAVED] {out_dir}")


if __name__ == "__main__":
    main()
