import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def safe_float_tag(x: float) -> str:
    return f"{x:.4f}"


def normalize_case_id(case_id) -> str:
    text = str(case_id).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def resolve_case_file(exports_dir: Path, case_id: str, inlet_u: float, sx: float, sy: float) -> Path:
    exact = exports_dir / (
        f"case_{case_id}_u{safe_float_tag(inlet_u)}_sx{safe_float_tag(sx)}_sy{safe_float_tag(sy)}.csv"
    )
    if exact.exists():
        return exact
    hits = sorted(exports_dir.glob(f"case_{case_id}_*.csv"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No exported CSV found for case_id={case_id} under {exports_dir}")


def load_case_csv(case_path: Path) -> pd.DataFrame:
    df = pd.read_csv(case_path)
    required = ["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Passive Scalar", "X (m)", "Y (m)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {case_path}. Got: {list(df.columns)}")
    return df[required].copy()


def df_to_grid(df: pd.DataFrame, Lx: float = 9.0, Ly: float = 3.0, nx: int = 128 * 3, ny: int = 128):
    x = np.linspace(0.0, Lx, nx, dtype=np.float32)
    y = np.linspace(0.0, Ly, ny, dtype=np.float32)
    xg, yg = np.meshgrid(x, y)
    pts = df[["X (m)", "Y (m)"]].to_numpy(dtype=np.float64)
    cc = df["Passive Scalar"].to_numpy(dtype=np.float64)

    lin = griddata(pts, cc, (xg, yg), method="linear")
    near = griddata(pts, cc, (xg, yg), method="nearest")
    c = np.where(np.isnan(lin), near, lin).astype(np.float32)
    return x, y, c


def split_indices_random(n: int, val_ratio=0.15, test_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test = idx[:n_test]
    val = idx[n_test:n_test + n_val]
    train = idx[n_test + n_val:]
    return train.tolist(), val.tolist(), test.tolist()


def split_indices_spatial_checkerboard(rows: pd.DataFrame):
    x_vals = np.array(sorted(pd.unique(rows["src_x"].astype(float))), dtype=np.float64)
    y_vals = np.array(sorted(pd.unique(rows["src_y"].astype(float))), dtype=np.float64)
    x_lookup = {float(v): i for i, v in enumerate(x_vals.tolist())}
    y_lookup = {float(v): i for i, v in enumerate(y_vals.tolist())}
    train, val, test = [], [], []
    for idx, row in rows.reset_index(drop=True).iterrows():
        ix = x_lookup[float(row["src_x"])]
        iy = y_lookup[float(row["src_y"])]
        parity = (ix + iy) % 2
        y_band = iy % 4
        if parity == 0:
            train.append(int(idx))
        elif y_band in (0, 1):
            test.append(int(idx))
        else:
            val.append(int(idx))
    return train, val, test


def _uniform_indices(n_total: int, n_pick: int) -> np.ndarray:
    if n_pick >= n_total:
        return np.arange(n_total, dtype=int)
    idx = np.round(np.linspace(0, n_total - 1, n_pick)).astype(int)
    idx = np.unique(idx)
    if len(idx) == n_pick:
        return idx.astype(int)
    missing = [i for i in range(n_total) if i not in set(idx.tolist())]
    idx = np.concatenate([idx, np.array(missing[: max(0, n_pick - len(idx))], dtype=int)])
    idx.sort()
    return idx.astype(int)


def split_indices_spatial_uniform(rows: pd.DataFrame, val_ratio=0.15, test_ratio=0.15, seed=42):
    rows = rows.reset_index(drop=True)
    x_vals = np.array(sorted(pd.unique(rows["src_x"].astype(float))), dtype=np.float64)
    y_vals = np.array(sorted(pd.unique(rows["src_y"].astype(float))), dtype=np.float64)
    nx_all, ny_all = len(x_vals), len(y_vals)
    n_total = len(rows)
    n_test_target = int(round(n_total * float(test_ratio)))
    n_val_target = int(round(n_total * float(val_ratio)))
    aspect = float(nx_all) / max(float(ny_all), 1.0)

    best = None
    for nx_pick in range(1, nx_all + 1):
        for ny_pick in range(1, ny_all + 1):
            n_pick = nx_pick * ny_pick
            score = (
                abs(n_pick - n_test_target),
                abs((float(nx_pick) / max(float(ny_pick), 1.0)) - aspect),
                abs(nx_pick - ny_pick),
            )
            if best is None or score < best[0]:
                best = (score, nx_pick, ny_pick)

    _, nx_pick, ny_pick = best
    x_sel = set(_uniform_indices(nx_all, nx_pick).tolist())
    y_sel = set(_uniform_indices(ny_all, ny_pick).tolist())
    x_lookup = {float(v): i for i, v in enumerate(x_vals.tolist())}
    y_lookup = {float(v): i for i, v in enumerate(y_vals.tolist())}

    test, remaining = [], []
    for idx, row in rows.iterrows():
        ix = x_lookup[float(row["src_x"])]
        iy = y_lookup[float(row["src_y"])]
        if ix in x_sel and iy in y_sel:
            test.append(int(idx))
        else:
            remaining.append(int(idx))

    rng = np.random.default_rng(seed)
    remaining = np.array(remaining, dtype=int)
    rng.shuffle(remaining)
    n_val = min(n_val_target, max(1, len(remaining) - 1))
    val = remaining[:n_val].tolist()
    train = remaining[n_val:].tolist()
    return train, val, test


def split_indices_spatial_left_right(rows: pd.DataFrame, val_ratio=0.15, test_ratio=0.15, seed=42):
    rows = rows.reset_index(drop=True)
    x_vals = np.array(sorted(pd.unique(rows["src_x"].astype(float))), dtype=np.float64)
    n_x = len(x_vals)
    n_total = len(rows)
    n_test_target = int(round(n_total * float(test_ratio)))
    n_val_target = int(round(n_total * float(val_ratio)))

    best = None
    n_y = int(rows["src_y"].nunique())
    for n_test_x in range(1, n_x):
        n_test = n_test_x * n_y
        score = abs(n_test - n_test_target)
        if best is None or score < best[0]:
            best = (score, n_test_x)

    _, n_test_x = best
    test_x_vals = set(x_vals[-n_test_x:].tolist())
    test, remaining = [], []
    for idx, row in rows.iterrows():
        if float(row["src_x"]) in test_x_vals:
            test.append(int(idx))
        else:
            remaining.append(int(idx))

    rng = np.random.default_rng(seed)
    remaining = np.array(remaining, dtype=int)
    rng.shuffle(remaining)
    n_val = min(n_val_target, max(1, len(remaining) - 1))
    val = remaining[:n_val].tolist()
    train = remaining[n_val:].tolist()
    return train, val, test


def resolve_split_indices(rows: pd.DataFrame, split_mode="random", val_ratio=0.15, test_ratio=0.15, seed=42):
    mode = str(split_mode).lower().strip()
    if mode == "random":
        return split_indices_random(len(rows), val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    if mode == "spatial_checkerboard":
        return split_indices_spatial_checkerboard(rows)
    if mode == "spatial_uniform":
        return split_indices_spatial_uniform(rows, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    if mode == "spatial_left_right":
        return split_indices_spatial_left_right(rows, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    raise ValueError("split_mode must be one of: random, spatial_checkerboard, spatial_uniform, spatial_left_right")


def l2_rel(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(a)) + eps
    return num / den


def rect_mask(xg: np.ndarray, yg: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
    ya, yb = (y0, y1) if y0 <= y1 else (y1, y0)
    return ((xg >= xa) & (xg <= xb) & (yg >= ya) & (yg <= yb)).astype(np.float32)


def masked_mean(field: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    w = float(np.sum(mask))
    if w <= 0.0:
        return float("nan")
    return float(np.sum(field * mask) / (w + eps))


def compute_cre(c_field: np.ndarray, outlet_mask: np.ndarray, bz_mask: np.ndarray):
    c_out = masked_mean(c_field, outlet_mask)
    c_bz = masked_mean(c_field, bz_mask)
    if not np.isfinite(c_out) or not np.isfinite(c_bz) or abs(c_bz) <= 1e-12:
        return float("nan"), c_out, c_bz
    return float(c_out / c_bz), c_out, c_bz


def make_xy_grids(Lx: float, Ly: float, nx: int, ny: int):
    xs = np.linspace(0.0, Lx, nx, dtype=np.float32)
    ys = np.linspace(0.0, Ly, ny, dtype=np.float32)
    return np.meshgrid(xs, ys)


def idw_weights(train_xy: np.ndarray, query_xy: np.ndarray, k: int, power: float, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    d2 = np.sum((train_xy - query_xy[None, :]) ** 2, axis=1)
    order = np.argsort(d2)
    idx = order[: min(k, len(order))]
    d = np.sqrt(d2[idx])
    if np.any(d <= eps):
        exact = np.argmin(d)
        w = np.zeros_like(d, dtype=np.float32)
        w[exact] = 1.0
        return idx, w
    w = 1.0 / np.power(d + eps, power)
    w = (w / np.sum(w)).astype(np.float32)
    return idx, w


def main():
    ap = argparse.ArgumentParser(description="Non-neural source-space interpolation baseline for fixed-u source-position task.")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--workdir", type=str, default="data_u0p2_dense_969")
    ap.add_argument("--exports", type=str, default="exports")
    ap.add_argument("--cases", type=str, default="cases.csv")
    ap.add_argument("--fixed_inlet_u", type=float, default=0.2)
    ap.add_argument("--inlet_u_tol", type=float, default=1e-8)
    ap.add_argument("--split_mode", type=str, default="random", choices=["random", "spatial_checkerboard", "spatial_uniform", "spatial_left_right"])
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--Lx", type=float, default=9.0)
    ap.add_argument("--Ly", type=float, default=3.0)
    ap.add_argument("--nx", type=int, default=128 * 3)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--outlet_width_x", type=float, default=0.05)
    ap.add_argument("--outlet_y0", type=float, default=0.01)
    ap.add_argument("--outlet_y1", type=float, default=0.25)
    ap.add_argument("--bz_x0", type=float, default=0.6)
    ap.add_argument("--bz_x1", type=float, default=8.4)
    ap.add_argument("--bz_y0", type=float, default=0.075)
    ap.add_argument("--bz_y1", type=float, default=1.8)
    ap.add_argument("--out_cases_csv", type=str, default="")
    ap.add_argument("--out_summary_csv", type=str, default="")
    args = ap.parse_args()

    workdir = (root / args.workdir).resolve() if not Path(args.workdir).is_absolute() else Path(args.workdir).resolve()
    exports_dir = workdir / args.exports
    cases_path = workdir / args.cases
    rows = pd.read_csv(cases_path)
    rows = rows[np.abs(rows["inlet_u"].astype(float) - float(args.fixed_inlet_u)) <= float(args.inlet_u_tol)].copy()
    rows = rows.reset_index(drop=True)

    train_idx, val_idx, test_idx = resolve_split_indices(
        rows,
        split_mode=args.split_mode,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    )
    train_rows = rows.iloc[train_idx].reset_index(drop=True)
    test_rows = rows.iloc[test_idx].reset_index(drop=True)

    print(f"[INFO] fixed_u={args.fixed_inlet_u:.4f} split={args.split_mode} split_seed={args.split_seed}")
    print(f"[INFO] n_total={len(rows)} n_train={len(train_rows)} n_test={len(test_rows)}")

    xg, yg = make_xy_grids(args.Lx, args.Ly, args.nx, args.ny)
    outlet_mask = rect_mask(xg, yg, args.Lx - args.outlet_width_x, args.Lx, args.outlet_y0, args.outlet_y1)
    bz_mask = rect_mask(xg, yg, args.bz_x0, args.bz_x1, args.bz_y0, args.bz_y1)

    train_xy = train_rows[["src_x", "src_y"]].to_numpy(dtype=np.float32)
    train_fields: List[np.ndarray] = []
    for _, row in train_rows.iterrows():
        case_id = normalize_case_id(row["case_id"])
        csv_path = resolve_case_file(exports_dir, case_id, float(row["inlet_u"]), float(row["src_x"]), float(row["src_y"]))
        _, _, c = df_to_grid(load_case_csv(csv_path), Lx=args.Lx, Ly=args.Ly, nx=args.nx, ny=args.ny)
        train_fields.append(c.astype(np.float32))
    train_fields_arr = np.stack(train_fields, axis=0)

    metrics = []
    for i, row in test_rows.iterrows():
        case_id = normalize_case_id(row["case_id"])
        sx = float(row["src_x"])
        sy = float(row["src_y"])
        csv_path = resolve_case_file(exports_dir, case_id, float(row["inlet_u"]), sx, sy)
        _, _, c_gt = df_to_grid(load_case_csv(csv_path), Lx=args.Lx, Ly=args.Ly, nx=args.nx, ny=args.ny)

        idx, w = idw_weights(train_xy, np.array([sx, sy], dtype=np.float32), k=args.k, power=args.power)
        c_pr = np.tensordot(w, train_fields_arr[idx], axes=(0, 0)).astype(np.float32)

        err = c_pr - c_gt
        mse = float(np.mean(err ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(err)))
        rel_l2 = float(l2_rel(c_gt, c_pr))
        cre_gt, c_out_gt, c_bz_gt = compute_cre(c_gt, outlet_mask, bz_mask)
        cre_pr, c_out_pr, c_bz_pr = compute_cre(c_pr, outlet_mask, bz_mask)
        cre_abs_err = float(abs(cre_pr - cre_gt)) if np.isfinite(cre_gt) and np.isfinite(cre_pr) else float("nan")
        cre_rel_err = float(abs(cre_pr - cre_gt) / (abs(cre_gt) + 1e-12)) if np.isfinite(cre_gt) and np.isfinite(cre_pr) else float("nan")

        c_gt_abs = np.abs(c_gt)
        p90 = np.percentile(c_gt_abs, 90.0)
        high_mask = c_gt_abs >= p90
        high_rmse = float(np.sqrt(np.mean((err[high_mask]) ** 2))) if np.any(high_mask) else float("nan")
        high_mae = float(np.mean(np.abs(err[high_mask]))) if np.any(high_mask) else float("nan")

        metrics.append(
            {
                "case_id": case_id,
                "inlet_u": float(row["inlet_u"]),
                "src_x": sx,
                "src_y": sy,
                "split_mode": args.split_mode,
                "split_seed": args.split_seed,
                "method": f"srcpos_idw_k{args.k}_p{args.power:g}",
                "k": int(args.k),
                "power": float(args.power),
                "rmse_c": rmse,
                "mae_c": mae,
                "rel_l2_c": rel_l2,
                "cre_gt": cre_gt,
                "cre_pred": cre_pr,
                "abs_err_cre": cre_abs_err,
                "rel_err_cre": cre_rel_err,
                "c_out_gt": c_out_gt,
                "c_out_pred": c_out_pr,
                "c_bz_gt": c_bz_gt,
                "c_bz_pred": c_bz_pr,
                "high_rmse_c": high_rmse,
                "high_mae_c": high_mae,
                "neighbor_case_ids": ",".join(train_rows.iloc[idx]["case_id"].astype(str).tolist()),
                "neighbor_weights": ",".join(f"{float(v):.6f}" for v in w.tolist()),
            }
        )
        if (i + 1) % 20 == 0 or (i + 1) == len(test_rows):
            print(f"[INFO] processed {i + 1}/{len(test_rows)} test cases")

    df_metrics = pd.DataFrame(metrics)
    df_summary = pd.DataFrame(
        [
            {
                "experiment": f"source_interp_{args.split_mode}_u{safe_float_tag(args.fixed_inlet_u)}_k{args.k}_p{args.power:g}",
                "method": "source_space_idw",
                "subset": "test",
                "n_cases": int(len(df_metrics)),
                "fixed_inlet_u": float(args.fixed_inlet_u),
                "split_mode": args.split_mode,
                "split_seed": int(args.split_seed),
                "k": int(args.k),
                "power": float(args.power),
                "mean_rel_l2_c": float(df_metrics["rel_l2_c"].mean()),
                "std_rel_l2_c": float(df_metrics["rel_l2_c"].std(ddof=0)),
                "mean_rmse_c": float(df_metrics["rmse_c"].mean()),
                "mean_mae_c": float(df_metrics["mae_c"].mean()),
                "mean_abs_err_cre": float(df_metrics["abs_err_cre"].mean()),
                "mean_rel_err_cre": float(df_metrics["rel_err_cre"].mean()),
                "mean_high_rmse_c": float(df_metrics["high_rmse_c"].mean()),
                "mean_high_mae_c": float(df_metrics["high_mae_c"].mean()),
            }
        ]
    )

    tag = f"{args.split_mode}_u{safe_float_tag(args.fixed_inlet_u)}_k{args.k}_p{args.power:g}".replace(".", "p")
    out_cases_csv = Path(args.out_cases_csv) if args.out_cases_csv else root / "analysis" / "results" / f"cases_source_interp_{tag}.csv"
    out_summary_csv = Path(args.out_summary_csv) if args.out_summary_csv else root / "analysis" / "results" / f"summary_source_interp_{tag}.csv"
    if not out_cases_csv.is_absolute():
        out_cases_csv = (root / out_cases_csv).resolve()
    if not out_summary_csv.is_absolute():
        out_summary_csv = (root / out_summary_csv).resolve()
    out_cases_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(out_cases_csv, index=False)
    df_summary.to_csv(out_summary_csv, index=False)
    print(f"[SAVED] {out_cases_csv}")
    print(f"[SAVED] {out_summary_csv}")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
