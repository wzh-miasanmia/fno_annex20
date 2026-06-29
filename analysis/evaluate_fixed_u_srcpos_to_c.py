import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import griddata


def resolve_dataset_context(
    default_workdir: Path,
    workdir_arg: str,
    exports_arg: str,
    cases_arg: str,
    meta: Dict[str, object],
):
    workdir = Path(workdir_arg).resolve()
    exports = str(exports_arg)
    cases = str(cases_arg)

    meta_workdir = meta.get("dataset_workdir", None)
    meta_exports_dir = meta.get("dataset_exports_dir", None)
    meta_cases_path = meta.get("dataset_cases_path", None)

    using_default_workdir = workdir == default_workdir.resolve()
    if using_default_workdir and meta_workdir:
        workdir = Path(str(meta_workdir)).resolve()
        if meta_exports_dir:
            try:
                exports = Path(str(meta_exports_dir)).resolve().relative_to(workdir).as_posix()
            except Exception:
                exports = str(Path(str(meta_exports_dir)).resolve())
        if meta_cases_path:
            try:
                cases = Path(str(meta_cases_path)).resolve().relative_to(workdir).as_posix()
            except Exception:
                cases = str(Path(str(meta_cases_path)).resolve())
        print(f"[INFO] Using dataset context from checkpoint meta: workdir={workdir}")
    elif meta_workdir:
        meta_workdir_path = Path(str(meta_workdir)).resolve()
        if workdir != meta_workdir_path:
            print(
                f"[WARN] Requested workdir={workdir} differs from checkpoint dataset_workdir={meta_workdir_path}. "
                "Make sure this is intentional."
            )

    return workdir, exports, cases


def safe_float_tag(x: float) -> str:
    return f"{x:.4f}"


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
    if not {"src_x", "src_y"}.issubset(rows.columns):
        raise ValueError("rows must contain src_x and src_y for spatial split.")

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

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError(
            f"Spatial checkerboard split produced empty subset(s): "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )
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


def split_indices_spatial_uniform(
    rows: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    if not {"src_x", "src_y"}.issubset(rows.columns):
        raise ValueError("rows must contain src_x and src_y for spatial split.")

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

    test = []
    remaining = []
    for idx, row in rows.iterrows():
        ix = x_lookup[float(row["src_x"])]
        iy = y_lookup[float(row["src_y"])]
        if ix in x_sel and iy in y_sel:
            test.append(int(idx))
        else:
            remaining.append(int(idx))

    if len(test) == 0 or len(remaining) == 0:
        raise RuntimeError(
            f"Spatial uniform split produced empty subset(s): train+val={len(remaining)}, test={len(test)}"
        )

    rng = np.random.default_rng(seed)
    remaining = np.array(remaining, dtype=int)
    rng.shuffle(remaining)
    n_val = min(n_val_target, max(1, len(remaining) - 1))
    val = remaining[:n_val].tolist()
    train = remaining[n_val:].tolist()

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError(
            f"Spatial uniform split produced empty subset(s): "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )
    return train, val, test


def split_indices_spatial_left_right(
    rows: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    if not {"src_x", "src_y"}.issubset(rows.columns):
        raise ValueError("rows must contain src_x and src_y for spatial split.")

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

    test = []
    remaining = []
    for idx, row in rows.iterrows():
        if float(row["src_x"]) in test_x_vals:
            test.append(int(idx))
        else:
            remaining.append(int(idx))

    if len(test) == 0 or len(remaining) == 0:
        raise RuntimeError(
            f"Spatial left-right split produced empty subset(s): train+val={len(remaining)}, test={len(test)}"
        )

    rng = np.random.default_rng(seed)
    remaining = np.array(remaining, dtype=int)
    rng.shuffle(remaining)
    n_val = min(n_val_target, max(1, len(remaining) - 1))
    val = remaining[:n_val].tolist()
    train = remaining[n_val:].tolist()

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError(
            f"Spatial left-right split produced empty subset(s): "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )
    return train, val, test


def resolve_split_indices(
    rows: pd.DataFrame,
    split_mode: str = "random",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
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


def infer_arch_from_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    fc0_w = state["fc0.weight"]
    fc2_w = state["fc2.weight"]
    spec0_w = state["spec_layers.0.weight"]
    depth = sum(1 for k in state.keys() if k.startswith("spec_layers.") and k.endswith(".weight"))
    return {
        "in_ch": int(fc0_w.shape[1]),
        "out_ch": int(fc2_w.shape[0]),
        "width": int(fc0_w.shape[0]),
        "modes1": int(spec0_w.shape[2]),
        "modes2": int(spec0_w.shape[3]),
        "depth": int(depth),
    }


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

    def interp(vals: np.ndarray) -> np.ndarray:
        lin = griddata(pts, vals, (xg, yg), method="linear")
        near = griddata(pts, vals, (xg, yg), method="nearest")
        return np.where(np.isnan(lin), near, lin).astype(np.float32)

    c = interp(cc)
    return x, y, c


def make_xy_grids(Lx: float, Ly: float, nx: int, ny: int):
    xs = np.linspace(0.0, Lx, nx, dtype=np.float32)
    ys = np.linspace(0.0, Ly, ny, dtype=np.float32)
    return np.meshgrid(xs, ys)


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


def build_source_mask_rect(xg: np.ndarray, yg: np.ndarray, src_x: float, src_y: float, src_size: float) -> np.ndarray:
    hx = 0.5 * src_size
    hy = 0.5 * src_size
    return (
        (xg >= (src_x - hx)) & (xg <= (src_x + hx)) &
        (yg >= (src_y - hy)) & (yg <= (src_y + hy))
    ).astype(np.float32)


def build_source_mask_gaussian(xg: np.ndarray, yg: np.ndarray, src_x: float, src_y: float, sigma: float) -> np.ndarray:
    r2 = (xg - src_x) ** 2 + (yg - src_y) ** 2
    return np.exp(-0.5 * r2 / (sigma ** 2)).astype(np.float32)


def build_source_mask_soft_rect_sigmoid(
    xg: np.ndarray,
    yg: np.ndarray,
    src_x: float,
    src_y: float,
    src_size: float,
    soft_k: float = 80.0,
) -> np.ndarray:
    hx = 0.5 * src_size
    hy = 0.5 * src_size
    dx = np.abs(xg - src_x) - hx
    dy = np.abs(yg - src_y) - hy
    sdf = np.maximum(dx, dy)
    z = np.clip(soft_k * sdf, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(z))).astype(np.float32)


def build_input_field(
    sx: float,
    sy: float,
    ny: int,
    nx: int,
    Lx: float,
    Ly: float,
    mask_kind: str,
    src_size: float,
    sigma: float,
    soft_k: float,
) -> np.ndarray:
    xx01 = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    yy01 = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    xc, yc = np.meshgrid(xx01, yy01)
    xc = xc[None, ...].astype(np.float32)
    yc = yc[None, ...].astype(np.float32)
    x_phys = np.linspace(0.0, Lx, nx, dtype=np.float32)
    y_phys = np.linspace(0.0, Ly, ny, dtype=np.float32)
    xg, yg = np.meshgrid(x_phys, y_phys)

    mk = str(mask_kind).lower().strip().replace("-", "_")
    if mk in ["softrect", "soft_rect(sigmoid)", "softrect(sigmoid)"]:
        mk = "soft_rect"
    if mk == "gaussian":
        mk = "gauss"
    if mk == "rect":
        s = build_source_mask_rect(xg, yg, sx, sy, src_size)
    elif mk == "gauss":
        s = build_source_mask_gaussian(xg, yg, sx, sy, sigma)
    elif mk == "soft_rect":
        s = build_source_mask_soft_rect_sigmoid(xg, yg, sx, sy, src_size, soft_k)
    else:
        raise ValueError("mask_kind must be one of: 'rect', 'gauss', 'soft_rect'")

    s = s[None, ...].astype(np.float32)
    x_field = np.concatenate([s, xc, yc], axis=0)
    return x_field[None, ...]


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, 2, dtype=torch.float32)
        )

    def compl_mul2d(self, input_fft, weight):
        return torch.einsum("bixy,ioxy->boxy", input_fft, weight)

    def forward(self, x):
        bsz, _, ny, nx = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        m1 = min(self.modes1, ny)
        m2 = min(self.modes2, x_ft.shape[-1])
        out_ft = torch.zeros(bsz, self.out_channels, ny, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        weight = torch.view_as_complex(self.weight)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], weight[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(ny, nx), norm="ortho")


class FNO2D(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, width=48, modes1=16, modes2=16, depth=4):
        super().__init__()
        self.fc0 = nn.Conv2d(in_ch, width, kernel_size=1)
        self.spec_layers = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)])
        self.w_layers = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)])
        self.act = nn.GELU()
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.fc0(x)
        for spec, w in zip(self.spec_layers, self.w_layers):
            x = self.act(spec(x) + w(x))
        x = self.act(self.fc1(x))
        return self.fc2(x)


def pick_rows(
    df_cases: pd.DataFrame,
    subset: str,
    case_ids_path: Path = None,
    split_mode: str = "random",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    if case_ids_path is not None:
        df_ids = pd.read_csv(case_ids_path)
        if "case_id" not in df_ids.columns:
            raise ValueError(f"{case_ids_path} must contain column 'case_id'.")
        ids = set(df_ids["case_id"].astype(str))
        out = df_cases[df_cases["case_id"].astype(str).isin(ids)].copy()
        out["split"] = "custom"
        return out.reset_index(drop=True)

    split = str(subset).lower().strip()
    if split == "all":
        out = df_cases.copy()
        out["split"] = "all"
        return out.reset_index(drop=True)

    train_idx, val_idx, test_idx = resolve_split_indices(
        df_cases.reset_index(drop=True),
        split_mode=split_mode,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    if split == "test":
        sel = df_cases.iloc[test_idx].copy()
    elif split == "val":
        sel = df_cases.iloc[val_idx].copy()
    elif split == "train":
        sel = df_cases.iloc[train_idx].copy()
    else:
        raise ValueError("subset must be one of: all, train, val, test")
    sel["split"] = split
    return sel.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    default_workdir = Path(__file__).resolve().parents[1] / "data"
    ap.add_argument("--workdir", type=str, default=str(default_workdir))
    ap.add_argument("--exports", type=str, default="exports")
    ap.add_argument("--cases", type=str, default="cases.csv")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--subset", type=str, default="test", choices=["all", "train", "val", "test"])
    ap.add_argument("--split_mode", type=str, default="auto", choices=["auto", "random", "spatial_checkerboard", "spatial_uniform", "spatial_left_right"])
    ap.add_argument("--split_seed", type=int, default=None)
    ap.add_argument("--val_ratio", type=float, default=None)
    ap.add_argument("--test_ratio", type=float, default=None)
    ap.add_argument("--case_ids_csv", type=str, default=None)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--summary_csv", type=str, default=None)
    ap.add_argument("--experiment_name", type=str, default=None)
    ap.add_argument("--Lx", type=float, default=9.0)
    ap.add_argument("--Ly", type=float, default=3.0)
    ap.add_argument("--nx", type=int, default=128 * 3)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--outlet_width_x", type=float, default=0.05)
    ap.add_argument("--outlet_y0", type=float, default=0.01)
    ap.add_argument("--outlet_y1", type=float, default=0.25)
    ap.add_argument("--bz_x0", type=float, default=0.6)
    ap.add_argument("--bz_x1", type=float, default=8.4)
    ap.add_argument("--bz_y0", type=float, default=0.075)
    ap.add_argument("--bz_y1", type=float, default=1.8)
    ap.add_argument("--src_size", type=float, default=0.20)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--soft_k", type=float, default=None)
    ap.add_argument("--mask_kind", type=str, default="auto", choices=["auto", "rect", "gauss", "soft_rect", "soft-rect"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--inlet_u_tol", type=float, default=1e-8)
    args = ap.parse_args()


    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"]
    meta = ckpt.get("meta", {})
    root, exports_name, cases_name = resolve_dataset_context(
        default_workdir=default_workdir,
        workdir_arg=args.workdir,
        exports_arg=args.exports,
        cases_arg=args.cases,
        meta=meta,
    )
    exports_dir = root / exports_name if not Path(exports_name).is_absolute() else Path(exports_name)
    cases_path = (root / cases_name) if not Path(cases_name).is_absolute() else Path(cases_name)
    case_ids_path = None if args.case_ids_csv is None else Path(args.case_ids_csv)

    df_cases = pd.read_csv(cases_path)
    required = ["case_id", "inlet_u", "src_x", "src_y"]
    missing = [c for c in required if c not in df_cases.columns]
    if missing:
        raise ValueError(f"cases.csv missing columns {missing}. Got: {list(df_cases.columns)}")
    df_cases["case_id"] = df_cases["case_id"].astype(str)

    split_mode = str(meta.get("split_mode", "random")) if args.split_mode == "auto" else str(args.split_mode)
    split_seed = int(meta.get("split_seed", 42)) if args.split_seed is None else int(args.split_seed)
    val_ratio = float(meta.get("split_val_ratio", 0.15)) if args.val_ratio is None else float(args.val_ratio)
    test_ratio = float(meta.get("split_test_ratio", 0.15)) if args.test_ratio is None else float(args.test_ratio)
    rows = pick_rows(
        df_cases,
        subset=args.subset,
        case_ids_path=case_ids_path,
        split_mode=split_mode,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )

    arch = infer_arch_from_state_dict(state)
    in_ch = int(meta.get("in_ch", arch["in_ch"]))
    if in_ch != 3:
        raise ValueError(f"Expected in_ch=3, got {in_ch}")

    width = int(meta.get("width", arch["width"]))
    modes1 = int(meta.get("modes1", arch["modes1"]))
    modes2 = int(meta.get("modes2", arch["modes2"]))
    depth = int(meta.get("depth", arch["depth"]))
    out_ch = int(meta.get("out_ch", arch["out_ch"]))

    ckpt_mask_kind = str(meta.get("mask_kind", "rect")).lower().strip().replace("-", "_")
    if ckpt_mask_kind in ["softrect", "soft_rect(sigmoid)", "softrect(sigmoid)"]:
        ckpt_mask_kind = "soft_rect"
    if ckpt_mask_kind == "gaussian":
        ckpt_mask_kind = "gauss"
    mask_kind = ckpt_mask_kind if args.mask_kind == "auto" else str(args.mask_kind).lower().strip().replace("-", "_")
    if mask_kind in ["softrect", "soft_rect(sigmoid)", "softrect(sigmoid)"]:
        mask_kind = "soft_rect"
    if mask_kind == "gaussian":
        mask_kind = "gauss"

    soft_k_meta = meta.get("soft_k", None)
    soft_k = float(args.soft_k) if args.soft_k is not None else (80.0 if soft_k_meta is None else float(soft_k_meta))
    fixed_u = meta.get("fixed_inlet_u", None)
    fixed_u = None if fixed_u is None else float(fixed_u)
    print(
        f"[INFO] split: mode={split_mode}, seed={split_seed}, "
        f"val_ratio={val_ratio}, test_ratio={test_ratio}, subset={args.subset}"
    )

    model = FNO2D(in_ch=in_ch, out_ch=out_ch, width=width, modes1=modes1, modes2=modes2, depth=depth).to(device)
    model.load_state_dict(state)
    model.eval()

    x_mean = ckpt["x_mean"].to(device)
    x_std = ckpt["x_std"].to(device)
    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)
    xg, yg = make_xy_grids(args.Lx, args.Ly, args.nx, args.ny)
    outlet_mask = rect_mask(xg, yg, args.Lx - args.outlet_width_x, args.Lx, args.outlet_y0, args.outlet_y1)
    bz_mask = rect_mask(xg, yg, args.bz_x0, args.bz_x1, args.bz_y0, args.bz_y1)

    metrics: List[dict] = []
    with torch.no_grad():
        for _, row in rows.iterrows():
            case_id = str(row["case_id"]).strip()
            inlet_u = float(row["inlet_u"])
            sx = float(row["src_x"])
            sy = float(row["src_y"])
            if fixed_u is not None and abs(inlet_u - fixed_u) > float(args.inlet_u_tol):
                continue

            case_file = resolve_case_file(exports_dir, case_id, inlet_u, sx, sy)
            _, _, c_gt = df_to_grid(load_case_csv(case_file), Lx=args.Lx, Ly=args.Ly, nx=args.nx, ny=args.ny)
            x_in = build_input_field(
                sx=sx,
                sy=sy,
                ny=args.ny,
                nx=args.nx,
                Lx=args.Lx,
                Ly=args.Ly,
                mask_kind=mask_kind,
                src_size=float(args.src_size),
                sigma=float(args.sigma),
                soft_k=float(soft_k),
            )
            x_t = torch.from_numpy(x_in).to(device)
            x_t_n = (x_t - x_mean) / (x_std + 1e-12)
            y_pred_n = model(x_t_n)
            y_pred = y_pred_n * (y_std + 1e-12) + y_mean
            c_pr = y_pred.cpu().numpy()[0, 0]

            err = c_pr - c_gt
            mse = float(np.mean(err ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(err)))
            rel_l2 = float(l2_rel(c_gt, c_pr))
            cre_gt, c_out_gt, c_bz_gt = compute_cre(c_gt, outlet_mask, bz_mask)
            cre_pr, c_out_pr, c_bz_pr = compute_cre(c_pr, outlet_mask, bz_mask)
            cre_abs_err = float(abs(cre_pr - cre_gt)) if np.isfinite(cre_gt) and np.isfinite(cre_pr) else float("nan")
            cre_rel_err = (
                float(abs(cre_pr - cre_gt) / (abs(cre_gt) + 1e-12))
                if np.isfinite(cre_gt) and np.isfinite(cre_pr)
                else float("nan")
            )
            c_gt_abs = np.abs(c_gt)
            p90 = float(np.percentile(c_gt_abs, 90.0))
            high_mask = c_gt_abs >= p90
            if np.any(high_mask):
                high_rmse = float(np.sqrt(np.mean((err[high_mask]) ** 2)))
                high_mae = float(np.mean(np.abs(err[high_mask])))
            else:
                high_rmse = float("nan")
                high_mae = float("nan")

            metrics.append(
                {
                    "experiment": args.experiment_name or ckpt_path.stem,
                    "checkpoint": str(ckpt_path),
                    "subset": str(row.get("split", args.subset)),
                    "case_id": case_id,
                    "inlet_u": inlet_u,
                    "src_x": sx,
                    "src_y": sy,
                    "width": width,
                    "modes1": modes1,
                    "modes2": modes2,
                    "depth": depth,
                    "mask_kind": mask_kind,
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
                    "mean_gt_c": float(np.mean(c_gt)),
                    "max_gt_c": float(np.max(c_gt)),
                }
            )

    if not metrics:
        raise RuntimeError("No cases were evaluated. Check subset/case_ids_csv/fixed_inlet_u.")

    df_metrics = pd.DataFrame(metrics).sort_values(["src_x", "src_y", "case_id"]).reset_index(drop=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(out_csv, index=False)

    summary = pd.DataFrame(
        [
            {
                "experiment": args.experiment_name or ckpt_path.stem,
                "checkpoint": str(ckpt_path),
                "subset": str(args.subset),
                "n_cases": int(len(df_metrics)),
                "fixed_inlet_u": fixed_u,
                "width": width,
                "modes1": modes1,
                "modes2": modes2,
                "depth": depth,
                "mask_kind": mask_kind,
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
    summary_csv = Path(args.summary_csv) if args.summary_csv is not None else out_csv.with_name(f"summary_{out_csv.name}")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)

    print(f"[SAVED] case metrics -> {out_csv.resolve()}")
    print(f"[SAVED] summary -> {summary_csv.resolve()}")


if __name__ == "__main__":
    main()
