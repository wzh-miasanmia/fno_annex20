#!/usr/bin/env python3
"""Evaluate a unit-strength FNO on off-grid, variable-strength CFD cases.

The checkpoint predicts the physical, de-normalized field at q=1. For each
validation strength, the prediction is scaled as C_pred(q) = q C_FNO(1).
No per-field normalization or clipping is applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from evaluate_fixed_u_srcpos_to_c_strict_ablation import (
    FNO2D,
    build_input_field,
    df_to_grid,
    infer_arch_from_state_dict,
    l2_rel,
    load_case_csv,
)


def case_file(exports: Path, case_id: int) -> Path:
    hits = sorted(exports.glob(f"case_{case_id}_*.csv"))
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one export for case {case_id}, found {len(hits)}")
    return hits[0]


def metrics(c_gt: np.ndarray, c_pred: np.ndarray) -> dict[str, float]:
    err = c_pred - c_gt
    denom = max(float(np.linalg.norm(c_gt)), 1e-30)
    return {
        "rel_l2_c": float(np.linalg.norm(err) / denom),
        "rmse_c": float(np.sqrt(np.mean(err**2))),
        "mae_c": float(np.mean(np.abs(err))),
        "max_abs_error_c": float(np.max(np.abs(err))),
        "mean_cfd_c": float(np.mean(c_gt)),
        "max_cfd_c": float(np.max(c_gt)),
        "mean_pred_c": float(np.mean(c_pred)),
        "max_pred_c": float(np.max(c_pred)),
        "norm_ratio_pred_to_cfd": float(np.linalg.norm(c_pred) / denom),
        "negative_pred_fraction": float(np.mean(c_pred < 0.0)),
    }


def save_error_plot(case_df: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> None:
    strengths = sorted(case_df["source_strength"].unique())
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for i, q in enumerate(strengths):
        values = case_df.loc[case_df.source_strength == q, "rel_l2_c"].to_numpy()
        jitter = rng.uniform(-0.055, 0.055, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, 100.0 * values, s=22, alpha=0.65)
        row = summary.loc[summary.source_strength == q].iloc[0]
        ax.errorbar(
            i,
            100.0 * row.rel_l2_c_mean,
            yerr=100.0 * row.rel_l2_c_std,
            fmt="o",
            color="black",
            markersize=5,
            capsize=4,
            linewidth=1.4,
            zorder=5,
        )
    ax.set_xticks(range(len(strengths)), [f"{q:g}" for q in strengths])
    ax.set_xlabel("Source strength, $q$")
    ax.set_ylabel("Relative $L_2$ error (%)")
    ax.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"source_strength_fno_error.{suffix}", dpi=300)
    plt.close(fig)


def save_field_plot(
    fields: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]],
    case_df: pd.DataFrame,
    outdir: Path,
) -> dict[str, float]:
    location_scores = case_df.groupby(["src_x", "src_y"])["rel_l2_c"].mean().sort_values()
    sx, sy = location_scores.index[len(location_scores) // 2]
    strengths = sorted(case_df.source_strength.unique())
    fig, axes = plt.subplots(len(strengths), 3, figsize=(10.2, 7.5), constrained_layout=True)
    extent = (0.0, 9.0, 0.0, 3.0)
    for row_i, q in enumerate(strengths):
        c_gt, c_pred = fields[(float(sx), float(sy), float(q))]
        vmax = float(max(c_gt.max(), c_pred.max()))
        image0 = axes[row_i, 0].imshow(c_gt, origin="lower", extent=extent, vmin=0.0, vmax=vmax, cmap="viridis", aspect="auto")
        axes[row_i, 1].imshow(c_pred, origin="lower", extent=extent, vmin=0.0, vmax=vmax, cmap="viridis", aspect="auto")
        image2 = axes[row_i, 2].imshow(np.abs(c_pred - c_gt), origin="lower", extent=extent, cmap="magma", aspect="auto")
        axes[row_i, 0].set_ylabel(f"$q={q:g}$\n$y$ (m)")
        fig.colorbar(image0, ax=axes[row_i, :2], fraction=0.025, pad=0.01)
        fig.colorbar(image2, ax=axes[row_i, 2], fraction=0.05, pad=0.02)
        for ax in axes[row_i]:
            ax.plot(sx, sy, marker="x", color="white", markersize=6, markeredgewidth=1.2)
            ax.set_xlabel("$x$ (m)")
    axes[0, 0].set_title("CFD")
    axes[0, 1].set_title("Scaled FNO")
    axes[0, 2].set_title("Absolute error")
    fig.suptitle(f"Representative off-grid source at ({sx:.4f}, {sy:.4f}) m", fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"source_strength_representative_fields.{suffix}", dpi=300)
    plt.close(fig)
    return {"representative_src_x": float(sx), "representative_src_y": float(sy)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cases = pd.read_csv(args.validation_root / "cases.csv")
    required = {"case_id", "inlet_u", "src_x", "src_y", "source_strength"}
    if required.difference(cases.columns):
        raise ValueError(f"Missing columns: {sorted(required.difference(cases.columns))}")
    if len(cases) != 45 or cases.groupby(["src_x", "src_y"]).ngroups != 15:
        raise ValueError("Expected 45 cases at 15 source locations")
    exports = args.validation_root / "exports"

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint["model_state"]
    meta = checkpoint.get("meta", {})
    arch = infer_arch_from_state_dict(state)
    in_ch = int(meta.get("in_ch", arch["in_ch"]))
    if in_ch != 3:
        raise ValueError(f"This validation expects the paper baseline with in_ch=3, got {in_ch}")
    model = FNO2D(
        in_ch=in_ch,
        out_ch=int(meta.get("out_ch", arch["out_ch"])),
        width=int(meta.get("width", arch["width"])),
        modes1=int(meta.get("modes1", arch["modes1"])),
        modes2=int(meta.get("modes2", arch["modes2"])),
        depth=int(meta.get("depth", arch["depth"])),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    x_mean, x_std = checkpoint["x_mean"].to(device), checkpoint["x_std"].to(device)
    y_mean, y_std = checkpoint["y_mean"].to(device), checkpoint["y_std"].to(device)

    Lx, Ly = float(meta.get("Lx", 9.0)), float(meta.get("Ly", 3.0))
    nx, ny = int(meta.get("nx", 384)), int(meta.get("ny", 128))
    mask_kind = str(meta.get("mask_kind", "soft_rect"))
    src_size, sigma = float(meta.get("src_size", 0.2)), float(meta.get("sigma", 0.12))
    soft_k = float(meta.get("soft_k", 30.0))
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    fields: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    with torch.no_grad():
        for (sx, sy), group in cases.groupby(["src_x", "src_y"], sort=True):
            by_strength = {float(row.source_strength): row for _, row in group.iterrows()}
            if set(by_strength) != {0.5, 1.0, 2.0}:
                raise ValueError(f"Unexpected strengths at ({sx}, {sy}): {sorted(by_strength)}")

            input_field = build_input_field(
                sx=float(sx), sy=float(sy), ny=ny, nx=nx, Lx=Lx, Ly=Ly,
                mask_kind=mask_kind, src_size=src_size, sigma=sigma, soft_k=soft_k,
            )
            x_tensor = torch.from_numpy(input_field).to(device)
            pred_n = model((x_tensor - x_mean) / (x_std + 1e-12))
            c_unit = (pred_n * (y_std + 1e-12) + y_mean).cpu().numpy()[0, 0]

            cfd_fields = {}
            for q, row in sorted(by_strength.items()):
                export = case_file(exports, int(row.case_id))
                _, _, _, _, c_gt = df_to_grid(load_case_csv(export), Lx=Lx, Ly=Ly, nx=nx, ny=ny)
                cfd_fields[q] = c_gt

            for q, row in sorted(by_strength.items()):
                c_gt = cfd_fields[q]
                c_pred = np.float32(q) * c_unit
                result = {
                    "case_id": int(row.case_id),
                    "inlet_u": float(row.inlet_u),
                    "src_x": float(sx),
                    "src_y": float(sy),
                    "source_strength": float(q),
                    "prediction_rule": "C_pred(q) = q * C_FNO(1)",
                    **metrics(c_gt, c_pred),
                    "cfd_linearity_residual": float(l2_rel(c_gt, np.float32(q) * cfd_fields[1.0])),
                }
                rows.append(result)
                fields[(float(sx), float(sy), float(q))] = (c_gt, c_pred)
            print(f"[OK] source=({sx:.4f}, {sy:.4f})")

    case_metrics = pd.DataFrame(rows).sort_values(["src_x", "src_y", "source_strength"])
    case_metrics.to_csv(args.outdir / "source_strength_fno_case_metrics.csv", index=False)
    summary = (
        case_metrics.groupby("source_strength", as_index=False)
        .agg(
            n_cases=("case_id", "size"),
            rel_l2_c_mean=("rel_l2_c", "mean"),
            rel_l2_c_std=("rel_l2_c", lambda x: x.std(ddof=1)),
            rel_l2_c_max=("rel_l2_c", "max"),
            rmse_c_mean=("rmse_c", "mean"),
            mae_c_mean=("mae_c", "mean"),
            norm_ratio_mean=("norm_ratio_pred_to_cfd", "mean"),
            cfd_linearity_residual_mean=("cfd_linearity_residual", "mean"),
            cfd_linearity_residual_max=("cfd_linearity_residual", "max"),
            negative_pred_fraction_mean=("negative_pred_fraction", "mean"),
        )
    )
    summary.to_csv(args.outdir / "source_strength_fno_summary.csv", index=False)
    save_error_plot(case_metrics, summary, args.outdir)
    representative = save_field_plot(fields, case_metrics, args.outdir)
    audit = {
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_root": str(args.validation_root.resolve()),
        "device": str(device),
        "n_locations": 15,
        "n_cases": 45,
        "strengths": [0.5, 1.0, 2.0],
        "scaling_rule": "C_pred(q) = q * C_FNO(1)",
        "per_field_normalization": False,
        "prediction_clipping": False,
        "model_meta": meta,
        **representative,
    }
    (args.outdir / "source_strength_fno_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("\n", summary.to_string(index=False))
    print(f"[SAVED] {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
