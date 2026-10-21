#!/usr/bin/env python3
"""Rebuild the representative zero-shot resolution-transfer field figure."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

# Import matplotlib before torch to avoid conflicting OpenMP runtimes in the
# workstation environment used for this reproducible plotting script.
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch


RESOLUTIONS = [(48, 16), (96, 32), (192, 64), (288, 96)]
CASE_ID = 900571
SOURCE_X = 4.625
SOURCE_Y = 1.625
REFERENCE_NX = 384
REFERENCE_NY = 128
ROOM_LX = 9.0
ROOM_LY = 3.0
ZOOM = (3.75, 5.50, 1.35, 1.95)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("zero_shot_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--case-csv",
        type=Path,
        default=Path(
            "data/dense969/exports/case_900571_u0.2000_sx4.6250_sy1.6250.csv"
        ),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=root
        / "analysis/results/zero_shot_multiseed_hpc_1945189/seed42_models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "analysis/figures/resolution_soft_rect",
    )
    return parser.parse_args()


def predict_reference_grid(trainer, checkpoint_path: Path) -> tuple[np.ndarray, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = checkpoint.get("meta", {})
    model = trainer.FNO2D(
        in_ch=int(meta.get("in_ch", 3)),
        out_ch=int(meta.get("out_ch", 1)),
        width=int(meta.get("width", 48)),
        modes1=int(meta.get("modes1", 16)),
        modes2=int(meta.get("modes2", 16)),
        depth=int(meta.get("depth", 4)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    x_phys = np.linspace(0.0, ROOM_LX, REFERENCE_NX, dtype=np.float32)
    y_phys = np.linspace(0.0, ROOM_LY, REFERENCE_NY, dtype=np.float32)
    xg, yg = np.meshgrid(x_phys, y_phys)
    xc, yc = np.meshgrid(
        np.linspace(0.0, 1.0, REFERENCE_NX, dtype=np.float32),
        np.linspace(0.0, 1.0, REFERENCE_NY, dtype=np.float32),
    )
    mask = trainer.build_source_mask_soft_rect_sigmoid(
        xg,
        yg,
        SOURCE_X,
        SOURCE_Y,
        float(meta.get("src_size", 0.20)),
        float(meta.get("soft_k", 30.0)),
    ).astype(np.float32)
    inputs = torch.from_numpy(np.stack([mask, xc, yc], axis=0)[None])
    x_mean = checkpoint["x_mean"].cpu()
    x_std = checkpoint["x_std"].cpu()
    y_mean = checkpoint["y_mean"].cpu()
    y_std = checkpoint["y_std"].cpu()
    with torch.inference_mode():
        prediction = model((inputs - x_mean) / (x_std + 1e-12)) * (y_std + 1e-12) + y_mean
    if str(meta.get("target_transform", "raw")) == "log10":
        eps = float(meta.get("target_transform_eps", 1e-8))
        prediction = torch.clamp(torch.pow(10.0, prediction) - eps, min=0.0)
    return prediction[0, 0].numpy().astype(np.float32), meta


def main() -> None:
    args = parse_args()
    trainer_path = args.project_root / "model/train_fixed_u_srcpos_to_c_strict_ablation.py"
    trainer = load_module(trainer_path)
    raw_case = trainer.load_case_csv(args.case_csv)

    x_ref, y_ref, _, _, gt_reference = trainer.df_to_grid(
        raw_case, Lx=ROOM_LX, Ly=ROOM_LY, nx=REFERENCE_NX, ny=REFERENCE_NY
    )
    native_fields: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    metric_records = []

    for nx, ny in RESOLUTIONS:
        label = f"{nx}x{ny}"
        x_native, y_native, _, _, gt_native = trainer.df_to_grid(
            raw_case, Lx=ROOM_LX, Ly=ROOM_LY, nx=nx, ny=ny
        )
        checkpoint_path = args.model_root / label / "model.pt"
        metrics_path = args.model_root / label / "common_grid_test.csv"
        prediction, meta = predict_reference_grid(trainer, checkpoint_path)
        error = prediction - gt_reference
        rel_l2 = float(np.linalg.norm(error.ravel()) / (np.linalg.norm(gt_reference.ravel()) + 1e-12))
        stored = pd.read_csv(metrics_path)
        stored_case = stored.loc[stored["case_id"].astype(int) == CASE_ID]
        if len(stored_case) != 1:
            raise RuntimeError(f"Expected one metric record for case {CASE_ID} at {label}")
        stored_rel_l2 = float(stored_case.iloc[0]["rel_l2_c"])
        # The downloaded HPC metric uses the cached 384x128 field, whereas this
        # script reconstructs the same field from the raw CFD CSV. Allow only
        # the small interpolation-rounding difference between those paths.
        if not np.isclose(rel_l2, stored_rel_l2, rtol=5e-4, atol=1e-5):
            raise RuntimeError(
                f"Metric mismatch at {label}: recomputed {rel_l2:.8f}, stored {stored_rel_l2:.8f}"
            )
        native_fields[label] = {"x": x_native, "y": y_native, "c": gt_native}
        predictions[label] = prediction
        metric_records.append(
            {
                "case_id": CASE_ID,
                "src_x": SOURCE_X,
                "src_y": SOURCE_Y,
                "seed": int(meta.get("seed", 42)),
                "training_resolution": label,
                "prediction_resolution": "384x128",
                "relative_l2": rel_l2,
                "stored_relative_l2": stored_rel_l2,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_records).to_csv(
        args.output_dir / "resolution_case_grid_to384_center_mixedgt_2x4_metrics.csv", index=False
    )
    source_arrays = {
        "case_id": np.asarray(CASE_ID),
        "src_x": np.asarray(SOURCE_X),
        "src_y": np.asarray(SOURCE_Y),
        "x_reference": x_ref,
        "y_reference": y_ref,
        "ground_truth_reference": gt_reference,
    }
    for nx, ny in RESOLUTIONS:
        label = f"{nx}x{ny}"
        source_arrays[f"x_{label}"] = native_fields[label]["x"]
        source_arrays[f"y_{label}"] = native_fields[label]["y"]
        source_arrays[f"ground_truth_{label}"] = native_fields[label]["c"]
        source_arrays[f"prediction_{label}_to384x128"] = predictions[label]
    np.savez_compressed(
        args.output_dir / "resolution_case_grid_to384_center_mixedgt_2x4_source_data.npz",
        **source_arrays,
    )

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    all_fields = [gt_reference]
    all_fields.extend(item["c"] for item in native_fields.values())
    all_fields.extend(predictions.values())
    vmin = 0.0
    vmax = max(float(np.nanmax(field)) for field in all_fields)
    cmap = "viridis"
    x0, x1, y0, y1 = ZOOM

    fig, axes = plt.subplots(4, 3, figsize=(11.8, 8.0))
    column_titles = ["CFD at training resolution", "Source-region CFD", r"FNO prediction at $384\times128$"]
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, pad=6)

    image = None
    for row, (nx, ny) in enumerate(RESOLUTIONS):
        label = f"{nx}x{ny}"
        native = native_fields[label]
        prediction = predictions[label]
        rel_l2 = metric_records[row]["relative_l2"]

        image = axes[row, 0].imshow(
            native["c"], origin="lower", extent=(0, ROOM_LX, 0, ROOM_LY),
            aspect="equal", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax
        )
        axes[row, 0].add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white",
                      linewidth=1.0, linestyle=(0, (4, 2)))
        )
        axes[row, 1].imshow(
            native["c"], origin="lower", extent=(0, ROOM_LX, 0, ROOM_LY),
            aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax
        )
        axes[row, 1].set_xlim(x0, x1)
        axes[row, 1].set_ylim(y0, y1)
        axes[row, 2].imshow(
            prediction, origin="lower", extent=(0, ROOM_LX, 0, ROOM_LY),
            aspect="equal", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax
        )
        axes[row, 2].text(
            0.98, 0.94, rf"relative $L_2={rel_l2:.3f}$", transform=axes[row, 2].transAxes,
            ha="right", va="top", fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
        axes[row, 0].text(
            -0.18, 0.5, f"({chr(97 + row)})  {nx}$\\times${ny}",
            transform=axes[row, 0].transAxes, rotation=90,
            ha="center", va="center", fontsize=12,
        )

        for column in range(3):
            axes[row, column].set_ylabel(r"$y$ (m)")
            axes[row, column].tick_params(direction="out", length=3, width=0.75)
            if row == len(RESOLUTIONS) - 1:
                axes[row, column].set_xlabel(r"$x$ (m)")
            else:
                axes[row, column].tick_params(labelbottom=False)

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.075, top=0.82, wspace=0.20, hspace=0.24)
    colorbar_axis = fig.add_axes([0.18, 0.925, 0.72, 0.027])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Concentration, $C$", labelpad=3)
    colorbar.ax.xaxis.set_label_position("top")

    stem = args.output_dir / "resolution_case_grid_to384_center_mixedgt_2x4"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    audit = {
        "figure_claim": "Increasing training resolution improves recovery of the source-proximal gradient during zero-shot evaluation on the 384x128 grid.",
        "figure_archetype": "image plate + quantitative annotations",
        "case_selection": "Fixed central held-out case retained from the original figure for direct comparability.",
        "case_id": CASE_ID,
        "source_location_m": [SOURCE_X, SOURCE_Y],
        "training_seed": 42,
        "training_resolutions": [f"{nx}x{ny}" for nx, ny in RESOLUTIONS],
        "prediction_resolution": "384x128",
        "shared_concentration_limits": [vmin, vmax],
        "zoom_extent_m": list(ZOOM),
        "checkpoint_root": str(args.model_root.resolve()),
        "case_csv": str(args.case_csv.resolve()),
    }
    (args.output_dir / "resolution_case_grid_to384_center_mixedgt_2x4_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(metric_records).to_string(index=False))
    print(f"Saved {stem}.svg/.pdf/.png/.tiff")


if __name__ == "__main__":
    main()
