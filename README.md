# FNO Annex20

Code and supporting analysis for:

**A Fourier Neural Operator Framework for Indoor Contaminant Transport Prediction and Source Localization**

This repository contains the paper-oriented subset extracted from the local `FNO_final` project. It is intended as the public code location for reproducing the main Fourier Neural Operator (FNO) experiments, figures, and inverse-source workflows used in the manuscript.

## Scope

The repository focuses on a fixed-inlet indoor contaminant transport setting (`u = 0.2 m/s`) with dense source-location sampling. The FNO surrogate learns the mapping from source representation to passive-scalar concentration field and is then used for:

- forward concentration-field prediction
- source-mask ablation
- random and spatial holdout evaluation
- resolution and zero-shot super-resolution analysis
- sparse-sensor inverse source localization

Large CFD exports, STAR-CCM simulation files, and trained PyTorch checkpoints are intentionally excluded from version control. Lightweight CSV summaries, training histories, plotting scripts, and final figures are kept.

## Layout

- `model/`: FNO training entry point and model definition.
- `analysis/`: evaluation, aggregation, and figure-generation scripts.
- `visualization/`: single-case inference and inverse source localization scripts.
- `analysis/results/`: lightweight summary and case-level CSV outputs used by figures.
- `analysis/figures/`: manuscript-oriented figures.
- `checkpoints/`: tracked training histories and curves; large `.pt` checkpoint files are ignored.
- `analysis/data_u0p2_dense_969/`: case descriptors and data-generation scripts; raw CFD exports and `.sim` files are ignored.

## Environment

Python dependencies are listed in `requirements.txt`. A typical setup is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching your CUDA runtime from the official PyTorch instructions.

## Main Commands

Train the fixed-inlet FNO model:

```powershell
python model/train_fixed_u_srcpos_to_c.py `
  --workdir analysis/data_u0p2_dense_969 `
  --exports exports `
  --cases cases.csv `
  --fixed_inlet_u 0.2 `
  --mask_kind soft_rect `
  --soft_k 30 `
  --width 48 `
  --modes1 16 `
  --modes2 16 `
  --depth 4 `
  --nx 384 `
  --ny 128 `
  --save checkpoints/baseline_soft_rect/dense969_u0p2_w48_m16_d4_soft_rect_sk30_rawtarget.pt
```

Evaluate a trained checkpoint:

```powershell
python analysis/evaluate_fixed_u_srcpos_to_c.py `
  --ckpt checkpoints/baseline_soft_rect/dense969_u0p2_w48_m16_d4_soft_rect_sk30_rawtarget.pt `
  --subset test `
  --out_dir analysis/results/baseline_soft_rect
```

Create example paper figures:

```powershell
python analysis/plot_baseline_training_validation_loss.py
python analysis/plot_baseline_case_grid.py
python analysis/plot_forward_multiseed_boxplot.py
python analysis/plot_spatial_test_heatmap_dense.py
python analysis/plot_resolution_case_grid_to384_mixedgt_2x4.py
python analysis/plot_inverse_three_cases_uniform_n16.py
```

## Data and Checkpoints

The raw CFD export directory is not committed because it is large. To reproduce training exactly, place the exported case CSVs under:

```text
analysis/data_u0p2_dense_969/exports/
```

Trained `.pt` checkpoints are also excluded. The repository keeps the associated history CSV files and figure outputs so the paper plots remain traceable without storing large binary weights.
