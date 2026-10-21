# FNO for indoor contaminant transport and source identification

Code accompanying the accepted *Building and Environment* manuscript:

**A Fourier Neural Operator Framework for Indoor Contaminant Transport Prediction,
Source Localization, and Release Strength Estimation**

Zihao Wang, Kevin H. Lausch, Eugen Lichtner, Eshwar Kumar Ramasetti, and Martin Kriegel.

The main setting is the two-dimensional IEA Annex 20 room with prescribed
ventilation and a spatially varying passive-scalar source. A soft rectangular
source mask conditions a full-field FNO surrogate; the surrogate also supports
continuous inverse localization and profiled release-strength estimation.

This release contains paper-related code and small case/split manifests, not
the complete research workspace. Unreported PDEformer, physical-splitting,
staged GINO and other exploratory experiments are intentionally excluded.

## Code map

| Paper component | Entry points |
| --- | --- |
| FNO and source encodings | `model/train_fixed_u_srcpos_to_c.py`, `analysis/run_forward_multiseed.py` |
| FNO / FCN / U-Net / DeepONet / GINO comparison | `analysis/unified_model_benchmark/` |
| Source-space IDW reference | `analysis/run_source_space_interp_baseline.py` |
| Zero-shot resolution transfer | `analysis/zero_shot_multiseed_20260915/` |
| Known-strength inverse comparison | `analysis/benchmark_inverse_three_methods_loss_time.py` |
| Off-grid scaling and joint inversion | `analysis/evaluate_source_strength_scaled_fno.py`, `analysis/joint_source_strength/` |
| Training-set-size ablation | `analysis/prepare_strict_datasize_splits.py`, `splits/strict_datasize/` |
| Sensor count/layout ablation | `analysis/sensor_count_unified_20260917/` |
| Variable-inlet-velocity appendix | `model/train_joint_fno_uvc.py` |
| Interactive application | `visualization/app_fixed_u_srcpos_to_c.py` |
| Inference timing | `analysis/inference_timing_20260916/benchmark.py` |

Publication plotting scripts are in `analysis/plot_*.py`. Geo-FNO and extended
trainer modules retained in `model/` supply shared dataset, quadrature and
architecture dependencies for the reported GINO implementation; their presence
does not indicate additional reported model comparisons.

## Installation

Use a separate Python environment. Install a PyTorch build appropriate for your
hardware, then install the scientific dependencies:

```bash
python -m pip install -r requirements.txt
```

The unified comparison and inference-timing runners require CUDA for their
recorded synchronization/profiling scopes. The core FNO trainer and many
evaluation scripts can use CPU. No system CUDA toolkit is installed or changed
by this repository.

## Data and checkpoints

Raw CFD exports, STAR-CCM+ `.sim` files, weights, caches, generated figures and
historical run outputs are **not bundled**. Checked-in CSVs contain coordinates
and split assignments, not concentration fields. See [the data specification](docs/DATA.md).

The main dataset contains 969 cases with fixed 679/145/145 partitions. The
three-seed benchmark uses **7, 42 and 123**. The independent off-grid campaign
contains 15 positions at three normalized release strengths. The variable-inlet
appendix uses a separate 9,900-case dataset and inlet-group split.

## Quick start: the paper FNO configuration

From the repository root, with CFD exports supplied in `data/dense969/exports/`:

```bash
python model/train_fixed_u_srcpos_to_c_strict_ablation.py \
  --workdir data/dense969 \
  --cases ../../splits/strict_datasize/cases_n679.csv \
  --split_csv splits/strict_datasize/split_n679.csv \
  --fixed_inlet_u 0.2 --no_velocity_input \
  --mask_kind soft_rect --src_size 0.20 --soft_k 30 \
  --target_transform raw --nx 384 --ny 128 \
  --width 48 --depth 4 --modes1 16 --modes2 16 \
  --epochs 300 --patience 30 --batch 4 --workers 0 --cache \
  --lr 0.001 --weight_decay 0.000001 --seed 7 --split_seed 42 \
  --device cuda --save runs/baseline/model.pt
```

This runs the core trainer, not the five-model profiling campaign. The `--cases`
path is relative to `--workdir`; use absolute paths when supplying data elsewhere.
See [reproduction notes](docs/REPRODUCIBILITY.md) and experiment READMEs for exact
protocols, checkpoint provenance and required arguments. Do not overwrite
historical checkpoints with a new campaign.

## Interactive application

```bash
python -m pip install -r requirements-app.txt
streamlit run visualization/app_fixed_u_srcpos_to_c.py
```

Select a trusted, compatible checkpoint and dataset. Checkpoint deserialization
is not a sandbox: do not load untrusted model files.

## Software checks

These checks require the installed scientific dependencies but no CFD fields,
checkpoints or GPU:

```bash
python -m unittest discover -s tests -p "test_*.py"
python -m unittest analysis.sensor_count_unified_20260917.test_output_safety
```

During packaging, CPU synthetic forward/backward checks also covered all five
architectures, and one-epoch synthetic CLI runs checked the four grid trainers
and checkpoint loading. These are software checks, not a rerun of the paper's
scientific experiments or GPU timing measurements.

## Scope and citation

The principal surrogate was evaluated for one two-dimensional room and a fixed
source shape/size. The variable-inlet appendix uses a separately trained model;
it does not establish arbitrary geometry or boundary-condition transfer. No
new scientific results are introduced by this code cleanup.

Citation metadata is in [CITATION.cff](CITATION.cff). A DOI will be added when the
final publication identifier is confirmed. No new code license has been assigned
as part of this cleanup.
