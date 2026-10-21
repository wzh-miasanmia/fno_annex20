# Unified model comparison

Reproduction runners for FNO, FCN, U-Net, DeepONet and the custom BasicGINO2D
used in the accepted paper. Use the frozen modules in `model/`, the same
679/145/145 split, and seeds 7, 42 and 123. No training data or checkpoint is
downloaded automatically.

## Inputs and commands

Run from the repository root. Replace all capitalized path placeholders with
your own paths; do not point an output at an existing completed run.
The dataset directory contains `exports/`; each export has the original
`case_ID_uU_sxX_syY.csv` name. The checked case manifest and split are in
`splits/strict_datasize/`. The shared CFD cache contains `rows.csv` and
`fields.npy` on the 384x128 grid (see the repository data documentation).
Use an absolute path for `--cases`: the frozen trainers resolve a relative
case-manifest path against the dataset directory, not the repository root.

Example FNO run, seed 7:

```bash
python analysis/unified_model_benchmark/grid.py fno \
  --workdir DATASET_DIR --cases "$PWD/splits/strict_datasize/cases_n679.csv" \
  --split_csv splits/strict_datasize/split_n679.csv --save RUN_DIR/model.pt \
  --fixed_inlet_u 0.2 --no_velocity_input --mask_kind soft_rect \
  --src_size 0.2 --soft_k 30 --target_transform raw --nx 384 --ny 128 \
  --width 48 --depth 4 --modes1 16 --modes2 16 \
  --epochs 300 --patience 30 --batch 4 --workers 0 --cache \
  --seed 7 --split_seed 42 --lr 0.001 --weight_decay 0.000001 --device cuda \
  --test_list_csv RUN_DIR/test_case_ids.csv
```

For the other grid models, change the positional kind and architecture flags:

| Kind | Architecture flags |
| --- | --- |
| fcn | `--width 128 --depth 10` |
| unet | `--width 32 --depth 4` |
| deeponet | `--width 64 --rank 128 --trunk_width 128` |

Remove FNO-only modes and test-list flags for those three models and add
`--out_csv RUN_DIR/common_grid_test.csv`. The wrapper supplies the matching
`--model_kind`; all remaining training options are forwarded unchanged.

Example GINO run:

```bash
python analysis/unified_model_benchmark/gino.py \
  --workdir DATASET_DIR --cases "$PWD/splits/strict_datasize/cases_n679.csv" \
  --split_csv splits/strict_datasize/split_n679.csv --save RUN_DIR/model.pt \
  --common-grid-cache CACHE_DIR --task concentration \
  --mask_kind soft_rect --src_size 0.2 --soft_k 30 --target_transform raw \
  --latent_ny 48 --latent_nx 144 --width 48 --depth 3 \
  --modes1 12 --modes2 12 --kernel_width 64 --batch 1 \
  --epochs 300 --patience 30 --workers 0 --cache --seed 7 \
  --lr 0.001 --weight_decay 0.000001 --device cuda \
  --nonnegative_weight 0 --concentration_weight_alpha 0 \
  --quadrature_mode auto --quadrature_cache_dir RUN_DIR/quadrature
```

`--common-grid-cache` belongs to the wrapper and is removed before invoking
the frozen GINO trainer. GINO uses native room sampling points and the three
source-mask/X/Y features, without teacher velocity.

Place the 15 completed runs at `BENCHMARK_DIR/MODEL/seedSEED/`, then run:

```bash
python analysis/unified_model_benchmark/aggregate.py BENCHMARK_DIR
```

Aggregation requires all 15 completion markers, the identical 145 case IDs,
finite metrics, and one recorded GPU/PyTorch configuration. It writes
`per_seed.csv`, `summary.csv` and `COMPLETE.txt`. All runners provide
`--help` without accessing a GPU. Training/profiling requires CUDA.

## Interpretation and preserved settings

- FNO: width 48, depth 4, modes 16; FCN: width 128, depth 10; U-Net:
  width 32, depth 4; DeepONet: width 64, rank 128, trunk width 128.
- BasicGINO2D: latent grid 48x144, width 48, depth 3, modes 12, kernel 64.
- Grid batch size is 4; GINO batch size is 1. Both use four CPU threads,
  no AMP, workers 0, matmul TF32 off and cuDNN TF32 on.
- The example commands explicitly set the published batch/worker settings.
  Profile metadata records the effective DataLoader values, so forwarded
  overrides (including `--batch=N`/`--workers=N`) and trainer defaults are
  reported accurately rather than relabeled as the published configuration.
- These are configuration-specific comparisons, not equal-batch or
  equal-parameter architecture benchmarks.
- Grid training retains a full-training-set metric pass each epoch;
  GINO does not. Accumulated epoch time includes the recorded training
  and evaluation work, not pure optimizer computation.
- Total wall time includes normalization, training, validation,
  checkpoint I/O and native test, but excludes detailed per-case
  evaluation, queueing and container startup.
- Memory is PyTorch peak allocated/reserved memory, not total process memory.
- GINO predictions are interpolated to the common 384x128 reference grid
  using linear interpolation with nearest-neighbor filling.
- High-concentration errors use points at or above the per-case 90th
  percentile of absolute ground-truth concentration.
- Standard deviations are across three seed-wise test means, not across cases.

## Inference latency

The separate synchronized batch-one timing runner uses the same three FNO
checkpoints and verifies one test prediction against each saved metric file:

```bash
python analysis/inference_timing_20260916/benchmark.py \
  --benchmark-root BENCHMARK_DIR --cache-dir CACHE_DIR \
  --test-ids splits/strict_datasize/test_case_ids.csv --output-dir TIMING_DIR
```

It preserves 50 warm-ups per seed/scope, three passes over 145 cases, and
the manuscript's model-only and source-to-CPU-field timing scopes. Model
loading, disk access and reusable coordinate-grid construction are excluded.
