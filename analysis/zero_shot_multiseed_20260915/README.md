# Multi-seed zero-shot super-resolution

Train the concentration FNO at 48x16, 96x32, 192x64 and 288x96 with seeds
7, 42 and 123; evaluate every checkpoint directly at 384x128. The 384x128
row reuses the three unified FNO comparison runs rather than retraining a
different full-resolution reference. All evaluations use the exact same
145 strict-test cases.

The reported metrics are relative L2, MAE and high-concentration RMSE.
High-concentration RMSE is computed per case over points at or above the
90th percentile of absolute ground-truth concentration. Standard deviations
are sample standard deviations across the three seed-wise test means.

## Paths and validation gate

Run commands from the repository root. Replace uppercase placeholders with
your own directories. `CACHE_DIR` contains the shared `rows.csv` and
384x128 `fields.npy`; `BENCHMARK_DIR/fno/seed7/` contains a completed
unified FNO checkpoint and its `common_grid_test.csv`.
The commands use an absolute `--cases` path because the frozen trainer
otherwise resolves it against the CFD dataset directory.

First evaluate the existing full-resolution seed-7 checkpoint with the
common-grid evaluator, then check that it agrees with the original
per-case metrics (maximum absolute difference 2e-5):

The validation gate requires 145 unique, non-null case IDs in each input,
identical ID sets, and finite values for every required metric in both files.

```bash
python analysis/zero_shot_multiseed_20260915/evaluate_common_grid.py \
  --checkpoint BENCHMARK_DIR/fno/seed7/model.pt \
  --rows CACHE_DIR/rows.csv --fields CACHE_DIR/fields.npy \
  --test-ids splits/strict_datasize/test_case_ids.csv \
  --output ZSR_DIR/validation/common_grid_test.csv \
  --train-nx 384 --train-ny 128 --seed 7 --batch 1 --device cuda

python analysis/zero_shot_multiseed_20260915/validate_reuse.py \
  --new ZSR_DIR/validation/common_grid_test.csv \
  --reference BENCHMARK_DIR/fno/seed7/common_grid_test.csv \
  --output ZSR_DIR/validation/COMPLETE.json
```

Both the evaluator and the latency runner default to the frozen trainer in
`model/train_fixed_u_srcpos_to_c_strict_ablation.py`; an explicit
`--trainer PATH` override is available. Do not substitute a different
architecture when evaluating a manuscript checkpoint.

## Training and fine-grid evaluation

Example 192x64, seed 7; repeat for the four resolutions and three seeds:

```bash
python model/train_fixed_u_srcpos_to_c_strict_ablation.py \
  --workdir DATASET_DIR --exports exports \
  --cases "$PWD/splits/strict_datasize/cases_n679.csv" \
  --split_csv splits/strict_datasize/split_n679.csv \
  --test_list_csv ZSR_DIR/192x64/seed7/test_ids.csv \
  --fixed_inlet_u 0.2 --no_velocity_input --target_transform raw \
  --mask_kind soft_rect --src_size 0.2 --soft_k 30 \
  --width 48 --depth 4 --modes1 16 --modes2 16 --nx 192 --ny 64 \
  --epochs 300 --patience 30 --batch 4 --workers 0 --cache \
  --seed 7 --split_seed 42 --lr 0.001 --weight_decay 0.000001 --device cuda \
  --save ZSR_DIR/192x64/seed7/model.pt

python analysis/zero_shot_multiseed_20260915/evaluate_common_grid.py \
  --checkpoint ZSR_DIR/192x64/seed7/model.pt \
  --rows CACHE_DIR/rows.csv --fields CACHE_DIR/fields.npy \
  --test-ids splits/strict_datasize/test_case_ids.csv \
  --output ZSR_DIR/192x64/seed7/common_grid_test.csv \
  --train-nx 192 --train-ny 64 --seed 7 --batch 1 --device cuda
```

The coarse training fields are projections of existing CFD outputs, not CFD
solutions generated on coarser simulation meshes. This experiment therefore
does not establish a reduction in CFD data-generation cost.

## Aggregate the table

After all twelve coarse-resolution evaluations and all three unified
full-resolution FNO runs have completed:

```bash
python analysis/zero_shot_multiseed_20260915/aggregate.py \
  --root ZSR_DIR --highres-root BENCHMARK_DIR/fno \
  --test-ids splits/strict_datasize/test_case_ids.csv
```

Expected output layout is `ZSR_DIR/NXxNY/seedSEED/common_grid_test.csv`.
Aggregation checks exact case IDs, finite metrics and three seeds per
resolution before writing `per_seed.csv`, `summary.csv` and `COMPLETE.txt`.
The validation, evaluation and aggregation scripts have no training side
effects on import; `--help` does not access a GPU.
