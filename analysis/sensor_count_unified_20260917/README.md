# Sensor-number and sensor-layout appendix

This public runner contains only the published 4-, 8-, and 16-sensor analysis on
the original 145 held-out source cases. Four predeclared layouts, three training
seeds (7, 42, 123), and noise levels 0, 0.02, and 0.05 are retained. There are
36 tasks, each evaluating three noise levels: 108 seed-wise conditions. Neither
the discarded 32-sensor study nor the separate 15 off-grid cases is evaluated.

The source-coordinate optimization, IDW weights, noise seeds, first 16 sensor
coordinates, four starting points, 150 optimization steps, learning rate 0.05,
and unit-strength assumption are unchanged from the published campaign.
Standard-normal noise still uses the first N entries of the historical 32-entry
draw to reproduce the published prefix-stable noise; this does not use 32 sensors.

## External inputs

- The original 969 CFD cases: `DATA/cases.csv` and `DATA/exports/*.csv`.
- Repository split manifests in `splits/strict_datasize/` (override with
  `--split-dir`). No off-grid CFD fields are needed.
- The **unified Table 2** FNO campaign at `CHECKPOINTS/seed7/`, `seed42/`,
  `seed123/`, each containing `model.pt`, `split_used.csv`, and
  `common_grid_test.csv`. Those per-case CSVs are required reproduction gates,
  not inputs to the optimizer. Do not substitute the separate data-size campaign.
- CUDA-capable PyTorch runtime for evaluation. Cache preparation is CPU-only.

## Commands (from the repository root)

Use your own data, cache, checkpoint, and output paths; no site-specific scheduler
or home-directory setup is required.

```bash
python analysis/sensor_count_unified_20260917/run_sensors.py prepare \
  --root runs/sensors --source runs/shared_cache --dataset /path/to/dense969

# Task 0 = layout 0, four sensors, seed 7. First use --smoke if desired.
python analysis/sensor_count_unified_20260917/run_sensors.py evaluate \
  --root runs/sensors --source runs/shared_cache \
  --checkpoints /path/to/unified/fno --task 0

# Run tasks 0 through 35, then aggregate and plot.
python analysis/sensor_count_unified_20260917/run_sensors.py aggregate --root runs/sensors
python analysis/sensor_count_unified_20260917/plot_sensor_ablation.py \
  --input runs/sensors/conditions.csv --output-dir runs/sensors/figures
```

Task index is `layout * 9 + count_index * 3 + seed_index`, where counts are
`[4, 8, 16]` and seeds `[7, 42, 123]`. The smoke test processes one inverse
target, but checks all 145 forward errors against the supplied historical CSV
first. Its outputs are not accepted by the production aggregation.

Smoke files are isolated under `ROOT/smoke/taskN/<condition>/`; production
conditions are under `ROOT/results/<condition>/`. Each condition directory is
reserved with an exclusive atomic directory creation before any result files
are written. Existing directories, including completed or partially failed runs,
are never overwritten. Use a fresh `--root` to rerun a condition; there is no
automatic destructive cleanup or implicit resumption.

An optional `--reference-sensors /path/to/sensor_responses.npz` in `prepare`
also verifies the historical eight-sensor cache (NPZ keys `ids` and `responses`).
When it is omitted, an explicit warning and `cache/audit.json` state that this
historical cache check was not requested; the supplied-checkpoint forward-error
gate during evaluation is still mandatory.

Only `FNO_coarse_gradient` is plotted in the paper appendix. FNO and IDW
Nelder--Mead control rows remain labeled in the output and are not pooled with
the plotted method. Error bars summarize training-seed variation, not 145-case
uncertainty or repeated noise realizations.
