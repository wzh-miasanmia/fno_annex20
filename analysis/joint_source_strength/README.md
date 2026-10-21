# Joint single-source location and release-strength validation

No retraining. Existing strict n679 FNO checkpoints, seeds7/42/123. IDW uses exactly
the same 679 training CFD responses. Independent 45 CFD cases: 15 off-grid source
positions × q=0.5/1/2. Only eight observed concentrations are supplied to joint
inference. Bounds x=[0.5,7.5], y=[0.5,2.5]; q>=0, no upper bound or candidate q list.
Assumes fixed airflow, passive scalar, and the previously validated scaling regime.
q is normalized release strength; this is NOT multiple-source inference.

## Frozen methods

- IDW k4,p2 and FNO use identical four initial coordinates selected by minimizing
  the amplitude-profiled observation loss over the shared 679-case training library.
- At each trial location, q=max(0, f dot observations / f dot f), after decoding
  the unit-strength concentration prediction. Response norm squared <=1e-24 maps
  to q=0 (no identifiable amplitude). No arbitrary upper strength prior is imposed.
- Bounded Nelder-Mead: same simplex and tolerances as prior comparisons, 150
  iterations per start. Joint location-strength is the primary matched comparison.
- FNO coarse-to-gradient additionally uses its own 0.25-m coarse sensor library,
  profiled loss for selection, four starts, 150 Adam steps, lr0.05. Different workflow.
- Known-q NM controls receive true strength explicitly and are labelled separately;
  they are never used to initialize or select joint estimates.

## Observations and reporting

Original eight horizontal-line sensor grid nodes. Noise 0/2%/5% of maximum absolute
clean sensor concentration per case. One fixed Gaussian draw per source position,
shared across strengths and models, rescaled by case amplitude. No clipping of
negative noisy observations. This is not a multi-realization noise study; the 45
cases cover only 15 independent spatial positions and must not be called 45
independent positions. All strengths, seeds, cases and failures are retained.

Report location error (m) and relative strength error, mean and median, separately
by true strength and noise. Seed SD describes FNO training-seed variability; IDW
has one deterministic result per observation, not artificial three-seed variance.
Known-q strength error is zero by construction, not evidence of strength recovery.
No superiority or identifiability claim before results; low sensor residual does
not guarantee correct source parameters. No significance tests are predeclared.

## Verification and execution

Input export SHA256 recorded; exact case/u/x/y/q filenames; source positions checked
absent from original grid; q=1 sensor responses must reproduce prior cache exactly.
Each checkpoint reproduces all 145 archived forward errors before inference.
Analytic amplitude tests cover scaling, nonnegative constraint, zero response, and
batch behavior. Three-case/all-noise smoke test checks gradient finiteness before
the full array. Aggregation checks 45 unique cases/method/noise and paired observations.
Old experiments untouched. Only one GPU used concurrently; no monitoring installed.

Outputs: seed*/cases_results.csv, observations.csv, COMPLETE.json; all_cases.csv,
per_seed.csv, summary.csv and COMPLETE.txt after automatic aggregation.

## Portable public-code entry points

Run from the repository root. The `--root` argument is an output directory;
`--inputs` is the directory containing the 45 CFD `cases.csv` and `exports/`.
The `--source` directory must contain `cache/rows.csv` and `cache/fields.npy`
for the 969 original cases followed by the 15 unit-strength off-grid cases.
Build it explicitly with the optional off-grid cache input:

```bash
python analysis/datasize_idw_comparison/run_comparison.py prepare \
  --root runs/joint_source --dataset /path/to/dense969 \
  --offgrid /path/to/offgrid45

python analysis/joint_source_strength/joint.py test
python analysis/joint_source_strength/joint.py prepare \
  --root runs/joint --source runs/joint_source --inputs /path/to/offgrid45

python analysis/joint_source_strength/joint.py evaluate \
  --root runs/joint --source runs/joint_source \
  --checkpoints /path/to/strict_datasize/checkpoints \
  --reference-metrics-dir /path/to/strict_datasize/reference --task 0

# Repeat evaluate with --task 1 and --task 2, then:
python analysis/joint_source_strength/joint.py aggregate --root runs/joint
```

`--task 0/1/2` selects seed 7/42/123. The checkpoint files must be
`n679_seed7.pt`, `n679_seed42.pt`, and `n679_seed123.pt`, from the separate
strict data-size campaign. The reference directory must contain
`cases_n679_seed7.csv`, `cases_n679_seed42.csv`, and `cases_n679_seed123.csv`,
with `case_id` and `rel_l2_c` columns for all 145 held-out cases. Those files
are required: missing references fail with instructions rather than silently
disabling checkpoint reproduction. They are not used as inverse observations.
Do not replace these with the unified Table 2 campaign: its training trajectories
and errors differ, even though the split and seeds are the same.

Optional `--smoke` evaluates the first three off-grid cases after the full
145-case forward-reproduction gate. Production aggregation rejects smoke outputs.
The known-q Nelder--Mead controls remain explicitly labeled `_known_q`; the
reported unknown-strength workflow is `FNO_coarse_gradient_joint`, compared with
`IDW_NM_joint`. Source strengths are never supplied to those joint optimizers.

CFD fields and pretrained checkpoints are external inputs, not included in this
code repository. The cache-builder's optional historical sensor-reference check
is reported separately in its audit. The joint prepare step still requires exact
q=1 agreement with the independent off-grid cache, and evaluation always retains
the archived strict-checkpoint forward-error gate.
