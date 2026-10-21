# Reproduction notes

This release selects accepted-paper code, not every research experiment. It
does not claim all scientific experiments were rerun during packaging. Personal
paths and launch mechanics are ported without changing model mathematics.

## Model comparison

Core dependencies were recovered from the unified benchmark's frozen snapshot,
not the later experimental working tree. The zero-shot campaign used the same
strict trainer (the original files have identical SHA256).

`model/train_fixed_u_srcpos_to_c.py` is a compatibility facade for that plain
FNO trainer. It is not the later FNO-CNN experiment that occupied the same
historical filename. The source-mask runner explicitly disables teacher velocity
inputs, and the standalone grid baselines share the canonical two-metric API.

| Model | Configuration | Batch |
| --- | --- | --- |
| FNO | width 48, four layers, 16 x 16 modes | 4 |
| FCN | width 128, depth 10 | 4 |
| U-Net | width 32, depth 4 | 4 |
| DeepONet | width 64, rank 128, trunk width 128 | 4 |
| GINO | latent 144 x 48, width 48, depth 3, 12 x 12 modes, kernel width 64 | 1 |

All use the same 679/145/145 split and seeds 7/42/123. Maximum epochs 300,
early-stopping patience 30, learning rate 1e-3, weight decay 1e-6. The benchmark
does not use teacher velocity fields. GINO's native-point output is evaluated
on the same 384 x 128 reference grid.

Accumulated epoch time sums recorded epoch durations. Grid and GINO runners
use different batch sizes and evaluation passes; times characterize evaluated
configurations, not a pure equal-batch optimizer comparison. Prediction metrics
are averaged over cases, then summarized across seeds. High-concentration RMSE
uses each case's points at or above the 90th percentile of absolute GT
concentration. Seed standard deviations are not case-level standard deviations.

## Source encoding and spatial holdouts

`analysis/run_forward_multiseed.py` supports the three masks and spatial split
modes. Its generic defaults are not the paper protocol: explicitly set seeds,
epoch budget, mask and split. Source encodings describe the same physical source;
they do not constitute a physical size/shape sensitivity study.

`analysis/run_source_space_interp_baseline.py` interpolates concentration
responses across source coordinates, not CFD mesh coordinates. Its response
library must exclude held-out target fields.

## Zero-shot super-resolution

Train at 48x16, 96x32, 192x64 or 288x96 and evaluate directly on 384x128 without
fine-resolution retraining. The full-resolution row reuses the main benchmark's
three FNO checkpoints. Inputs are resampled existing CFD fields, not separate
coarse-CFD simulations. Use `plot_zero_shot_resolution_case_multiseed.py` for
the updated representative figure. The original ZSR trainer did not explicitly
set TF32 flags; do not assume all campaign backend settings were identical.

## Known-strength inverse comparison

`analysis/benchmark_inverse_three_methods_loss_time.py` compares gradient-only,
fine-grid and coarse-to-gradient localization. The main table uses eight sensors,
145 held-out cases, noise 0/0.02/0.05 with `relative_max`, noise seed 42, restart
seed 42, 150 steps, top-k 4, coarse stride 2 and learning rate 0.05.

That table uses a legacy single checkpoint, not three training seeds. Supply its
checkpoint and corresponding test list. Do not silently replace it with a new
checkpoint while calling the output the original table.

## Off-grid scaling and joint inversion

Decode the physical q=1 prediction before multiplying by q; no strength
retraining is used. Joint inversion profiles nonnegative strength while
optimizing position, with an IDW plus bounded Nelder-Mead reference.

Joint inversion uses the three **strict n679** checkpoints, not the unified
Table 2 campaign. Its 45 cases have only 15 independent spatial positions. A
fixed paired noise realization is shared across strengths/models; this is not
a multi-realization noise study.

## Ablations

Training-size sets contain 50/100/200/300/400/600/679 cases selected through
min-max-normalized deterministic farthest-point ordering, with fixed validation
and test sets. Its n679 error 0.0311 +/- 0.0014 and the unified campaign's
0.0326 +/- 0.0022 are separate runs with different data-loader settings.

The public sensor runner uses only 4/8/16 sensors, four layouts and 145 original-
grid test cases. Earlier 32-sensor and off-grid layout tests are excluded. Its
README explains cache preparation and required checkpoint/reference artifacts.

## Variable-inlet appendix and timing

`model/train_joint_fno_uvc.py --target_mode concentration` predicts C from mask,
X, Y and inlet speed. `--target_mode joint_uvc` is the appendix's exploratory
Ux/Uy/C model. Use the recorded inlet-group split, not a random case split.
Other staged FNO/CNN or GINO workflows are not reported paper results.

Inference timing uses three unified FNO checkpoints, batch one, 50 warmups and
three passes over 145 test cases with GPU synchronization. Model-only and
source-to-CPU-field scopes differ; neither is a complete inverse runtime.
The CFD workflow runtime is not recreated by model inference code.

## Exclusions

No PDEformer, physical splitting, staged/curriculum GINO, GINO super-resolution,
adjoint/transport-response experiments, drafts, reviewer correspondence, logs,
raw CFD fields, weights, personal environments or generated figures. Original
research files remain untouched; only the publication checkout is curated.
