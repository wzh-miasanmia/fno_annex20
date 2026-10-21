# Required input data

Only code and small experiment manifests are bundled. Raw CFD fields and
checkpoints must be supplied separately; no public-data download is implied.

## Export schema

Field CSVs contain `Velocity[i] (m/s)`, `Velocity[j] (m/s)`, `Passive Scalar`,
`X (m)` and `Y (m)`; some also have `Z (m)`. Grid models interpolate CFD fields
to a Cartesian grid. GINO reads native points; reported common-grid metrics
evaluate its interpolated predictions on the same reference grid.

Recommended local layout (ignored by Git):

```text
data/
  dense969/
    cases.csv
    exports/case_<id>_u0.2000_sx<...>_sy<...>.csv
  offgrid_strength/
    cases.csv
    exports/case_<id>_u0.2000_sx<...>_sy<...>_q<...>.csv
  variable_inlet/
    cases.csv
    exports/case_<id>_u<...>_sx<...>_sy<...>.csv
```

Filename floats use four decimal places. Resolve cases by their complete
condition tuple, not ambiguous case IDs shared between datasets.

## Main candidate-source dataset

Manifest columns: `case_id,inlet_u,src_x,src_y`.

- `analysis/data_u0p2_dense_969/cases.csv`: main 969 source cases.
- `splits/strict_datasize/fixed_split_seed42.csv`: full assignment.
- `cases_n679.csv` and `split_n679.csv`: main 679/145/145 comparison inputs.
- `cases_n050.csv` through `cases_n600.csv`: nested training-size campaigns,
  each retaining the same 145 validation and 145 test cases.

The split reader accepts `validation` and `val`. Keep case IDs and row order
when matching historical runs. Do not redraw the test set for each training seed.

## Independent off-grid cases

`data_source_strength_validation/cases.csv` describes 15 positions at q=0.5, 1
and 2 (45 cases). Off-grid means absent from the original source-position grid,
not outside the sampled source region. Generator scripts and STAR-CCM+ Java
macros require the corresponding solver setup; no solver license or `.sim`
model is included.

Joint inversion requires cached physical unit-strength responses, three strict
n679 checkpoints and the corresponding reference metrics. Read its experiment
README for explicit paths and verification gates.

## Variable-inlet appendix

The separate dataset has 25 inlet speeds and 396 positions per speed (9,900
cases). `splits/variable_inlet/` preserves the recorded group assignment:
17 training, four validation and four test inlet speeds. These manifests do not
contain velocity or concentration targets.

## Checkpoints and derived files

PyTorch checkpoints include state dictionaries, normalization and model metadata.
Decode normalization/target transforms before physical error calculation or
release-strength scaling. Retain the original checkpoint mask parameters.

Cache `rows.csv`/`fields.npy`, reference metric CSVs and sensor-response archives
are external run artifacts. Supply them where verification gates require them;
missing references must not be described as a successful verification.
