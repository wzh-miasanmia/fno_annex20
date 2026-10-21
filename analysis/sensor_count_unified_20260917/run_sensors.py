import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
if __package__:
    from . import base_comparison as base
    from .sensors import COUNTS, SEEDS, NOISES, layout_nodes, validate
else:
    import base_comparison as base
    from sensors import COUNTS, SEEDS, NOISES, layout_nodes, validate


def evaluate(a):
    tasks_per_layout = len(COUNTS)*len(SEEDS)
    layout = a.task//tasks_per_layout
    count = COUNTS[(a.task%tasks_per_layout)//len(SEEDS)]
    seed = SEEDS[a.task%3]
    base.IX, base.IY = [v[:count] for v in layout_nodes(layout)]
    for noise in NOISES:
        tag = f'layout{layout}_n{count}_seed{seed}_noise{noise:g}'
        # Smoke outputs must never touch production forward/audit/observation
        # files, even after that production condition has completed.
        output_root = Path(a.root)/'smoke'/f'task{a.task}' if a.smoke else Path(a.root)/'results'
        args = SimpleNamespace(root=a.source, checkpoints=a.checkpoints, task=a.task%3,
            split_dir=a.split_dir, output_dir=str(output_root/tag),
            noise=noise, layout=layout, smoke=a.smoke)
        base.evaluate(args)


def aggregate(a):
    rows = []
    for layout in range(4):
        for count in COUNTS:
            for noise in NOISES:
                reference_obs = None
                for seed in SEEDS:
                    d = Path(a.root)/'results'/f'layout{layout}_n{count}_seed{seed}_noise{noise:g}'
                    assert json.loads((d/'COMPLETE.json').read_text())['n_targets'] == 145
                    f = pd.read_csv(d/'inverse.csv')
                    assert len(f) == 145*(3 if seed == 7 else 2)
                    assert set(f.dataset) == {'dense'}
                    for _, group in f.groupby(['dataset', 'method']):
                        assert group.case_id.nunique() == 145
                    obs = pd.read_csv(d/'observations.csv')
                    if reference_obs is None:
                        reference_obs = obs
                    else:
                        pd.testing.assert_frame_equal(obs, reference_obs)
                    rows.append(f.assign(layout=layout, n_sensors=count, noise=noise))
    all_rows = pd.concat(rows, ignore_index=True)
    root = Path(a.root)
    all_rows.to_csv(root/'all_inverse_cases.csv', index=False)
    groups = ['dataset', 'layout', 'n_sensors', 'noise', 'method', 'seed']
    per_seed = all_rows.groupby(groups).agg(mean_error_m=('loc_error_m', 'mean'),
        median_error_m=('loc_error_m', 'median'), mean_runtime_s=('runtime_s', 'mean')).reset_index()
    per_seed.to_csv(root/'per_seed.csv', index=False)
    output = []
    for keys, block in per_seed.groupby(groups[:4]):
        idw = float(block.loc[block.method == 'IDW_NM_shared', 'mean_error_m'].iloc[0])
        for method, fno in block[block.method != 'IDW_NM_shared'].groupby('method'):
            assert len(fno) == 3
            mean = float(fno.mean_error_m.mean())
            output.append(dict(zip(groups[:4], keys), method=method, idw_mean_error_m=idw,
                fno_mean_error_m=mean, fno_seed_std=float(fno.mean_error_m.std()),
                fno_lower_mean=mean < idw, fno_all_seeds_lower=bool((fno.mean_error_m < idw).all()),
                reduction_pct=100*(idw-mean)/max(idw,1e-12)))
    pd.DataFrame(output).to_csv(root/'conditions.csv', index=False)
    (root/'COMPLETE.txt').write_text('All 36 tasks / 108 seed-wise conditions validated; 145 held-out cases only. Layouts reported separately.\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare','evaluate','aggregate'])
    p.add_argument('--root', required=True)
    p.add_argument('--source', help='Directory containing cache/rows.csv and cache/fields.npy, or destination for prepare.')
    p.add_argument('--dataset', help='Original 969-case CFD directory (cases.csv and exports/); prepare only.')
    p.add_argument('--checkpoints', help='Unified Table 2 FNO root with seed7/, seed42/, seed123/.')
    p.add_argument('--split-dir', default=str(base.REPO_ROOT/'splits/strict_datasize'))
    p.add_argument('--reference-sensors', help='Optional historical sensor-response NPZ for the cache reproduction gate.')
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.mode == 'prepare' and (not a.source or not a.dataset):
        p.error('prepare requires --source and --dataset')
    if a.mode == 'evaluate' and (not a.source or not a.checkpoints):
        p.error('evaluate requires --source and --checkpoints')
    if not 0 <= a.task < 4*len(COUNTS)*len(SEEDS):
        p.error('--task must be in 0..35')
    validate()
    if a.mode == 'prepare':
        base.prepare(a)
    elif a.mode == 'evaluate':
        evaluate(a)
    else:
        aggregate(a)
