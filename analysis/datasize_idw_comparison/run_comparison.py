"""Frozen protocol: existing checkpoints, identical training libraries, no tuning."""
import argparse
import hashlib
import json
import time
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SIZES = [50, 100, 200, 300, 400, 600, 679]
SEEDS = [7, 42, 123]
IX = np.array([8, 60, 112, 165, 217, 270, 322, 375])
IY = np.full(8, 63)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def idw(q, xy, response):
    q = np.asarray(q, dtype=np.float32)
    d2 = np.sum((q[..., None, :] - xy)**2, axis=-1)
    idx = np.argsort(d2, axis=-1)[..., :4]
    dist = np.sqrt(np.take_along_axis(d2, idx, axis=-1))
    w = (dist + 1e-12)**-2
    w /= w.sum(axis=-1, keepdims=True)
    pred = (w[..., None]*response[idx]).sum(axis=-2)
    return np.where((dist[..., 0] <= 1e-12)[..., None], response[idx[..., 0]], pred)


def resolve_export(directory, row):
    name = (f'case_{int(row.case_id)}_u{float(row.inlet_u):.4f}'
            f'_sx{float(row.src_x):.4f}_sy{float(row.src_y):.4f}')
    if row.dataset == 'offgrid':
        name += f'_q{float(row.source_strength):.4f}'
    path = Path(directory) / (name + '.csv')
    if not path.is_file():
        raise FileNotFoundError(f'Exact CFD export required (no case-ID-only fallback): {path}')
    return path


def prepare(a):
    root = Path(a.root)
    cache = root / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    dense = pd.read_csv(Path(a.dataset)/'cases.csv').sort_values('case_id')
    assert len(dense) == 969 and dense.case_id.is_unique
    split_dir = Path(a.split_dir)
    split = pd.read_csv(split_dir/'fixed_split_seed42.csv')
    dense = dense.merge(split[['case_id', 'split']], on='case_id', validate='one_to_one')
    assert dense.split.value_counts().to_dict() == {'train': 679, 'validation': 145, 'test': 145}
    rows = dense.assign(dataset='dense').reset_index(drop=True)
    if a.offgrid:
        off = pd.read_csv(Path(a.offgrid)/'cases.csv')
        off = off[np.isclose(off.source_strength, 1) & np.isclose(off.inlet_u, .2)].sort_values('case_id').copy()
        assert len(off) == 15
        off['split'] = 'offgrid'
        assert not any(np.any(np.all(np.isclose(dense[['src_x', 'src_y']].values, q), axis=1)) for q in off[['src_x', 'src_y']].values)
        rows = pd.concat([rows, off.assign(dataset='offgrid')], ignore_index=True)
    # Resolve every input before allocating or interpolating the full cache.
    paths = [resolve_export(Path(a.dataset)/'exports' if r.dataset == 'dense'
                            else Path(a.offgrid)/'exports', r) for _, r in rows.iterrows()]
    rows.to_csv(cache/'rows.csv', index=False)
    fields = np.lib.format.open_memmap(cache/'fields.npy', mode='w+', dtype=np.float32, shape=(len(rows), 128, 384))
    x, y = np.meshgrid(np.linspace(0, 9, 384, dtype=np.float32), np.linspace(0, 3, 128, dtype=np.float32))
    provenance = []
    for i, r in rows.iterrows():
        path = paths[i]
        df = pd.read_csv(path)
        pts = df[['X (m)', 'Y (m)']].to_numpy(dtype=np.float64)
        c = df['Passive Scalar'].to_numpy(dtype=np.float64)
        lin = griddata(pts, c, (x, y), method='linear')
        near = griddata(pts, c, (x, y), method='nearest')
        fields[i] = np.where(np.isnan(lin), near, lin).astype(np.float32)
        assert np.isfinite(fields[i]).all()
        provenance.append({'dataset': r.dataset, 'case_id': int(r.case_id), 'file': str(path), 'sha256': digest(path)})
        if i % 50 == 0:
            fields.flush()
            print(f'PREP {i}/{len(rows)}', flush=True)
    fields.flush()
    discrepancy = None
    if a.reference_sensors:
        prior = np.load(a.reference_sensors)
        lookup = dict(zip(dense.case_id.astype(int), range(len(dense))))
        errors = [np.max(np.abs(fields[lookup[int(cid)], IY, IX]-response)) for cid, response in zip(prior['ids'], prior['responses'])]
        assert errors and max(errors) < 1e-6, errors
        discrepancy = float(max(errors))
    else:
        warnings.warn('No --reference-sensors supplied: cache was built from exact CFD exports, but historical sensor-response reproduction was NOT checked.', stacklevel=2)
    (cache/'audit.json').write_text(json.dumps({'inputs': provenance, 'prior_sensor_max_abs_error': discrepancy,
        'historical_sensor_reference': str(a.reference_sensors) if a.reference_sensors else None,
        'historical_sensor_check': 'passed' if a.reference_sensors else 'not_requested',
        'grid': [128, 384], 'sensor_ix': IX.tolist(), 'sensor_iy': IY.tolist()}, indent=2))
    print('PREP COMPLETE', flush=True)


def nelder_mead(predict, obs, starts):
    best = [np.inf, None]
    calls = 0
    def objective(q):
        nonlocal calls
        calls += 1
        loss = float(np.mean((predict(q)-obs)**2))
        if loss < best[0]:
            best[:] = [loss, np.array(q, dtype=float)]
        return loss
    successes = 0
    for q in starts:
        simplex = np.array([q, q+[.0625 if q[0]<7.5 else -.0625, 0], q+[0, .0625 if q[1]<2.5 else -.0625]])
        result = minimize(objective, q, method='Nelder-Mead', bounds=[(.5, 7.5), (.5, 2.5)],
            options={'maxiter': 150, 'xatol': 1e-6, 'fatol': 1e-12, 'initial_simplex': simplex})
        successes += int(result.success)
    return best[1], best[0], calls, successes


def evaluate(a):
    import torch
    from analysis.datasize_idw_comparison import inverse_model as inv, hybrid
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Matches the archived evaluation; verified on all 145 cases on 2026-09-12.
    torch.backends.cudnn.allow_tf32 = True
    root = Path(a.root)
    size, seed = SIZES[a.task//3], SEEDS[a.task%3]
    tag = f'n{size:03d}_seed{seed}'
    output = root/'results'/tag
    output.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(root/'cache/rows.csv')
    fields = np.load(root/'cache/fields.npy', mmap_mode='r')
    split_dir = Path(a.split_dir)
    split = pd.read_csv(split_dir/f'split_n{size:03d}.csv')
    train_ids = set(split.loc[split.split == 'train', 'case_id'].astype(int))
    assert len(train_ids) == size
    nested = pd.read_csv(split_dir/'train_nested_order_farthest.csv')
    assert train_ids == set(nested.case_id.iloc[:size].astype(int))
    perm = np.random.default_rng(42).permutation(969)[290:]
    train_idx = [int(i) for i in perm if int(rows.iloc[i].case_id) in train_ids]
    assert len(train_idx) == size and set(rows.iloc[train_idx].split) == {'train'}
    target_idx = rows.index[rows.split.isin(['test', 'offgrid'])].to_numpy()
    assert len(target_idx) in (145, 160)
    xy = rows.iloc[train_idx][['src_x', 'src_y']].to_numpy(dtype=np.float32)
    library = np.asarray(fields[train_idx]).reshape(size, -1)
    sensors = library.reshape(size, 128, 384)[:, IY, IX]
    np.testing.assert_allclose(idw(xy, xy, sensors), sensors, atol=1e-7)
    ckpt_path = Path(a.checkpoints)/f'{tag}.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    meta = ckpt.get('meta', {})
    assert meta.get('target_transform', 'raw') == 'raw'
    assert meta.get('mask_kind', 'soft_rect') == 'soft_rect'
    model = inv.FNO2D(in_ch=3, out_ch=1, width=48, modes1=16, modes2=16, depth=4).cuda().eval()
    model.load_state_dict(ckpt['model_state'], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    xm, xs, ym, ys = [ckpt[k].cuda() for k in ['x_mean', 'x_std', 'y_mean', 'y_std']]
    cfg = dict(ny=128, nx=384, Lx=9., Ly=3., mask_kind='soft_rect', src_size=.2, sigma=.1, soft_k=30.)
    @torch.no_grad()
    def predict(q):
        batch = np.concatenate([inv.build_input_field(float(t[0]), float(t[1]), **cfg) for t in np.atleast_2d(q)])
        pred = model((torch.from_numpy(batch).cuda()-xm)/(xs+1e-12))*(ys+1e-12)+ym
        return pred[:, 0].cpu().numpy()
    def sensor_predict(q):
        return predict(q)[0, IY, IX]
    forward = []
    for i in target_idx:
        r = rows.iloc[i]
        q = r[['src_x', 'src_y']].to_numpy(dtype=float)
        truth = np.asarray(fields[i])
        predictions = {'FNO': predict(q)[0]}
        if seed == 7:
            predictions['IDW'] = idw(q, xy, library).reshape(128, 384)
        for name, pred in predictions.items():
            forward.append(dict(dataset=r.dataset, case_id=int(r.case_id), method=name, n_train=size,
                seed=seed if name == 'FNO' else -1,
                field_rel_l2=float(np.linalg.norm(pred-truth)/(np.linalg.norm(truth)+1e-12)),
                sensor_rel_l2=float(np.linalg.norm(pred[IY, IX]-truth[IY, IX])/(np.linalg.norm(truth[IY, IX])+1e-12)),
                sensor_mse=float(np.mean((pred[IY, IX]-truth[IY, IX])**2))))
    forward = pd.DataFrame(forward)
    discrepancy = None
    if a.reference_metrics_dir:
        archive = pd.read_csv(Path(a.reference_metrics_dir)/f'cases_{tag}.csv')
        actual = forward[(forward.method == 'FNO') & (forward.dataset == 'dense')]
        match = actual.merge(archive[['case_id', 'rel_l2_c']], on='case_id', validate='one_to_one')
        assert len(match) == 145
        discrepancy = float(np.max(np.abs(match.field_rel_l2-match.rel_l2_c)))
        assert discrepancy < 2e-5, f'Checkpoint reproduction failed: {discrepancy}'
    else:
        warnings.warn('No --reference-metrics-dir supplied: historical checkpoint-metric reproduction was NOT checked. This run is not archive-validated.', stacklevel=2)
    forward.to_csv(output/'forward.csv', index=False)
    (output/'audit.json').write_text(json.dumps({'checkpoint_sha256': digest(ckpt_path),
        'train_ids_ordered': rows.iloc[train_idx].case_id.astype(int).tolist(), 'archive_max_abs_error': discrepancy,
        'historical_metric_check': 'passed' if a.reference_metrics_dir else 'not_requested',
        'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'protocol': 'q=1; noise=0; 8 sensors; IDW k=4 p=2; NM 4 identical training-library starts; 150 iterations/start',
        'encoding_source_size': .2, 'encoding_soft_k': 30}, indent=2))
    if a.smoke:
        target_idx = target_idx[:1]
    coarse_xy = np.array([(x, y) for x in np.arange(.5, 7.5001, .25) for y in np.arange(.5, 2.5001, .25)])
    torch.cuda.synchronize()
    start = time.perf_counter()
    coarse_obs = np.concatenate([predict(coarse_xy[j:j+4])[:, IY, IX] for j in range(0, len(coarse_xy), 4)])
    setup_seconds = time.perf_counter()-start
    inverse = []
    for ordinal, i in enumerate(target_idx):
        r = rows.iloc[i]
        obs = np.asarray(fields[i, IY, IX])
        truth_xy = r[['src_x', 'src_y']].to_numpy(dtype=float)
        starts = xy[np.argsort(np.mean((sensors-obs)**2, axis=1), kind='stable')[:4]].astype(float)
        methods = [('FNO_NM_shared', sensor_predict)]
        if seed == 7:
            methods.append(('IDW_NM_shared', lambda q: idw(q, xy, sensors)))
        for name, fun in methods:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            estimate, loss, calls, successes = nelder_mead(fun, obs, starts)
            inverse.append(dict(dataset=r.dataset, case_id=int(r.case_id), method=name, n_train=size,
                seed=seed if name.startswith('FNO') else -1, x_est=estimate[0], y_est=estimate[1],
                loc_error_m=float(np.linalg.norm(estimate-truth_xy)), loss=loss, runtime_s=time.perf_counter()-t0,
                evaluations=calls, converged_starts=successes))
        initials = coarse_xy[np.argsort(np.mean((coarse_obs-obs)**2, axis=1), kind='stable')[:4]]
        t0 = time.perf_counter()
        best = hybrid.optimize_from_initializations(inv, model=model, x_mean=xm, x_std=xs, y_mean=ym, y_std=ys,
            device=torch.device('cuda'), iy=IY, ix=IX, obs=obs, initial_xy=initials,
            ny=128, nx=384, lx=9., ly=3., mask_kind='soft_rect', src_size=.2, sigma=.1, soft_k=30.,
            x_min=.5, x_max=7.5, y_min=.5, y_max=2.5, steps=150, lr=.05)
        torch.cuda.synchronize()
        inverse.append(dict(dataset=r.dataset, case_id=int(r.case_id), method='FNO_coarse_gradient', n_train=size,
            seed=seed, x_est=best['sx'], y_est=best['sy'],
            loc_error_m=float(np.linalg.norm(np.array([best['sx'], best['sy']])-truth_xy)), loss=best['loss'],
            runtime_s=time.perf_counter()-t0, evaluations=600, converged_starts=np.nan))
        pd.DataFrame(inverse).to_csv(output/('inverse_smoke.csv' if a.smoke else 'inverse.csv'), index=False)
        print(f'{tag} inverse {ordinal+1}/{len(target_idx)}', flush=True)
    (output/('SMOKE_OK.json' if a.smoke else 'COMPLETE.json')).write_text(json.dumps({'n_targets': len(target_idx),
        'coarse_library_setup_s': setup_seconds}))


def aggregate(a):
    root = Path(a.root)
    frames = {'forward': [], 'inverse': []}
    for n in SIZES:
        for s in SEEDS:
            d = root/'results'/f'n{n:03d}_seed{s}'
            assert json.loads((d/'COMPLETE.json').read_text())['n_targets'] in (145, 160)
            for kind in frames:
                f = pd.read_csv(d/f'{kind}.csv')
                for _, group in f.groupby(['dataset', 'method']):
                    assert len(group) == (145 if group.dataset.iloc[0] == 'dense' else 15)
                frames[kind].append(f)
    for kind, parts in frames.items():
        all_rows = pd.concat(parts, ignore_index=True)
        all_rows.to_csv(root/f'{kind}_all_cases.csv', index=False)
        metrics = ['field_rel_l2', 'sensor_rel_l2', 'sensor_mse'] if kind == 'forward' else ['loc_error_m', 'runtime_s']
        per_seed = all_rows.groupby(['dataset', 'n_train', 'method', 'seed'])[metrics].mean().reset_index()
        per_seed.to_csv(root/f'{kind}_per_seed.csv', index=False)
        per_seed.groupby(['dataset', 'n_train', 'method'])[metrics].agg(['mean', 'std', 'count']).to_csv(root/f'{kind}_summary.csv')
        comparisons = []
        baseline = 'IDW' if kind == 'forward' else 'IDW_NM_shared'
        accuracy_metrics = metrics if kind == 'forward' else ['loc_error_m']
        for (dataset, n), block in per_seed.groupby(['dataset', 'n_train']):
            base = block[block.method == baseline]
            assert len(base) == 1
            for method, seeds in block[block.method != baseline].groupby('method'):
                assert len(seeds) == 3
                for metric in accuracy_metrics:
                    idw_error = float(base.iloc[0][metric])
                    fno_error = float(seeds[metric].mean())
                    paired = all_rows[(all_rows.dataset == dataset) & (all_rows.n_train == n)]
                    f = paired[paired.method == method].groupby('case_id')[metric].mean()
                    b = paired[paired.method == baseline].set_index('case_id')[metric]
                    assert set(f.index) == set(b.index)
                    comparisons.append(dict(dataset=dataset, n_train=n, method=method, metric=metric,
                        idw_mean=idw_error, fno_mean=fno_error, fno_seed_std=float(seeds[metric].std()),
                        fno_lower_mean=bool(fno_error < idw_error),
                        fno_lower_for_all_three_seeds=bool((seeds[metric] < idw_error).all()),
                        relative_error_reduction_pct=100*(idw_error-fno_error)/max(idw_error, 1e-12),
                        paired_case_win_fraction=float((f < b.reindex(f.index)).mean())))
        pd.DataFrame(comparisons).to_csv(root/f'{kind}_fno_vs_idw_conditions.csv', index=False)
    (root/'AGGREGATION_COMPLETE.txt').write_text('All 21 checkpoints and seven deterministic IDW libraries evaluated.\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'evaluate', 'aggregate'])
    p.add_argument('--root', required=True)
    p.add_argument('--dataset')
    p.add_argument('--checkpoints')
    p.add_argument('--split-dir', default=str(REPO_ROOT/'splits/strict_datasize'))
    p.add_argument('--offgrid', help='Optional directory with the independent CFD cases.csv and exports/. Omit for the 969-case cache only.')
    p.add_argument('--reference-sensors', help='Optional archived NPZ with ids and responses; supplied references are mandatory verification gates.')
    p.add_argument('--reference-metrics-dir', help='Optional archived cases_nNNN_seedSEED.csv directory; supplied references are mandatory verification gates.')
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.mode == 'prepare' and not args.dataset:
        p.error('prepare requires --dataset')
    if args.mode == 'evaluate' and not args.checkpoints:
        p.error('evaluate requires --checkpoints')
    if not 0 <= args.task < len(SIZES)*len(SEEDS):
        p.error('--task must be in 0..20')
    {'prepare': prepare, 'evaluate': evaluate, 'aggregate': aggregate}[args.mode](args)
