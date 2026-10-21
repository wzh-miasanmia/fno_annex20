"""Frozen protocol: existing checkpoints, identical training libraries, no tuning."""
import argparse
import hashlib
import json
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SIZES = [679]
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
    from analysis.datasize_idw_comparison import run_comparison as cache_builder
    from types import SimpleNamespace
    # Only the original 969 CFD fields are needed; no off-grid exports are read.
    cache_builder.prepare(SimpleNamespace(root=a.source, dataset=a.dataset,
        split_dir=a.split_dir, offgrid=None, reference_sensors=a.reference_sensors))


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


def claim_output_directory(path):
    """Atomically reserve a new condition directory; never overwrite a run.

    An existing directory may belong to a finished run or another worker. Refuse
    both cases rather than relying on a non-atomic COMPLETE.json existence test.
    Failed/partial runs also need a fresh output root or explicit manual recovery.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(exist_ok=False)
    except FileExistsError as error:
        state = 'completed' if (output/'COMPLETE.json').is_file() else 'existing or in-progress'
        raise FileExistsError(f'Refusing to overwrite {state} sensor results: {output}. Use a fresh --root for a rerun.') from error
    return output


def evaluate(a):
    output = claim_output_directory(a.output_dir)
    import torch
    from analysis.datasize_idw_comparison import inverse_model as inv, hybrid
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Matches the archived evaluation; verified on all 145 cases on 2026-09-12.
    torch.backends.cudnn.allow_tf32 = True
    root = Path(a.root)
    size, seed = SIZES[a.task//3], SEEDS[a.task%3]
    tag = f'n{size:03d}_seed{seed}'
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
    target_idx = rows.index[(rows.dataset == 'dense') & (rows.split == 'test')].to_numpy()
    assert len(target_idx) == 145
    xy = rows.iloc[train_idx][['src_x', 'src_y']].to_numpy(dtype=np.float32)
    library = np.asarray(fields[train_idx]).reshape(size, -1)
    sensors = library.reshape(size, 128, 384)[:, IY, IX]
    np.testing.assert_allclose(idw(xy, xy, sensors), sensors, atol=1e-7)
    # The sensor-ablation rerun uses the independently retrained unified-model
    # checkpoints rather than the earlier strict-data-size checkpoints.
    unified_dir = Path(a.checkpoints)/f'seed{seed}'
    ckpt_path = unified_dir/'model.pt'
    unified_split = pd.read_csv(unified_dir/'split_used.csv')
    assert unified_split.split.value_counts().to_dict() == {
        'train': 679, 'val': 145, 'test': 145
    }
    assert set(unified_split.loc[unified_split.split == 'train', 'case_id'].astype(int)) == train_ids
    assert set(unified_split.loc[unified_split.split == 'test', 'case_id'].astype(int)) == set(
        rows.loc[rows.split == 'test', 'case_id'].astype(int)
    )
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
    archive = pd.read_csv(unified_dir/'common_grid_test.csv')
    actual = forward[(forward.method == 'FNO') & (forward.dataset == 'dense')]
    match = actual.merge(archive[['case_id', 'rel_l2_c']], on='case_id', validate='one_to_one')
    assert len(match) == 145
    discrepancy = float(np.max(np.abs(match.field_rel_l2-match.rel_l2_c)))
    assert discrepancy < 2e-5, f'Checkpoint reproduction failed: {discrepancy}'
    forward.to_csv(output/'forward.csv', index=False)
    (output/'audit.json').write_text(json.dumps({'checkpoint_sha256': digest(ckpt_path),
        'train_ids_ordered': rows.iloc[train_idx].case_id.astype(int).tolist(), 'archive_max_abs_error': discrepancy,
        'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'protocol': 'q=1; IDW k=4 p=2; NM 4 identical training-library starts; 150 iterations/start',
        'noise': a.noise, 'layout': a.layout, 'sensor_ix': IX.tolist(), 'sensor_iy': IY.tolist(),
        'encoding_source_size': .2, 'encoding_soft_k': 30}, indent=2))
    if a.smoke:
        target_idx = target_idx[:1]
    coarse_xy = np.array([(x, y) for x in np.arange(.5, 7.5001, .25) for y in np.arange(.5, 2.5001, .25)])
    torch.cuda.synchronize()
    start = time.perf_counter()
    coarse_obs = np.concatenate([predict(coarse_xy[j:j+4])[:, IY, IX] for j in range(0, len(coarse_xy), 4)])
    setup_seconds = time.perf_counter()-start
    inverse = []
    observations = []
    for ordinal, i in enumerate(target_idx):
        r = rows.iloc[i]
        clean = np.asarray(fields[i, IY, IX]).copy()
        # Prefix-stable standard normal draws; identical for every model seed.
        rng = np.random.default_rng(np.random.SeedSequence([20260912, a.layout, int(r.case_id)]))
        noise = rng.normal(size=32)[:len(IX)].astype(np.float32)
        obs = clean + float(a.noise)*max(float(np.max(np.abs(clean))), 1e-12)*noise
        observations.extend(dict(dataset=r.dataset, case_id=int(r.case_id), sensor=j,
            ix=int(IX[j]), iy=int(IY[j]), clean=float(clean[j]), observed=float(obs[j])) for j in range(len(IX)))
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
    pd.DataFrame(observations).to_csv(output/'observations.csv', index=False)
    (output/('SMOKE_OK.json' if a.smoke else 'COMPLETE.json')).write_text(json.dumps({'n_targets': len(target_idx),
        'coarse_library_setup_s': setup_seconds}))


if __name__ == '__main__':
    raise SystemExit('Use analysis/sensor_count_unified_20260917/run_sensors.py instead.')
