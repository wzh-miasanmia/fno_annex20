"""Original grid architectures with strict data and shared profiling scope.

Additional arguments are forwarded unchanged to the frozen training module.
Explicit data, split and checkpoint paths prevent use of historical defaults.
"""
import argparse
import inspect
import json
import sys
import time
from pathlib import Path


def loader_settings(loader):
    """Report effective loader settings, including trainer defaults/overrides."""
    return {'batch': int(loader.batch_size), 'workers': int(loader.num_workers)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['fno', 'fcn', 'unet', 'deeponet'])
    parser.add_argument('--workdir', required=True, help='CFD dataset directory')
    parser.add_argument('--split_csv', required=True, help='Frozen 679/145/145 split CSV')
    parser.add_argument('--save', required=True, help='Output checkpoint, normally RUN/model.pt')
    args, remaining = parser.parse_known_args()

    import numpy as np
    import pandas as pd
    import torch

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import train_fixed_u_srcpos_to_c_strict_ablation as strict
    sys.modules['train_fixed_u_srcpos_to_c'] = strict
    from model import train_operator_baselines_srcpos_to_c as baseline

    kind = args.kind
    trainer = strict if kind == 'fno' else baseline
    sys.argv = [sys.argv[0], '--workdir', args.workdir, '--split_csv', args.split_csv,
                '--save', args.save, *remaining]
    if kind != 'fno' and '--model_kind' not in remaining:
        sys.argv.extend(['--model_kind', kind])
    if not torch.cuda.is_available():
        raise RuntimeError('The published profiling runner requires CUDA; --help needs no GPU.')
    output = Path(args.save).parent
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'COMPLETE.json').exists(), 'Completed run must not be overwritten'
    original_split = trainer.split_indices_from_csv

    def checked_split(rows, path):
        indices = original_split(rows, path)
        assert len(rows) == 969 and [len(i) for i in indices] == [679, 145, 145]
        pd.DataFrame([{'case_id': int(rows.iloc[i].case_id), 'split': tag}
            for tag, group in zip(['train', 'val', 'test'], indices) for i in group]
            ).to_csv(output / 'split_used.csv', index=False)
        return indices

    trainer.split_indices_from_csv = checked_split
    original_dataset = trainer.CasesFNOFieldDataset

    class CheckedDataset(original_dataset):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            assert len(self.rows) == 969 and not self.use_velocity_input
            for _, r in self.rows.iterrows():
                exact = self.exports / f'case_{int(r.case_id)}_u{r.inlet_u:.4f}_sx{r.src_x:.4f}_sy{r.src_y:.4f}.csv'
                assert exact.is_file() and self.file_map[str(int(r.case_id))] == exact

    trainer.CasesFNOFieldDataset = CheckedDataset
    last = {}
    original_eval = trainer.evaluate_metrics

    def capture_eval(*args, **kwargs):
        result = original_eval(*args, **kwargs)
        last['args'] = args
        return result

    trainer.evaluate_metrics = capture_eval
    original_train = trainer.train
    timing = {}

    def measured_train(*args, **kwargs):
        bound = inspect.signature(original_train).bind(*args, **kwargs)
        timing['parameters'] = sum(p.numel() for p in bound.arguments['model'].parameters())
        timing.update(loader_settings(bound.arguments['train_loader']))
        torch.cuda.synchronize()
        tick = time.perf_counter()
        result = original_train(*args, **kwargs)
        torch.cuda.synchronize()
        timing['training_loop_wall_s'] = time.perf_counter() - tick
        return result

    trainer.train = measured_train
    profile_written = False

    def write_profile():
        nonlocal profile_written
        if profile_written:
            return
        torch.cuda.synchronize()
        profile = dict(total_wall_s=time.perf_counter() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
            memory_scope='normalization, training, validation and native test; excludes per-case evaluation',
            kind=kind, cpu_threads=4, matmul_tf32=False, cudnn_tf32=True, **timing)
        (output / 'profile.json').write_text(json.dumps(profile, indent=2))
        profile_written = True

    original_cases = baseline.evaluate_case_metrics

    def measured_cases(*args, **kwargs):
        write_profile()
        return original_cases(*args, **kwargs)

    baseline.evaluate_case_metrics = measured_cases
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    trainer.main()
    write_profile()
    if kind == 'fno':
        model, loader, xn, yn, transform, device = last['args'][:6]
        details = original_cases(model, loader.dataset, xn, yn, transform, device,
                                 'fno', output / 'model.pt')
        details.to_csv(output / 'common_grid_test.csv', index=False)
    else:
        details = pd.read_csv(output / 'common_grid_test.csv')
    assert len(details) == 145 and details.case_id.nunique() == 145
    assert np.isfinite(details[['rel_l2_c', 'rmse_c', 'mae_c', 'high_rmse_c']]).all().all()
    hist = pd.read_csv(output / 'history_model.csv')
    profile = json.loads((output / 'profile.json').read_text())
    profile.update(epochs=len(hist), epoch_seconds_sum=float(hist.epoch_time_sec.sum()))
    (output / 'profile.json').write_text(json.dumps(profile, indent=2))
    (output / 'COMPLETE.json').write_text(json.dumps({'mean_rel_l2_c': float(details.rel_l2_c.mean())}))


if __name__ == '__main__':
    main()
