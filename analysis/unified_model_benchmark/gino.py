"""Fixed-geometry, three-input BasicGINO2D; no teacher velocity inputs."""
import argparse
import json
import sys
import time
from pathlib import Path


def loader_settings(loader):
    """Report effective loader settings, including trainer defaults/overrides."""
    return {'batch': int(loader.batch_size), 'workers': int(loader.num_workers)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-grid-cache', required=True,
                        help='Directory containing shared rows.csv and fields.npy')
    parser.add_argument('--workdir', required=True, help='Native-point CFD dataset directory')
    parser.add_argument('--split_csv', required=True, help='Frozen 679/145/145 split CSV')
    parser.add_argument('--save', required=True, help='Output checkpoint, normally RUN/model.pt')
    args, remaining = parser.parse_known_args()

    import numpy as np
    import pandas as pd
    import torch
    from scipy.interpolate import griddata

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from model import train_gino_extended_s as trainer
    from model import train_concentration_geofno_extended_s as raw

    # Consume only the wrapper's cache option; the frozen trainer receives its
    # original path options and all other caller-supplied arguments.
    sys.argv = [sys.argv[0], '--workdir', args.workdir, '--split_csv', args.split_csv,
                '--save', args.save, *remaining]
    if not torch.cuda.is_available():
        raise RuntimeError('The published profiling runner requires CUDA; --help needs no GPU.')
    output = Path(args.save).parent
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'COMPLETE.json').exists(), 'Completed run must not be overwritten'

    original_load = raw.load_case_csv
    def room_load(path):
        df = original_load(path)
        return df[df['X (m)'].between(0,9) & df['Y (m)'].between(0,3)].copy()
    raw.load_case_csv = room_load

    class ThreeInput(raw.RawPointCloudDataset):
        def __init__(self, **kwargs):
            kwargs.update(use_velocity_input=False, use_inlet_scalar=False, fixed_inlet_u=.2)
            super().__init__(**kwargs)
            assert len(self.rows) == 969
            for i, r in self.rows.iterrows():
                expected = self.exports/f'case_{int(r.case_id)}_u{r.inlet_u:.4f}_sx{r.src_x:.4f}_sy{r.src_y:.4f}.csv'
                assert expected.is_file() and Path(self.files[i]) == expected
        @property
        def in_channels(self):
            return 3
        @property
        def input_channels(self):
            return ['source_mask','X','Y']
        def __getitem__(self, i):
            sample = dict(super().__getitem__(i))
            sample['features'] = sample['features'][..., :3]
            return sample
    trainer.RawPointCloudDataset = ThreeInput

    original_split = trainer.split_indices_from_csv
    def strict_split(rows, path):
        result = original_split(rows,path)
        assert [len(x) for x in result] == [679,145,145]
        return result
    trainer.split_indices_from_csv = strict_split

    last = {}
    original_eval = trainer.evaluate
    def capture(*args, **kwargs):
        result = original_eval(*args, **kwargs)
        last.update(args=args, metrics=result)
        return result
    trainer.evaluate = capture

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    trainer.main()
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    output = Path(args.save).parent
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    model, loader, xn, yn, transform, device = last['args'][:6]
    hist = pd.read_csv(output/'history_gino_concentration.csv')
    profile = dict(total_wall_s=elapsed, epoch_seconds_sum=float(hist.seconds.sum()), epochs=len(hist),
        peak_allocated_bytes=peak, peak_reserved_bytes=reserved,
        memory_scope='normalization, training, validation and native test; excludes common-grid evaluation',
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, **loader_settings(loader),
        matmul_tf32=False, cudnn_tf32=True,
        parameters=sum(p.numel() for p in model.parameters()), native_test=last['metrics'])
    (output/'profile.json').write_text(json.dumps(profile,indent=2))
    source = Path(args.common_grid_cache)
    rows = pd.read_csv(source/'rows.csv')
    truth_fields = np.load(source/'fields.npy',mmap_mode='r')
    lookup = {int(r.case_id):i for i,r in rows[rows.dataset=='dense'].iterrows()}
    x,y = np.meshgrid(np.linspace(0,9,384,dtype=np.float32),np.linspace(0,3,128,dtype=np.float32))
    details=[]
    with torch.no_grad():
        for batch_raw in loader:
            batch = trainer._to_device(batch_raw,device)
            pred=transform.inverse(yn.decode(model(xn.encode(batch['features']),batch['coords'],batch['valid'],batch['quadrature'])))
            for j in range(len(pred)):
                idx=int(batch['case_index'][j])
                case_id=int(loader.dataset.dataset.rows.iloc[idx].case_id)
                mask=batch['valid'][j]
                pts=batch['coords'][j,mask].cpu().numpy()*np.array([9.,3.])
                values=pred[j,mask,0].cpu().numpy()
                linear=griddata(pts,values,(x,y),method='linear')
                nearest=griddata(pts,values,(x,y),method='nearest')
                field=np.where(np.isnan(linear),nearest,linear)
                truth=truth_fields[lookup[case_id]]
                err=field-truth
                high=np.abs(truth)>=np.percentile(np.abs(truth),90)
                details.append(dict(case_id=case_id,rel_l2_c=float(np.linalg.norm(err)/(np.linalg.norm(truth)+1e-12)),
                    rmse_c=float(np.sqrt(np.mean(err**2))),mae_c=float(np.mean(np.abs(err))),
                    high_rmse_c=float(np.sqrt(np.mean(err[high]**2))),high_mae_c=float(np.mean(np.abs(err[high])))))
    assert len(details)==145 and len({d['case_id'] for d in details})==145
    pd.DataFrame(details).to_csv(output/'common_grid_test.csv',index=False)
    (output/'COMPLETE.json').write_text(json.dumps(pd.DataFrame(details).drop(columns='case_id').mean().to_dict(),indent=2))


if __name__ == '__main__':
    main()
