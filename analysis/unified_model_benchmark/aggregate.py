"""Aggregate every model/seed; fail instead of silently dropping failures."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Benchmark output root with MODEL/seedSEED subdirectories')
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    root = args.root
    records=[]
    reference=None
    for model in ['fno','fcn','unet','deeponet','gino']:
        for seed in [7,42,123]:
            run=root/model/f'seed{seed}'
            assert (run/'COMPLETE.json').is_file(), run
            p=json.loads((run/'profile.json').read_text())
            df=pd.read_csv(run/'common_grid_test.csv')
            ids=set(df.case_id.astype(int))
            assert len(df)==len(ids)==145
            if reference is None: reference=ids
            assert ids==reference
            record=dict(model=model,seed=seed,gpu=p['gpu'],torch_version=p['torch_version'],
                parameters=p['parameters'],epochs=p['epochs'],batch=p['batch'],
                total_wall_min=p['total_wall_s']/60,epoch_sum_min=p['epoch_seconds_sum']/60,
                seconds_per_epoch=p['epoch_seconds_sum']/p['epochs'],
                peak_allocated_GiB=p['peak_allocated_bytes']/2**30,
                peak_reserved_GiB=p['peak_reserved_bytes']/2**30)
            for metric in ['rel_l2_c','rmse_c','mae_c','high_rmse_c','high_mae_c']:
                assert np.isfinite(df[metric]).all()
                record[metric]=float(df[metric].mean())
            records.append(record)
    data=pd.DataFrame(records)
    assert data.gpu.nunique()==data.torch_version.nunique()==1
    data.to_csv(root/'per_seed.csv',index=False)
    summary=data.groupby('model').agg({c:['mean','std'] for c in data.select_dtypes('number').columns if c!='seed'})
    summary.to_csv(root/'summary.csv')
    (root/'COMPLETE.txt').write_text('All 15 runs verified. SD is across seed means.\n')
    print(summary.to_string())


if __name__ == '__main__':
    main()
