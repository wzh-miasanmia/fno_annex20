"""Batch-one synchronized latency of manuscript FNO checkpoints."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-root', required=True,
                        help='Unified benchmark root containing fno/seed{7,42,123}/model.pt')
    parser.add_argument('--cache-dir', required=True,
                        help='Shared CFD cache directory containing rows.csv and fields.npy')
    parser.add_argument('--test-ids', required=True, help='Frozen strict test_case_ids.csv')
    parser.add_argument('--output-dir', required=True, help='Directory for latency CSVs and metadata')
    parser.add_argument('--trainer', default=str(Path(__file__).resolve().parents[2] /
                        'model/train_fixed_u_srcpos_to_c_strict_ablation.py'),
                        help='Frozen FNO module; defaults to the repository model snapshot')
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import torch

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tp = Path(args.trainer)
    sys.path.insert(0, str(tp.parent))
    spec = importlib.util.spec_from_file_location('strict_timing', tp)
    t = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = t
    spec.loader.exec_module(t)
    if not torch.cuda.is_available():
        raise RuntimeError('Synchronized manuscript latency measurement requires a CUDA GPU.')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    allrows = pd.read_csv(Path(args.cache_dir)/'rows.csv')
    ids = pd.read_csv(Path(args.test_ids)).case_id.astype(int)
    rows = allrows[(allrows.dataset=='dense') & allrows.case_id.isin(ids)].set_index('case_id').loc[ids]
    assert len(rows)==145
    fields = np.load(Path(args.cache_dir)/'fields.npy',mmap_mode='r')
    xg,yg = np.meshgrid(np.linspace(0,9,384,dtype=np.float32),np.linspace(0,3,128,dtype=np.float32))
    xc,yc = np.meshgrid(np.linspace(0,1,384,dtype=np.float32),np.linspace(0,1,128,dtype=np.float32))
    records,checks = [],[]
    with torch.inference_mode():
        for seed in (7,42,123):
            path = Path(args.benchmark_root)/'fno'/f'seed{seed}'/'model.pt'
            ck = torch.load(path,map_location='cpu')
            m = ck['meta']
            model = t.FNO2D(**{k:int(m.get(k,v)) for k,v in dict(in_ch=3,out_ch=1,width=48,modes1=16,modes2=16,depth=4).items()}).cuda().eval()
            model.load_state_dict(ck['model_state'])
            xm,xs,ym,ys = [ck[k].cuda() for k in ('x_mean','x_std','y_mean','y_std')]
            def inputs(r):
                mask = t.build_source_mask_soft_rect_sigmoid(xg,yg,float(r.src_x),float(r.src_y),float(m.get('src_size',.2)),float(m.get('soft_k',30))).astype(np.float32)
                return torch.from_numpy(np.stack([mask,xc,yc])[None])
            def decode(y):
                y = y*ys+ym
                if m.get('target_transform','raw')=='log10':
                    y = torch.clamp(torch.pow(10.,y)-float(m.get('target_transform_eps',1e-8)),min=0.)
                return y
            cases = list(rows.itertuples())
            normalized = [(inputs(r).cuda()-xm)/xs for r in cases]
            r = cases[0]
            idx = allrows.index[(allrows.dataset=='dense') & (allrows.case_id==r.Index)][0]
            pred = decode(model(normalized[0]))[0,0].cpu().numpy()
            truth = np.asarray(fields[idx])
            rel = float(np.linalg.norm(pred-truth)/(np.linalg.norm(truth)+1e-12))
            old = pd.read_csv(path.parent/'common_grid_test.csv').set_index('case_id').loc[r.Index,'rel_l2_c']
            assert abs(rel-old)<1e-4,(seed,rel,old)
            checks.append(dict(seed=seed,checkpoint=str(path),checked_rel_l2=rel))
            for mode in ('model_only','source_to_cpu_field'):
                def run(i):
                    if mode=='model_only':
                        return model(normalized[i])
                    return decode(model((inputs(cases[i]).cuda()-xm)/xs)).cpu().numpy()
                for i in range(50):
                    run(i%145)
                torch.cuda.synchronize()
                for repeat in range(3):
                    for i,r in enumerate(cases):
                        torch.cuda.synchronize()
                        start = time.perf_counter_ns()
                        result = run(i)
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter_ns()-start)/1e6
                        records.append(dict(seed=seed,mode=mode,repeat=repeat,case_id=int(r.Index),latency_ms=elapsed))
            del model,normalized
    df = pd.DataFrame(records)
    df.to_csv(out/'latencies.csv',index=False)
    summary = df.groupby(['seed','mode']).latency_ms.agg(['count','mean','std','median','min','max'])
    summary.to_csv(out/'summary.csv')
    (out/'metadata.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,batch=1,resolution=[384,128],warmup=50,repeats=3,n_test=145,cpu_threads=4,job_id=os.environ.get('SLURM_JOB_ID'),precision='FP32; matmul TF32 off; cudnn TF32 on; no autocast or compilation',scope='model_only: pre-normalized GPU input to GPU output; source_to_cpu_field: CPU mask construction, transfers, normalization, forward and decoding. Both exclude loading, disk IO and static coordinate grid setup.',checkpoints=checks),indent=2))
    print(summary.to_string(),flush=True)


if __name__ == '__main__':
    main()
