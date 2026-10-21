"""Joint single-source location/amplitude inference on independent CFD cases."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
if __package__:
    from .run_comparison import idw, nelder_mead, digest, IX, IY
else:
    from run_comparison import idw, nelder_mead, digest, IX, IY

SEEDS = [7,42,123]
NOISES = [0., .02, .05]


def profile(response, observed):
    f = np.asarray(response, dtype=np.float64)
    c = np.asarray(observed, dtype=np.float64)
    denominator = np.sum(f*f, axis=-1)
    q = np.maximum(0., np.sum(f*c,axis=-1)/np.maximum(denominator,1e-24))
    q = np.where(denominator <= 1e-24, 0., q)
    loss = np.mean((q[...,None]*f-c)**2,axis=-1)
    return q, loss


def tests():
    f=np.arange(1,9,dtype=float)
    for q in [.5,1.,2.,3.7]:
        est,loss=profile(f,q*f)
        np.testing.assert_allclose(est,q)
        assert loss < 1e-20
    assert profile(f,-f)[0] == 0
    assert np.isfinite(profile(np.zeros(8),f)[1])
    q,l=profile(np.stack([f,2*f]),3*f)
    np.testing.assert_allclose(q,[3,1.5])
    assert np.max(l) < 1e-20


def prepare(a):
    root=Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    inputs=Path(a.inputs) if a.inputs else root/'inputs'
    rows=pd.read_csv(inputs/'cases.csv').sort_values('case_id').reset_index(drop=True)
    assert len(rows)==45 and rows.case_id.is_unique
    assert set(rows.source_strength)=={.5,1.,2.} and np.allclose(rows.inlet_u,.2)
    groups=rows.groupby(['src_x','src_y'])
    assert len(groups)==15 and all(set(g.source_strength)=={.5,1.,2.} for _,g in groups)
    dense=pd.read_csv(Path(a.source)/'cache/rows.csv')
    dense=dense[dense.dataset=='dense']
    for xy in groups.groups:
        assert not np.any(np.all(np.isclose(dense[['src_x','src_y']].values,xy),axis=1))
    x,y=np.meshgrid(np.linspace(0,9,384,dtype=np.float32),np.linspace(0,3,128,dtype=np.float32))
    observations=[]; audit=[]
    for _,r in rows.iterrows():
        p=inputs/'exports'/f'case_{int(r.case_id)}_u{r.inlet_u:.4f}_sx{r.src_x:.4f}_sy{r.src_y:.4f}_q{r.source_strength:.4f}.csv'
        df=pd.read_csv(p)
        pts=df[['X (m)','Y (m)']].to_numpy(dtype=np.float64)
        values=df['Passive Scalar'].to_numpy(dtype=np.float64)
        linear=griddata(pts,values,(x,y),method='linear')
        near=griddata(pts,values,(x,y),method='nearest')
        field=np.where(np.isnan(linear),near,linear).astype(np.float32)
        observations.append(field[IY,IX])
        audit.append(dict(case_id=int(r.case_id),sha256=digest(p)))
    observations=np.array(observations)
    assert np.isfinite(observations).all()
    previous=np.load(Path(a.source)/'cache/fields.npy',mmap_mode='r')
    previous_rows=pd.read_csv(Path(a.source)/'cache/rows.csv')
    for i,r in rows[rows.source_strength==1].iterrows():
        j=previous_rows.index[(previous_rows.dataset=='offgrid') & (previous_rows.case_id==r.case_id)]
        assert len(j)==1
        np.testing.assert_array_equal(observations[i],previous[j[0],IY,IX])
    np.save(root/'observations.npy',observations)
    rows.to_csv(root/'cases.csv',index=False)
    (root/'inputs_audit.json').write_text(json.dumps(audit,indent=2))


def gradient_joint(inv, model, stats, obs, starts):
    import torch
    xm,xs,ym,ys=stats
    lo=torch.tensor([.5,.5],device='cuda'); span=torch.tensor([7.,2.],device='cuda')
    initial=np.clip((starts-np.array([.5,.5]))/np.array([7.,2.]),1e-5,1-1e-5)
    z=torch.nn.Parameter(torch.tensor(np.log(initial/(1-initial)),device='cuda',dtype=torch.float32))
    opt=torch.optim.Adam([z],lr=.05)
    target=torch.tensor(obs,device='cuda',dtype=torch.float32)
    best=(np.inf,None,None)
    for _ in range(150):
        opt.zero_grad(set_to_none=True)
        xy=lo+span*torch.sigmoid(z)
        batch=torch.cat([inv._build_input_field_torch(sx=p[0],sy=p[1],ny=128,nx=384,Lx=9.,Ly=3.,
            mask_kind='soft_rect',src_size=.2,sigma=.12,soft_k=30.,device=torch.device('cuda'),dtype=torch.float32) for p in xy])
        pred=model((batch-xm)/(xs+1e-12))*(ys+1e-12)+ym
        f=pred[:,0,IY,IX]
        den=(f*f).sum(dim=1)
        q=torch.clamp((f*target).sum(dim=1)/den.clamp_min(1e-24),min=0)
        q=torch.where(den>1e-24,q,torch.zeros_like(q))
        losses=((q[:,None]*f-target)**2).mean(dim=1)
        k=int(losses.argmin())
        if float(losses[k].detach()) < best[0]:
            best=(float(losses[k].detach()),xy[k].detach().cpu().numpy(),float(q[k].detach()))
        losses.sum().backward()
        assert torch.isfinite(z.grad).all()
        opt.step()
    return best[1],best[2],best[0]


def evaluate(a):
    import torch
    if __package__:
        from . import inverse_model as inv
    else:
        import inverse_model as inv
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path(a.root); source=Path(a.source); seed=SEEDS[a.task]
    out=root/(f'smoke_seed{seed}' if a.smoke else f'seed{seed}')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        raise FileExistsError(f'Completed results must not be overwritten: {out}')
    cases=pd.read_csv(root/'cases.csv'); obs_all=np.load(root/'observations.npy')
    rows=pd.read_csv(source/'cache/rows.csv'); fields=np.load(source/'cache/fields.npy',mmap_mode='r')
    train=np.random.default_rng(42).permutation(969)[290:]
    assert len(train)==679 and set(rows.iloc[train].split)=={'train'}
    xy=rows.iloc[train][['src_x','src_y']].to_numpy(dtype=np.float32)
    library=np.asarray(fields[train])[:,IY,IX]
    p=Path(a.checkpoints)/f'n679_seed{seed}.pt'
    ck=torch.load(p,map_location='cpu',weights_only=False)
    assert ck['meta']['in_ch']==3 and ck['meta']['target_transform']=='raw'
    model=inv.FNO2D(in_ch=3,out_ch=1,width=48,modes1=16,modes2=16,depth=4).cuda().eval()
    model.load_state_dict(ck['model_state'])
    for parameter in model.parameters(): parameter.requires_grad_(False)
    stats=[ck[k].cuda() for k in ['x_mean','x_std','y_mean','y_std']]
    xm,xs,ym,ys=stats
    @torch.no_grad()
    def forward(points):
        batch=np.concatenate([inv.build_input_field(float(pt[0]),float(pt[1]),ny=128,nx=384,Lx=9.,Ly=3.,
            mask_kind='soft_rect',src_size=.2,sigma=.12,soft_k=30.) for pt in np.atleast_2d(points)])
        pred=model((torch.from_numpy(batch).cuda()-xm)/(xs+1e-12))*(ys+1e-12)+ym
        return pred[:,0].cpu().numpy()
    def predict(point): return forward(point)[0,IY,IX]
    reference_dir=Path(a.reference_metrics_dir) if a.reference_metrics_dir else source/'reference'
    reference_path=reference_dir/f'cases_n679_seed{seed}.csv'
    if not reference_path.is_file():
        raise FileNotFoundError(f'Joint inversion requires the strict-campaign per-case forward reference: {reference_path}. Supply --reference-metrics-dir; do not substitute unified Table 2 metrics/checkpoints.')
    archive=pd.read_csv(reference_path).set_index('case_id')
    deltas=[]
    for i,r in rows[rows.split=='test'].iterrows():
        field=forward([r.src_x,r.src_y])[0]
        error=np.linalg.norm(field-fields[i])/np.linalg.norm(fields[i])
        deltas.append(abs(float(error)-archive.loc[int(r.case_id),'rel_l2_c']))
    assert len(deltas)==145 and max(deltas)<2e-5
    coarse=np.array([(x,y) for x in np.arange(.5,7.5001,.25) for y in np.arange(.5,2.5001,.25)])
    coarse_obs=np.concatenate([forward(coarse[j:j+4])[:,IY,IX] for j in range(0,len(coarse),4)])
    results=[]; recorded=[]
    selected=cases.iloc[:3] if a.smoke else cases
    for i,r in selected.iterrows():
        for noise in NOISES:
            # Shared noise draw within each source position across q; no q passed to inference.
            position_id=int(cases[(cases.src_x==r.src_x)&(cases.src_y==r.src_y)].case_id.min())
            rng=np.random.default_rng(np.random.SeedSequence([20260913,position_id]))
            clean=obs_all[i]
            obs=clean+noise*max(float(np.max(np.abs(clean))),1e-12)*rng.normal(size=8).astype(np.float32)
            recorded.extend(dict(case_id=int(r.case_id),noise=noise,sensor=j,observed=float(v)) for j,v in enumerate(obs))
            # Only observed concentrations and unit-strength training library enter initialization.
            _,scores=profile(library,obs)
            starts=xy[np.argsort(scores,kind='stable')[:4]].astype(float)
            methods=[('FNO',predict)]
            if seed==7: methods.append(('IDW',lambda point:idw(point,xy,library)))
            estimates=[]
            for name,fun in methods:
                t=time.perf_counter()
                def scaled(point):
                    f=fun(point); q,_=profile(f,obs); return q*f
                est,loss,calls,success=nelder_mead(scaled,obs,starts)
                q,_=profile(fun(est),obs)
                estimates.append((name+'_NM_joint',est,float(q),loss,time.perf_counter()-t,calls))
                # Separate oracle-strength control; never used to initialize the joint estimate.
                known_starts=xy[np.argsort(np.mean((float(r.source_strength)*library-obs)**2,axis=1),kind='stable')[:4]].astype(float)
                t=time.perf_counter()
                est,loss,calls,success=nelder_mead(lambda pt:float(r.source_strength)*fun(pt),obs,known_starts)
                estimates.append((name+'_NM_known_q',est,float(r.source_strength),loss,time.perf_counter()-t,calls))
            _,scores=profile(coarse_obs,obs)
            starts=coarse[np.argsort(scores,kind='stable')[:4]]
            t=time.perf_counter()
            est,q,loss=gradient_joint(inv,model,stats,obs,starts)
            estimates.append(('FNO_coarse_gradient_joint',est,q,loss,time.perf_counter()-t,600))
            for name,est,q,loss,seconds,calls in estimates:
                results.append(dict(case_id=int(r.case_id),q_true=float(r.source_strength),noise=noise,method=name,
                    seed=seed if name.startswith('FNO') else -1,x_true=r.src_x,y_true=r.src_y,x_est=est[0],y_est=est[1],
                    q_est=q,loc_error_m=float(np.linalg.norm(est-[r.src_x,r.src_y])),
                    q_relative_error=abs(q-float(r.source_strength))/float(r.source_strength),loss=loss,runtime_s=seconds,evaluations=calls))
            pd.DataFrame(results).to_csv(out/'cases_results.csv',index=False)
            print(f'seed={seed} case={r.case_id} noise={noise} complete',flush=True)
    pd.DataFrame(recorded).to_csv(out/'observations.csv',index=False)
    (out/'COMPLETE.json').write_text(json.dumps(dict(n_cases=len(selected),archive_discrepancy=max(deltas),checkpoint_sha256=digest(p))))


def aggregate(a):
    root=Path(a.root); parts=[]; reference=None
    for seed in SEEDS:
        d=root/f'seed{seed}'
        assert json.loads((d/'COMPLETE.json').read_text())['n_cases']==45
        rows=pd.read_csv(d/'cases_results.csv')
        for _,g in rows.groupby(['method','noise']):
            assert len(g)==45 and g.case_id.nunique()==45
        obs=pd.read_csv(d/'observations.csv')
        if reference is None: reference=obs
        else: pd.testing.assert_frame_equal(reference,obs)
        parts.append(rows)
    rows=pd.concat(parts,ignore_index=True)
    rows.to_csv(root/'all_cases.csv',index=False)
    groups=['method','noise','q_true','seed']
    summary=rows.groupby(groups).agg(mean_loc_error_m=('loc_error_m','mean'),median_loc_error_m=('loc_error_m','median'),
        mean_q_relative_error=('q_relative_error','mean'),median_q_relative_error=('q_relative_error','median')).reset_index()
    summary.to_csv(root/'per_seed.csv',index=False)
    summary.groupby(groups[:-1])[list(summary.columns[4:])].agg(['mean','std','count']).to_csv(root/'summary.csv')
    (root/'COMPLETE.txt').write_text('45 independent CFD cases, 3 noises, 3 FNO seeds; paired observations verified.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','evaluate','aggregate','test'])
    p.add_argument('--root'); p.add_argument('--source'); p.add_argument('--checkpoints')
    p.add_argument('--inputs', help='45-case off-grid CFD directory containing cases.csv and exports/; prepare only.')
    p.add_argument('--reference-metrics-dir', help='Strict n679 campaign directory with cases_n679_seed7/42/123.csv. Required gate; defaults to SOURCE/reference/.')
    p.add_argument('--task',type=int,default=0); p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); tests()
    if a.mode != 'test' and not a.root:
        p.error('--root is required for prepare/evaluate/aggregate')
    if a.mode in ('prepare','evaluate') and not a.source:
        p.error('--source is required for prepare/evaluate')
    if a.mode == 'evaluate' and not a.checkpoints:
        p.error('--checkpoints is required for evaluate; use the strict n679 campaign')
    if not 0 <= a.task < len(SEEDS):
        p.error('--task must be 0, 1, or 2 (seeds 7, 42, 123)')
    if a.mode!='test': globals()[a.mode](a)
