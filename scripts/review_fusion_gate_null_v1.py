"""Review synthetic paths, independent Kalman/GMM algebra and grouped counts."""
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[1]
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d=module(ROOT/'scripts/run_fusion_gate_null_v1.py','null_review_driver');OUT=d.OUT
vector=module(ROOT/'tests/test_fusion_trajectory_imm.py','null_vector_reference')


def norm(x):return (x-x.mean())/x.std()
def scalar_kalman(x,r):
    mean=0.;variance=1.;out=[]
    for value in x:
        pred=variance+.01*r;gain=pred/(pred+r);mean=mean+gain*(value-mean);variance=(1-gain)*pred;out.append(mean)
    return np.asarray(out)
def ci(k,n):
    if not n:return None
    return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]


def main():
    start=time.monotonic();m=d.verify();f=d.load(OUT/'FROZEN.json');assert f['status']=='COMPLETE' and f['trials']==768
    assert f['manifest_sha256']==d.sha(OUT/'MANIFEST.json')
    for p,h in f['files'].items():assert d.sha(p)==h,p
    records=[];counts=Counter();failures=Counter();maximum=0.
    for rec in m['selected']:
        uid=d.identity(rec);meta=d.load(OUT/'trials'/(uid+'.json'));assert meta['manifest_sha256']==f['manifest_sha256']
        for key,value in rec.items():assert meta[key]==value
        assert meta['array_sha256']==d.sha(OUT/'trials'/(uid+'.npz'))
        with np.load(OUT/'trials'/(uid+'.npz'),allow_pickle=False) as a:
            seed=int(hashlib.sha256(f'fusion-gate-null-v1/{rec["n"]}/{rec["replicate"]}'.encode()).hexdigest()[:8],16)
            z=np.random.default_rng(seed).normal(size=rec['n']+256);np.testing.assert_array_equal(a['innovations'],z);assert seed==meta['source']['seed']
            path=np.empty_like(z);path[0]=z[0];rho=rec['rho']
            for t in range(1,len(z)):path[t]=rho*path[t-1]+np.sqrt(1-rho*rho)*z[t]
            if rec['jump']:path[256+rec['n']//2:]+=3
            np.testing.assert_array_equal(path,a['path']);tail=path[-rec['n']:];context=(path-tail.mean())/tail.std()
            np.testing.assert_allclose(a['context'],context,atol=1e-14,rtol=0);np.testing.assert_allclose(a['tail'],norm(tail),atol=1e-14,rtol=0)
            scale=meta['source']['source_scale'];np.testing.assert_allclose([scale['mean'],scale['sd']],[tail.mean(),tail.std()],atol=1e-14,rtol=0)
            r=float(np.clip((np.median(np.abs(np.diff(context[-rec['n']:]))) /(.67448975*np.sqrt(2)))**2,.05,1))
            np.testing.assert_allclose(r,meta['model']['observation_variance'],atol=1e-14,rtol=0)
            for warm in (False,True):
                source=context if warm else context[-rec['n']:];suffix='_warm' if warm else '_cold'
                expected=scalar_kalman(source,r)[-rec['n']:];actual=a['kalman'+suffix+'__unscaled']
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12);maximum=max(maximum,float(np.max(np.abs(actual-expected))));counts['scalar_kalman_replays']+=1
                if rec['replicate']<2:
                    expected=vector.vector_imm(source[:,None],np.array([[r]]),[.01*r,r])['level'][-rec['n']:]
                    actual=a['imm'+suffix+'__unscaled'];np.testing.assert_allclose(actual,expected,atol=1e-11,rtol=1e-11);counts['direct_vector_imm_replays']+=1
            for name in m['readouts']:
                y=a[name];unscaled=a[name+'__unscaled'];np.testing.assert_allclose(y,norm(unscaled),atol=1e-14,rtol=0)
                scale=meta['model']['scales'][name];np.testing.assert_allclose([scale['mean'],scale['sd']],[unscaled.mean(),unscaled.std()],atol=1e-13,rtol=1e-13)
                method=meta['methods'][name]
                if not method['valid']:
                    assert method['reason'];failures[name+'/'+method['reason']]+=1
                else:
                    gate=method['gate'];n=len(y);onevar=y.var()+1e-4
                    ll1=-.5*float(np.sum(np.log(2*np.pi*onevar)+(y-y.mean())**2/onevar))
                    means=np.asarray(gate['means']);var=np.asarray(gate['variances']);weight=np.asarray(gate['weights'])
                    assert np.all(var>0) and np.all(weight>0);np.testing.assert_allclose(weight.sum(),1,atol=1e-12)
                    ll2=float(np.sum(logsumexp(np.log(weight)[None,:]-.5*(np.log(2*np.pi*var)[None,:]+(y[:,None]-means[None,:])**2/var[None,:]),axis=1)))
                    bic=[2*np.log(n)-2*ll1,5*np.log(n)-2*ll2]
                    np.testing.assert_allclose(gate['log_likelihood'],[ll1,ll2],atol=1e-9,rtol=1e-12);np.testing.assert_allclose(gate['bic'],bic,atol=1e-9,rtol=1e-12)
                    assert gate['two_components_selected']==(bic[1]<bic[0])
                    opened=bic[1]<bic[0] and np.any(y>means.mean());assert method['gate_open']==opened
                    if opened:assert gate['prediction']==int(np.flatnonzero(y>means.mean())[0])
                    else:assert gate['prediction']==-1
                    counts['independent_mixture_algebra']+=1
                    if rec['replicate']<2:
                        replay=d.core.inspect_mixture(y,y);np.testing.assert_allclose(replay['bic'],gate['bic'],atol=1e-10,rtol=1e-12)
                        assert replay['prediction']==gate['prediction'];counts['representative_gmm_refits']+=1
                counts['normalization_checks']+=1
            records.append(meta);counts['source_path_replays']+=1
    summaries=[];pairs=[]
    cells=sorted({(r['n'],r['rho'],r['jump']) for r in records})
    for n,rho,jump in cells:
        rows=[r for r in records if (r['n'],r['rho'],r['jump'])==(n,rho,jump)];assert len(rows)==64 and len({r['replicate'] for r in rows})==64
        for name in m['readouts']:
            valid=[r['methods'][name] for r in rows if r['methods'][name]['valid']];k=sum(v['gate_open'] for v in valid);fail=64-len(valid)
            summaries.append(dict(n=n,rho=rho,jump=jump,readout=name,trials=64,valid=len(valid),failures=fail,open=k,rate=k/len(valid) if valid else None,
                interval=ci(k,len(valid)),all_trial_bounds=[k/64,(k+fail)/64],two_components=sum(v['gate']['two_components_selected'] for v in valid),
                median_bic_gain=float(np.median([v['gate']['bic_gain'] for v in valid])) if valid else None))
        for left,right in [(name,'raw') for name in m['readouts'] if name!='raw']+[('imm_warm','imm_cold'),('kalman_warm','kalman_cold')]:
            common=[r for r in rows if r['methods'][left]['valid'] and r['methods'][right]['valid']]
            gained=sum(r['methods'][left]['gate_open'] and not r['methods'][right]['gate_open'] for r in common)
            lost=sum(not r['methods'][left]['gate_open'] and r['methods'][right]['gate_open'] for r in common)
            delta=sum(r['methods'][left]['gate_open']-r['methods'][right]['gate_open'] for r in common);assert gained-lost==delta
            pairs.append(dict(n=n,rho=rho,jump=jump,left=left,right=right,common=len(common),open_gained=gained,open_lost=lost))
    result=dict(status='REVIEWED_SIMULATION',summaries=summaries,pairs=pairs,counts=dict(counts),failures=dict(failures),
        interpretation='Synthetic gate-opening frequencies under declared generators, not hallucination false-positive rates or achieved benchmark results. Warm uses256 extra synthetic points. Conditional-valid intervals and all-trial failure bounds are separate; all cells retained.',
        parent_benchmark=str(ROOT/'results/fusion_trajectory_imm_v1/EVALUATION.json'))
    d.save(OUT/'RESULTS.json',result)
    review=dict(status='PASS',counts=dict(counts),failures=dict(failures),maximum_kalman_difference=maximum,seconds=time.monotonic()-start,
        hashes={str(OUT/n):d.sha(OUT/n) for n in ('MANIFEST.json','FROZEN.json','RESULTS.json')},dependencies={str(p):d.sha(p) for p in [Path(__file__),Path(vector.__file__)]},
        versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn']},
        scope='Same-session review: independent scalar Kalman, stationary source and mixture-likelihood/BIC algebra; representative direct vector IMM and shared actual GMM refits. Test vector reference reused; no external review, semantic-null guarantee or new model inference.')
    d.save(OUT/'REVIEW.json',review);print('REVIEW PASS',dict(counts),flush=True)
    for x in summaries:
        if x['readout'] in ('raw','imm_cold','imm_warm'):print(x['n'],x['rho'],x['jump'],x['readout'],x['open'],x['valid'],flush=True)


if __name__=='__main__':main()
