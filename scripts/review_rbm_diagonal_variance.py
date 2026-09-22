"""Post-fit algebra audit and independent Gaussian-mixture/step replay."""
import io,json,sqlite3,sys
from pathlib import Path
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.rbm_diagonal_variance import METHODS,VARIANCE_FLOOR,VARIANCE_PENALTY
OUT=ROOT/'results/rbm_diagonal_variance_v1'


def main():
    status=json.loads((OUT/'RUN_STATE.json').read_text())
    assert status['status']=='COMPLETE' and status['review']=='PASS'
    metrics=json.loads((OUT/'METRICS.json').read_text())
    from scripts.run_direct_probability_temporal import old,source_specs
    old.configure_source_root(ROOT.parents[1])
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    for path,digest in manifest['hashes'].items():assert old.sha256_file(Path(path))==digest,path
    con=sqlite3.connect('file:'+str(OUT/'CHECKPOINT.sqlite')+'?mode=ro',uri=True)
    tel={m:dict(valid=0,failed=0,nonconverged=0,floor_hits=0,variance_rmse=[],covariance_error=[]) for m in METHODS}
    checked=0;paired={m:dict(variance_better=0,covariance_better=0,n=0) for m in ('rbm_continued','rbm_diagonal')}
    for start in range(0,13769,128):
        rows=con.execute('SELECT idx,payload,info FROM answers WHERE idx>=? AND idx<? ORDER BY idx',(start,min(start+128,13769))).fetchall()
        assert [r[0] for r in rows]==list(range(start,min(start+128,13769)))
        for i,payload,info in rows:
            d=json.loads(info)
            with np.load(io.BytesIO(payload),allow_pickle=False) as zfile:z={k:zfile[k] for k in zfile.files}
            for j,m in enumerate(METHODS):
                t=tel[m]
                if m in d['failures']:
                    t['failed']+=1;assert np.isnan(z['steps'][:,j]).all();continue
                diag=d['diagnostics'][m];a,w,b,var=(z[m+'::'+k] for k in ('a','w','b','variance'))
                assert len(a)==len(w)==len(var)==diag['active_columns']
                assert np.isfinite(a).all() and np.isfinite(w).all() and np.isfinite(b) and np.isfinite(var).all()
                assert np.all(var>=VARIANCE_FLOOR-1e-12)
                expected=np.zeros(6);expected[diag['columns']]=diag['orientation']*w/var
                np.testing.assert_array_equal(expected,z['weights'][j,:6])
                assert np.all((z['steps'][:,j]>=0)&(z['steps'][:,j]<=1))
                prior=expit(b+np.sum((a*w+.5*w*w)/var))
                predicted=var+prior*(1-prior)*w*w
                np.testing.assert_allclose(predicted,diag['covariance']['predicted_variance'],atol=1e-12,rtol=0)
                if m!='rbm_diagonal':np.testing.assert_array_equal(var,1)
                else:
                    penalty=VARIANCE_PENALTY*np.sum(np.log(var)**2)
                    np.testing.assert_allclose(penalty,diag['penalty'],atol=1e-12,rtol=0)
                    np.testing.assert_allclose(diag['objective_final'],diag['nll_final']+penalty,atol=1e-12)
                    assert diag['regularization']==VARIANCE_PENALTY and diag['variance_floor']==VARIANCE_FLOOR
                    t['floor_hits']+=int(np.sum(var<=VARIANCE_FLOOR+1e-8))
                if m in ('rbm_continued','rbm_diagonal'):
                    np.testing.assert_allclose(diag['objective_initial'],d['diagnostics']['rbm']['nll_final'],atol=1e-12,rtol=0)
                    assert diag['objective_final']<=diag['objective_initial']+1e-8
                    base=d['diagnostics']['rbm']['covariance'];pair=paired[m];pair['n']+=1
                    pair['variance_better']+=diag['covariance']['variance_rmse']<base['variance_rmse']
                    pair['covariance_better']+=diag['covariance']['covariance_relative_error']<base['covariance_relative_error']
                if m=='rbm_initial':
                    np.testing.assert_array_equal(w,np.full(len(w),2/len(w)));np.testing.assert_array_equal(a,0);assert b==0
                t['valid']+=1;t['nonconverged']+=diag.get('converged') is False
                t['variance_rmse'].append(diag['covariance']['variance_rmse'])
                t['covariance_error'].append(diag['covariance']['covariance_relative_error']);checked+=1
    for m,t in tel.items():
        assert t['failed']==len(metrics['telemetry'][m]['failures']) and t['valid']+t['failed']==13769
        for k in ('variance_rmse','covariance_error'):t[k+'_median']=float(np.median(t.pop(k)))
    # Rebuild 27 deterministic real examples without calling the feature/scoring helpers.
    records=json.loads((old.BENCH/'evaluation/JOINED.json').read_text())['records']
    selected={r['uid'] for r in json.loads((OUT/'SMOKE.json').read_text())['rows']}
    replays=0;maxerr=0.
    for cell,path,kind,dataset in source_specs():
        rawrows=old._source_row_map(old.load_pickle(path),kind=kind,dataset=dataset)
        for i,r in enumerate(records):
            if r['cell']!=cell or r['uid'] not in selected:continue
            payload,info=con.execute('SELECT payload,info FROM answers WHERE idx=?',(i,)).fetchone();d=json.loads(info)
            raw=rawrows[r['row_id']];lp=np.asarray(old._topk_payload(raw)['logprobs'],float)[:,:15]
            p=np.exp(lp);q=p/(p.sum(axis=1,keepdims=True)+1e-12);y=-np.log(q+1e-12)
            H=np.sum(q*y,axis=1);V=np.sum(q*(y-H[:,None])**2,axis=1)
            chosen=np.maximum(np.asarray(raw['token_spilled_energies'],float),0)
            X=np.column_stack((H,V,np.sum(q*y**3,axis=1),chosen,chosen**2,chosen**3))
            with np.load(io.BytesIO(payload)) as zfile:z={k:zfile[k] for k in zfile.files}
            for j,m in enumerate(METHODS):
                if m in d['failures']:continue
                diag=d['diagnostics'][m];cols=diag['columns'];mean=np.asarray(diag['normalization_mean']);scale=np.asarray(diag['normalization_scale'])
                np.testing.assert_allclose(X.mean(axis=0),mean,rtol=1e-12,atol=1e-10)
                np.testing.assert_allclose(X.std(axis=0),scale,rtol=1e-12,atol=1e-10)
                Z=(X[:,cols]-mean[cols])/scale[cols]
                a,w,b,var=(z[m+'::'+k] for k in ('a','w','b','variance'))
                s=b+np.sum((a*w+.5*w*w)/var)
                l0=-.5*np.log(var).sum()-.5*np.sum((Z-a)**2/var,axis=1)-np.logaddexp(0,s)
                l1=-.5*np.log(var).sum()-.5*np.sum((Z-a-w)**2/var,axis=1)-np.logaddexp(0,-s)
                nll=-np.logaddexp(l0,l1).mean()
                if m!='rbm_initial':np.testing.assert_allclose(nll,diag['nll_final'],atol=1e-10,rtol=0)
                prob=expit(l1-l0)
                if diag['orientation']<0:prob=1-prob
                steps=np.array([np.sort(prob[u:v])[-10:].mean() for u,v in raw['step_token_spans']])
                np.testing.assert_allclose(steps,z['steps'][:,j],atol=1e-10,rtol=0)
                maxerr=max(maxerr,float(np.max(np.abs(steps-z['steps'][:,j]))));replays+=1
        del rawrows
    con.close()
    result=dict(status='PASS',answers=13769,model_records_checked=checked,telemetry=tel,paired_moment_fit=paired,
                independent_mixture_step_replays=replays,max_step_error=maxerr,
                scope='Post-fit state/algebra audit and independent mixture-likelihood/token-feature replay. No external scientist review or causal validation.')
    (OUT/'STATE_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
