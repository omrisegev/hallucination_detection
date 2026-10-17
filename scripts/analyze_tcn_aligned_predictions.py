"""Full held-answer TCN diagnostics and independent signed-readout replay."""
from pathlib import Path
import sys,json,sqlite3,io,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from spectral_utils.context_training import FeatureBundle
from spectral_utils.aligned_context_predictors import agreement,remove_current
from scripts.run_aligned_context_predictors import models,ridge_prediction
from scripts.run_tcn_aligned_study import OUT,MODELS,write,sha


def run():
    begin=time.perf_counter();bundle=FeatureBundle(ROOT/'results/temporal_context_data_v1','innovation5')
    ridge,_=models(bundle);diagnostics={};audit={};maxdelta=0.;seen=set()
    with np.load(ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:base=f['steps__append_innovation__H0lim']
    for fold in range(5):
        path=MODELS/f'tcn__innovation5__seed0__exclude{fold}'/'scoring'
        state=json.loads((path/'RUN_STATE.json').read_text())
        if state['status']!='SCORED':raise ValueError('Need complete outer scoring')
        con=sqlite3.connect(path/'SCORES.sqlite')
        count=0
        for i,blob,tokens in con.execute('SELECT idx,steps,tokens FROM answers ORDER BY idx'):
            if i in seen:raise ValueError('Repeated outer answer')
            m=bundle.metadata[i];n=m['tokens'];a=m['offset'];sl=slice(m['step_start'],m['step_stop'])
            if m['fold']!=fold:raise ValueError('Answer assigned wrong fold')
            x=np.asarray(bundle.features[a:a+n],float)[:,bundle.columns];z=(x-bundle.mean[i])/bundle.scale[i]
            rp=ridge_prediction(z,ridge[fold]);current=z.mean(1);rr=(z-rp).mean(1)
            spans=np.asarray(bundle.spans[sl])-a;row={}
            with np.load(io.BytesIO(tokens),allow_pickle=False) as t,np.load(io.BytesIO(blob),allow_pickle=False) as s:
                real=t['real__mean']
                for condition in ('real','shuffled','zero'):
                    mu=t[condition+'__mean'];var=t[condition+'__variance']
                    if mu.shape!=z.shape or var.shape!=z.shape or not np.isfinite(mu).all() or not np.isfinite(var).all() or np.any(var<=0):
                        raise ValueError('Invalid saved TCN prediction')
                    residual=z-mu;signed=residual.mean(1)
                    aux=np.array([np.sort(signed[u:v])[-min(10,v-u):].mean() for u,v in spans])
                    expected=base[sl].copy()
                    if aux.std()>1e-12:expected+=.25*base[sl].std()*(aux-aux.mean())/aux.std()
                    actual=s[condition+'__signed_residual_0.25']
                    delta=float(np.max(np.abs(expected-actual)));maxdelta=max(maxdelta,delta)
                    np.testing.assert_allclose(expected,actual,atol=2e-12,rtol=0)
                    row[condition]=dict(mse=np.square(residual).mean(0).tolist(),
                        mse_after16=np.square(residual[16:]).mean(0).tolist() if n>16 else None,
                        mean_variance=var.mean(0).tolist(),
                        gaussian_nll=float(np.mean(.5*(np.log(2*np.pi*var)+residual**2/var))),
                        prediction_correlation_ridge=agreement(mu.mean(1),rp.mean(1)),
                        signed_residual_correlation_ridge=agreement(signed,rr),
                        residual_correlation_after_removing_current=agreement(remove_current(signed,current),remove_current(rr,current)),
                        signed_residual_correlation_current=agreement(signed,current),
                        prediction_real_correlation=agreement(mu.mean(1),real.mean(1)),
                        signed_residual_real_correlation=agreement(signed,(z-real).mean(1)),
                        prediction_real_squared_difference=float(np.mean(np.square(mu-real))))
            diagnostics[i]=dict(uid=m['uid'],tokens=n,methods=row);seen.add(i);count+=1
        con.close()
        expected={i for i,m in enumerate(bundle.metadata) if m['fold']==fold}
        if count!=len(expected) or not expected<=seen:raise ValueError('Missing fold answers')
        audit[str(fold)]=dict(answers=count,sqlite_sha256=sha(path/'SCORES.sqlite'),scores_sha256=sha(path/'STEP_SCORES.npz'))
        print('[tcn-token-audit]',fold,count,flush=True)
    if len(seen)!=len(bundle.metadata):raise ValueError('Incomplete full population')
    write(OUT/'DIAGNOSTICS.json',[diagnostics[i] for i in range(len(bundle.metadata))])
    write(OUT/'PREDICTION_AUDIT.json',dict(status='PASS',answers=len(seen),tokens=int(bundle.length.sum()),
        max_scalar_readout_delta=maxdelta,folds=audit,seconds=time.perf_counter()-begin,
        diagnostics_sha256=sha(OUT/'DIAGNOSTICS.json')))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
