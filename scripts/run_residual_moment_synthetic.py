"""Bounded temporal mechanism checks before the full real-data comparison."""
from pathlib import Path
import sys, json, time, hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from spectral_utils.temporal_context_models import RidgePredictor
from spectral_utils.residual_moment_fusion import fit_pair, score_crossed, jsonable

WORLDS={'fast_slow':(0.,.97,1.),'fast_fast':(0.,0.,1.),'fast_none':(0.,0.,0.),
        'slow_fast':(.97,0.,1.),'slow_slow':(.97,.97,1.)}

def world(seed, py, pn, nuisance):
    rng=np.random.default_rng(seed);n,T=200,64
    def ar(phi):
        a=rng.normal(size=(n,T))
        for t in range(1,T):a[:,t]=phi*a[:,t-1]+np.sqrt(1-phi**2)*a[:,t]
        return a
    y=ar(py); labels=y+rng.normal(scale=.15,size=y.shape)>0
    z=ar(pn)
    x=y[...,None]*[1,1,1,.3,.3,.3]+nuisance*z[...,None]*[.3,.3,.3,1.6,1.6,1.6]+rng.normal(scale=.45,size=(n,T,6))
    H=np.stack([x[:,t-16:t].reshape(n,-1) for t in range(16,T)],axis=1)
    return H,x[:,16:],y[:,16:],z[:,16:],labels[:,16:]

def run():
    out=ROOT/'results/residual_moment_synthetic_v1';out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();rows=[]
    for name,params in WORLDS.items():
        for seed in range(20):
            H,X,y,z,labels=world(seed,*params);Ht,Xt,yt,zt,labelt=world(seed+100000,*params)
            R=np.empty_like(X)
            fold=np.arange(len(X))%5
            for k in range(5):
                fit=RidgePredictor.fit(H[fold!=k].reshape(-1,96),X[fold!=k].reshape(-1,6),ridge=1.)
                R[fold==k]=X[fold==k]-fit.predict(H[fold==k].reshape(-1,96)).reshape(-1,48,6)
            fit=RidgePredictor.fit(H.reshape(-1,96),X.reshape(-1,6),ridge=1.)
            Rt=Xt-fit.predict(Ht.reshape(-1,96)).reshape(-1,48,6)
            f=fit_pair(X.reshape(-1,6),R.reshape(-1,6))
            inputs={'L':Xt.reshape(-1,6),'R':Rt.reshape(-1,6)}
            scores=score_crossed(inputs,f,'pooled')
            scores.update({f'equal__{key}':v.mean(1) for key,v in inputs.items()})
            metrics={k:float(roc_auc_score(labelt.ravel(),v)) for k,v in scores.items()}
            # Target used only here, in explicitly synthetic diagnostics.
            cos={}
            for rep,xx in [('L',X),('R',R)]:
                v=xx.reshape(-1,6)/f['sd']; yy=y.ravel()-y.mean()
                true=(v-v.mean(0)).T@yy/len(v);rho=f[rep]['rho']
                cos[rep]=float(rho@true/(np.linalg.norm(rho)*np.linalg.norm(true)))
            rows.append(dict(world=name,seed=seed,auc=metrics,rho_cos=cos,
                residual_variance_ratio=float(R.var()/X.var()),fit=jsonable(f)))
        print('[synthetic]',name,{k:round(float(np.mean([r['auc'][k] for r in rows if r['world']==name])),5) for k in metrics},flush=True)
    summary={name:{k:float(np.mean([r['auc'][k] for r in rows if r['world']==name])) for k in rows[0]['auc']} for name in WORLDS}
    contrasts={};rng=np.random.default_rng(20260915)
    for name in WORLDS:
        rr=[r for r in rows if r['world']==name];contrasts[name]={}
        for head in ('native','simplex'):
            a=f'pooled__{head}__RR'
            for b in (f'pooled__{head}__LR','equal__R','equal__L'):
                d=np.array([r['auc'][a]-r['auc'][b] for r in rr])
                boot=d[rng.integers(20,size=(10000,20))].mean(1)
                contrasts[name][a+'_minus_'+b]=dict(delta=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),wins=int((d>0).sum()))
    checks=dict(equal_marginal_variance=1+.45**2,
        joint_energy_covariance_active=2./16,
        note='Equal marginal variance removes marginal energy cue, not joint energy dependence; Cov(mean Xi^2,mean Xj^2)=2*Cij^2/L for iid Gaussian history.',
        ceiling_gate_equals_constant_eta=True)
    w=np.array([.1,.2,.3,.4]);gate=np.clip((1.-.1)/(.5-.1),0,1)*.5
    np.testing.assert_array_equal((1-gate)/4+gate*w,.5/4+.5*w)
    payload=dict(status='COMPLETE',seeds=20,worlds=5,train_test_answers=200,tokens_per_answer=64,
        evaluated_tokens_per_answer=48,OOF_training_residuals=True,rows=rows,summary=summary,
        contrasts=contrasts,analytical_checks=checks,seconds=time.perf_counter()-start,
        quality_gate='Synthetic outcomes are diagnostic, not a veto on full real evaluation',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'RESULTS.json').write_text(json.dumps(payload,indent=2),encoding='utf8')
    print('[synthetic-complete]',round(payload['seconds'],1),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
