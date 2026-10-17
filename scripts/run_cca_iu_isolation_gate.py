"""Run the predeclared synthetic gate; original experiments remain immutable."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dsp_contextual_iu_synthetic as old
from spectral_utils.cca_iu_isolation import (CCAContext, covariance, regularize,
    fusion_weights, local_moments, choose_alpha)
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.contextual_iu import DEFAULT_IU_FIT

OUT = ROOT/'results/cca_iu_isolation_gate_v1'
WORLDS = ('informative','null','coherent_nuisance')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf8')
    tmp.replace(path)


def provenance():
    paths = ['docs/experiments/CCA_IU_ISOLATION_GATE_20260915.md',
             'scripts/run_cca_iu_isolation_gate.py','spectral_utils/cca_iu_isolation.py',
             'scripts/dsp_contextual_iu_synthetic.py','spectral_utils/contextual_iu.py',
             'spectral_utils/upcr.py',
             'results/dsp_contextual_iu_pilot_v1/STAGE_0_SYNTHETIC_PER_SEED.csv',
             'results/dsp_contextual_iu_pilot_v1/STAGE_0_DECISION.json']
    return {p:sha(ROOT/p) for p in paths}


def population(world, active):
    """Raw population covariance and Cov(X,target); no generated test labels."""
    a = np.asarray(active, float); m = len(a)
    if world == 'null': return np.ones((m,m))+.75**2*np.eye(m), np.ones(m)
    if world == 'informative':
        return np.outer(a,a)+np.diag(a*.45**2+(1-a)*1.5**2), a
    return np.outer(a,a)+1.6**2*np.outer(1-a,1-a)+np.diag(a*.7**2+(1-a)*.18**2), a


def history(world, active, seed):
    rng = np.random.default_rng(seed); n,m = active.shape
    a = active[:,None,:]; shape = (n,16,m)
    target = rng.normal(size=(n,16,1))
    if world == 'null': return target+rng.normal(scale=.75,size=shape)
    if world == 'informative':
        return np.where(a,target+rng.normal(scale=.45,size=shape),rng.normal(scale=1.5,size=shape))
    nuisance = rng.normal(size=(n,16,1))
    return np.where(a,target+rng.normal(scale=.7,size=shape),1.6*nuisance+rng.normal(scale=.18,size=shape))


def replay_legacy():
    reference = {(r['world'],int(r['seed'])):r for r in csv.DictReader(
        (old.OUT/'STAGE_0_SYNTHETIC_PER_SEED.csv').open(encoding='utf8'))}
    maximum = 0.
    for world in WORLDS:
        for seed in old.SEEDS:
            path = OUT/'legacy'/f'{world}_{seed}.json'
            row = json.loads(path.read_text()) if path.exists() else old._score_world(seed,world)
            for k,v in row.items():
                if isinstance(v,(float,int)) and k!='seed':
                    delta = abs(v-float(reference[(world,seed)][k])); maximum=max(maximum,delta)
                    if delta>1e-10: raise ValueError(f'Legacy mismatch {world} {seed} {k}: {delta}')
            save(path,row)
            print('legacy',world,seed,'max_delta',maximum,flush=True)
    mechanics = old._mechanical_checks()
    expected = json.loads((old.OUT/'STAGE_0_DECISION.json').read_text())['mechanics']
    for k,v in mechanics.items():
        if abs(float(v)-float(expected[k]))>1e-10: raise ValueError(k)
    save(OUT/'LEGACY_REPLAY.json',{'status':'PASS','rows':60,'max_abs_delta':maximum,
                                 'mechanics':mechanics,'historical_verdict':'STOP_NO_ROUTING_SIGNAL'})


def coordinates(mode, train, Htrain, test, Htest):
    if mode in ('linear','history_square_only','second_moment'):
        fit = CCAContext().fit(Htrain,train[0],mode)
        return fit.transform(Htrain),fit.transform(Htest),fit.diagnostics(Htest,test[0])
    if mode == 'oracle':
        return train[7][:,:1].astype(float),test[7][:,:1].astype(float),{}
    left,right = ((train[3],test[3]) if mode in ('dsp','random') else
                  ((Htrain**2).mean(axis=1),(Htest**2).mean(axis=1)))
    sd=left.std(axis=0); sd[sd<1e-8]=1
    return (left-left.mean(axis=0))/sd,(right-left.mean(axis=0))/sd,{}


def one_case(world, seed):
    train=old._world(seed,world,320); test=old._world(seed+100000,world,640)
    X=train[0]; Xt=test[0]; scale=X.std(axis=0); m=X.shape[1]
    Z=(X-X.mean(axis=0))/scale
    Cg=regularize(covariance(Z)); var_y=.25*np.trace(Cg)/m
    scores={}; weights={}; telemetry={}
    def register(name,C,kind='full',rho=None):
        w,t=fusion_weights(C,scale,var_y,kind,rho)
        weights[name]=np.broadcast_to(w,(len(Xt),m)).copy()
        scores[name]=np.sum(weights[name]*Xt,axis=1)
        telemetry[name]={'g2_ceiling_fraction':float(np.mean(t['g2']>=var_y*(1-1.5/300))),
                         'additive_residual_mean':float(np.mean(t['residual'])),
                         'rho_mean':t['rho'].mean(axis=0).tolist(),
                         'a_mean':t['a'].mean(axis=0).tolist()}
    scores['equal']=Xt.mean(axis=1)
    for kind in ('full','group','minvar','vertex'): register('static_'+kind,Cg,kind)
    covs=[]; rhos=[]
    for active in (np.r_[np.ones(3),np.zeros(3)],np.r_[np.zeros(3),np.ones(3)]):
        c,r=population(world,active);covs.append(c/scale[:,None]/scale[None,:]);rhos.append(r/scale/scale.mean())
    popglobal=np.mean(covs,axis=0); idx=(~test[7][:,0]).astype(int)
    pc=np.array(covs)[idx]; pr=np.array(rhos)[idx]
    for kind in ('full','group','minvar','vertex'):
        register('population_static_'+kind,popglobal,kind)
        register('population_context_'+kind,pc,kind)
    register('oracle_rho_context',pc,rho=pr)
    register('oracle_rho_static',popglobal,rho=np.mean(rhos,axis=0))
    kwargs=dict(DEFAULT_IU_FIT);kwargs['var_y']=var_y
    native_global=upcr_fit_covariance(popglobal,**kwargs).w
    native_local=np.array([upcr_fit_covariance(c,**kwargs).w for c in covs])
    native_local*=np.sum(abs(native_global))/np.sum(abs(native_local),axis=1)[:,None]
    scores['population_native_static']=((Xt-X.mean(axis=0))/scale)@native_global
    scores['population_native_context']=np.sum(((Xt-X.mean(axis=0))/scale)*native_local[idx],axis=1)
    H=history(world,train[7],seed+200000); Ht=history(world,test[7],seed+300000)
    inner=tuple(v[:240] for v in train); val=tuple(v[240:] for v in train)
    for mode in ('oracle','dsp','random','linear','history_square_only','second_moment','energy'):
        iz,vz,_=coordinates(mode,inner,H[:240],val,H[240:])
        inner_scale=X[:240].std(axis=0); inner_mean=X[:240].mean(axis=0)
        inner_x=(X[:240]-inner_mean)/inner_scale;val_x=(X[240:]-inner_mean)/inner_scale
        random_seed=seed+400000 if mode=='random' else None
        alpha,losses=choose_alpha(inner_x,iz,val_x,vz,random_seed=random_seed)
        tz,qz,diag=coordinates(mode,train,H,test,Ht)
        _,local,neff=local_moments(Z,tz,qz,random_seed=random_seed)
        C=(np.broadcast_to(Cg,local.shape).copy() if alpha==1 else
           regularize((1-alpha)*local+alpha*Cg))
        if np.min(neff)<32: raise ValueError('Insufficient effective groups')
        for kind in ('full','group'):
            name=mode+'_'+kind;register(name,C,kind)
            telemetry[name].update(alpha=alpha,validation_nll=losses,
                                   neff_min=float(neff.min()),neff_mean=float(neff.mean()),**diag)
    # No fitting routine below this boundary. Labels used solely for evaluation.
    rows={name:float(roc_auc_score(test[4],score)) for name,score in scores.items()}
    payload={'world':world,'seed':seed,'auc':rows,'telemetry':telemetry,
             'input_sha256':hashlib.sha256(X.tobytes()+Xt.tobytes()+H.tobytes()+Ht.tobytes()).hexdigest()}
    np.savez_compressed(OUT/'cases'/f'{world}_{seed}_arrays.npz',
        **{'score__'+k:v for k,v in scores.items()},**{'weight__'+k:v for k,v in weights.items()},
        labels=test[4],scale=scale)
    return payload


def summarize():
    rows=[json.loads((OUT/'cases'/f'{w}_{s}.json').read_text()) for w in WORLDS for s in old.SEEDS]
    summary=[]; rng=np.random.default_rng(190915)
    samples=rng.integers(0,20,size=(10000,20))
    for world in WORLDS:
        subset=[r for r in rows if r['world']==world]
        for name in subset[0]['auc']:
            baseline=('population_native_static' if name=='population_native_context' else
                      'oracle_rho_static' if name=='oracle_rho_context' else
                      'population_static_'+name.rsplit('_',1)[-1] if name.startswith('population_context_') else
                      'static_group' if name.endswith('_group') else 'static_full')
            values=np.array([r['auc'][name] for r in subset]); base=np.array([r['auc'][baseline] for r in subset])
            d=values-base; ci=np.quantile(d[samples].mean(axis=1),[.025,.975])
            summary.append(dict(world=world,method=name,baseline=baseline,auc=float(values.mean()),
                delta=float(d.mean()),ci95=ci.tolist(),wins=int(np.sum(d>0)),worst=float(d.min())))
    def gate(method):
        s={r['world']:r for r in summary if r['method']==method}
        return dict(informative_gain=s['informative']['delta']>=.005,
                    informative_wins=s['informative']['wins']>=18,
                    null_safety=abs(s['null']['delta'])<=.005,
                    nuisance_mean=s['coherent_nuisance']['delta']>=-.005,
                    nuisance_tail=s['coherent_nuisance']['worst']>=-.020)
    gates={k:gate(k) for k in ('population_context_full','oracle_full','dsp_full','linear_full',
                             'history_square_only_full','second_moment_full','energy_full')}
    save(OUT/'SUMMARY.json',{'rows':summary,'gates':gates,'scope':'synthetic diagnostic only'})
    with (OUT/'SUMMARY.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['legacy','new'],required=True)
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True);(OUT/'cases').mkdir(exist_ok=True)
    manifest=OUT/'PROVENANCE.json'; current=provenance()
    if manifest.exists() and json.loads(manifest.read_text())!=current: raise ValueError('Resume hash mismatch')
    save(manifest,current); started=time.time()
    with threadpool_limits(limits=1):
        if args.phase=='legacy': replay_legacy()
        else:
            for world in WORLDS:
                for seed in old.SEEDS:
                    path=OUT/'cases'/f'{world}_{seed}.json'
                    if not path.exists(): save(path,one_case(world,seed))
                    save(OUT/'RUN_STATE.json',dict(state='RUNNING',world=world,seed=seed,elapsed=time.time()-started))
                    print(world,seed,'elapsed',round(time.time()-started,1),flush=True)
            summarize()
    save(OUT/(args.phase.upper()+'_STATE.json'),dict(state='COMPLETE',elapsed=time.time()-started))


if __name__=='__main__': main()
