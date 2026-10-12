"""One fitted-AR Monte Carlo gate with a known-rho diagnostic reference."""
import hashlib
import numpy as np
from .fusion_gate_null import stationary_ar,normalize
from .fused_trajectory_readouts import imm_filter,noise_variance
from .fusion_gate_interface_audit import inspect_mixture

B=39
ALPHA=.05
READOUTS=('raw','imm')


def seed(role,identity):return int(hashlib.sha256(('fusion-gate-calibration-v1/'+role+'/'+identity).encode()).hexdigest()[:32],16)


def fit_rho(values):
    x=np.asarray(values,float)
    if x.ndim!=1 or len(x)<3 or not np.isfinite(x).all():raise ValueError('Invalid source')
    left=x[:-1]-x[:-1].mean();right=x[1:]-x[1:].mean();den=float(left@left)
    if den<=1e-12:raise ValueError('Degenerate lagged source')
    raw=float(left@right/den);value=float(np.clip(raw,-.95,.95))
    return dict(raw=raw,value=value,clipped=value!=raw)


def process(values):
    source,source_scale=normalize(values);r=noise_variance(source)
    level=imm_filter(source,r,[.01*r,r])['level'];curves={};methods={};scales={}
    for name,x in [('raw',source),('imm',level)]:
        curve,scale=normalize(x);curves[name]=curve;scales[name]=scale
        try:
            g=inspect_mixture(curve,curve);methods[name]=dict(valid=True,gate=g,native_open=g['prediction']!=-1)
        except ValueError as exc:methods[name]=dict(valid=False,reason=str(exc))
    return curves,dict(source_scale=source_scale,R=r,scales=scales,methods=methods)


def calibrated(observed,nulls):
    if not observed['valid'] or len(nulls)!=B or not all(x['valid'] for x in nulls):
        return dict(valid=False,reason='REQUIRED_GATE_FIT_UNAVAILABLE')
    statistic=observed['gate']['bic_gain'];values=np.array([x['gate']['bic_gain'] for x in nulls])
    tol=100*np.finfo(float).eps*abs(statistic);ge=int(np.sum(values>=statistic-tol));p=(1+ge)/(len(values)+1)
    # Finite nonconstant input and fitted convex Gaussian means imply that a
    # candidate window exceeds their mean; verify the actual condition upstream.
    return dict(valid=True,pvalue=p,exceedances=ge,tolerance=tol,open=p<=ALPHA)


def trial(n,rho,jump,replicate):
    identity=f'n{n}_rho{int(rho*100):02d}_jump{int(jump)}_rep{replicate:02d}'
    es=seed('evaluation',f'{n}/{replicate}');cs=seed('calibration',identity)
    innovation=np.random.default_rng(es).normal(size=n);source=stationary_ar(innovation,rho)
    if jump:source[n//2:]+=3
    estimate=fit_rho(source);observed,ometa=process(source)
    null_innovations=np.random.default_rng(cs).normal(size=(B,n));records={};arrays=dict(observed_source=source,observed_innovations=innovation,null_innovations=null_innovations)
    for name in READOUTS:arrays['observed_'+name]=observed[name]
    for model,param in [('fitted',estimate['value']),('known',rho)]:
        metas=[];collected={name:[] for name in READOUTS}
        for z in null_innovations:
            null=stationary_ar(z,param);curves,meta=process(null);metas.append(meta)
            for name in READOUTS:collected[name].append(curves[name])
        records[model]=metas
        for name in READOUTS:arrays[model+'_'+name]=np.asarray(collected[name])
    decisions={}
    for name in READOUTS:
        obs=ometa['methods'][name];decisions[name+'__native']=dict(valid=obs['valid'],open=obs.get('native_open'))
        for model in ('fitted','known'):
            decision=calibrated(obs,[rec['methods'][name] for rec in records[model]])
            if decision['valid']:
                eligible=bool(np.any(observed[name]>np.mean(obs['gate']['means'])))
                decision['eligible']=eligible;decision['open']&=eligible
            decisions[name+'__'+model]=decision
    return arrays,dict(uid=identity,n=n,rho=rho,jump=jump,replicate=replicate,evaluation_seed=es,calibration_seed=cs,
        rho_estimate=estimate,observed=ometa,nulls=records,decisions=decisions)
