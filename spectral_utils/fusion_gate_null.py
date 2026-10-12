"""Synthetic source controls for the existing fused-score mixture gate."""
import hashlib
import numpy as np
from .fused_trajectory_readouts import imm_filter,noise_variance
from .fusion_gate_interface_audit import inspect_mixture

READOUTS=('raw','kalman_cold','imm_cold','kalman_warm','imm_warm')
WARM=256


def stationary_ar(innovations,rho):
    z=np.asarray(innovations,float)
    if z.ndim!=1 or not len(z) or not np.isfinite(z).all() or not abs(rho)<1:raise ValueError('Invalid stationary source')
    x=np.empty_like(z);x[0]=z[0]
    for t in range(1,len(x)):x[t]=rho*x[t-1]+np.sqrt(1-rho*rho)*z[t]
    return x


def normalize(values):
    x=np.asarray(values,float);mean=float(x.mean());sd=float(x.std())
    if not np.isfinite(x).all() or sd<=1e-10:raise ValueError('Degenerate score')
    return (x-mean)/sd,dict(mean=mean,sd=sd)


def generate(n,rho,replicate,jump):
    seed=int(hashlib.sha256(f'fusion-gate-null-v1/{n}/{replicate}'.encode()).hexdigest()[:8],16)
    z=np.random.default_rng(seed).normal(size=n+WARM);path=stationary_ar(z,rho)
    if jump:path[WARM+n//2:]+=3
    tail,scale=normalize(path[-n:]);context=(path-scale['mean'])/scale['sd']
    return z,path,tail,context,dict(seed=seed,source_scale=scale)


def curves(context,n):
    tail=np.asarray(context)[-n:];r=noise_variance(tail);raw={'raw':tail};out={};scales={}
    for warm in (False,True):
        source=context if warm else tail
        for name,q in [('kalman',[.01*r]),('imm',[.01*r,r])]:
            key=name+('_warm' if warm else '_cold');raw[key]=imm_filter(source,r,q)['level'][-n:]
    for key,x in raw.items():out[key],scales[key]=normalize(x)
    return out,raw,dict(observation_variance=r,scales=scales)


def trial(n,rho,replicate,jump):
    z,path,tail,context,detail=generate(n,rho,replicate,jump);normalized,raw,model=curves(context,n)
    arrays=dict(innovations=z,path=path,tail=tail,context=context);methods={}
    for name in READOUTS:
        arrays[name]=normalized[name];arrays[name+'__unscaled']=raw[name]
        try:
            g=inspect_mixture(normalized[name],normalized[name]);methods[name]=dict(valid=True,gate_open=g['prediction']!=-1,gate=g)
        except ValueError as exc:methods[name]=dict(valid=False,reason=str(exc))
    return arrays,dict(n=n,rho=rho,replicate=replicate,jump=jump,source=detail,model=model,methods=methods)
