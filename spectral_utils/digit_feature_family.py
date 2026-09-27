"""Three decoding-independent numeric streams; labels are not accepted."""
import numpy as np
from .digit_alternative_probability import second_digit_probability,prefix_innovation

NAMES=('digit_alternative','digit_spread','digit_alternative_innovation')


def token_family(ids, logprobs, digit_ids):
    alt,upper,seen=second_digit_probability(ids,logprobs,digit_ids)
    lp=np.asarray(logprobs,float);mask=np.isin(ids,digit_ids)
    p=np.where(mask,np.exp(lp),0.);mass=p.sum(axis=1)
    # M*H(p/M) = -sum(p log p) + M log M, avoiding division at zero.
    mlm=np.zeros(len(mass));positive=mass>0;mlm[positive]=mass[positive]*np.log(mass[positive])
    spread=np.maximum(0.,(-(p*lp).sum(axis=1)+mlm)/np.log(10.))
    values=np.column_stack((alt,spread,prefix_innovation(alt)))
    active=np.ones(values.shape,bool)
    if len(values):active[0,2]=False
    return values,active,{'censored':seen<2,'digit_mass':mass,'alternative_upper':upper}


def step_family(values,active,spans):
    out=np.zeros((len(spans),3));available=np.zeros(out.shape,bool)
    for i,(a,b) in enumerate(spans):
        if not 0<=a<b<=len(values):raise ValueError('Invalid step span')
        for j in range(3):
            x=values[a:b,j][active[a:b,j]]
            if len(x):
                k=min(2,len(x));out[i,j]=np.partition(x,-k)[-k:].mean();available[i,j]=True
    return out,available


def answer_standardize(x,offsets,active=None):
    x=np.asarray(x,float);vector=x.ndim==1
    if vector:x=x[:,None]
    if not np.isfinite(x).all():raise ValueError('Nonfinite observations')
    if active is None:active=np.ones(x.shape,bool)
    out=np.zeros(x.shape)
    for a,b in zip(offsets[:-1],offsets[1:]):
        for j in range(x.shape[1]):
            valid=active[a:b,j];v=x[a:b,j][valid]
            if len(v) and v.std()>1e-12:
                out[a:b,j][valid]=(v-v.mean())/v.std()
    return out[:,0] if vector else out
