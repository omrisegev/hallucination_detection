"""Weighted Dawid--Skene and exact latent-group EM; no truth labels accepted."""
from dataclasses import dataclass, field
import numpy as np
from scipy.special import expit, logsumexp, logit
from .core import orient

EPS=1e-6
def clip(x):return np.clip(x,EPS,1-EPS)

def compress(x,w):
    """Binary sufficient observations: <=2**11 patterns, exactly same objective."""
    u,inv=np.unique(x,axis=0,return_inverse=True)
    return (u>0).astype(float),np.bincount(inv,weights=w)/np.sum(w)

def expectation(b,prior,emissions,groups=None,transition=None):
    """Returns log p(v), P(Y=1|v), P(Y,a_g|v); factors are marginalized exactly."""
    ll=b@np.log(emissions)+(1-b)@np.log1p(-emissions) if groups is None else None
    joints=[]
    if groups is not None:
        ll=np.zeros((len(b),2))
        for g in range(len(transition)):
            idx=np.flatnonzero(groups==g)
            obs=b[:,idx]@np.log(emissions[idx])+(1-b[:,idx])@np.log1p(-emissions[idx])
            joint=obs[:,None,:]+np.log(np.stack([1-transition[g],transition[g]],axis=1))[None,:,:]
            marg=logsumexp(joint,axis=2);ll+=marg
            joints.append(np.exp(joint-marg[:,:,None]))
    joint_y=ll+np.log([1-prior,prior]); marginal=logsumexp(joint_y,axis=1)
    py=np.exp(joint_y-marginal[:,None])
    return marginal,py[:,1],[a*py[:,:,None] for a in joints]

def initialize(b,w,q,groups):
    q=clip(q); prior=float(clip(w@q))
    if groups is None:
        resp=np.column_stack([1-q,q]); e=clip(b.T@(w[:,None]*resp)/(w@resp))
        return prior,e,None
    k=int(groups.max())+1; trans=np.empty((k,2)); e=np.empty((b.shape[1],2))
    for g in range(k):
        idx=np.flatnonzero(groups==g); a=clip(.1+.8*b[:,idx].mean(1))
        y=np.column_stack([1-q,q]); trans[g]=clip((w*a)@y/(w@y))
        resp=np.column_stack([1-a,a]);e[idx]=clip(b[:,idx].T@(w[:,None]*resp)/(w@resp))
    return prior,e,trans

@dataclass
class EMModel:
    kind:str
    prior:float
    emissions:object
    groups:object=None
    transition:object=None
    orientation:float=1.
    status:str='ok'
    diagnostics:dict=field(default_factory=dict)
    def raw_predict(self,x):
        if not np.isin(x,[-1,1]).all():raise ValueError('EM requires binary votes')
        return expectation((np.asarray(x)>0).astype(float),self.prior,self.emissions,self.groups,self.transition)[1]
    def predict(self,x):
        p=self.raw_predict(x);return p if self.orientation>0 else 1-p

def fit_em(x,w,kind,spectral,groups=None,seed=20260922,max_iter=1000,tol=1e-8):
    if not np.isin(x,[-1,1]).all():raise ValueError('binary observations required')
    b,ww=compress(x,w); votes=2*b-1
    if kind=='ds':groups=None
    elif groups is None:raise ValueError('hierarchical EM needs fixed training groups')
    elif not np.array_equal(np.unique(groups),np.arange(len(np.unique(groups)))):
        _,groups=np.unique(groups,return_inverse=True)
    s=spectral.predict(votes)
    sd=np.sqrt(np.average((s-np.average(s,weights=ww))**2,weights=ww))
    q=clip(expit(s/max(sd,1e-8)))
    base=initialize(b,ww,q,groups)
    initial=[base,initialize(b,ww,expit(votes.mean(1)),groups)]
    rng=np.random.default_rng(seed)
    for _ in range(3):
        initial.append(tuple(None if p is None else clip(expit(logit(clip(p))+rng.normal(0,.25,np.shape(p)))) for p in base))
    models=[]; runs=[]
    for start,(prior,e,t) in enumerate(initial):
        curve=[];stable=0;converged=False
        for iteration in range(max_iter+1):
            logp,q,joints=expectation(b,prior,e,groups,t);ll=float(ww@logp);curve.append(ll)
            if len(curve)>1:
                delta=ll-curve[-2]
                if delta < -1e-9*max(1,abs(curve[-2])):raise ArithmeticError(f'EM likelihood decreased {delta}')
                stable=stable+1 if delta/max(1,abs(curve[-2]))<tol else 0
                if stable>=3:converged=True;break
            if iteration==max_iter:break
            prior=float(clip(ww@q))
            if groups is None:
                resp=np.column_stack([1-q,q]);den=ww@resp
                e=clip(b.T@(ww[:,None]*resp)/np.maximum(den,1e-300))
            else:
                for g,joint in enumerate(joints):
                    weighted=joint*ww[:,None,None]
                    t[g]=clip(weighted[:,:,1].sum(0)/np.maximum(weighted.sum((0,2)),1e-300))
                    a=joint.sum(1);idx=np.flatnonzero(groups==g)
                    e[idx]=clip(b[:,idx].T@(ww[:,None]*a)/np.maximum(ww@a,1e-300))
        models.append((float(prior),e.copy(),None if t is None else t.copy()))
        runs.append({'start':start,'log_likelihood':curve,'converged':converged,'iterations':len(curve)-1})
    best=max(range(5),key=lambda k:runs[k]['log_likelihood'][-1])
    prior,e,t=models[best]
    diag={'starts':runs,'selected_start':best,'unique_patterns':len(b),'converged':runs[best]['converged'],
          'reliability_is_model_based_not_ground_truth':True,
          'small_groups':[] if groups is None else [int(g) for g in np.unique(groups) if sum(groups==g)<3],
          'boundary_emissions':int(np.sum((e<=EPS)|(e>=1-EPS)))}
    return orient(EMModel(kind,prior,e,groups,t,diagnostics=diag),x.shape[1])
