"""Frozen cumulative-vote experiment: profiles, weighted models and readouts.

No task labels enter fit(). Covariance uses normalized answer/threshold weights.
The continuous hierarchy is an empirical extension, not binary SML theory.
"""
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_laplace
from scipy.special import softmax

_spec = importlib.util.spec_from_file_location("cvf_canonical_fusion", Path(__file__).resolve().parents[3] / "spectral_utils/fusion_utils.py")
fu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fu)
CHANNELS = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level', 'energy_innovation', 'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0']
READOUTS = ['top5', 'top10', 'max', 'mean', 'log_top5', 'cusum_top5', 'onset80']
ARMS = [('hard','equal'), ('hard','spectral'), ('hard','binary_lsml'),
        ('soft','equal'), ('soft','spectral'), ('soft','continuous_lsml'),
        ('hard','continuous_lsml'), ('hard','ds'), ('hard','hem')]

def standardize(x, weights=None):
    x = np.asarray(x, float)
    mu = np.average(x, axis=0, weights=weights)
    sd = np.sqrt(np.average((x-mu)**2, axis=0, weights=weights))
    sd = np.where(sd > 1e-8, sd, 1.)
    return (x-mu)/sd, mu, sd

def profiles(tokens, spans):
    """S x 11 x 7; onset excludes suffix via -inf, never perturbs maxima."""
    x = np.asarray(tokens, float)
    med = np.median(x, axis=0)
    lo, hi = np.percentile(x, [25,75], axis=0)
    scale = (hi-lo)/1.349
    scale = np.where(scale>1e-8, scale, np.where(x.std(0)>1e-8,x.std(0),1.))
    x = (x-med)/scale
    log = x
    # Apply the derivative along time only, never across feature columns.
    if len(x)>3:
        log = np.column_stack([-gaussian_laplace(x[:,j],2.5) for j in range(x.shape[1])])
    cusum = np.abs(np.cumsum(x-x.mean(0),axis=0))
    out = np.empty((len(spans), x.shape[1], len(READOUTS)))
    def top(a,k):
        k = min(k,len(a)); return np.partition(a,len(a)-k,axis=0)[-k:].mean(0)
    for s,(a,b) in enumerate(spans):
        if not 0<=a<b<=len(x): raise ValueError(f'invalid span {(a,b)} / {len(x)}')
        v=x[a:b]
        out[s,:,:6]=np.stack([top(v,5),top(v,10),v.max(0),v.mean(0),top(log[a:b],5),top(cusum[a:b],5)],axis=1)
    out[:,:,6] = -np.inf
    for j in range(x.shape[1]):
        v=out[:,j,0]; threshold=v.min()+.8*(v.max()-v.min())
        hit=np.flatnonzero(v>=threshold)
        if not len(hit): raise AssertionError('relative onset threshold has no crossing')
        out[:hit[0]+1,j,6]=v[:hit[0]+1]
    return out

def encode(profile, encoding, task):
    p=np.asarray(profile,float); S,m=p.shape
    if task=='pb':
        if encoding=='hard': return 2*(np.argmax(p,axis=0)[None,:]<=np.arange(S-1)[:,None])-1.
        z=np.full_like(p,-np.inf)
        for j in range(m):
            finite=np.isfinite(p[:,j])
            if not finite.any(): raise ValueError('empty profile')
            z[finite,j]=standardize(p[finite,j])[0]
        return 2*np.cumsum(softmax(z,axis=0),axis=0)[:-1]-1
    if not np.isfinite(p).all(): raise ValueError('PRM cannot use onset profile')
    return 2*(p>np.median(p,axis=0))-1. if encoding=='hard' else standardize(p)[0]

def training_matrix(profiles_by_answer, indices, cells, encoding, task):
    usable=[int(i) for i in indices if task!='pb' or len(profiles_by_answer[i])>1]
    if not usable: raise ValueError('no trainable answers')
    counts={c:sum(cells[i]==c for i in usable) for c in set(cells[i] for i in usable)}
    xs=[]; ws=[]
    for i in usable:
        x=encode(profiles_by_answer[i],encoding,task)
        mass=1/(len(counts)*counts[cells[i]]) if task=='pb' else 1/len(usable)
        xs.append(x); ws.append(np.full(len(x),mass/len(x)))
    return np.concatenate(xs),np.concatenate(ws)

def covariance(x,w):
    w=np.asarray(w,float)/np.sum(w); z=x-np.average(x,axis=0,weights=w)
    return (z*w[:,None]).T@z

def spectral_weights(x,w):
    m=x.shape[1]
    if m==1:return np.ones(1)
    r=covariance(x,w); np.fill_diagonal(r,0)
    vals,vec=np.linalg.eigh(r); v=vec[:,-1]
    if np.linalg.norm(r)<1e-14: return np.ones(m)/np.sqrt(m)
    # Deterministic orientation of intermediate factors, without clipping negatives.
    if v.sum()<0 or (abs(v.sum())<1e-14 and v[np.argmax(abs(v))]<0): v=-v
    return v

def discover_groups(x,w):
    r=covariance(x,w); sim=fu._score_matrix_lsml(r); curve=[]
    for k in range(2,min(x.shape[1],9)):
        c=fu._spectral_cluster_precomputed(sim,k,seed=42)
        residual=float(fu._residual_lsml(r,c,loading_scale='unit'))
        curve.append((residual,k,c))
    if not curve: return np.arange(x.shape[1]),[]
    best=min(curve,key=lambda a:(a[0],a[1]))
    return best[2],[{'k':k,'residual':r,'groups':c.tolist()} for r,k,c in curve]

@dataclass
class FusionModel:
    kind: str
    mean: object=None
    scale: object=None
    weights: object=None
    groups: object=None
    inner: list=field(default_factory=list)
    cross: object=None
    orientation: float=1.
    status: str='ok'
    diagnostics: dict=field(default_factory=dict)

    def raw_predict(self,x):
        x=np.asarray(x,float)
        if self.kind=='equal': return x.mean(1)
        z=(x-self.mean)/self.scale
        if self.kind=='spectral':return z@self.weights
        virtual=[]
        for idx,w in self.inner:
            v=z[:,idx]@w
            if self.kind=='binary_lsml':v=np.where(v>=0,1.,-1.)
            virtual.append(v)
        return np.column_stack(virtual)@self.cross

    def predict(self,x): return self.orientation*self.raw_predict(x)

def orient(model,m):
    endpoints=model.raw_predict(np.stack([-np.ones(m),np.ones(m)]))
    delta=endpoints[1]-endpoints[0]
    model.diagnostics['endpoint_range']=float(abs(delta))
    if not np.isfinite(delta) or abs(delta)<=1e-12:model.status='invalid_endpoint_range'
    else:model.orientation=1. if delta>0 else -1.
    return model

def fit_spectral(x,w,kind):
    x=np.asarray(x,float); w=np.asarray(w,float)
    if not np.isfinite(x).all() or np.any(w<=0):raise ValueError('invalid training data')
    model=FusionModel(kind)
    if kind=='equal':return orient(model,x.shape[1])
    if kind=='binary_lsml':
        if not np.isin(x,[-1,1]).all():raise ValueError('binary L-SML requires raw votes')
        z=x; model.mean=np.zeros(x.shape[1]); model.scale=np.ones(x.shape[1])
    else:z,model.mean,model.scale=standardize(x,w)
    if kind=='spectral':model.weights=spectral_weights(z,w)
    else:
        model.groups,curve=discover_groups(z,w)
        model.diagnostics['group_search']=curve
        vs=[]
        for g in np.unique(model.groups):
            idx=np.flatnonzero(model.groups==g); weights=spectral_weights(z[:,idx],w)
            model.inner.append((idx,weights)); v=z[:,idx]@weights
            vs.append(np.where(v>=0,1.,-1.) if kind=='binary_lsml' else v)
        model.cross=spectral_weights(np.column_stack(vs),w)
        model.diagnostics['small_groups']=[int(g) for g in np.unique(model.groups) if sum(model.groups==g)<3]
    return orient(model,x.shape[1])

def pava(y):
    means=[]; counts=[]
    for v in np.asarray(y,float):
        means.append(float(v)); counts.append(1)
        while len(means)>1 and means[-2]>means[-1]:
            n=counts[-2]+counts[-1]
            means[-2:]=[(means[-2]*counts[-2]+means[-1]*counts[-1])/n]
            counts[-2:]=[n]
    return np.repeat(means,counts)

def location(model,profile,encoding):
    m=profile.shape[1]; S=len(profile)
    if S==1:return np.ones(1),False
    x=np.vstack([-np.ones(m),encode(profile,encoding,'pb'),np.ones(m)])
    scores=model.predict(x); corrected=pava(scores)
    mass=np.maximum(np.diff(corrected),0)
    fail=model.status!='ok' or not np.isfinite(mass).all() or mass.sum()<=1e-12
    if fail:
        mass=np.diff(pava(x.mean(1)))
    return mass/mass.sum(),fail
