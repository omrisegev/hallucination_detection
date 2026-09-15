"""Answer-local IU on comparable predictor innovations, without label access."""
from functools import lru_cache
from itertools import combinations
import numpy as np
from scipy.linalg import eigh
from .upcr import additive_design
from .laplacian_upcr import IU_FIT_DEFAULTS

PREDICTORS=('ridge','tcn','bocpd','noreset','mean16')
SUBSETS=tuple(s for k in (3,4,5) for s in combinations(range(5),k))
EXPECTED=dict(loss='l2',exclusion=False,difficulty_gate=False,simple_avg_fallback=False,
    recompute_after_exclusion=False,g2_projection_k=1,scale_ratio=.25,
    n_components=2,auto_components=False)

def subset_name(s):
    return '+'.join(PREDICTORS[i] for i in s)

METHODS=tuple(head+'__'+subset_name(s) for s in SUBSETS for head in ('iu','equal'))

@lru_cache(None)
def design(m):
    return additive_design(m)

def fast_iu(C):
    """Same300-point grid and two-PC scoring as canonical IU_FIT_DEFAULTS."""
    if dict(IU_FIT_DEFAULTS)!=EXPECTED:raise ValueError('IU defaults changed')
    C=np.asarray(C,float);m=len(C)
    if C.shape!=(m,m) or m<3 or not np.isfinite(C).all():raise ValueError('Invalid covariance')
    C=.5*(C+C.T);A,pairs=design(m)
    b=np.array([C[i,j] for i,j in pairs])
    rho0=np.linalg.lstsq(A,b,rcond=None)[0]
    ev,V=eigh(C,subset_by_index=[m-2,m-1]);ev=ev[::-1];V=V[:,::-1]
    var_y=.25*np.diag(C).mean();grid=np.linspace(0,var_y,300)
    rho=rho0[None,:]+.5*grid[:,None]
    proj=(rho@V[:,0])[:,None]*V[:,0][None,:]
    residual=np.linalg.norm(rho-proj,axis=1)/(np.linalg.norm(rho,axis=1)+1e-12)
    ix=int(np.argmin(residual));selected=rho[ix]
    w=sum((V[:,j]@selected)/(ev[j]+1e-12)*V[:,j] for j in range(2))
    if not np.isfinite(w).all():raise FloatingPointError('Nonfinite IU weights')
    return w,dict(g2=float(grid[ix]),g2_fraction=float(grid[ix]/(var_y+1e-12)),
        pair_residual=float(np.linalg.norm(A@rho0-b)/(np.linalg.norm(b)+1e-12)),
        spectral_residual=float(residual[ix]),second_eigenvalue=float(ev[1]),
        at_ceiling=bool(grid[ix]>=var_y*(1-1.5/300)))

def top10(values,spans):
    values=np.asarray(values,float)
    if values.ndim==1:values=values[:,None]
    out=[]
    for a,b in spans:
        if not 0<=a<b<=len(values):raise ValueError('Invalid step span')
        k=min(10,b-a);out.append(np.partition(values[a:b],b-a-k,axis=0)[-k:].mean(0))
    return np.array(out)

def correction(base,auxiliary):
    base=np.asarray(base,float);a=np.asarray(auxiliary,float)
    if a.ndim==1:a=a[:,None]
    scale=a.std(0);out=np.broadcast_to(base[:,None],a.shape).copy()
    live=scale>1e-12
    out[:,live]+=.25*base.std()*(a[:,live]-a[:,live].mean(0))/scale[live]
    return out,~live

def fuse(residuals,spans,base):
    r=np.asarray(residuals,float)
    if r.ndim!=2 or r.shape[1]!=5 or not np.isfinite(r).all():raise ValueError('Five finite residual columns required')
    scale=r.std(0)
    if np.any(scale<=1e-12):raise ValueError('Constant predictor column; fusion unavailable')
    Z=(r-r.mean(0))/scale;C=Z.T@Z/len(Z)
    W=np.zeros((5,len(METHODS)));diagnostics=[]
    for j,s in enumerate(SUBSETS):
        c=C[np.ix_(s,s)];w,diag=fast_iu(c);flipped=bool(w@c@np.ones(len(s))<0)
        if flipped:w=-w
        W[list(s),2*j]=w;W[list(s),2*j+1]=1/len(s)
        diag.update(weights=W[:,2*j].tolist(),flipped=flipped,
            negative_weights=int(np.sum(w<0)),weight_l1=float(np.abs(w).sum()))
        diagnostics.append(diag)
    scores,constant=correction(base,top10(Z@W,spans))
    single,_=correction(base,top10(r,spans))
    return scores,dict(subsets=diagnostics,residual_scale=scale.tolist(),
        correlation=C.tolist(),constant_auxiliary=constant.tolist()),single
