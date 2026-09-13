"""Unlabelled Gaussian factor fusion with low-rank position/feature loadings.

Rank refers to the 16 x P loading map, NOT the number of latent classes.
Each token has one Gaussian latent factor. No semantic labels enter this module.
"""
import hashlib
import numpy as np
from scipy.optimize import minimize

BINS = 16
RIDGE = 1e-4

def permutation(uid, step, length):
    seed = int.from_bytes(hashlib.sha256(f'{uid}:two-axis:{step}'.encode()).digest()[:8], 'little')
    return np.random.default_rng(seed).permutation(length)

def overlap(length, bins=BINS):
    if length < 1:
        raise ValueError('empty step')
    t = np.arange(length, dtype=float)[:, None]
    edges = np.linspace(0, length, bins + 1)
    return np.maximum(0., np.minimum(t + 1, edges[None, 1:]) - np.maximum(t, edges[None, :-1]))

def sufficient_statistics(x, spans, uid):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or not len(spans):
        raise ValueError('invalid feature data')
    p=x.shape[1]; real=np.zeros((BINS,p,p)); shuffled=real.copy()
    for j,(start,end) in enumerate(spans):
        v=x[int(start):int(end)]
        o=overlap(len(v)); o *= BINS/len(v)
        vs=v[permutation(uid,j,len(v))]
        real += np.einsum('ti,tj,tk->ijk', o, v, v, optimize=True)
        shuffled += np.einsum('ti,tj,tk->ijk', o, vs, vs, optimize=True)
    return real/len(spans), shuffled/len(spans)

def unpack(theta, bins, p, rank):
    n=bins*rank; k=p*rank
    return theta[:n].reshape(bins,rank), theta[n:n+k].reshape(p,rank), theta[n+k:]

def objective(theta, second, rank):
    j,p,_=second.shape
    u,v,logd=unpack(theta,j,p,rank)
    w=u@v.T; invd=np.exp(-logd); q=w*invd; den=1+np.sum(w*q,axis=1)
    inverse=np.broadcast_to(np.diag(invd),(j,p,p))-np.einsum('ji,jk->jik',q,q)/den[:,None,None]
    val=.5*np.mean(logd.sum()+np.log(den)+np.einsum('jik,jki->j',inverse,second))
    g=.5*(inverse-inverse@second@inverse)/j
    gw=2*np.einsum('jik,jk->ji',g,w)
    val += .5*RIDGE*np.mean(w*w)
    gw += RIDGE*w/w.size
    gd=np.einsum('jii->i',g)*np.exp(logd)
    return float(val),np.r_[(gw@v).ravel(),(gw.T@u).ravel(),gd]

def objective_parts(theta, second, rank):
    """Return the pure Gaussian NLL and the fixed optimization penalty."""
    j,p,_=second.shape
    u,v,_=unpack(theta,j,p,rank);w=u@v.T
    regularized=float(objective(theta,second,rank)[0])
    penalty=float(.5*RIDGE*np.mean(w*w))
    return regularized-penalty,penalty

def factorize(w,rank):
    a,s,b=np.linalg.svd(w,full_matrices=False)
    s=np.sqrt(np.maximum(s[:rank],1e-12))
    return a[:,:rank]*s,b[:rank].T*s

def fit(second, rank, stationary=False):
    c=np.asarray(second,dtype=np.float64)
    if c.ndim!=3 or c.shape[0]!=BINS or c.shape[1]!=c.shape[2] or not np.isfinite(c).all():
        raise ValueError('invalid position second moments')
    if rank not in (1,2) or (stationary and rank!=1):raise ValueError('unregistered rank')
    c=(c+c.transpose(0,2,1))/2
    if np.linalg.eigvalsh(c).min() < -1e-8*max(1.,np.trace(c,axis1=1,axis2=2).max()):
        raise ValueError('non-PSD second moment')
    if stationary:c=c.mean(axis=0,keepdims=True)
    j,p,_=c.shape; average=float(np.trace(c,axis1=1,axis2=2).mean()/p)
    floor=max(1e-8,1e-3*average)
    pooled=c.mean(axis=0); ev,vec=np.linalg.eigh(pooled)
    initial=vec[:,-1]*np.sqrt(max(ev[-1]-ev[:-1].mean(),.01*max(average,1e-8)))
    if initial[1]<0:initial=-initial
    constant=np.tile(initial,(j,1))
    if rank==2:
        constant += .05*np.outer(np.linspace(-1,1,j),vec[:,-2])*np.sqrt(max(average,1e-8))
    regional=[]
    for s in c:
        e,b=np.linalg.eigh(s); w=b[:,-1]*np.sqrt(max(e[-1]-e[:-1].mean(),.01*max(average,1e-8)))
        if w[1]<0:w=-w
        regional.append(w)
    fits=[];start_info=[]
    for start_index,(start_name,initial_w) in enumerate((('pooled',constant),('regional',np.asarray(regional)))):
        u,v=factorize(initial_w,rank); d=np.maximum(np.diag(pooled)-np.mean((u@v.T)**2,axis=0),floor)
        theta=np.r_[u.ravel(),v.ravel(),np.log(d)]
        try:
            res=minimize(objective,theta,args=(c,rank),jac=True,method='L-BFGS-B',
                         bounds=[(None,None)]*(len(theta)-p)+[(np.log(floor),None)]*p,
                         options=dict(maxiter=1000,ftol=1e-10,gtol=1e-6))
        except Exception as exc:
            start_info.append(dict(start_index=start_index,start_name=start_name,success=False,
                                   failure=type(exc).__name__+': '+str(exc)))
            continue
        if not np.isfinite(res.fun) or not np.isfinite(res.x).all():
            start_info.append(dict(start_index=start_index,start_name=start_name,success=False,
                                   failure='nonfinite optimizer result',message=str(res.message)))
            continue
        u,v,logd=unpack(res.x,j,p,rank); w=u@v.T
        # Fix latent sign by positive loading on the registered varentropy15 view.
        sign=np.where(w[:,1]<0,-1.,1.)
        tie=np.abs(w[:,1])<=1e-12
        for k in np.flatnonzero(tie):
            nz=np.flatnonzero(np.abs(w[k])>1e-12)
            sign[k]=np.sign(w[k,nz[0]]) if len(nz) else 1.
        w=w*sign[:,None]; d=np.exp(logd)
        coef=(w/d)/(1+np.sum(w*w/d,axis=1))[:,None]
        sigma=np.einsum('ji,jk->jik',w,w)+np.diag(d)
        bounds=np.r_[np.full(len(res.x)-p,-np.inf),np.full(p,np.log(floor))]
        grad=res.jac.copy();grad[(res.x<=bounds+1e-9)&(grad>0)]=0
        nll,penalty=objective_parts(res.x,c,rank)
        info=dict(start_index=start_index,start_name=start_name,success=True,
                                   converged=bool(res.success),iterations=int(res.nit),message=str(res.message),
                                   nll=float(nll),penalty=float(penalty),regularized_objective=float(res.fun),
                                   objective=float(res.fun),
                                   projected_gradient_max=float(np.abs(grad).max()),
                                   residual_relative=float(np.linalg.norm(c-sigma)/max(np.linalg.norm(c),1e-30)),
                                   condition_max=float(np.linalg.cond(sigma).max()),anchor_ties=int(tie.sum()))
        start_info.append(info)
        fits.append(dict(result=res,loadings=w,noise=d,coefficients=coef,info=info))
    if not fits:raise FloatingPointError('both factor fits nonfinite')
    chosen=min(range(len(fits)),key=lambda i:fits[i]['info']['nll']); best=fits[chosen]
    arrays={k:best[k] for k in ('loadings','noise','coefficients')}
    if stationary:
        for k in ('loadings','coefficients'):arrays[k]=np.repeat(arrays[k],BINS,axis=0)
    info=dict(rank=rank,stationary=stationary,noise_floor=floor,
              selected_start=best['info']['start_index'],selected_start_name=best['info']['start_name'],
              selected_by='pure_gaussian_nll',starts=start_info,**best['info'])
    info['coefficient_singular_values']=np.linalg.svd(arrays['coefficients'],compute_uv=False).tolist()
    info['coefficient_start_l1_difference']=(float(np.abs(fits[0]['coefficients']-fits[1]['coefficients']).sum())
                                             if len(fits)==2 else None)
    return arrays,info

def step_scores(x, spans, coefficients, uid, shuffled=False):
    if coefficients.shape!=(BINS,x.shape[1]) or not np.isfinite(coefficients).all():
        return np.full(len(spans),np.nan)
    result=[]
    for j,(start,end) in enumerate(spans):
        v=x[int(start):int(end)]
        if shuffled:v=v[permutation(uid,j,len(v))]
        token_weights=overlap(len(v))@coefficients
        token=np.sum(v*token_weights,axis=1)
        n=min(10,len(token));result.append(np.partition(token,len(token)-n)[-n:].mean())
    return np.asarray(result)
