"""One answer-local, position-conditioned correction to a saved Gaussian RBM.

Shared a,b,w0 stay frozen. w(c)=w0+c*d, with c=-1 early,+1 late.
The exact conditional density is normalized separately at each context.
No correctness labels, cross-answer parameters, or near-max readout enter here.
"""
import hashlib
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

MODES=('position','shared','permuted')


def contexts(n_tokens,spans,uid):
    spans=np.asarray(spans,int);n=len(spans)
    if n<1 or np.any(spans[:,0]<0) or np.any(spans[:,1]>n_tokens) or np.any(spans[:,1]<=spans[:,0]):
        raise ValueError('invalid spans')
    if n>1 and (np.any(spans[1:,0]<spans[:-1,0]) or np.any(spans[1:,1]<spans[:-1,1])):
        raise ValueError('out-of-order steps')
    step=np.clip(np.searchsorted(spans[:,0],np.arange(n_tokens),side='right')-1,0,n-1)
    signs=np.where(np.arange(n)<(n+1)//2,-1.,1.)
    seed=int.from_bytes(hashlib.sha256(('rbm-position-v1:'+uid).encode()).digest()[:8],'little')
    shuffled=np.random.default_rng(seed).permutation(signs)
    original=signs[step];counts=[int(np.sum(original==k)) for k in (-1,1)]
    covered=np.zeros(n_tokens,int)
    for start,end in spans:covered[start:end]+=1
    return dict(position=original,shared=np.ones(n_tokens),permuted=shuffled[step]),dict(
        step_signs=signs.tolist(),permuted_step_signs=shuffled.tolist(),seed=seed,
        early_tokens=counts[0],late_tokens=counts[1],outside_scored_spans=int((covered==0).sum()),shared_boundary_tokens=int((covered>1).sum()),
        single_step=n==1)


def objective(delta,X,c,a,w0,b,ridge):
    n,p=X.shape;loss=0.;gradient=np.zeros(p)
    for sign in np.unique(c):
        rows=X[c==sign];weight=len(rows)/n;w=w0+sign*delta
        ell=b+rows@w;s=b+a@w+.5*(w@w);prior=expit(s)
        # Constant visible-quadratic term retained for independent density replay.
        value=.5*np.mean(np.sum((rows-a)**2,axis=1))-np.logaddexp(0,ell).mean()+np.logaddexp(0,s)
        loss+=weight*value
        gradient+=weight*sign*(prior*(a+w)-rows.T@expit(ell)/len(rows))
    return float(loss+.5*ridge*(delta@delta)),gradient+ridge*delta


def fit_correction(X,c,a,w0,b,ridge,orientation,*,single_step=False,maxiter=100):
    start_time=time.perf_counter();p=X.shape[1];zero=np.zeros(p)
    if orientation not in (-1,1) or not all(np.isfinite(v).all() for v in (X,c,a,w0,np.asarray(b))):raise ValueError('invalid saved model/input')
    start,_=objective(zero,X,c,a,w0,b,ridge)
    if single_step:
        delta=zero;converged=True;message='one step: structural identity, no contextual contrast';nit=0
    else:
        fit=minimize(objective,zero,args=(X,c,a,w0,b,ridge),jac=True,method='L-BFGS-B',
            options=dict(maxiter=maxiter,ftol=1e-10,gtol=1e-6,maxls=40))
        delta=fit.x;converged=bool(fit.success);message=str(fit.message);nit=int(fit.nit)
    final,grad=objective(delta,X,c,a,w0,b,ridge)
    if not np.isfinite(delta).all() or not np.isfinite(final) or final>start+1e-8:raise ValueError('invalid/worse correction objective')
    score=orientation*(b+X@w0+c*(X@delta))
    if not np.isfinite(score).all():raise ValueError('nonfinite corrected logit')
    return score,delta,dict(converged=converged,message=message,iterations=nit,ridge=float(ridge),
        objective_initial=start,objective_final=final,gradient_max=float(np.max(np.abs(grad))),
        delta_norm=float(np.linalg.norm(delta)),relative_delta_norm=float(np.linalg.norm(delta)/max(np.linalg.norm(w0),1e-12)),
        nll_final=float(final-.5*ridge*(delta@delta)),seconds=time.perf_counter()-start_time)
