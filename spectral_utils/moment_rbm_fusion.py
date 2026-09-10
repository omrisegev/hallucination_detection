"""Same-answer continuous moment fusion, including an exact one-hidden-unit GRBM.

The visible variance is fixed to identity after answer-local standardization.
For E(x,h)=.5||x-a||^2-h(b+x.w), the partition function is analytic:
log Z - P/2 log(2pi) = softplus(b+a.w+.5||w||^2).
The latent posterior is a risk proxy, not a calibrated error probability.
"""
from dataclasses import asdict, replace
import hashlib
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .direct_probability_fusion import logprob_matrix, zscore_columns, _orient
from .direct_probability_fusion_v2 import selected_surprisal
from .varentropy_contribution_fusion import contributions
from .laplacian_upcr import IU_FIT_DEFAULTS
from .upcr import upcr_fit
from .deem_b3_contract_ablation import PreparedArm, GenericEnergy, fit_generic
from .residual_graph_deem import ContinuousDeemConfig

FEATURES=('entropy15','varentropy15','moment3_15','selected_surprisal','selected_squared','selected_cubed')
METHODS=('equal','iu','rbm_initial','rbm','b3_initial','b3')


def representation(logprobs,chosen):
    lp=logprob_matrix({'logprobs':logprobs},k=15)
    p=np.exp(lp);q=p/(p.sum(axis=1,keepdims=True)+1e-12)
    y=-np.log(q+1e-12)  # Frozen Top-K feature convention, including epsilons.
    h=(q*y).sum(axis=1);v=contributions(logprobs,15).sum(axis=1)
    m3=(q*y**3).sum(axis=1);a=selected_surprisal(chosen,len(lp))
    X=np.column_stack((h,v,m3,a,a**2,a**3))
    if not np.isfinite(X).all():raise ValueError('nonfinite moments')
    return X


def rbm_objective(theta,X):
    """Exact average negative log likelihood and analytic gradient, minus constant."""
    p=X.shape[1];a=theta[:p];w=theta[p:2*p];b=theta[-1]
    ell=b+X@w;post=expit(ell)
    s=b+a@w+.5*(w@w);prior=expit(s)
    loss=.5*np.mean(np.sum((X-a)**2,axis=1))-np.logaddexp(0.,ell).mean()+np.logaddexp(0.,s)
    grad=np.concatenate((a-X.mean(axis=0)+prior*w,
        prior*(a+w)-X.T@post/len(X),[prior-post.mean()]))
    return float(loss),grad


def fit_rbm(X,*,maxiter=100):
    p=X.shape[1];initial=np.r_[np.zeros(p),np.full(p,2./p),0.]
    start,_=rbm_objective(initial,X)
    result=minimize(rbm_objective,initial,args=(X,),jac=True,method='L-BFGS-B',
        options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-6,'maxls':40})
    final,grad=rbm_objective(result.x,X)
    if not np.isfinite(result.x).all() or not np.isfinite(final):raise ValueError('nonfinite RBM fit')
    if final>start+1e-8:raise ValueError('RBM optimizer increased objective')
    a,w,b=result.x[:p],result.x[p:2*p],result.x[-1]
    return expit(b+X@w),dict(a=a,w=w,b=np.asarray(b)),dict(
        optimizer='exact likelihood L-BFGS-B',converged=bool(result.success),
        message=str(result.message),iterations=int(result.nit),nll_initial=start,nll_final=final,
        gradient_max=float(np.max(np.abs(grad))),prior_hidden=float(expit(b+a@w+.5*w@w)))


def initial_b3(X,names,groups,config,seed):
    import torch
    model=GenericEnergy(names,groups,config,seed)
    with torch.no_grad():ell,_,_=model.logit(torch.as_tensor(X,dtype=torch.float64))
    return expit(ell.numpy()),model.state()


def fit_all(logprobs,chosen,uid,*,epochs=100,maxiter=100):
    X=representation(logprobs,chosen);Z,keep,mean,scale=zscore_columns(X)
    fits,failures,seconds={},{},{}
    seed=int.from_bytes(hashlib.sha256(('moment-rbm-v1:'+uid).encode()).digest()[:4],'little')%(2**31-1)
    for method in METHODS:
        started=time.perf_counter();state={}
        diag=dict(seed=seed,columns=np.flatnonzero(keep).tolist(),active_columns=int(keep.sum()),
            normalization_mean=mean.tolist(),normalization_scale=scale.tolist())
        try:
            if len(X)<3 or Z.shape[1]<3:raise ValueError('need three rows and varying columns')
            p=Z.shape[1];anchor=Z.mean(axis=1)
            if method in ('equal','iu'):
                if method=='equal':w=np.full(p,1./p)
                else:
                    f=upcr_fit(Z.T,**dict(IU_FIT_DEFAULTS))
                    if f.abstained or f.used_simple_average:raise ValueError('IU abstention/fallback')
                    w=np.asarray(f.w);diag.update(g2_hat=float(f.g2_hat),n_components=int(f.n_components_used))
                score=Z@w;state={'w':w.copy()}
            elif method=='rbm_initial':
                state=dict(a=np.zeros(p),w=np.full(p,2./p),b=np.asarray(0.))
                score=expit(Z@state['w']);diag['trained']=False
            elif method=='rbm':
                score,state,extra=fit_rbm(Z,maxiter=maxiter);diag.update(extra)
            else:
                names=tuple(np.asarray(FEATURES)[keep]);groups={'moments':tuple(range(p))}
                config=replace(ContinuousDeemConfig(),epochs=epochs)
                if method=='b3_initial':
                    score,state=initial_b3(Z,names,groups,config,seed);diag['trained']=False
                else:
                    prep=PreparedArm(Z,names,groups,frozenset(),mean,scale)
                    f=fit_generic(prep,seed=seed,config=config)
                    # Recover raw latent class before the common orientation below.
                    score=f.score if f.orientation>0 else 1-f.score
                    state=f.state;diag['health']=f.health
                diag.update(config=asdict(config),groups=groups)
            score,flipped,corr=_orient(score,anchor)
            # _orient negates scores; for posterior methods use class complement instead.
            if flipped and method not in ('equal','iu'):score=1+score
            diag.update(orientation=-1 if flipped else 1,
                anchor_correlation=float(corr) if corr is not None and np.isfinite(corr) else None,
                score_sd=float(np.std(score)),collapsed=bool(np.std(score)<1e-3))
            if not np.isfinite(score).all() or any(not np.isfinite(v).all() for v in state.values()):raise ValueError('nonfinite model')
            weights=np.full(6,np.nan)
            if method in ('equal','iu','rbm','rbm_initial'):
                weights=np.zeros(6);weights[keep]=state['w']*diag['orientation']
            fits[method]=dict(score=score,state=state,weights=weights,diagnostics=diag)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError,AssertionError) as e:
            failures[method]=f'{type(e).__name__}: {e}'
        seconds[method]=time.perf_counter()-started
    return fits,failures,seconds
