"""Exact one-hidden-unit Gaussian RBM with learned diagonal conditional variance."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .moment_rbm_fusion import representation, rbm_objective, fit_rbm
from .direct_probability_fusion import zscore_columns
from .rbm_m3_powers import orient_posterior

VARIANCE_FLOOR = 0.05  # 5% of each feature's standardized total variance.
VARIANCE_PENALTY = 0.1  # mean NLL + strength * sum(log conditional variance)^2.
METHODS = ('rbm', 'rbm_initial', 'rbm_continued', 'rbm_diagonal')


def diagonal_objective(theta, X, strength=VARIANCE_PENALTY):
    """E=.5 sum(log D)+.5(x-a)'D^-1(x-a)-h(b+x'D^-1 w).

    The log-D term is represented once in the normalized likelihood below.
    w is the shift between hidden-state means, not the posterior coefficient.
    D=1 exactly reduces the first 2P+1 objective derivatives to rbm_objective.
    """
    p=X.shape[1];a=theta[:p];w=theta[p:2*p];b=theta[2*p];logs=theta[2*p+1:]
    inv=np.exp(-logs); beta=inv*w
    ell=b+X@beta;post=expit(ell)
    s=b+np.sum(inv*(a*w+.5*w*w));prior=expit(s)
    second=np.mean((X-a)**2,axis=0)
    xp=X.T@post/len(X)
    nll=.5*logs.sum()+.5*np.sum(inv*second)-np.logaddexp(0.,ell).mean()+np.logaddexp(0.,s)
    penalty=strength*float(logs@logs)
    grad=np.r_[inv*(a-X.mean(axis=0)+prior*w),
               inv*(prior*(a+w)-xp),prior-post.mean(),
               .5-.5*inv*second+beta*xp-prior*inv*(a*w+.5*w*w)+2*strength*logs]
    return float(nll+penalty),grad


def fit_from_original(Z, original_state, diagonal):
    p=Z.shape[1]
    start=np.r_[original_state['a'],original_state['w'],original_state['b']]
    if diagonal:
        start=np.r_[start,np.zeros(p)]
        fun=diagonal_objective
        bounds=[(None,None)]*(2*p+1)+[(np.log(VARIANCE_FLOOR),None)]*p
    else:
        fun=rbm_objective;bounds=None
    initial,_=fun(start,Z)
    result=minimize(fun,start,args=(Z,),jac=True,method='L-BFGS-B',bounds=bounds,
                    options={'maxiter':100,'ftol':1e-10,'gtol':1e-6,'maxls':40})
    final,grad=fun(result.x,Z)
    if not np.isfinite(final) or not np.isfinite(result.x).all():
        raise ValueError('nonfinite fit')
    if final>initial+1e-8:raise ValueError('objective increased')
    a,w,b=result.x[:p],result.x[p:2*p],result.x[2*p]
    logs=result.x[2*p+1:] if diagonal else np.zeros(p)
    var=np.exp(logs);beta=w/var
    q=expit(b+Z@beta)
    penalty=VARIANCE_PENALTY*float(logs@logs) if diagonal else 0.
    projected=grad.copy()
    if diagonal:
        near=logs<=np.log(VARIANCE_FLOOR)+1e-8
        projected[2*p+1:][near & (grad[2*p+1:]>0)]=0
    return q,dict(a=a,w=w,b=np.asarray(b),variance=var),dict(
        converged=bool(result.success),iterations=int(result.nit),message=str(result.message),
        objective_initial=initial,objective_final=final,nll_final=final-penalty,penalty=penalty,
        gradient_max=float(np.max(np.abs(grad))),projected_gradient_max=float(np.max(np.abs(projected))),
        prior_hidden=float(expit(b+np.sum((a*w+.5*w*w)/var))),
        conditional_variance=var.tolist(),variance_floor_hits=int(np.sum(var<=VARIANCE_FLOOR+1e-8)),
        regularization=VARIANCE_PENALTY if diagonal else 0.,variance_floor=VARIANCE_FLOOR if diagonal else None)


def covariance_diagnostics(Z,state):
    """Moment compatibility, not a hallucination metric or goodness-of-fit p-value."""
    a,w,b=state['a'],state['w'],float(state['b'])
    var=np.asarray(state.get('variance',np.ones(Z.shape[1])))
    prior=float(expit(b+np.sum((a*w+.5*w*w)/var)))
    observed=(Z-Z.mean(axis=0)).T@(Z-Z.mean(axis=0))/len(Z)
    between=prior*(1-prior)*np.outer(w,w)
    predicted=np.diag(var)+between
    off=lambda C:C-np.diag(np.diag(C))
    return dict(prior=prior,observed_variance=np.diag(observed).tolist(),
                predicted_variance=np.diag(predicted).tolist(),
                mean_rmse=float(np.sqrt(np.mean((a+prior*w-Z.mean(axis=0))**2))),
                variance_rmse=float(np.sqrt(np.mean((np.diag(predicted)-np.diag(observed))**2))),
                covariance_relative_error=float(np.linalg.norm(predicted-observed)/max(np.linalg.norm(observed),1e-12)),
                offdiag_relative_error=float(np.linalg.norm(off(predicted-observed))/max(np.linalg.norm(off(observed)),1e-12)),
                empirical_eigenvalues=np.linalg.eigvalsh(observed).tolist(),
                covariance_excess_eigenvalues=np.linalg.eigvalsh(observed-np.diag(var)).tolist())


def fit_all(logprobs,chosen,entropy=None):
    X=representation(logprobs,chosen);Z,keep,mean,scale=zscore_columns(X)
    fits,failures,seconds={},{},{}
    if len(Z)<3 or Z.shape[1]<3:
        return {},{m:'ValueError: need three rows and varying columns' for m in METHODS},{m:0. for m in METHODS}
    original=None
    for method in METHODS:
        started=time.perf_counter()
        try:
            p=Z.shape[1]
            if method=='rbm':
                q,state,diag=fit_rbm(Z);original={k:v.copy() for k,v in state.items()}
                state['variance']=np.ones(p)
            elif method=='rbm_initial':
                state=dict(a=np.zeros(p),w=np.full(p,2/p),b=np.asarray(0.),variance=np.ones(p))
                q=expit(Z@state['w']);diag=dict(trained=False)
            else:
                if original is None:raise ValueError('original fit unavailable for paired warm start')
                q,state,diag=fit_from_original(Z,original,method=='rbm_diagonal')
            score,orientation=orient_posterior(q,Z.mean(axis=1));diag.update(orientation)
            diag.update(columns=np.flatnonzero(keep).tolist(),active_columns=p,
                        normalization_mean=mean.tolist(),normalization_scale=scale.tolist(),
                        anchor='mean6',covariance=covariance_diagnostics(Z,state))
            weights=np.zeros(6);weights[keep]=diag['orientation']*state['w']/state['variance']
            fits[method]=dict(score=score,state=state,weights=weights,diagnostics=diag)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError,RuntimeError) as error:
            failures[method]=f'{type(error).__name__}: {error}'
        seconds[method]=time.perf_counter()-started
    return fits,failures,seconds
