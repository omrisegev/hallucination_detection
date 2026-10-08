"""One frozen weight-shrinkage addition to the exact answer-local Gaussian RBM.

The six-column representation, initialization, orientation, optimizer and readout
are inherited unchanged. Only mean-NLL + 0.1 ||w - 2/P||^2 is new.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .moment_rbm_fusion import representation, rbm_objective, fit_rbm
from .direct_probability_fusion import zscore_columns
from .rbm_m3_powers import orient_posterior

LAMBDA = 0.1
METHODS = ('rbm', 'rbm_initial', 'rbm_shrinkage')


def penalized_objective(theta, X, strength=LAMBDA):
    """Average NLL, not summed NLL; penalize only raw latent weights."""
    if not np.isfinite(strength) or strength < 0:
        raise ValueError('regularization must be finite and nonnegative')
    loss, grad = rbm_objective(theta, X)
    p = X.shape[1]
    delta = theta[p:2*p] - 2.0/p
    return loss + strength * float(delta @ delta), grad + np.r_[
        np.zeros(p), 2*strength*delta, 0.0]


def fit_regularized(X, *, strength=LAMBDA, maxiter=100):
    p = X.shape[1]
    initial = np.r_[np.zeros(p), np.full(p, 2.0/p), 0.0]
    start, _ = penalized_objective(initial, X, strength)
    result = minimize(penalized_objective, initial, args=(X, strength), jac=True,
        method='L-BFGS-B', options={'maxiter':maxiter, 'ftol':1e-10,
                                  'gtol':1e-6, 'maxls':40})
    final, grad = penalized_objective(result.x, X, strength)
    if not np.isfinite(result.x).all() or not np.isfinite(final):
        raise ValueError('nonfinite regularized RBM fit')
    if final > start + 1e-8:
        raise ValueError('RBM optimizer increased penalized objective')
    a, w, b = result.x[:p], result.x[p:2*p], result.x[-1]
    return expit(b + X@w), dict(a=a, w=w, b=np.asarray(b)), dict(
        optimizer='exact penalized likelihood L-BFGS-B',
        converged=bool(result.success), message=str(result.message),
        iterations=int(result.nit), objective_initial=start, objective_final=final,
        nll_initial=rbm_objective(initial, X)[0],
        nll_final=rbm_objective(result.x, X)[0], regularization=float(strength),
        penalty=float(strength*np.sum((w-2.0/p)**2)),
        gradient_max=float(np.max(np.abs(grad))),
        prior_hidden=float(expit(b+a@w+.5*w@w)))


def fit_all(logprobs, chosen, entropy=None):
    # entropy accepted only for compatibility with the existing runner. The
    # anchor is the same answer-local six-column mean for every arm.
    X = representation(logprobs, chosen)
    Z, keep, mean, scale = zscore_columns(X)
    fits, failures, seconds = {}, {}, {}
    for method in METHODS:
        started = time.perf_counter()
        try:
            if len(X) < 3 or Z.shape[1] < 3:
                raise ValueError('need three rows and varying columns')
            p = Z.shape[1]
            if method == 'rbm':
                q, state, diag = fit_rbm(Z)
            elif method == 'rbm_shrinkage':
                q, state, diag = fit_regularized(Z)
            else:
                state = dict(a=np.zeros(p), w=np.full(p,2.0/p), b=np.asarray(0.0))
                q = expit(Z@state['w']); diag = {'trained':False}
            score, orientation = orient_posterior(q, Z.mean(axis=1))
            diag.update(orientation)
            diag.update(columns=np.flatnonzero(keep).tolist(), active_columns=int(keep.sum()),
                normalization_mean=mean.tolist(), normalization_scale=scale.tolist(),
                anchor='mean6', raw_weight_distance_from_initial=float(np.linalg.norm(state['w']-2.0/p)),
                oriented_weight_distance_from_initial=float(np.linalg.norm(orientation['orientation']*state['w']-2.0/p)))
            if not np.isfinite(score).all():
                raise ValueError('nonfinite score')
            weights = np.zeros(6)
            weights[keep] = orientation['orientation'] * state['w']
            fits[method] = dict(score=score, state=state, weights=weights, diagnostics=diag)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as e:
            failures[method] = f'{type(e).__name__}: {e}'
        seconds[method] = time.perf_counter()-started
    return fits, failures, seconds
