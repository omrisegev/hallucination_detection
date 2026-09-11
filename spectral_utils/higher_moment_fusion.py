"""Nested compact moment banks, orders 3..6; same-answer continuous fusion."""
import time
import numpy as np
from scipy.special import expit
from .moment_rbm_fusion import representation, fit_rbm
from .direct_probability_fusion import logprob_matrix, zscore_columns, _orient
from .laplacian_upcr import IU_FIT_DEFAULTS
from .upcr import upcr_fit

ORDERS = (3, 4, 5, 6)
SOLVERS = ('equal', 'iu', 'rbm_initial', 'rbm')
METHODS = tuple(f'd{d}__{s}' for d in ORDERS for s in SOLVERS)


def feature_names(order):
    if order not in ORDERS:
        raise ValueError('supported orders are 3 through 6')
    names = ['entropy15', 'varentropy15', 'moment3_15',
             'selected_surprisal', 'selected_squared', 'selected_cubed']
    for n in range(4, order + 1):
        names += [f'moment{n}_15', f'selected_power{n}']
    return tuple(names)


def representation_order(logprobs, chosen, order):
    feature_names(order)
    original = representation(logprobs, chosen)
    if order == 3:
        return original
    lp = logprob_matrix({'logprobs': logprobs}, k=15)
    p = np.exp(lp)
    q = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    y = -np.log(q + 1e-12)
    a = original[:, 3]
    columns = [original]
    for n in range(4, order + 1):
        columns += [(q * y**n).sum(axis=1)[:, None], (a**n)[:, None]]
    out = np.concatenate(columns, axis=1)
    if not np.isfinite(out).all():
        raise ValueError('nonfinite moment bank; no hidden clipping')
    return out


def fit_all(logprobs, chosen, entropy=None):
    # The anchor is fixed across orders; additional columns cannot redefine risk.
    original = representation_order(logprobs, chosen, 3)
    z0, _, _, _ = zscore_columns(original)
    if z0.shape[1] == 0:
        return {}, {m: 'ValueError: no varying original columns for risk orientation' for m in METHODS}, {m: 0. for m in METHODS}
    anchor = z0.mean(axis=1)
    fits, failures, seconds = {}, {}, {}
    for degree in ORDERS:
        X = representation_order(logprobs, chosen, degree)
        Z, keep, mean, scale = zscore_columns(X)
        for solver in SOLVERS:
            method = f'd{degree}__{solver}'
            started = time.perf_counter()
            try:
                if len(X) < 3 or Z.shape[1] < 3:
                    raise ValueError('need three rows and varying columns')
                p = Z.shape[1]
                diag = dict(order=degree, columns=np.flatnonzero(keep).tolist(),
                            feature_names=feature_names(degree), active_columns=p,
                            normalization_mean=mean.tolist(), normalization_scale=scale.tolist())
                if solver == 'rbm':
                    score, state, extra = fit_rbm(Z, maxiter=100)
                    diag.update(extra)
                else:
                    w = np.full(p, 2. / p if solver == 'rbm_initial' else 1. / p)
                    if solver == 'iu':
                        fit = upcr_fit(Z.T, **dict(IU_FIT_DEFAULTS))
                        if fit.abstained or fit.used_simple_average:
                            raise ValueError('IU abstention/fallback')
                        w = np.asarray(fit.w)
                        diag.update(g2_hat=float(fit.g2_hat), n_components=int(fit.n_components_used))
                    state = dict(w=w.copy())
                    score = Z @ w
                    if solver == 'rbm_initial':
                        state.update(a=np.zeros(p), b=np.asarray(0.))
                        score = expit(score)
                        diag['trained'] = False
                score, flipped, corr = _orient(score, anchor)
                if flipped and solver.startswith('rbm'):
                    score = 1 + score
                if not np.isfinite(score).all():
                    raise ValueError('nonfinite score')
                diag.update(orientation=-1 if flipped else 1,
                            anchor_correlation=float(corr) if corr is not None and np.isfinite(corr) else None,
                            score_sd=float(score.std()), collapsed=bool(score.std() < 1e-3))
                weights = np.zeros(X.shape[1])
                weights[keep] = state['w'] * diag['orientation']
                fits[method] = dict(score=score, state=state, weights=weights, diagnostics=diag)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, AssertionError) as error:
                failures[method] = f'{type(error).__name__}: {error}'
            seconds[method] = time.perf_counter() - started
    return fits, failures, seconds
