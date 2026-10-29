"""Shared, label-free pieces of the pre-registered answer-mixture gate
(docs/experiments/ANSWER_MIXTURE_GATE_PREREG_2026-10-29.md). Labels never enter any function here except `score`,
which only evaluates a finished fit.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer

SEED = 20261029
N_INIT = 10


def answer_max(step, off):
    """Primary answer score: the maximum step score of each answer."""
    return np.maximum.reduceat(np.asarray(step, float), off[:-1])


def answer_top2(step, off):
    """Secondary answer score: mean of the top-2 steps (the single step for 1-step answers)."""
    out = np.empty(len(off) - 1)
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        v = np.sort(step[a:b])[::-1]
        out[j] = v[:2].mean()
    return out


def residualize(s, covs):
    """Pooled OLS of s on [1, covariates]; returns the residual and the coefficients. No labels."""
    X = np.column_stack([np.ones(len(s))] + [np.asarray(c, float) for c in covs])
    beta, *_ = np.linalg.lstsq(X, s, rcond=None)
    return s - X @ beta, beta


def fit_mixture(x, transform=None):
    """1-D two-component Gaussian mixture (full covariance, n_init 10, seed 20261029) and the one-component BIC.

    transform=None: primary (on the residual itself); 'yeo-johnson': secondary (PowerTransformer, standardize=True).
    Returns pi_hat (weight of the higher-mean component), the posterior of that component per answer, BICs, params.
    """
    x = np.asarray(x, float)
    if transform == 'yeo-johnson':
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        x = pt.fit_transform(x[:, None])[:, 0]
        lam = float(pt.lambdas_[0])
    else:
        assert transform is None
        lam = None
    X = x[:, None]
    g2 = GaussianMixture(n_components=2, covariance_type='full', n_init=N_INIT, random_state=SEED).fit(X)
    g1 = GaussianMixture(n_components=1, covariance_type='full', n_init=N_INIT, random_state=SEED).fit(X)
    hi = int(np.argmax(g2.means_[:, 0]))
    post = g2.predict_proba(X)[:, hi]
    return {'pi_hat': float(g2.weights_[hi]), 'post_err': post,
            'bic1': float(g1.bic(X)), 'bic2': float(g2.bic(X)), 'bic2_minus_bic1': float(g2.bic(X) - g1.bic(X)),
            'bic_prefers_2': bool(g2.bic(X) < g1.bic(X)), 'converged': bool(g2.converged_ and g1.converged_),
            'means': g2.means_[:, 0].tolist(), 'sds': np.sqrt(g2.covariances_[:, 0, 0]).tolist(),
            'weights': g2.weights_.tolist(), 'yj_lambda': lam}


def auc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0:
        return float('nan')
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score(fit, has_err, s_a):
    """Endpoints (labels only score). Clean decision: posterior of the error component < .5."""
    y = np.asarray(has_err, bool); clean_call = fit['post_err'] < .5
    pi = float(y.mean())
    return {'n': int(len(y)), 'pi_true': pi, 'pi_hat': fit['pi_hat'], 'abs_err': abs(fit['pi_hat'] - pi),
            'bic1': fit['bic1'], 'bic2': fit['bic2'], 'bic2_minus_bic1': fit['bic2_minus_bic1'],
            'bic_prefers_2': fit['bic_prefers_2'], 'converged': fit['converged'],
            'clean_closed': float(clean_call[~y].mean()) if (~y).any() else float('nan'),
            'err_kept': float((~clean_call[y]).mean()) if y.any() else float('nan'),
            'share_called_clean': float(clean_call.mean()),
            'auroc_s_a': auc(y, s_a),
            'gate_useful': bool((clean_call[~y].mean() >= .30) and ((~clean_call[y]).mean() >= .85)) if (~y).any() and y.any() else False,
            'means': fit['means'], 'sds': fit['sds'], 'weights': fit['weights'], 'yj_lambda': fit['yj_lambda']}
