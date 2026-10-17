"""Renyi-order sweep (Stage 3b, authorized by Omri 2026-09-13): single-view localization as a function of alpha.

Each arm is ONE Renyi entropy of the frozen renormalized top-15 head (exactly the v1/v2 definitions:
``renyi_view_fusion.head_distribution`` / ``renyi_entropy``), natural high-is-risk sign, no
standardization, no fusion, no labels in scoring.  The sweep is a label-guided hyperparameter
search over ONE scalar on development data: the resulting curve and any chosen alpha are
development evidence; the selection rule must be frozen before an untouched confirmation.

Grid: alpha in {0.001, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 8, inf}
plus the analytic alpha -> 0 limit view: H_alpha = log(sum q^alpha)/(1-alpha) = log K + alpha (mean_i log q_i + log K)
+ O(alpha^2), so the within-answer ordering tends to that of ``mean_i log q_i`` (``H0lim``; higher = flatter head).
``view__H1`` is the frozen entropy15 (asserted by the evaluator); ``view__a0.1`` is the v2 ``view__H0.1``.

Second family (Omri, 2026-09-13: "is there a varentropy that uses these weights?"): the escort-weighted
varentropy VE_alpha = sum_i w_i s_i^2 - (sum_i w_i s_i)^2 with w_i = q_i^alpha / sum_j q_j^alpha (the Renyi
escort distribution; alpha = 0 -> uniform weights over the head, alpha = 1 -> exactly the frozen varentropy15,
alpha -> inf -> one-hot on q_1 and VE -> 0, so inf is excluded).  Grid: alpha in {0, 0.1, 0.25, 0.5, 0.75, 1,
1.5, 2, 3, 4, 8}.  ``view__ve1`` must reproduce the frozen Step-339 varentropy15 metrics (asserted).

Orientation of the VE family: the v2 smoke showed VE_alpha for small alpha anticorrelates with entropy
(uniform-weighted surprisal variance is LARGE when the head is peaked), i.e. its natural sign is not
high-is-risk.  Each VE arm is therefore oriented per answer by the sign of its within-answer Pearson
correlation with the answer's own raw varentropy15 (the label-free anchor rule of the fusion arms);
the flip is recorded (``orientation_flipped``).  At alpha = 1 the correlation is +1 and no flip occurs.
Renyi-entropy arms keep their natural high-is-risk sign (no flip), as in v2.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.stats import rankdata

from .renyi_view_fusion import head_distribution, renyi_entropy, EPS, K
from .varentropy_contribution_fusion import contributions

ALPHA_GRID = (0.001, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, np.inf)
VE_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0)


def _alpha_name(a):
    if np.isinf(a):
        return 'view__Hinf'
    if a == 1:
        return 'view__H1'
    return 'view__a' + ('%g' % a)


VIEW_NAMES = ('H0lim',) + tuple(_alpha_name(a)[len('view__'):] for a in ALPHA_GRID) + tuple('ve%g' % a for a in VE_GRID)
METHODS = tuple('view__' + v for v in VIEW_NAMES)
COLUMN_NAMES = VIEW_NAMES
MAX_WEIGHTS = len(COLUMN_NAMES)
ALPHA_OF = {'view__H0lim': 0.0, **{_alpha_name(a): a for a in ALPHA_GRID}, **{'view__ve%g' % a: a for a in VE_GRID}}
FAMILY_OF = {m: ('escort_varentropy' if m.startswith('view__ve') else 'renyi_entropy') for m in ALPHA_OF}
DIAG_NAMES = COLUMN_NAMES + ('varentropy15',)


def sweep_matrix(logprobs):
    q, _ = head_distribution(logprobs, K)
    lim = np.log(q + EPS).mean(axis=1)                      # alpha -> 0 limit ordering (mean log q over the head)
    V = np.column_stack([lim] + [renyi_entropy(q, a) for a in ALPHA_GRID] + [escort_varentropy(q, a) for a in VE_GRID])
    if not np.isfinite(V).all():
        raise ValueError('nonfinite sweep view')
    return V, VIEW_NAMES


def escort_varentropy(q, alpha):
    """Variance of the surprisal s = -log(q + eps) under the escort weights w ~ q^alpha (frozen epsilons)."""
    q = np.asarray(q, float); s = -np.log(q + EPS)
    if alpha == 0:
        w = np.full_like(q, 1.0 / q.shape[1])
    else:
        w = q ** float(alpha); w = w / w.sum(axis=1, keepdims=True)
    m = (w * s).sum(axis=1)
    return (w * s * s).sum(axis=1) - m * m


def anchor_stream(logprobs):
    return contributions(logprobs, K).sum(axis=1)


def _finite_or_none(x):
    x = float(x)
    return x if np.isfinite(x) else None


def view_diagnostics(logprobs, chosen):
    V, _ = sweep_matrix(logprobs); anchor = anchor_stream(logprobs)
    M = np.column_stack((V, anchor)); std = M.std(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        pearson = np.corrcoef(M, rowvar=False) if len(M) > 1 else np.full((M.shape[1],) * 2, np.nan)
        ranks = np.column_stack([rankdata(M[:, j]) for j in range(M.shape[1])])
        spearman = np.corrcoef(ranks, rowvar=False) if len(M) > 1 else np.full((M.shape[1],) * 2, np.nan)
        anchor_corr = [np.corrcoef(V[:, j], anchor)[0, 1] if len(V) > 1 else np.nan for j in range(V.shape[1])]
    # Spearman of every alpha with the limit view and with alpha = 0.1: how fast the ordering converges.
    return dict(columns=list(DIAG_NAMES), n_tokens=int(len(V)),
                std=[_finite_or_none(v) for v in std], near_constant=[bool(v <= 1e-10) for v in std[:MAX_WEIGHTS]],
                pearson=[[_finite_or_none(v) for v in row] for row in np.asarray(pearson, float)],
                spearman=[[_finite_or_none(v) for v in row] for row in np.asarray(spearman, float)],
                anchor_correlation=[_finite_or_none(v) for v in anchor_corr])


def fit_all(logprobs, chosen, methods=METHODS, *, uid='no-uid'):
    fits, failures, seconds = {}, {}, {}
    for name in methods:
        if name not in METHODS:
            raise ValueError(name)
    try:
        V, names = sweep_matrix(logprobs); anchor = anchor_stream(logprobs)
    except (ValueError, FloatingPointError) as error:
        reason = f'{type(error).__name__}: head invalid: {error}'
        return {}, {name: reason for name in methods}, {name: 0.0 for name in methods}
    for name in methods:
        started = time.perf_counter()
        try:
            j = names.index(name[len('view__'):]); x = V[:, j].copy(); sd = float(x.std())
            if not len(x):
                raise ValueError('empty view')
            if sd <= 1e-10:
                raise ValueError(f'constant view {name}; no within-answer localization signal')
            w = np.zeros(MAX_WEIGHTS); w[j] = 1.0
            with np.errstate(invalid='ignore', divide='ignore'):
                corr = float(np.corrcoef(x, anchor)[0, 1]) if len(x) > 1 and np.std(anchor) > EPS else np.nan
            flipped = False
            if FAMILY_OF[name] == 'escort_varentropy' and np.isfinite(corr) and corr < 0:
                x = -x; w = -w; flipped = True
            fits[name] = dict(score=x, weights=w, effective=w.copy(), intercept=0.0, state={},
                              diagnostics=dict(active_columns=1, orientation_flipped=flipped, alpha=ALPHA_OF[name],
                                               anchor_correlation=_finite_or_none(corr), std=sd,
                                               natural_sign='high_is_risk' if FAMILY_OF[name] == 'renyi_entropy' else 'anchor_oriented'))
        except (ValueError, FloatingPointError) as error:
            failures[name] = f'{type(error).__name__}: {error}'
        seconds[name] = time.perf_counter() - started
    return fits, failures, seconds


__all__ = ['ALPHA_GRID', 'ALPHA_OF', 'FAMILY_OF', 'VE_GRID', 'escort_varentropy', 'COLUMN_NAMES', 'DIAG_NAMES', 'MAX_WEIGHTS', 'METHODS', 'VIEW_NAMES',
           'anchor_stream', 'fit_all', 'sweep_matrix', 'view_diagnostics']
