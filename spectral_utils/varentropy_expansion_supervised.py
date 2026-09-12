"""Supervised linear diagnostic on the answer-locally standardized expansion banks.

Access label: supervised, other-answer access (labels of other answers in the
same cell, source-group-disjoint folds).  A matched diagnostic for the label-free
arms, not a ceiling.  The step score is the SAME top-10 token mean used by the
label-free readout; supervision acts on aggregated steps only (no token-label
broadcasting).  This module owns neither the fold contract nor the metrics.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit

from .direct_probability_fusion import zscore_columns
from .varentropy_expansion_fusion import BANK_COLUMNS, WIDTH, expansion_columns

RIDGE = 0.01
BANKS = ('B2_sel', 'B2d_sel')          # B2d_sel is a column subset of the standardized B2_sel matrix
EXCLUDED = -1


def standardized_bank(logprobs, chosen):
    """Answer-local z-scored 138-column bank; constant columns (scale<=1e-10) become 0."""
    X, _, _ = expansion_columns(logprobs, chosen)
    Z, keep, _, _ = zscore_columns(X)
    full = np.zeros(X.shape, dtype=np.float64)
    full[:, keep] = Z
    return full, keep


def step_labels(kind, n_steps, *, target=None, labels=None):
    """PB: prefix steps 0, first-error step 1, later steps excluded; clean all 0. PRMB: given labels, unknown excluded."""
    if kind == 'pb':
        y = np.zeros(int(n_steps), dtype=int)
        t = int(target)
        if t >= 0:
            if t >= n_steps: raise ValueError('first-error step outside the answer')
            y[t] = 1; y[t + 1:] = EXCLUDED
        return y
    if kind == 'prm':
        y = np.asarray(labels, dtype=int).copy()
        if y.shape != (int(n_steps),): raise ValueError('PRMB label length mismatch')
        y[y < 0] = EXCLUDED
        return y
    raise ValueError(kind)


def class_balanced_weights(y):
    """Each known class carries total mass 0.5; excluded steps carry 0. Raises if a class is absent."""
    y = np.asarray(y, int); w = np.zeros(len(y), float)
    for label in (0, 1):
        mask = y == label
        if not mask.any(): raise ValueError(f'training set has no step of class {label} (declared fold failure)')
        w[mask] = 0.5 / mask.sum()
    return w


class StepTop10:
    """Top-10 token mean per step with its exact subgradient, over concatenated answers.

    Ties select earlier tokens (stable argsort of -token), matching
    ``rbm_matched_top10.top10_value_gradient``.  ``spans`` are global token index pairs.
    """

    def __init__(self, x, spans):
        self.x = np.asarray(x)
        self.spans = np.asarray(spans, int)
        if self.x.ndim != 2 or self.spans.ndim != 2 or self.spans.shape[1] != 2: raise ValueError('bad shapes')
        if np.any(self.spans[:, 0] < 0) or np.any(self.spans[:, 1] <= self.spans[:, 0]) or np.any(self.spans[:, 1] > len(self.x)):
            raise ValueError('invalid step span')

    def selection(self, token):
        rows, cols, vals = [], [], []
        for step, (a, b) in enumerate(self.spans):
            chosen = np.argsort(-token[a:b], kind='stable')[:min(10, b - a)] + a
            rows.extend([step] * len(chosen)); cols.extend(chosen.tolist()); vals.extend([1.0 / len(chosen)] * len(chosen))
        return csr_matrix((np.asarray(vals, dtype=self.x.dtype), (rows, cols)), shape=(len(self.spans), len(self.x)))

    def evaluate(self, w, b, derivative=False):
        w = np.asarray(w, dtype=self.x.dtype)
        token = self.x @ w + self.x.dtype.type(b)
        S = self.selection(token)
        value = np.asarray(S @ token, dtype=np.float64)
        if not derivative: return value, None
        jac = np.empty((len(value), self.x.shape[1] + 1)); jac[:, :-1] = S @ self.x; jac[:, -1] = 1.0
        return value, jac


def objective(theta, top, y, weight, ridge=RIDGE):
    theta = np.asarray(theta, float)
    s, jac = top.evaluate(theta[:-1], theta[-1], derivative=True)
    known = y != EXCLUDED
    if np.any(weight[~known] != 0): raise ValueError('excluded steps must carry zero weight')
    yy = np.where(known, y, 0)
    loss = float(np.sum(weight[known] * (np.logaddexp(0, s[known]) - yy[known] * s[known])))
    residual = np.zeros_like(s); residual[known] = weight[known] * (expit(s[known]) - yy[known])
    penalty = np.r_[theta[:-1], 0.0]
    return loss + 0.5 * ridge * float(penalty @ penalty), jac.T @ residual + ridge * penalty


def fit(x, spans, y, *, ridge=RIDGE, maxiter=200):
    """L-BFGS-B fit of the linear step-supervised model; returns (theta, info)."""
    top = StepTop10(x, spans); y = np.asarray(y, int); weight = class_balanced_weights(y)
    fun = lambda t: objective(t, top, y, weight, ridge)
    initial = np.zeros(x.shape[1] + 1); start = fun(initial)[0]
    result = minimize(fun, initial, jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6, maxls=40))
    end, grad = fun(result.x)
    if not np.isfinite(result.x).all() or not np.isfinite(end) or end > start + 1e-8: raise RuntimeError('nonfinite or worsening fit')
    theta_norm = float(np.linalg.norm(result.x)); gradient_max = float(np.max(np.abs(grad)))
    # A fit that never left the origin or exits with a large gradient is a declared STALLED fit,
    # never reported as FIT/converged. No perturbed restart (it would be an undeclared second start).
    stalled = theta_norm < 1e-8 or gradient_max > 1e-2
    return result.x, dict(status='STALLED' if stalled else 'FIT', stalled=bool(stalled),
                          converged=bool(result.success and not stalled), message=str(result.message), iterations=int(result.nit),
                          initial_loss=float(start), final_loss=float(end), gradient_max=gradient_max,
                          theta_norm=theta_norm, n_train_steps=int(np.sum(y != EXCLUDED)),
                          n_positive_steps=int(np.sum(y == 1)), n_negative_steps=int(np.sum(y == 0)), ridge=float(ridge))


def score_steps(x, spans, theta):
    """Held-out step scores with the same top-10 readout (float64 token scores)."""
    x = np.asarray(x, np.float64); theta = np.asarray(theta, float)
    return StepTop10(x, spans).evaluate(theta[:-1], theta[-1])[0]


def bank_columns(bank):
    if bank not in BANKS: raise ValueError(bank)
    return BANK_COLUMNS[bank]


__all__ = ['BANKS', 'EXCLUDED', 'RIDGE', 'StepTop10', 'bank_columns', 'class_balanced_weights', 'fit', 'objective',
           'score_steps', 'standardized_bank', 'step_labels', 'WIDTH']
