"""Train-only linear fusion helpers for the advisor v11 diagnostic.

This module is a supervised diagnostic, not a new label-free method. Feature
selection accepts training data only; callers own source-group split validation.
Continuous L-SML and IU-PCR mirror the pinned Monday continuous-baseline entry
points. No evaluation-label sign correction, fallback or regrouping is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .fusion_utils import lsml_continuous
from .upcr import upcr_fit
from .laplacian_upcr import IU_FIT_DEFAULTS

C_GRID = (.001, .01, .1, 1., 10., 100., 1000.)
SELECTOR_C = 1.


@dataclass
class AffineFusion:
    """A fitted score on original answer-z columns: X @ weight + intercept."""
    weight: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray
    diagnostic: dict

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.weight + self.intercept

    @property
    def weight_std(self):
        """Contribution per training SD, common display/selection scale."""
        return self.weight * self.scale


def fit_lr(X, y, C=1.):
    X, y = np.asarray(X, float), np.asarray(y, int)
    if len(np.unique(y)) != 2:
        raise ValueError('Training fold has only one class')
    sc = StandardScaler().fit(X)
    estimator = LogisticRegression(C=float(C), class_weight='balanced', solver='lbfgs',
                                   max_iter=3000, random_state=20261003)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        estimator.fit(sc.transform(X), y)
    weight = estimator.coef_[0] / sc.scale_
    intercept = float(estimator.intercept_[0] - sc.mean_ @ weight)
    model = AffineFusion(weight, intercept, sc.mean_, sc.scale_,
                         {'C': float(C), 'coef_std': estimator.coef_[0].tolist(),
                          'n_iter': int(estimator.n_iter_[0])})
    error = float(np.max(np.abs(model.predict(X) -
                               estimator.decision_function(sc.transform(X)))))
    model.diagnostic['affine_max_abs'] = error
    if error > 1e-9:
        raise RuntimeError(f'LR affine reconstruction error {error}')
    return model


def fit_labelfree(X, method):
    X = np.asarray(X, float)
    if not np.isfinite(X).all():
        raise ValueError('Non-finite training input')
    mean, sd = X.mean(0), X.std(0)
    scale = np.where(sd > 0, sd, 1.)
    if method == 'LSML_CONT':
        score, meta = lsml_continuous(*[X[:, j] for j in range(X.shape[1])])
        if int(meta['K']) < 3:
            raise RuntimeError(f"L-SML selected K={meta['K']}: fewer than 3 groups")
        weight = np.zeros(X.shape[1])
        for cross, (indices, within) in zip(meta['cross_weights'], meta['group_weights']):
            weight[np.asarray(indices, int)] = float(cross) * np.asarray(within)
        intercept = 0.
        diag = {'K': int(meta['K']), 'degenerate': bool(meta['degenerate']),
                'group': np.asarray(meta['c'], int).tolist(),
                'group_weights': [{'columns': np.asarray(idx, int).tolist(),
                                   'weights': np.asarray(w, float).tolist()}
                                  for idx, w in meta['group_weights']],
                'cross_weights': np.asarray(meta['cross_weights'], float).tolist(),
                'normalization': 'original answer-z inputs; no global scaling'}
    elif method == 'IU_PCR':
        keep = sd > 0
        if int(keep.sum()) < 3:
            raise RuntimeError('Fewer than 3 non-constant channels')
        F = ((X[:, keep] - mean[keep]) / sd[keep]).T
        result = upcr_fit(F, **IU_FIT_DEFAULTS)
        if not np.isfinite(result.w).all():
            raise RuntimeError('Non-finite IU-PCR weights')
        weight = np.zeros(X.shape[1]); weight[keep] = result.w / sd[keep]
        intercept = -float(mean @ weight)
        score = result.w @ F
        diag = {'n_features': int(keep.sum()), 'constant_columns': np.flatnonzero(~keep).tolist(),
                'abstained': bool(result.abstained), 'IU_FIT_DEFAULTS': dict(IU_FIT_DEFAULTS),
                'normalization': 'training-only global mean/population SD'}
    else:
        raise ValueError(f'Unknown method {method}')
    model = AffineFusion(weight, intercept, mean, scale, diag)
    error = float(np.max(np.abs(model.predict(X) - score)))
    model.diagnostic['affine_max_abs'] = error
    if not np.isfinite(error) or error > 1e-9:
        raise RuntimeError(f'{method} affine reconstruction error {error}')
    return model


def balanced_log_loss(y, score):
    y = np.asarray(y, int)
    count = np.bincount(y, minlength=2)
    if (count == 0).any():
        raise ValueError('Inner validation lacks a class')
    # Stable log-loss on decision function, identically class-balanced.
    loss = np.logaddexp(0., score) - y * score
    return float(np.mean(loss * (len(y) / (2. * count[y]))))


def rank_half(weight):
    n = len(weight); k = math.ceil(n / 2)
    order = np.lexsort((np.arange(n), -np.abs(weight)))
    return np.sort(order[:k])


def restricted_sets(weight_std, X):
    """Historical R1/R2/R3 definitions; training-only selection inputs."""
    a = np.abs(weight_std); n = len(a)
    corr = np.corrcoef(X, rowvar=False)
    median = np.median(a); in_pair = set()
    for i in range(n):
        for j in range(i + 1, n):
            if (abs(corr[i, j]) >= .8 and
                    np.sign(weight_std[i]) * np.sign(weight_std[j]) * np.sign(corr[i, j]) < 0 and
                    a[i] > median and a[j] > median):
                in_pair.update((i, j))
    reference = max([a[i] for i in range(n) if i not in in_pair] or [a.max()])
    r3 = np.ones(n, bool)
    r3[np.argsort(a, kind='stable')[:math.ceil(n / 4)]] = False
    sets = {'R1': np.flatnonzero(a >= .05 * a.max()),
            'R2': np.flatnonzero(a >= .05 * reference), 'R3': np.flatnonzero(r3)}
    if any(len(c) == 0 for c in sets.values()):
        raise RuntimeError('Restricted selector returned no columns')
    return sets


def as_dict(model):
    return {'weight': model.weight.tolist(), 'intercept': model.intercept,
            'training_mean': model.mean.tolist(), 'training_scale': model.scale.tolist(),
            'weight_std': model.weight_std.tolist(), 'diagnostic': model.diagnostic}
