"""Donor-only L-SML clustering followed by two levels of filtered IU-PCR.

The baseline partition is supplied unchanged. The deployed relevance rule is
applied once per level; survivors are refitted without further deletion. Unlike
upcr_fit(exclusion=True), this never restores a top-three minimum. Singleton
and pair exceptions are explicit, including after filtering. No labels enter.
"""
from __future__ import annotations

import numpy as np
from .advisor_lr import AffineFusion
from .fusion_utils import sml_fuse_signed
from .laplacian_upcr import IU_FIT_DEFAULTS
from .upcr import upcr_fit


def _small_or_iu(Z):
    m = Z.shape[1]
    if m == 1:
        return np.ones(1), 'singleton_passthrough'
    if m == 2:
        _, w = sml_fuse_signed(Z[:, 0], Z[:, 1])
        return w, 'continuous_lsml_pair'
    fit = upcr_fit(Z.T, **IU_FIT_DEFAULTS)
    if not np.isfinite(fit.w).all():
        raise ValueError('Non-finite IU-PCR fit')
    return fit.w, 'IU_PCR'


def fit_filtered_level(X):
    """Train-only normalization, relevance selection and affine survivor refit."""
    X = np.asarray(X, float)
    if X.ndim != 2 or not all(X.shape) or not np.isfinite(X).all():
        raise ValueError('Expected a nonempty finite steps-by-inputs matrix')
    mean, sd = X.mean(0), X.std(0)
    available = np.flatnonzero(sd > 1e-12)
    if not len(available):
        raise ValueError('All level inputs are constant')
    Z = (X[:, available] - mean[available]) / sd[available]
    diag = {'constant_columns': np.flatnonzero(sd <= 1e-12).tolist(),
            'n_original': X.shape[1], 'available': available.tolist(),
            'rho_hat_full': None, 'threshold': None, 'raw_keep': None}
    if len(available) >= 3:
        probe = upcr_fit(Z.T, **IU_FIT_DEFAULTS)
        rho = probe.rho_hat_full
        threshold = max(.05 * probe.var_y, float(rho.max()) / 3.)
        keep = np.flatnonzero(rho >= threshold)
        diag.update(rho_hat_full=rho.tolist(), threshold=threshold,
                    raw_keep=keep.tolist(), probe_g2=probe.g2_hat,
                    probe_var_y=probe.var_y)
        if not len(keep):
            raise ValueError('No input survived relevance selection')
    else:
        keep = np.arange(len(available))
        diag['selection_exception'] = 'rho is unidentified for fewer than three inputs'
    surviving = available[keep]
    w, mode = _small_or_iu(Z[:, keep])
    weight = np.zeros(X.shape[1]); weight[surviving] = w / sd[surviving]
    intercept = -float(mean @ weight)
    diag.update(retained_columns=surviving.tolist(), n_retained=len(surviving),
                refit_mode=mode, survivor_weights=w.tolist(),
                forced_restoration=False, repeated_filtering=False)
    model = AffineFusion(weight, intercept, mean, np.where(sd > 1e-12, sd, 1.), diag)
    error = float(np.max(np.abs(model.predict(X) - Z[:, keep] @ w)))
    if not np.isfinite(error) or error > 1e-9:
        raise RuntimeError(f'Level affine reconstruction failed: {error}')
    diag['affine_max_abs'] = error
    return model


def fit_clustered_iu(X, groups):
    X = np.asarray(X, float); groups = np.asarray(groups)
    if groups.shape != (X.shape[1],) or groups.dtype.kind not in 'iu':
        raise ValueError('Invalid full feature partition')
    inner, columns = [], []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        inner.append(fit_filtered_level(X[:, idx])); columns.append(idx)
    virtual = np.column_stack([m.predict(X[:, idx]) for m, idx in zip(inner, columns)])
    outer = fit_filtered_level(virtual)
    weight = np.zeros(X.shape[1]); intercept = outer.intercept
    details = []
    for k, (m, idx) in enumerate(zip(inner, columns)):
        weight[idx] = outer.weight[k] * m.weight
        intercept += outer.weight[k] * m.intercept
        details.append({'cluster': k, 'columns': idx.tolist(), 'model': {
            'weight': m.weight.tolist(), 'intercept': m.intercept,
            'diagnostic': m.diagnostic}, 'outer_coefficient': float(outer.weight[k])})
    diag = {'group': groups.tolist(), 'K': len(inner), 'inner': details,
            'outer': {'weight': outer.weight.tolist(), 'intercept': outer.intercept,
                      'diagnostic': outer.diagnostic},
            'n_features_retained_within': sum(m.diagnostic['n_retained'] for m in inner),
            'n_clusters_retained': outer.diagnostic['n_retained'],
            'n_features_active_final': int(np.count_nonzero(weight))}
    sd = X.std(0)
    model = AffineFusion(weight, float(intercept), X.mean(0), np.where(sd > 1e-12, sd, 1.), diag)
    error = float(np.max(np.abs(model.predict(X) - outer.predict(virtual))))
    if not np.isfinite(error) or error > 1e-9:
        raise RuntimeError(f'Hierarchy affine reconstruction failed: {error}')
    diag['affine_max_abs'] = error
    return model
