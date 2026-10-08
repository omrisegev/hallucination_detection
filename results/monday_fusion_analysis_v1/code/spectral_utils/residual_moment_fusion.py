"""Label-free crossed level/residual moment fusion, frozen September15 protocol."""
from __future__ import annotations
import numpy as np
from .cca_iu_isolation import iu_moments
from .energy_context_stability import simplex_qp, moments

FLOOR = 1e-6


def context_design(bundle, ids, positions):
    ids = np.asarray(ids, int); positions = np.asarray(positions, int)
    if ids.shape != positions.shape or np.any(positions < 0) or np.any(positions >= bundle.length[ids]):
        raise ValueError('Invalid token coordinates')
    local = positions[:, None] - np.arange(16, 0, -1)
    mask = local >= 0
    index = bundle.offset[ids, None] + np.maximum(local, 0)
    raw = np.asarray(bundle.features[index])[:, :, bundle.columns]
    z = (raw - bundle.mean[ids, None, :]) / bundle.scale[ids, None, :]
    z[~mask] = 0.
    return np.column_stack((z.reshape(len(ids), -1), mask, (positions + .5) / bundle.length[ids]))


def residuals(bundle, ids, positions, model, batch=4096):
    ids = np.asarray(ids, int); positions = np.asarray(positions, int)
    output = np.empty((len(ids), len(bundle.columns)))
    for start in range(0, len(ids), batch):
        sl = slice(start, start + batch); ii = ids[sl]; pp = positions[sl]
        prediction = model.predict(context_design(bundle, ii, pp))
        raw = bundle.features[bundle.offset[ii] + pp][:, bundle.columns]
        output[sl] = raw - (bundle.mean[ii] + bundle.scale[ii] * prediction)
    if not np.isfinite(output).all():
        raise FloatingPointError('Nonfinite residual')
    return output


def paired_moments(level, residual, weights=None):
    level = np.asarray(level, float); residual = np.asarray(residual, float)
    if level.shape != residual.shape or level.ndim != 2 or len(level) < 2:
        raise ValueError('Aligned nontrivial observations required')
    if not np.isfinite(level).all() or not np.isfinite(residual).all():
        raise ValueError('Nonfinite observations')
    if weights is None: weights = np.ones(len(level))
    mu, raw_C = moments(level, weights)
    sd = np.sqrt(np.maximum(np.diag(raw_C), 1e-12))
    _, CL = moments(level / sd, weights)
    _, CR = moments(residual / sd, weights)
    eye = FLOOR * np.eye(level.shape[1])
    CL += eye; CR += eye
    return sd, CL, CR, .25 * np.trace(CL) / level.shape[1]


def fit_head(C, sd, var_y):
    C = np.asarray(C, float); sd = np.asarray(sd, float)
    rho, g2, error = iu_moments(C, var_y)
    rho = rho[0]
    values, vectors = np.linalg.eigh(C)
    u = vectors[:, -2:]; v = values[-2:]
    a = u @ ((u.T @ rho) / (v + 1e-12))
    raw = a / sd; norm = np.abs(raw).sum()
    fallback = bool(norm < 1e-12)
    native = np.full(len(sd), 1 / len(sd)) if fallback else raw / norm
    B = sd / sd.mean()
    Q = C * B[:, None] * B[None, :]; r = B * rho
    unconstrained_simplex = simplex_qp(Q, r)[0]
    simplex = .75 / len(sd) + .25 * unconstrained_simplex
    return dict(native=native, simplex=simplex, rho=rho, g2=float(g2[0]),
                additive_residual=float(error[0]), native_a=a,
                native_fallback=fallback, raw_simplex=unconstrained_simplex,
                simplex_a=simplex * B, condition=float(values[-1] / values[0]))


def fit_pair(level, residual, weights=None):
    sd, CL, CR, ceiling = paired_moments(level, residual, weights)
    return dict(sd=sd, C_L=CL, C_R=CR, var_y=ceiling,
                L=fit_head(CL, sd, ceiling), R=fit_head(CR, sd, ceiling))


def stream_top10(x, spans):
    """Select peaks before weights, including negative native coefficients."""
    x = np.asarray(x, float); spans = np.asarray(spans, int)
    result = []
    for start, stop in spans:
        if not 0 <= start < stop <= len(x): raise ValueError('Invalid step span')
        rows = x[start:stop]; k = min(10, len(rows))
        result.append(np.partition(rows, len(rows) - k, axis=0)[-k:].mean(axis=0))
    return np.asarray(result)


def score_crossed(summaries, fitted, prefix):
    return {f'{prefix}__{head}__{moment}{score}': summaries[score] @ fitted[moment][head]
            for head in ('native', 'simplex') for moment in ('L', 'R') for score in ('L', 'R')}


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [jsonable(v) for v in value]
    return value
