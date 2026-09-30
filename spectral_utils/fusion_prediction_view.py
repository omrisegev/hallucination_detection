"""A same-answer prediction-error view to SUPPORT existing feature fusion.

This is a small, explicit AR(1) adaptation, not KalmanNet or Diverging Flows.
Predictions use preceding tokens only. Downstream answer-fitted fusion remains
offline; causal feature construction does not make its fitted scores causal.
"""
from __future__ import annotations

import numpy as np

PSEUDOCOUNT = 16.0
EMA_SPAN = 32
KINDS = ('ar1', 'last', 'ema32')


def prediction_views(values):
    """Predict nine (or D) telemetry streams before observing each target.

At token t, fit pairs (x[k-1],x[k]) for k=1,...,t-1 only. Welford
statistics avoid subtracting large nearly equal sums. Let b be the clipped
OLS slope, mx/my the preceding pair means, n the number of pairs, and
eta=n/(n+16). Predict

    x[t-1] + eta * ((my-mx) + (b-1)*(x[t-1]-mx)).

This shrinks a stationary-slope AR(1) predictor towards the last observation.
The fixed pseudocount is an engineering choice, not an optimized parameter.
No pairs => last observation. Constant predictor column => b=1. The first
token has no prediction: a shared mask excludes its zero residual from
window averages. EMA32 is initialized at x[0] and is also read before update.
"""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError('FINITE_NONEMPTY_TRACE_WITH_TWO_TOKENS_REQUIRED')
    predictions = {kind: np.empty_like(x) for kind in KINDS}
    # Placeholder only; the first row is invalid for every predictor.
    for array in predictions.values():
        array[0] = x[0]
    n = 0
    mx = np.zeros(x.shape[1]); my = np.zeros(x.shape[1])
    sxx = np.zeros(x.shape[1]); sxy = np.zeros(x.shape[1])
    slow = x[0].copy()
    slope = np.ones_like(x); drift = np.zeros_like(x)
    for t in range(1, len(x)):
        previous = x[t - 1]
        beta = np.ones(x.shape[1])
        np.divide(sxy, sxx, out=beta, where=sxx > 0)
        beta = np.clip(beta, -1., 1.)
        eta = n / (n + PSEUDOCOUNT)
        correction = eta * ((my - mx) + (beta - 1.) * (previous - mx))
        predictions['ar1'][t] = previous + correction
        predictions['last'][t] = previous
        predictions['ema32'][t] = slow
        slope[t] = 1. + eta * (beta - 1.)
        drift[t] = eta * (my - mx)
        # Only now is the current target incorporated into the next fit.
        n += 1
        dx = previous - mx; dy = x[t] - my
        mx += dx / n; my += dy / n
        sxx += dx * (previous - mx)
        sxy += dx * (x[t] - my)
        slow += (2. / (EMA_SPAN + 1.)) * (x[t] - slow)
    mask = np.arange(len(x)) > 0
    residuals = {kind: x - prediction for kind, prediction in predictions.items()}
    if not all(np.isfinite(v).all() for v in (*predictions.values(), *residuals.values())):
        raise ValueError('NONFINITE_PREDICTION')
    return {'predictions': predictions, 'residuals': residuals,
            'mask': mask, 'slope': slope, 'drift': drift,
            'fit_pair_counts': np.maximum(np.arange(len(x)) - 1, 0)}


def residual_window_features(residual, mask, starts, ends):
    """Mean absolute prediction error, with the first unpredicted token masked."""
    residual = np.asarray(residual, float); mask = np.asarray(mask, bool)
    starts, ends = np.asarray(starts, int), np.asarray(ends, int)
    if (residual.ndim != 2 or mask.shape != (len(residual),)
            or starts.ndim != 1 or starts.shape != ends.shape or not len(starts)
            or np.any(starts < 0) or np.any(ends > len(residual)) or np.any(ends <= starts)
            or not np.isfinite(residual).all()):
        raise ValueError('INVALID_RESIDUAL_WINDOWS')
    counts = np.array([mask[a:b].sum() for a, b in zip(starts, ends)], dtype=int)
    if np.any(counts == 0):
        raise ValueError('WINDOW_HAS_NO_PREDICTED_TOKEN')
    values = np.array([np.abs(residual[a:b][mask[a:b]]).mean(axis=0)
                       for a, b in zip(starts, ends)])
    return values, counts


def augment_bank(base, names, residual_features, primitive_names):
    """Keep every original feature; append one error magnitude per primitive."""
    base = np.asarray(base, float); extra = np.asarray(residual_features, float)
    if (base.ndim != 2 or extra.shape != (len(base), len(primitive_names))
            or len(names) != base.shape[1] or not np.isfinite(base).all()
            or not np.isfinite(extra).all()):
        raise ValueError('AUGMENTATION_SCHEMA')
    return np.column_stack((base, extra)), list(names) + [
        name + '__prediction_abs_mean' for name in primitive_names]
