"""Answer-local fusion of probability-weighted varentropy contributions."""
import time
import numpy as np
from .direct_probability_fusion import logprob_matrix, zscore_columns, _orient
from .laplacian_upcr import IU_FIT_DEFAULTS
from .upcr import upcr_fit

METHODS = tuple(f'k{k}__{m}' for k in (15, 50) for m in ('raw', 'equal', 'iu'))


def contributions(logprobs, k):
    lp = logprob_matrix({'logprobs': logprobs}, k=k)
    p = np.exp(lp)
    # Exactly the frozen token_feature_views convention, including its epsilon.
    q = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    surprise = -np.log(q + 1e-12)
    entropy = (q * surprise).sum(axis=1, keepdims=True)
    return q * (surprise - entropy) ** 2


def fit_all(logprobs):
    fits, failures, seconds = {}, {}, {}
    for k in (15, 50):
        C = contributions(logprobs, k)
        raw = C.sum(axis=1)
        for solver in ('raw', 'equal', 'iu'):
            name = f'k{k}__{solver}'; started = time.perf_counter()
            try:
                if solver == 'raw':
                    score = raw.copy(); weights = np.ones(k)
                    effective = weights.copy(); intercept = 0.
                    diagnostic = dict(active_columns=k, orientation_flipped=False)
                else:
                    Z, keep, mean, scale = zscore_columns(C)
                    if len(C) < 3 or Z.shape[1] < 3:
                        raise ValueError('fewer than three tokens or varying contributions')
                    w = np.ones(Z.shape[1]) / Z.shape[1] if solver == 'equal' else upcr_fit(Z.T, **dict(IU_FIT_DEFAULTS)).w
                    score = Z @ w
                    flipped = False; corr = None
                    if solver == 'iu':
                        score, flipped, corr = _orient(score, raw)
                        w = w * (-1 if flipped else 1)
                    weights = np.zeros(k); weights[keep] = w
                    effective = np.zeros(k); effective[keep] = w / scale[keep]
                    intercept = -float(mean @ effective)
                    np.testing.assert_allclose(C @ effective + intercept, score, atol=1e-8, rtol=1e-8)
                    diagnostic = dict(active_columns=int(keep.sum()), orientation_flipped=bool(flipped),
                        anchor_correlation=float(corr) if corr is not None and np.isfinite(corr) else None)
                if not np.isfinite(score).all() or not np.isfinite(weights).all():
                    raise ValueError('nonfinite score or weights')
                fits[name] = dict(score=score, weights=weights, effective=effective,
                    intercept=intercept, diagnostics=diagnostic)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError, AssertionError) as e:
                failures[name] = f'{type(e).__name__}: {e}'
            seconds[name] = time.perf_counter() - started
    return fits, failures, seconds
