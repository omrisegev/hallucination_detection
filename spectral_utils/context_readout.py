"""Common fixed readouts for linear and neural feature predictors."""
import numpy as np
from .direct_probability_fusion import step_top_mean
from .temporal_context_models import residual_step_score


def prediction_readouts(raw, standardized, predicted, spans, base, variance=None):
    """Current observation is used for innovation, never to fit its predictor.

    Squared prediction error is a novelty diagnostic, not a correctness label.
    Raw units remain available in the unchanged baseline and variance fusion.
    """
    residual = standardized-predicted
    signals = {'signed': residual.mean(axis=1), 'squared': np.square(residual).mean(axis=1)}
    result = {}
    for name, values in signals.items():
        auxiliary = step_top_mean(values, spans[:,0], spans[:,1], 10)
        result[name] = auxiliary
        for gamma in (.25, 1.):
            result[f'{name}_residual_{gamma:g}'] = residual_step_score(base, auxiliary, gamma)
    if variance is not None:
        variance = np.asarray(variance)
        if variance.shape != raw.shape or not np.isfinite(variance).all() or np.any(variance<=0):
            raise ValueError('invalid predictive variances')
        precision = 1/variance
        weights = precision/precision.sum(axis=1, keepdims=True)
        # Per-stream Top10 first: constant equal weights reduce to raw mean Top10.
        result['variance_fusion'] = sum(step_top_mean(raw[:,j]*weights[:,j], spans[:,0], spans[:,1], 10)
                                        for j in range(raw.shape[1]))
    return result
