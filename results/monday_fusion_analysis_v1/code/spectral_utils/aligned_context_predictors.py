"""Fixed telemetry predictors; no correctness labels or fitting-group API."""
import numpy as np


def past_mean(values, width=16):
    x = np.asarray(values, float)
    if x.ndim != 2 or len(x) == 0 or width < 1 or not np.isfinite(x).all():
        raise ValueError('invalid predictor input')
    prefix = np.vstack((np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)))
    t = np.arange(len(x)); start = np.maximum(0, t-width)
    return (prefix[t]-prefix[start])/np.maximum(t-start, 1)[:, None]


def noreset_mean(values):
    x = np.asarray(values, float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('invalid predictor input')
    return np.vstack((np.zeros((1, x.shape[1])), np.cumsum(x[:-1], axis=0)))/(np.arange(len(x))+1)[:, None]


def bocpd_mean(values, hazard=1/32):
    """Independent scalar Gaussian filters, exact reset-before-observation.

    Unit observation/prior variance, zero prior mean. Return PRIOR predictive
    mean for x_t. Run-length states never truncated; no fitting on current x_t.
    Posterior variance after r observations is 1/(r+1), analytically exact.
    """
    x = np.asarray(values, float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or not 0 <= hazard < 1:
        raise ValueError('invalid predictor input')
    if hazard == 0:
        return noreset_mean(x)
    n, d = x.shape
    output = np.zeros_like(x)
    p = np.zeros((n, d)); mu = np.zeros_like(p)
    p[0] = 1
    # Candidate state r has r previous observations before its current update.
    v = 1/(np.arange(n, dtype=float)+1)
    variance = v+1
    lognorm = -.5*np.log(2*np.pi*variance)
    gain = v/variance
    for t in range(n):
        count = t+1
        if t:
            p[1:count] = (1-hazard)*p[:t]
            p[0] = hazard
            mu[1:count] = mu[:t]
            mu[0] = 0
        output[t] = np.sum(p[:count]*mu[:count], axis=0)
        loglike = lognorm[:count, None]-.5*(x[t]-mu[:count])**2/variance[:count, None]
        # Relative likelihoods avoid underflow of the complete mixture.
        likelihood = np.exp(loglike-loglike.max(axis=0))
        p[:count] *= likelihood
        p[:count] /= p[:count].sum(axis=0)
        mu[:count] += gain[:count, None]*(x[t]-mu[:count])
    if not np.isfinite(output).all():
        raise FloatingPointError('invalid BOCPD prediction')
    return output


def agreement(left, right):
    left = np.asarray(left, float); right = np.asarray(right, float)
    left = left-left.mean(); right = right-right.mean()
    den = np.linalg.norm(left)*np.linalg.norm(right)
    return float(left@right/den) if den > 1e-12 else None


def remove_current(signal, current):
    s = np.asarray(signal, float)-np.mean(signal)
    x = np.asarray(current, float)-np.mean(current)
    return s-x*(x@s/max(float(x@x), 1e-20))
