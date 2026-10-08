"""Nested supervised step-level diagnostics; not an unsupervised RBM."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

MODES = ('static', 'prior', 'conditional')
RIDGE = .01


def design(x, context, mode):
    x = np.asarray(x, float)
    c = np.asarray(context, float)
    if mode not in MODES or c.shape != (len(x),):
        raise ValueError('invalid design')
    if mode == 'static':
        return x
    if mode == 'prior':
        return np.column_stack((x, c))
    return np.column_stack((x, c, x*c[:, None]))


def listwise_loss(theta, x, offsets, targets, ridge=RIDGE):
    """Every training answer has one first-error target, not token labels."""
    starts = offsets[:-1]
    if len(targets) != len(starts) or np.any(targets < 0) or np.any(targets >= np.diff(offsets)):
        raise ValueError('invalid first-error target')
    s = x @ theta
    maxima = np.maximum.reduceat(s, starts)
    e = np.exp(s - np.repeat(maxima, np.diff(offsets)))
    sums = np.add.reduceat(e, starts)
    probs = e/np.repeat(sums, np.diff(offsets))
    target_rows = starts+targets
    value = np.mean(maxima+np.log(sums)-s[target_rows])
    probs[target_rows] -= 1
    grad = x.T @ probs/len(targets)
    return float(value+.5*ridge*(theta@theta)), grad+ridge*theta


def binary_loss(theta, x, y, weight, ridge=RIDGE):
    s = x @ theta[:-1]+theta[-1]
    loss = np.sum(weight*(np.logaddexp(0, s)-y*s))
    r = weight*(expit(s)-y)
    grad = np.r_[x.T@r+ridge*theta[:-1], r.sum()]
    return float(loss+.5*ridge*(theta[:-1]@theta[:-1])), grad


def balanced_answer_weights(y, answer):
    _, inv, n = np.unique(answer, return_inverse=True, return_counts=True)
    weight = 1./n[inv]
    for label in (0, 1):
        mask = y == label
        total = weight[mask].sum()
        if total <= 0:
            raise ValueError('training data lack a class')
        weight[mask] *= .5/total
    return weight


def optimize(fun, dimensions):
    initial = np.zeros(dimensions)
    result = minimize(fun, initial, jac=True, method='L-BFGS-B',
                      options=dict(maxiter=1000, ftol=1e-12, gtol=1e-7, maxls=40))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('fit failed: '+str(result.message))
    return result.x, dict(converged=True, iterations=int(result.nit),
                         initial_loss=float(fun(initial)[0]), loss=float(result.fun),
                         gradient_max=float(np.max(np.abs(result.jac))), message=str(result.message))
