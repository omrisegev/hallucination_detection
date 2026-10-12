"""Binary latent-group model and declared continuous likelihood continuation.

This fits the Jaffe et al. (2016) latent tree with fixed groups and EM, not their
exact published estimator. No labels, benchmark metrics, or I/O enter this module.
"""
from __future__ import annotations

import hashlib
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logit, logsumexp

EPS = 1e-6


def answer_standardize(x, offsets):
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    for a, b in zip(offsets[:-1], offsets[1:]):
        v = x[a:b]
        sd = v.std(axis=0)
        result[a:b] = np.divide(v - v.mean(axis=0), sd,
                               out=np.zeros_like(v), where=sd > 1e-8)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite standardized data')
    return result


def binary_tail(x, offsets):
    x = np.asarray(x, dtype=float)
    out = np.empty(x.shape, dtype=np.uint8)
    for a, b in zip(offsets[:-1], offsets[1:]):
        out[a:b] = x[a:b] > np.quantile(x[a:b], .8, axis=0)
    return out


def collapse_duplicates(binary, continuous, groups):
    """One binary classifier per within-family pattern; clones cannot add weight.

    Within a binary equivalence class, distinct continuous columns are averaged;
    exact continuous copies split their original column's contribution.
    """
    b, x, g = np.asarray(binary), np.asarray(continuous), np.asarray(groups)
    if b.shape != x.shape or b.shape[1] != len(g) or not np.isin(b, [0, 1]).all():
        raise ValueError('invalid duplicate inputs')
    buckets = {}
    for j in range(len(g)):
        key = (int(g[j]), hashlib.sha256(np.ascontiguousarray(b[:, j]).tobytes()).hexdigest())
        buckets.setdefault(key, []).append(j)
    expansion = np.zeros((len(g), len(buckets)))
    membership, reduced_groups, representatives = [], [], []
    for k, ((group, _), indices) in enumerate(buckets.items()):
        uniques = {}
        for j in indices:
            key = hashlib.sha256(np.ascontiguousarray(x[:, j]).tobytes()).hexdigest()
            uniques.setdefault(key, []).append(j)
        for cols in uniques.values():
            expansion[cols, k] = 1 / (len(uniques) * len(cols))
            for j in cols:
                np.testing.assert_array_equal(x[:, j], x[:, cols[0]])
        for j in indices:
            np.testing.assert_array_equal(b[:, j], b[:, indices[0]])
        membership.append(list(uniques.values()))
        reduced_groups.append(group)
        representatives.append(indices[0])
    return b[:, representatives], expansion, np.asarray(reduced_groups), membership


def spectral_initialization(b, groups, counts):
    n = counts.sum()
    mu = counts @ b / n
    sd = np.sqrt(mu * (1-mu))
    if np.any(sd < EPS):
        raise ValueError('inactive fitting classifier; no silent fallback')
    z = (b-mu) / sd
    r = (z.T * counts) @ z / n
    i, j = np.where(np.triu(groups[:, None] != groups[None, :], 1))
    if len(np.unique(groups)) < 3:
        raise ValueError('at least three distinct groups required')
    sizes = np.array([np.sum(groups == q) for q in groups])
    pair_weight = 1 / np.sqrt(sizes[i] * sizes[j])
    off = r * (groups[:, None] != groups[None, :])
    val, vec = np.linalg.eigh(off)
    start = vec[:, -1] * np.sqrt(max(val[-1], EPS))
    if start.sum() < 0:
        start *= -1
    def residual(v):
        return (v[i] * v[j] - r[i, j]) * pair_weight
    def jac(v):
        out = np.zeros((len(i), len(v)))
        out[np.arange(len(i)), i] = v[j] * pair_weight
        out[np.arange(len(i)), j] = v[i] * pair_weight
        return out
    fit = least_squares(residual, np.clip(start, -.98, .98), jac=jac,
                        bounds=(-.999, .999), max_nfev=1000,
                        ftol=1e-10, xtol=1e-10, gtol=1e-10)
    if not fit.success or np.linalg.matrix_rank(fit.jac) != len(mu):
        raise ValueError('off-block loading fit failed or is locally unidentified')
    v = fit.x
    if v[0] < 0:
        v *= -1
    pi = .2
    a = np.empty((len(np.unique(groups)), 2))
    theta = np.empty((len(mu), 2))
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        pg = float(np.clip(mu[idx].mean(), .05, .95))
        if len(idx) == 1:
            beta = v[idx[0]]
            theta[idx] = [0., 1.]
        else:
            ii, jj = np.triu_indices(len(idx), 1)
            cc = r[np.ix_(idx, idx)][ii, jj]
            vv = v[idx[ii]] * v[idx[jj]]
            usable = (cc > .01) & (vv > 0)
            beta = np.sqrt(np.clip(np.median(vv[usable] / cc[usable]) if usable.any() else .25, .01, .95))
            local = np.clip(v[idx] / beta, -.95, .95)
            theta[idx, 1] = mu[idx] + sd[idx] * local * np.sqrt((1-pg)/pg)
            theta[idx, 0] = mu[idx] - sd[idx] * local * np.sqrt(pg/(1-pg))
            theta[idx] = np.clip(theta[idx], .01, .99)
        a[group, 1] = pg + np.sqrt(pg*(1-pg)) * beta * np.sqrt((1-pi)/pi)
        a[group, 0] = pg - np.sqrt(pg*(1-pg)) * beta * np.sqrt(pi/(1-pi))
    return pi, np.clip(a, .01, .99), theta, {
        'offblock_relative_residual': float(np.linalg.norm(residual(v)) /
                                           max(np.linalg.norm(r[i, j]*pair_weight), EPS)),
        'loadings': v.tolist(), 'covariance': r.tolist(),
        'jacobian_condition': float(np.linalg.cond(fit.jac)),
    }


def _expectation(b, groups, pi, a, theta):
    n, ng = len(b), len(a)
    ll = np.empty((n, ng, 2))
    for g in range(ng):
        idx = np.flatnonzero(groups == g)
        if len(idx) == 1:
            obs = b[:, idx[0]]
            ll[:, g, 0] = np.where(obs == 0, 0., -np.inf)
            ll[:, g, 1] = np.where(obs == 1, 0., -np.inf)
        else:
            ll[:, g] = b[:, idx] @ (np.log(theta[idx])-np.log1p(-theta[idx])) + np.log1p(-theta[idx]).sum(0)
    logp = np.empty((n, 2))
    q = np.empty((n, ng, 2))
    for y in (0, 1):
        l0 = ll[:, :, 0] + np.log1p(-a[:, y])
        l1 = ll[:, :, 1] + np.log(a[:, y])
        norm = np.logaddexp(l0, l1)
        q[:, :, y] = np.exp(l1 - norm)
        logp[:, y] = norm.sum(1) + np.log(pi if y else 1-pi)
    norm = logsumexp(logp, axis=1)
    py = np.exp(logp - norm[:, None])
    return norm, py, q


def _penalty(pi, a, theta, groups):
    multi = np.array([np.sum(groups == g) > 1 for g in groups])
    t = theta[multi]
    return .5 * (np.log(pi) + np.log1p(-pi) + np.log(a).sum() +
                 np.log1p(-a).sum() + np.log(t).sum() + np.log1p(-t).sum())


def fit_binary_tree(binary, groups, *, max_iter=500, tolerance=1e-7):
    """Deterministic spectral initialization + MAP EM; no target labels accepted."""
    started = time.perf_counter()
    binary, groups = np.asarray(binary), np.asarray(groups, int)
    if binary.ndim != 2 or len(groups) != binary.shape[1] or not np.isin(binary, [0, 1]).all():
        raise ValueError('expected binary matrix and one group per column')
    if not np.array_equal(np.unique(groups), np.arange(len(np.unique(groups)))):
        raise ValueError('groups must be contiguous')
    b, counts = np.unique(binary, axis=0, return_counts=True)
    b, counts = b.astype(float), counts.astype(float)
    n = counts.sum()
    pi, a, theta, init = spectral_initialization(b, groups, counts)
    history = []
    converged = False
    for iteration in range(max_iter+1):
        norm, py, q = _expectation(b, groups, pi, a, theta)
        objective = float(counts @ norm + _penalty(pi, a, theta, groups))
        history.append(objective)
        if len(history) > 1:
            delta = history[-1]-history[-2]
            if delta < -1e-7 * n:
                raise ArithmeticError('MAP EM objective decreased')
            if delta/n < tolerance:
                converged = True
                break
        if iteration == max_iter:
            break
        weighted_y = counts[:, None] * py
        ny = weighted_y.sum(0)
        joint = weighted_y[:, None, :] * q
        a = np.clip((joint.sum(0)+.5)/(ny[None, :]+1), EPS, 1-EPS)
        pi = float(np.clip((ny[1]+.5)/(n+1), EPS, 1-EPS))
        palpha = joint.sum(2)
        for g in range(len(a)):
            idx = np.flatnonzero(groups == g)
            if len(idx) == 1:
                continue
            for alpha in (0, 1):
                weights = palpha[:, g] if alpha else counts-palpha[:, g]
                theta[idx, alpha] = np.clip((weights @ b[:, idx]+.5)/(weights.sum()+1), EPS, 1-EPS)
    # Local and global sign gauges: orientation is a convention, not accuracy.
    for g in range(len(a)):
        idx = np.flatnonzero(groups == g)
        if len(idx) > 1 and np.sum(theta[idx, 1]-theta[idx, 0]) < 0:
            theta[idx] = theta[idx, ::-1]
            a[g] = 1-a[g]
    _, py, _ = _expectation(b, groups, pi, a, theta)
    anchor_cov = float(counts @ ((b[:, 0] - counts @ b[:, 0]/n) * py[:, 1]))
    flipped = anchor_cov < 0
    if flipped:
        pi, a = 1-pi, a[:, ::-1]
    mu = counts @ b/n
    return {'pi': pi, 'a': a.tolist(), 'theta': theta.tolist(), 'groups': groups.tolist(),
            'mu': mu.tolist(), 'sd': np.sqrt(mu*(1-mu)).tolist(),
            'converged': converged, 'iterations': iteration, 'objective': history,
            'unique_patterns': len(b), 'rows': int(n), 'anchor_flipped': flipped,
            'initialization': init, 'seconds': time.perf_counter()-started}


def group_contributions(values, model, *, continuous=True, average_evidence=False):
    """Unnormalized risk log odds; continuous values are answer-z feature scores."""
    x = np.asarray(values, float)
    theta, a = np.asarray(model['theta']), np.asarray(model['a'])
    groups = np.asarray(model['groups'])
    h = np.asarray(model['mu']) + np.asarray(model['sd'])*x if continuous else x
    if h.ndim != 2 or h.shape[1] != len(groups) or not np.isfinite(h).all():
        raise ValueError('invalid application matrix')
    out = np.empty((len(x), len(a)))
    for g in range(len(a)):
        idx = np.flatnonzero(groups == g)
        if len(idx) == 1:
            out[:, g] = h[:, idx[0]]*(logit(a[g, 1])-logit(a[g, 0])) + np.log1p(-a[g, 1])-np.log1p(-a[g, 0])
        else:
            e = h[:, idx] @ (logit(theta[idx, 1])-logit(theta[idx, 0])) + (
                np.log1p(-theta[idx, 1])-np.log1p(-theta[idx, 0])).sum()
            if average_evidence:
                e /= len(idx)
            out[:, g] = np.logaddexp(np.log1p(-a[g, 1]), np.log(a[g, 1])+e) - np.logaddexp(np.log1p(-a[g, 0]), np.log(a[g, 0])+e)
    if not np.isfinite(out).all():
        raise ArithmeticError('nonfinite log evidence')
    return out


def score(values, model, *, continuous=True, average_evidence=False):
    return logit(model['pi']) + group_contributions(values, model, continuous=continuous,
                                                   average_evidence=average_evidence).sum(1)


def linearized_weights(model):
    theta, a = np.asarray(model['theta']), np.asarray(model['a'])
    groups, mu, sd = map(np.asarray, (model['groups'], model['mu'], model['sd']))
    w = np.empty(len(groups))
    for g in range(len(a)):
        idx = np.flatnonzero(groups == g)
        if len(idx) == 1:
            w[idx] = sd[idx]*(logit(a[g, 1])-logit(a[g, 0]))
        else:
            dw = logit(theta[idx, 1])-logit(theta[idx, 0])
            e0 = mu[idx] @ dw + (np.log1p(-theta[idx, 1])-np.log1p(-theta[idx, 0])).sum()
            w[idx] = sd[idx]*dw*(expit(logit(a[g, 1])+e0)-expit(logit(a[g, 0])+e0))
    return w
