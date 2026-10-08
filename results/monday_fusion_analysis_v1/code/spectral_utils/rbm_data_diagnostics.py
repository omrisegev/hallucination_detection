"""Saved-state diagnostics only. No fitting, label-driven orientation or selection."""
import hashlib
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

STRATA = ('all', 'entropy_low', 'entropy_high', 'short', 'long', 'early', 'late')
RESIDUAL_NAMES = ('mean_abs_correlation', 'largest_eigenvalue', 'top_eigen_share',
                  'mean_variance', 'mean_squared_residual_mean')


def first_near_max(s, sd_fraction=.25):
    # Verbatim numerical rule from Claude/provenance_readout.py. Whole-answer
    # invalidity is preserved by the caller, rather than filling missing scores.
    s = np.asarray(s, float)
    if not len(s):
        return s.copy()
    mx, sd = s.max(), s.std()
    near = np.flatnonzero(s >= mx - sd_fraction * sd)
    out = s.copy()
    eps = 1e-6 * max(sd, 1e-12)
    for rank, k in enumerate(near):
        out[k] = mx + eps * (len(near) - rank)
    return out


def step_means(x, spans):
    x = np.asarray(x, float)
    return np.array([x[a:b].mean(axis=0) if b > a else
                     np.full(x.shape[1:], np.nan) for a, b in spans])


def strata(entropy, lengths):
    n = len(lengths)
    return np.array([np.ones(n, bool), entropy <= np.median(entropy),
                     entropy > np.median(entropy), lengths <= np.median(lengths),
                     lengths > np.median(lengths), np.arange(n) < (n+1)//2,
                     np.arange(n) >= (n+1)//2])


def auc_columns(x, y):
    """Step labels: error=1, correct=0, unknown excluded; no sign selection."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    valid = np.isin(y, [0, 1]) & np.isfinite(x).all(axis=1)
    x, y = x[valid], np.asarray(y)[valid]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        return np.full(x.shape[1], np.nan), (n0, n1)
    ranks = rankdata(x, axis=0)
    return (ranks[y == 1].sum(axis=0) - n1*(n1+1)/2) / (n1*n0), (n0, n1)


def class_diagnostics(x, labels, masks):
    p = x.shape[1]
    # Difference, log variance ratio, both actual variances, counts, zero flags.
    mean_delta = np.full((len(masks), p), np.nan)
    log_ratio = mean_delta.copy()
    variance = np.full((len(masks), 2, p), np.nan)
    means = variance.copy()
    counts = np.zeros((len(masks), 2), int)
    zero = np.zeros((len(masks), 2, p), bool)
    for j, mask in enumerate(masks):
        groups = [x[mask & (labels == c)] for c in (0, 1)]
        groups = [g[np.isfinite(g).all(axis=1)] for g in groups]
        counts[j] = [len(g) for g in groups]
        if min(counts[j]) < 2:
            continue
        means[j] = [g.mean(axis=0) for g in groups]
        # Sample variance; these are step averages, not token-level truth labels.
        variance[j] = [g.var(axis=0, ddof=1) for g in groups]
        zero[j] = variance[j] <= 1e-12
        mean_delta[j] = means[j, 1] - means[j, 0]
        usable = ~zero[j].any(axis=0)
        log_ratio[j, usable] = np.log(variance[j, 1, usable] / variance[j, 0, usable])
    return dict(class_mean_delta=mean_delta, class_log_variance_ratio=log_ratio,
                class_variance=variance, class_means=means, class_counts=counts,
                class_zero_variance=zero)


def residuals(x, a, w, b):
    return x - a - expit(b + x @ w)[:, None] * w


def residual_summary(r):
    cov = np.atleast_2d(np.cov(r, rowvar=False, ddof=1))
    var = np.diag(cov)
    keep = var > 1e-12
    corr = np.full_like(cov, np.nan)
    eigen = np.full(len(var), np.nan)
    if keep.sum() >= 2:
        c = cov[np.ix_(keep, keep)] / np.sqrt(np.outer(var[keep], var[keep]))
        corr[np.ix_(keep, keep)] = c
        e = np.linalg.eigvalsh(c)[::-1]
        eigen[:len(e)] = e
        off = np.abs(c[~np.eye(len(c), dtype=bool)]).mean()
        values = [off, e[0], e[0]/e.sum(), var.mean(), np.mean(r.mean(axis=0)**2)]
    else:
        values = [np.nan]*3 + [var.mean(), np.mean(r.mean(axis=0)**2)]
    return np.asarray(values), corr, eigen


def seed_for(uid, bank, replica, purpose):
    return int.from_bytes(hashlib.sha256(
        f'rbm-diagnostics-v1|{uid}|{bank}|{replica}|{purpose}'.encode()).digest()[:8], 'little')


def sample_model(n, a, w, b, rng):
    pi = expit(b + a @ w + .5*(w @ w))
    h = rng.random(n) < pi
    return a + h[:, None]*w + rng.normal(size=(n, len(w)))


def lag_correlations(x, masks):
    out = np.full((len(masks), 3, x.shape[1]), np.nan)
    counts = np.zeros((len(masks), 3), int)
    for j, mask in enumerate(masks):
        for lag in (1, 2, 3):
            # Actual chronological gaps, never compress a stratum's subsequence.
            good = mask[:-lag] & mask[lag:]
            counts[j, lag-1] = good.sum()
            if good.sum() < 3:
                continue
            u, v = x[:-lag][good], x[lag:][good]
            u, v = u-u.mean(axis=0), v-v.mean(axis=0)
            den = np.sqrt((u*u).sum(axis=0)*(v*v).sum(axis=0))
            np.divide((u*v).sum(axis=0), den, out=out[j, lag-1], where=den>1e-12)
    return out, counts


def diagnose_bank(z, a, w, b, spans, labels, masks, uid, bank, fusion_steps):
    sm = step_means(z, spans)
    r = residuals(z, a, w, b)
    observed, corr, eig = residual_summary(r)
    synthetic = []
    for replica in range(4):
        rng = np.random.default_rng(seed_for(uid, bank, replica, 'model'))
        x = sample_model(len(z), a, w, b, rng)
        synthetic.append(residual_summary(residuals(x, a, w, b))[0])
    sr = step_means(r, spans)
    lag, lag_counts = lag_correlations(sr, masks)
    permutations = []
    for replica in range(4):
        rng = np.random.default_rng(seed_for(uid, bank, replica, 'order'))
        # Permute within each fixed length/position/entropy stratum separately.
        # All-stratum permutes all steps; strata retain their original positions.
        tmp = np.full_like(lag, np.nan)
        for j, mask in enumerate(masks):
            shuffled = sr.copy()
            selected = np.flatnonzero(mask)
            shuffled[selected] = sr[rng.permutation(selected)]
            tmp[j] = lag_correlations(shuffled, masks[j:j+1])[0][0]
        permutations.append(tmp)
    reliability = []
    rel_counts = []
    # Fusion uses top10; feature columns use step means. This is diagnostic,
    # not a claim isolating fusion from the aggregation operation.
    augmented = np.column_stack([sm, fusion_steps])
    for mask in masks:
        auc, count = auc_columns(augmented[mask], labels[mask])
        reliability.append(auc)
        rel_counts.append(count)
    result = dict(step_means=sm, residual_observed=observed,
                  residual_synthetic=np.array(synthetic), residual_correlation=corr,
                  residual_eigenvalues=eig, lag=lag, lag_counts=lag_counts,
                  lag_permutations=np.array(permutations), reliability=np.array(reliability),
                  reliability_counts=np.array(rel_counts))
    result.update(class_diagnostics(sm, labels, masks))
    return result
