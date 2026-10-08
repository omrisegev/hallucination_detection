"""Per-channel likelihood-ratio evidence for first-error localization (Step 432).

Idea.  The argmax over steps is invariant to any monotone transform of a single fused
score, so a per-step "how unusual" statistic computed from the fused score changes nothing.
What does change the decision is (a) a non-linear transform applied per channel BEFORE the
channels are summed, and (b) a null distribution that depends on the step's position in the
answer.  This module supplies both:

    evidence_s = sum_c  log f1_c(x_sc) / f0_c(x_sc | position bin of s)

where f1_c is the distribution of channel c on pseudo-positive steps and f0_c the
distribution on the remaining steps, estimated on training-fold answers only from smoothed
histograms on pooled-quantile edges (the construction of the Step 428 delay-floor
diagnostic).  Pseudo-positives come from a label-free seed (the argmax of an equal-weight
fusion); f0 may be estimated per relative-position bin (the label-free position prior: a
channel that drifts along the generation has a different null late in the answer).

A channel whose values are constant within an answer contributes a constant to every step and
therefore cannot move the argmax; a channel that is constant over the whole training
population has a single bin and contributes exactly zero.  Nothing here reads an error label;
the label-selected ceiling of the protocol passes true first-error indicators through the same
`fit_tables` and is labelled as such by the caller.
"""
import numpy as np

EPS = np.finfo(float).eps


def position_bins(n_steps, B=8):
    """Relative-position bin of every step of an answer with n_steps steps (a one-step answer
    is bin 0; the last step is always bin B-1)."""
    n = int(n_steps)
    if n <= 1:
        return np.zeros(n, int)
    rel = np.arange(n) / (n - 1)
    return np.minimum((rel * B).astype(int), B - 1)


def quantile_edges(x, bins=32):
    """Pooled-quantile edges with open ends; a constant input yields a single bin."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([-np.inf, np.inf])
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return np.array([-np.inf, np.inf])
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def bin_index(x, edges):
    x = np.asarray(x, float)
    k = np.searchsorted(edges, x, side='right') - 1
    k = np.clip(k, 0, len(edges) - 2)
    return np.where(np.isfinite(x), k, -1)


def smoothed_log_density(counts, alpha=0.5, prior=None, m=0.):
    """log of the smoothed bin probabilities.  `prior` (K,) with pseudo-count m shrinks a
    sparse histogram towards a pooled one; alpha is the usual additive smoothing."""
    c = np.asarray(counts, float)
    K = len(c)
    p = c + alpha
    if prior is not None and m > 0:
        p = p + m * np.asarray(prior, float)
    return np.log(p / p.sum()) if p.sum() > 0 else np.full(K, -np.log(K))


def pseudo_positive_weights(mass, rule='argmax'):
    """Per-step pseudo-positive weight in [0, 1] from a seed mass (S,)."""
    mass = np.asarray(mass, float)
    S = len(mass)
    if rule == 'argmax':
        tol = 8 * EPS * max(1., float(np.max(np.abs(mass)))) if S else 0.
        w = np.zeros(S)
        if S:
            w[int(np.flatnonzero(mass >= mass.max() - tol)[0])] = 1.
        return w
    if rule == 'mass':
        m = mass - mass.min()
        return m / m.sum() if m.sum() > 0 else np.full(S, 1. / max(S, 1))
    raise ValueError(rule)


def seed_mass_equal_softmax(profile):
    """Label-free, fit-free seed: equal-weight mean over channels of the per-channel softmax
    over steps of the answer-standardised readout (the pmf encoding of cvf_v2 without the
    population standardisation).  Non-finite entries (suffix-masked readouts) get zero mass."""
    p = np.asarray(profile, float)
    z = np.where(np.isfinite(p), p, -np.inf)
    z = z - np.max(np.where(np.isfinite(z), z, -np.inf), axis=0, keepdims=True)
    e = np.exp(z)
    e = np.where(np.isfinite(e), e, 0.)
    col = e.sum(0, keepdims=True)
    sm = np.divide(e, col, out=np.zeros_like(e), where=col > 0)
    return sm.mean(1)


def _tables_from_counts(edges, c1, c0, c0b, B, alpha, pseudo_count, extra):
    C, K = c1.shape
    log_lr = np.zeros((C, K)); log_lr_pos = np.zeros((C, B, K))
    for c in range(C):
        Kc = len(edges[c]) - 1
        if Kc < 2:
            continue
        l1 = smoothed_log_density(c1[c, :Kc], alpha)
        l0 = smoothed_log_density(c0[c, :Kc], alpha)
        log_lr[c, :Kc] = l1 - l0
        prior = np.exp(l0)
        for b in range(B):
            l0b = smoothed_log_density(c0b[c, b, :Kc], alpha, prior, pseudo_count)
            log_lr_pos[c, b, :Kc] = l1 - l0b
    padded = np.full((C, K + 1), np.inf)
    for c in range(C):
        padded[c, :len(edges[c])] = edges[c]
    return {'edges': padded, 'n_bins': np.array([len(e) - 1 for e in edges]), 'log_lr': log_lr, 'log_lr_pos': log_lr_pos,
            'count_pos': c1, 'count_neg': c0, 'count_neg_by_position': c0b, 'B': B, **extra}


def fit_tables(profiles, pos_weights, B=8, bins=32, alpha=0.5, pseudo_count=32., edges=None):
    """Estimate per-channel log-likelihood-ratio tables from a list of answers.

    profiles     : list of (S_i, C) arrays (training answers only).
    pos_weights  : list of (S_i,) pseudo-positive weights in [0, 1]; negative weight = 1 - w.
    Returns a dict with 'edges' (C, K+1) padded with +inf, 'log_lr' (C, K) for the plain null,
    'log_lr_pos' (C, B, K) for the position-conditional null (f1 pooled; f0 per position bin
    shrunk towards the pooled f0 with `pseudo_count`), and the raw counts.
    """
    C = profiles[0].shape[1]
    if edges is None:
        pooled = [np.concatenate([p[:, c] for p in profiles]) for c in range(C)]
        edges = [quantile_edges(x, bins) for x in pooled]
    K = max(len(e) - 1 for e in edges)
    c1 = np.zeros((C, K)); c0 = np.zeros((C, K)); c0b = np.zeros((C, B, K))
    for p, w in zip(profiles, pos_weights):
        w = np.asarray(w, float)
        pb = position_bins(len(w), B)
        for c in range(C):
            k = bin_index(p[:, c], edges[c])
            ok = k >= 0
            np.add.at(c1[c], k[ok], w[ok])
            np.add.at(c0[c], k[ok], (1 - w)[ok])
            np.add.at(c0b[c], (pb[ok], k[ok]), (1 - w)[ok])
    extra = {'positive_weight_total': float(sum(np.sum(w) for w in pos_weights)), 'answers': len(profiles)}
    return _tables_from_counts(edges, c1, c0, c0b, B, alpha, pseudo_count, extra)


def fit_tables_flat(X, w, pbin, train_steps, B=8, bins=32, alpha=0.5, pseudo_count=32., edges=None):
    """Vectorised `fit_tables` on a flat step matrix.  X (N, C) every step of the population,
    w (N,) pseudo-positive weights, pbin (N,) position bins, train_steps the indices of the
    training-fold steps (edges and counts use those only)."""
    X = np.asarray(X, float); w = np.asarray(w, float); pbin = np.asarray(pbin, int)
    Xt = X[train_steps]; wt = w[train_steps]; pt = pbin[train_steps]
    C = X.shape[1]
    if edges is None:
        edges = [quantile_edges(Xt[:, c], bins) for c in range(C)]
    K = max(len(e) - 1 for e in edges)
    c1 = np.zeros((C, K)); c0 = np.zeros((C, K)); c0b = np.zeros((C, B, K))
    for c in range(C):
        Kc = len(edges[c]) - 1
        k = bin_index(Xt[:, c], edges[c]); ok = k >= 0
        c1[c, :Kc] = np.bincount(k[ok], weights=wt[ok], minlength=Kc)[:Kc]
        c0[c, :Kc] = np.bincount(k[ok], weights=(1 - wt)[ok], minlength=Kc)[:Kc]
        c0b[c, :, :Kc] = np.bincount(pt[ok] * Kc + k[ok], weights=(1 - wt)[ok], minlength=B * Kc)[:B * Kc].reshape(B, Kc)
    extra = {'positive_weight_total': float(wt.sum()), 'training_steps': int(len(train_steps))}
    return _tables_from_counts(edges, c1, c0, c0b, B, alpha, pseudo_count, extra)


def evidence_flat(X, tables, pbin, conditional=False):
    """(N, C) per-channel log-likelihood ratios for a flat step matrix."""
    X = np.asarray(X, float); pbin = np.asarray(pbin, int)
    N, C = X.shape
    out = np.zeros((N, C))
    for c in range(C):
        Kc = int(tables['n_bins'][c])
        if Kc < 2:
            continue
        k = bin_index(X[:, c], tables['edges'][c, :Kc + 1]); ok = k >= 0
        out[ok, c] = tables['log_lr_pos'][c, pbin[ok], k[ok]] if conditional else tables['log_lr'][c, k[ok]]
    return out


def evidence_matrix(profile, tables, conditional=False):
    """(S, C) per-channel log-likelihood ratios for one answer."""
    p = np.asarray(profile, float)
    S, C = p.shape
    out = np.zeros((S, C))
    pb = position_bins(S, tables['B'])
    for c in range(C):
        Kc = int(tables['n_bins'][c])
        if Kc < 2:
            continue
        k = bin_index(p[:, c], tables['edges'][c, :Kc + 1])
        ok = k >= 0
        if conditional:
            out[ok, c] = tables['log_lr_pos'][c, pb[ok], k[ok]]
        else:
            out[ok, c] = tables['log_lr'][c, k[ok]]
    return out


def evidence(profile, tables, conditional=False, channel_weights=None):
    """(S,) evidence mass = weighted sum over channels of the log-likelihood ratios."""
    m = evidence_matrix(profile, tables, conditional)
    w = np.ones(m.shape[1]) if channel_weights is None else np.asarray(channel_weights, float)
    return m @ w


def position_prior_only(n_steps, tables):
    """(S,) the position-conditional null alone with a flat f1: sum over channels of
    -log f0_c(position) averaged over bins.  Used as the 'prior alone' control."""
    S = int(n_steps); pb = position_bins(S, tables['B'])
    C = tables['log_lr'].shape[0]
    out = np.zeros(S)
    for c in range(C):
        Kc = int(tables['n_bins'][c])
        if Kc < 2:
            continue
        # log f1 - log f0_b averaged over the bins of f1's own distribution = E_f1[log LR_b]
        p1 = np.exp(smoothed_log_density(tables['count_pos'][c, :Kc], 0.5))
        out += (tables['log_lr_pos'][c][:, :Kc] * p1[None, :]).sum(1)[pb]
    return out


def effective_channels(evidence_mat):
    """Number of channels whose evidence varies within the answer (can move the argmax)."""
    m = np.asarray(evidence_mat, float)
    return int(np.sum(np.ptp(m, axis=0) > 1e-12)) if len(m) else 0


def seed_agreement(pred_seed, pred_evidence):
    a = np.asarray(pred_seed); b = np.asarray(pred_evidence)
    return float(np.mean(a == b)) if len(a) else np.nan
