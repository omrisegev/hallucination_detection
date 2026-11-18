"""Bayesian first-error (change-point) rule for ProcessBench.

Per answer with steps s = 1..n and evidence vectors x_s (d channels). Latent first-error step tau in {1..n} or
tau = inf (no error). Two observation models share one form:

    log p(x | tau = t) - log p(x | tau = inf) = L_t = sum_{s in A(t)} lam_s,   lam_s = sum_j log f1j(x_sj) / f0j(x_sj)

    LEVEL (default): A(t) = {s >= t}  steps before tau are 'clean' (f0), steps from tau on are 'after error' (f1)
    DROP  (MtG):     A(t) = {t}       only the first-error step differs; x = first difference of the evidence

Channels are conditionally independent given the state (naive Bayes = Dawid-Skene across channels).
Prior: P(tau = inf) = pi; P(tau = t) = (1 - pi) h (1-h)^(t-1) / sum_{u<=n} h (1-h)^(u-1)  (truncated geometric).
Posterior: P(tau = t | x) ∝ prior(t) exp(L_t);  decision = MAP over {1..n, inf}. MAP is the Bayes rule for the
ProcessBench per-answer exact-match loss (predict the first error index, or -1).
Fit: EM on ALL answers without labels. Emissions are histograms on per-channel global quantile bins (no
distribution assumption); pi and h are updated in closed form (h by 1-D grid on the expected log prior).
"""
import numpy as np

H_GRID = np.linspace(.01, .6, 60)
MONO = True        # monotone likelihood ratio: higher risk bin never lowers log f1/f0 (identifiability constraint)


def pava(y, w):
    """Weighted isotonic (non-decreasing) regression."""
    y = list(map(float, y)); w = list(map(float, w)); blocks = []
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            y2, w2, n2 = blocks.pop(); y1, w1, n1 = blocks.pop()
            blocks.append([(y1 * w1 + y2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([[b[0]] * b[2] for b in blocks])


def qbins(x, B):
    """Per-channel quantile bin index (0..B-1); NaN -> -1 (ignored)."""
    out = np.full(x.shape, -1, np.int64)
    for j in range(x.shape[1]):
        v = x[:, j]; ok = np.isfinite(v)
        if ok.sum() < 2: continue
        e = np.unique(np.quantile(v[ok], np.linspace(0, 1, B + 1)[1:-1]))
        out[ok, j] = np.searchsorted(e, v[ok], side='right')
    return out


def hist_llr(bins, w1, B, alpha=1., w0=None):
    """lam_s = log f1/f0 from weighted histograms; w1 = P(state 1), w0 = P(state 0) (default 1 - w1)."""
    lam = np.zeros(len(bins)); w0 = 1 - w1 if w0 is None else w0
    for j in range(bins.shape[1]):
        b = bins[:, j]; ok = b >= 0
        c1 = np.bincount(b[ok], w1[ok], B) + alpha; c0 = np.bincount(b[ok], w0[ok], B) + alpha
        r = np.log(c1 / c1.sum()) - np.log(c0 / c0.sum())
        if MONO:
            r = pava(r, c0 + c1)
        lam[ok] += r[b[ok]]
    return lam


def log_prior(n, pi, h):
    t = np.arange(n); g = np.log(h) + t * np.log1p(-h)
    g = g - np.logaddexp.reduce(g)
    return np.r_[np.log1p(-pi) + g, np.log(pi)]                 # last entry = tau inf


def scores_L(l1, l2, model):
    """L_t for one answer. FIRST: L_t = l1_t + sum_{s>t} l2_s (3-state left-to-right chain clean -> error -> after)."""
    if model == 'level':
        return np.cumsum(l1[::-1])[::-1]
    if model == 'drop':
        return l1.copy()
    tail = np.r_[np.cumsum(l2[::-1])[::-1][1:], 0.]
    return l1 + tail


def posterior(lam, off, pi, h, model='level', lam2=None):
    """List of posterior vectors (length n+1, last = no error), P(state 1) per step, P(state 2) per step."""
    P, w1, w2 = [], np.zeros(len(lam)), np.zeros(len(lam))
    for a, b in zip(off[:-1], off[1:]):
        n = b - a
        L = scores_L(lam[a:b], None if lam2 is None else lam2[a:b], model)
        lp = log_prior(n, pi, h) + np.r_[L, 0.]
        p = np.exp(lp - np.logaddexp.reduce(lp)); P.append(p)
        if model == 'level':
            w1[a:b] = np.cumsum(p[:-1])
        else:
            w1[a:b] = p[:-1]; w2[a:b] = np.r_[0., np.cumsum(p[:-1])[:-1]]
    return P, w1, w2


def fit(x, off, model='level', B=20, iters=60, pi0=.5, h0=.15, init=None, tol=1e-7):
    """EM. x: steps x d evidence oriented so that HIGHER = MORE LIKELY AN ERROR STATE (risk). Returns dict."""
    x = np.asarray(x, float); x = x[:, None] if x.ndim == 1 else x
    bins = qbins(x, B)
    if init is None:     # start: state 1 = upper half of the mean risk
        m = np.nanmean((x - np.nanmean(x, 0)) / np.nanstd(x, 0), 1); init = (m > np.nanmedian(m)).astype(float)
    w1 = init.astype(float); w2 = np.zeros(len(w1)) if model != 'first' else init.astype(float)
    if model == 'first':
        w1 = init * .3
    pi, h = pi0, h0; ll_old = -np.inf
    n_ans = len(off) - 1; lens = np.diff(off)
    for it in range(iters):
        lam, lam2 = llrs(bins, w1, w2, B, model)
        P, w1, w2 = posterior(lam, off, pi, h, model, lam2)
        pinf = np.array([p[-1] for p in P]); pi = float(np.clip(pinf.mean(), 1e-3, 1 - 1e-3))
        # h: maximise sum_a sum_t P_a(t) log q_n(t) over the grid
        S = {}
        for p, n in zip(P, lens):
            S[n] = S.get(n, 0) + p[:-1]
        best, bh = -np.inf, h
        for hh in H_GRID:
            v = 0.
            for n, sv in S.items():
                g = np.log(hh) + np.arange(n) * np.log1p(-hh); g -= np.logaddexp.reduce(g); v += sv @ g
            if v > best: best, bh = v, hh
        h = float(bh)
        ll = 0.
        for (a, b), n in zip(zip(off[:-1], off[1:]), lens):
            L = scores_L(lam[a:b], None if lam2 is None else lam2[a:b], model)
            ll += np.logaddexp.reduce(log_prior(n, pi, h) + np.r_[L, 0.])
        if abs(ll - ll_old) < tol * abs(ll): break
        ll_old = ll
    lam, lam2 = llrs(bins, w1, w2, B, model); P, w1, w2 = posterior(lam, off, pi, h, model, lam2)
    return dict(P=P, lam=lam, lam2=lam2, pi=pi, h=h, iters=it + 1, w1=w1, w2=w2, loglik=ll, model=model)


def llrs(bins, w1, w2, B, model):
    if model != 'first':
        return hist_llr(bins, w1, B), None
    w0 = np.clip(1 - w1 - w2, 0, 1)
    return hist_llr(bins, w1, B, w0=w0), hist_llr(bins, w2, B, w0=w0)


def map_pred(P, delta=0.):
    """MAP first error (0-based) or -1. delta shifts the no-error log-odds (0 = pure MAP)."""
    out = []
    for p in P:
        lp = np.log(np.maximum(p, 1e-300)); lp[-1] += delta
        k = int(np.argmax(lp)); out.append(-1 if k == len(p) - 1 else k)
    return np.array(out)


def first_flag_pred(P, thr=.5):
    """Alternative readout: first step whose cumulative P(tau <= t) passes thr, -1 if none (median of tau)."""
    out = []
    for p in P:
        c = np.cumsum(p[:-1]); i = np.flatnonzero(c >= thr); out.append(int(i[0]) if len(i) else -1)
    return np.array(out)
