"""Shared functions of the one SSL experiment analysis (2026-11-18). CPU, numpy/scipy/sklearn only (no torch).

Every function that reproduces the method of record is a copy of the frozen repository code (source named per
function). The frozen L-SML is loaded from spectral_utils/fusion_utils.py by file path, so that the spectral_utils
package __init__ (which imports torch) is not executed.
"""
import importlib.util, os, sys
from pathlib import Path

import numpy as np
from scipy.special import expit, ndtri
from scipy.stats import rankdata

_here = Path(__file__).resolve()
HD_ROOT = Path(os.environ.get('HD_ROOT') or (_here.parents[2] if (_here.parents[2] / 'spectral_utils').exists() else '/home/user/ssl_exp/repo'))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, HD_ROOT / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


FU = _load('ssl_fusion_utils', 'spectral_utils/fusion_utils.py')

TOK11 = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level',
         'energy_innovation', 'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0']
STEP5 = ['realized_z', 'realized_drv', 'digit_alternative', 'digit_spread', 'digit_alternative_innovation']
N16 = TOK11 + STEP5
DIG = N16[13:]
SUBSETS = ('gsm8k', 'math', 'olympiadbench', 'omnimath')


# ------------------------------------------------------------------ evaluate_collection.py (prm_boost_collection_v1)
def answer_z(s, off):
    """Per-answer loop, verbatim (GAP is sensitive to 1e-16 changes)."""
    out = np.zeros(len(s))
    for a, b in zip(off[:-1], off[1:]):
        v = s[a:b]; out[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
    return out


def gap_counts(z, off):
    k = []
    for a, b in zip(off[:-1], off[1:]):
        m = b - a
        if m == 1: k.append(1); continue
        if m == 0: k.append(0); continue
        zz = np.sort(z[a:b])[::-1]; km = min(int(np.ceil(m / 2)), m - 1)
        k.append(int(np.argmax(zz[:km] - zz[1:km + 1]) + 1))
    return np.array(k)


def topk_flags(s, off, k):
    k = np.broadcast_to(np.asarray(k, int), (len(off) - 1,))
    f = np.zeros(off[-1], bool)
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        kk = int(min(max(k[j], 0), b - a))
        if kk: f[a + np.argsort(-s[a:b], kind='stable')[:kk]] = True
    return f


def gap_flags(s, off):
    """stage7_hygiene.gap_flags."""
    z = answer_z(s, off)
    return topk_flags(z, off, gap_counts(z, off))


def official_from_counts(C):
    """PRMBench PRMScore (0-100) from pooled [tp, fp, tn, fn] (positive = valid step)."""
    tp, fp, tn, fn = np.moveaxis(np.asarray(C, float), -1, 0)
    f1p = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    f1n = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn), where=(2 * tn + fp + fn) > 0)
    return 50 * (f1p + f1n)


def counts_by_answer(err, flags, keep, off):
    y_valid = ~err; p_valid = ~flags; m = keep
    cols = np.column_stack([y_valid & p_valid & m, ~y_valid & p_valid & m, ~y_valid & ~p_valid & m,
                            y_valid & ~p_valid & m]).astype(np.int64)
    C = np.add.reduceat(cols, off[:-1], axis=0)
    C[np.diff(off) == 0] = 0
    return C


def prmscore(err, flags, keep, off):
    return float(official_from_counts(counts_by_answer(err, flags, keep, off).sum(0)))


# ------------------------------------------------------------------ c4u_competitors.py
def answer_z_masked(x, off, m):
    z = np.zeros(len(x))
    for a, b in zip(off[:-1], off[1:]):
        ii = np.flatnonzero(m[a:b]) + a
        if not len(ii): continue
        xs = x[ii]; mu = xs.mean(); sd = xs.std()
        z[ii] = 0 if sd <= 1e-12 else (xs - mu) / sd
    return z


def rank_gauss(x, mask=None):
    out = np.zeros(len(x)); m = np.isfinite(x) if mask is None else (mask & np.isfinite(x))
    r = rankdata(x[m]); out[m] = ndtri((r - .5) / m.sum())
    return out


def r0_cols(raw16, act, off):
    """R0 = per-answer z of each raw channel; digit channels over digit-active steps only (0 elsewhere)."""
    return np.column_stack([answer_z_masked(raw16[:, j], off, act[:, DIG.index(n)] if n in DIG else np.ones(len(raw16), bool))
                            for j, n in enumerate(N16)])


def g16_cols(raw16, act):
    """G16 = global rank -> N(0,1), digit channels over active steps (0 elsewhere). No per-answer z."""
    return np.column_stack([rank_gauss(raw16[:, j], act[:, DIG.index(n)] if n in DIG else None) for j, n in enumerate(N16)])


def pb_predict(flags, off):
    pred = np.full(len(off) - 1, -1)
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        f = np.flatnonzero(flags[a:b])
        if len(f): pred[j] = int(f[0])
    return pred


def pb_metrics(pred, lab, sub):
    out = {}
    for s in SUBSETS:
        e = (sub == s) & (lab != -1); k = (sub == s) & (lab == -1)
        ea = 100 * float((pred[e] == lab[e]).mean()); ca = 100 * float((pred[k] == -1).mean())
        out[s] = dict(TPR=ea, TNR=ca, F1=2 * ea * ca / (ea + ca) if ea + ca else 0., SLA=ea)
    out['average F1'] = float(np.mean([out[s]['F1'] for s in SUBSETS]))
    out['macro SLA'] = float(np.mean([out[s]['SLA'] for s in SUBSETS]))
    out['error acc (macro)'] = float(np.mean([out[s]['TPR'] for s in SUBSETS]))
    out['clean acc (macro)'] = float(np.mean([out[s]['TNR'] for s in SUBSETS]))
    return out


# ------------------------------------------------------------------ L-SML (stage8_auto_forced_k.fit + c4u_t anchor)
def lsml(X, anchor=None, groups=None):
    """Frozen L-SML: lsml_continuous_guarded on the columns, sign by corr with the anchor (default: channel mean)."""
    s, d = FU.lsml_continuous_guarded(*[X[:, j] for j in range(X.shape[1])], groups=groups)
    s = np.asarray(s, float); ref = X.mean(1) if anchor is None else anchor
    return (-s if np.corrcoef(s, ref)[0, 1] < 0 else s), d


def lsml_fit_apply(X_fit, X_apply, anchor_fit=None):
    """L-SML fitted on X_fit, applied to X_apply with the same weights (used for the frozen test transfer).
    Within-group weights and cross weights from the fit; applied as a fixed linear map of the standardized columns."""
    s_fit, d = lsml(X_fit, anchor_fit)
    W = linear_weights(X_fit, s_fit)
    mu, sd = X_fit.mean(0), X_fit.std(0); sd[sd == 0] = 1
    return ((X_apply - mu) / sd) @ W, W, d


def linear_weights(X, s):
    """Least-squares weights w with ((X - mu) / sd) @ w ~= s (exact when s is a linear map of the columns)."""
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Z = (X - mu) / sd
    w, *_ = np.linalg.lstsq(Z, s - s.mean(), rcond=None)
    return w


# ------------------------------------------------------------------ readouts
def soft_topn_weights(N, topk, tau=1.0):
    """Weights over the sorted token positions i = 1..topk: sigmoid((N + .5 - i) / tau). N continuous >= 1."""
    i = np.arange(1, topk + 1)
    return expit((N + .5 - i) / tau)


def soft_topn(top, N, tau=1.0):
    """top: (S, C, K) token values sorted high -> low, NaN padded. N: scalar or (C,). Returns (S, C)."""
    S, C, K = top.shape
    N = np.broadcast_to(np.asarray(N, float), (C,))
    W = np.stack([soft_topn_weights(n, K, tau) for n in N])          # (C, K)
    ok = np.isfinite(top)
    w = np.where(ok, W[None], 0.)
    num = np.where(ok, top, 0.) * w
    den = w.sum(-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return num.sum(-1) / den


def fill_empty_answer_mean(V, off, nonempty):
    """Empty steps (PRMBench only) -> the answer's mean of that column (as build_bank.py)."""
    V = V.copy()
    for a, b in zip(off[:-1], off[1:]):
        ne = nonempty[a:b]
        if ne.all() or not ne.any():
            if not ne.any(): V[a:b] = 0.
            continue
        V[a:b][~ne] = np.nanmean(V[a:b][ne], 0)
    return V


# ------------------------------------------------------------------ answer statistics
def simes(sg, off, ref=None):
    """Simes per answer. Step p = share of the reference steps with score >= s_i (ref default: all steps)."""
    srt = np.sort(sg if ref is None else ref); n_all = len(srt)
    p = (n_all - np.searchsorted(srt, sg, side='left')) / n_all
    p = np.maximum(p, 1. / n_all)
    out = np.ones(len(off) - 1)
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        n = b - a
        if n == 0: continue
        q = np.sort(p[a:b]); out[j] = float(np.min(n * q / np.arange(1, n + 1)))
    return np.minimum(out, 1.)


def expand(x, off):
    return np.repeat(x, np.diff(off))


def auroc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0: return np.nan
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------ L-SML as a fixed linear map (fit on a pool, apply to others)
def lsml_linear(X, anchor=None):
    """Fit the frozen L-SML on X; return (s, W, meta) with s == X @ W (L-SML is linear in its input columns:
    within-group fused = X[:, idx] @ w_g, cross = virtual @ cross_w; dropped constant views get 0)."""
    s, d = lsml(X, anchor)
    W = np.zeros(X.shape[1]); cw = np.asarray(d['cross_weights'], float)
    for g, (idx, w) in enumerate(d['group_weights']):
        W[np.asarray(idx, int)] += cw[g] * np.asarray(w, float)
    t = X @ W
    if np.corrcoef(t, s)[0, 1] < 0: W = -W; t = -t
    err = float(np.abs(t - s).max() / max(np.abs(s).max(), 1e-12))
    assert err < 1e-8, f'L-SML is not reproduced by its linear map (rel err {err})'
    groups = [[int(i) for i in idx] for idx, _ in d['group_weights']]
    return s, W, dict(K=int(d['K']), groups=groups, dropped=list(map(int, d.get('dropped_constant', []))))


class RankGauss:
    """G16 transform fitted on a reference pool: value -> ndtri((mid-rank in the reference - .5) / n_ref).
    On the reference itself this equals rank_gauss (ties averaged). Digit channels: active steps only, 0 elsewhere."""

    def __init__(self, raw16, act):
        self.ref = []
        for j, n in enumerate(N16):
            m = act[:, DIG.index(n)] if n in DIG else np.ones(len(raw16), bool)
            self.ref.append(np.sort(raw16[m & np.isfinite(raw16[:, j]), j]))

    def __call__(self, raw16, act):
        out = np.zeros(raw16.shape)
        for j, n in enumerate(N16):
            m = act[:, DIG.index(n)] if n in DIG else np.ones(len(raw16), bool)
            r = self.ref[j]; x = raw16[m, j]
            mid = (np.searchsorted(r, x, 'left') + np.searchsorted(r, x, 'right')) / 2.     # = rank - .5 for ref values
            out[m, j] = ndtri(np.clip(mid / len(r), .5 / len(r), 1 - .5 / len(r)))
        return out


class ECDF:
    """Per-channel ECDF fitted on reference step values; digit channels over active steps, inactive -> 0."""

    def __init__(self, raw16, act):
        self.ref = []
        for j, n in enumerate(N16):
            m = act[:, DIG.index(n)] if n in DIG else np.ones(len(raw16), bool)
            self.ref.append(np.sort(raw16[m, j]))

    def __call__(self, raw16, act):
        out = np.zeros(raw16.shape)
        for j, n in enumerate(N16):
            m = act[:, DIG.index(n)] if n in DIG else np.ones(len(raw16), bool)
            out[m, j] = np.searchsorted(self.ref[j], raw16[m, j], 'right') / len(self.ref[j])
        return out


# ------------------------------------------------------------------ change point (pb_first_error_rule_v1/cp_rule.py + gsp_ablation pb_cp.py)
H_GRID = np.linspace(.01, .6, 60)


def pava(y, w):
    y = list(map(float, y)); w = list(map(float, w)); blocks = []
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            y2, w2, n2 = blocks.pop(); y1, w1, n1 = blocks.pop()
            blocks.append([(y1 * w1 + y2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([[b[0]] * b[2] for b in blocks])


def qbins(x, B, edges=None):
    """Quantile bins (per column). edges: fitted edges (list per column) to apply to new data."""
    x = x[:, None] if x.ndim == 1 else x
    out = np.full(x.shape, -1, np.int64); E = []
    for j in range(x.shape[1]):
        v = x[:, j]; ok = np.isfinite(v)
        e = np.unique(np.quantile(v[ok], np.linspace(0, 1, B + 1)[1:-1])) if edges is None else edges[j]
        out[ok, j] = np.searchsorted(e, v[ok], side='right'); E.append(e)
    return out, E


def hist_llr(bins, w1, B, alpha=1., w0=None, mono=True):
    lam = np.zeros(len(bins)); w0 = 1 - w1 if w0 is None else w0; tables = []
    for j in range(bins.shape[1]):
        b = bins[:, j]; ok = b >= 0
        c1 = np.bincount(b[ok], w1[ok], B) + alpha; c0 = np.bincount(b[ok], w0[ok], B) + alpha
        r = np.log(c1 / c1.sum()) - np.log(c0 / c0.sum())
        if mono: r = pava(r, c0 + c1)
        lam[ok] += r[b[ok]]; tables.append(r)
    return lam, tables


def cp_llrs(bins, w1, w2, B):
    w0 = np.clip(1 - w1 - w2, 0, 1)
    l1, t1 = hist_llr(bins, w1, B, w0=w0); l2, t2 = hist_llr(bins, w2, B, w0=w0)
    return l1, l2, (t1, t2)


def log_prior(n, pi, h):
    t = np.arange(n); g = np.log(h) + t * np.log1p(-h)
    g = g - np.logaddexp.reduce(g)
    return np.r_[np.log1p(-pi) + g, np.log(pi)]


def cp_posterior(l1, l2, off, pi, h):
    """3-state chain (model 'first'): L_t = l1_t + sum_{s>t} l2_s. pi: scalar or per-answer P(tau = inf)."""
    pi = np.broadcast_to(np.clip(np.asarray(pi, float), 1e-3, 1 - 1e-3), (len(off) - 1,))
    P, w1, w2 = [], np.zeros(len(l1)), np.zeros(len(l1))
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        n = b - a
        L = l1[a:b] + np.r_[np.cumsum(l2[a:b][::-1])[::-1][1:], 0.]
        lp = log_prior(n, pi[j], h) + np.r_[L, 0.]
        p = np.exp(lp - np.logaddexp.reduce(lp)); P.append(p)
        w1[a:b] = p[:-1]; w2[a:b] = np.r_[0., np.cumsum(p[:-1])[:-1]]
    return P, w1, w2


def cp_first_pass(flag, off):
    w1 = np.zeros(len(flag)); w2 = np.zeros(len(flag))
    for a, b in zip(off[:-1], off[1:]):
        f = np.flatnonzero(flag[a:b])
        if len(f): w1[a + f[0]] = 1; w2[a + f[0] + 1:b] = 1
    return w1, w2


def cp_free_init(m, off, frac=.5):
    mx = np.array([m[a:b].max() for a, b in zip(off[:-1], off[1:])]); q = np.quantile(mx, 1 - frac)
    return cp_first_pass(m >= q, off)


def cp_h_from(w1, off):
    tau = [np.flatnonzero(w1[a:b])[0] for a, b in zip(off[:-1], off[1:]) if w1[a:b].any()]
    lens = [b - a for a, b in zip(off[:-1], off[1:]) if w1[a:b].any()]
    best, bh = -np.inf, .15
    for h in H_GRID:
        v = sum(np.log(h) + t * np.log1p(-h) - np.log1p(-(1 - h) ** n) for t, n in zip(tau, lens))
        if v > best: best, bh = v, h
    return float(bh)


def cp_fit(s, off, pi, B=20, iters=30):
    """Free EM (gsp_ablation pb_cp.em with free_init and h_from), with the prior P(tau = inf) HELD FIXED at pi
    (scalar or per answer): pi is never re-estimated. Returns the emission tables (bin edges + llr tables) and h."""
    bins, edges = qbins(s, B)
    w1, w2 = cp_free_init(s, off); h = cp_h_from(w1, off)
    for _ in range(iters):
        l1, l2, tabs = cp_llrs(bins, w1, w2, B)
        P, w1, w2 = cp_posterior(l1, l2, off, pi, h)
    l1, l2, tabs = cp_llrs(bins, w1, w2, B)
    return dict(edges=edges, t1=tabs[0][0], t2=tabs[1][0], h=h, B=B)


def cp_predict(fit, s, off, pi):
    b, _ = qbins(s, fit['B'], fit['edges']); b = b[:, 0]
    l1 = fit['t1'][b]; l2 = fit['t2'][b]
    P, _, _ = cp_posterior(l1, l2, off, pi, fit['h'])
    return np.array([-1 if int(np.argmax(np.log(np.maximum(p, 1e-300)))) == len(p) - 1 else int(np.argmax(p)) for p in P])


# ------------------------------------------------------------------ gate: BBE (Garg et al. 2021) + isotonic PU posterior
def bbe(s_pos, s_unl, delta=.1, gamma=.01):
    """Best-bin estimate of the share of the positive component in the unlabeled sample. Higher score = positive."""
    s_pos = np.asarray(s_pos, float); s_unl = np.asarray(s_unl, float)
    nP, nU = len(s_pos), len(s_unl)
    cut = np.unique(np.r_[s_pos, s_unl])
    qP = 1 - np.searchsorted(np.sort(s_pos), cut, 'left') / nP          # share with score >= c
    qU = 1 - np.searchsorted(np.sort(s_unl), cut, 'left') / nU
    ok = qP > 0
    eps = np.sqrt(np.log(4 / delta) / (2 * nU)) + np.sqrt(np.log(4 / delta) / (2 * nP))
    obj = np.where(ok, qU / np.where(ok, qP, 1) + (1 + gamma) * eps / np.where(ok, qP, 1), np.inf)
    i = int(np.argmin(obj))
    return float(min(1., qU[i] / qP[i])), dict(cut=float(cut[i]), qP=float(qP[i]), qU=float(qU[i]))


def pu_posterior(s_pos, s_unl, pi_hat, s_eval):
    """P(wrong) = min(1, pi_hat * f_P/f_U) with f_P/f_U = g/(1-g) from an isotonic fit of 'positive' on the score,
    balanced weights (so n_U/n_P = 1 in the ratio)."""
    from sklearn.isotonic import IsotonicRegression
    x = np.r_[s_pos, s_unl]; y = np.r_[np.ones(len(s_pos)), np.zeros(len(s_unl))]
    w = np.r_[np.full(len(s_pos), .5 / len(s_pos)), np.full(len(s_unl), .5 / len(s_unl))]
    g = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds='clip').fit(x, y, sample_weight=w).predict(s_eval)
    return np.minimum(1., pi_hat * g / (1 - g))


# ------------------------------------------------------------------ checks
def c2st(Xa, Xb, ga, gb, folds=5, seed=0):
    """Classifier two-sample test: logistic regression (L2, default C), grouped K-fold; returns out-of-fold AUROC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    X = np.r_[Xa, Xb]; y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]; g = np.r_[ga, gb]
    p = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return auroc(y.astype(bool), p)


def balance_by_strata(sa, sb, rng):
    """Indices of a and b subsampled to equal counts inside each stratum (strata labels sa, sb)."""
    ia, ib = [], []
    for v in np.unique(np.r_[sa, sb]):
        a = np.flatnonzero(sa == v); b = np.flatnonzero(sb == v); n = min(len(a), len(b))
        if n: ia.append(rng.choice(a, n, replace=False)); ib.append(rng.choice(b, n, replace=False))
    return np.concatenate(ia), np.concatenate(ib)


def offdiag(C):
    iu = np.triu_indices(C.shape[0], 1)
    return C[iu]


def spearman_matrix(X):
    R = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])
    with np.errstate(invalid='ignore'):
        return np.nan_to_num(np.corrcoef(R.T))


def weighted_corr_from_sums(n, S1, S2):
    """Pearson matrices from totals: n (D,), S1 (D, p), S2 (D, p, p). Returns (D, p, p)."""
    mu = S1 / n[:, None]
    cov = S2 / n[:, None, None] - mu[:, :, None] * mu[:, None, :]
    sd = np.sqrt(np.maximum(np.einsum('dii->di', cov), 1e-300))
    return cov / sd[:, :, None] / sd[:, None, :]


def boot_corr(X, grp_idx, W):
    """Bootstrap Pearson matrices of X (n, p) with rows weighted by the group draw counts.
    grp_idx: group index of each row; W: (D, G) draw counts. Returns (D, p, p); row 0 = point estimate."""
    G = W.shape[1]; p = X.shape[1]
    n_g = np.bincount(grp_idx, minlength=G).astype(float)
    S1_g = np.zeros((G, p)); np.add.at(S1_g, grp_idx, X)
    S2_g = np.zeros((G, p * p)); np.add.at(S2_g, grp_idx, np.einsum('ni,nj->nij', X, X).reshape(len(X), -1))
    n = W @ n_g; S1 = W @ S1_g; S2 = (W @ S2_g).reshape(-1, p, p)
    return weighted_corr_from_sums(n, S1, S2)


def cond_abs_corr(Xr_w, gi_w, Xr_c, gi_c, W):
    """Conditional mean |corr| (answer 29): ranks (Spearman), mean |offdiag| inside wrong and inside clean,
    primary = mean of the two. Returns dict of arrays over draws (row 0 = point)."""
    Cw = boot_corr(Xr_w, gi_w, W); Cc = boot_corr(Xr_c, gi_c, W)
    iu = np.triu_indices(Xr_w.shape[1], 1)
    mw = np.abs(Cw[:, iu[0], iu[1]]).mean(1); mc = np.abs(Cc[:, iu[0], iu[1]]).mean(1)
    return dict(wrong=mw, clean=mc, primary=(mw + mc) / 2)


def ranks(X):
    return np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])


def ci(v):
    return [float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))]
