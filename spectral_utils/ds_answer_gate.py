"""Hierarchical two-coin Dawid-Skene model on binary group votes, with an answer-level clean/error state and a
position prior (pre-registration docs/experiments/PRM_DS_GATE_E1_PREREG_2026-10-29.md).

Model, for answer j with steps i (votes g_ik in {0,1}, k = 1..K groups; l = the answer's length group):
    z_j ~ Bernoulli(rho_l)                                   1 = the answer has an error
    z_j = 0:  y_ij = 0 for every step;  g_ik ~ Bernoulli(phi0_lk)          (votes that GAP forces in a clean answer)
    z_j = 1:  y_ij ~ Bernoulli(pi_b(i))  (b = relative-position bin)
              y_ij = 1: g_ik ~ Bernoulli(psi_lk);   y_ij = 0: g_ik ~ Bernoulli(phi1_lk)
Groups are conditionally independent given (z, y). Under z = 0 the votes of different groups do not concentrate on the
same step; under z = 1 they agree on the error steps. That agreement is what separates the two answer states, so the
model does not need raw channel levels. EM, no label. Steps outside `mask` add nothing to the likelihood.

Outputs: r_j = P(z_j = 1 | votes), q_ij = P(y_ij = 1 | z_j = 1, votes), p_ij = r_j q_ij.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-6


def _lse(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def _bern_ll(G, theta):
    """sum_k g log theta + (1-g) log(1-theta); G (n, K) bool, theta (n, K)."""
    return np.where(G, np.log(theta), np.log1p(-theta)).sum(1)


class Fit(dict):
    __getattr__ = dict.__getitem__


def _estep(G, ans, bins, lg_step, mask, P):
    logL1 = _bern_ll(G, P['psi'][lg_step])
    logL0 = _bern_ll(G, P['phi1'][lg_step])
    logC = _bern_ll(G, P['phi0'][lg_step])
    lpi, l1pi = np.log(P['pi'][bins]), np.log1p(-P['pi'][bins])
    a1, a0 = lpi + logL1, l1pi + logL0
    logm = _lse(a1, a0)
    q = np.exp(a1 - logm)
    nA = int(ans.max()) + 1
    A1 = np.bincount(ans, weights=np.where(mask, logm, 0.), minlength=nA)
    A0 = np.bincount(ans, weights=np.where(mask, logC, 0.), minlength=nA)
    lrho, l1rho = np.log(P['rho']), np.log1p(-P['rho'])
    lg_ans = P['_lg_ans']
    b1, b0 = lrho[lg_ans] + A1, l1rho[lg_ans] + A0
    tot = _lse(b1, b0)
    r = np.exp(b1 - tot)
    ll = float(tot[P['_fit_ans']].sum())
    return r, q, ll


def _mstep(G, ans, bins, lg_step, mask, r, q, P, n_bins, n_lg):
    rs = r[ans] * mask; w1 = rs * q; w0 = rs * (1 - q); wc = (1 - r[ans]) * mask
    Gf = G.astype(float); K = G.shape[1]
    psi = np.empty((n_lg, K)); phi1 = np.empty((n_lg, K)); phi0 = np.empty((n_lg, K)); rho = np.empty(n_lg)
    for l in range(n_lg):
        m = lg_step == l
        psi[l] = (w1[m] @ Gf[m]) / max(w1[m].sum(), EPS)
        phi1[l] = (w0[m] @ Gf[m]) / max(w0[m].sum(), EPS)
        phi0[l] = (wc[m] @ Gf[m]) / max(wc[m].sum(), EPS)
        a = P['_fit_ans'] & (P['_lg_ans'] == l)
        rho[l] = r[a].mean() if a.any() else .5
    num = np.bincount(bins, weights=w1, minlength=n_bins); den = np.bincount(bins, weights=rs, minlength=n_bins)
    pi = np.where(den > EPS, num / np.maximum(den, EPS), .5)
    c = lambda x: np.clip(x, EPS, 1 - EPS)
    return dict(P, psi=c(psi), phi1=c(phi1), phi0=c(phi0), rho=c(rho), pi=c(pi))


def fit(G, off, rel_pos, mask=None, length_group=None, n_bins: int = 10, restarts: int = 4, seed: int = 0,
        iters: int = 2000, tol: float = 1e-9):
    """Fit the model. G (n_steps, K) votes (>0 = suspicious); off answer offsets; rel_pos in [0, 1] per step;
    mask: steps that enter the likelihood (default all); length_group: int per answer (default one group).
    One deterministic start (q = share of groups voting at the step, r = .5) plus `restarts` seeded random starts;
    the fit with the highest log-likelihood is returned. Label switch inside z = 1: if mean(psi) < mean(phi1), the y
    labels are swapped (psi <-> phi1, pi -> 1 - pi)."""
    G = np.asarray(G) > 0; off = np.asarray(off, np.int64); n = int(off[-1]); K = G.shape[1]
    lens = np.diff(off); ans = np.repeat(np.arange(len(lens)), lens)
    mask = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    lg_ans = np.zeros(len(lens), int) if length_group is None else np.asarray(length_group, int)
    n_lg = int(lg_ans.max()) + 1; lg_step = lg_ans[ans]
    bins = np.minimum((np.asarray(rel_pos, float) * n_bins).astype(int), n_bins - 1)
    fit_ans = np.bincount(ans, weights=mask.astype(float), minlength=len(lens)) > 0
    base = dict(_lg_ans=lg_ans, _fit_ans=fit_ans)
    rng = np.random.default_rng(seed)
    best = None
    for s in range(restarts + 1):
        if s == 0:
            q0 = G.mean(1).astype(float); r0 = np.full(len(lens), .5)
        else:
            q0 = rng.uniform(.05, .95, n); r0 = rng.uniform(.2, .8, len(lens))
        P = _mstep(G, ans, bins, lg_step, mask, r0, q0, dict(base), n_bins, n_lg)
        ll_old = -np.inf
        for it in range(iters):
            r, q, ll = _estep(G, ans, bins, lg_step, mask, P)
            if abs(ll - ll_old) <= tol * max(1., abs(ll)): break
            ll_old = ll
            P = _mstep(G, ans, bins, lg_step, mask, r, q, P, n_bins, n_lg)
        if best is None or ll > best[0]:
            best = (ll, P, it + 1, s)
    ll, P, n_it, start = best
    swapped = bool(P['psi'].mean() < P['phi1'].mean())
    if swapped:
        P = dict(P, psi=P['phi1'], phi1=P['psi'], pi=1 - P['pi'])
    r, q, ll = _estep(G, ans, bins, lg_step, mask, P)
    return Fit(r=r, q=q, p=r[ans] * q, ll=ll, iters=n_it, start=start, swapped=swapped, rho_hat=float(r[fit_ans].mean()),
               **{k: v for k, v in P.items() if not k.startswith('_')})


# ----------------------------------------------------------------------------------------------- decisions
def prmscore_from_counts(tp, fp, tn, fn):
    """PRMBench PRMScore on pooled counts, POSITIVE = valid step (official orientation)."""
    f1p = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.
    f1n = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) > 0 else 0.
    return 50 * (f1p + f1n)


def cost_ratio_threshold(flags, p, keep, h: float = 1e-3):
    """Model-derived cutoff c for calling an answer clean under PRMScore (no label).
    Expected counts of the baseline flags under the model: TP = sum (1-f)(1-p), FP = sum (1-f) p, TN = sum f p,
    FN = sum f (1-p) (positive = valid; f = flagged as error). Unflagging a flagged step turns, in expectation,
    (1-p) of an FN into a TP (gain G = dS/dTP - dS/dFN) and p of a TN into an FP (loss L = dS/dTN - dS/dFP).
    Unflagging the whole flag set F_j of an answer raises the expected score iff mean_{i in F_j} p_i < G / (G + L)."""
    f = np.asarray(flags, bool) & keep; nf = ~np.asarray(flags, bool) & keep
    C = np.array([((1 - p) * nf).sum(), (p * nf).sum(), (p * f).sum(), ((1 - p) * f).sum()])
    S = lambda c: prmscore_from_counts(*c)
    d = lambda j: (S(C + h * np.eye(4)[j]) - S(C - h * np.eye(4)[j])) / (2 * h)
    gain = d(0) - d(3); loss = d(2) - d(1)
    return dict(c=float(gain / (gain + loss)), gain=float(gain), loss=float(loss), expected_counts=C.tolist())


def mean_p_on_flags(flags, p, off, keep=None):
    """Per answer: mean p over the flagged (and kept) steps; nan if none."""
    f = np.asarray(flags, bool) & (np.ones(len(p), bool) if keep is None else keep)
    s = np.add.reduceat(np.where(f, p, 0.), off[:-1]); n = np.add.reduceat(f.astype(float), off[:-1])
    s[np.diff(off) == 0] = 0; n[np.diff(off) == 0] = 0
    return np.divide(s, n, out=np.full(len(s), np.nan), where=n > 0)


def expected_counts(p, off):
    s = np.add.reduceat(p, off[:-1]); s[np.diff(off) == 0] = 0
    return s
