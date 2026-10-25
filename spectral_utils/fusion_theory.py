"""Theory helpers for label-free PRM boosting (docs/research_notes/math_rigor_2026-10-24/FRAMING_THEORY.md).

The latent-class view of PRM boosting: view 0 is the PRM risk, views 1..K are telemetry group scores, y is the
unobserved step error. This module holds the population formulas and the label-free estimators that the framing
note relies on:

    fused_auc_gaussian            P2: AUC of a linear fusion under Gaussian class-conditionals
    two_view_separation,
    improvement_threshold         P2: when a two-view fusion beats the PRM, as a minimum PRM share
    zero_diag_eigen_weights       P3: closed form of the shipped SML weights, v_k ∝ u_k / (lam1 + u_k^2)
    within_answer_center,
    within_answer_residualize     P1 / P5(f): answer fixed effects and observed covariates (position, length)
    anchored_loadings             P5: signal loadings identified by anchoring on view 0 (four estimators)
    gate_statistic                P5(e): G(w) = SNR(w) - SNR(view 0), free of the prevalence p
    gate_decision                 P5(g): answer-bootstrap decision fuse / prm / abstain

Labels never enter an estimator here. The functions take dense step-by-view arrays and answer offsets; they do not
read files. The unit tests (tests/test_fusion_theory.py) check every formula on synthetic data.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

ESTIMATORS = ('cumulant', 'cumulant_w01', 'rank', 'fa', 'cca')


# ----------------------------------------------------------------------------------------------- population formulas
def fused_auc_gaussian(w, delta, sigma) -> float:
    """AUC of s = w'x when x | y=1 ~ N(mu0 + delta, sigma) and x | y=0 ~ N(mu0, sigma) (P2).

    AUC = Phi( w'delta / sqrt(2 w' sigma w) ). Within-answer AUC has the same form once answer effects and observed
    covariates are removed and sigma is the within-class covariance of what remains.
    """
    w = np.asarray(w, float)
    num = float(w @ np.asarray(delta, float))
    var = float(w @ np.asarray(sigma, float) @ w)
    return float(norm.cdf(num / np.sqrt(2 * var)))


def two_view_separation(a: float, d_p: float, d_t: float, rho: float) -> float:
    """Separation D(a) of a * PRM + (1-a) * T for two views standardized to within-class SD 1 (P2)."""
    b = 1. - a
    return (a * d_p + b * d_t) / np.sqrt(a * a + b * b + 2 * a * b * rho)


def improvement_threshold(d_p: float, d_t: float, rho: float) -> float:
    """Smallest PRM share a in [0, 1] for which a * PRM + (1-a) * T beats the PRM alone (P2 iii).

    Returns 0.0 when d_t >= d_p (every share below 1 helps) and 1.0 when d_t <= rho * d_p (no positive telemetry
    weight helps). Otherwise a_min = R / (1 + R) with R = (d_p^2 - d_t^2) / (2 d_p (d_t - rho d_p)).
    """
    if d_t >= d_p:
        return 0.0
    if d_t <= rho * d_p:
        return 1.0
    r = (d_p ** 2 - d_t ** 2) / (2 * d_p * (d_t - rho * d_p))
    return float(r / (1 + r))


def zero_diag_eigen_weights(u):
    """Leading eigenvector of R_off = u u' - diag(u^2) in closed form (P3).

    From R_off v = lam v: v_k = s u_k / (lam + u_k^2) with s = u'v, so the leading eigenvalue lam1 is the largest root
    of sum_k u_k^2 / (lam + u_k^2) = 1. This is what ``fusion_utils.sml_fuse_signed`` computes on z-scored views that
    are conditionally independent given y. Compared with the Fisher weight u_k / (1 - u_k^2), the ratio
    (1 - u_k^2) / (lam1 + u_k^2) falls with u_k: the strongest view is under-weighted, and equal loadings give
    exactly equal weights.

    Returns:
        (weights L1-normalized, lam1)
    """
    u = np.abs(np.asarray(u, float))
    if np.count_nonzero(u) < 2:
        raise ValueError('need at least two non-zero loadings')
    f = lambda lam: float(np.sum(u ** 2 / (lam + u ** 2)) - 1.)
    lam1 = brentq(f, 1e-12, float(np.sum(u ** 2)) + 1.)
    v = u / (lam1 + u ** 2)
    return v / v.sum(), float(lam1)


# ----------------------------------------------------------------------------------------------- answer structure
def within_answer_center(X, offsets):
    """Subtract each answer's mean from every column (P1: removes answer effects, keeps the one-factor structure)."""
    X = np.asarray(X, float)
    out = np.empty_like(X)
    for a, b in zip(offsets[:-1], offsets[1:]):
        out[a:b] = X[a:b] - X[a:b].mean(0)
    return out


def covariate_basis(position, log_length=None, degree: int = 3):
    """Flexible basis of the observed step covariates: powers of relative position, plus log step length if given."""
    p = np.asarray(position, float)
    cols = [p ** d for d in range(1, degree + 1)]
    if log_length is not None:
        cols.append(np.asarray(log_length, float))
    return np.column_stack(cols)


def within_answer_residualize(X, Z, offsets):
    """Center X and the covariate basis Z within answers, then remove the pooled OLS fit of X on Z (P5 f).

    Used for ESTIMATION of loadings only: every loading then shrinks by the same factor sqrt(Var(y~)), so the ratios
    u_k / u_0 that set the Fisher weights and the sign of the gate do not change. Labels are not used.
    """
    Xc = within_answer_center(X, offsets)
    Zc = within_answer_center(np.atleast_2d(np.asarray(Z, float).T).T, offsets)
    beta, *_ = np.linalg.lstsq(Zc, Xc, rcond=None)
    return Xc - Zc @ beta


# ----------------------------------------------------------------------------------------------- anchored loadings
def _winsorize(X, q):
    lo, hi = np.quantile(X, q, axis=0), np.quantile(X, 1 - q, axis=0)
    return np.clip(X, lo, hi)


def _cumulant_u0sq(x0, Xt):
    """u_0^2 estimates C_0k * k(0,0,k) / k(0,k,k), one per telemetry view, from centered columns."""
    c = (x0[:, None] * Xt).mean(0)
    k00 = (x0[:, None] ** 2 * Xt).mean(0)
    k0k = (x0[:, None] * Xt ** 2).mean(0)
    ok = (c > 0) & (k00 > 0) & (k0k > 0)
    est = np.full(Xt.shape[1], np.nan)
    est[ok] = c[ok] * k00[ok] / k0k[ok]
    return est, c


def _rank_profile_u0sq(C, anchor):
    """u_0^2 from the second-order rank condition: C_TT,off - t c c' is the off-diagonal of a rank-one matrix at t = 1/u_0^2."""
    idx = [k for k in range(C.shape[0]) if k != anchor]
    c = C[anchor, idx]
    CT = C[np.ix_(idx, idx)]
    K = len(idx)
    if K < 4:
        return np.nan, {'reason': 'K<4'}
    off = ~np.eye(K, dtype=bool)

    def misfit(t):
        M = CT - t * np.outer(c, c)
        v = np.full(K, .3)
        for _ in range(60):
            for i in range(K):
                m = off[i]
                den = float(v[m] @ v[m])
                v[i] = max(0., float(M[i, m] @ v[m]) / den) if den > 1e-15 else 0.
        return float(((M - np.outer(v, v)) ** 2)[off].sum())

    t_min = max(1. / (.95 * C[anchor, anchor]), float(np.max(c ** 2 / (.95 * np.diag(CT)))))
    ts = t_min * 1.05 ** np.arange(120)
    ms = np.array([misfit(t) for t in ts])
    i = int(np.argmin(ms))
    flat = bool(i in (0, len(ts) - 1) or min(ms[max(i - 8, 0)], ms[min(i + 8, len(ts) - 1)]) < 1.5 * ms[i] + 1e-12)
    return 1. / ts[i], {'profile_flat': flat}


def _fa_anchor(C, anchor, iters=400):
    """Two-factor ULS fit of the off-diagonal entries, signal u and nuisance l with l[anchor] = 0 (P5, factor form)."""
    m = C.shape[0]
    off = ~np.eye(m, dtype=bool)
    w, V = np.linalg.eigh(np.where(off, C, 0.))
    u = np.abs(V[:, -1]) * np.sqrt(max(w[-1], 1e-12))
    lam = V[:, -2] * np.sqrt(max(w[-2], 1e-12)) if m > 2 else np.zeros(m)
    lam[anchor] = 0.
    for _ in range(iters):
        R = C - np.outer(lam, lam)
        for i in range(m):
            mi = off[i]
            den = float(u[mi] @ u[mi])
            u[i] = max(0., float(R[i, mi] @ u[mi]) / den) if den > 1e-15 else 0.
        R = C - np.outer(u, u)
        for i in range(m):
            if i == anchor:
                continue
            mi = off[i] & (np.arange(m) != anchor)
            den = float(lam[mi] @ lam[mi])
            lam[i] = float(R[i, mi] @ lam[mi]) / den if den > 1e-15 else 0.
    return u, lam


def anchored_loadings(X, anchor: int = 0, method: str = 'cumulant'):
    """Signal loadings u (Var(y)^(1/2) * Delta) of every view, identified by anchoring on view ``anchor`` (P5).

    X must already be centered within answers (and, for estimation, covariate-adjusted). Assumptions: the latent
    nuisance does not load on the anchor, and the anchor's noise is independent of the other views' noise.

    method:
        'cumulant'      u_0^2 = median_k C_0k k(0,0,k)/k(0,k,k)  (third-order route, P5 c)
        'cumulant_w01'  the same on columns winsorized at 1% / 99% (heavy tails)
        'rank'          second-order rank condition on the telemetry block (P5 b; needs K >= 4)
        'fa'            joint two-factor fit of the off-diagonal with the anchor's nuisance loading fixed at 0
        'cca'           composite T = C_TT^{-1} c (the regression of the anchor on the rest), then the cumulant ratio
                        between the anchor and T
    Then u_k = C_0k / u_0 for every k != anchor (except 'fa', which fits u directly).

    Returns:
        (u or None, info dict). u is None when the estimate is not available; info['abstain'] says why.
    """
    if method not in ESTIMATORS:
        raise ValueError(f'method must be one of {ESTIMATORS}')
    X = np.asarray(X, float)
    X = X - X.mean(0)
    if method == 'cumulant_w01':
        X = _winsorize(X, .01)
        X = X - X.mean(0)
    n, m = X.shape
    C = X.T @ X / (n - 1)
    idx = [k for k in range(m) if k != anchor]
    c = C[anchor, idx]
    info = {'C': C, 'n_negative_anchor_cov': int((c <= 0).sum())}
    if method == 'fa':
        u, lam = _fa_anchor(C, anchor)
        info['nuisance'] = lam
        u0sq = float(u[anchor] ** 2)
    elif method in ('cumulant', 'cumulant_w01'):
        est, _ = _cumulant_u0sq(X[:, anchor], X[:, idx])
        info['u0sq_per_view'] = est
        u0sq = float(np.nanmedian(est)) if np.isfinite(est).sum() >= 2 else np.nan
    elif method == 'rank':
        u0sq, extra = _rank_profile_u0sq(C, anchor)
        info.update(extra)
        if extra.get('profile_flat'):
            info['abstain'] = 'flat rank profile'
            return None, info
    else:  # 'cca'
        CT = C[np.ix_(idx, idx)]
        beta = np.linalg.solve(CT, c)
        T = X[:, idx] @ beta
        cT = float(X[:, anchor] @ T) / (n - 1)
        k1 = float(np.mean(X[:, anchor] ** 2 * T))
        k2 = float(np.mean(X[:, anchor] * T ** 2))
        u0sq = cT * k1 / k2 if (cT > 0 and k1 > 0 and k2 > 0) else np.nan
    if not np.isfinite(u0sq) or u0sq <= 0 or u0sq >= .95 * C[anchor, anchor]:
        info['abstain'] = 'anchor loading not identified or out of range'
        return None, info
    if method != 'fa':
        u = np.empty(m)
        u[anchor] = np.sqrt(u0sq)
        u[idx] = c / u[anchor]
    if np.any(u[idx] ** 2 >= .95 * np.diag(C)[idx]):
        info['abstain'] = 'a loading exceeds its view variance'
        return None, info
    return u, info


# ----------------------------------------------------------------------------------------------- gate
def fisher_weights(C, u):
    """Fisher / Bayes-linear weights w* ∝ Sigma^{-1} u ∝ C^{-1} u (Sherman-Morrison, C = Sigma + u u')."""
    return np.linalg.solve(np.asarray(C, float), np.asarray(u, float))


def gate_statistic(C, u, w, anchor: int = 0) -> dict:
    """G(w) = (w'u)^2 / (w'Sigma w) - u_0^2 / Sigma_00 with Sigma = C - u u' (P5 e).

    Under Gaussian class-conditionals, AUC is increasing in SNR(w) = (w'u)^2 / (w'Sigma w); the common factor Var(y)
    cancels, so G > 0 iff the fusion w beats the anchor view alone. Also returns G for the Fisher weights.
    """
    C, u, w = (np.asarray(a, float) for a in (C, u, w))
    S = C - np.outer(u, u)
    snr = lambda v: float((v @ u) ** 2 / (v @ S @ v))
    e0 = np.zeros(len(u)); e0[anchor] = 1.
    s0 = snr(e0)
    wf = fisher_weights(C, u)
    return {'G': snr(w) - s0, 'G_fisher': snr(wf) - s0, 'snr_w': snr(w), 'snr_anchor': s0, 'snr_fisher': snr(wf),
            'w_fisher': wf / np.abs(wf).sum()}


def gate_decision(X, offsets, w, anchor: int = 0, method: str = 'cumulant', B: int = 200, level: float = .90,
                  seed: int = 0) -> dict:
    """Label-free fuse / prm / abstain decision for the fixed fusion weights w (P5 g).

    X: estimation matrix (centered within answers and covariate-adjusted). The point estimate of G uses all rows; the
    interval resamples whole answers B times. Decision: 'fuse' if the lower quantile of G is > 0, 'prm' if the upper
    quantile is < 0, otherwise 'abstain' (the caller falls back to the PRM). Also 'abstain' when the point estimate is
    not identified or fewer than 80% of the resamples are.
    """
    X = np.asarray(X, float)
    u, info = anchored_loadings(X, anchor, method)
    out = {'method': method, 'n_negative_anchor_cov': info['n_negative_anchor_cov']}
    if u is None:
        return dict(out, decision='abstain', reason=info.get('abstain'))
    g = gate_statistic(info['C'], u, w, anchor)
    out.update(G=g['G'], G_fisher=g['G_fisher'], u=u, w_fisher=g['w_fisher'])
    if B <= 0:                                   # point-estimate decision, no interval
        return dict(out, decision='fuse' if g['G'] > 0 else 'prm', reason='point estimate (B=0)')
    rng = np.random.default_rng(seed)
    starts, ends = np.asarray(offsets[:-1]), np.asarray(offsets[1:])
    Gs = []
    for _ in range(B):
        pick = rng.integers(0, len(starts), len(starts))
        rows = np.concatenate([np.arange(starts[j], ends[j]) for j in pick])
        ub, ib = anchored_loadings(X[rows], anchor, method)
        if ub is not None:
            Gs.append(gate_statistic(ib['C'], ub, w, anchor)['G'])
    out['n_boot_identified'] = len(Gs)
    if len(Gs) < .8 * B:
        return dict(out, decision='abstain', reason='bootstrap rarely identified')
    lo, hi = np.quantile(Gs, [(1 - level) / 2, 1 - (1 - level) / 2])
    out.update(G_lo=float(lo), G_hi=float(hi))
    out['decision'] = 'fuse' if lo > 0 else ('prm' if hi < 0 else 'abstain')
    return out
