"""Label-free accuracy estimation for binary views (votes in {-1, +1}, +1 = error), and CI diagnostics.

Model (Parisi et al. 2014; Jaffe et al. 2015): a latent y in {-1, +1} with b = E[y]; view i flags with sensitivity
psi_i = P(v=+1 | y=+1) and specificity eta_i = P(v=-1 | y=-1); views are conditionally independent given y. Then, with
delta_i = psi_i + eta_i - 1 and centered votes,
    Cov(v_i, v_j)           = (1 - b^2) delta_i delta_j                       (i != j)
    E[c_i c_j c_k]          = -2 b (1 - b^2) delta_i delta_j delta_k          (distinct i, j, k)
    E[v_i]                  = psi_i - eta_i + delta_i b.
Used by results/prm_view_selection_v1/stage0 (pre-registration docs/experiments/PRM_VIEW_SELECTION_STAGE0_PREREG_2026-10-25.md).
"""
from __future__ import annotations
import numpy as np


def rank_one_offdiag(S, anchor: int = 0, iters: int = 500):
    """u with S_ij ~= u_i u_j off the diagonal (iterated diagonal replacement); sign fixed by u[anchor] > 0."""
    S = np.asarray(S, float)
    off = ~np.eye(len(S), dtype=bool)
    M = np.where(off, S, 0.)
    w, V = np.linalg.eigh(M)
    u = V[:, -1] * np.sqrt(max(w[-1], 0.))
    for _ in range(iters):
        M = np.where(off, S, np.outer(u, u))
        w, V = np.linalg.eigh(M)
        u_new = V[:, -1] * np.sqrt(max(w[-1], 0.))
        if u_new @ u < 0: u_new = -u_new
        if np.max(np.abs(u_new - u)) < 1e-12: u = u_new; break
        u = u_new
    if u[anchor] < 0: u = -u
    return u


def third_moments(Xc):
    """T[i, j, k] = mean(c_i c_j c_k) of centered columns."""
    n = len(Xc)
    return np.stack([Xc.T @ (Xc * Xc[:, [k]]) / n for k in range(Xc.shape[1])], axis=2)


def distinct_triples(m):
    i, j, k = np.meshgrid(np.arange(m), np.arange(m), np.arange(m), indexing='ij')
    sel = (i < j) & (j < k)
    return i[sel], j[sel], k[sel]


def spectral_binary(V, anchor: int = 0):
    """Spectral/moment estimates of (psi, eta, prevalence) for +-1 votes V (n x m). Returns a dict."""
    V = np.asarray(V, float)
    mu = V.mean(0); Xc = V - mu
    S = Xc.T @ Xc / len(V)
    u = rank_one_offdiag(S, anchor)
    T = third_moments(Xc)
    i, j, k = distinct_triples(V.shape[1])
    P = u[i] * u[j] * u[k]; t = T[i, j, k]
    s = float(t @ P / (P @ P)) if P @ P > 0 else 0.
    b = -s / np.sqrt(s * s + 4)
    delta = u / np.sqrt(1 - b * b)
    psi = np.clip((1 + delta + mu - delta * b) / 2, 0, 1)
    eta = np.clip((1 + delta - mu + delta * b) / 2, 0, 1)
    return dict(u=u, b=b, prevalence=(1 + b) / 2, delta=delta, psi=psi, eta=eta, bal_acc=(1 + delta) / 2,
                S=S, T=T, mu=mu)


def dawid_skene_binary(V, anchor: int = 0, iters: int = 200, tol: float = 1e-8, eps: float = 1e-6):
    """Two-coin Dawid-Skene EM for +-1 votes, initialized from the soft majority vote. Label switching is resolved
    by requiring the anchor's psi + eta > 1. Returns a dict with psi, eta, prevalence, posterior log-odds, loglik."""
    V = np.asarray(V, float); pos = V > 0
    q = np.clip((V.mean(1) + 1) / 2, .05, .95)
    ll_old = -np.inf
    for it in range(iters):
        p = np.clip(q.mean(), eps, 1 - eps)
        psi = np.clip((q[:, None] * pos).sum(0) / q.sum(), eps, 1 - eps)
        eta = np.clip(((1 - q)[:, None] * ~pos).sum(0) / (1 - q).sum(), eps, 1 - eps)
        a1 = np.where(pos, np.log(psi), np.log(1 - psi)).sum(1) + np.log(p)
        a0 = np.where(pos, np.log(1 - eta), np.log(eta)).sum(1) + np.log(1 - p)
        mx = np.maximum(a1, a0); ll = float(np.sum(mx + np.log(np.exp(a1 - mx) + np.exp(a0 - mx))))
        q = 1 / (1 + np.exp(a0 - a1))
        if abs(ll - ll_old) < tol * max(1., abs(ll)): break
        ll_old = ll
    if psi[anchor] + eta[anchor] < 1:                       # label switch: y <-> -y
        psi, eta, p, q = 1 - eta, 1 - psi, 1 - p, 1 - q
        a1, a0 = a0, a1
    return dict(psi=psi, eta=eta, prevalence=p, bal_acc=(psi + eta) / 2, logodds=a1 - a0, loglik=ll, iters=it + 1)


def rank_one_misfit(S, u, groups=None):
    """Share of the off-diagonal sum of squares not explained by u u'; overall, per view, and per block pair."""
    m = len(S); off = ~np.eye(m, dtype=bool)
    R = S - np.outer(u, u)
    out = dict(overall=float((R[off] ** 2).sum() / (S[off] ** 2).sum()),
               per_view=np.array([(R[k, off[k]] ** 2).sum() / max((S[k, off[k]] ** 2).sum(), 1e-30) for k in range(m)]))
    if groups is not None:
        g = np.asarray(groups)
        for a in np.unique(g):
            for b in np.unique(g):
                if a > b: continue
                msk = off & (g[:, None] == a) & (g[None, :] == b)
                if msk.any(): out[f'{a}-{b}'] = float((R[msk] ** 2).sum() / max((S[msk] ** 2).sum(), 1e-30))
    return out


def tci_residuals(S, T):
    """FUSE triplet-CI statistic per view k: relative RMS residual of T_ijk regressed through the origin on S_ij over
    pairs i<j with i, j != k. 0 under triplet conditional independence."""
    m = len(S); out = np.full(m, np.nan)
    iu, ju = np.triu_indices(m, 1)
    for k in range(m):
        sel = (iu != k) & (ju != k)
        x = S[iu[sel], ju[sel]]; y = T[iu[sel], ju[sel], k]
        if x @ x <= 0 or y @ y <= 0: continue
        slope = (x @ y) / (x @ x)
        out[k] = float(np.sqrt(((y - slope * x) ** 2).sum() / (y @ y)))
    return out


def triad_lambda(C, anchor: int = 0):
    """One-factor loading^2 of every view from triads with the anchor: median over g of C_kA C_kg / C_Ag, with IQR."""
    m = len(C); med = np.full(m, np.nan); iqr = np.full(m, np.nan)
    for k in range(m):
        if k == anchor: continue
        g = [x for x in range(m) if x not in (k, anchor) and abs(C[anchor, x]) > 1e-12]
        est = C[k, anchor] * C[k, g] / C[anchor, g]
        med[k] = float(np.median(est)); q1, q3 = np.percentile(est, [25, 75]); iqr[k] = float(q3 - q1)
    return med, iqr


def anchored_two_factor(C, anchor: int = 0, n_starts: int = 8, seed: int = 0):
    """Two-factor ULS fit of the off-diagonal of C: C_ij ~= u_i u_j + l_i l_j (i != j), with l[anchor] = 0.

    The zero fixes the rotation (just-identifying, so the anchor assumption itself is not testable from C). Signs:
    u[anchor] > 0, and l oriented so that its largest-|l| entry is positive. Multi-start least squares; returns
    (u, l, psi, info) with psi = diag(C) - u^2 - l^2 and info['rel_misfit'] = off-diagonal residual share."""
    from scipy.optimize import least_squares
    C = np.asarray(C, float); m = len(C)
    iu, ju = np.triu_indices(m, 1); tgt = C[iu, ju]
    rest = np.array([k for k in range(m) if k != anchor])

    def unpack(x):
        u = x[:m]; l = np.zeros(m); l[rest] = x[m:]
        return u, l

    def res(x):
        u, l = unpack(x)
        return u[iu] * u[ju] + l[iu] * l[ju] - tgt

    rng = np.random.default_rng(seed); best = None
    w, V = np.linalg.eigh(np.where(np.eye(m, dtype=bool), 0., C))
    x0s = [np.r_[V[:, -1] * np.sqrt(max(w[-1], 1e-6)), (V[:, -2] * np.sqrt(max(abs(w[-2]), 1e-6)))[rest]]]
    x0s += [rng.normal(0, .3, 2 * m - 1) for _ in range(n_starts - 1)]
    for x0 in x0s:
        r = least_squares(res, x0, method='lm', max_nfev=20000)
        if best is None or r.cost < best.cost: best = r
    u, l = unpack(best.x)
    if u[anchor] < 0: u = -u
    if l[np.argmax(np.abs(l))] < 0: l = -l
    psi = np.diag(C) - u ** 2 - l ** 2
    info = dict(rel_misfit=float(2 * best.cost / (tgt @ tgt)), heywood=int((psi <= 1e-6).sum()))
    return u, l, np.maximum(psi, 1e-6), info


def factor_information(u, lam, psi, drop=None):
    """Bartlett precision of the error factor with the nuisance projected out: 1 / [(L' Psi^-1 L)^-1]_11."""
    keep = np.ones(len(u), bool)
    if drop is not None: keep[drop] = False
    L = np.column_stack([u, lam])[keep]; w = 1 / psi[keep]
    M = L.T @ (L * w[:, None])
    try:
        return float(1 / np.linalg.inv(M)[0, 0])
    except np.linalg.LinAlgError:
        return float('nan')


def bartlett_scores(X, u, lam, psi):
    L = np.column_stack([u, lam]); W = L / psi[:, None]
    return X @ W @ np.linalg.inv(L.T @ W)
