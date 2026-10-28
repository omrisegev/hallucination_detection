"""PRM-anchored binary L-SML with continuous scoring (view selection stage 1).

Pre-registration: docs/experiments/PRM_VIEW_SELECTION_STAGE1_PREREG_2026-10-25.md. Every view is pre-oriented by
the PRM (anchor) and binarized per answer (GAP votes, +1 = error). For a telemetry subset S: groups from the vote
covariance (Jaffe-Fetaya-Nadler 2016 Algorithm 1, exactly as fusion_utils.detect_dependent_groups), SML within each
group (non-negative), two-coin Dawid-Skene EM between [PRM, group virtual votes] with ML log-odds weights (non-
negative), then the binary-estimated weights applied to the continuous answer-z values. No label enters.
"""
from __future__ import annotations
import numpy as np
from .fusion_utils import (_score_matrix_lsml, _residual_lsml, _spectral_cluster_precomputed,
                           _exact_small_m_groups, SMALL_M_EXACT)


# ----------------------------------------------------------------------------------------------- groups
def groups_from_cov(R):
    """(K, c) from a vote covariance R, the residual path of fusion_utils.detect_dependent_groups verbatim
    (default K_range 2..min(m, 9)-1, exact enumeration at m <= SMALL_M_EXACT, 'unit' loading scale)."""
    R = np.asarray(R, float); m = len(R)
    K_range = list(range(2, min(m, 9)))
    if not K_range:
        return 1, np.zeros(m, int)
    if m <= SMALL_M_EXACT:
        K, c, *_ = _exact_small_m_groups(R, K_range, 'unit')
        return (1, np.zeros(m, int)) if K is None else (int(K), np.asarray(c, int))
    s = _score_matrix_lsml(R)
    best = (None, None, np.inf)
    for K in K_range:
        try:
            c = _spectral_cluster_precomputed(s, K)
        except Exception:
            continue
        r = _residual_lsml(R, c, loading_scale='unit')
        if r < best[2]: best = (K, c, r)
    if best[0] is None:
        return 1, np.zeros(m, int)
    return int(best[0]), np.asarray(best[1], int)


# ----------------------------------------------------------------------------------------------- weights
def sml_weights_nonneg(R):
    """Leading eigenvector of the zero-diagonal covariance, sign so that the sum is positive, negatives clipped to 0.
    Returns (w, n_clipped)."""
    R = np.asarray(R, float)
    if len(R) == 1:
        return np.ones(1), 0
    M = R - np.diag(np.diag(R))
    w, V = np.linalg.eigh(M); v = V[:, -1]
    if v.sum() < 0: v = -v
    n_neg = int((v < 0).sum())
    return np.clip(v, 0, None), n_neg


def spectral_vote_weights(B):
    """Leading eigenvector of the zero-diagonal covariance of +-1 votes B, sign so that the sum is positive (the SML
    direction; used as an EM start, Zhang-Chen-Zhou-Jordan 2014)."""
    M = np.cov(np.asarray(B, float).T); M = M - np.diag(np.diag(M))
    v = np.linalg.eigh(M)[1][:, -1]
    return -v if v.sum() < 0 else v


def ds_em_votes(B, anchor=0, iters: int = 500, tol: float = 1e-9, eps: float = 1e-6, init='majority'):
    """Two-coin Dawid-Skene EM on +-1 votes B (n x m), on compressed patterns. Label switching resolved by the anchor
    (psi + eta > 1), or with anchor=None by the majority rule (flip if fewer than half the views have psi + eta > 1;
    Parisi et al. 2014 assumption iii; stage 2). Returns dict(psi, eta, prevalence, w) with ML log-odds weights w
    clipped at 0. init='majority' (stages 1-2) starts from the soft majority vote; init='spectral' from
    expit(s / sd(s)), s = votes @ spectral_vote_weights(B), both clipped to [.05, .95]."""
    B = np.asarray(B) > 0; n, m = B.shape
    code = B.astype(np.int64) @ (1 << np.arange(m, dtype=np.int64))
    u, cnt = np.unique(code, return_counts=True)
    P = ((u[:, None] >> np.arange(m)) & 1).astype(bool); wts = cnt / cnt.sum()
    if init == 'spectral':
        s = (2. * P - 1) @ spectral_vote_weights(2. * B - 1)
        sd = np.sqrt(wts @ (s - wts @ s) ** 2)
        q = np.clip(1 / (1 + np.exp(-s / max(sd, 1e-8))), .05, .95)
    elif init == 'majority':
        q = np.clip(((2 * P - 1).mean(1) + 1) / 2, .05, .95)
    else:
        raise ValueError(init)
    ll_old = -np.inf
    for it in range(iters):
        p = np.clip(wts @ q, eps, 1 - eps)
        psi = np.clip((wts * q) @ P / (wts @ q), eps, 1 - eps)
        eta = np.clip((wts * (1 - q)) @ ~P / (wts @ (1 - q)), eps, 1 - eps)
        a1 = np.where(P, np.log(psi), np.log(1 - psi)).sum(1) + np.log(p)
        a0 = np.where(P, np.log(1 - eta), np.log(eta)).sum(1) + np.log(1 - p)
        mx = np.maximum(a1, a0); ll = float(wts @ (mx + np.log(np.exp(a1 - mx) + np.exp(a0 - mx))))
        q = 1 / (1 + np.exp(a0 - a1))
        if abs(ll - ll_old) < tol: break
        ll_old = ll
    flip = (psi[anchor] + eta[anchor] < 1) if anchor is not None else (np.sum(psi + eta > 1) < m / 2)
    if flip:
        psi, eta, p = 1 - eta, 1 - psi, 1 - p
    w = np.log(psi * eta / ((1 - psi) * (1 - eta)))
    return dict(psi=psi, eta=eta, prevalence=float(p), w_raw=w, w=np.clip(w, 0, None), iters=it + 1,
                n_patterns=len(u))


# ----------------------------------------------------------------------------------------------- answers
class Answers:
    """Vectorized per-answer operations over a padded (n_answers x L) index; mirrors evaluate_collection's loops."""

    def __init__(self, off):
        off = np.asarray(off, np.int64); self.off = off
        lens = np.diff(off); self.lens = lens; L = int(lens.max())
        j = np.arange(L)[None, :]
        self.mask = j < lens[:, None]
        self.idx = np.where(self.mask, off[:-1, None] + j, 0)
        self.n = int(off[-1])

    def z(self, s):
        """Per-answer z-score with population std floored at 1e-8 (evaluate_collection.answer_z)."""
        X = np.where(self.mask, s[self.idx], 0.)
        mu = X.sum(1) / self.lens
        sd = np.sqrt(np.where(self.mask, (X - mu[:, None]) ** 2, 0.).sum(1) / self.lens)
        Zm = (X - mu[:, None]) / np.maximum(sd, 1e-8)[:, None]
        out = np.empty(self.n); out[self.idx[self.mask]] = Zm[self.mask]
        return out

    def gap_flags(self, z):
        """GAP count (gap_counts) then the top-k steps with stable ties by position (topk_flags)."""
        X = np.where(self.mask, z[self.idx], -np.inf)
        order = np.argsort(-X, axis=1, kind='stable')
        srt = np.take_along_axis(X, order, 1)
        m = self.lens; L = X.shape[1]
        km = np.minimum(np.ceil(m / 2).astype(int), m - 1)
        with np.errstate(invalid='ignore'):                  # -inf - -inf in the padding; masked out below
            gaps = srt[:, :-1] - srt[:, 1:] if L > 1 else np.zeros((len(m), 0))
        j = np.arange(gaps.shape[1])[None, :]
        gaps = np.where(j < km[:, None], gaps, -np.inf)
        k = np.where(m == 1, 1, np.argmax(gaps, axis=1) + 1) if gaps.shape[1] else np.ones(len(m), int)
        rank = np.empty_like(order); np.put_along_axis(rank, order, np.arange(L)[None, :].repeat(len(m), 0), 1)
        F = (rank < k[:, None]) & self.mask
        f = np.zeros(self.n, bool); f[self.idx[self.mask]] = F[self.mask]
        return f
