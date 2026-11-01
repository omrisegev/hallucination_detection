"""Neighbor-aware readouts of a fused step score (docs/experiments/NEIGHBOR_READOUT_PREREG_2026-11-01.md).

Every function takes s (one score per step, answer-z'd, higher = more likely an error) and off (answer offsets,
steps in answer order). No function accepts labels. Fits are pooled over all answers of one cell.

Stage-1 transforms (diagnostic curves): sym, lift, lap.
Stage-2 label-free fits: variogram_fit (V1 weight), gcv_lambda (G1), hmm_fit / hmm_logit (H1, H1-NOPOS).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import expit, logit

SYM_GRID = (0., .1, .2, .3, .5, .75, 1.)
LAP_GRID = (0., .1, .25, .5, 1., 2.)
HMM_STARTS = ((.5, .90), (.5, .95), (.6, .90), (.4, .85), (.5, .80))
HMM_RHO0, HMM_ENTRY0 = .4, .1
HMM_MAX_ITER, HMM_TOL, HMM_NEWTON = 500, 1e-7, 10
SD_FLOOR = 1e-3                                  # keeps EM away from a zero-variance spike (answers of 1-2 steps)
PROB_CLIP = 1e-6
VALID_SHARE, VALID_SD_RATIO, VALID_MU_GAP = (.01, .60), (.1, 10.), .1


# --------------------------------------------------------------------------------------------- neighbors
def _neighbors(off):
    """Boolean masks: step t has a previous / next step inside its own answer."""
    n = int(off[-1]); first = np.zeros(n, bool); last = np.zeros(n, bool)
    lens = np.diff(off); ne = lens > 0
    first[off[:-1][ne]] = True; last[off[1:][ne] - 1] = True
    return ~first, ~last


def neighbor_mean(s, off):
    """Mean of the existing neighbors t-1, t+1 within the answer; 0 where the answer has one step."""
    has_p, has_n = _neighbors(off)
    prev = np.where(has_p, np.roll(s, 1), 0.); nxt = np.where(has_n, np.roll(s, -1), 0.)
    cnt = has_p.astype(float) + has_n
    return np.divide(prev + nxt, cnt, out=np.zeros(len(s)), where=cnt > 0)


def neighbor_max(s, off):
    has_p, has_n = _neighbors(off)
    prev = np.where(has_p, np.roll(s, 1), -np.inf); nxt = np.where(has_n, np.roll(s, -1), -np.inf)
    return np.maximum(prev, nxt)


def sym(s, off, w):
    """SYM(w) = s_t + w * mean of neighbors."""
    return s + w * neighbor_mean(s, off) if w else s.copy()


def lift(s, off, w):
    """LIFT(w) = s_t + w * max(0, max neighbor - s_t): raises a step toward a stronger neighbor, never lowers it."""
    if not w: return s.copy()
    return s + w * np.maximum(0., np.nan_to_num(neighbor_max(s, off) - s, neginf=0.))


# --------------------------------------------------------------------------------------------- graph smoothing
def lap(s, off, lam):
    """LAP(lam) = (I + lam L)^-1 s, L the path-graph Laplacian of each answer (block-diagonal, one banded solve)."""
    if not lam: return s.copy()
    has_p, has_n = _neighbors(off)
    deg = has_p.astype(float) + has_n
    ab = np.zeros((3, len(s)))
    ab[0, 1:] = -lam * has_n[:-1]                # super-diagonal: (t, t+1)
    ab[1] = 1 + lam * deg
    ab[2, :-1] = -lam * has_n[:-1]               # sub-diagonal: (t+1, t)
    return solve_banded((1, 1), ab, s)


def lap_trace(n, lam):
    """tr (I + lam L_n)^-1 from the path-Laplacian eigenvalues 2 - 2 cos(pi k / n), k = 0..n-1."""
    k = np.arange(n)
    return float(np.sum(1. / (1. + lam * (2 - 2 * np.cos(np.pi * k / n)))))


def gcv_scores(s, off, grid=LAP_GRID):
    """Pooled GCV(lam) = sum ||s - S s||^2 / (sum (n - tr S))^2. At lam = 0 both terms vanish; the value used is
    the lam -> 0 limit sum ||L s||^2 / (sum tr L)^2 (implementation note A2 of the protocol)."""
    lens = np.diff(off); out = {}
    has_p, has_n = _neighbors(off)
    for lam in grid:
        if lam == 0:
            Ls = (has_p.astype(float) + has_n) * s - np.where(has_p, np.roll(s, 1), 0.) - np.where(has_n, np.roll(s, -1), 0.)
            trL = float(np.sum(2. * np.maximum(lens - 1, 0)))
            out[lam] = float(np.sum(Ls ** 2) / trL ** 2) if trL > 0 else np.nan
            continue
        rss = float(np.sum((s - lap(s, off, lam)) ** 2))
        uniq, cnt = np.unique(lens[lens > 0], return_counts=True)
        dof = float(sum(c * (n - lap_trace(int(n), lam)) for n, c in zip(uniq, cnt)))
        out[lam] = rss / dof ** 2 if dof > 0 else np.nan
    return out


def gcv_lambda(s, off, grid=LAP_GRID):
    g = gcv_scores(s, off, grid)
    lams = [l for l in grid if np.isfinite(g[l])]
    return dict(lam=float(min(lams, key=lambda l: (g[l], l))) if lams else 0., gcv={str(k): v for k, v in g.items()})


# --------------------------------------------------------------------------------------------- variogram (V1)
def rel_position(off):
    """r_t = (t - .5) / n per answer, t = 1..n."""
    lens = np.diff(off); n = np.repeat(lens, lens)
    t = np.arange(int(off[-1])) - np.repeat(off[:-1], lens) + 1
    return (t - .5) / n


def variogram_fit(s, off, max_lag=3, min_len=4):
    """Moment fit V(L) = sig2 + Vx (1 - lam^L), L = 1..3, on s minus its pooled relative-position-decile mean.
    Returns lam, Vx, sig2, the Wiener weight w (SYM form) and validity; invalid -> w = 0."""
    r = rel_position(off); dec = np.minimum((r * 10).astype(int), 9)
    means = np.array([s[dec == d].mean() if np.any(dec == d) else 0. for d in range(10)])
    x = s - means[dec]
    lens = np.diff(off); V = []
    for L in range(1, max_lag + 1):
        sq = [(x[a + L:b] - x[a:b - L]) ** 2 for a, b, n in zip(off[:-1], off[1:], lens) if n >= min_len]
        V.append(.5 * float(np.concatenate(sq).mean()) if sq else np.nan)
    V1, V2, V3 = V; d1, d2 = V2 - V1, V3 - V2
    out = dict(V=V, d1=d1, d2=d2, lam=np.nan, Vx=np.nan, sig2=np.nan, w=0., beta=None, valid=False, reason='')
    if not (np.isfinite(d1) and d1 > 0):
        out['reason'] = 'd1 <= 0'; return out
    lam = d2 / d1; out['lam'] = lam
    if not 0 < lam < 1:
        out['reason'] = 'lam outside (0,1)'; return out
    Vx = d1 / (lam * (1 - lam)); sig2 = V1 - Vx * (1 - lam); out.update(Vx=Vx, sig2=sig2)
    if not (Vx > 0 and sig2 > 0):
        out['reason'] = 'Vx or sig2 <= 0'; return out
    out.update(wiener_weight(lam, Vx, sig2), valid=True)
    return out


def wiener_weight(lam, Vx, sig2):
    """beta = Sigma^-1 c for (r_{t-1}, r_t, r_{t+1}); SYM uses the neighbor MEAN, so w = 2 beta_side / beta_center
    (amendment A1 of the protocol: the declared beta_0/beta_1 omits this factor 2)."""
    g = np.array([sig2 + Vx, Vx * lam, Vx * lam ** 2])
    Sig = np.array([[g[0], g[1], g[2]], [g[1], g[0], g[1]], [g[2], g[1], g[0]]])
    beta = np.linalg.solve(Sig, Vx * np.array([lam, 1., lam]))
    return dict(beta=beta.tolist(), w=float(2 * beta[0] / beta[1]))


# --------------------------------------------------------------------------------------------- HMM-POS (H1)
def _pad(v, off):
    lens = np.diff(off); A, T = len(lens), int(lens.max())
    M = np.arange(T)[None, :] < lens[:, None]
    P = np.zeros((A, T)); P[M] = v
    return P, M


def _forward_backward(S, R, M, p, position):
    """Scaled forward-backward of the 2-state chain. Returns loglik, gamma (A,T,2), xi (A,T,2,2) for t >= 1
    (xi[:, 0] unused), and the posterior log-odds of state 1 (finite: computed from mixtures, not from gamma)."""
    A, T = S.shape
    e = expit(p['a'] + (p['b'] * R if position else 0. * R))                    # entry prob, from state 0, per step
    logB = np.stack([-.5 * np.log(2 * np.pi * p['sd'][j] ** 2) - (S - p['mu'][j]) ** 2 / (2 * p['sd'][j] ** 2)
                     for j in (0, 1)], -1)
    off_ = logB.max(-1); B = np.exp(logB - off_[..., None])
    rho = p['rho']
    pred = np.zeros((A, T, 2)); alpha = np.zeros((A, T, 2)); c = np.ones((A, T))
    pred[:, 0, 1] = e[:, 0]; pred[:, 0, 0] = 1 - e[:, 0]
    for t in range(T):
        if t:
            ap = alpha[:, t - 1]
            pred[:, t, 1] = ap[:, 0] * e[:, t] + ap[:, 1] * rho
            pred[:, t, 0] = ap[:, 0] * (1 - e[:, t]) + ap[:, 1] * (1 - rho)
        a_ = pred[:, t] * B[:, t]; ct = a_.sum(-1)
        m = M[:, t]; c[m, t] = ct[m]
        alpha[:, t] = np.where(m[:, None], a_ / np.where(m, ct, 1.)[:, None], alpha[:, t - 1] if t else 0.)
    beta = np.ones((A, T, 2))
    for t in range(T - 2, -1, -1):
        m = M[:, t + 1]
        nb = B[:, t + 1] * beta[:, t + 1] / c[:, t + 1, None]
        b0 = (1 - e[:, t + 1]) * nb[:, 0] + e[:, t + 1] * nb[:, 1]
        b1 = (1 - rho) * nb[:, 0] + rho * nb[:, 1]
        beta[:, t, 0] = np.where(m, b0, 1.); beta[:, t, 1] = np.where(m, b1, 1.)
    gamma = alpha * beta; gamma /= gamma.sum(-1, keepdims=True)
    xi = np.zeros((A, T, 2, 2))
    nb = B[:, 1:] * beta[:, 1:] / c[:, 1:, None]
    ap = alpha[:, :-1]
    xi[:, 1:, 0, 0] = ap[..., 0] * (1 - e[:, 1:]) * nb[..., 0]
    xi[:, 1:, 0, 1] = ap[..., 0] * e[:, 1:] * nb[..., 1]
    xi[:, 1:, 1, 0] = ap[..., 1] * (1 - rho) * nb[..., 0]
    xi[:, 1:, 1, 1] = ap[..., 1] * rho * nb[..., 1]
    xi[:, 1:] *= M[:, 1:, None, None]
    ll = float(np.log(c[M]).sum() + off_[M].sum())
    lo = (np.log(pred[..., 1]) - np.log(pred[..., 0]) + logB[..., 1] - logB[..., 0]
          + np.log(beta[..., 1]) - np.log(beta[..., 0]))
    return ll, gamma, xi, lo, e


def _m_step(S, R, M, gamma, xi, p, position):
    g = gamma * M[..., None]
    mu = np.array([np.sum(g[..., j] * S) / g[..., j].sum() for j in (0, 1)])
    sd = np.array([max(np.sqrt(np.sum(g[..., j] * (S - mu[j]) ** 2) / g[..., j].sum()), SD_FLOOR) for j in (0, 1)])
    rho = float(np.clip(xi[..., 1, 1].sum() / (xi[..., 1, 0].sum() + xi[..., 1, 1].sum()), PROB_CLIP, 1 - PROB_CLIP))
    # entry regression: step 1 (target gamma_1(1), weight 1) + transitions out of state 0 (target xi01, weight xi00+xi01)
    y = np.concatenate([g[:, 0, 1], xi[:, 1:, 0, 1][M[:, 1:]]])
    W = np.concatenate([M[:, 0].astype(float), (xi[:, 1:, 0, 0] + xi[:, 1:, 0, 1])[M[:, 1:]]])
    r = np.concatenate([R[:, 0], R[:, 1:][M[:, 1:]]])
    X = np.column_stack([np.ones_like(r), r]) if position else np.ones((len(r), 1))
    th = np.array([p['a'], p['b']])[:X.shape[1]]

    def q(th_):
        z = X @ th_
        return float(np.sum(y * -np.logaddexp(0, -z) + (W - y) * -np.logaddexp(0, z)))
    for _ in range(HMM_NEWTON):
        pe = expit(X @ th); grad = X.T @ (y - W * pe)
        H = (X * (W * pe * (1 - pe))[:, None]).T @ X + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad); q0 = q(th); h = 1.
        while h > 1e-4 and q(th + h * step) < q0: h /= 2
        th = th + h * step
        if np.max(np.abs(h * step)) < 1e-10: break
    return dict(mu=mu, sd=sd, rho=rho, a=float(th[0]), b=float(th[1]) if position else 0.)


def _em(S, R, M, p, position):
    ll_old = -np.inf; it = 0; converged = False
    for it in range(1, HMM_MAX_ITER + 1):
        ll, gamma, xi, _, _ = _forward_backward(S, R, M, p, position)
        if np.isfinite(ll_old) and abs(ll - ll_old) / abs(ll_old) < HMM_TOL:
            converged = True; break
        p = _m_step(S, R, M, gamma, xi, p, position); ll_old = ll
    return p, ll, it, converged


def implied_share(e, rho, M):
    """Model-implied mean P(state 1) over the cell's steps (prior chain, no data): q_1 = e_1, q_t = q_{t-1} rho +
    (1 - q_{t-1}) e_t. Used for the stationary-share validity rule, since entry varies with position."""
    q = np.zeros_like(e); q[:, 0] = e[:, 0]
    for t in range(1, e.shape[1]):
        q[:, t] = q[:, t - 1] * rho + (1 - q[:, t - 1]) * e[:, t]
    return float(q[M].mean())


def hmm_fit(s, off, position=True):
    """H1 (position=True) / H1-NOPOS (position=False). Five deterministic starts, highest loglik kept.
    The error state is the state with the larger mu (err_state 1, or 0 if EM converged the other way round)."""
    S, M = _pad(s, off); R, _ = _pad(rel_position(off), off)
    sd0 = float(s.std()); runs = []
    for q0, q1 in HMM_STARTS:
        p0 = dict(mu=np.quantile(s, [q0, q1]), sd=np.array([sd0, sd0]), rho=HMM_RHO0, a=float(logit(HMM_ENTRY0)), b=0.)
        runs.append(_em(S, R, M, p0, position))
    best = int(np.argmax([r[1] for r in runs])); p, ll, it, conv = runs[best]
    ll2, gamma, _, lo, e = _forward_backward(S, R, M, p, position)
    err = int(p['mu'][1] >= p['mu'][0]); oth = 1 - err
    share1 = implied_share(e, p['rho'], M)
    share = share1 if err == 1 else 1 - share1
    ratio = p['sd'][err] / p['sd'][oth]; gap = p['mu'][err] - p['mu'][oth]
    reasons = []
    if not VALID_SHARE[0] <= share <= VALID_SHARE[1]: reasons.append(f'share {share:.3f}')
    if not VALID_SD_RATIO[0] <= ratio <= VALID_SD_RATIO[1]: reasons.append(f'sd ratio {ratio:.3f}')
    if gap < VALID_MU_GAP: reasons.append(f'mu gap {gap:.3f}')
    return dict(position=position, mu=p['mu'].tolist(), sd=p['sd'].tolist(), rho=p['rho'], a=p['a'], b=p['b'],
                loglik=ll2, n_iter=it, converged=conv, best_start=best, start_logliks=[r[1] for r in runs],
                start_converged=[r[3] for r in runs], err_state=err, implied_share=share,
                posterior_share=float(gamma[M][:, err].mean()), sd_ratio=float(ratio), mu_gap=float(gap),
                valid=not reasons, reason='; '.join(reasons))


def hmm_logit(s, off, fit):
    """Posterior log-odds that each step is in the error state (finite; monotone in p_t)."""
    S, M = _pad(s, off); R, _ = _pad(rel_position(off), off)
    p = dict(mu=np.asarray(fit['mu']), sd=np.asarray(fit['sd']), rho=fit['rho'], a=fit['a'], b=fit['b'])
    _, _, _, lo, _ = _forward_backward(S, R, M, p, fit['position'])
    lo = lo[M]
    return lo if fit['err_state'] == 1 else -lo
