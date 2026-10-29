"""ds_gate_v1b_fixed: the ds_gate_v1 hierarchical Dawid-Skene model with a BINNED position prior, and cleaning with
stored orientation + position residualization of every voter including the PRM.

Model (unchanged except the prior): answer state z in {clean, has-error}, P(has-error) = rho; inside a has-error
answer y_i ~ Bernoulli(pi_bin(i)), bin(i) = one of 10 equal-width bins of relative position (j+.5)/n, conditioned on
>= 1 error in the answer (exact likelihood); clean answer: all y = 0. Group votes independent given (z, y):
P(vote=1) = psi_g (error step), phi_g (correct step of a has-error answer), phi0_g (step of a clean answer).
No labels anywhere in this file.
"""
import numpy as np
from scipy.special import expit, logit

CLIP = 1e-4
NBINS = 10


def relative_position(off):
    n = np.diff(off)
    return np.concatenate([(np.arange(k) + .5) / k for k in n])


def bin_design(pos, nbins=NBINS):
    """One-hot design (steps x nbins): bin b = [b/10, (b+1)/10), last bin closed."""
    b = np.minimum((pos * nbins).astype(int), nbins - 1)
    D = np.zeros((len(pos), nbins)); D[np.arange(len(pos)), b] = 1.
    return D


# ---------------------------------------------------------------- steps 0-2
def clean_and_vote(E, X, off, pool, prm_risk=None):
    """X: stored channels (steps x m) in their stored pipeline orientation (stage2.TeleCell: no flips).
    prm_risk: optional PRM risk (higher = riskier). Every voter, the PRM included, is answer-z scored, residualized on
    a cubic in relative position fitted by pooled least squares on `pool` (Step 481), re-standardized per answer and
    GAP-binarized; a voter constant inside an answer votes 0 there. Returns V (steps x voters, {0,1}; PRM first)."""
    pos = relative_position(off); n = np.diff(off)
    B = np.column_stack([np.ones_like(pos), pos, pos ** 2, pos ** 3])
    cols = ([prm_risk] if prm_risk is not None else []) + [X[:, k] for k in range(X.shape[1])]
    R = np.column_stack(cols)
    Z = np.column_stack([E.answer_z(R[:, k], off) for k in range(R.shape[1])])
    Z = Z - B @ np.linalg.lstsq(B[pool], Z[pool], rcond=None)[0]
    sd = np.array([[R[a:b, k].std() for k in range(R.shape[1])] for a, b in zip(off[:-1], off[1:])])
    const = np.repeat(sd < 1e-8, n, axis=0)
    V = np.zeros(R.shape, int)
    for k in range(R.shape[1]):
        zk = E.answer_z(Z[:, k], off)
        f = E.topk_flags(zk, off, E.gap_counts(zk, off)); f[const[:, k]] = False
        V[:, k] = f
    return V


def group_votes(A, V, pool):
    """groups_from_cov on the +-1 vote covariance; group vote = sign of sml_weights_nonneg-weighted +-1 votes."""
    S = np.where(V > 0, 1., -1.)
    R = np.cov(np.ascontiguousarray(S[pool]).T)
    K, cg = A.groups_from_cov(R)
    G = np.zeros((len(V), K), int)
    for g in range(K):
        idx = np.flatnonzero(cg == g)
        if len(idx) == 1:
            G[:, g] = V[:, idx[0]]
        else:
            w, _ = A.sml_weights_nonneg(R[np.ix_(idx, idx)])
            G[:, g] = (S[:, idx] @ w) > 0
    return G, K, cg


# ---------------------------------------------------------------- step 3: the model
def _lv(G, p):
    return G @ np.log(p) + (1 - G) @ np.log1p(-p)


def e_step(G, off, D, prm):
    rho, theta, psi, phi, phi0 = prm
    l1, le, lc = _lv(G, psi), _lv(G, phi), _lv(G, phi0)
    pi = np.clip(expit(D @ theta), CLIP, 1 - CLIP)
    t1 = np.log(pi) + l1; t0 = np.log1p(-pi) + le
    lmix = np.logaddexp(t1, t0); starts = off[:-1]
    N = np.add.reduceat(t0, starts)
    Dd = np.add.reduceat(np.logaddexp(0, t1 - t0), starts)
    lnum = N + np.log(np.expm1(Dd))                              # log(e^M - e^N): configurations with >= 1 error
    Lz0 = np.add.reduceat(np.log1p(-pi), starts)
    L1 = lnum - np.log(-np.expm1(Lz0))                           # divided by P(>= 1 error | pi)
    L0 = np.add.reduceat(lc, starts)
    la1, la0 = np.log(rho) + L1, np.log1p(-rho) + L0
    ll = np.logaddexp(la1, la0)
    w = np.exp(la1 - ll); n = np.diff(off); W = np.repeat(w, n)
    qy = np.clip(np.exp(t1 - lmix) * np.repeat(1.0 / (-np.expm1(-Dd)), n), 0, 1)
    return w, W * qy, W, float(ll.sum())


def m_step(G, off, D, w, q, W, theta):
    eps = 1e-12
    rho = np.clip(w.mean(), CLIP, 1 - CLIP)
    psi = np.clip((q @ G) / (q.sum() + eps), CLIP, 1 - CLIP)
    r = np.clip(W - q, 0, None)
    phi = np.clip((r @ G) / (r.sum() + eps), CLIP, 1 - CLIP)
    c = 1 - W
    phi0 = np.clip((c @ G) / (c.sum() + eps), CLIP, 1 - CLIP)
    starts = off[:-1]; n = np.diff(off)

    def obj(t):                                                  # exact expected objective incl. the normalizer
        pi = np.clip(expit(D @ t), CLIP, 1 - CLIP)
        lz0 = np.add.reduceat(np.log1p(-pi), starts)
        return float(q @ np.log(pi) + (W - q) @ np.log1p(-pi) - w @ np.log(-np.expm1(lz0)))

    beta = np.asarray(theta, float).copy(); f0 = obj(beta)
    for _ in range(50):
        mu = expit(D @ beta)
        lz0 = np.add.reduceat(np.log1p(-np.clip(mu, CLIP, 1 - CLIP)), starts)
        ratio = np.repeat(w * np.exp(lz0) / (-np.expm1(lz0)), n)
        g = D.T @ (q - W * mu - ratio * mu)                      # exact gradient of obj
        H = (D * ((W + ratio) * mu * (1 - mu))[:, None]).T @ D + 1e-6 * np.eye(D.shape[1])
        step = np.linalg.solve(H, g); t = 1.0
        while t > 1e-4:                                          # backtracking on the exact objective
            cand = beta + t * step; f1 = obj(cand)
            if f1 >= f0 - 1e-9: break
            t /= 2
        if t <= 1e-4: break
        beta, done = cand, abs(f1 - f0) < 1e-10; f0 = f1
        if done: break
    return (rho, np.clip(beta, logit(CLIP), logit(1 - CLIP)), psi, phi, phi0)


def fit_hds(G, off, D, n_starts=5, iters=500, tol=1e-8, seed=20261029):
    rng = np.random.default_rng(seed)
    K = G.shape[1]; n = len(G); P = D.shape[1]; rate = np.clip(G.mean(0), .02, .98); share = G.mean(1)
    best = None
    for s in range(n_starts):
        if s == 0:                                               # start from the vote-share posterior
            q0 = np.clip(share, .01, .99); W0 = np.ones(n); w0 = np.full(len(off) - 1, .8)
            prm = m_step(G, off, D, w0, q0, W0, np.full(P, logit(np.clip(q0.mean(), .02, .98))))
        else:
            prm = (rng.uniform(.3, .9), logit(rng.uniform(.05, .4, P)), rng.uniform(.5, .95, K), rng.uniform(.02, .3, K),
                   np.clip(rate * rng.uniform(.7, 1.3, K), CLIP, 1 - CLIP))
        prev = -np.inf; drops = []
        for it in range(iters):
            w, q, W, ll = e_step(G, off, D, prm)
            if ll < prev - 1e-6: drops.append(prev - ll)
            prm = m_step(G, off, D, w, q, W, prm[1])
            if 0 <= ll - prev < tol * (len(off) - 1): break
            prev = ll
        w, q, W, ll = e_step(G, off, D, prm)
        if best is None or ll > best['loglik']:
            best = dict(params=prm, p_answer=w, q=q, loglik=ll, n_iter=it + 1, start=s, ll_drops=len(drops),
                        worst_drop=float(max(drops)) if drops else 0.0)
    rho, theta, psi, phi, phi0 = best['params']
    best['label_switch_warning'] = bool(psi.mean() < phi.mean())
    best['degenerate'] = dict(rho_at_clip=bool(rho <= 2 * CLIP or rho >= 1 - 2 * CLIP), phi0_above_psi=bool(phi0.mean() > psi.mean()))
    best['params'] = dict(rho=float(rho), pi_bins=expit(theta).tolist(), psi=psi.tolist(), phi=phi.tolist(), phi0=phi0.tolist())
    return best
