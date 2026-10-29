"""Hierarchical Dawid-Skene gate on binary GAP votes (PROTOCOL.json, ds_gate_v1).

Steps 0-2 (cleaning, GAP votes, grouping) turn a cell into group votes; step 3 (fit_hds) is the model:
  answer state z in {clean, has-error}, P(has-error) = rho;
  in a has-error answer, step label y ~ Bernoulli(pi(p)), pi(p) = sigmoid(a + b p), conditioned on >= 1 error;
  in a clean answer every y = 0;
  group votes independent given (z, y): P(vote=1) = psi on an error step, phi on a correct step of a has-error
  answer, phi0 on a step of a clean answer.
Outputs P(has-error | votes) per answer and q_i = P(y_i = 1 | votes) per step. No labels anywhere in this file.
"""
import numpy as np
from scipy.special import expit, logit

CLIP = 1e-4


# ---------------------------------------------------------------- steps 0-2: cell -> group votes
def relative_position(off):
    n = np.diff(off)
    return np.concatenate([(np.arange(k) + .5) / k for k in n])


def clean_and_vote(E, X, prm_risk, off, pool):
    """X: raw channels (n_steps x 16); prm_risk: PRM risk per step (higher = more likely error).
    Returns votes V (n_steps x 17, {0,1}; column 0 = PRM) and the cleaned, oriented answer-z channels Z."""
    pos = relative_position(off)
    zp = E.answer_z(prm_risk, off)
    Z = np.column_stack([E.answer_z(X[:, k], off) for k in range(X.shape[1])])
    B = np.column_stack([np.ones_like(pos), pos, pos ** 2, pos ** 3])
    beta = np.linalg.lstsq(B[pool], Z[pool], rcond=None)[0]
    Z = Z - B @ beta                                            # Step 481: cubic in relative position, pooled
    sign = np.sign(np.nan_to_num([np.corrcoef(Z[pool, k], zp[pool])[0, 1] for k in range(Z.shape[1])])); sign[sign == 0] = 1
    Z = Z * sign
    n = np.diff(off)
    sd = np.array([[X[a:b, k].std() for k in range(X.shape[1])] for a, b in zip(off[:-1], off[1:])])
    const = np.repeat(sd < 1e-8, n, axis=0)
    V = np.zeros((len(zp), 1 + Z.shape[1]), int)
    sdp = np.array([prm_risk[a:b].std() for a, b in zip(off[:-1], off[1:])])
    V[:, 0] = E.topk_flags(zp, off, E.gap_counts(zp, off)) & ~np.repeat(sdp < 1e-8, n)
    for k in range(Z.shape[1]):
        zk = E.answer_z(Z[:, k], off)
        f = E.topk_flags(zk, off, E.gap_counts(zk, off)); f[const[:, k]] = False
        V[:, k + 1] = f
    return V, Z, sign


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
def _loglik_terms(G, psi, phi, phi0):
    lv = lambda p: G @ np.log(p) + (1 - G) @ np.log1p(-p)
    return lv(psi), lv(phi), lv(phi0)                          # per step: l1 (error step), le (correct, has-error), lc (clean)


def e_step(G, off, pos, prm):
    rho, a, b, psi, phi, phi0 = prm
    l1, le, lc = _loglik_terms(G, psi, phi, phi0)
    pi = np.clip(expit(a + b * pos), CLIP, 1 - CLIP)
    t1 = np.log(pi) + l1; t0 = np.log1p(-pi) + le
    lmix = np.logaddexp(t1, t0)                                 # per step, unconstrained has-error mixture
    starts = off[:-1]
    N = np.add.reduceat(t0, starts)                             # log of the all-correct configuration
    Dd = np.add.reduceat(np.logaddexp(0, t1 - t0), starts)      # M - N, > 0
    lnum = N + np.log(np.expm1(Dd))                             # log(e^M - e^N), stable (review S1)
    Lz0 = np.add.reduceat(np.log1p(-pi), starts)                # log P(all y = 0 | pi)
    lnorm = np.log(-np.expm1(Lz0))                              # log(1 - prod(1 - pi))
    L1 = lnum - lnorm
    L0 = np.add.reduceat(lc, starts)
    la1, la0 = np.log(rho) + L1, np.log1p(-rho) + L0
    ll = np.logaddexp(la1, la0)
    w = np.exp(la1 - ll)                                        # P(has-error | votes)
    n = np.diff(off)
    W = np.repeat(w, n)
    qy_unc = np.exp(t1 - lmix)                                  # P(y=1 | votes, has-error, no constraint)
    corr = np.repeat(1.0 / (-np.expm1(-Dd)), n)                 # e^M / (e^M - e^N)
    qy = np.clip(qy_unc * corr, 0, 1)
    q = W * qy
    return w, q, W, float(ll.sum())


def m_step(G, off, pos, w, q, W, a, b):
    eps = 1e-12
    rho = np.clip(w.mean(), CLIP, 1 - CLIP)
    psi = np.clip((q @ G) / (q.sum() + eps), CLIP, 1 - CLIP)
    r = np.clip(W - q, 0, None)
    phi = np.clip((r @ G) / (r.sum() + eps), CLIP, 1 - CLIP)
    c = 1 - W
    phi0 = np.clip((c @ G) / (c.sum() + eps), CLIP, 1 - CLIP)
    # pi(p): maximize the exact expected objective in (a, b), including -w_a*log(1 - prod(1 - pi)) from the
    # at-least-one normalizer (review B1); Newton direction from the standard part, backtracking on the full objective
    X = np.column_stack([np.ones_like(pos), pos]); starts = off[:-1]

    def obj(bt):
        pi = np.clip(expit(X @ bt), CLIP, 1 - CLIP)
        lz0 = np.add.reduceat(np.log1p(-pi), starts)
        return float(q @ np.log(pi) + (W - q) @ np.log1p(-pi) - w @ np.log(-np.expm1(lz0)))

    beta = np.array([a, b], float); f0 = obj(beta)
    for _ in range(25):
        mu = expit(X @ beta)
        lz0 = np.add.reduceat(np.log1p(-np.clip(mu, CLIP, 1 - CLIP)), starts)
        P0 = np.exp(lz0); ratio = np.repeat(w * P0 / (-np.expm1(lz0)), np.diff(off))
        g = X.T @ (q - W * mu) - X.T @ (ratio * mu)
        H = (X * ((W + ratio) * mu * (1 - mu))[:, None]).T @ X + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g); t = 1.0
        while t > 1e-4:
            cand = beta + t * step; f1 = obj(cand)
            if f1 >= f0 - 1e-9: break
            t /= 2
        if t <= 1e-4: break
        beta, done = cand, abs(f1 - f0) < 1e-10; f0 = f1
        if done: break
    return (rho, float(beta[0]), float(beta[1]), psi, phi, phi0)


def fit_hds(G, off, pos, n_starts=5, iters=500, tol=1e-8, seed=20261029):
    """Returns dict(params, p_answer, q, loglik, n_iter, start) of the best start by final log-likelihood."""
    rng = np.random.default_rng(seed)
    K = G.shape[1]; n = len(G); rate = np.clip(G.mean(0), .02, .98)
    share = G.mean(1)
    best = None
    for s in range(n_starts):
        if s == 0:                                              # start from the vote-share posterior
            q0 = np.clip(share, .01, .99); W0 = np.ones(n); w0 = np.full(len(off) - 1, .8)
            prm = m_step(G, off, pos, w0, q0 * W0, W0, logit(np.clip(q0.mean(), .02, .98)), 0.)
        else:
            psi = rng.uniform(.5, .95, K); phi = rng.uniform(.02, .3, K); phi0 = np.clip(rate * rng.uniform(.7, 1.3, K), CLIP, 1 - CLIP)
            prm = (rng.uniform(.3, .9), logit(rng.uniform(.05, .4)), rng.uniform(-1, 2), psi, phi, phi0)
        prev = -np.inf; drops = []
        for it in range(iters):
            w, q, W, ll = e_step(G, off, pos, prm)
            if ll < prev - 1e-6: drops.append(prev - ll)          # monitored (review B1 / N5)
            prm = m_step(G, off, pos, w, q, W, prm[1], prm[2])
            if 0 <= ll - prev < tol * (len(off) - 1): break       # per-answer relative stop on an increase
            prev = ll
        w, q, W, ll = e_step(G, off, pos, prm)
        if best is None or ll > best['loglik']:
            best = dict(params=prm, p_answer=w, q=q, loglik=ll, n_iter=it + 1, start=s, ll_drops=len(drops),
                        worst_drop=float(max(drops)) if drops else 0.0)
    rho, a, b, psi, phi, phi0 = best['params']
    if psi.mean() < phi.mean():
        best['label_switch_warning'] = True                     # reported, never silently flipped after the fact
    best['degenerate'] = dict(rho_at_clip=bool(rho <= 2 * CLIP or rho >= 1 - 2 * CLIP), phi0_above_psi=bool(phi0.mean() > psi.mean()))
    best['params'] = dict(rho=float(rho), a=a, b=b, psi=psi.tolist(), phi=phi.tolist(), phi0=phi0.tolist())
    return best
