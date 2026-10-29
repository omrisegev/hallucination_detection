"""E2 (ds_gate_e2_posrel): the corrected-E1 hierarchical Dawid-Skene model (ds_gate_v1b_fixed/ds_gate_b.py) with
POSITION-DEPENDENT RELIABILITY: the emission probabilities depend on a coarse relative-position bin t(i) of the step
(first / middle / last third: p < 1/3, p < 2/3, else), in addition to the 10-bin prior pi_bin(i).

  answer state z in {clean, has-error}, P(has-error) = rho;
  has-error answer: y_i ~ Bernoulli(pi_bin(i)) conditioned on >= 1 error (exact likelihood); clean answer: all y = 0;
  group votes independent given (z, y): P(vote_g = 1) = psi[t(i), g] (error step), phi[t(i), g] (correct step of a
  has-error answer), phi0[t(i), g] (step of a clean answer).
The prior update is ds_gate_b.m_step's exact (normalizer-including) Newton/backtracking step, unchanged; the emission
M-step is closed form per (position third, group). No labels anywhere in this file.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ds_gate_v1b_fixed'))
import ds_gate_b as B  # noqa: E402  (read-only reuse: cleaning, votes, grouping, bin design, prior update)

CLIP = B.CLIP
NT = 3


def third(pos):
    """Position third per step: 0 if p < 1/3, 1 if p < 2/3, else 2."""
    return np.where(pos < 1 / 3, 0, np.where(pos < 2 / 3, 1, 2))


def _lv(G, P):
    """Per-step log-likelihood of the group votes G (steps x K) under per-step probabilities P (steps x K)."""
    return (G * np.log(P) + (1 - G) * np.log1p(-P)).sum(1)


def e_step(G, off, D, t, prm):
    rho, theta, psi, phi, phi0 = prm                            # psi/phi/phi0: (NT, K)
    l1, le, lc = _lv(G, psi[t]), _lv(G, phi[t]), _lv(G, phi0[t])
    pi = np.clip(expit(D @ theta), CLIP, 1 - CLIP)
    t1 = np.log(pi) + l1; t0 = np.log1p(-pi) + le
    lmix = np.logaddexp(t1, t0); starts = off[:-1]
    N = np.add.reduceat(t0, starts)
    Dd = np.add.reduceat(np.logaddexp(0, t1 - t0), starts)
    lnum = N + np.log(np.expm1(Dd))                              # configurations with >= 1 error
    Lz0 = np.add.reduceat(np.log1p(-pi), starts)
    L1 = lnum - np.log(-np.expm1(Lz0))                           # divided by P(>= 1 error | pi)
    L0 = np.add.reduceat(lc, starts)
    la1, la0 = np.log(rho) + L1, np.log1p(-rho) + L0
    ll = np.logaddexp(la1, la0)
    w = np.exp(la1 - ll); n = np.diff(off); W = np.repeat(w, n)
    qy = np.clip(np.exp(t1 - lmix) * np.repeat(1.0 / (-np.expm1(-Dd)), n), 0, 1)
    return w, W * qy, W, float(ll.sum())


def _per_third(G, t, wts):
    eps = 1e-12
    return np.clip(np.stack([(wts[t == b] @ G[t == b]) / (wts[t == b].sum() + eps) for b in range(NT)]), CLIP, 1 - CLIP)


def m_step(G, off, D, t, w, q, W, theta):
    rho, theta, _, _, _ = B.m_step(G, off, D, w, q, W, theta)   # rho and the exact prior update (emissions discarded)
    psi = _per_third(G, t, q)
    phi = _per_third(G, t, np.clip(W - q, 0, None))
    phi0 = _per_third(G, t, 1 - W)
    return (rho, theta, psi, phi, phi0)


def fit_e2(G, off, D, t, n_starts=5, iters=500, tol=1e-8, seed=20261029):
    rng = np.random.default_rng(seed)
    K = G.shape[1]; n = len(G); P = D.shape[1]; rate = np.clip(G.mean(0), .02, .98); share = G.mean(1)
    best = None
    for s in range(n_starts):
        if s == 0:                                               # start from the vote-share posterior
            q0 = np.clip(share, .01, .99); W0 = np.ones(n); w0 = np.full(len(off) - 1, .8)
            prm = m_step(G, off, D, t, w0, q0, W0, np.full(P, logit(np.clip(q0.mean(), .02, .98))))
        else:
            prm = (rng.uniform(.3, .9), logit(rng.uniform(.05, .4, P)), rng.uniform(.5, .95, (NT, K)),
                   rng.uniform(.02, .3, (NT, K)), np.clip(rate * rng.uniform(.7, 1.3, (NT, K)), CLIP, 1 - CLIP))
        prev = -np.inf; drops = []
        for it in range(iters):
            w, q, W, ll = e_step(G, off, D, t, prm)
            if ll < prev - 1e-6: drops.append(prev - ll)
            prm = m_step(G, off, D, t, w, q, W, prm[1])
            if 0 <= ll - prev < tol * (len(off) - 1): break
            prev = ll
        w, q, W, ll = e_step(G, off, D, t, prm)
        if best is None or ll > best['loglik']:
            best = dict(params=prm, p_answer=w, q=q, loglik=ll, n_iter=it + 1, start=s, ll_drops=len(drops),
                        worst_drop=float(max(drops)) if drops else 0.0)
    rho, theta, psi, phi, phi0 = best['params']
    best['label_switch_warning'] = bool(psi.mean() < phi.mean())
    best['label_switch_by_third'] = [bool(psi[b].mean() < phi[b].mean()) for b in range(NT)]
    best['degenerate'] = dict(rho_at_clip=bool(rho <= 2 * CLIP or rho >= 1 - 2 * CLIP), phi0_above_psi=bool(phi0.mean() > psi.mean()))
    best['params'] = dict(rho=float(rho), pi_bins=expit(theta).tolist(), psi=psi.tolist(), phi=phi.tolist(), phi0=phi0.tolist())
    return best
