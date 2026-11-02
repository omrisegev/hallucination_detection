"""Answer-level hurdle HMM on raw (not answer-standardized) step scores. Label-free.

Model, per answer j with steps i = 1..n (x = raw pooled score, oriented so higher = more error):
  b_j ~ N(0, tau^2)                                  answer effect ("difficulty"), integrated by Gauss-Hermite
  A_j ~ Bernoulli(pi)                                answer has at least one error
  A_j = 0: all y_i = 0
  A_j = 1: y follows a 2-state Markov chain (initial P(y_1=1)=e1, hazard P(0->1)=h, stay P(1->1)=rho),
           CONDITIONED on at least one error. Implemented exactly with 3 states:
           N (correct, no error yet), E (error), C (correct after an error); accept paths ending in E or C.
  x_i | y_i=0, b ~ N(mu0 + b, sig0^2);  x_i | y_i=1, b ~ N(mu0 + delta + b, sig1^2),  delta > 0.
Anchor (stage A, before stage B): marginal clean law N(mu0, s0^2) is fitted by truncated-normal MLE to the
steps below the pooled q_anchor quantile (the "certified clean" region). Stage B fits
theta = (pi, e1, h, rho, delta, sig1, f) by L-BFGS on the exact marginal likelihood, with sig0^2 = (1-f) s0^2,
tau^2 = f s0^2 (so the marginal clean law stays the anchored one).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

Q_NODES = 20
ADAPTIVE = True   # amendment candidate (2026-11-02): v1 used False (prior-centred nodes)


def pad(x, off):
    n = np.diff(off); A, L = len(n), int(n.max())
    X = np.zeros((A, L)); M = np.zeros((A, L), bool)
    for a in range(A):
        X[a, :n[a]] = x[off[a]:off[a + 1]]; M[a, :n[a]] = True
    return X, M, n


def anchor_fit(x, q):
    """Truncated-normal MLE of (mu0, s0) on x < t, t = q-quantile of x."""
    t = np.quantile(x, q); z = x[x < t]
    def nll(p):
        m, ls = p; s = np.exp(ls)
        return -(norm.logpdf(z, m, s).sum() - len(z) * norm.logcdf(t, m, s))
    r = minimize(nll, [np.median(z), np.log(z.std() + 1e-6)], method='Nelder-Mead',
                 options=dict(xatol=1e-7, fatol=1e-9, maxiter=4000))
    return float(r.x[0]), float(np.exp(r.x[1])), float(t)


def unpack(th):
    pi, e1, h, rho = expit(th[0]), expit(th[1]), expit(th[2]), expit(th[3])
    delta, sig1, f = np.exp(th[4]), np.exp(th[5]), expit(th[6])
    return dict(pi=pi, e1=e1, h=h, rho=rho, delta=delta, sig1=sig1, f=f)


def _lnorm(x, m, s):
    return -0.5 * ((x - m) / s) ** 2 - np.log(s) - 0.9189385332046727


def _forward(X, M, P, mu0, s0, need_backward=False):
    """Scaled linear-space forward(-backward). Returns per-answer log P(x) and, if asked,
    P(A=1|x) and P(y_i=1|x). Shapes: X, M (A, L)."""
    A, L = X.shape
    tk, wk = np.polynomial.hermite.hermgauss(Q_NODES)
    tau = np.sqrt(P['f']) * s0; sig0 = np.sqrt(1 - P['f']) * s0
    if ADAPTIVE:
        # adaptive Gauss-Hermite: centre and scale the nodes on the clean-only posterior of b per answer;
        # the weights carry the prior/proposal ratio, so the integral is unchanged in exact arithmetic.
        nn = M.sum(1); prec = nn / sig0 ** 2 + 1 / tau ** 2
        bc = ((np.where(M, X, 0).sum(1) - nn * mu0) / sig0 ** 2) / prec; bs = 1 / np.sqrt(prec)
        b2 = bc[:, None] + np.sqrt(2) * bs[:, None] * tk[None]                  # (A,Q)
        lw = (np.log(wk)[None] + tk[None] ** 2 + np.log(np.sqrt(2) * bs)[:, None]
              - 0.5 * (b2 / tau) ** 2 - np.log(tau) - 0.9189385332046727)
        Xq = X[:, None, :] - b2[:, :, None]
    else:
        b = np.sqrt(2) * tau * tk
        lw = np.broadcast_to(np.log(wk / np.sqrt(np.pi)), (A, Q_NODES))
        Xq = X[:, None, :] - b[None, :, None]                    # (A,Q,L)
    l0 = np.where(M[:, None, :], _lnorm(Xq, mu0, sig0), 0.0)
    l1 = np.where(M[:, None, :], _lnorm(Xq, mu0 + P['delta'], P['sig1']), 0.0)
    c = np.maximum(l0, l1)
    e0, e1v = np.exp(l0 - c), np.exp(l1 - c)                     # padded positions: e0 = e1 = 1
    logL0 = l0.sum(2)
    h, rho, e1 = P['h'], P['rho'], P['e1']
    T = np.array([[1 - h, h, 0.0], [0.0, rho, 1 - rho], [0.0, h, 1 - h]])
    al = np.empty((A, Q_NODES, L, 3)); ls = np.zeros((A, Q_NODES))
    a = np.stack([(1 - e1) * e0[:, :, 0], e1 * e1v[:, :, 0], np.zeros_like(e0[:, :, 0])], -1)
    sc = a.sum(-1); a /= sc[..., None]; ls += np.log(sc) + c[:, :, 0]; al[:, :, 0] = a
    for i in range(1, L):
        nxt = (a @ T) * np.stack([e0[:, :, i], e1v[:, :, i], e0[:, :, i]], -1)
        sc = nxt.sum(-1); nxt /= sc[..., None]
        m = M[:, None, i]
        a = np.where(m[..., None], nxt, a); ls += np.where(m, np.log(sc) + c[:, :, i], 0.0)
        al[:, :, i] = a
    n = M.sum(1)
    logpe = np.log1p(-(1 - e1) * (1 - h) ** (n - 1))
    acc = a[..., 1] + a[..., 2]
    chainL = ls + np.log(np.maximum(acc, 1e-300))                 # log numerator (A,Q)
    logL1 = chainL - logpe[:, None]
    lj0 = np.log(1 - P['pi']) + logL0; lj1 = np.log(P['pi']) + logL1
    lq = lw + np.logaddexp(lj0, lj1)
    mx = lq.max(1); ll = mx + np.log(np.exp(lq - mx[:, None]).sum(1))
    if not need_backward:
        return ll
    acc_v = np.array([0.0, 1.0, 1.0])
    be = np.empty((A, Q_NODES, L, 3)); bt = np.broadcast_to(acc_v, (A, Q_NODES, 3)).copy(); be[:, :, -1] = bt
    for i in range(L - 2, -1, -1):
        em = np.stack([e0[:, :, i + 1], e1v[:, :, i + 1], e0[:, :, i + 1]], -1)
        prv = (T[None, None] @ (em * bt)[..., None])[..., 0]
        prv /= np.maximum(prv.max(-1, keepdims=True), 1e-300)
        m = M[:, None, i + 1]
        bt = np.where(m[..., None], prv, acc_v); be[:, :, i] = bt
    g = al * be
    gE = g[..., 1] / np.maximum(g.sum(-1), 1e-300)               # P(y_i=1 | x, b_k, A=1)
    wq = np.exp(lq - ll[:, None])
    pA1_q = np.exp(lj1 - np.logaddexp(lj0, lj1))
    pA1 = (wq * pA1_q).sum(1)
    py = ((wq * pA1_q)[:, :, None] * gE).sum(1)
    return ll, pA1, np.where(M, py, np.nan)


def fit(x, off, q_anchor=0.30, max_fit_answers=2000, seed=0, th0=None, anchor=None, fixed_pi=None):
    """anchor=(mu0, s0) skips stage A (v2: clean law from labelled answers). fixed_pi fixes pi (v2)."""
    if anchor is None:
        mu0, s0, t = anchor_fit(x, q_anchor)
    else:
        (mu0, s0), t = anchor, float('nan')
    X, M, n = pad(x, off)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(n), min(len(n), max_fit_answers), replace=False))
    Xf, Mf = X[idx], M[idx]
    Xf, Mf = Xf[:, :Mf.sum(1).max()], Mf[:, :Mf.sum(1).max()]
    nll = lambda th: -_forward(Xf, Mf, unpack(th), mu0, s0).sum() / len(idx)
    starts = [th0] if th0 is not None else [
        np.array([0.0, -2.0, -2.5, 0.0, np.log(1.0), np.log(s0), 0.0]),
        np.array([1.5, -1.0, -2.0, 0.5, np.log(1.5), np.log(s0), -1.0]),
        np.array([-1.5, -2.5, -3.0, -0.5, np.log(0.7), np.log(s0), 1.0])]
    best = None
    for s in starts:
        bounds = [(-8, 8)] * 4 + [(-4, 3), (-4, 3), (-6, 6)]
        if fixed_pi is not None:
            lp = float(np.log(fixed_pi / (1 - fixed_pi))); bounds[0] = (lp, lp); s = np.array(s, float); s[0] = lp
        r = minimize(nll, s, method='L-BFGS-B', bounds=bounds,
                     options=dict(maxiter=300))
        if best is None or r.fun < best.fun:
            best = r
    return dict(theta=best.x, params=unpack(best.x), mu0=mu0, s0=s0, anchor_t=t, nll=best.fun,
                converged=bool(best.success), n_fit=len(idx))


def score(x, off, fitres, chunk=1500):
    X, M, n = pad(x, off)
    P = fitres['params']; out_pA, out_py, out_ll = [], [], []
    for a in range(0, len(n), chunk):
        Xc, Mc = X[a:a + chunk], M[a:a + chunk]
        Lc = Mc.sum(1).max(); Xc, Mc = Xc[:, :Lc], Mc[:, :Lc]
        ll, pA, py = _forward(Xc, Mc, P, fitres['mu0'], fitres['s0'], need_backward=True)
        out_ll.append(ll); out_pA.append(pA); out_py.extend(py[r, :n[a + r]] for r in range(len(pA)))
    return np.concatenate(out_ll), np.concatenate(out_pA), np.concatenate(out_py)


def simulate(n_steps, pi, e1, h, rho, delta, sig1, f, mu0=0.0, s0=1.0, seed=0):
    rng = np.random.default_rng(seed); xs, ys, As = [], [], []
    tau, sig0 = np.sqrt(f) * s0, np.sqrt(1 - f) * s0
    for n in n_steps:
        A = rng.random() < pi; b = rng.normal(0, tau)
        while True:
            y = np.zeros(n, bool)
            if A:
                y[0] = rng.random() < e1
                for i in range(1, n):
                    y[i] = rng.random() < (rho if y[i - 1] else h)
                if not y.any():
                    continue
            break
        x = np.where(y, rng.normal(mu0 + delta + b, sig1, n), rng.normal(mu0 + b, sig0, n))
        xs.append(x); ys.append(y); As.append(A)
    off = np.concatenate([[0], np.cumsum(n_steps)])
    return np.concatenate(xs), off, np.concatenate(ys), np.array(As)
