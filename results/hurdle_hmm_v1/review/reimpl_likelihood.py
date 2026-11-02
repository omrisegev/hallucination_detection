"""Independent re-implementation of the hurdle-HMM likelihood and posteriors, written from the PREREG text.
Two independent routes, neither copies hurdle_hmm._forward:
  (R1) brute-force enumeration over all 2^n step labels, b integrated by adaptive quadrature (scipy.integrate.quad),
       so the Gauss-Hermite rule of the module is NOT shared;
  (R2) for long answers: a standard 2-state HMM in log space minus the all-clean path
       (sum over y != 0 = sum over all y - P(y = 0)), integrated over b by quad.
Also reports how far the module's 20-node Gauss-Hermite rule is from the adaptive integral (quadrature error)."""
import importlib.util, itertools, pathlib, sys, json
import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import logsumexp
from scipy.stats import norm

sp = importlib.util.spec_from_file_location('hh', pathlib.Path(__file__).resolve().parents[3] / 'spectral_utils' / 'hurdle_hmm.py')
H = importlib.util.module_from_spec(sp); sp.loader.exec_module(H)


def chain_logp(y, e1, h, rho):
    lp = np.log(e1 if y[0] else 1 - e1)
    for i in range(1, len(y)):
        if y[i - 1]:
            lp += np.log(rho if y[i] else 1 - rho)
        else:
            lp += np.log(h if y[i] else 1 - h)
    return lp


_CACHE = {}


def cond_given_b_brute(x, b, P, mu0, s0):
    """Returns log p(x|b), log p(x, A=1|b), and p(x, y_i=1|b) (vector) by enumeration."""
    n = len(x); sig0 = np.sqrt(1 - P['f']) * s0
    l0 = norm.logpdf(x, mu0 + b, sig0); l1 = norm.logpdf(x, mu0 + P['delta'] + b, P['sig1'])
    p_any = 1 - (1 - P['e1']) * (1 - P['h']) ** (n - 1)
    key = (n, P['e1'], P['h'], P['rho'])
    if key not in _CACHE:
        ys = np.array([y for y in itertools.product([0, 1], repeat=n) if any(y)], bool)
        _CACHE[key] = (ys, np.array([chain_logp(y, P['e1'], P['h'], P['rho']) for y in ys]))
    ys, cl = _CACHE[key]
    terms = cl + ys @ l1 + (~ys) @ l0
    lA1 = np.log(P['pi']) - np.log(p_any) + logsumexp(terms)
    lA0 = np.log(1 - P['pi']) + l0.sum()
    lx = np.logaddexp(lA0, lA1)
    lyi = np.array([np.log(P['pi']) - np.log(p_any) + (logsumexp(terms[ys[:, i]]) if ys[:, i].any() else -np.inf)
                    for i in range(n)])
    return lx, lA1, lyi


def cond_given_b_hmm(x, b, P, mu0, s0):
    """Route R2: log-space 2-state forward-backward minus the all-zero path."""
    n = len(x); sig0 = np.sqrt(1 - P['f']) * s0
    le = np.stack([norm.logpdf(x, mu0 + b, sig0), norm.logpdf(x, mu0 + P['delta'] + b, P['sig1'])], 1)
    lT = np.log(np.array([[1 - P['h'], P['h']], [1 - P['rho'], P['rho']]]))
    la = np.zeros((n, 2)); la[0] = np.log([1 - P['e1'], P['e1']]) + le[0]
    for i in range(1, n):
        la[i] = logsumexp(la[i - 1][:, None] + lT, 0) + le[i]
    lb = np.zeros((n, 2))
    for i in range(n - 2, -1, -1):
        lb[i] = logsumexp(lT + (le[i + 1] + lb[i + 1])[None, :], 1)
    lall = logsumexp(la[-1]); lzero = np.log(1 - P['e1']) + (n - 1) * np.log(1 - P['h']) + le[:, 0].sum()
    # log(exp(lall) - exp(lzero)); exact as long as the difference is not lost to rounding
    lnz = lall + np.log1p(-np.exp(lzero - lall))
    p_any = 1 - (1 - P['e1']) * (1 - P['h']) ** (n - 1)
    lA1 = np.log(P['pi']) - np.log(p_any) + lnz
    lA0 = np.log(1 - P['pi']) + le[:, 0].sum()
    lyi = np.log(P['pi']) - np.log(p_any) + la[:, 1] + lb[:, 1]
    return np.logaddexp(lA0, lA1), lA1, lyi


def integrate(x, P, mu0, s0, route):
    tau = np.sqrt(P['f']) * s0
    f = cond_given_b_brute if route == 'brute' else cond_given_b_hmm
    # centre the integral on the posterior mode of b to keep quad accurate for narrow posteriors
    grid = np.linspace(-6 * tau, 6 * tau, 241)
    lg = np.array([f(x, b, P, mu0, s0)[0] + norm.logpdf(b, 0, tau) for b in grid])
    c = lg.max(); bm = grid[lg.argmax()]
    def g(b):
        lx, lA1, lyi = f(x, b, P, mu0, s0)
        return np.exp(np.concatenate([[lx, lA1], lyi]) + norm.logpdf(b, 0, tau) - c)
    pts = sorted({-12 * tau, bm - 2 * tau, bm - .3 * tau, bm, bm + .3 * tau, bm + 2 * tau, 12 * tau})
    pts = [p for p in pts if -12 * tau <= p <= 12 * tau]
    v = sum(quad_vec(g, a, bb, epsabs=0, epsrel=1e-11, limit=400)[0] for a, bb in zip(pts[:-1], pts[1:]))
    return np.log(v[0]) + c, v[1] / v[0], v[2:] / v[0]


def gh_reference(x, P, mu0, s0, route='hmm'):
    """My own Gauss-Hermite (same 20 nodes) over route R2: separates algebra error from quadrature error."""
    tau = np.sqrt(P['f']) * s0; tk, wk = np.polynomial.hermite.hermgauss(H.Q_NODES)
    f = cond_given_b_brute if route == 'brute' else cond_given_b_hmm
    out = [f(x, np.sqrt(2) * tau * t, P, mu0, s0) for t in tk]
    lw = np.log(wk / np.sqrt(np.pi))
    lx = np.array([o[0] for o in out]) + lw; LL = logsumexp(lx)
    pA = np.exp(logsumexp(np.array([o[1] for o in out]) + lw) - LL)
    py = np.exp(logsumexp(np.array([o[2] for o in out]) + lw[:, None], 0) - LL)
    return LL, pA, py


def rand_params(rng):
    return dict(pi=rng.uniform(.05, .95), e1=rng.uniform(.02, .6), h=rng.uniform(.02, .5), rho=rng.uniform(.05, .95),
                delta=rng.uniform(.2, 3), sig1=rng.uniform(.5, 2), f=rng.uniform(.05, .8))


def run():
    rng = np.random.default_rng(12345); rows = []
    # --- A: brute force, short answers, padded batches, random params
    for trial in range(25):
        P = rand_params(rng); mu0, s0 = rng.normal(0, .5), rng.uniform(.5, 1.5)
        lens = list(rng.integers(1, 9, 4)) + [1]
        x = rng.normal(mu0 + .5, 1.5, sum(lens)); off = np.concatenate([[0], np.cumsum(lens)])
        X, M, n = H.pad(x, off)
        ll, pA, py = H._forward(X, M, P, mu0, s0, need_backward=True)
        for a in range(len(lens)):
            xa = x[off[a]:off[a + 1]]
            gl, gA, gy = gh_reference(xa, P, mu0, s0, 'brute')
            ql, qA, qy = integrate(xa, P, mu0, s0, 'brute')
            rows.append(dict(set='A_brute', n=lens[a], f=P['f'], d_ll_gh=abs(ll[a] - gl), d_pA_gh=abs(pA[a] - gA),
                             d_py_gh=np.abs(py[a, :lens[a]] - gy).max(), d_ll_quad=abs(ll[a] - ql),
                             d_pA_quad=abs(pA[a] - qA), d_py_quad=np.abs(py[a, :lens[a]] - qy).max()))
    # --- B: long answers (route R2), realistic lengths 10..60, include large f
    for trial in range(30):
        P = rand_params(rng); P['f'] = [0.1, 0.3, 0.5, 0.7, 0.85][trial % 5]
        mu0, s0 = 0.0, 1.0
        lens = [int(v) for v in rng.integers(8, 61, 3)]
        x = rng.normal(0.3, 1.0, sum(lens)) + np.repeat(rng.normal(0, np.sqrt(P['f']), 3), lens)
        off = np.concatenate([[0], np.cumsum(lens)]); X, M, n = H.pad(x, off)
        ll, pA, py = H._forward(X, M, P, mu0, s0, need_backward=True)
        for a in range(3):
            xa = x[off[a]:off[a + 1]]
            gl, gA, gy = gh_reference(xa, P, mu0, s0, 'hmm')
            ql, qA, qy = integrate(xa, P, mu0, s0, 'hmm')
            rows.append(dict(set='B_hmm_long', n=lens[a], f=P['f'], d_ll_gh=abs(ll[a] - gl), d_pA_gh=abs(pA[a] - gA),
                             d_py_gh=np.abs(py[a, :lens[a]] - gy).max(), d_ll_quad=abs(ll[a] - ql),
                             d_pA_quad=abs(pA[a] - qA), d_py_quad=np.abs(py[a, :lens[a]] - qy).max()))
    # --- C: outliers (numerical stability)
    P = dict(pi=.5, e1=.1, h=.1, rho=.4, delta=1.0, sig1=1.0, f=.3)
    for xa in [np.array([25.0]), np.array([-25.0, 0.0, 30.0]), np.array([0, 0, 0, 40.0, 0, 0])]:
        X, M, _ = H.pad(xa, np.array([0, len(xa)]))
        ll, pA, py = H._forward(X, M, P, 0.0, 1.0, need_backward=True)
        gl, gA, gy = gh_reference(xa, P, 0.0, 1.0, 'brute')
        rows.append(dict(set='C_outlier', n=len(xa), f=P['f'], d_ll_gh=abs(ll[0] - gl), d_pA_gh=abs(pA[0] - gA),
                         d_py_gh=np.abs(py[0, :len(xa)] - gy).max(), d_ll_quad=np.nan, d_pA_quad=np.nan, d_py_quad=np.nan))
    import pandas as pd
    d = pd.DataFrame(rows)
    d.to_csv(pathlib.Path(__file__).with_name('REIMPL_LIKELIHOOD.csv'), index=False)
    pd.set_option('display.width', 200)
    print(d.groupby('set')[['d_ll_gh', 'd_pA_gh', 'd_py_gh', 'd_ll_quad', 'd_pA_quad', 'd_py_quad']].max().to_string())
    print(d[d.set == 'B_hmm_long'].groupby('f')[['d_ll_quad', 'd_pA_quad', 'd_py_quad']].max().to_string())


if __name__ == '__main__':
    run()
