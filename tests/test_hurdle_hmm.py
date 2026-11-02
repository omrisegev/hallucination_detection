"""Exact checks of spectral_utils.hurdle_hmm against brute-force enumeration (small n)."""
import itertools, numpy as np
from scipy.stats import norm
import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location('hurdle_hmm', pathlib.Path(__file__).resolve().parents[1] / 'spectral_utils' / 'hurdle_hmm.py')
H = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(H)  # avoids the package __init__ (torch)

P = dict(pi=0.4, e1=0.2, h=0.15, rho=0.5, delta=1.3, sig1=0.9, f=0.3)
MU0, S0 = 0.1, 1.1


def brute(x):
    tk, wk = np.polynomial.hermite.hermgauss(H.Q_NODES)
    tau, sig0 = np.sqrt(P['f']) * S0, np.sqrt(1 - P['f']) * S0
    n = len(x); tot = 0.0; post_y = np.zeros(n); post_A = 0.0
    for t, w in zip(tk, wk):
        b = np.sqrt(2) * tau * t; wq = w / np.sqrt(np.pi)
        L0 = np.prod(norm.pdf(x, MU0 + b, sig0))
        tot += wq * (1 - P['pi']) * L0
        num = 0.0; ny = np.zeros(n)
        for y in itertools.product([0, 1], repeat=n):
            y = np.array(y)
            if not y.any(): continue
            p = P['e1'] if y[0] else 1 - P['e1']
            for i in range(1, n):
                p *= (P['rho'] if y[i] else 1 - P['rho']) if y[i - 1] else (P['h'] if y[i] else 1 - P['h'])
            e = np.prod(np.where(y, norm.pdf(x, MU0 + P['delta'] + b, P['sig1']), norm.pdf(x, MU0 + b, sig0)))
            num += p * e; ny += p * e * y
        pe = 1 - (1 - P['e1']) * (1 - P['h']) ** (n - 1)
        tot += wq * P['pi'] * num / pe; post_A += wq * P['pi'] * num / pe; post_y += wq * P['pi'] * ny / pe
    return np.log(tot), post_A / tot, post_y / tot


def test_exact_small():
    """Algebra check: same prior-centred quadrature as the brute force."""
    H.ADAPTIVE = False
    rng = np.random.default_rng(1)
    lens = [1, 2, 3, 5, 6]
    x = rng.normal(0.5, 1.2, sum(lens)); off = np.concatenate([[0], np.cumsum(lens)])
    X, M, n = H.pad(x, off)
    ll, pA, py = H._forward(X, M, P, MU0, S0, need_backward=True)
    for a in range(len(lens)):
        bl, bA, by = brute(x[off[a]:off[a + 1]])
        assert abs(ll[a] - bl) < 1e-8, (a, ll[a], bl)
        assert abs(pA[a] - bA) < 1e-8
        assert np.allclose(py[a, :n[a]], by, atol=1e-8)
    H.ADAPTIVE = True


def test_padding_invariance():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 9); off = np.array([0, 4, 9])
    X, M, _ = H.pad(x, off)
    ll_all = H._forward(X, M, P, MU0, S0)
    ll_one = H._forward(X[:1, :4], M[:1, :4], P, MU0, S0)
    assert abs(ll_all[0] - ll_one[0]) < 1e-10


def test_adaptive_vs_dense_integral():
    """Long answer: adaptive 20-node rule vs a dense trapezoid integral over b (log-likelihood and P(A=1|x))."""
    from scipy.special import logsumexp
    rng = np.random.default_rng(3); n = 107
    x = rng.normal(0.4, 1.0, n); off = np.array([0, n]); X, M, _ = H.pad(x, off)
    ll, pA, _ = H._forward(X, M, P, MU0, S0, need_backward=True)
    tau = np.sqrt(P['f']) * S0; grid = np.linspace(-8 * tau, 8 * tau, 4001); db = grid[1] - grid[0]
    lls, l1s = [], []
    for bb in grid:
        Pz = dict(P, f=1e-12)  # tau -> 0: likelihood at a fixed b, by shifting x
        old = H.ADAPTIVE; H.ADAPTIVE = False
        l, pa, _ = H._forward(X - bb, M, dict(P, f=1e-12), MU0, S0 * np.sqrt((1 - P['f']) / (1 - 1e-12)), need_backward=True)
        H.ADAPTIVE = old
        lp = -0.5 * (bb / tau) ** 2 - np.log(tau * np.sqrt(2 * np.pi))
        lls.append(l[0] + lp); l1s.append(l[0] + lp + np.log(max(pa[0], 1e-300)))
    dense = logsumexp(lls) + np.log(db); dense_pA = np.exp(logsumexp(l1s) - logsumexp(lls))
    assert abs(ll[0] - dense) < 1e-4, (ll[0], dense)
    assert abs(pA[0] - dense_pA) < 1e-4, (pA[0], dense_pA)


if __name__ == '__main__':
    test_exact_small(); test_padding_invariance(); test_adaptive_vs_dense_integral(); print("3 tests pass")
