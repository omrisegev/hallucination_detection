"""Synthetic tests for spectral_utils.markov_ds_count (pre-registration amendment A8). No real data, no labels.

    python -m pytest tests/test_markov_ds_count.py -q -s
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils import markov_ds_count as M  # noqa: E402


def answer_z(s, off):
    out = np.zeros(len(s))
    for a, b in zip(off[:-1], off[1:]):
        v = s[a:b]; out[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
    return out


def gap_counts(z, off):
    """Same rule as evaluate_collection.gap_counts."""
    k = []
    for a, b in zip(off[:-1], off[1:]):
        m = b - a
        if m == 1: k.append(1); continue
        zz = np.sort(z[a:b])[::-1]; km = min(int(np.ceil(m / 2)), m - 1)
        k.append(int(np.argmax(zz[:km] - zz[1:km + 1]) + 1))
    return np.array(k)


def generate(N, delta, e, rho, seed, nlo=4, nhi=10, trend=0.):
    """Markov labels conditioned on >= 1 error; x = a + b (delta y + trend * position + eps), random a, b per voter."""
    rng = np.random.default_rng(seed); ys, xs, ns = [], [], []
    while len(ns) < N:
        n = int(rng.integers(nlo, nhi + 1)); y = np.zeros(n, bool); y[0] = rng.random() < e
        for t in range(1, n): y[t] = rng.random() < (rho if y[t - 1] else e)
        if not y.any(): continue
        a = rng.normal(0, 3, len(delta)); b = np.exp(rng.normal(0, .7, len(delta)))
        x = a + b * (np.outer(y, delta) + trend * np.linspace(0, 1, n)[:, None] + rng.normal(size=(n, len(delta))))
        ys.append(y); xs.append(x); ns.append(n)
    off = np.r_[0, np.cumsum(ns)]
    X = np.vstack(xs); Z = np.column_stack([answer_z(X[:, g], off) for g in range(len(delta))])
    return Z, np.concatenate(ys), off


DELTA, E0, RHO = np.array([1.5, 1., .8, .6]), .15, .5


@lru_cache(None)
def main_fit():
    Z, y, off = generate(1500, DELTA, E0, RHO, seed=11)
    X = M.Exact(Z, off, 16)
    return Z, y, off, X, M.fit_exact(Z, off, 16, X=X), M.fit_exact(Z, off, 16, markov=False, X=X)


def mae(k, y, off): return float(np.abs(k - np.add.reduceat(y.astype(int), off[:-1])).mean())


# ----------------------------------------------------------------------------------------------- numerics
def test_log_j_against_quadrature_and_recursion():
    worst = 0.
    for d in (2, 3, 5, 10, 15, 40, 150):
        for t in (-60., -20., -5., -1., -.1, 0., .3, 1., 3., 6., 15., 40., 60.):
            rm = (t + np.sqrt(t * t + 4 * (d - 1))) / 2; c = (d - 1) * np.log(rm) - rm * rm / 2 + rm * t
            f = lambda r: np.exp((d - 1) * np.log(r) - r * r / 2 + r * t - c) if r > 0 else 0.
            v = quad(f, 0, rm, limit=500, epsabs=0, epsrel=1e-13)[0] + quad(f, rm, np.inf, limit=500, epsabs=0, epsrel=1e-13)[0]
            worst = max(worst, abs(float(M.log_j(d, t)) - np.log(v) - c))
            if 0 <= t <= 6 and d <= 40:
                worst = max(worst, abs(float(M.log_j(d, t)) - M.log_j_recursion(d, t)))
    assert worst < 1e-9, worst


def test_table_against_direct():
    T = M.JTable(15); rng = np.random.default_rng(0)
    d = rng.integers(2, 16, 50000); tau = rng.uniform(-70, 70, 50000)        # includes values outside the table
    a, b = T(d, tau); c, p = M.log_j(d, tau, with_psi=True)
    assert np.abs(a - c).max() < 1e-8 and np.abs(b - p).max() < 1e-8


def test_direction_density_n4_monte_carlo():
    """n = 4 (d = 3): w = u . m_hat has density (1/2) exp(-|m|^2/2) J_3(|m| w) / J_3(0) on [-1, 1]."""
    rng = np.random.default_rng(1); y = np.array([0, 1, 1, 0.]); delta = 1.7
    x = 5 + 2 * (delta * y + rng.normal(size=(400000, 4)))
    v = x - x.mean(1, keepdims=True); u = v / np.linalg.norm(v, axis=1, keepdims=True)
    m = delta * (y - y.mean()); mn = np.linalg.norm(m); w = u @ (m / mn)
    edges = np.linspace(-1, 1, 41); h, _ = np.histogram(w, edges, density=True)
    mid = (edges[1:] + edges[:-1]) / 2
    pred = .5 * np.exp(-mn ** 2 / 2 + M.log_j(3, mn * mid) - M.log_j(3, 0.))
    assert np.abs(h - pred).max() < .03, np.abs(h - pred).max()
    fine = np.linspace(-1, 1, 20001)
    assert abs(np.trapezoid(.5 * np.exp(-mn ** 2 / 2 + M.log_j(3, mn * fine) - M.log_j(3, 0.)), fine) - 1) < 1e-6


@pytest.mark.parametrize('markov', [True, False])
def test_exact_gradient(markov):
    Z, y, off = generate(200, DELTA, E0, RHO, seed=3)
    X = M.Exact(Z, off, 16); th = np.r_[np.log(DELTA) + .1, -1.5, .2][:6 if markov else 5]
    _, g = X.objective(th, None, markov); h = 1e-6
    fd = [(X.objective(th + h * np.eye(len(th))[j], None, markov)[0] - X.objective(th - h * np.eye(len(th))[j], None, markov)[0]) / 2 / h
          for j in range(len(th))]
    assert np.allclose(g, fd, rtol=1e-4, atol=1e-4), (g, fd)


def test_topk_gradient():
    Z, y, off = generate(200, DELTA, E0, RHO, seed=4); f = Z.mean(1)
    T = M.TopK(Z, f, off); th = np.r_[np.log(DELTA), -1.5, .2]
    _, g = T.objective(th); h = 1e-6
    fd = [(T.objective(th + h * np.eye(6)[j])[0] - T.objective(th - h * np.eye(6)[j])[0]) / 2 / h for j in range(6)]
    assert np.allclose(g, fd, rtol=1e-4, atol=1e-4)


def test_exact_enumeration_matches_brute_force():
    """For one answer, the Exact log-likelihood equals a direct sum over configurations with per-configuration
    log_j (no table) and the explicit Markov prior."""
    Z, y, off = generate(1, DELTA, E0, RHO, seed=5, nlo=6, nhi=6)
    X = M.Exact(Z, off, 16); th = np.r_[np.log(DELTA), np.log(.2 / .8), np.log(.45 / .55)]
    nll, _ = X.objective(th)
    import itertools
    n = 6; e, rho = .2, .45; tot = []
    U = np.column_stack([M._unit_centered(Z[:, g])[0] for g in range(4)])
    for cfg in itertools.product([0, 1], repeat=n):
        yv = np.array(cfg)
        if not yv.any(): continue
        lp = np.log(e if yv[0] else 1 - e)
        for t in range(1, n):
            p1 = rho if yv[t - 1] else e; lp += np.log(p1 if yv[t] else 1 - p1)
        lp -= np.log(1 - (1 - e) ** n); k = yv.sum()
        for g in range(4):
            lp += -DELTA[g] ** 2 * k * (n - k) / (2 * n) + M.log_j(n - 1, DELTA[g] * U[yv == 1, g].sum()) - M.log_j(n - 1, 0.)
        tot.append(lp)
    assert abs(-nll - np.logaddexp.reduce(tot)) < 1e-7


# ----------------------------------------------------------------------------------------------- recovery and counting
def test_recovery_exact():
    Z, y, off, X, F, _ = main_fit()
    print('\nexact fit: delta', np.round(F['delta'], 3), 'e %.3f rho %.3f share %.3f (true %.3f) valid %s'
          % (F['e'], F['rho'], F['implied_share'], y.mean(), F['valid']))
    assert F['valid'], F['reason']
    assert np.all(np.abs(np.array(F['delta']) - DELTA) <= .2)
    assert abs(F['e'] - E0) <= .06 and abs(F['rho'] - RHO) <= .12


def test_count_beats_gap_and_share():
    Z, y, off, X, F, _ = main_fit()
    k, _ = M.counts_exact(F['theta'], X)
    assert (k >= 1).all()
    kg = gap_counts(answer_z(Z.mean(1), off), off); ks = M.share_counts(F['implied_share'], off)
    print('\nMAE exact %.3f GAP %.3f SHARE %.3f' % (mae(k, y, off), mae(kg, y, off), mae(ks, y, off)))
    assert mae(k, y, off) < mae(kg, y, off) and mae(k, y, off) < mae(ks, y, off)


def test_markov_not_worse_than_independent():
    Z, y, off, X, F, FI = main_fit()
    k, _ = M.counts_exact(F['theta'], X); ki, _ = M.counts_exact(FI['theta'], X, markov=False)
    print('\nMAE Markov %.3f independent %.3f (IND e %.3f)' % (mae(k, y, off), mae(ki, y, off), FI['e']))
    assert mae(k, y, off) <= mae(ki, y, off)


def test_topk_secondary_runs_and_is_documented():
    Z, y, off, X, F, _ = main_fit()
    T = M.TopK(Z, Z.mean(1), off); FT = M.fit_topk(Z, Z.mean(1), off, T=T)
    print('\nTOPK fit: e %.3f rho %.3f share %.3f, MAE %.3f' % (FT['e'], FT['rho'], FT['implied_share'], mae(T.counts(FT['theta']), y, off)))
    assert len(FT['starts']) == 5


def test_position_trend_negative_control_documented():
    """A trend shared by all voters violates conditional independence. Behavior is documented, not asserted."""
    Z, y, off = generate(800, DELTA, E0, RHO, seed=12, trend=2.)
    X = M.Exact(Z, off, 16); F = M.fit_exact(Z, off, 16, X=X); k, marg = M.counts_exact(F['theta'], X)
    pos = np.concatenate([np.linspace(0, 1, b - a) for a, b in zip(off[:-1], off[1:])])
    ok = np.isfinite(marg)
    print('\ntrend control: e %.3f rho %.3f share %.3f (true %.3f), MAE %.3f, corr(marginal, position) %.3f valid %s'
          % (F['e'], F['rho'], F['implied_share'], y.mean(), mae(k, y, off), np.corrcoef(marg[ok], pos[ok])[0, 1], F['valid']))
