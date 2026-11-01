"""Synthetic checks for spectral_utils.neighbor_readout (protocol docs/experiments/NEIGHBOR_READOUT_PREREG_2026-11-01.md,
section 5 step 1). No benchmark data, no labels."""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'results/prm_boost_collection_v1'))
from spectral_utils import neighbor_readout as NR  # noqa: E402
import evaluate_collection as E  # noqa: E402


def _offsets(rng, A, lo, hi):
    return np.r_[0, np.cumsum(rng.integers(lo, hi + 1, A))].astype(np.int64)


def _simulate_hmm(rng, off, mu, sd, rho, a, b):
    r = NR.rel_position(off); z = np.zeros(off[-1], int)
    for s0, s1 in zip(off[:-1], off[1:]):
        prev = 0
        for t in range(s0, s1):
            p1 = expit(a + b * r[t]) if prev == 0 else rho
            prev = z[t] = int(rng.random() < p1)
    s = np.where(z == 1, rng.normal(mu[1], sd[1], len(z)), rng.normal(mu[0], sd[0], len(z)))
    return s, z


def test_forward_backward_matches_enumeration():
    rng = np.random.default_rng(0); off = np.array([0, 4, 7, 8])
    s = rng.normal(size=8)
    fit = dict(position=True, mu=[-.2, 1.1], sd=[.9, 1.3], rho=.45, a=-1.2, b=.8, err_state=1)
    r = NR.rel_position(off); ll = 0.; post = np.zeros(8)
    for a0, b0 in zip(off[:-1], off[1:]):
        n = b0 - a0; tot = 0.; acc = np.zeros(n)
        for path in itertools.product((0, 1), repeat=n):
            pr = 1.; prev = 0
            for i, zt in enumerate(path):
                p1 = expit(fit['a'] + fit['b'] * r[a0 + i]) if (i == 0 or prev == 0) else fit['rho']
                pr *= (p1 if zt else 1 - p1) * norm.pdf(s[a0 + i], fit['mu'][zt], fit['sd'][zt]); prev = zt
            tot += pr; acc += pr * np.array(path)
        ll += np.log(tot); post[a0:b0] = acc / tot
    S, M = NR._pad(s, off); R, _ = NR._pad(r, off)
    p = dict(mu=np.array(fit['mu']), sd=np.array(fit['sd']), rho=fit['rho'], a=fit['a'], b=fit['b'])
    ll_fb, gamma, _, lo, _ = NR._forward_backward(S, R, M, p, True)
    assert abs(ll_fb - ll) < 1e-9
    assert np.allclose(gamma[M][:, 1], post, atol=1e-12)
    assert np.allclose(NR.hmm_logit(s, off, fit), np.log(post / (1 - post)), atol=1e-8)


def test_hmm_recovers_synthetic_markov_parameters():
    rng = np.random.default_rng(1); off = _offsets(rng, 3000, 6, 20)
    truth = dict(mu=(0., 2.), sd=(1., 1.), rho=.42, a=-2.6, b=1.6)
    s, z = _simulate_hmm(rng, off, **truth)
    fit = NR.hmm_fit(s, off, position=True)
    assert fit['valid'] and fit['err_state'] == 1 and fit['converged']
    assert abs(fit['rho'] - truth['rho']) < .05
    assert abs(fit['a'] - truth['a']) < .25 and abs(fit['b'] - truth['b']) < .4
    for r in (.05, .5, .95):                                   # entry probability across the answer
        assert abs(expit(fit['a'] + fit['b'] * r) - expit(truth['a'] + truth['b'] * r)) < .03
    assert np.allclose(fit['mu'], truth['mu'], atol=.12) and np.allclose(fit['sd'], truth['sd'], atol=.1)
    assert abs(fit['posterior_share'] - z.mean()) < .02
    nopos = NR.hmm_fit(s, off, position=False)
    assert nopos['b'] == 0 and nopos['loglik'] < fit['loglik']


def test_hmm_err_state_is_larger_mean_and_validity():
    rng = np.random.default_rng(2); off = _offsets(rng, 1500, 6, 16)
    s, _ = _simulate_hmm(rng, off, mu=(0., 2.), sd=(1., 1.), rho=.4, a=-2.5, b=1.)
    fit = NR.hmm_fit(s, off)
    lo = NR.hmm_logit(s, off, fit)
    assert np.corrcoef(lo, s)[0, 1] > .5                       # error posterior rises with the score
    flat = NR.hmm_fit(rng.normal(0, .01, off[-1]), off)        # no separable second state: mu gap < .1
    assert not flat['valid'] and 'mu gap' in flat['reason']
    # Declared risk (protocol section 6): on pure N(0,1) noise the declared validity rules still accept a spurious
    # second state (mu gap ~.4-.8, share ~.05-.12). This is recorded behavior, not a pass condition.
    noise = NR.hmm_fit(rng.normal(size=off[-1]), off)
    assert noise['mu_gap'] > 0 and noise['implied_share'] < .6


def test_identity_transforms_reproduce_gap_flags():
    rng = np.random.default_rng(3); off = _offsets(rng, 400, 1, 30)
    s = E.answer_z(rng.normal(size=off[-1]), off); k = E.gap_counts(s, off)
    ref = E.topk_flags(s, off, k)
    for f in (NR.sym(s, off, 0.), NR.lift(s, off, 0.), NR.lap(s, off, 0.)):
        assert np.array_equal(f, s) and np.array_equal(E.topk_flags(f, off, k), ref)


def test_sym_lift_definitions():
    off = np.array([0, 1, 4]); s = np.array([5., 1., 3., -1.])
    assert np.allclose(NR.neighbor_mean(s, off), [0., 3., 0., 3.])
    assert np.allclose(NR.sym(s, off, .5), [5., 2.5, 3., .5])
    assert np.allclose(NR.lift(s, off, .5), [5., 2., 3., 1.])   # step 3 is a peak: not lowered


def test_lap_matches_dense_solve_and_trace():
    rng = np.random.default_rng(4); off = _offsets(rng, 30, 1, 12); s = rng.normal(size=off[-1])
    for lam in (.1, .5, 2.):
        out = NR.lap(s, off, lam)
        for a, b in zip(off[:-1], off[1:]):
            n = b - a; Lp = np.diag(np.r_[1., 2 * np.ones(n - 2), 1.]) if n > 1 else np.zeros((1, 1))
            if n > 1: Lp -= np.eye(n, k=1) + np.eye(n, k=-1)
            Sm = np.linalg.inv(np.eye(n) + lam * Lp)
            assert np.allclose(out[a:b], Sm @ s[a:b])
            assert abs(NR.lap_trace(n, lam) - np.trace(Sm)) < 1e-10


def test_gcv_zero_is_small_lambda_limit():
    rng = np.random.default_rng(5); off = _offsets(rng, 200, 2, 15); s = rng.normal(size=off[-1])
    g0 = NR.gcv_scores(s, off, (0.,))[0.]; g = NR.gcv_scores(s, off, (1e-5,))[1e-5]
    assert abs(g - g0) / g0 < 1e-3
    out = NR.gcv_lambda(s, off)
    assert out['lam'] in NR.LAP_GRID


def _ar1_plus_noise(rng, off, lam, Vx, sig2):
    x = np.zeros(off[-1])
    for a, b in zip(off[:-1], off[1:]):
        x[a] = rng.normal(0, np.sqrt(Vx))
        for t in range(a + 1, b): x[t] = lam * x[t - 1] + rng.normal(0, np.sqrt(Vx * (1 - lam ** 2)))
    return x, x + rng.normal(0, np.sqrt(sig2), len(x))


def test_variogram_recovers_ar1_plus_noise():
    rng = np.random.default_rng(6); off = _offsets(rng, 6000, 10, 24)
    _, s = _ar1_plus_noise(rng, off, lam=.5, Vx=1., sig2=1.)
    v = NR.variogram_fit(s, off)
    assert v['valid'] and abs(v['lam'] - .5) < .06 and abs(v['Vx'] - 1.) < .2 and abs(v['sig2'] - 1.) < .2
    assert v['w'] > 0
    white = NR.variogram_fit(rng.normal(size=off[-1]), off)     # no persistence: should not produce a weight
    assert (not white['valid']) or white['w'] < .1


def test_wiener_weight_matches_regression():
    """Amendment A1: with the SYM neighbor MEAN, the Wiener weight is 2 beta_side / beta_center."""
    rng = np.random.default_rng(7); off = _offsets(rng, 8000, 12, 12)
    lam, Vx, sig2 = .6, 1., .8
    x, s = _ar1_plus_noise(rng, off, lam, Vx, sig2)
    has_p, has_n = NR._neighbors(off); i = np.flatnonzero(has_p & has_n)
    X = np.column_stack([s[i], NR.neighbor_mean(s, off)[i]])
    coef = np.linalg.lstsq(X, x[i], rcond=None)[0]
    assert abs(coef[1] / coef[0] - NR.wiener_weight(lam, Vx, sig2)['w']) < .05
