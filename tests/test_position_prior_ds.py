import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments'))
import position_prior_ds as PP  # noqa: E402


def _simulate(pi_b, psi, fpr, n_ans=6000, n=10, seed=0):
    rng = np.random.default_rng(seed)
    off = np.arange(0, (n_ans + 1) * n, n)
    bins = PP.position_bins(off, len(pi_b))
    y = rng.random(len(bins)) < np.asarray(pi_b)[bins]
    p = np.where(y[:, None], psi, fpr)
    votes = np.where(rng.random(p.shape) < p, 1, -1).astype(np.int8)
    return off, bins, y, votes


def _plain_ds(votes, psi, eta, prev, iters=20000):
    """Independent textbook DS EM (no compression) for the one-bin identity."""
    b = (votes > 0).astype(float); e1 = np.clip(psi, 1e-6, 1 - 1e-6); e0 = np.clip(1 - eta, 1e-6, 1 - 1e-6); pr = prev
    for _ in range(iters):
        l1 = b @ np.log(e1) + (1 - b) @ np.log1p(-e1) + np.log(pr); l0 = b @ np.log(e0) + (1 - b) @ np.log1p(-e0) + np.log1p(-pr)
        q = 1 / (1 + np.exp(l0 - l1))
        pr_new = q.mean(); e1n = np.clip(b.T @ q / q.sum(), 1e-6, 1 - 1e-6); e0n = np.clip(b.T @ (1 - q) / (1 - q).sum(), 1e-6, 1 - 1e-6)
        done = max(abs(pr_new - pr), np.abs(e1n - e1).max(), np.abs(e0n - e0).max()) < 1e-13
        pr, e1, e0 = pr_new, e1n, e0n
        if done:
            break
    return pr, e1, 1 - e0


def test_position_bins_exact():
    off = np.array([0, 11, 13, 14, 34])
    b = PP.position_bins(off, 10)
    assert b[:11].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
    assert b[11:13].tolist() == [0, 9]
    assert b[13] == 0
    assert b[14:34].tolist() == [min(t * 10 // 19, 9) for t in range(20)]
    assert PP.position_bins(off, 20)[:11].tolist() == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19]


def test_recovers_position_dependent_prevalence():
    pi_true = np.linspace(0.08, 0.40, 10); psi = np.linspace(0.60, 0.80, 8); fpr = np.linspace(0.10, 0.20, 8)
    off, bins, y, votes = _simulate(pi_true, psi, fpr)
    f = PP.fit_pds(votes, bins, 10, psi0=np.full(8, 0.6), eta0=np.full(8, 0.8), prev0=0.25)
    assert f['converged'] and f['oriented']
    assert np.max(np.abs(f['pi_bins'] - pi_true)) < 0.035
    assert np.max(np.abs(f['psi'] - psi)) < 0.035 and np.max(np.abs((1 - f['eta']) - fpr)) < 0.035


def test_constant_prevalence_gives_flat_prior():
    off, bins, y, votes = _simulate(np.full(10, 0.25), np.linspace(0.60, 0.80, 8), np.linspace(0.10, 0.20, 8), seed=1)
    f = PP.fit_pds(votes, bins, 10, psi0=np.full(8, 0.6), eta0=np.full(8, 0.8), prev0=0.25)
    assert np.ptp(f['pi_bins']) < 0.05


def test_one_bin_is_the_constant_model():
    off, bins, y, votes = _simulate(np.linspace(0.1, 0.3, 10), np.linspace(0.6, 0.8, 6), np.linspace(0.1, 0.2, 6), n_ans=1500, seed=2)
    f = PP.fit_pds(votes, np.zeros(len(votes), np.int64), 1, psi0=np.full(6, 0.6), eta0=np.full(6, 0.8), prev0=0.3, tol=1e-14, max_iter=20000)
    pr, psi, eta = _plain_ds(votes.astype(float), np.full(6, 0.6), np.full(6, 0.8), 0.3)
    assert abs(f['pi_bins'][0] - pr) < 1e-6 and np.max(np.abs(f['psi'] - psi)) < 1e-6 and np.max(np.abs(f['eta'] - eta)) < 1e-6


def test_nested_likelihood_rises_and_orientation_kept():
    off, bins, y, votes = _simulate(np.linspace(0.08, 0.40, 10), np.linspace(0.6, 0.8, 8), np.linspace(0.1, 0.2, 8), n_ans=2000, seed=3)
    one = PP.fit_pds(votes, np.zeros(len(votes), np.int64), 1, np.full(8, 0.6), np.full(8, 0.8), 0.25)
    ten = PP.fit_pds(votes, bins, 10, one['psi'], one['eta'], float(one['pi_bins'][0]))
    assert abs(ten['loglik_start'] - one['loglik']) < 1e-9 * abs(one['loglik'])   # the nested start is the one-bin optimum
    assert ten['loglik'] > one['loglik']
    assert ten['oriented'] and np.all(np.diff(ten['pi_bins'][[0, 4, 9]]) > 0)


def test_latent_slope_recovers_lda_slope():
    rng = np.random.default_rng(4); y = rng.random(200000) < 0.3
    S = np.where(y, 1.5, 0.0) + rng.normal(0, 0.5, len(y))
    a, d = PP.latent_slope(S, y.astype(float))
    assert abs(a - 1.5 / 0.25) < 0.1 and abs(d['class1_mass'] - 0.3) < 0.01


def test_content_log_odds_matches_posterior():
    off, bins, y, votes = _simulate(np.linspace(0.08, 0.40, 10), np.linspace(0.6, 0.8, 8), np.linspace(0.1, 0.2, 8), n_ans=500, seed=5)
    f = PP.fit_pds(votes, bins, 10, np.full(8, 0.6), np.full(8, 0.8), 0.25)
    L = PP.content_log_odds(votes, f['psi'], f['eta']) + np.log(f['pi_bins'][bins]) - np.log1p(-f['pi_bins'][bins])
    assert np.max(np.abs(1 / (1 + np.exp(-L)) - f['q'])) < 1e-8
