"""Known-result tests for scripts/experiments/er_stage_a.py (synthetic data only)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments'))
import er_stage_a as A  # noqa: E402


def _independent(seed=0, n=80_000, m=9, prev=0.15):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < prev
    psi = rng.uniform(0.35, 0.85, m); eta = rng.uniform(0.60, 0.95, m)
    p_pos = np.where(y[:, None], psi[None, :], 1 - eta[None, :])
    votes = np.where(rng.random((n, m)) < p_pos, 1, -1).astype(np.int8)
    return votes, y, psi, eta


def test_truth_on_a_tiny_table():
    votes = np.array([[1, -1], [1, 1], [-1, -1], [-1, 1]], np.int8); y = np.array([True, True, False, False])
    t = A.truth(votes, y)
    assert np.allclose(t['psi'], [1.0, 0.5]) and np.allclose(t['eta'], [1.0, 0.5]) and t['prevalence'] == 0.5


def test_binary_votes_mark_exactly_ceil_fraction():
    rng = np.random.default_rng(1); off = np.array([0, 7, 12, 30]); V = rng.normal(size=(30, 4))
    votes, info = A.binary_votes(V, off, 0.2)
    for a, b in zip(off[:-1], off[1:]):
        assert np.all((votes[a:b] > 0).sum(0) == int(np.ceil(0.2 * (b - a))))
    assert info['boundary_tie_rate'] == 0.0


def test_ds_recovers_independent_classifiers():
    votes, y, psi, eta = _independent()
    est = A.em_estimate(votes, 'ds'); tru = A.truth(votes, y)
    assert np.max(np.abs(est['psi'] - tru['psi'])) < 0.02 and np.max(np.abs(est['eta'] - tru['eta'])) < 0.02
    assert abs(est['prevalence'] - tru['prevalence']) < 0.01
    assert A.bar(est, tru)['passes']


def test_sml_ranks_and_scales_with_the_ds_imbalance():
    votes, y, psi, eta = _independent(seed=2)
    ds = A.em_estimate(votes, 'ds'); b_hat = 2 * ds['prevalence'] - 1
    sml = A.sml_estimate(votes, anchor=int(np.argmax((psi + eta) / 2)), b_hat=b_hat); tru = A.truth(votes, y)
    assert np.max(np.abs(sml['pi'] - tru['pi'])) < 0.03
    assert A.bar(sml, tru, prev_tol=None)['passes']


def test_hem_recovers_marginal_rates_under_group_dependence():
    rng = np.random.default_rng(3); n = 120_000; prev = 0.15; K, per = 3, 3
    y = rng.random(n) < prev
    trans = np.array([[0.15, 0.80], [0.20, 0.70], [0.10, 0.65]])         # P(a_g = 1 | Y = 0/1)
    a = rng.random((n, K)) < np.where(y[:, None], trans[:, 1][None, :], trans[:, 0][None, :])
    emis = rng.uniform(0.05, 0.25, (K * per, 2)); emis[:, 1] = rng.uniform(0.70, 0.95, K * per)   # P(vote +1 | a = 0/1)
    g = np.repeat(np.arange(K), per)
    p_pos = np.where(a[:, g], emis[:, 1][None, :], emis[:, 0][None, :])
    votes = np.where(rng.random(p_pos.shape) < p_pos, 1, -1).astype(np.int8)
    est = A.em_estimate(votes, 'hem', groups=g); tru = A.truth(votes, y)
    assert np.max(np.abs(est['psi'] - tru['psi'])) < 0.03 and np.max(np.abs(est['eta'] - tru['eta'])) < 0.03
    assert abs(est['prevalence'] - tru['prevalence']) < 0.02


def test_bar_flags_a_wrong_side():
    tru = {'pi': np.array([0.60, 0.55, 0.45, 0.70]), 'prevalence': 0.2}
    est = {'pi': np.array([0.62, 0.53, 0.52, 0.69]), 'prevalence': 0.21}
    r = A.bar(est, tru)
    assert r['side_wrong'] == 1 and not r['passes']


def test_tensor_mom_recovers_independent_classifiers():
    votes, y, psi, eta = _independent(seed=4, n=200_000)
    est = A.tensor_mom_estimate(votes); tru = A.truth(votes, y)
    assert abs(est['prevalence'] - tru['prevalence']) < 0.01
    assert np.max(np.abs(est['psi'] - tru['psi'])) < 0.03 and np.max(np.abs(est['eta'] - tru['eta'])) < 0.03
    assert est['out_of_range'] == 0 and A.bar(est, tru)['passes']


def test_tensor_mom_power_sums_equal_the_explicit_tensor():
    rng = np.random.default_rng(5); votes = np.where(rng.random((400, 6)) < 0.3, 1, -1).astype(np.int8)
    est = A.tensor_mom_estimate(votes); t = est['t']; z = votes - votes.mean(0)
    T = np.einsum('ni,nj,nk->ijk', z, z, z) / len(z); tt = np.einsum('i,j,k->ijk', t, t, t)
    i, j, k = np.indices(T.shape); distinct = (i != j) & (j != k) & (i != k)
    assert np.isclose(est['alpha'], np.sum(T[distinct] * tt[distinct]) / np.sum(tt[distinct] ** 2), rtol=1e-10)


def test_tensor_mom_side_is_the_sign_of_t_whatever_the_imbalance():
    for prev in (0.1, 0.5, 0.8):
        votes, y, psi, eta = _independent(seed=6, n=60_000, prev=prev)
        est = A.tensor_mom_estimate(votes)
        assert np.array_equal(est['pi'] > 0.5, est['t'] > 0)


def test_tensor_mom_anchor_sets_the_orientation():
    votes, y, psi, eta = _independent(seed=7)
    votes[:, 0] = -votes[:, 0]                     # a reversed channel: anchoring on it flips the classes
    a, b = A.tensor_mom_estimate(votes), A.tensor_mom_estimate(votes, anchor=0)
    assert a['t'][0] < 0 < b['t'][0]
    assert np.allclose(a['t'], -b['t']) and np.isclose(a['prevalence'], 1 - b['prevalence'])
