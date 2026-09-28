import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments'))
import ds_group_weights as GW  # noqa: E402
import er_stage_b as SB  # noqa: E402


def simulate(n, sizes, t_err, t_clean, e1, e0, prev, seed):
    """Hierarchical votes: group state a_g ~ Bernoulli(t_err[g] if error else t_clean[g]); member j of the group votes +1
    w.p. e1[j] if a_g else e0[j] (scalars broadcast)."""
    rng = np.random.default_rng(seed); y = rng.random(n) < prev; cols = []
    for g, m in enumerate(sizes):
        a = rng.random(n) < np.where(y, t_err[g], t_clean[g])
        for j in range(m):
            p1 = e1[j] if np.ndim(e1) else e1; p0 = e0[j] if np.ndim(e0) else e0
            cols.append(np.where(rng.random(n) < np.where(a, p1, p0), 1, -1))
    return np.column_stack(cols), y, np.repeat(np.arange(len(sizes)), sizes)


def test_latent_group_accuracy_is_recovered():
    t_err = [0.8, 0.7, 0.65]; t_clean = [0.15, 0.3, 0.3]
    x, y, g = simulate(60000, [4, 4, 3], t_err, t_clean, 0.9, 0.1, 0.25, 1)
    r = GW.hem_fit(x, g)
    assert r['source'] == ['latent'] * 3 and r['converged']
    assert np.allclose(r['psi'], t_err, atol=0.03) and np.allclose(r['eta'], 1 - np.array(t_clean), atol=0.03)
    assert abs(r['prevalence'] - 0.25) < 0.03


def test_within_weights_rank_members_by_how_well_they_track_the_state():
    e1 = [0.95, 0.85, 0.7, 0.6]; e0 = [0.05, 0.15, 0.3, 0.4]
    x, y, g = simulate(80000, [4, 4, 4], [0.8, 0.75, 0.7], [0.2, 0.25, 0.3], e1, e0, 0.3, 5)
    r = GW.hem_fit(x, g)
    for h in range(3):
        w = r['within'][h]; assert w is not None and np.all(np.diff(w) < 0)
    true_w = np.log(np.array(e1) * (1 - np.array(e0)) / (np.array(e0) * (1 - np.array(e1))))
    assert np.allclose(r['within'][0], true_w, rtol=0.2)


def test_small_groups_use_member_marginals_and_equal_within():
    x, y, g = simulate(40000, [4, 2, 3], [0.8, 0.7, 0.7], [0.2, 0.3, 0.3], 0.85, 0.15, 0.3, 2)
    r = GW.hem_fit(x, g)
    assert r['source'] == ['latent', 'member_marginal', 'latent'] and r['within'][1] is None
    idx = g == 1
    assert r['psi'][1] == pytest.approx(r['channel_psi'][idx].mean()) and r['eta'][1] == pytest.approx(r['channel_eta'][idx].mean())


def test_reversed_group_gets_zero_between_weight():
    x, y, g = simulate(50000, [4, 4, 4], [0.8, 0.75, 0.7], [0.2, 0.25, 0.3], 0.9, 0.1, 0.3, 3)
    x[:, g == 1] *= -1                                  # group 1 votes the other way
    r = GW.hem_fit(x, g)
    assert r['pi'][0] > 0.7 and r['pi'][2] > 0.65 and r['pi'][1] < 0.3
    assert GW.mle_group_weights(r['psi'], r['eta'])[1] == 0.0


def test_flipped_latent_state_gives_identical_estimates(monkeypatch):
    """The same model with the latent state relabelled (e -> e[:, ::-1], t -> 1 - t) must give identical group and within estimates."""
    x, y, g = simulate(30000, [4, 3, 3], [0.8, 0.7, 0.7], [0.2, 0.3, 0.3], 0.9, 0.1, 0.3, 7)
    ref = GW.hem_fit(x, g)
    core, em = GW.SA._cvf()
    class EmFlip:
        @staticmethod
        def fit_em(*a, **kw):
            m = em.fit_em(*a, **kw); m.emissions = np.asarray(m.emissions)[:, ::-1].copy(); m.transition = 1 - np.asarray(m.transition); return m
    monkeypatch.setattr(GW.SA, '_cvf', lambda: (core, EmFlip))
    alt = GW.hem_fit(x, g)
    assert all(alt['latent_flipped']) and not any(ref['latent_flipped'])
    assert np.allclose(alt['psi'], ref['psi']) and np.allclose(alt['eta'], ref['eta']) and alt['prevalence'] == pytest.approx(ref['prevalence'])
    assert all(np.allclose(a, b) for a, b in zip(alt['within'], ref['within']))


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        GW.hem_fit(np.zeros((10, 3)), [0, 1, 2])
    with pytest.raises(ValueError):
        GW.hem_fit(np.ones((10, 3)), [0, 1])


def test_group_matrix_equal_within_matches_stage_b_group_scores():
    rng = np.random.default_rng(0); off = np.array([0, 5, 12, 20]); X = rng.normal(size=(20, 5)); g = np.array([0, 0, 1, 1, 1])
    std = lambda v, o: np.vstack([(v[a:b] - v[a:b].mean(0)) / v[a:b].std(0) for a, b in zip(o[:-1], o[1:])])
    assert np.allclose(GW.group_matrix(X, off, g, None, std), SB.group_scores(X, off, g, std))
    assert np.allclose(GW.group_matrix(X, off, g, [None, None], std), SB.group_scores(X, off, g, std))
    Zw = GW.group_matrix(X, off, g, [np.array([1., 0.]), None], std)
    assert np.allclose(Zw[:, 0], std(X[:, [0]], off)[:, 0])
    with pytest.raises(ValueError):
        GW.group_matrix(X, off, g, [np.array([0., 0.]), None], std)


def test_weighted_group_score():
    Z = np.array([[1., 2., 3.], [0., 0., 6.]])
    assert np.allclose(GW.weighted_group_score(Z, np.array([1., 1., 2.])), [2.25, 3.0])
    assert np.allclose(GW.weighted_group_score(Z, np.array([1., -1., 2.]), signed=True), [1.25, 3.0])
    with pytest.raises(ValueError):
        GW.weighted_group_score(Z, np.zeros(3))
    with pytest.raises(ValueError):
        GW.weighted_group_score(Z, np.array([1., -1., 1.]))
