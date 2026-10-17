"""Known-result tests for scripts/experiments/er_stage_b.py (synthetic data only)."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments'))
import er_stage_b as B  # noqa: E402


def _std(x, off):
    out = np.zeros_like(x, float)
    for a, b in zip(off[:-1], off[1:]):
        blk = x[a:b]; sd = blk.std(0); out[a:b] = np.divide(blk - blk.mean(0), sd, out=np.zeros_like(blk), where=sd > 1e-12)
    return out


def test_marks_count_and_agree_with_value_order_without_ties():
    rng = np.random.default_rng(0); off = np.array([0, 7, 12, 30, 31]); V = rng.normal(size=(31, 3))
    m = B.random_tie_marks(V, off, 0.2, rng.random(V.shape))
    for a, b in zip(off[:-1], off[1:]):
        k = int(np.ceil(0.2 * (b - a))); assert np.all((m[a:b] > 0).sum(0) == k)
        for j in range(3):
            assert set(np.flatnonzero(m[a:b, j] > 0)) == set(np.argsort(-V[a:b, j])[:k])


def test_ties_are_broken_uniformly_not_by_position():
    n_ans, L = 4000, 10; off = np.arange(0, n_ans * L + 1, L); V = np.zeros((n_ans * L, 1))   # every step tied
    m = B.random_tie_marks(V, off, 0.2, np.random.default_rng(1).random(V.shape))
    rate = (m[:, 0] > 0).reshape(n_ans, L).mean(0)
    assert np.all(np.abs(rate - 0.2) < 0.03)                                   # a positional rule would give [1, 1, 0, ...]


def test_mle_weight_is_the_vote_log_likelihood_ratio():
    psi = np.array([0.7, 0.55, 0.4]); eta = np.array([0.8, 0.6, 0.9])
    w = np.log(psi * eta / ((1 - psi) * (1 - eta)))
    for f in itertools.product([-1, 1], repeat=3):
        f = np.array(f); llr = np.sum(np.where(f > 0, np.log(psi / (1 - eta)), np.log((1 - psi) / eta)))
        const = np.sum(np.log(psi / (1 - eta)) + np.log((1 - psi) / eta)) / 2
        assert abs(llr - (f @ w / 2 + const)) < 1e-12
    assert np.allclose(B.mle_weights(psi, eta), np.maximum(w, 0)) and B.mle_weights([0.4], [0.55])[0] == 0.0 and B.mle_weights([0.5], [0.5])[0] == 0.0


def test_group_scores_singleton_is_the_channel_and_standardized():
    rng = np.random.default_rng(2); off = np.array([0, 6, 15]); X = _std(rng.normal(size=(15, 4)), off)
    Z = B.group_scores(X, off, np.array([0, 1, 1, 2]), _std)
    assert np.allclose(Z[:, 0], X[:, 0]) and np.allclose(Z[:, 2], X[:, 3])
    for a, b in zip(off[:-1], off[1:]):
        assert np.allclose(Z[a:b].mean(0), 0) and np.allclose(Z[a:b].std(0), 1)


def test_position_profile_known_values_and_unseen_keys():
    off = np.array([0, 2, 4, 7]); V = np.array([[1.], [3.], [5.], [7.], [0.], [0.], [9.]])
    P = B.position_profile(V, off, np.array([0, 1, 2, 3]))                       # answers 0 and 1 (length 2) only
    assert np.allclose(P[:4, 0], [3, 5, 3, 5]) and np.allclose(P[4:, 0], 0)


def test_ari_and_canonical():
    assert B.ari([0, 0, 1, 1, 2], [5, 5, 3, 3, 9]) == 1.0 and B.canonical([2, 2, 0, 1]) == B.canonical([7, 7, 3, 4])
    assert B.ari([0, 0, 1, 1], [0, 1, 0, 1]) < 0.01


def test_group_bar_three_groups_needs_identical_order():
    tru = {'pi': np.array([0.62, 0.55, 0.67]), 'psi': np.array([.4, .3, .5]), 'eta': np.array([.8, .8, .8]), 'prevalence': 0.14}
    ok = B.group_bar({'pi': np.array([0.70, 0.60, 0.75]), 'psi': tru['psi'], 'eta': tru['eta'], 'prevalence': 0.15}, tru)
    bad = B.group_bar({'pi': np.array([0.80, 0.60, 0.75]), 'psi': tru['psi'], 'eta': tru['eta'], 'prevalence': 0.15}, tru)
    far = B.group_bar({'pi': np.array([0.70, 0.60, 0.75]), 'psi': tru['psi'], 'eta': tru['eta'], 'prevalence': 0.28}, tru)
    assert ok['passes'] and not bad['passes'] and not far['passes'] and far['rank_ok']


def test_ds_group_rule_beats_equal_on_unequal_independent_groups():
    """End to end on synthetic independent classifiers: DS estimates -> MLE weights recover the true weights'
    ordering and the weighted vote beats the equal vote."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments'))
    import er_stage_a as A
    rng = np.random.default_rng(3); n = 60_000; y = rng.random(n) < 0.15
    psi = np.array([0.85, 0.55, 0.52, 0.60]); eta = np.array([0.95, 0.62, 0.60, 0.70])
    votes = np.where(rng.random((n, 4)) < np.where(y[:, None], psi, 1 - eta), 1, -1).astype(np.int8)
    est = A.em_estimate(votes, 'ds'); w = B.mle_weights(est['psi'], est['eta']); wt = B.mle_weights(psi, eta)
    assert np.array_equal(np.argsort(w), np.argsort(wt))
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, votes @ w) > roc_auc_score(y, votes.sum(1)) + 0.01
