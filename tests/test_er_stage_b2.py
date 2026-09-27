"""Known-result tests for scripts/experiments/er_stage_b2.py (synthetic data only)."""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments'))
import er_stage_b2 as C  # noqa: E402


def _auc(y, s):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    return (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def test_merge_joins_only_groups_with_targets():
    g = C.merge_groups_containing([0, 0, 1, 2, 2, 3], ['a', 'b', 'c', 'd', 'e', 'f'], {'b', 'e'})
    assert g[0] == g[1] == g[3] == g[4] and len(set(g)) == 3 and set(g) == {0, 1, 2}
    assert np.array_equal(C.merge_groups_containing([0, 1, 2], ['a', 'b', 'c'], {'a'}), [0, 1, 2])


def test_vectorized_auc_equals_direct_auc_with_ties():
    rng = np.random.default_rng(0); off = np.array([0, 6, 15, 19, 30]); S = np.round(rng.normal(size=(30, 3)), 1)
    y = rng.random(30) < 0.3; y[[0, 7, 16, 20]] = True; y[[1, 8, 17, 21]] = False
    Rk, loc = C.within_ranks(S, off, np.arange(4)); A = C.auc_from_ranks(Rk, loc, y)
    for i in range(4):
        for j in range(3):
            assert abs(A[i, j] - _auc(y[off[i]:off[i + 1]], S[off[i]:off[i + 1], j])) < 1e-12


def test_class_conditional_corr_is_zero_for_conditionally_independent_columns():
    rng = np.random.default_rng(1); n = 200_000; y = rng.random(n) < 0.2
    X = rng.normal(size=(n, 3)) + 0.8 * y[:, None]                              # dependent marginally, independent given y
    r = C.class_conditional_corr(X, y)
    assert r['clean']['max_abs_offdiag'] < 0.02 and r['error']['max_abs_offdiag'] < 0.02
    assert abs(np.corrcoef(X, rowvar=False)[0, 1]) > 0.05


def test_permutations_preserve_counts_and_lengths():
    rng = np.random.default_rng(2); loc = np.array([0, 3, 6, 10, 13, 17]); y = rng.random(17) < 0.4
    ys = C.shuffle_within(y, loc, rng)
    assert all(ys[a:b].sum() == y[a:b].sum() for a, b in zip(loc[:-1], loc[1:]))
    yw = C.swap_same_length(y, loc, rng)                                         # lengths 3,3,4,3,4
    assert sorted(tuple(yw[a:b]) for a, b in zip(loc[:-1], loc[1:])) == sorted(tuple(y[a:b]) for a, b in zip(loc[:-1], loc[1:]))
    assert np.array_equal(yw[6:10], y[13:17]) and np.array_equal(yw[13:17], y[6:10])
