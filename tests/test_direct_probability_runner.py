import numpy as np

from scripts.run_direct_probability_fusion_v1 import (
    _paired_group_auc_bootstrap,
    _weighted_auc_batch,
    auc,
)


def test_weighted_auc_matches_point_auc_and_handles_ties():
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=bool)
    scores = np.array([0.1, 0.4, 0.4, 0.8, 0.2, 0.9])
    groups = np.arange(len(labels))
    counts = np.ones((1, len(labels)), dtype=int)
    observed = _weighted_auc_batch(labels, scores, groups, counts)[0]
    assert np.isclose(observed, auc(labels, scores), atol=1e-12, rtol=0.0)


def test_group_bootstrap_is_paired_and_reproducible():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
    left = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    right = left[::-1]
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    first = _paired_group_auc_bootstrap(
        labels, left, right, groups, draws=200, seed=23, batch_size=31
    )
    second = _paired_group_auc_bootstrap(
        labels, left, right, groups, draws=200, seed=23, batch_size=31
    )
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert first.mean() > 0


def test_group_bootstrap_replaces_single_class_draws_to_keep_frozen_count():
    labels = np.zeros(256, dtype=bool)
    labels[:6] = True
    left = np.linspace(0.0, 1.0, len(labels))
    right = left[::-1]
    groups = np.arange(len(labels)).astype(str)
    values = _paired_group_auc_bootstrap(
        labels,
        left,
        right,
        groups,
        draws=1_000,
        seed=2026091123,
        batch_size=73,
    )
    assert values.shape == (1_000,)
    assert np.isfinite(values).all()
