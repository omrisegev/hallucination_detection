"""Module-B invariants: B0 identity, replay rule, label-free fit, LR competitor."""

from __future__ import annotations

import unittest

import numpy as np

from spectral_utils.trajectory_reducer import (
    ORDERSTAT_K,
    blend_step_scores,
    equal_topk_weights,
    fit_lr_orderstats,
    fit_orderstat_weights,
    fit_position_bin_weights,
    max_weights,
    reduce_with_weights,
    score_lr_orderstats,
    step_order_statistics,
    step_position_bins,
)


def _steps(seed: int = 0, n_steps: int = 300):
    """Steps with variable lengths (3..40 tokens) and a planted tail signal."""
    rng = np.random.default_rng(seed)
    risk = []
    starts, ends = [], []
    labels = []
    cursor = 0
    for _ in range(n_steps):
        length = int(rng.integers(3, 41))
        base = rng.normal(size=length)
        label = int(rng.random() < 0.3)
        if label:
            spike = rng.choice(length, size=max(1, length // 10), replace=False)
            base[spike] += 2.5  # tail-concentrated signal
        risk.append(base)
        starts.append(cursor)
        ends.append(cursor + length)
        labels.append(label)
        cursor += length
    return np.concatenate(risk), np.asarray(starts), np.asarray(ends), np.asarray(labels)


class IncumbentIdentityTests(unittest.TestCase):
    def test_equal_weights_reproduce_top10_mean_including_short_steps(self) -> None:
        risk, starts, ends, _ = _steps()
        matrix, lengths = step_order_statistics(risk, starts, ends)
        learned = reduce_with_weights(matrix, lengths, equal_topk_weights())
        for row, (lo, hi) in enumerate(zip(starts, ends)):
            values = risk[lo:hi]
            k = min(ORDERSTAT_K, len(values))
            incumbent = float(np.sort(values)[::-1][:k].mean())
            self.assertAlmostEqual(learned[row], incumbent, places=12)

    def test_max_weights_reproduce_span_max(self) -> None:
        risk, starts, ends, _ = _steps(seed=1)
        matrix, lengths = step_order_statistics(risk, starts, ends)
        top1 = reduce_with_weights(matrix, lengths, max_weights())
        for row, (lo, hi) in enumerate(zip(starts, ends)):
            self.assertAlmostEqual(top1[row], float(risk[lo:hi].max()), places=12)

    def test_blend_alpha_zero_is_exactly_b0(self) -> None:
        risk, starts, ends, _ = _steps(seed=2)
        matrix, lengths = step_order_statistics(risk, starts, ends)
        blended = blend_step_scores(matrix, lengths, 0.0)
        incumbent = reduce_with_weights(matrix, lengths, equal_topk_weights())
        np.testing.assert_array_equal(blended, incumbent)


class LabelFreeFitTests(unittest.TestCase):
    def test_b1_fits_oriented_weights_on_full_steps(self) -> None:
        risk, starts, ends, _ = _steps(seed=3, n_steps=600)
        matrix, lengths = step_order_statistics(risk, starts, ends)
        weights, meta = fit_orderstat_weights(matrix, lengths)
        self.assertEqual(len(weights), ORDERSTAT_K)
        self.assertFalse(meta["labels_accessed"])
        self.assertGreaterEqual(meta["anchor_correlation"], 0.0)
        scores = reduce_with_weights(matrix, lengths, weights)
        self.assertTrue(np.isfinite(scores).all())

    def test_positional_bins_shape_and_replay(self) -> None:
        risk, starts, ends, _ = _steps(seed=4)
        bins = step_position_bins(risk, starts, ends)
        self.assertEqual(bins.shape, (len(starts), 5))
        weights, meta = fit_position_bin_weights(bins)
        self.assertAlmostEqual(float(np.sum(np.abs(weights))) > 0, True)
        self.assertFalse(meta["labels_accessed"])


class SupervisedCompetitorTests(unittest.TestCase):
    def test_lr_trains_balanced_and_scores_every_step(self) -> None:
        risk, starts, ends, labels = _steps(seed=5, n_steps=800)
        matrix, lengths = step_order_statistics(risk, starts, ends)
        model, meta = fit_lr_orderstats(matrix, lengths, labels, seed=0)
        self.assertTrue(meta["supervised"])
        scores = score_lr_orderstats(model, matrix)
        self.assertEqual(len(scores), len(starts))
        from sklearn.metrics import roc_auc_score

        self.assertGreater(roc_auc_score(labels, scores), 0.75)


if __name__ == "__main__":
    unittest.main()
