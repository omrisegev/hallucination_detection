"""Known-answer tests for A6's numeric-only factorial PTNI core."""

from __future__ import annotations

import unittest

import numpy as np

from spectral_utils.a6_features import A6_FEATURE_ROSTER, fit_natural_coordinate_system
from spectral_utils.a6_ptni import (
    QWEN_SOURCE_SCORERS,
    NamedCoordinateMatrix,
    NamedCoordinateVector,
    SourceQuartetBatch,
    SourceRiskDirection,
    anchor_source_direction,
    discover_exact_duplicate_quotient,
    factorial_effects,
    factorial_moments,
    fit_source_risk_direction,
    ordering_objective,
    quotient_batch,
    quartet_delta,
    structural_metrics,
)


def _metadata(n: int):
    domains = ("arithmetic", "relational", "finite_logic")
    mutations = ("value_leaf", "relation_operator", "constraint_condition")
    grammars = ("short", "certificate")
    cells = [
        (domain, mutation, grammar)
        for domain in domains for mutation in mutations for grammar in grammars
    ]
    if n % len(cells):
        raise ValueError("test batch size must be a multiple of 18")
    repeated = cells * (n // len(cells))
    return (
        tuple(f"group:{index}" for index in range(n)),
        tuple(cell[0] for cell in repeated),
        tuple(cell[1] for cell in repeated),
        tuple(cell[2] for cell in repeated),
    )


def _batch(values: np.ndarray) -> SourceQuartetBatch:
    groups = _metadata(values.shape[0])
    return SourceQuartetBatch(
        values, A6_FEATURE_ROSTER[:values.shape[-1]], QWEN_SOURCE_SCORERS,
        *groups,
    )


class A6PTNITests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(630_001)

    def test_factorial_target_effect_is_prompt_balanced_invalid_minus_valid(self) -> None:
        values = np.zeros((18, 2, 2, 2, 4, 17))
        # Off diagonal is invalid. Prompt-only offsets cancel exactly.
        values[:, :, 1, :, :, 1] += 9.0
        values[:, :, 1, 0, :, 0] += 2.0
        values[:, :, 0, 1, :, 0] += 2.0
        effects = factorial_effects(_batch(values))
        np.testing.assert_allclose(effects.tau[..., 0], 2.0)
        np.testing.assert_allclose(effects.tau[..., 1], 0.0)

    def test_nuisance_and_interaction_effects_are_exact(self) -> None:
        values = np.zeros((18, 2, 2, 2, 4, 17))
        values[..., 1, 2] = 3.0  # common paraphrase nuisance, no interaction
        values[:, :, 1, 0, 2, 0] = 2.0
        values[:, :, 0, 1, 2, 0] = 2.0
        effects = factorial_effects(_batch(values))
        np.testing.assert_allclose(effects.nuisance[..., 0, 2], 3.0)
        np.testing.assert_allclose(effects.interaction[:, :, 0, 0], 0.0)
        np.testing.assert_allclose(effects.interaction[:, :, 1, 0], 2.0)

    def test_population_moment_weights_sum_to_one(self) -> None:
        values = self.rng.normal(size=(18, 2, 2, 2, 4, 17))
        moments = factorial_moments(_batch(values))
        self.assertAlmostEqual(float(moments.target_weights.sum()), 1.0, places=14)
        self.assertAlmostEqual(float(moments.nuisance_weights.sum()), 1.0, places=14)
        self.assertAlmostEqual(float(moments.interaction_weights.sum()), 1.0, places=14)
        np.testing.assert_allclose(moments.total, moments.total.T, atol=1e-12)

    def test_leave_nuisance_render_is_absent_from_every_fitted_moment(self) -> None:
        values = self.rng.normal(size=(18, 2, 2, 2, 4, 17))
        batch = _batch(values)
        fitted = ("canonical", "layout", "notation")
        before = factorial_moments(batch, fitted_renderings=fitted)
        changed = values.copy()
        changed[..., 1, :] = self.rng.normal(
            loc=1_000.0, scale=100.0, size=changed[..., 1, :].shape
        )
        after = factorial_moments(_batch(changed), fitted_renderings=fitted)
        self.assertEqual(before.fitted_renderings, fitted)
        for field in (
            "mu_target", "target_covariance", "nuisance_second_moment",
            "interaction_second_moment", "total", "intervention_energy",
            "target_weights", "nuisance_weights", "interaction_weights",
        ):
            np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
        with self.assertRaisesRegex(ValueError, "fitted renderings"):
            factorial_moments(
                batch, fitted_renderings=("canonical", "paraphrase")
            )

    def test_source_direction_uses_trace_scaled_registered_ridge(self) -> None:
        values = self.rng.normal(scale=0.1, size=(36, 2, 2, 2, 4, 17))
        # Give all 17 coordinates nonzero mechanically oriented target energy.
        shift = np.linspace(0.2, 0.5, 17)
        values[:, :, 1, 0, :, :] += shift
        values[:, :, 0, 1, :, :] += shift
        moments = factorial_moments(_batch(values))
        direction = fit_source_risk_direction(
            moments, {
                scorer: NamedCoordinateVector(moments.feature_names, np.ones(17))
                for scorer in QWEN_SOURCE_SCORERS
            }, 0.1
        )
        self.assertIsNone(direction.zero_evidence_reason)
        self.assertEqual(int(direction.active.sum()), 17)
        self.assertGreater(direction.trace_scale, 0)
        self.assertTrue(np.isfinite(direction.weight).all())

    def test_fewer_than_seventeen_active_is_zero_evidence(self) -> None:
        values = self.rng.normal(size=(18, 2, 2, 2, 4, 17))
        moments = factorial_moments(_batch(values))
        variances = np.ones((2, 17))
        variances[:, 0] = 0.0
        direction = fit_source_risk_direction(
            moments, {
                scorer: NamedCoordinateVector(moments.feature_names, variances[index])
                for index, scorer in enumerate(QWEN_SOURCE_SCORERS)
            }, 0.1
        )
        self.assertEqual(direction.zero_evidence_reason, "fewer_than_17_active")
        np.testing.assert_array_equal(direction.weight, np.zeros(30))

    def test_alpha_zero_is_bit_exact_iu(self) -> None:
        raw = self.rng.normal(size=(240, 30))
        target = fit_natural_coordinate_system(raw)
        source = SourceRiskDirection(
            A6_FEATURE_ROSTER, 0.1, self.rng.normal(size=30),
            np.ones(30, dtype=bool), 1.0, None,
            tuple((name,) for name in A6_FEATURE_ROSTER),
        )
        score = anchor_source_direction(source, target, 0.0)
        self.assertTrue(np.array_equal(score.weight, target.iu.w))
        self.assertTrue(np.array_equal(score.unit_correction, np.zeros_like(target.iu.w)))
        z = target.transformer.training_output
        self.assertTrue(np.array_equal(score.score(z), z @ target.iu.w))

    def test_anchored_correction_is_covariance_orthogonal_and_affine(self) -> None:
        raw = self.rng.normal(size=(300, 30))
        target = fit_natural_coordinate_system(raw)
        source = SourceRiskDirection(
            A6_FEATURE_ROSTER, 0.1, self.rng.normal(size=30),
            np.ones(30, dtype=bool), 1.0, None,
            tuple((name,) for name in A6_FEATURE_ROSTER),
        )
        score = anchor_source_direction(source, target, 0.25)
        self.assertIsNone(score.zero_evidence_reason)
        z = target.transformer.training_output
        covariance = z.T @ z / len(z)
        self.assertLess(abs(float(score.iu_weight @ covariance @ score.unit_correction)), 1e-10)
        direct = z @ score.iu_weight - 0.25 * (z @ score.unit_correction)
        np.testing.assert_allclose(score.score(z), direct, atol=1e-12, rtol=0)

    def test_quartet_delta_ordering_and_tie_credit(self) -> None:
        values = np.zeros((18, 2, 2, 2, 4, 17))
        batch = _batch(values)
        scores = np.zeros(values.shape[:-1])
        scores[:, :, 0, 0, :] = 1.0
        scores[:, :, 1, 1, :] = 1.0
        self.assertTrue(np.all(quartet_delta(scores) == 1.0))
        self.assertAlmostEqual(ordering_objective(scores, batch), 1.0, places=14)
        self.assertAlmostEqual(
            ordering_objective(np.zeros_like(scores), batch), 0.5, places=14
        )

    def test_structural_metrics_reward_target_and_penalize_render_drift(self) -> None:
        values = np.zeros((18, 2, 2, 2, 4, 17))
        batch = _batch(values)
        unit_confidence = np.zeros(values.shape[:-1])
        unit_confidence[:, :, 0, 0, :] = 1.0
        unit_confidence[:, :, 1, 1, :] = 1.0
        metrics = structural_metrics(unit_confidence, batch)
        self.assertAlmostEqual(metrics.target_margin, 1.0, places=14)
        self.assertTrue(all(value == 0.0 for _, value in metrics.nuisance_ratios))
        self.assertTrue(all(value == 0.0 for _, value in metrics.interaction_ratios))
        drifted = unit_confidence.copy()
        drifted[..., 1] += np.array([[1.0, -1.0], [-1.0, 1.0]])
        drifted_metrics = structural_metrics(drifted, batch)
        self.assertGreater(dict(drifted_metrics.nuisance_ratios)["paraphrase"], 0.0)

    def test_public_batch_rejects_targets_and_partial_values_by_type_and_shape(self) -> None:
        with self.assertRaises(ValueError):
            _batch(np.zeros((2, 2, 2, 2, 4, 16)))
        values = np.zeros((18, 2, 2, 2, 4, 17))
        values[0, 0, 0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "complete and finite"):
            _batch(values)

    def test_incomplete_family_cartesian_and_wrong_scorers_are_rejected(self) -> None:
        values = np.zeros((18, 2, 2, 2, 4, 17))
        groups = _metadata(18)
        with self.assertRaisesRegex(ValueError, "cartesian"):
            SourceQuartetBatch(
                values, A6_FEATURE_ROSTER[:17], QWEN_SOURCE_SCORERS,
                groups[0], tuple("arithmetic" for _ in range(18)),
                groups[2], groups[3],
            )
        with self.assertRaisesRegex(ValueError, "exact two registered Qwen"):
            SourceQuartetBatch(
                values, A6_FEATURE_ROSTER[:17], ("wrong", "views"), *groups
            )

    def test_nonfinite_ordering_and_structural_scores_fail_closed(self) -> None:
        values = np.zeros((18, 2, 2, 2, 4, 17))
        batch = _batch(values)
        scores = np.zeros(values.shape[:-1])
        scores[0, 0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            ordering_objective(scores, batch)
        with self.assertRaisesRegex(ValueError, "finite"):
            structural_metrics(scores, batch)

    def test_source_subset_expands_to_nominal_target_roster(self) -> None:
        values = self.rng.normal(size=(18, 2, 2, 2, 4, 29))
        values[:, :, 1, 0, :, :] += 0.3
        values[:, :, 0, 1, :, :] += 0.3
        batch = _batch(values)
        moments = factorial_moments(batch)
        direction = fit_source_risk_direction(
            moments, {
                scorer: NamedCoordinateVector(moments.feature_names, np.ones(29))
                for scorer in QWEN_SOURCE_SCORERS
            }, 0.1,
        )
        self.assertEqual(direction.feature_names, A6_FEATURE_ROSTER)
        self.assertEqual(direction.weight.shape, (30,))
        raw = self.rng.normal(size=(240, 30))
        target = fit_natural_coordinate_system(raw)
        anchored = anchor_source_direction(direction, target, 0.25)
        self.assertNotEqual(anchored.zero_evidence_reason, "target_name_absent_from_source")

    def test_exact_duplicate_quotient_and_target_mismatch_fallback(self) -> None:
        values = self.rng.normal(size=(18, 2, 2, 2, 4, 18))
        values[..., 1] = values[..., 0]
        values[:, :, 1, 0, :, :] += 0.2
        values[:, :, 0, 1, :, :] += 0.2
        # Reassert the duplicate after the planted target shift.
        values[..., 1] = values[..., 0]
        batch = _batch(values)
        natural_values = self.rng.normal(size=(200, 18))
        natural_values[:, 1] = natural_values[:, 0]
        natural = {
            scorer: NamedCoordinateMatrix(batch.feature_names, natural_values.copy())
            for scorer in QWEN_SOURCE_SCORERS
        }
        quotient = discover_exact_duplicate_quotient(batch, natural)
        reduced = quotient_batch(batch, quotient)
        self.assertEqual(reduced.values.shape[-1], 17)
        moments = factorial_moments(reduced)
        variances = {
            scorer: NamedCoordinateVector(
                moments.feature_names, np.var(quotient.reduce(record.values), axis=0)
            )
            for scorer, record in natural.items()
        }
        direction = fit_source_risk_direction(
            moments, variances, 0.1, quotient=quotient
        )
        raw = self.rng.normal(size=(240, 30))
        raw[:, 1] = raw[:, 0]
        target = fit_natural_coordinate_system(raw)
        anchored = anchor_source_direction(direction, target, 0.0)
        z = target.transformer.training_output
        self.assertFalse(anchored.evaluation_bound)
        with self.assertRaisesRegex(ValueError, "bind the full frozen target"):
            anchored.score(z[:20])
        bound = anchored.bind_evaluation(z)
        self.assertTrue(bound.evaluation_bound)
        self.assertTrue(np.array_equal(bound.score(z), z @ bound.iu_weight))
        broken = raw.copy()
        broken[:, 1] += self.rng.normal(scale=0.1, size=len(broken))
        broken_target = fit_natural_coordinate_system(broken)
        fallback = anchor_source_direction(direction, broken_target, 0.25)
        self.assertEqual(fallback.zero_evidence_reason, "target_duplicate_equality_failed")
        self.assertTrue(np.array_equal(fallback.weight, broken_target.iu.w))

    def test_quotient_alpha_zero_matches_unaugmented_iu_score(self) -> None:
        base = self.rng.normal(size=(300, 30))
        unaugmented_raw = base.copy()
        unaugmented_raw[:, 1] = np.nan
        unaugmented = fit_natural_coordinate_system(unaugmented_raw)
        augmented_raw = base.copy()
        augmented_raw[:, 1] = augmented_raw[:, 0]
        augmented = fit_natural_coordinate_system(augmented_raw)
        components = ((A6_FEATURE_ROSTER[0], A6_FEATURE_ROSTER[1]),) + tuple(
            (name,) for name in A6_FEATURE_ROSTER[2:]
        )
        source = SourceRiskDirection(
            A6_FEATURE_ROSTER, 0.1, np.zeros(30), np.ones(30, dtype=bool),
            1.0, None, components,
        )
        quotient_score = anchor_source_direction(source, augmented, 0.0)
        self.assertFalse(quotient_score.evaluation_bound)
        quotient_score = quotient_score.bind_evaluation(
            augmented.transformer.training_output
        )
        score_augmented = quotient_score.score(augmented.transformer.training_output)
        score_unaugmented = unaugmented.iu_scores(unaugmented.transformer.training_output)
        np.testing.assert_allclose(score_augmented, score_unaugmented, atol=1e-10, rtol=0)

    def test_duplicate_evaluation_mismatch_binds_one_ordinary_iu_artifact(self) -> None:
        raw = self.rng.normal(size=(260, 30))
        raw[:, 1] = raw[:, 0]
        target = fit_natural_coordinate_system(raw)
        components = ((A6_FEATURE_ROSTER[0], A6_FEATURE_ROSTER[1]),) + tuple(
            (name,) for name in A6_FEATURE_ROSTER[2:]
        )
        source = SourceRiskDirection(
            A6_FEATURE_ROSTER, 0.1, self.rng.normal(size=30),
            np.ones(30, dtype=bool), 1.0, None, components,
        )
        unbound = anchor_source_direction(source, target, 0.25)
        self.assertFalse(unbound.evaluation_bound)
        held = target.transformer.training_output.copy()
        held[-1, 1] += 0.2
        bound = unbound.bind_evaluation(held)
        self.assertTrue(bound.evaluation_bound)
        self.assertEqual(
            bound.zero_evidence_reason, "target_duplicate_evaluation_failed"
        )
        self.assertEqual(bound.duplicate_components, ())
        np.testing.assert_array_equal(bound.weight, target.iu.w)
        np.testing.assert_array_equal(bound.unit_correction, np.zeros_like(target.iu.w))
        np.testing.assert_array_equal(bound.score(held[:13]), held[:13] @ target.iu.w)

    def test_duplicate_discovery_is_bit_exact_and_name_bound(self) -> None:
        values = self.rng.normal(size=(18, 2, 2, 2, 4, 18))
        values[..., 0] = 0.0
        values[..., 1] = -0.0
        batch = _batch(values)
        natural_values = self.rng.normal(size=(100, 18))
        natural_values[:, 0] = 0.0
        natural_values[:, 1] = -0.0
        natural = {
            scorer: NamedCoordinateMatrix(batch.feature_names, natural_values)
            for scorer in QWEN_SOURCE_SCORERS
        }
        quotient = discover_exact_duplicate_quotient(batch, natural)
        self.assertEqual(len(quotient.components), 18)
        wrong_names = tuple(reversed(batch.feature_names))
        bad = dict(natural)
        bad[QWEN_SOURCE_SCORERS[0]] = NamedCoordinateMatrix(
            wrong_names, natural_values
        )
        with self.assertRaisesRegex(ValueError, "names"):
            discover_exact_duplicate_quotient(batch, bad)


if __name__ == "__main__":
    unittest.main()
