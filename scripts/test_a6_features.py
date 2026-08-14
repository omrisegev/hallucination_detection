"""Tests for A6's strict 30-atom admission and target-local coordinates."""

from __future__ import annotations

import unittest

import numpy as np

from spectral_utils.a6_features import (
    A6_FEATURE_ROSTER,
    fit_natural_coordinate_system,
    validate_complete_quartet_tensor,
)
from spectral_utils.group_free_research import canonical_feature_names
from spectral_utils.paired_repeatability import FEATURE_ROSTER as A4_FEATURE_ROSTER


class A6FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(620_001)
        self.raw = self.rng.normal(size=(200, 30))

    def test_roster_is_exact_30_and_not_a4_29(self) -> None:
        self.assertEqual(A6_FEATURE_ROSTER, canonical_feature_names())
        self.assertEqual(len(A6_FEATURE_ROSTER), 30)
        self.assertIn("min_spilled", A6_FEATURE_ROSTER)
        self.assertEqual(len(A4_FEATURE_ROSTER), 29)
        self.assertNotEqual(A6_FEATURE_ROSTER, A4_FEATURE_ROSTER)

    def test_natural_fit_is_finite_and_scores_in_transformed_coordinates(self) -> None:
        coordinates = fit_natural_coordinate_system(self.raw)
        self.assertEqual(coordinates.names, A6_FEATURE_ROSTER)
        self.assertTrue(coordinates.candidate_eligible)
        transformed = coordinates.transform_natural_or_evaluation(self.raw)
        self.assertEqual(transformed.shape, self.raw.shape)
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_allclose(
            transformed, coordinates.transformer.training_output, atol=1e-12, rtol=0
        )
        scores = coordinates.iu_scores(transformed)
        np.testing.assert_allclose(scores, transformed @ coordinates.iu.w, atol=0, rtol=0)

    def test_presence_rule_imputes_at_most_one_percent_and_drops_below(self) -> None:
        raw = self.raw.copy()
        raw[:2, 0] = np.nan  # exactly 99%; retained and target-locally imputed
        raw[:3, 1] = np.nan  # 98.5%; absent for the entire target
        coordinates = fit_natural_coordinate_system(raw)
        self.assertIn(A6_FEATURE_ROSTER[0], coordinates.names)
        self.assertNotIn(A6_FEATURE_ROSTER[1], coordinates.names)
        self.assertIn((A6_FEATURE_ROSTER[1], "presence_below_0.99"), coordinates.excluded)
        transformed = coordinates.transform_natural_or_evaluation(raw)
        self.assertTrue(np.isfinite(transformed).all())

    def test_all_missing_fails_before_generic_transformer(self) -> None:
        with self.assertRaisesRegex(ValueError, "no feature meets"):
            fit_natural_coordinate_system(np.full((200, 30), np.nan))

    def test_candidate_eligibility_tracks_seventeen_present_coordinates(self) -> None:
        seventeen = self.raw.copy()
        seventeen[:, :13] = np.nan
        self.assertTrue(fit_natural_coordinate_system(seventeen).candidate_eligible)
        sixteen = self.raw.copy()
        sixteen[:, :14] = np.nan
        self.assertFalse(fit_natural_coordinate_system(sixteen).candidate_eligible)

    def test_complete_source_tensor_has_no_imputation_path(self) -> None:
        source = self.rng.normal(size=(7, 2, 2, 4, 30))
        validated = validate_complete_quartet_tensor(source)
        self.assertIs(validated, source)
        source[2, 1, 0, 3, 4] = np.nan
        with self.assertRaisesRegex(ValueError, "complete and finite"):
            validate_complete_quartet_tensor(source)

    def test_target_local_transform_is_not_refit_by_interventions(self) -> None:
        coordinates = fit_natural_coordinate_system(self.raw)
        source = self.rng.normal(size=(5, 2, 2, 4, 30))
        before_mean = coordinates.transformer.oriented_mean.copy()
        first = coordinates.transform_complete_source(source)
        changed = source.copy()
        changed[..., 0] += 1000.0
        second = coordinates.transform_complete_source(changed)
        np.testing.assert_array_equal(coordinates.transformer.oriented_mean, before_mean)
        self.assertFalse(np.array_equal(first, second))

    def test_wrong_roster_order_and_partial_source_shape_fail(self) -> None:
        wrong = tuple(reversed(A6_FEATURE_ROSTER))
        with self.assertRaisesRegex(ValueError, "canonical 30-feature roster"):
            fit_natural_coordinate_system(self.raw, wrong)
        with self.assertRaisesRegex(ValueError, "groups,2,2,4,30"):
            validate_complete_quartet_tensor(np.zeros((2, 2, 4, 30)))


if __name__ == "__main__":
    unittest.main()
