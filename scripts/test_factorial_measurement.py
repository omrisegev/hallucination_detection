"""Unit tests for the label-free A1 factorial measurement model."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from spectral_utils import factorial_measurement
from spectral_utils.factorial_measurement import (
    FactorialConfiguration,
    augment_correlated_duplicate,
    covariance_from_residuals,
    fit_factorial_measurement,
    masked_feature_reconstruction_rows,
    mechanical_design,
    pooled_mean_reconstruction_rows,
    reconstruction_mse,
    soft_quotient_weights,
)


def _dag(names):
    return [
        {
            "feature_name": name,
            "source_stream": f"channel_{index % 2}",
            "operator": f"operator_{index // 2}",
        }
        for index, name in enumerate(names)
    ]


class FactorialMeasurementTest(unittest.TestCase):
    def test_covariance_embedding_preserves_missingness(self):
        rng = np.random.default_rng(3)
        residuals = rng.normal(size=(200, 2))
        embedded = covariance_from_residuals(
            residuals, ("a", "c"), ("a", "b", "c")
        )
        self.assertTrue(np.isnan(embedded[1]).all())
        self.assertAlmostEqual(embedded[0, 0], 1.0)
        self.assertAlmostEqual(embedded[2, 2], 1.0)

    def test_random_axes_preserve_incidence_dimensions(self):
        names = tuple(f"f{index}" for index in range(8))
        dag = _dag(names)
        observed = mechanical_design(names, dag, axes="factorial")
        shuffled = mechanical_design(
            names, dag, axes="factorial", random_seed=17
        )
        self.assertEqual(observed.shape, shuffled.shape)
        np.testing.assert_allclose(observed.sum(axis=0), shuffled.sum(axis=0))

    def test_masked_pca_adapts_environment_scale(self):
        loadings = np.asarray([0.25, 0.45, 0.65, 0.8, 0.55, 0.35])
        covariances = []
        for scale in (0.25, 0.45, 0.65, 0.85):
            covariance = scale * np.outer(loadings, loadings)
            np.fill_diagonal(covariance, 1.0)
            covariances.append(covariance)
        covariances = np.asarray(covariances)
        names = tuple(f"f{index}" for index in range(len(loadings)))
        config = FactorialConfiguration("pca", 1, False, 1e-5)
        fit = fit_factorial_measurement(covariances[:3], names, _dag(names), config)
        adaptive = masked_feature_reconstruction_rows(fit, covariances[3], "held")
        pooled = pooled_mean_reconstruction_rows(
            covariances[:3], covariances[3], names, "held"
        )
        self.assertLess(reconstruction_mse(adaptive), reconstruction_mse(pooled))

    def test_exact_duplicate_conserves_soft_quotient_mass(self):
        loadings = np.asarray([0.2, 0.4, 0.6, 0.8, 0.5, 0.3])
        covariances = []
        for scale in (0.3, 0.5, 0.7):
            covariance = scale * np.outer(loadings, loadings)
            np.fill_diagonal(covariance, 1.0)
            covariances.append(covariance)
        covariances = np.asarray(covariances)
        names = tuple(f"f{index}" for index in range(len(loadings)))
        dag = _dag(names)
        config = FactorialConfiguration("hybrid", 2, True, 0.1, alpha=0.5)
        original = fit_factorial_measurement(covariances, names, dag, config)
        original_weights, _ = soft_quotient_weights(original)

        augmented_covariances = augment_correlated_duplicate(
            covariances, 0, correlation=1.0
        )
        augmented_names = names + ("f0_duplicate",)
        augmented_dag = dag + [{
            **dag[0], "feature_name": "f0_duplicate"
        }]
        augmented = fit_factorial_measurement(
            augmented_covariances, augmented_names, augmented_dag, config
        )
        augmented_weights, diagnostics = soft_quotient_weights(augmented)
        self.assertEqual(len(diagnostics["duplicate_classes"]), len(names))
        self.assertAlmostEqual(
            original_weights[0],
            augmented_weights[0] + augmented_weights[-1],
            places=12,
        )
        np.testing.assert_allclose(
            original_weights[1:], augmented_weights[1:-1], atol=1e-12
        )

    def test_public_fit_has_no_correctness_label_argument(self):
        signature = inspect.signature(fit_factorial_measurement)
        self.assertNotIn("labels", signature.parameters)
        source = inspect.getsource(factorial_measurement)
        self.assertNotIn("FEATURE_TO_VIEW", source)
        self.assertNotIn("specrage_views", source)
        from scripts import automatic_group_free_phase_a1
        script_source = inspect.getsource(automatic_group_free_phase_a1)
        self.assertNotIn("__labels", script_source)
        self.assertNotIn("roc_auc", script_source)


if __name__ == "__main__":
    unittest.main()
