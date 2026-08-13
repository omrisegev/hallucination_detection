#!/usr/bin/env python3
"""Development-seed tests for the Phase A5 numerical core."""

from __future__ import annotations

from collections import UserDict
import unittest

import numpy as np

from spectral_utils.anchored_sparse_latent_mixture import (
    anchored_direction,
    diagonal_support,
    fit_constrained_direction_mixture,
    fit_fixed_support_precision,
    fit_sparse_equal_covariance_mixture,
    fit_standardization,
    gaussian_log_density,
    held_mean_log_likelihood,
    posterior_log_odds,
)


def synthetic(seed: int = 510003, n: int = 1000, p: int = 6):
    rng = np.random.default_rng(seed)
    precision = np.eye(p)
    for index in range(p - 1):
        precision[index, index + 1] = precision[index + 1, index] = -0.16
        precision[index, index] += 0.16
        precision[index + 1, index + 1] += 0.16
    covariance = np.linalg.inv(precision)
    delta = np.zeros(p)
    delta[[0, 2, 5]] = [1.2, -0.8, 0.7]
    z = rng.choice([-1.0, 1.0], size=n)
    X = rng.multivariate_normal(np.zeros(p), covariance, size=n)
    X += z[:, None] * delta / 2.0
    return X, z, covariance, precision, delta


class TestPrecision(unittest.TestCase):
    def test_fixed_support_known_answer(self):
        X, _, _, precision, _ = synthetic()
        support = np.abs(precision) > 0
        empirical = np.cov(X, rowvar=False, bias=True)
        fit = fit_fixed_support_precision(empirical, support)
        self.assertTrue(fit.converged, fit)
        self.assertGreater(fit.minimum_eigenvalue, 1e-8)
        self.assertLess(np.max(np.abs(fit.precision[~support])), 1e-12)
        self.assertLess(np.linalg.norm(fit.covariance @ fit.precision - np.eye(6)), 1e-8)

    def test_diagonal_matches_inverse_variance(self):
        rng = np.random.default_rng(510004)
        variances = np.array([0.7, 1.1, 2.0, 3.5])
        X = rng.normal(size=(3000, 4)) * np.sqrt(variances)
        empirical = np.cov(X, rowvar=False, bias=True)
        fit = fit_fixed_support_precision(empirical, diagonal_support(4))
        expected = 1.0 / np.diag(empirical)
        self.assertTrue(fit.converged, fit)
        self.assertTrue(np.allclose(np.diag(fit.precision), expected, atol=1e-7))

    def test_exact_diagonal_optimum_converges_without_a_step(self):
        empirical = np.diag([0.5, 1.0, 2.0, 4.0])
        fit = fit_fixed_support_precision(empirical, diagonal_support(4))
        self.assertTrue(fit.converged, fit)
        self.assertEqual(fit.iterations, 1)

    def test_random_spd_free_entry_kkt_residuals(self):
        for seed in range(12):
            rng = np.random.default_rng(510100 + seed)
            p = 7
            factor = rng.normal(size=(p, p))
            empirical = factor @ factor.T / p + 0.2 * np.eye(p)
            support = np.eye(p, dtype=bool)
            for i in range(p):
                for j in range(i + 1, p):
                    if rng.random() < 0.35:
                        support[i, j] = support[j, i] = True
            fit = fit_fixed_support_precision(empirical, support)
            self.assertTrue(fit.converged, (seed, fit))
            gradient = empirical - fit.covariance
            free_residuals = list(np.diag(gradient))
            free_residuals.extend(
                2.0 * gradient[i, j]
                for i in range(p) for j in range(i + 1, p)
                if support[i, j]
            )
            self.assertLess(np.linalg.norm(free_residuals), 2e-6)
            self.assertGreater(fit.minimum_eigenvalue, 1e-8)

    def test_fixed_support_objective_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(510140)
        p = 5
        factor = rng.normal(size=(p, p))
        empirical = factor @ factor.T / p + np.eye(p)
        support = np.eye(p, dtype=bool)
        support[0, 2] = support[2, 0] = True
        support[1, 4] = support[4, 1] = True
        fit = fit_fixed_support_precision(empirical, support)
        self.assertTrue(fit.converged)
        # At the optimum, a symmetric free-edge perturbation has derivative
        # 2*(S-Sigma)_ij.  Verify the analytical KKT coordinate independently.
        edge = np.zeros((p, p))
        edge[0, 2] = edge[2, 0] = 1.0
        epsilon = 1e-6
        def objective(omega):
            return np.sum(empirical * omega) - np.linalg.slogdet(omega)[1]
        numeric = (
            objective(fit.precision + epsilon * edge)
            - objective(fit.precision - epsilon * edge)
        ) / (2 * epsilon)
        analytic = 2.0 * (empirical - fit.covariance)[0, 2]
        self.assertAlmostEqual(numeric, analytic, places=7)


class TestMixture(unittest.TestCase):
    def test_em_likelihood_monotone_and_affine(self):
        X, _, _, precision, delta = synthetic()
        support = np.abs(precision) > 0
        anchor = precision @ delta + np.array([0, 0.2, 0, 0, 0, 0])
        fit = fit_sparse_equal_covariance_mixture(X, support, anchor)
        self.assertTrue(fit.converged, fit.history[-10:])
        self.assertTrue(all(b + 1e-7 >= a for a, b in zip(fit.history, fit.history[1:])))
        direct = (
            gaussian_log_density(X, fit.centre + fit.delta / 2, fit.precision)
            - gaussian_log_density(X, fit.centre - fit.delta / 2, fit.precision)
            + np.log(fit.prior / (1-fit.prior))
        )
        affine = posterior_log_odds(
            X, fit.centre, fit.delta, fit.precision, fit.prior
        )
        self.assertLess(np.max(np.abs(direct - affine)), 1e-10)

    def test_anchor_direction_and_exact_zero_fallback(self):
        X, _, _, precision, delta = synthetic(n=1300)
        anchor = precision @ delta + np.array([0, 0.4, 0, 0, 0, 0])
        fit = fit_sparse_equal_covariance_mixture(X, np.abs(precision) > 0, anchor)
        weight0, correction, diagnostics = anchored_direction(fit, anchor, 0.0)
        self.assertTrue(np.array_equal(weight0, anchor))
        self.assertLess(abs(diagnostics["iu_correction_covariance"]), 1e-10)
        weight1, _, _ = anchored_direction(fit, anchor, 1.0)
        self.assertGreater(weight1 @ fit.covariance @ anchor, 0.0)
        self.assertGreater(np.linalg.norm(correction), 1e-5)

    def test_unorientable_and_near_zero_evidence_copy_iu_exactly(self):
        X = np.column_stack([
            np.tile([-1.0, 1.0], 100),
            np.repeat([-1.0, 1.0], 100),
        ])
        fit = fit_sparse_equal_covariance_mixture(
            X, diagonal_support(2), np.array([1.0, 0.0])
        )
        iu = np.array([0.0, 1.0])
        # Force an exact orthogonal mixture discriminant without relying on the
        # EM symmetry details.
        exact = type(fit)(**{
            **fit.__dict__,
            "delta": np.array([1.0, 0.0]),
            "covariance": np.eye(2),
            "precision": np.eye(2),
        })
        weight, correction, diagnostics = anchored_direction(exact, iu, 1.0)
        self.assertTrue(np.array_equal(weight, iu))
        self.assertTrue(np.array_equal(correction, np.zeros(2)))
        self.assertTrue(diagnostics["degenerate_mixture_direction"])
        near = type(exact)(**{**exact.__dict__, "delta": np.array([1.0, 1e-12])})
        weight, correction, diagnostics = anchored_direction(near, iu, 1.0)
        self.assertTrue(np.array_equal(weight, iu))
        self.assertTrue(np.array_equal(correction, np.zeros(2)))
        self.assertTrue(diagnostics["degenerate_mixture_direction"])

    def test_constrained_fit_and_held_likelihood(self):
        X, _, covariance, precision, delta = synthetic(n=1600)
        train, held = X[:800], X[800:]
        direction = precision @ delta
        fit = fit_constrained_direction_mixture(
            train, covariance, precision, direction
        )
        self.assertTrue(fit.converged, fit.history[-10:])
        score = held_mean_log_likelihood(held, fit, precision)
        self.assertTrue(np.isfinite(score))

    def test_constrained_likelihood_is_positive_scale_invariant(self):
        X, _, covariance, precision, delta = synthetic(seed=510006, n=900)
        direction = precision @ delta
        fits = [
            fit_constrained_direction_mixture(X, covariance, precision, scale * direction)
            for scale in (1e-4, 1.0, 1e4)
        ]
        for fit in fits:
            self.assertTrue(fit.converged, fit.history[-5:])
        for fit in fits[1:]:
            self.assertAlmostEqual(fit.mean_log_likelihood, fits[0].mean_log_likelihood, places=10)
            np.testing.assert_allclose(fit.delta, fits[0].delta, rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(
                fit.responsibilities, fits[0].responsibilities, rtol=1e-9, atol=1e-10
            )


class TestContracts(unittest.TestCase):
    def test_affine_raw_reconstruction(self):
        rng = np.random.default_rng(510005)
        raw = rng.normal(size=(120, 5)) * np.arange(1, 6) + np.arange(5)
        signs = np.array([1, -1, 1, -1, 1], dtype=float)
        standardization, transformed = fit_standardization(raw, signs)
        weight = rng.normal(size=5)
        intercept = 0.37
        raw_weight, raw_intercept = standardization.fold_affine(weight, intercept)
        self.assertLess(
            np.max(np.abs(transformed @ weight + intercept - (raw @ raw_weight + raw_intercept))),
            1e-10,
        )

    def test_feature_permutation_equivariance(self):
        X, _, _, precision, delta = synthetic(n=900)
        support = np.abs(precision) > 0
        anchor = precision @ delta
        fit = fit_sparse_equal_covariance_mixture(X, support, anchor)
        weight, _, _ = anchored_direction(fit, anchor, 0.5)
        permutation = np.array([3, 0, 5, 1, 4, 2])
        fit_p = fit_sparse_equal_covariance_mixture(
            X[:, permutation], support[np.ix_(permutation, permutation)], anchor[permutation]
        )
        weight_p, _, _ = anchored_direction(fit_p, anchor[permutation], 0.5)
        self.assertLess(np.max(np.abs(X @ weight - X[:, permutation] @ weight_p)), 1e-6)

    def test_target_like_mapping_rejected(self):
        with self.assertRaises(TypeError):
            fit_sparse_equal_covariance_mixture(
                {"X": np.ones((4, 3)), "label": object()},
                np.eye(3, dtype=bool),
                np.ones(3),
            )
        with self.assertRaises(TypeError):
            fit_standardization(UserDict({"X": np.ones((4, 3))}))


if __name__ == "__main__":
    unittest.main()
