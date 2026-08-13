"""Tests for Phase A2 multi-environment joint block diagonalization."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils import multi_environment_jbd
from spectral_utils.multi_environment_jbd import (
    JBDConfiguration,
    align_mixing,
    balanced_environment_row_null,
    blocks_from_coupling,
    complete_missing_covariances,
    fit_jbd_model,
    fit_pca_block_model,
    jacobi_refine,
    masked_reconstruction_rows,
    mechanism_subspace_overlap,
    missingness_preserving_stationary_null,
    off_diagonal_energy,
    randomized_joint_basis,
    residual_coupling,
    shuffled_environment_row_null,
    whiten_covariances,
)


def _joint_world(seed=7, environments=12, samples=10000):
    rng = np.random.default_rng(seed)
    p = 6
    mixing, _ = np.linalg.qr(rng.normal(size=(p, p)))
    covariances = []
    for _ in range(environments):
        variances = rng.uniform(0.3, 2.0, p)
        covariance = mixing @ np.diag(variances) @ mixing.T + 0.05 * np.eye(p)
        covariances.append(covariance)
    return np.asarray(covariances), mixing


class MultiEnvironmentJBDTest(unittest.TestCase):
    def test_missing_completion_is_psd_and_preserves_observed_blocks(self):
        covariances, _ = _joint_world(seed=19, environments=5)
        incomplete = covariances.copy()
        incomplete[0, 4:, :] = np.nan
        incomplete[0, :, 4:] = np.nan
        incomplete[1, 0, :] = np.nan
        incomplete[1, :, 0] = np.nan
        completed, diagnostics = complete_missing_covariances(incomplete)
        self.assertTrue(np.isfinite(completed).all())
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(completed))), -2e-9)
        for original, repaired, row in zip(incomplete, completed, diagnostics):
            observed = np.flatnonzero(np.isfinite(np.diag(original)))
            np.testing.assert_allclose(
                repaired[np.ix_(observed, observed)],
                original[np.ix_(observed, observed)],
                atol=1e-12,
            )
            self.assertLess(row["maximum_observed_entry_error"], 1e-12)

    def test_masked_reconstruction_scores_only_observed_features(self):
        covariances, _ = _joint_world(seed=20)
        names = tuple(f"f{index}" for index in range(6))
        model = fit_jbd_model(
            covariances[:-1], names, JBDConfiguration("jbd", ridge=0.1)
        )
        held = covariances[-1].copy()
        held[5, :] = np.nan
        held[:, 5] = np.nan
        rows = masked_reconstruction_rows(model, held, "held")
        self.assertEqual(len(rows), 5 * 4)
        self.assertNotIn("f5", {
            value
            for row in rows
            for value in (row["held_out_feature"], row["partner_feature"])
        })

    def test_stationary_null_preserves_missingness_and_psd_observed_blocks(self):
        covariances, _ = _joint_world(seed=21, environments=5)
        incomplete = covariances.copy()
        incomplete[0, 4:, :] = np.nan
        incomplete[0, :, 4:] = np.nan
        incomplete[1, 0, :] = np.nan
        incomplete[1, :, 0] = np.nan
        null = missingness_preserving_stationary_null(
            incomplete, (80, 90, 100, 110, 120), seed=41
        )
        self.assertTrue(np.array_equal(np.isnan(null), np.isnan(incomplete)))
        for original, covariance in zip(incomplete, null):
            observed = np.flatnonzero(np.isfinite(np.diag(original)))
            empirical = covariance[np.ix_(observed, observed)]
            self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(empirical))), -1e-10)

    def test_randomized_basis_recovers_joint_components(self):
        covariances, mixing = _joint_world()
        _, root, _, whitened, _ = whiten_covariances(covariances)
        basis, _ = randomized_joint_basis(whitened, draws=32, seed=11)
        recovered = root @ basis
        alignment = align_mixing(mixing, recovered)
        self.assertGreater(alignment["minimum_absolute_cosine"], 0.99)

    def test_jacobi_refinement_does_not_increase_off_diagonal_energy(self):
        covariances, _ = _joint_world(seed=8)
        _, _, _, whitened, _ = whiten_covariances(covariances)
        rng = np.random.default_rng(4)
        initial, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        before = off_diagonal_energy([
            initial.T @ covariance @ initial for covariance in whitened
        ])
        _, diagnostics = jacobi_refine(whitened, initial)
        self.assertLessEqual(diagnostics["final_off_diagonal_energy"], before + 1e-12)

    def test_block_graph_recovers_two_connected_components(self):
        transformed = np.zeros((8, 4, 4))
        for environment in range(8):
            transformed[environment] = np.diag([1.0, 1.1, 0.9, 1.2])
            transformed[environment, 0, 1] = transformed[environment, 1, 0] = 0.3
            transformed[environment, 2, 3] = transformed[environment, 3, 2] = 0.25
        coupling = residual_coupling(transformed)
        blocks = blocks_from_coupling(coupling, quantile=0.5)
        self.assertEqual(blocks, ((0, 1), (2, 3)))

    def test_masked_reconstruction_is_finite(self):
        covariances, _ = _joint_world(seed=9)
        names = tuple(f"f{index}" for index in range(6))
        config = JBDConfiguration("ajd", ridge=0.1, random_draws=16)
        model = fit_jbd_model(covariances[:-1], names, config)
        rows = masked_reconstruction_rows(model, covariances[-1], "held")
        self.assertEqual(len(rows), 30)
        self.assertTrue(np.isfinite([row["prediction"] for row in rows]).all())

    def test_enclosing_full_block_pca_uses_full_symmetric_space(self):
        covariances, _ = _joint_world(seed=15)
        names = tuple(f"f{index}" for index in range(6))
        model = fit_jbd_model(
            covariances,
            names,
            JBDConfiguration("pca_full", ridge=0.1),
        )
        self.assertEqual(model.diagnostics["block_sizes"], (6,))
        self.assertEqual(model.diagnostics["n_covariance_atoms"], 21)

    def test_pca_block_control_matches_jbd_mechanism_count(self):
        covariances, _ = _joint_world(seed=18)
        names = tuple(f"f{index}" for index in range(6))
        jbd = fit_jbd_model(
            covariances,
            names,
            JBDConfiguration("jbd", ridge=0.1, block_quantile=0.5),
        )
        control = fit_pca_block_model(
            covariances,
            names,
            ridge=0.1,
            block_sizes=tuple(len(block) for block in jbd.blocks),
        )
        self.assertEqual(control.atoms.shape, jbd.atoms.shape)
        self.assertEqual(control.configuration.ridge, jbd.configuration.ridge)

    def test_row_reassignment_null_is_psd_and_preserves_sample_counts(self):
        rng = np.random.default_rng(10)
        residuals = [rng.normal(size=(size, 6)) for size in (80, 110, 140)]
        shuffled = shuffled_environment_row_null(residuals, seed=22)
        self.assertEqual(shuffled.shape, (3, 6, 6))
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(shuffled))), -1e-10)
        np.testing.assert_allclose(
            np.diagonal(shuffled, axis1=1, axis2=2), 1.0, atol=1e-12
        )
        balanced = balanced_environment_row_null(residuals, seed=23)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(balanced))), -1e-10)
        np.testing.assert_allclose(
            np.diagonal(balanced, axis1=1, axis2=2), 1.0, atol=1e-12
        )

    def test_full_block_prediction_is_basis_invariant_after_frobenius_orthogonalization(self):
        covariances, _ = _joint_world(seed=16)
        names = tuple(f"f{index}" for index in range(6))
        config = JBDConfiguration("jbd", ridge=0.1, block_quantile=0.0)
        jbd = fit_jbd_model(covariances[:-1], names, config)
        pca = fit_jbd_model(
            covariances[:-1], names, JBDConfiguration("pca_full", ridge=0.1)
        )
        from dataclasses import replace
        from spectral_utils.multi_environment_jbd import covariance_atoms
        full_block = (tuple(range(6)),)
        atoms, labels = covariance_atoms(jbd.mixing, full_block)
        jbd = replace(jbd, blocks=full_block, atoms=atoms, atom_labels=labels)
        self.assertEqual(jbd.atoms.shape, pca.atoms.shape)
        jbd_rows = masked_reconstruction_rows(jbd, covariances[-1], "held")
        pca_rows = masked_reconstruction_rows(pca, covariances[-1], "held")
        self.assertLess(max(
            abs(left["prediction"] - right["prediction"])
            for left, right in zip(jbd_rows, pca_rows)
        ), 1e-10)

    def test_mechanism_overlap_is_invariant_to_within_block_rotation(self):
        covariances, _ = _joint_world(seed=12)
        names = tuple(f"f{index}" for index in range(6))
        config = JBDConfiguration("jbd", ridge=0.1, block_quantile=0.0)
        reference = fit_jbd_model(covariances, names, config)
        rng = np.random.default_rng(33)
        rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        from dataclasses import replace
        from spectral_utils.multi_environment_jbd import covariance_atoms
        full_block = ((0, 1, 2, 3, 4, 5),)
        reference_atoms, reference_labels = covariance_atoms(reference.mixing, full_block)
        reference = replace(
            reference, atoms=reference_atoms, atom_labels=reference_labels
        )
        atoms, labels = covariance_atoms(reference.mixing @ rotation, full_block)
        candidate = replace(
            reference,
            mixing=reference.mixing @ rotation,
            atoms=atoms,
            atom_labels=labels,
        )
        overlap = mechanism_subspace_overlap(reference, candidate)
        self.assertAlmostEqual(overlap["projector_overlap_on_smaller_span"], 1.0)
        self.assertAlmostEqual(overlap["rank_ratio"], 1.0)

    def test_public_model_fit_has_no_label_argument(self):
        self.assertNotIn("labels", inspect.signature(fit_jbd_model).parameters)
        source = inspect.getsource(multi_environment_jbd)
        self.assertNotIn("FEATURE_TO_VIEW", source)
        self.assertNotIn("roc_auc", source)

    def test_feature_order_permutation_equivariance(self):
        covariances, _ = _joint_world(seed=14)
        names = tuple(f"f{index}" for index in range(6))
        config = JBDConfiguration("jbd", ridge=0.1, block_quantile=0.8)
        reference = fit_jbd_model(covariances[:-1], names, config)
        reference_rows = masked_reconstruction_rows(reference, covariances[-1], "held")
        permutation = np.asarray([3, 0, 5, 2, 1, 4])
        permuted_names = tuple(names[index] for index in permutation)
        permuted_covariances = covariances[:, permutation][:, :, permutation]
        candidate = fit_jbd_model(permuted_covariances[:-1], permuted_names, config)
        candidate_rows = masked_reconstruction_rows(candidate, permuted_covariances[-1], "held")
        lookup = {
            (row["held_out_feature"], row["partner_feature"]): row["prediction"]
            for row in candidate_rows
        }
        error = max(
            abs(row["prediction"] - lookup[
                (row["held_out_feature"], row["partner_feature"])
            ])
            for row in reference_rows
        )
        self.assertLess(error, 1e-10)


if __name__ == "__main__":
    unittest.main()
