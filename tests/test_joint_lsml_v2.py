"""v2 invariants: identities, guards, SD/orientation, roster fitting.

Protocol: docs/experiments/JOINT_LSML_OPTIMIZATION_PLAN_V2.md, Section 8 item 3.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from spectral_utils.fusion_utils import lsml_continuous, sml_fuse_signed
from spectral_utils.joint_lsml import (
    covariance_matrix,
    effective_gates,
    fit_joint_lsml,
    gated_joint_hierarchical_fit,
    hierarchical_joint_weights,
    regularized_joint_map_weights,
)
from spectral_utils.joint_lsml_localization import prepare_active23
from spectral_utils.joint_lsml_v2_localization import (
    IU_ROSTER,
    LSML_ROSTER,
    donor_scale_orient,
    provenance_merged_labels,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "fusion_utils_v1_regression.npz"


def _planted(seed: int = 7, n_rows: int = 40, tokens_per_row: int = 60):
    """Synthetic 29-stream telemetry with a shared latent + group structure."""
    rng = np.random.default_rng(seed)
    n = n_rows * tokens_per_row
    latent = rng.normal(size=n)
    group_latent = rng.normal(size=(n, 4))
    raw = np.empty((n, 29))
    for feature in range(29):
        raw[:, feature] = (
            (0.5 + 0.02 * feature) * latent
            + 0.7 * group_latent[:, feature % 4]
            + 0.3 * rng.normal(size=n)
        )
    offsets = np.arange(0, n + 1, tokens_per_row)
    rows = [f"row{i}" for i in range(n_rows)]
    return raw, offsets, rows


FROZEN_RETAINED_23 = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 19, 20, 21, 23, 24, 25, 26, 27, 28]


def _preparation(**kwargs):
    raw, offsets, rows = _planted()
    from spectral_utils.feature_contract import confidence_sign_vector
    from spectral_utils.fixed_application_pipelines import (
        SHARED_GLOBAL_FEATURES,
        SHARED_TOKEN_VIEWS,
    )

    signs = confidence_sign_vector(SHARED_GLOBAL_FEATURES)
    return prepare_active23(
        raw, offsets, rows,
        retained_indices=FROZEN_RETAINED_23,
        confidence_signs_29=signs,
        stream_names_29=SHARED_TOKEN_VIEWS,
        raw_feature_names_29=SHARED_GLOBAL_FEATURES,
        **kwargs,
    )


class RegressionFixtureTests(unittest.TestCase):
    """The frozen pre-change behavior is reproduced bit-exactly by default."""

    def test_sml_and_continuous_match_frozen_fixture(self) -> None:
        data = np.load(FIXTURE)
        X = data["X"]
        score, weights = sml_fuse_signed(*[X[:, i] for i in range(X.shape[1])])
        np.testing.assert_array_equal(weights, data["sml_w"])
        np.testing.assert_array_equal(score, data["sml_score"])
        cont_score, meta = lsml_continuous(
            *[X[:, i] for i in range(X.shape[1])],
            groups=data["groups"], compute_score_matrix=False,
        )
        np.testing.assert_array_equal(cont_score, data["cont_score"])
        np.testing.assert_array_equal(np.asarray(meta["cross_weights"]), data["cross_w"])
        for position, (_, within) in enumerate(meta["group_weights"]):
            np.testing.assert_array_equal(np.asarray(within), data[f"gw_{position}"])


class GateIdentityTests(unittest.TestCase):
    def test_all_ones_gates_take_the_frozen_path_verbatim(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.normal(size=(300, 8))
        base_score, base_w = sml_fuse_signed(*[X[:, i] for i in range(8)])
        ones_score, ones_w = sml_fuse_signed(*[X[:, i] for i in range(8)], gates=np.ones(8))
        np.testing.assert_array_equal(base_w, ones_w)
        np.testing.assert_array_equal(base_score, ones_score)

    def test_gated_weights_are_the_congruence_pullback(self) -> None:
        rng = np.random.default_rng(4)
        X = rng.normal(size=(500, 6)) + rng.normal(size=(500, 1))
        q = np.asarray([1.3, 0.7, 1.0, 0.2, 0.9, 1.1])
        score, w = sml_fuse_signed(*[X[:, i] for i in range(6)], gates=q)
        R = np.cov((X * q[None, :]).T)
        R_off = R - np.diag(np.diag(R))
        from scipy.linalg import eigh

        _, vecs = eigh(R_off)
        v = vecs[:, -1]
        if np.sum(v > 0) < 3:
            v = -v
        np.testing.assert_allclose(np.abs(w), np.abs(q * v), atol=1e-10)
        np.testing.assert_allclose(score, X @ w, atol=1e-10)

    def test_effective_gates_lambda_zero_is_all_ones(self) -> None:
        q = np.asarray([0.2, 1.7, 0.9])
        np.testing.assert_array_equal(effective_gates(q, 0.0, 3), np.ones(3))
        np.testing.assert_allclose(effective_gates(q, 0.5, 3), 0.5 + 0.5 * q)

    def test_small_m_guard_replaces_three_unit_stage_with_equal_sd_weights(self) -> None:
        rng = np.random.default_rng(5)
        X = rng.normal(size=(200, 3)) * np.asarray([1.0, 2.0, 4.0])
        _, w = sml_fuse_signed(*[X[:, i] for i in range(3)], small_m_guard=True)
        expected = (1.0 / 3.0) / X.std(axis=0)
        np.testing.assert_allclose(w, expected, atol=1e-12)

    def test_lsml_continuous_gates_thread_within_only_and_flag_small_m(self) -> None:
        rng = np.random.default_rng(6)
        X = rng.normal(size=(400, 9)) + 0.6 * rng.normal(size=(400, 1))
        groups = np.repeat(np.arange(3), 3)
        q = np.linspace(0.5, 1.5, 9)
        _, meta = lsml_continuous(
            *[X[:, i] for i in range(9)], groups=groups,
            compute_score_matrix=False, gates=q, small_m_guard=True,
        )
        self.assertTrue(meta["gates_applied"])
        guarded = dict((kind, group) for kind, group in meta["small_m_guarded"])
        self.assertIn("cross", guarded)  # K=3 cross stage guarded
        self.assertEqual(sum(1 for kind, _ in meta["small_m_guarded"] if kind == "within"), 3)


class JointWrapperTests(unittest.TestCase):
    def _joint_inputs(self):
        rng = np.random.default_rng(9)
        labels = np.repeat(np.arange(3), 4)
        v = np.linspace(0.3, 0.7, 12)
        u = np.linspace(0.5, 1.0, 12)
        same = labels[:, None] == labels[None, :]
        cov = np.outer(v, v) + same * np.outer(u, u) + np.diag(np.linspace(0.6, 1.0, 12))
        X = rng.multivariate_normal(np.zeros(12), cov, size=1500)
        return X, labels

    def test_gated_joint_with_ones_matches_ungated_bitwise(self) -> None:
        X, labels = self._joint_inputs()
        cov = covariance_matrix(X)
        fit = fit_joint_lsml(cov, labels, anchor_index=0, seed=42)
        _, w_ref, _ = hierarchical_joint_weights(
            X, labels, fit.global_loading, anchor_index=0, small_m_guard=True
        )
        w_gated, _, _ = gated_joint_hierarchical_fit(
            X, labels, np.ones(12), anchor_index=0, seed=42, small_m_guard=True
        )
        np.testing.assert_array_equal(w_gated, np.asarray(w_ref))

    def test_regularized_map_lambda_zero_is_exact_identity(self) -> None:
        X, labels = self._joint_inputs()
        cov = covariance_matrix(X)
        fit = fit_joint_lsml(cov, labels, anchor_index=0, seed=42)
        from spectral_utils.dependency_fusion import regularized_covariance_weights

        reference, _ = regularized_covariance_weights(
            fit.model_covariance, fit.global_loading, target_condition=1e3
        )
        for mode in ("liu", "diag"):
            weight, meta = regularized_joint_map_weights(
                X, fit.model_covariance, fit.global_loading,
                mode=mode, lam=0.0, gates=np.ones(12),
            )
            np.testing.assert_array_equal(weight, np.asarray(reference))
        weight_liu, _ = regularized_joint_map_weights(
            X[:400], fit.model_covariance, fit.global_loading,
            mode="liu", lam=0.1, gates=np.linspace(0.5, 1.5, 12),
        )
        self.assertTrue(np.isfinite(weight_liu).all())
        self.assertGreater(np.max(np.abs(weight_liu - np.asarray(reference))), 0.0)


class InvariantTests(unittest.TestCase):
    def test_donor_scale_orient_sets_sd_one_and_orients(self) -> None:
        rng = np.random.default_rng(11)
        Z = rng.normal(size=(800, 23)) + 0.5 * rng.normal(size=(800, 1))
        w = rng.normal(size=23)
        oriented, meta = donor_scale_orient(w, Z, entropy_index=0)
        score = Z @ oriented
        self.assertAlmostEqual(float(score.std()), 1.0, places=9)
        self.assertGreaterEqual(meta["anchor_correlation"], 0.0)
        flipped, _ = donor_scale_orient(-oriented, Z, entropy_index=0)
        np.testing.assert_allclose(flipped, oriented, atol=1e-12)

    def test_donor_scale_orient_fails_closed_on_degenerate_sd(self) -> None:
        Z = np.zeros((100, 23))
        Z[:, 0] = 1.0
        with self.assertRaises(RuntimeError):
            donor_scale_orient(np.zeros(23), Z + 1e-15, entropy_index=0)

    def test_provenance_merge_produces_min_size_three(self) -> None:
        families = (
            ["entropy_level"]
            + ["entropy_dynamics"] * 3
            + ["topk_distribution"] * 2
            + ["sampled_token_energy"] * 8
            + ["partition_energy"] * 3
            + ["structural"] * 6
        )
        merged, meta = provenance_merged_labels(families)
        sizes = [int(np.sum(merged == label)) for label in np.unique(merged)]
        self.assertGreaterEqual(min(sizes), 3)
        self.assertEqual(meta["merged_group_size"], 3)

    def test_rosters_have_sixteen_rows_each(self) -> None:
        self.assertEqual(len(LSML_ROSTER), 16)
        self.assertEqual(len(IU_ROSTER), 16)
        self.assertEqual(len({row_id for row_id, _ in IU_ROSTER}), 16)


class FitRowMaskTests(unittest.TestCase):
    def test_default_path_matches_unmasked_behavior(self) -> None:
        full = _preparation()
        self.assertEqual(int(full.diagnostics["n_fit_tokens"]), 40 * 60)

    def test_mask_restricts_fit_population_to_selected_rows(self) -> None:
        mask = np.zeros(40, dtype=bool)
        mask[:25] = True
        prep = _preparation(fit_row_mask=mask)
        self.assertEqual(int(prep.diagnostics["n_fit_rows"]), 25)
        self.assertTrue(np.all(prep.fit_indices < 25 * 60))
        self.assertEqual(len(prep.raw), 40 * 60)  # projection still covers everything


if __name__ == "__main__":
    unittest.main()
