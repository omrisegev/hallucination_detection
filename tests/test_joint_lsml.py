from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from spectral_utils.fusion_utils import lsml_continuous, sml_fuse_signed
from spectral_utils.joint_lsml import (
    consensus_orientation_and_roster,
    covariance_matrix,
    discover_loao_consensus_groups,
    dispatch_alias,
    fit_joint_lsml,
    hard_lsml_misfit,
    pairwise_score_spearman,
    raw_orientation_cell,
    weight_maps,
)


class JointLSMLTests(unittest.TestCase):
    def test_aliases_dispatch_to_maintained_implementations_bit_exactly(self) -> None:
        rng = np.random.default_rng(31)
        values = rng.normal(size=(200, 9))
        flat_expected, flat_weight = sml_fuse_signed(*values.T)
        flat_score, alias_weight, flat_meta = dispatch_alias(
            values, np.zeros(9, dtype=int), mode="flat_sml"
        )
        self.assertTrue(np.array_equal(flat_score, flat_expected))
        self.assertTrue(np.array_equal(alias_weight, flat_weight))
        self.assertEqual(flat_meta["dispatch"], "sml_fuse_signed")

        groups = np.repeat(np.arange(3), 3)
        lsml_expected, _ = lsml_continuous(
            *values.T, groups=groups, compute_score_matrix=False
        )
        lsml_score, alias_weight, lsml_meta = dispatch_alias(
            values, groups, mode="two_stage_alias"
        )
        self.assertTrue(np.array_equal(lsml_score, lsml_expected))
        self.assertIsNone(alias_weight)
        self.assertEqual(lsml_meta["dispatch"], "lsml_continuous")

    def test_orientation_consensus_excludes_weak_unstable_and_low_degree(self) -> None:
        estimates = []
        for cell in range(9):
            signs = np.asarray([-1, -1, 1, -1], dtype=int)
            signs[2] = 1 if cell < 5 else -1
            estimates.append({
                "signs": signs,
                "absolute_loading": np.asarray([0.4, 0.2, 0.03, 0.001]),
                "degree_keep": np.asarray([True, cell < 7, True, True]),
            })
        result = consensus_orientation_and_roster(estimates, [-1, 1, -1, 1])
        self.assertTrue(result["active"][0])
        self.assertFalse(result["active"][1])
        self.assertFalse(result["active"][2])
        self.assertFalse(result["active"][3])
        self.assertTrue(result["degree_rejected"][1])
        self.assertTrue(result["unstable"][2])
        self.assertTrue(result["weak"][3])
        self.assertEqual(result["schema_signs"][1], 1)
        self.assertEqual(result["schema_signs"][3], 1)

    def test_raw_orientation_uses_entropy_negative_anchor(self) -> None:
        rng = np.random.default_rng(19)
        latent = rng.normal(size=800)
        raw = np.column_stack([
            -latent + 0.1 * rng.normal(size=len(latent)),
            latent + 0.1 * rng.normal(size=len(latent)),
            0.7 * latent + 0.2 * rng.normal(size=len(latent)),
        ])
        result = raw_orientation_cell(raw, entropy_index=0)
        self.assertEqual(int(result["signs"][0]), -1)
        self.assertEqual(int(result["signs"][1]), 1)
        self.assertGreater(float(result["absolute_loading"][0]), 0.4)

    @staticmethod
    def _planted_values() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(702)
        n_answers = 18
        tokens_per_answer = 24
        labels = np.repeat(np.arange(3), 4)
        rows = []
        for _ in range(n_answers):
            global_latent = rng.normal(size=tokens_per_answer)
            group_latent = rng.normal(size=(tokens_per_answer, 3))
            noise = rng.normal(scale=0.08, size=(tokens_per_answer, len(labels)))
            row = np.column_stack([
                (0.65 + 0.04 * feature) * global_latent
                + (0.75 + 0.03 * feature) * group_latent[:, labels[feature]]
                + noise[:, feature]
                for feature in range(len(labels))
            ])
            rows.append(row)
        return np.vstack(rows), np.repeat(np.arange(n_answers), tokens_per_answer)

    def test_loao_consensus_recovers_stable_nontrivial_partition(self) -> None:
        values, owners = self._planted_values()
        result = discover_loao_consensus_groups(
            values, owners, k_range=(3, 4), seed=77, minimum_group_size=3
        )
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["K"], 3)
        self.assertEqual(tuple(result["group_sizes"]), (4, 4, 4))
        self.assertGreater(result["median_ari"], 0.95)

    def test_loao_rejects_a_small_group_in_any_held_answer_partition(self) -> None:
        rng = np.random.default_rng(8)
        values = rng.normal(size=(80, 9))
        owners = np.repeat(np.arange(4), 20)
        balanced = np.repeat(np.arange(3), 3)
        singleton = np.asarray([0, 1, 1, 1, 1, 2, 2, 2, 2])
        calls = [balanced, singleton, balanced, balanced, balanced]
        with patch(
            "spectral_utils.joint_lsml._spectral_cluster_precomputed",
            side_effect=[row.copy() for row in calls],
        ):
            result = discover_loao_consensus_groups(
                values, owners, k_range=(3,), seed=4, minimum_group_size=3
            )
        self.assertEqual(result["status"], "BLOCKED_NO_ADMISSIBLE_PARTITION")
        self.assertFalse(result["candidates"][0]["admissible"])
        self.assertEqual(
            result["candidates"][0]["rejection_reason"],
            "GROUP_SIZE_LT3_IN_CONSENSUS_OR_LOAO",
        )

    def test_joint_fit_and_weight_maps_are_finite_and_monotone(self) -> None:
        rng = np.random.default_rng(99)
        labels = np.repeat(np.arange(3), 4)
        size = len(labels)
        v = np.linspace(0.25, 0.65, size)
        u = np.concatenate([
            np.linspace(0.7, 0.9, 4),
            np.linspace(0.5, 0.8, 4),
            np.linspace(0.6, 1.0, 4),
        ])
        same = labels[:, None] == labels[None, :]
        covariance = np.outer(v, v) + same * np.outer(u, u) + np.diag(np.linspace(0.7, 1.1, size))
        fit = fit_joint_lsml(
            covariance, labels, anchor_index=0, seed=71,
            max_sweeps=5000, relative_tolerance=1e-10,
        )
        self.assertTrue(fit.converged)
        self.assertEqual(fit.multistart_audit["status"], "PASS")
        self.assertLess(fit.relative_offdiag_misfit, 1e-6)
        self.assertGreater(fit.jacobian_audit["rank"], 0)
        for start in fit.starts:
            differences = np.diff(np.asarray(start.objective_trace))
            self.assertTrue(np.all(differences <= 1e-9))

        values = rng.multivariate_normal(np.zeros(size), covariance, size=1800)
        maps = weight_maps(
            values, covariance_matrix(values), labels, fit,
            anchor_index=0, target_condition=1e3,
        )
        self.assertEqual(len(maps["pairwise_score_spearman"]), 6)
        self.assertTrue(all(np.isfinite(value) for value in maps["pairwise_score_spearman"].values()))
        self.assertTrue(all(np.isfinite(weight).all() for weight in maps["weights"].values()))
        self.assertLess(
            fit.relative_offdiag_misfit,
            hard_lsml_misfit(covariance, labels)["relative_offdiag_misfit"],
        )

    def test_module_has_no_benchmark_outcome_imports(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "spectral_utils/joint_lsml.py").read_text()
        for forbidden in (
            "load_localization_targets", "load_processbench_labels",
            "load_prmbench_labels", "roc_auc_score", "average_precision_score",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
