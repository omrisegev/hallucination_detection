#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.graph_topology import (  # noqa: E402
    _knn_table,
    adaptive_knn_graph,
    adaptive_neighbor_counts,
    diffusion_edge_matched_graph,
    exact_length_permutations,
    extended_graph_diagnostics,
    holm_adjust,
    length_only_graph,
    matched_pair_permutations,
    propensity_crt_permutations,
    radius_edge_matched_graph,
    self_safe_knn_graph,
    smoothness_against_permutations,
)
from spectral_utils.laplacian_upcr import (  # noqa: E402
    laplacian_iu_path,
    self_tuning_knn_graph,
)
from scripts.direct_dufs_conditional_graph_topology_audit_v1 import (  # noqa: E402
    CANDIDATE_GRAPHS,
    DECISION_GRAPHS,
    GRAPH_ORDER,
    REPRESENTATIONS,
    TIE_SEEDS,
    _canonical_roundtrip,
    control_checks,
    decide,
    evaluate_cell,
)


class GraphTopologyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        dense = rng.normal(0.0, 0.12, size=(80, 4))
        sparse = rng.normal(2.0, 0.75, size=(40, 4))
        self.samples = np.vstack([dense, sparse])
        self.tie_keys = rng.random(len(self.samples))

    def test_radius_matches_self_safe_union_and_proves_boundary(self):
        union = self_safe_knn_graph(
            self.samples, k=7, tie_keys=self.tie_keys
        )
        expected = union.nnz // 2
        graph, diagnostic = radius_edge_matched_graph(
            self.samples,
            edge_count=expected,
            scale_k=7,
            initial_candidate_k=16,
            tie_keys=self.tie_keys,
        )
        self.assertEqual(graph.nnz // 2, expected)
        self.assertTrue(diagnostic["candidate_boundary_proven"])
        np.testing.assert_allclose(graph.toarray(), graph.toarray().T, atol=0.0)

    def test_radius_memory_cap_fails_closed(self):
        rng = np.random.default_rng(2)
        samples = rng.normal(size=(100, 3))
        with self.assertRaises(RuntimeError):
            radius_edge_matched_graph(
                samples,
                edge_count=3000,
                initial_candidate_k=8,
                max_candidate_k=16,
                tie_keys=rng.random(len(samples)),
            )

    def test_radius_cap_does_not_allow_uncapped_n_minus_one_exception(self):
        rng = np.random.default_rng(3)
        samples = rng.normal(size=(20, 3))
        with self.assertRaises(RuntimeError):
            radius_edge_matched_graph(
                samples,
                edge_count=180,
                initial_candidate_k=16,
                max_candidate_k=16,
                tie_keys=rng.random(len(samples)),
            )

    def test_radius_handles_large_duplicate_group_without_dense_expansion(self):
        rng = np.random.default_rng(12)
        samples = np.vstack([
            np.zeros((90, 3)),
            rng.normal(size=(30, 3)),
        ])
        keys = rng.random(len(samples))
        union = self_safe_knn_graph(samples, k=7, tie_keys=keys)
        graph, diagnostic = radius_edge_matched_graph(
            samples,
            edge_count=union.nnz // 2,
            initial_candidate_k=8,
            max_candidate_k=16,
            tie_keys=keys,
        )
        self.assertEqual(graph.nnz, union.nnz)
        self.assertTrue(diagnostic["candidate_boundary_proven"])
        self.assertGreater(diagnostic["exact_zero_pairs_total"], union.nnz // 2)
        self.assertEqual(
            diagnostic["exact_zero_pairs_materialized"], union.nnz // 2
        )

    def test_mixed_duplicates_keep_positive_structural_edge_weights(self):
        rng = np.random.default_rng(44)
        samples = np.vstack([
            np.zeros((10, 3)),
            rng.normal(scale=0.2, size=(20, 3)),
        ])
        keys = rng.random(len(samples))
        union = self_safe_knn_graph(samples, k=7, tie_keys=keys)
        radius, diagnostic = radius_edge_matched_graph(
            samples,
            edge_count=union.nnz // 2,
            tie_keys=keys,
        )
        self.assertEqual(radius.nnz, union.nnz)
        self.assertTrue(diagnostic["candidate_boundary_proven"])
        self.assertTrue(np.all(union.data > 0))
        self.assertTrue(np.all(radius.data > 0))

    def test_duplicate_rows_never_consume_a_nonself_slot(self):
        rng = np.random.default_rng(4)
        samples = np.vstack([
            np.zeros((24, 3)),
            rng.normal(size=(16, 3)),
        ])
        keys = rng.random(len(samples))
        distances, indices, _ = _knn_table(samples, 25, tie_keys=keys)
        self.assertEqual(indices.shape, (40, 25))
        self.assertEqual(distances.shape, (40, 25))
        for row in range(len(samples)):
            self.assertNotIn(row, indices[row])
        _, diagnostic = adaptive_knn_graph(samples, tie_keys=keys)
        self.assertEqual(diagnostic["directed_k_sum"], 7 * len(samples))

    def test_adaptive_allocation_has_exact_mean_and_tracks_sparsity(self):
        local_scale = np.linspace(0.1, 3.0, 120)
        counts = adaptive_neighbor_counts(
            local_scale,
            mean_k=7,
            min_k=3,
            max_k=25,
            rank_power=8.0,
            tie_keys=self.tie_keys,
        )
        self.assertEqual(int(np.sum(counts)), 7 * len(counts))
        self.assertGreater(np.mean(counts[-20:]), np.mean(counts[:20]))

    def test_new_graphs_are_row_permutation_equivariant(self):
        rng = np.random.default_rng(29)
        samples = np.vstack([self.samples[:70], self.samples[:10]])
        keys = rng.random(len(samples))
        permutation = rng.permutation(len(samples))
        inverse = np.argsort(permutation)

        def back(graph):
            array = graph.toarray()
            return array[np.ix_(inverse, inverse)]

        union = self_safe_knn_graph(samples, k=7, tie_keys=keys)
        permuted_union = self_safe_knn_graph(
            samples[permutation], k=7, tie_keys=keys[permutation]
        )
        np.testing.assert_allclose(union.toarray(), back(permuted_union), atol=1e-12)
        edge_count = union.nnz // 2

        constructors = (
            lambda x, k: radius_edge_matched_graph(
                x, edge_count=edge_count, tie_keys=k
            )[0],
            lambda x, k: adaptive_knn_graph(x, tie_keys=k)[0],
            lambda x, k: diffusion_edge_matched_graph(
                x,
                edge_count=edge_count,
                base_k=25,
                steps=2,
                row_keep=25,
                tie_keys=k,
            )[0],
        )
        for constructor in constructors:
            original = constructor(samples, keys)
            permuted = constructor(samples[permutation], keys[permutation])
            np.testing.assert_allclose(original.toarray(), back(permuted), atol=1e-10)

    def test_holm_missing_member_fails_closed(self):
        adjusted = holm_adjust(np.asarray([0.01, 0.04, 0.02, 0.8, np.nan]))
        np.testing.assert_array_equal(adjusted[:4], np.ones(4))
        self.assertTrue(np.isnan(adjusted[4]))

    def test_algebraic_connectivity_is_rebuild_deterministic(self):
        graph = self_safe_knn_graph(
            self.samples, k=7, tie_keys=self.tie_keys
        )
        first = extended_graph_diagnostics(graph)["algebraic_connectivity"]
        np.random.default_rng(999).normal(size=10000)
        second = extended_graph_diagnostics(graph)["algebraic_connectivity"]
        self.assertEqual(first, second)


class ConditionalNullTests(unittest.TestCase):
    def test_exact_swaps_preserve_every_exact_length_count(self):
        length = np.repeat(np.arange(24), 10).astype(float)
        target = np.asarray([
            int((row % 10) < (2 + (row // 10) // 5))
            for row in range(len(length))
        ])
        draws, diagnostic = exact_length_permutations(
            target, length, permutations=31, seed=9
        )
        for value in np.unique(length):
            indexes = length == value
            np.testing.assert_array_equal(
                np.sum(draws[indexes], axis=0), np.sum(target[indexes])
            )
        self.assertGreaterEqual(diagnostic["movable_fraction"], 0.80)

    def test_unique_length_adversary_cannot_pass_exact_or_crt(self):
        length = np.arange(800, dtype=float)
        target = (length >= 400).astype(int)
        exact, exact_diag = exact_length_permutations(
            target, length, permutations=199, seed=31
        )
        self.assertEqual(exact_diag["movable_fraction"], 0.0)
        crt, crt_diag = propensity_crt_permutations(
            target, length, permutations=199, seed=32
        )
        result = smoothness_against_permutations(
            length_only_graph(length, k=7, tie_keys=np.linspace(0.001, 0.999, 800)),
            target,
            crt,
        )
        crt_eligible = bool(
            crt_diag["overlap_fraction"] >= 0.20
            and crt_diag["brier"] <= crt_diag["constant_brier"] + 0.01
            and crt_diag["calibration_mae"] <= 0.10
        )
        self.assertTrue((not crt_eligible) or result["p_smoother"] > 0.05)
        self.assertEqual(exact.shape, crt.shape)

    def test_steep_within_pair_adversary_shows_pair_is_not_primary(self):
        base = np.arange(120, dtype=float) * 10.0
        length = np.ravel(np.column_stack([base, base + 0.01]))
        target = np.tile([0, 1], len(base))
        pairs, diagnostic = matched_pair_permutations(
            target, length, permutations=199, seed=77
        )
        # This coordinate is a deterministic function of length (position
        # inside each close pair), so it is nuisance-only despite perfectly
        # separating the target.
        graph = self_safe_knn_graph(
            np.mod(length, 10.0)[:, None],
            k=7,
            tie_keys=np.linspace(0.001, 0.999, len(length)),
        )
        result = smoothness_against_permutations(graph, target, pairs)
        self.assertEqual(diagnostic["discordant_pairs"], len(base))
        self.assertLessEqual(result["p_smoother"], 0.05)
        exact, exact_diag = exact_length_permutations(
            target, length, permutations=31, seed=78
        )
        self.assertEqual(exact_diag["movable_fraction"], 0.0)
        self.assertEqual(exact.shape[0], len(target))


class DecisionTests(unittest.TestCase):
    def _fixtures(self, geometry=True, utility=True):
        summaries = []
        intervals = []
        for lane in ("global24", "processbench", "ragtruth"):
            for representation in REPRESENTATIONS:
                for graph in CANDIDATE_GRAPHS:
                    for tie_seed in TIE_SEEDS:
                        summaries.append({
                            "lane": lane,
                            "representation": representation,
                            "graph": graph,
                            "tie_seed": tie_seed,
                            "exact_eligible_fraction": 1.0,
                            "crt_eligible_fraction": 1.0,
                            "fraction_exact_effect_positive": 1.0 if geometry else 0.0,
                            "fraction_exact_holm_p_le_0p05": 1.0 if geometry else 0.0,
                            "fraction_crt_effect_positive": 1.0 if geometry else 0.0,
                            "fraction_crt_holm_p_le_0p05": 1.0 if geometry else 0.0,
                            "healthy_graph_fraction": 1.0,
                        })
                        intervals.append({
                            "lane": lane,
                            "representation": representation,
                            "graph": graph,
                            "tie_seed": tie_seed,
                            "mean_liu_delta_auroc": 0.01 if utility else -0.01,
                            "ci_low": 0.001 if utility and lane == "global24" else -0.02,
                            "ci_high": 0.02,
                        })
        return summaries, intervals

    def test_no_geometry_decision(self):
        summaries, intervals = self._fixtures(geometry=False, utility=False)
        result = decide(summaries, intervals, {"all_controls_pass": True})
        self.assertEqual(
            result["decision"],
            "NO_GRAPH_REVEALS_LENGTH_CONDITIONAL_TARGET_GEOMETRY",
        )

    def test_joint_pass_decision(self):
        summaries, intervals = self._fixtures(geometry=True, utility=True)
        result = decide(summaries, intervals, {"all_controls_pass": True})
        self.assertEqual(
            result["decision"],
            "ROBUST_LENGTH_CONDITIONAL_GEOMETRY_AND_UTILITY",
        )

    def test_control_failure_invalidates_otherwise_positive_result(self):
        summaries, intervals = self._fixtures(geometry=True, utility=True)
        result = decide(summaries, intervals, {"all_controls_pass": False})
        self.assertEqual(
            result["decision"],
            "CONTROL_FAILURE_INVALIDATES_GEOMETRY_AUDIT",
        )

    def test_diffusion_t4_sensitivity_cannot_rescue_failed_t2(self):
        summaries, intervals = self._fixtures(geometry=False, utility=True)
        for row in summaries:
            if row["graph"] == "diffusion_edge_matched_base25_t4":
                row["fraction_exact_effect_positive"] = 1.0
                row["fraction_exact_holm_p_le_0p05"] = 1.0
                row["fraction_crt_effect_positive"] = 1.0
                row["fraction_crt_holm_p_le_0p05"] = 1.0
        result = decide(summaries, intervals, {"all_controls_pass": True})
        self.assertNotIn("diffusion_edge_matched_base25_t4", DECISION_GRAPHS)
        self.assertEqual(
            result["decision"],
            "NO_GRAPH_REVEALS_LENGTH_CONDITIONAL_TARGET_GEOMETRY",
        )


class ControlCheckTests(unittest.TestCase):
    def _rows(self):
        rows = []
        lane_cells = {
            "global24": [f"g{index}" for index in range(21)],
            "processbench": [f"p{index}" for index in range(4)],
            "ragtruth": ["r0"],
        }
        for lane, cells in lane_cells.items():
            for cell in cells:
                for representation in REPRESENTATIONS:
                    for tie_seed in TIE_SEEDS:
                        for graph_index, graph in enumerate(GRAPH_ORDER):
                            row = {
                                "lane": lane,
                                "cell": cell,
                                "representation": representation,
                                "tie_seed": tie_seed,
                                "graph": graph,
                                "n_edges": 100,
                                "exact_eligible": True,
                                "crt_eligible": True,
                                "pair_eligible": True,
                                "length_effect": 0.8,
                                "exact_target_p_smoother": 0.5,
                                "crt_target_p_smoother": 0.5,
                                "raw_target_p_smoother": 0.5,
                                "raw_target_effect": 0.1 - 0.01 * graph_index,
                            }
                            if graph == "radius_edge_matched_k7":
                                row["construction_candidate_boundary_proven"] = True
                            if graph == "adaptive_knn_mean7_k3_25":
                                row["construction_directed_k_mean"] = 7.0
                            if (
                                graph == "deployed_union_knn_k7"
                                and representation == "original"
                            ):
                                row["frozen_liu_max_abs_error"] = 0.0
                                row["frozen_liu_correlation"] = 1.0
                            rows.append(row)
        return rows

    def test_single_ragtruth_false_positive_fails_despite_low_pooled_rate(self):
        rows = self._rows()
        changed = False
        for row in rows:
            if (
                row["lane"] == "ragtruth"
                and row["representation"] == "original"
                and row["graph"] == "length_only_knn_k7"
                and row["tie_seed"] == TIE_SEEDS[0]
            ):
                row["exact_target_p_smoother"] = 0.005
                changed = True
                break
        self.assertTrue(changed)
        controls = control_checks(rows)
        self.assertLess(
            controls["length_positive_control"][
                "exact_target_false_positive_fraction"
            ],
            0.15,
        )
        rag = next(
            row for row in controls["false_positive_controls_by_lane"]
            if row["lane"] == "ragtruth"
        )
        self.assertFalse(rag["pass"])
        self.assertFalse(controls["all_controls_pass"])

    def test_multicell_false_positive_boundary_is_per_lane(self):
        rows = self._rows()
        global_length = [
            row for row in rows
            if row["lane"] == "global24"
            and row["representation"] == "original"
            and row["graph"] == "length_only_knn_k7"
        ]
        for row in global_length[:9]:
            row["exact_target_p_smoother"] = 0.005
        self.assertTrue(control_checks(rows)["all_controls_pass"])
        global_length[9]["exact_target_p_smoother"] = 0.005
        controls = control_checks(rows)
        global_control = next(
            row for row in controls["false_positive_controls_by_lane"]
            if row["lane"] == "global24"
        )
        self.assertGreater(
            global_control["length_exact_false_positive_fraction"], 0.15
        )
        self.assertFalse(controls["all_controls_pass"])

class EndToEndSyntheticTests(unittest.TestCase):
    def test_cell_evaluator_covers_factorial_and_reproduces_deployed_score(self):
        rng = np.random.default_rng(81)
        n = 120
        length = np.repeat(np.arange(12), 10).astype(float)
        base = rng.normal(size=(n, 5))
        length_z = (length - np.mean(length)) / np.std(length)
        matrix = np.column_stack([base, length_z])
        names = ("f0", "f1", "f2", "f3", "f4", "trace_length")
        target = np.asarray([
            int((index % 10) < (3 + (index // 10) % 4))
            for index in range(n)
        ])

        def gate_fitter(features):
            return np.ones(features.shape[0]), {
                "effective_feature_count": float(features.shape[0])
            }

        graph = self_tuning_knn_graph(matrix, k=7)
        path = laplacian_iu_path(matrix.T, (0.1,), graph=graph)
        frozen = -(path[0.1].w @ matrix.T)
        training = []
        for cell_index in range(2):
            shifted = matrix + rng.normal(scale=0.02, size=matrix.shape)
            training.append({
                "cell": f"train_{cell_index}",
                "matrix": shifted,
                "names": names,
                "length": length,
            })
        rows, diagnostics = evaluate_cell({
            "lane": "synthetic",
            "cell": "synthetic_cell",
            "split": "synthetic_validation",
            "matrix": matrix,
            "names": names,
            "length": length,
            "target": target,
            "training_cells": training,
            "original_gates": np.ones(matrix.shape[1]),
            "gate_fitter": gate_fitter,
            "frozen_liu": frozen,
            "frozen_tolerance": 1e-10,
        })
        self.assertEqual(
            len(rows), len(REPRESENTATIONS) * len(TIE_SEEDS) * len(GRAPH_ORDER)
        )
        self.assertEqual(diagnostics["feature_counts"]["drop_length"], 5)
        deployed = [
            row for row in rows
            if row["representation"] == "original"
            and row["graph"] == "deployed_union_knn_k7"
        ]
        self.assertEqual(len(deployed), len(TIE_SEEDS))
        self.assertTrue(all(
            row["frozen_liu_max_abs_error"] <= 1e-10 for row in deployed
        ))
        for representation in REPRESENTATIONS:
            for tie_seed in TIE_SEEDS:
                current = {
                    row["graph"]: row for row in rows
                    if row["representation"] == representation
                    and row["tie_seed"] == tie_seed
                }
                expected = current["union_knn_k7_self_safe"]["n_edges"]
                for graph_name in (
                    "radius_edge_matched_k7",
                    "diffusion_edge_matched_base25_t2",
                    "diffusion_edge_matched_base25_t4",
                ):
                    self.assertEqual(current[graph_name]["n_edges"], expected)

    def test_json_canonicalization_matches_resume_semantics(self):
        value = [{"finite": 1.0, "missing": float("nan")}]
        fresh = _canonical_roundtrip(value)
        resumed = _canonical_roundtrip(fresh)
        self.assertEqual(fresh, resumed)
        self.assertIsNone(fresh[0]["missing"])


if __name__ == "__main__":
    unittest.main()
