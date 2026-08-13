#!/usr/bin/env python3
"""Development-only tests for the frozen A5 synthetic pipeline."""

from __future__ import annotations

import unittest

import numpy as np

from spectral_utils.a5_simulation import (
    DuplicateQuotient,
    duplicate_pair_diagnostics,
    degree_matched_random_supports,
    feature_deletion_indices,
    fit_graph_pipeline,
    fit_held_directions,
    held_mechanism_controls,
    induce_held_boundary,
    project_diagnostic_truth,
    run_synthetic_repetition,
    select_diagonal_control,
    select_random_graph_control,
    select_graph_and_alpha,
    simulate_synthetic_world,
)
from spectral_utils.a5_target_free_data import CORE_FEATURES
from spectral_utils.feature_contract import CONFIDENCE_FEATURE_SIGNS_V1


class SimulatorTests(unittest.TestCase):
    def test_observational_equivalence_is_bit_exact(self):
        left = simulate_synthetic_world(11, 510011, semantic_swap=False)
        right = simulate_synthetic_world(11, 510011, semantic_swap=True)
        for a, b in zip(left.observed, right.observed):
            np.testing.assert_array_equal(a.adaptation, b.adaptation)
            np.testing.assert_array_equal(a.evaluation, b.evaluation)
            np.testing.assert_array_equal(a.iu_weight, b.iu_weight)
        for a, b in zip(left.diagnostics, right.diagnostics):
            np.testing.assert_array_equal(a.adaptation_y, b.adaptation_z)
            np.testing.assert_array_equal(a.evaluation_y, b.evaluation_z)
            np.testing.assert_array_equal(a.target_weight, b.nuisance_weight)
        # Full fitting and selection see byte-identical ObservedEnvironment
        # tuples, so parameters, chosen alpha and scores must be identical.
        lgraph, lalpha, lrecords = select_graph_and_alpha(
            left.observed[:3], left.observed[3:4], penalties=(0.1,), alphas=(0.0, 0.5)
        )
        rgraph, ralpha, rrecords = select_graph_and_alpha(
            right.observed[:3], right.observed[3:4], penalties=(0.1,), alphas=(0.0, 0.5)
        )
        np.testing.assert_array_equal(lgraph.support, rgraph.support)
        self.assertEqual(lalpha, ralpha)
        self.assertEqual(lrecords, rrecords)
        lf = fit_held_directions(left.observed[4], lgraph, alphas=(lalpha,))
        rf = fit_held_directions(right.observed[4], rgraph, alphas=(ralpha,))
        np.testing.assert_array_equal(lf.alpha_scores[lalpha], rf.alpha_scores[ralpha])

    def test_graph_and_held_fit_do_not_accept_diagnostic_bits(self):
        world = simulate_synthetic_world(1, 510001)
        graph = fit_graph_pipeline(world.observed[:3], 0.1)
        held = fit_held_directions(world.observed[3], graph)
        self.assertEqual(set(held.alpha_scores), {0.0, 0.125, 0.25, 0.5, 1.0})
        self.assertTrue(held.mixture.converged)
        with self.assertRaises(TypeError):
            fit_held_directions(world.diagnostics[3], graph)

    def test_selection_is_deterministic_and_alpha_zero_is_exact(self):
        world = simulate_synthetic_world(2, 510002)
        first = run_synthetic_repetition(world)
        second = run_synthetic_repetition(world)
        self.assertEqual(first["selected_penalty"], second["selected_penalty"])
        self.assertEqual(first["selected_alpha"], second["selected_alpha"])
        self.assertEqual(first["candidate_auroc"], second["candidate_auroc"])
        if first["selected_alpha"] == 0:
            self.assertEqual(first["fallback_error"], 0.0)

    def test_world6_dimensions_covariance_and_end_to_end(self):
        for variant in ("exact", "near"):
            world = simulate_synthetic_world(6, 510060, duplicate_variant=variant)
            self.assertEqual(world.observed[0].adaptation.shape[1], 18)
            self.assertEqual(world.diagnostics[0].covariance.shape, (18, 18))
            self.assertEqual(world.diagnostics[0].centre.shape, (18,))
            rho = 1.0 if variant == "exact" else 0.999
            np.testing.assert_allclose(
                world.diagnostics[0].covariance[-1, :-1],
                rho * world.diagnostics[0].covariance[0, :-1],
            )
            graph = fit_graph_pipeline(world.observed[:3], 0.2)
            held = fit_held_directions(world.observed[3], graph, alphas=(0.0,))
            self.assertIn((0, 17), graph.quotient.groups)
            self.assertEqual(graph.quotient.quotient_dimension, 18)
            self.assertEqual(held.alpha_scores[0.0].shape[0], 400)
            np.testing.assert_array_equal(held.alpha_scores[0.0], held.iu_scores)

    def test_registered_split_and_deletion_are_deterministic(self):
        first = simulate_synthetic_world(3, 510003)
        second = simulate_synthetic_world(3, 510003)
        np.testing.assert_array_equal(first.observed[0].adaptation, second.observed[0].adaptation)
        self.assertEqual(len(first.observed[0].adaptation), 250)
        graph = fit_graph_pipeline(first.observed[:3], 0.2)
        graph_train_copy = first.observed[0].graph_population.copy()
        for count in (1, 2, 3):
            projected, induced, keep = induce_held_boundary(
                first.observed[3], graph, seed=first.seed, count=count
            )
            self.assertEqual(projected.adaptation.shape[1], 17-count)
            self.assertEqual(induced.support.shape[0], 17-count)
            self.assertEqual(
                feature_deletion_indices(first.seed, first.observed[3].environment_id, 17, count),
                tuple(sorted(set(range(17)) - set(keep))),
            )
            np.testing.assert_array_equal(first.observed[0].graph_population, graph_train_copy)
            truth = project_diagnostic_truth(first.diagnostics[3], keep)
            expected = np.linalg.solve(
                truth.covariance,
                (first.diagnostics[3].covariance @ first.diagnostics[3].target_weight)[keep],
            )
            np.testing.assert_allclose(truth.target_weight, expected)

    def test_illegal_variant_and_swap_combinations_close(self):
        with self.assertRaises(ValueError):
            simulate_synthetic_world(6, 510006)
        with self.assertRaises(ValueError):
            simulate_synthetic_world(1, 510001, duplicate_variant="exact")
        with self.assertRaises(ValueError):
            simulate_synthetic_world(1, 510001, semantic_swap=True)

    def test_world8_constructs_and_hash_reduces_k10_prompt_pools(self):
        first = simulate_synthetic_world(8, 510008)
        second = simulate_synthetic_world(8, 510008)
        for index, audit in enumerate(first.sampling_audits):
            self.assertEqual(audit.prompt_count, 800)
            expected_k = 1 if index < 6 else 10
            self.assertEqual(audit.candidates_per_prompt, expected_k)
            self.assertEqual(sum(audit.selected_ordinal_counts), 800)
            self.assertEqual(
                audit.selected_ordinals_sha256,
                second.sampling_audits[index].selected_ordinals_sha256,
            )
            if expected_k == 10:
                self.assertGreater(sum(value > 0 for value in audit.selected_ordinal_counts), 1)
            self.assertEqual(len(first.observed[index].graph_population), 800)

    def test_paired_duplicate_diagnostic_reports_registered_statistics(self):
        baseline = simulate_synthetic_world(1, 510061)
        duplicate = simulate_synthetic_world(6, 510061, duplicate_variant="exact")
        result = duplicate_pair_diagnostics(
            baseline, duplicate, penalties=(0.2,), alphas=(0.0,)
        )
        self.assertLessEqual(result["median_combined_mass_ratio"], 1.10)
        self.assertGreater(result["median_score_spearman"], 0.999999)
        near = duplicate_pair_diagnostics(
            baseline, simulate_synthetic_world(6, 510061, duplicate_variant="near"),
            penalties=(0.2,), alphas=(0.0, 0.5, 1.0),
        )
        self.assertLessEqual(near["median_combined_mass_ratio"], 1.10)
        self.assertGreaterEqual(near["median_score_spearman"], 0.995)
        self.assertLessEqual(near["selected_alpha_absolute_difference"], 0.125)

    def test_degree_matched_random_supports_are_unique_and_deterministic(self):
        world = simulate_synthetic_world(1, 510071)
        graph = fit_graph_pipeline(world.observed[:4], 0.1)
        names = tuple(f"feature_{index:02d}" for index in range(17))
        first = degree_matched_random_supports(
            graph.support, split_seed=510071, penalty=0.1, count=4,
            feature_names=names,
        )
        second = degree_matched_random_supports(
            graph.support, split_seed=510071, penalty=0.1, count=4,
            feature_names=names,
        )
        degree = graph.support.sum(axis=0)
        self.assertEqual(len({np.packbits(value).tobytes() for value in first}), 4)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
            np.testing.assert_array_equal(left.sum(axis=0), degree)
            self.assertEqual(left.sum(), graph.support.sum())
        permutation = np.asarray([3, 0, 8, 1, 6, 2, 9, 4, 7, 5, 16, 11, 10, 15, 12, 14, 13])
        permuted = degree_matched_random_supports(
            graph.support[np.ix_(permutation, permutation)], split_seed=510071,
            penalty=0.1, count=4,
            feature_names=tuple(names[index] for index in permutation),
        )
        inverse = np.argsort(permutation)
        for reference, candidate in zip(first, permuted):
            np.testing.assert_array_equal(
                reference, candidate[np.ix_(inverse, inverse)]
            )

    def test_learned_graph_is_permutation_equivariant_and_repeatable(self):
        world = simulate_synthetic_world(1, 510073)
        environments = world.observed[:4]
        first = fit_graph_pipeline(environments, 0.1)
        second = fit_graph_pipeline(environments, 0.1)
        np.testing.assert_array_equal(first.support, second.support)
        permutation = np.asarray([3, 0, 8, 1, 6, 2, 9, 4, 7, 5, 16, 11, 10, 15, 12, 14, 13])
        permuted = tuple(type(value)(
            value.environment_id, value.graph_population[:, permutation],
            value.adaptation[:, permutation], value.evaluation[:, permutation],
            value.iu_weight[permutation],
            tuple(value.feature_names[index] for index in permutation),
            value.confidence_signs[permutation],
        ) for value in environments)
        fitted = fit_graph_pipeline(permuted, 0.1)
        np.testing.assert_array_equal(
            first.support, fitted.support
        )

    def test_diagonal_and_random_controls_select_their_own_alpha(self):
        world = simulate_synthetic_world(1, 510072)
        graph, _, _ = select_graph_and_alpha(
            world.observed[:4], world.observed[4:5], penalties=(0.05,),
            alphas=(0.0, 0.5),
        )
        diagonal, diagonal_alpha, diagonal_records = select_diagonal_control(
            world.observed[:4], world.observed[4:5], alphas=(0.0, 0.5)
        )
        self.assertTrue(np.array_equal(diagonal.support, np.eye(17, dtype=bool)))
        self.assertIn(diagonal_alpha, (0.0, 0.5))
        self.assertEqual(len(diagonal_records), 2)
        best, arms = select_random_graph_control(
            world.observed[:4], world.observed[4:5], graph,
            split_seed=510072, alphas=(0.0, 0.5), arm_count=2,
        )
        self.assertEqual(len(arms), 2)
        self.assertIn(best["alpha"], (0.0, 0.5))
        for arm in arms:
            selected = next(
                record for record in arm["records"]
                if record["alpha"] == arm["alpha"]
            )
            self.assertEqual(arm["mean_log_likelihood"],
                             selected["mean_log_likelihood"])
            self.assertGreaterEqual(arm["empirical_best_mean_log_likelihood"],
                                    arm["mean_log_likelihood"])
        usable = [arm for arm in arms if arm["usable"]]
        expected = min(
            [arm for arm in usable if max(x["mean_log_likelihood"] for x in usable)
             - arm["mean_log_likelihood"] <= 1e-8],
            key=lambda arm: (arm["alpha"], arm["arm"]),
        )
        self.assertEqual(best["arm"], expected["arm"])
        controls = held_mechanism_controls(
            world.observed[5], graph, diagonal, best["graph"],
            candidate_alpha=0.0, diagonal_alpha=diagonal_alpha,
            random_alpha=best["alpha"],
        )
        for key in (
            "candidate_log_likelihood", "capacity_identical_alpha_zero_log_likelihood",
            "diagonal_log_likelihood", "random_graph_log_likelihood",
            "one_gaussian_log_likelihood", "unanchored_mixture_log_likelihood",
        ):
            self.assertTrue(np.isfinite(controls[key]), key)
        self.assertEqual(
            controls["candidate_log_likelihood"],
            controls["capacity_identical_alpha_zero_log_likelihood"],
        )
        np.testing.assert_array_equal(controls["candidate_scores"], controls["iu_scores"])

    def test_train_exact_contrast_retains_held_iu_identity(self):
        rng = np.random.default_rng(510074)
        names = ("a", "b", "c", "d", "e")
        signs = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0])
        graph_environments = []
        for index in range(4):
            X = rng.normal(size=(300, 5)); X[:, 1] = -X[:, 0]
            graph_environments.append(type(simulate_synthetic_world(1, 510001).observed[0])(
                f"graph_{index}", X, X[:150], X[150:],
                np.asarray([1.0, -1.0, .2, -.1, .3]), names, signs,
            ))
        held = rng.normal(size=(300, 5))
        environment = type(graph_environments[0])(
            "held", held, held[:150], held[150:],
            np.asarray([1.0, -1.0, .2, -.1, .3]), names, signs,
        )
        graph = fit_graph_pipeline(graph_environments, 0.2)
        self.assertEqual(graph.quotient.quotient_dimension, 5)
        fitted = fit_held_directions(environment, graph, alphas=(0.0,))
        np.testing.assert_allclose(fitted.alpha_scores[0.0], fitted.iu_scores, atol=1e-12)

    def test_grouped_pipeline_is_feature_permutation_equivariant(self):
        rng = np.random.default_rng(510075)
        names = tuple(CORE_FEATURES[:5])
        signs = np.asarray([CONFIDENCE_FEATURE_SIGNS_V1[name] for name in names])
        environments = []
        for index in range(5):
            latent = rng.choice((-1.0, 1.0), size=360)
            source = 1.2 * latent + rng.normal(size=360)
            X = rng.normal(size=(360, 5))
            X[:, 0] = source + .04 * rng.normal(size=360)
            X[:, 1] = source + .04 * rng.normal(size=360)
            X[:, 2] = source + .04 * rng.normal(size=360)
            X[:, 3:] += latent[:, None] * np.asarray([.5, -.4])
            anchor = np.asarray([.5, .5, .5, .2, -.2])
            environments.append(type(simulate_synthetic_world(1, 510001).observed[0])(
                f"grouped_{index}", X, X[:180], X[180:], anchor, names, signs,
            ))
        graph = fit_graph_pipeline(environments[:4], 0.2)
        fitted = fit_held_directions(environments[4], graph, alphas=(0.0, .5))
        permutation = np.asarray([2, 4, 0, 3, 1])
        permuted = tuple(type(value)(
            value.environment_id, value.graph_population[:, permutation],
            value.adaptation[:, permutation], value.evaluation[:, permutation],
            value.iu_weight[permutation], tuple(names[i] for i in permutation),
            signs[permutation],
        ) for value in environments)
        pgraph = fit_graph_pipeline(permuted[:4], 0.2)
        pfitted = fit_held_directions(permuted[4], pgraph, alphas=(0.0, .5))
        self.assertEqual(graph.quotient.coordinate_names, pgraph.quotient.coordinate_names)
        np.testing.assert_array_equal(graph.support, pgraph.support)
        for alpha in (0.0, .5):
            np.testing.assert_allclose(
                fitted.alpha_scores[alpha], pfitted.alpha_scores[alpha], atol=1e-9
            )
        projected, induced, _ = induce_held_boundary(
            environments[4], graph, seed=510075, count=1
        )
        self.assertEqual(induced.quotient.quotient_dimension, projected.adaptation.shape[1])
        self.assertEqual(np.linalg.matrix_rank(induced.quotient.transform),
                         induced.quotient.quotient_dimension)


if __name__ == "__main__":
    unittest.main()
