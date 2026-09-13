"""CPU-only mathematical checks; no experiment data or package-wide imports.

Run: python scripts/test_conditional_iu_graph.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse


MODULE_PATH = Path(__file__).resolve().parents[1] / "spectral_utils" / "conditional_iu_graph.py"
SPEC = importlib.util.spec_from_file_location("conditional_iu_graph", MODULE_PATH)
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


class TokenGraphTests(unittest.TestCase):
    def test_window_and_local_bandwidth_formula(self):
        x = np.arange(43, dtype=float)[:, None]
        graph = core.build_token_graph(x)
        expected = np.zeros((len(x), len(x)))
        bandwidth = np.zeros(len(x))
        for i in range(len(x)):
            neighbors = [j for j in range(len(x)) if 0 < abs(i - j) <= 16]
            bandwidth[i] = sorted(abs(i - j) for j in neighbors)[7]
        for i in range(len(x)):
            for j in range(len(x)):
                if 0 < abs(i - j) <= 16:
                    expected[i, j] = np.exp(-(i - j)**2 / (bandwidth[i] * bandwidth[j]))
        self.assertIsInstance(graph, sparse.csr_matrix)
        assert_allclose(graph.toarray(), expected, atol=1e-15)
        self.assertEqual(graph[0, 17], 0)
        # A hypothetical step boundary between rows 20 and 21 has no effect.
        self.assertGreater(graph[20, 21], 0)

    def test_uniform_control_and_canonical_edges(self):
        rng = np.random.default_rng(42)
        graph = core.build_token_graph(rng.normal(size=(40, 12)), mode="uniform")
        i, j, a = core.graph_edges(graph)
        self.assertTrue(np.all(i < j))
        self.assertTrue(np.all(j - i <= 16))
        assert_array_equal(a, np.ones_like(a))
        self.assertEqual(len(a), sum(40 - offset for offset in range(1, 17)))
        self.assertEqual(graph.nnz, 2 * len(a))
        self.assertEqual((graph - graph.T).nnz, 0)

    def test_single_short_duplicate_and_disconnected(self):
        one = core.build_token_graph([[2.0, 5.0]])
        self.assertEqual(one.shape, (1, 1))
        self.assertEqual(one.nnz, 0)
        short = core.build_token_graph([[0.0], [2.0]])
        assert_allclose(short.toarray(), [[0, np.exp(-1)], [np.exp(-1), 0]])
        duplicate = core.build_token_graph(np.ones((9, 3)))
        assert_array_equal(duplicate.toarray(), np.ones((9, 9)) - np.eye(9))
        clusters = core.build_token_graph(np.repeat([[0.0], [100.0]], 10, axis=0))
        self.assertEqual(clusters[:10, 10:].nnz, 0)
        self.assertTrue(np.isfinite(clusters.data).all())

    def test_permutation_is_stable_isomorphism_only(self):
        x = np.random.default_rng(15).normal(size=(67, 5))
        original_x = x.copy()
        graph = core.build_token_graph(x)
        shuffled, permutation = core.permute_token_graph(graph, "pb/question-17")
        repeated, p_again = core.permute_token_graph(graph, "pb/question-17")
        assert_array_equal(permutation, p_again)
        assert_array_equal(shuffled.toarray(), repeated.toarray())
        assert_array_equal(shuffled.toarray(), graph.toarray()[permutation][:, permutation])
        assert_array_equal(np.sort(graph.data), np.sort(shuffled.data))
        assert_array_equal(np.sort(np.diff(graph.indptr)), np.sort(np.diff(shuffled.indptr)))
        assert_allclose(np.sort(np.asarray(graph.sum(axis=1)).ravel()),
                        np.sort(np.asarray(shuffled.sum(axis=1)).ravel()), atol=1e-14)
        assert_array_equal(x, original_x)
        self.assertFalse(np.array_equal(permutation, np.arange(len(x))))

    def test_sparse_moments_match_brute_force_population(self):
        rng = np.random.default_rng(34)
        x = rng.normal(size=(21, 12))
        graph = core.build_token_graph(x)
        for adjacency in (graph, core.permute_token_graph(graph, "moments")[0],
                          core.build_token_graph(x, mode="uniform")):
            output = core.weighted_neighborhood_moments(x, adjacency)
            weights = adjacency.toarray() + np.eye(len(x))
            for i, weight in enumerate(weights):
                mean = np.average(x, weights=weight, axis=0)
                difference = x - mean
                covariance = (difference.T * weight) @ difference / weight.sum()
                assert_allclose(output["mean"][i], mean, atol=2e-14)
                assert_allclose(output["covariance"][i], covariance, atol=2e-14)
                self.assertAlmostEqual(output["effective_weight_count"][i],
                                       weight.sum()**2 / (weight @ weight))
                self.assertAlmostEqual(output["weight_sum"][i], weight.sum())

    def test_isolated_moments_and_translation(self):
        x = np.array([[1.0, 3.0], [3.0, -1.0], [2.0, 4.0]])
        zero = sparse.csr_matrix((len(x), len(x)))
        isolated = core.weighted_neighborhood_moments(x, zero)
        assert_array_equal(isolated["mean"], x)
        assert_array_equal(isolated["covariance"], np.zeros((3, 2, 2)))
        assert_array_equal(isolated["effective_weight_count"], np.ones(3))
        graph = core.build_token_graph(x, mode="uniform")
        ordinary = core.weighted_neighborhood_moments(x, graph)
        translated = core.weighted_neighborhood_moments(x + 1e9, graph)
        assert_allclose(translated["mean"] - 1e9, ordinary["mean"], atol=1e-7)
        assert_allclose(translated["covariance"], ordinary["covariance"], atol=1e-14)

    def test_long_graph_never_densifies(self):
        n = 2048
        x = np.random.default_rng(29).normal(size=(n, 12))
        with patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("dense adjacency")):
            graph = core.build_token_graph(x)
            moments = core.weighted_neighborhood_moments(x, graph)
            shuffled, _ = core.permute_token_graph(graph, "long")
            core.graph_edges(shuffled)
        self.assertLessEqual(graph.nnz, 32 * n)
        self.assertEqual(moments["covariance"].shape, (n, 12, 12))

    def test_invalid_graph_inputs_rejected(self):
        for invalid in ([[np.nan]], [[np.inf]], [[None]], [], np.zeros((2, 0)),
                        np.ma.array([[1.0]], mask=[[True]])):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                core.build_token_graph(invalid)
        for kwargs in ({"window": 0}, {"bandwidth_neighbor": 0}, {"bandwidth_floor": 0},
                       {"window": 1.5}, {"mode": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                core.build_token_graph([[0.0]], **kwargs)
        for bad in (np.zeros((2, 2)), sparse.eye(2),
                    sparse.csr_matrix([[0, 1], [0, 0]]),
                    sparse.csr_matrix([[0, -1], [-1, 0]]),
                    sparse.csr_matrix([[0, np.nan], [np.nan, 0]])):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                core.weighted_neighborhood_moments([[0.0], [1.0]], bad)


class NetworkLassoTests(unittest.TestCase):
    def check_certificate(self, result):
        self.assertTrue(np.isfinite(result["theta"]).all())
        self.assertGreaterEqual(result["primal_dual_gap"], 0)
        assert_allclose(result["primal_objective"] - result["dual_objective"],
                        result["primal_dual_gap"], atol=2e-11, rtol=2e-8)
        self.assertLess(result["tau"] * result["sigma"] * result["incidence_norm_squared_bound"], 1)

    def test_two_node_analytic_group_lasso(self):
        g = np.repeat(np.eye(2)[None, :, :], 2, axis=0)
        r = np.array([[0.0, 0.0], [3.0, 4.0]])
        difference = r[1] - r[0]
        center = r.mean(axis=0)
        for eta, a in ((0.1, 2.0), (1.0, 1.0), (2.5, 1.0), (7.0, 1.0)):
            with self.subTest(eta=eta, a=a):
                factor = max(0.0, 1.0 - 2 * eta * a / np.linalg.norm(difference))
                expected = np.array([center - factor * difference / 2,
                                     center + factor * difference / 2])
                result = core.solve_network_lasso(g, r, [0], [1], [a], eta)
                self.assertTrue(result["converged"])
                assert_allclose(result["theta"], expected, atol=2e-5)
                self.check_certificate(result)

    def test_eta_zero_exact_batched_solve(self):
        rng = np.random.default_rng(24)
        b = rng.normal(size=(12, 2, 2))
        g = b @ b.swapaxes(1, 2) + 0.2 * np.eye(2)
        r = rng.normal(size=(12, 2))
        expected = np.linalg.solve(g, r[..., None])[..., 0]
        result = core.solve_network_lasso(g, r, np.arange(11), np.arange(1, 12), np.ones(11), 0)
        assert_array_equal(result["theta"], expected)
        self.assertEqual(result["iterations"], 0)
        self.assertTrue(result["converged"])
        self.check_certificate(result)

    def test_constant_optimum_unchanged_with_heterogeneous_quadratics(self):
        rng = np.random.default_rng(65)
        b = rng.normal(size=(30, 2, 2))
        g = b @ b.swapaxes(1, 2) + np.eye(2)
        expected = np.tile([1.25, -0.5], (len(g), 1))
        r = np.einsum("tij,tj->ti", g, expected)
        graph = core.build_token_graph(r)
        for adjacency in (graph, core.permute_token_graph(graph, "constant")[0]):
            result = core.solve_network_lasso(g, r, *core.graph_edges(adjacency), 10.0)
            assert_allclose(result["theta"], expected, atol=1e-13)
            self.assertTrue(result["converged"])
            self.check_certificate(result)

    def test_anisotropic_problem_with_constructed_kkt_solution(self):
        rng = np.random.default_rng(76)
        n = 15
        b = rng.normal(size=(n, 2, 2))
        g = b @ b.swapaxes(1, 2) + np.eye(2)
        expected = rng.normal(size=(n, 2))
        i, j = np.arange(n - 1), np.arange(1, n)
        a = np.linspace(0.2, 1.5, n - 1)
        eta = 0.8
        diff = expected[i] - expected[j]
        dual = eta * a[:, None] * diff / np.linalg.norm(diff, axis=1)[:, None]
        r = np.einsum("tij,tj->ti", g, expected)
        np.add.at(r, i, dual)
        np.add.at(r, j, -dual)
        result = core.solve_network_lasso(g, r, i, j, a, eta)
        self.assertTrue(result["converged"])
        assert_allclose(result["theta"], expected, atol=5e-5)
        self.check_certificate(result)

    def test_single_disconnected_and_zero_weight(self):
        g = np.repeat(np.eye(2)[None], 4, axis=0)
        r = np.array([[0., 0.], [3., 4.], [17., -2.], [6., 13.]])
        result = core.solve_network_lasso(g, r, [0], [1], [1.0], 1.0)
        assert_allclose(result["theta"][2:], r[2:], atol=1e-13)
        assert_allclose(result["theta"][:2], [[0.6, 0.8], [2.4, 3.2]], atol=2e-5)
        single = core.solve_network_lasso(g[:1], r[:1], [], [], [], 10)
        assert_array_equal(single["theta"], r[:1])
        self.assertEqual(single["iterations"], 0)
        zero = core.solve_network_lasso(g, r, [0], [1], [0.0], 10)
        assert_array_equal(zero["theta"], r)
        self.assertEqual(zero["iterations"], 0)

    def test_iteration_cap_retains_finite_solution_with_failure_flag(self):
        g = np.repeat(np.eye(2)[None], 2, axis=0)
        r = np.array([[0., 0.], [3., 4.]])
        result = core.solve_network_lasso(g, r, [0], [1], [1.0], 1.0, max_iter=1)
        self.assertFalse(result["converged"])
        self.assertEqual(result["status"], "max_iter")
        self.assertEqual(result["iterations"], 1)
        self.assertGreater(result["relative_gap"], 1e-6)
        self.check_certificate(result)

    def test_missing_invalid_and_numerical_inputs_rejected(self):
        g = np.repeat(np.eye(2)[None], 2, axis=0)
        r = np.zeros((2, 2))
        calls = [
            (g * np.nan, r, [0], [1], [1], 1),
            (g, [[None, 0], [0, 0]], [0], [1], [1], 1),
            (-g, r, [0], [1], [1], 1),
            (g * 0, r, [0], [1], [1], 1),
            (g, r, [1], [0], [1], 1),
            (g, r, [0], [2], [1], 1),
            (g, r, [0.5], [1], [1], 1),
            (g, r, [0], [1], [-1], 1),
            (g, r, [0], [1], [np.nan], 1),
            (g, r, [0, 0], [1, 1], [1, 1], 1),
            (g, r, [0], [1], [1], -1),
        ]
        for args in calls:
            with self.subTest(args=args), self.assertRaises(ValueError):
                core.solve_network_lasso(*args)
        with self.assertRaises(FloatingPointError):
            core.solve_network_lasso(g, r, [0], [1], [1e308], 1e308)


if __name__ == "__main__":
    unittest.main(verbosity=2)
