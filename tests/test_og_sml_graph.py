from __future__ import annotations

import unittest

import numpy as np

from spectral_utils.og_sml_graph import (
    exclusive_graphs,
    fiedler,
    free_graph,
    graph_identifiability_report,
    groups_from_partition,
    is_admissible,
    is_bipartite,
)


class OGSMLGraphTests(unittest.TestCase):
    def test_partition_k3_is_identifiable_when_groups_have_triangles(self) -> None:
        groups = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
        report = graph_identifiability_report(groups, p=9, global_loading=np.ones(9))
        self.assertTrue(report.admissible)
        self.assertTrue(report.free_connected)
        self.assertFalse(report.free_bipartite)
        self.assertEqual([item["exclusive_edge_count"] for item in report.exclusive], [3, 3, 3])

    def test_partition_k2_free_graph_is_bipartite(self) -> None:
        groups = ((0, 1, 2), (3, 4, 5))
        report = graph_identifiability_report(groups, p=6, global_loading=np.ones(6))
        self.assertFalse(report.admissible)
        self.assertTrue(report.free_bipartite)
        self.assertIn("FREE_GRAPH_BIPARTITE", report.blockers)

    def test_small_group_fails_exclusive_triangle_condition(self) -> None:
        groups = ((0, 1), (2, 3, 4), (5, 6, 7))
        report = graph_identifiability_report(groups, p=8, global_loading=np.ones(8))
        self.assertFalse(report.admissible)
        self.assertIn("GROUP_0_EXCLUSIVE_SUPPORT_LT3", report.blockers)

    def test_overlap_without_exclusive_edges_fails(self) -> None:
        groups = ((0, 1, 2), (0, 1, 2), (3, 4, 5))
        report = graph_identifiability_report(groups, p=6, global_loading=np.ones(6))
        self.assertFalse(report.admissible)
        self.assertEqual(report.exclusive[0]["exclusive_edge_count"], 0)
        self.assertEqual(report.exclusive[1]["exclusive_edge_count"], 0)

    def test_free_and_exclusive_edge_construction(self) -> None:
        groups = ((0, 1, 2), (1, 2, 3))
        h = free_graph(groups, p=5)
        exclusive = exclusive_graphs(groups, p=5)
        self.assertFalse(h[0, 1])
        self.assertFalse(h[2, 3])
        self.assertTrue(h[0, 3])
        self.assertFalse(exclusive[0][1, 2])
        self.assertTrue(exclusive[0][0, 2])

    def test_fiedler_and_bipartite_known_graphs(self) -> None:
        triangle = np.ones((3, 3), dtype=float) - np.eye(3)
        path = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        self.assertAlmostEqual(fiedler(triangle), 3.0)
        self.assertFalse(is_bipartite(triangle.astype(bool)))
        self.assertTrue(is_bipartite(path.astype(bool)))

    def test_partition_conversion_and_public_boolean(self) -> None:
        groups = groups_from_partition([2, 2, 7, 7, 9, 9])
        self.assertEqual(groups, ((0, 1), (2, 3), (4, 5)))
        self.assertFalse(is_admissible(groups, p=6))


if __name__ == "__main__":
    unittest.main()
