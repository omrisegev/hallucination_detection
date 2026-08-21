#!/usr/bin/env python3
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from scripts.direct_dufs_graph_semantics_audit_v1 import (
    encode,
    purity_test,
    smoothness_test,
)


class DirectGraphAuditTests(unittest.TestCase):
    def setUp(self):
        self.graph = csr_matrix(np.asarray([
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ], dtype=float))

    def test_target_clusters_are_smoother_than_permutation(self):
        result = smoothness_test(self.graph, [0, 0, 0, 1, 1, 1], categorical=True, seed=7, permutations=200)
        self.assertGreater(result["smoothness_z"], 1.5)

    def test_target_clusters_have_high_purity(self):
        result = purity_test(self.graph, [0, 0, 0, 1, 1, 1], seed=9, permutations=200)
        self.assertEqual(result["purity"], 1.0)
        self.assertGreater(result["purity_z"], 1.5)

    def test_categorical_encoding_is_centered(self):
        matrix = encode(["a", "b", "a", "c"], categorical=True)
        np.testing.assert_allclose(matrix.mean(axis=0), 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
