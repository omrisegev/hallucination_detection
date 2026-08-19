#!/usr/bin/env python3
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.test_whitebox_depth_consensus import synthetic_cell  # noqa: E402
from spectral_utils.whitebox_depth_tail_consensus import (  # noqa: E402
    COMPONENTS,
    TAIL_COMPONENTS,
    _aggregate,
    extract_depth_tail_consensus,
)
from spectral_utils.whitebox_depth_tail_organic_consensus import (  # noqa: E402
    ORGANIC_COMPONENT,
    extract_depth_tail_organic_consensus,
)
from spectral_utils.whitebox_depth_distributed_pure import (  # noqa: E402
    KL_COMPONENT,
    extract_depth_distributed_pure,
)
from spectral_utils.whitebox_layer_fusion import fit_core_spectral  # noqa: E402


class DepthTailConsensusTests(unittest.TestCase):
    def test_label_free_seven_component_contract(self) -> None:
        self.assertEqual(tuple(inspect.signature(extract_depth_tail_consensus).parameters), ("cell",))
        matrix = extract_depth_tail_consensus(synthetic_cell())
        self.assertEqual(matrix.feature_names, COMPONENTS)
        self.assertEqual(matrix.groups, COMPONENTS)
        self.assertEqual(matrix.values.shape, (48, 7))
        self.assertTrue(np.isfinite(matrix.values).all())
        self.assertFalse(matrix.metadata["labels_seen_during_fit"])

    def test_spread_components_cover_every_depth_quartile(self) -> None:
        matrix = extract_depth_tail_consensus(synthetic_cell(layers=8))
        diagnostics = matrix.metadata["tail_component_fits"]
        for name in ("max_top1_surprisal_spread8", "max_target_nll_spread8"):
            selected = diagnostics[name]["selected_layers"]
            self.assertEqual(len(selected), 8)
            for band in ((0, 1), (2, 3), (4, 5), (6, 7)):
                self.assertEqual(sum(layer in band for layer in selected), 2)
        self.assertEqual(
            len(diagnostics["max_entropy_excess_over_top1_top8"]["selected_layers"]),
            8,
        )

    def test_variable_token_lengths_and_fit_are_finite(self) -> None:
        matrix = extract_depth_tail_consensus(synthetic_cell(seed=19, n=51, layers=12))
        scores, diagnostics = fit_core_spectral(matrix)
        self.assertFalse(diagnostics["labels_seen_during_fit"])
        self.assertEqual(set(scores), {"upcr", "iu_pcr", "dufs_liu_pcr"})
        for score in scores.values():
            self.assertEqual(score.shape, (51,))
            self.assertTrue(np.isfinite(score).all())

    def test_tail_components_are_not_degenerate_duplicates(self) -> None:
        matrix = extract_depth_tail_consensus(synthetic_cell(seed=23, layers=8))
        tail = matrix.values[:, -len(TAIL_COMPONENTS):]
        correlation = np.corrcoef(tail, rowvar=False)
        off_diagonal = correlation[np.triu_indices(len(TAIL_COMPONENTS), 1)]
        self.assertTrue(np.all(np.abs(off_diagonal) < 0.999))

    def test_layer_organic_extension_has_real_layer_groups(self) -> None:
        matrix = extract_depth_tail_organic_consensus(synthetic_cell(seed=29, layers=8))
        self.assertEqual(matrix.values.shape, (48, 8))
        self.assertEqual(matrix.feature_names[-1], ORGANIC_COMPONENT)
        fit = matrix.metadata["organic_fit"]
        self.assertEqual(fit["n_groups"], 8)
        self.assertEqual(fit["group_names"], [f"layer_{layer:02d}" for layer in range(8)])
        for inner in fit["inner"]:
            self.assertEqual(len(inner["feature_names"]), 3)
        self.assertFalse(matrix.metadata["labels_seen_during_fit"])

    def test_spread4_and_organic_modes_are_depth_distributed(self) -> None:
        rng = np.random.default_rng(31)
        values = rng.normal(size=(80, 24))
        anchor = rng.normal(size=80)
        _score, spread = _aggregate(values, anchor, n_layers=8, mode="spread4")
        self.assertEqual(len(spread["selected_layers"]), 4)
        for band in ((0, 1), (2, 3), (4, 5), (6, 7)):
            self.assertEqual(sum(layer in band for layer in spread["selected_layers"]), 1)
        _score, organic = _aggregate(values, anchor, n_layers=8, mode="organic_all")
        self.assertEqual(sorted(organic["selected_layers"]), list(range(8)))
        self.assertTrue(organic["virtual_layer_aggregation"])
        self.assertIsNone(organic["selected_modules"])

    def test_pure_distributed_contract_has_no_output_expert(self) -> None:
        matrix = extract_depth_distributed_pure(synthetic_cell(seed=37, layers=8))
        self.assertEqual(matrix.values.shape, (48, 13))
        self.assertEqual(matrix.feature_names[-1], KL_COMPONENT)
        self.assertNotIn("generation_entropy_mean", matrix.feature_names)
        selected = matrix.metadata["kl_component_fit"]["selected_layers"]
        self.assertEqual(len(selected), 8)
        for band in ((0, 1), (2, 3), (4, 5), (6, 7)):
            self.assertEqual(sum(layer in band for layer in selected), 2)
        self.assertFalse(matrix.metadata["labels_seen_during_fit"])


if __name__ == "__main__":
    unittest.main()
