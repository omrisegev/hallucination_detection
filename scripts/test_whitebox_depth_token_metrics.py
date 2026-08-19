#!/usr/bin/env python3
"""Synthetic tests for iteration-2 layer-local token dynamics."""

from __future__ import annotations

import unittest

import numpy as np

from spectral_utils.whitebox_depth_token_metrics import EXTRACTORS, TOKEN_METRIC_REGISTRY, registry_hash
from spectral_utils.whitebox_layer_fusion import LayerCell


def fixture() -> LayerCell:
    rng = np.random.default_rng(11)
    n, layers = 15, 8
    records = []
    for row in range(n):
        tokens = 2 + row % 6
        entropy = np.abs(rng.normal(size=(3, layers, tokens)) + np.linspace(1.2, .2, layers)[None, :, None])
        target = -np.abs(rng.normal(size=(3, layers, tokens)) + .5)
        top1 = -np.abs(rng.normal(size=(3, layers, tokens)) + .2)
        kl = np.abs(rng.normal(size=(3, layers, tokens)) * .1 + np.linspace(.8, 0, layers)[None, :, None])
        kl[:, -1] = 0.0
        records.append({
            "lens_H": entropy, "lens_logp_tgt": target, "lens_logp_top1": top1,
            "lens_kl_final": kl, "resid_norm": np.ones((layers, tokens)),
            "cov_eigs": np.ones((layers, 4)), "hid_proj": rng.normal(size=(layers, 16)),
            "n_gen_tokens": tokens,
        })
    return LayerCell(
        cell_id="token-fixture", row_ids=tuple(f"{i}:0" for i in range(n)),
        problem_ids=tuple(str(i) for i in range(n)), n_gen_tokens=np.asarray([2+i%6 for i in range(n)]),
        records=tuple(records), modules=("attn","mlp","resid"), n_layers=layers,
        projection_dim=16, covariance_rank=4,
        provenance={"model":"fixture","version":"layer-lens-v1","proj_seed":11},
    )


class TokenMetricTests(unittest.TestCase):
    def test_registry_and_hash(self) -> None:
        self.assertEqual(set(EXTRACTORS), set(TOKEN_METRIC_REGISTRY))
        self.assertEqual(len(registry_hash()), 64)

    def test_variable_token_lengths_and_shapes(self) -> None:
        cell = fixture()
        for contract, extractor in EXTRACTORS.items():
            matrix = extractor(cell)
            self.assertEqual(matrix.n_samples, cell.n_samples)
            self.assertGreaterEqual(matrix.n_features, cell.n_layers - 1)
            self.assertTrue(np.isfinite(matrix.values).all(), contract)

    def test_one_scalar_expert_per_retained_layer(self) -> None:
        cell = fixture()
        for extractor in EXTRACTORS.values():
            matrix = extractor(cell)
            self.assertEqual(matrix.n_features, len(set(matrix.groups)))
            self.assertTrue(all(name.rsplit(".",1)[-1] == group for name,group in zip(matrix.feature_names,matrix.groups)))

    def test_final_kl_is_mechanically_dropped(self) -> None:
        matrix = EXTRACTORS["resid_kl_tail"](fixture())
        self.assertEqual(matrix.n_features, 7)
        self.assertFalse(any(name.endswith("layer_07") for name in matrix.feature_names))

    def test_outcomes_cannot_enter_contract(self) -> None:
        cell = fixture()
        for extractor in EXTRACTORS.values():
            matrix = extractor(cell)
            self.assertFalse(any("label" in key.lower() for key in matrix.metadata))
            self.assertFalse(any("label" in name.lower() for name in matrix.feature_names))


if __name__ == "__main__":
    unittest.main()
