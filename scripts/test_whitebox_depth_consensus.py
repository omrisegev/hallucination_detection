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

from spectral_utils.whitebox_depth_consensus import (  # noqa: E402
    COMPONENTS,
    anchor_rank_consensus,
    extract_depth_consensus,
)
from spectral_utils.whitebox_layer_fusion import (  # noqa: E402
    LayerCell,
    fit_core_spectral,
)


def synthetic_cell(seed: int = 7, n: int = 48, layers: int = 8) -> LayerCell:
    rng = np.random.default_rng(seed)
    records = []
    token_counts = []
    for row in range(n):
        tokens = 3 + row % 6
        token_counts.append(tokens)
        entropy = rng.normal(4.0, 0.4, size=(3, layers, tokens))
        target = -rng.uniform(0.1, 4.0, size=(3, layers, tokens))
        top1 = target + rng.uniform(0.0, 1.0, size=(3, layers, tokens))
        kl = rng.uniform(0.0, 1.0, size=(3, layers, tokens))
        kl[2, -1] = 0.0
        records.append({
            "lens_H": entropy.astype(np.float16),
            "lens_logp_tgt": target.astype(np.float16),
            "lens_logp_top1": top1.astype(np.float16),
            "lens_kl_final": kl.astype(np.float16),
            "resid_norm": rng.uniform(1, 2, size=(layers, tokens)).astype(np.float16),
            "cov_eigs": rng.uniform(0, 1, size=(layers, 16)).astype(np.float16),
            "hid_proj": rng.normal(size=(layers, 256)).astype(np.float16),
            "n_gen_tokens": tokens,
        })
    return LayerCell(
        cell_id="synthetic",
        row_ids=tuple(f"{i}:0" for i in range(n)),
        problem_ids=tuple(str(i) for i in range(n)),
        n_gen_tokens=np.asarray(token_counts),
        records=tuple(records),
        modules=("attn", "mlp", "resid"),
        n_layers=layers,
        projection_dim=256,
        covariance_rank=16,
        provenance={"model": "synthetic", "version": "layer-lens-v1", "proj_seed": 20260811},
    )


class DepthConsensusTests(unittest.TestCase):
    def test_consensus_is_monotone_rank_invariant(self) -> None:
        rng = np.random.default_rng(3)
        values = rng.normal(size=(100, 12))
        anchor = rng.normal(size=100)
        left, left_diag = anchor_rank_consensus(
            values, anchor, k=5, reliability_power=2.0
        )
        right, right_diag = anchor_rank_consensus(
            np.exp(values), anchor, k=5, reliability_power=2.0
        )
        np.testing.assert_allclose(left, right, atol=1e-12)
        self.assertEqual(
            left_diag["selected_original_indices"],
            right_diag["selected_original_indices"],
        )

    def test_fixed_four_component_contract(self) -> None:
        matrix = extract_depth_consensus(synthetic_cell())
        self.assertEqual(matrix.feature_names, COMPONENTS)
        self.assertEqual(matrix.groups, COMPONENTS)
        self.assertEqual(matrix.values.shape, (48, 4))
        self.assertTrue(np.isfinite(matrix.values).all())
        self.assertFalse(matrix.metadata["labels_seen_during_fit"])
        for component in COMPONENTS:
            self.assertIn(component, matrix.metadata["component_fits"])

    def test_core_solvers_receive_no_labels(self) -> None:
        signature = inspect.signature(extract_depth_consensus)
        self.assertEqual(tuple(signature.parameters), ("cell",))
        matrix = extract_depth_consensus(synthetic_cell(seed=11))
        scores, diagnostics = fit_core_spectral(matrix)
        self.assertEqual(set(scores), {"upcr", "iu_pcr", "dufs_liu_pcr"})
        self.assertFalse(diagnostics["labels_seen_during_fit"])
        for score in scores.values():
            self.assertEqual(score.shape, (48,))
            self.assertTrue(np.isfinite(score).all())

    def test_outcome_like_record_is_rejected_upstream(self) -> None:
        cell = synthetic_cell()
        bad = dict(cell.records[0])
        bad["label"] = True
        with self.assertRaises(ValueError):
            LayerCell(
                cell_id=cell.cell_id,
                row_ids=cell.row_ids,
                problem_ids=cell.problem_ids,
                n_gen_tokens=cell.n_gen_tokens,
                records=(bad,) + cell.records[1:],
                modules=cell.modules,
                n_layers=cell.n_layers,
                projection_dim=cell.projection_dim,
                covariance_rank=cell.covariance_rank,
                provenance=cell.provenance,
            )


if __name__ == "__main__":
    unittest.main()
