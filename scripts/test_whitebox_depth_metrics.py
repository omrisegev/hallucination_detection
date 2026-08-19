#!/usr/bin/env python3
"""Synthetic invariants for the frozen depth-distributed metric registry."""

from __future__ import annotations

import unittest

import numpy as np

from spectral_utils.whitebox_depth_metrics import (
    DEPTH_METRIC_REGISTRY,
    extract_ghost_geometry,
    extract_module_conflict,
    extract_prediction_revision,
    extract_target_commitment,
    ghost_layer_interval,
    registry_hash,
)
from spectral_utils.whitebox_layer_fusion import LayerCell


def fixture_cell(n: int = 12, layers: int = 32, tokens: int = 5) -> LayerCell:
    rng = np.random.default_rng(17)
    records = []
    for row in range(n):
        base = rng.normal(size=(3, layers, tokens))
        logp_tgt = -np.abs(base + np.linspace(2.0, 0.2, layers)[None, :, None])
        logp_top1 = -np.abs(0.3 * base + 0.2)
        entropy = np.abs(0.2 * base + np.linspace(1.5, 0.3, layers)[None, :, None])
        kl = np.abs(0.1 * base + np.linspace(1.0, 0.0, layers)[None, :, None])
        kl[:, -1, :] = 0.0
        hid = np.cumsum(rng.normal(scale=0.1, size=(layers, 16)), axis=0)
        records.append({
            "lens_H": entropy.astype(np.float32),
            "lens_logp_tgt": logp_tgt.astype(np.float32),
            "lens_logp_top1": logp_top1.astype(np.float32),
            "lens_kl_final": kl.astype(np.float32),
            "resid_norm": np.ones((layers, tokens), dtype=np.float32),
            "cov_eigs": np.ones((layers, 4), dtype=np.float32),
            "hid_proj": hid.astype(np.float32),
            "n_gen_tokens": tokens,
        })
    return LayerCell(
        cell_id="fixture",
        row_ids=tuple(f"{index}:0" for index in range(n)),
        problem_ids=tuple(str(index) for index in range(n)),
        n_gen_tokens=np.full(n, tokens),
        records=tuple(records),
        modules=("attn", "mlp", "resid"),
        n_layers=layers,
        projection_dim=16,
        covariance_rank=4,
        provenance={"model": "fixture", "version": "layer-lens-v1", "proj_seed": 17},
    )


class DepthMetricTests(unittest.TestCase):
    def test_registry_is_stable_and_complete(self) -> None:
        self.assertEqual(set(DEPTH_METRIC_REGISTRY), {
            "target_commitment", "module_conflict", "prediction_revision", "ghost_geometry"
        })
        self.assertEqual(len(registry_hash()), 64)

    def test_ghost_window_matches_paper_32_layer_example(self) -> None:
        self.assertEqual(ghost_layer_interval(32), tuple(range(3, 30)))

    def test_target_commitment_is_one_expert_per_layer(self) -> None:
        matrix = extract_target_commitment(fixture_cell())
        self.assertEqual(matrix.values.shape, (12, 32))
        self.assertEqual(len(set(matrix.groups)), 32)
        self.assertTrue(np.allclose(matrix.values[:, -1], matrix.risk_anchor))

    def test_module_conflict_is_three_features_per_layer(self) -> None:
        matrix = extract_module_conflict(fixture_cell())
        self.assertEqual(matrix.values.shape, (12, 96))
        self.assertEqual(len(set(matrix.groups)), 32)
        self.assertTrue(np.all(matrix.values >= 0.0))

    def test_prediction_revision_drops_degenerate_final_kl(self) -> None:
        matrix = extract_prediction_revision(fixture_cell())
        self.assertEqual(len(set(matrix.groups)), 31)
        self.assertNotIn("prediction_revision.kl_to_final.transition_30_31", matrix.feature_names)
        self.assertIn(
            "prediction_revision.kl_to_final.transition_29_30", matrix.feature_names
        )

    def test_ghost_metrics_are_rotation_invariant(self) -> None:
        cell = fixture_cell()
        original = extract_ghost_geometry(cell)
        rng = np.random.default_rng(23)
        q, _ = np.linalg.qr(rng.normal(size=(cell.projection_dim, cell.projection_dim)))
        rotated_records = []
        for record in cell.records:
            changed = dict(record)
            changed["hid_proj"] = np.asarray(record["hid_proj"]) @ q
            rotated_records.append(changed)
        rotated = LayerCell(
            cell_id=cell.cell_id,
            row_ids=cell.row_ids,
            problem_ids=cell.problem_ids,
            n_gen_tokens=cell.n_gen_tokens,
            records=tuple(rotated_records),
            modules=cell.modules,
            n_layers=cell.n_layers,
            projection_dim=cell.projection_dim,
            covariance_rank=cell.covariance_rank,
            provenance=cell.provenance,
        )
        after = extract_ghost_geometry(rotated)
        self.assertTrue(np.allclose(original.values, after.values, atol=1e-10))
        self.assertEqual(len(set(original.groups)), len(ghost_layer_interval(32)) - 1)

    def test_extractors_never_create_label_fields(self) -> None:
        cell = fixture_cell()
        for extractor in (
            extract_target_commitment,
            extract_module_conflict,
            extract_prediction_revision,
            extract_ghost_geometry,
        ):
            matrix = extractor(cell)
            self.assertFalse(any("label" in name.lower() for name in matrix.feature_names))
            self.assertFalse(any("label" in key.lower() for key in matrix.metadata))


if __name__ == "__main__":
    unittest.main()
