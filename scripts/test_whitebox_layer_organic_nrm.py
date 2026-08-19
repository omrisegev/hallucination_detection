#!/usr/bin/env python3
"""Integrity tests for the layer-organic white-box NRM addendum."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import whitebox_layer_organic_nrm_experiment as experiment  # noqa: E402
from spectral_utils.whitebox_layer_fusion import FeatureMatrix  # noqa: E402
from spectral_utils.whitebox_layer_organic_nrm import (  # noqa: E402
    KL_SENSITIVITY_METRICS,
    LOCAL_RESID_METRICS,
    assert_layer_organic_contract,
    layer_organic_residual_matrix,
)


RESULTS = REPO / "results" / "whitebox_layer_organic_nrm_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fixture_grid(*, missing: str | None = None) -> FeatureMatrix:
    rng = np.random.default_rng(14)
    names, values, groups = [], [], []
    for module in ("attn", "mlp", "resid"):
        for metric in KL_SENSITIVITY_METRICS:
            for layer in range(4):
                name = f"{module}.{metric}.layer_{layer:02d}"
                if name == missing or (module == "resid" and metric == "lens_kl_final" and layer == 3):
                    continue
                names.append(name)
                values.append(rng.normal(size=48) + layer / 5)
                groups.append(f"{module}.{metric}")
    return FeatureMatrix(
        values=np.column_stack(values),
        feature_names=tuple(names),
        risk_anchor=rng.normal(size=48),
        groups=tuple(groups),
        protocol_signature="synthetic-grid",
        metadata={"contract": "lens-grid"},
    )


class OrganicContractTests(unittest.TestCase):
    def test_local_triad_is_three_features_per_layer(self):
        matrix = layer_organic_residual_matrix(fixture_grid(), metrics=LOCAL_RESID_METRICS)
        self.assertEqual(matrix.values.shape, (48, 12))
        self.assertEqual(tuple(dict.fromkeys(matrix.groups)), tuple(f"layer_{i:02d}" for i in range(4)))
        self.assertEqual([matrix.groups.count(f"layer_{i:02d}") for i in range(4)], [3, 3, 3, 3])
        self.assertTrue(all(name.startswith("resid.") for name in matrix.feature_names))
        self.assertFalse(matrix.metadata["kl_is_nonlocal_sensitivity"])
        assert_layer_organic_contract(matrix, n_layers=4)

    def test_kl_is_separate_and_final_degenerate_is_allowed(self):
        matrix = layer_organic_residual_matrix(fixture_grid(), metrics=KL_SENSITIVITY_METRICS)
        self.assertEqual(matrix.values.shape, (48, 15))
        self.assertEqual([matrix.groups.count(f"layer_{i:02d}") for i in range(4)], [4, 4, 4, 3])
        self.assertTrue(matrix.metadata["kl_is_nonlocal_sensitivity"])
        self.assertEqual(matrix.metadata["missing_mechanical_features"], ["resid.lens_kl_final.layer_03"])
        assert_layer_organic_contract(matrix, n_layers=4)

    def test_missing_local_metric_fails_closed(self):
        grid = fixture_grid(missing="resid.lens_H.layer_02")
        with self.assertRaisesRegex(ValueError, "missing non-degenerate organic feature"):
            layer_organic_residual_matrix(grid, metrics=LOCAL_RESID_METRICS)

    def test_feature_order_permutation_does_not_change_contract_values(self):
        grid = fixture_grid()
        reference = layer_organic_residual_matrix(grid)
        order = np.random.default_rng(8).permutation(grid.n_features)
        permuted = FeatureMatrix(
            values=grid.values[:, order],
            feature_names=tuple(grid.feature_names[index] for index in order),
            risk_anchor=grid.risk_anchor,
            groups=tuple(grid.groups[index] for index in order),
            protocol_signature=grid.protocol_signature,
            metadata=grid.metadata,
        )
        observed = layer_organic_residual_matrix(permuted)
        np.testing.assert_array_equal(reference.values, observed.values)
        self.assertEqual(reference.feature_names, observed.feature_names)
        self.assertEqual(reference.groups, observed.groups)


class OrganicExperimentTests(unittest.TestCase):
    def test_exact_depth_rosters(self):
        self.assertEqual(len(experiment.L32_CELLS), 10)
        self.assertEqual(len(experiment.LLAMA6_CELLS), 6)
        self.assertEqual(len(experiment.GSM8K_L32_CELLS), 5)
        self.assertEqual(set(experiment.NON_L32_CELLS), {
            "gsm8k_mistral24b_t1.0", "gsm8k_nemo_t1.0", "triviaqa_qwen3_t0.6"
        })
        for target in experiment.L32_CELLS:
            for strategy in ("lodo", "lomo", "loco"):
                sources = experiment._source_cells(target, strategy)
                self.assertNotIn(target, sources)
                self.assertGreaterEqual(len(sources), 3)
                self.assertTrue(all(int(experiment.CELLS[cell].get("n_layers", 32)) == 32 for cell in sources))

    def test_fit_phase_cannot_open_outcomes(self):
        source = inspect.getsource(experiment.phase_fit)
        self.assertNotIn("load_evaluation_labels", source)
        self.assertNotIn("roc_auc", source.lower())
        self.assertNotIn("average_precision", source.lower())

    def test_frozen_artifacts_are_label_free_and_hashed(self):
        fit = json.loads((RESULTS / "FIT_COMPLETE.json").read_text())
        freeze = json.loads((RESULTS / "SCORE_FREEZE_MANIFEST.json").read_text())
        self.assertFalse(fit["labels_seen_during_fit"])
        self.assertTrue(freeze["scores_frozen_before_labels"])
        self.assertEqual([row["cell"] for row in fit["score_manifest"]], list(experiment.L32_CELLS))
        for row in fit["score_manifest"]:
            path = RESULTS / row["score_file"]
            self.assertEqual(sha256_file(path), row["score_sha256"])
            with np.load(path, allow_pickle=False) as bundle:
                self.assertFalse(any("label" in key.lower() for key in bundle.files))
                for method in row["score_keys"]:
                    self.assertTrue(np.isfinite(bundle[method]).all())

    def test_score_freeze_is_immutable_on_repeat_verification(self):
        path = RESULTS / "SCORE_FREEZE_MANIFEST.json"
        before = sha256_file(path)
        experiment._verify_score_freeze(RESULTS)
        self.assertEqual(before, sha256_file(path))

    def test_diagnostics_use_32_groups_and_exclude_target(self):
        for path in sorted((RESULTS / "diagnostics").glob("*.json")):
            payload = json.loads(path.read_text())
            self.assertFalse(payload["labels_seen_during_fit"])
            self.assertEqual(payload["contracts"]["triad"]["n_families"], 32)
            self.assertEqual(payload["contracts"]["kl_sensitivity"]["n_families"], 32)
            self.assertLess(payload["contracts"]["triad"]["reconstruction_error"], 1e-10)
            for fit in payload["fits"].values():
                self.assertNotIn(payload["target"], fit["source_cells"])
                self.assertEqual(len(fit["calibration"]["direction"]), 32)

    def test_report_is_semantic_self_contained_and_hash_verified(self):
        report = (RESULTS / "REPORT.html").read_text()
        self.assertIn("PRELIMINARY / VALIDATION BLOCKED", report)
        self.assertGreaterEqual(report.count("<table>"), 4)
        self.assertGreaterEqual(report.count("<caption>"), 4)
        self.assertEqual(report.count("data:image/svg+xml;base64,"), 5)
        self.assertNotIn("<pre", report.lower())
        self.assertNotIn('src="http', report.lower())
        self.assertNotIn('href="http', report.lower())
        manifest = json.loads((RESULTS / "REPORT_MANIFEST.json").read_text())
        self.assertTrue(manifest["self_contained"])
        self.assertFalse(manifest["network_assets"])
        self.assertEqual(
            manifest["report_generator_sha256"],
            sha256_file(REPO / "scripts" / "whitebox_layer_organic_nrm_report.py"),
        )
        for relative, expected in manifest["inputs"].items():
            self.assertEqual(sha256_file(RESULTS / relative), expected)
        for relative, expected in manifest["generated"].items():
            self.assertEqual(sha256_file(RESULTS / relative), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
