#!/usr/bin/env python3
"""Integrity gates for the white-box NRM addendum and its report."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import whitebox_layer_fusion_nrm_experiment as experiment  # noqa: E402


RESULTS = REPO / "results" / "whitebox_layer_fusion_nrm_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class WhiteboxNRMTests(unittest.TestCase):
    def test_source_strategies_are_axis_clean(self):
        for target in experiment.PRIMARY_CELLS:
            target_spec = experiment.CELLS[target]
            lodo = experiment._source_cells(target, "lodo")
            lomo = experiment._source_cells(target, "lomo")
            loco = experiment._source_cells(target, "loco")
            self.assertGreaterEqual(len(lodo), 3)
            self.assertGreaterEqual(len(lomo), 3)
            self.assertEqual(len(loco), len(experiment.PRIMARY_CELLS) - 1)
            self.assertNotIn(target, lodo)
            self.assertNotIn(target, lomo)
            self.assertNotIn(target, loco)
            self.assertTrue(all(
                experiment.CELLS[cell]["dataset"] != target_spec["dataset"]
                for cell in lodo
            ))
            self.assertTrue(all(
                experiment.CELLS[cell].get("model", experiment.MODEL)
                != target_spec.get("model", experiment.MODEL)
                for cell in lomo
            ))

    def test_fit_phase_has_no_outcome_loader(self):
        source = inspect.getsource(experiment.phase_fit)
        self.assertNotIn("load_evaluation_labels", source)
        self.assertNotIn("roc_auc", source.lower())
        self.assertNotIn("average_precision", source.lower())

    def test_frozen_score_roster_is_label_free_and_finite(self):
        fit = json.loads((RESULTS / "FIT_COMPLETE.json").read_text())
        self.assertFalse(fit["labels_seen_during_fit"])
        self.assertEqual(
            [row["cell"] for row in fit["score_manifest"]],
            list(experiment.CELLS),
        )
        expected_methods = {
            "nrm_depth_lodo", "nrm_depth_lomo", "nrm_depth_loco",
            "nrm_lens_lodo", "nrm_lens_lomo", "nrm_lens_loco",
        }
        for row in fit["score_manifest"]:
            path = RESULTS / row["score_file"]
            self.assertEqual(sha256_file(path), row["score_sha256"])
            with np.load(path, allow_pickle=False) as bundle:
                self.assertTrue(expected_methods.issubset(bundle.files))
                self.assertFalse(any("label" in key.lower() for key in bundle.files))
                n = len(bundle["row_ids"])
                for method in expected_methods:
                    self.assertEqual(bundle[method].shape, (n,))
                    self.assertTrue(np.isfinite(bundle[method]).all())

    def test_diagnostics_reconstruct_iu_and_exclude_target(self):
        for path in sorted((RESULTS / "diagnostics").glob("*.json")):
            payload = json.loads(path.read_text())
            self.assertFalse(payload["labels_seen_during_fit"])
            target = payload["target"]
            for key, fit in payload["fits"].items():
                self.assertNotIn(target, fit["source_cells"])
                self.assertLess(float(fit["iu_reconstruction_error"]), 1e-10)
                self.assertFalse(fit["calibration"]["labels_seen_during_fit"])
                self.assertFalse(fit["target_fit"]["labels_seen_during_fit"])

    def test_report_is_self_contained_semantic_and_hash_verified(self):
        report = (RESULTS / "REPORT.html").read_text()
        self.assertIn("PRELIMINARY / VALIDATION BLOCKED", report)
        self.assertGreaterEqual(report.count("<table>"), 4)
        self.assertGreaterEqual(report.count("<caption>"), 4)
        self.assertNotIn("<pre", report.lower())
        self.assertNotIn('src="http', report.lower())
        self.assertNotIn('href="http', report.lower())
        self.assertEqual(report.count("data:image/svg+xml;base64,"), 4)
        manifest = json.loads((RESULTS / "REPORT_MANIFEST.json").read_text())
        self.assertTrue(manifest["self_contained"])
        self.assertFalse(manifest["network_assets"])
        for relative, expected in manifest["generated"].items():
            self.assertEqual(sha256_file(RESULTS / relative), expected)

        with (RESULTS / "headline_summary.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        depth = next(row for row in rows if row["method"] == "nrm_depth_lodo")
        self.assertIn(f"{float(depth['macro_auroc']):.4f}", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
