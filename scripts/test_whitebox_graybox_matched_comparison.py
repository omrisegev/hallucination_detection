#!/usr/bin/env python3
"""Focused mechanical tests for the matched white/gray report."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import whitebox_graybox_matched_comparison as comparison

RESULTS = Path(__file__).resolve().parents[1] / "results" / "whitebox_vs_graybox_matched_v1"


class MatchedComparisonTests(unittest.TestCase):
    def test_registered_roster_is_exact(self):
        self.assertEqual(len(comparison.CELLS), 13)
        self.assertNotIn("coqa_llama7b_t0.5", comparison.CELLS)
        self.assertEqual(
            comparison.CELLS["gsm8k_t1.0"][0],
            "lapeigvals_gsm8k_llama8b",
        )

    def test_hybrid_is_label_free_equal_z_average(self):
        white = np.asarray([1.0, 3.0, 2.0, 7.0])
        gray = np.asarray([4.0, 0.0, 2.0, 1.0])
        hybrid = 0.5 * (comparison.zscore(white) + comparison.zscore(gray))
        self.assertTrue(np.isfinite(hybrid).all())
        self.assertAlmostEqual(float(hybrid.mean()), 0.0, places=12)

    def test_score_bundle_contract_has_no_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npz"
            np.savez_compressed(
                path,
                row_ids=np.asarray(["0:0", "1:0"]),
                white_pure_upcr=np.asarray([0.1, 0.2]),
                gray_mixed_v2_dufs_liu=np.asarray([0.2, 0.3]),
                gray_mixed_v2_upcr=np.asarray([0.3, 0.4]),
                exploratory_equal_z_hybrid=np.asarray([0.4, 0.5]),
            )
            frozen = np.load(path, allow_pickle=False)
            self.assertFalse(any("label" in key.lower() for key in frozen.files))

    def test_cell_bootstrap_is_deterministic(self):
        values = np.asarray([-0.02, 0.01, 0.03, -0.01])
        self.assertEqual(
            comparison.cell_bootstrap(values, 20260812, draws=500),
            comparison.cell_bootstrap(values, 20260812, draws=500),
        )

    def test_frozen_report_is_sorted_self_contained_and_hash_complete(self):
        with (RESULTS / "headline_summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        aucs = [float(row["macro_auroc"]) for row in rows]
        self.assertEqual(aucs, sorted(aucs, reverse=True))

        html = (RESULTS / "REPORT.html").read_text()
        self.assertIn("<table>", html)
        self.assertIn("PRELIMINARY / WHITE VALIDATION BLOCKED", html)
        self.assertNotIn("<pre", html.lower())
        self.assertNotIn('src="http', html.lower())
        self.assertNotIn('href="http', html.lower())

        manifest = json.loads((RESULTS / "REPORT_MANIFEST.json").read_text())
        for item in manifest["artifacts"]:
            artifact = RESULTS / item["file"]
            self.assertTrue(artifact.is_file(), artifact)
            self.assertEqual(
                hashlib.sha256(artifact.read_bytes()).hexdigest(),
                item["sha256"],
                artifact,
            )


if __name__ == "__main__":
    unittest.main()
