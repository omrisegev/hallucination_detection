#!/usr/bin/env python3
"""Acceptance gates for the frozen 14-cell white-box data audit."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.whitebox_layer_fusion_experiment import (  # noqa: E402
    CELLS,
    EXPECTED_COHORTS,
    EXPECTED_EXCLUSIONS,
)


class FullWhiteboxDataAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = Path(
            os.environ.get(
                "WHITEBOX_RESULTS_DIR",
                REPO / "results" / "whitebox_layer_fusion_v2",
            )
        )
        cls.audit = json.loads((cls.results / "data_audit.json").read_text())
        cls.prepared = json.loads(
            (cls.results / "PREPARED_FEATURE_MANIFEST.json").read_text()
        )
        cls.sources = json.loads(
            (cls.results / "SOURCE_FREEZE_MANIFEST.json").read_text()
        )

    def test_exact_roster_and_totals(self) -> None:
        self.assertEqual(list(self.audit["cells"]), sorted(CELLS))
        self.assertEqual(self.audit["n_cells"], 14)
        self.assertEqual(self.audit["n_source_rows"], 47265)
        self.assertEqual(self.audit["n_evaluable_rows"], 47238)
        self.assertEqual(self.audit["n_excluded_rows"], 27)

    def test_cell_groups_shapes_finiteness_and_join_identity(self) -> None:
        for name, spec in CELLS.items():
            with self.subTest(cell=name):
                cell = self.audit["cells"][name]
                cohort = EXPECTED_COHORTS[name]
                self.assertEqual(cell["n_source_rows"], spec["source_rows"])
                self.assertEqual(cell["source_n_problems"], cohort["source_groups"])
                self.assertEqual(cell["n_problems"], cohort["valid_groups"])
                self.assertEqual(cell["n_layers"], spec.get("n_layers", 32))
                self.assertTrue(cell["all_registered_tensor_shapes_valid"])
                self.assertTrue(cell["all_core_lens_values_finite"])
                self.assertTrue(cell["labels_equal_between_raw_and_sidecar"])
                self.assertTrue(cell["token_lengths_equal_for_evaluable_rows"])
                self.assertEqual(
                    {row["row_id"] for row in cell["excluded_rows"]},
                    set(EXPECTED_EXCLUSIONS.get(name, {})),
                )

    def test_geometry_overflow_is_exact_and_quarantined(self) -> None:
        expected = {
            "gsm8k_phi35_t1.0": 733,
            "gsm8k_phi3mini_t1.0": 4675,
            "triviaqa_qwen3_t0.6": 47008,
        }
        for name, cell in self.audit["cells"].items():
            count = cell["nonfinite_geometry_counts"]["cov_eigs"]
            self.assertEqual(count, expected.get(name, 0))
            self.assertEqual(cell["geometry_tensors_finite"], count == 0)
            for contract in cell["feature_contracts"].values():
                self.assertTrue(contract["finite"])

    def test_prepared_and_source_freezes_are_label_free_and_complete(self) -> None:
        self.assertFalse(self.prepared["labels_present"])
        self.assertEqual(self.prepared["n_files"], 155)
        self.assertEqual(len(self.prepared["files"]), 155)
        for item in self.prepared["files"]:
            self.assertFalse(any("label" in field.lower() for field in item["fields"]))
        self.assertEqual(self.sources["n_source_files"], 28)
        self.assertEqual(len(self.sources["sources"]), 28)
        for item in self.sources["sources"]:
            self.assertTrue(item["remote_local_sha256_equal"])
            self.assertEqual(item["remote_hash"], item["local_sha256"])
            self.assertEqual(item["remote_size"], item["local_size"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
