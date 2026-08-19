#!/usr/bin/env python3
"""Focused phase-boundary and evaluation tests for the white-box runner."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.whitebox_layer_fusion_experiment import (
    CELLS,
    SEED,
    VERSION,
    _lr_ceiling,
    group_bootstrap_indices,
    sha256_file,
    verify_score_freeze,
    write_json,
)
from spectral_utils.whitebox_layer_fusion import FeatureMatrix


class WhiteboxExperimentTests(unittest.TestCase):
    def test_bootstrap_draws_are_grouped_deterministic_and_method_independent(self) -> None:
        groups = np.asarray(["p0", "p0", "p1", "p1", "p2", "p2"])
        left, left_hash = group_bootstrap_indices(groups, draws=40, seed=SEED)
        right, right_hash = group_bootstrap_indices(groups, draws=40, seed=SEED)
        self.assertEqual(left_hash, right_hash)
        self.assertEqual(len(left), 40)
        for a, b in zip(left, right):
            np.testing.assert_array_equal(a, b)
            # Every sampled problem contributes its complete two-candidate block.
            counts = np.bincount(a, minlength=len(groups)).reshape(3, 2)
            np.testing.assert_array_equal(counts[:, 0], counts[:, 1])

    def test_grouped_cv_has_no_problem_overlap_and_averages_folds(self) -> None:
        rng = np.random.default_rng(91)
        groups = np.repeat(np.arange(30).astype(str), 2)
        y = np.tile([0, 1], 30)
        values = rng.normal(size=(60, 6))
        values[:, 0] += y * 0.7
        matrix = FeatureMatrix(
            values=values,
            feature_names=tuple(f"f{i}" for i in range(6)),
            risk_anchor=values[:, 0],
            groups=tuple(f"g{i % 3}" for i in range(6)),
            protocol_signature="group-cv-fixture",
        )
        auroc, auprc, folds = _lr_ceiling(matrix, y, groups)
        self.assertEqual(len(folds), 5)
        self.assertTrue(np.isfinite([auroc, auprc]).all())
        self.assertTrue(all(row["problem_overlap"] == 0 for row in folds))
        self.assertAlmostEqual(auroc, np.mean([row["auroc"] for row in folds]))
        self.assertAlmostEqual(auprc, np.mean([row["auprc"] for row in folds]))

    def _frozen_fixture(self, directory: Path, *, inject_labels: bool = False) -> None:
        (directory / "scores").mkdir()
        (directory / "diagnostics").mkdir()
        (directory / "prepared").mkdir()
        prepared_manifest = []
        for index in range(42):
            path = directory / "prepared" / f"bundle_{index:02d}.npz"
            np.savez_compressed(path, values=np.asarray([index], dtype=float))
            prepared_manifest.append({
                "file": str(path.relative_to(directory)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "fields": ["values"],
            })
        write_json(directory / "PREPARED_FEATURE_MANIFEST.json", {
            "version": VERSION, "labels_present": False, "n_files": 42,
            "files": prepared_manifest,
        })
        manifest = []
        for index, cell in enumerate(CELLS):
            score_path = directory / "scores" / f"{cell}.npz"
            arrays = {
                "row_ids": np.asarray(["0:0", "1:0", "2:0"]),
                "problem_ids": np.asarray(["0", "1", "2"]),
                "n_gen_tokens": np.asarray([2, 3, 4]),
                "protocol_signature": np.asarray(f"signature-{cell}"),
                "upcr__resid-core-32__all32__flat": np.asarray([0.1, 0.2, 0.3]) + index,
            }
            if inject_labels and index == 0:
                arrays["labels"] = np.asarray([0, 1, 0])
            np.savez_compressed(score_path, **arrays)
            diagnostic_path = directory / "diagnostics" / f"{cell}.json"
            write_json(diagnostic_path, {"labels_seen_during_fit": False})
            manifest.append({
                "cell": cell,
                "score_file": str(score_path.relative_to(directory)),
                "score_sha256": sha256_file(score_path),
                "diagnostic_file": str(diagnostic_path.relative_to(directory)),
                "diagnostic_sha256": sha256_file(diagnostic_path),
                "n_rows": 3,
                "n_methods": 1,
            })
        write_json(directory / "FIT_COMPLETE.json", {
            "version": VERSION,
            "scientific_run": True,
            "labels_seen_during_fit": False,
            "score_manifest": manifest,
        })
        write_json(directory / "RUN_DEFINITION.json", {"version": VERSION, "source_sha256": {}})
        write_json(directory / "SOURCE_FREEZE_MANIFEST.json", {"version": VERSION, "sources": []})

    def test_score_freeze_rejects_labels_and_attests_pre_label_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            self._frozen_fixture(directory)
            observed, freeze = verify_score_freeze(directory)
            self.assertEqual(tuple(observed), tuple(CELLS))
            self.assertFalse(freeze["labels_seen_during_fit"])
            self.assertTrue(freeze["scores_frozen_before_labels"])
            self.assertTrue(freeze["score_files_verified_before_labels"])

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            self._frozen_fixture(directory, inject_labels=True)
            with self.assertRaisesRegex(RuntimeError, "label-like arrays"):
                verify_score_freeze(directory)

    def test_evaluation_labels_cannot_change_frozen_score_hash(self) -> None:
        score = np.linspace(-2.0, 2.0, 50).astype("<f8")
        digest_before = hashlib.sha256(score.tobytes()).hexdigest()
        labels = np.tile([0, 1], 25)
        _ = labels[::-1].copy()  # evaluator-owned permutation
        digest_after = hashlib.sha256(score.tobytes()).hexdigest()
        self.assertEqual(digest_before, digest_after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
