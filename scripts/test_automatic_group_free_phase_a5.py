#!/usr/bin/env python3
"""Tests for the frozen A5 nuisance-first boundary/runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import automatic_group_free_phase_a5 as runner


class BoundaryTests(unittest.TestCase):
    def test_prepare_and_verify_bind_sources_protocol_configuration_and_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            boundary = runner.prepare(directory)
            self.assertEqual(boundary, runner.load_and_verify_boundary(directory))
            self.assertFalse(boundary["configuration"]["real_cache_accessed"])
            self.assertFalse(boundary["configuration"]["retrospective_labels_accessed"])
            path = Path(directory) / "A5_BOUNDARY.json"
            changed = json.loads(path.read_text())
            changed["configuration"]["repetitions"] = 99
            path.write_text(json.dumps(changed))
            with self.assertRaises(RuntimeError):
                runner.load_and_verify_boundary(directory)

    def test_prepare_is_append_only_and_status_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            runner.prepare(directory)
            sentinel = Path(directory) / "A5_NUISANCE_COMPLETE.json"
            sentinel.write_text("sentinel")
            with self.assertRaises(RuntimeError):
                runner.prepare(directory)
            self.assertEqual(sentinel.read_text(), "sentinel")
        with tempfile.TemporaryDirectory() as directory:
            runner.prepare(directory)
            path = Path(directory) / "A5_BOUNDARY.json"
            value = json.loads(path.read_text())
            value["status"] = "tampered"
            path.write_text(json.dumps(value))
            with self.assertRaises(RuntimeError):
                runner.load_and_verify_boundary(directory)

    def test_sealed_seed_formula_and_gate_summary(self):
        self.assertEqual(runner.sealed_world_seed(8, 0), 521600)
        self.assertEqual(runner.sealed_world_seed(8, 99), 521699)
        passing = [{
            "seed": runner.sealed_world_seed(8, repetition), "usable": True,
            "target_preferred_final": True,
            "target_preferred_correction": True,
            "candidate_minus_iu": 0.01,
        } for repetition in range(100)]
        summary = runner.summarize_nuisance(passing)
        self.assertTrue(summary["gate_pass"])
        self.assertEqual(summary["verdict"], "PASS_NUISANCE_ANTI_REPACKAGING_GATE")
        failing = list(passing)
        for record in failing[:11]:
            record["target_preferred_final"] = False
        summary = runner.summarize_nuisance(failing)
        self.assertFalse(summary["gate_pass"])
        self.assertEqual(summary["verdict"], "CLOSE_NUISANCE_REPACKAGING")
        with self.assertRaises(ValueError):
            runner.summarize_nuisance(passing[:90])

    def test_numerical_failure_records_all_scheduled_seeds_and_binds_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            runner.prepare(directory)
            with patch.object(runner, "run_synthetic_repetition",
                              side_effect=RuntimeError("CLOSE_TEST")):
                summary = runner.run_nuisance(directory)
            self.assertEqual(summary["verdict"], "CLOSE_NUMERICAL_NONCONVERGENCE")
            self.assertEqual(summary["failure_count"], 100)
            records = json.loads(
                (Path(directory) / "nuisance_repetitions.json").read_text()
            )
            self.assertEqual(len(records), 100)
            self.assertEqual(records[0]["seed"], 521600)
            self.assertEqual(records[-1]["seed"], 521699)
            self.assertEqual(
                summary["boundary_sha256"],
                runner.sha256_file(Path(directory) / "A5_BOUNDARY.json"),
            )
            self.assertEqual(
                summary["repetitions_sha256"],
                runner.sha256_file(Path(directory) / "nuisance_repetitions.json"),
            )
            self.assertEqual(summary, runner.verify_nuisance_artifacts(directory))
            with self.assertRaises(RuntimeError):
                runner.run_nuisance(directory)

    def test_unexpected_failure_is_invalid_and_tampered_summary_cannot_unlock(self):
        with tempfile.TemporaryDirectory() as directory:
            runner.prepare(directory)
            with patch.object(runner, "run_synthetic_repetition",
                              side_effect=KeyError("implementation bug")):
                summary = runner.run_nuisance(directory)
            self.assertEqual(summary["verdict"], "INVALID_IMPLEMENTATION")
            path = Path(directory) / "A5_NUISANCE_COMPLETE.json"
            tampered = json.loads(path.read_text())
            tampered["gate_pass"] = True
            path.write_text(json.dumps(tampered))
            with self.assertRaises(RuntimeError):
                runner.verify_nuisance_artifacts(directory)
            with self.assertRaises(RuntimeError):
                runner.run_remaining(directory)

    def test_complete_continuation_schedule_and_failure_summary_are_frozen(self):
        schedule = runner.remaining_schedule()
        self.assertEqual(len(schedule), 1700)
        self.assertEqual(len(set(schedule)), 1700)
        records = [{
            "world": world, "repetition": repetition,
            "duplicate_variant": variant, "deletion_count": deletion,
            "seed": runner.sealed_world_seed(world, repetition),
            "usable": False, "failure": "synthetic-test",
        } for world, repetition, variant, deletion in schedule]
        duplicates = [{
            "variant": variant, "repetition": repetition,
            "seed": runner.sealed_world_seed(6, repetition),
            "usable": False, "failure": "synthetic-test",
        } for variant in ("exact", "near") for repetition in range(100)]
        summary = runner.summarize_remaining(records, duplicates)
        self.assertFalse(summary["gate_pass"])
        self.assertEqual(summary["verdict"], "CLOSE_NUMERICAL_NONCONVERGENCE")
        records[0]["seed"] = 999
        with self.assertRaises(ValueError):
            runner.summarize_remaining(records, duplicates)
        records[0]["seed"] = runner.sealed_world_seed(
            records[0]["world"], records[0]["repetition"]
        )
        duplicates[0]["seed"] = 999
        with self.assertRaises(ValueError):
            runner.summarize_remaining(records, duplicates)


if __name__ == "__main__":
    unittest.main()
