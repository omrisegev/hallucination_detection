#!/usr/bin/env python3
"""Tests for automatic group-free IU Phase-A0 infrastructure."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from spectral_utils.group_free_research import (
    canonical_feature_names,
    derive_feature_dag,
    factorial_world_diagnostics,
    resolve_local_lfs_object,
    simulate_factorial_world,
)
import spectral_utils.group_free_research as group_free_research


class FeatureDagTests(unittest.TestCase):
    def test_dag_covers_exact_contract_without_manual_group_field(self):
        rows = derive_feature_dag()
        self.assertEqual(len(rows), 30)
        self.assertEqual(tuple(row["feature_name"] for row in rows), canonical_feature_names())
        self.assertTrue(all(row["manual_provenance_registry_used"] is False for row in rows))
        self.assertFalse(any("family" in row or "view" in row for row in rows))

    def test_crossed_axes_are_not_one_hard_group(self):
        rows = derive_feature_dag()
        self.assertGreaterEqual(len({row["source_stream"] for row in rows}), 5)
        self.assertGreaterEqual(len({row["operator"] for row in rows}), 12)
        self.assertTrue(all(row["implementation"]["source_line"] for row in rows))

    def test_module_does_not_import_manual_provenance_registry(self):
        source = inspect.getsource(group_free_research)
        self.assertNotIn("specrage_views", source)
        self.assertNotIn("FEATURE_TO_VIEW", source)


class LfsResolutionTests(unittest.TestCase):
    def test_resolves_and_verifies_local_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"paired telemetry fixture"
            oid = hashlib.sha256(payload).hexdigest()
            obj = root / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
            obj.parent.mkdir(parents=True)
            obj.write_bytes(payload)
            pointer = root / "cache.pkl"
            pointer.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                f"oid sha256:{oid}\nsize {len(payload)}\n",
                encoding="ascii",
            )
            resolved, diagnostics = resolve_local_lfs_object(pointer, root)
            self.assertEqual(resolved, obj)
            self.assertEqual(diagnostics["sha256"], oid)
            self.assertEqual(diagnostics["storage"], "local_git_lfs_object")


class FactorialSimulatorTests(unittest.TestCase):
    def test_reproducible_crossed_world_and_duplicate(self):
        left = simulate_factorial_world(seed=17, n_environments=3, n_samples=80)
        right = simulate_factorial_world(seed=17, n_environments=3, n_samples=80)
        self.assertEqual(left.feature_names, right.feature_names)
        for a, b in zip(left.environments, right.environments):
            np.testing.assert_array_equal(a["matrix"], b["matrix"])
            np.testing.assert_array_equal(a["target"], b["target"])
        diagnostics = factorial_world_diagnostics(left)
        self.assertEqual(diagnostics["feature_count"], 30)
        self.assertEqual(diagnostics["maximum_duplicate_error"], 0.0)
        self.assertGreater(diagnostics["mean_missing_fraction"], 0.0)

    def test_environment_target_drift_is_observable(self):
        stable = factorial_world_diagnostics(simulate_factorial_world(
            seed=23, n_environments=4, n_samples=100,
        ))
        drifting = factorial_world_diagnostics(simulate_factorial_world(
            seed=23, n_environments=4, n_samples=100,
            environment_specific_target=True,
        ))
        self.assertAlmostEqual(stable["minimum_target_loading_cosine"], 1.0)
        self.assertLess(drifting["minimum_target_loading_cosine"], 1.0)


if __name__ == "__main__":
    unittest.main()
