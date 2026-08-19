#!/usr/bin/env python3
"""Focused fail-closed tests for independent fair-package rebuild verification."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_utils.fair_comparisons.registry import (  # noqa: E402
    build_hash_manifest,
    write_canonical_json,
)


SPEC = importlib.util.spec_from_file_location(
    "verify_fair_comparison_rebuild_v1",
    REPO_ROOT / "scripts" / "verify_fair_comparison_rebuild_v1.py",
)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def _write_package(root: Path, *, value: str = "same") -> None:
    (root / "lanes" / "global").mkdir(parents=True)
    (root / "REPORT.md").write_text(f"report:{value}\n", encoding="utf-8")
    (root / "lanes" / "global" / "metrics.json").write_text(
        json.dumps({"value": value}, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    write_canonical_json(root / "HASH_MANIFEST.json", build_hash_manifest(root))


class RebuildVerifierTests(unittest.TestCase):
    def test_accepts_identical_independent_trees_path_freely(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first = base / "arbitrary-reference-name"
            second = base / "unrelated-candidate-name"
            _write_package(first)
            _write_package(second)
            result = VERIFIER.verify_rebuild(first, second)
            reversed_result = VERIFIER.verify_rebuild(second, first)
        self.assertEqual(result, reversed_result)
        self.assertTrue(result["byte_identical"])
        self.assertTrue(result["directories_independent"])
        self.assertEqual(result["file_count"], 2)
        self.assertNotIn("reference", result)
        self.assertNotIn("candidate", result)

    def test_rejects_different_but_individually_valid_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first = base / "first"
            second = base / "second"
            _write_package(first, value="one")
            _write_package(second, value="two")
            with self.assertRaisesRegex(RuntimeError, "independent rebuild mismatch"):
                VERIFIER.verify_rebuild(first, second)

    def test_rejects_stale_or_incomplete_scope_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first = base / "first"
            second = base / "second"
            _write_package(first)
            _write_package(second)
            (second / "REPORT.md").write_text("mutated after manifest\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "input package manifest failure"):
                VERIFIER.verify_rebuild(first, second)

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first = base / "first"
            second = base / "second"
            _write_package(first)
            _write_package(second)
            partial = build_hash_manifest(second, include=[second / "REPORT.md"])
            write_canonical_json(second / "HASH_MANIFEST.json", partial)
            with self.assertRaisesRegex(ValueError, "complete result tree"):
                VERIFIER.verify_rebuild(first, second)

    def test_rejects_noncanonical_manifest_or_same_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "package"
            other = Path(temporary) / "other"
            _write_package(root)
            _write_package(other)
            manifest_path = other / "HASH_MANIFEST.json"
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not canonical JSON"):
                VERIFIER.verify_rebuild(root, other)
            with self.assertRaisesRegex(ValueError, "independently written"):
                VERIFIER.verify_rebuild(root, root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
