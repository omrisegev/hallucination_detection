#!/usr/bin/env python3
"""Fail-closed tests for pre-fit source attribution and resume behavior."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.reconstruction_benchmark.run_24cell_methods import (
    _fit_source_snapshot_record,
    _freeze_or_verify_prefit_snapshot,
)
from spectral_utils.reconstruction_benchmark.fit_validation import (
    validate_static_sources,
)
from spectral_utils.dufs_liu_feature_contract import CONTRACT_VERSION


def source_snapshot(tag: str = "a") -> dict:
    return {
        "schema_version": "reconstruction-source-snapshot-v1",
        "git_head": tag,
        "git_status_sha256": "1" * 64,
        "git_status_clean": True,
        "files": [],
        "snapshot_sha256": ("2" if tag == "a" else "3") * 64,
    }


def expected_record(*, build_id: str = "A", source_tag: str = "a") -> dict:
    return _fit_source_snapshot_record(
        release_id="synthetic-release",
        build_id=build_id,
        input_manifest={"manifest_payload_sha256": "4" * 64},
        source_snapshot=source_snapshot(source_tag),
        method_ids=("iu_pcr",),
        cell_ids=("cell-1",),
    )


class FitRunnerProvenanceTest(unittest.TestCase):
    def test_orientation_registry_hash_mismatch_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            paths = {
                "input_artifact": "cells.npz",
                "input_manifest": "manifest.csv",
                "transform_source": "transform.py",
                "orientation_source": "orientation.py",
                "roster_source": "roster.py",
            }
            for relative in paths.values():
                (repo / relative).write_text("frozen\n")
            import hashlib

            config = {
                "schema_version": "reconstruction-feature-contract-v1",
                "contract_id": CONTRACT_VERSION,
            }
            for key, relative in paths.items():
                config[key] = relative
                hash_key = "input_sha256" if key == "input_artifact" else f"{key}_sha256"
                config[hash_key] = hashlib.sha256(b"frozen\n").hexdigest()
            config["orientation_source_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "orientation_source"):
                validate_static_sources(repo, config)

    def test_new_run_writes_snapshot_before_any_fit_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fit"
            expected = expected_record()
            observed = _freeze_or_verify_prefit_snapshot(
                fit_root=root, expected=expected, resume=False
            )
            self.assertEqual(observed, expected)
            self.assertEqual(
                json.loads((root / "FIT_SOURCE_SNAPSHOT.json").read_text()),
                expected,
            )

    def test_resume_refuses_source_or_roster_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fit"
            _freeze_or_verify_prefit_snapshot(
                fit_root=root, expected=expected_record(), resume=False
            )
            with self.assertRaisesRegex(RuntimeError, "differs"):
                _freeze_or_verify_prefit_snapshot(
                    fit_root=root,
                    expected=expected_record(source_tag="b"),
                    resume=True,
                )

    def test_resume_refuses_legacy_outputs_without_prefit_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fit"
            root.mkdir(parents=True)
            (root / "legacy-record.json").write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "requires FIT_SOURCE_SNAPSHOT"):
                _freeze_or_verify_prefit_snapshot(
                    fit_root=root, expected=expected_record(), resume=True
                )

    def test_frozen_release_can_never_be_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fit"
            _freeze_or_verify_prefit_snapshot(
                fit_root=root, expected=expected_record(), resume=False
            )
            (root / "SCORE_FREEZE_MANIFEST.json").write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "already scientifically frozen"):
                _freeze_or_verify_prefit_snapshot(
                    fit_root=root, expected=expected_record(), resume=True
                )


if __name__ == "__main__":
    unittest.main()
