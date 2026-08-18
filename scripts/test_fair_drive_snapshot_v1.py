#!/usr/bin/env python3
"""Focused tests for the frozen read-only Drive metadata observation."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_utils.fair_comparisons import drive_snapshot as D  # noqa: E402
from spectral_utils.fair_comparisons.registry import canonical_sha256  # noqa: E402


def _rehash(observation: dict) -> None:
    observation["metadata_members_sha256"] = canonical_sha256(
        observation["metadata_members"]
    )
    projection = dict(observation)
    projection.pop("observation_sha256", None)
    observation["observation_sha256"] = canonical_sha256(projection)


class DriveMetadataObservationTests(unittest.TestCase):
    def test_exact_observation_is_self_consistent(self) -> None:
        first = D.build_drive_metadata_observation()
        second = D.build_drive_metadata_observation()
        self.assertEqual(first, second)
        self.assertEqual(D.validate_drive_metadata_observation(first), first)
        self.assertEqual(first["metadata_member_count"], 66)
        self.assertEqual(first["metadata_total_size_bytes"], 694812)
        self.assertEqual(
            set(first["metadata_members"]),
            {
                "l0",
                "l1_uprm",
                "s1_refrain",
                "s2_leash_complete",
                "m2_deepconf_status",
            },
        )
        self.assertEqual(len(first["metadata_members"]["l0"]), 1)
        self.assertEqual(len(first["metadata_members"]["l1_uprm"]), 6)
        self.assertEqual(len(first["metadata_members"]["s1_refrain"]), 5)
        self.assertEqual(len(first["metadata_members"]["s2_leash_complete"]), 30)
        self.assertEqual(len(first["metadata_members"]["m2_deepconf_status"]), 24)
        expected_size = sum(
            row["size_bytes"]
            for rows in first["metadata_members"].values()
            for row in rows
        )
        self.assertEqual(first["metadata_total_size_bytes"], expected_size)
        projection = dict(first)
        projection.pop("observation_sha256")
        self.assertEqual(first["observation_sha256"], canonical_sha256(projection))
        self.assertEqual(
            first["metadata_members"]["l1_uprm"][0],
            {
                "path": "paper_exact/l1_uprm_judge_full/GATE_L1-uprm-judge-full.json",
                "size_bytes": 598,
                "sha256": "d5dc93943fde859bfabe6f9dbdf86bd7576cfdfa60af6f530cf3819824a2c459",
            },
        )
        s2 = first["metadata_members"]["s2_leash_complete"]
        run_counts: dict[str, int] = {}
        for row in s2:
            run_id = row["path"].split("/")[1]
            run_counts[run_id] = run_counts.get(run_id, 0) + 1
        self.assertEqual(len(run_counts), 6)
        self.assertEqual(set(run_counts.values()), {5})
        self.assertEqual(
            s2[-1],
            {
                "path": "paper_exact/s2_leash_Qwen2.5-7B-Instruct_gsm8k/SUMMARY.json",
                "size_bytes": 1186,
                "sha256": "ebfcd87dda090f90f08f7da5e1d22b371e462336126ccdcc2fd02083646d028f",
            },
        )

    def test_exact_member_drift_fails_even_when_attacker_rehashes_ledger(self) -> None:
        tampered = copy.deepcopy(D.build_drive_metadata_observation())
        tampered["metadata_members"]["s1_refrain"][0]["path"] = (
            "paper_exact/s1_refrain_full/ALTERNATE_STATE.json"
        )
        _rehash(tampered)
        with self.assertRaisesRegex(D.DriveObservationError, "exact frozen"):
            D.validate_drive_metadata_observation(tampered)

    def test_member_paths_are_canonical_remote_relative_and_cwd_free(self) -> None:
        before = D.build_drive_metadata_observation()
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path.cwd()
            try:
                os.chdir(temporary)
                after = D.build_drive_metadata_observation()
            finally:
                os.chdir(previous)
        self.assertEqual(before, after)
        serialized = json.dumps(before, sort_keys=True)
        self.assertNotIn(str(REPO_ROOT), serialized)
        for rows in before["metadata_members"].values():
            for row in rows:
                path = row["path"]
                self.assertFalse(PurePosixPath(path).is_absolute())
                self.assertNotIn("..", PurePosixPath(path).parts)
                self.assertFalse(path.startswith("gdrive:"))

        absolute = copy.deepcopy(before)
        absolute["metadata_members"]["l0"][0]["path"] = "/tmp/local/L0_INVENTORY.json"
        _rehash(absolute)
        with self.assertRaisesRegex(D.DriveObservationError, "remote-prefix-relative"):
            D.validate_drive_metadata_observation(absolute)

    def test_summary_or_content_hash_drift_fails_closed(self) -> None:
        l1_summary = copy.deepcopy(D.build_drive_metadata_observation())
        l1_summary["summaries"]["l1_uprm"]["metadata_members"] = 5
        _rehash(l1_summary)
        with self.assertRaisesRegex(D.DriveObservationError, "L1 summary"):
            D.validate_drive_metadata_observation(l1_summary)

        s2_summary = copy.deepcopy(D.build_drive_metadata_observation())
        s2_summary["summaries"]["s2_leash_complete"]["complete_cells"] = 5
        _rehash(s2_summary)
        with self.assertRaisesRegex(D.DriveObservationError, "S2 summary"):
            D.validate_drive_metadata_observation(s2_summary)

        summary = copy.deepcopy(D.build_drive_metadata_observation())
        summary["summaries"]["m2_deepconf"]["status_members"] = 23
        _rehash(summary)
        with self.assertRaisesRegex(D.DriveObservationError, "status_members"):
            D.validate_drive_metadata_observation(summary)

        content_hash = copy.deepcopy(D.build_drive_metadata_observation())
        content_hash["metadata_members_sha256"] = "0" * 64
        projection = dict(content_hash)
        projection.pop("observation_sha256")
        content_hash["observation_sha256"] = canonical_sha256(projection)
        with self.assertRaisesRegex(D.DriveObservationError, "metadata_members_sha256"):
            D.validate_drive_metadata_observation(content_hash)


if __name__ == "__main__":
    unittest.main(verbosity=2)
