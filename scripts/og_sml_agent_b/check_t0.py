#!/usr/bin/env python3
"""Verify the frozen OG-SML Agent B T0 artifacts and label firewall."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/og_sml_agent_b_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    registry = json.loads((RESULTS / "T0_EXECUTION_REGISTRY.json").read_text())
    for relative, expected in registry["source_hashes"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"source hash mismatch for {relative}: {actual}")

    manifest = json.loads((RESULTS / "T0_MANIFEST.json").read_text())
    for name, expected in manifest["artifacts"].items():
        actual = sha256_file(RESULTS / name)
        if actual != expected:
            raise RuntimeError(f"artifact hash mismatch for {name}: {actual}")
    renderer = manifest["renderer"]
    if sha256_file(Path(renderer["source"])) != renderer["source_sha256"]:
        raise RuntimeError("renderer source hash mismatch")
    if sha256_file(RESULTS / "T0_REPORT.json") != renderer["input_report_sha256"]:
        raise RuntimeError("renderer input-report hash mismatch")

    report = json.loads((RESULTS / "T0_REPORT.json").read_text())
    if report["terminal_status"] != "T0_FALSIFIED_STOP_BEFORE_STEPS_0_6":
        raise RuntimeError("unexpected T0 terminal status")
    for key in ("labels_seen", "targets_loaded", "outcome_metrics_computed", "fused_score_arrays_created"):
        if report[key] is not False:
            raise RuntimeError(f"firewall mismatch: {key}")
    expected_cross_tab = {
        "prior_pass_and_admissible": 0,
        "prior_pass_and_inadmissible": 3,
        "prior_fail_and_admissible": 6,
        "prior_fail_and_inadmissible": 9,
    }
    if report["cross_tab"] != expected_cross_tab:
        raise RuntimeError("unexpected T0 cross-tab")
    closed_form = json.loads((RESULTS / "T0_PARTITION_CLOSED_FORM_AUDIT.json").read_text())
    if closed_form["cross_tab"] != expected_cross_tab or closed_form["status"] != "PASS":
        raise RuntimeError("closed-form partition audit mismatch")

    print("OG-SML Agent B T0 check: PASS")
    print(f"terminal_status={report['terminal_status']}")
    print("labels_seen=false targets_loaded=false outcome_metrics_computed=false fused_score_arrays_created=false")


if __name__ == "__main__":
    main()
