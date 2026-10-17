#!/usr/bin/env python3
"""Fail-closed final manifest for the Joint L-SML existing-data experiment."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from spectral_utils.reconstruction_benchmark.localization_contract import payload_sha256  # noqa: E402


ROOT = REPO / "results/joint_lsml_existing_localization_v1"
FINAL = ROOT / "FINAL_COMPLETE.json"
CANVAS = Path(
    "/Users/osegev/.cursor/projects/1788418645651/canvases/"
    "joint-lsml-localization-results.canvas.tsx"
)


def _checked_payload(path: Path, expected_status: str | None = None) -> dict:
    data = json.loads(path.read_text())
    body = {key: value for key, value in data.items() if key != "payload_sha256"}
    if payload_sha256(body) != data.get("payload_sha256"):
        raise RuntimeError(f"payload mismatch: {path}")
    if expected_status is not None and data.get("status") != expected_status:
        raise RuntimeError(f"status mismatch: {path}")
    return data


def main() -> None:
    if FINAL.exists():
        raise RuntimeError("FINAL_COMPLETE.json already exists")
    files = {
        "execution_registry": ROOT / "EXECUTION_REGISTRY.json",
        "structural_ledger": ROOT / "STRUCTURAL_LEDGER.json",
        "score_manifest": ROOT / "SCORE_FREEZE_MANIFEST.json",
        "score_freeze_audit": ROOT / "INDEPENDENT_SCORE_FREEZE_AUDIT.json",
        "evaluation_r1_registry": ROOT / "EVALUATION_AMENDMENT_R1_REGISTRY.json",
        "evaluation_r1_audit": ROOT / "INDEPENDENT_EVALUATION_AMENDMENT_R1_AUDIT.json",
        "evaluation_r2_registry": ROOT / "EVALUATION_AMENDMENT_R2_REGISTRY.json",
        "evaluation_r2_audit": ROOT / "INDEPENDENT_EVALUATION_AMENDMENT_R2_AUDIT.json",
        "evaluation_summary": ROOT / "evaluation_r2/EVALUATION_SUMMARY.json",
        "evaluation_result_audit": ROOT / "INDEPENDENT_EVALUATION_RESULT_AUDIT.json",
        "presentation_manifest": ROOT / "presentation/PRESENTATION_MANIFEST.json",
        "report": ROOT / "REPORT.md",
        "progress": REPO / "PROGRESS.md",
        "history": REPO / "HISTORY.md",
        "canvas": CANVAS,
    }
    for path in files.values():
        if not path.is_file():
            raise RuntimeError(f"missing final artifact: {path}")
    structural = _checked_payload(files["structural_ledger"], "COMPLETE")
    result = _checked_payload(files["evaluation_summary"])
    audit = _checked_payload(files["evaluation_result_audit"], "PASS")
    presentation = _checked_payload(files["presentation_manifest"], "COMPLETE")
    if audit["evaluation_summary_sha256"] != sha256_file(files["evaluation_summary"]):
        raise RuntimeError("result audit is stale")
    if presentation["evaluation_summary_sha256"] != sha256_file(files["evaluation_summary"]):
        raise RuntimeError("presentation is stale")
    for artifact in presentation["artifacts"]:
        path = ROOT / artifact["path"]
        if sha256_file(path) != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
            raise RuntimeError(f"presentation artifact mismatch: {path}")
    if structural["processbench_panel_status"] != "STRUCTURAL_NO_SCORE":
        raise RuntimeError("ProcessBench must remain structural no-score")
    if result["ProcessBench"]["status"] != "STRUCTURAL_NO_SCORE":
        raise RuntimeError("ProcessBench evaluation unexpectedly exists")
    if result["PRMBench"]["decision_state"] != "HARM":
        raise RuntimeError("registered PRMBench decision state changed")
    payload = {
        "schema": "joint-lsml-existing-localization-final-complete-v1",
        "status": "COMPLETE_RETROSPECTIVE_HARM_NO_PROMOTION",
        "scope": "RETROSPECTIVE_OPENED_DEVELOPMENT",
        "processbench_status": "STRUCTURAL_NO_SCORE",
        "prmbench_status": "HARM",
        "promotion_allowed": False,
        "generalization_run_recommended": False,
        "artifacts": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in files.items()},
    }
    payload["payload_sha256"] = payload_sha256(payload)
    atomic_write_json(FINAL, payload)


if __name__ == "__main__":
    main()

