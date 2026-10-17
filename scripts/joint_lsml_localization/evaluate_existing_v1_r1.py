#!/usr/bin/env python3
"""Versioned PRMBench evaluator amendment for the canonical error-only join."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json, load_npz_no_pickle, sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import payload_sha256  # noqa: E402


RESULT_ROOT = REPO / "results/joint_lsml_existing_localization_v1"
ORIGINAL_EVALUATOR = REPO / "scripts/joint_lsml_localization/evaluate_existing_v1.py"
SELF = Path(__file__).resolve()
TEST = REPO / "tests/test_joint_lsml_evaluation_amendment_r1.py"
PROTOCOL = REPO / "docs/experiments/JOINT_LSML_EXISTING_LOCALIZATION_EVALUATION_AMENDMENT_R1.md"
REGISTRY = RESULT_ROOT / "EVALUATION_AMENDMENT_R1_REGISTRY.json"
AUDIT = RESULT_ROOT / "INDEPENDENT_EVALUATION_AMENDMENT_R1_AUDIT.json"
EVALUATION_ROOT = RESULT_ROOT / "evaluation_r1"
SUMMARY = EVALUATION_ROOT / "EVALUATION_SUMMARY.json"
DEFAULT_RELEASE = Path(
    "/Users/osegev/Desktop/hallucination_detection/.worktrees/"
    "reconstruction-science-run-v1/results/reconstruction_benchmark_v1/"
    "releases/2026-08-24_localization_v1"
)
EXPECTED_SCORE_RESPONSES = 6966
EXPECTED_LABEL_RESPONSES = 6208
EXPECTED_SCORE_ONLY_RESPONSES = 758
EXPECTED_SCORE_SPANS = 94112
EXPECTED_LABEL_STEPS = 83280
LABEL_SHA256 = "7911225fcb4092a4cfebb6ba981ed9b0cfc8ed0a653c6800107945e771362b5a"


class AmendmentError(RuntimeError):
    pass


def _base_module():
    spec = importlib.util.spec_from_file_location("joint_existing_evaluator_v0", ORIGINAL_EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:  # pragma: no cover
        raise AmendmentError("cannot load registered v0 evaluator")
    spec.loader.exec_module(module)
    return module


def validate_prm_subset_join(
    scores: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    *,
    expected_score_responses: int = EXPECTED_SCORE_RESPONSES,
    expected_label_responses: int = EXPECTED_LABEL_RESPONSES,
    expected_score_only_responses: int = EXPECTED_SCORE_ONLY_RESPONSES,
    expected_score_spans: int = EXPECTED_SCORE_SPANS,
    expected_label_steps: int = EXPECTED_LABEL_STEPS,
) -> dict[str, Any]:
    score_ids = tuple(np.asarray(scores["row_ids"]).astype(str))
    label_ids = tuple(np.asarray(labels["response_row_ids"]).astype(str))
    score_index = {row_id: index for index, row_id in enumerate(score_ids)}
    if len(score_ids) != expected_score_responses or len(score_index) != len(score_ids):
        raise AmendmentError("PRMBench score response roster/count mismatch")
    if len(label_ids) != expected_label_responses or len(set(label_ids)) != len(label_ids):
        raise AmendmentError("PRMBench label response roster/count mismatch")
    if not set(label_ids) <= set(score_ids):
        raise AmendmentError("PRMBench label IDs are not a subset of frozen score IDs")
    score_only = set(score_ids) - set(label_ids)
    if len(score_only) != expected_score_only_responses:
        raise AmendmentError("PRMBench score-only response count mismatch")
    selected_indices = np.asarray([score_index[row_id] for row_id in label_ids], dtype=np.int64)
    if np.any(np.diff(selected_indices) <= 0):
        raise AmendmentError("PRMBench label roster is not the frozen score-roster subsequence")

    score_offsets = np.asarray(scores["segment_offsets"], dtype=np.int64)
    label_offsets = np.asarray(labels["step_offsets"], dtype=np.int64)
    if (
        score_offsets.shape != (len(score_ids) + 1,)
        or score_offsets[0] != 0
        or score_offsets[-1] != expected_score_spans
        or np.any(np.diff(score_offsets) <= 0)
    ):
        raise AmendmentError("PRMBench frozen score spans are incomplete or malformed")
    if (
        label_offsets.shape != (len(label_ids) + 1,)
        or label_offsets[0] != 0
        or label_offsets[-1] != expected_label_steps
        or label_offsets[-1] != len(labels["step_labels"])
        or np.any(np.diff(label_offsets) <= 0)
    ):
        raise AmendmentError("PRMBench official label steps are incomplete or malformed")
    selected_counts = np.diff(score_offsets)[selected_indices]
    label_counts = np.diff(label_offsets)
    if not np.array_equal(selected_counts, label_counts):
        raise AmendmentError("PRMBench response-level score/label step counts differ")
    return {
        "score_index": score_index,
        "score_ids": score_ids,
        "label_ids": label_ids,
        "selected_indices": selected_indices,
        "n_score_responses": len(score_ids),
        "n_label_responses": len(label_ids),
        "n_score_only_responses": len(score_only),
        "n_score_spans": int(score_offsets[-1]),
        "n_label_steps": int(label_offsets[-1]),
    }


def _prm_groups_r1(base: Any, by_cell: Mapping[str, Mapping[str, Any]], release: Path):
    if base.PRM_CELL not in by_cell:
        raise AmendmentError("PRMBench panel is not score-frozen")
    scores = load_npz_no_pickle(base.SCORE_ROOT / by_cell[base.PRM_CELL]["artifact_path"])
    labels = load_npz_no_pickle(release / "build_A/localization/evaluation/prmbench_steps.npz")
    joined = validate_prm_subset_join(scores, labels)
    score_index = joined["score_index"]
    label_ids = joined["label_ids"]
    label_offsets = np.asarray(labels["step_offsets"], dtype=np.int64)
    groups: dict[str, dict[str, Any]] = {}
    strata: dict[str, str] = {}
    for label_row, (row_id, group_id, family) in enumerate(zip(
        label_ids, labels["group_ids"].astype(str), labels["error_families"].astype(str),
    )):
        score_row = score_index[row_id]
        slo, shi = map(int, scores["segment_offsets"][score_row:score_row + 2])
        llo, lhi = map(int, label_offsets[label_row:label_row + 2])
        payload = groups.setdefault(
            group_id, {"family": family, "labels": [], "scores": {method: [] for method in base.METHODS}},
        )
        if payload["family"] != family:
            raise AmendmentError("PRMBench source group spans error families")
        payload["labels"].extend(np.asarray(labels["step_labels"][llo:lhi], dtype=np.int64).tolist())
        for method_index, method in enumerate(base.METHODS):
            payload["scores"][method].extend(
                np.asarray(scores["step_risk"][slo:shi, method_index], dtype=float).tolist()
            )
        strata[group_id] = family
    return groups, strata


def _source_hashes() -> dict[str, str]:
    paths = (
        "scripts/joint_lsml_localization/evaluate_existing_v1.py",
        "scripts/joint_lsml_localization/evaluate_existing_v1_r1.py",
        "tests/test_joint_lsml_evaluation_amendment_r1.py",
        "docs/experiments/JOINT_LSML_EXISTING_LOCALIZATION_EVALUATION_AMENDMENT_R1.md",
        "configs/reconstruction_benchmark_v1/localization.json",
        "spectral_utils/reconstruction_benchmark/localization_postfreeze.py",
        "scripts/reasoning_localization/run_phase1_baseline.py",
        "scripts/reasoning_localization/run_h3_prmbench_diagnostic.py",
    )
    return {path: sha256_file(REPO / path) for path in paths}


def _prior_hashes() -> dict[str, str]:
    names = (
        "EXECUTION_REGISTRY.json", "STRUCTURAL_LEDGER.json", "SCORE_FREEZE_MANIFEST.json",
        "INDEPENDENT_SCORE_FREEZE_AUDIT.json",
    )
    return {name: sha256_file(RESULT_ROOT / name) for name in names}


def register(release: Path) -> None:
    if REGISTRY.exists() or EVALUATION_ROOT.exists():
        raise AmendmentError("R1 evaluation registry/namespace already exists")
    base = _base_module()
    manifest, by_cell = base._verified_score_manifest()
    if manifest["processbench_panel_status"] != "STRUCTURAL_NO_SCORE":
        raise AmendmentError("R1 amendment is PRMBench-only")
    label_path = release / "build_A/localization/evaluation/prmbench_steps.npz"
    if sha256_file(label_path) != LABEL_SHA256:
        raise AmendmentError("canonical PRMBench label hash changed")
    scores = load_npz_no_pickle(base.SCORE_ROOT / by_cell[base.PRM_CELL]["artifact_path"])
    labels = load_npz_no_pickle(label_path)
    joined = validate_prm_subset_join(scores, labels)
    original_registry = json.loads((RESULT_ROOT / "EXECUTION_REGISTRY.json").read_text())
    original_expected = original_registry["source_hashes"][
        "scripts/joint_lsml_localization/evaluate_existing_v1.py"
    ]
    if sha256_file(ORIGINAL_EVALUATOR) != original_expected:
        raise AmendmentError("registered v0 evaluator was modified")
    payload = {
        "schema": "joint-lsml-existing-evaluation-amendment-registry-r1",
        "status": "REGISTERED_AFTER_V0_PREMETRIC_JOIN_FAILURE",
        "scope": "RETROSPECTIVE_OPENED_DEVELOPMENT_PRMBENCH_ONLY",
        "reason": "v0 required full-score/evaluator ID equality instead of the canonical error-only subset join",
        "v0_failure": {
            "exception": "EvaluationError: PRMBench score/label response coverage mismatch",
            "metrics_computed": False,
            "evaluation_artifact_created": False,
            "labels_opened": True,
        },
        "method_scores_refit": False,
        "score_freeze_changed": False,
        "processbench_labels_allowed": False,
        "processbench_status": "STRUCTURAL_NO_SCORE",
        "prmbench_label_sha256": LABEL_SHA256,
        "join_contract": {key: value for key, value in joined.items() if key not in {
            "score_index", "score_ids", "label_ids", "selected_indices",
        }},
        "source_hashes": _source_hashes(),
        "prior_artifact_hashes": _prior_hashes(),
        "independent_amendment_audit_required": True,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    atomic_write_json(REGISTRY, payload)


def _verified_registry(release: Path) -> dict[str, Any]:
    if not REGISTRY.exists() or not AUDIT.exists():
        raise AmendmentError("registered and independently audited R1 amendment is required")
    registry = json.loads(REGISTRY.read_text())
    body = {key: value for key, value in registry.items() if key != "payload_sha256"}
    if payload_sha256(body) != registry.get("payload_sha256"):
        raise AmendmentError("R1 amendment registry payload hash mismatch")
    for path, expected in registry["source_hashes"].items():
        if sha256_file(REPO / path) != expected:
            raise AmendmentError(f"R1 amendment source changed: {path}")
    for name, expected in registry["prior_artifact_hashes"].items():
        if sha256_file(RESULT_ROOT / name) != expected:
            raise AmendmentError(f"R1 prior artifact changed: {name}")
    label_path = release / "build_A/localization/evaluation/prmbench_steps.npz"
    if sha256_file(label_path) != registry["prmbench_label_sha256"]:
        raise AmendmentError("R1 PRMBench label source changed")
    audit = json.loads(AUDIT.read_text())
    audit_body = {key: value for key, value in audit.items() if key != "payload_sha256"}
    if (
        audit.get("status") != "PASS"
        or audit.get("amendment_registry_sha256") != sha256_file(REGISTRY)
        or payload_sha256(audit_body) != audit.get("payload_sha256")
    ):
        raise AmendmentError("R1 amendment independent audit is absent or stale")
    return registry


def evaluate(release: Path) -> None:
    registry = _verified_registry(release)
    base = _base_module()
    base.EVALUATION_ROOT = EVALUATION_ROOT
    base.SUMMARY = SUMMARY
    base._prm_groups = lambda by_cell, root: _prm_groups_r1(base, by_cell, root)
    base.evaluate(release)
    output = json.loads(SUMMARY.read_text())
    output["schema"] = "joint-lsml-existing-localization-evaluation-r1"
    output["evaluation_amendment_registry_sha256"] = sha256_file(REGISTRY)
    output["evaluation_amendment_audit_sha256"] = sha256_file(AUDIT)
    output["join_contract"] = registry["join_contract"]
    output["v0_metrics_computed"] = False
    output["payload_sha256"] = payload_sha256({k: v for k, v in output.items() if k != "payload_sha256"})
    atomic_write_json(SUMMARY, output)


def check(release: Path) -> None:
    _verified_registry(release)
    if SUMMARY.exists():
        output = json.loads(SUMMARY.read_text())
        body = {key: value for key, value in output.items() if key != "payload_sha256"}
        if payload_sha256(body) != output.get("payload_sha256"):
            raise AmendmentError("R1 evaluation summary payload mismatch")
        if output.get("ProcessBench", {}).get("status") != "STRUCTURAL_NO_SCORE":
            raise AmendmentError("R1 unexpectedly evaluated ProcessBench")
        if output.get("PRMBench", {}).get("status") != "COMPLETE":
            raise AmendmentError("R1 PRMBench evaluation is incomplete")
    print("PASS")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    release = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_RELEASE
    if command == "register":
        register(release)
    elif command == "evaluate":
        evaluate(release)
    elif command == "check":
        check(release)
    else:
        raise SystemExit("usage: evaluate_existing_v1_r1.py [register|evaluate|check] [release]")

