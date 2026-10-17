#!/usr/bin/env python3
"""Open existing outcomes only after the Joint L-SML score-freeze audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.fair_comparisons.evaluator import (  # noqa: E402
    crossfit_localization_threshold, detection_metrics, paired_grouped_bootstrap,
)
from spectral_utils.fair_comparisons.folds import assign_group_folds  # noqa: E402
from spectral_utils.joint_lsml_localization import (  # noqa: E402
    EQUAL_FAMILY_METHOD, FIXED_FAMILY_METHOD, IU_METHOD, JOINT_METHOD, METHODS,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json, atomic_write_npz, load_npz_no_pickle, sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import payload_sha256  # noqa: E402


RESULT_ROOT = REPO / "results/joint_lsml_existing_localization_v1"
REGISTRY = RESULT_ROOT / "EXECUTION_REGISTRY.json"
STRUCTURAL_LEDGER = RESULT_ROOT / "STRUCTURAL_LEDGER.json"
SCORE_ROOT = RESULT_ROOT / "score_freeze"
SCORE_MANIFEST = RESULT_ROOT / "SCORE_FREEZE_MANIFEST.json"
AUDIT = RESULT_ROOT / "INDEPENDENT_SCORE_FREEZE_AUDIT.json"
EVALUATION_ROOT = RESULT_ROOT / "evaluation"
SUMMARY = EVALUATION_ROOT / "EVALUATION_SUMMARY.json"
PLAN = REPO / "configs/joint_lsml_existing_localization_v1.json"
DEFAULT_RELEASE = Path(
    "/Users/osegev/Desktop/hallucination_detection/.worktrees/"
    "reconstruction-science-run-v1/results/reconstruction_benchmark_v1/"
    "releases/2026-08-24_localization_v1"
)
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
MODELS = ("qwen3_4b", "qwen3_8b")
PB_CELLS = tuple(f"processbench_{subset}_{model}" for model in MODELS for subset in SUBSETS)
PRM_CELL = "prmbench_response_qwen3_8b"


class EvaluationError(RuntimeError):
    pass


def _verified_score_manifest() -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    if not SCORE_MANIFEST.exists() or not AUDIT.exists() or not REGISTRY.exists():
        raise EvaluationError("score freeze and independent audit are required")
    registry = json.loads(REGISTRY.read_text())
    registry_body = {key: value for key, value in registry.items() if key != "payload_sha256"}
    if payload_sha256(registry_body) != registry.get("payload_sha256"):
        raise EvaluationError("execution registry payload hash mismatch")
    for relative, expected in registry.get("source_hashes", {}).items():
        if sha256_file(REPO / relative) != expected:
            raise EvaluationError(f"registered source changed: {relative}")
    if registry.get("analysis_plan_sha256") != sha256_file(PLAN):
        raise EvaluationError("analysis plan changed after registration")
    manifest = json.loads(SCORE_MANIFEST.read_text())
    body = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if payload_sha256(body) != manifest.get("payload_sha256"):
        raise EvaluationError("score manifest payload hash mismatch")
    if manifest.get("registry_sha256") != sha256_file(REGISTRY):
        raise EvaluationError("score manifest does not bind the current registry")
    if manifest.get("structural_ledger_sha256") != sha256_file(STRUCTURAL_LEDGER):
        raise EvaluationError("score manifest does not bind the structural ledger")
    structural = json.loads(STRUCTURAL_LEDGER.read_text())
    structural_body = {key: value for key, value in structural.items() if key != "payload_sha256"}
    if structural.get("status") != "COMPLETE" or payload_sha256(structural_body) != structural.get("payload_sha256"):
        raise EvaluationError("structural ledger payload/status mismatch")
    if manifest.get("labels_accessed") is not False or structural.get("labels_accessed") is not False:
        raise EvaluationError("pre-label artifact claims target access")
    audit = json.loads(AUDIT.read_text())
    audit_body = {key: value for key, value in audit.items() if key != "payload_sha256"}
    if (
        audit.get("status") != "PASS"
        or audit.get("labels_accessed") is not False
        or payload_sha256(audit_body) != audit.get("payload_sha256")
        or audit.get("score_manifest_sha256") != sha256_file(SCORE_MANIFEST)
    ):
        raise EvaluationError("independent score-freeze audit is absent or stale")
    by_cell = {row["cell_id"]: row for row in manifest["cells"]}
    for row in manifest["cells"]:
        path = SCORE_ROOT / row["artifact_path"]
        if sha256_file(path) != row["artifact_sha256"]:
            raise EvaluationError("score artifact changed after freeze")
        arrays = load_npz_no_pickle(path)
        if tuple(arrays["method_ids"].astype(str)) != METHODS:
            raise EvaluationError("score method order changed")
    return manifest, by_cell


def _pb_labels(release: Path) -> dict[str, dict[str, tuple[str, int]]]:
    path = release / "build_A/localization/evaluation/localization_decisions.csv"
    labels = {cell: {} for cell in PB_CELLS}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cell = row["cell_id"]
            if cell not in labels or row["system_id"] != "deem_b3__loc_geomean_v1":
                continue
            row_id = row["row_id"]
            if row_id in labels[cell]:
                raise EvaluationError("duplicate ProcessBench label")
            labels[cell][row_id] = (row["group_id"], int(row["true_first_error"]))
    return labels


def _pb_rows(by_cell: Mapping[str, Mapping[str, Any]], release: Path) -> list[dict[str, Any]]:
    labels = _pb_labels(release)
    rows = []
    for model in MODELS:
        for subset in SUBSETS:
            cell = f"processbench_{subset}_{model}"
            if cell not in by_cell:
                raise EvaluationError("ProcessBench panel is not fully score-frozen")
            arrays = load_npz_no_pickle(SCORE_ROOT / by_cell[cell]["artifact_path"])
            row_ids = tuple(arrays["row_ids"].astype(str))
            if set(row_ids) != set(labels[cell]) or len(row_ids) != len(labels[cell]):
                raise EvaluationError("ProcessBench score/label coverage mismatch")
            detector = np.asarray(arrays["detector_scores"], dtype=np.float64)
            locators = np.asarray(arrays["locators"], dtype=np.int64)
            for row_index, row_id in enumerate(row_ids):
                group_id, first_error = labels[cell][row_id]
                source_key = f"{subset}::{group_id}"
                for method_index, method in enumerate(METHODS):
                    rows.append({
                        "source_key": source_key, "group_id": source_key,
                        "source_group_id": group_id, "family": subset, "subset": subset,
                        "model_id": model, "cell_id": cell, "row_id": row_id,
                        "stratify_label": int(first_error != -1), "first_error": int(first_error),
                        "method_id": method,
                        "step_scores": [float(detector[row_index, method_index])],
                        "step_indices": [int(locators[row_index, method_index])],
                    })
    fold_rows = [row for row in rows if row["method_id"] == JOINT_METHOD and row["model_id"] == MODELS[0]]
    folds = assign_group_folds(
        fold_rows, n_folds=5, group_key="source_key", family_key="family",
        stratum_key="stratify_label", namespace="joint-lsml-existing-localization-v1",
    )
    for row in rows:
        row["fold"] = int(folds[row["source_key"]])
    _assert_pb_pairing(rows)
    return rows


def _assert_pb_pairing(rows: Sequence[Mapping[str, Any]]) -> None:
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source_key"]), []).append(row)
    expected = {(model, method) for model in MODELS for method in METHODS}
    for source_key, payload in by_source.items():
        observed = {(str(row["model_id"]), str(row["method_id"])) for row in payload}
        if len(payload) != len(expected) or observed != expected:
            raise EvaluationError(f"incomplete paired ProcessBench payload: {source_key}")
        for model in MODELS:
            selected = [row for row in payload if row["model_id"] == model]
            invariant = {
                (str(row["row_id"]), str(row["subset"]), str(row["source_group_id"]),
                 int(row["first_error"]), int(row["stratify_label"]), int(row["fold"]))
                for row in selected
            }
            if len(invariant) != 1:
                raise EvaluationError(f"method-specific ProcessBench identity/label drift: {source_key}/{model}")
        # These are two teacher-forced scorer copies of the same ProcessBench
        # response. The canonical post-freeze contract requires target identity
        # across scorer models; opaque row IDs remain cell-scoped.
        if len({
            (row["subset"], row["source_group_id"], row["first_error"], row["stratify_label"], row["fold"])
            for row in payload
        }) != 1:
            raise EvaluationError(f"model-copy ProcessBench target/pairing drift: {source_key}")


def _pb_fit(sample_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = {}
    ledgers = {}
    for method in METHODS:
        per_model = []
        for model in MODELS:
            selected = [row for row in sample_rows if row["method_id"] == method and row["model_id"] == model]
            result = crossfit_localization_threshold(selected, expected_subsets=SUBSETS)
            per_model.append(float(result["official_oof_metrics"]["equal_subset_macro_f1"]))
            ledgers[f"{method}::{model}"] = result["calibration_ledgers"]
        metrics[method] = float(np.mean(per_model))
    return {"metrics": metrics, "calibration_ledgers": ledgers}


def _pb_statistic(_sample: list[Any], fit: Mapping[str, Any]) -> dict[str, float]:
    values = {f"macro_f1::{method}": float(fit["metrics"][method]) for method in METHODS}
    for control in (IU_METHOD, EQUAL_FAMILY_METHOD, FIXED_FAMILY_METHOD):
        values[f"delta_joint_vs::{control}"] = float(fit["metrics"][JOINT_METHOD] - fit["metrics"][control])
    return values


def _evaluate_pb(by_cell: Mapping[str, Mapping[str, Any]], release: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    rows = _pb_rows(by_cell, release)
    groups: dict[str, list[Mapping[str, Any]]] = {}
    strata = {}
    for row in rows:
        groups.setdefault(row["source_key"], []).append(row)
        strata[row["source_key"]] = f"{row['subset']}::fold{row['fold']}"

    def recompute(payloads: list[Any]) -> dict[str, Any]:
        return _pb_fit([row for payload in payloads for row in payload])

    result = paired_grouped_bootstrap(
        groups, _pb_statistic, strata=strata, recompute=recompute,
        n_boot=int(plan["processbench"]["bootstrap_draws"]),
        seed=int(plan["processbench"]["bootstrap_seed"]), alpha=0.05,
    )
    point_fit = _pb_fit(rows)
    return {
        "status": "COMPLETE", "scope": "RETROSPECTIVE_OPENED_DEVELOPMENT",
        "point_metrics": point_fit["metrics"],
        "point_calibration_ledgers": point_fit["calibration_ledgers"],
        "paired_bootstrap": result,
        "n_source_questions": len(groups), "n_model_rows": len(rows) // len(METHODS),
    }


def _prm_groups(by_cell: Mapping[str, Mapping[str, Any]], release: Path) -> tuple[dict[str, Any], dict[str, str]]:
    if PRM_CELL not in by_cell:
        raise EvaluationError("PRMBench panel is not score-frozen")
    scores = load_npz_no_pickle(SCORE_ROOT / by_cell[PRM_CELL]["artifact_path"])
    labels = load_npz_no_pickle(release / "build_A/localization/evaluation/prmbench_steps.npz")
    score_ids = tuple(scores["row_ids"].astype(str))
    score_index = {row_id: index for index, row_id in enumerate(score_ids)}
    label_ids = tuple(labels["response_row_ids"].astype(str))
    if len(score_index) != len(score_ids) or len(set(label_ids)) != len(label_ids) or set(score_ids) != set(label_ids):
        raise EvaluationError("PRMBench score/label response coverage mismatch")
    label_offsets = np.asarray(labels["step_offsets"], dtype=np.int64)
    if (
        label_offsets.shape != (len(label_ids) + 1,)
        or label_offsets[0] != 0
        or label_offsets[-1] != len(labels["step_labels"])
        or np.any(np.diff(label_offsets) <= 0)
    ):
        raise EvaluationError("PRMBench label step offsets are incomplete or malformed")
    groups: dict[str, dict[str, Any]] = {}
    strata: dict[str, str] = {}
    for label_row, (row_id, group_id, family) in enumerate(zip(
        label_ids, labels["group_ids"].astype(str), labels["error_families"].astype(str),
    )):
        score_row = score_index[row_id]
        slo, shi = map(int, scores["segment_offsets"][score_row:score_row + 2])
        llo, lhi = map(int, labels["step_offsets"][label_row:label_row + 2])
        if shi - slo != lhi - llo:
            raise EvaluationError("PRMBench step coverage mismatch")
        payload = groups.setdefault(group_id, {"family": family, "labels": [], "scores": {method: [] for method in METHODS}})
        if payload["family"] != family:
            raise EvaluationError("PRMBench source group spans error families")
        payload["labels"].extend(np.asarray(labels["step_labels"][llo:lhi], dtype=np.int64).tolist())
        for method_index, method in enumerate(METHODS):
            payload["scores"][method].extend(np.asarray(scores["step_risk"][slo:shi, method_index], dtype=float).tolist())
        strata[group_id] = family
    return groups, strata


def _prm_statistic(payloads: list[Any], _fit: Any) -> dict[str, float]:
    labels = np.asarray([value for payload in payloads for value in payload["labels"]], dtype=np.int64)
    output = {}
    for method in METHODS:
        scores = np.asarray([value for payload in payloads for value in payload["scores"][method]], dtype=np.float64)
        metrics = detection_metrics(labels, scores)
        output[f"auroc::{method}"] = float(metrics["auroc"])
        output[f"auprc::{method}"] = float(metrics["error_auprc"])
        output[f"normalized_ap::{method}"] = float(metrics["prevalence_normalized_ap"])
    for control in (IU_METHOD, EQUAL_FAMILY_METHOD, FIXED_FAMILY_METHOD):
        output[f"delta_auroc_joint_vs::{control}"] = output[f"auroc::{JOINT_METHOD}"] - output[f"auroc::{control}"]
    return output


def _evaluate_prm(by_cell: Mapping[str, Mapping[str, Any]], release: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    groups, strata = _prm_groups(by_cell, release)
    result = paired_grouped_bootstrap(
        groups, _prm_statistic, strata=strata,
        n_boot=int(plan["prmbench"]["bootstrap_draws"]),
        seed=int(plan["prmbench"]["bootstrap_seed"]), alpha=0.05,
    )
    point = _prm_statistic(list(groups.values()), None)
    per_family = {}
    for family in sorted(set(strata.values())):
        selected = [groups[key] for key, value in strata.items() if value == family]
        per_family[family] = _prm_statistic(selected, None)
    return {
        "status": "COMPLETE", "scope": "RETROSPECTIVE_OPENED_DEVELOPMENT",
        "point_metrics": point, "per_family": per_family,
        "paired_bootstrap": result, "n_source_groups": len(groups),
        "n_steps": sum(len(payload["labels"]) for payload in groups.values()),
    }


def _panel_state(result: Mapping[str, Any], *, panel: str) -> str:
    prefix = "delta_joint_vs::" if panel == "ProcessBench" else "delta_auroc_joint_vs::"
    stats = result["paired_bootstrap"]["statistics"]
    iu = stats[prefix + IU_METHOD]
    if float(iu["ci_high"]) < 0.0:
        return "HARM"
    no_control_harm = all(
        float(stats[prefix + control]["ci_high"]) >= 0.0
        for control in (EQUAL_FAMILY_METHOD, FIXED_FAMILY_METHOD)
    )
    if float(iu["point"]) > 0.0 and no_control_harm:
        return "DEVELOPMENT_SUPPORTED"
    return "INCONCLUSIVE"


def evaluate(release: Path) -> None:
    if EVALUATION_ROOT.exists():
        raise EvaluationError("evaluation namespace already exists")
    manifest, by_cell = _verified_score_manifest()
    plan = json.loads(PLAN.read_text())
    output: dict[str, Any] = {
        "schema": "joint-lsml-existing-localization-evaluation-v1",
        "scope": "RETROSPECTIVE_OPENED_DEVELOPMENT",
        "promotion_allowed": False, "generalization_claim_allowed": False,
        "score_manifest_sha256": sha256_file(SCORE_MANIFEST),
        "independent_audit_sha256": sha256_file(AUDIT),
        "analysis_plan_sha256": sha256_file(PLAN),
        "label_sources": {},
    }
    if manifest["processbench_panel_status"] == "SCORES_FROZEN":
        path = release / "build_A/localization/evaluation/localization_decisions.csv"
        output["label_sources"]["ProcessBench"] = {"path": str(path), "sha256": sha256_file(path)}
        output["ProcessBench"] = _evaluate_pb(by_cell, release, plan)
        output["ProcessBench"]["decision_state"] = _panel_state(output["ProcessBench"], panel="ProcessBench")
    else:
        output["ProcessBench"] = {"status": "STRUCTURAL_NO_SCORE", "decision_state": "STRUCTURAL_NO_SCORE"}
    if manifest["prmbench_panel_status"] == "SCORES_FROZEN":
        path = release / "build_A/localization/evaluation/prmbench_steps.npz"
        output["label_sources"]["PRMBench"] = {"path": str(path), "sha256": sha256_file(path)}
        output["PRMBench"] = _evaluate_prm(by_cell, release, plan)
        output["PRMBench"]["decision_state"] = _panel_state(output["PRMBench"], panel="PRMBench")
    else:
        output["PRMBench"] = {"status": "STRUCTURAL_NO_SCORE", "decision_state": "STRUCTURAL_NO_SCORE"}
    output["fresh_generalization_recommended"] = bool(
        output["ProcessBench"]["decision_state"] == "DEVELOPMENT_SUPPORTED"
        and output["PRMBench"]["decision_state"] == "DEVELOPMENT_SUPPORTED"
    )
    output["labels_accessed_only_after_score_freeze_audit"] = True
    output["payload_sha256"] = payload_sha256(output)
    atomic_write_json(SUMMARY, output)


if __name__ == "__main__":
    release = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RELEASE
    evaluate(release)
