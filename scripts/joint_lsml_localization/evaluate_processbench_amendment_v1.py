#!/usr/bin/env python3
"""Registered ProcessBench-only evaluation for the Joint-or-flat amendment."""

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

from scripts.joint_lsml_localization import run_processbench_amendment_v1 as freeze  # noqa: E402
from spectral_utils.fair_comparisons.evaluator import (  # noqa: E402
    crossfit_localization_threshold,
    paired_grouped_bootstrap,
)
from spectral_utils.fair_comparisons.folds import assign_group_folds  # noqa: E402
from spectral_utils.joint_lsml_processbench_amendment import (  # noqa: E402
    COVERAGE_METHOD,
    COVERAGE_METHODS,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    load_npz_no_pickle,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import payload_sha256  # noqa: E402


PLAN = freeze.PLAN
RESULT_ROOT = freeze.RESULT_ROOT
AUDIT = RESULT_ROOT / "INDEPENDENT_SCORE_FREEZE_AUDIT.json"
EVALUATION_REGISTRY = RESULT_ROOT / "EVALUATION_REGISTRY.json"
EVALUATION_REGISTRATION_COMPLETE = RESULT_ROOT / "EVALUATION_REGISTRATION_COMPLETE.json"
EVALUATION_REGISTRY_AUDIT = RESULT_ROOT / "INDEPENDENT_EVALUATION_REGISTRY_AUDIT.json"
EVALUATION_ROOT = RESULT_ROOT / "evaluation"
SUMMARY = EVALUATION_ROOT / "EVALUATION_SUMMARY.json"
DEFAULT_RELEASE = Path(
    "/Users/osegev/Desktop/hallucination_detection/.worktrees/"
    "reconstruction-science-run-v1/results/reconstruction_benchmark_v1/"
    "releases/2026-08-24_localization_v1"
)
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
MODELS = ("qwen3_4b", "qwen3_8b")
FALLBACK_CELL = "processbench_math_qwen3_4b"


class ProcessBenchEvaluationError(RuntimeError):
    pass


def _json_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if "payload_sha256" in payload and payload_sha256(body) != payload["payload_sha256"]:
        raise ProcessBenchEvaluationError(f"noncanonical payload: {path}")
    return payload


def _evaluation_source_paths() -> tuple[str, ...]:
    return (
        "configs/joint_lsml_processbench_amendment_v1.json",
        "docs/experiments/JOINT_LSML_PROCESSBENCH_AMENDMENT_V1.md",
        "docs/experiments/PRIOR_ORDER_AUDIT_JOINT_LSML_PROCESSBENCH_AMENDMENT_V1.md",
        "scripts/joint_lsml_localization/evaluate_processbench_amendment_v1.py",
        "scripts/joint_lsml_localization/run_processbench_amendment_v1.py",
        "spectral_utils/fair_comparisons/evaluator.py",
        "spectral_utils/fair_comparisons/folds.py",
        "spectral_utils/paper_exact/evaluator.py",
        "spectral_utils/reconstruction_benchmark/io.py",
        "spectral_utils/reconstruction_benchmark/localization_contract.py",
        "tests/test_joint_lsml_processbench_evaluation.py",
    )


def _evaluation_source_hashes() -> dict[str, str]:
    return {relative: sha256_file(REPO / relative) for relative in _evaluation_source_paths()}


def _verified_score_freeze() -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    freeze.check()
    manifest = _json_payload(freeze.SCORE_MANIFEST)
    audit = _json_payload(AUDIT)
    required_audit = {
        "schema": "joint-lsml-processbench-amendment-independent-score-freeze-audit-v1",
        "status": "PASS",
        "score_manifest_sha256": sha256_file(freeze.SCORE_MANIFEST),
        "registry_sha256": sha256_file(freeze.REGISTRY),
        "policy_ledger_sha256": sha256_file(freeze.POLICY_LEDGER),
        "all_eight_replayed": True,
        "max_abs_detector_replay_error": 0.0,
        "locator_mismatch_count": 0,
        "processbench_labels_accessed": False,
    }
    observed = {key: audit.get(key) for key in required_audit}
    if observed != required_audit:
        raise ProcessBenchEvaluationError(f"independent score-freeze audit is absent or stale: {observed}")
    if manifest.get("processbench_labels_accessed") is not False:
        raise ProcessBenchEvaluationError("score freeze claims ProcessBench label access")
    by_cell = {row["cell_id"]: row for row in manifest["cells"]}
    if tuple(by_cell) != freeze.parent.PB_CELLS:
        raise ProcessBenchEvaluationError("score freeze is not exact ordered all-eight ProcessBench")
    return manifest, by_cell


def register_evaluation(release: Path) -> None:
    if EVALUATION_REGISTRY.exists() or EVALUATION_ROOT.exists():
        raise ProcessBenchEvaluationError("evaluation namespace already exists")
    manifest, _ = _verified_score_freeze()
    plan = json.loads(PLAN.read_text())
    label_path = release / "build_A/localization/evaluation/localization_decisions.csv"
    payload = {
        "schema": "joint-lsml-processbench-amendment-evaluation-registry-v1",
        "status": "REGISTERED_BEFORE_PROCESSBENCH_LABEL_PARSE",
        "scope": plan["scope"],
        "methods": list(COVERAGE_METHODS),
        "candidate": COVERAGE_METHOD,
        "score_manifest_sha256": sha256_file(freeze.SCORE_MANIFEST),
        "score_audit_sha256": sha256_file(AUDIT),
        "freeze_registry_sha256": sha256_file(freeze.REGISTRY),
        "analysis_plan_sha256": sha256_file(PLAN),
        "source_hashes": _evaluation_source_hashes(),
        "processbench_label_path": str(label_path.resolve()),
        "processbench_label_sha256": sha256_file(label_path),
        "processbench_cells": list(freeze.parent.PB_CELLS),
        "folds": int(plan["processbench"]["folds"]),
        "fold_namespace": plan["processbench"]["fold_namespace"],
        "bootstrap_draws": int(plan["processbench"]["bootstrap_draws"]),
        "bootstrap_seed": int(plan["processbench"]["bootstrap_seed"]),
        "bootstrap_strata": plan["processbench"]["bootstrap_strata"],
        "primary_metric": plan["processbench"]["primary_metric"],
        "decision_state": plan["reporting"]["decision_state"],
        "processbench_labels_parsed": False,
        "prmbench_labels_accessed_by_this_evaluator": False,
    }
    if manifest.get("fallback_cells") != [FALLBACK_CELL]:
        raise ProcessBenchEvaluationError("score freeze fallback cell drift")
    payload["payload_sha256"] = payload_sha256(payload)
    registry_hash = atomic_write_json(EVALUATION_REGISTRY, payload)
    atomic_write_json(EVALUATION_REGISTRATION_COMPLETE, {
        "status": "PASS",
        "evaluation_registry_sha256": registry_hash,
        "processbench_labels_parsed": False,
    })


def _verified_evaluation_registry(
    release: Path, *, require_independent_audit: bool = True
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    if not EVALUATION_REGISTRY.exists() or not EVALUATION_REGISTRATION_COMPLETE.exists():
        raise ProcessBenchEvaluationError("evaluation is not registered")
    registry = _json_payload(EVALUATION_REGISTRY)
    completion = json.loads(EVALUATION_REGISTRATION_COMPLETE.read_text())
    if completion.get("evaluation_registry_sha256") != sha256_file(EVALUATION_REGISTRY):
        raise ProcessBenchEvaluationError("evaluation registration completion mismatch")
    if registry.get("source_hashes") != _evaluation_source_hashes():
        raise ProcessBenchEvaluationError("registered evaluator source changed")
    if registry.get("analysis_plan_sha256") != sha256_file(PLAN):
        raise ProcessBenchEvaluationError("analysis plan changed after registration")
    _, by_cell = _verified_score_freeze()
    if registry.get("score_manifest_sha256") != sha256_file(freeze.SCORE_MANIFEST):
        raise ProcessBenchEvaluationError("registered score freeze changed")
    if registry.get("score_audit_sha256") != sha256_file(AUDIT):
        raise ProcessBenchEvaluationError("registered score audit changed")
    label_path = release / "build_A/localization/evaluation/localization_decisions.csv"
    if str(label_path.resolve()) != registry.get("processbench_label_path"):
        raise ProcessBenchEvaluationError("ProcessBench label path changed")
    if sha256_file(label_path) != registry.get("processbench_label_sha256"):
        raise ProcessBenchEvaluationError("ProcessBench label source changed")
    if require_independent_audit:
        if not EVALUATION_REGISTRY_AUDIT.exists():
            raise ProcessBenchEvaluationError("independent evaluation-registry audit is required")
        audit = _json_payload(EVALUATION_REGISTRY_AUDIT)
        required = {
            "schema": "joint-lsml-processbench-amendment-independent-evaluation-registry-audit-v1",
            "status": "PASS",
            "evaluation_registry_sha256": sha256_file(EVALUATION_REGISTRY),
            "score_manifest_sha256": sha256_file(freeze.SCORE_MANIFEST),
            "processbench_label_sha256": registry["processbench_label_sha256"],
            "processbench_labels_parsed": False,
            "prmbench_labels_accessed": False,
        }
        if {key: audit.get(key) for key in required} != required:
            raise ProcessBenchEvaluationError("independent evaluation-registry audit is absent or stale")
    return registry, by_cell


def _load_pb_labels(path: Path) -> dict[str, dict[str, tuple[str, int]]]:
    labels = {cell: {} for cell in freeze.parent.PB_CELLS}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cell = row["cell_id"]
            if cell not in labels or row["system_id"] != "deem_b3__loc_geomean_v1":
                continue
            row_id = row["row_id"]
            if row_id in labels[cell]:
                raise ProcessBenchEvaluationError("duplicate ProcessBench label")
            labels[cell][row_id] = (row["group_id"], int(row["true_first_error"]))
    return labels


def _assert_pairing(rows: Sequence[Mapping[str, Any]]) -> None:
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source_key"]), []).append(row)
    expected = {(model, method) for model in MODELS for method in COVERAGE_METHODS}
    for source_key, payload in by_source.items():
        observed = {(str(row["model_id"]), str(row["method_id"])) for row in payload}
        if len(payload) != len(expected) or observed != expected:
            raise ProcessBenchEvaluationError(f"incomplete paired payload: {source_key}")
        if len({
            (row["subset"], row["source_group_id"], row["first_error"], row["stratify_label"], row["fold"])
            for row in payload
        }) != 1:
            raise ProcessBenchEvaluationError(f"q4/q8 target or fold drift: {source_key}")
        for model in MODELS:
            selected = [row for row in payload if row["model_id"] == model]
            if len({
                (row["row_id"], row["first_error"], row["fold"])
                for row in selected
            }) != 1:
                raise ProcessBenchEvaluationError(f"method identity drift: {source_key}/{model}")


def _pb_rows(by_cell: Mapping[str, Mapping[str, Any]], label_path: Path, namespace: str) -> list[dict[str, Any]]:
    labels = _load_pb_labels(label_path)
    rows = []
    for model in MODELS:
        for subset in SUBSETS:
            cell = f"processbench_{subset}_{model}"
            arrays = load_npz_no_pickle(freeze.SCORE_ROOT / by_cell[cell]["artifact_path"])
            row_ids = tuple(arrays["row_ids"].astype(str))
            if len(set(row_ids)) != len(row_ids) or set(row_ids) != set(labels[cell]):
                raise ProcessBenchEvaluationError(f"score/label coverage mismatch: {cell}")
            for row_index, row_id in enumerate(row_ids):
                group_id, first_error = labels[cell][row_id]
                source_key = f"{subset}::{group_id}"
                for method_index, method in enumerate(COVERAGE_METHODS):
                    rows.append({
                        "source_key": source_key,
                        "group_id": source_key,
                        "source_group_id": group_id,
                        "family": subset,
                        "subset": subset,
                        "model_id": model,
                        "cell_id": cell,
                        "row_id": row_id,
                        "stratify_label": int(first_error != -1),
                        "first_error": int(first_error),
                        "method_id": method,
                        "step_scores": [float(arrays["detector_scores"][row_index, method_index])],
                        "step_indices": [int(arrays["locators"][row_index, method_index])],
                    })
    fold_rows = [
        row for row in rows
        if row["method_id"] == COVERAGE_METHOD and row["model_id"] == MODELS[0]
    ]
    folds = assign_group_folds(
        fold_rows,
        n_folds=5,
        group_key="source_key",
        family_key="family",
        stratum_key="stratify_label",
        namespace=namespace,
    )
    for row in rows:
        row["fold"] = int(folds[row["source_key"]])
    _assert_pairing(rows)
    return rows


def _fit(sample_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = {}
    ledgers = {}
    per_cell = {}
    for method in COVERAGE_METHODS:
        model_macros = []
        for model in MODELS:
            selected = [row for row in sample_rows if row["method_id"] == method and row["model_id"] == model]
            result = crossfit_localization_threshold(selected, expected_subsets=SUBSETS)
            model_macros.append(float(result["official_oof_metrics"]["equal_subset_macro_f1"]))
            ledgers[f"{method}::{model}"] = result["calibration_ledgers"]
            for subset in SUBSETS:
                cell = f"processbench_{subset}_{model}"
                per_cell[f"{method}::{cell}"] = float(result["official_oof_metrics"]["per_subset"][subset]["f1"])
        metrics[method] = float(np.mean(model_macros))
    fitted7 = _fitted7_diagnostic(per_cell)
    return {
        "metrics": metrics,
        "calibration_ledgers": ledgers,
        "per_cell_oof_f1": per_cell,
        "fitted7_selection_conditioned_equal_cell_mean_f1": fitted7,
    }


def _fitted7_diagnostic(per_cell: Mapping[str, float]) -> dict[str, float]:
    """Selection-conditioned view of primary OOF predictions; no threshold refit."""
    fitted_cells = [cell for cell in freeze.parent.PB_CELLS if cell != FALLBACK_CELL]
    return {
        method: float(np.mean([per_cell[f"{method}::{cell}"] for cell in fitted_cells]))
        for method in COVERAGE_METHODS
    }


def _statistic(_sample: list[Any], fitted: Mapping[str, Any]) -> dict[str, float]:
    metrics = fitted["metrics"]
    output = {f"macro_f1::{method}": float(metrics[method]) for method in COVERAGE_METHODS}
    for control in COVERAGE_METHODS[1:]:
        output[f"delta_candidate_vs::{control}"] = float(metrics[COVERAGE_METHOD] - metrics[control])
    return output


def _decision_state(statistics: Mapping[str, Mapping[str, Any]]) -> str:
    contrasts = [statistics[f"delta_candidate_vs::{control}"] for control in COVERAGE_METHODS[1:]]
    iu = contrasts[0]
    if float(iu["ci_high"]) < 0.0:
        return "HARM"
    if float(iu["point"]) > 0.0 and all(float(row["ci_low"]) >= 0.0 for row in contrasts):
        return "DEVELOPMENT_SUPPORTED"
    return "INCONCLUSIVE"


def evaluate(release: Path) -> None:
    if EVALUATION_ROOT.exists():
        raise ProcessBenchEvaluationError("evaluation result namespace already exists")
    registry, by_cell = _verified_evaluation_registry(release)
    rows = _pb_rows(by_cell, Path(registry["processbench_label_path"]), registry["fold_namespace"])
    groups: dict[str, list[Mapping[str, Any]]] = {}
    strata = {}
    for row in rows:
        groups.setdefault(row["source_key"], []).append(row)
        strata[row["source_key"]] = f"{row['subset']}::fold{row['fold']}"

    def recompute(payloads: list[Any]) -> dict[str, Any]:
        return _fit([row for payload in payloads for row in payload])

    point = _fit(rows)
    bootstrap = paired_grouped_bootstrap(
        groups,
        _statistic,
        strata=strata,
        recompute=recompute,
        n_boot=registry["bootstrap_draws"],
        seed=registry["bootstrap_seed"],
        alpha=0.05,
    )
    output = {
        "schema": "joint-lsml-processbench-amendment-evaluation-v1",
        "status": "COMPLETE",
        "scope": registry["scope"],
        "candidate": COVERAGE_METHOD,
        "candidate_is_pure_joint_on_all_cells": False,
        "fallback_cells": [FALLBACK_CELL],
        "point_metrics": point["metrics"],
        "point_calibration_ledgers": point["calibration_ledgers"],
        "per_cell_oof_f1": point["per_cell_oof_f1"],
        "fitted7_selection_conditioned_equal_cell_mean_f1": point["fitted7_selection_conditioned_equal_cell_mean_f1"],
        "fitted7_thresholds_refit": False,
        "fitted7_complete_panel_efficacy": False,
        "fitted7_fallback_independent": False,
        "paired_bootstrap": bootstrap,
        "decision_state": _decision_state(bootstrap["statistics"]),
        "n_source_questions": len(groups),
        "n_model_rows": len(rows) // len(COVERAGE_METHODS),
        "evaluation_registry_sha256": sha256_file(EVALUATION_REGISTRY),
        "score_manifest_sha256": sha256_file(freeze.SCORE_MANIFEST),
        "label_sources": {
            "ProcessBench": {
                "path": registry["processbench_label_path"],
                "sha256": registry["processbench_label_sha256"],
            }
        },
        "processbench_labels_accessed_after_score_freeze_audit": True,
        "prmbench_labels_accessed_by_this_evaluator": False,
        "prmbench_prior_result": "HARM__NOT_REEVALUATED",
        "fresh_generalization_recommended": False,
        "promotion_allowed": False,
        "generalization_claim_allowed": False,
    }
    output["payload_sha256"] = payload_sha256(output)
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=False)
    atomic_write_json(SUMMARY, output)


def check(release: Path) -> None:
    registry, _ = _verified_evaluation_registry(release)
    if not SUMMARY.exists():
        print("PASS_REGISTERED")
        return
    summary = _json_payload(SUMMARY)
    if summary.get("evaluation_registry_sha256") != sha256_file(EVALUATION_REGISTRY):
        raise ProcessBenchEvaluationError("evaluation summary registry mismatch")
    if summary.get("score_manifest_sha256") != sha256_file(freeze.SCORE_MANIFEST):
        raise ProcessBenchEvaluationError("evaluation summary score-freeze mismatch")
    if summary.get("candidate") != COVERAGE_METHOD or summary.get("fallback_cells") != [FALLBACK_CELL]:
        raise ProcessBenchEvaluationError("evaluation candidate/fallback mismatch")
    if summary.get("n_source_questions") != 3400 or summary.get("n_model_rows") != 6800:
        raise ProcessBenchEvaluationError("ProcessBench population count mismatch")
    stats = summary["paired_bootstrap"]["statistics"]
    if any(int(row["n_valid"]) != registry["bootstrap_draws"] for row in stats.values()):
        raise ProcessBenchEvaluationError("incomplete bootstrap interval")
    if summary.get("decision_state") != _decision_state(stats):
        raise ProcessBenchEvaluationError("decision state mismatch")
    print("PASS")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    release = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_RELEASE
    if command == "register":
        register_evaluation(release)
    elif command == "evaluate":
        evaluate(release)
    elif command == "check":
        check(release)
    else:
        raise SystemExit("usage: evaluate_processbench_amendment_v1.py [register|evaluate|check] [release]")
