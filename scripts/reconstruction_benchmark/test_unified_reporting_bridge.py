#!/usr/bin/env python3
"""Focused adversarial tests for the certified unified reporting bridge."""

from __future__ import annotations

import csv
import copy
from dataclasses import replace
import hashlib
import importlib.util
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_reporting.unified_reporting_publish import (  # noqa: E402
    build_unified_release,
    verify_completed_release,
    verify_unified_release_ab,
)
import spectral_utils.reconstruction_reporting.unified_reporting_publish as unified_publish  # noqa: E402
import spectral_utils.reconstruction_reporting.unified_reporting_sources as unified_sources  # noqa: E402
from spectral_utils.reconstruction_reporting.unified_reporting_bridge import (  # noqa: E402
    advisor_inputs,
    build_unified_rows,
)
from spectral_utils.reconstruction_reporting.unified_reporting_schemas import (  # noqa: E402
    COMPARISON_SIGNATURE_FIELDS,
    UnifiedReportingError,
    assert_csv_parquet_parity,
    canonical_json_bytes,
    canonical_sha256,
    csv_bytes,
    derive_comparison_group_id,
    parquet_bytes,
    read_csv_bytes,
    validate_row,
)
from spectral_utils.reconstruction_reporting.unified_reporting_sources import (  # noqa: E402
    AuthenticatedSource,
    authenticate_sources,
    load_source_lock,
    parse_contract_bytes,
    read_locked_file,
    validate_contract_source_lock,
)


HAS_ARROW = importlib.util.find_spec("pyarrow") is not None

LEASH_RELEASE_RELATIVE = Path(
    "results/reconstruction_benchmark_v1/releases/2026-08-25_leash_v1/leash"
)
SCIENCE_REPO = next(
    (
        candidate
        for candidate in (REPO.parent / "reconstruction-science-run-v1", REPO)
        if (candidate / LEASH_RELEASE_RELATIVE / "EVALUATION_AB_VERIFICATION.json").is_file()
    ),
    REPO.parent / "reconstruction-science-run-v1",
)
REAL_SOURCE_LOCK_AVAILABLE = (
    (REPO / "results/reconstruction_benchmark_v1/releases/2026-08-24_frozen24_v1"
     / "reporting_inputs/BRIDGE_MANIFEST.json").is_file()
    and (SCIENCE_REPO / LEASH_RELEASE_RELATIVE / "EVALUATION_AB_VERIFICATION.json").is_file()
)

LEASH_LOCKED_FILES = {
    "aggregate_metrics": (
        "aggregate_metrics.csv",
        "6f6c2dceadd64e75136d80fbc6e8a1ddc6a038a4b94e100c9f9a6547e1b2134d",
    ),
    "bootstrap_intervals": (
        "bootstrap_intervals.csv",
        "59b04949b36072a6157e5b8ce403a749f11b8887ad5e2a9ce4ce267c4326707e",
    ),
    "cell_metrics": (
        "cell_metrics.csv",
        "7c51844e722ffef2746c417686912eef6965482caa45486336cb113ffee9d9f6",
    ),
    "contrasts": (
        "contrasts.csv",
        "c9cd49dcbfd4a88202c02a843f007a1a4a8f170dd425e2e3b3dc148fb2f16e51",
    ),
    "coverage": (
        "coverage.csv",
        "9171c217fc25268e4320bd053d3517a429dc5bc0f6598a11a64723fdc82a1438",
    ),
    "frontier": (
        "frontier.csv",
        "a3c97e038d07627c6fd091a180ddc974715ee7762655a914e131b8788af2f583",
    ),
}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _csv_payload(fieldnames: list[str], rows: list[dict[str, object]]) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    source = root / "source"
    source.mkdir()
    metric_fields = [
        "task_id", "dataset_id", "population_id", "cell_id", "slice_id",
        "access_contract_id", "fidelity", "aggregation_level", "aggregation_id",
        "system_id", "method_id", "metric_id", "metric_unit", "positive_class",
        "better_direction", "comparison_group_id", "value", "ci_low", "ci_high",
        "n_rows", "n_groups", "n_positive", "n_negative", "bootstrap_unit",
        "bootstrap_draws", "status", "status_detail", "cohort_id",
    ]
    metrics = _csv_payload(metric_fields, [{
        "task_id": "final_answer_detection", "dataset_id": "fixture_dataset",
        "population_id": "fixture_population", "cell_id": "fixture_cell",
        "slice_id": "fixture_slice", "access_contract_id": "saved_telemetry",
        "fidelity": "fixture_certified", "aggregation_level": "cell",
        "aggregation_id": "fixture_cell_native", "system_id": "system_v1",
        "method_id": "method", "metric_id": "auroc", "metric_unit": "fraction",
        "positive_class": "incorrect final answer", "better_direction": "higher",
        "comparison_group_id": "producer_group", "value": 0.75, "ci_low": 0.6,
        "ci_high": 0.9, "n_rows": 4, "n_groups": 4, "n_positive": 2,
        "n_negative": 2, "bootstrap_unit": "source_group", "bootstrap_draws": 20,
        "status": "OK", "status_detail": "fixture", "cohort_id": "fixture_cohort",
    }])
    contrasts = _csv_payload(["left_system_id", "right_system_id"], [])
    coverage_fields = [
        "task_id", "dataset_id", "population_id", "cell_id", "slice_id",
        "access_contract_id", "system_id", "method_id", "expected_n", "eligible_n",
        "scored_n", "fallback_n", "excluded_n", "failed_n", "coverage_fraction",
        "cohort_id", "status", "status_detail",
    ]
    coverage = _csv_payload(coverage_fields, [{
        "task_id": "final_answer_detection", "dataset_id": "fixture_dataset",
        "population_id": "fixture_population", "cell_id": "fixture_cell",
        "slice_id": "fixture_slice", "access_contract_id": "saved_telemetry",
        "system_id": "system_v1", "method_id": "method", "expected_n": 4,
        "eligible_n": 4, "scored_n": 4, "fallback_n": 0, "excluded_n": 0,
        "failed_n": 0, "coverage_fraction": 1, "cohort_id": "fixture_cohort",
        "status": "OK", "status_detail": "fixture",
    }])
    payloads = {"metrics": metrics, "contrasts": contrasts, "coverage": coverage}
    filenames = {
        "metrics": "metrics_long.csv", "contrasts": "contrasts_long.csv",
        "coverage": "coverage_long.csv",
    }
    for name, payload in payloads.items():
        (source / filenames[name]).write_bytes(payload)
    certificate_body = {
        "schema_version": "reconstruction-24cell-reporting-bridge-v1",
        "scientific_publication_eligible": True,
        "artifacts": [
            {"path": filenames[name], "file_sha256": _sha(payload)}
            for name, payload in payloads.items()
        ],
    }
    certificate = dict(certificate_body)
    certificate["payload_sha256"] = canonical_sha256(certificate_body)
    certificate_payload = canonical_json_bytes(certificate) + b"\n"
    (source / "BRIDGE_MANIFEST.json").write_bytes(certificate_payload)
    source_lock = {
        "schema_version": "reconstruction-unified-reporting-source-lock-v1",
        "sources": [
            {
                "source_id": "frozen24", "source_release_id": "fixture_frozen24",
                "source_root_id": "fixture", "certified": True,
                "certificate": {
                    "path": "BRIDGE_MANIFEST.json", "file_sha256": _sha(certificate_payload),
                    "schema_version": certificate["schema_version"],
                    "self_hash_field": "payload_sha256",
                    "self_hash": certificate["payload_sha256"],
                    "status_field": "scientific_publication_eligible", "status_value": True,
                },
                "files": {
                    name: {
                        "path": filenames[name], "format": "csv",
                        "file_sha256": _sha(payload),
                    }
                    for name, payload in payloads.items()
                },
            },
            {
                "source_id": "rag_evidence", "source_release_id": "NOT_CERTIFIED",
                "certified": False, "status": "NOT_CERTIFIED",
            },
            {
                "source_id": "leash_stopping", "source_release_id": "NOT_CERTIFIED",
                "certified": False, "status": "NOT_CERTIFIED",
            },
        ],
    }
    contract = {
        "schema_version": "reconstruction-unified-reporting-contract-v1",
        "access_partitions": {}, "claim_boundaries": {},
        "lanes": {
            "frozen24": {
                "adapter": "frozen24_v1", "lane_id": "frozen24_response",
                "task_id": "final_answer_detection", "default_prediction_unit": "response",
                "default_estimand_id": "response_error_ranking", "report_partition": "primary",
            },
            "rag_evidence": {
                "adapter": "rag_evidence_v1", "lane_id": "rag_evidence",
                "task_id": "rag_evidence_evaluation",
                "default_prediction_unit": "panel_registered_unit",
                "default_estimand_id": "panel_registered_rag_evidence",
                "bootstrap_draws": 20000,
                "panel_ids": [
                    "ragtruth_evidence_contrast_answer",
                    "ragtruth_evidence_contrast_sentence",
                    "ragtruth_evidence_contrast_token",
                    "gasp_protocol_sentence",
                    "lettucedetect_example",
                    "refchecker_threeway",
                    "refchecker_binary_claim",
                ],
                "refchecker_subgroups": [
                    "accurate_context", "noisy_context", "zero_context",
                ],
                "report_partition": "context",
            },
            "leash_stopping": {
                "adapter": "leash_v1", "lane_id": "leash_actual_stopping",
                "task_id": "adaptive_stopping", "default_prediction_unit": "source_question",
                "default_estimand_id": "accuracy_vs_realized_compute",
                "allowed_arms": ["cot", "leash", "nocot"],
                "reference_arm": "cot", "bootstrap_draws": 2000,
                "report_partition": "context",
            },
        },
    }
    contract_path, lock_path = root / "contract.json", root / "source_lock.json"
    _write_json(contract_path, contract)
    _write_json(lock_path, source_lock)
    return source, contract_path, lock_path


RAG_PANELS = [
    "ragtruth_evidence_contrast_answer",
    "ragtruth_evidence_contrast_sentence",
    "ragtruth_evidence_contrast_token",
    "gasp_protocol_sentence",
    "lettucedetect_example",
    "refchecker_threeway",
    "refchecker_binary_claim",
]


def _typed_contract(kind: str) -> dict[str, object]:
    if kind == "leash":
        source_id = "leash_stopping"
        lane = {
            "adapter": "leash_v1", "lane_id": "leash_actual_stopping",
            "task_id": "adaptive_stopping", "default_prediction_unit": "source_question",
            "default_estimand_id": "accuracy_vs_realized_compute",
            "allowed_arms": ["cot", "leash", "nocot"], "reference_arm": "cot",
            "bootstrap_draws": 2000, "report_partition": "context",
        }
    elif kind == "rag":
        source_id = "rag_evidence"
        lane = {
            "adapter": "rag_evidence_v1", "lane_id": "rag_evidence",
            "task_id": "rag_evidence_evaluation",
            "default_prediction_unit": "panel_registered_unit",
            "default_estimand_id": "panel_registered_rag_evidence",
            "panel_ids": RAG_PANELS,
            "refchecker_subgroups": [
                "accurate_context", "noisy_context", "zero_context",
            ],
            "bootstrap_draws": 20000, "report_partition": "context",
        }
    else:
        raise AssertionError(kind)
    return {
        "schema_version": "reconstruction-unified-reporting-contract-v1",
        "access_partitions": {}, "claim_boundaries": {}, "lanes": {source_id: lane},
    }


def _leash_payloads() -> dict[str, bytes]:
    datasets = {"aqua": 254, "gsm8k": 300}
    models = (
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "microsoft/Phi-3-mini-128k-instruct",
    )
    arms = ("cot", "leash", "nocot")
    metric_names = (
        "early_stop_rate", "forced_closure_rate", "mean_closure_tokens",
        "mean_reasoning_tokens", "mean_total_tokens", "mean_wall_s",
        "parser_failure_rate", "pass_at_1",
    )
    values: dict[tuple[str, str, str], dict[str, float]] = {}
    cell_rows: list[dict[str, object]] = []
    for dataset, n_questions in datasets.items():
        for model_index, model in enumerate(models):
            for arm in arms:
                closure = 10.0 + model_index
                reasoning = {"cot": 320.0, "leash": 160.0, "nocot": 0.0}[arm]
                total = closure + reasoning
                row_values = {
                    "early_stop_rate": 0.5 if arm == "leash" else 0.0,
                    "forced_closure_rate": 0.5 if arm == "leash" else 0.0,
                    "mean_closure_tokens": closure,
                    "mean_reasoning_tokens": reasoning,
                    "mean_total_tokens": total,
                    "mean_wall_s": total / 100.0,
                    "parser_failure_rate": 0.1,
                    "pass_at_1": {"cot": 0.7, "leash": 0.6, "nocot": 0.2}[arm],
                }
                values[(dataset, model, arm)] = row_values
                cell_rows.append({
                    "actual_stopping_claim_eligible": arm == "leash", "arm": arm,
                    "cell_id": f"s2::{dataset}::{model}", "closure_tokens": int(closure * n_questions),
                    "dataset": dataset, "dataset_revision": "test", **row_values,
                    "mean_tokens_per_question": total, "fidelity": "paper-specified-partial",
                    "method_id": f"{arm}|central", "model": model,
                    "n_forced_closure": int(0.5 * n_questions) if arm == "leash" else 0,
                    "n_parser_failures": int(0.1 * n_questions), "n_questions": n_questions,
                    "n_stopped_early": int(0.5 * n_questions) if arm == "leash" else 0,
                    "n_stopped_without_closure": 0, "parser_revision": "fixture_parser_v1",
                    "realized_savings_valid": True, "reasoning_tokens": int(reasoning * n_questions),
                    "schema": "s2_stopping_cell_metric_v1", "total_tokens": int(total * n_questions),
                    "total_wall_s": total * n_questions / 100.0,
                })
    cell_fields = [
        "actual_stopping_claim_eligible", "arm", "cell_id", "closure_tokens", "dataset",
        "dataset_revision", "early_stop_rate", "fidelity", "forced_closure_rate",
        "mean_closure_tokens", "mean_reasoning_tokens", "mean_tokens_per_question",
        "mean_total_tokens", "mean_wall_s", "method_id", "model", "n_forced_closure",
        "n_parser_failures", "n_questions", "n_stopped_early", "n_stopped_without_closure",
        "parser_failure_rate", "parser_revision", "pass_at_1", "realized_savings_valid",
        "reasoning_tokens", "schema", "total_tokens", "total_wall_s",
    ]

    contrast_rows: list[dict[str, object]] = []
    contrast_values: dict[tuple[str, str, str], dict[str, float]] = {}
    for dataset in datasets:
        for model in models:
            cot = values[(dataset, model, "cot")]
            for arm in ("leash", "nocot"):
                current = values[(dataset, model, arm)]
                deltas = {f"{metric}_delta_vs_cot": current[metric] - cot[metric] for metric in metric_names}
                deltas["token_reduction_vs_cot"] = 1.0 - current["mean_total_tokens"] / cot["mean_total_tokens"]
                contrast_values[(dataset, model, arm)] = deltas
                contrast_rows.append({
                    "arm": arm, "cell_id": f"s2::{dataset}::{model}",
                    "contrast_direction": "arm_minus_cot", "dataset": dataset, **deltas,
                    "matched_accuracy_claim": False, "model": model, "reference_arm": "cot",
                })
    contrast_fields = [
        "arm", "cell_id", "contrast_direction", "dataset",
        "early_stop_rate_delta_vs_cot", "forced_closure_rate_delta_vs_cot",
        "matched_accuracy_claim", "mean_closure_tokens_delta_vs_cot",
        "mean_reasoning_tokens_delta_vs_cot", "mean_total_tokens_delta_vs_cot",
        "mean_wall_s_delta_vs_cot", "model", "parser_failure_rate_delta_vs_cot",
        "pass_at_1_delta_vs_cot", "reference_arm", "token_reduction_vs_cot",
    ]

    def mean(rows: list[float]) -> float:
        return sum(rows) / len(rows)

    aggregate_rows: list[dict[str, object]] = []
    aggregate_values: dict[tuple[str, str, str, str], float] = {}
    for scope, dataset_roster in (
        ("equal_model_within_dataset", tuple(datasets)),
        ("equal_dataset_after_equal_model", ("",)),
    ):
        for dataset in dataset_roster:
            for arm in arms:
                for metric in metric_names:
                    if scope == "equal_model_within_dataset":
                        point = mean([values[(dataset, model, arm)][metric] for model in models])
                    else:
                        point = mean([
                            mean([values[(name, model, arm)][metric] for model in models])
                            for name in datasets
                        ])
                    aggregate_values[(scope, dataset, arm, metric)] = point
                    aggregate_rows.append({
                        "arm": arm, "dataset": dataset, "fidelity": "paper-specified-partial",
                        "metric": metric, "scope": scope, "value": point,
                    })

    bootstrap_rows: list[dict[str, object]] = []
    grouping = "source_question_stratified_within_dataset_shared_across_arms_and_models"

    def interval_row(
        *, arm: str, dataset: str, model: str, metric: str, point: float,
        reference: str, scope: str, groups: int,
    ) -> dict[str, object]:
        return {
            "arm": arm, "dataset": dataset, "grouping": grouping,
            "hi": point + 0.01, "lo": point - 0.01, "metric": metric, "model": model,
            "n_boot": 2000, "n_groups": groups, "point": point,
            "reference_arm": reference, "scope": scope, "seed": 2026082406,
        }

    for dataset, groups in datasets.items():
        for model in models:
            for arm in arms:
                for metric in metric_names:
                    bootstrap_rows.append(interval_row(
                        arm=arm, dataset=dataset, model=model, metric=metric,
                        point=values[(dataset, model, arm)][metric], reference="",
                        scope="cell", groups=groups,
                    ))
            for arm in ("leash", "nocot"):
                for metric, point in contrast_values[(dataset, model, arm)].items():
                    bootstrap_rows.append(interval_row(
                        arm=arm, dataset=dataset, model=model, metric=metric,
                        point=point, reference="cot", scope="cell", groups=groups,
                    ))
        for arm in arms:
            for metric in metric_names:
                bootstrap_rows.append(interval_row(
                    arm=arm, dataset=dataset, model="", metric=metric,
                    point=aggregate_values[("equal_model_within_dataset", dataset, arm, metric)],
                    reference="", scope="equal_model_within_dataset", groups=groups,
                ))
        for arm in ("leash", "nocot"):
            for metric in (*[f"{name}_delta_vs_cot" for name in metric_names], "token_reduction_vs_cot"):
                point = mean([contrast_values[(dataset, model, arm)][metric] for model in models])
                bootstrap_rows.append(interval_row(
                    arm=arm, dataset=dataset, model="", metric=metric, point=point,
                    reference="cot", scope="equal_model_within_dataset", groups=groups,
                ))
    total_groups = sum(datasets.values())
    for arm in arms:
        for metric in metric_names:
            bootstrap_rows.append(interval_row(
                arm=arm, dataset="", model="", metric=metric,
                point=aggregate_values[("equal_dataset_after_equal_model", "", arm, metric)],
                reference="", scope="equal_dataset_after_equal_model", groups=total_groups,
            ))
    for arm in ("leash", "nocot"):
        for metric in (*[f"{name}_delta_vs_cot" for name in metric_names], "token_reduction_vs_cot"):
            point = mean([
                mean([contrast_values[(dataset, model, arm)][metric] for model in models])
                for dataset in datasets
            ])
            bootstrap_rows.append(interval_row(
                arm=arm, dataset="", model="", metric=metric, point=point,
                reference="cot", scope="equal_dataset_after_equal_model", groups=total_groups,
            ))
    bootstrap_fields = [
        "arm", "dataset", "grouping", "hi", "lo", "metric", "model", "n_boot",
        "n_groups", "point", "reference_arm", "scope", "seed",
    ]

    frontier_rows: list[dict[str, object]] = []
    for dataset in datasets:
        for model in models:
            for arm in arms:
                point = values[(dataset, model, arm)]
                delta = 0.0 if arm == "cot" else contrast_values[(dataset, model, arm)]["pass_at_1_delta_vs_cot"]
                reduction = 0.0 if arm == "cot" else contrast_values[(dataset, model, arm)]["token_reduction_vs_cot"]
                frontier_rows.append({
                    "accuracy_delta_vs_cot": delta, "arm": arm,
                    "cell_id": f"s2::{dataset}::{model}", "dataset": dataset,
                    "dataset_revision": "test", "dominated_by": "[]",
                    "mean_tokens_per_question": point["mean_total_tokens"],
                    "mean_wall_s": point["mean_wall_s"], "model": model,
                    "pareto_efficient_within_cell": True, "pass_at_1": point["pass_at_1"],
                    "schema": "s2_accuracy_compute_frontier_point_v1",
                    "token_reduction_vs_cot": reduction,
                })
    frontier_fields = [
        "accuracy_delta_vs_cot", "arm", "cell_id", "dataset", "dataset_revision",
        "dominated_by", "mean_tokens_per_question", "mean_wall_s", "model",
        "pareto_efficient_within_cell", "pass_at_1", "schema", "token_reduction_vs_cot",
    ]

    coverage_rows: list[dict[str, object]] = []
    for dataset, questions in datasets.items():
        for model in (*models, "mistralai/Mistral-7B-v0.1"):
            ready = not model.startswith("mistralai/")
            expected = questions * 3
            coverage_rows.append({
                "actual_policy_execution_observed": ready,
                "actual_stopping_claim_eligible": ready,
                "coverage_status": "READY" if ready else "PROTOCOL_GATE_FAILED",
                "dataset": dataset, "fidelity": "paper-specified-partial", "model": model,
                "n_expected": expected, "n_failed": 0 if ready else expected,
                "n_finished": expected if ready else 0,
                "n_leash_policy_stops": questions // 2 if ready else "",
                "n_leash_rows_replayed": questions if ready else "",
                "n_policy_replay_mismatches": 0 if ready else "",
                "reason": "" if ready else "fixture protocol gate",
                "run_id": f"fixture::{dataset}::{model}", "usable_for_evaluation": ready,
            })
    coverage_fields = [
        "actual_policy_execution_observed", "actual_stopping_claim_eligible",
        "coverage_status", "dataset", "fidelity", "model", "n_expected", "n_failed",
        "n_finished", "n_leash_policy_stops", "n_leash_rows_replayed",
        "n_policy_replay_mismatches", "reason", "run_id", "usable_for_evaluation",
    ]
    return {
        "aggregate_metrics": _csv_payload([*aggregate_rows[0]], aggregate_rows),
        "bootstrap_intervals": _csv_payload(bootstrap_fields, bootstrap_rows),
        "cell_metrics": _csv_payload(cell_fields, cell_rows),
        "contrasts": _csv_payload(contrast_fields, contrast_rows),
        "coverage": _csv_payload(coverage_fields, coverage_rows),
        "frontier": _csv_payload(frontier_fields, frontier_rows),
    }


def _rag_payloads() -> tuple[dict[str, bytes], list[dict[str, object]], bytes]:
    contracts = {
        "ragtruth_evidence_contrast_answer": ("RAGTruth", "answer", "teacher_forced_full_noctx_loo_where_available", "response_hallucination_ranking", ("fixed_rag_iu_pcr",), ("auroc", "auprc"), ("dev", "test"), ("all", "QA")),
        "ragtruth_evidence_contrast_sentence": ("RAGTruth", "sentence", "teacher_forced_full_noctx_loo_where_available", "sentence_hallucination_ranking", ("fixed_rag_iu_pcr",), ("auroc", "auprc"), ("dev", "test"), ("all", "QA")),
        "ragtruth_evidence_contrast_token": ("RAGTruth", "scorer_token", "teacher_forced_full_noctx_loo_where_available", "token_overlap_hallucination_ranking", ("fixed_rag_iu_pcr",), ("auroc", "auprc"), ("dev", "test"), ("all", "QA")),
        "gasp_protocol_sentence": ("RAGTruth balanced GASP cohort", "sentence", "teacher_forced_full_noctx_loo_exact_full_vocab_jsd", "sentence_hallucination_ranking_on_local_protocol_sample", ("gasp_threshold", "fixed_rag_iu_pcr_matched"), ("auroc", "auprc"), ("local_400_response_sample",), ("all", "QA")),
        "lettucedetect_example": ("RAGTruth", "example", "supervised_ragtruth_token_classifier", "any_predicted_span_vs_any_gold_span", ("lettucedetect_large_modernbert",), ("f1", "precision", "recall"), ("test",), ("all", "QA")),
        "refchecker_threeway": ("KnowHalBench fixed claims", "fixed_claim", "supervised_nli_checker", "three_way_claim_checking", ("refchecker_nli",), ("accuracy", "macro_f1"), ("official_fixed_claims",), ("accurate_context", "noisy_context", "zero_context")),
        "refchecker_binary_claim": ("KnowHalBench fixed claims", "fixed_claim", "teacher_forced_full_noctx", "unsupported_claim_ranking_binary_collapse", ("fixed_rag_iu_pcr_transfer",), ("auroc", "auprc"), ("official_fixed_claims",), ("accurate_context", "noisy_context", "zero_context")),
    }
    metric_rows: list[dict[str, object]] = []
    for panel_id, (dataset, unit, access, estimand, methods, metrics, splits, subgroups) in contracts.items():
        for split in splits:
            for subgroup in subgroups:
                for method in methods:
                    for metric in metrics:
                        metric_rows.append({
                            "panel_id": panel_id, "dataset": dataset, "unit": unit,
                            "access": access, "estimand": estimand, "split": split,
                            "subgroup": subgroup, "method_id": method, "metric": metric,
                            "value": 0.7, "ci_low": 0.6, "ci_high": 0.8, "n": 20,
                            "n_groups": 10, "positive_rate": "" if panel_id == "refchecker_threeway" else 0.5,
                            "bootstrap_draws": 20000, "status": "OK",
                        })
    metric_fields = [
        "panel_id", "dataset", "unit", "access", "estimand", "split", "subgroup",
        "method_id", "metric", "value", "ci_low", "ci_high", "n", "n_groups",
        "positive_rate", "bootstrap_draws", "status",
    ]
    contrast_rows = [
        {
            "panel_id": "gasp_protocol_sentence", "split": "local_400_response_sample",
            "subgroup": subgroup, "left_method": "gasp_threshold",
            "right_method": "fixed_rag_iu_pcr_matched", "metric": metric,
            "delta": 0.05, "ci_low": 0.01, "ci_high": 0.09, "n": 20,
            "n_groups": 10, "bootstrap_draws": 20000, "status": "OK",
        }
        for subgroup in ("all", "QA") for metric in ("auroc", "auprc")
    ]
    contrast_fields = [
        "panel_id", "split", "subgroup", "left_method", "right_method", "metric",
        "delta", "ci_low", "ci_high", "n", "n_groups", "bootstrap_draws", "status",
    ]
    status_rows: list[dict[str, object]] = []
    for panel_id in RAG_PANELS:
        status_rows.append({
            "panel_id": panel_id, "status": "PASS",
            "metric_rows": sum(row["panel_id"] == panel_id for row in metric_rows),
            "prediction_rows": 20, "cross_panel_macro_contribution": "FORBIDDEN",
        })
    status_fields = [
        "panel_id", "status", "metric_rows", "prediction_rows",
        "cross_panel_macro_contribution",
    ]
    predictions = b"fixture predictions are certificate-bound but reporting-private\n"
    return {
        "metrics": _csv_payload(metric_fields, metric_rows),
        "contrasts": _csv_payload(contrast_fields, contrast_rows),
        "panel_status": _csv_payload(status_fields, status_rows),
    }, status_rows, predictions


def _certified_lane_fixture(
    root: Path, kind: str,
) -> tuple[Path, dict[str, object], dict[str, object], AuthenticatedSource]:
    source_root = root / f"{kind}_source"
    source_root.mkdir()
    contract = _typed_contract(kind)
    if kind == "leash":
        payloads = _leash_payloads()
        filenames = {name: f"{name}.csv" for name in payloads}
        for name, payload in payloads.items():
            (source_root / filenames[name]).write_bytes(payload)
        row_counts = {
            name: len(list(csv.DictReader(StringIO(payload.decode("utf-8")))))
            for name, payload in payloads.items()
        }
        row_counts["per_question"] = 4986
        tables = {
            name: {"files": {filenames[name]: _sha(payload)}, "row_count": row_counts[name]}
            for name, payload in payloads.items()
        }
        tables["per_question"] = {
            "files": {"per_question.csv": "9" * 64}, "row_count": row_counts["per_question"],
        }
        manifest_body = {
            "schema_version": "reconstruction-leash-stopping-evaluation-v1",
            "lane_id": "leash_actual_stopping_v1", "policy_execution_evaluated": True,
            "all_policy_stops_have_realized_closure": True,
            "claim_scope": "six ready cells only", "claim_status": "ACTUAL_POLICY_EXECUTION_EVALUATED_FOR_SIX_READY_CELLS",
            "fidelity": "paper-specified-partial", "paper_exact_claim": False,
            "conceptual_objective_reproduced_as_equation": False,
            "matched_accuracy_claim": False, "cross_task_or_access_macro": False,
            "proxy_stopping": False, "registry_sha256": "1" * 64,
            "bootstrap": {
                "draws": 2000, "paired_across_arms_and_model_copies": True,
                "seed": 2026082406, "stratification": "within dataset",
                "unit": "source question",
            },
            "preparation_ab_certificate_sha256": "2" * 64,
            "fit_ab_certificate_sha256": "3" * 64, "tables": tables,
        }
        manifest = dict(manifest_body); manifest["payload_sha256"] = canonical_sha256(manifest_body)
        manifest_payload = canonical_json_bytes(manifest) + b"\n"
        certificate_body = {
            "schema_version": "reconstruction-leash-stopping-evaluation-ab-v1",
            "status": "PASS", "lane_id": "leash_actual_stopping_v1",
            "registry_sha256": "1" * 64, "preparation_ab_certificate_sha256": "2" * 64,
            "fit_ab_certificate_sha256": "3" * 64,
            "evaluation_tree_sha256": {"A": "4" * 64, "B": "4" * 64},
            "rederived_evaluation_tree_sha256": "4" * 64, "rows_by_table": row_counts,
            "grouped_bootstrap_rederived": True, "private_outcomes_reparsed": True,
            "searchable_output_contract_verified": True, "byte_identical": True,
            "transitive_rederivation": True, "paper_exact_claim": False,
            "conceptual_objective_reproduced_as_equation": False,
            "matched_accuracy_claim": False,
        }
        source_id, release_id = "leash_stopping", "fixture_leash_v1"
    else:
        payloads, status_rows, predictions = _rag_payloads()
        filenames = {"metrics": "metrics.csv", "contrasts": "contrasts.csv", "panel_status": "panel_status.csv"}
        for name, payload in payloads.items():
            (source_root / filenames[name]).write_bytes(payload)
        reporting_files = {filenames[name]: _sha(payload) for name, payload in payloads.items()}
        reporting_files["predictions.csv"] = _sha(predictions)
        descriptors = [
            {"path": name, "sha256": digest, "size_bytes": (
                len(predictions) if name == "predictions.csv" else len(payloads[name.removesuffix(".csv")])
            )}
            for name, digest in reporting_files.items()
        ]
        manifest_body = {
            "schema_version": "reconstruction-rag-evidence-evaluation-v1",
            "release_id": "fixture_rag_v1", "build_id": "A",
            "lane_id": "rag_evidence_benchmark_v1", "scientific_full": True,
            "score_ab_certificate_sha256": "5" * 64,
            "score_sha256": "6" * 64, "private_label_sha256": "7" * 64,
            "source_binding_sha256": "8" * 64,
            "evaluation_repo_snapshot": {"snapshot_sha256": "9" * 64},
            "score_verifier_repo_snapshot": {"snapshot_sha256": "a" * 64},
            "isolated_score_authentication": {"status": "PASS"},
            "shared_repository_contract": {"status": "BOUND"},
            "bootstrap": {"draws_requested": 20000, "group": "panel-registered source group", "paired_contrasts": True, "seed": 2026082407},
            "files": descriptors, "panel_status": status_rows,
            "cross_panel_macro_computed": False, "refchecker_settings_pooled": False,
            "historical_scores_copied": False,
        }
        manifest = dict(manifest_body); manifest["payload_sha256"] = canonical_sha256(manifest_body)
        manifest_payload = canonical_json_bytes(manifest) + b"\n"
        certificate_body = {
            "schema_version": "reconstruction-rag-evidence-evaluation-ab-v1",
            "release_id": "fixture_rag_v1", "status": "PASS",
            "scientific_full_required": True,
            "transitive_source_rederivation": True,
            "independent_postfreeze_reevaluation": True,
            "score_ab_sha256": "5" * 64,
            "score_sha256": "6" * 64, "private_label_sha256": "7" * 64,
            "source_binding_sha256": "8" * 64,
            "evaluation_repo_snapshot": {"snapshot_sha256": "9" * 64},
            "score_verifier_repo_snapshot": {"snapshot_sha256": "a" * 64},
            "isolated_score_authentication": {"status": "PASS"},
            "shared_repository_contract": {"status": "BOUND"},
            "cross_panel_macro_computed": False, "refchecker_settings_pooled": False,
            "comparisons": {
                **{name: True for name in (
                    "metrics.csv", "predictions.csv", "contrasts.csv",
                    "panel_status.csv", "source_snapshot_identity",
                    "bootstrap_identity", "panel_status_identity",
                    "independent_panel_status_matches_A",
                    "independent_panel_status_matches_B",
                )},
                **{
                    f"independent_{name}_matches_{build_id}": True
                    for name in (
                        "metrics.csv", "predictions.csv", "contrasts.csv",
                        "panel_status.csv",
                    )
                    for build_id in ("A", "B")
                },
            },
            "reporting_files": reporting_files,
            "builds": {"A": {
                "evaluation_manifest_sha256": _sha(manifest_payload),
                "evaluation_manifest_payload_sha256": manifest["payload_sha256"],
            }},
        }
        source_id, release_id = "rag_evidence", "fixture_rag_v1"
    certificate = dict(certificate_body); certificate["payload_sha256"] = canonical_sha256(certificate_body)
    certificate_payload = canonical_json_bytes(certificate) + b"\n"
    (source_root / "CERTIFICATE.json").write_bytes(certificate_payload)
    (source_root / "MANIFEST.json").write_bytes(manifest_payload)
    source_lock = {
        "schema_version": "reconstruction-unified-reporting-source-lock-v1",
        "sources": [{
            "source_id": source_id, "source_release_id": release_id,
            "source_root_id": "fixture", "certified": True,
            "certificate": {
                "path": "CERTIFICATE.json", "file_sha256": _sha(certificate_payload),
                "schema_version": certificate["schema_version"],
                "self_hash_field": "payload_sha256", "self_hash": certificate["payload_sha256"],
                "status_field": "status", "status_value": "PASS",
            },
            "manifest": {
                "path": "MANIFEST.json", "file_sha256": _sha(manifest_payload),
                "schema_version": manifest["schema_version"],
                "self_hash_field": "payload_sha256", "self_hash": manifest["payload_sha256"],
            },
            "files": {
                name: {"path": filenames[name], "format": "csv", "file_sha256": _sha(payload)}
                for name, payload in payloads.items()
            },
        }],
    }
    parsed_contract = parse_contract_bytes(canonical_json_bytes(contract))
    parsed_lock = load_source_lock(_write_lock(root / f"{kind}_lock.json", source_lock))
    validate_contract_source_lock(parsed_contract, parsed_lock)
    authenticated = authenticate_sources(parsed_lock, source_roots={"fixture": source_root})
    return source_root, parsed_contract, parsed_lock, authenticated[0]


def _write_lock(path: Path, value: object) -> Path:
    _write_json(path, value)
    return path


def _forge_metric_value(build: Path, value: float) -> None:
    """Coordinate a self-consistent output-only forgery without changing sources."""

    rows = read_csv_bytes("metrics", (build / "tables/metrics.csv").read_bytes())
    rows[0]["value"] = value
    csv_payload, normalized = csv_bytes("metrics", rows)
    parquet_payload, _ = parquet_bytes("metrics", normalized)
    logical = assert_csv_parquet_parity("metrics", csv_payload, parquet_payload)
    (build / "tables/metrics.csv").write_bytes(csv_payload)
    (build / "tables/metrics.parquet").write_bytes(parquet_payload)

    manifest_path = build / "BRIDGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = manifest["content"]
    for relative, payload in (
        ("tables/metrics.csv", csv_payload),
        ("tables/metrics.parquet", parquet_payload),
    ):
        content["files"][relative] = {"file_sha256": _sha(payload), "bytes": len(payload)}
    record = content["tables"]["metrics"]
    record["logical_sha256"] = logical
    record["csv_file_sha256"] = _sha(csv_payload)
    record["parquet_file_sha256"] = _sha(parquet_payload)
    release_hash = canonical_sha256(content)
    manifest["release_content_sha256"] = release_hash
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = canonical_sha256(manifest)
    manifest_payload = canonical_json_bytes(manifest) + b"\n"
    manifest_path.write_bytes(manifest_payload)

    completion_path = build / "RELEASE_COMPLETE.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["manifest_file_sha256"] = _sha(manifest_payload)
    completion["manifest_payload_sha256"] = manifest["payload_sha256"]
    completion["release_content_sha256"] = release_hash
    completion.pop("completion_sha256")
    completion["completion_sha256"] = canonical_sha256(completion)
    completion_path.write_bytes(canonical_json_bytes(completion) + b"\n")


def _decision() -> dict[str, object]:
    return {
        "release_id": "release", "lane_id": "localization",
        "task_id": "first_error_localization", "source_dataset_id": "dataset",
        "dataset_id": "localization::dataset::dataset",
        "population_id": "localization::population::population",
        "cell_id": "localization::cell::cell", "slice_id": "localization::slice::slice",
        "prediction_unit": "trace_first_error_index",
        "estimand_id": "first_error_index_decision",
        "access_level": "saved_output_probability_telemetry_one_pass",
        "supervision": "unsupervised", "fidelity": "saved_telemetry",
        "report_partition": "primary", "model_id": "model",
        "system_id": "localization::system::method", "row_id": "row",
        "cohort_id": "cohort", "group_id": "group", "fold": 0,
        "predicted_first_error": -1, "true_first_error": 3,
        "comparison_group_id": "decisionv1_fixture",
        "source_comparison_group_id": "producer_group", "source_status": "OK",
        "source_binding_id": "sourcev1_fixture", "source_table": "localization_decisions",
        "source_row_locator": "row:2", "source_row_sha256": "0" * 64,
    }


def _winner_set() -> dict[str, object]:
    scope = {key: _decision()[key] for key in (
        "release_id", "lane_id", "task_id", "source_dataset_id", "dataset_id",
        "population_id", "cell_id", "slice_id", "prediction_unit", "estimand_id",
        "access_level", "supervision", "fidelity", "report_partition",
    )}
    return {
        **scope, "comparison_group_id": "cmpv1_fixture",
        "source_comparison_group_id": "producer_group", "aggregation_id": "native",
        "aggregation_level": "cell", "metric_id": "auroc", "better_direction": "higher",
        "winner_reference_method_id": "winner", "method_id": "candidate",
        "method_value": 0.7, "membership_status": "NOT_SEPARATED_FROM_POINT_WINNER_95CI",
        "in_winner_reference_set": True,
        "interpretation_code": "DIRECT_PAIRED_NONSEPARATION_95",
        "equivalence_claim": False, "simultaneous_coverage": False,
        "winner_selection_adjusted": False, "multiplicity_adjustment": "NONE",
        "source_binding_id": "sourcev1_fixture", "source_table": "winner_reference_sets",
        "source_row_locator": "row:2", "source_row_sha256": "0" * 64,
    }


class SchemaTests(unittest.TestCase):
    def test_comparison_group_separates_every_scientific_boundary(self) -> None:
        signature = {field: f"value::{field}" for field in COMPARISON_SIGNATURE_FIELDS}
        baseline = derive_comparison_group_id(signature)
        for field in ("task_id", "prediction_unit", "access_level", "fidelity", "slice_id"):
            changed = dict(signature)
            changed[field] = f"different::{field}"
            self.assertNotEqual(baseline, derive_comparison_group_id(changed), field)

    @unittest.skipUnless(HAS_ARROW, "PyArrow is required")
    def test_localization_int64_abstention_and_csv_parquet_parity(self) -> None:
        row = _decision()
        csv_payload, _ = csv_bytes("localization_decisions", [row])
        parquet_payload, _ = parquet_bytes("localization_decisions", [row])
        assert_csv_parquet_parity("localization_decisions", csv_payload, parquet_payload)
        self.assertEqual(read_csv_bytes("localization_decisions", csv_payload)[0]["predicted_first_error"], -1)
        import pyarrow.parquet as pq
        self.assertEqual(str(pq.read_schema(BytesIO(parquet_payload)).field("predicted_first_error").type), "int64")
        invalid = dict(row); invalid["predicted_first_error"] = False
        with self.assertRaises(UnifiedReportingError):
            validate_row("localization_decisions", invalid)
        invalid = dict(row); invalid["predicted_first_error"] = -2
        with self.assertRaises(UnifiedReportingError):
            validate_row("localization_decisions", invalid)
        invalid = dict(row); invalid["report_partition"] = "context"
        with self.assertRaises(UnifiedReportingError):
            validate_row("localization_decisions", invalid)

    def test_winner_reference_cannot_be_upgraded_to_equivalence(self) -> None:
        row = _winner_set()
        validate_row("winner_reference_sets", row)
        invalid = dict(row); invalid["equivalence_claim"] = True
        with self.assertRaises(UnifiedReportingError):
            validate_row("winner_reference_sets", invalid)
        invalid = dict(row); invalid["multiplicity_adjustment"] = "BONFERRONI"
        with self.assertRaises(UnifiedReportingError):
            validate_row("winner_reference_sets", invalid)
        invalid = dict(row); invalid["membership_status"] = "TIE"
        with self.assertRaises(UnifiedReportingError):
            validate_row("winner_reference_sets", invalid)


class SourceAuthenticationTests(unittest.TestCase):
    def test_locked_file_rejects_tamper_traversal_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            payload = b"reviewed\n"
            (root / "reviewed.txt").write_bytes(payload)
            self.assertEqual(read_locked_file(root, "reviewed.txt", _sha(payload)), payload)
            (root / "reviewed.txt").write_bytes(b"tampered\n")
            with self.assertRaises(UnifiedReportingError):
                read_locked_file(root, "reviewed.txt", _sha(payload))
            with self.assertRaises(UnifiedReportingError):
                read_locked_file(root, "../reviewed.txt", _sha(payload))
            (root / "target.txt").write_bytes(payload)
            (root / "link.txt").symlink_to(root / "target.txt")
            with self.assertRaises(UnifiedReportingError):
                read_locked_file(root, "link.txt", _sha(payload))

    def test_locked_file_detects_change_during_open_descriptor_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = root / "reviewed.bin"
            payload = b"a" * (2 << 20)
            target.write_bytes(payload)
            real_read = __import__("os").read
            changed = False

            def racing_read(descriptor: int, size: int) -> bytes:
                nonlocal changed
                block = real_read(descriptor, size)
                if block and not changed:
                    changed = True
                    with target.open("ab") as stream:
                        stream.write(b"changed")
                return block

            with patch(
                "spectral_utils.reconstruction_reporting.unified_reporting_sources.os.read",
                side_effect=racing_read,
            ):
                with self.assertRaisesRegex(UnifiedReportingError, "changed while being read"):
                    read_locked_file(root, "reviewed.bin", _sha(payload))

    def test_reviewed_certificate_chain_authenticates_then_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, _, lock_path = _fixture(root)
            lock = load_source_lock(lock_path)
            authenticated = authenticate_sources(lock, source_roots={"fixture": source})
            self.assertEqual([item.source_status for item in authenticated], [
                "CERTIFIED", "NOT_CERTIFIED", "NOT_CERTIFIED",
            ])
            (source / "metrics_long.csv").write_bytes(b"tampered\n")
            with self.assertRaises(UnifiedReportingError):
                authenticate_sources(lock, source_roots={"fixture": source})

    def test_only_registered_future_adapters_may_be_typed_placeholders(self) -> None:
        contract = {
            "schema_version": "reconstruction-unified-reporting-contract-v1",
            "access_partitions": {}, "claim_boundaries": {},
            "lanes": {
                "frozen24": {
                    "adapter": "frozen24_v1", "lane_id": "frozen24_response",
                    "task_id": "final_answer_error_detection",
                    "default_prediction_unit": "response",
                    "default_estimand_id": "response_error_ranking",
                    "report_partition": "primary",
                },
            },
        }
        source_lock = {
            "schema_version": "reconstruction-unified-reporting-source-lock-v1",
            "sources": [{
                "source_id": "frozen24", "source_release_id": "NOT_CERTIFIED",
                "certified": False, "status": "NOT_CERTIFIED",
            }],
        }
        parsed = parse_contract_bytes(canonical_json_bytes(contract))
        with self.assertRaisesRegex(UnifiedReportingError, "certification/adapter mismatch"):
            validate_contract_source_lock(parsed, source_lock)

    def test_leash_chain_rejects_self_consistent_ab_and_claim_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source_root, _, lock, _ = _certified_lane_fixture(root, "leash")
            certificate_path = source_root / "CERTIFICATE.json"
            certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
            certificate.pop("payload_sha256")
            certificate["evaluation_tree_sha256"]["B"] = "5" * 64
            certificate["payload_sha256"] = canonical_sha256(certificate)
            payload = canonical_json_bytes(certificate) + b"\n"
            certificate_path.write_bytes(payload)
            ab_tampered_lock = copy.deepcopy(lock)
            binding = ab_tampered_lock["sources"][0]["certificate"]
            binding["file_sha256"] = _sha(payload)
            binding["self_hash"] = certificate["payload_sha256"]
            with self.assertRaisesRegex(UnifiedReportingError, "passing rederivation"):
                authenticate_sources(
                    ab_tampered_lock, source_roots={"fixture": source_root},
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source_root, _, lock, _ = _certified_lane_fixture(root, "leash")
            manifest_path = source_root / "MANIFEST.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("payload_sha256")
            manifest["paper_exact_claim"] = True
            manifest["payload_sha256"] = canonical_sha256(manifest)
            payload = canonical_json_bytes(manifest) + b"\n"
            manifest_path.write_bytes(payload)
            claim_tampered_lock = copy.deepcopy(lock)
            binding = claim_tampered_lock["sources"][0]["manifest"]
            binding["file_sha256"] = _sha(payload)
            binding["self_hash"] = manifest["payload_sha256"]
            with self.assertRaisesRegex(UnifiedReportingError, "claim boundary drift"):
                authenticate_sources(
                    claim_tampered_lock, source_roots={"fixture": source_root},
                )


@unittest.skipUnless(
    REAL_SOURCE_LOCK_AVAILABLE,
    "immutable canonical science/frozen source releases are unavailable",
)
class CanonicalLeashSourceLockTests(unittest.TestCase):
    def test_real_lock_authenticates_leash_and_keeps_rag_source_closed(self) -> None:
        contract = parse_contract_bytes(
            (REPO / "configs/reconstruction_benchmark_v1/unified_reporting_v1.json")
            .read_bytes()
        )
        source_lock = load_source_lock(
            REPO / "configs/reconstruction_benchmark_v1/unified_reporting_source_lock_v1.json"
        )
        validate_contract_source_lock(contract, source_lock)
        records = {record["source_id"]: record for record in source_lock["sources"]}
        self.assertEqual(len(records), 9)
        self.assertEqual(records["rag_evidence"], {
            "certified": False,
            "source_id": "rag_evidence",
            "source_release_id": "NOT_CERTIFIED",
            "status": "NOT_CERTIFIED",
        })
        leash = records["leash_stopping"]
        self.assertEqual(
            {
                "certified": leash["certified"],
                "source_release_id": leash["source_release_id"],
                "source_root_id": leash["source_root_id"],
            },
            {
                "certified": True,
                "source_release_id": "2026-08-25_leash_v1",
                "source_root_id": "science",
            },
        )
        self.assertEqual(leash["certificate"], {
            "file_sha256": "a98c4084e93d62b88227e1a78a16f86fdab071e39664440a6b609bf911ec3b88",
            "path": str(LEASH_RELEASE_RELATIVE / "EVALUATION_AB_VERIFICATION.json"),
            "schema_version": "reconstruction-leash-stopping-evaluation-ab-v1",
            "self_hash": "3ce4fb378ea308bfdb132d5758fab967567209be3743fef8b6639a723b90af15",
            "self_hash_field": "payload_sha256",
            "status_field": "status", "status_value": "PASS",
        })
        self.assertEqual(leash["manifest"], {
            "file_sha256": "7cd65d462793f9f60f6b5698b4f308b9565899e55dab3576264e7789e99f87c2",
            "path": str(LEASH_RELEASE_RELATIVE / "A/evaluation/EVALUATION_MANIFEST.json"),
            "schema_version": "reconstruction-leash-stopping-evaluation-v1",
            "self_hash": "3b4de40d3c1387c9d85fef23dacc0ba1143204ed4f6a440b18532484f72950c9",
            "self_hash_field": "payload_sha256",
        })
        self.assertEqual(set(leash["files"]), set(LEASH_LOCKED_FILES))
        for role, (filename, digest) in LEASH_LOCKED_FILES.items():
            self.assertEqual(leash["files"][role], {
                "file_sha256": digest, "format": "csv",
                "path": str(LEASH_RELEASE_RELATIVE / "A/evaluation" / filename),
            })

        with tempfile.TemporaryDirectory() as temporary:
            poison_root = Path(temporary).resolve()
            poison = b"PRIVATE_RAG_PREDICTION_LABEL_SENTINEL"
            (poison_root / "predictions.csv").write_bytes(poison)
            reads: list[tuple[Path, str]] = []
            real_read = unified_sources.read_locked_file

            def audited_read(root: str | Path, relative: str, digest: str) -> bytes:
                reads.append((Path(root).resolve(), relative))
                return real_read(root, relative, digest)

            with patch(
                "spectral_utils.reconstruction_reporting.unified_reporting_sources.read_locked_file",
                side_effect=audited_read,
            ):
                sources = authenticate_sources(
                    source_lock,
                    source_roots={
                        "frozen24": REPO,
                        "science": SCIENCE_REPO,
                        "rag_evidence": poison_root,
                    },
                )
            self.assertFalse(any(root == poison_root for root, _ in reads))
            self.assertFalse(any("predictions" in relative for _, relative in reads))
            self.assertFalse(any("per_question" in relative for _, relative in reads))
            self.assertEqual(
                [source.source_status for source in sources].count("CERTIFIED"), 8,
            )
            self.assertEqual(
                [source.source_status for source in sources].count("NOT_CERTIFIED"), 1,
            )
            rag_source = next(source for source in sources if source.source_id == "rag_evidence")
            self.assertEqual(rag_source.files, {})
            self.assertIsNone(rag_source.certificate)
            self.assertIsNone(rag_source.manifest)
            leash_source = next(
                source for source in sources if source.source_id == "leash_stopping"
            )
            certificate = leash_source.certificate or {}
            manifest = leash_source.manifest or {}
            self.assertEqual({
                certificate["evaluation_tree_sha256"]["A"],
                certificate["evaluation_tree_sha256"]["B"],
                certificate["rederived_evaluation_tree_sha256"],
            }, {
                "256cfd7cdf07339d78752d3887d25becaf929ee111f4f58b9361a9a93a644496",
            })
            self.assertTrue(all(certificate[field] is True for field in (
                "byte_identical", "transitive_rederivation",
                "grouped_bootstrap_rederived", "private_outcomes_reparsed",
                "searchable_output_contract_verified",
            )))
            self.assertTrue(all(certificate[field] is False for field in (
                "paper_exact_claim", "conceptual_objective_reproduced_as_equation",
                "matched_accuracy_claim",
            )))
            self.assertEqual(certificate["rows_by_table"], {
                "aggregate_metrics": 72, "bootstrap_intervals": 378,
                "cell_metrics": 18, "contrasts": 12, "coverage": 8,
                "frontier": 18, "per_question": 4986,
            })
            self.assertEqual(manifest["fidelity"], "paper-specified-partial")
            self.assertTrue(manifest["policy_execution_evaluated"])
            self.assertTrue(manifest["all_policy_stops_have_realized_closure"])
            self.assertFalse(manifest["cross_task_or_access_macro"])
            self.assertFalse(manifest["proxy_stopping"])
            self.assertFalse(manifest["paper_exact_claim"])
            self.assertFalse(manifest["matched_accuracy_claim"])

        tables = build_unified_rows(
            release_id="canonical-real-lock-check", contract=contract, sources=sources,
        )
        self.assertEqual(len(tables["source_bindings"]), 9)
        self.assertEqual(sum(
            row["lane_id"] == "leash_actual_stopping"
            for row in tables["context_metrics"]
        ), 306)
        self.assertEqual(sum(
            row["lane_id"] == "leash_actual_stopping"
            for row in tables["context_contrasts"]
        ), 162)
        self.assertEqual(sum(
            row["lane_id"] == "leash_actual_stopping"
            for row in tables["context_coverage"]
        ), 8)
        leash_artifacts = {
            row["logical_name"]
            for row in tables["source_artifacts"]
            if row["source_id"] == "leash_stopping"
        }
        self.assertEqual(leash_artifacts, {
            "certificate", "manifest", *LEASH_LOCKED_FILES,
        })
        self.assertNotIn("per_question", leash_artifacts)
        self.assertFalse(any(
            row["source_id"] == "rag_evidence"
            for row in tables["source_artifacts"]
        ))
        rag_statuses = [
            row for row in tables["status"] if row["lane_id"] == "rag_evidence"
        ]
        self.assertEqual(len(rag_statuses), 1)
        self.assertEqual(rag_statuses[0]["status_class"], "NOT_CERTIFIED")


class CertifiedContextLaneTests(unittest.TestCase):
    def test_leash_aggregate_only_adapter_is_context_and_reference_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _, contract, _, source = _certified_lane_fixture(root, "leash")
            tables = build_unified_rows(
                release_id="fixture-release", contract=contract, sources=[source],
            )

            self.assertEqual(len(tables["context_metrics"]), 306)
            self.assertEqual(len(tables["context_contrasts"]), 162)
            self.assertEqual(len(tables["context_coverage"]), 8)
            self.assertFalse(tables["metrics"])
            self.assertFalse(tables["contrasts"])
            self.assertFalse(tables["coverage"])
            self.assertTrue(all(
                row["report_partition"] == "context"
                and row["access_level"]
                == "actual_callback_execution_with_cot_nocot_references"
                and row["fidelity"] == "paper-specified-partial"
                for table in ("context_metrics", "context_contrasts", "context_coverage")
                for row in tables[table]
            ))
            interval_metrics = [
                row for row in tables["context_metrics"]
                if not row["metric_id"].startswith("frontier_")
            ]
            frontier_metrics = [
                row for row in tables["context_metrics"]
                if row["metric_id"].startswith("frontier_")
            ]
            # The 216 point intervals enrich (and do not duplicate) the
            # 18x8 cell plus 72 aggregate point estimates.  Frontier is a
            # separate 18x5 descriptive relation without inferential fields.
            self.assertEqual(len(interval_metrics), 216)
            self.assertEqual(len(frontier_metrics), 90)
            self.assertTrue(all(
                row["source_table"] == "bootstrap_intervals"
                and row["bootstrap_draws"] == 2000
                and row["ci_low"] is not None
                and row["ci_high"] is not None
                for row in interval_metrics
            ))
            self.assertTrue(all(
                row["source_table"] == "frontier"
                and row["bootstrap_draws"] is None
                and row["ci_low"] is None
                and row["ci_high"] is None
                and not row["rankable"]
                and "descriptive_only_no_interval" in (row["status_detail"] or "")
                for row in frontier_metrics
            ))
            self.assertEqual(
                len({
                    (row["comparison_group_id"], row["system_id"])
                    for row in tables["context_metrics"]
                }),
                306,
            )
            self.assertTrue(all(
                row["source_table"] == "bootstrap_intervals"
                for row in tables["context_contrasts"]
            ))
            self.assertTrue(all(
                row["right_method_id"] == "cot|central" and row["paired"]
                for row in tables["context_contrasts"]
            ))
            self.assertEqual(set(source.files), {
                "aggregate_metrics", "bootstrap_intervals", "cell_metrics",
                "contrasts", "coverage", "frontier",
            })
            self.assertFalse(any(
                row["logical_name"] in {"per_question", "predictions"}
                for row in tables["source_artifacts"]
            ))
            self.assertFalse(any(
                row["report_partition"] == "primary" or row["rankable"]
                for row in tables["status"]
                if row["lane_id"] == "leash_actual_stopping"
            ))
            self.assertFalse(advisor_inputs(
                release_id="fixture-release", contract=contract, tables=tables,
            )["scientific_cross_task_macro_computed"])

    def test_rag_seven_panel_adapter_preserves_units_subgroups_and_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _, contract, _, source = _certified_lane_fixture(root, "rag")
            tables = build_unified_rows(
                release_id="fixture-release", contract=contract, sources=[source],
            )

            metrics = tables["context_metrics"]
            contrasts = tables["context_contrasts"]
            self.assertEqual(len(metrics), 50)
            self.assertEqual(len(contrasts), 4)
            self.assertFalse(tables["metrics"])
            self.assertFalse(tables["contrasts"])
            self.assertEqual(
                {row["population_id"].rsplit("::", 1)[-1] for row in metrics},
                set(RAG_PANELS),
            )
            self.assertTrue(all(
                row["report_partition"] == "context"
                and row["bootstrap_draws"] == 20000
                and row["aggregation_level"] == "cell"
                for row in metrics
            ))
            refchecker_slices = {
                row["slice_id"].rsplit("::", 1)[-1] for row in metrics
                if row["population_id"].rsplit("::", 1)[-1].startswith("refchecker_")
            }
            self.assertEqual(refchecker_slices, {
                "accurate_context", "noisy_context", "zero_context",
            })
            self.assertNotIn("all", refchecker_slices)
            self.assertTrue(all(
                row["population_id"].endswith("::gasp_protocol_sentence")
                and row["left_method_id"] == "gasp_threshold"
                and row["right_method_id"] == "fixed_rag_iu_pcr_matched"
                and row["paired"]
                for row in contrasts
            ))
            self.assertEqual(set(source.files), {"metrics", "contrasts", "panel_status"})
            self.assertFalse(any(
                row["logical_name"] == "predictions"
                for row in tables["source_artifacts"]
            ))
            panel_statuses = [
                row for row in tables["status"]
                if row["lane_id"] == "rag_evidence"
                and row["status_scope"] == "cell"
                and row["source_table"] == "panel_status"
            ]
            self.assertEqual(len(panel_statuses), 7)
            self.assertTrue(all(not row["rankable"] for row in panel_statuses))
            self.assertFalse(advisor_inputs(
                release_id="fixture-release", contract=contract, tables=tables,
            )["scientific_cross_task_macro_computed"])

    def test_rag_pooling_cross_panel_contrast_and_primary_partition_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _, contract, _, source = _certified_lane_fixture(root, "rag")

            rows = list(csv.DictReader(StringIO(source.files["metrics"].decode("utf-8"))))
            for row in rows:
                if row["panel_id"] == "refchecker_threeway":
                    row["subgroup"] = "all"
                    break
            pooled = replace(
                source,
                files={**source.files, "metrics": _csv_payload(list(rows[0]), rows)},
            )
            with self.assertRaisesRegex(UnifiedReportingError, "RefChecker setting pooling"):
                build_unified_rows(
                    release_id="fixture-release", contract=contract, sources=[pooled],
                )

            rows = list(csv.DictReader(StringIO(source.files["contrasts"].decode("utf-8"))))
            rows[0]["panel_id"] = "ragtruth_evidence_contrast_sentence"
            cross_panel = replace(
                source,
                files={**source.files, "contrasts": _csv_payload(list(rows[0]), rows)},
            )
            with self.assertRaisesRegex(UnifiedReportingError, "cross-panel contrast"):
                build_unified_rows(
                    release_id="fixture-release", contract=contract, sources=[cross_panel],
                )

            primary_contract = copy.deepcopy(contract)
            primary_contract["lanes"]["rag_evidence"]["report_partition"] = "primary"
            with self.assertRaisesRegex(UnifiedReportingError, "RAG unified-reporting contract drift"):
                parse_contract_bytes(canonical_json_bytes(primary_contract))

    def test_rag_single_class_metric_remains_typed_undefined_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _, contract, _, source = _certified_lane_fixture(root, "rag")
            rows = list(csv.DictReader(StringIO(source.files["metrics"].decode("utf-8"))))
            rows[0]["status"] = "METRIC_UNDEFINED_SINGLE_CLASS"
            rows[0]["value"] = ""
            rows[0]["ci_low"] = ""
            rows[0]["ci_high"] = ""
            undefined = replace(
                source,
                files={**source.files, "metrics": _csv_payload(list(rows[0]), rows)},
            )
            tables = build_unified_rows(
                release_id="fixture-release", contract=contract, sources=[undefined],
            )
            row = next(
                item for item in tables["context_metrics"]
                if item["source_row_locator"] == "row:2"
            )
            self.assertEqual(row["status_class"], "UNDEFINED")
            self.assertIsNone(row["value"])
            self.assertIsNone(row["ci_low"])
            self.assertIsNone(row["ci_high"])
            self.assertFalse(row["rankable"])

    def test_leash_unregistered_reference_comparison_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _, contract, _, source = _certified_lane_fixture(root, "leash")
            rows = list(csv.DictReader(StringIO(
                source.files["bootstrap_intervals"].decode("utf-8"),
            )))
            for row in rows:
                if row["reference_arm"] == "cot":
                    row["reference_arm"] = "nocot"
                    break
            escaped = replace(
                source,
                files={
                    **source.files,
                    "bootstrap_intervals": _csv_payload(list(rows[0]), rows),
                },
            )
            with self.assertRaisesRegex(UnifiedReportingError, "unregistered comparison"):
                build_unified_rows(
                    release_id="fixture-release", contract=contract, sources=[escaped],
                )

    def test_rag_chain_rejects_hash_schema_and_self_consistent_claim_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source_root, _, lock, _ = _certified_lane_fixture(root, "rag")
            metrics = source_root / "metrics.csv"
            metrics.write_bytes(metrics.read_bytes() + b"\n")
            with self.assertRaisesRegex(UnifiedReportingError, "hash mismatch"):
                authenticate_sources(lock, source_roots={"fixture": source_root})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source_root, _, lock, _ = _certified_lane_fixture(root, "rag")
            certificate_path = source_root / "CERTIFICATE.json"
            certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
            certificate.pop("payload_sha256")
            certificate["schema_version"] = "unreviewed-rag-schema"
            certificate["payload_sha256"] = canonical_sha256(certificate)
            payload = canonical_json_bytes(certificate) + b"\n"
            certificate_path.write_bytes(payload)
            bad_lock = copy.deepcopy(lock)
            binding = bad_lock["sources"][0]["certificate"]
            binding["file_sha256"] = _sha(payload)
            binding["self_hash"] = certificate["payload_sha256"]
            with self.assertRaisesRegex(UnifiedReportingError, "schema drift"):
                authenticate_sources(bad_lock, source_roots={"fixture": source_root})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source_root, _, lock, _ = _certified_lane_fixture(root, "rag")
            certificate_path = source_root / "CERTIFICATE.json"
            certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
            certificate.pop("payload_sha256")
            certificate["cross_panel_macro_computed"] = True
            certificate["payload_sha256"] = canonical_sha256(certificate)
            payload = canonical_json_bytes(certificate) + b"\n"
            certificate_path.write_bytes(payload)
            forged_lock = copy.deepcopy(lock)
            binding = forged_lock["sources"][0]["certificate"]
            binding["file_sha256"] = _sha(payload)
            binding["self_hash"] = certificate["payload_sha256"]
            with self.assertRaisesRegex(UnifiedReportingError, "not a passing rederivation"):
                authenticate_sources(forged_lock, source_roots={"fixture": source_root})


@unittest.skipUnless(HAS_ARROW, "PyArrow is required")
class PublicationTests(unittest.TestCase):
    def test_no_clobber_completion_ab_and_status_only_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, contract, lock = _fixture(root)
            private_sentinel = b"PRIVATE_RAG_LEASH_SENTINEL_DO_NOT_INGEST"
            (source / "predictions.csv").write_bytes(private_sentinel)
            (source / "per_question.csv").write_bytes(private_sentinel)
            typed = json.loads(contract.read_text(encoding="utf-8"))
            self.assertEqual(typed["lanes"]["rag_evidence"]["adapter"], "rag_evidence_v1")
            self.assertEqual(typed["lanes"]["leash_stopping"]["adapter"], "leash_v1")
            authenticated = authenticate_sources(
                load_source_lock(lock), source_roots={"fixture": source},
            )
            self.assertTrue(all(
                item.files == {}
                for item in authenticated
                if item.source_id in {"rag_evidence", "leash_stopping"}
            ))
            build_a, build_b = root / "build_A", root / "build_B"
            result_a = build_unified_release(
                release_id="fixture-release", build_id="A", output_root=build_a,
                contract_path=contract, source_lock_path=lock,
                source_roots={"fixture": source},
            )
            result_b = build_unified_release(
                release_id="fixture-release", build_id="B", output_root=build_b,
                contract_path=contract, source_lock_path=lock,
                source_roots={"fixture": source},
            )
            self.assertEqual(result_a["release_content_sha256"], result_b["release_content_sha256"])
            verified = verify_completed_release(build_a)
            self.assertEqual(verified.completion["status"], "COMPLETE")
            statuses = read_csv_bytes("status", (build_a / "tables/status.csv").read_bytes())
            placeholders = [row for row in statuses if row["status_class"] == "NOT_CERTIFIED"]
            self.assertEqual(len(placeholders), 2)
            self.assertTrue(all(row["status_scope"] == "lane" and not row["rankable"] for row in placeholders))
            for table in (
                "metrics", "context_metrics", "contrasts", "context_contrasts",
                "coverage", "context_coverage", "localization_decisions",
                "context_localization_decisions", "winner_reference_sets",
                "context_winner_reference_sets", "winner_reference_contrasts",
                "context_winner_reference_contrasts",
            ):
                rows = read_csv_bytes(table, (build_a / f"tables/{table}.csv").read_bytes())
                self.assertFalse(any(row["source_binding_id"] in {
                    placeholder["source_binding_id"] for placeholder in placeholders
                } for row in rows), table)
            rendered = "\n".join(
                path.read_text(encoding="utf-8")
                for path in (
                    build_a / "REPORT_CONTRACT.json", build_a / "SOURCE_LOCK.json",
                    build_a / "BRIDGE_MANIFEST.json", build_a / "RELEASE_COMPLETE.json",
                    build_a / "tables/source_artifacts.csv",
                )
            )
            self.assertNotIn(str(source.resolve()), rendered)
            self.assertFalse(any(
                private_sentinel in path.read_bytes()
                for path in build_a.rglob("*")
                if path.is_file()
            ))
            with self.assertRaises(UnifiedReportingError):
                build_unified_release(
                    release_id="fixture-release", build_id="A2", output_root=build_a,
                    contract_path=contract, source_lock_path=lock,
                    source_roots={"fixture": source},
                )
            certificate = root / "AB_VERIFICATION.json"
            issued = verify_unified_release_ab(
                build_a=build_a, build_b=build_b, certificate_path=certificate,
                contract_path=contract, source_lock_path=lock,
                source_roots={"fixture": source},
            )
            self.assertEqual(issued["status"], "PASS")
            with self.assertRaises(UnifiedReportingError):
                verify_unified_release_ab(
                    build_a=build_a, build_b=build_b, certificate_path=certificate,
                    contract_path=contract, source_lock_path=lock,
                    source_roots={"fixture": source},
                )
            with (build_b / "tables/metrics.csv").open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaises(UnifiedReportingError):
                verify_completed_release(build_b)

    def test_missing_completion_and_extra_file_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            incomplete = root / "incomplete"
            incomplete.mkdir()
            with self.assertRaises(UnifiedReportingError):
                verify_completed_release(incomplete)
            source, contract, lock = _fixture(root)
            release = root / "release"
            build_unified_release(
                release_id="fixture-release", build_id="A", output_root=release,
                contract_path=contract, source_lock_path=lock,
                source_roots={"fixture": source},
            )
            (release / "UNREVIEWED.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaises(UnifiedReportingError):
                verify_completed_release(release)

    def test_ab_rejects_coordinated_fabrication_by_source_rederivation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, contract, lock = _fixture(root)
            build_a, build_b = root / "build_A", root / "build_B"
            for build_id, output in (("A", build_a), ("B", build_b)):
                build_unified_release(
                    release_id="fixture-release", build_id=build_id, output_root=output,
                    contract_path=contract, source_lock_path=lock,
                    source_roots={"fixture": source},
                )
            _forge_metric_value(build_a, 0.123456789)
            _forge_metric_value(build_b, 0.123456789)
            # Output-only verification sees a self-consistent forged tree.  The
            # A/B trust path must independently reopen the authenticated source.
            self.assertEqual(
                read_csv_bytes("metrics", (build_a / "tables/metrics.csv").read_bytes())[0]["value"],
                0.123456789,
            )
            verify_completed_release(build_a)
            certificate = root / "AB_FORGED.json"
            with self.assertRaisesRegex(UnifiedReportingError, "independent authenticated-source rederivation"):
                verify_unified_release_ab(
                    build_a=build_a, build_b=build_b, certificate_path=certificate,
                    contract_path=contract, source_lock_path=lock,
                    source_roots={"fixture": source},
                )
            self.assertFalse(certificate.exists())

    def test_late_inventory_injection_fails_end_rescan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, contract, lock = _fixture(root)
            release = root / "release"
            build_unified_release(
                release_id="fixture-release", build_id="A", output_root=release,
                contract_path=contract, source_lock_path=lock,
                source_roots={"fixture": source},
            )
            real_read = unified_publish._read_held_regular_fd
            injected = False

            def inject_then_read(
                descriptor: int, expected: tuple[int, ...], relative: str,
            ) -> bytes:
                nonlocal injected
                if not injected:
                    injected = True
                    (release / "LATE_UNREVIEWED.txt").write_text("late", encoding="utf-8")
                return real_read(descriptor, expected, relative)

            with patch.object(
                unified_publish, "_read_held_regular_fd", side_effect=inject_then_read,
            ):
                with self.assertRaisesRegex(UnifiedReportingError, "inventory drift"):
                    verify_completed_release(release)

    def test_parent_and_stage_swaps_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, contract, lock = _fixture(root)
            real_assert = unified_publish._assert_directory_path_identity
            moved = root.with_name(f"{root.name}.moved")
            swapped = False

            def swap_parent(path: Path, expected: os.stat_result, *, where: str) -> None:
                nonlocal swapped
                if where == "publication parent" and not swapped:
                    swapped = True
                    path.rename(moved)
                    path.mkdir()
                real_assert(path, expected, where=where)

            try:
                with patch.object(
                    unified_publish, "_assert_directory_path_identity", side_effect=swap_parent,
                ):
                    with self.assertRaisesRegex(UnifiedReportingError, "publication parent changed identity"):
                        build_unified_release(
                            release_id="fixture-release", build_id="A", output_root=root / "release",
                            contract_path=contract, source_lock_path=lock,
                            source_roots={"fixture": source},
                        )
            finally:
                if swapped:
                    root.rmdir()
                    moved.rename(root)
            self.assertFalse((root / "release").exists())
            self.assertFalse(any(path.name.startswith(".release.staging-") for path in root.iterdir()))

            real_rename = unified_publish._rename_noreplace
            stage_swapped = False
            displaced_stage: str | None = None

            def swap_stage(parent_fd: int, source_name: str, target_name: str) -> None:
                nonlocal displaced_stage, stage_swapped
                if target_name == "release" and ".staging-" in source_name and not stage_swapped:
                    stage_swapped = True
                    held_name = f"{source_name}.held"
                    displaced_stage = held_name
                    os.rename(
                        source_name, held_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
                    )
                    os.mkdir(source_name, mode=0o755, dir_fd=parent_fd)
                real_rename(parent_fd, source_name, target_name)

            with patch.object(unified_publish, "_rename_noreplace", side_effect=swap_stage):
                with self.assertRaisesRegex(UnifiedReportingError, "published release changed identity"):
                    build_unified_release(
                        release_id="fixture-release", build_id="A", output_root=root / "release",
                        contract_path=contract, source_lock_path=lock,
                        source_roots={"fixture": source},
                    )
            self.assertTrue(stage_swapped)
            self.assertFalse((root / "release").exists())
            quarantined = list(root.glob(".release.invalid-*"))
            self.assertEqual(len(quarantined), 1)
            self.assertTrue(quarantined[0].is_dir())
            self.assertEqual(list(quarantined[0].iterdir()), [])
            self.assertIsNotNone(displaced_stage)
            genuine_evidence = root / str(displaced_stage)
            self.assertTrue(genuine_evidence.is_dir())
            verify_completed_release(genuine_evidence)

    def test_release_postrename_injection_evacuates_canonical_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, contract, lock = _fixture(root)
            release = root / "release"
            real_rename = unified_publish._rename_noreplace
            injected = False

            def inject_after_release_rename(
                parent_fd: int, source_name: str, target_name: str,
            ) -> None:
                nonlocal injected
                real_rename(parent_fd, source_name, target_name)
                if target_name == release.name and ".staging-" in source_name and not injected:
                    injected = True
                    (release / "late.bin").write_bytes(b"preserve late evidence")

            with patch.object(
                unified_publish, "_rename_noreplace", side_effect=inject_after_release_rename,
            ):
                with self.assertRaisesRegex(UnifiedReportingError, "inventory drift"):
                    build_unified_release(
                        release_id="fixture-release", build_id="A", output_root=release,
                        contract_path=contract, source_lock_path=lock,
                        source_roots={"fixture": source},
                    )
            self.assertTrue(injected)
            self.assertFalse(release.exists())
            quarantined = list(root.glob(".release.invalid-*"))
            self.assertEqual(len(quarantined), 1)
            self.assertTrue(quarantined[0].is_dir())
            self.assertEqual(
                (quarantined[0] / "late.bin").read_bytes(), b"preserve late evidence",
            )
            self.assertTrue((quarantined[0] / "RELEASE_COMPLETE.json").is_file())

    def test_release_rename_commit_uncertainty_quarantines_canonical(self) -> None:
        for replace_after_commit in (False, True):
            with self.subTest(replace_after_commit=replace_after_commit):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary).resolve()
                    source, contract, lock = _fixture(root)
                    release = root / "release"
                    displaced = root / "DISPLACED_COMMITTED_RELEASE"
                    real_rename = unified_publish._rename_noreplace
                    committed = False

                    def commit_then_raise(
                        parent_fd: int, source_name: str, target_name: str,
                    ) -> None:
                        nonlocal committed
                        if target_name == release.name and ".staging-" in source_name and not committed:
                            real_rename(parent_fd, source_name, target_name)
                            committed = True
                            if replace_after_commit:
                                os.rename(
                                    target_name, displaced.name,
                                    src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
                                )
                                with open(release, "wb") as stream:
                                    stream.write(b"preserve commit-window replacement")
                            raise UnifiedReportingError("injected exception after kernel rename commit")
                        real_rename(parent_fd, source_name, target_name)

                    with patch.object(
                        unified_publish, "_rename_noreplace", side_effect=commit_then_raise,
                    ):
                        with self.assertRaisesRegex(
                            UnifiedReportingError, "injected exception after kernel rename commit",
                        ):
                            build_unified_release(
                                release_id="fixture-release", build_id="A", output_root=release,
                                contract_path=contract, source_lock_path=lock,
                                source_roots={"fixture": source},
                            )
                    self.assertTrue(committed)
                    self.assertFalse(release.exists())
                    quarantined = list(root.glob(".release.invalid-*"))
                    self.assertEqual(len(quarantined), 1)
                    if replace_after_commit:
                        self.assertTrue(quarantined[0].is_file())
                        self.assertEqual(
                            quarantined[0].read_bytes(), b"preserve commit-window replacement",
                        )
                        verify_completed_release(displaced)
                    else:
                        self.assertTrue(quarantined[0].is_dir())
                        verify_completed_release(quarantined[0])

    def test_release_commit_uncertainty_with_restored_stage_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, contract, lock = _fixture(root)
            release = root / "release"
            real_rename = unified_publish._rename_noreplace
            committed = False

            def commit_restore_stage_replace_then_raise(
                parent_fd: int, source_name: str, target_name: str,
            ) -> None:
                nonlocal committed
                if target_name == release.name and ".staging-" in source_name and not committed:
                    real_rename(parent_fd, source_name, target_name)
                    os.rename(
                        target_name, source_name,
                        src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
                    )
                    release.write_bytes(b"preserve restored-stage replacement")
                    committed = True
                    raise UnifiedReportingError(
                        "injected exception after release commit and stage restoration"
                    )
                real_rename(parent_fd, source_name, target_name)

            with patch.object(
                unified_publish, "_rename_noreplace",
                side_effect=commit_restore_stage_replace_then_raise,
            ):
                with self.assertRaisesRegex(
                    UnifiedReportingError,
                    "injected exception after release commit and stage restoration",
                ):
                    build_unified_release(
                        release_id="fixture-release", build_id="A", output_root=release,
                        contract_path=contract, source_lock_path=lock,
                        source_roots={"fixture": source},
                    )
            self.assertTrue(committed)
            self.assertFalse(release.exists())
            quarantined = list(root.glob(".release.invalid-*"))
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(
                quarantined[0].read_bytes(), b"preserve restored-stage replacement",
            )
            staging_evidence = list(root.glob(".release.staging-*"))
            self.assertEqual(len(staging_evidence), 1)
            verify_completed_release(staging_evidence[0])

    def test_release_final_replacement_after_publish_is_quarantined(self) -> None:
        for replacement_kind in ("directory", "file"):
            with self.subTest(replacement_kind=replacement_kind):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary).resolve()
                    source, contract, lock = _fixture(root)
                    release = root / "release"
                    displaced = root / f"DISPLACED_GENUINE_RELEASE_{replacement_kind}"
                    real_assert = unified_publish._assert_held_file_states
                    replaced = False

                    def replace_during_first_postpublish_check(
                        file_fds: dict[str, int], inventory: dict[str, tuple[int, ...]],
                    ) -> None:
                        nonlocal replaced
                        if release.exists() and not replaced:
                            os.rename(release, displaced)
                            if replacement_kind == "directory":
                                release.mkdir()
                                (release / "unrelated.txt").write_text(
                                    "preserve replacement directory", encoding="utf-8",
                                )
                            else:
                                release.write_text("preserve replacement file", encoding="utf-8")
                            replaced = True
                        real_assert(file_fds, inventory)

                    with patch.object(
                        unified_publish, "_assert_held_file_states",
                        side_effect=replace_during_first_postpublish_check,
                    ):
                        with self.assertRaisesRegex(
                            UnifiedReportingError, "published release before success changed identity",
                        ):
                            build_unified_release(
                                release_id="fixture-release", build_id="A", output_root=release,
                                contract_path=contract, source_lock_path=lock,
                                source_roots={"fixture": source},
                            )
                    self.assertTrue(replaced)
                    self.assertFalse(release.exists())
                    verify_completed_release(displaced)
                    quarantined = list(root.glob(".release.invalid-*"))
                    self.assertEqual(len(quarantined), 1)
                    if replacement_kind == "directory":
                        self.assertTrue(quarantined[0].is_dir())
                        self.assertEqual(
                            (quarantined[0] / "unrelated.txt").read_text(encoding="utf-8"),
                            "preserve replacement directory",
                        )
                    else:
                        self.assertTrue(quarantined[0].is_file())
                        self.assertEqual(
                            quarantined[0].read_text(encoding="utf-8"),
                            "preserve replacement file",
                        )

    def test_certificate_inside_build_is_rejected_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, contract, lock = _fixture(root)
            build_a, build_b = root / "build_A", root / "build_B"
            for build_id, output in (("A", build_a), ("B", build_b)):
                build_unified_release(
                    release_id="fixture-release", build_id=build_id, output_root=output,
                    contract_path=contract, source_lock_path=lock,
                    source_roots={"fixture": source},
                )
            inside = build_a / "AB_VERIFICATION.json"
            with self.assertRaisesRegex(UnifiedReportingError, "inside an authenticated build tree"):
                verify_unified_release_ab(
                    build_a=build_a, build_b=build_b, certificate_path=inside,
                    contract_path=contract, source_lock_path=lock,
                    source_roots={"fixture": source},
                )
            self.assertFalse(inside.exists())
            verify_completed_release(build_a)
            verify_completed_release(build_b)

    def test_certificate_commit_window_rescans_builds_and_rolls_back_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, contract, lock = _fixture(root)
            build_a, build_b = root / "build_A", root / "build_B"
            for build_id, output in (("A", build_a), ("B", build_b)):
                build_unified_release(
                    release_id="fixture-release", build_id=build_id, output_root=output,
                    contract_path=contract, source_lock_path=lock,
                    source_roots={"fixture": source},
                )
            certificate = root / "AB_LATE_FILE.json"
            real_rename = unified_publish._rename_noreplace
            injected = False

            def inject_inside_certificate_rename(
                parent_fd: int, source_name: str, target_name: str,
            ) -> None:
                nonlocal injected
                if target_name == certificate.name and not injected:
                    injected = True
                    (build_a / "LATE_UNREVIEWED.txt").write_text(
                        "late unreviewed payload", encoding="utf-8",
                    )
                real_rename(parent_fd, source_name, target_name)

            with patch.object(
                unified_publish, "_rename_noreplace",
                side_effect=inject_inside_certificate_rename,
            ):
                with self.assertRaisesRegex(UnifiedReportingError, "inventory drift"):
                    verify_unified_release_ab(
                        build_a=build_a, build_b=build_b,
                        certificate_path=certificate, contract_path=contract,
                        source_lock_path=lock, source_roots={"fixture": source},
                    )
            self.assertTrue(injected)
            self.assertFalse(certificate.exists())
            quarantined = list(root.glob(f".{certificate.name}.invalid-*"))
            self.assertEqual(len(quarantined), 1)
            quarantined_payload = json.loads(quarantined[0].read_text(encoding="utf-8"))
            self.assertEqual(quarantined_payload["status"], "PASS")
            self.assertTrue((build_a / "LATE_UNREVIEWED.txt").is_file())

    def test_certificate_rename_commit_uncertainty_quarantines_replacement(self) -> None:
        for replacement_kind in ("file", "directory"):
            with self.subTest(replacement_kind=replacement_kind):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary).resolve()
                    source, contract, lock = _fixture(root)
                    build_a, build_b = root / "build_A", root / "build_B"
                    for build_id, output in (("A", build_a), ("B", build_b)):
                        build_unified_release(
                            release_id="fixture-release", build_id=build_id, output_root=output,
                            contract_path=contract, source_lock_path=lock,
                            source_roots={"fixture": source},
                        )
                    certificate = root / f"AB_COMMIT_UNCERTAIN_{replacement_kind}.json"
                    real_rename = unified_publish._rename_noreplace
                    committed = False

                    def commit_unlink_replace_then_raise(
                        parent_fd: int, source_name: str, target_name: str,
                    ) -> None:
                        nonlocal committed
                        if target_name == certificate.name and not committed:
                            real_rename(parent_fd, source_name, target_name)
                            os.unlink(target_name, dir_fd=parent_fd)
                            if replacement_kind == "file":
                                certificate.write_bytes(b"preserve attacker certificate file")
                            else:
                                os.mkdir(target_name, mode=0o755, dir_fd=parent_fd)
                                (certificate / "attacker.txt").write_text(
                                    "preserve attacker certificate directory", encoding="utf-8",
                                )
                            committed = True
                            raise UnifiedReportingError(
                                "injected certificate exception after kernel rename commit"
                            )
                        real_rename(parent_fd, source_name, target_name)

                    with patch.object(
                        unified_publish, "_rename_noreplace",
                        side_effect=commit_unlink_replace_then_raise,
                    ):
                        with self.assertRaisesRegex(
                            UnifiedReportingError,
                            "injected certificate exception after kernel rename commit",
                        ):
                            verify_unified_release_ab(
                                build_a=build_a, build_b=build_b,
                                certificate_path=certificate, contract_path=contract,
                                source_lock_path=lock, source_roots={"fixture": source},
                            )
                    self.assertTrue(committed)
                    self.assertFalse(certificate.exists())
                    quarantined = list(root.glob(f".{certificate.name}.invalid-*"))
                    self.assertEqual(len(quarantined), 1)
                    if replacement_kind == "file":
                        self.assertTrue(quarantined[0].is_file())
                        self.assertEqual(
                            quarantined[0].read_bytes(), b"preserve attacker certificate file",
                        )
                    else:
                        self.assertTrue(quarantined[0].is_dir())
                        self.assertEqual(
                            (quarantined[0] / "attacker.txt").read_text(encoding="utf-8"),
                            "preserve attacker certificate directory",
                        )
                    self.assertFalse(any(
                        path.name.startswith(f".{certificate.name}.staging-")
                        for path in root.iterdir()
                    ))

    def test_certificate_commit_uncertainty_with_restored_stage_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, contract, lock = _fixture(root)
            build_a, build_b = root / "build_A", root / "build_B"
            for build_id, output in (("A", build_a), ("B", build_b)):
                build_unified_release(
                    release_id="fixture-release", build_id=build_id, output_root=output,
                    contract_path=contract, source_lock_path=lock,
                    source_roots={"fixture": source},
                )
            certificate = root / "AB_RESTORED_STAGE.json"
            real_rename = unified_publish._rename_noreplace
            committed = False

            def commit_restore_stage_replace_then_raise(
                parent_fd: int, source_name: str, target_name: str,
            ) -> None:
                nonlocal committed
                if target_name == certificate.name and not committed:
                    real_rename(parent_fd, source_name, target_name)
                    os.rename(
                        target_name, source_name,
                        src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
                    )
                    os.mkdir(target_name, mode=0o755, dir_fd=parent_fd)
                    (certificate / "attacker.txt").write_text(
                        "preserve restored-stage certificate replacement", encoding="utf-8",
                    )
                    committed = True
                    raise UnifiedReportingError(
                        "injected exception after certificate commit and stage restoration"
                    )
                real_rename(parent_fd, source_name, target_name)

            with patch.object(
                unified_publish, "_rename_noreplace",
                side_effect=commit_restore_stage_replace_then_raise,
            ):
                with self.assertRaisesRegex(
                    UnifiedReportingError,
                    "injected exception after certificate commit and stage restoration",
                ):
                    verify_unified_release_ab(
                        build_a=build_a, build_b=build_b,
                        certificate_path=certificate, contract_path=contract,
                        source_lock_path=lock, source_roots={"fixture": source},
                    )
            self.assertTrue(committed)
            self.assertFalse(certificate.exists())
            quarantined = list(root.glob(f".{certificate.name}.invalid-*"))
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(
                (quarantined[0] / "attacker.txt").read_text(encoding="utf-8"),
                "preserve restored-stage certificate replacement",
            )
            staging_evidence = list(root.glob(f".{certificate.name}.staging-*"))
            self.assertEqual(len(staging_evidence), 1)
            staged_payload = json.loads(staging_evidence[0].read_text(encoding="utf-8"))
            self.assertEqual(staged_payload["status"], "PASS")

    def test_replaced_certificate_target_is_type_agnostically_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, contract, lock = _fixture(root)
            build_a, build_b = root / "build_A", root / "build_B"
            for build_id, output in (("A", build_a), ("B", build_b)):
                build_unified_release(
                    release_id="fixture-release", build_id=build_id, output_root=output,
                    contract_path=contract, source_lock_path=lock,
                    source_roots={"fixture": source},
                )
            certificate = root / "AB_REPLACED.json"
            displaced = root / "DISPLACED_GENUINE_CERT.json"
            real_rename = unified_publish._rename_noreplace
            replaced = False

            def replace_after_certificate_rename(
                parent_fd: int, source_name: str, target_name: str,
            ) -> None:
                nonlocal replaced
                if target_name == certificate.name and not replaced:
                    real_rename(parent_fd, source_name, target_name)
                    os.rename(
                        target_name, displaced.name,
                        src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
                    )
                    os.mkdir(target_name, mode=0o755, dir_fd=parent_fd)
                    (certificate / "unrelated.txt").write_text(
                        "preserve me", encoding="utf-8",
                    )
                    replaced = True
                    return
                real_rename(parent_fd, source_name, target_name)

            with patch.object(
                unified_publish, "_rename_noreplace",
                side_effect=replace_after_certificate_rename,
            ):
                with self.assertRaisesRegex(
                    UnifiedReportingError, "published certificate changed identity",
                ):
                    verify_unified_release_ab(
                        build_a=build_a, build_b=build_b,
                        certificate_path=certificate, contract_path=contract,
                        source_lock_path=lock, source_roots={"fixture": source},
                    )
            self.assertTrue(replaced)
            self.assertFalse(certificate.exists())
            displaced_payload = json.loads(displaced.read_text(encoding="utf-8"))
            self.assertEqual(displaced_payload["status"], "PASS")
            quarantined = list(root.glob(f".{certificate.name}.invalid-*"))
            self.assertEqual(len(quarantined), 1)
            self.assertTrue(quarantined[0].is_dir())
            self.assertEqual(
                (quarantined[0] / "unrelated.txt").read_text(encoding="utf-8"),
                "preserve me",
            )

    def test_release_hardlinks_fail_at_capture_and_certificate_end_rescan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, contract, lock = _fixture(root)
            build_a, build_b = root / "build_A", root / "build_B"
            for build_id, output in (("A", build_a), ("B", build_b)):
                build_unified_release(
                    release_id="fixture-release", build_id=build_id, output_root=output,
                    contract_path=contract, source_lock_path=lock,
                    source_roots={"fixture": source},
                )

            capture_link = root / "OUTSIDE_CAPTURE_HARDLINK.json"
            capture_source = build_a / "REPORT_CONTRACT.json"
            os.link(capture_source, capture_link)
            with self.assertRaisesRegex(UnifiedReportingError, "exactly one hard link"):
                verify_completed_release(build_a)
            self.assertTrue(capture_link.samefile(capture_source))
            capture_link.unlink()
            verify_completed_release(build_a)

            certificate = root / "AB_LATE_HARDLINK.json"
            late_link = root / "OUTSIDE_LATE_HARDLINK.csv"
            late_source = build_b / "tables/metrics.csv"
            real_rename = unified_publish._rename_noreplace
            injected = False

            def hardlink_inside_certificate_rename(
                parent_fd: int, source_name: str, target_name: str,
            ) -> None:
                nonlocal injected
                if target_name == certificate.name and not injected:
                    injected = True
                    os.link(late_source, late_link)
                real_rename(parent_fd, source_name, target_name)

            with patch.object(
                unified_publish, "_rename_noreplace",
                side_effect=hardlink_inside_certificate_rename,
            ):
                with self.assertRaisesRegex(
                    UnifiedReportingError, "held release file inode changed",
                ):
                    verify_unified_release_ab(
                        build_a=build_a, build_b=build_b,
                        certificate_path=certificate, contract_path=contract,
                        source_lock_path=lock, source_roots={"fixture": source},
                    )
            self.assertTrue(injected)
            self.assertFalse(certificate.exists())
            self.assertTrue(late_link.samefile(late_source))
            quarantined = list(root.glob(f".{certificate.name}.invalid-*"))
            self.assertEqual(len(quarantined), 1)
            late_link.unlink()
            verify_completed_release(build_a)
            verify_completed_release(build_b)

    def test_atomic_publish_failure_leaves_no_final_or_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, contract, lock = _fixture(root)
            final = root / "release"
            with patch.object(
                unified_publish, "_rename_noreplace",
                side_effect=UnifiedReportingError("injected no-replace failure"),
            ):
                with self.assertRaisesRegex(UnifiedReportingError, "injected no-replace failure"):
                    build_unified_release(
                        release_id="fixture-release", build_id="A", output_root=final,
                        contract_path=contract, source_lock_path=lock,
                        source_roots={"fixture": source},
                    )
            self.assertFalse(final.exists())
            self.assertFalse(any(path.name.startswith(".release.staging-") for path in root.iterdir()))


if __name__ == "__main__":
    unittest.main()
