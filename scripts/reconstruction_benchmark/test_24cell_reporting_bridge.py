#!/usr/bin/env python3
"""Synthetic boundary tests for the frozen-24 evaluator/reporting bridge."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
from html.parser import HTMLParser
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    canonical_tree_manifest,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.reporting_bridge import (  # noqa: E402
    GRAPH_DIAGNOSTIC_VERSION,
    GRAPH_EXAMPLE_RULE,
    GRAPH_EXAMPLE_SCHEMA,
    GRAPH_MANIFEST_SCHEMA,
    GRAPH_METHOD_IDS,
    GRAPH_PAYLOAD_SCHEMA,
    GRAPH_PLOT_SCHEMA,
    NONGRAPH_DIAGNOSTIC_METHOD_IDS,
    REQUIRED_GRAPH_PANELS_BY_METHOD,
    ReportingBridgeError,
    _auprc,
    _auroc,
    _expected_scopes,
    build_bridge_inputs,
    publish_bridge_inputs,
)
from spectral_utils.reconstruction_reporting.io import read_tidy_csv  # noqa: E402
from spectral_utils.reconstruction_reporting.report import default_plot_manifest, render_report  # noqa: E402
from spectral_utils.reconstruction_reporting.registry import (  # noqa: E402
    expected_coverage_rows,
    validate_result_references,
)
from spectral_utils.reconstruction_reporting.schemas import (  # noqa: E402
    SchemaError,
    validate_equal_unit_aggregates,
    validate_expected_coverage,
)


CELL_REGISTRY = REPO / "configs" / "reconstruction_benchmark_v1" / "frozen24_cells.json"
METHOD_REGISTRY = REPO / "configs" / "reconstruction_benchmark_v1" / "methods.json"
FEATURE_REGISTRY = REPO / "configs" / "reconstruction_benchmark_v1" / "feature_contract.json"


class _ReportStructure(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.class_counts: dict[str, int] = {}

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        for name in str(values.get("class", "")).split():
            self.class_counts[name] = self.class_counts.get(name, 0) + 1


def _hash(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


def _interval(point: float) -> dict:
    return {
        "bootstrap_draws_requested": 20_000,
        "bootstrap_draws_valid": 20_000,
        "ci_lower": max(0.0, point - 0.05),
        "bootstrap_median": point,
        "ci_upper": min(1.0, point + 0.05),
        "ci_level": 0.95,
        "ci_quantile_rule": "linear_type7",
    }


def _with_payload(value: dict) -> dict:
    output = copy.deepcopy(value)
    output.pop("payload_sha256", None)
    output["payload_sha256"] = sha256_bytes(canonical_json_bytes(output))
    return output


def _publish_fixture(root: Path) -> dict[str, Path]:
    cells_config = json.loads(CELL_REGISTRY.read_text(encoding="utf-8"))
    methods_config = json.loads(METHOD_REGISTRY.read_text(encoding="utf-8"))
    cell_rows = cells_config["cells"]
    method_rows = methods_config["methods"]
    cell_ids = [row["cell_id"] for row in cell_rows]
    method_ids = [row["method_id"] for row in method_rows]
    versions = {row["method_id"]: row["method_version_id"] for row in method_rows}
    row_meta = {row["cell_id"]: row for row in cell_rows}
    y_error = np.asarray([0, 1, 0, 1], dtype="<i1")
    scores = np.asarray([0.1, 0.8, 0.7, 0.6], dtype="<f8")
    point = {"auroc": _auroc(y_error, scores), "auprc": _auprc(y_error, scores)}

    arrays: dict[str, np.ndarray] = {}
    fit_outcomes = []
    labels = []
    cell_metrics = []
    for cell_index, cell_id in enumerate(cell_ids):
        row_ids = np.asarray([f"{cell_id}::row::{index}" for index in range(4)])
        group_ids = np.asarray([f"{cell_id}::question::0", f"{cell_id}::question::0", f"{cell_id}::question::1", f"{cell_id}::question::1"])
        arrays[f"{cell_id}__row_ids"] = row_ids
        arrays[f"{cell_id}__group_ids"] = group_ids
        arrays[f"{cell_id}__y_error"] = y_error
        labels.append({
            "cell_id": cell_id,
            "n_rows": 4,
            "n_correct": 2,
            "n_error": 2,
            "error_prevalence": 0.5,
            "y_correct_sha256": _hash("unused-correct-" + cell_id),
            "y_error_sha256": sha256_bytes(y_error.tobytes(order="C")),
            "conversion": "y_error=1-y_correct",
        })
        for method_id in method_ids:
            arrays[f"{cell_id}__{method_id}__score"] = scores
            fallback = cell_index == 0 and method_id == "family_nrm_a"
            fit_outcomes.append({
                "cell_id": cell_id,
                "method_id": method_id,
                "method_version_id": versions[method_id],
                "fit_status": "OK_FALLBACK" if fallback else "OK",
                "fallback_used": fallback,
                "fallback_reason": "synthetic registered fallback" if fallback else None,
                "score_file_sha256": _hash(f"score-file::{cell_id}::{method_id}"),
                "prepared_matrix_sha256": _hash(f"matrix::{cell_id}"),
            })
            meta = row_meta[cell_id]
            for metric in ("auroc", "auprc"):
                cell_metrics.append({
                    "status": "OK",
                    "population_id": "frozen24_response_v1",
                    "cell_id": cell_id,
                    "domain": meta["domain"],
                    "dataset_id": meta["dataset_id"],
                    "dataset_family": meta["dataset_family"],
                    "model_id": meta["model_id"],
                    "model_family": meta["model_family"],
                    "method_id": method_id,
                    "method_version_id": versions[method_id],
                    "metric": metric,
                    "estimate": point[metric],
                    "n_rows": 4,
                    "n_groups": 2,
                    "positive_class": "incorrect",
                    "score_semantics": "higher_is_incorrect",
                    **_interval(point[metric]),
                })

    scopes = _expected_scopes(cell_rows)
    aggregates = []
    contrasts = []
    for scope in scopes:
        for method_id in method_ids:
            for metric in ("auroc", "auprc"):
                aggregates.append({
                    "status": "OK",
                    "population_id": "frozen24_response_v1",
                    "scope_type": scope["scope_type"],
                    "scope_value": scope["scope_value"],
                    "cell_ids": list(scope["cell_ids"]),
                    "n_cells": len(scope["cell_ids"]),
                    "aggregation": "equal_cell_mean",
                    "headline_eligible": scope["scope_type"] == "macro24",
                    "method_id": method_id,
                    "method_version_id": versions[method_id],
                    "metric": metric,
                    "estimate": point[metric],
                    **_interval(point[metric]),
                })
        for candidate in method_ids:
            if candidate == "iu_pcr":
                continue
            for metric in ("auroc", "auprc"):
                contrasts.append({
                    "status": "OK",
                    "population_id": "frozen24_response_v1",
                    "scope_type": scope["scope_type"],
                    "scope_value": scope["scope_value"],
                    "cell_ids": list(scope["cell_ids"]),
                    "n_cells": len(scope["cell_ids"]),
                    "aggregation": "equal_cell_mean_of_paired_deltas",
                    "reference_method_id": "iu_pcr",
                    "candidate_method_id": candidate,
                    "metric": metric,
                    "delta": 0.0,
                    "wins": 0,
                    "ties": len(scope["cell_ids"]),
                    "losses": 0,
                    "tie_tolerance": 1e-12,
                    "bootstrap_probability_delta_positive": 0.0,
                    "bootstrap_draws_requested": 20_000,
                    "bootstrap_draws_valid": 20_000,
                    "ci_lower": 0.0,
                    "bootstrap_median": 0.0,
                    "ci_upper": 0.0,
                    "ci_level": 0.95,
                    "ci_quantile_rule": "linear_type7",
                })

    provenance = {
        "cell_registry_sha256": sha256_file(CELL_REGISTRY),
        "method_registry_sha256": sha256_file(METHOD_REGISTRY),
        "label_bundle_sha256": _hash("label-bundle"),
        "group_manifest_sha256": _hash("group-manifest"),
        "score_ab_verification_sha256": _hash("ab"),
        "freeze_A_sha256": _hash("freeze-a"),
        "freeze_B_sha256": _hash("freeze-b"),
        "input_manifest_A_sha256": _hash("input-a"),
        "input_manifest_B_sha256": _hash("input-b"),
        "evaluation_module_sha256": _hash("evaluator-module"),
        "numpy_version": np.__version__,
        "labels_opened": True,
        "verified_cell_method_pairs": 312,
    }
    macro_headline = [
        row for row in aggregates
        if row["scope_type"] == "macro24" and row["metric"] == "auroc"
    ]
    evaluation = _with_payload({
        "schema_version": "reconstruction-24cell-evaluation-v1",
        "status": "OK",
        "headline_status": "OK",
        "population_id": "frozen24_response_v1",
        "positive_class": "incorrect",
        "label_conversion": "y_error=1-y_correct",
        "score_semantics": "higher_is_incorrect",
        "metric_definitions": {
            "auroc": "weighted Mann-Whitney AUROC with half credit for score ties",
            "auprc": "weighted non-interpolated average precision (sklearn average_precision convention)",
        },
        "n_cells": 24,
        "n_methods": 13,
        "method_ids": method_ids,
        "reference_method_id": "iu_pcr",
        "bootstrap": {
            "draws": 20_000,
            "canonical_draw_count": 20_000,
            "minimum_valid_fraction": 0.95,
            "minimum_valid_draws": 19_000,
            "base_seed": 20_260_824,
            "rng": "numpy.PCG64",
            "resampling_unit": "verified_source_group_within_cell",
            "shared_draws": "synthetic shared draws",
            "aggregate_rule": "synthetic same-index equal-cell mean",
            "inference_boundary": "synthetic fixture",
            "cell_draw_manifests": [],
        },
        "provenance": provenance,
        "fit_outcomes": fit_outcomes,
        "label_provenance": labels,
        "cell_metrics": cell_metrics,
        "aggregate_metrics": aggregates,
        "paired_contrasts_vs_iu_pcr": contrasts,
        "headline_macro24_auroc": macro_headline,
    })

    evaluation_dir = root / "evaluation"
    evaluation_dir.mkdir(parents=True)
    evaluation_path = evaluation_dir / "EVALUATION.json"
    bootstrap_path = evaluation_dir / "BOOTSTRAP_DRAWS.npz"
    snapshot_path = evaluation_dir / "PREDICTION_SNAPSHOT.npz"
    evaluation_sha = atomic_write_json(evaluation_path, evaluation)
    bootstrap_sha = atomic_write_npz(bootstrap_path, {"synthetic": np.asarray([1], dtype=np.int8)})
    snapshot_sha = atomic_write_npz(snapshot_path, arrays)
    manifest = _with_payload({
        "schema_version": "reconstruction-evaluation-manifest-v1",
        "status": "OK",
        "headline_status": "OK",
        "population_id": "frozen24_response_v1",
        "n_cells": 24,
        "n_methods": 13,
        "bootstrap_draws": 20_000,
        "canonical_bootstrap_draws": 20_000,
        "evaluation_path": evaluation_path.name,
        "evaluation_sha256": evaluation_sha,
        "bootstrap_path": bootstrap_path.name,
        "bootstrap_sha256": bootstrap_sha,
        "prediction_snapshot_path": snapshot_path.name,
        "prediction_snapshot_sha256": snapshot_sha,
        "prediction_snapshot_schema": "reconstruction-prediction-snapshot-v1",
        "evaluator_cli_sha256": _hash("evaluator-cli"),
        "input_provenance": provenance,
    })
    manifest_path = evaluation_dir / "EVALUATION_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    return {"evaluation_dir": evaluation_dir, "evaluation": evaluation_path, "manifest": manifest_path, "snapshot": snapshot_path}


def _rewrite_evaluation(paths: dict[str, Path], evaluation: dict) -> None:
    evaluation = _with_payload(evaluation)
    evaluation_sha = atomic_write_json(paths["evaluation"], evaluation)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["evaluation_sha256"] = evaluation_sha
    manifest["status"] = evaluation["status"]
    manifest["headline_status"] = evaluation["headline_status"]
    manifest["input_provenance"] = evaluation["provenance"]
    atomic_write_json(paths["manifest"], _with_payload(manifest))


def _text(values) -> np.ndarray:
    values = [str(value) for value in values]
    width = max([1] + [len(value) for value in values])
    return np.asarray(values, dtype=f"<U{width}")


def _publish_graph_fixture(
    root: Path,
    paths: dict[str, Path],
    *,
    unavailable_nuisance_method: str | None = None,
) -> Path:
    """Publish a small but structurally complete signed graph package."""

    source_root = root / "synthetic_sources"
    source_root.mkdir()
    common_files = {
        "score_ab_verification": (root / "SCORE_AB_VERIFICATION.json", {"kind": "ab"}),
        "score_freeze": (root / "build_A" / "fit" / "SCORE_FREEZE_MANIFEST.json", {"kind": "freeze"}),
        "input_manifest": (root / "build_A" / "inputs" / "MANIFEST.json", {"kind": "input"}),
        "prepared": (source_root / "prepared.json", {"kind": "prepared"}),
        "score_record": (source_root / "RESULT.json", {"kind": "score-record"}),
        "artifact_index": (source_root / "ARTIFACT_INDEX.json", {"kind": "index"}),
    }
    hashes = {}
    for name, (path, payload) in common_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        hashes[name] = atomic_write_json(path, payload)
    score_path = source_root / "SCORES.npz"
    artifact_path = source_root / "artifacts.npz"
    hashes["score"] = atomic_write_npz(score_path, {"score": np.asarray([0.1, 0.8, 0.7, 0.6])})
    hashes["artifact"] = atomic_write_npz(artifact_path, {"synthetic": np.asarray([1], dtype=np.int8)})

    evaluation = json.loads(paths["evaluation"].read_text())
    evaluation["provenance"]["score_ab_verification_sha256"] = hashes["score_ab_verification"]
    evaluation["provenance"]["freeze_A_sha256"] = hashes["score_freeze"]
    evaluation["provenance"]["input_manifest_A_sha256"] = hashes["input_manifest"]
    for row in evaluation["fit_outcomes"]:
        row["score_file_sha256"] = hashes["score"]
    _rewrite_evaluation(paths, evaluation)
    evaluation = json.loads(paths["evaluation"].read_text())
    evaluation_manifest_hash = sha256_file(paths["manifest"])

    cells = json.loads(CELL_REGISTRY.read_text())["cells"]
    methods = json.loads(METHOD_REGISTRY.read_text())["methods"]
    versions = {row["method_id"]: row["method_version_id"] for row in methods}
    fit = {(row["cell_id"], row["method_id"]): row for row in evaluation["fit_outcomes"]}
    release_paths = {
        "score_ab_verification_path": "SCORE_AB_VERIFICATION.json",
        "score_ab_verification_sha256": hashes["score_ab_verification"],
        "score_freeze_A_path": "build_A/fit/SCORE_FREEZE_MANIFEST.json",
        "score_freeze_A_sha256": hashes["score_freeze"],
        "input_manifest_A_path": "build_A/inputs/MANIFEST.json",
        "input_manifest_A_sha256": hashes["input_manifest"],
        "evaluation_manifest_path": "evaluation/EVALUATION_MANIFEST.json",
        "evaluation_manifest_sha256": evaluation_manifest_hash,
        "evaluation_path": "evaluation/EVALUATION.json",
        "evaluation_sha256": sha256_file(paths["evaluation"]),
        "prediction_snapshot_path": "evaluation/PREDICTION_SNAPSHOT.npz",
        "prediction_snapshot_sha256": sha256_file(paths["snapshot"]),
        "raw_label_bundle_opened": False,
        "targets_source": "hashed evaluator prediction snapshot only",
    }

    producer = REPO / "scripts" / "reconstruction_benchmark" / "build_24cell_graph_diagnostics.py"
    module = REPO / "spectral_utils" / "reconstruction_benchmark" / "graph_diagnostics.py"
    source_files = sorted(
        (
            {"path": producer.relative_to(REPO).as_posix(), "sha256": sha256_file(producer)},
            {"path": module.relative_to(REPO).as_posix(), "sha256": sha256_file(module)},
        ),
        key=lambda row: row["path"].encode("utf-8"),
    )
    producer_snapshot = {
        "schema_version": "graph-diagnostics-source-environment-snapshot-v1",
        "git_head": "a" * 40,
        "git_status_porcelain": "",
        "source_files": source_files,
        "environment": {
            "python_version": "synthetic",
            "python_implementation": "synthetic",
            "python_executable": "synthetic",
            "platform": "synthetic",
            "machine": "synthetic",
            "packages": {},
        },
    }
    producer_snapshot["snapshot_sha256"] = sha256_bytes(canonical_json_bytes(producer_snapshot))

    bindings = []
    binding_by_pair = {}
    records = []
    graph_methods = set(GRAPH_METHOD_IDS)
    diagnostic_methods = GRAPH_METHOD_IDS + NONGRAPH_DIAGNOSTIC_METHOD_IDS
    first_cell_id = cells[0]["cell_id"]
    unavailable_slots = 0
    for cell in cells:
        cell_id = cell["cell_id"]
        feature_hash = _hash(f"features::{cell_id}")
        for method_id in diagnostic_methods:
            body = {
                "binding_type": "single_method_artifact",
                "cell_id": cell_id,
                "method_id": method_id,
                "method_version_id": versions[method_id],
                "feature_matrix_sha256": feature_hash,
                "prepared_matrix_sha256": fit[(cell_id, method_id)]["prepared_matrix_sha256"],
                "prepared_artifact_path": "synthetic_sources/prepared.json",
                "prepared_artifact_sha256": hashes["prepared"],
                "score_record_path": "synthetic_sources/RESULT.json",
                "score_record_sha256": hashes["score_record"],
                "score_path": "synthetic_sources/SCORES.npz",
                "score_sha256": hashes["score"],
                "method_artifact_path": "synthetic_sources/artifacts.npz",
                "method_artifact_sha256": hashes["artifact"],
                "artifact_index_path": "synthetic_sources/ARTIFACT_INDEX.json",
                "artifact_index_sha256": hashes["artifact_index"],
                "score_freeze_A_path": release_paths["score_freeze_A_path"],
                "score_freeze_A_sha256": release_paths["score_freeze_A_sha256"],
                "score_ab_verification_path": release_paths["score_ab_verification_path"],
                "score_ab_verification_sha256": release_paths["score_ab_verification_sha256"],
                "evaluation_manifest_path": release_paths["evaluation_manifest_path"],
                "evaluation_manifest_sha256": release_paths["evaluation_manifest_sha256"],
                "prediction_snapshot_path": release_paths["prediction_snapshot_path"],
                "prediction_snapshot_sha256": release_paths["prediction_snapshot_sha256"],
                "producer_snapshot_sha256": producer_snapshot["snapshot_sha256"],
            }
            binding_id = "binding_" + sha256_bytes(canonical_json_bytes(body))[:20]
            bindings.append({"source_binding_id": binding_id, **body})
            binding_by_pair[(cell_id, method_id)] = binding_id
            graph_hash = _hash(f"graph::{cell_id}::{method_id}") if method_id in graph_methods else None
            operator_hash = _hash(f"operator::{cell_id}::{method_id}") if method_id in graph_methods else None
            primary_panel = "graph_health" if method_id in graph_methods else REQUIRED_GRAPH_PANELS_BY_METHOD[method_id][0]
            identity = {
                "diagnostic_version": GRAPH_DIAGNOSTIC_VERSION,
                "scope_type": "cell",
                "scope_value": cell_id,
                "cell_id": cell_id,
                "method_id": method_id,
                "method_version_id": versions[method_id],
                "compared_method_id": None,
                "compared_method_version_id": None,
                "stage": "target_free",
                "panel_id": primary_panel,
                "metric_id": "n_edges" if method_id in graph_methods else "available_components",
                "series_id": "observed",
                "x_index": 0,
                "x_value": 0.0,
                "null_id": None,
                "seed": None,
                "draw_index": None,
                "feature_matrix_sha256": feature_hash,
                "graph_sha256": graph_hash,
                "operator_sha256": operator_hash,
                "compared_graph_sha256": None,
                "compared_operator_sha256": None,
                "source_binding_id": binding_id,
            }
            records.append({
                "diagnostic_id": "diag_" + sha256_bytes(canonical_json_bytes(identity))[:24],
                **identity,
                "status": "OK",
                "value": 2.0 if method_id in graph_methods else 1.0,
                "unit": "undirected_edges" if method_id in graph_methods else "count",
                "note": "synthetic signed diagnostic",
            })
            if method_id in graph_methods:
                relation_identity = {
                    **identity,
                    "stage": "post_freeze",
                    "panel_id": "alignment_vs_improvement",
                    "metric_id": "published_cell_auroc_delta_vs_iu_pcr",
                    "series_id": "cell_relation",
                    "x_value": 0.1 + 0.001 * len(records),
                }
                records.append({
                    "diagnostic_id": "diag_" + sha256_bytes(canonical_json_bytes(relation_identity))[:24],
                    **relation_identity,
                    "status": "OK",
                    "value": 0.01,
                    "unit": "AUROC_delta_copied_from_evaluation",
                    "note": "copied from the frozen evaluator; synthetic fixture",
                })
            reserved_panels = {primary_panel}
            if method_id in graph_methods:
                reserved_panels.add("alignment_vs_improvement")
            if method_id == "dufs_liu" and cell_id == first_cell_id:
                reserved_panels.update(("target_vs_nuisance_roughness", "node_permutation_null", "roughness_null_summary"))
            post_freeze_panels = {
                "target_vs_nuisance_roughness",
                "node_permutation_null",
                "roughness_null_summary",
                "alignment_vs_improvement",
                "length_only_graph_control",
                "random_family_graph_control",
            }
            for panel_id in REQUIRED_GRAPH_PANELS_BY_METHOD[method_id]:
                if panel_id in reserved_panels:
                    continue
                unavailable_identity = {
                    **identity,
                    "stage": "post_freeze" if panel_id in post_freeze_panels else "target_free",
                    "panel_id": panel_id,
                    "metric_id": "diagnostic_available",
                    "series_id": "observed",
                    "x_index": 0,
                    "x_value": 0.0,
                    "null_id": None,
                    "seed": None,
                    "draw_index": None,
                }
                records.append({
                    "diagnostic_id": "diag_" + sha256_bytes(canonical_json_bytes(unavailable_identity))[:24],
                    **unavailable_identity,
                    "status": "NOT_AVAILABLE_REQUESTED_ARTIFACT_MISSING",
                    "value": None,
                    "unit": "boolean",
                    "note": "synthetic explicit unavailable panel",
                })
                unavailable_slots += 1
    release_binding_by_method = {}
    ordered_cell_ids = sorted((cell["cell_id"] for cell in cells), key=lambda value: value.encode("utf-8"))
    for method_id in GRAPH_METHOD_IDS:
        multi_body = {
            "binding_type": "multi_cell_method_artifacts",
            "release_id": "synthetic-frozen24-reporting-v1",
            "method_id": method_id,
            "method_version_id": versions[method_id],
            "cell_source_bindings": [
                {"cell_id": cell_id, "source_binding_id": binding_by_pair[(cell_id, method_id)]}
                for cell_id in ordered_cell_ids
            ],
            "evaluation_manifest_sha256": release_paths["evaluation_manifest_sha256"],
            "prediction_snapshot_sha256": release_paths["prediction_snapshot_sha256"],
            "producer_snapshot_sha256": producer_snapshot["snapshot_sha256"],
        }
        multi_binding_id = "binding_" + sha256_bytes(canonical_json_bytes(multi_body))[:20]
        bindings.append({"source_binding_id": multi_binding_id, **multi_body})
        release_binding_by_method[method_id] = multi_binding_id

    for method_id in GRAPH_METHOD_IDS:
        aggregate_feature_hash = sha256_bytes(canonical_json_bytes([
            {"cell_id": cell_id, "feature_matrix_sha256": _hash(f"features::{cell_id}")}
            for cell_id in ordered_cell_ids
        ]))
        aggregate_graph_hash = sha256_bytes(canonical_json_bytes([
            {
                "cell_id": cell_id,
                "graph_sha256": _hash(f"graph::{cell_id}::{method_id}"),
            }
            for cell_id in ordered_cell_ids
        ]))
        aggregate_operator_hash = sha256_bytes(canonical_json_bytes([
            {
                "cell_id": cell_id,
                "operator_sha256": _hash(f"operator::{cell_id}::{method_id}"),
            }
            for cell_id in ordered_cell_ids
        ]))
        identity = {
            "diagnostic_version": GRAPH_DIAGNOSTIC_VERSION,
            "scope_type": "release",
            "scope_value": "synthetic-frozen24-reporting-v1",
            "cell_id": "__release__",
            "method_id": method_id,
            "method_version_id": versions[method_id],
            "compared_method_id": None,
            "compared_method_version_id": None,
            "stage": "post_freeze",
            "panel_id": "alignment_vs_improvement_summary",
            "metric_id": "spearman_error_alignment_vs_auroc_delta",
            "series_id": "descriptive_relation",
            "x_index": 0,
            "x_value": 0.0,
            "null_id": None,
            "seed": None,
            "draw_index": None,
            "feature_matrix_sha256": aggregate_feature_hash,
            "graph_sha256": aggregate_graph_hash,
            "operator_sha256": aggregate_operator_hash,
            "compared_graph_sha256": None,
            "compared_operator_sha256": None,
            "source_binding_id": release_binding_by_method[method_id],
        }
        for index, metric_id in enumerate((
            "spearman_error_alignment_vs_auroc_delta",
            "pearson_error_alignment_vs_auroc_delta",
        )):
            metric_identity = {**identity, "metric_id": metric_id, "x_index": index, "x_value": float(index)}
            records.append({
                "diagnostic_id": "diag_" + sha256_bytes(canonical_json_bytes(metric_identity))[:24],
                **metric_identity,
                "status": "OK",
                "value": 0.2,
                "unit": "correlation_across_24_cells",
                "note": "descriptive across-cell association; not a causal or inferential claim",
            })
    null_values = [0.5 + draw / 100.0 for draw in range(32)]
    base_roughness_identity = {
        "diagnostic_version": GRAPH_DIAGNOSTIC_VERSION,
        "scope_type": "cell",
        "scope_value": first_cell_id,
        "cell_id": first_cell_id,
        "method_id": "dufs_liu",
        "method_version_id": versions["dufs_liu"],
        "compared_method_id": None,
        "compared_method_version_id": None,
        "stage": "post_freeze",
        "panel_id": "target_vs_nuisance_roughness",
        "metric_id": "error_label_roughness",
        "series_id": "observed",
        "x_index": 0,
        "x_value": 0.0,
        "null_id": None,
        "seed": None,
        "draw_index": None,
        "feature_matrix_sha256": _hash(f"features::{first_cell_id}"),
        "graph_sha256": _hash(f"graph::{first_cell_id}::dufs_liu"),
        "operator_sha256": _hash(f"operator::{first_cell_id}::dufs_liu"),
        "compared_graph_sha256": None,
        "compared_operator_sha256": None,
        "source_binding_id": binding_by_pair[(first_cell_id, "dufs_liu")],
    }
    records.append({
        "diagnostic_id": "diag_" + sha256_bytes(canonical_json_bytes(base_roughness_identity))[:24],
        **base_roughness_identity,
        "status": "OK",
        "value": 0.2,
        "unit": "centered_normalized_laplacian_rayleigh_quotient",
        "note": "synthetic observed roughness",
    })
    for draw, null_value in enumerate(null_values):
        identity = {
            **base_roughness_identity,
            "panel_id": "node_permutation_null",
            "series_id": "error_label",
            "x_index": draw,
            "x_value": float(draw),
            "null_id": "node_permutation_fixed_signal_v1",
            "seed": 10_000 + draw,
            "draw_index": draw,
            "graph_sha256": _hash(f"permuted-graph::{first_cell_id}::{draw}"),
            "operator_sha256": _hash(f"permuted-operator::{first_cell_id}::{draw}"),
        }
        records.append({
            "diagnostic_id": "diag_" + sha256_bytes(canonical_json_bytes(identity))[:24],
            **identity,
            "status": "OK",
            "value": null_value,
            "unit": "centered_normalized_laplacian_rayleigh_quotient",
            "note": None,
        })
    null_median = float(np.median(null_values))
    summary_identity = {
        **base_roughness_identity,
        "panel_id": "roughness_null_summary",
        "metric_id": "error_alignment_null_minus_real",
        "series_id": "error_label",
        "null_id": "node_permutation_fixed_signal_v1",
    }
    records.append({
        "diagnostic_id": "diag_" + sha256_bytes(canonical_json_bytes(summary_identity))[:24],
        **summary_identity,
        "status": "OK",
        "value": null_median - 0.2,
        "unit": "roughness_difference",
        "note": "positive null-minus-real means more graph alignment",
    })
    bindings.sort(key=lambda row: row["source_binding_id"].encode("utf-8"))
    records.sort(key=lambda row: row["diagnostic_id"].encode("utf-8"))
    selected_cell = first_cell_id
    selected = {method_id: selected_cell for method_id in sorted(graph_methods)}
    expected_panel_slots = 24 * sum(len(panels) for panels in REQUIRED_GRAPH_PANELS_BY_METHOD.values())
    payload = _with_payload({
        "schema_version": GRAPH_PAYLOAD_SCHEMA,
        "diagnostic_version": GRAPH_DIAGNOSTIC_VERSION,
        "release_id": "synthetic-frozen24-reporting-v1",
        "status": "OK",
        "scope": {
            "population_id": "frozen24_response_v1",
            "n_cells": 24,
            "graph_methods": list(GRAPH_METHOD_IDS),
            "non_graph_methods": list(NONGRAPH_DIAGNOSTIC_METHOD_IDS),
            "performance_metrics_recomputed": False,
            "raw_label_bundle_opened": False,
        },
        "example_selection": {
            "rule_id": GRAPH_EXAMPLE_RULE,
            "labels_used": False,
            "selected_cell_by_method": selected,
        },
        "coverage": {
            "coverage_axis": "cell_x_method_x_preregistered_panel",
            "expected_panel_slots": expected_panel_slots,
            "observed_panel_slots": expected_panel_slots,
            "explicit_unavailable_slots_added": unavailable_slots,
            "complete": True,
            "required_panels_by_method": {
                method_id: list(panels)
                for method_id, panels in REQUIRED_GRAPH_PANELS_BY_METHOD.items()
            },
        },
        "null_registry": {
            "node_permutation": {
                "null_id": "node_permutation_fixed_signal_v1",
                "draws_per_cell_method": 32,
            },
            "ca_alpha_controls": {
                "null_id": "ca_alpha_control_v1",
                "controls": ["learned", "equal_view", "provenance_prior", "global_mean_alpha", "permuted"],
            },
        },
        "provenance": release_paths,
        "producer_source_environment_snapshot": producer_snapshot,
        "source_bindings": bindings,
        "records": records,
    })

    graph_dir = root / "graph_diagnostics"
    graph_dir.mkdir()
    diagnostics_path = graph_dir / "GRAPH_DIAGNOSTICS.json"
    plot_path = graph_dir / "PLOT_DATA.npz"
    example_path = graph_dir / "EXAMPLE_GRAPH_DATA.npz"
    diagnostics_sha = atomic_write_json(diagnostics_path, payload)
    ok = [row for row in records if row["status"] == "OK"]
    text_fields = (
        "diagnostic_id", "scope_type", "scope_value", "cell_id", "method_id", "method_version_id", "stage",
        "compared_method_id", "compared_method_version_id", "panel_id", "metric_id",
        "series_id", "null_id", "feature_matrix_sha256", "graph_sha256",
        "operator_sha256", "compared_graph_sha256", "compared_operator_sha256", "source_binding_id",
    )
    plot_arrays = {}
    for field in text_fields:
        values = []
        for row in ok:
            value = row[field]
            if field in ("compared_method_id", "compared_method_version_id", "graph_sha256", "operator_sha256", "compared_graph_sha256", "compared_operator_sha256"):
                value = value or "not_applicable"
            elif field == "null_id":
                value = value or "observed"
            values.append(value)
        plot_arrays[field] = _text(values)
    plot_arrays.update({
        "x_index": np.asarray([row["x_index"] for row in ok], dtype="<i8"),
        "seed": _text([str(row["seed"] if row["seed"] is not None else -1) for row in ok]),
        "draw_index": np.asarray([row["draw_index"] if row["draw_index"] is not None else -1 for row in ok], dtype="<i8"),
        "x_value": np.asarray([row["x_value"] for row in ok], dtype="<f8"),
        "y_value": np.asarray([row["value"] for row in ok], dtype="<f8"),
        "schema_version": _text([GRAPH_PLOT_SCHEMA] * len(ok)),
        "diagnostic_version": _text([GRAPH_DIAGNOSTIC_VERSION] * len(ok)),
    })
    plot_sha = atomic_write_npz(plot_path, plot_arrays)
    with np.load(paths["snapshot"], allow_pickle=False) as snapshot_bundle:
        rows = np.asarray(snapshot_bundle[f"{selected_cell}__row_ids"])
        labels = np.asarray(snapshot_bundle[f"{selected_cell}__y_error"], dtype="<i1")
    example_arrays = {
        "schema_version": _text([GRAPH_EXAMPLE_SCHEMA]),
        "diagnostic_version": _text([GRAPH_DIAGNOSTIC_VERSION]),
        "selection_rule_id": _text([GRAPH_EXAMPLE_RULE]),
    }
    for method_id in ("dufs_liu", "ca_specrage_atomic", "pgrd_a"):
        prefix = method_id
        nuisance_available = method_id != unavailable_nuisance_method
        method_arrays = {
            f"{prefix}__cell_id": _text([selected_cell]),
            f"{prefix}__row_ids": _text(rows),
            f"{prefix}__embedding_x": np.asarray([0.0, 1.0, 2.0, 3.0]),
            f"{prefix}__embedding_y": np.asarray([0.0, 1.0, 0.0, 1.0]),
            f"{prefix}__y_error": labels,
            f"{prefix}__trace_length_available": np.asarray([nuisance_available], dtype=bool),
            f"{prefix}__edge_source": np.asarray([0, 1], dtype="<i8"),
            f"{prefix}__edge_target": np.asarray([1, 2], dtype="<i8"),
            f"{prefix}__edge_weight": np.asarray([1.0, 1.0]),
            f"{prefix}__feature_matrix_sha256": _text([_hash(f"features::{selected_cell}")]),
            f"{prefix}__graph_sha256": _text([_hash(f"graph::{selected_cell}::{method_id}")]),
            f"{prefix}__operator_sha256": _text([_hash(f"operator::{selected_cell}::{method_id}")]),
        }
        if nuisance_available:
            method_arrays[f"{prefix}__trace_length_coordinate"] = np.asarray([-1.0, -0.5, 0.5, 1.0])
        example_arrays.update(method_arrays)
    example_sha = atomic_write_npz(example_path, example_arrays)
    manifest = _with_payload({
        "schema_version": GRAPH_MANIFEST_SCHEMA,
        "diagnostic_version": GRAPH_DIAGNOSTIC_VERSION,
        "release_id": "synthetic-frozen24-reporting-v1",
        "status": "OK",
        "n_records": len(records),
        "n_source_bindings": len(bindings),
        "node_permutation_draws_per_cell_method": 32,
        "diagnostics_path": diagnostics_path.name,
        "diagnostics_sha256": diagnostics_sha,
        "diagnostics_payload_sha256": payload["payload_sha256"],
        "plot_data_path": plot_path.name,
        "plot_data_sha256": plot_sha,
        "example_graph_data_path": example_path.name,
        "example_graph_data_sha256": example_sha,
        "selected_examples": selected,
        "source_provenance": release_paths,
        "source_environment_snapshot": producer_snapshot,
        "source_environment_snapshot_sha256": producer_snapshot["snapshot_sha256"],
        "producer_path": producer.relative_to(REPO).as_posix(),
        "producer_sha256": sha256_file(producer),
        "diagnostics_module_path": module.relative_to(REPO).as_posix(),
        "diagnostics_module_sha256": sha256_file(module),
    })
    atomic_write_json(graph_dir / "GRAPH_DIAGNOSTICS_MANIFEST.json", manifest)
    atomic_write_json(graph_dir / "TREE_MANIFEST.json", canonical_tree_manifest(graph_dir))
    return graph_dir


class BridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = _publish_fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build(self):
        return build_bridge_inputs(
            evaluation_dir=self.paths["evaluation_dir"],
            release_id="synthetic-frozen24-reporting-v1",
            cell_registry_path=CELL_REGISTRY,
            method_registry_path=METHOD_REGISTRY,
            feature_registry_path=FEATURE_REGISTRY,
            allow_empty_graph_diagnostics=True,
        )

    def build_with_graph(self, graph_dir: Path):
        return build_bridge_inputs(
            evaluation_dir=self.paths["evaluation_dir"],
            release_id="synthetic-frozen24-reporting-v1",
            cell_registry_path=CELL_REGISTRY,
            method_registry_path=METHOD_REGISTRY,
            feature_registry_path=FEATURE_REGISTRY,
            graph_diagnostics_dir=graph_dir,
        )

    def test_complete_bridge_is_schema_valid_and_metric_specific_predictions_exist(self):
        inputs = self.build()
        self.assertEqual(len(inputs.registry["methods"]), 13)
        self.assertEqual(len(inputs.rows["predictions"]), 24 * 13 * 4 * 2)
        self.assertEqual(inputs.rows["graph_diagnostics"], ())
        self.assertEqual(inputs.rows["graph_examples"], ())
        first_cell = json.loads(CELL_REGISTRY.read_text())["cells"][0]["cell_id"]
        first_system = next(row for row in inputs.registry["systems"] if row["method_version_id"].startswith("equal-feature"))["system_id"]
        groups = {
            row["comparison_group_id"]
            for row in inputs.rows["predictions"]
            if row["cell_id"] == first_cell and row["system_id"] == first_system
        }
        self.assertEqual(len(groups), 2, "AUROC and AUPRC predictions must not share a forged comparison group")
        validate_result_references(inputs.registry, inputs.rows)
        validate_expected_coverage(expected_coverage_rows(inputs.registry), inputs.rows["coverage"])
        validate_equal_unit_aggregates(inputs.rows["metrics"], inputs.registry["aggregations"])

    def test_fit_fallback_is_preserved_in_predictions_and_coverage(self):
        inputs = self.build()
        first_cell = json.loads(CELL_REGISTRY.read_text())["cells"][0]["cell_id"]
        fallback_system = next(system["system_id"] for system in inputs.registry["systems"] if system["method_version_id"].startswith("family-nrm-a"))
        predictions = [row for row in inputs.rows["predictions"] if row["cell_id"] == first_cell and row["system_id"] == fallback_system]
        self.assertTrue(predictions and all(row["status"] == "OK_FALLBACK" and row["fallback_used"] for row in predictions))
        coverage = next(row for row in inputs.rows["coverage"] if row["cell_id"] == first_cell and row["system_id"] == fallback_system)
        self.assertEqual((coverage["fallback_n"], coverage["scored_n"]), (4, 4))

    def test_publication_is_immutable_and_graph_file_is_explicit_header_only(self):
        inputs = self.build()
        output = publish_bridge_inputs(self.root / "bridge", inputs)
        graph_rows = read_tidy_csv(output / "graph_diagnostics_long.csv", "graph_diagnostics")
        self.assertEqual(graph_rows, [])
        self.assertTrue((output / "BRIDGE_MANIFEST.json").is_file())
        with self.assertRaises(FileExistsError):
            publish_bridge_inputs(output, inputs)

    def test_manifest_hash_drift_is_rejected_before_semantic_read(self):
        with self.paths["evaluation"].open("ab") as handle:
            handle.write(b"drift")
        with self.assertRaisesRegex(ReportingBridgeError, "file hash drift"):
            self.build()

    def test_missing_fit_outcome_is_rejected_even_when_evaluation_is_rehashed(self):
        evaluation = json.loads(self.paths["evaluation"].read_text())
        evaluation["fit_outcomes"].pop()
        _rewrite_evaluation(self.paths, evaluation)
        with self.assertRaisesRegex(ReportingBridgeError, "fit outcomes roster drift"):
            self.build()

    def test_snapshot_point_metric_mismatch_is_rejected_even_when_files_are_rehashed(self):
        evaluation = json.loads(self.paths["evaluation"].read_text())
        evaluation["cell_metrics"][0]["estimate"] -= 0.1
        _rewrite_evaluation(self.paths, evaluation)
        with self.assertRaisesRegex(ReportingBridgeError, "point metric disagrees with snapshot"):
            self.build()

    def test_graph_artifact_is_required_without_explicit_nonpublication_opt_out(self):
        with self.assertRaisesRegex(ReportingBridgeError, "scientific reporting requires"):
            build_bridge_inputs(
                evaluation_dir=self.paths["evaluation_dir"],
                release_id="synthetic-frozen24-reporting-v1",
                cell_registry_path=CELL_REGISTRY,
                method_registry_path=METHOD_REGISTRY,
                feature_registry_path=FEATURE_REGISTRY,
            )

    def test_final_report_builder_requires_the_signed_scientific_bridge(self):
        script = REPO / "scripts" / "reconstruction_benchmark" / "build_reporting_release.py"
        spec = importlib.util.spec_from_file_location("scientific_report_builder", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        debug_dir = publish_bridge_inputs(self.root / "debug-bridge", self.build())
        debug_args = argparse.Namespace(
            bridge_manifest=debug_dir / "BRIDGE_MANIFEST.json",
            registry=debug_dir / "research_registry.json",
            predictions=debug_dir / "predictions.jsonl",
            metrics=debug_dir / "metrics_long.csv",
            contrasts=debug_dir / "contrasts_long.csv",
            coverage=debug_dir / "coverage_long.csv",
            graph_diagnostics=debug_dir / "graph_diagnostics_long.csv",
            graph_examples=debug_dir / "graph_examples_long.csv",
            plot_manifest=None,
        )
        with self.assertRaisesRegex(SchemaError, "ineligible for scientific publication"):
            module.load_and_validate(debug_args)

        graph_dir = _publish_graph_fixture(self.root, self.paths)
        scientific_dir = publish_bridge_inputs(
            self.root / "scientific-bridge",
            self.build_with_graph(graph_dir),
        )
        scientific_args = argparse.Namespace(
            bridge_manifest=scientific_dir / "BRIDGE_MANIFEST.json",
            registry=scientific_dir / "research_registry.json",
            predictions=scientific_dir / "predictions.jsonl",
            metrics=scientific_dir / "metrics_long.csv",
            contrasts=scientific_dir / "contrasts_long.csv",
            coverage=scientific_dir / "coverage_long.csv",
            graph_diagnostics=scientific_dir / "graph_diagnostics_long.csv",
            graph_examples=scientific_dir / "graph_examples_long.csv",
            plot_manifest=None,
        )
        registry, rows, plots = module.load_and_validate(scientific_args)
        self.assertEqual(registry["release_id"], "synthetic-frozen24-reporting-v1")
        self.assertTrue(rows["graph_diagnostics"] and rows["graph_examples"])
        self.assertTrue(plots["plots"])
        with scientific_args.metrics.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(SchemaError, "bridge artifact hash mismatch"):
            module.load_and_validate(scientific_args)

    def test_signed_graph_package_is_converted_and_auxiliary_sources_are_preserved(self):
        graph_dir = _publish_graph_fixture(self.root, self.paths)
        inputs = self.build_with_graph(graph_dir)
        self.assertEqual(
            {row["method_id"] for row in inputs.rows["graph_diagnostics"]},
            set(GRAPH_METHOD_IDS + NONGRAPH_DIAGNOSTIC_METHOD_IDS),
        )
        self.assertEqual(
            sum("alignment_vs_improvement_summary" in row["graph_variant"] for row in inputs.rows["graph_diagnostics"]),
            2 * len(GRAPH_METHOD_IDS),
        )
        self.assertGreater(len(inputs.rows["graph_diagnostics"]), 24 * len(REQUIRED_GRAPH_PANELS_BY_METHOD))
        self.assertTrue(
            any(
                row["diagnostic_label"].startswith(
                    "Fixed fitted-graph weight sensitivity under source-group resampling"
                )
                for row in inputs.rows["graph_diagnostics"]
            ),
        )
        self.assertEqual(len(inputs.rows["graph_examples"]), 3 * (4 + 2))
        self.assertTrue(inputs.source_provenance["scientific_publication_eligible"])
        first = next(row for row in inputs.rows["graph_diagnostics"] if row["method_id"] == "dufs_liu" and "panel=graph_health" in row["graph_variant"])
        self.assertEqual((first["n_nodes"], first["n_edges"], first["effect"]), (4, 2, 2.0))
        relation = next(row for row in inputs.rows["graph_diagnostics"] if "alignment_vs_improvement_summary" in row["graph_variant"])
        self.assertTrue(relation["cell_id"].startswith("aggregate::macro24"))
        self.assertEqual((relation["n_nodes"], relation["n_edges"]), (24 * 4, 24 * 2))
        roughness = next(
            row for row in inputs.rows["graph_diagnostics"]
            if row["method_id"] == "dufs_liu"
            and row["value"] is not None
            and "panel=target_vs_nuisance_roughness" in row["graph_variant"]
        )
        self.assertAlmostEqual(roughness["null_value"], float(np.median([0.5 + draw / 100.0 for draw in range(32)])))
        self.assertAlmostEqual(roughness["effect"], roughness["null_value"] - roughness["value"])
        self.assertEqual(roughness["permutation_count"], 32)
        self.assertFalse(any("node_permutation_null" in row["graph_variant"] for row in inputs.rows["graph_diagnostics"]))
        output = publish_bridge_inputs(self.root / "bridge-with-graphs", inputs)
        self.assertTrue((output / "graph_sources" / "PLOT_DATA.npz").is_file())
        self.assertTrue((output / "graph_sources" / "EXAMPLE_GRAPH_DATA.npz").is_file())
        self.assertEqual(len(read_tidy_csv(output / "graph_examples_long.csv", "graph_examples")), 18)
        manifest = json.loads((output / "BRIDGE_MANIFEST.json").read_text())
        copied = {row["path"]: row for row in manifest["artifacts"]}
        self.assertEqual(copied["graph_sources/PLOT_DATA.npz"]["file_sha256"], sha256_file(graph_dir / "PLOT_DATA.npz"))

    def test_graph_plot_tamper_is_rejected(self):
        graph_dir = _publish_graph_fixture(self.root, self.paths)
        with (graph_dir / "PLOT_DATA.npz").open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ReportingBridgeError, "plot data file hash drift|graph tree (hash|size) drift"):
            self.build_with_graph(graph_dir)

    def test_verified_graph_visuals_have_deterministic_browser_structure(self):
        graph_dir = _publish_graph_fixture(self.root, self.paths)
        inputs = self.build_with_graph(graph_dir)
        manifest = default_plot_manifest(inputs.registry["release_id"], inputs.rows)
        first = render_report(registry=inputs.registry, rows_by_table=inputs.rows, plot_manifest=manifest)
        second = render_report(registry=inputs.registry, rows_by_table=inputs.rows, plot_manifest=manifest)
        self.assertEqual(first, second)
        structure = _ReportStructure()
        structure.feed(first)
        self.assertIn("alignment-scatter-svg", structure.ids)
        self.assertIn("diagnostic-selector", structure.ids)
        self.assertEqual(structure.class_counts.get("graph-example"), 3)
        self.assertEqual(structure.class_counts.get("example-graph-svg"), 6)
        self.assertEqual(sum(plot["kind"] == "graph_embedding_pair" for plot in manifest["plots"]), 3)
        self.assertEqual(sum(plot["kind"] == "diagnostic_scatter" for plot in manifest["plots"]), 1)
        self.assertIn("Both panels use the identical frozen two-dimensional spectral embedding", first)
        self.assertIn("does not refit edges", first)

    def test_missing_nuisance_renders_explicit_unavailable_panel(self):
        graph_dir = _publish_graph_fixture(
            self.root,
            self.paths,
            unavailable_nuisance_method="dufs_liu",
        )
        inputs = self.build_with_graph(graph_dir)
        dufs_nodes = [
            row for row in inputs.rows["graph_examples"]
            if row["method_id"] == "dufs_liu" and row["row_kind"] == "node"
        ]
        self.assertTrue(dufs_nodes and all(not row["nuisance_available"] and row["nuisance_value"] is None for row in dufs_nodes))
        manifest = default_plot_manifest(inputs.registry["release_id"], inputs.rows)
        rendered = render_report(registry=inputs.registry, rows_by_table=inputs.rows, plot_manifest=manifest)
        self.assertIn("Trace-length nuisance coordinate unavailable", rendered)
        self.assertIn("No substitute feature was used", rendered)

    def test_opaque_bound_score_tamper_is_rejected(self):
        graph_dir = _publish_graph_fixture(self.root, self.paths)
        with (self.root / "synthetic_sources" / "SCORES.npz").open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ReportingBridgeError, "file SHA-256 mismatch"):
            self.build_with_graph(graph_dir)


if __name__ == "__main__":
    unittest.main()
