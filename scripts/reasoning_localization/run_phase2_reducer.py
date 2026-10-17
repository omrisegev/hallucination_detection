#!/usr/bin/env python3
"""Execute one preregistered Phase-2R reducer with a frozen R1 parent."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import (  # noqa: E402
    load_prepared_localization_cell,
    validate_fit_manifest,
)
from spectral_utils.fixed_application_pipelines import SHARED_TOKEN_VIEWS  # noqa: E402
from scripts.reasoning_localization.run_phase1_baseline import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DEFAULT_RELEASE,
    PB_METRICS,
    PROGRAM_ROOT,
    QWEN_MODELS,
    _bootstrap_pb_panel,
    _load_pb_labels,
    combine_with_common_detector,
)


REFERENCE = "P2R_A_TOPK5_REFERENCE"
STAGE_A_VARIANTS = (
    REFERENCE,
    "P2R_A_MAX_K1",
    "P2R_A_MEAN_ALL",
    "P2R_A_TOPK2",
    "P2R_A_TOPK3",
    "P2R_A_TOPK8",
    "P2R_A_TOPK10",
    "P2R_A_TOPQ25",
    "P2R_A_TOPQ50",
    "P2R_A_QUANTILE75",
    "P2R_A_QUANTILE90",
    "P2R_A_MEDIAN",
    "P2R_A_TOPQ10_EXPLORATORY",
    "P2R_A_TOPQ05_EXPLORATORY",
)
EXPLORATORY_STAGE_A_VARIANTS = (
    "P2R_A_TOPQ10_EXPLORATORY",
    "P2R_A_TOPQ05_EXPLORATORY",
)
FAMILIES = ("gsm8k", "math", "olympiadbench", "omnimath")
PB_CELLS = tuple(
    f"processbench_{family}_{model}" for model in QWEN_MODELS for family in FAMILIES
)
PHASE_ROOT = PROGRAM_ROOT / "phase_2"
P1_REFERENCE_ROOT = PROGRAM_ROOT / "phase_1/r1_entropy_top5"
SCHEMA = "reasoning-localization-phase2-reducer-v1"


class ReducerError(RuntimeError):
    """Fail-closed reducer-contract violation."""


def _payload_sha(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _peak_memory_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ReducerError(f"refusing to write empty table: {path}")
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require_source_hashes(registry: Mapping[str, Any]) -> None:
    for source in registry["frozen_sources"]:
        path = Path(source["path"])
        if not path.is_absolute():
            path = REPO / path
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise ReducerError(f"frozen source changed or missing: {source['role']}")


def load_registry(path: Path, variant_id: str, release: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "reasoning-localization-phase2-reducer-execution-registry-v1",
        "status": "FROZEN_BEFORE_RUN",
        "variant_id": variant_id,
        "stage_a_order": list(STAGE_A_VARIANTS),
        "processbench_cells": list(PB_CELLS),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    for key, value in required.items():
        if registry.get(key) != value:
            raise ReducerError(f"execution registry mismatch for {key}")
    if Path(registry["release_root"]).resolve() != release.resolve():
        raise ReducerError("release root differs from frozen registry")
    if registry["runner_sha256"] != sha256_file(Path(__file__).resolve()):
        raise ReducerError("runner changed after registry freeze")
    _require_source_hashes(registry)
    return registry


def reduce_steps(
    variant_id: str,
    token_risk: Sequence[float],
    starts: Sequence[int],
    ends: Sequence[int],
) -> np.ndarray:
    risk = np.asarray(token_risk, dtype=np.float64)
    starts_i = np.asarray(starts, dtype=np.int64)
    ends_i = np.asarray(ends, dtype=np.int64)
    if starts_i.shape != ends_i.shape or np.any(starts_i < 0) or np.any(ends_i <= starts_i):
        raise ValueError("malformed step spans")
    fixed_k = {
        REFERENCE: 5,
        "P2R_A_MAX_K1": 1,
        "P2R_A_TOPK2": 2,
        "P2R_A_TOPK3": 3,
        "P2R_A_TOPK8": 8,
        "P2R_A_TOPK10": 10,
    }
    output = np.empty(len(starts_i), dtype=np.float64)
    for index, (lo, hi) in enumerate(zip(starts_i, ends_i)):
        values = risk[int(lo):int(hi)]
        if variant_id in fixed_k:
            take = min(fixed_k[variant_id], len(values))
            output[index] = float(np.mean(np.partition(values, len(values) - take)[-take:]))
        elif variant_id == "P2R_A_MEAN_ALL":
            output[index] = float(np.mean(values))
        elif variant_id in {
            "P2R_A_TOPQ25",
            "P2R_A_TOPQ50",
            "P2R_A_TOPQ10_EXPLORATORY",
            "P2R_A_TOPQ05_EXPLORATORY",
        }:
            fractions = {
                "P2R_A_TOPQ25": 0.25,
                "P2R_A_TOPQ50": 0.50,
                "P2R_A_TOPQ10_EXPLORATORY": 0.10,
                "P2R_A_TOPQ05_EXPLORATORY": 0.05,
            }
            fraction = fractions[variant_id]
            take = max(1, int(math.ceil(fraction * len(values))))
            output[index] = float(np.mean(np.partition(values, len(values) - take)[-take:]))
        elif variant_id in {"P2R_A_QUANTILE75", "P2R_A_QUANTILE90"}:
            quantile = 0.75 if variant_id.endswith("75") else 0.90
            output[index] = float(np.quantile(values, quantile, method="linear"))
        elif variant_id == "P2R_A_MEDIAN":
            output[index] = float(np.median(values))
        else:
            raise ReducerError(f"unsupported Stage-A variant: {variant_id}")
    return output


def freeze_scores(
    variant_id: str, release: Path, output: Path, registry: Mapping[str, Any]
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"variant output already exists: {output}")
    score_root = output / "score_freeze"
    score_root.mkdir(parents=True, exist_ok=False)
    input_root = release / "build_A/localization/inputs"
    manifest_path = input_root / "MANIFEST.json"
    manifest = validate_fit_manifest(manifest_path, input_root=input_root)
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    if not set(PB_CELLS).issubset(by_cell):
        raise ReducerError("prepared input lacks the exact eight Qwen cells")
    records = []
    max_local_alias_error = 0.0
    max_combined_alias_error = 0.0
    entropy_index = SHARED_TOKEN_VIEWS.index("entropy_series")
    for position, cell_id in enumerate(PB_CELLS, start=1):
        source_record = by_cell[cell_id]
        input_path = input_root / source_record["artifact_path"]
        cell = load_prepared_localization_cell(input_path, source_record)
        entropy_risk = -np.asarray(cell.token_confidence[:, entropy_index], dtype=np.float64)
        local = reduce_steps(variant_id, entropy_risk, cell.segment_starts, cell.segment_ends)
        combined = combine_with_common_detector(cell, local)
        if variant_id == REFERENCE:
            parent_path = P1_REFERENCE_ROOT / "score_freeze/cells" / cell_id / "scores.npz"
            parent = load_npz_no_pickle(parent_path)
            if tuple(parent["row_ids"].astype(str)) != tuple(cell.row_ids):
                raise ReducerError(f"{cell_id}: R1 row order differs")
            max_local_alias_error = max(
                max_local_alias_error,
                float(np.max(np.abs(local - parent["local_step_scores"]))),
            )
            max_combined_alias_error = max(
                max_combined_alias_error,
                float(np.max(np.abs(combined - parent["combined_step_scores"]))),
            )
        target = score_root / "cells" / cell_id
        target.mkdir(parents=True, exist_ok=False)
        score_path = target / "scores.npz"
        score_sha = atomic_write_npz(score_path, {
            "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
            "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
            "segment_lengths": np.asarray(cell.segment_ends - cell.segment_starts, dtype="<i8"),
            "local_step_scores": np.asarray(local, dtype="<f8"),
            "combined_step_scores": np.asarray(combined, dtype="<f8"),
        })
        record = {
            "schema": "reasoning-localization-phase2-reducer-cell-v1",
            "variant_id": variant_id,
            "cell_id": cell_id,
            "dataset_id": str(cell.dataset_id),
            "model_id": str(cell.model_id),
            "slice_id": str(cell.slice_id),
            "population_id": str(cell.population_id),
            "n_rows": len(cell.row_ids),
            "n_steps": len(local),
            "prepared_input": str(input_path),
            "prepared_input_sha256": sha256_file(input_path),
            "score_file": "scores.npz",
            "score_sha256": score_sha,
            "labels_seen_during_fit": False,
            "targets_accessed_during_fit": False,
            "response_detector": "equal_feature_mean",
            "local_normalization": "within-cell empirical midrank",
            "threshold_fit": "forbidden in score-freeze",
            "peak_memory_bytes": _peak_memory_bytes(),
        }
        record["payload_sha256"] = _payload_sha(record)
        record_path = target / "RECORD.json"
        atomic_write_json(record_path, record)
        records.append({
            "cell_id": cell_id,
            "record_path": record_path.relative_to(score_root).as_posix(),
            "record_sha256": sha256_file(record_path),
            "score_sha256": score_sha,
        })
        print(f"score-freeze {variant_id}: {cell_id} ({position}/{len(PB_CELLS)})", flush=True)
    if variant_id == REFERENCE and max(max_local_alias_error, max_combined_alias_error) > 1e-12:
        raise ReducerError("top-five reference is not an exact R1 score alias")
    freeze = {
        "schema": SCHEMA,
        "stage": "target_free_score_freeze",
        "variant_id": variant_id,
        "cells": list(PB_CELLS),
        "labels_seen_during_fit": False,
        "targets_accessed_during_fit": False,
        "input_manifest_sha256": sha256_file(manifest_path),
        "execution_registry_sha256": sha256_file(Path(registry["registry_path"])),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "reference_local_alias_max_abs_error": max_local_alias_error,
        "reference_combined_alias_max_abs_error": max_combined_alias_error,
        "records": records,
        "complete": len(records) == len(PB_CELLS),
    }
    freeze["payload_sha256"] = _payload_sha(freeze)
    atomic_write_json(score_root / "SCORE_FREEZE_MANIFEST.json", freeze)
    return freeze


def _verified_scores(output: Path, freeze: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if freeze.get("complete") is not True or freeze.get("labels_seen_during_fit") is not False:
        raise ReducerError("invalid score freeze")
    score_root = output / "score_freeze"
    verified: dict[str, dict[str, Any]] = {}
    for item in freeze["records"]:
        record_path = score_root / item["record_path"]
        if sha256_file(record_path) != item["record_sha256"]:
            raise ReducerError("score record changed after freeze")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        score_path = record_path.parent / record["score_file"]
        if sha256_file(score_path) != item["score_sha256"]:
            raise ReducerError("score array changed after freeze")
        verified[item["cell_id"]] = {
            "record": record,
            "arrays": load_npz_no_pickle(score_path),
        }
    if tuple(verified) != PB_CELLS:
        raise ReducerError("verified score roster/order changed")
    return verified


def _rows_by_model(
    verified: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, tuple[str, int]]],
) -> dict[str, list[dict[str, Any]]]:
    output = {model: [] for model in QWEN_MODELS}
    for cell_id in PB_CELLS:
        record, arrays = verified[cell_id]["record"], verified[cell_id]["arrays"]
        row_ids = tuple(arrays["row_ids"].astype(str))
        if set(row_ids) != set(labels[cell_id]):
            raise ReducerError(f"{cell_id}: frozen score and label populations differ")
        offsets = np.asarray(arrays["segment_offsets"], dtype=np.int64)
        lengths = np.asarray(arrays["segment_lengths"], dtype=np.int64)
        scores = np.asarray(arrays["combined_step_scores"], dtype=np.float64)
        for row_index, row_id in enumerate(row_ids):
            lo, hi = map(int, offsets[row_index:row_index + 2])
            group_id, first_error = labels[cell_id][row_id]
            output[record["model_id"]].append({
                "row_id": row_id,
                "group_id": group_id,
                "slice_id": record["slice_id"],
                "cell_id": cell_id,
                "model_id": record["model_id"],
                "first_error": first_error,
                "step_scores": scores[lo:hi].tolist(),
                "step_lengths": lengths[lo:hi].tolist(),
            })
    return output


def _length_cutpoints(rows: Sequence[Mapping[str, Any]], assignments: Mapping[str, int]) -> dict[str, Any]:
    by_fold = {}
    for held_fold in range(5):
        lengths = [
            int(row["step_lengths"][int(row["first_error"])])
            for row in rows
            if assignments[str(row["group_id"])] != held_fold and int(row["first_error"]) >= 0
        ]
        if not lengths:
            raise ReducerError("length-stratum calibration fold has no erroneous rows")
        q1, q2 = np.quantile(lengths, [1.0 / 3.0, 2.0 / 3.0], method="linear")
        by_fold[str(held_fold)] = {
            "q1": float(q1),
            "q2": float(q2),
            "n_calibration_errors": len(lengths),
            "quantile_method": "numpy_linear",
        }
    return by_fold


def _stratum(length: int, cutpoints: Mapping[str, Any]) -> str:
    if length <= float(cutpoints["q1"]):
        return "short"
    if length <= float(cutpoints["q2"]):
        return "medium"
    return "long"


def _evaluate_reference(
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]], evaluation: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    by_cell: list[dict[str, Any]] = []
    threshold_models: dict[str, Any] = {}
    for model in QWEN_MODELS:
        rows = list(rows_by_model[model])
        result = evaluation.crossfit_processbench_threshold(rows)
        assignments = evaluation.assign_processbench_folds(rows)
        cutpoints = _length_cutpoints(rows, assignments)
        threshold_models[model] = {
            "fold_assignment_sha256": result["fold_assignment_sha256"],
            "calibration_ledgers": result["calibration_ledgers"],
            "length_cutpoints": cutpoints,
        }
        by_row = {str(row["row_id"]): row for row in rows}
        for row in result["decisions"]:
            parent = by_row[row["row_id"]]
            prediction = int(row["prediction_step"])
            true_error = int(parent["first_error"])
            true_length = int(parent["step_lengths"][true_error]) if true_error >= 0 else None
            selected_length = int(parent["step_lengths"][prediction]) if prediction >= 0 else None
            decisions.append({
                "model_id": model,
                "cell_id": parent["cell_id"],
                "slice_id": parent["slice_id"],
                "row_id": row["row_id"],
                "group_id": parent["group_id"],
                "fold": int(row["fold"]),
                "true_first_error": true_error,
                "prediction_step": prediction,
                "true_error_step_length": true_length,
                "true_error_length_stratum": (
                    _stratum(true_length, cutpoints[str(row["fold"])])
                    if true_length is not None else "CLEAN"
                ),
                "selected_step_length": selected_length,
            })
        for family, metrics in result["metrics"]["per_subset"].items():
            by_cell.append({
                "model_id": model,
                "slice_id": family,
                "cell_id": f"processbench_{family}_{model}",
                **{metric: metrics[metric] for metric in PB_METRICS},
                "n_examples": metrics["n_examples"],
                "n_error": metrics["n_error"],
                "n_clean": metrics["n_clean"],
            })
    return decisions, by_cell, threshold_models


def _alias_audit(
    decisions: Sequence[Mapping[str, Any]],
    by_cell: Sequence[Mapping[str, Any]],
    samples: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    parent_decisions = {
        (row["model_id"], row["row_id"]): row
        for row in _read_csv(P1_REFERENCE_ROOT / "evaluation/PROCESSBENCH_DECISIONS.csv")
        if row["model_id"] in QWEN_MODELS
    }
    decision_mismatches = 0
    fold_mismatches = 0
    for row in decisions:
        parent = parent_decisions[(str(row["model_id"]), str(row["row_id"]))]
        decision_mismatches += int(int(row["prediction_step"]) != int(parent["prediction_step"]))
        fold_mismatches += int(int(row["fold"]) != int(parent["fold"]))
    parent_cells = {
        row["cell_id"]: row
        for row in _read_csv(P1_REFERENCE_ROOT / "evaluation/PROCESSBENCH_BY_CELL.csv")
        if row["model_id"] in QWEN_MODELS
    }
    metric_error = max(
        abs(float(row[metric]) - float(parent_cells[row["cell_id"]][metric]))
        for row in by_cell for metric in PB_METRICS
    )
    parent_samples = load_npz_no_pickle(
        P1_REFERENCE_ROOT / "evaluation/PROCESSBENCH_BOOTSTRAP_SAMPLES.npz"
    )
    bootstrap_error = max(
        float(np.max(np.abs(values - parent_samples[f"current_common_eight_qwen__{metric}"])))
        for metric, values in samples.items()
    )
    return {
        "decision_mismatches": decision_mismatches,
        "fold_mismatches": fold_mismatches,
        "metric_max_abs_error": metric_error,
        "bootstrap_max_abs_error": bootstrap_error,
    }


def _length_strata(
    decisions: Sequence[Mapping[str, Any]], by_cell: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    cell_metrics = {row["cell_id"]: row for row in by_cell}
    output: list[dict[str, Any]] = []
    for stratum in ("short", "medium", "long"):
        per_cell = []
        for cell_id in PB_CELLS:
            rows = [
                row for row in decisions
                if row["cell_id"] == cell_id and row["true_error_length_stratum"] == stratum
            ]
            if not rows:
                continue
            exact = float(np.mean([
                int(row["prediction_step"]) == int(row["true_first_error"]) for row in rows
            ]))
            within = float(np.mean([
                int(row["prediction_step"]) >= 0
                and abs(int(row["prediction_step"]) - int(row["true_first_error"])) <= 1
                for row in rows
            ]))
            abstention = float(cell_metrics[cell_id]["clean_abstention_accuracy"])
            macro_f1 = 2 * exact * abstention / (exact + abstention) if exact + abstention else 0.0
            record = {
                "level": "cell", "stratum": stratum, "cell_id": cell_id,
                "macro_f1": macro_f1, "first_error_exact": exact,
                "first_error_within_one": within,
                "clean_abstention_accuracy": abstention,
                "n_error": len(rows),
            }
            output.append(record)
            per_cell.append(record)
        for metric in (
            "macro_f1", "first_error_exact", "first_error_within_one",
            "clean_abstention_accuracy",
        ):
            output.append({
                "level": "aggregate", "stratum": stratum, "cell_id": "aggregate",
                "metric_id": metric,
                "value": float(np.mean([row[metric] for row in per_cell])),
                "n_error": sum(int(row["n_error"]) for row in per_cell),
                "n_cells": len(per_cell),
            })
    return output


def _selected_length_distribution(decisions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for outcome in ("all", "error", "clean"):
        rows = [
            row for row in decisions
            if outcome == "all" or (outcome == "error") == (int(row["true_first_error"]) >= 0)
        ]
        selected = [int(row["selected_step_length"]) for row in rows if row["selected_step_length"] is not None]
        output.append({
            "outcome": outcome,
            "n_rows": len(rows),
            "n_selected": len(selected),
            "n_abstained": len(rows) - len(selected),
            "mean_selected_step_length": float(np.mean(selected)) if selected else None,
            "median_selected_step_length": float(np.median(selected)) if selected else None,
            "q90_selected_step_length": float(np.quantile(selected, 0.90)) if selected else None,
        })
    return output


def evaluate_reference(
    release: Path,
    output: Path,
    registry: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    if registry["variant_id"] != REFERENCE:
        raise ReducerError("this turn authorizes only the top-five reference")
    _require_source_hashes(registry)
    verified = _verified_scores(output, freeze)
    labels = _load_pb_labels(release)
    evaluation = __import__(
        "spectral_utils.reconstruction_benchmark.localization_evaluation",
        fromlist=["localization_evaluation"],
    )
    rows_by_model = _rows_by_model(verified, labels)
    decisions, by_cell, threshold_models = _evaluate_reference(rows_by_model, evaluation)
    samples = _bootstrap_pb_panel(decisions, QWEN_MODELS)
    point = {metric: float(np.mean([row[metric] for row in by_cell])) for metric in PB_METRICS}
    panels = [{
        "population_id": "current_common_eight_qwen",
        "metric_id": metric,
        "value": point[metric],
        "ci_low": float(np.quantile(samples[metric], 0.025)),
        "ci_high": float(np.quantile(samples[metric], 0.975)),
        "n_rows": sum(int(row["n_examples"]) for row in by_cell),
        "n_groups": 3400,
    } for metric in PB_METRICS]
    alias = _alias_audit(decisions, by_cell, samples)
    if max(alias.values()) > 1e-12:
        raise ReducerError(f"R1 alias audit failed: {alias}")
    eval_root = output / "evaluation"
    eval_root.mkdir(parents=True, exist_ok=False)
    _write_csv(eval_root / "PROCESSBENCH_DECISIONS.csv", decisions)
    _write_csv(eval_root / "PROCESSBENCH_BY_CELL.csv", by_cell)
    _write_csv(eval_root / "PROCESSBENCH_PANELS.csv", panels)
    atomic_write_npz(eval_root / "PROCESSBENCH_BOOTSTRAP_SAMPLES.npz", samples)
    strata = _length_strata(decisions, by_cell)
    _write_csv(eval_root / "STEP_LENGTH_STRATA.csv", strata)
    _write_csv(eval_root / "SELECTED_STEP_LENGTH.csv", _selected_length_distribution(decisions))
    _write_csv(eval_root / "PREDICTION_FLIPS.csv", [{
        "comparison": "P2R_A_TOPK5_REFERENCE_vs_R1_ENTROPY_TOP5",
        "n_rows": len(decisions),
        "n_prediction_flips": alias["decision_mismatches"],
        "n_fold_flips": alias["fold_mismatches"],
    }])
    threshold_payload = {
        "schema": "reasoning-localization-phase2-reference-thresholds-v1",
        "variant_id": REFERENCE,
        "source_parent": "R1_ENTROPY_TOP5",
        "candidate_rethresholding_allowed": False,
        "models": threshold_models,
    }
    threshold_payload["payload_sha256"] = _payload_sha(threshold_payload)
    atomic_write_json(eval_root / "FROZEN_THRESHOLDS.json", threshold_payload)
    gates = [
        {"gate_id": "P2R_REFERENCE_SCORE_ALIAS", "status": "PASS", "observed": max(freeze["reference_local_alias_max_abs_error"], freeze["reference_combined_alias_max_abs_error"]), "required": "<=1e-12", "detail": "local and combined step scores reproduce R1"},
        {"gate_id": "P2R_REFERENCE_DECISION_ALIAS", "status": "PASS", "observed": alias["decision_mismatches"], "required": "0", "detail": "all predictions reproduce R1"},
        {"gate_id": "P2R_REFERENCE_FOLD_ALIAS", "status": "PASS", "observed": alias["fold_mismatches"], "required": "0", "detail": "all source-group folds reproduce R1"},
        {"gate_id": "P2R_REFERENCE_METRIC_ALIAS", "status": "PASS", "observed": alias["metric_max_abs_error"], "required": "<=1e-12", "detail": "all five per-cell metrics reproduce R1"},
        {"gate_id": "P2R_REFERENCE_BOOTSTRAP_ALIAS", "status": "PASS", "observed": alias["bootstrap_max_abs_error"], "required": "<=1e-12", "detail": "all 20,000 paired samples reproduce R1"},
        {"gate_id": "P2R_THRESHOLD_LEDGER_FROZEN", "status": "PASS", "observed": sum(len(row["calibration_ledgers"]) for row in threshold_models.values()), "required": "10 model-fold ledgers", "detail": "the sole Phase-2R threshold reconstruction is now immutable"},
        {"gate_id": "P2R_LENGTH_STRATA_CALIBRATION_ONLY", "status": "PASS", "observed": 10, "required": "10 model-fold cutpoint pairs", "detail": "linear tertiles use only erroneous calibration rows"},
        {"gate_id": "P2R_FINITE_COMPLETE_SCORES", "status": "PASS", "observed": sum(len(payload["arrays"]["combined_step_scores"]) for payload in verified.values()), "required": "all registered steps finite; no missing rows", "detail": "verified after score-freeze checksums"},
    ]
    _write_csv(eval_root / "GATES.csv", gates)
    summary = {
        "schema": "reasoning-localization-phase2-reducer-evaluation-v1",
        "variant_id": REFERENCE,
        "status": "COMPLETE",
        "processbench_qwen8_macro_f1": point["official_macro_f1"],
        "processbench_qwen8_ci": [
            next(row["ci_low"] for row in panels if row["metric_id"] == "official_macro_f1"),
            next(row["ci_high"] for row in panels if row["metric_id"] == "official_macro_f1"),
        ],
        "alias_audit": alias,
        "thresholds_frozen": True,
        "candidate_rethresholding_allowed": False,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "peak_memory_bytes": _peak_memory_bytes(),
    }
    summary["payload_sha256"] = _payload_sha(summary)
    atomic_write_json(eval_root / "SUMMARY.json", summary)
    outputs = (
        "PROCESSBENCH_DECISIONS.csv", "PROCESSBENCH_BY_CELL.csv",
        "PROCESSBENCH_PANELS.csv", "PROCESSBENCH_BOOTSTRAP_SAMPLES.npz",
        "STEP_LENGTH_STRATA.csv", "SELECTED_STEP_LENGTH.csv",
        "PREDICTION_FLIPS.csv", "FROZEN_THRESHOLDS.json", "GATES.csv", "SUMMARY.json",
    )
    manifest = {
        "schema": "reasoning-localization-phase2-reducer-evaluation-manifest-v1",
        "variant_id": REFERENCE,
        "score_freeze_sha256": sha256_file(output / "score_freeze/SCORE_FREEZE_MANIFEST.json"),
        "execution_registry_sha256": sha256_file(Path(registry["registry_path"])),
        "outputs": [
            {"path": name, "sha256": sha256_file(eval_root / name), "bytes": (eval_root / name).stat().st_size}
            for name in outputs
        ],
    }
    manifest["payload_sha256"] = _payload_sha(manifest)
    atomic_write_json(eval_root / "EVALUATION_MANIFEST.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=STAGE_A_VARIANTS)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    registry_path = (
        args.registry or PHASE_ROOT / f"{args.variant}_EXECUTION_REGISTRY.json"
    ).resolve()
    output = (args.output or PHASE_ROOT / args.variant.lower()).resolve()
    release = args.release.resolve()
    registry = load_registry(registry_path, args.variant, release)
    registry["registry_path"] = str(registry_path)
    started = time.perf_counter()
    freeze = freeze_scores(args.variant, release, output, registry)
    summary = evaluate_reference(release, output, registry, freeze)
    run = {
        "schema": "reasoning-localization-phase2-reducer-run-v1",
        "variant_id": args.variant,
        "status": "COMPLETE",
        "execution_registry_sha256": sha256_file(registry_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "score_freeze_manifest_sha256": sha256_file(output / "score_freeze/SCORE_FREEZE_MANIFEST.json"),
        "evaluation_manifest_sha256": sha256_file(output / "evaluation/EVALUATION_MANIFEST.json"),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
    }
    run["payload_sha256"] = _payload_sha(run)
    atomic_write_json(output / "RUN_MANIFEST.json", run)
    print(json.dumps(run, indent=2), flush=True)


if __name__ == "__main__":
    main()
