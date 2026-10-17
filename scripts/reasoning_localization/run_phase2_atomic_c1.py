#!/usr/bin/env python3
"""Freeze and evaluate the atomic entropy plus causal SWVar16 candidate."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
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
    empirical_midrank,
    load_prepared_localization_cell,
    validate_fit_manifest,
)
from spectral_utils.fixed_application_pipelines import SHARED_TOKEN_VIEWS  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402


REFERENCE = "P2A_TOPK10_REFERENCE"
CANDIDATE = "C1_ENT_SW16"
ARMS = (REFERENCE, CANDIDATE)
ATOMIC_ROOT = p1.PROGRAM_ROOT / "phase_2/atomic"
OUTPUT_ROOT = ATOMIC_ROOT / CANDIDATE.lower()
REGISTRY_PATH = ATOMIC_ROOT / f"{CANDIDATE}_EXECUTION_REGISTRY.json"
P2R_TOP10_ROOT = p1.PROGRAM_ROOT / "phase_2/p2r_a_topk10"
P1_TOP5_ROOT = p1.PROGRAM_ROOT / "phase_1/r1_entropy_top5"
WINDOW = 16
FUSION_WEIGHT = 0.5
PRIMARY_COMPARISON_FAMILY = 2
HARD_WORST_CELL_BOUND = -0.030
BENEFIT = 0.005
HARM = -0.005
COMPONENT_BOUND = -0.010
PROMOTION_WORST_CELL_BOUND = -0.020


class AtomicC1Error(RuntimeError):
    """Fail-closed C1 contract error."""


def payload_sha(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def peak_memory_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise AtomicC1Error(f"refusing to write empty table: {path}")
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def trailing_population_variance(values: Sequence[float], window: int = WINDOW) -> np.ndarray:
    """Causal available-prefix population variance with one-response reset."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("SWVar input must be a nonempty finite vector")
    if int(window) != window or int(window) < 1:
        raise ValueError("SWVar window must be a positive integer")
    prefix = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(x * x, dtype=np.float64)))
    indices = np.arange(len(x), dtype=np.int64)
    starts = np.maximum(0, indices - int(window) + 1)
    ends = indices + 1
    counts = (ends - starts).astype(np.float64)
    totals = prefix[ends] - prefix[starts]
    totals_sq = prefix_sq[ends] - prefix_sq[starts]
    result = totals_sq / counts - (totals / counts) ** 2
    # Roundoff can produce tiny negative values for a mathematically nonnegative variance.
    result = np.maximum(result, 0.0)
    result[0] = 0.0
    if not np.isfinite(result).all():
        raise AtomicC1Error("SWVar transform produced non-finite values")
    return result


def response_reset_swvar(values: Sequence[float], offsets: Sequence[int]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    bounds = np.asarray(offsets, dtype=np.int64)
    if bounds.ndim != 1 or bounds[0] != 0 or bounds[-1] != len(x) or np.any(np.diff(bounds) <= 0):
        raise ValueError("response offsets do not partition the token curve")
    output = np.empty_like(x)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        output[int(lo):int(hi)] = trailing_population_variance(x[int(lo):int(hi)])
    return output


def fuse_step_channels(
    entropy_step: Sequence[float], swvar_step: Sequence[float], *, sw_weight: float = FUSION_WEIGHT
) -> np.ndarray:
    entropy = np.asarray(entropy_step, dtype=np.float64)
    variance = np.asarray(swvar_step, dtype=np.float64)
    if entropy.shape != variance.shape or entropy.ndim != 1 or not len(entropy):
        raise ValueError("C1 step channels must be aligned nonempty vectors")
    if not 0.0 <= float(sw_weight) <= 1.0:
        raise ValueError("SWVar fusion weight must lie in [0,1]")
    return (
        (1.0 - float(sw_weight)) * empirical_midrank(entropy)
        + float(sw_weight) * empirical_midrank(variance)
    )


def suffix_invariance_audit(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    full = trailing_population_variance(x)
    cuts = sorted({1, min(2, len(x)), min(WINDOW - 1, len(x)), min(WINDOW, len(x)),
                   max(1, len(x) // 2), max(1, len(x) - 1), len(x)})
    return max(
        float(np.max(np.abs(full[:cut] - trailing_population_variance(x[:cut]))))
        for cut in cuts
    )


def require_source_hashes(registry: Mapping[str, Any]) -> None:
    for source in registry["frozen_sources"]:
        path = Path(source["path"])
        if not path.is_absolute():
            path = REPO / path
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise AtomicC1Error(f"frozen source changed or missing: {source['role']}")


def load_registry(path: Path, release: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "reasoning-localization-phase2-atomic-c1-execution-registry-v1",
        "status": "FROZEN_BEFORE_RUN",
        "candidate": CANDIDATE,
        "atomic_reference": REFERENCE,
        "processbench_cells": list(p2r.PB_CELLS),
        "window": WINDOW,
        "variance_ddof": 0,
        "fusion_weight": FUSION_WEIGHT,
        "bootstrap_draws": p1.BOOTSTRAP_DRAWS,
        "bootstrap_seed": p1.BOOTSTRAP_SEED,
    }
    for key, value in expected.items():
        if registry.get(key) != value:
            raise AtomicC1Error(f"execution registry mismatch for {key}")
    if Path(registry["release_root"]).resolve() != release.resolve():
        raise AtomicC1Error("release root differs from frozen registry")
    if registry["runner_sha256"] != sha256_file(Path(__file__).resolve()):
        raise AtomicC1Error("runner changed after registry freeze")
    require_source_hashes(registry)
    return registry


def freeze_scores(release: Path, output: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite C1 output: {output}")
    score_root = output / "score_freeze"
    score_root.mkdir(parents=True, exist_ok=False)
    input_root = release / "build_A/localization/inputs"
    manifest_path = input_root / "MANIFEST.json"
    manifest = validate_fit_manifest(manifest_path, input_root=input_root)
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    if not set(p2r.PB_CELLS).issubset(by_cell):
        raise AtomicC1Error("prepared input lacks the exact eight Qwen cells")
    entropy_index = SHARED_TOKEN_VIEWS.index("entropy_series")
    records = []
    alias_local = 0.0
    alias_combined = 0.0
    suffix_error = 0.0
    started = time.perf_counter()
    for position, cell_id in enumerate(p2r.PB_CELLS, start=1):
        source_record = by_cell[cell_id]
        input_path = input_root / source_record["artifact_path"]
        cell = load_prepared_localization_cell(input_path, source_record)
        entropy_risk = -np.asarray(cell.token_confidence[:, entropy_index], dtype=np.float64)
        swvar_risk = response_reset_swvar(entropy_risk, cell.token_offsets)
        for lo, hi in zip(cell.token_offsets[:-1], cell.token_offsets[1:]):
            suffix_error = max(
                suffix_error,
                suffix_invariance_audit(entropy_risk[int(lo):int(hi)]),
            )
        entropy_step = p1.topk_step_mean(
            entropy_risk, cell.segment_starts, cell.segment_ends, k=10
        )
        swvar_step = p1.topk_step_mean(
            swvar_risk, cell.segment_starts, cell.segment_ends, k=10
        )
        candidate_local = fuse_step_channels(entropy_step, swvar_step)
        reference_combined = p1.combine_with_common_detector(cell, entropy_step)
        candidate_combined = p1.combine_with_common_detector(cell, candidate_local)
        prior = load_npz_no_pickle(
            P2R_TOP10_ROOT / "score_freeze/cells" / cell_id / "scores.npz"
        )
        alias_local = max(
            alias_local,
            float(np.max(np.abs(entropy_step - prior["local_step_scores"]))),
        )
        alias_combined = max(
            alias_combined,
            float(np.max(np.abs(reference_combined - prior["combined_step_scores"]))),
        )
        target = score_root / "cells" / cell_id
        target.mkdir(parents=True, exist_ok=False)
        score_path = target / "scores.npz"
        score_sha = atomic_write_npz(score_path, {
            "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
            "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
            "segment_lengths": np.asarray(cell.segment_ends - cell.segment_starts, dtype="<i8"),
            "reference_local_step_scores": np.asarray(entropy_step, dtype="<f8"),
            "reference_combined_step_scores": np.asarray(reference_combined, dtype="<f8"),
            "candidate_local_step_scores": np.asarray(candidate_local, dtype="<f8"),
            "candidate_combined_step_scores": np.asarray(candidate_combined, dtype="<f8"),
            "swvar_step_scores": np.asarray(swvar_step, dtype="<f8"),
        })
        record = {
            "schema": "reasoning-localization-phase2-atomic-c1-cell-v1",
            "cell_id": cell_id,
            "model_id": str(cell.model_id),
            "slice_id": str(cell.slice_id),
            "population_id": str(cell.population_id),
            "n_rows": len(cell.row_ids),
            "n_steps": len(entropy_step),
            "prepared_input": str(input_path),
            "prepared_input_sha256": sha256_file(input_path),
            "score_file": "scores.npz",
            "score_sha256": score_sha,
            "labels_seen_during_fit": False,
            "targets_accessed_during_fit": False,
        }
        record["payload_sha256"] = payload_sha(record)
        atomic_write_json(target / "RECORD.json", record)
        records.append({
            "cell_id": cell_id,
            "record_path": f"cells/{cell_id}/RECORD.json",
            "record_sha256": sha256_file(target / "RECORD.json"),
            "score_sha256": score_sha,
        })
        print(f"score-freeze {CANDIDATE}: {cell_id} ({position}/8)", flush=True)
    require_source_hashes(registry)
    freeze = {
        "schema": "reasoning-localization-phase2-atomic-c1-score-freeze-v1",
        "status": "COMPLETE",
        "candidate": CANDIDATE,
        "atomic_reference": REFERENCE,
        "cells": list(p2r.PB_CELLS),
        "labels_seen_during_fit": False,
        "targets_accessed_during_fit": False,
        "reference_local_alias_max_abs_error": alias_local,
        "reference_combined_alias_max_abs_error": alias_combined,
        "suffix_invariance_max_abs_error": suffix_error,
        "input_manifest_sha256": sha256_file(manifest_path),
        "execution_registry_sha256": sha256_file(Path(registry["registry_path"])),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "records": records,
    }
    freeze["payload_sha256"] = payload_sha(freeze)
    atomic_write_json(score_root / "SCORE_FREEZE_MANIFEST.json", freeze)
    return freeze


def verified_scores(output: Path, freeze: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for item in freeze["records"]:
        record_path = output / "score_freeze" / item["record_path"]
        if sha256_file(record_path) != item["record_sha256"]:
            raise AtomicC1Error("score record changed after freeze")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        score_path = record_path.parent / record["score_file"]
        if sha256_file(score_path) != item["score_sha256"]:
            raise AtomicC1Error("score array changed after freeze")
        result[item["cell_id"]] = {"record": record, "arrays": load_npz_no_pickle(score_path)}
    if tuple(result) != p2r.PB_CELLS:
        raise AtomicC1Error("verified score roster differs from frozen cell order")
    return result


def rows_by_model(
    verified: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, tuple[str, int]]],
    arm: str,
) -> dict[str, list[dict[str, Any]]]:
    key = "reference_combined_step_scores" if arm == REFERENCE else "candidate_combined_step_scores"
    result = {model: [] for model in p1.QWEN_MODELS}
    for cell_id in p2r.PB_CELLS:
        record, arrays = verified[cell_id]["record"], verified[cell_id]["arrays"]
        row_ids = tuple(arrays["row_ids"].astype(str))
        if set(row_ids) != set(labels[cell_id]):
            raise AtomicC1Error(f"{cell_id}: score/label population mismatch")
        offsets = np.asarray(arrays["segment_offsets"], dtype=np.int64)
        lengths = np.asarray(arrays["segment_lengths"], dtype=np.int64)
        scores = np.asarray(arrays[key], dtype=np.float64)
        for index, row_id in enumerate(row_ids):
            lo, hi = map(int, offsets[index:index + 2])
            group_id, first_error = labels[cell_id][row_id]
            result[record["model_id"]].append({
                "row_id": row_id,
                "group_id": group_id,
                "slice_id": record["slice_id"],
                "cell_id": cell_id,
                "model_id": record["model_id"],
                "first_error": first_error,
                "step_scores": scores[lo:hi].tolist(),
                "step_lengths": lengths[lo:hi].tolist(),
            })
    return result


def evaluate_arm(
    arm: str,
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    evaluation: Any,
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    by_cell: list[dict[str, Any]] = []
    ledgers = {}
    for model in p1.QWEN_MODELS:
        model_rows = list(rows[model])
        result = evaluation.crossfit_processbench_threshold(model_rows)
        assignments = evaluation.assign_processbench_folds(model_rows)
        cutpoints = p2r._length_cutpoints(model_rows, assignments)
        ledgers[model] = {
            "fold_assignment_sha256": result["fold_assignment_sha256"],
            "calibration_ledgers": result["calibration_ledgers"],
            "length_cutpoints": cutpoints,
        }
        by_row = {str(row["row_id"]): row for row in model_rows}
        for row in result["decisions"]:
            parent = by_row[row["row_id"]]
            prediction = int(row["prediction_step"])
            target = int(parent["first_error"])
            true_length = int(parent["step_lengths"][target]) if target >= 0 else None
            selected_length = int(parent["step_lengths"][prediction]) if prediction >= 0 else None
            decisions.append({
                "arm_id": arm,
                "model_id": model,
                "cell_id": parent["cell_id"],
                "slice_id": parent["slice_id"],
                "row_id": row["row_id"],
                "group_id": parent["group_id"],
                "fold": int(row["fold"]),
                "true_first_error": target,
                "prediction_step": prediction,
                "true_error_step_length": true_length,
                "true_error_length_stratum": (
                    p2r._stratum(true_length, cutpoints[str(row["fold"])])
                    if true_length is not None else "CLEAN"
                ),
                "selected_step_length": selected_length,
            })
        for family, metrics in result["metrics"]["per_subset"].items():
            by_cell.append({
                "arm_id": arm,
                "model_id": model,
                "slice_id": family,
                "cell_id": f"processbench_{family}_{model}",
                **{metric: metrics[metric] for metric in p1.PB_METRICS},
                "n_examples": metrics["n_examples"],
                "n_error": metrics["n_error"],
                "n_clean": metrics["n_clean"],
            })
    samples = p1._bootstrap_pb_panel(decisions, p1.QWEN_MODELS)
    panels = []
    for metric in p1.PB_METRICS:
        values = np.asarray(samples[metric], dtype=np.float64)
        panels.append({
            "arm_id": arm,
            "population_id": "current_common_eight_qwen",
            "metric_id": metric,
            "value": float(np.mean([float(row[metric]) for row in by_cell])),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "n_rows": sum(int(row["n_examples"]) for row in by_cell),
            "n_groups": 3400,
        })
    return {"decisions": decisions, "by_cell": by_cell, "panels": panels, "samples": samples, "ledgers": ledgers}


def comparator_top5() -> dict[str, Any]:
    cells = [
        row for row in read_csv(P1_TOP5_ROOT / "evaluation/PROCESSBENCH_BY_CELL.csv")
        if row["model_id"] in p1.QWEN_MODELS
    ]
    arrays = load_npz_no_pickle(P1_TOP5_ROOT / "evaluation/PROCESSBENCH_BOOTSTRAP_SAMPLES.npz")
    samples = {
        metric: np.asarray(arrays[f"current_common_eight_qwen__{metric}"], dtype=np.float64)
        for metric in p1.PB_METRICS
    }
    return {"by_cell": cells, "samples": samples}


def build_contrasts(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    comparators = {
        REFERENCE: reference,
        "R1_ENTROPY_TOP5": comparator_top5(),
    }
    output = []
    for comparator_id, comparator in comparators.items():
        right_cells = {str(row["cell_id"]): row for row in comparator["by_cell"]}
        for metric in p1.PB_METRICS:
            left_point = float(np.mean([float(row[metric]) for row in candidate["by_cell"]]))
            right_point = float(np.mean([float(right_cells[row["cell_id"]][metric]) for row in candidate["by_cell"]]))
            draws = np.asarray(candidate["samples"][metric]) - np.asarray(comparator["samples"][metric])
            q = 0.025 / PRIMARY_COMPARISON_FAMILY if metric == "official_macro_f1" else 0.025
            cell_deltas = {
                str(row["cell_id"]): float(row[metric]) - float(right_cells[str(row["cell_id"])][metric])
                for row in candidate["by_cell"]
            }
            family_deltas = {
                family: float(np.mean([
                    value for cell_id, value in cell_deltas.items()
                    if str(right_cells[cell_id]["slice_id"]) == family
                ]))
                for family in p1.FAMILIES
            }
            eps = 1e-12
            output.append({
                "contrast_id": f"pb::{CANDIDATE}::{comparator_id}::{metric}",
                "left_variant_id": CANDIDATE,
                "right_variant_id": comparator_id,
                "metric_id": "macro_f1" if metric == "official_macro_f1" else metric,
                "source_metric_id": metric,
                "delta": left_point - right_point,
                "ci_low": float(np.quantile(draws, q)),
                "ci_high": float(np.quantile(draws, 1.0 - q)),
                "wins": sum(value > eps for value in cell_deltas.values()),
                "ties": sum(abs(value) <= eps for value in cell_deltas.values()),
                "losses": sum(value < -eps for value in cell_deltas.values()),
                "worst_unit_delta": min(cell_deltas.values()),
                "worst_unit_id": min(cell_deltas, key=cell_deltas.get),
                "family_wins": sum(value > eps for value in family_deltas.values()),
                "family_ties": sum(abs(value) <= eps for value in family_deltas.values()),
                "family_losses": sum(value < -eps for value in family_deltas.values()),
                "worst_family_delta": min(family_deltas.values()),
                "worst_family_id": min(family_deltas, key=family_deltas.get),
                "multiplicity_family_size": PRIMARY_COMPARISON_FAMILY if metric == "official_macro_f1" else 1,
                "inference": (
                    "Bonferroni simultaneous percentile interval across C1 versus both required primary comparators"
                    if metric == "official_macro_f1" else "unadjusted paired diagnostic percentile interval"
                ),
            })
    return output


def prediction_flips(
    candidate: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    right = {(str(row["model_id"]), str(row["row_id"])): row for row in reference}
    rows = []
    counts: dict[tuple[str, str], int] = {}
    for row in candidate:
        parent = right[(str(row["model_id"]), str(row["row_id"]))]
        left_prediction = int(row["prediction_step"])
        right_prediction = int(parent["prediction_step"])
        target = int(row["true_first_error"])
        left_correct = left_prediction == target
        right_correct = right_prediction == target
        if left_prediction == right_prediction:
            transition = "NO_FLIP_CORRECT" if left_correct else "NO_FLIP_INCORRECT"
        elif left_correct and not right_correct:
            transition = "FLIP_GAIN"
        elif right_correct and not left_correct:
            transition = "FLIP_LOSS"
        else:
            transition = "FLIP_LATERAL"
        counts[(str(row["cell_id"]), transition)] = counts.get((str(row["cell_id"]), transition), 0) + 1
        rows.append({
            "cell_id": row["cell_id"], "model_id": row["model_id"],
            "slice_id": row["slice_id"], "row_id": row["row_id"],
            "group_id": row["group_id"], "true_first_error": target,
            "reference_prediction_step": right_prediction,
            "candidate_prediction_step": left_prediction,
            "changed": str(left_prediction != right_prediction).lower(),
            "transition": transition,
        })
    summary = [
        {"cell_id": cell, "transition": transition, "count": count}
        for (cell, transition), count in sorted(counts.items())
    ]
    return rows, summary


def evaluate_scores(
    release: Path, output: Path, registry: Mapping[str, Any], freeze: Mapping[str, Any]
) -> dict[str, Any]:
    require_source_hashes(registry)
    verified = verified_scores(output, freeze)
    # This is the first target-bearing operation in the run.
    labels = p1._load_pb_labels(release)
    evaluation = importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation")
    arm_results = {
        arm: evaluate_arm(arm, rows_by_model(verified, labels, arm), evaluation)
        for arm in ARMS
    }
    contrasts = build_contrasts(arm_results[CANDIDATE], arm_results[REFERENCE])
    primary = {
        row["right_variant_id"]: row for row in contrasts if row["metric_id"] == "macro_f1"
    }
    by_metric = {
        (row["right_variant_id"], row["metric_id"]): row for row in contrasts
    }
    hard_failure = (
        freeze["reference_local_alias_max_abs_error"] > 1e-12
        or freeze["reference_combined_alias_max_abs_error"] > 1e-12
        or freeze["suffix_invariance_max_abs_error"] > 1e-12
        or min(float(row["worst_unit_delta"]) for row in primary.values()) < HARD_WORST_CELL_BOUND
    )
    promotion = all(
        float(row["delta"]) >= BENEFIT
        and float(row["ci_low"]) > BENEFIT
        and int(row["wins"]) + int(row["ties"]) >= 6
        and float(row["worst_unit_delta"]) >= PROMOTION_WORST_CELL_BOUND
        and float(by_metric[(comparator, "first_error_exact")]["delta"]) >= COMPONENT_BOUND
        and float(by_metric[(comparator, "clean_abstention_accuracy")]["delta"]) >= COMPONENT_BOUND
        for comparator, row in primary.items()
    ) and not hard_failure
    gates = [
        {"gate_id": "P2A_SCORE_FREEZE_COMPLETE", "status": "PASS", "observed": len(verified), "required": "8 cells", "detail": "both atomic reference and C1 scores froze before labels opened"},
        {"gate_id": "P2A_LABEL_FIREWALL", "status": "PASS", "observed": "labels opened after complete score-freeze manifest", "required": "no fit-side labels or targets", "detail": "C1 transform, reduction, and fusion are label-free"},
        {"gate_id": "P2A_TOP10_LOCAL_ALIAS", "status": "PASS" if freeze["reference_local_alias_max_abs_error"] <= 1e-12 else "HARD_FAIL", "observed": freeze["reference_local_alias_max_abs_error"], "required": "<=1e-12", "detail": "atomic top-ten local scores reproduce Stage-A top-ten"},
        {"gate_id": "P2A_TOP10_COMBINED_ALIAS", "status": "PASS" if freeze["reference_combined_alias_max_abs_error"] <= 1e-12 else "HARD_FAIL", "observed": freeze["reference_combined_alias_max_abs_error"], "required": "<=1e-12", "detail": "atomic top-ten combined scores reproduce Stage-A top-ten before threshold refit"},
        {"gate_id": "C1_SUFFIX_INVARIANCE", "status": "PASS" if freeze["suffix_invariance_max_abs_error"] <= 1e-12 else "HARD_FAIL", "observed": freeze["suffix_invariance_max_abs_error"], "required": "<=1e-12", "detail": "every response passed deterministic prefix replay cuts"},
        {"gate_id": "C1_WORST_CELL_HARD_BOUND", "status": "HARD_FAIL" if hard_failure else "PASS", "observed": min(float(row["worst_unit_delta"]) for row in primary.values()), "required": f">={HARD_WORST_CELL_BOUND}", "detail": "minimum across both required comparators"},
        {"gate_id": "C1_PREMISE_PROMOTION", "status": "PASS" if promotion else "FAIL", "observed": str(promotion).lower(), "required": "all promotion gates versus top-ten and top-five", "detail": "Phase-2R-B SWVar template opens only if this premise passes"},
    ]
    for comparator, row in primary.items():
        prefix = "C1_VS_" + comparator
        checks = (
            ("POINT_BENEFIT", row["delta"], f">={BENEFIT}", float(row["delta"]) >= BENEFIT),
            ("CI_BENEFIT", row["ci_low"], f">{BENEFIT}", float(row["ci_low"]) > BENEFIT),
            ("NONNEGATIVE_CELLS", int(row["wins"]) + int(row["ties"]), ">=6", int(row["wins"]) + int(row["ties"]) >= 6),
            ("WORST_CELL", row["worst_unit_delta"], f">={PROMOTION_WORST_CELL_BOUND}", float(row["worst_unit_delta"]) >= PROMOTION_WORST_CELL_BOUND),
            ("EXACT_ERROR", by_metric[(comparator, "first_error_exact")]["delta"], f">={COMPONENT_BOUND}", float(by_metric[(comparator, "first_error_exact")]["delta"]) >= COMPONENT_BOUND),
            ("CLEAN_ABSTENTION", by_metric[(comparator, "clean_abstention_accuracy")]["delta"], f">={COMPONENT_BOUND}", float(by_metric[(comparator, "clean_abstention_accuracy")]["delta"]) >= COMPONENT_BOUND),
        )
        gates.extend({
            "gate_id": f"{prefix}_{name}", "status": "PASS" if passed else "FAIL",
            "observed": observed, "required": required, "detail": f"C1 versus {comparator}",
        } for name, observed, required, passed in checks)

    flips, flip_summary = prediction_flips(
        arm_results[CANDIDATE]["decisions"], arm_results[REFERENCE]["decisions"]
    )
    eval_root = output / "evaluation"
    eval_root.mkdir(parents=True, exist_ok=False)
    write_csv(eval_root / "PROCESSBENCH_DECISIONS.csv", [
        row for arm in ARMS for row in arm_results[arm]["decisions"]
    ])
    write_csv(eval_root / "PROCESSBENCH_BY_CELL.csv", [
        row for arm in ARMS for row in arm_results[arm]["by_cell"]
    ])
    write_csv(eval_root / "PROCESSBENCH_PANELS.csv", [
        row for arm in ARMS for row in arm_results[arm]["panels"]
    ])
    atomic_write_npz(eval_root / "PROCESSBENCH_BOOTSTRAP_SAMPLES.npz", {
        f"{arm}__{metric}": values
        for arm in ARMS for metric, values in arm_results[arm]["samples"].items()
    })
    atomic_write_json(eval_root / "CALIBRATION_LEDGERS.json", {
        "schema": "reasoning-localization-phase2-atomic-c1-calibration-v1",
        "arms": {arm: arm_results[arm]["ledgers"] for arm in ARMS},
    })
    write_csv(eval_root / "PAIRWISE_CONTRASTS.csv", contrasts)
    write_csv(eval_root / "STEP_LENGTH_STRATA.csv", p2r._length_strata(
        arm_results[CANDIDATE]["decisions"], arm_results[CANDIDATE]["by_cell"]
    ))
    write_csv(eval_root / "SELECTED_STEP_LENGTH.csv", p2r._selected_length_distribution(
        arm_results[CANDIDATE]["decisions"]
    ))
    write_csv(eval_root / "PREDICTION_FLIPS.csv", flips)
    write_csv(eval_root / "PREDICTION_FLIP_SUMMARY.csv", flip_summary)
    write_csv(eval_root / "GATES.csv", gates)
    candidate_panel = next(
        row for row in arm_results[CANDIDATE]["panels"] if row["metric_id"] == "official_macro_f1"
    )
    reference_panel = next(
        row for row in arm_results[REFERENCE]["panels"] if row["metric_id"] == "official_macro_f1"
    )
    status = "HARD_FAIL" if hard_failure else "COMPLETE"
    summary = {
        "schema": "reasoning-localization-phase2-atomic-c1-evaluation-v1",
        "variant_id": CANDIDATE,
        "status": status,
        "premise_gate_passed": promotion,
        "candidate_macro_f1": candidate_panel["value"],
        "candidate_macro_f1_ci": [candidate_panel["ci_low"], candidate_panel["ci_high"]],
        "atomic_reference_macro_f1": reference_panel["value"],
        "primary_contrasts": {key: value for key, value in primary.items()},
        "prediction_flips_vs_atomic_reference": sum(row["changed"] == "true" for row in flips),
        "reference_local_alias_max_abs_error": freeze["reference_local_alias_max_abs_error"],
        "reference_combined_alias_max_abs_error": freeze["reference_combined_alias_max_abs_error"],
        "suffix_invariance_max_abs_error": freeze["suffix_invariance_max_abs_error"],
        "bootstrap_draws": p1.BOOTSTRAP_DRAWS,
        "bootstrap_seed": p1.BOOTSTRAP_SEED,
        "peak_memory_bytes": peak_memory_bytes(),
    }
    summary["payload_sha256"] = payload_sha(summary)
    atomic_write_json(eval_root / "SUMMARY.json", summary)
    outputs = (
        "PROCESSBENCH_DECISIONS.csv", "PROCESSBENCH_BY_CELL.csv", "PROCESSBENCH_PANELS.csv",
        "PROCESSBENCH_BOOTSTRAP_SAMPLES.npz", "CALIBRATION_LEDGERS.json",
        "PAIRWISE_CONTRASTS.csv", "STEP_LENGTH_STRATA.csv", "SELECTED_STEP_LENGTH.csv",
        "PREDICTION_FLIPS.csv", "PREDICTION_FLIP_SUMMARY.csv", "GATES.csv", "SUMMARY.json",
    )
    manifest = {
        "schema": "reasoning-localization-phase2-atomic-c1-evaluation-manifest-v1",
        "variant_id": CANDIDATE,
        "status": status,
        "score_freeze_sha256": sha256_file(output / "score_freeze/SCORE_FREEZE_MANIFEST.json"),
        "execution_registry_sha256": sha256_file(Path(registry["registry_path"])),
        "outputs": [
            {"path": name, "sha256": sha256_file(eval_root / name), "bytes": (eval_root / name).stat().st_size}
            for name in outputs
        ],
    }
    manifest["payload_sha256"] = payload_sha(manifest)
    atomic_write_json(eval_root / "EVALUATION_MANIFEST.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=p1.DEFAULT_RELEASE)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    release, registry_path, output = args.release.resolve(), args.registry.resolve(), args.output.resolve()
    registry = load_registry(registry_path, release)
    registry["registry_path"] = str(registry_path)
    started = time.perf_counter()
    freeze = freeze_scores(release, output, registry)
    summary = evaluate_scores(release, output, registry, freeze)
    run = {
        "schema": "reasoning-localization-phase2-atomic-c1-run-v1",
        "variant_id": CANDIDATE,
        "status": summary["status"],
        "execution_registry_sha256": sha256_file(registry_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "score_freeze_manifest_sha256": sha256_file(output / "score_freeze/SCORE_FREEZE_MANIFEST.json"),
        "evaluation_manifest_sha256": sha256_file(output / "evaluation/EVALUATION_MANIFEST.json"),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
    }
    run["payload_sha256"] = payload_sha(run)
    atomic_write_json(output / "RUN_MANIFEST.json", run)
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
