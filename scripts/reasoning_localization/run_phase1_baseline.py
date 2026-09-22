#!/usr/bin/env python3
"""Freeze and evaluate one registered Reasoning Localization Phase-1 baseline.

The score-freeze half imports no target-bearing module or label artifact.  The
evaluation half runs only after the complete score manifest has been written
and its source hashes have been rechecked.  R0--R4 share the same prepared
rows, response detector, five-fold evaluator, step spans, and bootstrap stream.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
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
from spectral_utils.token_local_fusion import (  # noqa: E402
    fit_local_equal_family,
    prepare_localization_cell,
)
from spectral_utils.fixed_application_pipelines import SHARED_TOKEN_VIEWS  # noqa: E402


PROGRAM_ROOT = REPO / "results/reasoning_localization_03662_v1"
DEFAULT_RELEASE = Path(
    "/Users/osegev/Desktop/hallucination_detection/.worktrees/"
    "reconstruction-science-run-v1/results/reconstruction_benchmark_v1/"
    "releases/2026-08-24_localization_v1"
)
VARIANTS = (
    "R0_ENTROPY_MAX",
    "R1_ENTROPY_TOP5",
    "R2_FAMILY6_TOP5_CURRENT",
    "R3_IU29",
    "R4_MIND_GAP",
)
MODELS = ("qwen3_4b", "qwen3_8b", "llama31_8b")
QWEN_MODELS = MODELS[:2]
FAMILIES = ("gsm8k", "math", "olympiadbench", "omnimath")
PB_CELLS = tuple(
    f"processbench_{family}_{model}" for model in MODELS for family in FAMILIES
)
PRM_CELL = "prmbench_response_qwen3_8b"
CELLS = PB_CELLS + (PRM_CELL,)
GLOBAL_METHOD = "equal_feature_mean"
STRICT_R3_SYSTEM = "equal_feature_mean__loc_geomean_v1"
PB_METRICS = (
    "official_macro_f1",
    "first_error_exact",
    "first_error_within_one",
    "clean_abstention_accuracy",
    "overall_decision_accuracy",
)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026083001
SCHEMA = "reasoning-localization-03662-phase1-baseline-v1"
EVIDENCE_MODULE = REPO / "scripts/gl_liu_v1/localization/evidence_drop.py"


class Phase1Error(RuntimeError):
    """Fail-closed Phase-1 contract error."""


def _payload_sha(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _peak_memory_bytes() -> int:
    """Return the process high-water mark in bytes on supported hosts."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise Phase1Error(f"refusing to write empty table: {path.name}")
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_evidence_module() -> Any:
    spec = importlib.util.spec_from_file_location("reasoning_phase1_evidence_drop", EVIDENCE_MODULE)
    if spec is None or spec.loader is None:
        raise Phase1Error(f"cannot load Evidence-Drop helper: {EVIDENCE_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_source_hashes(registry: Mapping[str, Any]) -> None:
    for row in registry["frozen_sources"]:
        path = Path(row["path"])
        if not path.is_absolute():
            path = REPO / path
        if not path.is_file():
            raise Phase1Error(f"missing frozen source {row['role']}: {path}")
        observed = sha256_file(path)
        if observed != row["sha256"]:
            raise Phase1Error(
                f"frozen source changed for {row['role']}: expected {row['sha256']} observed {observed}"
            )


def load_execution_registry(path: Path, variant_id: str, release: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "reasoning-localization-phase1-execution-registry-v1",
        "status": "FROZEN_BEFORE_RUN",
        "variant_id": variant_id,
        "variant_order": list(VARIANTS),
        "processbench_cells": list(PB_CELLS),
        "prmbench_cell": PRM_CELL,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    for key, value in required.items():
        if registry.get(key) != value:
            raise Phase1Error(f"execution registry mismatch for {key}")
    if Path(registry["release_root"]).resolve() != release.resolve():
        raise Phase1Error("execution registry release root mismatch")
    if registry.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise Phase1Error("execution registry runner hash mismatch")
    _require_source_hashes(registry)
    return registry


def topk_step_mean(
    token_risk: Sequence[float], starts: Sequence[int], ends: Sequence[int], *, k: int = 5
) -> np.ndarray:
    risk = np.asarray(token_risk, dtype=np.float64)
    starts_i = np.asarray(starts, dtype=np.int64)
    ends_i = np.asarray(ends, dtype=np.int64)
    if starts_i.shape != ends_i.shape or np.any(starts_i < 0) or np.any(ends_i <= starts_i):
        raise ValueError("malformed step spans")
    output = np.empty(len(starts_i), dtype=np.float64)
    for index, (lo, hi) in enumerate(zip(starts_i, ends_i)):
        values = risk[int(lo):int(hi)]
        take = min(int(k), len(values))
        output[index] = float(np.mean(np.partition(values, len(values) - take)[-take:]))
    return output


def mindgap_step_scores(cell: Any, evidence_module: Any) -> np.ndarray:
    """Apply the frozen EMA-drop rule, resetting state at every response.

    The sealed mixed-v2 input retains a confidence-oriented affine transform of
    renormalized top-k entropy rather than raw top-20 log probabilities.  A
    positive affine transform preserves EMA drop locations and, after the
    mandatory within-cell empirical rank, the direct reference ordering.  This
    is a repository same-access adaptation, not a paper-exact score in raw nats.
    """

    index = SHARED_TOKEN_VIEWS.index("topk_entropy_series")
    evidence = np.asarray(cell.token_confidence[:, index], dtype=np.float64)
    output = np.empty(len(cell.segment_starts), dtype=np.float64)
    for row_index, (token_lo, token_hi) in enumerate(
        zip(cell.token_offsets[:-1], cell.token_offsets[1:])
    ):
        token_lo_i, token_hi_i = int(token_lo), int(token_hi)
        trace = evidence[token_lo_i:token_hi_i]
        smooth = evidence_module.ema(trace, span=5)
        flux = np.diff(smooth)
        tolerance = evidence_module._flux_tol(trace)
        segment_lo, segment_hi = map(
            int, cell.segment_offsets[row_index:row_index + 2]
        )
        for segment_index in range(segment_lo, segment_hi):
            lo = int(cell.segment_starts[segment_index]) - token_lo_i
            hi = int(cell.segment_ends[segment_index]) - token_lo_i
            a, b = max(lo - 1, 0), min(hi - 1, len(flux))
            values = flux[a:b]
            negative = values[values < -tolerance]
            output[segment_index] = float(-np.min(negative)) if len(negative) else 0.0
    if not np.isfinite(output).all():
        raise Phase1Error(f"{cell.cell_id}: Mind-the-Gap produced non-finite step scores")
    return output


def combine_with_common_detector(cell: Any, local_step: Sequence[float]) -> np.ndarray:
    local = np.asarray(local_step, dtype=np.float64)
    if local.shape != np.asarray(cell.segment_starts).shape or not np.isfinite(local).all():
        raise Phase1Error(f"{cell.cell_id}: malformed local step score")
    method_ids = tuple(map(str, cell.method_ids))
    if GLOBAL_METHOD not in method_ids:
        raise Phase1Error(f"{cell.cell_id}: missing {GLOBAL_METHOD} response detector")
    response = np.asarray(
        cell.response_scores[method_ids.index(GLOBAL_METHOD)], dtype=np.float64
    )
    counts = np.diff(np.asarray(cell.segment_offsets, dtype=np.int64))
    score = np.sqrt(
        empirical_midrank(local) * np.repeat(empirical_midrank(response), counts)
    )
    if not np.isfinite(score).all():
        raise Phase1Error(f"{cell.cell_id}: combined scores are non-finite")
    return score


def _strict_r3_scores(release: Path, cell: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = release / "build_A/localization/fit/cells" / cell.cell_id / "scores.npz"
    arrays = load_npz_no_pickle(path)
    row_ids = tuple(map(str, arrays["row_ids"].tolist()))
    if row_ids != tuple(cell.row_ids) or not np.array_equal(arrays["segment_offsets"], cell.segment_offsets):
        raise Phase1Error(f"{cell.cell_id}: strict R3 row/span alignment failed")
    systems = tuple(map(str, arrays["system_ids"].tolist()))
    if STRICT_R3_SYSTEM not in systems:
        raise Phase1Error(f"{cell.cell_id}: strict R3 system missing")
    local = np.asarray(arrays["token_step_score"], dtype=np.float64)
    strict = np.asarray(arrays["system_scores"][systems.index(STRICT_R3_SYSTEM)], dtype=np.float64)
    reconstructed = combine_with_common_detector(cell, local)
    error = float(np.max(np.abs(strict - reconstructed)))
    if error > 1e-12:
        raise Phase1Error(f"{cell.cell_id}: strict R3 reconstruction error {error}")
    return local, strict, {
        "strict_score_artifact": str(path),
        "strict_score_sha256": sha256_file(path),
        "strict_system_id": STRICT_R3_SYSTEM,
        "reconstruction_max_abs_error": error,
    }


def score_cell(variant_id: str, cell: Any, release: Path, evidence_module: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    entropy_index = SHARED_TOKEN_VIEWS.index("entropy_series")
    entropy_risk = -np.asarray(cell.token_confidence[:, entropy_index], dtype=np.float64)
    if variant_id == "R0_ENTROPY_MAX":
        local = np.asarray([
            np.max(entropy_risk[int(lo):int(hi)])
            for lo, hi in zip(cell.segment_starts, cell.segment_ends)
        ], dtype=np.float64)
        diagnostics = {"representation": "mixed-v2 entropy coordinate; raw-entropy order equivalent", "reducer": "step_max"}
    elif variant_id == "R1_ENTROPY_TOP5":
        local = topk_step_mean(entropy_risk, cell.segment_starts, cell.segment_ends, k=5)
        diagnostics = {"representation": "mixed-v2 entropy coordinate; raw-entropy order equivalent", "reducer": "step_top5mean"}
    elif variant_id == "R2_FAMILY6_TOP5_CURRENT":
        preparation = prepare_localization_cell(cell)
        fitted = fit_local_equal_family(preparation)
        local = topk_step_mean(
            fitted.token_risk, cell.segment_starts, cell.segment_ends, k=5
        )
        diagnostics = {
            "representation": "current equal non-structural provenance-family mean",
            "reducer": "step_top5mean",
            "fit": dict(fitted.diagnostics),
            "preparation": dict(preparation.diagnostics),
        }
    elif variant_id == "R3_IU29":
        return _strict_r3_scores(release, cell)
    elif variant_id == "R4_MIND_GAP":
        local = mindgap_step_scores(cell, evidence_module)
        diagnostics = {
            "representation": "confidence-oriented affine top-k entropy coordinate",
            "temporal_transform": "adjusted EMA span 5; strongest negative flux in each step",
            "fidelity": "same-access repository adaptation; not paper-exact raw top-20 nats",
        }
    else:  # pragma: no cover - guarded by argparse and registry
        raise Phase1Error(f"unknown variant {variant_id}")
    return local, combine_with_common_detector(cell, local), diagnostics


def freeze_scores(
    variant_id: str, release: Path, output: Path, registry: Mapping[str, Any]
) -> dict[str, Any]:
    score_root = output / "score_freeze"
    if output.exists():
        raise FileExistsError(f"variant output already exists: {output}")
    score_root.mkdir(parents=True, exist_ok=False)
    input_root = release / "build_A/localization/inputs"
    manifest_path = input_root / "MANIFEST.json"
    manifest = validate_fit_manifest(manifest_path, input_root=input_root)
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    if set(CELLS) != set(by_cell):
        raise Phase1Error("prepared localization cell roster differs from the exact 12+PRM contract")
    evidence_module = _load_evidence_module()
    records = []
    start_time = time.perf_counter()
    for position, cell_id in enumerate(CELLS, start=1):
        record = by_cell[cell_id]
        input_path = input_root / str(record["artifact_path"])
        cell = load_prepared_localization_cell(input_path, record)
        started = time.perf_counter()
        local, combined, diagnostics = score_cell(variant_id, cell, release, evidence_module)
        target = score_root / "cells" / cell_id
        target.mkdir(parents=True, exist_ok=False)
        arrays = {
            "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
            "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
            "local_step_scores": np.asarray(local, dtype="<f8"),
            "combined_step_scores": np.asarray(combined, dtype="<f8"),
        }
        score_path = target / "scores.npz"
        score_sha = atomic_write_npz(score_path, arrays)
        cell_record = {
            "schema": "reasoning-localization-phase1-cell-score-v1",
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
            "token_transform_sha256": str(cell.token_transform_sha256),
            "score_file": "scores.npz",
            "score_sha256": score_sha,
            "fit_seconds": time.perf_counter() - started,
            "peak_memory_bytes": _peak_memory_bytes(),
            "labels_seen_during_fit": False,
            "targets_accessed_during_fit": False,
            "common_response_detector": GLOBAL_METHOD,
            "combination": "sqrt(empirical_midrank(local_step) * empirical_midrank(equal_feature_mean_response))",
            "diagnostics": diagnostics,
        }
        cell_record["payload_sha256"] = _payload_sha(cell_record)
        record_path = target / "RECORD.json"
        atomic_write_json(record_path, cell_record)
        records.append({
            "cell_id": cell_id,
            "record_path": record_path.relative_to(score_root).as_posix(),
            "record_sha256": sha256_file(record_path),
            "score_sha256": score_sha,
        })
        print(f"score-freeze {variant_id}: {cell_id} ({position}/{len(CELLS)})", flush=True)
    _require_source_hashes(registry)
    freeze = {
        "schema": SCHEMA,
        "stage": "target_free_score_freeze",
        "variant_id": variant_id,
        "cells": list(CELLS),
        "processbench_cells": list(PB_CELLS),
        "prmbench_cell": PRM_CELL,
        "common_response_detector": GLOBAL_METHOD,
        "labels_seen_during_fit": False,
        "targets_accessed_during_fit": False,
        "input_manifest_sha256": sha256_file(manifest_path),
        "execution_registry_sha256": sha256_file(Path(registry["registry_path"])),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.perf_counter() - start_time,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "records": records,
        "complete": len(records) == len(CELLS),
    }
    freeze["payload_sha256"] = _payload_sha(freeze)
    atomic_write_json(score_root / "SCORE_FREEZE_MANIFEST.json", freeze)
    return freeze


def _verified_scores(score_root: Path, freeze: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if freeze.get("complete") is not True or freeze.get("labels_seen_during_fit") is not False:
        raise Phase1Error("score freeze is incomplete or label-contaminated")
    output: dict[str, dict[str, Any]] = {}
    for item in freeze["records"]:
        record_path = score_root / item["record_path"]
        if sha256_file(record_path) != item["record_sha256"]:
            raise Phase1Error(f"cell record changed: {item['cell_id']}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        score_path = record_path.parent / record["score_file"]
        if sha256_file(score_path) != item["score_sha256"]:
            raise Phase1Error(f"cell scores changed: {item['cell_id']}")
        arrays = load_npz_no_pickle(score_path)
        if set(arrays) != {"row_ids", "segment_offsets", "local_step_scores", "combined_step_scores"}:
            raise Phase1Error(f"cell score schema changed: {item['cell_id']}")
        output[item["cell_id"]] = {"record": record, "arrays": arrays}
    if tuple(output) != CELLS:
        raise Phase1Error("verified score order differs from frozen cell order")
    return output


def _load_pb_labels(release: Path) -> dict[str, dict[str, tuple[str, int]]]:
    path = release / "build_A/localization/evaluation/localization_decisions.csv"
    labels = {cell_id: {} for cell_id in PB_CELLS}
    for row in _read_csv(path):
        cell_id = row["cell_id"]
        if cell_id not in labels or row["system_id"] != "deem_b3__loc_geomean_v1":
            continue
        row_id = row["row_id"]
        if row_id in labels[cell_id]:
            raise Phase1Error(f"duplicate ProcessBench label: {cell_id}/{row_id}")
        labels[cell_id][row_id] = (row["group_id"], int(row["true_first_error"]))
    return labels


def _bootstrap_pb_panel(
    decisions: Sequence[Mapping[str, Any]], models: Sequence[str]
) -> dict[str, np.ndarray]:
    models = tuple(models)
    indicators = {}
    for family in FAMILIES:
        ids = sorted({
            str(row["group_id"]) for row in decisions
            if row["slice_id"] == family and row["model_id"] in models
        })
        index = {group_id: value for value, group_id in enumerate(ids)}
        values = np.zeros((len(ids), len(models), 5), dtype=np.float64)
        seen = set()
        for row in decisions:
            if row["slice_id"] != family or row["model_id"] not in models:
                continue
            key = (row["group_id"], row["model_id"])
            if key in seen:
                raise Phase1Error("duplicate ProcessBench bootstrap unit")
            seen.add(key)
            target, prediction = int(row["true_first_error"]), int(row["prediction_step"])
            error = target != -1
            values[index[row["group_id"]], models.index(row["model_id"])] = (
                float(error), float(not error), float(error and prediction == target),
                float(error and prediction != -1 and abs(prediction - target) <= 1),
                float((not error) and prediction == -1),
            )
        if len(seen) != len(ids) * len(models):
            raise Phase1Error(f"incomplete ProcessBench paired panel for {family}")
        indicators[family] = values
    result = {metric: np.empty(BOOTSTRAP_DRAWS, dtype=np.float64) for metric in PB_METRICS}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    offset, chunk = 0, 200
    while offset < BOOTSTRAP_DRAWS:
        size = min(chunk, BOOTSTRAP_DRAWS - offset)
        per_family = []
        for family in FAMILIES:
            values = indicators[family]
            picks = rng.integers(0, len(values), size=(size, len(values)))
            counts = values[picks].sum(axis=1)
            n_error, n_clean = counts[:, :, 0], counts[:, :, 1]
            exact = counts[:, :, 2] / n_error
            within = counts[:, :, 3] / n_error
            abstain = counts[:, :, 4] / n_clean
            f1 = np.divide(2 * exact * abstain, exact + abstain, out=np.zeros_like(exact), where=(exact + abstain) > 0)
            accuracy = (counts[:, :, 2] + counts[:, :, 4]) / (n_error + n_clean)
            per_family.append(np.stack((f1, exact, within, abstain, accuracy), axis=-1))
        sample = np.stack(per_family).mean(axis=(0, 2))
        for metric_index, metric in enumerate(PB_METRICS):
            result[metric][offset:offset + size] = sample[:, metric_index]
        offset += size
    return result


def _evaluate_processbench(
    verified: Mapping[str, Mapping[str, Any]], labels: Mapping[str, Mapping[str, tuple[str, int]]], evaluation: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    rows_by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
    for cell_id in PB_CELLS:
        record, arrays = verified[cell_id]["record"], verified[cell_id]["arrays"]
        row_ids = tuple(map(str, arrays["row_ids"].tolist()))
        if set(row_ids) != set(labels[cell_id]):
            raise Phase1Error(f"{cell_id}: score/label population mismatch")
        offsets = np.asarray(arrays["segment_offsets"], dtype=np.int64)
        scores = np.asarray(arrays["combined_step_scores"], dtype=np.float64)
        for row_index, row_id in enumerate(row_ids):
            lo, hi = map(int, offsets[row_index:row_index + 2])
            group_id, first_error = labels[cell_id][row_id]
            rows_by_model[record["model_id"]].append({
                "row_id": row_id, "group_id": group_id, "slice_id": record["slice_id"],
                "first_error": first_error, "step_scores": scores[lo:hi].tolist(),
                "cell_id": cell_id, "model_id": record["model_id"],
            })
    decisions, by_cell, by_model = [], [], []
    for model in MODELS:
        result = evaluation.crossfit_processbench_threshold(rows_by_model[model])
        by_model.append({"model_id": model, **result["metrics"]["aggregate"]})
        source = {row["row_id"]: row for row in rows_by_model[model]}
        for row in result["decisions"]:
            parent = source[row["row_id"]]
            decisions.append({
                "model_id": model, "cell_id": parent["cell_id"], "slice_id": parent["slice_id"],
                "row_id": row["row_id"], "group_id": parent["group_id"],
                "fold": int(row["fold"]), "true_first_error": int(parent["first_error"]),
                "prediction_step": int(row["prediction_step"]),
            })
        for family, metrics in result["metrics"]["per_subset"].items():
            by_cell.append({
                "model_id": model, "slice_id": family,
                "cell_id": f"processbench_{family}_{model}",
                **{name: metrics[name] for name in PB_METRICS},
                "n_examples": metrics["n_examples"], "n_error": metrics["n_error"], "n_clean": metrics["n_clean"],
            })
    panels, samples = [], {}
    for panel_id, panel_models in (("current_common_eight_qwen", QWEN_MODELS), ("current_full_twelve_cell", MODELS)):
        selected = [row for row in by_cell if row["model_id"] in panel_models]
        point = {metric: float(np.mean([row[metric] for row in selected])) for metric in PB_METRICS}
        sample = _bootstrap_pb_panel(decisions, panel_models)
        samples[panel_id] = sample
        for metric in PB_METRICS:
            panels.append({
                "population_id": panel_id, "metric_id": metric, "value": point[metric],
                "ci_low": float(np.quantile(sample[metric], 0.025)),
                "ci_high": float(np.quantile(sample[metric], 0.975)),
                "n_rows": sum(row["n_examples"] for row in selected),
                "n_groups": sum(400 if family == "gsm8k" else 1000 for family in FAMILIES),
            })
    return decisions, by_cell, panels, samples


def _evaluate_prmbench(verified: Mapping[str, Mapping[str, Any]], release: Path, evaluation: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = load_npz_no_pickle(release / "build_A/localization/evaluation/prmbench_steps.npz")
    arrays = verified[PRM_CELL]["arrays"]
    score_row_ids = tuple(map(str, arrays["row_ids"].tolist()))
    score_index = {row_id: index for index, row_id in enumerate(score_row_ids)}
    offsets = np.asarray(arrays["segment_offsets"], dtype=np.int64)
    all_scores = np.asarray(arrays["combined_step_scores"], dtype=np.float64)
    rows = []
    for response_index, (row_id, group_id, family, n_steps) in enumerate(zip(
        labels["response_row_ids"].astype(str), labels["group_ids"].astype(str),
        labels["error_families"].astype(str), np.diff(labels["step_offsets"]),
    )):
        index = score_index[row_id]
        lo, hi = map(int, offsets[index:index + 2])
        if hi - lo != int(n_steps):
            raise Phase1Error("PRMBench score/label step count mismatch")
        label_lo, label_hi = map(int, labels["step_offsets"][response_index:response_index + 2])
        for step_offset, (label, score) in enumerate(zip(labels["step_labels"][label_lo:label_hi], all_scores[lo:hi])):
            rows.append({
                "group_id": group_id, "response_row_id": row_id, "error_family": family,
                "step_index": step_offset, "step_label": int(label), "step_score": float(score),
            })
    panel = evaluation.prmbench_panel_metrics(rows)
    output = [{"slice_type": "overall", "slice_id": "all", **panel["overall"]}]
    output.extend({"slice_type": "error_family", "slice_id": family, **metrics} for family, metrics in panel["per_family"].items())
    output.extend([
        {"slice_type": "source_stratum", "slice_id": "prm_train", "status": "BLOCKED_METADATA_NOT_IN_SEALED_EVALUATOR", "auroc": None, "auprc": None, "n_examples": 0, "n_positive": 0, "n_negative": 0, "mean_risk": None, "risk_q90": None, "coverage": None, "positive_class": "annotated_error_step"},
        {"slice_type": "source_stratum", "slice_id": "prm_test", "status": "BLOCKED_METADATA_NOT_IN_SEALED_EVALUATOR", "auroc": None, "auprc": None, "n_examples": 0, "n_positive": 0, "n_negative": 0, "mean_risk": None, "risk_q90": None, "coverage": None, "positive_class": "annotated_error_step"},
    ])
    audit = {
        "n_scored_input_responses": len(score_row_ids), "n_evaluable_error_responses": len(labels["response_row_ids"]),
        "n_evaluable_steps": len(labels["step_labels"]), "all_nine_families_visible": panel["all_nine_families_visible"],
        "source_strata_available": False,
        "source_strata_reason": "the sealed prmbench_steps.npz evaluator exposes response IDs, source_idx groups, error families, and labels but not prm_train/prm_test membership",
    }
    return output, audit


def evaluate_scores(
    variant_id: str, release: Path, output: Path, registry: Mapping[str, Any], freeze: Mapping[str, Any]
) -> dict[str, Any]:
    _require_source_hashes(registry)
    score_root = output / "score_freeze"
    if sha256_file(score_root / "SCORE_FREEZE_MANIFEST.json") == "":  # pragma: no cover
        raise Phase1Error("unreachable empty score-freeze hash")
    verified = _verified_scores(score_root, freeze)
    evaluation = importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation")
    labels = _load_pb_labels(release)
    decisions, by_cell, panels, samples = _evaluate_processbench(verified, labels, evaluation)
    prm, prm_audit = _evaluate_prmbench(verified, release, evaluation)
    eval_root = output / "evaluation"
    eval_root.mkdir(parents=True, exist_ok=False)
    _write_csv(eval_root / "PROCESSBENCH_DECISIONS.csv", decisions)
    _write_csv(eval_root / "PROCESSBENCH_BY_CELL.csv", by_cell)
    _write_csv(eval_root / "PROCESSBENCH_PANELS.csv", panels)
    atomic_write_npz(eval_root / "PROCESSBENCH_BOOTSTRAP_SAMPLES.npz", {
        f"{population}__{metric}": values for population, by_metric in samples.items() for metric, values in by_metric.items()
    })
    _write_csv(eval_root / "PRMBENCH_SLICES.csv", prm)
    atomic_write_json(eval_root / "PRMBENCH_AUDIT.json", prm_audit)
    all_combined = [
        np.asarray(payload["arrays"]["combined_step_scores"], dtype=np.float64)
        for payload in verified.values()
    ]
    fold_by_group: dict[tuple[str, str], set[int]] = {}
    for row in decisions:
        key = (str(row["slice_id"]), str(row["group_id"]))
        fold_by_group.setdefault(key, set()).add(int(row["fold"]))
    r3_alias_error = max(
        float(payload["record"]["diagnostics"].get("reconstruction_max_abs_error", 0.0))
        for payload in verified.values()
    )
    gate_rows = [
        {"gate_id": "P1_SCORE_FREEZE_COMPLETE", "status": "PASS", "observed": len(verified), "required": len(CELLS), "detail": "exact registered 12 ProcessBench cells plus PRMBench cell"},
        {"gate_id": "P1_LABEL_FIREWALL", "status": "PASS", "observed": "labels opened after score-freeze manifest", "required": "no fit-side labels or targets", "detail": "score records and manifest declare labels_seen_during_fit=false and targets_accessed_during_fit=false"},
        {"gate_id": "P1_FINITE_AND_COMPLETE_SCORES", "status": "PASS" if all(np.isfinite(values).all() for values in all_combined) else "HARD_FAIL", "observed": sum(len(values) for values in all_combined), "required": "all registered step scores finite; no missing-score population change", "detail": "validated after frozen-score checksum verification"},
        {"gate_id": "P1_GROUP_SAFE_FOLDS", "status": "PASS" if all(len(folds) == 1 for folds in fold_by_group.values()) else "HARD_FAIL", "observed": max(map(len, fold_by_group.values())), "required": "one fold per source-question group across scorer copies", "detail": "checked on family x source-group keys"},
        {"gate_id": "P1_R3_EXACT_ALIAS", "status": "PASS" if variant_id != "R3_IU29" or r3_alias_error <= 1e-12 else "HARD_FAIL", "observed": r3_alias_error if variant_id == "R3_IU29" else "NOT_APPLICABLE", "required": "max absolute reconstruction error <= 1e-12", "detail": "strict-release alias check applies only to R3"},
        {"gate_id": "P1_PRM_SOURCE_STRATA", "status": "BLOCKED", "observed": "membership absent", "required": "prm_train/prm_test membership for source-stratum reporting", "detail": prm_audit["source_strata_reason"]},
    ]
    _write_csv(eval_root / "GATES.csv", gate_rows)
    overall_prm = next(row for row in prm if row["slice_type"] == "overall")
    qwen_f1 = next(row for row in panels if row["population_id"] == "current_common_eight_qwen" and row["metric_id"] == "official_macro_f1")
    full_f1 = next(row for row in panels if row["population_id"] == "current_full_twelve_cell" and row["metric_id"] == "official_macro_f1")
    summary = {
        "schema": "reasoning-localization-phase1-evaluation-v1", "variant_id": variant_id,
        "score_freeze_sha256": sha256_file(score_root / "SCORE_FREEZE_MANIFEST.json"),
        "labels_opened_only_after_complete_score_freeze": True,
        "processbench_qwen8_macro_f1": qwen_f1["value"], "processbench_qwen8_ci": [qwen_f1["ci_low"], qwen_f1["ci_high"]],
        "processbench_full12_macro_f1": full_f1["value"], "processbench_full12_ci": [full_f1["ci_low"], full_f1["ci_high"]],
        "prmbench_auroc": overall_prm["auroc"], "prmbench_auprc": overall_prm["auprc"],
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
        "source_strata_available": False, "new_model_inference": False,
        "peak_memory_bytes": _peak_memory_bytes(),
        "missing_score_count": int(sum(np.size(values) - np.count_nonzero(np.isfinite(values)) for values in all_combined)),
    }
    summary["payload_sha256"] = _payload_sha(summary)
    atomic_write_json(eval_root / "SUMMARY.json", summary)
    outputs = ("PROCESSBENCH_DECISIONS.csv", "PROCESSBENCH_BY_CELL.csv", "PROCESSBENCH_PANELS.csv", "PROCESSBENCH_BOOTSTRAP_SAMPLES.npz", "PRMBENCH_SLICES.csv", "PRMBENCH_AUDIT.json", "GATES.csv", "SUMMARY.json")
    manifest = {
        "schema": "reasoning-localization-phase1-evaluation-manifest-v1", "variant_id": variant_id,
        "score_freeze_sha256": summary["score_freeze_sha256"], "execution_registry_sha256": sha256_file(Path(registry["registry_path"])),
        "outputs": [{"path": name, "sha256": sha256_file(eval_root / name), "bytes": (eval_root / name).stat().st_size} for name in outputs],
    }
    manifest["payload_sha256"] = _payload_sha(manifest)
    atomic_write_json(eval_root / "EVALUATION_MANIFEST.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    registry_path = (args.registry or PROGRAM_ROOT / "phase_1" / f"{args.variant}_EXECUTION_REGISTRY.json").resolve()
    output = (args.output or PROGRAM_ROOT / "phase_1" / args.variant.lower()).resolve()
    release = args.release.resolve()
    registry = load_execution_registry(registry_path, args.variant, release)
    registry["registry_path"] = str(registry_path)
    freeze = freeze_scores(args.variant, release, output, registry)
    summary = evaluate_scores(args.variant, release, output, registry, freeze)
    run_manifest = {
        "schema": "reasoning-localization-phase1-run-manifest-v1", "variant_id": args.variant,
        "status": "COMPLETE", "execution_registry_sha256": sha256_file(registry_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "score_freeze_manifest_sha256": sha256_file(output / "score_freeze/SCORE_FREEZE_MANIFEST.json"),
        "evaluation_manifest_sha256": sha256_file(output / "evaluation/EVALUATION_MANIFEST.json"),
        "summary": summary,
    }
    run_manifest["payload_sha256"] = _payload_sha(run_manifest)
    atomic_write_json(output / "RUN_MANIFEST.json", run_manifest)
    print(json.dumps({"status": "COMPLETE", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
