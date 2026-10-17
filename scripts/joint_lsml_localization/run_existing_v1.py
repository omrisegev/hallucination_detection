#!/usr/bin/env python3
"""Sanitize, register, and freeze retrospective Joint L-SML localization scores.

The score path has no outcome import. Benchmark outcomes live in a separate
evaluator and may be opened only after an independent score-freeze audit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.dufs_liu_feature_contract import FEATURE_TRANSFORMS  # noqa: E402
from spectral_utils.feature_contract import confidence_sign_vector  # noqa: E402
from spectral_utils.fixed_application_pipelines import (  # noqa: E402
    SHARED_GLOBAL_FEATURES, SHARED_TOKEN_VIEWS, raw_token_feature_matrix,
)
from spectral_utils.joint_lsml_localization import (  # noqa: E402
    METHODS, fit_active23_arms, prepare_active23, score_active23_arms,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json, atomic_write_npz, load_npz_no_pickle, sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import (  # noqa: E402
    load_prepared_localization_token_cell, payload_sha256, validate_fit_manifest,
)


EXPERIMENT_ID = "JOINT_LSML_EXISTING_LOCALIZATION_V1"
RESULT_ROOT = REPO / "results/joint_lsml_existing_localization_v1"
SANITIZED_ROOT = RESULT_ROOT / "sanitized_target_free_inputs"
SANITIZED_MANIFEST = SANITIZED_ROOT / "MANIFEST.json"
REGISTRY = RESULT_ROOT / "EXECUTION_REGISTRY.json"
REGISTRATION_COMPLETE = RESULT_ROOT / "REGISTRATION_COMPLETE.json"
STRUCTURAL_LEDGER = RESULT_ROOT / "STRUCTURAL_LEDGER.json"
SCORE_ROOT = RESULT_ROOT / "score_freeze"
SCORE_MANIFEST = RESULT_ROOT / "SCORE_FREEZE_MANIFEST.json"
RUN_COMPLETE = RESULT_ROOT / "RUN_COMPLETE.json"

PROTOCOL = REPO / "docs/experiments/JOINT_LSML_EXISTING_LOCALIZATION_V1.md"
PRIOR_AUDIT = REPO / "docs/experiments/PRIOR_ORDER_AUDIT_JOINT_LSML_EXISTING_V1.md"
CORE = REPO / "spectral_utils/joint_lsml_localization.py"
JOINT_CORE = REPO / "spectral_utils/joint_lsml.py"
TEST_SOURCE = REPO / "tests/test_joint_lsml_localization.py"
ANALYSIS_PLAN = REPO / "configs/joint_lsml_existing_localization_v1.json"
EVALUATOR = REPO / "scripts/joint_lsml_localization/evaluate_existing_v1.py"
RUNNER = Path(__file__).resolve()
ORIENTATION = REPO / "results/joint_lsml_v1_r2/V2_ABSOLUTE_ORIENTATION_REGISTRY.json"
ROSTER = REPO / "results/joint_lsml_v1_r2/V2_GLOBAL_PRUNED_ROSTER.json"
DEFAULT_RELEASE = Path(
    "/Users/osegev/Desktop/hallucination_detection/.worktrees/"
    "reconstruction-science-run-v1/results/reconstruction_benchmark_v1/"
    "releases/2026-08-24_localization_v1"
)
SOURCE_ROOT = Path(
    "/Users/osegev/Desktop/hallucination_detection/.worktrees/"
    "reconstruction-benchmark-v1/results/reconstruction_benchmark_v1/"
    "source_overlays/external_final_answer_v1"
)

PB_CELLS = tuple(
    f"processbench_{subset}_{model}"
    for model in ("qwen3_4b", "qwen3_8b")
    for subset in ("gsm8k", "math", "olympiadbench", "omnimath")
)
PRM_CELL = "prmbench_response_qwen3_8b"
CELLS = PB_CELLS + (PRM_CELL,)
SEED = 2026090405
MATCH_TOLERANCE = 1e-9

RAW_RELATIVE = {
    **{
        f"processbench_{subset}_{model}":
        f"dataset_cache/repgrid/pb_{model}/processbench_{subset}.pkl"
        for model in ("qwen3_4b", "qwen3_8b")
        for subset in ("gsm8k", "math", "olympiadbench", "omnimath")
    },
    PRM_CELL: "dataset_cache/four_localization/"
    "prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl",
}


class ProtocolError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _source_hashes() -> dict[str, str]:
    paths = (
        RUNNER, CORE, JOINT_CORE, TEST_SOURCE, PROTOCOL, PRIOR_AUDIT,
        REPO / "tests/test_joint_lsml.py",
        ANALYSIS_PLAN, EVALUATOR,
        REPO / "spectral_utils/fixed_application_pipelines.py",
        REPO / "spectral_utils/token_local_fusion.py",
        REPO / "spectral_utils/fusion_utils.py",
        REPO / "spectral_utils/upcr.py",
        REPO / "spectral_utils/dependency_fusion.py",
        REPO / "spectral_utils/specrage_views.py",
        REPO / "spectral_utils/dufs_liu_feature_contract.py",
        REPO / "spectral_utils/feature_contract.py",
        REPO / "spectral_utils/fair_comparisons/evaluator.py",
        REPO / "spectral_utils/fair_comparisons/folds.py",
        REPO / "spectral_utils/paper_exact/evaluator.py",
        REPO / "spectral_utils/token_feature_views.py",
        REPO / "spectral_utils/feature_utils.py",
        REPO / "spectral_utils/reconstruction_benchmark/io.py",
        REPO / "spectral_utils/reconstruction_benchmark/localization_contract.py",
        REPO / "spectral_utils/reconstruction_benchmark/localization_postfreeze.py",
        REPO / "scripts/reasoning_localization/run_phase2_reducer.py",
    )
    return {str(path.relative_to(REPO)): sha256_file(path) for path in paths}


def _load_release(release: Path) -> tuple[Path, dict[str, Mapping[str, Any]]]:
    root = release / "build_A/localization/inputs"
    manifest = validate_fit_manifest(root / "MANIFEST.json", input_root=root)
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    if not set(CELLS).issubset(by_cell):
        raise ProtocolError("fit-safe release lacks the exact nine Qwen cells")
    return root, by_cell


def _allowed_raw_row(row: Mapping[str, Any], dataset_id: str) -> dict[str, Any]:
    identity = "id" if dataset_id == "processbench" else "idx"
    required = (
        identity, "token_entropies", "token_spilled_energies", "token_logsumexp",
        "top_k_logprobs", "gen_token_ids", "step_token_spans",
    )
    if any(name not in row for name in required):
        raise ProtocolError("raw telemetry row lacks a target-free member")
    return {name: row[name] for name in required}


def _prepared_signature(cell: Any, index: int) -> tuple[int, tuple[tuple[int, int], ...]]:
    token_lo, token_hi = map(int, cell.token_offsets[index:index + 2])
    seg_lo, seg_hi = map(int, cell.segment_offsets[index:index + 2])
    starts = np.asarray(cell.segment_starts[seg_lo:seg_hi], dtype=np.int64) - token_lo
    ends = np.asarray(cell.segment_ends[seg_lo:seg_hi], dtype=np.int64) - token_lo
    return token_hi - token_lo, tuple((int(lo), int(hi)) for lo, hi in zip(starts, ends))


def _raw_signature(row: Mapping[str, Any]) -> tuple[int, tuple[tuple[int, int], ...]]:
    return len(row["token_entropies"]), tuple(
        (int(lo), int(hi)) for lo, hi in row["step_token_spans"]
    )


def _correlation_match_cost(raw: np.ndarray, prepared: np.ndarray) -> float:
    current = confidence_sign_vector(SHARED_GLOBAL_FEATURES).astype(np.float64)
    oriented = np.asarray(raw, dtype=np.float64) * current[None, :]
    nonlinear = {
        index for index, name in enumerate(SHARED_GLOBAL_FEATURES)
        if FEATURE_TRANSFORMS.get(name, "raw") != "raw"
    }
    correlations = []
    for column in range(oriented.shape[1]):
        if column == 0 or column in nonlinear:
            continue
        left, right = oriented[:, column], np.asarray(prepared[:, column], dtype=np.float64)
        if np.std(left) <= 1e-10 or np.std(right) <= 1e-10:
            continue
        value = float(np.corrcoef(left, right)[0, 1])
        if np.isfinite(value):
            correlations.append(value)
    return 1.0 if len(correlations) < 10 else float(1.0 - np.median(correlations))


def _match_raw_rows(cell: Any, raw_rows: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    prepared_by_signature: dict[Any, list[int]] = {}
    for index in range(len(cell.row_ids)):
        prepared_by_signature.setdefault(_prepared_signature(cell, index), []).append(index)
    raw_by_signature: dict[Any, list[Mapping[str, Any]]] = {}
    for row in raw_rows:
        signature = _raw_signature(row)
        if signature in prepared_by_signature:
            raw_by_signature.setdefault(signature, []).append(row)
    mapping: dict[int, Mapping[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    for signature, prepared_indices in prepared_by_signature.items():
        candidates = raw_by_signature.get(signature, [])
        if len(candidates) < len(prepared_indices):
            raise ProtocolError("raw/prepared signature cohort cannot be joined")
        raw_matrices = [raw_token_feature_matrix(dict(row)) for row in candidates]
        cost = np.empty((len(prepared_indices), len(candidates)), dtype=np.float64)
        for left, index in enumerate(prepared_indices):
            lo, hi = map(int, cell.token_offsets[index:index + 2])
            prepared = np.asarray(cell.token_confidence[lo:hi], dtype=np.float64)
            for right, raw in enumerate(raw_matrices):
                cost[left, right] = _correlation_match_cost(raw, prepared)
        row_ind, col_ind = linear_sum_assignment(cost)
        if len(row_ind) != len(prepared_indices):
            raise ProtocolError("raw/prepared assignment is incomplete")
        for left, right in zip(row_ind.tolist(), col_ind.tolist()):
            index = prepared_indices[left]
            score = float(cost[left, right])
            if score > MATCH_TOLERANCE:
                raise ProtocolError(f"raw/prepared match exceeded tolerance: {score}")
            mapping[index] = candidates[right]
            audit.append({
                "opaque_row_id": str(cell.row_ids[index]),
                "match_cost": score,
                "signature_token_count": int(signature[0]),
                "signature_step_count": len(signature[1]),
            })
    if set(mapping) != set(range(len(cell.row_ids))):
        raise ProtocolError("raw/prepared mapping is incomplete")
    return [mapping[index] for index in range(len(cell.row_ids))], audit


def sanitize(release: Path, source_root: Path) -> None:
    if SANITIZED_ROOT.exists():
        raise ProtocolError("sanitized namespace already exists")
    input_root, by_cell = _load_release(release)
    records = []
    for cell_id in CELLS:
        record = by_cell[cell_id]
        cell = load_prepared_localization_token_cell(input_root / record["artifact_path"], record)
        source = source_root / RAW_RELATIVE[cell_id]
        with source.open("rb") as handle:
            container = pickle.load(handle)
        if not isinstance(container, Mapping):
            raise ProtocolError(f"{cell_id}: raw telemetry is not a mapping")
        raw_rows = [
            _allowed_raw_row(row, str(cell.dataset_id))
            for row in container.values() if isinstance(row, Mapping)
        ]
        matched, audit = _match_raw_rows(cell, raw_rows)
        parts = []
        offsets = [0]
        for row in matched:
            raw = raw_token_feature_matrix(dict(row))
            parts.append(raw)
            offsets.append(offsets[-1] + len(raw))
        arrays = {
            "raw": np.asarray(np.vstack(parts), dtype="<f8"),
            "token_offsets": np.asarray(offsets, dtype="<i8"),
            "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
            "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
            "segment_starts": np.asarray(cell.segment_starts, dtype="<i8"),
            "segment_ends": np.asarray(cell.segment_ends, dtype="<i8"),
        }
        target = SANITIZED_ROOT / f"{cell_id}.npz"
        artifact_hash = atomic_write_npz(target, arrays)
        records.append({
            "cell_id": cell_id,
            "dataset_id": str(cell.dataset_id),
            "model_id": str(cell.model_id),
            "slice_id": str(cell.slice_id),
            "artifact_path": target.name,
            "artifact_sha256": artifact_hash,
            "raw_source_path": str(source),
            "raw_source_sha256": sha256_file(source),
            "fit_safe_source_path": str(input_root / record["artifact_path"]),
            "fit_safe_source_sha256": sha256_file(input_root / record["artifact_path"]),
            "n_rows": len(cell.row_ids),
            "n_tokens": int(len(arrays["raw"])),
            "n_segments": int(len(arrays["segment_starts"])),
            "maximum_match_cost": max(float(row["match_cost"]) for row in audit),
            "matching_audit_sha256": payload_sha256(audit),
            "members": sorted(arrays),
        })
        del container, raw_rows, matched, parts, arrays
        print(f"sanitized {cell_id}", flush=True)
    manifest = {
        "schema": "joint-lsml-existing-target-free-sanitized-v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE",
        "cells": records,
        "raw_container_target_fields_materialized_but_never_indexed": True,
        "sanitized_members_are_target_free": True,
        "labels_accessed": False,
    }
    manifest["payload_sha256"] = payload_sha256(manifest)
    atomic_write_json(SANITIZED_MANIFEST, manifest)


def _contracts() -> tuple[dict[str, Any], dict[str, Any], list[int], list[int], list[str], list[str]]:
    orientation = json.loads(ORIENTATION.read_text())
    roster = json.loads(ROSTER.read_text())
    for payload, name in ((orientation, "orientation"), (roster, "roster")):
        claim = payload.get("payload_sha256")
        body = {key: value for key, value in payload.items() if key != "payload_sha256"}
        if not isinstance(claim, str) or payload_sha256(body) != claim:
            raise ProtocolError(f"{name} payload hash mismatch")
        if payload.get("status") != "FROZEN_BEFORE_V2_REGISTRATION":
            raise ProtocolError(f"{name} is not frozen")
        if payload.get("labels_accessed") is not False or payload.get("response_scores_materialized") is not False:
            raise ProtocolError(f"{name} crossed the target-free boundary")
    if orientation.get("schema") != "within-answer-confidence-orientation-v1":
        raise ProtocolError("orientation schema mismatch")
    if orientation.get("input_domain") != "raw_token_feature_matrix(SHARED_GLOBAL_FEATURES)":
        raise ProtocolError("orientation is not absolute raw-domain")
    if orientation.get("relative_to") is not None:
        raise ProtocolError("relative orientation is forbidden")
    if orientation.get("output_semantics") != "HIGHER_IS_MORE_CONFIDENT":
        raise ProtocolError("orientation output semantics mismatch")
    if roster.get("schema") != "within-answer-global-pruned-roster-v1":
        raise ProtocolError("roster schema mismatch")
    if roster.get("input_domain") != orientation.get("input_domain"):
        raise ProtocolError("orientation and roster input domains differ")
    if roster.get("source_artifact_sha256") != sha256_file(ORIENTATION):
        raise ProtocolError("roster does not bind the absolute orientation artifact")
    retained = list(map(int, roster["retained_global_indices"]))
    expected_retained = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 19, 20, 21, 23, 24, 25, 26, 27, 28]
    expected_excluded = [0, 5, 12, 17, 18, 22]
    if retained != expected_retained or list(map(int, roster["excluded_global_indices"])) != expected_excluded:
        raise ProtocolError("active-23 roster drifted or retained trace length")
    signs = list(map(int, orientation["confidence_signs"]))
    if len(signs) != 29 or any(sign not in (-1, 1) for sign in signs):
        raise ProtocolError("orientation signs are not exact 29 +/-1 values")
    ordered = orientation["ordered_features"]
    if [int(row["global_index"]) for row in ordered] != list(range(29)):
        raise ProtocolError("orientation global-index order drifted")
    streams = [str(row["stream_name"]) for row in ordered]
    raw_names = [str(row["raw_feature_name"]) for row in ordered]
    if streams != list(SHARED_TOKEN_VIEWS) or raw_names != list(SHARED_GLOBAL_FEATURES):
        raise ProtocolError("orientation feature order differs from runtime")
    if [streams[index] for index in retained] != list(roster["retained_stream_names"]):
        raise ProtocolError("active roster names and indices disagree")
    return orientation, roster, retained, signs, streams, raw_names


def register() -> None:
    if REGISTRY.exists() or REGISTRATION_COMPLETE.exists():
        raise ProtocolError("registration namespace already exists")
    if not SANITIZED_MANIFEST.exists():
        raise ProtocolError("sanitize must complete before registration")
    orientation, roster, retained, signs, streams, raw_names = _contracts()
    analysis_plan = json.loads(ANALYSIS_PLAN.read_text())
    if (
        analysis_plan.get("candidate") != METHODS[0]
        or tuple(analysis_plan.get("controls", ())) != tuple(METHODS[1:])
        or analysis_plan.get("candidate_slots") != 1
        or analysis_plan.get("efficacy_control_slots") != 3
        or analysis_plan.get("fit_seed") != SEED
        or analysis_plan.get("pairwise_diagnostic_cap") != 32768
        or analysis_plan.get("minimum_held_admissible_fraction") != 0.95
        or analysis_plan.get("minimum_weight_map_score_spearman") != 0.5
        or analysis_plan.get("promotion_allowed") is not False
        or analysis_plan.get("generalization_claim_allowed") is not False
    ):
        raise ProtocolError("analysis plan differs from the executable contract")
    sanitized = json.loads(SANITIZED_MANIFEST.read_text())
    for row in sanitized["cells"]:
        path = SANITIZED_ROOT / row["artifact_path"]
        if sha256_file(path) != row["artifact_sha256"]:
            raise ProtocolError("sanitized input hash drift")
    payload = {
        "schema": "joint-lsml-existing-localization-execution-registry-v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "REGISTERED_RETROSPECTIVE_OPENED_DEVELOPMENT",
        "git_head": _git_head(),
        "cells": list(CELLS),
        "processbench_cells": list(PB_CELLS),
        "prmbench_cell": PRM_CELL,
        "methods": list(METHODS),
        "candidate_count": 1,
        "control_count": 3,
        "fit_token_cap": 60_000,
        "K_range": [3, 4, 6, 8],
        "K_selection": "median_ari_then_mean_ari_then_minimum_ari_then_smaller_k",
        "minimum_held_admissible_fraction": 0.95,
        "minimum_weight_map_score_spearman": 0.50,
        "primary_weight_map": "hierarchical_joint_irrevocable",
        "seed": SEED,
        "pairwise_diagnostic_cap": 32_768,
        "pairwise_diagnostic_sampling": "deterministic_uniform_pair_sample_when_population_exceeds_cap",
        "analysis_plan_sha256": sha256_file(ANALYSIS_PLAN),
        "multiplicity_ledger_sha256": sha256_file(ANALYSIS_PLAN),
        "runtime_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "processbench_reducer": "fixed top min(10, step_length) mean; detector=max token risk",
        "prmbench_reducer": "max token risk inside official step span",
        "opened_population": True,
        "promotion_allowed": False,
        "generalization_claim_allowed": False,
        "claude_review_commit": "45f8b572e221164ff6ebe3fe9fff96c25828a49d",
        "label_import_allowed_in_this_runner": False,
        "orientation_sha256": sha256_file(ORIENTATION),
        "roster_sha256": sha256_file(ROSTER),
        "sanitized_manifest_sha256": sha256_file(SANITIZED_MANIFEST),
        "source_hashes": _source_hashes(),
        "retained_indices": retained,
        "confidence_signs": signs,
        "stream_names": streams,
        "raw_feature_names": raw_names,
        "orientation_payload_sha256": orientation["payload_sha256"],
        "roster_payload_sha256": roster["payload_sha256"],
        "labels_accessed": False,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    registry_hash = atomic_write_json(REGISTRY, payload)
    atomic_write_json(REGISTRATION_COMPLETE, {
        "status": "PASS", "registry_sha256": registry_hash,
        "labels_accessed": False,
    })


def _verify_registration() -> dict[str, Any]:
    if not REGISTRY.exists() or not REGISTRATION_COMPLETE.exists():
        raise ProtocolError("registration is incomplete")
    registry = json.loads(REGISTRY.read_text())
    completion = json.loads(REGISTRATION_COMPLETE.read_text())
    if completion.get("registry_sha256") != sha256_file(REGISTRY):
        raise ProtocolError("registration completion hash mismatch")
    expected = dict(registry)
    claimed = expected.pop("payload_sha256")
    if payload_sha256(expected) != claimed:
        raise ProtocolError("registry payload hash mismatch")
    if registry["source_hashes"] != _source_hashes():
        raise ProtocolError("registered source changed")
    if registry["orientation_sha256"] != sha256_file(ORIENTATION):
        raise ProtocolError("orientation changed")
    if registry["roster_sha256"] != sha256_file(ROSTER):
        raise ProtocolError("roster changed")
    if registry["sanitized_manifest_sha256"] != sha256_file(SANITIZED_MANIFEST):
        raise ProtocolError("sanitized manifest changed")
    runtime = {
        "python": sys.version.split()[0], "numpy": np.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
    }
    if registry.get("runtime_versions") != runtime:
        raise ProtocolError("registered numerical runtime changed")
    return registry


def _load_sanitized(cell_id: str, manifest_by_cell: Mapping[str, Any]) -> dict[str, np.ndarray]:
    record = manifest_by_cell[cell_id]
    path = SANITIZED_ROOT / record["artifact_path"]
    if sha256_file(path) != record["artifact_sha256"]:
        raise ProtocolError(f"{cell_id}: sanitized artifact hash drift")
    arrays = load_npz_no_pickle(path)
    expected = {"raw", "token_offsets", "row_ids", "segment_offsets", "segment_starts", "segment_ends"}
    if set(arrays) != expected:
        raise ProtocolError(f"{cell_id}: unsafe sanitized member roster")
    return arrays


def _candidate_summary(grouping: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in grouping.get("candidates", []):
        pairwise = np.asarray(row["pairwise_ari"], dtype=np.float64)
        output.append({
            "K": int(row["K"]), "group_sizes": list(row["group_sizes"]),
            "admissible": bool(row["admissible"]),
            "rejection_reason": row["rejection_reason"],
            "minimum_held_answer_group_size": int(min(min(sizes) for sizes in row["held_answer_group_sizes"])),
            "held_admissible_fraction": float(row["held_admissible_fraction"]),
            "all_held_admissible": bool(row["all_held_admissible"]),
            "median_ari": float(row["median_ari"]), "mean_ari": float(row["mean_ari"]),
            "minimum_ari": float(row["minimum_ari"]), "exact_fraction": float(row["exact_fraction"]),
            "pairwise_ari_population_count": int(row.get("pairwise_ari_population_count", len(pairwise))),
            "pairwise_ari_sampling": str(row.get("pairwise_ari_sampling", "all_pairs")),
            "pairwise_ari_sample_count": int(len(pairwise)),
            "pairwise_ari_summary": {
                "minimum": float(pairwise.min()), "median": float(np.median(pairwise)),
                "mean": float(pairwise.mean()), "maximum": float(pairwise.max()),
            },
        })
    return output


def _compact_fit(cell_id: str, fitted: Mapping[str, Any]) -> dict[str, Any]:
    if fitted["status"] != "FIT_COMPLETE":
        grouping = fitted["grouping"]
        return {
            "cell_id": cell_id, "panel": "PRMBench" if cell_id == PRM_CELL else "ProcessBench",
            "status": fitted["status"], "structural_fit_pass": False,
            "preparation": fitted["preparation"],
            "preprocessing_parameters": fitted["preprocessing_parameters"],
            "grouping": {"status": grouping["status"], "selection_rule": grouping["selection_rule"],
                         "candidates": _candidate_summary(grouping)},
            "labels_accessed": False,
        }
    grouping, joint = fitted["grouping"], fitted["joint_fit"]
    return {
        "cell_id": cell_id, "panel": "PRMBench" if cell_id == PRM_CELL else "ProcessBench",
        "status": fitted["status"], "structural_fit_pass": bool(fitted["structural_fit_pass"]),
        "preparation": fitted["preparation"],
        "preprocessing_parameters": fitted["preprocessing_parameters"],
        "grouping": {
            "status": grouping["status"], "K": int(grouping["K"]),
            "labels": grouping["labels"], "group_sizes": grouping["group_sizes"],
            "median_ari": float(grouping["median_ari"]), "mean_ari": float(grouping["mean_ari"]),
            "minimum_ari": float(grouping["minimum_ari"]), "exact_fraction": float(grouping["exact_fraction"]),
            "held_admissible_fraction": float(grouping["held_admissible_fraction"]),
            "selection_rule": grouping["selection_rule"], "candidates": _candidate_summary(grouping),
        },
        "joint_fit": {
            "converged": bool(joint.converged), "converged_starts": int(joint.converged_starts),
            "selected_start": int(joint.selected_start), "objective": float(joint.objective),
            "relative_offdiag_misfit": float(joint.relative_offdiag_misfit),
            "global_loading": joint.global_loading, "group_loading": joint.group_loading,
            "multistart_audit": joint.multistart_audit, "jacobian_audit": joint.jacobian_audit,
            "diagonal_audit": joint.diagonal_audit,
            "starts": [{
                "start": int(row.start), "converged": bool(row.converged),
                "failed_monotonicity": bool(row.failed_monotonicity), "sweeps": int(row.sweeps),
                "objective_trace": row.objective_trace, "model_change_trace": row.model_change_trace,
            } for row in joint.starts],
        },
        "hard_lsml_relative_offdiag_misfit": fitted["hard_lsml_relative_offdiag_misfit"],
        "joint_lower_misfit": fitted["joint_lower_misfit"],
        "weight_map_agreement": fitted["weight_map_agreement"],
        "weights": fitted["weights"], "diagnostics": fitted["diagnostics"],
        "labels_accessed": False,
    }


def _preparation(arrays: Mapping[str, np.ndarray], registry: Mapping[str, Any]):
    return prepare_active23(
        arrays["raw"], arrays["token_offsets"], arrays["row_ids"].astype(str),
        retained_indices=registry["retained_indices"],
        confidence_signs_29=registry["confidence_signs"],
        stream_names_29=registry["stream_names"], raw_feature_names_29=registry["raw_feature_names"],
    )


def _top10_step_scores(risk: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    output = np.empty(len(starts), dtype=np.float64)
    for index, (lo, hi) in enumerate(zip(starts, ends)):
        values = risk[int(lo):int(hi)]
        take = min(10, len(values))
        output[index] = float(np.mean(np.partition(values, len(values) - take)[-take:]))
    return output


def _fit_cell_worker(cell_id: str, position: int, registry: Mapping[str, Any], manifest_by_cell: Mapping[str, Any]):
    """Fit one target-free cell in an isolated process and return compact state."""
    arrays = _load_sanitized(cell_id, manifest_by_cell)
    prep = _preparation(arrays, registry)
    fitted = fit_active23_arms(prep, seed=SEED + 100_000 * position)
    compact = _compact_fit(cell_id, fitted)
    weights = (
        {name: np.asarray(value, dtype=np.float64) for name, value in fitted["weights"].items()}
        if fitted["status"] == "FIT_COMPLETE" and fitted.get("structural_fit_pass") else None
    )
    return cell_id, compact, weights


def _score_cell(cell_id: str, arrays: Mapping[str, np.ndarray], prep: Any, weights: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fitted_stub = {"status": "FIT_COMPLETE", "structural_fit_pass": True, "weights": weights}
    curves = score_active23_arms(prep, fitted_stub)
    methods = list(METHODS)
    if cell_id != PRM_CELL:
        detector = np.empty((len(prep.row_ids), len(methods)), dtype=np.float64)
        locator = np.empty((len(prep.row_ids), len(methods)), dtype=np.int64)
        for column, method in enumerate(methods):
            risk = curves[method]
            detector[:, column] = [
                float(np.max(risk[int(lo):int(hi)]))
                for lo, hi in zip(arrays["token_offsets"][:-1], arrays["token_offsets"][1:])
            ]
            steps = _top10_step_scores(risk, arrays["segment_starts"], arrays["segment_ends"])
            for row in range(len(prep.row_ids)):
                lo, hi = map(int, arrays["segment_offsets"][row:row + 2])
                locator[row, column] = int(np.argmax(steps[lo:hi]))
        step_counts = np.diff(arrays["segment_offsets"])
        if (
            not np.isfinite(detector).all()
            or np.any(locator < 0)
            or np.any(locator >= step_counts[:, None])
        ):
            raise ProtocolError(f"{cell_id}: invalid ProcessBench score freeze")
        frozen = {
            "row_ids": np.asarray(prep.row_ids, dtype="<U80"),
            "method_ids": np.asarray(methods, dtype="<U64"),
            "detector_scores": detector, "locators": locator,
        }
        semantics = "PB detector=max token risk; locator=argmax fixed-top10-mean step risk"
    else:
        step_risk = np.empty((len(arrays["segment_starts"]), len(methods)), dtype=np.float64)
        for column, method in enumerate(methods):
            risk = curves[method]
            step_risk[:, column] = [
                float(np.max(risk[int(lo):int(hi)]))
                for lo, hi in zip(arrays["segment_starts"], arrays["segment_ends"])
            ]
        if not np.isfinite(step_risk).all():
            raise ProtocolError("PRMBench score freeze contains non-finite values")
        frozen = {
            "row_ids": np.asarray(prep.row_ids, dtype="<U80"),
            "method_ids": np.asarray(methods, dtype="<U64"),
            "segment_offsets": np.asarray(arrays["segment_offsets"], dtype="<i8"),
            "step_risk": step_risk,
        }
        semantics = "PRM step risk=max token risk inside official span"
    return frozen, {"semantics": semantics, "n_rows": len(prep.row_ids)}


def _panel_gate_status(compact_by_cell: Mapping[str, Mapping[str, Any]]) -> tuple[bool, bool]:
    pb_pass = all(
        compact_by_cell[cell]["status"] == "FIT_COMPLETE"
        and compact_by_cell[cell]["structural_fit_pass"]
        for cell in PB_CELLS
    )
    prm_pass = (
        compact_by_cell[PRM_CELL]["status"] == "FIT_COMPLETE"
        and compact_by_cell[PRM_CELL]["structural_fit_pass"]
    )
    return pb_pass, prm_pass


def score() -> None:
    registry = _verify_registration()
    if STRUCTURAL_LEDGER.exists() or SCORE_ROOT.exists() or SCORE_MANIFEST.exists():
        raise ProtocolError("score namespace already exists")
    sanitized = json.loads(SANITIZED_MANIFEST.read_text())
    by_cell = {row["cell_id"]: row for row in sanitized["cells"]}
    compact_by_cell: dict[str, Mapping[str, Any]] = {}
    weights_by_cell: dict[str, Mapping[str, np.ndarray] | None] = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_fit_cell_worker, cell_id, position, registry, by_cell): cell_id
            for position, cell_id in enumerate(CELLS)
        }
        for future in as_completed(futures):
            cell_id, compact, weights = future.result()
            compact_by_cell[cell_id] = compact
            weights_by_cell[cell_id] = weights
            print(f"fit {cell_id}: {compact['status']}", flush=True)
    records = [compact_by_cell[cell_id] for cell_id in CELLS]
    pb_pass, prm_pass = _panel_gate_status(compact_by_cell)
    ledger = {
        "schema": "joint-lsml-existing-structural-ledger-v1", "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE", "processbench_panel_status": "PASS" if pb_pass else "STRUCTURAL_NO_SCORE",
        "prmbench_panel_status": "PASS" if prm_pass else "STRUCTURAL_NO_SCORE",
        "cells": records, "labels_accessed": False, "score_arrays_persisted": False,
    }
    SCORE_ROOT.mkdir(parents=True, exist_ok=False)
    score_records = []
    for cell_id in CELLS:
        panel_allowed = prm_pass if cell_id == PRM_CELL else pb_pass
        if not panel_allowed:
            continue
        arrays = _load_sanitized(cell_id, by_cell)
        prep = _preparation(arrays, registry)
        assert weights_by_cell[cell_id] is not None
        frozen, meta = _score_cell(cell_id, arrays, prep, weights_by_cell[cell_id])
        path = SCORE_ROOT / f"{cell_id}.npz"
        artifact_hash = atomic_write_npz(path, frozen)
        score_records.append({
            "cell_id": cell_id, "panel": "PRMBench" if cell_id == PRM_CELL else "ProcessBench",
            "artifact_path": path.name, "artifact_sha256": artifact_hash,
            "members": sorted(frozen), **meta,
        })
        print(f"froze {cell_id}", flush=True)
    ledger["score_arrays_persisted"] = bool(score_records)
    ledger["payload_sha256"] = payload_sha256(_jsonable(ledger))
    atomic_write_json(STRUCTURAL_LEDGER, _jsonable(ledger))
    manifest = {
        "schema": "joint-lsml-existing-score-freeze-v1", "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE", "development_scope": "RETROSPECTIVE_OPENED_DEVELOPMENT",
        "processbench_panel_status": "SCORES_FROZEN" if pb_pass else "STRUCTURAL_NO_SCORE",
        "prmbench_panel_status": "SCORES_FROZEN" if prm_pass else "STRUCTURAL_NO_SCORE",
        "methods": list(METHODS), "cells": score_records,
        "structural_ledger_sha256": sha256_file(STRUCTURAL_LEDGER),
        "registry_sha256": sha256_file(REGISTRY), "labels_accessed": False,
    }
    manifest["payload_sha256"] = payload_sha256(manifest)
    manifest_hash = atomic_write_json(SCORE_MANIFEST, manifest)
    atomic_write_json(RUN_COMPLETE, {
        "status": "PASS_SCORE_FREEZE_PENDING_INDEPENDENT_AUDIT",
        "score_manifest_sha256": manifest_hash, "labels_accessed": False,
    })


def check() -> None:
    registry = _verify_registration()
    sanitized = json.loads(SANITIZED_MANIFEST.read_text())
    for row in sanitized["cells"]:
        if sha256_file(SANITIZED_ROOT / row["artifact_path"]) != row["artifact_sha256"]:
            raise ProtocolError("sanitized artifact hash mismatch")
    if SCORE_MANIFEST.exists():
        manifest = json.loads(SCORE_MANIFEST.read_text())
        body = {key: value for key, value in manifest.items() if key != "payload_sha256"}
        if payload_sha256(body) != manifest.get("payload_sha256") or manifest.get("status") != "COMPLETE":
            raise ProtocolError("score manifest payload/status mismatch")
        if manifest.get("registry_sha256") != sha256_file(REGISTRY):
            raise ProtocolError("score manifest registry mismatch")
        for row in manifest["cells"]:
            path = SCORE_ROOT / row["artifact_path"]
            if sha256_file(path) != row["artifact_sha256"]:
                raise ProtocolError("score artifact hash mismatch")
            arrays = load_npz_no_pickle(path)
            if sorted(arrays) != sorted(row["members"]):
                raise ProtocolError("score artifact member roster mismatch")
            if tuple(arrays["method_ids"].astype(str)) != tuple(registry["methods"]):
                raise ProtocolError("score method roster mismatch")
            for key, value in arrays.items():
                if key not in {"row_ids", "method_ids"} and np.issubdtype(value.dtype, np.number):
                    if not np.isfinite(value).all():
                        raise ProtocolError("score artifact contains non-finite values")
        if manifest["structural_ledger_sha256"] != sha256_file(STRUCTURAL_LEDGER):
            raise ProtocolError("structural ledger hash mismatch")
        run_complete = json.loads(RUN_COMPLETE.read_text())
        if run_complete.get("status") != "PASS_SCORE_FREEZE_PENDING_INDEPENDENT_AUDIT":
            raise ProtocolError("run completion status mismatch")
        if run_complete.get("score_manifest_sha256") != sha256_file(SCORE_MANIFEST):
            raise ProtocolError("run completion score hash mismatch")
    print("PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("sanitize", "register", "score", "check"))
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args()
    if args.command == "sanitize":
        sanitize(args.release.resolve(), args.source_root.resolve())
    elif args.command == "register":
        register()
    elif args.command == "score":
        score()
    else:
        check()


if __name__ == "__main__":
    main()
