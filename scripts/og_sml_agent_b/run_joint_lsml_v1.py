#!/usr/bin/env python3
"""Freeze and execute Agent-B Joint L-SML v1 without outcome access."""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict, is_dataclass
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.dependency_fusion import regularized_covariance_weights  # noqa: E402
from spectral_utils.feature_contract import confidence_sign_vector  # noqa: E402
from spectral_utils.fixed_application_pipelines import (  # noqa: E402
    SHARED_GLOBAL_FEATURES,
    SHARED_TOKEN_VIEWS,
)
from spectral_utils.fusion_utils import lsml_continuous, sml_fuse_signed  # noqa: E402
from spectral_utils.joint_lsml import (  # noqa: E402
    consensus_orientation_and_roster,
    covariance_matrix,
    discover_loao_consensus_groups,
    dispatch_alias,
    fit_joint_lsml,
    global_degree_roster,
    hard_lsml_misfit,
    pairwise_score_spearman,
    raw_orientation_cell,
    weight_maps,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    load_npz_no_pickle,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import payload_sha256  # noqa: E402
from spectral_utils.token_local_fusion import TOKEN_FAMILY_NAMES, prepare_token_fusion  # noqa: E402

from render_joint_lsml_v1 import render  # noqa: E402


EXPERIMENT_ID = "JOINT_LSML_V1_R2"
BASE_COMMIT = "250e092e1a0f5b2e460e2fd0221bcbded28069dc"
SEED = 2026090401
CONDITIONS = (1e2, 1e3, 1e4)
K_RANGE = (3, 4, 6, 8)
DEGREE_TAU = 0.1
MINIMUM_DEGREE_CELLS = 8
WEAK_LOADING_THRESHOLD = 0.01
MINIMUM_SIGN_VOTES = 6
RESULT_ROOT = REPO / "results/joint_lsml_v1_r2"
REGISTRY_PATH = RESULT_ROOT / "EXECUTION_REGISTRY.json"
REGISTRATION_COMPLETE_PATH = RESULT_ROOT / "REGISTRATION_COMPLETE.json"
PROTOCOL = REPO / "docs/experiments/JOINT_LSML_V1.md"
PRIOR_AUDIT = REPO / "docs/experiments/PRIOR_ORDER_AUDIT_JOINT_LSML_V1.md"
CORE = REPO / "spectral_utils/joint_lsml.py"
TEST_SOURCE = REPO / "tests/test_joint_lsml.py"
RUNNER = Path(__file__).resolve()
RENDERER = REPO / "scripts/og_sml_agent_b/render_joint_lsml_v1.py"
C_V2_ROOT = REPO.parent / "structured_fusion_c_v2/results/label_free_structured_fusion_c_v2_raw"
SANITIZED_ROOT = C_V2_ROOT / "sanitized_raw_inputs"
SANITIZED_MANIFEST = SANITIZED_ROOT / "MANIFEST.json"
C_V2_LEDGER = C_V2_ROOT / "REAL_STRUCTURAL_LEDGER.json"
R1_FAILURE = REPO / "results/joint_lsml_v1_r1/RUN_FAILURE.json"
PB_CELLS = tuple(
    f"processbench_{family}_{model}"
    for model in ("qwen3_4b", "qwen3_8b")
    for family in ("gsm8k", "math", "olympiadbench", "omnimath")
)
CELLS = PB_CELLS + ("prmbench_response_qwen3_8b",)
LANES = ("v2_active28", "h2_24")


class JointProtocolError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise JointProtocolError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, _jsonable(value))


def _write_new_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    if path.exists():
        raise JointProtocolError(f"immutable artifact already exists: {path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name) for name in fields})
    atomic_write_bytes(path, stream.getvalue().encode("utf-8"))


def _source_hashes() -> dict[str, str]:
    return {
        "base_commit": BASE_COMMIT,
        "protocol": sha256_file(PROTOCOL),
        "prior_order_audit": sha256_file(PRIOR_AUDIT),
        "core": sha256_file(CORE),
        "runner": sha256_file(RUNNER),
        "renderer": sha256_file(RENDERER),
        "tests": sha256_file(TEST_SOURCE),
        "fusion_utils": sha256_file(REPO / "spectral_utils/fusion_utils.py"),
        "dependency_fusion": sha256_file(REPO / "spectral_utils/dependency_fusion.py"),
        "feature_contract": sha256_file(REPO / "spectral_utils/feature_contract.py"),
        "token_local_fusion": sha256_file(REPO / "spectral_utils/token_local_fusion.py"),
        "reconstruction_io": sha256_file(REPO / "spectral_utils/reconstruction_benchmark/io.py"),
        "localization_contract": sha256_file(REPO / "spectral_utils/reconstruction_benchmark/localization_contract.py"),
        "sanitized_manifest": sha256_file(SANITIZED_MANIFEST),
        "c_v2_structural_ledger": sha256_file(C_V2_LEDGER),
        "r1_failed_run_record": sha256_file(R1_FAILURE),
    }


def _firewall_audit() -> dict[str, Any]:
    sources = (CORE, RUNNER, RENDERER, TEST_SOURCE)
    imported: list[str] = []
    called: list[str] = []
    forbidden = {
        "load_prepared_localization_cell", "load_localization_targets",
        "load_processbench_labels", "load_prmbench_labels", "roc_auc_score",
        "average_precision_score", "processbench_f1",
    }
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called.append(node.func.attr)
    violations = sorted((set(imported) | set(called)) & forbidden)
    return {
        "status": "PASS" if not violations else "FAIL",
        "audited_files": [str(path) for path in sources],
        "forbidden_symbols": violations,
        "sanitized_members_allowed": ["raw", "token_offsets", "row_ids"],
        "labels_accessed": False,
        "response_scores_materialized": False,
    }


def _verify_payload(payload: Mapping[str, Any], *, name: str) -> None:
    body = dict(payload)
    recorded = body.pop("payload_sha256", None)
    if recorded != payload_sha256(body):
        raise JointProtocolError(f"{name} payload hash mismatch")


def register() -> dict[str, Any]:
    if RESULT_ROOT.exists():
        raise JointProtocolError(f"result namespace already exists: {RESULT_ROOT}")
    firewall = _firewall_audit()
    if firewall["status"] != "PASS":
        raise JointProtocolError(f"label firewall failed: {firewall}")
    sanitized = json.loads(SANITIZED_MANIFEST.read_text(encoding="utf-8"))
    _verify_payload(sanitized, name="sanitized manifest")
    records = {str(row["cell_id"]): row for row in sanitized["cells"]}
    if tuple(records) != CELLS:
        raise JointProtocolError("sanitized cell order/roster drift")
    input_files = {}
    for cell_id in CELLS:
        path = SANITIZED_ROOT / str(records[cell_id]["artifact_path"])
        if sha256_file(path) != records[cell_id]["artifact_sha256"]:
            raise JointProtocolError(f"{cell_id}: sanitized artifact hash mismatch")
        input_files[cell_id] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "allowed_members": ["raw", "token_offsets", "row_ids"],
        }
    registry = {
        "schema": "joint-lsml-v1-execution-registry-v1",
        "status": "FROZEN_BEFORE_DONOR_STRUCTURAL_COMPUTATION",
        "experiment_id": EXPERIMENT_ID,
        "source_hashes": _source_hashes(),
        "input_files": input_files,
        "cell_order": list(CELLS),
        "lane_order": list(LANES),
        "task1_conditions": list(CONDITIONS),
        "task1_pass_threshold": 0.99,
        "orientation": {
            "entropy_raw_sign": -1,
            "weak_loading_threshold": WEAK_LOADING_THRESHOLD,
            "minimum_sign_votes": MINIMUM_SIGN_VOTES,
            "weighted_degree_tau": DEGREE_TAU,
            "minimum_degree_cells": MINIMUM_DEGREE_CELLS,
        },
        "grouping": {
            "k_range": list(K_RANGE),
            "selection": "median LOAO-to-consensus ARI; mean ARI; smaller K",
            "minimum_group_size": 3,
        },
        "joint_fit": {
            "starts": 5, "max_sweeps": 5000,
            "relative_tolerance": 1e-10, "consecutive_stable_sweeps": 5,
            "monotonicity_tolerance": 1e-12,
        },
        "score_arrays_persisted": False,
        "benchmark_outcomes_loaded": False,
        "firewall": firewall,
    }
    registry["payload_sha256"] = payload_sha256(registry)
    _write_new_json(REGISTRY_PATH, registry)
    completion = {
        "schema": "joint-lsml-v1-registration-complete-v1",
        "status": "COMPLETE",
        "execution_registry_sha256": sha256_file(REGISTRY_PATH),
        "source_hashes": registry["source_hashes"],
    }
    completion["payload_sha256"] = payload_sha256(completion)
    _write_new_json(REGISTRATION_COMPLETE_PATH, completion)
    return completion


def _verify_registration() -> dict[str, Any]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    _verify_payload(registry, name="execution registry")
    completion = json.loads(REGISTRATION_COMPLETE_PATH.read_text(encoding="utf-8"))
    _verify_payload(completion, name="registration completion")
    if completion["execution_registry_sha256"] != sha256_file(REGISTRY_PATH):
        raise JointProtocolError("execution registry file hash drift")
    if registry["source_hashes"] != _source_hashes():
        raise JointProtocolError("registered source/input hash drift")
    for cell_id, record in registry["input_files"].items():
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise JointProtocolError(f"{cell_id}: registered input changed")
    if registry["firewall"]["status"] != "PASS" or _firewall_audit()["status"] != "PASS":
        raise JointProtocolError("label firewall is not closed")
    return registry


def _load_inputs() -> tuple[dict[str, dict[str, np.ndarray]], dict[tuple[str, str], Mapping[str, Any]]]:
    manifest = json.loads(SANITIZED_MANIFEST.read_text(encoding="utf-8"))
    records = {str(row["cell_id"]): row for row in manifest["cells"]}
    inputs = {}
    for cell_id in CELLS:
        path = SANITIZED_ROOT / str(records[cell_id]["artifact_path"])
        arrays = load_npz_no_pickle(path, members=("raw", "token_offsets", "row_ids"))
        if set(arrays) != {"raw", "token_offsets", "row_ids"}:
            raise JointProtocolError(f"{cell_id}: unexpected sanitized member")
        inputs[cell_id] = arrays
    ledger = json.loads(C_V2_LEDGER.read_text(encoding="utf-8"))
    if ledger.get("labels_seen") or ledger.get("targets_loaded") or ledger.get("outcome_metrics_computed"):
        raise JointProtocolError("C-v2 ledger is not target-free")
    lanes = {(str(row["cell_id"]), str(row["lane"])): row for row in ledger["cells"]}
    if tuple(lanes) != tuple((cell_id, lane) for cell_id in CELLS for lane in LANES):
        raise JointProtocolError("C-v2 lane roster/order drift")
    return inputs, lanes


def _prepare(arrays: Mapping[str, np.ndarray]) -> Any:
    preparation = prepare_token_fusion(
        np.asarray(arrays["raw"], dtype=np.float64),
        np.asarray(arrays["token_offsets"], dtype=np.int64),
        np.asarray(arrays["row_ids"]).astype(str).tolist(),
    )
    if not np.all(preparation.keep) or tuple(preparation.kept_stream_names) != tuple(SHARED_TOKEN_VIEWS):
        raise JointProtocolError("raw 29-stream preparation unexpectedly removed a coordinate")
    return preparation


def _edis_onset(trace: Sequence[float]) -> np.ndarray:
    values = np.asarray(trace, dtype=np.float64)
    burst = np.zeros_like(values)
    if len(values) > 1:
        burst[1:] = np.maximum(np.diff(values) - 1.36, 0.0)
    running_min = np.minimum.accumulate(values)
    rebound = np.maximum(values - running_min - 1.33, 0.0)
    onset = np.maximum(np.diff(np.concatenate(([0.0], rebound))), 0.0)
    return np.maximum(burst, onset)


def _response_map(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    output = np.empty_like(values)
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        output[int(lo):int(hi)] = _edis_onset(values[int(lo):int(hi)])
    return output


def _standardize_selected(values: np.ndarray, fit_indices: np.ndarray) -> np.ndarray:
    fit = np.asarray(values[fit_indices], dtype=np.float64)
    medians = np.nanmedian(fit, axis=0)
    clean = np.where(np.isfinite(fit), fit, medians[None, :])
    means = clean.mean(axis=0)
    scales = clean.std(axis=0)
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or np.any(scales <= 1e-8):
        raise JointProtocolError("selected H2 lane contains a degenerate stream")
    return (clean - means[None, :]) / scales[None, :]


def _h2_values(preparation: Any, token_offsets: np.ndarray, signs: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    oriented = np.asarray(preparation.values, dtype=np.float64) * signs[None, :]
    selected: list[np.ndarray] = []
    names: list[str] = []
    families: list[str] = []
    for index, (name, family) in enumerate(zip(SHARED_TOKEN_VIEWS, TOKEN_FAMILY_NAMES)):
        eligible = family in {"entropy_level", "entropy_dynamics", "partition_energy", "topk_distribution"}
        if family == "partition_energy" and name == "energy_series":
            eligible = False
        if eligible:
            selected.append(oriented[:, index])
            names.append(str(name))
            families.append(str(family))
    entropy_index = SHARED_TOKEN_VIEWS.index("entropy_series")
    entropy_risk = -oriented[:, entropy_index]
    c7_confidence = -_response_map(entropy_risk, np.asarray(token_offsets, dtype=np.int64))
    insertion = max(index for index, family in enumerate(families) if family == "entropy_dynamics") + 1
    selected.insert(insertion, c7_confidence)
    names.insert(insertion, "C7_EDIS_ONSET")
    matrix = _standardize_selected(np.column_stack(selected), preparation.fit_indices)
    if matrix.shape[1] != 24:
        raise JointProtocolError(f"H2 roster drifted to {matrix.shape[1]}")
    return matrix, tuple(names)


def _frozen_lane_values(preparation: Any, offsets: np.ndarray, lane: str, signs: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    if lane == "v2_active28":
        return np.asarray(preparation.standardized_fit)[:, 1:] * signs[None, 1:], tuple(SHARED_TOKEN_VIEWS[1:])
    if lane == "h2_24":
        return _h2_values(preparation, offsets, signs)
    raise ValueError(lane)


def _task1(
    prepared: Mapping[str, Any], inputs: Mapping[str, Mapping[str, np.ndarray]],
    frozen_lanes: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    current_signs = confidence_sign_vector(SHARED_GLOBAL_FEATURES).astype(np.float64)
    rows = []
    for cell_id in CELLS:
        for lane in LANES:
            source = frozen_lanes[(cell_id, lane)]
            values, names = _frozen_lane_values(
                prepared[cell_id], np.asarray(inputs[cell_id]["token_offsets"], dtype=np.int64), lane, current_signs
            )
            if list(names) != list(source["feature_names"]):
                raise JointProtocolError(f"{cell_id}/{lane}: frozen feature roster mismatch")
            if prepared[cell_id].diagnostics["fit_index_sha256"] != source["fit_index_sha256"]:
                raise JointProtocolError(f"{cell_id}/{lane}: frozen fit index mismatch")
            observed_covariance = covariance_matrix(values)
            source_covariance = np.asarray(source["covariance"], dtype=np.float64)
            covariance_max_error = float(np.max(np.abs(observed_covariance - source_covariance)))
            if covariance_max_error > 1e-12:
                raise JointProtocolError(f"{cell_id}/{lane}: C-v2 covariance reconstruction drift {covariance_max_error}")
            model = np.asarray(source["structured_fit"]["model_covariance"], dtype=np.float64)
            loading = np.asarray(source["structured_fit"]["global_loading"], dtype=np.float64)
            score_map = {}
            diagnostics = {}
            for condition in CONDITIONS:
                key = f"condition_{int(condition):d}"
                weight, audit = regularized_covariance_weights(model, loading, target_condition=condition)
                score_map[key] = values @ weight
                diagnostics[key] = {
                    **audit,
                    "weight_l2_norm": float(np.linalg.norm(weight)),
                    "finite_weight": bool(np.isfinite(weight).all()),
                    "finite_donor_score": bool(np.isfinite(score_map[key]).all()),
                }
            correlations = pairwise_score_spearman(score_map)
            minimum = float(min(correlations.values()))
            rows.append({
                "cell_id": cell_id,
                "benchmark_panel": "PRMBench" if cell_id.startswith("prmbench") else "ProcessBench",
                "lane": lane,
                "n_donor_tokens": int(len(values)),
                "n_streams": int(values.shape[1]),
                "covariance_reconstruction_max_abs_error": covariance_max_error,
                "pairwise_score_spearman": correlations,
                "minimum_pairwise_score_spearman": minimum,
                "pass_minimum_ge_0_99": bool(minimum >= 0.99),
                "solver_diagnostics": diagnostics,
                "score_arrays_persisted": False,
            })
    return {
        "schema": "joint-lsml-v1-task1-ridge-score-stability-v1",
        "status": "COMPLETE",
        "conditions": list(CONDITIONS),
        "pass_threshold": 0.99,
        "pass_count": int(sum(row["pass_minimum_ge_0_99"] for row in rows)),
        "lane_count": len(rows),
        "lanes": rows,
        "score_arrays_persisted": False,
        "labels_accessed": False,
    }


def _ordered_features() -> list[dict[str, Any]]:
    return [
        {"global_index": index, "raw_feature_name": str(SHARED_GLOBAL_FEATURES[index]), "stream_name": str(SHARED_TOKEN_VIEWS[index])}
        for index in range(29)
    ]


def _candidate_summary(grouping: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for candidate in grouping["candidates"]:
        pairwise = np.asarray(candidate["pairwise_ari"], dtype=np.float64)
        output.append({
            "K": int(candidate["K"]),
            "group_sizes": list(candidate["group_sizes"]),
            "admissible": bool(candidate["admissible"]),
            "rejection_reason": candidate["rejection_reason"],
            "minimum_held_answer_group_size": int(min(min(row) for row in candidate["held_answer_group_sizes"])),
            "median_ari": float(candidate["median_ari"]),
            "mean_ari": float(candidate["mean_ari"]),
            "minimum_ari": float(candidate["minimum_ari"]),
            "exact_fraction": float(candidate["exact_fraction"]),
            "ari_to_consensus": np.asarray(candidate["ari_to_consensus"], dtype=float).tolist(),
            "pairwise_ari_summary": {
                "count": int(len(pairwise)), "minimum": float(pairwise.min()),
                "median": float(np.median(pairwise)), "mean": float(pairwise.mean()),
                "maximum": float(pairwise.max()),
            },
        })
    return output


def _fit_record(fit: Any) -> dict[str, Any]:
    return {
        "converged": fit.converged,
        "converged_starts": fit.converged_starts,
        "selected_start": fit.selected_start,
        "objective": fit.objective,
        "relative_offdiag_misfit": fit.relative_offdiag_misfit,
        "global_loading": fit.global_loading,
        "group_loading": fit.group_loading,
        "fitted_offdiag": fit.fitted_offdiag,
        "model_covariance": fit.model_covariance,
        "multistart_audit": fit.multistart_audit,
        "jacobian_audit": fit.jacobian_audit,
        "diagonal_audit": fit.diagonal_audit,
        "starts": [
            {
                "start": row.start, "converged": row.converged,
                "failed_monotonicity": row.failed_monotonicity, "sweeps": row.sweeps,
                "objective_trace": row.objective_trace,
                "model_change_trace": row.model_change_trace,
            }
            for row in fit.starts
        ],
    }


def _joint_lane(
    cell_id: str, lane: str, values: np.ndarray, names: Sequence[str], owners: np.ndarray,
    *, seed: int, registry_sign_by_stream: Mapping[str, int], current_sign_by_stream: Mapping[str, int],
) -> dict[str, Any]:
    if "entropy_series" not in names:
        raise JointProtocolError(f"{cell_id}/{lane}: protected entropy anchor missing")
    anchor = list(names).index("entropy_series")
    grouping = discover_loao_consensus_groups(values, owners, k_range=K_RANGE, seed=seed)
    if grouping["status"] != "SELECTED":
        return {
            "cell_id": cell_id,
            "benchmark_panel": "PRMBench" if cell_id.startswith("prmbench") else "ProcessBench",
            "lane": lane,
            "status": "BLOCKED_NO_ADMISSIBLE_PARTITION",
            "n_donor_tokens": int(len(values)),
            "n_streams": int(values.shape[1]),
            "feature_names": list(map(str, names)),
            "grouping": {
                "status": grouping["status"],
                "selection_rule": grouping["selection_rule"],
                "candidates": _candidate_summary(grouping),
            },
            "structural_fit_pass": False,
            "labels_accessed": False,
            "score_arrays_persisted": False,
        }
    labels = np.asarray(grouping["labels"], dtype=np.int64)
    covariance = covariance_matrix(values)
    fit = fit_joint_lsml(covariance, labels, anchor_index=anchor, seed=seed + 10000)
    hard = hard_lsml_misfit(covariance, labels)
    maps = weight_maps(values, covariance, labels, fit, anchor_index=anchor, target_condition=1e3)
    score_correlations = maps["pairwise_score_spearman"]
    minimum_score_correlation = float(min(score_correlations.values()))
    weights = {name: np.asarray(weight, dtype=float) for name, weight in maps["weights"].items()}
    finite_weights = bool(all(np.isfinite(weight).all() for weight in weights.values()))
    oriented_loading_sign = np.where(fit.global_loading < 0.0, -1, 1)
    implied_raw_signs = []
    sign_comparison = []
    for name, coefficient_sign in zip(names, oriented_loading_sign):
        if name == "C7_EDIS_ONSET":
            implied = 1
            existing = 1
        else:
            implied = int(registry_sign_by_stream[name]) * int(coefficient_sign)
            existing = int(current_sign_by_stream[name])
        implied_raw_signs.append(implied)
        sign_comparison.append({
            "stream_name": str(name),
            "oriented_global_loading_sign": int(coefficient_sign),
            "implied_raw_confidence_sign": int(implied),
            "current_dict_sign": int(existing),
            "differs_from_current_dict": bool(implied != existing),
        })
    structural_pass = bool(
        fit.converged and fit.multistart_audit["status"] == "PASS"
        and fit.jacobian_audit["full_global_rank"] and finite_weights
    )
    return {
        "cell_id": cell_id,
        "benchmark_panel": "PRMBench" if cell_id.startswith("prmbench") else "ProcessBench",
        "lane": lane,
        "status": "FIT_COMPLETE",
        "n_donor_tokens": int(len(values)),
        "n_streams": int(values.shape[1]),
        "feature_names": list(map(str, names)),
        "covariance": covariance,
        "grouping": {
            "status": grouping["status"], "K": grouping["K"],
            "labels": grouping["labels"], "group_sizes": grouping["group_sizes"],
            "coassignment": grouping["coassignment"],
            "mean_loao_coassignment": grouping["mean_loao_coassignment"],
            "median_ari": grouping["median_ari"], "mean_ari": grouping["mean_ari"],
            "minimum_ari": grouping["minimum_ari"], "exact_fraction": grouping["exact_fraction"],
            "pairwise_ari_summary": grouping["pairwise_ari_summary"],
            "selection_rule": grouping["selection_rule"],
            "candidates": _candidate_summary(grouping),
        },
        "joint_fit": _fit_record(fit),
        "hard_lsml_relative_offdiag_misfit": hard["relative_offdiag_misfit"],
        "joint_minus_hard_misfit": float(fit.relative_offdiag_misfit - hard["relative_offdiag_misfit"]),
        "joint_lower_misfit": bool(fit.relative_offdiag_misfit < hard["relative_offdiag_misfit"]),
        "weight_maps": {
            "weights": weights,
            "weight_l2_norms": {name: float(np.linalg.norm(weight)) for name, weight in weights.items()},
            "pairwise_score_spearman": score_correlations,
            "diagnostics": maps["diagnostics"],
            "finite_weights": finite_weights,
            "scores_persisted": False,
        },
        "minimum_weight_map_score_spearman": minimum_score_correlation,
        "sign_comparison_to_current_dict": sign_comparison,
        "negative_oriented_loading_streams": [
            str(name) for name, sign in zip(names, oriented_loading_sign) if int(sign) < 0
        ],
        "implied_raw_signs": implied_raw_signs,
        "structural_fit_pass": structural_pass,
        "labels_accessed": False,
        "score_arrays_persisted": False,
    }


def execute() -> dict[str, Any]:
    registry = _verify_registration()
    forbidden_outputs = (
        "TASK1_RIDGE_SCORE_STABILITY.json", "ORIENTATION_CELL_LEDGER.json",
        "V2_ABSOLUTE_ORIENTATION_REGISTRY.json", "PER_CELL_REMOVAL_LEDGER.json",
        "V2_GLOBAL_PRUNED_ROSTER.json", "JOINT_STRUCTURAL_LEDGER.json", "SUMMARY.json",
    )
    if any((RESULT_ROOT / name).exists() for name in forbidden_outputs):
        raise JointProtocolError("refusing to overwrite an existing Joint L-SML result")
    inputs, frozen_lanes = _load_inputs()
    prepared = {cell_id: _prepare(inputs[cell_id]) for cell_id in CELLS}

    task1 = _task1(prepared, inputs, frozen_lanes)
    current_signs = confidence_sign_vector(SHARED_GLOBAL_FEATURES).astype(np.int64)
    current_active = current_signs[1:]
    active_names = tuple(SHARED_TOKEN_VIEWS[1:])
    entropy_active_index = active_names.index("entropy_series")
    orientation_estimates = []
    orientation_cells = []
    for cell_id in CELLS:
        raw_active = np.asarray(prepared[cell_id].standardized_fit, dtype=np.float64)[:, 1:]
        estimate = raw_orientation_cell(raw_active, entropy_index=entropy_active_index, tau=DEGREE_TAU)
        orientation_estimates.append(estimate)
        orientation_cells.append({
            "cell_id": cell_id,
            "benchmark_panel": "PRMBench" if cell_id.startswith("prmbench") else "ProcessBench",
            "signs": estimate["signs"], "loading": estimate["loading"],
            "absolute_loading": estimate["absolute_loading"],
            "weighted_degree": estimate["weighted_degree"],
            "median_weighted_degree": estimate["median_weighted_degree"],
            "degree_threshold": estimate["degree_threshold"],
            "degree_keep": estimate["degree_keep"],
            "leading_eigenvalue": estimate["leading_eigenvalue"],
        })
    consensus = consensus_orientation_and_roster(
        orientation_estimates, current_active,
        weak_loading_threshold=WEAK_LOADING_THRESHOLD,
        minimum_sign_votes=MINIMUM_SIGN_VOTES,
        minimum_degree_cells=MINIMUM_DEGREE_CELLS,
    )
    if not bool(consensus["active"][entropy_active_index]):
        raise JointProtocolError("protected entropy anchor failed orientation/pruning rules")
    final_signs = np.concatenate((np.asarray([-1], dtype=np.int64), consensus["schema_signs"]))
    active_global_indices = tuple(
        index + 1 for index, keep in enumerate(consensus["active"]) if bool(keep)
    )
    active_stream_names = tuple(SHARED_TOKEN_VIEWS[index] for index in active_global_indices)
    if len(active_global_indices) < 8:
        raise JointProtocolError("global V2 roster is too small")

    orientation_ledger = {
        "schema": "joint-lsml-v1-orientation-cell-ledger-v1",
        "status": "COMPLETE",
        "active_stream_names": list(active_names),
        "cells": orientation_cells,
        "consensus": consensus,
        "current_dict_active_signs": current_active,
        "final_29_schema_signs": final_signs,
        "flips_against_current_dict": [
            name for name, old, new in zip(active_names, current_active, final_signs[1:]) if int(old) != int(new)
        ],
        "weak_streams": [name for name, flag in zip(active_names, consensus["weak"]) if bool(flag)],
        "unstable_streams": [name for name, flag in zip(active_names, consensus["unstable"]) if bool(flag)],
        "degree_rejected_streams": [name for name, flag in zip(active_names, consensus["degree_rejected"]) if bool(flag)],
        "retained_streams": list(active_stream_names),
        "labels_accessed": False,
    }

    # Build confidence-oriented full H2 matrices, then freeze one global H2 roster.
    h2_full = {}
    h2_names: tuple[str, ...] | None = None
    h2_anchor_indices = []
    for cell_id in CELLS:
        matrix, names = _h2_values(
            prepared[cell_id], np.asarray(inputs[cell_id]["token_offsets"], dtype=np.int64), final_signs
        )
        if h2_names is None:
            h2_names = names
        elif names != h2_names:
            raise JointProtocolError("H2 roster differs across cells")
        h2_full[cell_id] = matrix
        h2_anchor_indices.append(names.index("entropy_series"))
    assert h2_names is not None
    h2_degree = global_degree_roster(
        [h2_full[cell_id] for cell_id in CELLS], anchor_indices=h2_anchor_indices,
        tau=DEGREE_TAU, minimum_cells=MINIMUM_DEGREE_CELLS,
    )
    active_v2_set = set(active_stream_names)
    h2_active = np.asarray([
        bool(degree_keep) and (name == "C7_EDIS_ONSET" or name in active_v2_set)
        for name, degree_keep in zip(h2_names, h2_degree["active"])
    ], dtype=bool)
    if not h2_active[h2_names.index("entropy_series")]:
        raise JointProtocolError("protected H2 entropy anchor failed pruning rules")
    h2_retained_names = tuple(name for name, keep in zip(h2_names, h2_active) if keep)

    removal_cells = []
    for cell_index, cell_id in enumerate(CELLS):
        active_rows = []
        for feature_index, name in enumerate(active_names):
            reasons = []
            if bool(consensus["weak"][feature_index]):
                reasons.append("WEAK_MEAN_ABS_LOADING_LT_0_01")
            if bool(consensus["unstable"][feature_index]):
                reasons.append("SIGN_WINNING_VOTES_LT_6")
            if bool(consensus["degree_rejected"][feature_index]):
                reasons.append("DEGREE_KEEP_IN_LT_8_OF_9_CELLS")
            active_rows.append({
                "stream_name": name,
                "local_degree_keep": bool(orientation_estimates[cell_index]["degree_keep"][feature_index]),
                "degree_pass_count": int(consensus["degree_pass_count"][feature_index]),
                "global_retained": bool(consensus["active"][feature_index]),
                "removal_reasons": reasons,
            })
        h2_rows = []
        for feature_index, name in enumerate(h2_names):
            reasons = []
            if not bool(h2_degree["active"][feature_index]):
                reasons.append("H2_DEGREE_KEEP_IN_LT_8_OF_9_CELLS")
            if name != "C7_EDIS_ONSET" and name not in active_v2_set:
                reasons.append("REMOVED_BY_GLOBAL_RAW_V2_ROSTER")
            h2_rows.append({
                "stream_name": name,
                "local_degree_keep": bool(h2_degree["cell_estimates"][cell_index]["degree_keep"][feature_index]),
                "degree_pass_count": int(h2_degree["degree_pass_count"][feature_index]),
                "global_retained": bool(h2_active[feature_index]),
                "removal_reasons": reasons,
            })
        removal_cells.append({"cell_id": cell_id, "v2_active28_source": active_rows, "h2_24_source": h2_rows})
    removal_ledger = {
        "schema": "joint-lsml-v1-per-cell-removal-ledger-v1",
        "status": "COMPLETE", "tau": DEGREE_TAU,
        "global_keep_rule": "kept in at least 8 of 9 target-free donor cells and neither weak nor sign-unstable",
        "cells": removal_cells,
        "v2_retained_global_indices": list(active_global_indices),
        "v2_retained_stream_names": list(active_stream_names),
        "h2_retained_stream_names": list(h2_retained_names),
        "labels_accessed": False,
    }

    # Finish every numerical fit in memory before materializing any result payload.
    current_sign_by_stream = {
        name: int(sign) for name, sign in zip(SHARED_TOKEN_VIEWS, current_signs)
    }
    registry_sign_by_stream = {
        name: int(sign) for name, sign in zip(SHARED_TOKEN_VIEWS, final_signs)
    }
    structural_rows = []
    lane_data: dict[tuple[str, str], tuple[np.ndarray, tuple[str, ...]]] = {}
    lane_index = 0
    for cell_id in CELLS:
        owners = np.asarray(prepared[cell_id].fit_row_indices, dtype=np.int64)
        v2_full = np.asarray(prepared[cell_id].standardized_fit, dtype=np.float64)[:, 1:] * final_signs[None, 1:]
        v2_values = v2_full[:, np.asarray(consensus["active"], dtype=bool)]
        h2_values = h2_full[cell_id][:, h2_active]
        for lane, values, names in (
            ("v2_active28", v2_values, active_stream_names),
            ("h2_24", h2_values, h2_retained_names),
        ):
            lane_data[(cell_id, lane)] = (values, tuple(names))
            structural_rows.append(_joint_lane(
                cell_id, lane, values, names, owners, seed=SEED + 1000 * lane_index,
                registry_sign_by_stream=registry_sign_by_stream,
                current_sign_by_stream=current_sign_by_stream,
            ))
            lane_index += 1

    # Explicit dispatch controls on a real donor matrix.
    control_record = next(
        (row for row in structural_rows if row["status"] == "FIT_COMPLETE"), None
    )
    if control_record is None:
        raise JointProtocolError("all 18 lanes lack an admissible consensus partition")
    control_values, _ = lane_data[(control_record["cell_id"], control_record["lane"])]
    direct_flat, direct_flat_weight = sml_fuse_signed(*control_values.T)
    alias_flat, alias_flat_weight, _ = dispatch_alias(
        control_values, np.zeros(control_values.shape[1], dtype=int), mode="flat_sml"
    )
    control_groups = np.asarray(control_record["grouping"]["labels"], dtype=int)
    direct_lsml, _ = lsml_continuous(*control_values.T, groups=control_groups, compute_score_matrix=False)
    alias_lsml, _, _ = dispatch_alias(control_values, control_groups, mode="two_stage_alias")
    alias_audit = {
        "schema": "joint-lsml-v1-alias-audit-v1", "status": "PASS",
        "k1_flat_score_bit_exact": bool(np.array_equal(direct_flat, alias_flat)),
        "k1_flat_weight_bit_exact": bool(np.array_equal(direct_flat_weight, alias_flat_weight)),
        "two_stage_score_bit_exact": bool(np.array_equal(direct_lsml, alias_lsml)),
        "score_arrays_persisted": False, "labels_accessed": False,
    }
    if not all(alias_audit[key] for key in (
        "k1_flat_score_bit_exact", "k1_flat_weight_bit_exact", "two_stage_score_bit_exact"
    )):
        raise JointProtocolError("explicit alias audit failed")

    structural = {
        "schema": "joint-lsml-v1-structural-ledger-v1", "status": "COMPLETE",
        "lane_count": len(structural_rows), "lanes": structural_rows,
        "fit_lane_count": int(sum(row["status"] == "FIT_COMPLETE" for row in structural_rows)),
        "blocked_lane_count": int(sum(row["status"] != "FIT_COMPLETE" for row in structural_rows)),
        "joint_lower_misfit_count": int(sum(row.get("joint_lower_misfit", False) for row in structural_rows)),
        "structural_fit_pass_count": int(sum(row["structural_fit_pass"] for row in structural_rows)),
        "score_arrays_persisted": False, "labels_accessed": False,
    }

    # Materialize the signed source ledgers first, then bind Agent-A registries to their file hashes.
    _write_new_json(RESULT_ROOT / "TASK1_RIDGE_SCORE_STABILITY.json", task1)
    _write_new_json(RESULT_ROOT / "ORIENTATION_CELL_LEDGER.json", orientation_ledger)
    _write_new_json(RESULT_ROOT / "PER_CELL_REMOVAL_LEDGER.json", removal_ledger)
    ordered = _ordered_features()
    orientation_registry = {
        "schema": "within-answer-confidence-orientation-v1",
        "status": "FROZEN_BEFORE_V2_REGISTRATION",
        "scope": "GLOBAL",
        "orientation_domain": "RAW_TOKEN_FEATURE_MATRIX_TO_ABSOLUTE_CONFIDENCE",
        "input_domain": "raw_token_feature_matrix(SHARED_GLOBAL_FEATURES)",
        "ordered_features": ordered,
        "ordered_features_sha256": payload_sha256(ordered),
        "output_semantics": "HIGHER_IS_MORE_CONFIDENT",
        "relative_to": None,
        "confidence_signs": final_signs.tolist(),
        "source_contract_sha256": sha256_file(RESULT_ROOT / "ORIENTATION_CELL_LEDGER.json"),
        "labels_accessed": False,
        "response_scores_materialized": False,
    }
    orientation_registry["payload_sha256"] = payload_sha256(orientation_registry)
    _write_new_json(RESULT_ROOT / "V2_ABSOLUTE_ORIENTATION_REGISTRY.json", orientation_registry)

    anchor = {
        "global_index": SHARED_TOKEN_VIEWS.index("entropy_series"),
        "raw_feature_name": SHARED_GLOBAL_FEATURES[SHARED_TOKEN_VIEWS.index("entropy_series")],
        "stream_name": "entropy_series", "required_retained": True,
    }
    excluded_indices = tuple(index for index in range(29) if index not in active_global_indices)
    selection_population = [
        {"cell_id": cell_id, "sanitized_input_sha256": registry["input_files"][cell_id]["sha256"]}
        for cell_id in CELLS
    ]
    pruned_registry = {
        "schema": "within-answer-global-pruned-roster-v1",
        "status": "FROZEN_BEFORE_V2_REGISTRATION",
        "scope": "GLOBAL_SINGLE_ROSTER",
        "input_domain": "raw_token_feature_matrix(SHARED_GLOBAL_FEATURES)",
        "ordered_input_features": ordered,
        "ordered_input_features_sha256": payload_sha256(ordered),
        "retained_global_indices": list(active_global_indices),
        "retained_stream_names": list(active_stream_names),
        "excluded_global_indices": list(excluded_indices),
        "excluded_stream_names": [SHARED_TOKEN_VIEWS[index] for index in excluded_indices],
        "selection_rule": "KEPT_IN_AT_LEAST_8_OF_9_TARGET_FREE_CELLS",
        "selection_population_sha256": payload_sha256(selection_population),
        "per_cell_removal_ledger_sha256": sha256_file(RESULT_ROOT / "PER_CELL_REMOVAL_LEDGER.json"),
        "protected_semantic_anchor_sha256": payload_sha256(anchor),
        "protected_semantic_anchor": anchor,
        "source_artifact_sha256": sha256_file(RESULT_ROOT / "V2_ABSOLUTE_ORIENTATION_REGISTRY.json"),
        "trace_length_role": "NUISANCE_ONLY__EXCLUDED_FROM_ACTIVE_ROSTER",
        "not_an_orientation_registry": True,
        "labels_accessed": False,
        "response_scores_materialized": False,
    }
    pruned_registry["payload_sha256"] = payload_sha256(pruned_registry)
    _write_new_json(RESULT_ROOT / "V2_GLOBAL_PRUNED_ROSTER.json", pruned_registry)
    _write_new_json(RESULT_ROOT / "ALIAS_AUDIT.json", alias_audit)
    _write_new_json(RESULT_ROOT / "JOINT_STRUCTURAL_LEDGER.json", structural)

    task1_csv = [{
        "cell_id": row["cell_id"], "benchmark_panel": row["benchmark_panel"], "lane": row["lane"],
        "n_streams": row["n_streams"],
        "minimum_pairwise_score_spearman": row["minimum_pairwise_score_spearman"],
        "pass_minimum_ge_0_99": row["pass_minimum_ge_0_99"],
    } for row in task1["lanes"]]
    _write_new_csv(
        RESULT_ROOT / "TASK1_RIDGE_SCORE_STABILITY.csv", task1_csv,
        ("cell_id", "benchmark_panel", "lane", "n_streams", "minimum_pairwise_score_spearman", "pass_minimum_ge_0_99"),
    )
    structural_csv = [{
        "cell_id": row["cell_id"], "benchmark_panel": row["benchmark_panel"], "lane": row["lane"],
        "status": row["status"], "n_streams": row["n_streams"],
        "K": row["grouping"].get("K"),
        "minimum_group_size": min(row["grouping"]["group_sizes"]) if "group_sizes" in row["grouping"] else None,
        "median_loao_ari": row["grouping"].get("median_ari"),
        "minimum_loao_ari": row["grouping"].get("minimum_ari"),
        "joint_misfit": row.get("joint_fit", {}).get("relative_offdiag_misfit"),
        "hard_lsml_misfit": row.get("hard_lsml_relative_offdiag_misfit"),
        "joint_lower_misfit": row.get("joint_lower_misfit"),
        "minimum_weight_map_score_spearman": row.get("minimum_weight_map_score_spearman"),
        "structural_fit_pass": row["structural_fit_pass"],
    } for row in structural_rows]
    _write_new_csv(
        RESULT_ROOT / "JOINT_STRUCTURAL_LANES.csv", structural_csv,
        ("cell_id", "benchmark_panel", "lane", "status", "n_streams", "K", "minimum_group_size", "median_loao_ari", "minimum_loao_ari", "joint_misfit", "hard_lsml_misfit", "joint_lower_misfit", "minimum_weight_map_score_spearman", "structural_fit_pass"),
    )
    no_scoring = {
        "schema": "joint-lsml-v1-no-scoring-declaration-v1",
        "status": "CLOSED_NO_SCORING",
        "candidate_slots_opened": 0,
        "benchmark_outcomes_loaded": False,
        "outcome_metrics_computed": False,
        "score_arrays_persisted": False,
        "overlap_or_lag_experiment_opened": False,
    }
    no_scoring["payload_sha256"] = payload_sha256(no_scoring)
    _write_new_json(RESULT_ROOT / "NO_SCORING_DECLARATION.json", no_scoring)

    summary = {
        "schema": "joint-lsml-v1-summary-v1", "status": "STRUCTURAL_RESULTS_ONLY_NO_SCORING",
        "task1_pass_count": task1["pass_count"], "task1_lane_count": task1["lane_count"],
        "v2_retained_count": len(active_stream_names), "v2_removed_count": 28 - len(active_stream_names),
        "v2_retained_stream_names": list(active_stream_names),
        "v2_removed_stream_names": [name for name in active_names if name not in active_stream_names],
        "h2_retained_count": len(h2_retained_names), "h2_retained_stream_names": list(h2_retained_names),
        "orientation_flips_against_current_dict": orientation_ledger["flips_against_current_dict"],
        "weak_streams": orientation_ledger["weak_streams"],
        "unstable_streams": orientation_ledger["unstable_streams"],
        "degree_rejected_streams": orientation_ledger["degree_rejected_streams"],
        "joint_fit_lane_count": structural["fit_lane_count"],
        "joint_blocked_lane_count": structural["blocked_lane_count"],
        "joint_blocked_lanes": [
            f"{row['cell_id']}::{row['lane']}" for row in structural_rows
            if row["status"] != "FIT_COMPLETE"
        ],
        "joint_lower_misfit_count": structural["joint_lower_misfit_count"],
        "structural_fit_pass_count": structural["structural_fit_pass_count"],
        "lane_count": len(structural_rows),
        "minimum_weight_map_score_spearman_range": [
            float(min(row["minimum_weight_map_score_spearman"] for row in structural_rows if row["status"] == "FIT_COMPLETE")),
            float(max(row["minimum_weight_map_score_spearman"] for row in structural_rows if row["status"] == "FIT_COMPLETE")),
        ],
        "agent_a_orientation_registry": str(RESULT_ROOT / "V2_ABSOLUTE_ORIENTATION_REGISTRY.json"),
        "agent_a_pruned_roster_registry": str(RESULT_ROOT / "V2_GLOBAL_PRUNED_ROSTER.json"),
        "labels_accessed": False, "score_arrays_persisted": False,
    }
    summary["payload_sha256"] = payload_sha256(summary)
    _write_new_json(RESULT_ROOT / "SUMMARY.json", summary)

    plots = render(RESULT_ROOT)
    report_lines = [
        "# Joint L-SML v1 — Agent B structural report", "",
        "This run is label-free structural development only. It does not compute benchmark efficacy, open a scoring arm, or support promotion.", "",
        "## Task 1 — ridge target diagnostic", "",
        f"{task1['pass_count']}/{task1['lane_count']} frozen C-v2 lanes have minimum pairwise donor-score Spearman >= 0.99 across target conditions 1e2, 1e3 and 1e4.", "",
        "- Observation: the plot reports rank stability of the actual donor fused scores Xw, replacing coefficient-space cosine.",
        "- Inference: a passing lane is insensitive in ranking to this frozen regularization range on its donor rows.",
        "- Limitation: this is neither outcome performance nor out-of-population stability.", "",
        "## Orientation and global pruning", "",
        f"The global V2 roster retains {len(active_stream_names)}/28 active raw streams. Removed streams: {', '.join(summary['v2_removed_stream_names']) or 'none'}.", "",
        f"Weak: {', '.join(summary['weak_streams']) or 'none'}. Sign-unstable: {', '.join(summary['unstable_streams']) or 'none'}. Degree-rejected: {', '.join(summary['degree_rejected_streams']) or 'none'}.", "",
        "- Observation: sign is estimated independently in nine cells and opacity in the heatmap tracks |v|.",
        "- Inference: streams without stable/meaningful orientation are excluded from geometry and fusion rather than repaired ad hoc.",
        "- Limitation: signs are gauge-fixed by entropy_series and remain donor-population estimates.", "",
        "## Joint disjoint-group estimator", "",
        f"Joint L-SML produced an admissible fit in {structural['fit_lane_count']}/{len(structural_rows)} lanes; {structural['blocked_lane_count']} lanes had no K whose consensus and every LOAO fold kept all groups at size >=3. Among all 18 lanes, joint misfit is lower in {structural['joint_lower_misfit_count']} and {structural['structural_fit_pass_count']} pass convergence, multistart, profiled-Jacobian and finite-weight checks.", "",
        "- Observation: every fitted lane uses a K>=3 LOAO-consensus partition chosen by ARI stability, never by residual fit; blocked lanes have no fitted model.",
        "- Inference: the joint estimator directly tests whether a shared factor plus disjoint group factors explains donor covariance more faithfully than the historical two-stage factorization.",
        "- Limitation: lower covariance misfit does not imply better localization; no labels or benchmark scores were accessed.", "",
        "## Weight-map comparison", "",
        f"The per-lane minimum pairwise donor-score Spearman among hierarchical-joint, model-inverse, sample-inverse and existing continuous L-SML spans {summary['minimum_weight_map_score_spearman_range'][0]:.6f} to {summary['minimum_weight_map_score_spearman_range'][1]:.6f}.", "",
        "- Observation: the comparison is made in score/ranking space on identical donor rows.",
        "- Inference: disagreement localizes the practical consequence of changing the loading estimator or covariance map.",
        "- Limitation: these are retrospective donor diagnostics and no fused score arrays were saved.", "",
        "## Agent A handoff", "",
        f"- Absolute orientation registry: `{summary['agent_a_orientation_registry']}`",
        f"- Global pruned roster: `{summary['agent_a_pruned_roster_registry']}`",
        "- trace_length is nuisance-only and excluded from the active roster.",
        "- This handoff does not authorize overlap, LAG, T1/T2, or scoring.", "",
        "## Plots", "",
        *[f"- `{name}`" for name in plots["plot_files"]], "",
    ]
    report_path = RESULT_ROOT / "REPORT.md"
    if report_path.exists():
        raise JointProtocolError("refusing to overwrite REPORT.md")
    atomic_write_bytes(report_path, "\n".join(report_lines).encode("utf-8"))
    claim_audit = {
        "schema": "joint-lsml-v1-claim-audit-v1", "status": "PASS",
        "allowed_claim": "label-free donor structural and rank-stability results only",
        "forbidden_claims": ["benchmark improvement", "localization efficacy", "promotion", "new leader", "transfer"],
        "benchmark_outcomes_loaded": False, "score_arrays_persisted": False,
    }
    claim_audit["payload_sha256"] = payload_sha256(claim_audit)
    _write_new_json(RESULT_ROOT / "CLAIM_AUDIT.json", claim_audit)

    artifact_paths = sorted(
        path for path in RESULT_ROOT.iterdir()
        if path.is_file() and path.name not in {"COMPLETE.json"}
    )
    complete = {
        "schema": "joint-lsml-v1-complete-v1", "status": "COMPLETE",
        "execution_registry_sha256": sha256_file(REGISTRY_PATH),
        "registration_complete_sha256": sha256_file(REGISTRATION_COMPLETE_PATH),
        "artifacts": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifact_paths
        ],
        "source_hashes": registry["source_hashes"],
        "labels_accessed": False, "score_arrays_persisted": False,
    }
    complete["payload_sha256"] = payload_sha256(complete)
    _write_new_json(RESULT_ROOT / "COMPLETE.json", complete)
    return summary


def check() -> dict[str, Any]:
    registry = _verify_registration()
    complete_path = RESULT_ROOT / "COMPLETE.json"
    if not complete_path.exists():
        raise JointProtocolError("COMPLETE.json is missing")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    _verify_payload(complete, name="complete manifest")
    if complete["execution_registry_sha256"] != sha256_file(REGISTRY_PATH):
        raise JointProtocolError("complete manifest registry hash drift")
    for record in complete["artifacts"]:
        path = RESULT_ROOT / str(record["name"])
        if not path.is_file() or sha256_file(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise JointProtocolError(f"artifact drift: {path}")
    if complete["source_hashes"] != registry["source_hashes"]:
        raise JointProtocolError("complete source hashes differ from preregistration")
    summary = json.loads((RESULT_ROOT / "SUMMARY.json").read_text(encoding="utf-8"))
    _verify_payload(summary, name="summary")
    if summary.get("labels_accessed") is not False or summary.get("score_arrays_persisted") is not False:
        raise JointProtocolError("summary violates no-label/no-score-array contract")
    return {
        "status": "PASS", "artifact_count": len(complete["artifacts"]),
        "execution_registry_sha256": sha256_file(REGISTRY_PATH),
        "complete_sha256": sha256_file(complete_path),
        "task1_pass_count": summary["task1_pass_count"],
        "structural_fit_pass_count": summary["structural_fit_pass_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("register", "run", "check"))
    args = parser.parse_args()
    result = register() if args.command == "register" else execute() if args.command == "run" else check()
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
