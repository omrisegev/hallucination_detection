#!/usr/bin/env python3
"""Strict evaluation-only boundary for the frozen DEEM-B3 pair router.

The label-sidecar module is deliberately imported only after a complete
cryptographic and mechanical preflight of the run, its exact Cartesian fit
manifest, all target-free bundles, all selected-seed B3 baselines, and every
fit artifact in the frozen run.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.deem_b3_pair_router import (  # noqa: E402
    PairRouterConfig,
    _validate_config,
)
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ResidualGraphDeemError,
    atomic_write_json,
    canonical_sha256,
    family_index_map,
    sha256_file,
)
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    load_registry,
    load_target_free_bundle,
    registry_cell,
)


DEFAULT_CONFIG = ROOT / "configs/deem_b3_pair_router_v1.json"
DEFAULT_REGISTRY = ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"
LABEL_MODULE = "spectral_utils.residual_graph_deem_labels"
FIT_SCHEMA = "deem_b3_pair_router_fit_artifact_v1"
RUN_SCHEMA = "deem_b3_pair_router_run_definition_v1"
MANIFEST_SCHEMA = "deem_b3_pair_router_fit_manifest_v1"
TOLERANCE = 5e-4

REQUIRED_ARRAYS = {
    "score",
    "posterior",
    "logit",
    "contributions",
    "base_family_contributions",
    "routed_family_contributions",
    "gates",
    "family_probabilities",
    "pair_transfers",
    "pair_open_probabilities",
    "pair_context_residuals",
    "family_order",
    "pair_order_left",
    "pair_order_right",
    "feature_names",
    "row_id",
    "group_id",
    "raw_trace_length",
    "standardization_mean",
    "standardization_scale",
    "standardization_constant_mask",
    "baseline_score",
    "baseline_orientation",
    "baseline_aligned_bias",
    "graph_residual_coordinates",
    "graph_residual_family_order",
    "graph_global_family_order",
    "graph_present_family_mask",
    "graph_grouped_folds",
    "graph_row_tie_keys",
    "graph_loo_predictability",
    "graph_data",
    "graph_indices",
    "graph_indptr",
    "graph_shape",
    "laplacian_data",
    "laplacian_indices",
    "laplacian_indptr",
    "laplacian_shape",
}


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResidualGraphDeemError(f"JSON object required: {path}")
    return value


def _verify_content_hash(value: Mapping[str, Any], *, context: str) -> str:
    payload = dict(value)
    expected = payload.pop("content_sha256", None)
    actual = canonical_sha256(payload)
    if not isinstance(expected, str) or actual != expected:
        raise ResidualGraphDeemError(f"content hash mismatch: {context}")
    return expected


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _ndarray_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_sha256(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _expit(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


def _safe_logit(score: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(score, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return np.log(values) - np.log1p(-values)


def _load_config(path: Path) -> tuple[dict, dict[str, dict]]:
    value = _read_json(path)
    if value.get("schema") != "deem_b3_pair_router_v1_config":
        raise ResidualGraphDeemError("pair-router config schema mismatch")
    boundary = value.get("scientific_boundary", {})
    if (
        boundary.get("fit_is_label_free") is not True
        or boundary.get("baseline_is_frozen") is not True
        or boundary.get(
            "nongraph_router_arms_are_strictly_self_free_for_each_routed_pair"
        )
        is not True
        or boundary.get(
            "graph_arms_are_functional_inference_loo2_not_end_to_end_strict_loo2"
        )
        is not True
        or boundary.get("graph_laplacian_uses_all_present_family_loo_residual_coordinates")
        is not True
        or boundary.get("natural_24cell_targets_previously_opened") is not True
    ):
        raise ResidualGraphDeemError("pair-router scientific boundary is incomplete")
    rows = value.get("variants")
    if not isinstance(rows, list) or not rows:
        raise ResidualGraphDeemError("pair-router variant roster is empty")
    lookup: dict[str, dict] = {}
    for row in rows:
        identifier = str(row.get("id", ""))
        if not identifier or identifier in lookup:
            raise ResidualGraphDeemError("pair-router variant IDs are empty or duplicated")
        try:
            parsed = PairRouterConfig(**row.get("config", {}))
            _validate_config(parsed)
        except (TypeError, ValueError) as exc:
            raise ResidualGraphDeemError(f"invalid pair-router config: {identifier}") from exc
        lookup[identifier] = row
    screen = value.get("screen_cells")
    if not isinstance(screen, list) or len(screen) != 8 or len(set(screen)) != 8:
        raise ResidualGraphDeemError("frozen pair-router screen must contain eight unique cells")
    rule = value.get("frozen_screen_rule", {})
    if str(rule.get("mechanistic_primary", "")) not in lookup:
        raise ResidualGraphDeemError("frozen screen primary is absent from the roster")
    thresholds = rule.get("primary_survives_only_if", {})
    required_thresholds = {
        "equal_family_auroc_delta_min",
        "descriptive_family_bootstrap_lower_min",
        "exact_family_signflip_one_sided_p_max",
        "wins_plus_ties_min_of_8",
        "worst_cell_delta_min",
    }
    if set(thresholds) != required_thresholds:
        raise ResidualGraphDeemError("frozen screen threshold contract mismatch")
    if not all(np.isfinite(float(thresholds[key])) for key in required_thresholds):
        raise ResidualGraphDeemError("non-finite frozen screen threshold")
    return value, lookup


def _parse_selection(
    raw: str,
    available: Sequence[str],
    *,
    special: Sequence[str] | None = None,
    special_name: str | None = None,
) -> list[str]:
    allowed = [str(value) for value in available]
    if raw == "all":
        selected = list(allowed)
    elif special_name is not None and raw == special_name:
        selected = [str(value) for value in (special or ())]
    else:
        selected = [value.strip() for value in raw.split(",") if value.strip()]
    if not selected or len(selected) != len(set(selected)) or not set(selected).issubset(set(allowed)):
        raise ValueError(f"invalid selection: {raw}")
    return selected


def _parse_seeds(raw: str, available: Sequence[int], development: Sequence[int]) -> list[int]:
    if raw == "all":
        selected = [int(value) for value in available]
    elif raw == "development":
        selected = [int(value) for value in development]
    else:
        selected = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if (
        not selected
        or len(selected) != len(set(selected))
        or not set(selected).issubset(set(int(value) for value in available))
    ):
        raise ValueError("invalid seed selection")
    return selected


def _validate_bundle(bundle, registered: Mapping[str, Any], path: Path) -> dict:
    expected = {
        "cell_id": str(registered["cell_id"]),
        "dataset_family": str(registered["dataset_family"]),
        "task_type": str(registered["task_type"]),
        "inventory_sha256": str(registered["inventory_sha256"]),
        "source_sha256": str(registered["source"]["source_sha256"]),
        "manifest_sha256": str(registered["source"]["manifest_sha256"]),
        "admission_sha256": str(registered["source"]["admission_contract_sha256"]),
    }
    for field, wanted in expected.items():
        if str(getattr(bundle, field)) != wanted:
            raise ResidualGraphDeemError(
                f"bundle/registry {field} mismatch: {registered['cell_id']}"
            )
    if (
        len(bundle.row_ids) != int(registered["n_rows"])
        or len(bundle.feature_names) != int(registered["n_features"])
        or tuple(bundle.feature_names) != tuple(registered["feature_names"])
    ):
        raise ResidualGraphDeemError(f"bundle/registry shape mismatch: {bundle.cell_id}")
    manifest_path = path.with_suffix(".manifest.json")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema") != "residual_graph_deem_target_free_bundle_v1"
        or manifest.get("cell_id") != bundle.cell_id
        or int(manifest.get("n_rows", -1)) != len(bundle.row_ids)
        or int(manifest.get("n_features", -1)) != len(bundle.feature_names)
        or manifest.get("bundle_sha256") != bundle.bundle_sha256
        or manifest.get("ordered_row_id_sha256")
        != canonical_sha256(list(bundle.row_ids))
        or manifest.get("inventory_sha256") != bundle.inventory_sha256
        or manifest.get("source_sha256") != bundle.source_sha256
        or manifest.get("manifest_sha256") != bundle.manifest_sha256
        or manifest.get("admission_sha256") != bundle.admission_sha256
        or manifest.get("labels_accessed") is not False
        or manifest.get("allow_pickle") is not False
    ):
        raise ResidualGraphDeemError(f"invalid target-free bundle manifest: {bundle.cell_id}")
    return {
        "bundle_sha256": bundle.bundle_sha256,
        "bundle_manifest_file_sha256": sha256_file(manifest_path),
        "ordered_row_id_sha256": canonical_sha256(list(bundle.row_ids)),
        "n_rows": len(bundle.row_ids),
    }


def _expected_full_config(row: Mapping[str, Any]) -> dict:
    return asdict(PairRouterConfig(**row.get("config", {})))


def _fit_paths(run_dir: Path, variant: str, cell: str, seed: int) -> tuple[Path, Path]:
    stem = f"{variant}__seed{int(seed)}"
    root = (run_dir / "fits" / variant / cell).resolve()
    run_root = run_dir.resolve()
    try:
        root.relative_to(run_root)
    except ValueError as exc:
        raise ResidualGraphDeemError("fit path escaped run directory") from exc
    return root / f"{stem}.npz", root / f"{stem}.json"


def _csr_from_arrays(arrays: Mapping[str, np.ndarray], prefix: str):
    shape_raw = np.asarray(arrays[f"{prefix}_shape"], dtype=np.int64)
    if shape_raw.shape != (2,):
        raise ResidualGraphDeemError(f"invalid {prefix} shape vector")
    shape = tuple(int(value) for value in shape_raw)
    data = np.asarray(arrays[f"{prefix}_data"], dtype=np.float64)
    indices = np.asarray(arrays[f"{prefix}_indices"], dtype=np.int64)
    indptr = np.asarray(arrays[f"{prefix}_indptr"], dtype=np.int64)
    if not np.isfinite(data).all() or data.ndim != 1 or indices.shape != data.shape:
        raise ResidualGraphDeemError(f"invalid {prefix} CSR payload")
    if shape == (0, 0) and data.size == indices.size == indptr.size == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float64)
    try:
        matrix = sparse.csr_matrix((data, indices, indptr), shape=shape)
        matrix.check_format(full_check=True)
    except Exception as exc:
        raise ResidualGraphDeemError(f"invalid {prefix} CSR structure") from exc
    return matrix


def _validate_fit_arrays(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    bundle,
    variant_row: Mapping[str, Any],
    baseline_score: np.ndarray,
) -> dict[str, float]:
    missing = sorted(REQUIRED_ARRAYS - set(arrays))
    unexpected = sorted(
        name for name in arrays if name not in REQUIRED_ARRAYS and not name.startswith("state__")
    )
    if missing or unexpected:
        raise ResidualGraphDeemError(
            f"fit array schema mismatch; missing={missing}, unexpected={unexpected}"
        )
    n = len(bundle.row_ids)
    feature_names = tuple(str(value) for value in arrays["feature_names"].tolist())
    family_order = tuple(str(value) for value in arrays["family_order"].tolist())
    groups = family_index_map(feature_names)
    if feature_names != tuple(bundle.feature_names) or family_order != tuple(groups):
        raise ResidualGraphDeemError("fit feature/family order mismatch")
    family_count = len(family_order)
    pair_order = tuple(
        zip(
            (str(value) for value in arrays["pair_order_left"].tolist()),
            (str(value) for value in arrays["pair_order_right"].tolist()),
        )
    )
    expected_pairs = tuple(itertools.combinations(family_order, 2))
    if pair_order != expected_pairs or family_count < 4:
        raise ResidualGraphDeemError("fit pair order mismatch")
    pair_count = len(expected_pairs)

    required_shapes = {
        "score": (n,),
        "posterior": (n, 2),
        "logit": (n,),
        "contributions": (n, len(feature_names)),
        "base_family_contributions": (n, family_count),
        "routed_family_contributions": (n, family_count),
        "gates": (n, family_count),
        "family_probabilities": (n, family_count),
        "pair_transfers": (n, pair_count),
        "pair_open_probabilities": (pair_count,),
        "pair_context_residuals": (n, pair_count, family_count),
        "standardization_mean": (len(feature_names),),
        "standardization_scale": (len(feature_names),),
        "standardization_constant_mask": (len(feature_names),),
        "baseline_score": (n,),
    }
    numeric: dict[str, np.ndarray] = {}
    for name, shape in required_shapes.items():
        value = np.asarray(arrays[name], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ResidualGraphDeemError(f"invalid fit array {name}")
        numeric[name] = value
    if np.any(numeric["standardization_scale"] <= 0.0):
        raise ResidualGraphDeemError("nonpositive standardization scale")
    score = numeric["score"]
    logit = numeric["logit"]
    gates = numeric["gates"]
    probabilities = numeric["family_probabilities"]
    if np.max(np.abs(score - _expit(logit))) > 1e-12:
        raise ResidualGraphDeemError("score/logit reconstruction failed")
    expected_posterior = np.column_stack([1.0 - score, score])
    if np.max(np.abs(numeric["posterior"] - expected_posterior)) > 1e-12:
        raise ResidualGraphDeemError("score/posterior reconstruction failed")
    if tuple(str(value) for value in arrays["row_id"].tolist()) != tuple(bundle.row_ids):
        raise ResidualGraphDeemError("fit row IDs do not match the bound bundle order")
    if tuple(str(value) for value in arrays["group_id"].tolist()) != tuple(bundle.group_ids):
        raise ResidualGraphDeemError("fit group IDs do not match the bound bundle order")
    if not np.array_equal(
        np.asarray(arrays["raw_trace_length"], dtype=np.int64),
        np.asarray(bundle.raw_trace_length, dtype=np.int64),
    ):
        raise ResidualGraphDeemError("fit trace lengths do not match the bound bundle")
    constant_mask = np.asarray(arrays["standardization_constant_mask"], dtype=np.int8)
    if not set(np.unique(constant_mask)).issubset({0, 1}):
        raise ResidualGraphDeemError("invalid standardization constant mask")
    stored_orientation = np.asarray(arrays["baseline_orientation"])
    stored_bias = np.asarray(arrays["baseline_aligned_bias"], dtype=np.float64)
    if (
        stored_orientation.shape != ()
        or int(stored_orientation.item()) != int(metadata.get("orientation", 0))
        or stored_bias.shape != ()
        or not np.isfinite(stored_bias.item())
        or float(stored_bias.item()) != float(metadata.get("aligned_bias", np.nan))
    ):
        raise ResidualGraphDeemError("stored baseline orientation/bias mismatch")
    if not np.array_equal(numeric["baseline_score"], baseline_score):
        raise ResidualGraphDeemError("fit does not bind the exact selected-seed B3 score")
    bias = float(metadata.get("aligned_bias", np.nan))
    if not np.isfinite(bias) or np.max(
        np.abs(bias + numeric["contributions"].sum(axis=1) - logit)
    ) > 1e-8:
        raise ResidualGraphDeemError("atomic contribution reconstruction failed")
    if np.max(
        np.abs(
            numeric["routed_family_contributions"]
            - numeric["base_family_contributions"] * gates
        )
    ) > 1e-10:
        raise ResidualGraphDeemError("routed family contribution reconstruction failed")
    for column, family in enumerate(family_order):
        indices = groups[family]
        grouped = numeric["contributions"][:, list(indices)].sum(axis=1)
        if np.max(np.abs(grouped - numeric["routed_family_contributions"][:, column])) > 1e-8:
            raise ResidualGraphDeemError("atomic/family contribution mismatch")

    full_config = _expected_full_config(variant_row)
    rho = float(full_config["rho"])
    gate_sum_error = float(np.max(np.abs(gates.sum(axis=1) - family_count)))
    gate_bound_error = float(
        max(
            np.max((1.0 - rho) - gates),
            np.max(gates - (1.0 + rho)),
            0.0,
        )
    )
    probability_error = float(np.max(np.abs(probabilities - gates / family_count)))
    if gate_sum_error > 1e-10 or gate_bound_error > 1e-10 or probability_error > 1e-12:
        raise ResidualGraphDeemError("pair-router gate invariant failed")
    if rho == 0.0 and (
        not np.array_equal(score, baseline_score)
        or not np.array_equal(gates, np.ones_like(gates))
    ):
        raise ResidualGraphDeemError("rho=0 artifact is not an exact B3 alias")
    if np.min(numeric["pair_open_probabilities"]) < 0.0 or np.max(
        numeric["pair_open_probabilities"]
    ) > 1.0:
        raise ResidualGraphDeemError("invalid pair-open probabilities")
    if np.max(np.abs(numeric["pair_transfers"])) > 1.0 + 1e-12:
        raise ResidualGraphDeemError("pair transfer escaped [-1,1]")
    context = numeric["pair_context_residuals"]
    for pair_index, pair in enumerate(expected_pairs):
        left = family_order.index(pair[0])
        right = family_order.index(pair[1])
        if np.any(context[:, pair_index, left] != 0.0) or np.any(
            context[:, pair_index, right] != 0.0
        ):
            raise ResidualGraphDeemError("pair context contains an endpoint family")

    graph_coordinates = np.asarray(arrays["graph_residual_coordinates"], dtype=np.float64)
    if not np.isfinite(graph_coordinates).all() or graph_coordinates.ndim != 2:
        raise ResidualGraphDeemError("invalid graph residual coordinates")
    graph_family_order = tuple(
        str(value) for value in arrays["graph_residual_family_order"].tolist()
    )
    graph_global_order = tuple(
        str(value) for value in arrays["graph_global_family_order"].tolist()
    )
    graph_present = np.asarray(arrays["graph_present_family_mask"], dtype=np.int8)
    graph_folds = np.asarray(arrays["graph_grouped_folds"], dtype=np.int64)
    graph_ties = np.asarray(arrays["graph_row_tie_keys"], dtype=np.float64)
    graph_predictability = np.asarray(arrays["graph_loo_predictability"], dtype=np.float64)
    if (
        graph_global_order != family_order
        or graph_present.shape != (family_count,)
        or not set(np.unique(graph_present)).issubset({0, 1})
        or not np.isfinite(graph_ties).all()
        or not np.isfinite(graph_predictability).all()
    ):
        raise ResidualGraphDeemError("invalid graph family/presence payload")
    graph = _csr_from_arrays(arrays, "graph")
    laplacian = _csr_from_arrays(arrays, "laplacian")
    graph_weight = float(full_config["graph_weight"])
    graph_metadata = metadata.get("graph", {})
    if graph_weight > 0.0:
        if graph.shape != (n, n) or laplacian.shape != (n, n):
            raise ResidualGraphDeemError("graph-weighted fit lacks an aligned graph")
        if (
            graph_metadata.get("used") is not True
            or graph_metadata.get("functional_inference_loo2") is not True
            or graph_metadata.get("end_to_end_strict_loo2") is not False
            or graph_metadata.get("uses_labels") is not False
            or graph_coordinates.shape != (n, len(graph_family_order))
            or graph_family_order
            != tuple(
                family
                for family, is_present in zip(graph_global_order, graph_present)
                if is_present
            )
            or len(graph_family_order) < 4
            or graph_folds.shape != (n,)
            or graph_ties.shape != (n,)
            or len(np.unique(graph_ties)) != n
            or graph_predictability.shape != (family_count,)
        ):
            raise ResidualGraphDeemError("graph-weighted fit lacks residual coordinates")
        for name, matrix in (("graph", graph), ("laplacian", laplacian)):
            asymmetry = matrix - matrix.T
            error = float(np.max(np.abs(asymmetry.data))) if asymmetry.nnz else 0.0
            if error > 1e-10:
                raise ResidualGraphDeemError(f"{name} is not symmetric")
        if abs(float(np.sum(laplacian.diagonal())) / n - 1.0) > 1e-12:
            raise ResidualGraphDeemError("normalized Laplacian trace contract failed")
        graph_payload_names = sorted(
            name
            for name in REQUIRED_ARRAYS
            if name.startswith("graph_") or name.startswith("laplacian_")
        )
        graph_payload_hash = canonical_sha256(
            {name: _ndarray_sha256(arrays[name]) for name in graph_payload_names}
        )
        if (
            graph_metadata.get("coordinate_sha256") != _ndarray_sha256(graph_coordinates)
            or graph_metadata.get("grouped_fold_sha256") != _ndarray_sha256(graph_folds)
            or graph_metadata.get("graph_payload_sha256") != graph_payload_hash
        ):
            raise ResidualGraphDeemError("graph payload hash closure failed")
    elif (
        graph_metadata.get("used") is not False
        or graph_metadata.get("end_to_end_strict_loo2") is not True
        or graph_metadata.get("uses_labels") is not False
        or graph.nnz
        or laplacian.nnz
        or graph.shape != (0, 0)
        or laplacian.shape != (0, 0)
        or graph_coordinates.shape != (n, 0)
        or graph_family_order
        or np.any(graph_present)
        or graph_folds.size
        or graph_ties.size
        or graph_predictability.size
    ):
        raise ResidualGraphDeemError("graph-free fit stores an active graph")

    baseline_logit = _safe_logit(baseline_score)
    diagnostics = {
        "gate_mean_abs_deviation_from_one": float(np.mean(np.abs(gates - 1.0))),
        "gate_min": float(np.min(gates)),
        "gate_max": float(np.max(gates)),
        "gate_sum_max_abs_error": gate_sum_error,
        "gate_bound_max_violation": gate_bound_error,
        "family_probability_max_abs_error": probability_error,
        "pair_transfer_mean_abs": float(np.mean(np.abs(numeric["pair_transfers"]))),
        "pair_transfer_max_abs": float(np.max(np.abs(numeric["pair_transfers"]))),
        "pair_open_mean": float(np.mean(numeric["pair_open_probabilities"])),
        "pair_open_min": float(np.min(numeric["pair_open_probabilities"])),
        "pair_open_max": float(np.max(numeric["pair_open_probabilities"])),
        "context_residual_sd": float(np.std(context)),
        "router_delta_logit_sd": float(np.std(logit - baseline_logit)),
        "score_pearson_vs_b3": float(np.corrcoef(score, baseline_score)[0, 1]),
    }
    for column, family in enumerate(family_order):
        diagnostics[f"gate_mean__{family}"] = float(np.mean(gates[:, column]))
        diagnostics[f"gate_sd__{family}"] = float(np.std(gates[:, column]))
    return diagnostics


def _load_baseline(
    baseline_dir: Path,
    cell: str,
    seed: int,
    bundle,
    baseline_contract: Mapping[str, Any],
) -> tuple[np.ndarray, dict]:
    array_path = baseline_dir / "fits" / cell / f"B3__seed{int(seed)}.npz"
    metadata_path = array_path.with_suffix(".json")
    metadata = _read_json(metadata_path)
    _verify_content_hash(metadata, context=f"B3 metadata {cell}/seed{seed}")
    array_sha = sha256_file(array_path)
    relative_array = array_path.relative_to(baseline_dir).as_posix()
    relative_metadata = metadata_path.relative_to(baseline_dir).as_posix()
    artifact_map = baseline_contract["artifact_map"]
    row_sha = canonical_sha256(list(bundle.row_ids))
    if (
        metadata.get("status") != "complete"
        or metadata.get("arm_id") != "B3"
        or metadata.get("cell_id") != cell
        or metadata.get("stem") != f"B3__seed{int(seed)}"
        or int(metadata.get("seed", -1)) != int(seed)
        or metadata.get("array_sha256") != array_sha
        or metadata.get("bundle_sha256") != bundle.bundle_sha256
        or metadata.get("inventory_sha256") != bundle.inventory_sha256
        or metadata.get("source_sha256") != bundle.source_sha256
        or metadata.get("ordered_row_id_sha256") != row_sha
        or metadata.get("health", {}).get("healthy") is not True
        or int(metadata.get("orientation", 0)) not in {-1, 1}
        or artifact_map.get(relative_array) != array_sha
        or artifact_map.get(relative_metadata) != sha256_file(metadata_path)
    ):
        raise ResidualGraphDeemError(f"invalid frozen B3 binding: {cell}/seed{seed}")
    with np.load(array_path, allow_pickle=False) as data:
        if "score" not in data.files:
            raise ResidualGraphDeemError("B3 artifact has no score")
        score = np.asarray(data["score"], dtype=np.float64)
    if score.shape != (len(bundle.row_ids),) or not np.isfinite(score).all():
        raise ResidualGraphDeemError(f"invalid frozen B3 score: {cell}/seed{seed}")
    return score, {
        "array_sha256": array_sha,
        "metadata_sha256": sha256_file(metadata_path),
        "orientation": int(metadata["orientation"]),
        "baseline_score_freeze_manifest_sha256": baseline_contract[
            "manifest_sha256"
        ],
        "baseline_score_freeze_content_sha256": baseline_contract[
            "content_sha256"
        ],
    }


def _validate_metadata(
    metadata: Mapping[str, Any],
    *,
    variant: str,
    experiment_id: str,
    cell: str,
    seed: int,
    bundle,
    variant_row: Mapping[str, Any],
    run_definition_sha256: str,
    array_path: Path,
    metadata_path: Path,
    manifest_row: Mapping[str, Any],
    baseline_audit: Mapping[str, Any],
    bundle_audit: Mapping[str, Any],
    run_dir: Path,
) -> None:
    _verify_content_hash(metadata, context=f"fit metadata {variant}/{cell}/seed{seed}")
    expected_config = _expected_full_config(variant_row)
    expected_array_path = array_path.relative_to(run_dir.resolve()).as_posix()
    expected_metadata_path = metadata_path.relative_to(run_dir.resolve()).as_posix()
    baseline = metadata.get("baseline", {})
    if (
        metadata.get("schema") != FIT_SCHEMA
        or metadata.get("status") != "complete"
        or metadata.get("experiment_id") != experiment_id
        or metadata.get("variant_id") != variant
        or metadata.get("variant_role") != str(variant_row.get("role", ""))
        or metadata.get("cell_id") != cell
        or metadata.get("dataset_family") != bundle.dataset_family
        or metadata.get("task_type") != bundle.task_type
        or int(metadata.get("seed", -1)) != int(seed)
        or int(metadata.get("n_rows", -1)) != len(bundle.row_ids)
        or int(metadata.get("n_features", -1)) != len(bundle.feature_names)
        or metadata.get("bundle_sha256") != bundle.bundle_sha256
        or metadata.get("bundle_manifest_file_sha256")
        != bundle_audit["bundle_manifest_file_sha256"]
        or metadata.get("inventory_sha256") != bundle.inventory_sha256
        or metadata.get("ordered_row_id_sha256") != canonical_sha256(list(bundle.row_ids))
        or metadata.get("source_sha256") != bundle.source_sha256
        or metadata.get("source_manifest_sha256") != bundle.manifest_sha256
        or metadata.get("admission_sha256") != bundle.admission_sha256
        or metadata.get("run_definition_sha256") != run_definition_sha256
        or metadata.get("array_path") != expected_array_path
        or metadata.get("array_sha256") != sha256_file(array_path)
        or metadata.get("config") != expected_config
        or metadata.get("config_sha256") != canonical_sha256(expected_config)
        or int(metadata.get("orientation", 0)) not in {-1, 1}
        or metadata.get("targets_accessed_during_fit") is not False
        or metadata.get("labels_module_imported") is not False
        or metadata.get("diagnostics", {}).get("uses_labels") is not False
        or metadata.get("health", {}).get("healthy") is not True
        or baseline.get("baseline_array_sha256") != baseline_audit["array_sha256"]
        or baseline.get("baseline_metadata_sha256") != baseline_audit["metadata_sha256"]
        or int(baseline.get("baseline_orientation", 0)) != baseline_audit["orientation"]
        or baseline.get("baseline_score_freeze_manifest_sha256")
        != baseline_audit["baseline_score_freeze_manifest_sha256"]
        or baseline.get("baseline_score_freeze_content_sha256")
        != baseline_audit["baseline_score_freeze_content_sha256"]
        or manifest_row.get("array_path") != expected_array_path
        or manifest_row.get("metadata_path") != expected_metadata_path
        or manifest_row.get("variant_id") != variant
        or manifest_row.get("cell_id") != cell
        or int(manifest_row.get("seed", -1)) != int(seed)
        or manifest_row.get("bundle_sha256") != bundle.bundle_sha256
        or manifest_row.get("ordered_row_id_sha256")
        != canonical_sha256(list(bundle.row_ids))
        or manifest_row.get("array_sha256") != sha256_file(array_path)
        or manifest_row.get("metadata_sha256") != sha256_file(metadata_path)
    ):
        raise ResidualGraphDeemError(f"fit metadata binding failed: {variant}/{cell}/seed{seed}")


def _load_run_contract(
    run_dir: Path,
    manifest_path: Path | None,
    baseline_dir: Path,
    config_path: Path,
    registry_path: Path,
    config: Mapping[str, Any],
    variant_lookup: Mapping[str, Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> tuple[dict, str, dict, list[dict], dict]:
    if manifest_path is None:
        manifest_candidates = sorted((run_dir / "fit_manifests").glob("*.json"))
    else:
        selected_manifest = (
            manifest_path if manifest_path.is_absolute() else run_dir / manifest_path
        )
        manifest_candidates = [selected_manifest]
    if len(manifest_candidates) != 1:
        raise ResidualGraphDeemError(
            "exactly one fit manifest must be selected with --fit-manifest"
        )
    manifest_path = manifest_candidates[0].resolve()
    try:
        manifest_path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ResidualGraphDeemError("fit manifest escaped run directory") from exc
    manifest = _read_json(manifest_path)
    _verify_content_hash(manifest, context="pair-router fit manifest")
    definition_sha = str(manifest.get("run_definition_sha256", ""))
    definition_path = run_dir / "run_definitions" / f"{definition_sha}.json"
    definition = _read_json(definition_path)
    verified_definition_sha = _verify_content_hash(
        definition, context="pair-router run definition"
    )
    if verified_definition_sha != definition_sha:
        raise ResidualGraphDeemError("run-definition filename/content binding failed")
    source_manifest = definition.get("source_manifest")
    environment = definition.get("environment")
    if (
        not isinstance(source_manifest, dict)
        or canonical_sha256(source_manifest) != definition.get("source_sha256")
        or not isinstance(environment, dict)
        or canonical_sha256(
            {key: value for key, value in environment.items() if key != "environment_sha256"}
        )
        != environment.get("environment_sha256")
    ):
        raise ResidualGraphDeemError("run source/environment hash closure failed")
    baseline_freeze_path = baseline_dir / "SCORE_FREEZE_MANIFEST.json"
    baseline_freeze = _read_json(baseline_freeze_path)
    baseline_freeze_content = _verify_content_hash(
        baseline_freeze, context="baseline score-freeze manifest"
    )
    baseline_artifacts = baseline_freeze.get("artifacts")
    if not isinstance(baseline_artifacts, list) or not baseline_artifacts:
        raise ResidualGraphDeemError("baseline score-freeze has no artifact inventory")
    baseline_artifact_map = {
        str(row.get("path")): str(row.get("sha256")) for row in baseline_artifacts
    }
    if len(baseline_artifact_map) != len(baseline_artifacts):
        raise ResidualGraphDeemError("duplicate baseline score-freeze artifact path")
    baseline_contract = {
        "artifact_map": baseline_artifact_map,
        "manifest_sha256": sha256_file(baseline_freeze_path),
        "content_sha256": baseline_freeze_content,
    }
    if (
        definition.get("schema") != RUN_SCHEMA
        or definition.get("status") != config.get("status")
        or definition.get("experiment_id") != config.get("experiment_id")
        or definition.get("config_sha256") != sha256_file(config_path)
        or definition.get("registry_file_sha256") != sha256_file(registry_path)
        or definition.get("registry_content_sha256") != registry.get("registry_content_sha256")
        or definition.get("targets_accessed_during_fit") is not False
        or definition.get("labels_module_imported") is not False
        or definition.get("scientific_boundary") != config.get("scientific_boundary")
        or definition.get("frozen_screen_rule") != config.get("frozen_screen_rule")
        or definition.get("baseline_score_freeze_manifest_sha256")
        != sha256_file(baseline_freeze_path)
        or definition.get("baseline_score_freeze_content_sha256")
        != baseline_freeze_content
    ):
        raise ResidualGraphDeemError("invalid pair-router run definition")
    rows = manifest.get("artifacts")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("run_definition_sha256") != definition_sha
        or not isinstance(rows, list)
        or int(manifest.get("n_artifacts", -1)) != len(rows)
        or manifest.get("all_healthy") is not True
        or manifest.get("targets_accessed_during_fit") is not False
        or manifest.get("labels_module_imported") is not False
    ):
        raise ResidualGraphDeemError("invalid pair-router fit manifest")
    invocation_sha = str(manifest.get("invocation_sha256", ""))
    if not invocation_sha or manifest_path.stem != invocation_sha:
        raise ResidualGraphDeemError("fit-manifest invocation binding failed")
    variants = [str(value) for value in manifest.get("selected_variants", [])]
    cells = [str(value) for value in manifest.get("selected_cells", [])]
    seeds = [int(value) for value in manifest.get("selected_seeds", [])]
    registry_cells = {str(row["cell_id"]) for row in registry["cells"]}
    if (
        not variants
        or not cells
        or not seeds
        or len(variants) != len(set(variants))
        or len(cells) != len(set(cells))
        or len(seeds) != len(set(seeds))
        or not set(variants).issubset(set(variant_lookup))
        or not set(cells).issubset(registry_cells)
    ):
        raise ResidualGraphDeemError("invalid run Cartesian roster")
    expected_list = list(itertools.product(variants, cells, seeds))
    expected = set(expected_list)
    actual: set[tuple[str, str, int]] = set()
    actual_list: list[tuple[str, str, int]] = []
    for row in rows:
        key = (
            str(row.get("variant_id", "")),
            str(row.get("cell_id", "")),
            int(row.get("seed", -1)),
        )
        if key in actual:
            raise ResidualGraphDeemError(f"duplicate fit-manifest key: {key}")
        actual.add(key)
        actual_list.append(key)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ResidualGraphDeemError(
            f"fit manifest is not the exact run Cartesian product; missing={missing}, extra={extra}"
        )
    if actual_list != expected_list:
        raise ResidualGraphDeemError("fit manifest artifact order is not canonical")
    expected_keys = [
        f"{variant}|{cell}|{seed}" for variant, cell, seed in expected_list
    ]
    if manifest.get("expected_cartesian_keys") != expected_keys:
        raise ResidualGraphDeemError("manifest expected-Cartesian key list mismatch")
    completion_path = run_dir / "fit_completions" / f"{invocation_sha}.json"
    completion = _read_json(completion_path)
    _verify_content_hash(completion, context="pair-router fit completion")
    if (
        completion.get("schema") != "deem_b3_pair_router_fit_complete_v1"
        or completion.get("status") != "complete"
        or completion.get("run_definition_sha256") != definition_sha
        or completion.get("invocation_sha256") != invocation_sha
        or completion.get("fit_manifest_path")
        != manifest_path.relative_to(run_dir.resolve()).as_posix()
        or completion.get("fit_manifest_sha256") != sha256_file(manifest_path)
        or completion.get("fit_manifest_content_sha256")
        != manifest.get("content_sha256")
        or int(completion.get("n_records", -1)) != len(expected_list)
        or completion.get("variants") != variants
        or completion.get("cells") != cells
        or completion.get("seeds") != seeds
        or completion.get("all_healthy") is not True
        or completion.get("targets_accessed_during_fit") is not False
        or completion.get("labels_module_imported") is not False
    ):
        raise ResidualGraphDeemError("invalid pair-router fit completion")
    return definition, definition_sha, manifest, rows, baseline_contract


def _preflight(
    *,
    config_path: Path,
    registry_path: Path,
    bundle_dir: Path,
    baseline_dir: Path,
    run_dir: Path,
    manifest_path: Path | None,
    variants_raw: str,
    cells_raw: str,
    seeds_raw: str,
) -> dict:
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module was imported before pair-router preflight")
    config, variant_lookup = _load_config(config_path)
    registry = load_registry(registry_path)
    registry_ids = [str(row["cell_id"]) for row in registry["cells"]]
    if len(registry_ids) != len(set(registry_ids)):
        raise ResidualGraphDeemError("registry cell IDs are duplicated")
    (
        definition,
        definition_sha,
        manifest,
        manifest_rows,
        baseline_contract,
    ) = _load_run_contract(
        run_dir,
        manifest_path,
        baseline_dir,
        config_path,
        registry_path,
        config,
        variant_lookup,
        registry,
    )
    run_variants = [str(value) for value in manifest["selected_variants"]]
    run_cells = [str(value) for value in manifest["selected_cells"]]
    run_seeds = [int(value) for value in manifest["selected_seeds"]]
    variants = _parse_selection(variants_raw, run_variants)
    cells = _parse_selection(
        cells_raw,
        run_cells,
        special=config["screen_cells"],
        special_name="screen",
    )
    development_seeds = config["baseline"]["development_score_seeds"]
    seeds = _parse_seeds(seeds_raw, run_seeds, development_seeds)

    bundles = {}
    bundle_audit = {}
    for cell in run_cells:
        path = bundle_dir / f"{cell}.npz"
        bundle = load_target_free_bundle(path)
        audit = _validate_bundle(bundle, registry_cell(registry, cell), path)
        bundles[cell] = bundle
        bundle_audit[cell] = audit

    baseline_scores: dict[tuple[str, int], np.ndarray] = {}
    baseline_audits: dict[tuple[str, int], dict] = {}
    for cell, seed in itertools.product(run_cells, run_seeds):
        score, audit = _load_baseline(
            baseline_dir,
            cell,
            seed,
            bundles[cell],
            baseline_contract,
        )
        baseline_scores[(cell, seed)] = score
        baseline_audits[(cell, seed)] = audit

    manifest_lookup = {
        (str(row["variant_id"]), str(row["cell_id"]), int(row["seed"])): row
        for row in manifest_rows
    }
    selected_keys = set(itertools.product(variants, cells, seeds))
    selected_fits = {}
    selected_metadata = {}
    gate_diagnostics = {}
    for variant, cell, seed in itertools.product(run_variants, run_cells, run_seeds):
        array_path, metadata_path = _fit_paths(run_dir, variant, cell, seed)
        if not array_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"missing pair-router fit: {variant}/{cell}/seed{seed}")
        metadata = _read_json(metadata_path)
        row = manifest_lookup[(variant, cell, seed)]
        _validate_metadata(
            metadata,
            variant=variant,
            experiment_id=str(config["experiment_id"]),
            cell=cell,
            seed=seed,
            bundle=bundles[cell],
            variant_row=variant_lookup[variant],
            run_definition_sha256=definition_sha,
            array_path=array_path.resolve(),
            metadata_path=metadata_path.resolve(),
            manifest_row=row,
            baseline_audit=baseline_audits[(cell, seed)],
            bundle_audit=bundle_audit[cell],
            run_dir=run_dir,
        )
        with np.load(array_path, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
        diagnostic = _validate_fit_arrays(
            arrays,
            metadata,
            bundles[cell],
            variant_lookup[variant],
            baseline_scores[(cell, seed)],
        )
        key = (variant, cell, seed)
        if key in selected_keys:
            selected_fits[key] = arrays
            selected_metadata[key] = metadata
            gate_diagnostics[key] = diagnostic
    if set(selected_fits) != selected_keys:
        raise ResidualGraphDeemError("selected fit cache is incomplete after preflight")
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module entered import closure during preflight")
    return {
        "config": config,
        "variant_lookup": variant_lookup,
        "registry": registry,
        "definition": definition,
        "definition_sha256": definition_sha,
        "manifest": manifest,
        "variants": variants,
        "cells": cells,
        "seeds": seeds,
        "bundles": {cell: bundles[cell] for cell in cells},
        "baseline_scores": {
            (cell, seed): baseline_scores[(cell, seed)]
            for cell, seed in itertools.product(cells, seeds)
        },
        "fits": selected_fits,
        "fit_metadata": selected_metadata,
        "gate_diagnostics": gate_diagnostics,
        "bundle_audit": {cell: bundle_audit[cell] for cell in cells},
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "registry_sha256": sha256_file(registry_path),
            "run_definition_sha256": definition_sha,
            "run_source_sha256": definition["source_sha256"],
            "run_environment_sha256": definition["environment"][
                "environment_sha256"
            ],
            "baseline_score_freeze_manifest_sha256": definition[
                "baseline_score_freeze_manifest_sha256"
            ],
            "fit_manifest_sha256": sha256_file(
                run_dir
                / "fit_manifests"
                / f"{manifest['invocation_sha256']}.json"
            ),
            "fit_completion_sha256": sha256_file(
                run_dir
                / "fit_completions"
                / f"{manifest['invocation_sha256']}.json"
            ),
            "evaluator_sha256": sha256_file(Path(__file__)),
        },
    }


def _load_targets_after_preflight(preflight: Mapping[str, Any], sidecar_dir: Path):
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module imported before explicit evaluation phase")
    from spectral_utils.residual_graph_deem_labels import (  # noqa: PLC0415
        SIDECAR_SCHEMA,
        join_labels_by_id,
        load_label_sidecar,
    )

    targets = {}
    audit = {}
    for cell in preflight["cells"]:
        bundle = preflight["bundles"][cell]
        path = sidecar_dir / f"{cell}.npz"
        manifest_path = path.with_suffix(".manifest.json")
        manifest = _read_json(manifest_path)
        sidecar = load_label_sidecar(path)
        if (
            manifest.get("schema") != SIDECAR_SCHEMA
            or manifest.get("cell_id") != cell
            or int(manifest.get("n_rows", -1)) != len(bundle.row_ids)
            or manifest.get("sidecar_sha256") != sidecar.sidecar_sha256
            or manifest.get("unordered_row_id_sha256")
            != canonical_sha256(sorted(bundle.row_ids))
        ):
            raise ResidualGraphDeemError(f"invalid label-sidecar manifest: {cell}")
        target = join_labels_by_id(bundle, sidecar)
        if len(np.unique(target)) != 2:
            raise ResidualGraphDeemError(f"single-class evaluation target: {cell}")
        targets[cell] = target
        audit[cell] = {
            "sidecar_sha256": sidecar.sidecar_sha256,
            "sidecar_manifest_sha256": sha256_file(manifest_path),
        }
    return targets, audit


def _metrics(target: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=np.int8)
    values = np.asarray(score, dtype=np.float64)
    if (
        y.shape != values.shape
        or not np.isfinite(values).all()
        or np.min(values) < 0.0
        or np.max(values) > 1.0
        or len(np.unique(y)) != 2
    ):
        raise ResidualGraphDeemError("invalid target/score pair")
    return {
        "auroc": float(roc_auc_score(y, values)),
        "auprc": float(average_precision_score(y, values)),
    }


def _summary(rows: Sequence[Mapping[str, Any]], method: str, metric: str) -> dict:
    selected = [row for row in rows if row["method"] == method]
    if not selected:
        raise ResidualGraphDeemError(f"no rows for summary: {method}/{metric}")
    grouped = defaultdict(list)
    for row in selected:
        grouped[str(row["dataset_family"])].append(float(row[metric]))
    family_means = {
        family: float(np.mean(values)) for family, values in sorted(grouped.items())
    }
    return {
        "method": method,
        "metric": metric,
        "n_cells": len(selected),
        "n_families": len(family_means),
        "cell_macro": float(np.mean([float(row[metric]) for row in selected])),
        "equal_family_macro": float(np.mean(list(family_means.values()))),
        "worst_cell": float(min(float(row[metric]) for row in selected)),
        "family_means": family_means,
    }


def _family_deltas(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    metric: str,
    *,
    reference: str = "B3",
) -> tuple[dict[str, float], dict[str, float]]:
    lookup = {(str(row["cell_id"]), str(row["method"])): row for row in rows}
    cells = sorted({str(row["cell_id"]) for row in rows})
    grouped = defaultdict(list)
    cell_delta = {}
    for cell in cells:
        try:
            base = lookup[(cell, reference)]
            cand = lookup[(cell, candidate)]
        except KeyError as exc:
            raise ResidualGraphDeemError(f"incomplete paired metric rows: {candidate}/{cell}") from exc
        delta = float(cand[metric]) - float(base[metric])
        grouped[str(base["dataset_family"])].append(delta)
        cell_delta[cell] = delta
    family_delta = {
        family: float(np.mean(values)) for family, values in sorted(grouped.items())
    }
    return family_delta, cell_delta


def _comparison(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    *,
    reference: str = "B3",
    draws: int,
    seed: int,
) -> dict:
    if int(draws) < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    family_delta, cell_delta = _family_deltas(
        rows, candidate, "auroc", reference=reference
    )
    family_auprc, _ = _family_deltas(
        rows, candidate, "auprc", reference=reference
    )
    families = tuple(family_delta)
    values = np.asarray([family_delta[family] for family in families], dtype=np.float64)
    observed = float(np.mean(values))
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    distribution = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        selected = rng.integers(0, len(values), size=len(values))
        distribution[draw] = float(np.mean(values[selected]))
    if len(values) > 20:
        raise ResidualGraphDeemError("exact sign-flip enumeration is unexpectedly large")
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = np.mean(signs * values[None, :], axis=1)
    p_value = float(np.mean(null >= observed - 1e-15))
    return {
        "candidate": candidate,
        "reference": reference,
        "equal_family_auroc_delta": observed,
        "equal_family_auprc_delta": float(np.mean(list(family_auprc.values()))),
        "descriptive_family_bootstrap_lower": float(np.quantile(distribution, 0.025)),
        "descriptive_family_bootstrap_upper": float(np.quantile(distribution, 0.975)),
        "exact_family_signflip_one_sided_p": p_value,
        "exact_family_signflip_assignments": int(2 ** len(values)),
        "wins": int(sum(value > TOLERANCE for value in cell_delta.values())),
        "ties": int(sum(abs(value) <= TOLERANCE for value in cell_delta.values())),
        "losses": int(sum(value < -TOLERANCE for value in cell_delta.values())),
        "worst_cell_delta": float(min(cell_delta.values())),
        "family_delta": family_delta,
        "cell_delta": cell_delta,
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "bootstrap_is_descriptive_not_a_null_test": True,
    }


def _screen_verdict(
    config: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
    *,
    variants: Sequence[str],
    cells: Sequence[str],
    seeds: Sequence[int],
    dataset_families: Sequence[str],
) -> dict:
    rule = config["frozen_screen_rule"]
    primary = str(rule["mechanistic_primary"])
    official_cells = [str(value) for value in config["screen_cells"]]
    official_seeds = [int(value) for value in config["baseline"]["development_score_seeds"]]
    eligible = (
        list(cells) == official_cells
        and len(cells) == 8
        and list(seeds) == official_seeds
        and primary in variants
        and len(set(dataset_families)) == 8
    )
    if not eligible:
        reasons = []
        if list(cells) != official_cells:
            reasons.append("cell selection is not the ordered frozen eight-cell screen")
        if list(seeds) != official_seeds:
            reasons.append("seed selection is not the frozen development-score seed roster")
        if primary not in variants:
            reasons.append("mechanistic primary was not evaluated")
        if len(set(dataset_families)) != 8:
            reasons.append("frozen screen does not span eight unique dataset families")
        return {
            "status": "not_applicable_sensitivity_analysis",
            "verdict": "NOT_APPLICABLE",
            "mechanistic_primary": primary,
            "reasons": reasons,
            "secondary_variants_cannot_substitute_for_primary": True,
        }
    row = next(item for item in comparisons if item["candidate"] == primary)
    thresholds = rule["primary_survives_only_if"]
    clauses = [
        {
            "name": "equal_family_auroc_delta_min",
            "observed": row["equal_family_auroc_delta"],
            "threshold": thresholds["equal_family_auroc_delta_min"],
            "passed": row["equal_family_auroc_delta"]
            >= float(thresholds["equal_family_auroc_delta_min"]),
        },
        {
            "name": "descriptive_family_bootstrap_lower_min",
            "observed": row["descriptive_family_bootstrap_lower"],
            "threshold": thresholds["descriptive_family_bootstrap_lower_min"],
            "passed": row["descriptive_family_bootstrap_lower"]
            >= float(thresholds["descriptive_family_bootstrap_lower_min"]),
        },
        {
            "name": "exact_family_signflip_one_sided_p_max",
            "observed": row["exact_family_signflip_one_sided_p"],
            "threshold": thresholds["exact_family_signflip_one_sided_p_max"],
            "passed": row["exact_family_signflip_one_sided_p"]
            <= float(thresholds["exact_family_signflip_one_sided_p_max"]),
        },
        {
            "name": "wins_plus_ties_min_of_8",
            "observed": int(row["wins"] + row["ties"]),
            "threshold": int(thresholds["wins_plus_ties_min_of_8"]),
            "passed": int(row["wins"] + row["ties"])
            >= int(thresholds["wins_plus_ties_min_of_8"]),
        },
        {
            "name": "worst_cell_delta_min",
            "observed": row["worst_cell_delta"],
            "threshold": thresholds["worst_cell_delta_min"],
            "passed": row["worst_cell_delta"]
            >= float(thresholds["worst_cell_delta_min"]),
        },
    ]
    passed = all(bool(clause["passed"]) for clause in clauses)
    return {
        "status": "official_frozen_eight_cell_screen",
        "verdict": "PASS" if passed else "FAIL",
        "mechanistic_primary": primary,
        "n_cells": 8,
        "n_dataset_families": 8,
        "clauses": clauses,
        "secondary_variants_cannot_substitute_for_primary": True,
    }


def _control_scores(
    arrays: Mapping[str, np.ndarray],
    *,
    aligned_bias: float,
    variant: str,
    cell: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return label-free static-mean and deterministic row-permuted controls."""

    base = np.asarray(arrays["base_family_contributions"], dtype=np.float64)
    gates = np.asarray(arrays["gates"], dtype=np.float64)
    if base.shape != gates.shape or base.ndim != 2:
        raise ResidualGraphDeemError("control family/gate shape mismatch")
    mean_gate = gates.mean(axis=0, keepdims=True)
    static_logit = float(aligned_bias) + np.sum(base * mean_gate, axis=1)
    permutation_seed = _stable_seed(
        "pair_router_row_permuted_gate_v1", variant, cell, str(int(seed))
    )
    rng = np.random.Generator(np.random.PCG64(permutation_seed))
    permutation = rng.permutation(len(gates))
    if len(gates) > 1 and np.array_equal(permutation, np.arange(len(gates))):
        permutation = np.roll(permutation, 1)
    permuted_logit = float(aligned_bias) + np.sum(base * gates[permutation], axis=1)
    return _expit(static_logit), _expit(permuted_logit), permutation_seed


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--fit-manifest",
        type=Path,
        default=None,
        help="specific run_dir/fit_manifests/*.json; auto-selects only when unique",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--cells", default="screen")
    parser.add_argument("--seeds", default="development")
    parser.add_argument("--bootstrap-draws", type=int, default=9999)
    args = parser.parse_args()

    # Phase A: no code capable of reading labels has been imported yet.
    preflight = _preflight(
        config_path=args.config,
        registry_path=args.registry,
        bundle_dir=args.bundle_dir,
        baseline_dir=args.baseline_dir,
        run_dir=args.run_dir,
        manifest_path=args.fit_manifest,
        variants_raw=args.variants,
        cells_raw=args.cells,
        seeds_raw=args.seeds,
    )
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module imported before preflight completed")

    # Phase B: labels are opened only after every frozen artifact has passed.
    targets, sidecar_audit = _load_targets_after_preflight(preflight, args.sidecar_dir)
    variants = preflight["variants"]
    cells = preflight["cells"]
    seeds = preflight["seeds"]
    per_fit = []
    per_cell = []
    gate_rows = []
    for cell in cells:
        bundle = preflight["bundles"][cell]
        target = targets[cell]
        baseline_seed_scores = [
            preflight["baseline_scores"][(cell, seed)] for seed in seeds
        ]
        for seed, score in zip(seeds, baseline_seed_scores):
            per_fit.append(
                {
                    "cell_id": cell,
                    "dataset_family": bundle.dataset_family,
                    "task_type": bundle.task_type,
                    "method": "B3",
                    "seed": seed,
                    **_metrics(target, score),
                }
            )
        baseline = np.mean(np.stack(baseline_seed_scores), axis=0)
        per_cell.append(
            {
                "cell_id": cell,
                "dataset_family": bundle.dataset_family,
                "task_type": bundle.task_type,
                "method": "B3",
                "n_seeds_averaged": len(seeds),
                **_metrics(target, baseline),
            }
        )
        for variant in variants:
            candidate_scores = []
            static_scores = []
            permuted_scores = []
            for seed in seeds:
                arrays = preflight["fits"][(variant, cell, seed)]
                metadata = preflight["fit_metadata"][(variant, cell, seed)]
                score = np.asarray(arrays["score"], dtype=np.float64)
                static_score, permuted_score, permutation_seed = _control_scores(
                    arrays,
                    aligned_bias=float(metadata["aligned_bias"]),
                    variant=variant,
                    cell=cell,
                    seed=seed,
                )
                candidate_scores.append(score)
                static_scores.append(static_score)
                permuted_scores.append(permuted_score)
                for method, values in (
                    (variant, score),
                    (f"{variant}::MEAN_GATE", static_score),
                    (f"{variant}::ROW_PERMUTED_GATE", permuted_score),
                ):
                    per_fit.append(
                        {
                            "cell_id": cell,
                            "dataset_family": bundle.dataset_family,
                            "task_type": bundle.task_type,
                            "method": method,
                            "seed": seed,
                            **_metrics(target, values),
                        }
                    )
                gate_rows.append(
                    {
                        "cell_id": cell,
                        "dataset_family": bundle.dataset_family,
                        "task_type": bundle.task_type,
                        "variant": variant,
                        "seed": seed,
                        "row_permutation_seed": permutation_seed,
                        **preflight["gate_diagnostics"][(variant, cell, seed)],
                    }
                )
            for method, matrix in (
                (variant, candidate_scores),
                (f"{variant}::MEAN_GATE", static_scores),
                (f"{variant}::ROW_PERMUTED_GATE", permuted_scores),
            ):
                averaged = np.mean(np.stack(matrix), axis=0)
                per_cell.append(
                    {
                        "cell_id": cell,
                        "dataset_family": bundle.dataset_family,
                        "task_type": bundle.task_type,
                        "method": method,
                        "n_seeds_averaged": len(seeds),
                        **_metrics(target, averaged),
                    }
                )

    methods = ["B3"]
    for variant in variants:
        methods.extend(
            [
                variant,
                f"{variant}::MEAN_GATE",
                f"{variant}::ROW_PERMUTED_GATE",
            ]
        )
    summaries = [
        _summary(per_cell, method, metric)
        for method in methods
        for metric in ("auroc", "auprc")
    ]
    cell_seed_hash = canonical_sha256({"cells": cells, "seeds": seeds})
    comparisons = []
    primary_comparisons = []
    moe_specificity = []
    for variant in variants:
        rows_for_variant = []
        for reference in (
            "B3",
            f"{variant}::MEAN_GATE",
            f"{variant}::ROW_PERMUTED_GATE",
        ):
            comparison_seed = _stable_seed(
                str(preflight["config"]["experiment_id"]),
                variant,
                reference,
                cell_seed_hash,
                "descriptive_family_block_bootstrap_v1",
            )
            comparison = _comparison(
                per_cell,
                variant,
                reference=reference,
                draws=int(args.bootstrap_draws),
                seed=comparison_seed,
            )
            comparisons.append(comparison)
            rows_for_variant.append(comparison)
            if reference == "B3":
                primary_comparisons.append(comparison)
        versus_static = rows_for_variant[1]
        versus_permuted = rows_for_variant[2]
        moe_specificity.append(
            {
                "variant": variant,
                "equal_family_auroc_delta_vs_mean_gate": versus_static[
                    "equal_family_auroc_delta"
                ],
                "equal_family_auroc_delta_vs_row_permuted_gate": versus_permuted[
                    "equal_family_auroc_delta"
                ],
                "beats_both_controls_on_equal_family_point_estimate": bool(
                    versus_static["equal_family_auroc_delta"] > 0.0
                    and versus_permuted["equal_family_auroc_delta"] > 0.0
                ),
                "interpretation": (
                    "descriptive control contrast only; a router gain over B3 is not "
                    "specific evidence for sample-dependent MoE behavior unless it also "
                    "beats both controls"
                ),
            }
        )
    verdict = _screen_verdict(
        preflight["config"],
        primary_comparisons,
        variants=variants,
        cells=cells,
        seeds=seeds,
        dataset_families=[
            preflight["bundles"][cell].dataset_family for cell in cells
        ],
    )
    gate_summary = []
    for variant in variants:
        selected = [row for row in gate_rows if row["variant"] == variant]
        gate_summary.append(
            {
                "variant": variant,
                "n_fits": len(selected),
                "gate_mean_abs_deviation_from_one": float(
                    np.mean([row["gate_mean_abs_deviation_from_one"] for row in selected])
                ),
                "gate_min": float(min(row["gate_min"] for row in selected)),
                "gate_max": float(max(row["gate_max"] for row in selected)),
                "pair_transfer_mean_abs": float(
                    np.mean([row["pair_transfer_mean_abs"] for row in selected])
                ),
                "pair_open_mean": float(np.mean([row["pair_open_mean"] for row in selected])),
                "router_delta_logit_sd_mean": float(
                    np.mean([row["router_delta_logit_sd"] for row in selected])
                ),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "PER_FIT_METRICS.csv", per_fit)
    _write_csv(args.out_dir / "PER_CELL_METRICS.csv", per_cell)
    _write_csv(args.out_dir / "GATE_DIAGNOSTICS.csv", gate_rows)
    atomic_write_json(args.out_dir / "SUMMARY.json", summaries)
    atomic_write_json(args.out_dir / "COMPARISONS.json", comparisons)
    atomic_write_json(args.out_dir / "MOE_SPECIFICITY.json", moe_specificity)
    atomic_write_json(args.out_dir / "GATE_SUMMARY.json", gate_summary)
    atomic_write_json(args.out_dir / "SCREEN_VERDICT.json", verdict)
    report = {
        "schema": "deem_b3_pair_router_evaluation_v1",
        "status": "complete",
        "scientific_tier": "retrospective_exploratory",
        "natural_24cell_targets_previously_opened": True,
        "strict_two_pass_preflight_before_label_import": True,
        "label_module_imported_only_after_preflight": True,
        "experiment_id": preflight["config"]["experiment_id"],
        "variants": variants,
        "cells": cells,
        "seeds_averaged": seeds,
        "bootstrap_draws": int(args.bootstrap_draws),
        "hashes": preflight["hashes"],
        "bundle_audit": preflight["bundle_audit"],
        "sidecar_audit": sidecar_audit,
        "summaries": summaries,
        "comparisons": comparisons,
        "moe_specificity": moe_specificity,
        "graph_interpretation_boundary": {
            "graph_arms_are_functional_loo2_at_inference": True,
            "graph_arms_are_not_claimed_strict_loo2_end_to_end": True,
            "reason": (
                "the fixed global Laplacian is built from all-family LOO residual "
                "geometry and can therefore encode a routed pair during training"
            ),
        },
        "router_identifiability_boundary": {
            "family_multipliers_gamma_are_interpretable": True,
            "individual_pair_transfers_are_not_identifiable": True,
            "reason": (
                "antisymmetric pair flows contain a cycle-space gauge; only their "
                "divergence into the family multipliers is identified"
            ),
        },
        "screen_verdict": verdict,
    }
    atomic_write_json(args.out_dir / "REPORT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
