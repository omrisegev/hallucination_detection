#!/usr/bin/env python3
"""Freeze label-free pairwise-residual routing scores over exact B3 states.

The default invocation is deliberately the eight-cell development screen at
seed 0.  This process has a physical target firewall: it never imports the
label-sidecar module, and checks that boundary before and after every fit.
Graph-weighted variants receive one fixed union-k7 Laplacian built from true
grouped-fold leave-one-family-out residuals of the frozen five-seed B3
ensemble.  Nongraph variants do not build or load that graph.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_deem_b3_moe_v1 import load_frozen_b3  # noqa: E402
from spectral_utils.deem_b3_pair_router import (  # noqa: E402
    PairRouterConfig,
    _validate_config,
    fit_pair_residual_router,
)
from spectral_utils.deem_b3_residual_moe import (  # noqa: E402
    FAMILY_ORDER,
    build_residual_cell,
    load_frozen_b3_ensemble,
)
from spectral_utils.graph_topology import self_safe_knn_graph  # noqa: E402
from spectral_utils.laplacian_upcr import (  # noqa: E402
    graph_diagnostics,
    symmetric_normalized_laplacian,
)
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ResidualGraphDeemError,
    assign_grouped_length_folds,
    atomic_save_npz,
    atomic_write_json,
    canonical_sha256,
    donor_risk_matrix,
    environment_fingerprint,
    jsonable,
    sha256_file,
)
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    BUNDLE_SCHEMA,
    assert_no_target_fields,
    load_registry,
    load_target_free_bundle,
    registry_cell,
)


DEFAULT_CONFIG = ROOT / "configs/deem_b3_pair_router_v1.json"
DEFAULT_REGISTRY = ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"
LABEL_MODULE = "spectral_utils.residual_graph_deem_labels"
EXPECTED_VARIANTS = (
    "A0_B3_EXACT_ALIAS",
    "A1_PAIR_RESIDUAL_CD_A25",
    "A2_PAIR_RESIDUAL_CD_GEO_A25",
    "A3_PAIR_RESIDUAL_GEO_ONLY_A25",
    "A4_PAIR_RESIDUAL_CD_A50",
    "A5_PAIR_RESIDUAL_CD_GEO_A50",
)
SOURCE_DEPENDENCIES = (
    ROOT / "spectral_utils/deem_b3_pair_router.py",
    ROOT / "spectral_utils/deem_b3_moe.py",
    ROOT / "spectral_utils/deem_b3_residual_moe.py",
    ROOT / "spectral_utils/adapted_dufs.py",
    ROOT / "spectral_utils/residual_graph_deem.py",
    ROOT / "spectral_utils/residual_graph_deem_data.py",
    ROOT / "spectral_utils/graph_topology.py",
    ROOT / "spectral_utils/laplacian_upcr.py",
    ROOT / "spectral_utils/feature_contract.py",
    ROOT / "spectral_utils/specrage_views.py",
    ROOT / "scripts/run_deem_b3_moe_v1.py",
    ROOT / "scripts/run_deem_b3_pair_router_v1.py",
)


@dataclass(frozen=True)
class GraphInput:
    laplacian: csr_matrix
    arrays: dict[str, np.ndarray]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class PreparedCell:
    bundle: Any
    bundle_audit: dict[str, Any]
    X_risk: np.ndarray
    transform: Any
    graph: GraphInput | None


def _assert_label_firewall() -> None:
    imported = sorted(
        name
        for name in sys.modules
        if name == LABEL_MODULE or name.startswith(LABEL_MODULE + ".")
    )
    if imported:
        raise ResidualGraphDeemError(
            "label module crossed the fit boundary: " + ", ".join(imported)
        )


def _verified_content_json(path: Path, *, expected_schema: str | None = None) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.get("content_sha256")
    unhashed = dict(value)
    unhashed.pop("content_sha256", None)
    if expected is None or canonical_sha256(unhashed) != expected:
        raise ResidualGraphDeemError(f"JSON content hash mismatch: {path}")
    if expected_schema is not None and value.get("schema") != expected_schema:
        raise ResidualGraphDeemError(f"JSON schema mismatch: {path}")
    return value


def load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "deem_b3_pair_router_v1_config":
        raise ResidualGraphDeemError("pair-router config schema mismatch")
    variants = value.get("variants", [])
    identifiers = tuple(str(row.get("id")) for row in variants)
    if identifiers != EXPECTED_VARIANTS:
        raise ResidualGraphDeemError(
            "pair-router roster/order must be the frozen A0--A5 menu"
        )
    allowed = {field.name for field in fields(PairRouterConfig)}
    for row in variants:
        settings = row.get("config", {})
        unknown = sorted(set(settings) - allowed)
        if unknown:
            raise ResidualGraphDeemError(
                f"unknown pair-router fields for {row['id']}: {unknown}"
            )
        parsed = PairRouterConfig(**settings)
        try:
            _validate_config(parsed)
        except ValueError as exc:
            raise ResidualGraphDeemError(
                f"invalid pair-router config for {row['id']}"
            ) from exc
    if value.get("frozen_screen_rule", {}).get("mechanistic_primary") != (
        "A4_PAIR_RESIDUAL_CD_A50"
    ):
        raise ResidualGraphDeemError("the frozen mechanistic primary changed")
    graph = value.get("graph", {})
    if (
        int(graph.get("k", -1)) != 7
        or graph.get("topology") != "self_safe_self_tuning_union_knn"
        or graph.get("laplacian") != "symmetric_normalized"
        or not graph.get("fixed_during_router_fit")
    ):
        raise ResidualGraphDeemError("the frozen graph contract changed")
    boundary = value.get("scientific_boundary", {})
    required = (
        "fit_is_label_free",
        "baseline_is_frozen",
        "nongraph_router_arms_are_strictly_self_free_for_each_routed_pair",
        "graph_arms_are_functional_inference_loo2_not_end_to_end_strict_loo2",
        "graph_is_built_only_for_graph_weighted_variants",
        "natural_24cell_targets_previously_opened",
    )
    if any(not boundary.get(name) for name in required):
        raise ResidualGraphDeemError("pair-router scientific boundary is incomplete")
    screen = [str(cell) for cell in value.get("screen_cells", [])]
    if len(screen) != 8 or len(screen) != len(set(screen)):
        raise ResidualGraphDeemError("development screen must contain eight unique cells")
    baseline = value.get("baseline", {})
    available = [int(seed) for seed in baseline.get("available_fit_seeds", [])]
    development = [int(seed) for seed in baseline.get("development_score_seeds", [])]
    graph_seeds = [int(seed) for seed in baseline.get("residual_graph_ensemble_seeds", [])]
    if available != [0, 1, 2, 3, 4] or development != [0] or graph_seeds != available:
        raise ResidualGraphDeemError("the frozen B3 seed contract changed")
    return value


def _source_manifest(config_path: Path) -> dict[str, str]:
    paths = list(SOURCE_DEPENDENCIES) + [config_path.resolve()]
    output: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            name = path.relative_to(ROOT).as_posix()
        except ValueError:
            name = str(path)
        if name in output:
            raise ResidualGraphDeemError(f"duplicate source dependency: {name}")
        output[name] = sha256_file(path)
    return dict(sorted(output.items()))


def _ndarray_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_sha256(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_baseline_freeze(baseline_dir: Path) -> tuple[dict, dict[str, str], dict]:
    path = baseline_dir / "SCORE_FREEZE_MANIFEST.json"
    manifest = _verified_content_json(
        path, expected_schema="deem_vs_iupcr_score_freeze_v1"
    )
    if (
        manifest.get("status") != "complete"
        or bool(manifest.get("debug"))
        or "B3" not in manifest.get("arms", [])
    ):
        raise ResidualGraphDeemError("frozen B3 score manifest is not admissible")
    entries = manifest.get("artifacts", [])
    artifact_map = {str(row["path"]): str(row["sha256"]) for row in entries}
    if len(artifact_map) != len(entries):
        raise ResidualGraphDeemError("duplicate paths in frozen B3 score manifest")
    audit = {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": sha256_file(path),
        "manifest_content_sha256": manifest["content_sha256"],
        "run_definition_sha256": manifest["run_definition_sha256"],
    }
    return manifest, artifact_map, audit


def _verify_baseline_member(
    baseline_dir: Path,
    artifact_map: Mapping[str, str],
    cell_id: str,
    seed: int,
) -> None:
    for suffix in ("npz", "json"):
        relative = f"fits/{cell_id}/B3__seed{int(seed)}.{suffix}"
        path = baseline_dir / relative
        expected = artifact_map.get(relative)
        if expected is None or not path.is_file() or sha256_file(path) != expected:
            raise ResidualGraphDeemError(
                f"B3 artifact is not bound to its freeze manifest: {relative}"
            )


def _load_bound_bundle(
    bundle_dir: Path,
    registry: Mapping[str, Any],
    cell_id: str,
) -> tuple[Any, dict[str, Any]]:
    path = bundle_dir / f"{cell_id}.npz"
    sidecar_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"missing target-free bundle or manifest: {cell_id}")
    bundle = load_target_free_bundle(path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    registered = registry_cell(registry, cell_id)
    ordered_row_hash = canonical_sha256(list(bundle.row_ids))
    source = registered["source"]
    checks = {
        "bundle cell": bundle.cell_id == cell_id,
        "bundle family": bundle.dataset_family == str(registered["dataset_family"]),
        "bundle task": bundle.task_type == str(registered["task_type"]),
        "bundle rows": len(bundle.row_ids) == int(registered["n_rows"]),
        "bundle features": tuple(bundle.feature_names) == tuple(registered["feature_names"]),
        "bundle confidence signs": np.array_equal(
            bundle.confidence_signs, np.asarray(registered["confidence_signs"], dtype=np.int8)
        ),
        "bundle inventory": bundle.inventory_sha256 == registered["inventory_sha256"],
        "bundle source": bundle.source_sha256 == source["source_sha256"],
        "bundle source manifest": bundle.manifest_sha256 == source["manifest_sha256"],
        "bundle admission": bundle.admission_sha256 == source["admission_contract_sha256"],
        "sidecar schema": sidecar.get("schema") == BUNDLE_SCHEMA,
        "sidecar cell": sidecar.get("cell_id") == cell_id,
        "sidecar bundle": sidecar.get("bundle_sha256") == bundle.bundle_sha256,
        "sidecar rows": int(sidecar.get("n_rows", -1)) == len(bundle.row_ids),
        "sidecar features": int(sidecar.get("n_features", -1)) == len(bundle.feature_names),
        "sidecar row order": sidecar.get("ordered_row_id_sha256") == ordered_row_hash,
        "sidecar inventory": sidecar.get("inventory_sha256") == bundle.inventory_sha256,
        "sidecar source": sidecar.get("source_sha256") == bundle.source_sha256,
        "sidecar source manifest": sidecar.get("manifest_sha256") == bundle.manifest_sha256,
        "sidecar admission": sidecar.get("admission_sha256") == bundle.admission_sha256,
        "sidecar target firewall": sidecar.get("labels_accessed") is False,
        "sidecar pickle firewall": sidecar.get("allow_pickle") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ResidualGraphDeemError(
            f"target-free bundle binding failed for {cell_id}: {failed}"
        )
    return bundle, {
        "bundle_path": str(path.resolve()),
        "bundle_sha256": bundle.bundle_sha256,
        "bundle_manifest_path": str(sidecar_path.resolve()),
        "bundle_manifest_file_sha256": sha256_file(sidecar_path),
        "ordered_row_id_sha256": ordered_row_hash,
        "inventory_sha256": bundle.inventory_sha256,
        "source_sha256": bundle.source_sha256,
        "source_manifest_sha256": bundle.manifest_sha256,
        "admission_sha256": bundle.admission_sha256,
    }


def _audit_baseline_metadata(
    metadata: Mapping[str, Any],
    bundle: Any,
    seed: int,
) -> None:
    expected = {
        "status": "complete",
        "arm_id": "B3",
        "cell_id": bundle.cell_id,
        "stem": f"B3__seed{int(seed)}",
        "seed": int(seed),
        "bundle_sha256": bundle.bundle_sha256,
        "inventory_sha256": bundle.inventory_sha256,
        "ordered_row_id_sha256": canonical_sha256(list(bundle.row_ids)),
        "source_sha256": bundle.source_sha256,
    }
    mismatches = [name for name, value in expected.items() if metadata.get(name) != value]
    if mismatches:
        raise ResidualGraphDeemError(
            f"frozen B3/bundle binding mismatch for {bundle.cell_id}/seed{seed}: "
            + ", ".join(mismatches)
        )
    if not metadata.get("health", {}).get("healthy"):
        raise ResidualGraphDeemError(
            f"unhealthy frozen B3 baseline: {bundle.cell_id}/seed{seed}"
        )


def _lexical_row_ranks(row_ids: Sequence[str]) -> np.ndarray:
    values = np.asarray(row_ids, dtype=str)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(np.unique(ranks)) != len(values):
        raise AssertionError("row-ID tie keys are not unique")
    return ranks


def _csr_payload(prefix: str, matrix: csr_matrix) -> dict[str, np.ndarray]:
    value = csr_matrix(matrix, dtype=np.float64).copy()
    value.sort_indices()
    return {
        f"{prefix}_data": np.asarray(value.data, dtype=np.float64),
        f"{prefix}_indices": np.asarray(value.indices, dtype=np.int64),
        f"{prefix}_indptr": np.asarray(value.indptr, dtype=np.int64),
        f"{prefix}_shape": np.asarray(value.shape, dtype=np.int64),
    }


def _empty_graph_arrays(
    n_rows: int, family_order: Sequence[str]
) -> dict[str, np.ndarray]:
    order = tuple(str(name) for name in family_order)
    output = {
        "graph_residual_coordinates": np.empty((n_rows, 0), dtype=np.float64),
        "graph_residual_family_order": np.asarray([], dtype=str),
        "graph_global_family_order": np.asarray(order, dtype=str),
        "graph_present_family_mask": np.zeros(len(order), dtype=np.int8),
        "graph_grouped_folds": np.asarray([], dtype=np.int64),
        "graph_row_tie_keys": np.asarray([], dtype=np.float64),
        "graph_loo_predictability": np.asarray([], dtype=np.float64),
    }
    for prefix in ("graph", "laplacian"):
        output[f"{prefix}_data"] = np.asarray([], dtype=np.float64)
        output[f"{prefix}_indices"] = np.asarray([], dtype=np.int64)
        output[f"{prefix}_indptr"] = np.asarray([], dtype=np.int64)
        output[f"{prefix}_shape"] = np.asarray([0, 0], dtype=np.int64)
    return output


def _build_graph_input(
    bundle: Any,
    baseline_dir: Path,
    artifact_map: Mapping[str, str],
    ensemble_seeds: Sequence[int],
    k: int,
) -> GraphInput:
    for seed in ensemble_seeds:
        _verify_baseline_member(baseline_dir, artifact_map, bundle.cell_id, int(seed))
    row_hash = canonical_sha256(list(bundle.row_ids))
    ensemble = load_frozen_b3_ensemble(
        baseline_dir,
        bundle.cell_id,
        seeds=tuple(int(seed) for seed in ensemble_seeds),
        expected_bundle_sha256=bundle.bundle_sha256,
        expected_ordered_row_id_sha256=row_hash,
    )
    if ensemble.score.shape != (len(bundle.row_ids),):
        raise ResidualGraphDeemError(f"B3 ensemble row mismatch: {bundle.cell_id}")
    folds = assign_grouped_length_folds(bundle.group_ids, bundle.raw_trace_length)
    residual = build_residual_cell(
        ensemble,
        baseline_score=ensemble.score,
        folds=folds,
    )
    present = np.asarray(residual.present_mask, dtype=bool)
    coordinates = np.asarray(residual.loo_residuals[:, present], dtype=np.float64)
    present_order = tuple(
        name for name, is_present in zip(residual.family_order, present) if is_present
    )
    if coordinates.shape != (len(bundle.row_ids), len(present_order)):
        raise ResidualGraphDeemError(f"LOO graph-coordinate mismatch: {bundle.cell_id}")
    if coordinates.shape[1] < 4 or not np.isfinite(coordinates).all():
        raise ResidualGraphDeemError(f"invalid LOO graph coordinates: {bundle.cell_id}")
    tie_keys = _lexical_row_ranks(bundle.row_ids)
    graph = self_safe_knn_graph(coordinates, k=int(k), tie_keys=tie_keys)
    laplacian = symmetric_normalized_laplacian(graph)
    graph_diag = graph_diagnostics(graph, laplacian)
    lap_delta = laplacian - laplacian.T
    symmetry_error = float(
        np.max(np.abs(lap_delta.data)) if lap_delta.nnz else 0.0
    )
    trace = float(np.sum(laplacian.diagonal()))
    trace_ratio = trace / len(bundle.row_ids)
    if (
        graph_diag["n_nodes"] != len(bundle.row_ids)
        or graph_diag["n_edges"] <= 0
        or graph_diag["degree_min"] <= 0.0
        or symmetry_error > 1e-10
        or not np.isfinite(laplacian.data).all()
        or abs(trace_ratio - 1.0) > 1e-12
    ):
        raise ResidualGraphDeemError(f"fixed residual graph failed health checks: {bundle.cell_id}")
    arrays = {
        "graph_residual_coordinates": coordinates,
        "graph_residual_family_order": np.asarray(present_order, dtype=str),
        # The pair router contains only families present in this cell.  Persist
        # that exact local canonical order so every graph coordinate has a
        # one-to-one family interpretation even for five-family Math cells.
        "graph_global_family_order": np.asarray(present_order, dtype=str),
        "graph_present_family_mask": np.ones(len(present_order), dtype=np.int8),
        "graph_grouped_folds": np.asarray(folds, dtype=np.int64),
        "graph_row_tie_keys": tie_keys,
        "graph_loo_predictability": np.asarray(
            residual.loo_predictability[present], dtype=np.float64
        ),
        **_csr_payload("graph", graph),
        **_csr_payload("laplacian", laplacian),
    }
    payload_hashes = {
        name: _ndarray_sha256(value) for name, value in sorted(arrays.items())
    }
    diagnostics = {
        "used": True,
        "coordinate_source": "true_grouped_fold_loo_b3_family_contribution_residuals",
        "functional_inference_loo2": True,
        "end_to_end_strict_loo2": False,
        "topology": "self_safe_self_tuning_union_knn",
        "k": int(k),
        "tie_break": "lexical_row_id_rank",
        "laplacian": "symmetric_normalized",
        "fixed_during_router_fit": True,
        "ensemble_seeds": [int(seed) for seed in ensemble_seeds],
        "family_order": list(present_order),
        "grouped_fold_sha256": _ndarray_sha256(folds),
        "coordinate_sha256": _ndarray_sha256(coordinates),
        "graph_payload_sha256": canonical_sha256(payload_hashes),
        "laplacian_trace": trace,
        "laplacian_trace_per_row": trace_ratio,
        "laplacian_symmetry_max_abs": symmetry_error,
        "residual_diagnostics": residual.diagnostics,
        "ensemble_diagnostics": ensemble.diagnostics,
        **graph_diag,
        "uses_labels": False,
    }
    return GraphInput(
        laplacian=csr_matrix(laplacian, dtype=np.float64),
        arrays=arrays,
        diagnostics=diagnostics,
    )


def _prepare_cell(
    *,
    bundle_dir: Path,
    baseline_dir: Path,
    registry: Mapping[str, Any],
    artifact_map: Mapping[str, str],
    cell_id: str,
    need_graph: bool,
    graph_seeds: Sequence[int],
    graph_k: int,
) -> PreparedCell:
    bundle, bundle_audit = _load_bound_bundle(bundle_dir, registry, cell_id)
    X_risk, _, transform = donor_risk_matrix(
        bundle.X_raw, bundle.X_raw, bundle.feature_names
    )
    graph = None
    if need_graph:
        graph = _build_graph_input(
            bundle, baseline_dir, artifact_map, graph_seeds, graph_k
        )
    return PreparedCell(
        bundle=bundle,
        bundle_audit=bundle_audit,
        X_risk=X_risk,
        transform=transform,
        graph=graph,
    )


def _parse_selection(
    value: str,
    available: Sequence[str],
    *,
    screen: Sequence[str] | None = None,
) -> list[str]:
    if value == "all":
        return list(available)
    if value == "screen":
        if screen is None:
            raise ValueError("screen selection is unavailable")
        requested = list(screen)
    else:
        requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(available))
    if not requested or unknown or len(requested) != len(set(requested)):
        raise ValueError(f"invalid selection; unknown={unknown}")
    return requested


def _parse_seeds(value: str, config: Mapping[str, Any]) -> list[int]:
    baseline = config["baseline"]
    available = [int(seed) for seed in baseline["available_fit_seeds"]]
    if value == "all":
        selected = available
    elif value == "screen":
        selected = [int(seed) for seed in baseline["development_score_seeds"]]
    else:
        selected = [int(item.strip()) for item in value.split(",") if item.strip()]
    if (
        not selected
        or not set(selected).issubset(set(available))
        or len(selected) != len(set(selected))
    ):
        raise ValueError(f"seeds must be unique members of {available}")
    return selected


def output_paths(
    out_dir: Path, variant_id: str, cell_id: str, seed: int
) -> tuple[Path, Path]:
    stem = f"{variant_id}__seed{int(seed)}"
    directory = out_dir / "fits" / variant_id / cell_id
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def _valid_existing(
    array_path: Path,
    metadata_path: Path,
    *,
    definition_hash: str,
    variant_id: str,
    cell_id: str,
    seed: int,
    bundle_audit: Mapping[str, Any],
) -> bool:
    if not array_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = _verified_content_json(
            metadata_path, expected_schema="deem_b3_pair_router_fit_artifact_v1"
        )
        return bool(
            metadata.get("status") == "complete"
            and metadata.get("run_definition_sha256") == definition_hash
            and metadata.get("variant_id") == variant_id
            and metadata.get("cell_id") == cell_id
            and int(metadata.get("seed", -1)) == int(seed)
            and metadata.get("bundle_sha256") == bundle_audit["bundle_sha256"]
            and metadata.get("inventory_sha256") == bundle_audit["inventory_sha256"]
            and metadata.get("ordered_row_id_sha256")
            == bundle_audit["ordered_row_id_sha256"]
            and metadata.get("health", {}).get("healthy")
            and metadata.get("targets_accessed_during_fit") is False
            and metadata.get("labels_module_imported") is False
            and sha256_file(array_path) == metadata.get("array_sha256")
        )
    except Exception:
        return False


def _fit_one(
    *,
    prepared: PreparedCell,
    baseline_dir: Path,
    baseline_artifact_map: Mapping[str, str],
    baseline_freeze_audit: Mapping[str, Any],
    out_dir: Path,
    variant: Mapping[str, Any],
    seed: int,
    definition_hash: str,
    experiment_id: str,
) -> dict:
    _assert_label_firewall()
    cell_id = str(prepared.bundle.cell_id)
    variant_id = str(variant["id"])
    array_path, metadata_path = output_paths(out_dir, variant_id, cell_id, seed)
    if _valid_existing(
        array_path,
        metadata_path,
        definition_hash=definition_hash,
        variant_id=variant_id,
        cell_id=cell_id,
        seed=seed,
        bundle_audit=prepared.bundle_audit,
    ):
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    if array_path.exists() or metadata_path.exists():
        raise ResidualGraphDeemError(
            f"refusing to overwrite incomplete/mismatched artifact: {metadata_path}"
        )

    _verify_baseline_member(baseline_dir, baseline_artifact_map, cell_id, seed)
    state, baseline_score, baseline_metadata, baseline_audit = load_frozen_b3(
        baseline_dir,
        cell_id,
        seed,
        prepared.X_risk,
        prepared.bundle.feature_names,
    )
    _audit_baseline_metadata(baseline_metadata, prepared.bundle, seed)
    settings = PairRouterConfig(**variant.get("config", {}))
    uses_graph = float(settings.graph_weight) > 0.0
    if uses_graph != (prepared.graph is not None):
        if uses_graph:
            raise ResidualGraphDeemError(f"missing fixed graph for {variant_id}/{cell_id}")
    laplacian = prepared.graph.laplacian if uses_graph else None
    result = fit_pair_residual_router(
        prepared.X_risk,
        prepared.bundle.feature_names,
        state,
        baseline_orientation=int(baseline_metadata["orientation"]),
        baseline_score=np.asarray(baseline_score, dtype=np.float64),
        seed=int(seed),
        config=settings,
        laplacian=laplacian,
    )
    _assert_label_firewall()
    if not result.health.get("healthy"):
        raise ResidualGraphDeemError(
            f"unhealthy pair-router fit: {variant_id}/{cell_id}/seed{seed}"
        )
    if result.orientation != int(baseline_metadata["orientation"]):
        raise ResidualGraphDeemError("pair router changed the frozen B3 orientation")
    if float(settings.rho) == 0.0 and not np.array_equal(result.score, baseline_score):
        raise ResidualGraphDeemError(
            f"exact B3 alias failed: {variant_id}/{cell_id}/seed{seed}"
        )
    if tuple(result.family_order) != tuple(
        name for name in FAMILY_ORDER if name in result.family_order
    ):
        raise ResidualGraphDeemError("pair-router family order is not canonical")
    family_reconstruction = float(
        np.max(
            np.abs(
                result.aligned_bias
                + result.routed_family_contributions.sum(axis=1)
                - result.logit
            )
        )
    )
    if family_reconstruction > 1e-8:
        raise ResidualGraphDeemError("routed-family reconstruction failed")

    graph_arrays = (
        dict(prepared.graph.arrays)
        if uses_graph and prepared.graph is not None
        else _empty_graph_arrays(len(prepared.bundle.row_ids), result.family_order)
    )
    arrays = {
        "score": np.asarray(result.score, dtype=np.float64),
        "posterior": np.column_stack([1.0 - result.score, result.score]).astype(np.float64),
        "logit": np.asarray(result.logit, dtype=np.float64),
        "contributions": np.asarray(result.contributions, dtype=np.float64),
        "base_family_contributions": np.asarray(
            result.base_family_contributions, dtype=np.float64
        ),
        "routed_family_contributions": np.asarray(
            result.routed_family_contributions, dtype=np.float64
        ),
        "gates": np.asarray(result.gates, dtype=np.float64),
        "family_probabilities": np.asarray(
            result.family_probabilities, dtype=np.float64
        ),
        "pair_transfers": np.asarray(result.pair_transfers, dtype=np.float64),
        "pair_open_probabilities": np.asarray(
            result.pair_open_probabilities, dtype=np.float64
        ),
        "pair_context_residuals": np.asarray(
            result.pair_context_residuals, dtype=np.float64
        ),
        "family_order": np.asarray(result.family_order, dtype=str),
        "pair_order_left": np.asarray([pair[0] for pair in result.pair_order], dtype=str),
        "pair_order_right": np.asarray([pair[1] for pair in result.pair_order], dtype=str),
        "feature_names": np.asarray(prepared.bundle.feature_names, dtype=str),
        "row_id": np.asarray(prepared.bundle.row_ids, dtype=str),
        "group_id": np.asarray(prepared.bundle.group_ids, dtype=str),
        "raw_trace_length": np.asarray(
            prepared.bundle.raw_trace_length, dtype=np.int64
        ),
        "standardization_mean": np.asarray(prepared.transform.mean, dtype=np.float64),
        "standardization_scale": np.asarray(prepared.transform.scale, dtype=np.float64),
        "standardization_constant_mask": np.asarray(
            prepared.transform.constant_mask, dtype=np.int8
        ),
        "baseline_score": np.asarray(baseline_score, dtype=np.float64),
        "baseline_orientation": np.asarray(result.orientation, dtype=np.int8),
        "baseline_aligned_bias": np.asarray(result.aligned_bias, dtype=np.float64),
        **graph_arrays,
    }
    for name, value in result.state.items():
        arrays[f"state__{name}"] = np.asarray(value)
    assert_no_target_fields(arrays)
    _assert_label_firewall()
    array_hash = atomic_save_npz(array_path, **arrays)
    graph_record = (
        prepared.graph.diagnostics
        if uses_graph and prepared.graph is not None
        else {
            "used": False,
            "reason": "variant_graph_weight_is_zero",
            "functional_inference_loo2": True,
            "end_to_end_strict_loo2": True,
            "uses_labels": False,
        }
    )
    record = {
        "schema": "deem_b3_pair_router_fit_artifact_v1",
        "status": "complete",
        "experiment_id": experiment_id,
        "variant_id": variant_id,
        "variant_role": str(variant.get("role", "")),
        "cell_id": cell_id,
        "dataset_family": prepared.bundle.dataset_family,
        "task_type": prepared.bundle.task_type,
        "seed": int(seed),
        "n_rows": len(prepared.bundle.row_ids),
        "n_features": len(prepared.bundle.feature_names),
        "bundle_sha256": prepared.bundle.bundle_sha256,
        "bundle_manifest_file_sha256": prepared.bundle_audit[
            "bundle_manifest_file_sha256"
        ],
        "inventory_sha256": prepared.bundle.inventory_sha256,
        "ordered_row_id_sha256": prepared.bundle_audit["ordered_row_id_sha256"],
        "source_sha256": prepared.bundle.source_sha256,
        "source_manifest_sha256": prepared.bundle.manifest_sha256,
        "admission_sha256": prepared.bundle.admission_sha256,
        "run_definition_sha256": definition_hash,
        "array_path": array_path.relative_to(out_dir).as_posix(),
        "array_sha256": array_hash,
        "config": jsonable(result.config),
        "config_sha256": canonical_sha256(result.config),
        "orientation": int(result.orientation),
        "aligned_bias": float(result.aligned_bias),
        "health": jsonable(result.health),
        "diagnostics": {
            **jsonable(result.diagnostics),
            "routed_family_reconstruction_max_abs": family_reconstruction,
        },
        "objective_history": jsonable(result.objective_history),
        "baseline": {
            **baseline_audit,
            "baseline_metadata_content_sha256": baseline_metadata["content_sha256"],
            "baseline_score_freeze_manifest_sha256": baseline_freeze_audit[
                "manifest_sha256"
            ],
            "baseline_score_freeze_content_sha256": baseline_freeze_audit[
                "manifest_content_sha256"
            ],
        },
        "graph": jsonable(graph_record),
        "control_reconstruction_support": {
            "mean_gate_control": "base_family_contributions_times_column_mean_gates",
            "static_gate_control": "base_family_contributions_times_saved_fixed_gate_vector",
            "row_permuted_gate_control": "base_family_contributions_times_row_permuted_saved_gates",
            "all_required_arrays_persisted": True,
        },
        "targets_accessed_during_fit": False,
        "labels_module_imported": False,
    }
    record["content_sha256"] = canonical_sha256(record)
    atomic_write_json(metadata_path, record)
    _assert_label_firewall()
    return record


def _write_verified_or_new(path: Path, value: Mapping[str, Any]) -> None:
    serializable = jsonable(value)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != serializable:
            raise ResidualGraphDeemError(f"existing immutable record mismatch: {path}")
    else:
        atomic_write_json(path, serializable)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variants", default="all", help="all or comma-separated IDs")
    parser.add_argument("--cells", default="screen", help="screen, all, or comma-separated IDs")
    parser.add_argument("--seeds", default="screen", help="screen, all, or comma-separated seeds")
    args = parser.parse_args()

    _assert_label_firewall()
    config = load_config(args.config)
    registry = load_registry(args.registry)
    variant_map = {str(row["id"]): row for row in config["variants"]}
    selected_variants = _parse_selection(args.variants, list(variant_map))
    all_cells = [str(row["cell_id"]) for row in registry["cells"]]
    selected_cells = _parse_selection(
        args.cells, all_cells, screen=[str(cell) for cell in config["screen_cells"]]
    )
    selected_seeds = _parse_seeds(args.seeds, config)
    baseline_manifest, baseline_artifact_map, baseline_freeze_audit = (
        _load_baseline_freeze(args.baseline_dir)
    )
    if not set(selected_cells).issubset(set(baseline_manifest["cells"])):
        raise ResidualGraphDeemError("selected cells are absent from the B3 freeze")
    required_baseline_seeds = set(selected_seeds)
    if any(
        float(PairRouterConfig(**variant_map[name]["config"]).graph_weight) > 0.0
        for name in selected_variants
    ):
        required_baseline_seeds.update(
            int(seed) for seed in config["baseline"]["residual_graph_ensemble_seeds"]
        )
    if not required_baseline_seeds.issubset(set(int(seed) for seed in baseline_manifest["seeds"])):
        raise ResidualGraphDeemError("required seeds are absent from the B3 freeze")

    sources = _source_manifest(args.config)
    source_hash = canonical_sha256(sources)
    definition = {
        "schema": "deem_b3_pair_router_run_definition_v1",
        "status": str(config["status"]),
        "experiment_id": str(config["experiment_id"]),
        "config_sha256": sha256_file(args.config),
        "registry_file_sha256": sha256_file(args.registry),
        "registry_content_sha256": registry["registry_content_sha256"],
        "source_manifest": sources,
        "source_sha256": source_hash,
        "baseline_score_freeze_manifest_sha256": baseline_freeze_audit[
            "manifest_sha256"
        ],
        "baseline_score_freeze_content_sha256": baseline_freeze_audit[
            "manifest_content_sha256"
        ],
        "baseline_score_freeze_run_definition_sha256": baseline_freeze_audit[
            "run_definition_sha256"
        ],
        "environment": environment_fingerprint(),
        "scientific_boundary": config["scientific_boundary"],
        "frozen_screen_rule": config["frozen_screen_rule"],
        "targets_accessed_during_fit": False,
        "labels_module_imported": False,
    }
    definition["content_sha256"] = canonical_sha256(definition)
    definition_hash = str(definition["content_sha256"])
    definition_path = args.out_dir / "run_definitions" / f"{definition_hash}.json"
    _write_verified_or_new(definition_path, definition)

    graph_variant_ids = {
        name
        for name in selected_variants
        if float(PairRouterConfig(**variant_map[name]["config"]).graph_weight) > 0.0
    }
    need_any_graph = bool(graph_variant_ids)
    prepared: dict[str, PreparedCell] = {}
    for cell_id in selected_cells:
        print(
            f"[prepare] {cell_id} graph={'yes' if need_any_graph else 'no'}",
            flush=True,
        )
        prepared[cell_id] = _prepare_cell(
            bundle_dir=args.bundle_dir,
            baseline_dir=args.baseline_dir,
            registry=registry,
            artifact_map=baseline_artifact_map,
            cell_id=cell_id,
            need_graph=need_any_graph,
            graph_seeds=config["baseline"]["residual_graph_ensemble_seeds"],
            graph_k=int(config["graph"]["k"]),
        )
    _assert_label_firewall()

    records = []
    for variant_id in selected_variants:
        for cell_id in selected_cells:
            for seed in selected_seeds:
                print(f"[{variant_id}] {cell_id} seed={seed}", flush=True)
                records.append(
                    _fit_one(
                        prepared=prepared[cell_id],
                        baseline_dir=args.baseline_dir,
                        baseline_artifact_map=baseline_artifact_map,
                        baseline_freeze_audit=baseline_freeze_audit,
                        out_dir=args.out_dir,
                        variant=variant_map[variant_id],
                        seed=seed,
                        definition_hash=definition_hash,
                        experiment_id=str(config["experiment_id"]),
                    )
                )
    _assert_label_firewall()

    expected_keys = [
        f"{variant_id}|{cell_id}|{seed}"
        for variant_id in selected_variants
        for cell_id in selected_cells
        for seed in selected_seeds
    ]
    actual_keys = [
        f"{row['variant_id']}|{row['cell_id']}|{row['seed']}" for row in records
    ]
    if actual_keys != expected_keys or len(actual_keys) != len(set(actual_keys)):
        raise ResidualGraphDeemError("fit records do not match the exact Cartesian invocation")
    invocation = {
        "run_definition_sha256": definition_hash,
        "selected_variants": selected_variants,
        "selected_cells": selected_cells,
        "selected_seeds": selected_seeds,
    }
    invocation_hash = canonical_sha256(invocation)
    artifact_rows = []
    for row in records:
        metadata_path = output_paths(
            args.out_dir, row["variant_id"], row["cell_id"], int(row["seed"])
        )[1]
        artifact_rows.append(
            {
                "variant_id": row["variant_id"],
                "cell_id": row["cell_id"],
                "seed": int(row["seed"]),
                "array_path": row["array_path"],
                "array_sha256": row["array_sha256"],
                "metadata_path": metadata_path.relative_to(args.out_dir).as_posix(),
                "metadata_sha256": sha256_file(metadata_path),
                "bundle_sha256": row["bundle_sha256"],
                "ordered_row_id_sha256": row["ordered_row_id_sha256"],
            }
        )
    manifest = {
        "schema": "deem_b3_pair_router_fit_manifest_v1",
        "status": "complete",
        "run_definition_sha256": definition_hash,
        "invocation_sha256": invocation_hash,
        "selected_variants": selected_variants,
        "selected_cells": selected_cells,
        "selected_seeds": selected_seeds,
        "expected_cartesian_keys": expected_keys,
        "n_artifacts": len(artifact_rows),
        "artifacts": artifact_rows,
        "all_healthy": all(row.get("health", {}).get("healthy") for row in records),
        "targets_accessed_during_fit": False,
        "labels_module_imported": False,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    manifest_path = args.out_dir / "fit_manifests" / f"{invocation_hash}.json"
    _write_verified_or_new(manifest_path, manifest)
    completion = {
        "schema": "deem_b3_pair_router_fit_complete_v1",
        "status": "complete",
        "run_definition_sha256": definition_hash,
        "invocation_sha256": invocation_hash,
        "fit_manifest_path": manifest_path.relative_to(args.out_dir).as_posix(),
        "fit_manifest_sha256": sha256_file(manifest_path),
        "fit_manifest_content_sha256": manifest["content_sha256"],
        "n_records": len(records),
        "variants": selected_variants,
        "cells": selected_cells,
        "seeds": selected_seeds,
        "all_healthy": manifest["all_healthy"],
        "targets_accessed_during_fit": False,
        "labels_module_imported": False,
    }
    completion["content_sha256"] = canonical_sha256(completion)
    completion_path = args.out_dir / "fit_completions" / f"{invocation_hash}.json"
    _write_verified_or_new(completion_path, completion)
    _assert_label_firewall()
    print(
        f"complete: {len(records)} fit artifacts; definition={definition_hash}; "
        f"invocation={invocation_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
