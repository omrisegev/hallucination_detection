#!/usr/bin/env python3
"""Strict evaluation boundary for frozen local-descent PGRD scores.

Phase A validates and mechanically replays the complete label-free fit.  The
label-sidecar module is imported only after every hash, row binding, B3 member,
donor-family exclusion, graph, gate, control, and score reconstruction passes.
Phase B joins labels by row ID and reports the retrospective eight-family
screen.  The natural targets were previously opened, so all inferential
statistics remain exploratory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
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

from scripts.run_deem_b3_local_descent_pgrd_v1 import (  # noqa: E402
    EXPECTED_VARIANTS,
    load_config as load_runner_config,
)
from spectral_utils.deem_b3_local_descent_pgrd import (  # noqa: E402
    LocalDescentPGRDConfig,
    build_common_residual_graph,
    fit_leave_dataset_family_out_direction,
    score_local_descent_pgrd,
)
from spectral_utils.deem_b3_residual_moe import (  # noqa: E402
    FAMILY_ORDER,
    build_residual_cell,
    graph_roughness_moment,
    load_frozen_b3_ensemble,
)
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ResidualGraphDeemError,
    assign_grouped_length_folds,
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    BUNDLE_SCHEMA,
    assert_no_target_fields,
    load_registry,
    load_target_free_bundle,
    registry_cell,
)


DEFAULT_CONFIG = ROOT / "configs/deem_b3_local_descent_pgrd_v1.json"
DEFAULT_REGISTRY = ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"
DEFAULT_BUNDLE_DIR = ROOT / "local_cache/deem_b3_moe_v1/bundles"
DEFAULT_BASELINE_DIR = ROOT / "local_cache/deem_b3_moe_v1/b3_frozen"
DEFAULT_SIDECAR_DIR = ROOT / "local_cache/deem_b3_moe_v1/label_sidecars"
DEFAULT_RUN_DIR = ROOT / "local_cache/deem_b3_moe_v1/local_descent_pgrd_screen_v1"
DEFAULT_OUT_DIR = ROOT / "local_cache/deem_b3_moe_v1/local_descent_pgrd_screen_v1_eval"

LABEL_MODULE = "spectral_utils.residual_graph_deem_labels"
RUN_SCHEMA = "deem_b3_local_descent_pgrd_run_definition_v1"
FREEZE_SCHEMA = "deem_b3_local_descent_pgrd_score_freeze_v1"
FIT_SCHEMA = "deem_b3_local_descent_pgrd_fit_artifact_v1"
TOLERANCE = 5e-4

# This is the same five-clause exploratory survival rule used by the adjacent
# IU-PGRD and pair-router screens.  The local-descent fit config did not encode
# numeric thresholds, so the report marks this as a shared retrospective rule,
# not a new confirmatory preregistration.
SHARED_EXPLORATORY_RULE = {
    "equal_family_auroc_delta_min": 0.0025,
    "descriptive_family_bootstrap_lower_min": 0.0,
    "exact_family_signflip_one_sided_p_max": 0.05,
    "wins_plus_ties_min_of_8": 6,
    "worst_cell_delta_min": -0.02,
}

REQUIRED_ARRAYS = {
    "score",
    "posterior",
    "baseline_score",
    "baseline_seed_scores",
    "baseline_logit",
    "baseline_z",
    "logit",
    "correction_z",
    "local_gradient",
    "expert_terms",
    "activation",
    "local_gate_probabilities",
    "gate_probabilities",
    "raw_direction",
    "direction",
    "row_permutation",
    "alpha",
    "tau",
    "cross_term",
    "quadratic_term",
    "roughness_before",
    "roughness_after",
    "family_direction",
    "family_stability",
    "present_family_mask",
    "family_order",
    "loo_residuals",
    "loo_seed_instability",
    "loo_predictability",
    "grouped_folds",
    "row_id",
    "group_id",
    "raw_trace_length",
    "graph_coordinates",
    "graph_family_order",
    "graph_tie_keys",
    "pooled_direction",
    "pooled_stability",
    "donor_group_directions",
    "donor_group_presence",
    "moment_a0",
    "moment_c0",
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
    if not isinstance(expected, str) or canonical_sha256(payload) != expected:
        raise ResidualGraphDeemError(f"content hash mismatch: {context}")
    return expected


def _write_content_json(path: Path, payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    if "content_sha256" in value:
        raise ValueError("content_sha256 is reserved")
    value["content_sha256"] = canonical_sha256(value)
    return atomic_write_json(path, value)


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _fit_paths(run_dir: Path, variant: str, cell: str) -> tuple[Path, Path]:
    path = run_dir / "scores" / variant / cell / f"{variant}.npz"
    return path, path.with_suffix(".json")


def _safe_child(root: Path, relative: str) -> Path:
    base = root.resolve()
    path = (base / str(relative)).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ResidualGraphDeemError(f"artifact escaped run directory: {relative}") from exc
    return path


def _csr_payload(prefix: str, matrix: sparse.spmatrix) -> dict[str, np.ndarray]:
    value = sparse.csr_matrix(matrix, dtype=np.float64).copy()
    value.sort_indices()
    return {
        f"{prefix}_data": np.asarray(value.data, dtype=np.float64),
        f"{prefix}_indices": np.asarray(value.indices, dtype=np.int64),
        f"{prefix}_indptr": np.asarray(value.indptr, dtype=np.int64),
        f"{prefix}_shape": np.asarray(value.shape, dtype=np.int64),
    }


def _array_close(actual: np.ndarray, expected: np.ndarray, *, context: str) -> None:
    left = np.asarray(actual)
    right = np.asarray(expected)
    if left.shape != right.shape or left.dtype.kind == "O" or right.dtype.kind == "O":
        raise ResidualGraphDeemError(f"array shape/object mismatch: {context}")
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        if not np.allclose(left, right, rtol=2e-12, atol=2e-12, equal_nan=False):
            error = float(np.max(np.abs(left.astype(float) - right.astype(float))))
            raise ResidualGraphDeemError(
                f"numeric replay mismatch ({error:.3e}): {context}"
            )
    elif not np.array_equal(left, right):
        raise ResidualGraphDeemError(f"exact array replay mismatch: {context}")


def _validate_bundle(bundle: Any, registered: Mapping[str, Any], path: Path) -> dict:
    sidecar_path = path.with_suffix(".manifest.json")
    sidecar = _read_json(sidecar_path)
    source = registered["source"]
    row_hash = canonical_sha256(list(bundle.row_ids))
    checks = {
        "cell": bundle.cell_id == str(registered["cell_id"]),
        "family": bundle.dataset_family == str(registered["dataset_family"]),
        "task": bundle.task_type == str(registered["task_type"]),
        "rows": len(bundle.row_ids) == int(registered["n_rows"]),
        "features": tuple(bundle.feature_names) == tuple(registered["feature_names"]),
        "inventory": bundle.inventory_sha256 == str(registered["inventory_sha256"]),
        "source": bundle.source_sha256 == str(source["source_sha256"]),
        "source manifest": bundle.manifest_sha256 == str(source["manifest_sha256"]),
        "admission": bundle.admission_sha256
        == str(source["admission_contract_sha256"]),
        "sidecar schema": sidecar.get("schema") == BUNDLE_SCHEMA,
        "sidecar cell": sidecar.get("cell_id") == bundle.cell_id,
        "sidecar bundle": sidecar.get("bundle_sha256") == bundle.bundle_sha256,
        "sidecar rows": int(sidecar.get("n_rows", -1)) == len(bundle.row_ids),
        "sidecar row order": sidecar.get("ordered_row_id_sha256") == row_hash,
        "sidecar inventory": sidecar.get("inventory_sha256")
        == bundle.inventory_sha256,
        "sidecar source": sidecar.get("source_sha256") == bundle.source_sha256,
        "sidecar source manifest": sidecar.get("manifest_sha256")
        == bundle.manifest_sha256,
        "sidecar admission": sidecar.get("admission_sha256")
        == bundle.admission_sha256,
        "sidecar labels": sidecar.get("labels_accessed") is False,
        "sidecar pickle": sidecar.get("allow_pickle") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ResidualGraphDeemError(
            f"bundle/registry binding failed for {bundle.cell_id}: {failed}"
        )
    return {
        "bundle_sha256": bundle.bundle_sha256,
        "bundle_manifest_file_sha256": sha256_file(sidecar_path),
        "ordered_row_id_sha256": row_hash,
        "inventory_sha256": bundle.inventory_sha256,
        "source_sha256": bundle.source_sha256,
        "source_manifest_sha256": bundle.manifest_sha256,
        "admission_sha256": bundle.admission_sha256,
    }


def _load_run_contract(
    *,
    config_path: Path,
    registry_path: Path,
    baseline_dir: Path,
    run_dir: Path,
) -> tuple[dict, dict, dict, dict[tuple[str, str], dict]]:
    config = load_runner_config(config_path)
    registry = load_registry(registry_path)
    definition = _read_json(run_dir / "RUN_DEFINITION.json")
    freeze = _read_json(run_dir / "SCORE_FREEZE_MANIFEST.json")
    freeze_content = _verify_content_hash(freeze, context="score freeze")

    if definition.get("schema") != RUN_SCHEMA or definition.get("status") != "complete":
        raise ResidualGraphDeemError("local-descent run definition is incomplete")
    definition_body = dict(definition)
    for field in (
        "definition_sha256",
        "status",
        "n_score_artifacts",
        "score_freeze_content_sha256",
    ):
        definition_body.pop(field, None)
    definition_sha = str(definition.get("definition_sha256", ""))
    if canonical_sha256(definition_body) != definition_sha:
        raise ResidualGraphDeemError("run definition body hash mismatch")

    screen_cells = [str(value) for value in config["screen_cells"]]
    variants = [str(row["id"]) for row in config["variants"]]
    registry_cells = [str(row["cell_id"]) for row in registry["cells"]]
    if (
        definition.get("experiment_id") != config["experiment_id"]
        or definition.get("config_sha256") != sha256_file(config_path)
        or definition.get("registry_sha256") != sha256_file(registry_path)
        or definition.get("score_cells") != screen_cells
        or definition.get("variants") != variants
        or definition.get("seeds") != [0, 1, 2, 3, 4]
        or definition.get("donor_cells") != registry_cells
        or int(definition.get("n_score_artifacts", -1))
        != len(screen_cells) * len(variants)
        or definition.get("score_freeze_content_sha256") != freeze_content
        or definition.get("targets_accessed_during_fit") is not False
        or definition.get("labels_module_imported") is not False
        or definition.get("fit_only_no_evaluation") is not True
        or definition.get("transductive_target_cell_geometry") is not True
    ):
        raise ResidualGraphDeemError("run definition frozen contract mismatch")

    source_manifest = definition.get("source_manifest")
    if not isinstance(source_manifest, dict) or not source_manifest:
        raise ResidualGraphDeemError("run source manifest is missing")
    for relative, expected in source_manifest.items():
        path = Path(relative)
        path = path if path.is_absolute() else ROOT / path
        if not path.is_file() or sha256_file(path) != expected:
            raise ResidualGraphDeemError(f"run source dependency changed: {relative}")
    if canonical_sha256(source_manifest) != definition.get("source_sha256"):
        raise ResidualGraphDeemError("run source-manifest aggregate hash mismatch")

    baseline_freeze = baseline_dir / "SCORE_FREEZE_MANIFEST.json"
    baseline_audit = definition.get("baseline_freeze", {})
    if (
        not baseline_freeze.is_file()
        or sha256_file(baseline_freeze)
        != baseline_audit.get("manifest_file_sha256")
    ):
        raise ResidualGraphDeemError("frozen B3 manifest file binding mismatch")
    baseline_manifest = _read_json(baseline_freeze)
    baseline_content = _verify_content_hash(
        baseline_manifest, context="frozen B3 score manifest"
    )
    if (
        baseline_manifest.get("schema") != "deem_vs_iupcr_score_freeze_v1"
        or baseline_manifest.get("status") != "complete"
        or bool(baseline_manifest.get("debug"))
        or "B3" not in baseline_manifest.get("arms", [])
        or baseline_content != baseline_audit.get("manifest_content_sha256")
        or baseline_manifest.get("run_definition_sha256")
        != baseline_audit.get("baseline_run_definition_sha256")
    ):
        raise ResidualGraphDeemError("frozen B3 manifest contract mismatch")

    if (
        freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("status") != "complete"
        or freeze.get("experiment_id") != config["experiment_id"]
        or freeze.get("definition_sha256") != definition_sha
        or freeze.get("cells") != screen_cells
        or freeze.get("variants") != variants
        or int(freeze.get("n_artifacts", -1)) != len(screen_cells) * len(variants)
        or freeze.get("targets_accessed_during_fit") is not False
        or freeze.get("labels_module_imported") is not False
        or freeze.get("fit_only_no_evaluation") is not True
    ):
        raise ResidualGraphDeemError("score freeze contract mismatch")
    rows = freeze.get("artifacts")
    if not isinstance(rows, list):
        raise ResidualGraphDeemError("score freeze artifact rows are missing")
    keys = [(str(row.get("variant_id")), str(row.get("cell_id"))) for row in rows]
    expected_keys = [(variant, cell) for cell in screen_cells for variant in variants]
    if keys != expected_keys or len(keys) != len(set(keys)):
        raise ResidualGraphDeemError("score freeze is not the exact Cartesian roster")
    lookup = {}
    for row in rows:
        variant, cell = str(row["variant_id"]), str(row["cell_id"])
        array_path, metadata_path = _fit_paths(run_dir, variant, cell)
        if (
            str(row.get("array_path")) != array_path.relative_to(run_dir).as_posix()
            or str(row.get("metadata_path"))
            != metadata_path.relative_to(run_dir).as_posix()
            or not array_path.is_file()
            or not metadata_path.is_file()
            or sha256_file(array_path) != row.get("array_sha256")
            or sha256_file(metadata_path) != row.get("metadata_sha256")
        ):
            raise ResidualGraphDeemError(f"score-freeze artifact mismatch: {variant}/{cell}")
        _safe_child(run_dir, str(row["array_path"]))
        _safe_child(run_dir, str(row["metadata_path"]))
        lookup[(variant, cell)] = row
    return config, registry, definition, lookup


def _load_and_rebuild_label_free(
    *,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    definition: Mapping[str, Any],
    bundle_dir: Path,
    baseline_dir: Path,
) -> dict:
    artifact_rows = _read_json(baseline_dir / "SCORE_FREEZE_MANIFEST.json")[
        "artifacts"
    ]
    baseline_artifacts = {
        str(row["path"]): str(row["sha256"]) for row in artifact_rows
    }
    if len(baseline_artifacts) != len(artifact_rows):
        raise ResidualGraphDeemError("frozen B3 manifest has duplicate paths")

    bundles: dict[str, Any] = {}
    bundle_audit: dict[str, dict] = {}
    ensembles: dict[str, Any] = {}
    residuals: dict[str, Any] = {}
    moments: dict[str, Any] = {}
    folds: dict[str, np.ndarray] = {}
    family_by_cell: dict[str, str] = {}
    donor_cells = [str(value) for value in definition["donor_cells"]]
    run_bindings = definition.get("bundle_bindings", {})
    for cell in donor_cells:
        path = bundle_dir / f"{cell}.npz"
        bundle = load_target_free_bundle(path)
        audit = _validate_bundle(bundle, registry_cell(registry, cell), path)
        if run_bindings.get(cell) != {
            **audit,
            "bundle_path": run_bindings.get(cell, {}).get("bundle_path"),
            "bundle_manifest_path": run_bindings.get(cell, {}).get(
                "bundle_manifest_path"
            ),
        }:
            # Absolute paths may differ after a content-preserving move; all
            # cryptographic and semantic fields must still match exactly.
            for field, wanted in audit.items():
                if run_bindings.get(cell, {}).get(field) != wanted:
                    raise ResidualGraphDeemError(
                        f"run/bundle binding mismatch: {cell}/{field}"
                    )
        for seed in (0, 1, 2, 3, 4):
            for suffix in ("npz", "json"):
                relative = f"fits/{cell}/B3__seed{seed}.{suffix}"
                candidate = baseline_dir / relative
                if (
                    relative not in baseline_artifacts
                    or not candidate.is_file()
                    or sha256_file(candidate) != baseline_artifacts[relative]
                ):
                    raise ResidualGraphDeemError(
                        f"unbound frozen B3 member: {cell}/seed{seed}/{suffix}"
                    )
        ensemble = load_frozen_b3_ensemble(
            baseline_dir,
            cell,
            seeds=(0, 1, 2, 3, 4),
            expected_bundle_sha256=bundle.bundle_sha256,
            expected_ordered_row_id_sha256=audit["ordered_row_id_sha256"],
        )
        if not np.array_equal(ensemble.score, np.mean(ensemble.seed_scores, axis=0)):
            raise ResidualGraphDeemError(f"mean-five B3 identity failed: {cell}")
        grouped_folds = assign_grouped_length_folds(
            bundle.group_ids, bundle.raw_trace_length
        )
        residual = build_residual_cell(
            ensemble, baseline_score=ensemble.score, folds=grouped_folds
        )
        moment = graph_roughness_moment(
            residual, bundle.row_ids, residual_source="loo", k=7
        )
        bundles[cell] = bundle
        bundle_audit[cell] = audit
        ensembles[cell] = ensemble
        residuals[cell] = residual
        moments[cell] = moment
        folds[cell] = grouped_folds
        family_by_cell[cell] = bundle.dataset_family

    calibrations = {}
    for held_family in sorted(
        {family_by_cell[cell] for cell in config["screen_cells"]}
    ):
        calibration = fit_leave_dataset_family_out_direction(
            [moments[cell] for cell in donor_cells],
            [family_by_cell[cell] for cell in donor_cells],
            target_dataset_family=held_family,
        )
        if (
            calibration.target_dataset_family != held_family
            or held_family in calibration.donor_dataset_families
        ):
            raise ResidualGraphDeemError(
                f"whole-dataset-family calibration exclusion failed: {held_family}"
            )
        calibrations[held_family] = calibration
    graphs = {
        cell: build_common_residual_graph(
            residuals[cell], bundles[cell].row_ids, k=7
        )
        for cell in config["screen_cells"]
    }
    return {
        "bundles": bundles,
        "bundle_audit": bundle_audit,
        "ensembles": ensembles,
        "residuals": residuals,
        "moments": moments,
        "folds": folds,
        "family_by_cell": family_by_cell,
        "calibrations": calibrations,
        "graphs": graphs,
    }


def _validate_fit_metadata(
    metadata: Mapping[str, Any],
    *,
    variant: str,
    variant_row: Mapping[str, Any],
    cell: str,
    bundle: Any,
    bundle_audit: Mapping[str, Any],
    definition: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    array_path: Path,
    metadata_path: Path,
    calibration: Any,
) -> None:
    _verify_content_hash(metadata, context=f"fit metadata {variant}/{cell}")
    checks = {
        "schema": metadata.get("schema") == FIT_SCHEMA,
        "status": metadata.get("status") == "complete",
        "experiment": metadata.get("experiment_id") == definition["experiment_id"],
        "definition": metadata.get("definition_sha256")
        == definition["definition_sha256"],
        "source": metadata.get("source_sha256") == definition["source_sha256"],
        "variant": metadata.get("variant_id") == variant,
        "variant config": metadata.get("variant_config") == variant_row["config"],
        "cell": metadata.get("cell_id") == cell,
        "family": metadata.get("dataset_family") == bundle.dataset_family,
        "task": metadata.get("task_type") == bundle.task_type,
        "rows": int(metadata.get("n_rows", -1)) == len(bundle.row_ids),
        "features": int(metadata.get("n_features", -1)) == len(bundle.feature_names),
        "bundle": metadata.get("bundle_sha256") == bundle.bundle_sha256,
        "bundle manifest": metadata.get("bundle_manifest_file_sha256")
        == bundle_audit["bundle_manifest_file_sha256"],
        "inventory": metadata.get("inventory_sha256") == bundle.inventory_sha256,
        "row order": metadata.get("ordered_row_id_sha256")
        == bundle_audit["ordered_row_id_sha256"],
        "baseline seeds": metadata.get("baseline_seeds") == [0, 1, 2, 3, 4],
        "baseline ensemble": metadata.get("baseline_score_ensemble")
        == "exact_mean_of_seed_posteriors",
        "calibration held family": metadata.get("calibration_held_dataset_family")
        == calibration.target_dataset_family,
        "calibration donor families": metadata.get(
            "calibration_donor_dataset_families"
        )
        == list(calibration.donor_dataset_families),
        "calibration exclusion": metadata.get(
            "calibration_excludes_entire_held_dataset_family"
        )
        is True,
        "graph binding": metadata.get("graph_binding_sha256")
        == metadata.get("graph_diagnostics", {}).get("binding_sha256"),
        "healthy": metadata.get("health", {}).get("healthy") is True,
        "array hash": metadata.get("array_sha256") == sha256_file(array_path),
        "manifest array hash": metadata.get("array_sha256")
        == manifest_row.get("array_sha256"),
        "manifest metadata hash": sha256_file(metadata_path)
        == manifest_row.get("metadata_sha256"),
        "no targets": metadata.get("targets_accessed_during_fit") is False,
        "no labels": metadata.get("labels_module_imported") is False,
        "fit only": metadata.get("fit_only_no_evaluation") is True,
        "transductive": metadata.get("target_cell_geometry_is_transductive")
        is True,
        "no self-free claim": metadata.get("per_row_self_free_inference_claimed")
        is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ResidualGraphDeemError(
            f"fit metadata contract failed for {variant}/{cell}: {failed}"
        )


def _validate_and_replay_scores(
    *,
    config: Mapping[str, Any],
    definition: Mapping[str, Any],
    manifest: Mapping[tuple[str, str], Mapping[str, Any]],
    rebuilt: Mapping[str, Any],
    run_dir: Path,
) -> tuple[dict[tuple[str, str], dict[str, np.ndarray]], list[dict]]:
    variant_lookup = {str(row["id"]): row for row in config["variants"]}
    arrays_by_key: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    diagnostics = []
    for cell in config["screen_cells"]:
        bundle = rebuilt["bundles"][cell]
        ensemble = rebuilt["ensembles"][cell]
        residual = rebuilt["residuals"][cell]
        moment = rebuilt["moments"][cell]
        graph = rebuilt["graphs"][cell]
        calibration = rebuilt["calibrations"][bundle.dataset_family]
        for variant in EXPECTED_VARIANTS:
            variant_row = variant_lookup[variant]
            array_path, metadata_path = _fit_paths(run_dir, variant, cell)
            metadata = _read_json(metadata_path)
            _validate_fit_metadata(
                metadata,
                variant=variant,
                variant_row=variant_row,
                cell=cell,
                bundle=bundle,
                bundle_audit=rebuilt["bundle_audit"][cell],
                definition=definition,
                manifest_row=manifest[(variant, cell)],
                array_path=array_path,
                metadata_path=metadata_path,
                calibration=calibration,
            )
            with np.load(array_path, allow_pickle=False) as data:
                if set(data.files) != REQUIRED_ARRAYS:
                    missing = sorted(REQUIRED_ARRAYS - set(data.files))
                    extra = sorted(set(data.files) - REQUIRED_ARRAYS)
                    raise ResidualGraphDeemError(
                        f"fit array schema mismatch {variant}/{cell}: "
                        f"missing={missing}, extra={extra}"
                    )
                arrays = {name: np.asarray(data[name]) for name in data.files}
            if any(value.dtype.hasobject for value in arrays.values()):
                raise ResidualGraphDeemError(f"object array found: {variant}/{cell}")
            assert_no_target_fields(arrays)

            result = score_local_descent_pgrd(
                residual,
                calibration,
                graph,
                config=LocalDescentPGRDConfig(**variant_row["config"]),
            )
            expected = {
                "score": result.score,
                "posterior": np.column_stack([1.0 - result.score, result.score]),
                "baseline_score": ensemble.score,
                "baseline_seed_scores": ensemble.seed_scores,
                "baseline_logit": residual.baseline_logit,
                "baseline_z": result.baseline_z,
                "logit": result.logit,
                "correction_z": result.correction_z,
                "local_gradient": result.local_gradient,
                "expert_terms": result.expert_terms,
                "activation": result.activation,
                "local_gate_probabilities": result.local_gate_probabilities,
                "gate_probabilities": result.gate_probabilities,
                "raw_direction": result.raw_direction,
                "direction": result.direction,
                "row_permutation": result.row_permutation,
                "alpha": np.asarray(result.alpha),
                "tau": np.asarray(result.tau),
                "cross_term": np.asarray(result.cross_term),
                "quadratic_term": np.asarray(result.quadratic_term),
                "roughness_before": np.asarray(result.roughness_before),
                "roughness_after": np.asarray(result.roughness_after),
                "family_direction": result.family_direction,
                "family_stability": result.family_stability,
                "present_family_mask": result.present_mask.astype(np.int8),
                "family_order": np.asarray(result.family_order, dtype=str),
                "loo_residuals": residual.loo_residuals,
                "loo_seed_instability": residual.loo_seed_instability,
                "loo_predictability": residual.loo_predictability,
                "grouped_folds": rebuilt["folds"][cell],
                "row_id": np.asarray(bundle.row_ids, dtype=str),
                "group_id": np.asarray(bundle.group_ids, dtype=str),
                "raw_trace_length": bundle.raw_trace_length,
                "graph_coordinates": graph.coordinates,
                "graph_family_order": np.asarray(graph.family_order, dtype=str),
                "graph_tie_keys": graph.tie_keys,
                "pooled_direction": calibration.direction,
                "pooled_stability": calibration.stability,
                "donor_group_directions": calibration.donor_group_directions,
                "donor_group_presence": calibration.donor_group_presence.astype(np.int8),
                "moment_a0": moment.a0,
                "moment_c0": moment.c0,
                **_csr_payload("graph", graph.graph),
                **_csr_payload("laplacian", graph.laplacian),
            }
            for name, wanted in expected.items():
                _array_close(arrays[name], np.asarray(wanted), context=f"{variant}/{cell}/{name}")
            if not np.array_equal(arrays["baseline_score"], ensemble.score):
                raise ResidualGraphDeemError(f"B3 baseline bytes changed: {variant}/{cell}")
            if variant == "L0_B3_EXACT_ALIAS" and not np.array_equal(
                arrays["score"], ensemble.score
            ):
                raise ResidualGraphDeemError(f"L0 is not an exact B3 alias: {cell}")
            if metadata["graph_binding_sha256"] != graph.binding_sha256:
                raise ResidualGraphDeemError(f"graph binding replay failed: {variant}/{cell}")
            alpha = float(arrays["alpha"])
            tau = float(arrays["tau"])
            gate_mass = np.sum(arrays["gate_probabilities"], axis=1)
            local_mass = np.sum(arrays["local_gate_probabilities"], axis=1)
            diagnostics.append(
                {
                    "cell_id": cell,
                    "dataset_family": bundle.dataset_family,
                    "task_type": bundle.task_type,
                    "variant": variant,
                    "alpha": alpha,
                    "tau": tau,
                    "alpha_at_cap": bool(alpha > 0.0 and abs(alpha - tau) <= 1e-12),
                    "correction_z_sd": float(np.std(arrays["correction_z"])),
                    "roughness_delta": float(
                        arrays["roughness_after"] - arrays["roughness_before"]
                    ),
                    "local_active_row_fraction": float(np.mean(local_mass > 0.0)),
                    "applied_active_row_fraction": float(np.mean(gate_mass > 0.0)),
                    "gate_zero_fraction_present": float(
                        np.mean(
                            arrays["gate_probabilities"][:,
                            arrays["present_family_mask"].astype(bool)]
                            == 0.0
                        )
                    ),
                    "gate_row_sum_min": float(np.min(gate_mass)),
                    "gate_row_sum_max": float(np.max(gate_mass)),
                    "cross_term": float(arrays["cross_term"]),
                    "quadratic_term": float(arrays["quadratic_term"]),
                    "graph_binding_sha256": graph.binding_sha256,
                }
            )
            arrays_by_key[(variant, cell)] = arrays

        primary = arrays_by_key[("L1_LOCAL_DESCENT_PRIMARY", cell)]
        static = arrays_by_key[("L2_STATIC_GATE_CONTROL", cell)]
        permuted = arrays_by_key[("L3_ROW_PERMUTED_GATE_CONTROL", cell)]
        expected_static = np.repeat(
            np.mean(primary["local_gate_probabilities"], axis=0)[None, :],
            len(bundle.row_ids),
            axis=0,
        )
        if not np.array_equal(static["gate_probabilities"], expected_static):
            raise ResidualGraphDeemError(f"static gate is not mean matched: {cell}")
        if not np.array_equal(
            permuted["gate_probabilities"],
            primary["local_gate_probabilities"][permuted["row_permutation"]],
        ):
            raise ResidualGraphDeemError(f"row-permuted gate is not matched: {cell}")
        for variant in EXPECTED_VARIANTS:
            arrays = arrays_by_key[(variant, cell)]
            for shared in (
                "baseline_score",
                "baseline_seed_scores",
                "baseline_logit",
                "baseline_z",
                "loo_residuals",
                "local_gradient",
                "expert_terms",
                "activation",
                "local_gate_probabilities",
                "graph_data",
                "graph_indices",
                "graph_indptr",
                "laplacian_data",
                "laplacian_indices",
                "laplacian_indptr",
            ):
                if not np.array_equal(arrays[shared], primary[shared]):
                    raise ResidualGraphDeemError(
                        f"variant paths do not share frozen base/graph: {cell}/{shared}"
                    )
    return arrays_by_key, diagnostics


def _preflight(
    *,
    config_path: Path,
    registry_path: Path,
    bundle_dir: Path,
    baseline_dir: Path,
    run_dir: Path,
) -> dict:
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module imported before local-descent preflight")
    config, registry, definition, manifest = _load_run_contract(
        config_path=config_path,
        registry_path=registry_path,
        baseline_dir=baseline_dir,
        run_dir=run_dir,
    )
    rebuilt = _load_and_rebuild_label_free(
        config=config,
        registry=registry,
        definition=definition,
        bundle_dir=bundle_dir,
        baseline_dir=baseline_dir,
    )
    arrays, diagnostics = _validate_and_replay_scores(
        config=config,
        definition=definition,
        manifest=manifest,
        rebuilt=rebuilt,
        run_dir=run_dir,
    )
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module entered the preflight import closure")
    return {
        "config": config,
        "registry": registry,
        "definition": definition,
        "manifest": manifest,
        "rebuilt": rebuilt,
        "arrays": arrays,
        "router_diagnostics": diagnostics,
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "registry_sha256": sha256_file(registry_path),
            "run_definition_sha256": sha256_file(run_dir / "RUN_DEFINITION.json"),
            "run_definition_content_sha256": definition["definition_sha256"],
            "score_freeze_manifest_sha256": sha256_file(
                run_dir / "SCORE_FREEZE_MANIFEST.json"
            ),
            "score_freeze_content_sha256": definition[
                "score_freeze_content_sha256"
            ],
            "baseline_score_freeze_manifest_sha256": sha256_file(
                baseline_dir / "SCORE_FREEZE_MANIFEST.json"
            ),
            "fit_source_sha256": definition["source_sha256"],
            "evaluator_sha256": sha256_file(Path(__file__)),
        },
    }


def _load_targets_after_preflight(preflight: Mapping[str, Any], sidecar_dir: Path):
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module imported before explicit label phase")
    from spectral_utils.residual_graph_deem_labels import (  # noqa: PLC0415
        SIDECAR_SCHEMA,
        join_labels_by_id,
        load_label_sidecar,
    )

    targets = {}
    audit = {}
    for cell in preflight["config"]["screen_cells"]:
        bundle = preflight["rebuilt"]["bundles"][cell]
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
            raise ResidualGraphDeemError(f"single-class target: {cell}")
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
        "n_dataset_families": len(family_means),
        "cell_macro": float(np.mean([float(row[metric]) for row in selected])),
        "equal_family_macro": float(np.mean(list(family_means.values()))),
        "worst_cell": float(min(float(row[metric]) for row in selected)),
        "family_means": family_means,
    }


def _comparison(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    reference: str,
    *,
    draws: int,
    seed: int,
) -> dict:
    if int(draws) < 100:
        raise ValueError("bootstrap draws must be at least 100")
    lookup = {(str(row["cell_id"]), str(row["method"])): row for row in rows}
    cells = sorted({str(row["cell_id"]) for row in rows})
    by_family_auroc = defaultdict(list)
    by_family_auprc = defaultdict(list)
    cell_delta = {}
    for cell in cells:
        candidate_row = lookup[(cell, candidate)]
        reference_row = lookup[(cell, reference)]
        family = str(reference_row["dataset_family"])
        delta = float(candidate_row["auroc"] - reference_row["auroc"])
        by_family_auroc[family].append(delta)
        by_family_auprc[family].append(
            float(candidate_row["auprc"] - reference_row["auprc"])
        )
        cell_delta[cell] = delta
    family_delta = {
        family: float(np.mean(values))
        for family, values in sorted(by_family_auroc.items())
    }
    family_auprc_delta = {
        family: float(np.mean(values))
        for family, values in sorted(by_family_auprc.items())
    }
    values = np.asarray(list(family_delta.values()), dtype=np.float64)
    observed = float(np.mean(values))
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    distribution = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        selected = rng.integers(0, len(values), size=len(values))
        distribution[draw] = float(np.mean(values[selected]))
    if len(values) > 20:
        raise ResidualGraphDeemError("too many families for exact sign-flip test")
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = np.mean(signs * values[None, :], axis=1)
    return {
        "candidate": candidate,
        "reference": reference,
        "equal_family_auroc_delta": observed,
        "equal_family_auprc_delta": float(np.mean(list(family_auprc_delta.values()))),
        "descriptive_family_bootstrap_lower": float(
            np.quantile(distribution, 0.025)
        ),
        "descriptive_family_bootstrap_upper": float(
            np.quantile(distribution, 0.975)
        ),
        "exact_family_signflip_one_sided_p": float(
            np.mean(null >= observed - 1e-15)
        ),
        "exact_family_signflip_assignments": int(2 ** len(values)),
        "wins": int(sum(value > TOLERANCE for value in cell_delta.values())),
        "ties": int(sum(abs(value) <= TOLERANCE for value in cell_delta.values())),
        "losses": int(sum(value < -TOLERANCE for value in cell_delta.values())),
        "worst_cell_delta": float(min(cell_delta.values())),
        "family_delta": family_delta,
        "family_auprc_delta": family_auprc_delta,
        "cell_delta": cell_delta,
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "bootstrap_is_descriptive_not_a_null_test": True,
    }


def _verdict(comparisons: Sequence[Mapping[str, Any]]) -> dict:
    lookup = {
        (str(row["candidate"]), str(row["reference"])): row
        for row in comparisons
    }
    primary = lookup[("L1_LOCAL_DESCENT_PRIMARY", "B3")]
    versus_static = lookup[
        ("L1_LOCAL_DESCENT_PRIMARY", "L2_STATIC_GATE_CONTROL")
    ]
    versus_permuted = lookup[
        ("L1_LOCAL_DESCENT_PRIMARY", "L3_ROW_PERMUTED_GATE_CONTROL")
    ]
    thresholds = SHARED_EXPLORATORY_RULE
    clauses = [
        {
            "name": "equal_family_auroc_delta_min",
            "observed": primary["equal_family_auroc_delta"],
            "threshold": thresholds["equal_family_auroc_delta_min"],
            "passed": primary["equal_family_auroc_delta"]
            >= thresholds["equal_family_auroc_delta_min"],
        },
        {
            "name": "descriptive_family_bootstrap_lower_min",
            "observed": primary["descriptive_family_bootstrap_lower"],
            "threshold": thresholds["descriptive_family_bootstrap_lower_min"],
            "passed": primary["descriptive_family_bootstrap_lower"]
            >= thresholds["descriptive_family_bootstrap_lower_min"],
        },
        {
            "name": "exact_family_signflip_one_sided_p_max",
            "observed": primary["exact_family_signflip_one_sided_p"],
            "threshold": thresholds["exact_family_signflip_one_sided_p_max"],
            "passed": primary["exact_family_signflip_one_sided_p"]
            <= thresholds["exact_family_signflip_one_sided_p_max"],
        },
        {
            "name": "wins_plus_ties_min_of_8",
            "observed": int(primary["wins"] + primary["ties"]),
            "threshold": thresholds["wins_plus_ties_min_of_8"],
            "passed": int(primary["wins"] + primary["ties"])
            >= thresholds["wins_plus_ties_min_of_8"],
        },
        {
            "name": "worst_cell_delta_min",
            "observed": primary["worst_cell_delta"],
            "threshold": thresholds["worst_cell_delta_min"],
            "passed": primary["worst_cell_delta"]
            >= thresholds["worst_cell_delta_min"],
        },
        {
            "name": "local_gate_beats_static_gate_equal_family_point_estimate",
            "observed": versus_static["equal_family_auroc_delta"],
            "threshold": 0.0,
            "passed": versus_static["equal_family_auroc_delta"] > 0.0,
        },
        {
            "name": "local_gate_beats_row_permuted_gate_equal_family_point_estimate",
            "observed": versus_permuted["equal_family_auroc_delta"],
            "threshold": 0.0,
            "passed": versus_permuted["equal_family_auroc_delta"] > 0.0,
        },
    ]
    passed = all(bool(clause["passed"]) for clause in clauses)
    return {
        "status": "retrospective_shared_rule_eight_family_screen",
        "verdict": "PASS" if passed else "FAIL",
        "mechanistic_primary": "L1_LOCAL_DESCENT_PRIMARY",
        "n_cells": 8,
        "n_dataset_families": 8,
        "clauses": clauses,
        "rule_provenance": (
            "the five baseline clauses are the common adjacent-screen rule; "
            "the local fit config did not encode numeric thresholds, and the two "
            "control-specificity clauses are descriptive mechanism requirements"
        ),
        "not_confirmatory": True,
        "secondary_or_control_variants_cannot_substitute_for_primary": True,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--sidecar-dir", type=Path, default=DEFAULT_SIDECAR_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap-draws", type=int, default=9999)
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"evaluation output directory must be empty: {args.out_dir}")
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module imported before evaluation preflight")

    # Phase A: complete target-free replay and cryptographic preflight.
    preflight = _preflight(
        config_path=args.config,
        registry_path=args.registry,
        bundle_dir=args.bundle_dir,
        baseline_dir=args.baseline_dir,
        run_dir=args.run_dir,
    )
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module imported before preflight completed")

    # Phase B: labels are imported and joined only after Phase A succeeds.
    targets, sidecar_audit = _load_targets_after_preflight(
        preflight, args.sidecar_dir
    )
    config = preflight["config"]
    cells = [str(value) for value in config["screen_cells"]]
    variants = list(EXPECTED_VARIANTS)
    per_cell = []
    for cell in cells:
        bundle = preflight["rebuilt"]["bundles"][cell]
        target = targets[cell]
        baseline = preflight["rebuilt"]["ensembles"][cell].score
        per_cell.append(
            {
                "cell_id": cell,
                "dataset_family": bundle.dataset_family,
                "task_type": bundle.task_type,
                "method": "B3",
                "n_seeds_averaged": 5,
                **_metrics(target, baseline),
            }
        )
        for variant in variants:
            per_cell.append(
                {
                    "cell_id": cell,
                    "dataset_family": bundle.dataset_family,
                    "task_type": bundle.task_type,
                    "method": variant,
                    "n_seeds_averaged": 5,
                    **_metrics(target, preflight["arrays"][(variant, cell)]["score"]),
                }
            )

    methods = ["B3", *variants]
    summaries = [
        _summary(per_cell, method, metric)
        for method in methods
        for metric in ("auroc", "auprc")
    ]
    comparison_pairs = [(variant, "B3") for variant in variants]
    comparison_pairs.extend(
        [
            ("L1_LOCAL_DESCENT_PRIMARY", "L2_STATIC_GATE_CONTROL"),
            ("L1_LOCAL_DESCENT_PRIMARY", "L3_ROW_PERMUTED_GATE_CONTROL"),
        ]
    )
    cell_hash = canonical_sha256(cells)
    comparisons = [
        _comparison(
            per_cell,
            candidate,
            reference,
            draws=int(args.bootstrap_draws),
            seed=_stable_seed(
                str(config["experiment_id"]),
                candidate,
                reference,
                cell_hash,
                "descriptive_family_block_bootstrap_v1",
            ),
        )
        for candidate, reference in comparison_pairs
    ]
    verdict = _verdict(comparisons)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "PER_CELL_METRICS.csv", per_cell)
    _write_csv(
        args.out_dir / "ROUTER_DIAGNOSTICS.csv",
        preflight["router_diagnostics"],
    )
    _write_content_json(
        args.out_dir / "SUMMARY.json",
        {"schema": "deem_b3_local_descent_pgrd_summary_v1", "rows": summaries},
    )
    _write_content_json(
        args.out_dir / "COMPARISONS.json",
        {
            "schema": "deem_b3_local_descent_pgrd_comparisons_v1",
            "rows": comparisons,
        },
    )
    _write_content_json(args.out_dir / "SCREEN_VERDICT.json", verdict)
    output_hashes = {
        name: sha256_file(args.out_dir / name)
        for name in (
            "PER_CELL_METRICS.csv",
            "ROUTER_DIAGNOSTICS.csv",
            "SUMMARY.json",
            "COMPARISONS.json",
            "SCREEN_VERDICT.json",
        )
    }
    report = {
        "schema": "deem_b3_local_descent_pgrd_evaluation_v1",
        "status": "complete",
        "scientific_tier": "retrospective_exploratory",
        "natural_24cell_targets_previously_opened": True,
        "strict_two_phase_preflight_before_label_import": True,
        "label_module_imported_only_after_preflight": True,
        "fit_was_label_free": True,
        "target_geometry_is_transductive": True,
        "per_row_or_end_to_end_self_free_inference_claimed": False,
        "experiment_id": config["experiment_id"],
        "cells": cells,
        "variants": variants,
        "n_seeds_averaged": 5,
        "n_dataset_families": len(
            {preflight["rebuilt"]["bundles"][cell].dataset_family for cell in cells}
        ),
        "bootstrap_draws": int(args.bootstrap_draws),
        "hashes": preflight["hashes"],
        "bundle_audit": {
            cell: preflight["rebuilt"]["bundle_audit"][cell] for cell in cells
        },
        "sidecar_audit": sidecar_audit,
        "output_hashes": output_hashes,
        "summaries": summaries,
        "comparisons": comparisons,
        "screen_verdict": verdict,
        "mechanism_interpretation": (
            "a gain over B3 is not specific evidence for sample-local routing unless "
            "L1 also beats both the exact mean-gate and row-permuted-gate controls"
        ),
        "rule_boundary": (
            "numeric survival thresholds were not stored in the local fit config; "
            "the shared adjacent-screen rule is therefore retrospective and cannot "
            "upgrade this result to confirmation"
        ),
        "fit_source_manifest_scope": (
            "the runner-declared explicit dependency set, verified byte-for-byte; "
            "it is not an automatically discovered transitive import closure"
        ),
    }
    _write_content_json(args.out_dir / "REPORT.json", report)
    print(json.dumps({**report, "report_path": str((args.out_dir / 'REPORT.json').resolve())}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
