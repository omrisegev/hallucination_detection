#!/usr/bin/env python3
"""Fit and freeze the target-free local-descent PGRD screen.

This process never imports the label-sidecar module.  It verifies every
target-free bundle and every one of the five frozen B3 members against their
content/hash manifests, constructs true grouped-fold LOO residuals, fits the
pooled PGRD direction after excluding the target dataset family, and writes
only score/geometry artifacts.  Evaluation is deliberately a separate future
process.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.deem_b3_local_descent_pgrd import (  # noqa: E402
    LocalDescentPGRDConfig,
    build_common_residual_graph,
    fit_leave_dataset_family_out_direction,
    score_local_descent_pgrd,
    validate_config,
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
    atomic_save_npz,
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
DEFAULT_OUT_DIR = ROOT / "local_cache/deem_b3_moe_v1/local_descent_pgrd_screen_v1"
LABEL_MODULE = "spectral_utils.residual_graph_deem_labels"
EXPECTED_VARIANTS = (
    "L0_B3_EXACT_ALIAS",
    "L1_LOCAL_DESCENT_PRIMARY",
    "L2_STATIC_GATE_CONTROL",
    "L3_ROW_PERMUTED_GATE_CONTROL",
)
SOURCE_DEPENDENCIES = (
    ROOT / "spectral_utils/deem_b3_local_descent_pgrd.py",
    ROOT / "spectral_utils/deem_b3_residual_moe.py",
    ROOT / "spectral_utils/residual_graph_deem.py",
    ROOT / "spectral_utils/residual_graph_deem_data.py",
    ROOT / "spectral_utils/graph_topology.py",
    ROOT / "spectral_utils/laplacian_upcr.py",
    ROOT / "scripts/run_deem_b3_local_descent_pgrd_v1.py",
)


def _assert_label_firewall() -> None:
    imported = sorted(
        name
        for name in sys.modules
        if name == LABEL_MODULE or name.startswith(LABEL_MODULE + ".")
    )
    if imported:
        raise ResidualGraphDeemError(
            "label module crossed local-descent fit boundary: " + ", ".join(imported)
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
    if value.get("schema") != "deem_b3_local_descent_pgrd_v1_config":
        raise ResidualGraphDeemError("local-descent config schema mismatch")
    identifiers = tuple(str(row.get("id")) for row in value.get("variants", []))
    if identifiers != EXPECTED_VARIANTS:
        raise ResidualGraphDeemError("local-descent L0--L3 roster/order changed")
    allowed = {field.name for field in fields(LocalDescentPGRDConfig)}
    modes = []
    for row in value["variants"]:
        settings = row.get("config", {})
        unknown = sorted(set(settings) - allowed)
        if unknown:
            raise ResidualGraphDeemError(
                f"unknown local-descent config fields for {row['id']}: {unknown}"
            )
        parsed = LocalDescentPGRDConfig(**settings)
        validate_config(parsed)
        modes.append(parsed.gate_mode)
    if tuple(modes) != ("alias", "local", "static", "row_permuted"):
        raise ResidualGraphDeemError("local-descent mechanism controls changed")
    if value.get("baseline", {}).get("seeds") != [0, 1, 2, 3, 4]:
        raise ResidualGraphDeemError("mean-five B3 seed contract changed")
    boundary = value.get("scientific_boundary", {})
    required = (
        "fit_is_label_free",
        "baseline_is_frozen",
        "target_dataset_family_is_excluded_from_direction_calibration",
        "target_cell_rows_enter_only_target_free_transductive_geometry",
        "method_is_not_per_row_or_end_to_end_self_free_inference",
        "gates_and_direction_are_frozen_at_the_b3_base",
        "one_additive_stage_only",
        "natural_24cell_targets_previously_opened",
        "screen_is_exploratory",
        "evaluation_is_a_separate_explicit_process",
    )
    if any(boundary.get(name) is not True for name in required):
        raise ResidualGraphDeemError("local-descent scientific boundary is incomplete")
    graph = value.get("graph", {})
    if (
        int(graph.get("k", -1)) != 7
        or graph.get("topology") != "self_safe_self_tuning_union_knn"
        or graph.get("laplacian") != "symmetric_normalized"
        or graph.get("one_common_graph_for_gate_and_line_step") is not True
        or graph.get("fixed_at_b3_base") is not True
    ):
        raise ResidualGraphDeemError("local-descent graph contract changed")
    screen = [str(value) for value in value.get("screen_cells", [])]
    if len(screen) != 8 or len(screen) != len(set(screen)):
        raise ResidualGraphDeemError("screen must contain eight unique cells")
    return value


def _source_manifest(config_path: Path) -> dict[str, str]:
    paths = list(SOURCE_DEPENDENCIES) + [config_path.resolve()]
    output = {}
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


def _load_baseline_freeze(baseline_dir: Path) -> tuple[dict[str, str], dict]:
    path = baseline_dir / "SCORE_FREEZE_MANIFEST.json"
    manifest = _verified_content_json(
        path, expected_schema="deem_vs_iupcr_score_freeze_v1"
    )
    if (
        manifest.get("status") != "complete"
        or bool(manifest.get("debug"))
        or "B3" not in manifest.get("arms", [])
    ):
        raise ResidualGraphDeemError("frozen B3 score manifest is inadmissible")
    entries = manifest.get("artifacts", [])
    artifact_map = {str(row["path"]): str(row["sha256"]) for row in entries}
    if len(artifact_map) != len(entries):
        raise ResidualGraphDeemError("duplicate frozen B3 artifact paths")
    return artifact_map, {
        "manifest_path": str(path.resolve()),
        "manifest_file_sha256": sha256_file(path),
        "manifest_content_sha256": manifest["content_sha256"],
        "baseline_run_definition_sha256": manifest["run_definition_sha256"],
    }


def _verify_baseline_member(
    baseline_dir: Path,
    artifact_map: Mapping[str, str],
    bundle: Any,
    seed: int,
) -> dict:
    relative_npz = f"fits/{bundle.cell_id}/B3__seed{int(seed)}.npz"
    relative_json = f"fits/{bundle.cell_id}/B3__seed{int(seed)}.json"
    npz_path = baseline_dir / relative_npz
    json_path = baseline_dir / relative_json
    for relative, path in ((relative_npz, npz_path), (relative_json, json_path)):
        if (
            artifact_map.get(relative) is None
            or not path.is_file()
            or sha256_file(path) != artifact_map[relative]
        ):
            raise ResidualGraphDeemError(
                f"B3 member is not bound to freeze manifest: {relative}"
            )
    metadata = _verified_content_json(
        json_path, expected_schema="deem_vs_iupcr_fit_artifact_v1"
    )
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
        "array_sha256": artifact_map[relative_npz],
    }
    mismatch = sorted(name for name, wanted in expected.items() if metadata.get(name) != wanted)
    if mismatch or metadata.get("health", {}).get("healthy") is not True:
        raise ResidualGraphDeemError(
            f"B3/bundle binding failed for {bundle.cell_id}/seed{seed}: {mismatch}"
        )
    return metadata


def _load_bound_bundle(
    bundle_dir: Path,
    registry: Mapping[str, Any],
    cell_id: str,
) -> tuple[Any, dict[str, Any]]:
    path = bundle_dir / f"{cell_id}.npz"
    sidecar_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"missing target-free bundle/manifest: {cell_id}")
    bundle = load_target_free_bundle(path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    registered = registry_cell(registry, cell_id)
    row_hash = canonical_sha256(list(bundle.row_ids))
    source = registered["source"]
    checks = {
        "bundle cell": bundle.cell_id == cell_id,
        "bundle family": bundle.dataset_family == str(registered["dataset_family"]),
        "bundle task": bundle.task_type == str(registered["task_type"]),
        "bundle rows": len(bundle.row_ids) == int(registered["n_rows"]),
        "bundle features": tuple(bundle.feature_names) == tuple(registered["feature_names"]),
        "bundle signs": np.array_equal(
            bundle.confidence_signs,
            np.asarray(registered["confidence_signs"], dtype=np.int8),
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
        "sidecar row order": sidecar.get("ordered_row_id_sha256") == row_hash,
        "sidecar inventory": sidecar.get("inventory_sha256") == bundle.inventory_sha256,
        "sidecar source": sidecar.get("source_sha256") == bundle.source_sha256,
        "sidecar source manifest": sidecar.get("manifest_sha256") == bundle.manifest_sha256,
        "sidecar admission": sidecar.get("admission_sha256") == bundle.admission_sha256,
        "sidecar label firewall": sidecar.get("labels_accessed") is False,
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
        "ordered_row_id_sha256": row_hash,
        "inventory_sha256": bundle.inventory_sha256,
        "source_sha256": bundle.source_sha256,
        "source_manifest_sha256": bundle.manifest_sha256,
        "admission_sha256": bundle.admission_sha256,
    }


def _csr_payload(prefix: str, matrix: sparse.spmatrix) -> dict[str, np.ndarray]:
    value = sparse.csr_matrix(matrix, dtype=np.float64).copy()
    value.sort_indices()
    return {
        f"{prefix}_data": np.asarray(value.data, dtype=np.float64),
        f"{prefix}_indices": np.asarray(value.indices, dtype=np.int64),
        f"{prefix}_indptr": np.asarray(value.indptr, dtype=np.int64),
        f"{prefix}_shape": np.asarray(value.shape, dtype=np.int64),
    }


def _parse_selection(value: str, available: Sequence[str]) -> list[str]:
    if value == "all" or value == "screen":
        return list(available)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(available))
    if not selected or unknown or len(selected) != len(set(selected)):
        raise ValueError(f"invalid selection; unknown={unknown}")
    return selected


def _valid_existing(
    array_path: Path,
    metadata_path: Path,
    *,
    definition_sha256: str,
    variant_id: str,
    cell_id: str,
    bundle_audit: Mapping[str, Any],
) -> bool:
    if not array_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = _verified_content_json(
            metadata_path,
            expected_schema="deem_b3_local_descent_pgrd_fit_artifact_v1",
        )
        return bool(
            metadata.get("status") == "complete"
            and metadata.get("definition_sha256") == definition_sha256
            and metadata.get("variant_id") == variant_id
            and metadata.get("cell_id") == cell_id
            and metadata.get("bundle_sha256") == bundle_audit["bundle_sha256"]
            and metadata.get("inventory_sha256") == bundle_audit["inventory_sha256"]
            and metadata.get("ordered_row_id_sha256")
            == bundle_audit["ordered_row_id_sha256"]
            and metadata.get("targets_accessed_during_fit") is False
            and metadata.get("labels_module_imported") is False
            and metadata.get("health", {}).get("healthy") is True
            and sha256_file(array_path) == metadata.get("array_sha256")
        )
    except Exception:
        return False


def _output_paths(out_dir: Path, variant_id: str, cell_id: str) -> tuple[Path, Path]:
    directory = out_dir / "scores" / variant_id / cell_id
    array_path = directory / f"{variant_id}.npz"
    return array_path, array_path.with_suffix(".json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cells", default="screen")
    parser.add_argument("--variants", default="all")
    args = parser.parse_args()

    _assert_label_firewall()
    config = load_config(args.config)
    registry = load_registry(args.registry)
    all_cells = [str(row["cell_id"]) for row in registry["cells"]]
    screen_cells = [str(cell) for cell in config["screen_cells"]]
    if not set(screen_cells).issubset(set(all_cells)):
        raise ResidualGraphDeemError("screen contains unregistered cells")
    cells = _parse_selection(args.cells, screen_cells)
    variant_ids = _parse_selection(
        args.variants, [str(row["id"]) for row in config["variants"]]
    )
    variants = [
        row for row in config["variants"] if str(row["id"]) in set(variant_ids)
    ]
    seeds = tuple(int(seed) for seed in config["baseline"]["seeds"])
    artifact_map, baseline_freeze_audit = _load_baseline_freeze(args.baseline_dir)
    source_manifest = _source_manifest(args.config)

    bundles = {}
    bundle_audits = {}
    ensembles = {}
    residual_cells = {}
    moments = {}
    folds = {}
    family_by_cell = {}
    for cell_id in all_cells:
        _assert_label_firewall()
        bundle, audit = _load_bound_bundle(args.bundle_dir, registry, cell_id)
        for seed in seeds:
            _verify_baseline_member(
                args.baseline_dir, artifact_map, bundle, int(seed)
            )
        ensemble = load_frozen_b3_ensemble(
            args.baseline_dir,
            cell_id,
            seeds=seeds,
            expected_bundle_sha256=bundle.bundle_sha256,
            expected_ordered_row_id_sha256=audit["ordered_row_id_sha256"],
        )
        if not np.array_equal(ensemble.score, np.mean(ensemble.seed_scores, axis=0)):
            raise ResidualGraphDeemError(f"mean-five B3 identity failed: {cell_id}")
        grouped_folds = assign_grouped_length_folds(
            bundle.group_ids, bundle.raw_trace_length
        )
        residual = build_residual_cell(
            ensemble, baseline_score=ensemble.score, folds=grouped_folds
        )
        moment = graph_roughness_moment(
            residual, bundle.row_ids, residual_source="loo", k=7
        )
        bundles[cell_id] = bundle
        bundle_audits[cell_id] = audit
        ensembles[cell_id] = ensemble
        residual_cells[cell_id] = residual
        moments[cell_id] = moment
        folds[cell_id] = grouped_folds
        family_by_cell[cell_id] = bundle.dataset_family

    calibrations = {}
    for held_family in sorted({family_by_cell[cell] for cell in cells}):
        calibration = fit_leave_dataset_family_out_direction(
            [moments[cell] for cell in all_cells],
            [family_by_cell[cell] for cell in all_cells],
            target_dataset_family=held_family,
        )
        if calibration.target_dataset_family != held_family:
            raise ResidualGraphDeemError("calibration held-family identity mismatch")
        if held_family in calibration.donor_dataset_families:
            raise ResidualGraphDeemError("held dataset family leaked into donor calibration")
        calibrations[held_family] = calibration

    graphs = {
        cell_id: build_common_residual_graph(
            residual_cells[cell_id], bundles[cell_id].row_ids, k=7
        )
        for cell_id in cells
    }
    for cell_id, graph in graphs.items():
        if graph.row_ids != tuple(str(value) for value in bundles[cell_id].row_ids):
            raise ResidualGraphDeemError(
                f"target-cell graph row binding failed: {cell_id}"
            )
    definition_body = {
        "schema": "deem_b3_local_descent_pgrd_run_definition_v1",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "registry_sha256": sha256_file(args.registry),
        "source_manifest": source_manifest,
        "source_sha256": canonical_sha256(source_manifest),
        "baseline_freeze": baseline_freeze_audit,
        "bundle_dir": str(args.bundle_dir.resolve()),
        "baseline_dir": str(args.baseline_dir.resolve()),
        "donor_cells": all_cells,
        "score_cells": cells,
        "variants": [str(row["id"]) for row in variants],
        "seeds": list(seeds),
        "bundle_bindings": {
            cell: bundle_audits[cell] for cell in all_cells
        },
        "targets_accessed_during_fit": False,
        "labels_module_imported": False,
        "fit_only_no_evaluation": True,
        "transductive_target_cell_geometry": True,
    }
    definition_sha256 = canonical_sha256(definition_body)
    run_definition = {
        **definition_body,
        "definition_sha256": definition_sha256,
        "status": "running",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out_dir / "RUN_DEFINITION.json", run_definition)

    manifest_rows = []
    for cell_id in cells:
        _assert_label_firewall()
        bundle = bundles[cell_id]
        residual = residual_cells[cell_id]
        ensemble = ensembles[cell_id]
        graph = graphs[cell_id]
        calibration = calibrations[bundle.dataset_family]
        if calibration.target_dataset_family != bundle.dataset_family:
            raise ResidualGraphDeemError(
                f"wrong held-family calibration applied to {cell_id}"
            )
        results = {
            str(variant["id"]): score_local_descent_pgrd(
                residual,
                calibration,
                graph,
                config=LocalDescentPGRDConfig(**variant["config"]),
            )
            for variant in variants
        }
        if "L0_B3_EXACT_ALIAS" in results and not np.array_equal(
            results["L0_B3_EXACT_ALIAS"].score, ensemble.score
        ):
            raise ResidualGraphDeemError(f"exact B3 alias failed: {cell_id}")
        if {
            "L1_LOCAL_DESCENT_PRIMARY",
            "L2_STATIC_GATE_CONTROL",
            "L3_ROW_PERMUTED_GATE_CONTROL",
        }.issubset(results):
            primary = results["L1_LOCAL_DESCENT_PRIMARY"]
            static = results["L2_STATIC_GATE_CONTROL"]
            permuted = results["L3_ROW_PERMUTED_GATE_CONTROL"]
            expected_static = np.repeat(
                np.mean(primary.local_gate_probabilities, axis=0)[None, :],
                len(bundle.row_ids),
                axis=0,
            )
            if not np.array_equal(static.gate_probabilities, expected_static):
                raise ResidualGraphDeemError("static control is not mean-gate matched")
            if not np.array_equal(
                permuted.gate_probabilities,
                primary.local_gate_probabilities[permuted.row_permutation],
            ):
                raise ResidualGraphDeemError("row-permuted control is not gate matched")

        for variant in variants:
            variant_id = str(variant["id"])
            result = results[variant_id]
            array_path, metadata_path = _output_paths(
                args.out_dir, variant_id, cell_id
            )
            if _valid_existing(
                array_path,
                metadata_path,
                definition_sha256=definition_sha256,
                variant_id=variant_id,
                cell_id=cell_id,
                bundle_audit=bundle_audits[cell_id],
            ):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                manifest_rows.append(
                    {
                        "variant_id": variant_id,
                        "cell_id": cell_id,
                        "dataset_family": bundle.dataset_family,
                        "array_path": str(array_path.relative_to(args.out_dir)),
                        "array_sha256": metadata["array_sha256"],
                        "metadata_path": str(metadata_path.relative_to(args.out_dir)),
                        "metadata_sha256": sha256_file(metadata_path),
                    }
                )
                continue
            if array_path.exists() or metadata_path.exists():
                raise ResidualGraphDeemError(
                    f"refusing to overwrite mismatched artifact: {metadata_path}"
                )
            arrays = {
                "score": np.asarray(result.score, dtype=np.float64),
                "posterior": np.column_stack(
                    [1.0 - result.score, result.score]
                ).astype(np.float64),
                "baseline_score": np.asarray(ensemble.score, dtype=np.float64),
                "baseline_seed_scores": np.asarray(
                    ensemble.seed_scores, dtype=np.float64
                ),
                "baseline_logit": np.asarray(residual.baseline_logit, dtype=np.float64),
                "baseline_z": np.asarray(result.baseline_z, dtype=np.float64),
                "logit": np.asarray(result.logit, dtype=np.float64),
                "correction_z": np.asarray(result.correction_z, dtype=np.float64),
                "local_gradient": np.asarray(result.local_gradient, dtype=np.float64),
                "expert_terms": np.asarray(result.expert_terms, dtype=np.float64),
                "activation": np.asarray(result.activation, dtype=np.float64),
                "local_gate_probabilities": np.asarray(
                    result.local_gate_probabilities, dtype=np.float64
                ),
                "gate_probabilities": np.asarray(
                    result.gate_probabilities, dtype=np.float64
                ),
                "raw_direction": np.asarray(result.raw_direction, dtype=np.float64),
                "direction": np.asarray(result.direction, dtype=np.float64),
                "row_permutation": np.asarray(result.row_permutation, dtype=np.int64),
                "alpha": np.asarray(result.alpha, dtype=np.float64),
                "tau": np.asarray(result.tau, dtype=np.float64),
                "cross_term": np.asarray(result.cross_term, dtype=np.float64),
                "quadratic_term": np.asarray(result.quadratic_term, dtype=np.float64),
                "roughness_before": np.asarray(
                    result.roughness_before, dtype=np.float64
                ),
                "roughness_after": np.asarray(
                    result.roughness_after, dtype=np.float64
                ),
                "family_direction": np.asarray(
                    result.family_direction, dtype=np.float64
                ),
                "family_stability": np.asarray(
                    result.family_stability, dtype=np.float64
                ),
                "present_family_mask": np.asarray(
                    result.present_mask, dtype=np.int8
                ),
                "family_order": np.asarray(result.family_order, dtype=str),
                "loo_residuals": np.asarray(
                    residual.loo_residuals, dtype=np.float64
                ),
                "loo_seed_instability": np.asarray(
                    residual.loo_seed_instability, dtype=np.float64
                ),
                "loo_predictability": np.asarray(
                    residual.loo_predictability, dtype=np.float64
                ),
                "grouped_folds": np.asarray(folds[cell_id], dtype=np.int64),
                "row_id": np.asarray(bundle.row_ids, dtype=str),
                "group_id": np.asarray(bundle.group_ids, dtype=str),
                "raw_trace_length": np.asarray(
                    bundle.raw_trace_length, dtype=np.int64
                ),
                "graph_coordinates": np.asarray(
                    graph.coordinates, dtype=np.float64
                ),
                "graph_family_order": np.asarray(graph.family_order, dtype=str),
                "graph_tie_keys": np.asarray(graph.tie_keys, dtype=np.float64),
                "pooled_direction": np.asarray(
                    calibration.direction, dtype=np.float64
                ),
                "pooled_stability": np.asarray(
                    calibration.stability, dtype=np.float64
                ),
                "donor_group_directions": np.asarray(
                    calibration.donor_group_directions, dtype=np.float64
                ),
                "donor_group_presence": np.asarray(
                    calibration.donor_group_presence, dtype=np.int8
                ),
                "moment_a0": np.asarray(moments[cell_id].a0, dtype=np.float64),
                "moment_c0": np.asarray(moments[cell_id].c0, dtype=np.float64),
                **_csr_payload("graph", graph.graph),
                **_csr_payload("laplacian", graph.laplacian),
            }
            assert_no_target_fields(arrays)
            _assert_label_firewall()
            array_sha256 = atomic_save_npz(array_path, **arrays)
            graph_payload_names = sorted(
                name
                for name in arrays
                if name.startswith("graph_") or name.startswith("laplacian_")
            )
            graph_payload_sha256 = canonical_sha256(
                {
                    name: _ndarray_sha256(arrays[name])
                    for name in graph_payload_names
                }
            )
            health = {
                "healthy": bool(
                    np.isfinite(result.score).all()
                    and result.diagnostics["alpha_within_bounds"]
                    and result.diagnostics["roughness_nonincreasing"]
                    and result.diagnostics["logit_reconstruction_max_abs"] <= 1e-12
                    and (
                        variant_id != "L0_B3_EXACT_ALIAS"
                        or np.array_equal(result.score, ensemble.score)
                    )
                ),
                "score_finite": bool(np.isfinite(result.score).all()),
                "alpha_within_bounds": result.diagnostics["alpha_within_bounds"],
                "roughness_nonincreasing": result.diagnostics[
                    "roughness_nonincreasing"
                ],
                "exact_alias": bool(np.array_equal(result.score, ensemble.score)),
            }
            if not health["healthy"]:
                raise ResidualGraphDeemError(
                    f"unhealthy local-descent score: {variant_id}/{cell_id}"
                )
            metadata = {
                "schema": "deem_b3_local_descent_pgrd_fit_artifact_v1",
                "status": "complete",
                "experiment_id": config["experiment_id"],
                "definition_sha256": definition_sha256,
                "source_sha256": definition_body["source_sha256"],
                "variant_id": variant_id,
                "variant_role": str(variant.get("role", "")),
                "variant_config": variant["config"],
                "cell_id": cell_id,
                "dataset_family": bundle.dataset_family,
                "task_type": bundle.task_type,
                "n_rows": len(bundle.row_ids),
                "n_features": len(bundle.feature_names),
                "bundle_sha256": bundle.bundle_sha256,
                "bundle_manifest_file_sha256": bundle_audits[cell_id][
                    "bundle_manifest_file_sha256"
                ],
                "inventory_sha256": bundle.inventory_sha256,
                "ordered_row_id_sha256": bundle_audits[cell_id][
                    "ordered_row_id_sha256"
                ],
                "baseline_seeds": list(seeds),
                "baseline_score_ensemble": "exact_mean_of_seed_posteriors",
                "baseline_diagnostics": ensemble.diagnostics,
                "residual_diagnostics": residual.diagnostics,
                "graph_diagnostics": graph.diagnostics,
                "graph_payload_sha256": graph_payload_sha256,
                "graph_binding_sha256": graph.binding_sha256,
                "calibration_diagnostics": calibration.diagnostics,
                "calibration_held_dataset_family": calibration.target_dataset_family,
                "calibration_donor_dataset_families": list(
                    calibration.donor_dataset_families
                ),
                "calibration_excludes_entire_held_dataset_family": True,
                "score_diagnostics": result.diagnostics,
                "health": health,
                "array_sha256": array_sha256,
                "targets_accessed_during_fit": False,
                "labels_module_imported": False,
                "fit_only_no_evaluation": True,
                "target_cell_geometry_is_transductive": True,
                "per_row_self_free_inference_claimed": False,
            }
            metadata["content_sha256"] = canonical_sha256(metadata)
            atomic_write_json(metadata_path, metadata)
            manifest_rows.append(
                {
                    "variant_id": variant_id,
                    "cell_id": cell_id,
                    "dataset_family": bundle.dataset_family,
                    "array_path": str(array_path.relative_to(args.out_dir)),
                    "array_sha256": array_sha256,
                    "metadata_path": str(metadata_path.relative_to(args.out_dir)),
                    "metadata_sha256": sha256_file(metadata_path),
                }
            )

    expected = len(cells) * len(variants)
    if len(manifest_rows) != expected:
        raise ResidualGraphDeemError(
            f"score manifest has {len(manifest_rows)} rows; expected {expected}"
        )
    _assert_label_firewall()
    freeze = {
        "schema": "deem_b3_local_descent_pgrd_score_freeze_v1",
        "status": "complete",
        "experiment_id": config["experiment_id"],
        "definition_sha256": definition_sha256,
        "n_artifacts": len(manifest_rows),
        "variants": [str(row["id"]) for row in variants],
        "cells": cells,
        "targets_accessed_during_fit": False,
        "labels_module_imported": False,
        "fit_only_no_evaluation": True,
        "artifacts": manifest_rows,
    }
    freeze["content_sha256"] = canonical_sha256(freeze)
    atomic_write_json(args.out_dir / "SCORE_FREEZE_MANIFEST.json", freeze)
    run_definition["status"] = "complete"
    run_definition["n_score_artifacts"] = len(manifest_rows)
    run_definition["score_freeze_content_sha256"] = freeze["content_sha256"]
    atomic_write_json(args.out_dir / "RUN_DEFINITION.json", run_definition)
    print(json.dumps(run_definition, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
