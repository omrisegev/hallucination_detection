#!/usr/bin/env python3
"""Freeze the one-stage B3-orthogonal IU-PGRD boost without labels."""

from __future__ import annotations

import argparse
import ast
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

from spectral_utils.deem_b3_iupgrd_boost import (  # noqa: E402
    fit_b3_iupgrd_cell,
    permute_family_direction,
    pooled_cross_only_direction,
    score_b3_iupgrd_boost,
)
from spectral_utils.deem_b3_residual_moe import (  # noqa: E402
    load_frozen_b3_ensemble,
)
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ResidualGraphDeemError,
    atomic_save_npz,
    atomic_write_json,
    canonical_sha256,
    donor_risk_matrix,
    environment_fingerprint,
    sha256_file,
)
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    BUNDLE_SCHEMA,
    load_registry,
    load_target_free_bundle,
    registry_cell,
)
from spectral_utils.specrage_views import VIEW_ORDER  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/deem_b3_iupgrd_boost_v1.json"
DEFAULT_REGISTRY = ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"
LABEL_MODULE = "spectral_utils.residual_graph_deem_labels"
EXPECTED_VARIANTS = (
    "E0_B3_EXACT_ALIAS",
    "E1_B3_ORTH_IUPGRD_FULL",
    "E2_B3_ORTH_IUPGRD_HALF",
    "E3_B3_UNPROJECTED_IUPGRD_FULL",
    "E4_B3_ORTH_FAMILY_PERMUTED_FULL",
    "E5_B3_ORTH_ROW_PERMUTED_FULL",
)


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


def _content_json(path: Path, payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    if "content_sha256" in value:
        raise ValueError("content_sha256 is reserved")
    value["content_sha256"] = canonical_sha256(value)
    return atomic_write_json(path, value)


def _verified_content_json(path: Path, expected_schema: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.get("content_sha256")
    unhashed = dict(value)
    unhashed.pop("content_sha256", None)
    if value.get("schema") != expected_schema or canonical_sha256(unhashed) != expected:
        raise ResidualGraphDeemError(f"content-bound JSON mismatch: {path}")
    return value


def load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "deem_b3_iupgrd_boost_v1_config":
        raise ResidualGraphDeemError("IU-PGRD boost config schema mismatch")
    identifiers = tuple(str(row.get("id")) for row in value.get("variants", []))
    if identifiers != EXPECTED_VARIANTS:
        raise ResidualGraphDeemError("frozen E0--E5 roster/order changed")
    if value.get("frozen_screen_rule", {}).get("mechanistic_primary") != (
        "E1_B3_ORTH_IUPGRD_FULL"
    ):
        raise ResidualGraphDeemError("mechanistic primary changed")
    if value.get("baseline", {}).get("seeds") != [0, 1, 2, 3, 4]:
        raise ResidualGraphDeemError("mean-of-five B3 seed contract changed")
    graph = value.get("graph", {})
    if int(graph.get("k", -1)) != 7 or graph.get("direction") != (
        "cross_only_negative_pooled_c"
    ):
        raise ResidualGraphDeemError("graph/direction contract changed")
    permutation = value.get("family_axis_permutation", [])
    if sorted(int(item) for item in permutation) != list(range(len(VIEW_ORDER))):
        raise ResidualGraphDeemError("invalid frozen family-axis permutation")
    screen = [str(item) for item in value.get("screen_cells", [])]
    if len(screen) != 8 or len(screen) != len(set(screen)):
        raise ResidualGraphDeemError("screen must contain eight unique cells")
    boundary = value.get("scientific_boundary", {})
    required = (
        "fit_is_label_free",
        "baseline_is_frozen_mean_of_five_b3",
        "iu_is_coordinate_generator_not_baseline",
        "pooled_calibration_excludes_entire_target_dataset_family",
        "one_stage_only",
    )
    if any(not boundary.get(name) for name in required):
        raise ResidualGraphDeemError("scientific boundary is incomplete")
    return value


def _resolve_local_import(path: Path, node: ast.AST) -> list[Path]:
    relative = path.relative_to(ROOT).with_suffix("")
    package = list(relative.parts[:-1])
    modules: list[list[str]] = []
    if isinstance(node, ast.Import):
        modules = [alias.name.split(".") for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            base = package[: len(package) - (node.level - 1)]
            tail = [] if node.module is None else node.module.split(".")
            modules = [base + tail]
        elif node.module:
            modules = [node.module.split(".")]
    output = []
    for parts in modules:
        if not parts or parts[0] not in {"spectral_utils", "scripts"}:
            continue
        candidate = ROOT.joinpath(*parts).with_suffix(".py")
        if not candidate.is_file():
            candidate = ROOT.joinpath(*parts) / "__init__.py"
        if candidate.is_file():
            output.append(candidate.resolve())
    return output


def _source_dependency_manifest(seed_paths: Sequence[Path]) -> dict[str, str]:
    pending = [path.resolve() for path in seed_paths]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                pending.extend(_resolve_local_import(path, node))
    label_path = ROOT / "spectral_utils/residual_graph_deem_labels.py"
    if label_path.resolve() in visited:
        raise ResidualGraphDeemError("source closure unexpectedly contains labels")
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted(visited)
    }


def _load_baseline_freeze(
    baseline_dir: Path,
) -> tuple[dict, dict[str, str], dict[str, Any]]:
    path = baseline_dir / "SCORE_FREEZE_MANIFEST.json"
    value = _verified_content_json(path, "deem_vs_iupcr_score_freeze_v1")
    if value.get("status") != "complete" or bool(value.get("debug")) or (
        "B3" not in value.get("arms", [])
    ):
        raise ResidualGraphDeemError("B3 freeze manifest is inadmissible")
    rows = value.get("artifacts", [])
    artifact_map = {str(row["path"]): str(row["sha256"]) for row in rows}
    if len(artifact_map) != len(rows):
        raise ResidualGraphDeemError("duplicate B3 freeze artifact paths")
    return value, artifact_map, {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "content_sha256": value["content_sha256"],
        "run_definition_sha256": value["run_definition_sha256"],
    }


def _verify_baseline_member(
    baseline_dir: Path,
    artifact_map: Mapping[str, str],
    bundle: Any,
    seed: int,
) -> dict:
    paths = {}
    for suffix in ("npz", "json"):
        relative = f"fits/{bundle.cell_id}/B3__seed{int(seed)}.{suffix}"
        path = baseline_dir / relative
        expected = artifact_map.get(relative)
        if expected is None or not path.is_file() or sha256_file(path) != expected:
            raise ResidualGraphDeemError(f"unbound B3 member: {relative}")
        paths[suffix] = path
    metadata = json.loads(paths["json"].read_text(encoding="utf-8"))
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
    failed = [name for name, item in expected.items() if metadata.get(name) != item]
    if failed or not metadata.get("health", {}).get("healthy"):
        raise ResidualGraphDeemError(
            f"B3/bundle metadata mismatch for {bundle.cell_id}/seed{seed}: {failed}"
        )
    return {
        "seed": int(seed),
        "array_path": str(paths["npz"].resolve()),
        "array_sha256": sha256_file(paths["npz"]),
        "metadata_path": str(paths["json"].resolve()),
        "metadata_sha256": sha256_file(paths["json"]),
    }


def _load_bound_bundle(
    bundle_dir: Path, registry: Mapping[str, Any], cell_id: str
) -> tuple[Any, dict[str, Any]]:
    path = bundle_dir / f"{cell_id}.npz"
    sidecar_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"missing bundle/manifest: {cell_id}")
    bundle = load_target_free_bundle(path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    registered = registry_cell(registry, cell_id)
    source = registered["source"]
    row_hash = canonical_sha256(list(bundle.row_ids))
    checks = {
        "cell": bundle.cell_id == cell_id,
        "family": bundle.dataset_family == str(registered["dataset_family"]),
        "task": bundle.task_type == str(registered["task_type"]),
        "rows": len(bundle.row_ids) == int(registered["n_rows"]),
        "features": tuple(bundle.feature_names) == tuple(registered["feature_names"]),
        "signs": np.array_equal(
            bundle.confidence_signs,
            np.asarray(registered["confidence_signs"], dtype=np.int8),
        ),
        "inventory": bundle.inventory_sha256 == registered["inventory_sha256"],
        "source": bundle.source_sha256 == source["source_sha256"],
        "source_manifest": bundle.manifest_sha256 == source["manifest_sha256"],
        "admission": bundle.admission_sha256 == source["admission_contract_sha256"],
        "sidecar_schema": sidecar.get("schema") == BUNDLE_SCHEMA,
        "sidecar_cell": sidecar.get("cell_id") == cell_id,
        "sidecar_bundle": sidecar.get("bundle_sha256") == bundle.bundle_sha256,
        "sidecar_row_hash": sidecar.get("ordered_row_id_sha256") == row_hash,
        "sidecar_inventory": sidecar.get("inventory_sha256") == bundle.inventory_sha256,
        "sidecar_source": sidecar.get("source_sha256") == bundle.source_sha256,
        "sidecar_source_manifest": sidecar.get("manifest_sha256") == bundle.manifest_sha256,
        "sidecar_admission": sidecar.get("admission_sha256") == bundle.admission_sha256,
        "sidecar_no_labels": sidecar.get("labels_accessed") is False,
        "sidecar_no_pickle": sidecar.get("allow_pickle") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ResidualGraphDeemError(f"bundle binding failed for {cell_id}: {failed}")
    return bundle, {
        "bundle_path": str(path.resolve()),
        "bundle_sha256": bundle.bundle_sha256,
        "bundle_manifest_path": str(sidecar_path.resolve()),
        "bundle_manifest_sha256": sha256_file(sidecar_path),
        "ordered_row_id_sha256": row_hash,
        "inventory_sha256": bundle.inventory_sha256,
        "source_sha256": bundle.source_sha256,
        "source_manifest_sha256": bundle.manifest_sha256,
        "admission_sha256": bundle.admission_sha256,
    }


def _csr_payload(prefix: str, matrix: Any) -> dict[str, np.ndarray]:
    value = csr_matrix(matrix, dtype=np.float64).copy()
    value.sort_indices()
    return {
        f"{prefix}_data": np.asarray(value.data, dtype=np.float64),
        f"{prefix}_indices": np.asarray(value.indices, dtype=np.int64),
        f"{prefix}_indptr": np.asarray(value.indptr, dtype=np.int64),
        f"{prefix}_shape": np.asarray(value.shape, dtype=np.int64),
    }


def _ndarray_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_sha256(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _selection(value: str, config: Mapping[str, Any], registry: Mapping[str, Any]) -> list[str]:
    available = [str(row["cell_id"]) for row in registry["cells"]]
    if value == "screen":
        selected = [str(item) for item in config["screen_cells"]]
    elif value == "all":
        selected = available
    else:
        selected = [item.strip() for item in value.split(",") if item.strip()]
    if not selected or len(selected) != len(set(selected)) or not set(selected).issubset(available):
        raise ValueError("invalid unique registered cell selection")
    if len(set(registry_cell(registry, cell)["dataset_family"] for cell in selected)) < 2:
        raise ValueError("donor-family exclusion requires at least two families")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cells", default="screen")
    args = parser.parse_args()

    _assert_label_firewall()
    config = load_config(args.config)
    registry = load_registry(args.registry)
    cells = _selection(args.cells, config, registry)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _, baseline_artifacts, baseline_audit = _load_baseline_freeze(args.baseline_dir)
    bundles: dict[str, Any] = {}
    bundle_audits: dict[str, dict] = {}
    baseline_member_audits: dict[str, list[dict]] = {}
    ensembles: dict[str, Any] = {}
    seeds = tuple(int(seed) for seed in config["baseline"]["seeds"])
    for cell in cells:
        bundle, audit = _load_bound_bundle(args.bundle_dir, registry, cell)
        bundles[cell] = bundle
        bundle_audits[cell] = audit
        baseline_member_audits[cell] = [
            _verify_baseline_member(args.baseline_dir, baseline_artifacts, bundle, seed)
            for seed in seeds
        ]
        ensembles[cell] = load_frozen_b3_ensemble(
            args.baseline_dir,
            cell,
            seeds=seeds,
            expected_bundle_sha256=bundle.bundle_sha256,
            expected_ordered_row_id_sha256=audit["ordered_row_id_sha256"],
        )

    family_by_cell = {cell: bundles[cell].dataset_family for cell in cells}
    donor_cells_by_family = {
        held_family: [
            cell for cell in cells if family_by_cell[cell] != held_family
        ]
        for held_family in sorted(set(family_by_cell.values()))
    }
    source_manifest = _source_dependency_manifest(
        [
            ROOT / "spectral_utils/deem_b3_iupgrd_boost.py",
            ROOT / "scripts/run_deem_b3_iupgrd_boost_v1.py",
        ]
    )
    source_manifest["configs/deem_b3_iupgrd_boost_v1.json"] = sha256_file(args.config)
    source_manifest = dict(sorted(source_manifest.items()))
    run_definition = {
        "schema": "deem_b3_iupgrd_boost_v1_run_definition",
        "status": "frozen_before_fit",
        "experiment_id": config["experiment_id"],
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "registry_path": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry),
        "bundle_dir": str(args.bundle_dir.resolve()),
        "baseline_dir": str(args.baseline_dir.resolve()),
        "cells": cells,
        "dataset_family_by_cell": family_by_cell,
        "donor_cells_by_held_dataset_family": donor_cells_by_family,
        "variants": [row["id"] for row in config["variants"]],
        "baseline_seeds": list(seeds),
        "bundle_audits": bundle_audits,
        "baseline_freeze_audit": baseline_audit,
        "baseline_member_audits": baseline_member_audits,
        "source_dependencies": source_manifest,
        "source_dependency_sha256": canonical_sha256(source_manifest),
        "environment": environment_fingerprint(),
        "orientation_note": (
            "simultaneously orienting donor IU score and residuals leaves R'Lu "
            "invariant; held-cell residual orientation aligns q to B3"
        ),
        "scientific_qualifier": (
            "the direction descends donor IU graph roughness, not held B3 graph "
            "roughness; transfer is not a B3-descent guarantee"
        ),
        "targets_accessed_during_fit": False,
        "labels_module_imported": False,
    }
    run_path = args.out_dir / "RUN_DEFINITION.json"
    run_definition_sha256 = _content_json(run_path, run_definition)

    artifacts: list[dict[str, Any]] = []

    def record(path: Path, kind: str, **extra: Any) -> None:
        artifacts.append(
            {
                "path": path.relative_to(args.out_dir).as_posix(),
                "sha256": sha256_file(path),
                "kind": kind,
                **extra,
            }
        )

    fitted = {}
    state_hashes = {}
    for cell in cells:
        bundle = bundles[cell]
        X_risk, held_risk, transform = donor_risk_matrix(
            bundle.X_raw, bundle.X_raw, bundle.feature_names
        )
        if not np.array_equal(X_risk, held_risk):
            raise AssertionError("within-cell risk transform is not identical")
        fitted[cell] = fit_b3_iupgrd_cell(
            cell,
            X_risk,
            bundle.feature_names,
            ensembles[cell].score,
            bundle.row_ids,
            k=int(config["graph"]["k"]),
        )
        item = fitted[cell]
        state_path = args.out_dir / "states" / f"{cell}.npz"
        state_sha = atomic_save_npz(
            state_path,
            row_id=np.asarray(bundle.row_ids, dtype=str),
            feature_names=np.asarray(bundle.feature_names, dtype=str),
            family_order=np.asarray(item.families, dtype=str),
            global_family_order=np.asarray(VIEW_ORDER, dtype=str),
            risk_transform_mean=np.asarray(transform.mean, dtype=np.float64),
            risk_transform_scale=np.asarray(transform.scale, dtype=np.float64),
            risk_transform_constant_mask=np.asarray(transform.constant_mask, dtype=np.int8),
            baseline_score=np.asarray(item.baseline_score, dtype=np.float64),
            baseline_logit=np.asarray(item.baseline_logit, dtype=np.float64),
            baseline_z=np.asarray(item.baseline_z, dtype=np.float64),
            baseline_logit_mean=np.asarray(item.baseline_mean, dtype=np.float64),
            baseline_logit_scale=np.asarray(item.baseline_scale, dtype=np.float64),
            iu_score=np.asarray(item.iu_score, dtype=np.float64),
            iu_score_aligned=np.asarray(item.iu_score_aligned, dtype=np.float64),
            iu_orientation=np.asarray(item.iu_orientation, dtype=np.int8),
            iu_weights=np.asarray(item.iu_weights, dtype=np.float64),
            raw_family_contributions=np.asarray(item.raw_contributions, dtype=np.float64),
            standardized_family_contributions=np.asarray(
                item.standardized_contributions, dtype=np.float64
            ),
            iu_family_residuals=np.asarray(item.residuals, dtype=np.float64),
            moment_A=np.asarray(item.moment.A, dtype=np.float64),
            moment_c=np.asarray(item.moment.c, dtype=np.float64),
            moment_presence=np.asarray(item.moment.presence, dtype=np.int8),
            **{
                f"iu_transform_{name}": np.asarray(value)
                for name, value in item.transform_arrays.items()
            },
            **_csr_payload("graph", item.graph),
            **_csr_payload("laplacian", item.laplacian),
        )
        state_meta_path = state_path.with_suffix(".json")
        _content_json(
            state_meta_path,
            {
                "schema": "deem_b3_iupgrd_boost_v1_state",
                "status": "complete",
                "cell_id": cell,
                "dataset_family": family_by_cell[cell],
                "run_definition_sha256": run_definition_sha256,
                "array_sha256": state_sha,
                "bundle_audit": bundle_audits[cell],
                "baseline_member_audits": baseline_member_audits[cell],
                "diagnostics": item.diagnostics,
                "moment_diagnostics": item.moment.diagnostics,
                "row_id_array_sha256": _ndarray_sha256(
                    np.asarray(bundle.row_ids, dtype=str)
                ),
                "uses_labels": False,
            },
        )
        state_hashes[cell] = {
            "array_sha256": state_sha,
            "metadata_sha256": sha256_file(state_meta_path),
        }
        record(state_path, "cell_state", cell_id=cell)
        record(state_meta_path, "cell_state_metadata", cell_id=cell)

    calibration_by_family = {}
    calibration_hashes = {}
    for held_family, donor_cells in donor_cells_by_family.items():
        direction, diagnostics = pooled_cross_only_direction(
            [fitted[cell] for cell in donor_cells],
            [family_by_cell[cell] for cell in donor_cells],
        )
        calibration_by_family[held_family] = direction
        path = args.out_dir / "calibrations" / f"held_{held_family}.npz"
        array_sha = atomic_save_npz(
            path,
            direction=np.asarray(direction, dtype=np.float64),
            donor_cells=np.asarray(donor_cells, dtype=str),
            donor_dataset_families=np.asarray(
                [family_by_cell[cell] for cell in donor_cells], dtype=str
            ),
            donor_moment_A=np.stack([fitted[cell].moment.A for cell in donor_cells]),
            donor_moment_c=np.stack([fitted[cell].moment.c for cell in donor_cells]),
            donor_moment_presence=np.stack(
                [fitted[cell].moment.presence for cell in donor_cells]
            ).astype(np.int8),
        )
        meta_path = path.with_suffix(".json")
        _content_json(
            meta_path,
            {
                "schema": "deem_b3_iupgrd_boost_v1_calibration",
                "status": "complete",
                "held_dataset_family": held_family,
                "donor_cells": donor_cells,
                "donor_dataset_families": [family_by_cell[cell] for cell in donor_cells],
                "whole_held_dataset_family_excluded": all(
                    family_by_cell[cell] != held_family for cell in donor_cells
                ),
                "run_definition_sha256": run_definition_sha256,
                "array_sha256": array_sha,
                "donor_state_hashes": {cell: state_hashes[cell] for cell in donor_cells},
                "diagnostics": diagnostics,
                "uses_labels": False,
            },
        )
        calibration_hashes[held_family] = {
            "array_sha256": array_sha,
            "metadata_sha256": sha256_file(meta_path),
        }
        record(path, "calibration", held_dataset_family=held_family)
        record(meta_path, "calibration_metadata", held_dataset_family=held_family)

    permutation = [int(item) for item in config["family_axis_permutation"]]
    for variant in config["variants"]:
        variant_id = str(variant["id"])
        settings = dict(variant["config"])
        for cell in cells:
            held_family = family_by_cell[cell]
            base_direction = calibration_by_family[held_family]
            applied_direction = (
                permute_family_direction(base_direction, permutation)
                if settings["family_axis_permuted"]
                else np.asarray(base_direction, dtype=float)
            )
            result = score_b3_iupgrd_boost(
                fitted[cell],
                applied_direction,
                trust_factor=float(settings["trust_factor"]),
                project_against_b3=bool(settings["project_against_b3"]),
                row_ids=bundles[cell].row_ids,
                row_permutation_salt=(
                    str(config["row_permutation_salt"])
                    if settings["row_permuted"]
                    else None
                ),
            )
            if variant_id == "E0_B3_EXACT_ALIAS" and not np.array_equal(
                result.score, ensembles[cell].score
            ):
                raise ResidualGraphDeemError(f"exact B3 alias failed: {cell}")
            if not np.isfinite(result.score).all():
                raise ResidualGraphDeemError(f"non-finite score: {variant_id}/{cell}")
            directory = args.out_dir / "scores" / variant_id / cell
            path = directory / f"{variant_id}.npz"
            array_sha = atomic_save_npz(
                path,
                row_id=np.asarray(bundles[cell].row_ids, dtype=str),
                score=np.asarray(result.score, dtype=np.float64),
                baseline_score=np.asarray(fitted[cell].baseline_score, dtype=np.float64),
                logit=np.asarray(result.logit, dtype=np.float64),
                baseline_logit=np.asarray(fitted[cell].baseline_logit, dtype=np.float64),
                correction_z=np.asarray(result.correction_z, dtype=np.float64),
                raw_correction=np.asarray(result.raw_correction, dtype=np.float64),
                projected_correction=np.asarray(
                    result.projected_correction, dtype=np.float64
                ),
                projection_coefficients=np.asarray(
                    result.projection_coefficients, dtype=np.float64
                ),
                calibration_direction=np.asarray(base_direction, dtype=np.float64),
                applied_direction=np.asarray(applied_direction, dtype=np.float64),
                family_axis_permutation=np.asarray(permutation, dtype=np.int64),
                row_permutation=np.asarray(result.row_permutation, dtype=np.int64),
            )
            meta_path = path.with_suffix(".json")
            present = fitted[cell].families
            _content_json(
                meta_path,
                {
                    "schema": "deem_b3_iupgrd_boost_v1_fit",
                    "status": "complete",
                    "variant_id": variant_id,
                    "role": variant["role"],
                    "cell_id": cell,
                    "dataset_family": held_family,
                    "run_definition_sha256": run_definition_sha256,
                    "array_sha256": array_sha,
                    "state_hashes": state_hashes[cell],
                    "calibration_hashes": calibration_hashes[held_family],
                    "bundle_audit": bundle_audits[cell],
                    "baseline_member_audits": baseline_member_audits[cell],
                    "donor_cells": donor_cells_by_family[held_family],
                    "variant_config": settings,
                    "present_families": list(present),
                    "missing_global_families": [
                        name for name in VIEW_ORDER if name not in present
                    ],
                    "applied_direction_present": [
                        float(applied_direction[VIEW_ORDER.index(name)])
                        for name in present
                    ],
                    "family_axis_permutation": permutation,
                    "diagnostics": result.diagnostics,
                    "uses_labels": False,
                },
            )
            record(path, "score", variant_id=variant_id, cell_id=cell)
            record(meta_path, "score_metadata", variant_id=variant_id, cell_id=cell)

    _assert_label_firewall()
    manifest_path = args.out_dir / "FIT_ARTIFACT_MANIFEST.json"
    manifest_payload = {
        "schema": "deem_b3_iupgrd_boost_v1_fit_artifact_manifest",
        "status": "complete",
        "experiment_id": config["experiment_id"],
        "run_definition_path": "RUN_DEFINITION.json",
        "run_definition_sha256": run_definition_sha256,
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
        "artifact_count": len(artifacts),
        "targets_accessed_during_fit": False,
        "labels_module_imported": False,
    }
    manifest_sha256 = _content_json(manifest_path, manifest_payload)
    complete_path = args.out_dir / "FIT_COMPLETE.json"
    _content_json(
        complete_path,
        {
            "schema": "deem_b3_iupgrd_boost_v1_fit_complete",
            "status": "complete",
            "experiment_id": config["experiment_id"],
            "run_definition_sha256": run_definition_sha256,
            "fit_artifact_manifest_path": "FIT_ARTIFACT_MANIFEST.json",
            "fit_artifact_manifest_sha256": manifest_sha256,
            "cells": cells,
            "variants": [row["id"] for row in config["variants"]],
            "targets_accessed_during_fit": False,
            "labels_module_imported": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "out_dir": str(args.out_dir.resolve()),
                "n_cells": len(cells),
                "n_variants": len(config["variants"]),
                "run_definition_sha256": run_definition_sha256,
                "fit_artifact_manifest_sha256": manifest_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
