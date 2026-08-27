#!/usr/bin/env python3
"""Fit the frozen, label-free DEEM-B3 Crossed-Rook v1 experiment.

This is the physical target firewall.  It imports neither label sidecars nor
evaluation code.  Every fit is bound to a target-free bundle, its ordered row
IDs, the same-cell/same-seed frozen B3 state, the registry, config, and source
hashes before any downstream evaluator may open natural targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_deem_b3_moe_v1 import load_frozen_b3  # noqa: E402
from spectral_utils.deem_b3_crossed_rook import (  # noqa: E402
    CrossedRookConfig,
    fit_deem_b3_crossed_rook,
)
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ResidualGraphDeemError,
    atomic_save_npz,
    atomic_write_json,
    canonical_sha256,
    donor_risk_matrix,
    environment_fingerprint,
    jsonable,
    sha256_file,
)
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    load_registry,
    load_target_free_bundle,
    registry_cell,
)


SCHEMA = "deem_b3_crossed_rook_v1"
DEFAULT_CONFIG = ROOT / "configs/deem_b3_crossed_rook_v1.json"
DEFAULT_REGISTRY = ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"
DEFAULT_BUNDLE_DIR = ROOT / "local_cache/deem_b3_moe_v1/bundles"
DEFAULT_BASELINE_DIR = ROOT / "local_cache/deem_b3_moe_v1/b3_frozen"
DEFAULT_OUT_DIR = ROOT / "local_cache/deem_b3_moe_v1/crossed_rook_v1"
EXPECTED_ARMS = (
    "A0_B3_EXACT_ALIAS",
    "A1_ROW_9",
    "A2_COLUMN_9",
    "A3_CROSSED_ROOK_18",
    "A4_NONROOK_18_CONTROL",
)
CORE_SOURCES = (
    ROOT / "spectral_utils/deem_b3_crossed_rook.py",
    ROOT / "scripts/run_deem_b3_crossed_rook_v1.py",
    ROOT / "scripts/run_deem_b3_moe_v1.py",
)


def load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "deem_b3_crossed_rook_v1_config":
        raise ResidualGraphDeemError("crossed-rook config schema mismatch")
    variants = value.get("variants", [])
    ids = tuple(str(row.get("id")) for row in variants)
    if ids != EXPECTED_ARMS:
        raise ResidualGraphDeemError(
            f"crossed-rook arm roster/order mismatch: {ids!r}"
        )
    for row in variants:
        CrossedRookConfig(**row.get("config", {}))
    boundary = value.get("scientific_boundary", {})
    if not (
        boundary.get("fit_is_label_free")
        and boundary.get("baseline_is_frozen")
        and boundary.get("only_edge_theta_is_trainable")
        and boundary.get("structural_evidence_was_length_conditioned_but_v1_fit_is_unconditional")
    ):
        raise ResidualGraphDeemError("crossed-rook scientific boundary is incomplete")
    return value


def source_hash(config_path: Path) -> tuple[str, dict[str, str]]:
    manifest = {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in CORE_SOURCES
    }
    manifest[config_path.relative_to(ROOT).as_posix()] = sha256_file(config_path)
    return canonical_sha256(manifest), manifest


def parse_selection(
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
    if unknown or not requested or len(requested) != len(set(requested)):
        raise ValueError(f"invalid selection; unknown={unknown}, requested={requested}")
    return requested


def variant_lookup(config: Mapping) -> dict[str, Mapping]:
    return {str(row["id"]): row for row in config["variants"]}


def output_paths(out_dir: Path, arm_id: str, cell_id: str, seed: int) -> tuple[Path, Path]:
    directory = out_dir / "fits" / arm_id / cell_id
    stem = f"{arm_id}__seed{int(seed)}"
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def valid_existing(
    array_path: Path,
    metadata_path: Path,
    *,
    definition_hash: str,
) -> bool:
    if not array_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_content = metadata.get("content_sha256")
        unhashed = dict(metadata)
        unhashed.pop("content_sha256", None)
        return bool(
            metadata.get("schema") == "deem_b3_crossed_rook_fit_artifact_v1"
            and metadata.get("status") == "complete"
            and metadata.get("run_definition_sha256") == definition_hash
            and metadata.get("health", {}).get("healthy")
            and canonical_sha256(unhashed) == expected_content
            and sha256_file(array_path) == metadata.get("array_sha256")
        )
    except Exception:
        return False


def _score_hash(value: np.ndarray) -> str:
    score = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return canonical_sha256(
        {"dtype": str(score.dtype), "shape": list(score.shape), "bytes": score.tobytes().hex()}
    )


def fit_one(
    *,
    bundle_dir: Path,
    baseline_dir: Path,
    out_dir: Path,
    registry: Mapping,
    variant: Mapping,
    seed: int,
    cell_id: str,
    definition_hash: str,
    experiment_id: str,
) -> dict:
    arm_id = str(variant["id"])
    array_path, metadata_path = output_paths(out_dir, arm_id, cell_id, seed)
    if valid_existing(array_path, metadata_path, definition_hash=definition_hash):
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    if array_path.exists() or metadata_path.exists():
        raise ResidualGraphDeemError(
            f"incomplete/mismatched artifact exists; refusing overwrite: {metadata_path}"
        )

    bundle_path = bundle_dir / f"{cell_id}.npz"
    bundle = load_target_free_bundle(bundle_path)
    registered = registry_cell(registry, cell_id)
    if (
        len(bundle.row_ids) != int(registered["n_rows"])
        or bundle.inventory_sha256 != registered["inventory_sha256"]
        or bundle.cell_id != cell_id
    ):
        raise ResidualGraphDeemError(f"registry/bundle mismatch: {cell_id}")
    X_risk, _, transform = donor_risk_matrix(
        bundle.X_raw, bundle.X_raw, bundle.feature_names
    )
    baseline_state, baseline_score, baseline_metadata, baseline_audit = load_frozen_b3(
        baseline_dir, cell_id, seed, X_risk, bundle.feature_names
    )
    result = fit_deem_b3_crossed_rook(
        X_risk,
        bundle.feature_names,
        baseline_state,
        baseline_orientation=int(baseline_metadata["orientation"]),
        baseline_score=baseline_score if arm_id == "A0_B3_EXACT_ALIAS" else None,
        seed=seed,
        config=CrossedRookConfig(**variant.get("config", {})),
    )
    if not result.health["healthy"]:
        raise ResidualGraphDeemError(f"unhealthy crossed-rook fit: {arm_id}/{cell_id}/{seed}")

    reconstructed_base = 1.0 / (
        1.0 + np.exp(-np.clip(result.base_logit, -700.0, 700.0))
    )
    base_identity = float(np.max(np.abs(reconstructed_base - baseline_score)))
    if base_identity > 1e-12:
        raise ResidualGraphDeemError(
            f"frozen B3 base changed before/inside fit: {base_identity:.3e}"
        )
    if arm_id == "A0_B3_EXACT_ALIAS" and not np.array_equal(
        result.score, baseline_score
    ):
        raise ResidualGraphDeemError("A0 did not preserve saved B3 score bytes")

    edge_pairs = np.asarray(result.edge_pairs, dtype=np.int64)
    expected_edges = 9 if arm_id in {"A1_ROW_9", "A2_COLUMN_9"} else 18
    if edge_pairs.shape != (expected_edges, 2):
        raise ResidualGraphDeemError(f"edge cardinality mismatch: {arm_id}")

    arrays = {
        "score": np.asarray(result.score, dtype=np.float64),
        "posterior": np.asarray(result.posterior, dtype=np.float64),
        "logit": np.asarray(result.logit, dtype=np.float64),
        "base_score": np.asarray(baseline_score, dtype=np.float64),
        "base_logit": np.asarray(result.base_logit, dtype=np.float64),
        "correction": np.asarray(result.correction, dtype=np.float64),
        "contributions": np.asarray(result.contributions, dtype=np.float64),
        "base_contributions": np.asarray(result.base_contributions, dtype=np.float64),
        "cell_delta": np.asarray(result.cell_delta, dtype=np.float64),
        "edge_values": np.asarray(result.edge_values, dtype=np.float64),
        "edge_raw_contribution": np.asarray(
            result.edge_raw_contribution, dtype=np.float64
        ),
        "edge_pairs": edge_pairs,
        "edge_kinds": np.asarray(result.edge_kinds, dtype=str),
        "edge_weights": np.asarray(result.edge_weights, dtype=np.float64),
        "core_features": np.asarray(result.core_features, dtype=str),
        "core_indices": np.asarray(result.core_indices, dtype=np.int64),
        "feature_names": np.asarray(result.feature_names, dtype=str),
        "row_id": np.asarray(bundle.row_ids, dtype=str),
        "standardization_mean": np.asarray(transform.mean, dtype=np.float64),
        "standardization_scale": np.asarray(transform.scale, dtype=np.float64),
    }
    for name, value in result.state.items():
        arrays[f"state__{name}"] = np.asarray(value)
    array_hash = atomic_save_npz(array_path, **arrays)
    record = {
        "schema": "deem_b3_crossed_rook_fit_artifact_v1",
        "status": "complete",
        "experiment_id": experiment_id,
        "arm_id": arm_id,
        "arm_role": str(variant.get("role", "")),
        "cell_id": cell_id,
        "dataset_family": bundle.dataset_family,
        "task_type": bundle.task_type,
        "seed": int(seed),
        "n_rows": len(bundle.row_ids),
        "n_features": len(bundle.feature_names),
        "n_edges": len(result.edge_pairs),
        "bundle_path": bundle_path.relative_to(ROOT).as_posix(),
        "bundle_sha256": bundle.bundle_sha256,
        "inventory_sha256": bundle.inventory_sha256,
        "ordered_row_id_sha256": canonical_sha256(list(bundle.row_ids)),
        "run_definition_sha256": definition_hash,
        "array_path": array_path.relative_to(out_dir).as_posix(),
        "array_sha256": array_hash,
        "score_sha256": _score_hash(result.score),
        "base_score_sha256": _score_hash(baseline_score),
        "config": jsonable(result.config),
        "config_sha256": canonical_sha256(result.config),
        "orientation": int(result.orientation),
        "orientation_policy": "fixed_same_seed_frozen_b3",
        "aligned_bias": float(result.aligned_bias),
        "risk_anchor_difference": float(result.risk_anchor_difference),
        "health": jsonable(result.health),
        "diagnostics": jsonable(result.diagnostics),
        "objective_history": jsonable(result.objective_history),
        "baseline": {
            **baseline_audit,
            "base_reconstruction_max_abs": base_identity,
            "score_sha256": _score_hash(baseline_score),
        },
        "targets_accessed_during_fit": False,
        "label_module_imported_during_fit": False,
        "length_residualization": "none_unconditional_atomic_core_v1",
    }
    record["content_sha256"] = canonical_sha256(record)
    atomic_write_json(metadata_path, record)
    return record


def _write_score_freeze(
    *,
    out_dir: Path,
    invocation_hash: str,
    definition_hash: str,
    records: Sequence[Mapping],
    arms: Sequence[str],
    cells: Sequence[str],
    seeds: Sequence[int],
    publish_root: bool,
) -> dict:
    artifact_map = {
        f"{row['arm_id']}|{row['cell_id']}|{row['seed']}": {
            "array_sha256": row["array_sha256"],
            "metadata_sha256": sha256_file(
                out_dir / Path(row["array_path"]).with_suffix(".json")
            ),
            "score_sha256": row["score_sha256"],
            "base_score_sha256": row["base_score_sha256"],
            "bundle_sha256": row["bundle_sha256"],
            "ordered_row_id_sha256": row["ordered_row_id_sha256"],
            "baseline_array_sha256": row["baseline"]["baseline_array_sha256"],
            "baseline_metadata_sha256": row["baseline"]["baseline_metadata_sha256"],
        }
        for row in records
    }
    freeze = {
        "schema": "deem_b3_crossed_rook_score_freeze_v1",
        "status": "frozen",
        "run_definition_sha256": definition_hash,
        "invocation_sha256": invocation_hash,
        "arms": list(arms),
        "cells": list(cells),
        "seeds": [int(seed) for seed in seeds],
        "n_artifacts": len(records),
        "artifact_sha256": artifact_map,
        "targets_accessed_during_fit": False,
        "label_module_imported_during_fit": False,
    }
    freeze["content_sha256"] = canonical_sha256(freeze)
    path = out_dir / "score_freezes" / f"{invocation_hash}.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != jsonable(freeze):
            raise ResidualGraphDeemError("existing score freeze differs")
    else:
        atomic_write_json(path, freeze, immutable=True)
    if publish_root:
        root_path = out_dir / "SCORE_FREEZE_MANIFEST.json"
        if root_path.is_file():
            if json.loads(root_path.read_text(encoding="utf-8")) != jsonable(freeze):
                raise ResidualGraphDeemError("root score freeze already binds another invocation")
        else:
            atomic_write_json(root_path, freeze, immutable=True)
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--arms", default="all", help="all or comma-separated IDs")
    parser.add_argument("--cells", default="screen", help="all, screen, or comma-separated IDs")
    parser.add_argument("--seeds", default="0", help="comma-separated members of 0..4")
    parser.add_argument(
        "--publish-freeze",
        action="store_true",
        help="also publish this invocation as immutable OUT/SCORE_FREEZE_MANIFEST.json",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    registry = load_registry(args.registry)
    variants = variant_lookup(config)
    selected_arms = parse_selection(args.arms, EXPECTED_ARMS)
    all_cells = [str(row["cell_id"]) for row in registry["cells"]]
    selected_cells = parse_selection(
        args.cells, all_cells, screen=config["screen_cells"]
    )
    selected_seeds = [
        int(value.strip()) for value in args.seeds.split(",") if value.strip()
    ]
    allowed_seeds = set(int(value) for value in config["baseline"]["available_fit_seeds"])
    if (
        not selected_seeds
        or len(selected_seeds) != len(set(selected_seeds))
        or not set(selected_seeds).issubset(allowed_seeds)
    ):
        raise ValueError(f"seeds must be unique members of {sorted(allowed_seeds)}")

    code_hash, source_manifest = source_hash(args.config)
    baseline_freeze_path = args.baseline_dir / "SCORE_FREEZE_MANIFEST.json"
    definition = {
        "schema": "deem_b3_crossed_rook_run_definition_v1",
        "experiment_id": config["experiment_id"],
        "status": config["status"],
        "config_sha256": sha256_file(args.config),
        "registry_content_sha256": registry["registry_content_sha256"],
        "source_sha256": code_hash,
        "source_manifest": source_manifest,
        "baseline_score_freeze_sha256": sha256_file(baseline_freeze_path),
        "arm_order": list(EXPECTED_ARMS),
        "representation": config["representation"],
        "scientific_boundary": config["scientific_boundary"],
        "explicit_caveat": (
            "The motivating reconstruction evidence conditioned on cubic trace length. "
            "V1 is an unconditional EBM on centered/scaled frozen-B3 atomic core "
            "coordinates; it does not claim length-conditioned mechanism isolation."
        ),
        "environment": environment_fingerprint(),
    }
    definition["content_sha256"] = canonical_sha256(definition)
    definition_hash = canonical_sha256(definition)
    definition_path = args.out_dir / "run_definitions" / f"{definition_hash}.json"
    if definition_path.is_file():
        if json.loads(definition_path.read_text(encoding="utf-8")) != jsonable(definition):
            raise ResidualGraphDeemError("existing crossed-rook run definition mismatch")
    else:
        atomic_write_json(definition_path, definition, immutable=True)

    records = []
    for arm_id in selected_arms:
        for cell_id in selected_cells:
            for seed in selected_seeds:
                print(f"[{arm_id}] {cell_id} seed={seed}", flush=True)
                records.append(
                    fit_one(
                        bundle_dir=args.bundle_dir,
                        baseline_dir=args.baseline_dir,
                        out_dir=args.out_dir,
                        registry=registry,
                        variant=variants[arm_id],
                        seed=seed,
                        cell_id=cell_id,
                        definition_hash=definition_hash,
                        experiment_id=str(config["experiment_id"]),
                    )
                )

    invocation = {
        "run_definition_sha256": definition_hash,
        "selected_arms": selected_arms,
        "selected_cells": selected_cells,
        "selected_seeds": selected_seeds,
    }
    invocation_hash = canonical_sha256(invocation)
    completion = {
        "schema": "deem_b3_crossed_rook_fit_complete_v1",
        "status": "complete",
        "run_definition_sha256": definition_hash,
        "invocation_sha256": invocation_hash,
        "n_records": len(records),
        "arms": selected_arms,
        "cells": selected_cells,
        "seeds": selected_seeds,
        "all_healthy": all(row.get("health", {}).get("healthy") for row in records),
        "artifact_sha256": {
            f"{row['arm_id']}|{row['cell_id']}|{row['seed']}": {
                "array_sha256": row["array_sha256"],
                "metadata_sha256": sha256_file(
                    args.out_dir / Path(row["array_path"]).with_suffix(".json")
                ),
            }
            for row in records
        },
        "targets_accessed_during_fit": False,
        "label_module_imported_during_fit": False,
    }
    completion["content_sha256"] = canonical_sha256(completion)
    completion_path = args.out_dir / "fit_completions" / f"{invocation_hash}.json"
    if completion_path.is_file():
        if json.loads(completion_path.read_text(encoding="utf-8")) != jsonable(completion):
            raise ResidualGraphDeemError("existing fit completion differs")
    else:
        atomic_write_json(completion_path, completion, immutable=True)
    _write_score_freeze(
        out_dir=args.out_dir,
        invocation_hash=invocation_hash,
        definition_hash=definition_hash,
        records=records,
        arms=selected_arms,
        cells=selected_cells,
        seeds=selected_seeds,
        publish_root=bool(args.publish_freeze),
    )
    print(
        f"complete: {len(records)} artifacts; definition={definition_hash}; "
        f"invocation={invocation_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
