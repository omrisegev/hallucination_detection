#!/usr/bin/env python3
"""Fit label-free DEEM-B3 family-routing variants from frozen B3 states.

This is the target-firewalled fit boundary.  It intentionally imports neither
the label-sidecar module nor any evaluator.  Natural targets are opened only by
``evaluate_deem_b3_moe_v1.py`` after fit artifacts have been written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.deem_b3_moe import DeemMoEConfig, fit_deem_b3_moe  # noqa: E402
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ContinuousDeemConfig,
    ResidualGraphDeemError,
    _FamilyAdditiveEnergy,
    atomic_save_npz,
    atomic_write_json,
    canonical_sha256,
    donor_risk_matrix,
    jsonable,
    sha256_file,
)
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    load_registry,
    load_target_free_bundle,
    registry_cell,
)


DEFAULT_CONFIG = ROOT / "configs/deem_b3_moe_v1.json"
DEFAULT_REGISTRY = ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"
CORE_SOURCES = (
    ROOT / "spectral_utils/deem_b3_moe.py",
    ROOT / "scripts/run_deem_b3_moe_v1.py",
)


def load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "deem_b3_moe_v1_config":
        raise ResidualGraphDeemError("DEEM-B3 MoE config schema mismatch")
    variants = value.get("variants", [])
    ids = [str(row.get("id")) for row in variants]
    if not variants or len(ids) != len(set(ids)):
        raise ResidualGraphDeemError("DEEM-B3 MoE variant roster is empty or duplicated")
    for row in variants:
        DeemMoEConfig(**row.get("config", {}))
    if not value.get("scientific_boundary", {}).get("natural_24cell_targets_previously_opened"):
        raise ResidualGraphDeemError("retrospective natural-target boundary is not declared")
    return value


def source_hash(config_path: Path) -> str:
    payload = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in CORE_SOURCES}
    payload["experiment_config"] = sha256_file(config_path)
    return canonical_sha256(payload)


def variant_lookup(config: Mapping) -> dict[str, Mapping]:
    return {str(row["id"]): row for row in config["variants"]}


def parse_selection(value: str, available: list[str], *, screen: list[str] | None = None) -> list[str]:
    if value == "all":
        return list(available)
    if value == "screen":
        if screen is None:
            raise ValueError("screen selection is not defined here")
        unknown = sorted(set(screen) - set(available))
        if unknown:
            raise ValueError("unknown screen entries: " + ", ".join(unknown))
        return list(screen)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError("unknown selection entries: " + ", ".join(unknown))
    return requested


def load_frozen_b3(
    baseline_dir: Path,
    cell_id: str,
    seed: int,
    X_risk: np.ndarray,
    feature_names,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict, dict]:
    stem = f"B3__seed{int(seed)}"
    directory = baseline_dir / "fits" / cell_id
    array_path = directory / f"{stem}.npz"
    metadata_path = directory / f"{stem}.json"
    if not array_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing frozen B3 artifact: {cell_id}/{stem}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_content = metadata.get("content_sha256")
    if expected_content is not None:
        unhashed = dict(metadata)
        unhashed.pop("content_sha256", None)
        if canonical_sha256(unhashed) != expected_content:
            raise ResidualGraphDeemError(f"frozen B3 metadata content mismatch: {cell_id}/{stem}")
    if sha256_file(array_path) != metadata.get("array_sha256"):
        raise ResidualGraphDeemError(f"frozen B3 array hash mismatch: {cell_id}/{stem}")
    with np.load(array_path, allow_pickle=False) as data:
        state = {
            name.removeprefix("state__"): np.asarray(data[name])
            for name in data.files
            if name.startswith("state__")
        }
        score = np.asarray(data["score"], dtype=np.float64)
        stored_names = tuple(str(name) for name in data["feature_names"].tolist())
    if stored_names != tuple(feature_names) or score.shape != (len(X_risk),):
        raise ResidualGraphDeemError(f"frozen B3 inventory/row mismatch: {cell_id}/{stem}")

    # Reconstruct the warm-start score before fitting.  This catches a wrong
    # transform, seed, state orientation, or mixed-v2 artifact immediately.
    import torch

    continuous = ContinuousDeemConfig(**metadata["config"]["continuous"])
    model = _FamilyAdditiveEnergy(feature_names, continuous, seed)
    model.load_state_numpy(state)
    with torch.no_grad():
        ell, _, _ = model.logit(torch.as_tensor(X_risk, dtype=torch.float64))
    aligned = int(metadata["orientation"]) * ell.cpu().numpy()
    reconstructed = 1.0 / (1.0 + np.exp(-np.clip(aligned, -700.0, 700.0)))
    identity_error = float(np.max(np.abs(reconstructed - score)))
    if identity_error > 1e-12:
        raise ResidualGraphDeemError(
            f"frozen B3 identity reconstruction failed: {cell_id}/{stem} {identity_error:.3e}"
        )
    audit = {
        "baseline_array_sha256": metadata["array_sha256"],
        "baseline_metadata_sha256": sha256_file(metadata_path),
        "baseline_identity_max_abs": identity_error,
        "baseline_orientation": int(metadata["orientation"]),
    }
    return state, score, metadata, audit


def output_paths(out_dir: Path, variant_id: str, cell_id: str, seed: int) -> tuple[Path, Path]:
    stem = f"{variant_id}__seed{int(seed)}"
    directory = out_dir / "fits" / variant_id / cell_id
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def valid_existing(array_path: Path, metadata_path: Path, *, definition_hash: str) -> bool:
    if not array_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_content = metadata.get("content_sha256")
        unhashed = dict(metadata)
        unhashed.pop("content_sha256", None)
        return bool(
            metadata.get("status") == "complete"
            and metadata.get("run_definition_sha256") == definition_hash
            and metadata.get("health", {}).get("healthy")
            and canonical_sha256(unhashed) == expected_content
            and sha256_file(array_path) == metadata.get("array_sha256")
        )
    except Exception:
        return False


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
    array_path, metadata_path = output_paths(out_dir, variant["id"], cell_id, seed)
    if valid_existing(array_path, metadata_path, definition_hash=definition_hash):
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    if array_path.exists() or metadata_path.exists():
        raise ResidualGraphDeemError(
            f"incomplete or mismatched existing artifact (refusing overwrite): {metadata_path}"
        )

    bundle = load_target_free_bundle(bundle_dir / f"{cell_id}.npz")
    registered = registry_cell(registry, cell_id)
    if len(bundle.row_ids) != int(registered["n_rows"]):
        raise ResidualGraphDeemError(f"registered row-count mismatch: {cell_id}")
    if bundle.inventory_sha256 != registered["inventory_sha256"]:
        raise ResidualGraphDeemError(f"registered inventory mismatch: {cell_id}")
    X_risk, _, transform = donor_risk_matrix(
        bundle.X_raw, bundle.X_raw, bundle.feature_names
    )
    baseline_state, baseline_score, _, baseline_audit = load_frozen_b3(
        baseline_dir, cell_id, seed, X_risk, bundle.feature_names
    )
    moe_config = DeemMoEConfig(**variant.get("config", {}))
    result = fit_deem_b3_moe(
        X_risk,
        bundle.feature_names,
        baseline_state,
        seed=seed,
        config=moe_config,
    )
    if not result.health["healthy"]:
        raise ResidualGraphDeemError(f"unhealthy DEEM-B3 MoE fit: {variant['id']}/{cell_id}/{seed}")
    base_family = np.column_stack(
        [result.base_family_contributions[name] for name in result.family_order]
    )
    routed_family = np.column_stack(
        [result.family_contributions[name] for name in result.family_order]
    )
    arrays = {
        "score": np.asarray(result.score, dtype=np.float64),
        "posterior": np.asarray(result.posterior, dtype=np.float64),
        "logit": np.asarray(result.logit, dtype=np.float64),
        "gates": np.asarray(result.gates, dtype=np.float64),
        "router_probabilities": np.asarray(result.router_probabilities, dtype=np.float64),
        "router_logits": np.asarray(result.router_logits, dtype=np.float64),
        "interaction": np.asarray(result.interaction, dtype=np.float64),
        "family_state_input": np.asarray(result.family_state_input, dtype=np.float64),
        "family_state_output": np.asarray(result.family_state_output, dtype=np.float64),
        "family_state_delta": np.asarray(result.family_state_delta, dtype=np.float64),
        "base_family_contributions": np.asarray(base_family, dtype=np.float64),
        "routed_family_contributions": np.asarray(routed_family, dtype=np.float64),
        "family_order": np.asarray(result.family_order, dtype=str),
        "feature_names": np.asarray(result.feature_names, dtype=str),
        "standardization_mean": np.asarray(transform.mean, dtype=np.float64),
        "standardization_scale": np.asarray(transform.scale, dtype=np.float64),
        "baseline_score": np.asarray(baseline_score, dtype=np.float64),
    }
    for name, value in result.state.items():
        arrays[f"state__{name}"] = np.asarray(value)
    array_hash = atomic_save_npz(array_path, **arrays)
    record = {
        "schema": "deem_b3_moe_fit_artifact_v1",
        "status": "complete",
        "experiment_id": experiment_id,
        "variant_id": str(variant["id"]),
        "variant_role": str(variant.get("role", "")),
        "cell_id": cell_id,
        "dataset_family": bundle.dataset_family,
        "task_type": bundle.task_type,
        "seed": int(seed),
        "n_rows": len(bundle.row_ids),
        "n_features": len(bundle.feature_names),
        "bundle_sha256": bundle.bundle_sha256,
        "inventory_sha256": bundle.inventory_sha256,
        "ordered_row_id_sha256": canonical_sha256(list(bundle.row_ids)),
        "run_definition_sha256": definition_hash,
        "array_path": array_path.relative_to(out_dir).as_posix(),
        "array_sha256": array_hash,
        "config": jsonable(result.config),
        "config_sha256": canonical_sha256(result.config),
        "orientation": int(result.orientation),
        "aligned_bias": float(result.aligned_bias),
        "health": jsonable(result.health),
        "diagnostics": jsonable(result.diagnostics),
        "objective_history": jsonable(result.objective_history),
        "baseline": baseline_audit,
        "score_pearson_vs_same_seed_b3": float(
            np.corrcoef(result.score, baseline_score)[0, 1]
        ),
        "targets_accessed_during_fit": False,
    }
    record["content_sha256"] = canonical_sha256(record)
    atomic_write_json(metadata_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variants", default="all", help="all or comma-separated IDs")
    parser.add_argument("--cells", default="all", help="all, screen, or comma-separated IDs")
    parser.add_argument("--seeds", default="0", help="comma-separated integer seeds")
    args = parser.parse_args()

    config = load_config(args.config)
    registry = load_registry(args.registry)
    variants = variant_lookup(config)
    selected_variants = parse_selection(args.variants, list(variants))
    all_cells = [row["cell_id"] for row in registry["cells"]]
    selected_cells = parse_selection(
        args.cells, all_cells, screen=list(config.get("screen_cells", []))
    )
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    allowed_seeds = set(int(value) for value in config["baseline"]["seeds"])
    if not seeds or not set(seeds).issubset(allowed_seeds) or len(seeds) != len(set(seeds)):
        raise ValueError(f"seeds must be unique members of {sorted(allowed_seeds)}")

    definition = {
        "schema": "deem_b3_moe_run_definition_v1",
        "experiment_id": config["experiment_id"],
        "status": config["status"],
        "config_sha256": sha256_file(args.config),
        "registry_content_sha256": registry["registry_content_sha256"],
        "source_sha256": source_hash(args.config),
        "baseline_score_freeze_sha256": sha256_file(
            args.baseline_dir / "SCORE_FREEZE_MANIFEST.json"
        ),
        "scientific_boundary": config["scientific_boundary"],
    }
    definition["content_sha256"] = canonical_sha256(definition)
    definition_hash = canonical_sha256(definition)
    definition_path = args.out_dir / "run_definitions" / f"{definition_hash}.json"
    if definition_path.is_file():
        if json.loads(definition_path.read_text(encoding="utf-8")) != jsonable(definition):
            raise ResidualGraphDeemError("existing DEEM-B3 MoE run definition mismatch")
    else:
        atomic_write_json(definition_path, definition)

    records = []
    for variant_id in selected_variants:
        for cell_id in selected_cells:
            for seed in seeds:
                print(f"[{variant_id}] {cell_id} seed={seed}", flush=True)
                records.append(
                    fit_one(
                        bundle_dir=args.bundle_dir,
                        baseline_dir=args.baseline_dir,
                        out_dir=args.out_dir,
                        registry=registry,
                        variant=variants[variant_id],
                        seed=seed,
                        cell_id=cell_id,
                        definition_hash=definition_hash,
                        experiment_id=str(config["experiment_id"]),
                    )
                )
    invocation = {
        "run_definition_sha256": definition_hash,
        "selected_variants": selected_variants,
        "selected_cells": selected_cells,
        "selected_seeds": seeds,
    }
    invocation_hash = canonical_sha256(invocation)
    complete = {
        "schema": "deem_b3_moe_fit_complete_v1",
        "status": "complete",
        "run_definition_sha256": definition_hash,
        "invocation_sha256": invocation_hash,
        "n_records": len(records),
        "variants": selected_variants,
        "cells": selected_cells,
        "seeds": seeds,
        "all_healthy": all(row.get("health", {}).get("healthy") for row in records),
        "artifact_sha256": {
            f"{row['variant_id']}|{row['cell_id']}|{row['seed']}": row["array_sha256"]
            for row in records
        },
    }
    complete["content_sha256"] = canonical_sha256(complete)
    atomic_write_json(args.out_dir / "fit_completions" / f"{invocation_hash}.json", complete)
    print(
        f"complete: {len(records)} fit artifacts; definition={definition_hash}; "
        f"invocation={invocation_hash}"
    )


if __name__ == "__main__":
    main()
