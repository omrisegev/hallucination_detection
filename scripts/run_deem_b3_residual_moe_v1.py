#!/usr/bin/env python3
"""Freeze label-free B3 residual-family MoE scores before evaluation.

This process intentionally cannot import the label-sidecar module.  Every
target cell is calibrated from cells outside its entire dataset family; the
cell itself is used only for its transductive, label-free B3 residual transform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.deem_b3_residual_moe import (  # noqa: E402
    build_residual_cell,
    fit_residual_calibration,
    load_frozen_b3_ensemble,
    score_residual_moe,
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
    load_registry,
    load_target_free_bundle,
)


DEFAULT_CONFIG = ROOT / "configs/deem_b3_residual_moe_v1.json"
DEFAULT_REGISTRY = ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"
SOURCES = (
    ROOT / "spectral_utils/deem_b3_residual_moe.py",
    ROOT / "scripts/run_deem_b3_residual_moe_v1.py",
)


def load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "deem_b3_residual_moe_v1_config":
        raise ResidualGraphDeemError("residual-MoE config schema mismatch")
    variants = value.get("variants", [])
    identifiers = [str(row.get("id")) for row in variants]
    if not variants or len(identifiers) != len(set(identifiers)):
        raise ResidualGraphDeemError("empty or duplicated residual-MoE roster")
    allowed = {
        "iterations", "trust", "survival", "gate_strength",
        "novelty_strength", "stability_strength", "temperature",
        "dufs_epochs",
    }
    for row in variants:
        config = row.get("config", {})
        unknown = sorted(set(config) - allowed)
        if unknown:
            raise ResidualGraphDeemError(
                f"unknown residual-MoE config fields for {row['id']}: {unknown}"
            )
        if int(config.get("iterations", 1)) < 1:
            raise ResidualGraphDeemError("iterations must be positive")
    boundary = value.get("scientific_boundary", {})
    if not boundary.get("fit_is_label_free") or not boundary.get(
        "natural_24cell_targets_previously_opened"
    ):
        raise ResidualGraphDeemError("scientific boundary is incomplete")
    return value


def _source_hash(config_path: Path) -> str:
    payload = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in SOURCES}
    payload["config"] = sha256_file(config_path)
    return canonical_sha256(payload)


def _parse_variants(value: str, config: dict) -> list[dict]:
    lookup = {str(row["id"]): row for row in config["variants"]}
    if value == "all":
        return list(config["variants"])
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(lookup))
    if unknown:
        raise ValueError("unknown variants: " + ", ".join(unknown))
    return [lookup[item] for item in requested]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variants", default="all")
    args = parser.parse_args()

    config = load_config(args.config)
    registry = load_registry(args.registry)
    variants = _parse_variants(args.variants, config)
    cells = [str(row["cell_id"]) for row in registry["cells"]]
    family_by_cell = {
        str(row["cell_id"]): str(row["dataset_family"])
        for row in registry["cells"]
    }
    dataset_families = sorted(set(family_by_cell.values()))
    seeds = tuple(int(seed) for seed in config["baseline"]["seeds"])
    bundles = {
        cell: load_target_free_bundle(args.bundle_dir / f"{cell}.npz")
        for cell in cells
    }
    ensembles = {
        cell: load_frozen_b3_ensemble(
            args.baseline_dir,
            cell,
            seeds=seeds,
            expected_bundle_sha256=bundles[cell].bundle_sha256,
            expected_ordered_row_id_sha256=canonical_sha256(
                list(bundles[cell].row_ids)
            ),
        )
        for cell in cells
    }
    grouped_folds = {
        cell: assign_grouped_length_folds(
            bundles[cell].group_ids, bundles[cell].raw_trace_length
        )
        for cell in cells
    }
    initial_scores = {cell: ensembles[cell].score for cell in cells}
    source_sha256 = _source_hash(args.config)
    run_definition = {
        "schema": "deem_b3_residual_moe_run_definition_v1",
        "status": "running",
        "experiment_id": config["experiment_id"],
        "source_sha256": source_sha256,
        "config_sha256": sha256_file(args.config),
        "registry_sha256": sha256_file(args.registry),
        "baseline_dir": str(args.baseline_dir.resolve()),
        "bundle_dir": str(args.bundle_dir.resolve()),
        "variants": [str(row["id"]) for row in variants],
        "cells": cells,
        "dataset_families": dataset_families,
        "seeds": list(seeds),
        "targets_accessed_during_fit": False,
        "labels_module_imported": False,
        "calibration_excludes_entire_target_dataset_family": True,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out_dir / "RUN_DEFINITION.json", run_definition)

    manifest = []
    for held_family in dataset_families:
        held_cells = [cell for cell in cells if family_by_cell[cell] == held_family]
        donor_cells = [cell for cell in cells if family_by_cell[cell] != held_family]
        if set(held_cells).intersection(donor_cells):
            raise AssertionError("dataset-family calibration overlap")
        for variant in variants:
            variant_id = str(variant["id"])
            settings = dict(variant.get("config", {}))
            iterations = int(settings.pop("iterations", 1))
            dufs_epochs = int(settings.pop("dufs_epochs", 120))
            survival = str(settings.pop("survival", "uniform"))
            current_scores = {
                cell: np.asarray(initial_scores[cell], dtype=float).copy()
                for cell in cells
            }
            final_records = None
            final_calibration = None
            final_results = None
            iteration_diagnostics = []
            for iteration in range(iterations):
                records = {
                    cell: build_residual_cell(
                        ensembles[cell],
                        baseline_score=current_scores[cell],
                        folds=grouped_folds[cell],
                    )
                    for cell in cells
                }
                calibration = fit_residual_calibration(
                    [records[cell] for cell in donor_cells],
                    survival=survival,
                    dufs_epochs=dufs_epochs,
                )
                results = {
                    cell: score_residual_moe(
                        records[cell], calibration, **settings
                    )
                    for cell in cells
                }
                current_scores = {cell: results[cell].score for cell in cells}
                iteration_diagnostics.append(
                    {
                        "iteration": iteration + 1,
                        "calibration": calibration.diagnostics,
                        "direction": calibration.direction,
                        "family_survival": calibration.family_survival,
                    }
                )
                final_records = records
                final_calibration = calibration
                final_results = results
            assert final_records is not None
            assert final_calibration is not None
            assert final_results is not None

            for cell in held_cells:
                result = final_results[cell]
                record = final_records[cell]
                baseline = initial_scores[cell]
                if result.score.shape != baseline.shape or not np.isfinite(result.score).all():
                    raise ResidualGraphDeemError(f"invalid residual-MoE score: {variant_id}/{cell}")
                if float(variant["config"].get("trust", 0.0)) == 0.0 and not np.array_equal(
                    result.score, baseline
                ):
                    raise ResidualGraphDeemError(f"B3 identity failed: {variant_id}/{cell}")
                directory = args.out_dir / "scores" / variant_id / cell
                array_path = directory / f"{variant_id}.npz"
                array_sha256 = atomic_save_npz(
                    array_path,
                    score=np.asarray(result.score, dtype=np.float64),
                    baseline_score=np.asarray(baseline, dtype=np.float64),
                    logit=np.asarray(result.logit, dtype=np.float64),
                    correction_z=np.asarray(result.correction_z, dtype=np.float64),
                    raw_correction=np.asarray(result.raw_correction, dtype=np.float64),
                    expert_terms=np.asarray(result.expert_terms, dtype=np.float64),
                    gate_probabilities=np.asarray(result.gate_probabilities, dtype=np.float64),
                    gates=np.asarray(result.gates, dtype=np.float64),
                    residuals=np.asarray(record.residuals, dtype=np.float64),
                    seed_instability=np.asarray(record.seed_instability, dtype=np.float64),
                    loo_residuals=np.asarray(record.loo_residuals, dtype=np.float64),
                    loo_seed_instability=np.asarray(
                        record.loo_seed_instability, dtype=np.float64
                    ),
                    loo_predictability=np.asarray(
                        record.loo_predictability, dtype=np.float64
                    ),
                    calibration_direction=np.asarray(final_calibration.direction, dtype=np.float64),
                    family_survival=np.asarray(final_calibration.family_survival, dtype=np.float64),
                )
                metadata = {
                    "schema": "deem_b3_residual_moe_score_v1",
                    "status": "complete",
                    "experiment_id": config["experiment_id"],
                    "variant_id": variant_id,
                    "cell_id": cell,
                    "dataset_family": held_family,
                    "donor_cells": donor_cells,
                    "held_family_cells": held_cells,
                    "source_sha256": source_sha256,
                    "array_sha256": array_sha256,
                    "config": variant["config"],
                    "iterations": iterations,
                    "iteration_diagnostics": iteration_diagnostics,
                    "score_diagnostics": result.diagnostics,
                    "residual_diagnostics": record.diagnostics,
                    "baseline_diagnostics": ensembles[cell].diagnostics,
                    "targets_accessed_during_fit": False,
                    "labels_module_imported": False,
                    "calibration_excludes_entire_target_dataset_family": True,
                }
                metadata["content_sha256"] = canonical_sha256(metadata)
                metadata_path = array_path.with_suffix(".json")
                atomic_write_json(metadata_path, metadata)
                manifest.append(
                    {
                        "variant_id": variant_id,
                        "cell_id": cell,
                        "dataset_family": held_family,
                        "array_path": str(array_path.relative_to(args.out_dir)),
                        "array_sha256": array_sha256,
                        "metadata_path": str(metadata_path.relative_to(args.out_dir)),
                        "metadata_sha256": sha256_file(metadata_path),
                    }
                )

    expected = len(variants) * len(cells)
    if len(manifest) != expected:
        raise AssertionError(f"score manifest has {len(manifest)} rows, expected {expected}")
    run_definition["status"] = "complete"
    run_definition["n_score_artifacts"] = len(manifest)
    run_definition["manifest_sha256"] = canonical_sha256(manifest)
    atomic_write_json(args.out_dir / "SCORE_FREEZE_MANIFEST.json", manifest)
    atomic_write_json(args.out_dir / "RUN_DEFINITION.json", run_definition)
    print(json.dumps(run_definition, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
