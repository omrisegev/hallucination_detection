#!/usr/bin/env python3
"""Freeze B3 + true-LOO-residual + PGRD family-expert scores label-free."""

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
    FAMILY_ORDER,
    build_residual_cell,
    graph_roughness_moment,
    load_frozen_b3_ensemble,
    pooled_graph_roughness_direction,
    score_graph_roughness_direction,
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


DEFAULT_CONFIG = ROOT / "configs/deem_b3_residual_pgrd_v1.json"
DEFAULT_REGISTRY = ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"
SOURCES = (
    ROOT / "spectral_utils/deem_b3_residual_moe.py",
    ROOT / "scripts/run_deem_b3_residual_pgrd_v1.py",
)


def load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "deem_b3_residual_pgrd_v1_config":
        raise ResidualGraphDeemError("PGRD config schema mismatch")
    variants = value.get("variants", [])
    identifiers = [str(row.get("id")) for row in variants]
    if not variants or len(identifiers) != len(set(identifiers)):
        raise ResidualGraphDeemError("empty or duplicated PGRD roster")
    for row in variants:
        settings = row.get("config", {})
        if settings.get("pooling") not in {"within_cell", "donor_family"}:
            raise ResidualGraphDeemError(f"invalid PGRD pooling: {row['id']}")
        if settings.get("residual_source") not in {"baseline", "loo"}:
            raise ResidualGraphDeemError(f"invalid PGRD residual source: {row['id']}")
        if int(settings.get("iterations", 1)) < 1:
            raise ResidualGraphDeemError("PGRD iterations must be positive")
    return value


def _source_hash(config_path: Path) -> str:
    payload = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in SOURCES}
    payload["config"] = sha256_file(config_path)
    return canonical_sha256(payload)


def _variants(value: str, config: dict) -> list[dict]:
    lookup = {str(row["id"]): row for row in config["variants"]}
    if value == "all":
        return list(config["variants"])
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested or not set(requested).issubset(set(lookup)):
        raise ValueError("invalid PGRD variant selection")
    return [lookup[item] for item in requested]


def _is_zero_trust(settings: dict) -> bool:
    if "trust_factor" in settings:
        return float(settings["trust_factor"]) == 0.0
    if "trust" in settings and settings["trust"] is not None:
        return float(settings["trust"]) == 0.0
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variants", default="all")
    args = parser.parse_args()

    config = load_config(args.config)
    registry = load_registry(args.registry)
    variants = _variants(args.variants, config)
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
    base_records = {
        cell: build_residual_cell(
            ensembles[cell],
            baseline_score=initial_scores[cell],
            folds=grouped_folds[cell],
        )
        for cell in cells
    }
    required_base_sources = sorted(
        {
            str(row["config"]["residual_source"])
            for row in variants
            if not _is_zero_trust(row["config"])
        }
    )
    base_moments = {
        source: {
            cell: graph_roughness_moment(
                base_records[cell],
                bundles[cell].row_ids,
                residual_source=source,
            )
            for cell in cells
        }
        for source in required_base_sources
    }
    source_sha256 = _source_hash(args.config)
    run_definition = {
        "schema": "deem_b3_residual_pgrd_run_definition_v1",
        "status": "running",
        "experiment_id": config["experiment_id"],
        "source_sha256": source_sha256,
        "config_sha256": sha256_file(args.config),
        "registry_sha256": sha256_file(args.registry),
        "bundle_dir": str(args.bundle_dir.resolve()),
        "baseline_dir": str(args.baseline_dir.resolve()),
        "variants": [str(row["id"]) for row in variants],
        "cells": cells,
        "dataset_families": dataset_families,
        "seeds": list(seeds),
        "targets_accessed_during_fit": False,
        "labels_module_imported": False,
        "contains_within_cell_transductive_variants": any(
            row["config"]["pooling"] == "within_cell" for row in variants
        ),
        "contains_donor_family_exclusion_variants": any(
            row["config"]["pooling"] == "donor_family" for row in variants
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out_dir / "RUN_DEFINITION.json", run_definition)

    manifest = []
    for held_family in dataset_families:
        held_cells = [cell for cell in cells if family_by_cell[cell] == held_family]
        donor_cells = [cell for cell in cells if family_by_cell[cell] != held_family]
        for variant in variants:
            variant_id = str(variant["id"])
            settings = dict(variant["config"])
            iterations = int(settings.pop("iterations", 1))
            pooling = str(settings.pop("pooling"))
            residual_source = str(settings.pop("residual_source"))
            current_scores = {
                cell: np.asarray(initial_scores[cell], dtype=float).copy()
                for cell in cells
            }
            final_records = None
            final_results = None
            final_diagnostics = None
            final_directions = None
            final_moments = None
            iteration_diagnostics = []
            for iteration in range(iterations):
                active_cells = (
                    cells
                    if pooling == "donor_family" and iterations > 1
                    else held_cells
                )
                if iteration == 0:
                    records = base_records
                else:
                    records = {
                        cell: build_residual_cell(
                            ensembles[cell],
                            baseline_score=current_scores[cell],
                            folds=grouped_folds[cell],
                        )
                        for cell in cells
                    }
                identity_only = _is_zero_trust(settings)
                if identity_only:
                    # The mechanical control must not depend on graph health or
                    # on a nonzero cross-gradient.  It still traverses the same
                    # scoring function, but with an inert direction.
                    moments = None
                    directions = {
                        cell: np.zeros(len(FAMILY_ORDER), dtype=float)
                        for cell in active_cells
                    }
                    diagnostics = {
                        cell: {
                            "identity_only": True,
                            "n_moments": 0,
                            "residual_source": residual_source,
                            "uses_labels": False,
                        }
                        for cell in active_cells
                    }
                else:
                    moment_cells = cells if pooling == "within_cell" else donor_cells
                    if iteration == 0:
                        moments = {
                            cell: base_moments[residual_source][cell]
                            for cell in moment_cells
                        }
                    else:
                        moments = {
                            cell: graph_roughness_moment(
                                records[cell], bundles[cell].row_ids,
                                residual_source=residual_source,
                            )
                            for cell in moment_cells
                        }
                    directions = {}
                    diagnostics = {}
                    if pooling == "within_cell":
                        for cell in active_cells:
                            directions[cell], diagnostics[cell] = (
                                pooled_graph_roughness_direction([moments[cell]])
                            )
                    else:
                        direction, diagnostic = pooled_graph_roughness_direction(
                            [moments[cell] for cell in donor_cells],
                            [family_by_cell[cell] for cell in donor_cells],
                        )
                        directions = {cell: direction for cell in active_cells}
                        diagnostics = {cell: diagnostic for cell in active_cells}
                results = {
                    cell: score_graph_roughness_direction(
                        records[cell],
                        directions[cell],
                        residual_source=residual_source,
                        **settings,
                    )
                    for cell in active_cells
                }
                current_scores.update(
                    {cell: results[cell].score for cell in active_cells}
                )
                iteration_diagnostics.append(
                    {
                        "iteration": iteration + 1,
                        "pooling": pooling,
                        "residual_source": residual_source,
                        "donor_cells": donor_cells if pooling == "donor_family" else [],
                        "calibration_by_cell": diagnostics,
                    }
                )
                final_records = records
                final_results = results
                final_diagnostics = diagnostics
                final_directions = directions
                final_moments = moments
            assert final_records is not None
            assert final_results is not None
            assert final_diagnostics is not None
            assert final_directions is not None

            for cell in held_cells:
                result = final_results[cell]
                record = final_records[cell]
                baseline = initial_scores[cell]
                if result.score.shape != baseline.shape or not np.isfinite(
                    result.score
                ).all():
                    raise ResidualGraphDeemError(
                        f"invalid PGRD score: {variant_id}/{cell}"
                    )
                if _is_zero_trust(variant["config"]) and not np.array_equal(
                    result.score, baseline
                ):
                    raise ResidualGraphDeemError(f"PGRD B3 identity failed: {cell}")
                moment = (
                    None
                    if final_moments is None
                    else final_moments.get(cell)
                )
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
                    loo_seed_instability=np.asarray(record.loo_seed_instability, dtype=np.float64),
                    loo_predictability=np.asarray(record.loo_predictability, dtype=np.float64),
                    calibration_direction=np.asarray(
                        final_directions[cell], dtype=np.float64
                    ),
                    graph_moment_a0=np.asarray(
                        np.zeros((len(FAMILY_ORDER), len(FAMILY_ORDER)))
                        if moment is None else moment.a0,
                        dtype=np.float64,
                    ),
                    graph_moment_c0=np.asarray(
                        np.zeros(len(FAMILY_ORDER)) if moment is None else moment.c0,
                        dtype=np.float64,
                    ),
                    graph_moment_present=np.asarray(
                        record.present_mask if moment is None else moment.present_mask,
                        dtype=np.int8,
                    ),
                )
                graph_diagnostic = final_diagnostics[cell]
                metadata = {
                    "schema": "deem_b3_residual_pgrd_score_v1",
                    "status": "complete",
                    "experiment_id": config["experiment_id"],
                    "variant_id": variant_id,
                    "cell_id": cell,
                    "dataset_family": held_family,
                    "config": variant["config"],
                    "iterations": iterations,
                    "pooling": pooling,
                    "residual_source": residual_source,
                    "donor_cells": donor_cells if pooling == "donor_family" else [],
                    "held_family_cells": held_cells,
                    "source_sha256": source_sha256,
                    "array_sha256": array_sha256,
                    "iteration_diagnostics": [
                        {
                            **{
                                key: value
                                for key, value in row.items()
                                if key != "calibration_by_cell"
                            },
                            "calibration": row["calibration_by_cell"][cell],
                        }
                        for row in iteration_diagnostics
                    ],
                    "score_diagnostics": result.diagnostics,
                    "residual_diagnostics": record.diagnostics,
                    "baseline_diagnostics": ensembles[cell].diagnostics,
                    "targets_accessed_during_fit": False,
                    "labels_module_imported": False,
                    "calibration_excludes_entire_target_dataset_family": (
                        pooling == "donor_family"
                    ),
                    "within_cell_transductive_graph": pooling == "within_cell",
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
        raise AssertionError(f"PGRD manifest has {len(manifest)} rows, expected {expected}")
    run_definition["status"] = "complete"
    run_definition["n_score_artifacts"] = len(manifest)
    run_definition["manifest_sha256"] = canonical_sha256(manifest)
    atomic_write_json(args.out_dir / "SCORE_FREEZE_MANIFEST.json", manifest)
    atomic_write_json(args.out_dir / "RUN_DEFINITION.json", run_definition)
    print(json.dumps(run_definition, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
