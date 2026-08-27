#!/usr/bin/env python3
"""Label-free score-bank fit for Pooled Graph-Roughness Direction V1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    DEFAULT_BUNDLE,
    family as dataset_family,
    load_contract,
    validate_bundle_without_labels,
)
from scripts.inscope_cells import INSCOPE  # noqa: E402
from spectral_utils.family_residual_graph import (  # noqa: E402
    fit_family_residual_state,
    graphs_from_coordinates,
)
from spectral_utils.pooled_graph_roughness import (  # noqa: E402
    apply_pooled_roughness,
    fit_pooled_roughness_calibration,
    graph_roughness_moment,
)


VERSION = "pooled-graph-roughness-direction-v2-2026-08-23"
DEFAULT_OUT = REPO / "results" / "pooled_graph_roughness_direction_v2"
SPEC = REPO / "docs" / "experiments" / "POOLED_GRAPH_ROUGHNESS_DIRECTION_V1.md"
EXCLUDED_MIN_POSITIVE = "spilled_triviaqa_llama8b"
ELIGIBLE_CELLS = tuple(cell for cell in INSCOPE if cell != EXCLUDED_MIN_POSITIVE)
K = 7
LAMBDAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
TRUST_FACTORS = (0.5, 1.0, 2.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def code(value: float) -> str:
    return f"{int(round(100 * float(value))):05d}"


def candidate_key(lambda_: float, trust_factor: float) -> str:
    return f"l{code(lambda_)}__t{code(trust_factor)}"


def candidates():
    return tuple(
        (lambda_, trust_factor)
        for lambda_ in LAMBDAS
        for trust_factor in TRUST_FACTORS
    )


def source_hashes() -> dict[str, str]:
    paths = {
        "fit_script": Path(__file__),
        "report_script": REPO / "scripts" / "pooled_graph_roughness_report.py",
        "mechanical_test_script": REPO / "scripts" / "test_pooled_graph_roughness.py",
        "core_module": REPO / "spectral_utils" / "pooled_graph_roughness.py",
        "family_graph_module": REPO / "spectral_utils" / "family_residual_graph.py",
        "graph_topology_module": REPO / "spectral_utils" / "graph_topology.py",
        "laplacian_module": REPO / "spectral_utils" / "laplacian_upcr.py",
        "contribution_module": REPO / "spectral_utils" / "contribution_subspace.py",
        "upcr_module": REPO / "spectral_utils" / "upcr.py",
        "family_registry_module": REPO / "spectral_utils" / "specrage_views.py",
        "feature_contract_module": REPO / "spectral_utils" / "dufs_liu_feature_contract.py",
        "base_feature_contract_module": REPO / "spectral_utils" / "feature_contract.py",
        "fusion_utils_module": REPO / "spectral_utils" / "fusion_utils.py",
        "contract_loader_script": REPO / "scripts" / "hard_filter_dufs_liu_benchmark.py",
        "roster_script": REPO / "scripts" / "inscope_cells.py",
        "spec": SPEC,
        "family_nrm_reference": (
            REPO / "results" / "neutral_residual_mode_cs_iu_v1"
            / "cell_results.csv"
        ),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def run_definition(bundle: Path) -> dict:
    payload = {
        "version": VERSION,
        "status": "retrospective_reconstruction",
        "bundle": str(Path(bundle).resolve()),
        "bundle_sha256": sha256_file(bundle),
        "eligible_cells": list(ELIGIBLE_CELLS),
        "excluded_historical_min_positive_cell": EXCLUDED_MIN_POSITIVE,
        "dataset_families": {
            cell: dataset_family(cell) for cell in ELIGIBLE_CELLS
        },
        "graph": {
            "coordinates": "standardized_family_residuals",
            "topology": "duplicate_safe_symmetric_union_knn",
            "k": K,
            "tie_keys": "literal_row_index",
        },
        "pooling": "equal_cell_within_dataset_family_then_equal_family",
        "lambdas": list(LAMBDAS),
        "trust_factors": list(TRUST_FACTORS),
        "candidate_count": len(candidates()),
        "labels_accessed_by_fit": False,
        "target_fields_received_by_fit": [],
        "source_hashes": source_hashes(),
    }
    payload["definition_hash"] = canonical_hash(payload)
    return payload


def load_label_free_cells(bundle: Path):
    output = []
    with np.load(bundle, allow_pickle=True) as data:
        validate_bundle_without_labels(data)
        for index, cell in enumerate(ELIGIBLE_CELLS, start=1):
            print(f"[{index}/{len(ELIGIBLE_CELLS)}] moment {cell}", flush=True)
            F, names = load_contract(data, cell, "mixed_v2")
            state = fit_family_residual_state(F, names)
            families = tuple(state.contribution_space.families)
            graph = graphs_from_coordinates(
                state.residuals,
                (K,),
                topology="union",
                tie_keys=np.arange(F.shape[1], dtype=float),
            )[K]
            moment = graph_roughness_moment(
                state.baseline, state.residuals, families, graph
            )
            identity_cross = np.asarray(
                state.residuals.T @ state.baseline / len(state.baseline),
                dtype=float,
            )
            output.append({
                "cell": cell,
                "group": dataset_family(cell),
                "baseline": np.asarray(state.baseline, dtype=float),
                "residuals": np.asarray(state.residuals, dtype=float),
                "families": families,
                "moment": moment,
                "identity_cross_max_abs": float(np.max(np.abs(identity_cross))),
                "state_diagnostics": state.diagnostics,
                "n_features": int(F.shape[0]),
            })
    return output


def calibration_cache(cells):
    groups = tuple(sorted({cell["group"] for cell in cells}))
    exclusions = {()}
    exclusions.update((group,) for group in groups)
    exclusions.update(
        tuple(sorted((left, right)))
        for index, left in enumerate(groups)
        for right in groups[index + 1:]
    )
    cache = {}
    for excluded in sorted(exclusions, key=lambda value: (len(value), value)):
        source = [cell for cell in cells if cell["group"] not in excluded]
        for lambda_ in LAMBDAS:
            cache[(excluded, lambda_)] = fit_pooled_roughness_calibration(
                [cell["moment"] for cell in source],
                [cell["group"] for cell in source],
                lambda_,
                pooling="equal_group",
            )
    return cache


def fit(args):
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    (args.out / "states").mkdir()
    (args.out / "scores").mkdir()
    definition = run_definition(args.bundle)
    write_json(args.out / "RUN_DEFINITION.json", definition)
    cells = load_label_free_cells(args.bundle)
    groups = tuple(sorted({cell["group"] for cell in cells}))
    if len(cells) != 23 or len(groups) != 8:
        raise RuntimeError("development roster must contain 23 cells / 8 families")
    cache = calibration_cache(cells)

    calibration_payload = {}
    for (excluded, lambda_), calibration in cache.items():
        key = f"exclude={'+'.join(excluded) if excluded else 'none'}__lambda={lambda_:g}"
        calibration_payload[key] = {
            "excluded_groups": list(excluded),
            "lambda": lambda_,
            "source_groups": list(calibration.source_groups),
            "A": calibration.A.tolist(),
            "c": calibration.c.tolist(),
            "direction": calibration.direction.tolist(),
            "diagnostics": calibration.diagnostics,
        }
    write_json(args.out / "CALIBRATIONS.json", calibration_payload)

    state_hashes, score_hashes = {}, {}
    max_identity_error = 0.0
    for index, cell in enumerate(cells, start=1):
        print(f"[{index}/{len(cells)}] scores {cell['cell']}", flush=True)
        state_path = args.out / "states" / f"{cell['cell']}.npz"
        np.savez_compressed(
            state_path,
            baseline=cell["baseline"],
            residuals=cell["residuals"],
            families=np.asarray(cell["families"]),
            moment_A=cell["moment"].A,
            moment_c=cell["moment"].c,
            moment_presence=cell["moment"].presence,
        )
        state_hashes[cell["cell"]] = sha256_file(state_path)
        max_identity_error = max(
            max_identity_error, cell["identity_cross_max_abs"]
        )
        values = {
            "iu": np.asarray(cell["baseline"], dtype=np.float64),
            "sample_index": np.arange(len(cell["baseline"]), dtype=np.int64),
        }
        target_group = cell["group"]
        for lambda_, trust in candidates():
            candidate = candidate_key(lambda_, trust)
            full = cache[((), lambda_)]
            outer = cache[((target_group,), lambda_)]
            values[f"full__{candidate}"] = apply_pooled_roughness(
                cell["baseline"], cell["residuals"], cell["families"],
                full, trust,
            ).score
            values[f"outer__{candidate}"] = apply_pooled_roughness(
                cell["baseline"], cell["residuals"], cell["families"],
                outer, trust,
            ).score
            for outer_group in groups:
                if outer_group == target_group:
                    continue
                excluded = tuple(sorted((outer_group, target_group)))
                inner = cache[(excluded, lambda_)]
                values[f"inner={outer_group}__{candidate}"] = (
                    apply_pooled_roughness(
                        cell["baseline"], cell["residuals"],
                        cell["families"], inner, trust,
                    ).score
                )
        score_path = args.out / "scores" / f"{cell['cell']}.npz"
        np.savez_compressed(score_path, **values)
        score_hashes[cell["cell"]] = sha256_file(score_path)

    diagnostics = {
        "version": VERSION,
        "n_cells": len(cells),
        "n_groups": len(groups),
        "groups": list(groups),
        "max_identity_no_laplacian_cross_abs": max_identity_error,
        "cells": [{
            "cell": cell["cell"],
            "group": cell["group"],
            "n": len(cell["baseline"]),
            "n_features": cell["n_features"],
            "families": list(cell["families"]),
            "identity_cross_max_abs": cell["identity_cross_max_abs"],
            "moment": cell["moment"].diagnostics,
            "state": cell["state_diagnostics"],
        } for cell in cells],
    }
    write_json(args.out / "DIAGNOSTICS.json", diagnostics)
    complete = {
        "version": VERSION,
        "definition_hash": definition["definition_hash"],
        "labels_accessed_by_fit": False,
        "target_fields_received_by_fit": [],
        "state_hashes": state_hashes,
        "score_hashes": score_hashes,
        "calibrations_sha256": sha256_file(args.out / "CALIBRATIONS.json"),
        "diagnostics_sha256": sha256_file(args.out / "DIAGNOSTICS.json"),
    }
    complete["manifest_hash"] = canonical_hash(complete)
    write_json(args.out / "FIT_COMPLETE.json", complete)
    print(json.dumps({
        "status": "label_free_score_bank_frozen",
        "n_cells": len(cells),
        "n_candidates": len(candidates()),
        "max_identity_cross_abs": max_identity_error,
        "manifest_hash": complete["manifest_hash"],
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    fit(args)


if __name__ == "__main__":
    main()
