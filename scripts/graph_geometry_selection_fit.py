#!/usr/bin/env python3
"""Fit and freeze the target-free graph-geometry score bank.

The process accepts only a physically sanitized NPZ whose member whitelist is
validated before any array is read.  It constructs graphs, freezes intrinsic
selections, fits every leave-family-out calibration, and hashes every
geometry/lambda/trust score.  Correctness outcomes are neither present nor
accepted by this command.
"""

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
    DUFS_EPOCHS,
    DUFS_SEEDS,
    family as dataset_family,
    load_contract,
)
from scripts.inscope_cells import INSCOPE  # noqa: E402
from spectral_utils.family_residual_graph import fit_family_residual_state  # noqa: E402
from spectral_utils.graph_geometry_selection import (  # noqa: E402
    GEOMETRIES,
    INTRINSIC_LAMBDA,
    INTRINSIC_RULE_VERSION,
    INTRINSIC_TRUST,
    aggregate_geometry_similarity,
    build_geometry_graph,
    choose_intrinsic_geometry,
    deduplicate_geometries,
    graph_energy,
    graph_health,
    geometry_base_coordinates,
    intrinsic_geometry_summary,
    perturbation_diagnostics,
    phase_a_geometry_ids,
    selector_geometry_ids,
    stable_rng,
    validate_physically_label_free_members,
)
from spectral_utils.laplacian_upcr import dufs_soft_gates  # noqa: E402
from spectral_utils.pooled_graph_roughness import (  # noqa: E402
    GraphRoughnessMoment,
    apply_pooled_roughness,
    direction_cosine,
    fit_pooled_roughness_calibration,
    graph_roughness_moment,
    pool_graph_roughness_moments,
)


VERSION = "graph-geometry-selection-research-v1-2026-08-23"
DEFAULT_OUT = REPO / "results" / "graph_geometry_selection_research_v1" / "development_fit"
DEFAULT_BUNDLE = (
    REPO / "results" / "graph_geometry_selection_research_v1"
    / "label_free_input" / "cells_target_free.npz"
)
ORIGINAL_BUNDLE = REPO / "results" / "dependency_fusion_raw" / "cells.npz"
SPEC = REPO / "docs" / "experiments" / "GRAPH_GEOMETRY_SELECTION_RESEARCH_V1.md"
EXCLUDED_MIN_POSITIVE = "spilled_triviaqa_llama8b"
ELIGIBLE_CELLS = tuple(cell for cell in INSCOPE if cell != EXCLUDED_MIN_POSITIVE)
LAMBDAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
TRUST_FACTORS = (0.25, 0.5, 1.0, 2.0)
TRUST_CLASSES = {
    "canonical": (0.5, 1.0, 2.0),
    "v1": (0.25, 0.5, 1.0),
    "expanded": TRUST_FACTORS,
}
ACTUATORS = ("full", "cross")
N_NODE_PERMUTATIONS = 20
NODE_CONTROL_LAMBDA = 0.03
NODE_CONTROL_TRUST = 0.5

LAUNCH_HASHES = {
    "pooled_spec": (
        REPO / "docs" / "experiments" / "POOLED_GRAPH_ROUGHNESS_DIRECTION_V1.md",
        "56d0f90ade36a7e5f2df31b286a64e0cfd4dce14c58a009b339147d735f93af8",
    ),
    "pooled_report": (
        REPO / "results" / "pooled_graph_roughness_direction_v2" / "REPORT.md",
        "ad15df53a5f399d5727665a02aa6dd13230ffdb14064c6810a7cb50f2da05185",
    ),
    "pooled_result": (
        REPO / "results" / "pooled_graph_roughness_direction_v2" / "RESULT.json",
        "afd81cf5f3bf50fce2d7e4e312c194604928aa06300c922ea8014c960000484b",
    ),
    "pooled_selection": (
        REPO / "results" / "pooled_graph_roughness_direction_v2" / "FROZEN_SELECTION.json",
        "ff0b6e824d0140b7e5fbdab0d10f97b7a32ff80217d6b740915436c5ce8d1aa3",
    ),
    "pooled_controls_report": (
        REPO / "results" / "pooled_graph_roughness_direction_v2" / "controls" / "REPORT.md",
        "8f4263a73017a0179995acdaaadb7fb3852011af406336f139b285c6d5e5982a",
    ),
    "su_v1_spec": (
        REPO / "docs" / "experiments" / "SU_POOLED_GRAPH_ADAPTATION_SIDECAR_V1.md",
        "814ab6a0e9d7babd21cc17dfafe547d1c5eaac403142b8a16c3f9e857bf51ef9",
    ),
    "su_v1_report": (
        REPO / "results" / "su_pooled_graph_adaptation_sidecar_v1" / "REPORT.md",
        "f48f57864141b31abe2809d8b1a26d45ae05485533940b5e4dae56f2f981a77a",
    ),
    "su_v2_spec": (
        REPO / "docs" / "experiments" / "SU_POOLED_GRAPH_ADAPTATION_CONSERVATIVE_V2.md",
        "ca8d80dc92466016ce1623710eed1680aca3ebe173efc55c9493abf0dad8e365",
    ),
    "su_v2_report": (
        REPO / "results" / "su_pooled_graph_adaptation_conservative_v2" / "REPORT.md",
        "7d213a47d06a7b2fedbfbfb833b8579f9450438bca2de4f88548346e503af6d8",
    ),
    "pooled_module_at_launch": (
        REPO / "spectral_utils" / "pooled_graph_roughness.py",
        "d33dd89e61fb44d56f4c6e26b89a4e4835e542c17d8f7c8d7deefa99d6c1eb61",
    ),
    "family_module_at_launch": (
        REPO / "spectral_utils" / "family_residual_graph.py",
        "f07f05e41fe8de275045fe6ae018e9a1254398c2c615a5cefb9bb060f9f38ba9",
    ),
}


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


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(json.dumps(value.shape, separators=(",", ":")).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def bundle_member_hash(value: np.ndarray) -> str:
    """Match the physical-isolation manifest's dtype-preserving member hash."""
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    if value.dtype == object:
        digest.update(json.dumps(
            [str(item) for item in value.tolist()],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode())
    else:
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def code(value: float) -> str:
    return f"{int(round(100 * float(value))):05d}"


def calibration_key(
    geometry_id: str, excluded, lambda_: float | None, actuator: str = "full"
) -> str:
    exclusion = "+".join(sorted(excluded)) if excluded else "none"
    lambda_code = f"{lambda_:g}" if lambda_ is not None else "direction_only"
    return f"g={geometry_id}__a={actuator}__exclude={exclusion}__l={lambda_code}"


def basis_key(
    prefix: str, geometry_id: str, lambda_: float | None, actuator: str = "full"
) -> str:
    lambda_code = code(lambda_) if lambda_ is not None else "direction_only"
    return f"{prefix}__a={actuator}__g={geometry_id}__l={lambda_code}"


def candidate_key(
    geometry_id: str,
    lambda_: float | None,
    trust: float,
    actuator: str = "full",
) -> str:
    lambda_code = code(lambda_) if lambda_ is not None else "direction_only"
    return f"a={actuator}__g={geometry_id}__l={lambda_code}__t={code(trust)}"


def node_basis_key(replicate: int, geometry_id: str, actuator: str) -> str:
    lambda_code = (
        code(NODE_CONTROL_LAMBDA) if actuator == "full" else "direction_only"
    )
    return (
        f"control=nodeperm{int(replicate):02d}__outer__a={actuator}"
        f"__g={geometry_id}__l={lambda_code}"
    )


def node_score_key(replicate: int, geometry_id: str, actuator: str) -> str:
    return (
        f"{node_basis_key(replicate, geometry_id, actuator)}"
        f"__t={code(NODE_CONTROL_TRUST)}"
    )


def launch_hash_audit() -> dict:
    output = {}
    for name, (path, expected) in LAUNCH_HASHES.items():
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"canonical launch artifact changed: {name}: {observed} != {expected}"
            )
        output[name] = {"path": str(path.resolve()), "sha256": observed}
    return output


def source_hashes() -> dict[str, str]:
    paths = {
        "fit_script": Path(__file__),
        "report_script": REPO / "scripts" / "graph_geometry_selection_report.py",
        "test_script": REPO / "scripts" / "test_graph_geometry_selection.py",
        "isolation_script": REPO / "scripts" / "build_graph_geometry_label_free_bundle.py",
        "core_module": REPO / "spectral_utils" / "graph_geometry_selection.py",
        "pooled_module": REPO / "spectral_utils" / "pooled_graph_roughness.py",
        "family_graph_module": REPO / "spectral_utils" / "family_residual_graph.py",
        "graph_topology_module": REPO / "spectral_utils" / "graph_topology.py",
        "laplacian_module": REPO / "spectral_utils" / "laplacian_upcr.py",
        "dufs_trainer_module": REPO / "spectral_utils" / "selectors" / "a2_groupfs.py",
        "mixed_feature_contract_module": REPO / "spectral_utils" / "dufs_liu_feature_contract.py",
        "base_feature_contract_module": REPO / "spectral_utils" / "feature_contract.py",
        "family_registry_module": REPO / "spectral_utils" / "specrage_views.py",
        "contribution_module": REPO / "spectral_utils" / "contribution_subspace.py",
        "upcr_module": REPO / "spectral_utils" / "upcr.py",
        "fusion_utils_module": REPO / "spectral_utils" / "fusion_utils.py",
        "family_nrm_reference": (
            REPO / "results" / "neutral_residual_mode_cs_iu_v1"
            / "cell_results.csv"
        ),
        "contract_loader": REPO / "scripts" / "hard_filter_dufs_liu_benchmark.py",
        "roster": REPO / "scripts" / "inscope_cells.py",
        "spec": SPEC,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def verify_isolated_bundle(bundle: Path) -> dict:
    manifest_path = bundle.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != "graph-geometry-physical-label-isolation-v1-2026-08-23":
        raise RuntimeError("physical-isolation manifest version changed")
    if manifest.get("output_path") != str(bundle.resolve()):
        raise RuntimeError("physical-isolation output path changed")
    if manifest.get("allowed_suffixes") != ["V", "pool", "hand_signs"]:
        raise RuntimeError("physical-isolation suffix whitelist changed")
    if manifest.get("output_member_count") != 3 * len(INSCOPE):
        raise RuntimeError("physical-isolation member count changed")
    if sha256_file(bundle) != manifest["output_sha256"]:
        raise RuntimeError("physically label-free bundle hash changed")
    if manifest.get("source_target_arrays_loaded") is not False:
        raise RuntimeError("isolation manifest does not certify target non-access")
    if manifest.get("output_contains_target_like_members") is not False:
        raise RuntimeError("isolated bundle contains target-like members")
    with np.load(bundle, allow_pickle=True) as data:
        validate_physically_label_free_members(data.files, INSCOPE)
        if set(manifest.get("output_member_hashes", {})) != set(data.files):
            raise RuntimeError("physical-isolation member-hash registry changed")
        for key in data.files:
            if bundle_member_hash(data[key]) != manifest["output_member_hashes"][key]:
                raise RuntimeError(f"physical-isolation member changed: {key}")
    return manifest


def run_definition(bundle: Path) -> dict:
    isolated = verify_isolated_bundle(bundle)
    payload = {
        "version": VERSION,
        "status": "retrospective_opened-development_geometry_identification",
        "spec": str(SPEC.resolve()),
        "physically_label_free_bundle": str(bundle.resolve()),
        "physically_label_free_bundle_sha256": sha256_file(bundle),
        "isolation_manifest_sha256": sha256_file(bundle.with_suffix(".manifest.json")),
        "original_bundle_sha256": isolated["source_sha256"],
        "eligible_cells": list(ELIGIBLE_CELLS),
        "excluded_historical_min_positive_cell": EXCLUDED_MIN_POSITIVE,
        "dataset_families": {cell: dataset_family(cell) for cell in ELIGIBLE_CELLS},
        "geometries": [spec.__dict__ for spec in GEOMETRIES],
        "phase_a_geometry_ids": list(phase_a_geometry_ids()),
        "phase_b_selector_geometry_ids": list(selector_geometry_ids()),
        "control_only_geometry_ids": ["dufs_union_k7"],
        "lambda_grid": list(LAMBDAS),
        "trust_grid_union": list(TRUST_FACTORS),
        "trust_classes": {key: list(value) for key, value in TRUST_CLASSES.items()},
        "selector_tail_floor_auroc": -0.005,
        "actuators": {
            "full": "-lambda*(I+lambda*Abar)^-1*cbar",
            "cross": "-cbar; lambda absent because score scaling identifies direction only",
            "selector_can_choose_actuator": False,
        },
        "node_permutations_per_geometry": N_NODE_PERMUTATIONS,
        "intrinsic_rule_version": INTRINSIC_RULE_VERSION,
        "intrinsic_fixed_calibration": {
            "lambda": INTRINSIC_LAMBDA,
            "trust": INTRINSIC_TRUST,
        },
        "pooling": "equal_cell_within_dataset_family_then_equal_family",
        "duplicate_handling": "distance_then_literal_row_index",
        "deduplication_thresholds": {
            "mean_edge_jaccard": 0.995,
            "mean_operator_cosine": 0.999,
        },
        "labels_accessed_by_fit": False,
        "target_fields_physically_present_in_fit_input": False,
        "target_fields_received_by_fit": [],
        "su_covariance_or_rho_arms": [],
        "launch_hash_audit": launch_hash_audit(),
        "source_hashes": source_hashes(),
    }
    payload["definition_hash"] = canonical_hash(payload)
    return payload


def load_label_free_cells(bundle: Path):
    cells = []
    graphs_by_cell = {}
    with np.load(bundle, allow_pickle=True) as data:
        validate_physically_label_free_members(data.files, INSCOPE)
        for index, cell in enumerate(ELIGIBLE_CELLS, start=1):
            print(f"[{index}/{len(ELIGIBLE_CELLS)}] geometries {cell}", flush=True)
            F, names = load_contract(data, cell, "mixed_v2")
            state = fit_family_residual_state(F, names)
            gates, gate_diag = dufs_soft_gates(
                F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
            )
            length = (
                np.asarray(F[names.index("trace_length")], dtype=float)
                if "trace_length" in names else None
            )
            cell_graphs = {}
            moments = {}
            health = {}
            perturb = {}
            coordinates_by_geometry = {}
            node_permutation_moments = {}
            permutations = [
                stable_rng(VERSION, "node", replicate, cell).permutation(F.shape[1])
                for replicate in range(N_NODE_PERMUTATIONS)
            ]
            for spec in GEOMETRIES:
                graph, coordinates = build_geometry_graph(
                    spec, F, state, dufs_gates=gates,
                    tie_keys=np.arange(F.shape[1], dtype=float),
                )
                base_coordinates = geometry_base_coordinates(
                    spec, F, state, dufs_gates=gates
                )
                cell_graphs[spec.geometry_id] = graph
                coordinates_by_geometry[spec.geometry_id] = coordinates
                moments[spec.geometry_id] = graph_roughness_moment(
                    state.baseline,
                    state.residuals,
                    tuple(state.contribution_space.families),
                    graph,
                )
                health[spec.geometry_id] = graph_health(graph)
                perturb[spec.geometry_id] = perturbation_diagnostics(
                    spec, base_coordinates, graph, cell=cell
                )
                node_permutation_moments[spec.geometry_id] = [
                    graph_roughness_moment(
                        state.baseline,
                        state.residuals,
                        tuple(state.contribution_space.families),
                        graph[permutation][:, permutation],
                    )
                    for permutation in permutations
                ]
            length_ratios = {}
            if length is None:
                length_ratios = {
                    spec.geometry_id: None for spec in GEOMETRIES
                }
            else:
                canonical_energy = graph_energy(
                    cell_graphs["residual_union_k7"], length
                )
                for spec in GEOMETRIES:
                    energy = graph_energy(cell_graphs[spec.geometry_id], length)
                    if canonical_energy <= 1e-12:
                        ratio = 1.0 if energy <= 1e-12 else None
                    else:
                        ratio = energy / canonical_energy
                    length_ratios[spec.geometry_id] = (
                        None if ratio is None else float(ratio)
                    )
            identity_cross = np.asarray(
                state.residuals.T @ state.baseline / len(state.baseline),
                dtype=float,
            )
            cells.append({
                "cell": cell,
                "group": dataset_family(cell),
                "baseline": np.asarray(state.baseline, dtype=float),
                "residuals": np.asarray(state.residuals, dtype=float),
                "families": tuple(state.contribution_space.families),
                "moments": moments,
                "node_permutation_moments": node_permutation_moments,
                "graph_diagnostics": health,
                "perturbation_diagnostics": perturb,
                "length_energy_ratio": length_ratios,
                "identity_cross_max_abs": float(np.max(np.abs(identity_cross))),
                "state_diagnostics": state.diagnostics,
                "dufs_gate_diagnostics": {
                    "effective_feature_count": float(gate_diag["effective_feature_count"]),
                    "mean_probability": float(gate_diag["mean_probability"]),
                    "mean_seed_std": float(gate_diag["mean_seed_std"]),
                },
                "n_features": int(F.shape[0]),
            })
            graphs_by_cell[cell] = cell_graphs
    return cells, graphs_by_cell


def exclusion_sets(groups):
    output = {()}
    output.update((group,) for group in groups)
    output.update(
        tuple(sorted((left, right)))
        for index, left in enumerate(groups)
        for right in groups[index + 1:]
    )
    return tuple(sorted(output, key=lambda value: (len(value), value)))


def fit_calibrations(cells, active_geometry_ids, groups):
    cache = {}
    for geometry_index, geometry_id in enumerate(active_geometry_ids, start=1):
        print(
            f"[{geometry_index}/{len(active_geometry_ids)}] calibrations {geometry_id}",
            flush=True,
        )
        for excluded in exclusion_sets(groups):
            source = [cell for cell in cells if cell["group"] not in excluded]
            for lambda_ in LAMBDAS:
                cache[("full", geometry_id, excluded, lambda_)] = (
                    fit_pooled_roughness_calibration(
                        [cell["moments"][geometry_id] for cell in source],
                        [cell["group"] for cell in source],
                        lambda_,
                        pooling="equal_group",
                    )
                )
            cache[("cross", geometry_id, excluded, None)] = (
                fit_pooled_roughness_calibration(
                    [cell["moments"][geometry_id] for cell in source],
                    [cell["group"] for cell in source],
                    1.0,
                    pooling="equal_group",
                    cross_only=True,
                )
            )
    return cache


def actuator_diagnostics(cells, active_geometry_ids, groups):
    """Target-free diagnostics isolating A-curvature from the cross gradient."""
    contexts = {"all_source": ()}
    contexts.update({f"outer_held={group}": (group,) for group in groups})
    output = {"node_permutations": N_NODE_PERMUTATIONS, "contexts": {}}
    for context_name, excluded in contexts.items():
        selected_cells = [cell for cell in cells if cell["group"] not in excluded]
        context_rows = {}
        for geometry_id in active_geometry_ids:
            moments = [cell["moments"][geometry_id] for cell in selected_cells]
            source_groups = [cell["group"] for cell in selected_cells]
            Abar, cbar, used_groups = pool_graph_roughness_moments(
                moments, source_groups, pooling="equal_group"
            )
            group_c = []
            for group in used_groups:
                group_c.append(np.mean([
                    moment.c for moment, source_group in zip(moments, source_groups)
                    if source_group == group
                ], axis=0))
            group_c = np.asarray(group_c, dtype=float)
            leave_one_cosines = []
            for held_group in used_groups:
                leave_moments = [
                    moment for moment, source_group in zip(moments, source_groups)
                    if source_group != held_group
                ]
                leave_groups = [
                    source_group for source_group in source_groups
                    if source_group != held_group
                ]
                _, leave_c, _ = pool_graph_roughness_moments(
                    leave_moments, leave_groups, pooling="equal_group"
                )
                leave_one_cosines.append(direction_cosine(cbar, leave_c))
            full_cross_cosines = {}
            for lambda_ in LAMBDAS:
                direction = -lambda_ * np.linalg.solve(
                    np.eye(len(cbar)) + lambda_ * Abar, cbar
                )
                full_cross_cosines[f"{lambda_:g}"] = direction_cosine(direction, -cbar)
            permuted_c = []
            for replicate in range(N_NODE_PERMUTATIONS):
                permuted_moments = [
                    cell["node_permutation_moments"][geometry_id][replicate]
                    for cell in selected_cells
                ]
                _, value, _ = pool_graph_roughness_moments(
                    permuted_moments, source_groups, pooling="equal_group"
                )
                permuted_c.append(value)
            permuted_c = np.asarray(permuted_c, dtype=float)
            permuted_mean = np.mean(permuted_c, axis=0)
            distances = np.linalg.norm(permuted_c - cbar[None, :], axis=1)
            null_spread = np.linalg.norm(
                permuted_c - permuted_mean[None, :], axis=1
            )
            context_rows[geometry_id] = {
                "source_groups": list(used_groups),
                "cbar": cbar.tolist(),
                "cbar_norm": float(np.linalg.norm(cbar)),
                "group_c_dispersion_relative": float(
                    np.mean(np.linalg.norm(group_c - cbar[None, :], axis=1))
                    / max(np.linalg.norm(cbar), 1e-12)
                ),
                "leave_one_source_c_cosine_min": float(np.min(leave_one_cosines)),
                "leave_one_source_c_cosine_mean": float(np.mean(leave_one_cosines)),
                "full_vs_cross_direction_cosine_by_lambda": full_cross_cosines,
                "node_permuted_c_norm_mean": float(np.mean(np.linalg.norm(permuted_c, axis=1))),
                "real_c_minus_permuted_mean_norm": float(np.linalg.norm(cbar - permuted_mean)),
                "real_c_distance_to_permutations_mean": float(np.mean(distances)),
                "node_permutation_null_spread_mean": float(np.mean(null_spread)),
                "real_minus_permuted_mean_separation_ratio": float(
                    np.linalg.norm(cbar - permuted_mean)
                    / max(float(np.mean(null_spread)), 1e-12)
                ),
                "real_vs_node_permuted_c_cosines": [
                    direction_cosine(cbar, value) for value in permuted_c
                ],
            }
        output["contexts"][context_name] = context_rows
    return output


def fit_node_permutation_controls(cells, active_geometry_ids, groups):
    """Freeze outer-only node-permutation controls at canonical strength."""
    cache = {}
    payload = {
        "protocol": {
            "outer_only": True,
            "lambda_full": NODE_CONTROL_LAMBDA,
            "lambda_cross": None,
            "trust_factor": NODE_CONTROL_TRUST,
            "selector_used": False,
            "actuator_selected": False,
            "replicates": N_NODE_PERMUTATIONS,
        },
        "calibrations": {},
    }
    for held in groups:
        source = [cell for cell in cells if cell["group"] != held]
        for geometry_id in active_geometry_ids:
            for replicate in range(N_NODE_PERMUTATIONS):
                moments = [
                    cell["node_permutation_moments"][geometry_id][replicate]
                    for cell in source
                ]
                source_groups = [cell["group"] for cell in source]
                full = fit_pooled_roughness_calibration(
                    moments, source_groups, NODE_CONTROL_LAMBDA,
                    pooling="equal_group",
                )
                cross = fit_pooled_roughness_calibration(
                    moments, source_groups, 1.0,
                    pooling="equal_group", cross_only=True,
                )
                for actuator, calibration in (("full", full), ("cross", cross)):
                    cache[(held, geometry_id, replicate, actuator)] = calibration
                    key = (
                        f"held={held}__g={geometry_id}__r={replicate:02d}"
                        f"__a={actuator}"
                    )
                    payload["calibrations"][key] = {
                        "held_group": held,
                        "geometry_id": geometry_id,
                        "replicate": replicate,
                        "actuator": actuator,
                        "direction_families": list(calibration.families),
                        "source_groups": list(calibration.source_groups),
                        "direction": calibration.direction.tolist(),
                        "A": calibration.A.tolist(),
                        "c": calibration.c.tolist(),
                    }
    return cache, payload


def freeze_intrinsic_selector(cells, active_geometry_ids, groups, fit_hash=None):
    contexts = {"all_source": ()}
    contexts.update({f"outer_held={group}": (group,) for group in groups})
    selections = {}
    for name, excluded in contexts.items():
        summaries = [
            intrinsic_geometry_summary(
                cells, geometry_id, excluded_groups=excluded
            )
            for geometry_id in active_geometry_ids
        ]
        selected, diagnostics = choose_intrinsic_geometry(summaries)
        selections[name] = {
            "excluded_groups": list(excluded),
            "selected_geometry": selected,
            "fixed_lambda": INTRINSIC_LAMBDA,
            "fixed_trust": INTRINSIC_TRUST,
            "diagnostics": diagnostics,
            "all_geometry_summaries": summaries,
        }
    payload = {
        "version": VERSION,
        "rule_version": INTRINSIC_RULE_VERSION,
        "outcome_labels_accessed": False,
        "selection_frozen_by_fit": True,
        "calibration_policy": (
            "use the canonical frozen conservative lambda/trust; intrinsic "
            "criteria select geometry only"
        ),
        "active_geometry_ids": list(active_geometry_ids),
        "contexts": selections,
    }
    payload["selection_hash"] = canonical_hash(payload)
    return payload


def fit(args) -> None:
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    (args.out / "states").mkdir()
    (args.out / "score_basis").mkdir()
    definition = run_definition(args.bundle)
    write_json(args.out / "RUN_DEFINITION.json", definition)

    cells, graphs_by_cell = load_label_free_cells(args.bundle)
    groups = tuple(sorted({cell["group"] for cell in cells}))
    if len(cells) != 23 or len(groups) != 8:
        raise RuntimeError("development roster must contain exactly 23 cells / 8 families")
    geometry_ids = tuple(spec.geometry_id for spec in GEOMETRIES)
    similarity = aggregate_geometry_similarity(graphs_by_cell, geometry_ids)
    active_geometry_ids, duplicate_of = deduplicate_geometries(
        similarity, geometry_ids
    )
    missing_phase_a = sorted(set(phase_a_geometry_ids()) - set(active_geometry_ids))
    if missing_phase_a:
        raise RuntimeError(f"Phase-A geometry unexpectedly deduplicated: {missing_phase_a}")
    diversity = {
        "candidate_geometry_count": len(geometry_ids),
        "effective_geometry_count": len(active_geometry_ids),
        "active_geometry_ids": list(active_geometry_ids),
        "duplicate_of": duplicate_of,
        "thresholds": definition["deduplication_thresholds"],
        "pairwise": similarity,
    }
    write_json(args.out / "GRAPH_DIVERSITY.json", diversity)

    selector_active_geometry_ids = tuple(
        geometry_id for geometry_id in selector_geometry_ids()
        if geometry_id in active_geometry_ids
    )
    intrinsic = freeze_intrinsic_selector(
        cells, selector_active_geometry_ids, groups
    )
    write_json(args.out / "FROZEN_LABELFREE_SELECTION.json", intrinsic)
    calibrations = fit_calibrations(cells, active_geometry_ids, groups)
    calibration_payload = {}
    for (actuator, geometry_id, excluded, lambda_), calibration in calibrations.items():
        calibration_payload[calibration_key(
            geometry_id, excluded, lambda_, actuator
        )] = {
            "actuator": actuator,
            "geometry_id": geometry_id,
            "excluded_groups": list(excluded),
            "lambda": lambda_,
            "source_groups": list(calibration.source_groups),
            "direction_families": list(calibration.families),
            "A": calibration.A.tolist(),
            "c": calibration.c.tolist(),
            "direction": calibration.direction.tolist(),
            "diagnostics": calibration.diagnostics,
        }
    write_json(args.out / "CALIBRATIONS.json", calibration_payload)
    actuator_payload = actuator_diagnostics(
        cells, active_geometry_ids, groups
    )
    write_json(args.out / "ACTUATOR_DIAGNOSTICS.json", actuator_payload)
    node_controls, node_control_payload = fit_node_permutation_controls(
        cells, active_geometry_ids, groups
    )
    write_json(
        args.out / "NODE_PERMUTATION_CONTROLS.json",
        node_control_payload,
    )

    state_hashes = {}
    basis_hashes = {}
    score_hashes = {}
    max_identity_error = 0.0
    for index, cell in enumerate(cells, start=1):
        name = cell["cell"]
        print(f"[{index}/{len(cells)}] score basis {name}", flush=True)
        state_values = {
            "baseline": cell["baseline"],
            "residuals": cell["residuals"],
            "families": np.asarray(cell["families"]),
        }
        for geometry_id in active_geometry_ids:
            moment = cell["moments"][geometry_id]
            state_values[f"moment_A__{geometry_id}"] = moment.A
            state_values[f"moment_c__{geometry_id}"] = moment.c
            state_values[f"moment_presence__{geometry_id}"] = moment.presence
        state_path = args.out / "states" / f"{name}.npz"
        np.savez_compressed(state_path, **state_values)
        state_hashes[name] = sha256_file(state_path)

        target_group = cell["group"]
        basis = {
            "iu": np.asarray(cell["baseline"], dtype=np.float64),
            "sample_index": np.arange(len(cell["baseline"]), dtype=np.int64),
        }
        hashes = {"iu": array_hash(cell["baseline"])}
        prefixes = [("full", ()) , ("outer", (target_group,))]
        prefixes.extend(
            (f"inner={outer_group}", tuple(sorted((outer_group, target_group))))
            for outer_group in groups if outer_group != target_group
        )
        for prefix, excluded in prefixes:
            for geometry_id in active_geometry_ids:
                for lambda_ in LAMBDAS:
                    calibration = calibrations[("full", geometry_id, excluded, lambda_)]
                    correction = apply_pooled_roughness(
                        cell["baseline"], cell["residuals"], cell["families"],
                        calibration, 1.0,
                    ).correction
                    key = basis_key(prefix, geometry_id, lambda_, "full")
                    basis[key] = np.asarray(correction, dtype=np.float64)
                    for trust in TRUST_FACTORS:
                        candidate = candidate_key(geometry_id, lambda_, trust, "full")
                        hashes[f"{prefix}__{candidate}"] = array_hash(
                            cell["baseline"] + trust * correction
                        )
                cross_calibration = calibrations[("cross", geometry_id, excluded, None)]
                cross_correction = apply_pooled_roughness(
                    cell["baseline"], cell["residuals"], cell["families"],
                    cross_calibration, 1.0,
                ).correction
                cross_key = basis_key(prefix, geometry_id, None, "cross")
                basis[cross_key] = np.asarray(cross_correction, dtype=np.float64)
                for trust in TRUST_FACTORS:
                    candidate = candidate_key(geometry_id, None, trust, "cross")
                    hashes[f"{prefix}__{candidate}"] = array_hash(
                        cell["baseline"] + trust * cross_correction
                    )
        for geometry_id in active_geometry_ids:
            for replicate in range(N_NODE_PERMUTATIONS):
                for actuator in ACTUATORS:
                    calibration = node_controls[
                        (target_group, geometry_id, replicate, actuator)
                    ]
                    correction = apply_pooled_roughness(
                        cell["baseline"], cell["residuals"], cell["families"],
                        calibration, 1.0,
                    ).correction
                    key = node_basis_key(replicate, geometry_id, actuator)
                    basis[key] = np.asarray(correction, dtype=np.float64)
                    hashes[node_score_key(
                        replicate, geometry_id, actuator
                    )] = array_hash(
                        cell["baseline"] + NODE_CONTROL_TRUST * correction
                    )
        basis_path = args.out / "score_basis" / f"{name}.npz"
        np.savez_compressed(basis_path, **basis)
        basis_hashes[name] = sha256_file(basis_path)
        score_hashes[name] = hashes
        max_identity_error = max(max_identity_error, cell["identity_cross_max_abs"])

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
            "state": cell["state_diagnostics"],
            "dufs_gates": cell["dufs_gate_diagnostics"],
            "graph": cell["graph_diagnostics"],
            "perturbation": cell["perturbation_diagnostics"],
            "length_energy_ratio": cell["length_energy_ratio"],
        } for cell in cells],
    }
    write_json(args.out / "DIAGNOSTICS.json", diagnostics)
    write_json(args.out / "SCORE_HASHES.json", score_hashes)
    complete = {
        "version": VERSION,
        "definition_hash": definition["definition_hash"],
        "labels_accessed_by_fit": False,
        "target_fields_physically_present_in_fit_input": False,
        "target_fields_received_by_fit": [],
        "active_geometry_ids": list(active_geometry_ids),
        "selector_geometry_ids": list(selector_active_geometry_ids),
        "candidate_score_count": int(sum(len(value) - 1 for value in score_hashes.values())),
        "state_hashes": state_hashes,
        "basis_hashes": basis_hashes,
        "calibrations_sha256": sha256_file(args.out / "CALIBRATIONS.json"),
        "diagnostics_sha256": sha256_file(args.out / "DIAGNOSTICS.json"),
        "actuator_diagnostics_sha256": sha256_file(
            args.out / "ACTUATOR_DIAGNOSTICS.json"
        ),
        "node_permutation_controls_sha256": sha256_file(
            args.out / "NODE_PERMUTATION_CONTROLS.json"
        ),
        "diversity_sha256": sha256_file(args.out / "GRAPH_DIVERSITY.json"),
        "label_free_selection_sha256": sha256_file(
            args.out / "FROZEN_LABELFREE_SELECTION.json"
        ),
        "score_hashes_sha256": sha256_file(args.out / "SCORE_HASHES.json"),
    }
    complete["manifest_hash"] = canonical_hash(complete)
    write_json(args.out / "FIT_COMPLETE.json", complete)
    print(json.dumps({
        "status": "physically_isolated_label_free_score_bank_frozen",
        "manifest_hash": complete["manifest_hash"],
        "n_cells": len(cells),
        "candidate_geometries": len(geometry_ids),
        "effective_geometries": len(active_geometry_ids),
        "candidate_scores_hashed": complete["candidate_score_count"],
        "intrinsic_selection_hash": intrinsic["selection_hash"],
        "max_identity_cross_abs": max_identity_error,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    fit(args)


if __name__ == "__main__":
    main()
