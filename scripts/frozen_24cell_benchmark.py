#!/usr/bin/env python3
"""Fit the frozen 24-cell unsupervised fusion benchmark.

This file deliberately does not compute AUROC, AUPRC, or any other label-based
quantity.  It writes one label-free score checkpoint per registered cell.  The
separate ``frozen_24cell_report.py`` program verifies and freezes every score
file before it opens the labels and builds the evaluation report.

The scientific run always uses the exact roster in ``scripts.inscope_cells``.
``--debug-cells`` and ``--debug-cell`` exist only for execution tests and write
a manifest marked ``scientific_run=false``; such output cannot be consumed by
the report script.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
import platform
import sys
import time
import types

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from scripts.inscope_cells import GROUP, INSCOPE                 # noqa: E402
from spectral_utils.laplacian_upcr import (                      # noqa: E402
    build_graph_from_features,
    dufs_soft_gates,
    laplacian_iu_path,
    symmetric_normalized_laplacian,
)
from spectral_utils.fusion_aware_views import (                  # noqa: E402
    DEFAULT_CLUSTER_BOOTSTRAPS,
    DEFAULT_GRAPH_K as IMPACT_GRAPH_K,
    DEFAULT_K_VALUES as MICRO_K_VALUES,
    DEFAULT_MAX_SAMPLES as IMPACT_MAX_SAMPLES,
    DEFAULT_SAMPLE_FRACTION as IMPACT_SAMPLE_FRACTION,
    DEFAULT_SAMPLE_REPLICATES as IMPACT_SAMPLE_REPLICATES,
    SCHEMA_VERSION as FUSION_VIEW_SCHEMA_VERSION,
    build_view_schemas,
    cell_impact_profile,
    learn_loco_micro_partition,
)
from spectral_utils.specrage_laplacian import (                  # noqa: E402
    SpecRaGEConfig,
    fit_specrage_graph,
    graph_for_control,
)
from spectral_utils.specrage_views import (                      # noqa: E402
    VIEW_SCHEMA_VERSION,
    fixed_stable_from_bundle,
    provenance_views,
    view_members,
)
from spectral_utils.upcr import upcr_fit                         # noqa: E402


VERSION = "frozen-24cell-unsupervised-fusion-v1-2026-08-07"
DEFAULT_BUNDLE = os.path.join(
    REPO, "results", "dependency_fusion_raw", "cells.npz"
)
DEFAULT_OUT = os.path.join(REPO, "results", "frozen_24cell_benchmark")

# These values are registered before the all-cell result is opened.  Lambda=10
# is the CA-SpecRaGE synthetic-transfer value.  The other values are reported as
# a sensitivity path; the report never selects a winner from this path.
LAMBDAS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
FROZEN_LAMBDA = {
    "dufs_liu": 0.1,
    "adapted_specrage_y_liu": 10.0,
    "ca_specrage_alpha_liu": 10.0,
}

MODEL_SEEDS = (11, 23)
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
DUFS_K = 7
GRAPH_K = 15

CA_CONFIG = SpecRaGEConfig(
    output_dim=2,
    n_neighbors=GRAPH_K,
    temperature=1.0,
    learning_rate=1e-2,
    batch_size=128,
    max_epochs=60,
    min_epochs=60,
    patience=61,
    lr_patience=20,
    encoder_hidden=(32,),
    fusion_hidden=(50,),
    checkpoint_mode="final",
    orthogonalization="svd_floor",
    orthogonal_floor=1e-3,
    agreement_strength=2.0,
    agreement_temperature=0.08,
    edge_mass_strength=0.1,
    view_mass_normalization=True,
    fit_sample_cap=1500,
)

# This is an adapted plain-loss baseline, not a bit-for-bit reproduction of the
# published SpecRaGE experiments.  It uses the paper's unmodified Rayleigh loss,
# temperature used by the prior repository baseline, and exactly the same
# architecture, epochs, seeds, and numerical stabilization as the CA fit.
PLAIN_CONFIG = replace(
    CA_CONFIG,
    temperature=90.0,
    agreement_strength=0.0,
    edge_mass_strength=0.0,
)
UNIFORM_CONFIG = replace(PLAIN_CONFIG, fusion_mode="uniform")

DEPLOYED_FIT = {
    "loss": "l2",
    "exclusion": True,
    "difficulty_gate": False,
    "simple_avg_fallback": True,
    "recompute_after_exclusion": True,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
}

GRAPH_ARMS = (
    "dufs_liu",
    "raw_uniform_liu",
)
VIEW_SCHEMAS = ("manual", "atomic", "micro")
SCHEMA_GRAPH_INTERFACES = (
    "adapted_specrage_y_liu",
    "ca_specrage_alpha_liu",
    "ca_specrage_y_liu",
    "uniform_y_liu",
    "ca_uniform_alpha_control",
    "ca_global_alpha_control",
    "ca_permuted_alpha_control",
)


def schema_arm(schema: str, interface: str) -> str:
    if schema not in VIEW_SCHEMAS:
        raise ValueError(f"unknown view schema: {schema}")
    if interface not in SCHEMA_GRAPH_INTERFACES:
        raise ValueError(f"unknown graph interface: {interface}")
    return f"{schema}__{interface}"


ALL_GRAPH_ARMS = GRAPH_ARMS + tuple(
    schema_arm(schema, interface)
    for schema in VIEW_SCHEMAS
    for interface in SCHEMA_GRAPH_INTERFACES
)

SOURCE_FILES = (
    "scripts/frozen_24cell_benchmark.py",
    "scripts/frozen_24cell_report.py",
    "scripts/inscope_cells.py",
    "spectral_utils/upcr.py",
    "spectral_utils/laplacian_upcr.py",
    "spectral_utils/fusion_aware_views.py",
    "spectral_utils/specrage_laplacian.py",
    "spectral_utils/specrage_views.py",
    "spectral_utils/feature_contract.py",
    "spectral_utils/selectors/a2_groupfs.py",
    "docs/experiments/FROZEN_24_CELL_BENCHMARK.md",
    "docs/methods/README.md",
    "docs/methods/deployed_upcr.md",
    "docs/methods/iu_pcr.md",
    "docs/methods/dufs_liu.md",
    "docs/methods/adapted_specrage_y_liu.md",
    "docs/methods/ca_specrage_alpha_liu.md",
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_version(distribution: str) -> str:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, np.floating):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        # JSON null means the numerical diagnostic was unavailable.  The exact
        # paths are recorded separately in every cell diagnostic, so this is
        # auditable and never silently converted to a favorable numeric value.
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _nonfinite_paths(value, path="$") -> list[str]:
    """Locate unavailable numerical diagnostics before JSON conversion."""
    output = []
    if isinstance(value, dict):
        for key, item in value.items():
            output.extend(_nonfinite_paths(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            output.extend(_nonfinite_paths(item, f"{path}[{index}]"))
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            for index in np.argwhere(~np.isfinite(value)):
                suffix = "".join(f"[{int(item)}]" for item in index)
                output.append(path + suffix)
        else:
            output.extend(_nonfinite_paths(value.tolist(), path))
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        output.append(path)
    return output


def _finalize_diagnostics(payload: dict) -> dict:
    """Attach nonfinite paths while every nested diagnostic is still numeric."""
    if "nonfinite_diagnostic_paths" in payload:
        raise ValueError("diagnostic payload is already finalized")
    payload = dict(payload)
    payload["nonfinite_diagnostic_paths"] = _nonfinite_paths(payload)
    return payload


def lambda_token(value: float) -> str:
    return format(float(value), ".12g").replace(".", "p")


def score_key(arm: str, lambda_: float) -> str:
    return f"{arm}__lambda_{lambda_token(lambda_)}"


def bundle_cells(data) -> set[str]:
    suffixes = ("__V", "__labels", "__pool", "__hand_signs")
    cells = set()
    for key in data.files:
        for suffix in suffixes:
            if key.endswith(suffix):
                cells.add(key[: -len(suffix)])
                break
    return cells


def validate_bundle(data) -> tuple[str, ...]:
    observed = bundle_cells(data)
    expected = set(INSCOPE)
    if observed != expected:
        raise RuntimeError(
            "bundle cell roster differs from scripts.inscope_cells: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    required_suffixes = ("V", "labels", "pool", "hand_signs")
    missing = [
        f"{cell}__{suffix}"
        for cell in INSCOPE
        for suffix in required_suffixes
        if f"{cell}__{suffix}" not in data.files
    ]
    if missing:
        raise RuntimeError("bundle is missing required arrays: " + ", ".join(missing))
    if len(INSCOPE) != 24 or sum(GROUP[cell] == "QA" for cell in INSCOPE) != 9:
        raise RuntimeError("registered roster is not the expected 24 cells (9 QA, 15 math)")
    return tuple(INSCOPE)


def run_definition(bundle: str, scientific_run: bool, cells: tuple[str, ...]) -> dict:
    payload = {
        "version": VERSION,
        "scientific_run": scientific_run,
        "bundle": os.path.relpath(bundle, REPO),
        "bundle_sha256": sha256_file(bundle),
        "cells": list(cells),
        "domains": {cell: GROUP[cell] for cell in cells},
        "feature_contract": "fixed_stable_v1",
        "view_schema": VIEW_SCHEMA_VERSION,
        "fusion_view_schema": FUSION_VIEW_SCHEMA_VERSION,
        "view_grouping": list(VIEW_SCHEMAS),
        "micro_view_learning": {
            "protocol": "leave-one-cell-out projected-impact clustering",
            "impact_graph_k": IMPACT_GRAPH_K,
            "impact_max_samples": IMPACT_MAX_SAMPLES,
            "impact_sample_replicates": IMPACT_SAMPLE_REPLICATES,
            "impact_sample_fraction": IMPACT_SAMPLE_FRACTION,
            "cluster_bootstraps": DEFAULT_CLUSTER_BOOTSTRAPS,
            "candidate_k": list(MICRO_K_VALUES),
            "labels_used": False,
        },
        "lambdas": list(LAMBDAS),
        "frozen_lambda": FROZEN_LAMBDA,
        "model_seeds": list(MODEL_SEEDS),
        "dufs_seeds": list(DUFS_SEEDS),
        "dufs_epochs": DUFS_EPOCHS,
        "dufs_k": DUFS_K,
        "graph_k": GRAPH_K,
        "ca_config": asdict(CA_CONFIG),
        "plain_config": asdict(PLAIN_CONFIG),
        "uniform_config": asdict(UNIFORM_CONFIG),
        "deployed_fit": DEPLOYED_FIT,
        "graph_arms": list(ALL_GRAPH_ARMS),
        "source_sha256": {
            path: sha256_file(os.path.join(REPO, path)) for path in SOURCE_FILES
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "dependency_versions": {
            distribution: dependency_version(distribution)
            for distribution in ("scipy", "scikit-learn", "torch", "matplotlib")
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["run_fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def _fit_histories(result, prefix: str) -> list[dict]:
    rows = []
    for seed_result in result.seed_results:
        for record in seed_result.history:
            rows.append({
                "fit": prefix,
                "seed": int(seed_result.seed),
                **record,
            })
    return rows


def _schema_fusion_diagnostics(F, result):
    """Measure how differently a schema's view graphs actuate IU-PCR."""
    F = np.asarray(F, dtype=float)
    m, n = F.shape
    covariance = F @ F.T / n
    _, basis = np.linalg.eigh(covariance)
    basis = basis[:, -2:]
    signatures = []
    for graph in result.base_graphs:
        laplacian = symmetric_normalized_laplacian(graph)
        roughness = np.asarray(F @ (laplacian @ F.T) / n, dtype=float)
        projected = basis.T @ (0.5 * (roughness + roughness.T)) @ basis
        projected = 0.5 * (projected + projected.T)
        trace = float(np.trace(projected))
        signatures.append(
            projected / trace if trace > 1e-12 else np.eye(2) / 2.0
        )
    signatures = np.asarray(signatures)
    distances = []
    for left in range(len(signatures)):
        for right in range(left + 1, len(signatures)):
            distances.append(float(
                np.linalg.norm(signatures[left] - signatures[right]) / np.sqrt(2.0)
            ))
    return {
        "projected_roughness_distance_mean": float(np.mean(distances))
            if distances else 0.0,
        "projected_roughness_distance_median": float(np.median(distances))
            if distances else 0.0,
        "projected_roughness_distance_max": float(np.max(distances))
            if distances else 0.0,
    }


def fit_cell(
    cell: str, data, permutation_seed: int, micro_partition, partition_diagnostics
) -> tuple[dict, dict]:
    """Return label-free scores and diagnostics for one cell.

    The labels array is intentionally never read here.
    """
    started = time.time()
    stored = np.asarray(data[f"{cell}__V"], dtype=float)
    names = tuple(str(name) for name in data[f"{cell}__pool"])
    legacy = np.asarray(data[f"{cell}__hand_signs"], dtype=float)
    matrix, stable_names = fixed_stable_from_bundle(stored, names, legacy)
    schemas = build_view_schemas(matrix, stable_names, micro_partition)
    F = matrix.T

    stage_runtime = {}
    stamp = time.time()
    dufs_gates, dufs_diagnostics = dufs_soft_gates(
        F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
    )
    stage_runtime["dufs_seconds"] = time.time() - stamp

    permutation = (
        permutation_seed
        + int(hashlib.sha256(cell.encode("utf-8")).hexdigest()[:8], 16)
    )
    graphs = {
        "dufs_liu": build_graph_from_features(F, gates=dufs_gates, k=DUFS_K),
        "raw_uniform_liu": build_graph_from_features(F, k=GRAPH_K),
    }
    schema_fits = {}
    for schema, definition in schemas.items():
        views = definition["views"]
        prior = definition["prior"]
        stamp = time.time()
        plain = fit_specrage_graph(
            views, config=PLAIN_CONFIG, seeds=MODEL_SEEDS, view_prior=prior
        )
        stage_runtime[f"{schema}_adapted_specrage_seconds"] = time.time() - stamp
        stamp = time.time()
        ca = fit_specrage_graph(
            views, config=CA_CONFIG, seeds=MODEL_SEEDS, view_prior=prior
        )
        stage_runtime[f"{schema}_ca_specrage_seconds"] = time.time() - stamp
        stamp = time.time()
        uniform = fit_specrage_graph(
            views, config=UNIFORM_CONFIG, seeds=MODEL_SEEDS, view_prior=prior
        )
        stage_runtime[f"{schema}_uniform_specrage_seconds"] = time.time() - stamp
        graphs.update({
            schema_arm(schema, "adapted_specrage_y_liu"): plain.embedding_graph,
            schema_arm(schema, "ca_specrage_alpha_liu"): ca.graph,
            schema_arm(schema, "ca_specrage_y_liu"): ca.embedding_graph,
            schema_arm(schema, "uniform_y_liu"): uniform.embedding_graph,
            schema_arm(schema, "ca_uniform_alpha_control"): graph_for_control(
                ca, "uniform"
            ),
            schema_arm(schema, "ca_global_alpha_control"): graph_for_control(ca, "global"),
            schema_arm(schema, "ca_permuted_alpha_control"): graph_for_control(
                ca, "permuted", seed=permutation
            ),
        })
        schema_fits[schema] = {
            "view_names": list(views),
            "view_members": (
                view_members(stable_names) if schema == "manual" else
                ({name: [name] for name in views} if schema == "atomic" else
                 {name: list(micro_partition[name]) for name in views})
            ),
            "view_dimensions": {name: int(value.shape[1]) for name, value in views.items()},
            "view_prior": {name: float(prior[name]) for name in views},
            "adapted_specrage": plain.diagnostics,
            "ca_specrage": ca.diagnostics,
            "uniform_specrage": uniform.diagnostics,
            "fusion_geometry": _schema_fusion_diagnostics(F, ca),
            "histories": (
                _fit_histories(plain, f"{schema}_adapted_specrage")
                + _fit_histories(ca, f"{schema}_ca_specrage")
                + _fit_histories(uniform, f"{schema}_uniform_specrage")
            ),
        }

    stamp = time.time()
    paths = {
        arm: laplacian_iu_path(F, LAMBDAS, graph=graph)
        for arm, graph in graphs.items()
    }
    stage_runtime["liu_paths_seconds"] = time.time() - stamp

    baseline = paths["dufs_liu"][0.0].baseline
    deployed = upcr_fit(F, **DEPLOYED_FIT)
    score_arrays = {
        "sample_index": np.arange(F.shape[1], dtype=np.int64),
        "feature_names": np.asarray(stable_names, dtype=str),
        "deployed_upcr": np.asarray(deployed.w @ F, dtype=np.float64),
        "iu_pcr": np.asarray(baseline.w @ F, dtype=np.float64),
    }
    for arm, path in paths.items():
        for lambda_, fit in path.items():
            values = np.asarray(fit.w @ F, dtype=np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError(f"non-finite scores for {cell}/{arm}/{lambda_}")
            if lambda_ == 0.0 and not np.array_equal(values, score_arrays["iu_pcr"]):
                raise RuntimeError(f"lambda=0 identity failed for {cell}/{arm}")
            score_arrays[score_key(arm, lambda_)] = values

    liu_diagnostics = {
        arm: {
            lambda_token(lambda_): fit.diagnostics
            for lambda_, fit in path.items()
        }
        for arm, path in paths.items()
    }
    diagnostics = {
        "cell": cell,
        "domain": GROUP[cell],
        "n_samples": int(F.shape[1]),
        "n_features_stable": int(F.shape[0]),
        "stable_feature_names": list(stable_names),
        "manual_provenance_members": view_members(stable_names),
        "micro_partition": {key: list(value) for key, value in micro_partition.items()},
        "micro_partition_diagnostics": partition_diagnostics,
        "deployed": {
            "n_kept": int(np.sum(deployed.keep)),
            "keep_mask": deployed.keep.tolist(),
            "abstained": bool(deployed.abstained),
            "used_simple_average": bool(deployed.used_simple_average),
            "n_components_used": int(deployed.n_components_used),
            "lambda2_frac": float(deployed.lambda2_frac),
            "projection_residual": float(deployed.proj_residual),
            "g2_fraction": float(deployed.g2_frac_of_var_y),
        },
        "iu": {
            "n_features": int(np.sum(baseline.keep)),
            "n_components_used": int(baseline.n_components_used),
            "lambda2_frac": float(baseline.lambda2_frac),
            "projection_residual": float(baseline.proj_residual),
        },
        "dufs": dufs_diagnostics,
        "schemas": schema_fits,
        "liu": liu_diagnostics,
        "histories": [
            record for schema in VIEW_SCHEMAS
            for record in schema_fits[schema]["histories"]
        ],
        "runtime": {
            **stage_runtime,
            "total_seconds": time.time() - started,
        },
    }
    return score_arrays, _finalize_diagnostics(diagnostics)


def _profile_checkpoint(path, profile=None):
    if profile is not None:
        np.savez_compressed(
            path,
            cell=np.asarray(profile["cell"]),
            names=np.asarray(profile["names"], dtype=str),
            distances=profile["distances"],
            mean_distance=profile["mean_distance"],
            sample_sizes=profile["sample_sizes"],
            graph_edges=profile["graph_edges"],
            pair_distance_bootstrap_std_mean=np.asarray(
                profile["pair_distance_bootstrap_std_mean"]
            ),
            pair_distance_bootstrap_std_p95=np.asarray(
                profile["pair_distance_bootstrap_std_p95"]
            ),
        )
        return profile
    with np.load(path, allow_pickle=False) as stored:
        return {
            "cell": str(stored["cell"].item()),
            "names": tuple(str(name) for name in stored["names"]),
            "distances": np.asarray(stored["distances"], dtype=float),
            "mean_distance": np.asarray(stored["mean_distance"], dtype=float),
            "sample_sizes": np.asarray(stored["sample_sizes"], dtype=int),
            "graph_edges": np.asarray(stored["graph_edges"], dtype=int),
            "pair_distance_bootstrap_std_mean": float(
                stored["pair_distance_bootstrap_std_mean"]
            ),
            "pair_distance_bootstrap_std_p95": float(
                stored["pair_distance_bootstrap_std_p95"]
            ),
        }


def prepare_view_construction(data, cells, out_dir, *, resume):
    root = os.path.join(out_dir, "view_construction")
    profile_dir = os.path.join(root, "profiles")
    partition_dir = os.path.join(root, "partitions")
    os.makedirs(profile_dir, exist_ok=True)
    os.makedirs(partition_dir, exist_ok=True)
    profiles = []
    for index, cell in enumerate(INSCOPE, start=1):
        path = os.path.join(profile_dir, f"{cell}.npz")
        if resume and os.path.exists(path):
            profile = _profile_checkpoint(path)
            status = "reused"
        else:
            stored = np.asarray(data[f"{cell}__V"], dtype=float)
            names = tuple(str(name) for name in data[f"{cell}__pool"])
            legacy = np.asarray(data[f"{cell}__hand_signs"], dtype=float)
            matrix, stable_names = fixed_stable_from_bundle(stored, names, legacy)
            profile = cell_impact_profile(matrix, stable_names, cell=cell)
            _profile_checkpoint(path, profile)
            status = "fit"
        profiles.append(profile)
        print(json.dumps({
            "view_profile": cell,
            "progress": f"{index}/{len(INSCOPE)}",
            "status": status,
        }), flush=True)
    output = {}
    for cell in cells:
        profile = next(item for item in profiles if item["cell"] == cell)
        partition, diagnostics = learn_loco_micro_partition(
            profiles, held_cell=cell, held_feature_names=profile["names"]
        )
        write_json(os.path.join(partition_dir, f"{cell}.json"), diagnostics)
        output[cell] = (partition, diagnostics)
    return output


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def fit(args) -> None:
    data = np.load(args.bundle, allow_pickle=True)
    canonical_cells = validate_bundle(data)
    scientific_run = args.debug_cells is None and args.debug_cell is None
    cells = canonical_cells
    if args.debug_cells is not None:
        if args.debug_cells < 1 or args.debug_cells >= len(canonical_cells):
            raise ValueError("--debug-cells must be between 1 and 23")
        cells = canonical_cells[: args.debug_cells]
    elif args.debug_cell is not None:
        if args.debug_cell not in canonical_cells:
            raise ValueError(f"--debug-cell is not registered: {args.debug_cell}")
        cells = (args.debug_cell,)

    definition = run_definition(args.bundle, scientific_run, cells)
    os.makedirs(args.out_dir, exist_ok=True)
    scores_dir = os.path.join(args.out_dir, "scores")
    diagnostics_dir = os.path.join(args.out_dir, "diagnostics")
    os.makedirs(scores_dir, exist_ok=True)
    os.makedirs(diagnostics_dir, exist_ok=True)
    definition_path = os.path.join(args.out_dir, "RUN_DEFINITION.json")
    if os.path.exists(definition_path):
        with open(definition_path, encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("run_fingerprint") != definition["run_fingerprint"]:
            raise RuntimeError(
                "output directory contains a different run definition; choose a new --out-dir"
            )
    else:
        write_json(definition_path, definition)

    view_construction = prepare_view_construction(
        data, cells, args.out_dir, resume=args.resume
    )

    started = time.time()
    score_manifest = []
    for index, cell in enumerate(cells, start=1):
        score_path = os.path.join(scores_dir, f"{cell}.npz")
        diagnostic_path = os.path.join(diagnostics_dir, f"{cell}.json")
        if args.resume and os.path.exists(score_path) and os.path.exists(diagnostic_path):
            with np.load(score_path, allow_pickle=False) as checkpoint:
                expected_score_keys = {
                    "sample_index", "feature_names", "iu_pcr", "deployed_upcr",
                    *(
                        score_key(arm, lambda_)
                        for arm in ALL_GRAPH_ARMS
                        for lambda_ in LAMBDAS
                    ),
                }
                missing = expected_score_keys - set(checkpoint.files)
                if missing:
                    raise RuntimeError(
                        f"incomplete score checkpoint {score_path}: "
                        f"missing={sorted(missing)}"
                    )
                if any("label" in key.lower() for key in checkpoint.files):
                    raise RuntimeError(f"label-like array found in checkpoint: {score_path}")
            status = "reused"
        else:
            partition, partition_diagnostics = view_construction[cell]
            score_arrays, diagnostics = fit_cell(
                cell, data, args.permutation_seed,
                partition, partition_diagnostics,
            )
            np.savez_compressed(score_path, **score_arrays)
            write_json(diagnostic_path, diagnostics)
            status = "fit"
        score_manifest.append({
            "cell": cell,
            "domain": GROUP[cell],
            "score_file": os.path.relpath(score_path, args.out_dir),
            "score_sha256": sha256_file(score_path),
            "diagnostic_file": os.path.relpath(diagnostic_path, args.out_dir),
            "diagnostic_sha256": sha256_file(diagnostic_path),
        })
        print(
            json.dumps({
                "cell": cell,
                "progress": f"{index}/{len(cells)}",
                "status": status,
                "elapsed_seconds": round(time.time() - started, 2),
            }),
            flush=True,
        )

    fit_complete = {
        "version": VERSION,
        "scientific_run": scientific_run,
        "run_fingerprint": definition["run_fingerprint"],
        "n_cells": len(cells),
        "runtime_seconds": time.time() - started,
        "labels_opened_by_fit": False,
        "score_manifest": score_manifest,
    }
    write_json(os.path.join(args.out_dir, "FIT_COMPLETE.json"), fit_complete)
    print(
        f"Frozen label-free scores written for {len(cells)} cells. "
        "Run scripts/frozen_24cell_report.py only after this step completes.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    debug = parser.add_mutually_exclusive_group()
    debug.add_argument("--debug-cells", type=int, default=None)
    debug.add_argument("--debug-cell", default=None)
    parser.add_argument("--permutation-seed", type=int, default=8_104_113)
    args = parser.parse_args()
    fit(args)


if __name__ == "__main__":
    main()
