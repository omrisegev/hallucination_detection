#!/usr/bin/env python3
"""Fit and freeze the Family-residual graph LIU development grid.

This command deliberately never accesses correctness labels.  Labels are read
only by ``family_residual_graph_liu_report.py`` after every score file and hash
has been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    DEFAULT_BUNDLE,
    DUFS_EPOCHS,
    DUFS_SEEDS,
    load_contract,
    validate_bundle_without_labels,
)
from scripts.inscope_cells import INSCOPE  # noqa: E402
from spectral_utils.contribution_subspace import (  # noqa: E402
    cardinality_balanced_contribution_score,
)
from spectral_utils.family_residual_graph import (  # noqa: E402
    build_family_graphs,
    contribution_laplacian_path,
    fit_family_residual_state,
)
from spectral_utils.laplacian_upcr import (  # noqa: E402
    dufs_soft_gates,
    laplacian_iu_path,
)


VERSION = "family-residual-graph-liu-v3-2026-08-23"
DEFAULT_OUT = REPO / "results" / "family_residual_graph_liu_v3"
ETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
KS = (5, 7, 15)
LAMBDAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
TRUST_FACTORS = (0.5, 1.0, 2.0)
TOPOLOGIES = ("union", "adaptive")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def code(value):
    return f"{int(round(100 * float(value))):04d}"


def score_key(
    readout, eta, beta, k, lambda_, trust_factor=None, *, topology="union"
):
    key = (
        f"{readout}__g{topology}__e{code(eta)}__b{code(beta)}__k{int(k):02d}"
        f"__l{code(lambda_)}"
    )
    if trust_factor is not None:
        key += f"__t{code(trust_factor)}"
    return key


def graph_pairs():
    return tuple(
        (eta, 0.5) if eta == 0.0 else (eta, beta)
        for eta in ETAS
        for beta in ((0.5,) if eta == 0.0 else BETAS)
    )


def graph_settings():
    return tuple(("union", k) for k in KS) + (("adaptive", 7),)


def configuration_index():
    output = {
        "iu": {"readout": "iu"},
        "cardinality": {"readout": "cardinality"},
    }
    for eta, beta in graph_pairs():
        for topology, k in graph_settings():
            for lambda_ in LAMBDAS:
                output[score_key(
                    "u2", eta, beta, k, lambda_, topology=topology
                )] = {
                    "readout": "u2",
                    "topology": topology,
                    "eta": eta,
                    "beta": beta,
                    "k": k,
                    "lambda": lambda_,
                }
                for trust_factor in TRUST_FACTORS:
                    output[score_key(
                        "cs", eta, beta, k, lambda_, trust_factor,
                        topology=topology,
                    )] = {
                        "readout": "cs",
                        "topology": topology,
                        "eta": eta,
                        "beta": beta,
                        "k": k,
                        "lambda": lambda_,
                        "trust_factor": trust_factor,
                    }
    return output


def run_definition(bundle):
    configs = configuration_index()
    payload = {
        "version": VERSION,
        "bundle": str(Path(bundle).resolve()),
        "bundle_sha256": sha256_file(bundle),
        "roster": list(INSCOPE),
        "etas": list(ETAS),
        "betas": list(BETAS),
        "ks": list(KS),
        "lambdas": list(LAMBDAS),
        "trust_factors": list(TRUST_FACTORS),
        "topologies": list(TOPOLOGIES),
        "dufs_seeds": list(DUFS_SEEDS),
        "dufs_epochs": DUFS_EPOCHS,
        "score_configuration_count": len(configs),
        "sources": {
            "fit_script": sha256_file(Path(__file__)),
            "report_script": sha256_file(
                REPO / "scripts" / "family_residual_graph_liu_report.py"
            ),
            "controls_script": sha256_file(
                REPO / "scripts" / "family_residual_graph_liu_controls.py"
            ),
            "prmbench_script": sha256_file(
                REPO / "scripts" / "family_residual_graph_liu_prmbench.py"
            ),
            "hle_script": sha256_file(
                REPO / "scripts" / "family_residual_graph_liu_hle.py"
            ),
            "hard_filter_contract_script": sha256_file(
                REPO / "scripts" / "hard_filter_dufs_liu_benchmark.py"
            ),
            "inscope_roster_script": sha256_file(
                REPO / "scripts" / "inscope_cells.py"
            ),
            "dufs_trainer_module": sha256_file(
                REPO / "spectral_utils" / "selectors" / "a2_groupfs.py"
            ),
            "transfer_contract_script": sha256_file(
                REPO / "scripts" / "leverage_balanced_processbench_transfer.py"
            ),
            "prmbench_loader_script": sha256_file(
                REPO / "scripts" / "neutral_residual_mode_prmbench_confirmation.py"
            ),
            "hle_loader_script": sha256_file(
                REPO / "scripts" / "neutral_residual_mode_hle_confirmation.py"
            ),
            "core_module": sha256_file(
                REPO / "spectral_utils" / "family_residual_graph.py"
            ),
            "contribution_module": sha256_file(
                REPO / "spectral_utils" / "contribution_subspace.py"
            ),
            "graph_topology_module": sha256_file(
                REPO / "spectral_utils" / "graph_topology.py"
            ),
            "laplacian_module": sha256_file(
                REPO / "spectral_utils" / "laplacian_upcr.py"
            ),
            "upcr_module": sha256_file(
                REPO / "spectral_utils" / "upcr.py"
            ),
            "feature_contract_module": sha256_file(
                REPO / "spectral_utils" / "dufs_liu_feature_contract.py"
            ),
            "family_registry_module": sha256_file(
                REPO / "spectral_utils" / "specrage_views.py"
            ),
            "fusion_utils_module": sha256_file(
                REPO / "spectral_utils" / "fusion_utils.py"
            ),
            "base_feature_contract_module": sha256_file(
                REPO / "spectral_utils" / "feature_contract.py"
            ),
            "feature_utils_module": sha256_file(
                REPO / "spectral_utils" / "feature_utils.py"
            ),
            "repgrid_scoring_module": sha256_file(
                REPO / "spectral_utils" / "repgrid_scoring.py"
            ),
            "spec": sha256_file(
                REPO / "docs" / "experiments" / "FAMILY_RESIDUAL_GRAPH_LIU_V3.md"
            ),
            "base_v2_spec": sha256_file(
                REPO / "docs" / "experiments" / "FAMILY_RESIDUAL_GRAPH_LIU_V2.md"
            ),
        },
        "labels_accessed_by_fit": False,
        "comparators": {
            "family_nrm_cell_results_sha256": sha256_file(
                REPO / "results" / "neutral_residual_mode_cs_iu_v1"
                / "cell_results.csv"
            ),
        },
    }
    payload["definition_hash"] = canonical_hash(payload)
    return payload, configs


def fit_cell(data, cell, configs):
    started = time.time()
    F, names = load_contract(data, cell, "mixed_v2")
    state = fit_family_residual_state(F, names)
    gates, gate_diag = dufs_soft_gates(
        F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
    )
    cardinality = cardinality_balanced_contribution_score(
        state.contribution_space, state.baseline_fit.w
    )
    scores = {
        "iu": np.asarray(state.baseline, dtype=np.float64),
        "cardinality": np.asarray(cardinality.score, dtype=np.float64),
        "sample_index": np.arange(F.shape[1], dtype=np.int64),
    }
    graph_rows = []
    max_identity_error = 0.0
    for eta, beta in graph_pairs():
        for topology, requested_k in graph_settings():
            graph_fit = build_family_graphs(
                F,
                gates,
                state,
                eta=eta,
                beta=beta,
                ks=(requested_k,),
                family_mode="residual",
                topology=topology,
                scale_seed=1729,
            )[requested_k]
            k = requested_k
            W = graph_fit.graph
            u2 = laplacian_iu_path(F, LAMBDAS, graph=W)
            cs = contribution_laplacian_path(
                state.baseline,
                state.residuals,
                W,
                LAMBDAS,
                trust_caps=tuple(
                    factor / len(state.contribution_space.families)
                    for factor in TRUST_FACTORS
                ),
            )
            for lambda_ in LAMBDAS:
                if lambda_ == 0.0:
                    u2_score = state.baseline.copy()
                else:
                    raw = np.asarray(u2[lambda_].w @ F, dtype=float)
                    u2_score = (
                        raw - state.transform.baseline_mean
                    ) / state.transform.baseline_scale
                scores[score_key(
                    "u2", eta, beta, k, lambda_, topology=topology
                )] = np.asarray(
                    u2_score, dtype=np.float64
                )
                if lambda_ == 0.0:
                    max_identity_error = max(
                        max_identity_error,
                        float(np.max(np.abs(u2_score - state.baseline))),
                    )
                for trust_factor in TRUST_FACTORS:
                    cap = trust_factor / len(state.contribution_space.families)
                    result = cs[(lambda_, cap)]
                    scores[score_key(
                        "cs", eta, beta, k, lambda_, trust_factor,
                        topology=topology,
                    )] = np.asarray(result.score, dtype=np.float64)
                    if lambda_ == 0.0:
                        max_identity_error = max(
                            max_identity_error,
                            float(np.max(np.abs(
                                result.score - state.baseline
                            ))),
                        )
            graph_rows.append({
                **graph_fit.diagnostics,
                "u2_lambda_01_weight_cosine": float(
                    u2[0.1].diagnostics["weight_cosine_vs_iu"]
                ),
            })
    expected = set(configs) | {"sample_index"}
    if set(scores) != expected:
        missing = sorted(expected - set(scores))
        extra = sorted(set(scores) - expected)
        raise RuntimeError(f"score registry mismatch: missing={missing}, extra={extra}")
    if max_identity_error != 0.0:
        raise RuntimeError(f"lambda=0 identity error: {max_identity_error:.3e}")
    diagnostics = {
        "version": VERSION,
        "cell": cell,
        "n_samples": int(F.shape[1]),
        "n_features": int(F.shape[0]),
        "feature_names": list(names),
        "families": list(state.contribution_space.families),
        "state": state.diagnostics,
        "dufs": {
            "raw_probabilities": np.asarray(
                gate_diag["raw_probabilities"], dtype=float
            ).tolist(),
            "effective_feature_count": float(
                gate_diag["effective_feature_count"]
            ),
            "mean_seed_std": float(gate_diag["mean_seed_std"]),
        },
        "graphs": graph_rows,
        "lambda_zero_max_identity_error": max_identity_error,
        "runtime_seconds": float(time.time() - started),
    }
    return scores, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug-cell", choices=INSCOPE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scores").mkdir(exist_ok=True)
    (args.out / "diagnostics").mkdir(exist_ok=True)

    definition, configs = run_definition(args.bundle)
    definition_path = args.out / "RUN_DEFINITION.json"
    if definition_path.exists():
        existing = json.loads(definition_path.read_text())
        if existing != definition:
            raise RuntimeError("run definition changed; use a new output directory")
    else:
        write_json(definition_path, definition)
        write_json(args.out / "CONFIG_INDEX.json", configs)

    roster = (args.debug_cell,) if args.debug_cell else tuple(INSCOPE)
    score_hashes = {}
    diagnostic_hashes = {}
    with np.load(args.bundle, allow_pickle=True) as data:
        validate_bundle_without_labels(data)
        for index, cell in enumerate(roster, start=1):
            score_path = args.out / "scores" / f"{cell}.npz"
            diag_path = args.out / "diagnostics" / f"{cell}.json"
            if args.resume and score_path.exists() and diag_path.exists():
                score_hashes[cell] = sha256_file(score_path)
                diagnostic_hashes[cell] = sha256_file(diag_path)
                print(f"[{index}/{len(roster)}] reuse {cell}", flush=True)
                continue
            print(f"[{index}/{len(roster)}] fit {cell}", flush=True)
            scores, diagnostics = fit_cell(data, cell, configs)
            temporary = score_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **scores)
            temporary.replace(score_path)
            write_json(diag_path, diagnostics)
            score_hashes[cell] = sha256_file(score_path)
            diagnostic_hashes[cell] = sha256_file(diag_path)

    complete = {
        "version": VERSION,
        "definition_hash": definition["definition_hash"],
        "scientific_run": args.debug_cell is None,
        "roster": list(roster),
        "score_hashes": score_hashes,
        "diagnostic_hashes": diagnostic_hashes,
        "config_index_sha256": sha256_file(args.out / "CONFIG_INDEX.json"),
        "labels_accessed_by_fit": False,
    }
    complete["manifest_hash"] = canonical_hash(complete)
    write_json(args.out / "FIT_COMPLETE.json", complete)
    print(json.dumps({
        "status": "fit_complete",
        "cells": len(roster),
        "manifest_hash": complete["manifest_hash"],
    }, indent=2))


if __name__ == "__main__":
    main()
