#!/usr/bin/env python3
"""Registered Phase-1 cross-view graph audit experiment.

This runner executes only the low-dimensional premise worlds P1-A--P1-F.  The
candidate API never receives labels.  Labels are joined to frozen scores by the
evaluator in this file.  Smoke freezes source/config hashes; development refuses
to run if either changed.  Confirmation is additionally approval-locked and is
not part of the current development cycle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
import types

import numpy as np
import scipy
import sklearn
from scipy.linalg import eigh
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.cross_view_graph import (  # noqa: E402
    cross_view_consensus,
    mmdufs_shared_graph,
    standardize_columns,
)
from spectral_utils.laplacian_upcr import (  # noqa: E402
    IU_FIT_DEFAULTS,
    laplacian_iu_path,
    permute_graph,
    self_tuning_knn_graph,
)
from spectral_utils.upcr import upcr_fit  # noqa: E402


VERSION = "graybox-cross-view-phase1-v1-2026-08-06"
DEFAULT_OUT = os.path.join(REPO, "results", "graybox_cross_view_phase1")
WORLDS = ("P1-A", "P1-B", "P1-C", "P1-D", "P1-E", "P1-F")
WORLD_LABELS = {
    "P1-A": "Aligned target",
    "P1-B": "Discovery-specific nuisance",
    "P1-C": "Measured shared nuisance",
    "P1-D": "Paired targets",
    "P1-E": "Pure noise",
    "P1-F": "Unmeasured shared nuisance",
}
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
PRIMARY_LAMBDA = 0.1
PRIMARY_K = 7
K_VALUES = (5, 7, 11)
PERMUTATIONS = 199
SMOKE_REPLICATES = 2
DEVELOPMENT_REPLICATES = 8
CONFIRMATION_REPLICATES = 16
SMOKE_N = 180
DEVELOPMENT_N = 360
CONFIRMATION_N = 500
SEED_BASES = {
    "smoke": 3_100_000,
    "development": 3_200_000,
    "confirmation": 3_800_000,
}
ARMS = (
    "iu",
    "direct_g",
    "direct_a",
    "g_to_a",
    "a_to_g",
    "consensus",
    "mmdufs_shared",
    "projected_ridge",
    "permuted_consensus",
    "nuisance_only",
    "oracle",
)
EPS = 1e-12


def balanced_labels(latent, rng):
    decision = np.asarray(latent) + 0.75 * rng.standard_normal(len(latent))
    return (decision > np.median(decision)).astype(int)


def zscore_rows(matrix):
    values = np.asarray(matrix, dtype=float)
    centered = values - values.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    if np.any(scale < EPS):
        raise ValueError("fusion matrix contains a constant row")
    return centered / scale


def signal12(g, rng):
    rows = [
        g + sigma * rng.standard_normal(len(g))
        for sigma in (0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.85, 1.00)
    ]
    rows += [0.55 * g + 1.15 * rng.standard_normal(len(g)) for _ in range(2)]
    rows += [rng.standard_normal(len(g)) for _ in range(2)]
    return zscore_rows(np.vstack(rows))


def nuisance12(g, q, rng):
    rows = [g + 0.55 * rng.standard_normal(len(g)) for _ in range(6)]
    rows += [q + 0.25 * rng.standard_normal(len(g)) for _ in range(6)]
    return zscore_rows(np.vstack(rows))


def noise12(n, rng):
    return zscore_rows(np.vstack([rng.standard_normal(n) for _ in range(12)]))


def nuisance_basis(q, rng):
    return np.column_stack([
        q + 0.10 * rng.standard_normal(len(q)),
        np.tanh(q) + 0.10 * rng.standard_normal(len(q)),
        np.arctan(q) + 0.10 * rng.standard_normal(len(q)),
        (q > 0.8).astype(float),
    ])


def make_world(world, seed, n):
    """Return observed method inputs plus evaluator-only targets and latents."""
    rng = np.random.default_rng(int(seed))
    g = rng.standard_normal(n)
    q = rng.standard_normal(n)
    r = rng.standard_normal(n)
    y_g = balanced_labels(g, rng)
    y_r = balanced_labels(r, rng)

    if world in ("P1-A", "P1-D"):
        G = np.column_stack([
            g + 0.35 * rng.standard_normal(n),
            np.tanh(g) + 0.20 * rng.standard_normal(n),
            np.arctan(g) + 0.20 * rng.standard_normal(n),
        ])
        A = np.column_stack([
            g + 0.45 * rng.standard_normal(n),
            np.tanh(0.8 * g) + 0.25 * rng.standard_normal(n),
            np.arctan(0.8 * g) + 0.25 * rng.standard_normal(n),
        ])
        N = np.column_stack([rng.standard_normal(n), rng.standard_normal(n)])
        F = signal12(g, rng)
    elif world == "P1-B":
        G = np.column_stack([
            q + 0.20 * rng.standard_normal(n),
            np.tanh(q) + 0.20 * rng.standard_normal(n),
            0.35 * g + 1.00 * rng.standard_normal(n),
        ])
        A = np.column_stack([
            g + 0.45 * rng.standard_normal(n),
            np.tanh(0.8 * g) + 0.30 * rng.standard_normal(n),
            np.arctan(0.8 * g) + 0.30 * rng.standard_normal(n),
        ])
        N = nuisance_basis(q, rng)
        F = nuisance12(g, q, rng)
    elif world == "P1-C":
        G = np.column_stack([
            q + 0.20 * rng.standard_normal(n),
            np.tanh(q) + 0.20 * rng.standard_normal(n),
            0.25 * g + 1.00 * rng.standard_normal(n),
        ])
        A = np.column_stack([
            q + 0.25 * rng.standard_normal(n),
            np.arctan(q) + 0.20 * rng.standard_normal(n),
            0.25 * g + 1.00 * rng.standard_normal(n),
        ])
        N = nuisance_basis(q, rng)
        F = nuisance12(g, q, rng)
    elif world == "P1-E":
        G = np.column_stack([rng.standard_normal(n) for _ in range(3)])
        A = np.column_stack([rng.standard_normal(n) for _ in range(3)])
        N = np.column_stack([rng.standard_normal(n) for _ in range(2)])
        F = noise12(n, rng)
    elif world == "P1-F":
        G = np.column_stack([
            q + 0.20 * rng.standard_normal(n),
            np.tanh(q) + 0.20 * rng.standard_normal(n),
            np.arctan(q) + 0.20 * rng.standard_normal(n),
        ])
        A = np.column_stack([
            q + 0.25 * rng.standard_normal(n),
            np.tanh(0.8 * q) + 0.25 * rng.standard_normal(n),
            np.arctan(0.8 * q) + 0.25 * rng.standard_normal(n),
        ])
        N = np.column_stack([rng.standard_normal(n), rng.standard_normal(n)])
        F = nuisance12(g, q, rng)
    else:
        raise ValueError(f"unknown world: {world}")

    targets = [{"name": "g", "labels": y_g, "oracle_latent": g}]
    if world == "P1-D":
        targets.append({"name": "r", "labels": y_r, "oracle_latent": r})
    return {
        "G": standardize_columns(G),
        "A": standardize_columns(A),
        "N": standardize_columns(N),
        "F": F,
        "targets": targets,
    }


def projected_ridge_path(F, baseline):
    m, n = F.shape
    covariance = F @ F.T / n
    values, basis = eigh(covariance, subset_by_index=[m - 2, m - 1])
    basis = basis[:, np.argsort(values)[::-1]]
    projected = 0.5 * (basis.T @ covariance @ basis + basis.T @ covariance.T @ basis)
    ridge = np.eye(2) * np.trace(projected) / 2.0
    rhs = basis.T @ baseline.rho_hat
    output = {}
    for lambda_ in LAMBDAS:
        if lambda_ == 0.0:
            weight = baseline.w.copy()
        else:
            weight = basis @ np.linalg.solve(projected + lambda_ * ridge, rhs)
        output[lambda_] = {
            "weight": weight,
            "scores": weight @ F,
            "roughness_min_eigenvalue": 0.0,
            "condition_number": float(np.linalg.cond(projected + lambda_ * ridge)),
        }
    return output


def graph_path(F, baseline, graph):
    if graph is None:
        return {
            lambda_: {
                "weight": baseline.w.copy(),
                "scores": baseline.w @ F,
                "roughness_min_eigenvalue": 0.0,
                "condition_number": float("nan"),
            }
            for lambda_ in LAMBDAS
        }
    fitted = laplacian_iu_path(F, LAMBDAS, graph=graph, k=PRIMARY_K)
    return {
        lambda_: {
            "weight": result.w,
            "scores": result.w @ F,
            "roughness_min_eigenvalue": result.diagnostics[
                "roughness_min_eigenvalue"
            ],
            "condition_number": result.diagnostics[
                "projected_condition_number"
            ],
        }
        for lambda_, result in fitted.items()
    }


def hash_method_output(consensus, arm_paths):
    digest = hashlib.sha256()
    graph = consensus["graph"]
    digest.update(str(consensus["accepted_count"]).encode())
    for direction in (consensus["forward"], consensus["reverse"]):
        digest.update(json.dumps(direction.diagnostics, sort_keys=True).encode())
    if graph is not None:
        digest.update(graph.data.tobytes())
        digest.update(graph.indices.tobytes())
        digest.update(graph.indptr.tobytes())
    for arm in sorted(arm_paths):
        if arm == "oracle":
            continue
        digest.update(arm.encode())
        digest.update(arm_paths[arm][PRIMARY_LAMBDA]["scores"].tobytes())
    return digest.hexdigest()


def flatten_audit(world, replicate, seed, direction_name, direction):
    output = []
    for k_text, item in direction.diagnostics["per_k"].items():
        output.append({
            "world": world,
            "replicate": replicate,
            "seed": seed,
            "direction": direction_name,
            "k": int(k_text),
            "direction_accepted": direction.accepted,
            "decision_agreement": direction.diagnostics["decision_agreement"],
            "stability_cka": direction.diagnostics["stability_cka"],
            "audit_row_permutation_statistic": direction.diagnostics[
                "audit_row_permutation_statistic"
            ],
            "audit_row_permutation_p": direction.diagnostics[
                "audit_row_permutation_p"
            ],
            **item,
        })
    return output


def run_dataset(stage, world, world_index, replicate, n):
    seed = SEED_BASES[stage] + 10_000 * world_index + replicate
    data = make_world(world, seed, n)
    F, G, A, N = data["F"], data["G"], data["A"], data["N"]
    baseline = upcr_fit(F, **IU_FIT_DEFAULTS)
    baseline_scores = baseline.w @ F

    candidate = cross_view_consensus(
        G,
        A,
        N,
        seed=seed,
        permutation_count=PERMUTATIONS,
        ks=K_VALUES,
        primary_k=PRIMARY_K,
    )
    graph_g = self_tuning_knn_graph(G, k=PRIMARY_K)
    graph_a = self_tuning_knn_graph(A, k=PRIMARY_K)
    graph_n = self_tuning_knn_graph(N, k=PRIMARY_K)
    graph_mm = mmdufs_shared_graph(G, A, k=PRIMARY_K)
    graph_permuted = None
    if candidate["graph"] is not None:
        rng = np.random.default_rng(seed + 501_119)
        graph_permuted = permute_graph(candidate["graph"], rng.permutation(n))

    arm_graphs = {
        "direct_g": graph_g,
        "direct_a": graph_a,
        "g_to_a": candidate["forward"].graph if candidate["forward"].accepted else None,
        "a_to_g": candidate["reverse"].graph if candidate["reverse"].accepted else None,
        "consensus": candidate["graph"],
        "mmdufs_shared": graph_mm,
        "permuted_consensus": graph_permuted,
        "nuisance_only": graph_n,
    }
    arm_paths = {
        "iu": {
            lambda_: {
                "weight": baseline.w.copy(),
                "scores": baseline_scores.copy(),
                "roughness_min_eigenvalue": 0.0,
                "condition_number": float("nan"),
            }
            for lambda_ in LAMBDAS
        },
        "projected_ridge": projected_ridge_path(F, baseline),
    }
    for arm, graph in arm_graphs.items():
        arm_paths[arm] = graph_path(F, baseline, graph)

    method_hash = hash_method_output(candidate, arm_paths)
    lambda0_error = max(
        float(np.max(np.abs(path[0.0]["scores"] - baseline_scores)))
        for path in arm_paths.values()
    )
    min_roughness = min(
        item["roughness_min_eigenvalue"]
        for path in arm_paths.values() for item in path.values()
    )

    result_rows = []
    for target in data["targets"]:
        labels = target["labels"]
        oracle_graph = self_tuning_knn_graph(
            target["oracle_latent"][:, None], k=PRIMARY_K
        )
        oracle_path = graph_path(F, baseline, oracle_graph)
        target_paths = {**arm_paths, "oracle": oracle_path}
        baseline_auc = float(roc_auc_score(labels, baseline_scores))
        baseline_ap = float(average_precision_score(labels, baseline_scores))
        for arm in ARMS:
            for lambda_ in LAMBDAS:
                scores = target_paths[arm][lambda_]["scores"]
                auc = float(roc_auc_score(labels, scores))
                ap = float(average_precision_score(labels, scores))
                accepted = {
                    "g_to_a": candidate["forward"].accepted,
                    "a_to_g": candidate["reverse"].accepted,
                    "consensus": candidate["accepted_count"] > 0,
                    "permuted_consensus": candidate["accepted_count"] > 0,
                }.get(arm, arm != "iu")
                result_rows.append({
                    "version": VERSION,
                    "stage": stage,
                    "world": world,
                    "world_label": WORLD_LABELS[world],
                    "replicate": replicate,
                    "seed": seed,
                    "target": target["name"],
                    "arm": arm,
                    "lambda": lambda_,
                    "auroc": auc,
                    "auprc": ap,
                    "auroc_delta": auc - baseline_auc,
                    "auprc_delta": ap - baseline_ap,
                    "accepted": bool(accepted),
                    "fallback": bool(
                        arm in ("g_to_a", "a_to_g", "consensus", "permuted_consensus")
                        and not accepted
                    ),
                    "accepted_direction_count": candidate["accepted_count"],
                    "roughness_min_eigenvalue": target_paths[arm][lambda_][
                        "roughness_min_eigenvalue"
                    ],
                    "condition_number": target_paths[arm][lambda_]["condition_number"],
                    "method_hash": method_hash,
                })

    audit_rows = []
    audit_rows.extend(flatten_audit(
        world, replicate, seed, "g_to_a", candidate["forward"]
    ))
    audit_rows.extend(flatten_audit(
        world, replicate, seed, "a_to_g", candidate["reverse"]
    ))
    dataset_row = {
        "version": VERSION,
        "stage": stage,
        "world": world,
        "replicate": replicate,
        "seed": seed,
        "n": n,
        "accepted_direction_count": candidate["accepted_count"],
        "consensus_accepted": candidate["accepted_count"] > 0,
        "method_hash": method_hash,
        "method_call_count": 1,
        "lambda0_max_score_error": lambda0_error,
        "roughness_min_eigenvalue": min_roughness,
        "label_parameter_absent": "labels" not in inspect.signature(
            cross_view_consensus
        ).parameters,
    }
    return result_rows, audit_rows, dataset_row


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hashes():
    paths = (
        "docs/research_notes/graybox_cross_view_manifold_proposal.md",
        "spectral_utils/cross_view_graph.py",
        "spectral_utils/laplacian_upcr.py",
        "spectral_utils/manifests/graybox_cross_view_v1.json",
        "results/graybox_cross_view_phase1/preregistration.json",
        "scripts/graybox_cross_view_phase1.py",
        "scripts/graybox_cross_view_report.py",
        "scripts/test_cross_view_graph.py",
    )
    output = {}
    for relative in paths:
        with open(os.path.join(REPO, relative), "rb") as handle:
            output[relative] = hashlib.sha256(handle.read()).hexdigest()
    return output


def experiment_config(stage):
    resources = {
        "smoke": (SMOKE_REPLICATES, SMOKE_N),
        "development": (DEVELOPMENT_REPLICATES, DEVELOPMENT_N),
        "confirmation": (CONFIRMATION_REPLICATES, CONFIRMATION_N),
    }
    replicates, n = resources[stage]
    return {
        "version": VERSION,
        "stage": stage,
        "worlds": list(WORLDS),
        "replicates": replicates,
        "n": n,
        "seed_base": SEED_BASES[stage],
        "k_values": list(K_VALUES),
        "primary_k": PRIMARY_K,
        "lambdas": list(LAMBDAS),
        "primary_lambda": PRIMARY_LAMBDA,
        "permutations": PERMUTATIONS,
        "transfer_p_threshold": 0.025,
        "transfer_z_threshold": 2.0,
        "stability_cka_threshold": 0.75,
        "nuisance_ridge_alpha": 1.0,
        "nuisance_folds": 5,
    }


def combined_hash(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def metadata(stage, started, config_hash):
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        head = "unavailable"
    return {
        "version": VERSION,
        "stage": stage,
        "command": list(sys.argv),
        "config": experiment_config(stage),
        "config_hash": config_hash,
        "source_hashes": source_hashes(),
        "git_head": head,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "elapsed_seconds": time.time() - started,
    }


def run_stage(stage, out_dir):
    resources = {
        "smoke": (SMOKE_REPLICATES, SMOKE_N),
        "development": (DEVELOPMENT_REPLICATES, DEVELOPMENT_N),
        "confirmation": (CONFIRMATION_REPLICATES, CONFIRMATION_N),
    }
    replicates, n = resources[stage]
    rows, audits, datasets = [], [], []
    total = len(WORLDS) * replicates
    complete = 0
    for world_index, world in enumerate(WORLDS):
        for replicate in range(replicates):
            batch, audit_batch, dataset = run_dataset(
                stage, world, world_index, replicate, n
            )
            rows.extend(batch)
            audits.extend(audit_batch)
            datasets.append(dataset)
            complete += 1
            print(
                f"[{complete:02d}/{total:02d}] {stage} {world} replicate={replicate}",
                flush=True,
            )
    prefix = os.path.join(out_dir, stage)
    write_csv(prefix + "_per_run.csv", rows)
    write_csv(prefix + "_audit_diagnostics.csv", audits)
    write_csv(prefix + "_dataset_diagnostics.csv", datasets)
    return rows, audits, datasets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=("smoke", "development", "confirmation")
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--allow-confirmation", action="store_true")
    parser.add_argument("--frozen-hash", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()
    frozen_path = os.path.join(args.out_dir, "frozen_after_smoke.json")
    development_lock_path = os.path.join(args.out_dir, "development_lock.json")
    current_sources = source_hashes()
    development_config = experiment_config("development")
    smoke_freeze_payload = {
        "version": VERSION,
        "development_config": development_config,
        "source_hashes": current_sources,
    }
    smoke_freeze_hash = combined_hash(smoke_freeze_payload)

    if args.stage == "smoke":
        run_stage("smoke", args.out_dir)
        frozen = {**smoke_freeze_payload, "frozen_hash": smoke_freeze_hash}
        with open(frozen_path, "w", encoding="utf-8") as handle:
            json.dump(frozen, handle, indent=2, sort_keys=True)
        meta = metadata("smoke", started, smoke_freeze_hash)
        with open(os.path.join(args.out_dir, "smoke_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)
        print(json.dumps({
            "stage": "smoke",
            "frozen_hash": smoke_freeze_hash,
            "elapsed_seconds": time.time() - started,
        }, indent=2))
        return

    if not os.path.exists(frozen_path):
        raise FileNotFoundError("run --stage smoke before development")
    with open(frozen_path, encoding="utf-8") as handle:
        frozen = json.load(handle)
    if frozen != {**smoke_freeze_payload, "frozen_hash": smoke_freeze_hash}:
        raise ValueError("source or development config changed after smoke freeze")

    if args.stage == "development":
        run_stage("development", args.out_dir)
        lock_payload = {
            "version": VERSION,
            "development_freeze_hash": smoke_freeze_hash,
            "confirmation_config": experiment_config("confirmation"),
            "source_hashes": current_sources,
        }
        lock_hash = combined_hash(lock_payload)
        with open(development_lock_path, "w", encoding="utf-8") as handle:
            json.dump({**lock_payload, "confirmation_lock_hash": lock_hash}, handle,
                      indent=2, sort_keys=True)
        meta = metadata("development", started, smoke_freeze_hash)
        with open(os.path.join(args.out_dir, "development_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)
        print(json.dumps({
            "stage": "development",
            "development_freeze_hash": smoke_freeze_hash,
            "confirmation_lock_hash": lock_hash,
            "elapsed_seconds": time.time() - started,
            "confirmation_not_run": True,
        }, indent=2))
        return

    if not args.allow_confirmation:
        raise PermissionError("confirmation requires --allow-confirmation")
    if not os.path.exists(development_lock_path):
        raise FileNotFoundError("run development before confirmation")
    with open(development_lock_path, encoding="utf-8") as handle:
        lock = json.load(handle)
    expected = lock["confirmation_lock_hash"]
    if args.frozen_hash != expected:
        raise PermissionError("--frozen-hash does not match the development lock")
    if lock["source_hashes"] != current_sources:
        raise ValueError("source changed after development")
    run_stage("confirmation", args.out_dir)
    meta = metadata("confirmation", started, expected)
    with open(os.path.join(args.out_dir, "confirmation_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
