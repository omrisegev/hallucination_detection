#!/usr/bin/env python3
"""SU-aware dependency-cleaning factorial for pooled graph roughness.

The ``fit`` phase never reads correctness labels.  It freezes compact
family-residual coordinates and graph roughness moments.  The ``report`` phase
verifies their hashes before opening labels for nested leave-dataset-family-out
selection and evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigh


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    DEFAULT_BUNDLE,
    family as dataset_family,
    load_contract,
)
from scripts.inscope_cells import INSCOPE  # noqa: E402
from spectral_utils.contribution_subspace import (  # noqa: E402
    fit_contribution_transform,
    iu_family_contributions,
)
from spectral_utils.dependency_fusion import sparse_upcr_fit  # noqa: E402
from spectral_utils.family_residual_graph import graphs_from_coordinates  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.pooled_graph_roughness import (  # noqa: E402
    graph_roughness_moment,
)
from spectral_utils.specrage_views import (  # noqa: E402
    FEATURE_TO_VIEW,
    VIEW_ORDER,
)
from spectral_utils.upcr import upcr_fit  # noqa: E402


VERSION = "su-pooled-graph-adaptation-sidecar-v1-2026-08-23"
DEFAULT_OUT = REPO / "results" / "su_pooled_graph_adaptation_sidecar_v1"
PROTOCOL = REPO / "docs" / "experiments" / "SU_POOLED_GRAPH_ADAPTATION_SIDECAR_V1.md"
ALPHAS = (0.25, 0.5, 1.0)
GRAPH_SETTINGS = ("union_k5", "union_k7", "union_k15", "adaptive_k7")
CALIBRATION_LAMBDAS = (0.1, 0.3, 1.0, 3.0, 10.0)
TRUST_FACTORS = (0.25, 0.5, 1.0)
CLEAN_MODES = ("observed", "all_sparse", "cross_sparse", "shared_cross")
RHO_MODES = ("iu", "su")
EPS = 1e-12


ARM_SPECS = (
    ("iu_observed_mean", "iu", "observed", "mean"),
    ("su_observed_mean", "su", "observed", "mean"),
    ("iu_all_sparse_mean", "iu", "all_sparse", "mean"),
    ("su_all_sparse_mean", "su", "all_sparse", "mean"),
    ("iu_cross_sparse_mean", "iu", "cross_sparse", "mean"),
    ("su_cross_sparse_mean", "su", "cross_sparse", "mean"),
    ("iu_shared_cross_mean", "iu", "shared_cross", "mean"),
    ("su_shared_cross_mean", "su", "shared_cross", "mean"),
    ("iu_observed_geomedian", "iu", "observed", "geomedian"),
    ("iu_cross_sparse_geomedian", "iu", "cross_sparse", "geomedian"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sym(matrix) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    return 0.5 * (values + values.T)


def psd_projection(matrix) -> tuple[np.ndarray, dict]:
    values, vectors = eigh(sym(matrix))
    clipped = np.maximum(values, 0.0)
    projected = sym((vectors * clipped) @ vectors.T)
    return projected, {
        "raw_min_eigenvalue": float(values[0]),
        "n_negative_eigenvalues": int(np.sum(values < -1e-10)),
        "psd_rank": int(np.sum(clipped > 1e-10)),
        "projection_relative_frobenius": float(
            np.linalg.norm(projected - matrix, ord="fro")
            / (np.linalg.norm(matrix, ord="fro") + EPS)
        ),
    }


def pcr_weights(covariance, rho, n_components: int = 2) -> np.ndarray:
    covariance = sym(covariance)
    rho = np.asarray(rho, dtype=float)
    values, vectors = eigh(covariance)
    order = np.argsort(values)[::-1][: min(int(n_components), len(values))]
    basis = vectors[:, order]
    reduced = sym(basis.T @ covariance @ basis)
    return np.asarray(basis @ (np.linalg.pinv(reduced, rcond=1e-10) @ (basis.T @ rho)))


def variant_id(rho_mode: str, clean_mode: str, alpha: float) -> str:
    alpha_text = f"{float(alpha):g}".replace(".", "p")
    return f"{rho_mode}__{clean_mode}__a{alpha_text}"


def alphas_for(clean_mode: str) -> tuple[float, ...]:
    return (0.0,) if clean_mode == "observed" else ALPHAS


def feature_cross_mask(names) -> np.ndarray:
    families = np.asarray([FEATURE_TO_VIEW[str(name)] for name in names])
    mask = families[:, None] != families[None, :]
    np.fill_diagonal(mask, False)
    return mask


def shared_cross_sparse(cell_records: list[dict]) -> tuple[tuple[str, ...], np.ndarray, dict]:
    roster = tuple(name for name in FEATURE_TO_VIEW if any(name in row["names"] for row in cell_records))
    index = {name: idx for idx, name in enumerate(roster)}
    groups = sorted({row["group"] for row in cell_records})
    group_matrices = []
    group_presence = []
    for group in groups:
        numerator = np.zeros((len(roster), len(roster)), dtype=float)
        denominator = np.zeros_like(numerator)
        for row in cell_records:
            if row["group"] != group:
                continue
            local = np.asarray([index[name] for name in row["names"]], dtype=int)
            cross = feature_cross_mask(row["names"])
            values = np.where(cross, row["S"], 0.0)
            numerator[np.ix_(local, local)] += values
            denominator[np.ix_(local, local)] += cross.astype(float)
        present = denominator > 0
        matrix = np.zeros_like(numerator)
        matrix[present] = numerator[present] / denominator[present]
        group_matrices.append(matrix)
        group_presence.append(present)
    numerator = np.zeros((len(roster), len(roster)), dtype=float)
    denominator = np.zeros_like(numerator)
    for matrix, present in zip(group_matrices, group_presence):
        numerator[present] += matrix[present]
        denominator[present] += 1.0
    pooled = np.zeros_like(numerator)
    present = denominator > 0
    pooled[present] = numerator[present] / denominator[present]
    pooled = sym(pooled)
    np.fill_diagonal(pooled, 0.0)
    return roster, pooled, {
        "n_groups": len(groups),
        "n_features": len(roster),
        "nonzero_pairs": int(np.sum(np.abs(pooled[np.triu_indices(len(roster), 1)]) > 0)),
    }


def local_shared_matrix(names, roster, shared) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(roster)}
    local = np.asarray([index[str(name)] for name in names], dtype=int)
    return np.asarray(shared[np.ix_(local, local)], dtype=float)


def contribution_state(F, names, weights) -> dict:
    space = iu_family_contributions(F, names, weights)
    transform = fit_contribution_transform(space, np.arange(F.shape[1], dtype=int))
    baseline, residuals = transform.apply(space.baseline_score, space.contributions)
    aligned = np.zeros((F.shape[1], len(VIEW_ORDER)), dtype=float)
    presence = np.zeros(len(VIEW_ORDER), dtype=bool)
    for local_idx, family_name in enumerate(space.families):
        global_idx = VIEW_ORDER.index(family_name)
        aligned[:, global_idx] = residuals[:, local_idx]
        presence[global_idx] = True
    return {
        "baseline": np.asarray(baseline, dtype=float),
        "residuals": aligned,
        "presence": presence,
        "families": tuple(space.families),
        "local_residuals": np.asarray(residuals, dtype=float),
        "reconstruction_error": float(space.diagnostics["reconstruction_error"]),
    }


def graph_moments(state: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    coordinates = state["local_residuals"]
    graphs = graphs_from_coordinates(coordinates, (5, 7, 15), topology="union")
    adaptive = graphs_from_coordinates(coordinates, (7,), topology="adaptive")[7]
    graph_by_setting = {
        "union_k5": graphs[5],
        "union_k7": graphs[7],
        "union_k15": graphs[15],
        "adaptive_k7": adaptive,
    }
    output = {}
    for setting, graph in graph_by_setting.items():
        moment = graph_roughness_moment(
            state["baseline"],
            state["local_residuals"],
            state["families"],
            graph,
        )
        output[setting] = (moment.A, moment.c)
    return output


def fit_command(args) -> None:
    bundle_path = Path(args.bundle).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    (out / "cells").mkdir()

    records = []
    with np.load(bundle_path, allow_pickle=True) as bundle:
        for cell in INSCOPE:
            F, names = load_contract(bundle, cell, "mixed_v2")
            iu = upcr_fit(F, **IU_FIT_DEFAULTS)
            su = sparse_upcr_fit(F)
            records.append({
                "cell": cell,
                "group": dataset_family(cell),
                "F": np.asarray(F, dtype=float),
                "names": tuple(str(name) for name in names),
                "C": sym(F @ F.T / F.shape[1]),
                "S": sym(su.decomposition.sparse),
                "rho_iu": np.asarray(iu.rho_hat, dtype=float),
                "rho_su": np.asarray(su.rho_hat, dtype=float),
                "w_iu": np.asarray(iu.w, dtype=float),
                "w_su": np.asarray(su.w_pcr, dtype=float),
                "su": su,
            })

    roster, shared_sparse, shared_diag = shared_cross_sparse(records)
    np.savez_compressed(out / "SHARED_CROSS_SPARSE.npz", names=np.asarray(roster), matrix=shared_sparse)
    cell_manifest = {}
    for cell_index, row in enumerate(records, 1):
        print(f"[{cell_index:02d}/{len(records)}] {row['cell']}", flush=True)
        F, names, C, S = row["F"], row["names"], row["C"], row["S"]
        cross = feature_cross_mask(names)
        shared_local = local_shared_matrix(names, roster, shared_sparse)
        payload = {
            "feature_names": np.asarray(names),
            "n_samples": np.asarray([F.shape[1]], dtype=np.int64),
        }
        diagnostics = {
            "cell": row["cell"],
            "group": row["group"],
            "n_samples": int(F.shape[1]),
            "n_features": int(F.shape[0]),
            "labels_read": False,
            "su_sparse_fraction": float(row["su"].decomposition.sparse_fraction),
            "su_g2_at_ceiling": bool(row["su"].meta["g2_at_ceiling"]),
            "variants": {},
        }
        for clean_mode in CLEAN_MODES:
            for alpha in alphas_for(clean_mode):
                if clean_mode == "observed":
                    covariance, psd_diag = C, {
                        "raw_min_eigenvalue": float(np.min(eigh(C, eigvals_only=True))),
                        "n_negative_eigenvalues": 0,
                        "psd_rank": int(F.shape[0]),
                        "projection_relative_frobenius": 0.0,
                    }
                else:
                    nuisance = {
                        "all_sparse": S,
                        "cross_sparse": np.where(cross, S, 0.0),
                        "shared_cross": shared_local,
                    }[clean_mode]
                    covariance, psd_diag = psd_projection(C - float(alpha) * nuisance)
                for rho_mode in RHO_MODES:
                    key = variant_id(rho_mode, clean_mode, alpha)
                    if clean_mode == "observed" and rho_mode == "iu":
                        weights = row["w_iu"]
                    elif clean_mode == "observed" and rho_mode == "su":
                        weights = row["w_su"]
                    else:
                        weights = pcr_weights(covariance, row[f"rho_{rho_mode}"], 2)
                    state = contribution_state(F, names, weights)
                    moments = graph_moments(state)
                    payload[f"baseline__{key}"] = state["baseline"]
                    payload[f"residuals__{key}"] = state["residuals"]
                    payload[f"presence__{key}"] = state["presence"].astype(np.int8)
                    for setting, (A, c) in moments.items():
                        payload[f"A__{key}__{setting}"] = A
                        payload[f"c__{key}__{setting}"] = c
                    diagnostics["variants"][key] = {
                        "clean_mode": clean_mode,
                        "rho_mode": rho_mode,
                        "alpha": float(alpha),
                        "weight_norm": float(np.linalg.norm(weights)),
                        "reconstruction_error": state["reconstruction_error"],
                        **psd_diag,
                    }
        path = out / "cells" / f"{row['cell']}.npz"
        np.savez_compressed(path, **payload)
        diag_path = out / "cells" / f"{row['cell']}.json"
        write_json(diag_path, diagnostics)
        cell_manifest[row["cell"]] = {
            "group": row["group"],
            "npz_sha256": sha256_file(path),
            "diagnostics_sha256": sha256_file(diag_path),
        }

    definition = {
        "version": VERSION,
        "status": "retrospective_isolated_sidecar",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_sha256": sha256_file(Path(__file__)),
        "bundle": str(bundle_path),
        "bundle_sha256": sha256_file(bundle_path),
        "cells": list(INSCOPE),
        "alphas": list(ALPHAS),
        "graph_settings": list(GRAPH_SETTINGS),
        "calibration_lambdas": list(CALIBRATION_LAMBDAS),
        "trust_factors": list(TRUST_FACTORS),
        "arm_specs": [list(row) for row in ARM_SPECS],
        "shared_cross_diagnostics": shared_diag,
        "labels_read_during_fit": False,
    }
    write_json(out / "RUN_DEFINITION.json", definition)
    write_json(out / "FIT_MANIFEST.json", {
        "version": VERSION,
        "cells": cell_manifest,
        "shared_cross_sha256": sha256_file(out / "SHARED_CROSS_SPARSE.npz"),
    })
    write_json(out / "FIT_COMPLETE.json", {
        "version": VERSION,
        "n_cells": len(records),
        "n_variants_per_cell": len(diagnostics["variants"]),
        "labels_read": False,
    })


def group_operator(records, variant: str, setting: str, group: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in records if row["group"] == group]
    return (
        np.mean([row["payload"][f"A__{variant}__{setting}"] for row in selected], axis=0),
        np.mean([row["payload"][f"c__{variant}__{setting}"] for row in selected], axis=0),
    )


def geomedian_weights(vectors: np.ndarray, max_iter: int = 200) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median[None, :]), axis=0)
    scale = np.where(mad > 1e-8, 1.4826 * mad, 1.0)
    standardized = (values - median[None, :]) / scale[None, :]
    center = np.mean(standardized, axis=0)
    weights = np.full(len(values), 1.0 / len(values))
    for _ in range(max_iter):
        distance = np.linalg.norm(standardized - center[None, :], axis=1)
        if np.min(distance) <= 1e-10:
            weights = np.zeros(len(values))
            weights[int(np.argmin(distance))] = 1.0
            break
        weights = 1.0 / np.maximum(distance, 1e-10)
        weights /= np.sum(weights)
        updated = weights @ standardized
        if np.linalg.norm(updated - center) <= 1e-10:
            center = updated
            break
        center = updated
    return weights


def pool_operator(records, groups, variant: str, setting: str, pooling: str):
    group_values = [group_operator(records, variant, setting, group) for group in groups]
    if pooling == "mean":
        weights = np.full(len(groups), 1.0 / len(groups))
    elif pooling == "geomedian":
        vectors = np.row_stack([
            np.concatenate([A[np.triu_indices(len(A))], c]) for A, c in group_values
        ])
        weights = geomedian_weights(vectors)
    else:
        raise ValueError(pooling)
    A = sym(sum(weight * value[0] for weight, value in zip(weights, group_values)))
    c = np.asarray(sum(weight * value[1] for weight, value in zip(weights, group_values)))
    return A, c, weights


def direction_for(records, groups, variant, setting, lambda_, pooling):
    A, c, weights = pool_operator(records, groups, variant, setting, pooling)
    direction = -float(lambda_) * np.linalg.solve(np.eye(len(c)) + float(lambda_) * A, c)
    return direction, weights


def candidate_score(row, variant: str, direction, trust: float) -> np.ndarray:
    baseline = np.asarray(row["payload"][f"baseline__{variant}"], dtype=float)
    residuals = np.asarray(row["payload"][f"residuals__{variant}"], dtype=float)
    presence = np.asarray(row["payload"][f"presence__{variant}"], dtype=bool)
    raw = residuals @ np.asarray(direction, dtype=float)
    sd = float(np.std(raw))
    correction = np.zeros_like(raw) if sd <= EPS else float(trust) * raw / (int(np.sum(presence)) * sd)
    return baseline + correction


def auc(row, score) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(row["labels"], score))


def canonical_auc(row) -> float:
    key = variant_id("iu", "observed", 0.0)
    return auc(row, row["payload"][f"baseline__{key}"])


def arm_variants(rho_mode: str, clean_mode: str) -> tuple[str, ...]:
    return tuple(variant_id(rho_mode, clean_mode, alpha) for alpha in alphas_for(clean_mode))


def parse_alpha(variant: str) -> float:
    return float(variant.rsplit("__a", 1)[1].replace("p", "."))


def graph_candidates(rho_mode: str, clean_mode: str):
    return tuple(
        (variant, setting, lambda_, trust)
        for variant in arm_variants(rho_mode, clean_mode)
        for setting in GRAPH_SETTINGS
        for lambda_ in CALIBRATION_LAMBDAS
        for trust in TRUST_FACTORS
    )


def mean_group_delta(records, group: str, score_fn) -> float:
    selected = [row for row in records if row["group"] == group]
    return float(np.mean([auc(row, score_fn(row)) - canonical_auc(row) for row in selected]))


def choose_graph_candidate(records, training_groups, candidates, pooling):
    values = {}
    for candidate in candidates:
        variant, setting, lambda_, trust = candidate
        held_values = []
        for held in training_groups:
            source_groups = [group for group in training_groups if group != held]
            direction, _ = direction_for(records, source_groups, variant, setting, lambda_, pooling)
            held_values.append(mean_group_delta(
                records, held, lambda row, v=variant, d=direction, t=trust: candidate_score(row, v, d, t)
            ))
        values[candidate] = float(np.mean(held_values))
    return max(
        candidates,
        key=lambda item: (
            values[item], -item[3], -item[2], -parse_alpha(item[0]),
            -int(item[1].split("k")[-1]), item[1],
        ),
    ), values


def choose_upstream_variant(records, training_groups, variants):
    values = {}
    for variant in variants:
        values[variant] = float(np.mean([
            mean_group_delta(
                records, held,
                lambda row, v=variant: row["payload"][f"baseline__{v}"],
            )
            for held in training_groups
        ]))
    return max(variants, key=lambda value: (values[value], -parse_alpha(value))), values


def bootstrap_ci(values, seed_text: str, draws: int = 50000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    return tuple(float(value) for value in np.quantile(np.mean(sampled, axis=1), (0.025, 0.975)))


def report_command(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bundle_path = Path(args.bundle).resolve()
    out = Path(args.out).resolve()
    definition = json.loads((out / "RUN_DEFINITION.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "FIT_MANIFEST.json").read_text(encoding="utf-8"))
    if definition["protocol_sha256"] != sha256_file(PROTOCOL):
        raise RuntimeError("protocol changed after fit")
    if definition["source_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("source changed after fit")
    records = []
    with np.load(bundle_path, allow_pickle=True) as bundle:
        for cell in INSCOPE:
            path = out / "cells" / f"{cell}.npz"
            if sha256_file(path) != manifest["cells"][cell]["npz_sha256"]:
                raise RuntimeError(f"fit hash mismatch: {cell}")
            labels = np.asarray(bundle[f"{cell}__labels"], dtype=int)
            if int(np.sum(labels)) < 20:
                continue
            records.append({
                "cell": cell,
                "group": dataset_family(cell),
                "labels": labels,
                "payload": np.load(path, allow_pickle=False),
            })
    groups = sorted({row["group"] for row in records})
    outer_rows = []
    selection_rows = []
    for arm_index, (arm, rho_mode, clean_mode, pooling) in enumerate(ARM_SPECS, 1):
        print(f"[{arm_index:02d}/{len(ARM_SPECS)}] nested {arm}", flush=True)
        candidates = graph_candidates(rho_mode, clean_mode)
        variants = arm_variants(rho_mode, clean_mode)
        for held in groups:
            training_groups = [group for group in groups if group != held]
            selected, _ = choose_graph_candidate(records, training_groups, candidates, pooling)
            selected_variant, setting, lambda_, trust = selected
            upstream_variant, _ = choose_upstream_variant(records, training_groups, variants)
            direction, group_weights = direction_for(
                records, training_groups, selected_variant, setting, lambda_, pooling
            )
            held_records = [row for row in records if row["group"] == held]
            graph_delta, matched_upstream_delta, graph_increment, upstream_selected_delta = [], [], [], []
            for row in held_records:
                iu_auc = canonical_auc(row)
                matched_auc = auc(row, row["payload"][f"baseline__{selected_variant}"])
                graph_auc = auc(row, candidate_score(row, selected_variant, direction, trust))
                upstream_auc = auc(row, row["payload"][f"baseline__{upstream_variant}"])
                graph_delta.append(graph_auc - iu_auc)
                matched_upstream_delta.append(matched_auc - iu_auc)
                graph_increment.append(graph_auc - matched_auc)
                upstream_selected_delta.append(upstream_auc - iu_auc)
            outer_rows.append({
                "arm": arm,
                "held_group": held,
                "rho_mode": rho_mode,
                "clean_mode": clean_mode,
                "pooling": pooling,
                "selected_variant": selected_variant,
                "selected_alpha": parse_alpha(selected_variant),
                "selected_graph": setting,
                "selected_lambda": lambda_,
                "selected_trust": trust,
                "independent_upstream_variant": upstream_variant,
                "independent_upstream_alpha": parse_alpha(upstream_variant),
                "graph_delta_vs_iu_pp": 100 * float(np.mean(graph_delta)),
                "matched_upstream_delta_vs_iu_pp": 100 * float(np.mean(matched_upstream_delta)),
                "graph_increment_pp": 100 * float(np.mean(graph_increment)),
                "independent_upstream_delta_vs_iu_pp": 100 * float(np.mean(upstream_selected_delta)),
                "direction": json.dumps(direction.tolist()),
                "source_group_weights": json.dumps(dict(zip(training_groups, map(float, group_weights)))),
            })

        full_selected, full_values = choose_graph_candidate(records, groups, candidates, pooling)
        full_upstream, upstream_values = choose_upstream_variant(records, groups, variants)
        selection_rows.append({
            "arm": arm,
            "selected_variant": full_selected[0],
            "selected_alpha": parse_alpha(full_selected[0]),
            "selected_graph": full_selected[1],
            "selected_lambda": full_selected[2],
            "selected_trust": full_selected[3],
            "cross_validated_delta_vs_iu_pp": 100 * full_values[full_selected],
            "independent_upstream_variant": full_upstream,
            "independent_upstream_alpha": parse_alpha(full_upstream),
            "upstream_cross_validated_delta_vs_iu_pp": 100 * upstream_values[full_upstream],
        })

    write_csv(out / "NESTED_OUTER.csv", outer_rows)
    write_csv(out / "FULL_SELECTION.csv", selection_rows)
    summary_rows = []
    for arm, _, _, _ in ARM_SPECS:
        selected = [row for row in outer_rows if row["arm"] == arm]
        graph_values = np.asarray([row["graph_delta_vs_iu_pp"] for row in selected])
        upstream_values = np.asarray([row["matched_upstream_delta_vs_iu_pp"] for row in selected])
        increment_values = np.asarray([row["graph_increment_pp"] for row in selected])
        independent_values = np.asarray([row["independent_upstream_delta_vs_iu_pp"] for row in selected])
        graph_ci = bootstrap_ci(graph_values, VERSION + arm + "graph")
        increment_ci = bootstrap_ci(increment_values, VERSION + arm + "increment")
        summary_rows.append({
            "arm": arm,
            "nested_graph_delta_vs_iu_pp": float(np.mean(graph_values)),
            "graph_ci_low_pp": graph_ci[0],
            "graph_ci_high_pp": graph_ci[1],
            "positive_groups": int(np.sum(graph_values > 0)),
            "worst_group_pp": float(np.min(graph_values)),
            "matched_upstream_delta_vs_iu_pp": float(np.mean(upstream_values)),
            "graph_increment_pp": float(np.mean(increment_values)),
            "increment_ci_low_pp": increment_ci[0],
            "increment_ci_high_pp": increment_ci[1],
            "independent_upstream_delta_vs_iu_pp": float(np.mean(independent_values)),
            "nrm_gain_recovery_fraction": float(np.mean(graph_values) / 0.277),
        })
    write_csv(out / "SUMMARY.csv", summary_rows)

    labels = [row[0] for row in ARM_SPECS]
    graph_means = np.asarray([next(row for row in summary_rows if row["arm"] == arm)["nested_graph_delta_vs_iu_pp"] for arm in labels])
    graph_low = np.asarray([next(row for row in summary_rows if row["arm"] == arm)["graph_ci_low_pp"] for arm in labels])
    graph_high = np.asarray([next(row for row in summary_rows if row["arm"] == arm)["graph_ci_high_pp"] for arm in labels])
    fig, ax = plt.subplots(figsize=(12, 6.5))
    y = np.arange(len(labels))
    ax.errorbar(graph_means, y, xerr=np.vstack([graph_means - graph_low, graph_high - graph_means]), fmt="o", capsize=3)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(0.277, color="#888888", linestyle="--", linewidth=1, label="Family-NRM +0.277pp")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Nested equal-dataset-family AUROC delta vs IU-PCR (pp)")
    ax.set_title("SU-aware covariance cleaning + pooled graph roughness")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "NESTED_METHOD_COMPARISON.png", dpi=180)
    plt.close(fig)

    matched = np.asarray([next(row for row in summary_rows if row["arm"] == arm)["matched_upstream_delta_vs_iu_pp"] for arm in labels])
    increments = np.asarray([next(row for row in summary_rows if row["arm"] == arm)["graph_increment_pp"] for arm in labels])
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.barh(y, matched, label="selected upstream covariance/rho", color="#4C78A8")
    ax.barh(y, increments, left=matched, label="pooled graph increment", color="#F58518")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC delta vs canonical IU-PCR (pp)")
    ax.set_title("Mechanism decomposition on outer folds")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "MECHANISM_DECOMPOSITION.png", dpi=180)
    plt.close(fig)

    heat = np.asarray([
        [next(row for row in outer_rows if row["arm"] == arm and row["held_group"] == group)["graph_delta_vs_iu_pp"] for group in groups]
        for arm in labels
    ])
    fig, ax = plt.subplots(figsize=(12, 7))
    limit = max(0.25, float(np.max(np.abs(heat))))
    image = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(groups)), groups, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Outer-family AUROC delta vs IU-PCR (pp)")
    fig.colorbar(image, ax=ax, label="pp")
    fig.tight_layout()
    fig.savefig(out / "OUTER_FAMILY_HEATMAP.png", dpi=180)
    plt.close(fig)

    by_arm = {row["arm"]: row for row in summary_rows}
    current = by_arm["iu_observed_mean"]
    primary = by_arm["iu_cross_sparse_mean"]
    report = [
        "# SU-aware pooled graph adaptation sidecar v1",
        "",
        "This is retrospective development evidence and does not alter a frozen baseline.",
        "",
        "## Headline",
        "",
        f"- Reproduced observed-IU pooled graph: **{current['nested_graph_delta_vs_iu_pp']:+.3f}pp** "
        f"[{current['graph_ci_low_pp']:+.3f},{current['graph_ci_high_pp']:+.3f}], "
        f"{current['positive_groups']}/8 positive families.",
        f"- Prespecified IU + cross-family sparse cleaning: **{primary['nested_graph_delta_vs_iu_pp']:+.3f}pp** "
        f"[{primary['graph_ci_low_pp']:+.3f},{primary['graph_ci_high_pp']:+.3f}], "
        f"graph increment {primary['graph_increment_pp']:+.3f}pp.",
        f"- Direct primary-minus-current point contrast: "
        f"**{primary['nested_graph_delta_vs_iu_pp'] - current['nested_graph_delta_vs_iu_pp']:+.3f}pp**.",
        "",
        "## All arms",
        "",
        "| arm | graph vs IU (pp) | 95% family bootstrap | wins | matched upstream vs IU | graph increment | independent no-graph |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report.append(
            f"| `{row['arm']}` | {row['nested_graph_delta_vs_iu_pp']:+.3f} | "
            f"[{row['graph_ci_low_pp']:+.3f},{row['graph_ci_high_pp']:+.3f}] | "
            f"{row['positive_groups']}/8 | {row['matched_upstream_delta_vs_iu_pp']:+.3f} | "
            f"{row['graph_increment_pp']:+.3f} | {row['independent_upstream_delta_vs_iu_pp']:+.3f} |"
        )
    report.extend([
        "",
        "## Interpretation boundary",
        "",
        "The graph/covariance directions are label-free, but the hyperparameters are selected retrospectively inside nested folds. "
        "With eight development families, the bootstrap intervals are descriptive. A positive arm must still be frozen and transferred; "
        "the table cannot be used to choose an unregistered maximum and call it confirmation.",
        "",
    ])
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json(out / "REPORT_COMPLETE.json", {
        "version": VERSION,
        "n_evaluable_cells": len(records),
        "n_groups": len(groups),
        "labels_opened_after_fit_hash_verification": True,
        "current_reproduction_pp": current["nested_graph_delta_vs_iu_pp"],
        "primary_pp": primary["nested_graph_delta_vs_iu_pp"],
    })
    print("\n".join(report[:12]))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "report"):
        command = sub.add_parser(name)
        command.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
        command.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.command == "fit":
        fit_command(args)
    else:
        report_command(args)


if __name__ == "__main__":
    main()
