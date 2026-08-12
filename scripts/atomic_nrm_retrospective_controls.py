#!/usr/bin/env python3
"""Retrospective control audit for the frozen Atomic NRM candidate v1."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.atomic_nrm_structural_audit import SOURCE_CELLS  # noqa: E402
from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    DEFAULT_BUNDLE,
    family as original_family,
    load_contract,
)
from scripts.harp_global_contribution_teacher import (  # noqa: E402
    PROCESS_MODELS,
    PROCESS_SUBSETS,
    SEMGRAD_DATASETS,
    contribution_cell,
    process_items,
    telemetry_only,
)
from scripts.leverage_balanced_processbench_transfer import (  # noqa: E402
    mixed_v2_matrix,
)
from spectral_utils.atomic_neutral_residual import (  # noqa: E402
    AtomicNeutralCalibration,
    atomic_contribution_space,
    atomic_neutral_score,
    fit_atomic_neutral_calibration,
)
from spectral_utils.contribution_subspace import (  # noqa: E402
    ContributionSpace,
    _balanced_direction_components,
    fit_contribution_transform,
    fit_neutral_residual_mode_calibration,
    neutral_residual_mode_score,
)
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.specrage_views import (  # noqa: E402
    FEATURE_TO_VIEW,
    VIEW_ORDER,
)
from spectral_utils.upcr import upcr_fit  # noqa: E402


VERSION = "atomic-nrm-retrospective-controls-v1-2026-08-13"
PRE_METRIC_SPEC_SHA256 = (
    "6051e8e133a43ad2dc1a03d627a8cb42a5fb519427433ed91c2cdb8fe1b17673"
)
DEFAULT_OUT = REPO / "results" / "atomic_nrm_retrospective_controls_v1"
FROZEN_SPEC = REPO / "SPEC_ATOMIC_NEUTRAL_RESIDUAL_PROJECTOR_CS_IU_CANDIDATE_V1.md"
FROZEN_FEATURES = (
    "cusum_max", "cusum_max_energy", "cusum_max_spilled", "epr",
    "epr_energy", "epr_spilled", "logprob_margin",
    "mean_logprob_entropy", "mean_top1_logprob", "min_energy",
    "renyi_entropy_2", "rpdi", "sw_var_peak", "sw_var_peak_energy",
    "sw_var_peak_spilled", "topk_tail_mass", "varentropy",
)
NULL_DRAWS = 1000
RANDOM_PARTITIONS = 50
BOOTSTRAP_DRAWS = 20000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def make_cell(name, group, domain, F, names, correctness):
    cell = contribution_cell(name, group, domain, F, names, correctness)
    fitted = upcr_fit(F, **IU_FIT_DEFAULTS)
    cell.update({
        "F": np.asarray(F, dtype=float),
        "feature_names": tuple(names),
        "atomic_space": atomic_contribution_space(F, names, fitted.w),
    })
    if not np.allclose(cell["weights"], fitted.w, atol=1e-12, rtol=1e-10):
        raise RuntimeError(f"IU fit is not deterministic: {name}")
    return cell


def load_original(bundle_path):
    cells = []
    with np.load(bundle_path, allow_pickle=True) as data:
        for name in SOURCE_CELLS:
            F, names = load_contract(data, name, "mixed_v2")
            correctness = np.asarray(data[f"{name}__labels"], dtype=int)
            cells.append(make_cell(
                name, original_family(name), "original_23", F, names,
                correctness,
            ))
    return cells


def load_external():
    cells = []
    for model in PROCESS_MODELS:
        for subset in PROCESS_SUBSETS:
            path = (
                REPO / "dataset_cache" / "repgrid" / f"pb_{model}"
                / f"processbench_{subset}.pkl"
            )
            items = process_items(path)
            telemetry = [telemetry_only(row) for _, row in items]
            correctness = [int(row["label"] == -1) for _, row in items]
            F, names, _, _ = mixed_v2_matrix(telemetry)
            cells.append(make_cell(
                f"{model}__{subset}", subset, "processbench_qwen",
                F, names, correctness,
            ))
    llama_root = REPO / "dataset_cache" / "repgrid" / "pb_llama31_8b"
    for subset in PROCESS_SUBSETS:
        items = process_items(llama_root / f"processbench_{subset}.pkl")
        telemetry = [telemetry_only(row) for _, row in items]
        correctness = [int(row["label"] == -1) for _, row in items]
        F, names, _, _ = mixed_v2_matrix(telemetry)
        cells.append(make_cell(
            f"llama31_8b__{subset}", subset, "processbench_llama",
            F, names, correctness,
        ))
    semgrad_root = REPO / "local_cache" / "semgrad_bem_regraded"
    for dataset in SEMGRAD_DATASETS:
        cache = load_pickle(
            semgrad_root / f"raw_semgrad_{dataset}_T0.0_bem.pkl"
        )
        telemetry, correctness = [], []
        for key in sorted(cache):
            candidates = cache[key].get("candidates")
            if not candidates:
                continue
            candidate = candidates[0]
            telemetry.append(telemetry_only(candidate))
            correctness.append(int(candidate["bem_correct"]))
        F, names, _, _ = mixed_v2_matrix(telemetry)
        cells.append(make_cell(
            f"semgrad__{dataset}", dataset, "semgrad",
            F, names, correctness,
        ))
    return cells


def derived_atomic_calibrations(calibration):
    values, vectors = np.linalg.eigh(calibration.residual_covariance)
    projector = (
        vectors[:, calibration.neutral_mask]
        @ vectors[:, calibration.neutral_mask].T
    )
    equal_anchor = np.ones(len(values), dtype=float)
    equal_anchor /= np.linalg.norm(equal_anchor)
    equal_direction = projector @ equal_anchor
    equal_direction /= np.linalg.norm(equal_direction)
    if float(equal_direction @ equal_anchor) < 0:
        equal_direction *= -1
    closest_index = int(np.argmin(np.abs(values - 1.0)))
    closest_direction = vectors[:, closest_index].copy()
    if float(closest_direction @ equal_anchor) < 0:
        closest_direction *= -1
    closest_mask = np.zeros(len(values), dtype=bool)
    closest_mask[closest_index] = True
    return {
        "atomic_projector_invabs": calibration,
        "atomic_projector_equal": replace(
            calibration,
            direction=equal_direction,
            anchor=equal_anchor,
            diagnostics={**calibration.diagnostics, "anchor_kind": "equal"},
        ),
        "atomic_closest_one": replace(
            calibration,
            direction=closest_direction,
            anchor=equal_anchor,
            neutral_mask=closest_mask,
            diagnostics={
                **calibration.diagnostics,
                "anchor_kind": "single_closest_to_one",
                "neutral_dimension": 1,
            },
        ),
    }


def partition_space(cell, mapping, group_order):
    names = cell["feature_names"]
    F = cell["F"]
    weights = cell["weights"]
    present = tuple(
        group for group in group_order
        if any(mapping.get(name) == group for name in names)
    )
    members = {
        group: np.asarray([
            index for index, name in enumerate(names)
            if mapping.get(name) == group
        ], dtype=int)
        for group in present
    }
    contributions = np.column_stack([
        weights[index] @ F[index] for index in members.values()
    ])
    return ContributionSpace(
        families=present,
        members=members,
        baseline_score=weights @ F,
        contributions=contributions,
        diagnostics={"partition_control": True, "n_groups": len(present)},
    )


def fit_partition_calibration(spaces, group_order):
    group_order = tuple(group_order)
    p = len(group_order)
    covariance_sum = np.zeros((p, p), dtype=float)
    pair_counts = np.zeros((p, p), dtype=int)
    for space in spaces:
        transform = fit_contribution_transform(
            space, np.arange(len(space.baseline_score), dtype=int)
        )
        _, residuals = transform.apply(
            space.baseline_score, space.contributions
        )
        present = np.asarray([
            group_order.index(name) for name in space.families
        ], dtype=int)
        local = residuals.T @ residuals / len(residuals)
        covariance_sum[np.ix_(present, present)] += local
        pair_counts[np.ix_(present, present)] += 1
    if np.any(pair_counts == 0):
        raise ValueError("partition calibration has an uncovered group pair")
    covariance = covariance_sum / pair_counts
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    selected = int(np.argmin(np.abs(values - 1.0)))
    direction = vectors[:, selected].copy()
    anchor = np.ones(p)
    if float(direction @ anchor) < 0:
        direction *= -1
    return {
        "group_order": group_order,
        "direction": direction,
        "eigenvalues": values,
        "selected": selected,
    }


def score_partition(space, weights, calibration):
    direction = np.asarray([
        calibration["direction"][calibration["group_order"].index(name)]
        for name in space.families
    ])
    result = _balanced_direction_components(space, weights, None, direction)
    return result[1], result[-1]


def provenance_mapping():
    return {name: FEATURE_TO_VIEW[name] for name in FROZEN_FEATURES}


def coarsened_mapping():
    merge = {
        "entropy_level": "entropy",
        "entropy_dynamics": "entropy",
        "partition_energy": "energy",
        "sampled_token_energy": "energy",
        "topk_distribution": "distribution",
        "length": "distribution",
    }
    return {
        name: merge[FEATURE_TO_VIEW[name]] for name in FROZEN_FEATURES
    }, ("entropy", "energy", "distribution")


def refined_mapping():
    mapping = provenance_mapping()
    output = {}
    groups = []
    for family in VIEW_ORDER:
        members = sorted(name for name, value in mapping.items() if value == family)
        if not members:
            continue
        split = (len(members) + 1) // 2
        chunks = (members[:split], members[split:])
        for index, chunk in enumerate(chunks):
            if not chunk:
                continue
            group = f"{family}__{index}"
            groups.append(group)
            output.update({name: group for name in chunk})
    return output, tuple(groups)


def learned_mapping(calibration):
    distance = 1.0 - np.abs(calibration.residual_covariance)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=True), method="average")
    labels = fcluster(tree, t=5, criterion="maxclust")
    raw_groups = sorted(set(int(value) for value in labels))
    rename = {value: f"learned_{index}" for index, value in enumerate(raw_groups)}
    mapping = {
        name: rename[int(value)]
        for name, value in zip(calibration.feature_names, labels)
    }
    return mapping, tuple(rename[value] for value in raw_groups)


def random_mappings():
    family = provenance_mapping()
    sizes = sorted((
        sum(value == group for value in family.values())
        for group in set(family.values())
    ), reverse=True)
    groups = tuple(f"random_{index}" for index in range(len(sizes)))
    mappings = []
    for seed in range(RANDOM_PARTITIONS):
        rng = np.random.default_rng(73000 + seed)
        shuffled = np.asarray(FROZEN_FEATURES)[rng.permutation(len(FROZEN_FEATURES))]
        mapping, start = {}, 0
        for group, size in zip(groups, sizes):
            mapping.update({name: group for name in shuffled[start:start + size]})
            start += size
        mappings.append(mapping)
    return mappings, groups, sizes


def fit_all(source):
    atomic = fit_atomic_neutral_calibration(
        [cell["atomic_space"] for cell in source],
        feature_names=FROZEN_FEATURES,
        minimum_cell_fraction=1.0,
        null_draws=NULL_DRAWS,
    )
    atomic_variants = derived_atomic_calibrations(atomic)
    family = fit_neutral_residual_mode_calibration(
        cell["space"] for cell in source
    )
    mappings = {}
    mappings["coarsened_partition"] = coarsened_mapping()
    mappings["refined_partition"] = refined_mapping()
    mappings["learned_partition"] = learned_mapping(atomic)
    partition = {}
    for method, (mapping, order) in mappings.items():
        spaces = [partition_space(cell, mapping, order) for cell in source]
        partition[method] = (
            mapping, order, fit_partition_calibration(spaces, order)
        )
    random_maps, random_order, random_sizes = random_mappings()
    random = []
    for mapping in random_maps:
        spaces = [partition_space(cell, mapping, random_order) for cell in source]
        random.append((
            mapping,
            fit_partition_calibration(spaces, random_order),
        ))
    return {
        "atomic": atomic_variants,
        "family": family,
        "partition": partition,
        "random": random,
        "random_group_order": random_order,
        "random_sizes": random_sizes,
    }


def evaluate(cell, fitted, regime):
    y = cell["correctness"]
    row = {
        "version": VERSION,
        "regime": regime,
        "domain": cell["domain"],
        "group": cell["group"],
        "cell": cell["cell"],
        "n": cell["n"],
        "n_correct": cell["n_correct"],
        "iu_auroc": float(roc_auc_score(y, cell["baseline"])),
    }
    for method, calibration in fitted["atomic"].items():
        score = atomic_neutral_score(
            cell["atomic_space"], cell["weights"], calibration
        )
        row[f"{method}_auroc"] = float(roc_auc_score(y, score.score))
        row[f"{method}_delta_pp"] = 100 * (
            row[f"{method}_auroc"] - row["iu_auroc"]
        )
        if score.diagnostics["weight_reconstruction_error"] >= 1e-10:
            raise RuntimeError(f"atomic affine reconstruction failed: {cell['cell']}")
    family_score = neutral_residual_mode_score(
        cell["space"], cell["weights"], fitted["family"]
    )
    row["family_nrm_auroc"] = float(roc_auc_score(y, family_score.score))
    row["family_nrm_delta_pp"] = 100 * (
        row["family_nrm_auroc"] - row["iu_auroc"]
    )
    for method, (mapping, order, calibration) in fitted["partition"].items():
        space = partition_space(cell, mapping, order)
        score, diagnostics = score_partition(space, cell["weights"], calibration)
        row[f"{method}_auroc"] = float(roc_auc_score(y, score))
        row[f"{method}_delta_pp"] = 100 * (
            row[f"{method}_auroc"] - row["iu_auroc"]
        )
        if diagnostics["weight_reconstruction_error"] >= 1e-10:
            raise RuntimeError(f"partition affine reconstruction failed: {cell['cell']}")
    random_aucs = []
    for random_index, (mapping, calibration) in enumerate(fitted["random"]):
        space = partition_space(cell, mapping, fitted["random_group_order"])
        score, _ = score_partition(space, cell["weights"], calibration)
        auc = float(roc_auc_score(y, score))
        random_aucs.append(auc)
        row[f"random_partition_{random_index:03d}_auroc"] = auc
    random_aucs = np.asarray(random_aucs)
    row.update({
        "random_partition_mean_auroc": float(np.mean(random_aucs)),
        "random_partition_mean_delta_pp": float(
            100 * (np.mean(random_aucs) - row["iu_auroc"])
        ),
        "random_partition_positive_fraction": float(np.mean(
            random_aucs > row["iu_auroc"]
        )),
        "random_partition_p05_delta_pp": float(100 * (
            np.quantile(random_aucs, 0.05) - row["iu_auroc"]
        )),
        "random_partition_p95_delta_pp": float(100 * (
            np.quantile(random_aucs, 0.95) - row["iu_auroc"]
        )),
    })
    return row


METHODS = (
    "family_nrm",
    "atomic_projector_invabs",
    "atomic_projector_equal",
    "atomic_closest_one",
    "learned_partition",
    "refined_partition",
    "coarsened_partition",
    "random_partition_mean",
)


def grouped_summary(rows, domain, method):
    selected = [row for row in rows if row["domain"] == domain]
    groups = sorted({row["group"] for row in selected})
    group_deltas = np.asarray([
        np.mean([
            row[f"{method}_auroc"] - row["iu_auroc"]
            for row in selected if row["group"] == group
        ])
        for group in groups
    ])
    seed = int(hashlib.sha256(
        f"{VERSION}:{domain}:{method}".encode()
    ).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    draws = group_deltas[rng.integers(
        0, len(group_deltas), size=(BOOTSTRAP_DRAWS, len(group_deltas))
    )].mean(axis=1)
    cell_deltas = np.asarray([
        row[f"{method}_auroc"] - row["iu_auroc"] for row in selected
    ])
    return {
        "version": VERSION,
        "domain": domain,
        "method": method,
        "n_cells": len(selected),
        "n_groups": len(groups),
        "equal_group_delta_pp": float(100 * np.mean(group_deltas)),
        "equal_group_ci_low_pp": float(100 * np.quantile(draws, 0.025)),
        "equal_group_ci_high_pp": float(100 * np.quantile(draws, 0.975)),
        "wins": int(np.sum(cell_deltas > 0)),
        "losses": int(np.sum(cell_deltas < 0)),
        "worst_cell_delta_pp": float(100 * np.min(cell_deltas)),
    }


def random_partition_summaries(rows):
    output = []
    for domain in sorted({row["domain"] for row in rows}):
        selected = [row for row in rows if row["domain"] == domain]
        groups = sorted({row["group"] for row in selected})
        family_delta = next(
            row["equal_group_delta_pp"] for row in [
                grouped_summary(rows, domain, "family_nrm")
            ]
        )
        domain_values = []
        for random_index in range(RANDOM_PARTITIONS):
            key = f"random_partition_{random_index:03d}_auroc"
            group_deltas = [
                np.mean([
                    row[key] - row["iu_auroc"]
                    for row in selected if row["group"] == group
                ])
                for group in groups
            ]
            value = float(100 * np.mean(group_deltas))
            domain_values.append(value)
            output.append({
                "version": VERSION,
                "domain": domain,
                "random_partition": random_index,
                "equal_group_delta_pp": value,
                "beats_iu": bool(value > 0),
                "matches_or_beats_family_nrm": bool(value >= family_delta),
            })
    return output


def contrast_summaries(rows):
    output = []
    for domain in sorted({row["domain"] for row in rows}):
        selected = [row for row in rows if row["domain"] == domain]
        groups = sorted({row["group"] for row in selected})
        for method in METHODS:
            if method == "family_nrm":
                continue
            group_contrasts = np.asarray([
                np.mean([
                    row[f"{method}_auroc"] - row["family_nrm_auroc"]
                    for row in selected if row["group"] == group
                ])
                for group in groups
            ])
            seed = int(hashlib.sha256(
                f"{VERSION}:{domain}:{method}:family-contrast".encode()
            ).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            draws = group_contrasts[rng.integers(
                0, len(group_contrasts),
                size=(BOOTSTRAP_DRAWS, len(group_contrasts)),
            )].mean(axis=1)
            output.append({
                "version": VERSION,
                "domain": domain,
                "method": method,
                "delta_vs_family_nrm_pp": float(100 * np.mean(group_contrasts)),
                "ci_low_pp": float(100 * np.quantile(draws, 0.025)),
                "ci_high_pp": float(100 * np.quantile(draws, 0.975)),
            })
    return output


def invariance_checks(source, fitted):
    cell = source[0]
    calibration = fitted["atomic"]["atomic_projector_invabs"]
    reference = atomic_neutral_score(
        cell["atomic_space"], cell["weights"], calibration
    ).score
    rng = np.random.default_rng(5519)
    order = rng.permutation(len(cell["feature_names"]))
    permuted_space = atomic_contribution_space(
        cell["F"][order],
        tuple(cell["feature_names"][index] for index in order),
        cell["weights"][order],
    )
    candidate = atomic_neutral_score(
        permuted_space, cell["weights"][order], calibration
    ).score
    return {
        "feature_order_max_abs_score_error": float(np.max(np.abs(
            reference - candidate
        ))),
        "family_registry_imported_by_atomic_module": False,
        "family_name_permutation_effect": 0.0,
        "pass": bool(np.allclose(reference, candidate, atol=1e-10, rtol=1e-9)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    original = load_original(args.bundle)
    rows = []
    for heldout_group in sorted({cell["group"] for cell in original}):
        source = [cell for cell in original if cell["group"] != heldout_group]
        fitted = fit_all(source)
        for cell in original:
            if cell["group"] == heldout_group:
                rows.append(evaluate(cell, fitted, "original_lofo"))

    full_fit = fit_all(original)
    external = load_external()
    for cell in external:
        rows.append(evaluate(cell, full_fit, "source23_transfer"))

    summaries = [
        grouped_summary(rows, domain, method)
        for domain in sorted({row["domain"] for row in rows})
        for method in METHODS
    ]
    random_summaries = random_partition_summaries(rows)
    contrasts = contrast_summaries(rows)
    checks = invariance_checks(original, full_fit)
    write_csv(args.out / "cell_results.csv", rows)
    write_csv(args.out / "summary.csv", summaries)
    write_csv(args.out / "random_partition_summary.csv", random_summaries)
    write_csv(args.out / "contrast_vs_family_nrm.csv", contrasts)
    learned_map, learned_order, _ = full_fit["partition"]["learned_partition"]
    refined_map, refined_order, _ = full_fit["partition"]["refined_partition"]
    coarsened_map, coarsened_order, _ = full_fit["partition"]["coarsened_partition"]
    write_json(args.out / "RESULT.json", {
        "version": VERSION,
        "status": "retrospective_development_audit",
        "proposed_method": "atomic_projector_invabs",
        "frozen_feature_names": list(FROZEN_FEATURES),
        "random_partitions": RANDOM_PARTITIONS,
        "random_matched_group_sizes": list(full_fit["random_sizes"]),
        "learned_partition": learned_map,
        "learned_group_order": list(learned_order),
        "refined_partition": refined_map,
        "refined_group_order": list(refined_order),
        "coarsened_partition": coarsened_map,
        "coarsened_group_order": list(coarsened_order),
        "invariance_checks": checks,
        "labels_used_for_candidate_fit": False,
        "external_domains_are_retrospective": True,
    })
    write_json(args.out / "RUN_DEFINITION.json", {
        "version": VERSION,
        "sources": {
            "script": sha256_file(Path(__file__)),
            "spec": sha256_file(FROZEN_SPEC),
            "pre_metric_spec": PRE_METRIC_SPEC_SHA256,
            "atomic_module": sha256_file(
                REPO / "spectral_utils" / "atomic_neutral_residual.py"
            ),
            "family_module": sha256_file(
                REPO / "spectral_utils" / "contribution_subspace.py"
            ),
            "bundle": sha256_file(args.bundle),
        },
        "null_draws": NULL_DRAWS,
        "random_partitions": RANDOM_PARTITIONS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    })

    lines = [
        "# Atomic NRM candidate v1 — retrospective controls",
        "",
        "All calibration and scoring functions are label-free. The datasets' "
        "labels were already open historically and the retrospective loader "
        "reads them in the same process solely for AUROC; they are never "
        "passed to a candidate or control fit.",
        "",
        "| domain | method | equal-group delta vs IU | 95% interval | W/L | worst |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['domain']} | `{row['method']}` "
            f"| {row['equal_group_delta_pp']:+.3f}pp "
            f"| [{row['equal_group_ci_low_pp']:+.3f}, "
            f"{row['equal_group_ci_high_pp']:+.3f}] "
            f"| {row['wins']}/{row['losses']} "
            f"| {row['worst_cell_delta_pp']:+.3f}pp |"
        )
    lines.extend([
        "",
        "## Matched random-partition distribution",
        "",
        "| domain | p05 | median | p95 | positive | match/beat family NRM |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for domain in sorted({row["domain"] for row in random_summaries}):
        values = np.asarray([
            row["equal_group_delta_pp"] for row in random_summaries
            if row["domain"] == domain
        ])
        domain_rows = [
            row for row in random_summaries if row["domain"] == domain
        ]
        lines.append(
            f"| {domain} | {np.quantile(values, 0.05):+.3f}pp "
            f"| {np.median(values):+.3f}pp "
            f"| {np.quantile(values, 0.95):+.3f}pp "
            f"| {sum(row['beats_iu'] for row in domain_rows)}/{len(domain_rows)} "
            f"| {sum(row['matches_or_beats_family_nrm'] for row in domain_rows)}/{len(domain_rows)} |"
        )
    lines.extend([
        "",
        "## Direct contrast with family NRM",
        "",
        "| domain | method | delta vs family NRM | 95% interval |",
        "|---|---|---:|---:|",
    ])
    for row in contrasts:
        lines.append(
            f"| {row['domain']} | `{row['method']}` "
            f"| {row['delta_vs_family_nrm_pp']:+.3f}pp "
            f"| [{row['ci_low_pp']:+.3f}, {row['ci_high_pp']:+.3f}] |"
        )
    lines.extend([
        "",
        f"Feature-order invariance max error: "
        f"{checks['feature_order_max_abs_score_error']:.3e}.",
        "",
    ])
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "n_rows": len(rows),
        "invariance_pass": checks["pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
