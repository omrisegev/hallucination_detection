#!/usr/bin/env python3
"""Outcome-facing report for Graph Geometry Selection Research V1."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.graph_geometry_selection_fit import (  # noqa: E402
    ACTUATORS,
    DEFAULT_BUNDLE,
    DEFAULT_OUT,
    ELIGIBLE_CELLS,
    LAMBDAS,
    NODE_CONTROL_LAMBDA,
    NODE_CONTROL_TRUST,
    N_NODE_PERMUTATIONS,
    ORIGINAL_BUNDLE,
    TRUST_CLASSES,
    TRUST_FACTORS,
    VERSION,
    array_hash,
    basis_key,
    calibration_key,
    candidate_key,
    canonical_hash,
    node_basis_key,
    node_score_key,
    run_definition,
    sha256_file,
    write_json,
)
from scripts.hard_filter_dufs_liu_benchmark import family as dataset_family  # noqa: E402
from spectral_utils.graph_geometry_selection import (  # noqa: E402
    GEOMETRIES,
    INTRINSIC_LAMBDA,
    INTRINSIC_TRUST,
    phase_a_geometry_ids,
)
from spectral_utils.specrage_views import VIEW_ORDER  # noqa: E402


TAIL_FLOOR = -0.005
BOOTSTRAPS = 200_000
BOOTSTRAP_SEED = 20260823
ANCHOR_ONE_SE_PP = 0.25147679442711046
ANCHOR_MAX_MEAN_PP = 0.449629196668661
LEGACY_V1_PP = 0.4516058351238263
LEGACY_V1_LAMBDAS = (0.1, 0.3, 1.0, 3.0, 10.0)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_once(path: Path, payload) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f"refusing to overwrite changed frozen file: {path}")
        return
    write_json(path, payload)


def geometry_priority(geometry_id: str) -> int:
    return next(
        index for index, spec in enumerate(GEOMETRIES)
        if spec.geometry_id == geometry_id
    )


def candidates(geometry_ids, trusts, lambdas=LAMBDAS):
    return tuple(
        (geometry_id, lambda_, trust)
        for geometry_id in geometry_ids
        for lambda_ in lambdas
        for trust in trusts
    )


def all_prefixes(cell: str, groups):
    target = dataset_family(cell)
    return ("full", "outer") + tuple(
        f"inner={outer}" for outer in groups if outer != target
    )


def verify_fit(out: Path, bundle: Path) -> tuple[dict, dict, dict]:
    definition = json.loads((out / "RUN_DEFINITION.json").read_text())
    current = run_definition(bundle)
    if definition != current:
        raise RuntimeError("run definition or frozen source hash changed")
    complete = json.loads((out / "FIT_COMPLETE.json").read_text())
    payload = dict(complete)
    recorded = payload.pop("manifest_hash")
    if canonical_hash(payload) != recorded:
        raise RuntimeError("fit manifest is not self-consistent")
    if complete["definition_hash"] != definition["definition_hash"]:
        raise RuntimeError("fit/run-definition mismatch")
    if complete.get("labels_accessed_by_fit") is not False:
        raise RuntimeError("fit does not certify label non-access")
    if complete.get("target_fields_physically_present_in_fit_input") is not False:
        raise RuntimeError("fit input was not physically target-free")
    if complete.get("target_fields_received_by_fit") != []:
        raise RuntimeError("fit received target fields")
    if set(complete["state_hashes"]) != set(ELIGIBLE_CELLS):
        raise RuntimeError("state roster changed")
    if set(complete["basis_hashes"]) != set(ELIGIBLE_CELLS):
        raise RuntimeError("score-basis roster changed")
    for cell, expected in complete["state_hashes"].items():
        if sha256_file(out / "states" / f"{cell}.npz") != expected:
            raise RuntimeError(f"state hash changed: {cell}")
    for cell, expected in complete["basis_hashes"].items():
        if sha256_file(out / "score_basis" / f"{cell}.npz") != expected:
            raise RuntimeError(f"score-basis hash changed: {cell}")
    named = {
        "CALIBRATIONS.json": "calibrations_sha256",
        "DIAGNOSTICS.json": "diagnostics_sha256",
        "ACTUATOR_DIAGNOSTICS.json": "actuator_diagnostics_sha256",
        "NODE_PERMUTATION_CONTROLS.json": "node_permutation_controls_sha256",
        "GRAPH_DIVERSITY.json": "diversity_sha256",
        "FROZEN_LABELFREE_SELECTION.json": "label_free_selection_sha256",
        "SCORE_HASHES.json": "score_hashes_sha256",
    }
    for filename, field in named.items():
        if sha256_file(out / filename) != complete[field]:
            raise RuntimeError(f"frozen fit artifact changed: {filename}")
    intrinsic = json.loads((out / "FROZEN_LABELFREE_SELECTION.json").read_text())
    for name, context in intrinsic["contexts"].items():
        diagnostics = context["diagnostics"]
        if diagnostics["valid_count"] <= 0 or diagnostics["fallback_to_invalid_pool"]:
            raise RuntimeError(f"intrinsic hard filter failed closed: {name}")
    score_hashes = json.loads((out / "SCORE_HASHES.json").read_text())
    return complete, definition, score_hashes


def verify_every_candidate_score_without_labels(
    out: Path, complete: dict, score_hashes: dict
) -> dict:
    """Reconstruct and hash every candidate before the label archive is opened."""

    active = tuple(complete["active_geometry_ids"])
    groups = tuple(sorted({dataset_family(cell) for cell in ELIGIBLE_CELLS}))
    verified = 0
    for index, cell in enumerate(ELIGIBLE_CELLS, start=1):
        print(f"[{index}/{len(ELIGIBLE_CELLS)}] pre-label score verification {cell}", flush=True)
        expected_hashes = score_hashes[cell]
        with np.load(out / "score_basis" / f"{cell}.npz") as stored:
            baseline = np.asarray(stored["iu"], dtype=float)
            if array_hash(baseline) != expected_hashes["iu"]:
                raise RuntimeError(f"IU array hash changed: {cell}")
            sample_index = np.asarray(stored["sample_index"], dtype=int)
            if not np.array_equal(sample_index, np.arange(len(baseline))):
                raise RuntimeError(f"sample identity changed: {cell}")
            expected_basis = {"iu", "sample_index"}
            for prefix in all_prefixes(cell, groups):
                for geometry_id in active:
                    for lambda_ in LAMBDAS:
                        key = basis_key(prefix, geometry_id, lambda_)
                        expected_basis.add(key)
                        correction = np.asarray(stored[key], dtype=float)
                        if correction.shape != baseline.shape or not np.isfinite(correction).all():
                            raise RuntimeError(f"invalid correction basis: {cell}/{key}")
                        for trust in TRUST_FACTORS:
                            score_name = f"{prefix}__{candidate_key(geometry_id, lambda_, trust)}"
                            if array_hash(baseline + trust * correction) != expected_hashes[score_name]:
                                raise RuntimeError(f"candidate score hash changed: {cell}/{score_name}")
                            verified += 1
                    cross_key = basis_key(prefix, geometry_id, None, "cross")
                    expected_basis.add(cross_key)
                    cross_correction = np.asarray(stored[cross_key], dtype=float)
                    if (
                        cross_correction.shape != baseline.shape
                        or not np.isfinite(cross_correction).all()
                    ):
                        raise RuntimeError(
                            f"invalid cross correction basis: {cell}/{cross_key}"
                        )
                    for trust in TRUST_FACTORS:
                        score_name = (
                            f"{prefix}__"
                            f"{candidate_key(geometry_id, None, trust, 'cross')}"
                        )
                        if array_hash(
                            baseline + trust * cross_correction
                        ) != expected_hashes[score_name]:
                            raise RuntimeError(
                                f"cross candidate score hash changed: {cell}/{score_name}"
                            )
                        verified += 1
            if set(stored.files) != expected_basis:
                # Node-control bases are added below before the registry check.
                pass
            for geometry_id in active:
                for replicate in range(N_NODE_PERMUTATIONS):
                    for actuator in ACTUATORS:
                        key = node_basis_key(replicate, geometry_id, actuator)
                        expected_basis.add(key)
                        correction = np.asarray(stored[key], dtype=float)
                        if correction.shape != baseline.shape or not np.isfinite(correction).all():
                            raise RuntimeError(f"invalid node-control basis: {cell}/{key}")
                        score_name = node_score_key(replicate, geometry_id, actuator)
                        if array_hash(
                            baseline + NODE_CONTROL_TRUST * correction
                        ) != expected_hashes[score_name]:
                            raise RuntimeError(
                                f"node-control score hash changed: {cell}/{score_name}"
                            )
                        verified += 1
            if set(stored.files) != expected_basis:
                raise RuntimeError(f"score-basis registry changed: {cell}")
        expected_score_hashes = {"iu"}
        expected_score_hashes.update(
            f"{prefix}__{candidate_key(geometry_id, lambda_, trust, 'full')}"
            for prefix in all_prefixes(cell, groups)
            for geometry_id in active
            for lambda_ in LAMBDAS
            for trust in TRUST_FACTORS
        )
        expected_score_hashes.update(
            node_score_key(replicate, geometry_id, actuator)
            for geometry_id in active
            for replicate in range(N_NODE_PERMUTATIONS)
            for actuator in ACTUATORS
        )
        expected_score_hashes.update(
            f"{prefix}__{candidate_key(geometry_id, None, trust, 'cross')}"
            for prefix in all_prefixes(cell, groups)
            for geometry_id in active
            for trust in TRUST_FACTORS
        )
        if set(expected_hashes) != expected_score_hashes:
            raise RuntimeError(f"candidate score-hash registry changed: {cell}")
    if verified != complete["candidate_score_count"]:
        raise RuntimeError("candidate score verification count mismatch")
    return {
        "candidate_scores_verified_before_label_open": verified,
        "all_score_hashes_matched": True,
        "labels_available_to_verifier": False,
    }


def load_labels_after_verification(definition: dict) -> dict[str, np.ndarray]:
    if sha256_file(ORIGINAL_BUNDLE) != definition["original_bundle_sha256"]:
        raise RuntimeError("label archive changed after fit")
    labels = {}
    with np.load(ORIGINAL_BUNDLE, allow_pickle=True) as data:
        for cell in ELIGIBLE_CELLS:
            values = np.asarray(data[f"{cell}__labels"], dtype=int)
            if values.ndim != 1 or not np.all(np.isin(values, (0, 1))):
                raise RuntimeError(f"invalid labels: {cell}")
            labels[cell] = values
    return labels


def build_metric_index(out: Path, complete: dict, labels: dict):
    active = tuple(complete["active_geometry_ids"])
    groups = tuple(sorted({dataset_family(cell) for cell in ELIGIBLE_CELLS}))
    index = {}
    rows = []
    for cell_index, cell in enumerate(ELIGIBLE_CELLS, start=1):
        print(f"[{cell_index}/{len(ELIGIBLE_CELLS)}] outcome metrics {cell}", flush=True)
        y = labels[cell]
        with np.load(out / "score_basis" / f"{cell}.npz") as stored:
            baseline = np.asarray(stored["iu"], dtype=float)
            if baseline.shape != y.shape:
                raise RuntimeError(f"score/label shape mismatch: {cell}")
            iu_auc = float(roc_auc_score(y, baseline))
            iu_ap = float(average_precision_score(y, baseline))
            index[(cell, "iu", "auroc")] = iu_auc
            index[(cell, "iu", "auprc")] = iu_ap
            for prefix in all_prefixes(cell, groups):
                for geometry_id in active:
                    for lambda_ in LAMBDAS:
                        correction = np.asarray(
                            stored[basis_key(prefix, geometry_id, lambda_)], dtype=float
                        )
                        for trust in TRUST_FACTORS:
                            score = baseline + trust * correction
                            auc = float(roc_auc_score(y, score))
                            ap = float(average_precision_score(y, score))
                            candidate = (geometry_id, lambda_, trust)
                            index[(cell, prefix, candidate, "auroc")] = auc - iu_auc
                            index[(cell, prefix, candidate, "auprc")] = ap - iu_ap
                            rows.append({
                                "cell": cell,
                                "group": dataset_family(cell),
                                "prefix": prefix,
                                "actuator": "full",
                                "geometry_id": geometry_id,
                                "lambda": lambda_,
                                "trust_factor": trust,
                                "iu_auroc": iu_auc,
                                "candidate_auroc": auc,
                                "delta_auroc": auc - iu_auc,
                                "iu_auprc": iu_ap,
                                "candidate_auprc": ap,
                                "delta_auprc": ap - iu_ap,
                            })
                    cross_correction = np.asarray(
                        stored[basis_key(prefix, geometry_id, None, "cross")],
                        dtype=float,
                    )
                    for trust in TRUST_FACTORS:
                        score = baseline + trust * cross_correction
                        auc = float(roc_auc_score(y, score))
                        ap = float(average_precision_score(y, score))
                        candidate = (geometry_id, None, trust, "cross")
                        index[(cell, prefix, candidate, "auroc")] = auc - iu_auc
                        index[(cell, prefix, candidate, "auprc")] = ap - iu_ap
                        rows.append({
                            "cell": cell,
                            "group": dataset_family(cell),
                            "prefix": prefix,
                            "actuator": "cross",
                            "geometry_id": geometry_id,
                            "lambda": None,
                            "trust_factor": trust,
                            "iu_auroc": iu_auc,
                            "candidate_auroc": auc,
                            "delta_auroc": auc - iu_auc,
                            "iu_auprc": iu_ap,
                            "candidate_auprc": ap,
                            "delta_auprc": ap - iu_ap,
                        })
            for geometry_id in active:
                for replicate in range(N_NODE_PERMUTATIONS):
                    for actuator in ACTUATORS:
                        correction = np.asarray(
                            stored[node_basis_key(
                                replicate, geometry_id, actuator
                            )],
                            dtype=float,
                        )
                        score = baseline + NODE_CONTROL_TRUST * correction
                        auc = float(roc_auc_score(y, score))
                        ap = float(average_precision_score(y, score))
                        index[(
                            cell, "node_control", geometry_id, replicate,
                            actuator, "auroc",
                        )] = auc - iu_auc
                        index[(
                            cell, "node_control", geometry_id, replicate,
                            actuator, "auprc",
                        )] = ap - iu_ap
                        rows.append({
                            "cell": cell,
                            "group": dataset_family(cell),
                            "prefix": f"nodeperm={replicate:02d}_outer",
                            "actuator": actuator,
                            "geometry_id": geometry_id,
                            "lambda": (
                                NODE_CONTROL_LAMBDA
                                if actuator == "full" else None
                            ),
                            "trust_factor": NODE_CONTROL_TRUST,
                            "iu_auroc": iu_auc,
                            "candidate_auroc": auc,
                            "delta_auroc": auc - iu_auc,
                            "iu_auprc": iu_ap,
                            "candidate_auprc": ap,
                            "delta_auprc": ap - iu_ap,
                        })
    return index, rows


def group_delta(index, group, prefix, candidate, metric="auroc") -> float:
    cells = [
        cell for cell in ELIGIBLE_CELLS if dataset_family(cell) == group
    ]
    return float(np.mean([
        index[(cell, prefix, candidate, metric)] for cell in cells
    ]))


def node_control_group_delta(
    index, group, geometry_id, replicate, actuator, metric="auroc"
) -> float:
    cells = [
        cell for cell in ELIGIBLE_CELLS if dataset_family(cell) == group
    ]
    return float(np.mean([
        index[(
            cell, "node_control", geometry_id, replicate, actuator, metric,
        )]
        for cell in cells
    ]))


def summarize_node_permutation_controls(index, active, groups):
    rows = []
    summaries = []
    for geometry_id in active:
        for actuator in ACTUATORS:
            real_candidate = (
                (geometry_id, NODE_CONTROL_LAMBDA, NODE_CONTROL_TRUST)
                if actuator == "full"
                else (geometry_id, None, NODE_CONTROL_TRUST, "cross")
            )
            real = np.asarray([
                group_delta(index, group, "outer", real_candidate, "auroc")
                for group in groups
            ])
            replicate_vectors = []
            for replicate in range(N_NODE_PERMUTATIONS):
                values = np.asarray([
                    node_control_group_delta(
                        index, group, geometry_id, replicate, actuator, "auroc"
                    )
                    for group in groups
                ])
                replicate_vectors.append(values)
                rows.append({
                    "geometry_id": geometry_id,
                    "actuator": actuator,
                    "replicate": replicate,
                    "delta_auroc_pp": 100 * float(np.mean(values)),
                    "group_values_pp": json.dumps((100 * values).tolist()),
                })
            replicate_vectors = np.asarray(replicate_vectors, dtype=float)
            null_mean_by_group = np.mean(replicate_vectors, axis=0)
            paired = real - null_mean_by_group
            replicate_means = np.mean(replicate_vectors, axis=1)
            summaries.append({
                "geometry_id": geometry_id,
                "actuator": actuator,
                "lambda_full": (
                    NODE_CONTROL_LAMBDA if actuator == "full" else None
                ),
                "trust_factor": NODE_CONTROL_TRUST,
                "real_delta_pp": 100 * float(np.mean(real)),
                "permutation_mean_delta_pp": 100 * float(np.mean(replicate_means)),
                "permutation_min_delta_pp": 100 * float(np.min(replicate_means)),
                "permutation_max_delta_pp": 100 * float(np.max(replicate_means)),
                "randomization_p_greater_or_equal": float(
                    (1 + np.sum(replicate_means >= np.mean(real) - 1e-15))
                    / (N_NODE_PERMUTATIONS + 1)
                ),
                **{
                    f"real_minus_permutation_mean_{key}": value
                    for key, value in bootstrap_mean(
                        paired,
                        seed_offset=(
                            1500 + 100 * geometry_priority(geometry_id)
                            + (actuator == "cross")
                        ),
                    ).items()
                },
                "real_group_values_pp": json.dumps((100 * real).tolist()),
                "permutation_mean_group_values_pp": json.dumps(
                    (100 * null_mean_by_group).tolist()
                ),
            })
    return rows, summaries


def candidate_values(
    index, validation_groups, *, outer_held, geometry_ids, trusts,
    lambdas=LAMBDAS,
):
    prefix = "full" if outer_held is None else f"inner={outer_held}"
    output = {}
    for candidate in candidates(geometry_ids, trusts, lambdas):
        output[candidate] = {
            group: group_delta(index, group, prefix, candidate, "auroc")
            for group in validation_groups
        }
    return output


def candidate_summary(candidate, values, groups):
    vector = np.asarray([values[candidate][group] for group in groups], dtype=float)
    return {
        "geometry_id": candidate[0],
        "lambda": float(candidate[1]),
        "trust_factor": float(candidate[2]),
        "mean": float(np.mean(vector)),
        "se": float(np.std(vector, ddof=1) / np.sqrt(len(vector))),
        "worst": float(np.min(vector)),
        "values": {group: float(value) for group, value in zip(groups, vector)},
    }


def choose_one_se(values, groups):
    summaries = {
        candidate: candidate_summary(candidate, values, groups)
        for candidate in values
    }
    best = max(
        summaries,
        key=lambda candidate: (
            summaries[candidate]["mean"],
            -candidate[2], -candidate[1], -geometry_priority(candidate[0]),
        ),
    )
    threshold = summaries[best]["mean"] - summaries[best]["se"]
    eligible = [
        candidate for candidate in summaries
        if summaries[candidate]["mean"] >= threshold - 1e-15
    ]
    tail_safe = [
        candidate for candidate in eligible
        if summaries[candidate]["worst"] >= TAIL_FLOOR
    ]
    pool = tail_safe if tail_safe else eligible
    selected = min(
        pool,
        key=lambda candidate: (
            candidate[2], candidate[1],
            -summaries[candidate]["mean"],
            geometry_priority(candidate[0]),
        ),
    )
    return selected, {
        "policy": "one_se_then_tail_then_min_trust_lambda_then_mean_geometry_priority",
        "best": summaries[best],
        "threshold": float(threshold),
        "eligible_count": len(eligible),
        "tail_safe_count": len(tail_safe),
        "selected": summaries[selected],
    }


def choose_max_mean(values, groups):
    summaries = {
        candidate: candidate_summary(candidate, values, groups)
        for candidate in values
    }
    selected = max(
        summaries,
        key=lambda candidate: (
            summaries[candidate]["mean"],
            -candidate[2], -candidate[1], -geometry_priority(candidate[0]),
        ),
    )
    return selected, {
        "policy": "max_inner_family_mean",
        "selected": summaries[selected],
    }


def nested_arm(index, geometry_ids, trusts, selector, groups, *, lambdas=LAMBDAS):
    rows = []
    for held in groups:
        training = tuple(group for group in groups if group != held)
        values = candidate_values(
            index, training, outer_held=held,
            geometry_ids=geometry_ids, trusts=trusts, lambdas=lambdas,
        )
        selected, diagnostics = selector(values, training)
        rows.append({
            "held_group": held,
            "geometry_id": selected[0],
            "lambda": selected[1],
            "trust_factor": selected[2],
            "held_delta_auroc": group_delta(index, held, "outer", selected, "auroc"),
            "held_delta_auprc": group_delta(index, held, "outer", selected, "auprc"),
            "inner_selected_mean": diagnostics["selected"]["mean"],
            "selection": diagnostics,
        })
    return rows


def cross_candidate_values(index, validation_groups, *, outer_held, geometry_id, trusts):
    prefix = "full" if outer_held is None else f"inner={outer_held}"
    output = {}
    for trust in trusts:
        candidate = (geometry_id, None, trust, "cross")
        output[trust] = {
            group: group_delta(index, group, prefix, candidate, "auroc")
            for group in validation_groups
        }
    return output


def cross_summary(trust, values, groups, geometry_id):
    vector_ = np.asarray([values[trust][group] for group in groups], dtype=float)
    return {
        "geometry_id": geometry_id,
        "actuator": "cross",
        "lambda": None,
        "trust_factor": float(trust),
        "mean": float(np.mean(vector_)),
        "se": float(np.std(vector_, ddof=1) / np.sqrt(len(vector_))),
        "worst": float(np.min(vector_)),
        "values": {group: float(value) for group, value in zip(groups, vector_)},
    }


def choose_cross_one_se(values, groups, geometry_id):
    summaries = {
        trust: cross_summary(trust, values, groups, geometry_id)
        for trust in values
    }
    best = max(summaries, key=lambda trust: (summaries[trust]["mean"], -trust))
    threshold = summaries[best]["mean"] - summaries[best]["se"]
    eligible = [
        trust for trust in summaries
        if summaries[trust]["mean"] >= threshold - 1e-15
    ]
    tail_safe = [
        trust for trust in eligible if summaries[trust]["worst"] >= TAIL_FLOOR
    ]
    pool = tail_safe if tail_safe else eligible
    selected = min(pool, key=lambda trust: (trust, -summaries[trust]["mean"]))
    return selected, {
        "policy": "cross_one_se_then_tail_then_min_trust;lambda_absent",
        "best": summaries[best],
        "threshold": float(threshold),
        "eligible_count": len(eligible),
        "tail_safe_count": len(tail_safe),
        "selected": summaries[selected],
    }


def choose_cross_max_mean(values, groups, geometry_id):
    summaries = {
        trust: cross_summary(trust, values, groups, geometry_id)
        for trust in values
    }
    selected = max(summaries, key=lambda trust: (summaries[trust]["mean"], -trust))
    return selected, {
        "policy": "cross_max_inner_family_mean;lambda_absent",
        "selected": summaries[selected],
    }


def nested_cross_arm(index, geometry_id, trusts, selector_name, groups):
    rows = []
    selector = (
        choose_cross_one_se if selector_name == "one_se" else choose_cross_max_mean
    )
    for held in groups:
        training = tuple(group for group in groups if group != held)
        values = cross_candidate_values(
            index, training, outer_held=held,
            geometry_id=geometry_id, trusts=trusts,
        )
        trust, diagnostics = selector(values, training, geometry_id)
        candidate = (geometry_id, None, trust, "cross")
        rows.append({
            "held_group": held,
            "geometry_id": geometry_id,
            "lambda": None,
            "trust_factor": trust,
            "held_delta_auroc": group_delta(index, held, "outer", candidate, "auroc"),
            "held_delta_auprc": group_delta(index, held, "outer", candidate, "auprc"),
            "inner_selected_mean": diagnostics["selected"]["mean"],
            "selection": diagnostics,
        })
    return rows


def run_actuator_factor(index, active, groups):
    """Run full and cross as disjoint arms; never select between actuators."""
    arm_rows = []
    paired_rows = []
    selector_functions = {"one_se": choose_one_se, "max_mean": choose_max_mean}
    for geometry_id in active:
        for selector_name, full_selector in selector_functions.items():
            for trust_name, trusts in TRUST_CLASSES.items():
                full_rows = nested_arm(
                    index, (geometry_id,), trusts, full_selector, groups
                )
                cross_rows = nested_cross_arm(
                    index, geometry_id, trusts, selector_name, groups
                )
                for actuator, values in (("full", full_rows), ("cross", cross_rows)):
                    summary = bootstrap_mean(
                        vector(values),
                        seed_offset=(
                            900 + 100 * geometry_priority(geometry_id)
                            + 10 * (selector_name == "max_mean")
                            + list(TRUST_CLASSES).index(trust_name)
                            + (actuator == "cross")
                        ),
                    )
                    arm_rows.append({
                        "geometry_id": geometry_id,
                        "selector": selector_name,
                        "trust_class": trust_name,
                        "actuator": actuator,
                        "lambda_is_a_cross_parameter": actuator == "full",
                        **summary,
                    })
                for full_row, cross_row in zip(full_rows, cross_rows):
                    # The paired mechanism contrast uses the FULL arm's selected
                    # trust for both directions. Cross is not separately selected.
                    common_trust = full_row["trust_factor"]
                    cross_candidate = (geometry_id, None, common_trust, "cross")
                    cross_delta = group_delta(
                        index, full_row["held_group"], "outer",
                        cross_candidate, "auroc",
                    )
                    paired_rows.append({
                        "held_group": full_row["held_group"],
                        "geometry_id": geometry_id,
                        "selector": selector_name,
                        "trust_class": trust_name,
                        "full_selected_lambda": full_row["lambda"],
                        "common_trust_factor": common_trust,
                        "full_delta_auroc_pp": 100 * full_row["held_delta_auroc"],
                        "cross_delta_auroc_pp": 100 * cross_delta,
                        "full_minus_cross_pp": 100 * (
                            full_row["held_delta_auroc"] - cross_delta
                        ),
                        "cross_lambda": None,
                        "actuator_was_selected": False,
                    })
    return arm_rows, paired_rows


def vector(rows, field="held_delta_auroc"):
    return np.asarray([row[field] for row in rows], dtype=float)


def bootstrap_mean(values, *, seed_offset=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(seed_offset))
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))].mean(axis=1)
    return {
        "mean_pp": 100 * float(np.mean(values)),
        "ci_pp": [
            100 * float(np.quantile(draws, 0.025)),
            100 * float(np.quantile(draws, 0.975)),
        ],
        "positive_groups": int(np.sum(values > 0)),
        "worst_group_pp": 100 * float(np.min(values)),
    }


def arm_name(capacity, selector, trust_class):
    return f"capacity={capacity}__selector={selector}__trust={trust_class}"


def run_factorial(index, active, groups):
    arms = {}
    rows = []
    selector_functions = {"one_se": choose_one_se, "max_mean": choose_max_mean}
    for capacity, geometry_ids in (
        ("fixed", ("residual_union_k7",)),
        ("searched", tuple(g for g in phase_a_geometry_ids() if g in active)),
    ):
        for selector_name, selector in selector_functions.items():
            for trust_name, trusts in TRUST_CLASSES.items():
                name = arm_name(capacity, selector_name, trust_name)
                nested = nested_arm(index, geometry_ids, trusts, selector, groups)
                arms[name] = nested
                summary = bootstrap_mean(vector(nested), seed_offset=len(arms))
                rows.append({
                    "arm": name,
                    "capacity": capacity,
                    "selector": selector_name,
                    "trust_class": trust_name,
                    **summary,
                    "selected_geometries": json.dumps(
                        [row["geometry_id"] for row in nested]
                    ),
                    "selected_lambdas": json.dumps([row["lambda"] for row in nested]),
                    "selected_trusts": json.dumps([row["trust_factor"] for row in nested]),
                })
    return arms, rows


def contrast_row(name, kind, left_name, right_name, arms, seed):
    paired = vector(arms[left_name]) - vector(arms[right_name])
    return {
        "contrast": name,
        "kind": kind,
        "left_arm": left_name,
        "right_arm": right_name,
        **bootstrap_mean(paired, seed_offset=seed),
        "group_values_pp": json.dumps((100 * paired).tolist()),
    }


def factorial_contrasts(arms):
    rows = []
    seed = 100
    for capacity in ("fixed", "searched"):
        for trust in TRUST_CLASSES:
            left = arm_name(capacity, "max_mean", trust)
            right = arm_name(capacity, "one_se", trust)
            rows.append(contrast_row(
                f"selector_max_minus_one_se__{capacity}__{trust}",
                "selector_main", left, right, arms, seed,
            ))
            seed += 1
    for selector in ("one_se", "max_mean"):
        for trust in TRUST_CLASSES:
            left = arm_name("searched", selector, trust)
            right = arm_name("fixed", selector, trust)
            rows.append(contrast_row(
                f"searched_minus_fixed__{selector}__{trust}",
                "geometry_capacity_main", left, right, arms, seed,
            ))
            seed += 1
    for capacity in ("fixed", "searched"):
        for selector in ("one_se", "max_mean"):
            for trust in ("v1", "expanded"):
                left = arm_name(capacity, selector, trust)
                right = arm_name(capacity, selector, "canonical")
                rows.append(contrast_row(
                    f"trust_{trust}_minus_canonical__{capacity}__{selector}",
                    "trust_main", left, right, arms, seed,
                ))
                seed += 1
    for trust in TRUST_CLASSES:
        search_max = vector(arms[arm_name("searched", "max_mean", trust)])
        fixed_max = vector(arms[arm_name("fixed", "max_mean", trust)])
        search_one = vector(arms[arm_name("searched", "one_se", trust)])
        fixed_one = vector(arms[arm_name("fixed", "one_se", trust)])
        values = (search_max - fixed_max) - (search_one - fixed_one)
        rows.append({
            "contrast": f"geometry_x_selector__{trust}",
            "kind": "interaction",
            "left_arm": "(searched-fixed|max_mean)",
            "right_arm": "(searched-fixed|one_se)",
            **bootstrap_mean(values, seed_offset=seed),
            "group_values_pp": json.dumps((100 * values).tolist()),
        })
        seed += 1
    return rows


def intrinsic_rows(index, intrinsic, groups):
    output = []
    for held in groups:
        context = intrinsic["contexts"][f"outer_held={held}"]
        candidate = (
            context["selected_geometry"],
            context["fixed_lambda"],
            context["fixed_trust"],
        )
        output.append({
            "held_group": held,
            "geometry_id": candidate[0],
            "lambda": candidate[1],
            "trust_factor": candidate[2],
            "held_delta_auroc": group_delta(index, held, "outer", candidate, "auroc"),
            "held_delta_auprc": group_delta(index, held, "outer", candidate, "auprc"),
            "inner_selected_mean": None,
            "selection": context["diagnostics"],
        })
    return output


def geometry_diagnostics(index, active, intrinsic, groups):
    matrix_rows = []
    geometry_rows_by_held = {}
    rank_rows = []
    for held in groups:
        training = tuple(group for group in groups if group != held)
        per_geometry = []
        for geometry_id in active:
            values = candidate_values(
                index, training, outer_held=held,
                geometry_ids=(geometry_id,), trusts=TRUST_CLASSES["canonical"],
            )
            selected, diagnostics = choose_one_se(values, training)
            held_delta = group_delta(index, held, "outer", selected, "auroc")
            row = {
                "held_group": held,
                "geometry_id": geometry_id,
                "donor_selected_lambda": selected[1],
                "donor_selected_trust": selected[2],
                "donor_inner_mean": diagnostics["selected"]["mean"],
                "held_delta_auroc": held_delta,
            }
            matrix_rows.append(row)
            per_geometry.append(row)
        geometry_rows_by_held[held] = per_geometry

        context = intrinsic["contexts"][f"outer_held={held}"]
        summaries = {row["geometry_id"]: row for row in context["all_geometry_summaries"]}
        intrinsic_order = sorted(
            active,
            key=lambda geometry_id: (
                summaries[geometry_id]["valid"],
                summaries[geometry_id]["minimum_perturbation_stability"],
                summaries[geometry_id]["minimum_direction_cosine"],
                -summaries[geometry_id]["moment_dispersion"],
                summaries[geometry_id]["predicted_roughness_decrease"],
                -geometry_priority(geometry_id),
            ),
            reverse=True,
        )
        intrinsic_rank = {geometry_id: len(active) - rank for rank, geometry_id in enumerate(intrinsic_order)}
        donor_values = [row["donor_inner_mean"] for row in per_geometry]
        held_values = [row["held_delta_auroc"] for row in per_geometry]
        intrinsic_values = [intrinsic_rank[row["geometry_id"]] for row in per_geometry]
        donor_rho = spearmanr(donor_values, held_values).statistic
        intrinsic_rho = spearmanr(intrinsic_values, held_values).statistic
        rank_rows.append({
            "held_group": held,
            "donor_rank_spearman": float(donor_rho) if np.isfinite(donor_rho) else None,
            "intrinsic_rank_spearman": float(intrinsic_rho) if np.isfinite(intrinsic_rho) else None,
        })
    return matrix_rows, geometry_rows_by_held, rank_rows


def oracle_rows(index, active, groups, geometry_rows_by_held):
    geometry_oracle = []
    full_tuple_ceiling = []
    for held in groups:
        per_geometry = geometry_rows_by_held[held]
        best_geometry = max(
            per_geometry,
            key=lambda row: (
                row["held_delta_auroc"], -geometry_priority(row["geometry_id"])
            ),
        )
        geometry_oracle.append({
            "held_group": held,
            "geometry_id": best_geometry["geometry_id"],
            "lambda": best_geometry["donor_selected_lambda"],
            "trust_factor": best_geometry["donor_selected_trust"],
            "held_delta_auroc": best_geometry["held_delta_auroc"],
            "oracle_scope": "held labels choose geometry only; calibration donor-selected",
        })
        all_candidates = candidates(active, TRUST_CLASSES["canonical"])
        best_tuple = max(
            all_candidates,
            key=lambda candidate: (
                group_delta(index, held, "outer", candidate, "auroc"),
                -candidate[2], -candidate[1], -geometry_priority(candidate[0]),
            ),
        )
        full_tuple_ceiling.append({
            "held_group": held,
            "geometry_id": best_tuple[0],
            "lambda": best_tuple[1],
            "trust_factor": best_tuple[2],
            "held_delta_auroc": group_delta(index, held, "outer", best_tuple, "auroc"),
            "oracle_scope": "held labels choose geometry and correction strength; optimism ceiling",
        })
    return geometry_oracle, full_tuple_ceiling


def selector_comparison_rows(method_rows: dict[str, list[dict]], oracle):
    oracle_by_group = {row["held_group"]: row for row in oracle}
    rows = []
    for method, values in method_rows.items():
        for row in values:
            reference = oracle_by_group[row["held_group"]]
            rows.append({
                "method": method,
                "held_group": row["held_group"],
                "geometry_id": row["geometry_id"],
                "lambda": row["lambda"],
                "trust_factor": row["trust_factor"],
                "held_delta_auroc_pp": 100 * row["held_delta_auroc"],
                "held_delta_auprc_pp": (
                    100 * row["held_delta_auprc"]
                    if "held_delta_auprc" in row else None
                ),
                "oracle_geometry": reference["geometry_id"],
                "oracle_delta_auroc_pp": 100 * reference["held_delta_auroc"],
                "regret_pp": 100 * (
                    reference["held_delta_auroc"] - row["held_delta_auroc"]
                ),
                "oracle_geometry_agreement": row["geometry_id"] == reference["geometry_id"],
            })
    return rows


def family_nrm_vector(index, groups):
    path = REPO / "results" / "neutral_residual_mode_cs_iu_v1" / "cell_results.csv"
    if sha256_file(path) != "0c4f836ac2ae046c228b943c3afcf26a6abe20f84dc138d4ada9dc6a3d96c278":
        raise RuntimeError("Family-NRM comparator artifact changed")
    per_cell = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["regime"] != "original_lofo" or row["cell"] not in ELIGIBLE_CELLS:
                continue
            cell = row["cell"]
            recorded_iu = float(row["iu_auroc"])
            current_iu = index[(cell, "iu", "auroc")]
            if abs(recorded_iu - current_iu) > 1e-12:
                raise RuntimeError(f"Family-NRM IU comparator drift: {cell}")
            per_cell[cell] = float(row["nrm_auroc"]) - recorded_iu
    if set(per_cell) != set(ELIGIBLE_CELLS):
        raise RuntimeError("Family-NRM comparator roster changed")
    return np.asarray([
        np.mean([per_cell[cell] for cell in ELIGIBLE_CELLS if dataset_family(cell) == group])
        for group in groups
    ], dtype=float)


def historical_control_audit() -> dict:
    root = REPO / "results" / "pooled_graph_roughness_direction_v2" / "controls"
    result_path = root / "RESULT.json"
    report_path = root / "REPORT.md"
    fit_manifest_path = root / "FIT_MANIFEST.json"
    config_path = root / "FROZEN_CONTROL_CONFIG.json"
    if sha256_file(report_path) != "8f4263a73017a0179995acdaaadb7fb3852011af406336f139b285c6d5e5982a":
        raise RuntimeError("canonical control report changed")
    expected_hashes = {
        result_path: "495aa7c53fbd12a14b6b50229c7e61e5d841ba9d94999220fa54ca5a42d1d6b2",
        fit_manifest_path: "00f86c7182f0bf38e511f36f7e223b73b80d423a5d334cedd1d55925338839d1",
        config_path: "5ec2b8c691c57d9f80d3c2437e7aac4c83091f6703799014a348385c6694c2cc",
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"canonical control artifact changed: {path.name}")
    result = json.loads(result_path.read_text())
    required = (
        "contribution_graph", "dufs_graph", "cross_only",
        "equal_cell_pooling", "family_axis_permuted",
    )
    matched = result["primary_hyperparameter_matched"]
    if not all(name in matched for name in required):
        raise RuntimeError("canonical control registry incomplete")
    null = result["node_permutation_null"]
    if null["n_permutations"] < 20 or len(null["replicate_delta_pp"]) < 20:
        raise RuntimeError("canonical node-permutation control count changed")
    return {
        "scope": (
            "inherited and hash-verified for the exact residual_union_k7 canonical arm; "
            "not evidence for a different selected geometry"
        ),
        "report_sha256": sha256_file(report_path),
        "result_sha256": sha256_file(result_path),
        "fit_manifest_sha256": sha256_file(fit_manifest_path),
        "control_config_sha256": sha256_file(config_path),
        "status": result["status"],
        "primary_real_delta_pp": result["primary_real_delta_pp"],
        "matched_controls_pp": {
            name: matched[name]["control_delta_vs_iu_pp"] for name in required
        },
        "node_permutation_null": null,
        "mechanism_gates": result["mechanism_gates"],
        "complete_registered_attribution_passed": result["status"] == "PASS",
    }


def full_source_selection(index, active, groups, selector, geometry_ids=None):
    geometry_ids = tuple(active if geometry_ids is None else geometry_ids)
    values = candidate_values(
        index, groups, outer_held=None,
        geometry_ids=geometry_ids, trusts=TRUST_CLASSES["canonical"],
    )
    return selector(values, groups)


def transfer_registry(
    out, complete, intrinsic, index, active, groups, calibrations
):
    label_free_context = intrinsic["contexts"]["all_source"]
    supervised, supervised_diag = full_source_selection(
        index, active, groups, choose_one_se
    )
    supervised_max, supervised_max_diag = full_source_selection(
        index, active, groups, choose_max_mean
    )
    canonical = ("residual_union_k7", 0.03, 0.5)
    canonical_selection_path = (
        REPO / "results" / "pooled_graph_roughness_direction_v2"
        / "FROZEN_SELECTION.json"
    )
    if sha256_file(canonical_selection_path) != (
        "ff0b6e824d0140b7e5fbdab0d10f97b7a32ff80217d6b740915436c5ce8d1aa3"
    ):
        raise RuntimeError("canonical transfer selection artifact changed")
    canonical_frozen = json.loads(canonical_selection_path.read_text())
    frozen_config = canonical_frozen["selected_config"]
    if (
        frozen_config["lambda"] != canonical[1]
        or frozen_config["trust_factor"] != canonical[2]
        or frozen_config["k"] != 7
        or frozen_config["topology"] != "union"
    ):
        raise RuntimeError("canonical transfer configuration drift")
    entries = {}
    for name, candidate, selector_type, diagnostics in (
        ("canonical", canonical, "frozen_canonical", None),
        (
            "label_free",
            (
                label_free_context["selected_geometry"],
                label_free_context["fixed_lambda"],
                label_free_context["fixed_trust"],
            ),
            "intrinsic_label_free_geometry",
            label_free_context["diagnostics"],
        ),
        ("supervised_one_se", supervised, "supervised_donor_label", supervised_diag),
        ("supervised_max_mean", supervised_max, "supervised_donor_label_sensitivity", supervised_max_diag),
    ):
        cal = calibrations[calibration_key(candidate[0], (), candidate[1], "full")]
        direction = np.asarray(cal["direction"], dtype=float)
        if (
            direction.shape != (len(VIEW_ORDER),)
            or tuple(cal["direction_families"]) != tuple(VIEW_ORDER)
        ):
            raise RuntimeError(f"transfer direction registry mismatch: {name}")
        entries[name] = {
            "selector_type": selector_type,
            "actuator": "full",
            "geometry_id": candidate[0],
            "lambda": candidate[1],
            "trust_factor": candidate[2],
            "direction_families": list(VIEW_ORDER),
            "direction": direction.tolist(),
            "calibration_key": calibration_key(candidate[0], (), candidate[1], "full"),
            "selection_diagnostics": diagnostics,
        }
        if name == "canonical" and not np.allclose(
            direction,
            np.asarray(canonical_frozen["direction"], dtype=float),
            rtol=0.0,
            atol=1e-15,
        ):
            raise RuntimeError("new canonical transfer direction does not match frozen V2")
    for name in tuple(entries):
        base = entries[name]
        cross_cal = calibrations[calibration_key(
            base["geometry_id"], (), None, "cross"
        )]
        cross_direction = np.asarray(cross_cal["direction"], dtype=float)
        if (
            cross_direction.shape != (len(VIEW_ORDER),)
            or tuple(cross_cal["direction_families"]) != tuple(VIEW_ORDER)
        ):
            raise RuntimeError(f"cross transfer direction registry mismatch: {name}")
        entries[f"{name}_cross"] = {
            "selector_type": base["selector_type"] + "_matched_cross_actuator",
            "actuator": "cross",
            "geometry_id": base["geometry_id"],
            "lambda": None,
            "trust_factor": base["trust_factor"],
            "direction_families": list(VIEW_ORDER),
            "direction": cross_direction.tolist(),
            "calibration_key": calibration_key(
                base["geometry_id"], (), None, "cross"
            ),
            "selection_diagnostics": {
                "actuator_was_selected": False,
                "trust_matched_to_full_entry": name,
                "lambda_absent_because_sd_normalization_identifies_direction_only": True,
            },
        }
    payload = {
        "version": VERSION,
        "fit_manifest_hash": complete["manifest_hash"],
        "fit_label_free_selection_sha256": sha256_file(
            out / "FROZEN_LABELFREE_SELECTION.json"
        ),
        "canonical_frozen_selection_sha256": sha256_file(
            canonical_selection_path
        ),
        "canonical_frozen_selection_hash": canonical_frozen["selection_hash"],
        "development_outcomes_opened_for_supervised_entries": True,
        "held_family_oracle_exported": False,
        "retrospective_transfer_only": True,
        "entries": entries,
    }
    payload["selection_hash"] = canonical_hash(payload)
    return payload


def exact_sign_flip_pvalue(values):
    values = np.asarray(values, dtype=float)
    observed = float(np.mean(values))
    draws = np.asarray([
        np.mean(values * np.asarray(signs))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ])
    return float(np.mean(draws >= observed - 1e-15))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    complete, definition, score_hashes = verify_fit(args.out, args.bundle)
    verification = verify_every_candidate_score_without_labels(
        args.out, complete, score_hashes
    )
    # This is the first outcome-array access in this reporting process.
    labels = load_labels_after_verification(definition)
    index, metric_rows = build_metric_index(args.out, complete, labels)
    write_csv(args.out / "candidate_cell_metrics.csv", metric_rows)

    active = tuple(complete["active_geometry_ids"])
    selector_active = tuple(complete["selector_geometry_ids"])
    groups = tuple(sorted({dataset_family(cell) for cell in ELIGIBLE_CELLS}))
    intrinsic = json.loads((args.out / "FROZEN_LABELFREE_SELECTION.json").read_text())
    calibrations = json.loads((args.out / "CALIBRATIONS.json").read_text())
    diversity = json.loads((args.out / "GRAPH_DIVERSITY.json").read_text())
    node_control_rows, node_control_summaries = (
        summarize_node_permutation_controls(index, active, groups)
    )
    write_csv(
        args.out / "node_permutation_outcome_controls.csv",
        node_control_rows,
    )
    write_csv(
        args.out / "node_permutation_outcome_summaries.csv",
        node_control_summaries,
    )

    arms, phase_a_rows = run_factorial(index, active, groups)
    contrasts = factorial_contrasts(arms)
    write_csv(args.out / "phase_a_factorial.csv", phase_a_rows)
    write_csv(args.out / "phase_a_contrasts.csv", contrasts)
    actuator_arm_rows, actuator_paired_rows = run_actuator_factor(
        index, active, groups
    )
    write_csv(args.out / "actuator_arms.csv", actuator_arm_rows)
    write_csv(args.out / "actuator_paired_outer.csv", actuator_paired_rows)
    actuator_fit_diagnostics = json.loads(
        (args.out / "ACTUATOR_DIAGNOSTICS.json").read_text()
    )
    actuator_diagnostic_rows = []
    for context_name, values in actuator_fit_diagnostics["contexts"].items():
        for geometry_id, diagnostics in values.items():
            actuator_diagnostic_rows.append({
                "context": context_name,
                "geometry_id": geometry_id,
                **{
                    key: (json.dumps(value) if isinstance(value, (dict, list)) else value)
                    for key, value in diagnostics.items()
                    if key != "cbar"
                },
                "cbar": json.dumps(diagnostics["cbar"]),
            })
    write_csv(args.out / "actuator_diagnostics.csv", actuator_diagnostic_rows)

    canonical_rows = arms[arm_name("fixed", "one_se", "canonical")]
    fixed_max_rows = arms[arm_name("fixed", "max_mean", "canonical")]
    fixed_max_v1_rows = arms[arm_name("fixed", "max_mean", "v1")]
    matched_v1_rows = arms[arm_name("searched", "max_mean", "v1")]
    legacy_v1_rows = nested_arm(
        index,
        tuple(g for g in phase_a_geometry_ids() if g in active),
        TRUST_CLASSES["v1"],
        choose_max_mean,
        groups,
        lambdas=LEGACY_V1_LAMBDAS,
    )
    canonical_pp = 100 * float(np.mean(vector(canonical_rows)))
    fixed_max_pp = 100 * float(np.mean(vector(fixed_max_rows)))
    matched_v1_pp = 100 * float(np.mean(vector(matched_v1_rows)))
    legacy_v1_pp = 100 * float(np.mean(vector(legacy_v1_rows)))
    if abs(canonical_pp - ANCHOR_ONE_SE_PP) > 1e-10:
        raise RuntimeError(f"canonical +0.251pp anchor failed: {canonical_pp}")
    if abs(fixed_max_pp - ANCHOR_MAX_MEAN_PP) > 1e-10:
        raise RuntimeError(f"fixed max-mean +0.450pp anchor failed: {fixed_max_pp}")
    if abs(legacy_v1_pp - LEGACY_V1_PP) > 1e-10:
        raise RuntimeError(f"legacy V1 +0.452pp anchor failed: {legacy_v1_pp}")

    label_free_rows = intrinsic_rows(index, intrinsic, groups)
    supervised_rows = nested_arm(
        index, selector_active, TRUST_CLASSES["canonical"],
        choose_one_se, groups,
    )
    supervised_max_rows = nested_arm(
        index, selector_active, TRUST_CLASSES["canonical"],
        choose_max_mean, groups,
    )
    matrix_rows, per_held, rank_rows = geometry_diagnostics(
        index, selector_active, intrinsic, groups
    )
    geometry_oracle, full_tuple_ceiling = oracle_rows(
        index, selector_active, groups, per_held
    )
    write_csv(args.out / "geometry_family_matrix.csv", matrix_rows)
    write_csv(args.out / "selector_rank_correlations.csv", rank_rows)

    methods = {
        "canonical_fixed_one_se": canonical_rows,
        "fixed_max_mean": fixed_max_rows,
        "supervised_geometry_one_se": supervised_rows,
        "supervised_geometry_max_mean": supervised_max_rows,
        "intrinsic_label_free": label_free_rows,
        "held_family_geometry_oracle": geometry_oracle,
        "held_family_full_tuple_ceiling": full_tuple_ceiling,
    }
    selector_rows = selector_comparison_rows(methods, geometry_oracle)
    write_csv(args.out / "selector_results.csv", selector_rows)

    intrinsic_diag_rows = []
    for context_name, context in intrinsic["contexts"].items():
        for summary in context["all_geometry_summaries"]:
            intrinsic_diag_rows.append({
                "context": context_name,
                "selected": summary["geometry_id"] == context["selected_geometry"],
                **{key: value for key, value in summary.items() if key != "direction"},
            })
    write_csv(args.out / "intrinsic_diagnostics.csv", intrinsic_diag_rows)

    control_audit = historical_control_audit()
    write_json(args.out / "CONTROL_AUDIT.json", control_audit)
    transfer = transfer_registry(
        args.out, complete, intrinsic, index, selector_active, groups, calibrations
    )
    write_once(args.out / "FROZEN_TRANSFER_SELECTIONS.json", transfer)

    nrm = family_nrm_vector(index, groups)
    selector_summaries = {
        name: {
            **bootstrap_mean(vector(rows), seed_offset=400 + index_),
            "group_values_pp": {
                group: 100 * float(value)
                for group, value in zip(groups, vector(rows))
            },
            "mean_regret_to_geometry_oracle_pp": 100 * float(np.mean(
                vector(geometry_oracle) - vector(rows)
            )),
            "oracle_geometry_agreement": int(sum(
                row["geometry_id"] == oracle["geometry_id"]
                for row, oracle in zip(rows, geometry_oracle)
            )),
        }
        for index_, (name, rows) in enumerate(methods.items())
    }
    canonical = vector(canonical_rows)
    fixed_max = vector(fixed_max_rows)
    supervised = vector(supervised_rows)
    supervised_max = vector(supervised_max_rows)
    label_free = vector(label_free_rows)
    geometry_oracle_vector = vector(geometry_oracle)
    full_ceiling_vector = vector(full_tuple_ceiling)
    selector_effect = fixed_max - canonical
    trust_grid_effect = vector(fixed_max_v1_rows) - fixed_max
    geometry_effect_v1 = vector(matched_v1_rows) - vector(fixed_max_v1_rows)
    lambda_grid_effect = vector(legacy_v1_rows) - vector(matched_v1_rows)
    geometry_effect_one = supervised - canonical
    geometry_effect_max = supervised_max - fixed_max
    label_free_effect = label_free - canonical
    selection_optimism = np.asarray([
        row["inner_selected_mean"] - row["held_delta_auroc"]
        for row in supervised_max_rows
    ])
    actuator_paired_summaries = []
    for geometry_id in active:
        for selector_name in ("one_se", "max_mean"):
            for trust_name in TRUST_CLASSES:
                selected_rows = [
                    row for row in actuator_paired_rows
                    if row["geometry_id"] == geometry_id
                    and row["selector"] == selector_name
                    and row["trust_class"] == trust_name
                ]
                values = np.asarray([
                    row["full_minus_cross_pp"] / 100.0 for row in selected_rows
                ])
                actuator_paired_summaries.append({
                    "geometry_id": geometry_id,
                    "selector": selector_name,
                    "trust_class": trust_name,
                    **bootstrap_mean(
                        values,
                        seed_offset=(
                            1200 + 100 * geometry_priority(geometry_id)
                            + 10 * (selector_name == "max_mean")
                            + list(TRUST_CLASSES).index(trust_name)
                        ),
                    ),
                    "full_minus_cross_sign_flip_p": exact_sign_flip_pvalue(values),
                })
    write_csv(
        args.out / "actuator_paired_summaries.csv",
        actuator_paired_summaries,
    )

    if (
        100 * np.mean(label_free_effect) > 0.05
        and transfer["entries"]["label_free"]["geometry_id"] != "residual_union_k7"
    ):
        provisional_decision = "INCONCLUSIVE_GEOMETRY_IDENTIFICATION"
        decision_reason = "label-free development gain requires the frozen transfer stress test"
    elif 100 * np.mean(geometry_effect_one) > 0.05:
        provisional_decision = "SUPERVISED_GEOMETRY_HEADROOM_ONLY"
        decision_reason = "supervised donor selection improved matched outer folds but intrinsic selection did not"
    elif 100 * np.mean(geometry_effect_max) < 0 and 100 * np.mean(selection_optimism) > 0:
        provisional_decision = "GEOMETRY_SEARCH_SELECTION_OPTIMISM"
        decision_reason = "the expanded class improved inner selected means but lost on outer families"
    else:
        provisional_decision = "SELECTOR_EFFECT_WITHOUT_GEOMETRY_GAIN"
        decision_reason = "max-mean changed correction strength; matched geometry capacity added no material gain"

    provenance = {
        "new_fit_physical_isolation": verification,
        "fit_input_target_fields_physically_present": False,
        "report_first_opened_outcomes_after_all_score_hashes_verified": True,
        "canonical_historical_boundary_correction": (
            "The canonical fit did not index correctness arrays and its emitted score/state "
            "banks are label-free and hash-consistent. Its input NPZ nevertheless physically "
            "contained 24 __labels members, so target_fields_received_by_fit=[] and an "
            "unqualified 'labels unopened' were too strong. Canonical separation was logical "
            "field whitelisting; this study repairs it with physical isolation."
        ),
        "representation_claim_boundary": (
            "Outer LOFO is strict for the new graph/selector stage conditional on the frozen "
            "mixed-v2 and confidence-orientation contract. That representation was previously "
            "selected using these opened families, so this is not end-to-end unseen-family validation."
        ),
        "retrospective": True,
        "su_arms_present": False,
    }
    write_json(args.out / "PROVENANCE_AUDIT.json", provenance)

    result = {
        "version": VERSION,
        "status": "DEVELOPMENT_COMPLETE_TRANSFER_PENDING",
        "provisional_decision": provisional_decision,
        "decision_reason": decision_reason,
        "claim_boundary": provenance["representation_claim_boundary"],
        "anchors": {
            "canonical_one_se_pp": canonical_pp,
            "canonical_expected_pp": ANCHOR_ONE_SE_PP,
            "fixed_max_mean_pp": fixed_max_pp,
            "fixed_max_mean_expected_pp": ANCHOR_MAX_MEAN_PP,
            "matched_searched_max_v1_trust_common_lambda_pp": matched_v1_pp,
            "legacy_v1_separate_reproduction_pp": legacy_v1_pp,
            "matched_minus_legacy_v1_pp": matched_v1_pp - legacy_v1_pp,
            "known_selector_effect_pp": fixed_max_pp - canonical_pp,
            "matched_trust_grid_effect_pp": 100 * float(np.mean(trust_grid_effect)),
            "matched_geometry_effect_under_v1_pp": 100 * float(np.mean(geometry_effect_v1)),
            "legacy_lambda_grid_effect_pp": 100 * float(np.mean(lambda_grid_effect)),
            "decomposition_sum_check_pp": 100 * float(np.mean(
                selector_effect + trust_grid_effect
                + geometry_effect_v1 + lambda_grid_effect
            )),
        },
        "phase_a": {
            "arm_count": len(arms),
            "factorial_rows": phase_a_rows,
            "contrasts": contrasts,
            "primary_effects": {
                "fixed_max_mean_minus_one_se": bootstrap_mean(selector_effect, seed_offset=700),
                "searched_minus_fixed_one_se": bootstrap_mean(geometry_effect_one, seed_offset=701),
                "searched_minus_fixed_max_mean": bootstrap_mean(geometry_effect_max, seed_offset=702),
            },
        },
        "phase_b": {
            "candidate_geometry_count": diversity["candidate_geometry_count"],
            "effective_geometry_count": len(selector_active),
            "active_geometry_ids": list(selector_active),
            "control_only_geometry_ids": sorted(set(active) - set(selector_active)),
            "duplicate_of": diversity["duplicate_of"],
            "hybrid_omission": (
                "omitted because six family coordinates and 19-30 DUFS coordinates cannot be "
                "capacity matched without adding another tuned projection"
            ),
            "weighting_omission": (
                "all candidates retain the reviewed duplicate-safe self-tuning heat kernel; "
                "a second weighting was omitted to avoid an unrestricted product after prior null sweeps"
            ),
            "selector_summaries": selector_summaries,
            "effects": {
                "intrinsic_label_free_minus_canonical": bootstrap_mean(label_free_effect, seed_offset=710),
                "supervised_one_se_minus_fixed_one_se": bootstrap_mean(geometry_effect_one, seed_offset=711),
                "supervised_max_minus_fixed_max": bootstrap_mean(geometry_effect_max, seed_offset=712),
                "geometry_oracle_minus_canonical": bootstrap_mean(geometry_oracle_vector - canonical, seed_offset=713),
                "full_tuple_ceiling_minus_canonical": bootstrap_mean(full_ceiling_vector - canonical, seed_offset=714),
                "supervised_max_inner_minus_outer_optimism": bootstrap_mean(selection_optimism, seed_offset=715),
            },
            "rank_correlations": rank_rows,
        },
        "actuator_factor": {
            "selector_can_choose_actuator": False,
            "cross_lambda_parameter": None,
            "cross_identification_note": (
                "Because every R*d correction is normalized to the same requested SD, "
                "cross-only identifies only direction -cbar; lambda is absent."
            ),
            "paired_full_minus_cross_summaries": actuator_paired_summaries,
            "target_free_diagnostics_sha256": sha256_file(
                args.out / "ACTUATOR_DIAGNOSTICS.json"
            ),
            "node_permutations_per_geometry": actuator_fit_diagnostics[
                "node_permutations"
            ],
        },
        "family_nrm": {
            "delta_pp": 100 * float(np.mean(nrm)),
            "group_values_pp": {group: 100 * float(value) for group, value in zip(groups, nrm)},
        },
        "controls": control_audit,
        "new_node_permutation_controls": {
            "protocol": (
                "20 deterministic node permutations per geometry and actuator; "
                "outer-only, selector-free, fixed lambda=.03 for full and trust=.5"
            ),
            "summaries": node_control_summaries,
        },
        "controls_scope_gate": (
            transfer["entries"]["label_free"]["geometry_id"] == "residual_union_k7"
        ),
        "inference": {
            "exact_one_sided_sign_flip_p_label_free_minus_canonical": exact_sign_flip_pvalue(label_free_effect),
            "exact_one_sided_sign_flip_p_supervised_minus_fixed": exact_sign_flip_pvalue(geometry_effect_one),
        },
        "fit_manifest_hash": complete["manifest_hash"],
        "label_free_selection_hash": intrinsic["selection_hash"],
        "transfer_selection_hash": transfer["selection_hash"],
        "candidate_scores_verified_before_labels": verification["candidate_scores_verified_before_label_open"],
        "plots_pending": True,
        "transfer_pending": True,
    }
    write_json(args.out / "RESULT.json", result)

    lines = [
        "# Graph Geometry Selection Research V1 — development", "",
        f"**Provisional decision: `{provisional_decision}`.** {decision_reason}.", "",
        "## Anchor decomposition", "",
        f"- Fixed union-k7 + one-SE/canonical: **{canonical_pp:+.6f}pp** (exact anchor).",
        f"- Fixed union-k7 + max-mean/canonical: **{fixed_max_pp:+.6f}pp** (exact anchor).",
        f"- Selector effect: **{fixed_max_pp-canonical_pp:+.6f}pp**.",
        f"- Matched searched/max-mean/V1-trust with the common eight-lambda grid: **{matched_v1_pp:+.6f}pp**.",
        f"- Separate exact legacy V1 reproduction (five-lambda grid): **{legacy_v1_pp:+.6f}pp**.",
        f"- Matched trust-grid effect (fixed/max-mean, V1 minus canonical): **{100*np.mean(trust_grid_effect):+.6f}pp**.",
        f"- Matched geometry-capacity effect (max-mean/V1 trust): **{100*np.mean(geometry_effect_v1):+.6f}pp**.",
        f"- Legacy five-lambda minus common eight-lambda effect: **{100*np.mean(lambda_grid_effect):+.6f}pp**.", "",
        "## Selector comparison", "",
        "| method | mean ΔAUROC (pp) | 95% family bootstrap | oracle regret (pp) | oracle geometry agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "canonical_fixed_one_se", "fixed_max_mean", "supervised_geometry_one_se",
        "supervised_geometry_max_mean", "intrinsic_label_free",
        "held_family_geometry_oracle", "held_family_full_tuple_ceiling",
    ):
        summary = selector_summaries[name]
        lines.append(
            f"| `{name}` | {summary['mean_pp']:+.3f} | "
            f"[{summary['ci_pp'][0]:+.3f}, {summary['ci_pp'][1]:+.3f}] | "
            f"{summary['mean_regret_to_geometry_oracle_pp']:+.3f} | "
            f"{summary['oracle_geometry_agreement']}/8 |"
        )
    lines += [
        "", "## Actuator decomposition", "",
        "`full` and `cross` were frozen and evaluated as separate arms; no selector could choose between them. Each paired full−cross row uses the full arm's selected trust for both directions.", "",
        "Because every correction `R d` is normalized to a fixed requested SD, `cross = -cbar` has no lambda parameter and identifies direction only. Target-free diagnostics include `cosine(d_full,-cbar)`, leave-source stability and dispersion of `cbar`, plus 20 deterministic node permutations per geometry. If full approximately equals cross throughout the bank, the mechanism is a pooled graph cross-gradient rather than a quadratic graph solve.", "",
        "", "## Boundaries", "",
        "The new fit consumed a physically target-free archive and every candidate score hash was verified before this report opened outcomes. The canonical historical fit was logically label-whitelisted but not physically isolated; its score bank remains hash-consistent.", "",
        "The inherited 20-node-permutation, contribution, DUFS, cross-only, equal-cell, and family-axis controls apply exactly to the fixed residual union-k7 arm. They do not validate a newly selected geometry. No SU covariance cleaning or SU-rho arm appears here.", "",
        "These eight-family comparisons are retrospective and conditional on an already outcome-informed frozen feature contract. External datasets are also opened stress tests, not confirmation.", "",
    ]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "provisional_decision": provisional_decision,
        "canonical_anchor_pp": canonical_pp,
        "fixed_max_mean_anchor_pp": fixed_max_pp,
        "matched_v1_pp": matched_v1_pp,
        "label_free_pp": selector_summaries["intrinsic_label_free"]["mean_pp"],
        "supervised_one_se_pp": selector_summaries["supervised_geometry_one_se"]["mean_pp"],
        "geometry_oracle_pp": selector_summaries["held_family_geometry_oracle"]["mean_pp"],
        "candidate_scores_verified_before_labels": verification["candidate_scores_verified_before_label_open"],
        "transfer_selection_hash": transfer["selection_hash"],
    }, indent=2))


if __name__ == "__main__":
    main()
