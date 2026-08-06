#!/usr/bin/env python3
"""Run the frozen U2-prior reconciliation checkpoint.

This script regenerates only the consumed selective-target development pair and
reads only saved CSV/JSON real-study artifacts.  It never opens the raw real
feature bundle and never touches the reserved confirmation seed block.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import types
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.feature_contract import consensus_anchor                    # noqa: E402
from spectral_utils.semi_supervised_fusion import (                            # noqa: E402
    orient_weight,
    spectral_score_basis,
)
from spectral_utils.target_anchored_laplacian import (                         # noqa: E402
    fixed_logistic_scores,
    target_anchored_laplacian_fit,
)
from spectral_utils.u2_prior_reconciliation import (                           # noqa: E402
    CURRENT_UPCR_KWARGS,
    FIT_TOLERANCES,
    GEOMETRY_TOLERANCES,
    HISTORICAL_UPCR_KWARGS,
    basis_alignment,
    covariance_normalized_u2_basis,
    fit_equivalence_diagnostics,
    fit_prior_head,
    optimistic_endpoint_controls,
)
from spectral_utils.upcr import upcr_fit                                        # noqa: E402
from scripts.target_anchored_laplacian_synthetic import (                       # noqa: E402
    array_hash,
    calibration_permutations,
    dataset_seed,
    make_selective_pair,
)


VERSION = "u2-prior-reconciliation-v1-2026-08-06"
SPEC = "SPEC_U2_PRIOR_RECONCILIATION_V1.md"
OUT_DEFAULT = os.path.join(REPO, "results", "u2_prior_reconciliation")
REAL_REPLICATES = os.path.join(
    REPO, "results", "semi_supervised_spectral_v1", "replicates.csv"
)
MIX_ROWS = os.path.join(
    REPO, "results", "upcr_study", "11_posthoc_controls", "mix_splithalf.csv"
)
MIX_SUMMARY = os.path.join(
    REPO, "results", "upcr_study", "11_posthoc_controls", "summary.json"
)
DEVELOPMENT_SEED_OFFSET = 40_000
CONFIRMATION_SEED_OFFSET = 2_600_000
REPLICATES = 8
N = 360
CALIBRATION_DRAWS = 16
BUDGETS = (4, 8, 16, 32, 64)
PRIOR = np.array([1.0, 0.0])
CANDIDATE_REAL_METHODS = ("gold_pcr2", "anchored_pcr2", "anchored_pcr6")
EXPECTED_REAL_METHODS = (
    "upcr",
    "platt_upcr",
    "gold_pcr2",
    "gold_pcr6",
    "gold_ridge_all",
    "anchored_pcr2",
    "anchored_pcr6",
    "pseudo_gold_pcr6",
)
CORE_SYNTHETIC_METHODS = (
    "iu",
    "ta_liu",
    "u2_logistic",
    "anchored_pcr2_historical",
    "anchored_pcr2_current",
    "anchored_pcr2_current_reparameterized",
    "optimistic_endpoint_switch",
    "fixed_average",
    "optimistic_interpolation",
)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_metrics(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def validate_cartesian_rows(rows, expected_cells, expected_repetitions, expected_methods):
    """Require one row for every cell × repetition × method combination."""
    cells = set(expected_cells)
    repetitions = set(expected_repetitions)
    methods = set(expected_methods)
    observed_cells = {row["unit"] for row in rows}
    observed_repetitions = {int(row["repetition"]) for row in rows}
    observed_methods = {row["method"] for row in rows}
    if observed_cells != cells:
        raise RuntimeError("saved rows do not contain the exact expected cell set")
    if observed_repetitions != repetitions:
        raise RuntimeError("saved rows do not contain the exact expected repetition set")
    if observed_methods != methods:
        raise RuntimeError("saved rows do not contain the exact expected method set")

    counts = Counter(
        (row["unit"], int(row["repetition"]), row["method"]) for row in rows
    )
    expected_keys = {
        (cell, repetition, method)
        for cell in cells
        for repetition in repetitions
        for method in methods
    }
    if set(counts) != expected_keys or any(counts[key] != 1 for key in expected_keys):
        raise RuntimeError(
            "saved rows are not an exact cell × repetition × method Cartesian product"
        )
    for cell in cells:
        metadata = {
            (row["group"], row["domain"]) for row in rows if row["unit"] == cell
        }
        if len(metadata) != 1:
            raise RuntimeError(f"saved cell metadata is unstable for {cell}")


def real_cell_method_means(rows, *, repetitions_per_cell):
    """Average repetitions inside each cell before any cross-cell comparison."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["unit"], row["method"])].append(float(row["auc"]))
    wrong = {
        key: len(values) for key, values in grouped.items()
        if len(values) != repetitions_per_cell
    }
    if wrong:
        raise RuntimeError(f"wrong repetition count within cell/method: {wrong}")
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def pack_and_validate_row_scores(
    row_arrays,
    *,
    expected_rows,
    expected_groups,
    samples_per_group,
):
    """Concatenate row-score chunks and validate sample/calibration alignment."""
    packed = {key: np.concatenate(chunks) for key, chunks in row_arrays.items()}
    required = {
        "replicate", "task", "draw", "budget", "sample_index",
        "is_calibration", "label",
    }
    required.update(f"score__{method}" for method in CORE_SYNTHETIC_METHODS)
    missing = required - set(packed)
    if missing:
        raise RuntimeError("packed row scores are missing: " + ", ".join(sorted(missing)))
    lengths = {key: len(value) for key, value in packed.items()}
    if set(lengths.values()) != {expected_rows}:
        raise RuntimeError(f"packed row-score arrays are misaligned: {lengths}")

    groups = defaultdict(list)
    for index, key in enumerate(zip(
        packed["replicate"], packed["task"], packed["draw"], packed["budget"]
    )):
        groups[tuple(int(value) for value in key)].append(index)
    if len(groups) != expected_groups:
        raise RuntimeError(
            f"expected {expected_groups} packed score groups, found {len(groups)}"
        )
    expected_indices = np.arange(samples_per_group)
    for key, indices in groups.items():
        indices = np.asarray(indices, dtype=int)
        if len(indices) != samples_per_group:
            raise RuntimeError(f"packed score group {key} has the wrong sample count")
        order = np.argsort(packed["sample_index"][indices])
        ordered = indices[order]
        if not np.array_equal(packed["sample_index"][ordered], expected_indices):
            raise RuntimeError(f"packed score group {key} does not cover every sample")
        budget = key[3]
        if int(np.sum(packed["is_calibration"][ordered])) != budget:
            raise RuntimeError(f"packed score group {key} has a wrong calibration mask")
        for field in required:
            if field.startswith("score__") and not np.isfinite(packed[field][ordered]).all():
                raise RuntimeError(f"packed score field {field} contains non-finite values")
    return packed


def oriented_upcr(F, kwargs):
    matrix = F.T
    result = upcr_fit(F, **kwargs)
    weight = orient_weight(result.w, matrix, consensus_anchor(matrix))
    return result, weight, matrix @ weight


def basis_row(replicate, seed, basis_name, alignment):
    row = {
        "version": VERSION,
        "replicate": replicate,
        "seed": seed,
        "basis": basis_name,
    }
    for key, value in alignment.as_dict().items():
        row[key] = json.dumps(value) if isinstance(value, list) else value
    return row


def metric_row(
    replicate,
    seed,
    task,
    draw,
    budget,
    calibration_hash,
    method,
    labels,
    scores,
    evaluation,
    iu_auc,
    iu_ap,
    *,
    fallback=False,
    fallback_reason="",
    selected="",
    alpha_u2="",
):
    auc, ap = safe_metrics(labels[evaluation], np.asarray(scores)[evaluation])
    return {
        "version": VERSION,
        "split": "consumed_development",
        "replicate": replicate,
        "seed": seed,
        "task": task,
        "draw": draw,
        "budget": budget,
        "calibration_index_hash": calibration_hash,
        "n_evaluation": len(evaluation),
        "method": method,
        "auroc": auc,
        "auprc": ap,
        "auroc_delta_vs_iu": auc - iu_auc,
        "auprc_delta_vs_iu": ap - iu_ap,
        "fallback": bool(fallback),
        "fallback_reason": fallback_reason,
        "selected": selected,
        "alpha_u2": alpha_u2,
    }


def run_synthetic():
    metrics = []
    geometry = []
    fit_equality = []
    invariants = []
    row_arrays = defaultdict(list)

    for replicate in range(REPLICATES):
        seed = dataset_seed(DEVELOPMENT_SEED_OFFSET, replicate, 2)
        if seed >= CONFIRMATION_SEED_OFFSET:
            raise RuntimeError("development seed entered reserved confirmation block")
        F, views = make_selective_pair(seed, N)
        if float(np.max(np.abs(F.mean(axis=1)))) > 1e-10:
            raise RuntimeError("frozen synthetic feature contract is not row-centred")
        reference_basis, u2_coordinates, _ = covariance_normalized_u2_basis(F)
        current_result, current_weight, iu_scores = oriented_upcr(
            F, CURRENT_UPCR_KWARGS
        )
        historical_result, historical_weight, _ = oriented_upcr(
            F, HISTORICAL_UPCR_KWARGS
        )
        matrix = F.T
        current_basis = spectral_score_basis(matrix, current_weight, rank=2)
        historical_basis = spectral_score_basis(matrix, historical_weight, rank=2)
        current_alignment = basis_alignment(
            F, reference_basis, current_basis, PRIOR
        )
        historical_alignment = basis_alignment(
            F, reference_basis, historical_basis, PRIOR
        )
        geometry.append(basis_row(replicate, seed, "current", current_alignment))
        geometry.append(
            basis_row(replicate, seed, "historical", historical_alignment)
        )
        if not current_alignment.geometrically_equivalent:
            raise RuntimeError(
                f"replicate {replicate}: current anchored basis is not U2-equivalent"
            )

        permutations = calibration_permutations(seed, N, CALIBRATION_DRAWS)
        calibration_hashes = [
            array_hash(permutation[:budget])
            for permutation in permutations
            for budget in BUDGETS
        ]
        invariants.append({
            "version": VERSION,
            "replicate": replicate,
            "seed": seed,
            "F_hash_signal": array_hash(F),
            "F_hash_nuisance": array_hash(F.copy()),
            "equal_F": bool(np.array_equal(F, F.copy())),
            "calibration_hash_bundle_signal": array_hash(
                np.asarray(calibration_hashes, dtype="S64")
            ),
            "calibration_hash_bundle_nuisance": array_hash(
                np.asarray(calibration_hashes, dtype="S64")
            ),
            "equal_calibration_indices": True,
            "current_geometry_equivalent": (
                current_alignment.geometrically_equivalent
            ),
            "historical_geometry_equivalent": (
                historical_alignment.geometrically_equivalent
            ),
            "current_upcr_used_simple_average": bool(
                current_result.used_simple_average
            ),
            "historical_upcr_used_simple_average": bool(
                historical_result.used_simple_average
            ),
            "historical_upcr_n_kept": int(historical_result.keep.sum()),
        })

        for task_index, task in enumerate(
            ("selective_target_signal", "selective_target_nuisance")
        ):
            labels, _, _ = views[task]
            labels = np.asarray(labels, dtype=int)
            for draw, permutation in enumerate(permutations):
                for budget in BUDGETS:
                    calibration = np.asarray(permutation[:budget], dtype=int)
                    evaluation_mask = np.ones(N, dtype=bool)
                    evaluation_mask[calibration] = False
                    evaluation = np.flatnonzero(evaluation_mask)
                    calibration_hash = array_hash(calibration)

                    ta = target_anchored_laplacian_fit(
                        F, labels, calibration, lambda_=0.1, k=7
                    )
                    ta_weight = orient_weight(
                        ta.fit.w, matrix, consensus_anchor(matrix)
                    )
                    ta_scores = matrix @ ta_weight
                    u2_scores, u2_diagnostic = fixed_logistic_scores(
                        u2_coordinates, labels, calibration
                    )
                    historical_head, historical_scores = fit_prior_head(
                        matrix,
                        labels,
                        calibration,
                        historical_basis,
                        PRIOR,
                    )
                    current_head, current_scores = fit_prior_head(
                        matrix, labels, calibration, current_basis, PRIOR
                    )
                    reference_head, reference_scores = fit_prior_head(
                        matrix,
                        labels,
                        calibration,
                        reference_basis,
                        current_alignment.source_prior_in_reference,
                    )
                    equality = fit_equivalence_diagnostics(
                        current_head,
                        current_scores,
                        reference_head,
                        reference_scores,
                    )
                    fit_equality.append({
                        "version": VERSION,
                        "replicate": replicate,
                        "seed": seed,
                        "task": task,
                        "draw": draw,
                        "budget": budget,
                        "calibration_index_hash": calibration_hash,
                        "one_class": bool(len(np.unique(labels[calibration])) == 1),
                        **equality,
                    })
                    if not equality["fit_equivalent"]:
                        raise RuntimeError(
                            "current-basis reparameterization failed frozen fit "
                            f"tolerances at r={replicate}, {task}, d={draw}, k={budget}"
                        )

                    iu_auc, iu_ap = safe_metrics(
                        labels[evaluation], iu_scores[evaluation]
                    )
                    controls = optimistic_endpoint_controls(
                        labels, iu_scores, u2_scores, evaluation
                    )
                    scores_by_method = {
                        "iu": iu_scores,
                        "ta_liu": ta_scores,
                        "u2_logistic": u2_scores,
                        "anchored_pcr2_historical": historical_scores,
                        "anchored_pcr2_current": current_scores,
                        "anchored_pcr2_current_reparameterized": reference_scores,
                        **controls["scores"],
                    }
                    for method in CORE_SYNTHETIC_METHODS:
                        fallback = False
                        fallback_reason = ""
                        selected = ""
                        alpha = ""
                        if method == "ta_liu":
                            fallback = ta.gate_result.fallback
                            fallback_reason = ta.gate_result.fallback_reason
                        elif method == "u2_logistic":
                            fallback = u2_diagnostic["fallback"]
                            fallback_reason = u2_diagnostic["fallback_reason"]
                        elif method == "optimistic_endpoint_switch":
                            selected = controls["metrics"][method]["selected"]
                        elif method == "optimistic_interpolation":
                            alpha = controls["metrics"][method]["alpha_u2"]
                        metrics.append(metric_row(
                            replicate,
                            seed,
                            task,
                            draw,
                            budget,
                            calibration_hash,
                            method,
                            labels,
                            scores_by_method[method],
                            evaluation,
                            iu_auc,
                            iu_ap,
                            fallback=fallback,
                            fallback_reason=fallback_reason,
                            selected=selected,
                            alpha_u2=alpha,
                        ))

                    count = N
                    row_arrays["replicate"].append(
                        np.full(count, replicate, dtype=np.int16)
                    )
                    row_arrays["task"].append(
                        np.full(count, task_index, dtype=np.int8)
                    )
                    row_arrays["draw"].append(np.full(count, draw, dtype=np.int8))
                    row_arrays["budget"].append(
                        np.full(count, budget, dtype=np.int16)
                    )
                    row_arrays["sample_index"].append(
                        np.arange(count, dtype=np.int16)
                    )
                    row_arrays["is_calibration"].append(
                        np.isin(np.arange(count), calibration)
                    )
                    row_arrays["label"].append(labels.astype(np.int8))
                    for method, scores in scores_by_method.items():
                        row_arrays[f"score__{method}"].append(
                            np.asarray(scores, dtype=np.float64)
                        )
        print(
            f"[{replicate + 1:02d}/{REPLICATES:02d}] consumed selective pair "
            f"seed={seed}",
            flush=True,
        )

    packed_scores = pack_and_validate_row_scores(
        row_arrays,
        expected_rows=(
            REPLICATES * 2 * CALIBRATION_DRAWS * len(BUDGETS) * N
        ),
        expected_groups=REPLICATES * 2 * CALIBRATION_DRAWS * len(BUDGETS),
        samples_per_group=N,
    )
    packed_scores["task_names"] = np.asarray(
        ["selective_target_signal", "selective_target_nuisance"]
    )
    return metrics, geometry, fit_equality, invariants, packed_scores


def aggregate_synthetic(metrics):
    grouped = defaultdict(list)
    for row in metrics:
        key = (row["task"], row["replicate"], row["seed"], row["budget"], row["method"])
        grouped[key].append(row)
    datasets = []
    for key, rows in sorted(grouped.items()):
        task, replicate, seed, budget, method = key
        datasets.append({
            "version": VERSION,
            "task": task,
            "replicate": replicate,
            "seed": seed,
            "budget": budget,
            "method": method,
            "n_calibration_draws": len(rows),
            "auroc": float(np.mean([r["auroc"] for r in rows])),
            "auprc": float(np.mean([r["auprc"] for r in rows])),
            "auroc_delta_vs_iu": float(
                np.mean([r["auroc_delta_vs_iu"] for r in rows])
            ),
            "auprc_delta_vs_iu": float(
                np.mean([r["auprc_delta_vs_iu"] for r in rows])
            ),
            "fallback_fraction": float(np.mean([bool(r["fallback"]) for r in rows])),
            "mean_alpha_u2": (
                float(np.mean([float(r["alpha_u2"]) for r in rows]))
                if method == "optimistic_interpolation" else ""
            ),
        })

    summary_groups = defaultdict(list)
    for row in datasets:
        summary_groups[(row["task"], row["budget"], row["method"])].append(row)
    summaries = []
    for key, rows in sorted(summary_groups.items()):
        task, budget, method = key
        values = np.asarray([r["auroc_delta_vs_iu"] for r in rows]) * 100.0
        mean = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
        radius = float(student_t.ppf(0.975, len(values) - 1) * se)
        summaries.append({
            "version": VERSION,
            "task": task,
            "budget": budget,
            "method": method,
            "n_datasets": len(values),
            "mean_auroc_delta_pp": mean,
            "se_pp": se,
            "ci95_low_pp": mean - radius,
            "ci95_high_pp": mean + radius,
            "wins_vs_iu": int(np.sum(values > 0)),
            "losses_vs_iu": int(np.sum(values < 0)),
            "mean_auroc": float(np.mean([r["auroc"] for r in rows])),
            "mean_auprc": float(np.mean([r["auprc"] for r in rows])),
        })
    return datasets, summaries


def bootstrap_interval(values, *, seed, draws=10_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def literal_cell_stop_rule(method_summaries, replay):
    """Apply the frozen stop gate to every individual cell-level switch."""
    no_method_improves = not any(
        row["strictly_improves_cell_macro"] for row in method_summaries
    )
    every_cell_switch_below_one = all(
        row["optimistic_endpoint_switch_gain_pp"] < 1.0 for row in replay
    )
    return no_method_improves, every_cell_switch_below_one, bool(
        no_method_improves and every_cell_switch_below_one
    )


def reconcile_saved_real():
    with open(REAL_REPLICATES, newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    if len(all_rows) != 39_600:
        raise RuntimeError(f"unexpected semi-supervised row count: {len(all_rows)}")
    real_rows = [row for row in all_rows if row["source"] == "real"]
    budget_rows = [row for row in real_rows if int(row["budget"]) == 20]
    if len(real_rows) != 32_400 or len(budget_rows) != 5_760:
        raise RuntimeError("saved real semi-supervised row counts changed")
    real_cells = {row["unit"] for row in budget_rows}
    if len(real_cells) != 24:
        raise RuntimeError("expected exactly 24 saved real cells")
    validate_cartesian_rows(
        budget_rows,
        expected_cells=real_cells,
        expected_repetitions=range(30),
        expected_methods=EXPECTED_REAL_METHODS,
    )

    metadata = {}
    for row in budget_rows:
        metadata[row["unit"]] = {"group": row["group"], "domain": row["domain"]}
    means = real_cell_method_means(budget_rows, repetitions_per_cell=30)
    replay = []
    method_summaries = []
    for method_index, method in enumerate(CANDIDATE_REAL_METHODS):
        deltas = []
        switches = []
        for cell in sorted(metadata):
            incumbent = means[(cell, "upcr")]
            candidate = means[(cell, method)]
            delta_pp = 100.0 * (candidate - incumbent)
            switch_pp = max(0.0, delta_pp)
            deltas.append(delta_pp)
            switches.append(switch_pp)
            replay.append({
                "version": VERSION,
                "cell": cell,
                "group": metadata[cell]["group"],
                "domain": metadata[cell]["domain"],
                "budget": 20,
                "n_repetitions": 30,
                "method": method,
                "upcr_auroc": incumbent,
                "candidate_auroc": candidate,
                "delta_pp": delta_pp,
                "optimistic_endpoint_switch_gain_pp": switch_pp,
            })
        delta_ci = bootstrap_interval(deltas, seed=2200 + method_index)
        switch_ci = bootstrap_interval(switches, seed=2270 + method_index)
        group_means = {}
        for group in ("QA", "math"):
            selected = [
                row["delta_pp"] for row in replay
                if row["method"] == method and row["group"] == group
            ]
            group_means[group] = float(np.mean(selected))
        method_summaries.append({
            "method": method,
            "n_cells": len(deltas),
            "mean_delta_pp": float(np.mean(deltas)),
            "delta_bootstrap_ci95_pp": delta_ci,
            "wins": int(np.sum(np.asarray(deltas) > 0)),
            "losses": int(np.sum(np.asarray(deltas) < 0)),
            "strictly_improves_cell_macro": bool(np.mean(deltas) > 0),
            "mean_optimistic_endpoint_switch_gain_pp": float(np.mean(switches)),
            "max_optimistic_endpoint_switch_gain_pp": float(np.max(switches)),
            "n_cells_endpoint_switch_at_least_1pp": int(
                np.sum(np.asarray(switches) >= 1.0)
            ),
            "cells_endpoint_switch_at_least_1pp": [
                row["cell"] for row in replay
                if row["method"] == method
                and row["optimistic_endpoint_switch_gain_pp"] >= 1.0
            ],
            "endpoint_switch_bootstrap_ci95_pp": switch_ci,
            "qa_delta_pp": group_means["QA"],
            "math_delta_pp": group_means["math"],
        })

    with open(MIX_ROWS, newline="", encoding="utf-8") as handle:
        mix_rows = list(csv.DictReader(handle))
    with open(MIX_SUMMARY, encoding="utf-8") as handle:
        mix_summary = json.load(handle)
    mix_cells = [row["cell"] for row in mix_rows]
    if (
        len(mix_rows) != 24
        or len(set(mix_cells)) != 24
        or set(mix_cells) != real_cells
        or {int(row["n_splits"]) for row in mix_rows} != {5}
    ):
        raise RuntimeError("historical full-angle mix artifact changed")
    if mix_summary.get("n_cells") != 24 or "mix_splithalf" not in mix_summary:
        raise RuntimeError("historical full-angle mix summary changed")
    mix_delta = float(np.mean([float(row["delta_pp"]) for row in mix_rows]))
    recorded_mix_delta = float(mix_summary["mix_splithalf"]["mean_delta_pp"])
    if abs(mix_delta - recorded_mix_delta) > 1e-12:
        raise RuntimeError("mix row and summary artifacts disagree")

    (
        no_method_improves,
        every_endpoint_switch_below_one,
        stop_tested_family,
    ) = literal_cell_stop_rule(method_summaries, replay)
    decision = {
        "version": VERSION,
        "decision_name": "tested_u2_anchored_head_family",
        "candidate_methods": list(CANDIDATE_REAL_METHODS),
        "improves_definition": "strictly positive 24-cell macro mean delta versus U-PCR",
        "endpoint_switch_gate_definition": (
            "every individual candidate×cell optimistic endpoint switch is "
            "strictly below +1.00 AUROC point"
        ),
        "stop_tested_family": stop_tested_family,
        "no_candidate_strictly_improves": no_method_improves,
        "all_cell_level_optimistic_endpoint_switches_below_1pp": (
            every_endpoint_switch_below_one
        ),
        "does_not_close_every_u2_angle_or_future_estimator": True,
        "normalized_real_row_score_interpolation_available": False,
        "historical_full_angle_context": {
            "mean_delta_pp": mix_delta,
            "ci95_pp": mix_summary["mix_splithalf"]["ci95"],
            "wins": mix_summary["mix_splithalf"]["wins"],
            "losses": mix_summary["mix_splithalf"]["losses"],
            "compatible_with_current_fixed_stable_schema": False,
            "reason": (
                "historical full feature pool, per-split sign(rho) orientation, "
                "and historical deployed U-PCR configuration"
            ),
        },
        "method_summaries": method_summaries,
    }
    return replay, decision


def summary_lookup(summaries, task, budget, method):
    return next(
        row for row in summaries
        if row["task"] == task and row["budget"] == budget and row["method"] == method
    )


def make_plots(out, summaries, geometry, decision):
    tasks = ("selective_target_signal", "selective_target_nuisance")
    labels = {tasks[0]: "Target g", tasks[1]: "Target u"}
    methods = (
        "ta_liu",
        "u2_logistic",
        "anchored_pcr2_historical",
        "anchored_pcr2_current",
    )
    colors = {
        "ta_liu": "#4c78a8",
        "u2_logistic": "#f58518",
        "anchored_pcr2_historical": "#54a24b",
        "anchored_pcr2_current": "#b279a2",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, task in zip(axes, tasks):
        for method in methods:
            rows = [summary_lookup(summaries, task, b, method) for b in BUDGETS]
            axis.plot(
                BUDGETS,
                [row["mean_auroc_delta_pp"] for row in rows],
                marker="o",
                label=method,
                color=colors[method],
            )
            axis.fill_between(
                BUDGETS,
                [row["ci95_low_pp"] for row in rows],
                [row["ci95_high_pp"] for row in rows],
                color=colors[method],
                alpha=0.12,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(labels[task])
        axis.set_xlabel("Trusted labels")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("AUROC change vs IU-PCR (points)")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Consumed paired synthetic replay")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "synthetic_method_deltas.png"), dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for basis_name, color in (("current", "#4c78a8"), ("historical", "#e45756")):
        rows = sorted(
            [row for row in geometry if row["basis"] == basis_name],
            key=lambda row: row["replicate"],
        )
        degrees = np.degrees(
            [max(float(row["max_principal_angle_rad"]), 1e-16) for row in rows]
        )
        axis.plot(
            [row["replicate"] for row in rows],
            degrees,
            marker="o",
            label=basis_name,
            color=color,
        )
    axis.axhline(
        np.degrees(GEOMETRY_TOLERANCES["max_principal_angle_rad"]),
        color="black",
        linestyle="--",
        label="equivalence tolerance",
    )
    axis.set_yscale("log")
    axis.set_xlabel("Synthetic dataset replicate")
    axis.set_ylabel("Maximum principal angle (degrees, log scale)")
    axis.set_title("Does each anchored basis span ordinary U2?")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "basis_geometry.png"), dpi=170)
    plt.close(fig)

    controls = (
        "optimistic_endpoint_switch",
        "fixed_average",
        "optimistic_interpolation",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, task in zip(axes, tasks):
        for method in controls:
            rows = [summary_lookup(summaries, task, b, method) for b in BUDGETS]
            axis.plot(
                BUDGETS,
                [row["mean_auroc_delta_pp"] for row in rows],
                marker="o",
                label=method,
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(labels[task])
        axis.set_xlabel("Trusted labels")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Optimistic AUROC gain vs IU-PCR (points)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Synthetic score-combination diagnostics (evaluation-label optimistic)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "synthetic_optimistic_controls.png"), dpi=170)
    plt.close(fig)

    rows = decision["method_summaries"]
    x = np.arange(len(rows))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(
        x - width / 2,
        [row["mean_delta_pp"] for row in rows],
        width,
        label="candidate mean delta",
        color="#4c78a8",
    )
    axis.bar(
        x + width / 2,
        [row["mean_optimistic_endpoint_switch_gain_pp"] for row in rows],
        width,
        label="macro mean endpoint switch",
        color="#f58518",
    )
    axis.scatter(
        x + width / 2,
        [row["max_optimistic_endpoint_switch_gain_pp"] for row in rows],
        marker="D",
        color="#e45756",
        label="maximum cell endpoint switch",
        zorder=4,
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="1-point gate")
    axis.set_xticks(x, [row["method"] for row in rows], rotation=15, ha="right")
    axis.set_ylabel("24-cell macro AUROC change (points)")
    axis.set_title("Saved real-artifact reconciliation at 20 labels")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "saved_real_reconciliation.png"), dpi=170)
    plt.close(fig)


def write_report(out, summaries, geometry, fit_equality, decision):
    current_angles = [
        float(row["max_principal_angle_rad"])
        for row in geometry if row["basis"] == "current"
    ]
    historical_angles = [
        float(row["max_principal_angle_rad"])
        for row in geometry if row["basis"] == "historical"
    ]
    lines = [
        "# U2-prior reconciliation — consumed development checkpoint",
        "",
        f"Version: `{VERSION}`",
        "",
        "## Scope",
        "",
        "This run regenerated only the already consumed paired synthetic development",
        "matrices and read saved CSV/JSON experiment results. It did not open raw real",
        "hallucination features or labels. It generated no confirmation data and every",
        f"synthetic seed remained below the reserved `{CONFIRMATION_SEED_OFFSET:,}` block.",
        "",
        "## Reconciliation result",
        "",
        f"- Current anchored basis: max principal angle `{max(current_angles):.3e}` radians;",
        "  all eight datasets pass the frozen geometry gates.",
        f"- Historical anchored basis: max principal angle `{max(historical_angles):.3e}` radians;",
        "  it differed under the frozen historical U-PCR configuration, which includes",
        "  exclusion, recomputation, fallback, and component-setting differences. This",
        "  checkpoint does not isolate which setting caused the difference.",
        f"- Current reparameterized fits passing all equality tolerances: "
        f"`{sum(bool(row['fit_equivalent']) for row in fit_equality)}/{len(fit_equality)}`.",
        "",
        "Therefore current IU-prior logistic in two dimensions is a coordinate change",
        "of the ordinary U2 head on these matrices. It is not a distinct estimator class.",
        "Historical `anchored_pcr2` must be judged separately when its excluded-feature",
        "basis does not span full-matrix U2.",
        "",
        "## Synthetic mechanism at 16 labels",
        "",
        "All values below are AUROC points versus ordinary IU-PCR, after averaging the",
        "16 calibration draws within each of eight independent matrices.",
        "",
        "| target | TA-LIU | U2 logistic | historical anchored2 | current anchored2 | optimistic interpolation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, label in (
        ("selective_target_signal", "g"),
        ("selective_target_nuisance", "u"),
    ):
        values = [
            summary_lookup(summaries, task, 16, method)["mean_auroc_delta_pp"]
            for method in (
                "ta_liu",
                "u2_logistic",
                "anchored_pcr2_historical",
                "anchored_pcr2_current",
                "optimistic_interpolation",
            )
        ]
        lines.append(
            f"| {label} | " + " | ".join(f"{value:+.3f}" for value in values) + " |"
        )
    lines.extend([
        "",
        "The interpolation is selected and scored on the same evaluation labels. It is",
        "an optimistic mechanism diagnostic, not a deployable result.",
        "",
        "## Saved real artifacts at 20 labels",
        "",
        "The current-schema semi-supervised CSV was averaged over 30 repetitions within",
        "each cell and method before comparing 24 cells.",
        "",
        "| tested method | mean delta | 95% cell bootstrap | W/L | endpoint switch mean (95% bootstrap) | maximum cell switch | cells >=1 point |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in decision["method_summaries"]:
        lines.append(
            f"| {row['method']} | {row['mean_delta_pp']:+.3f} | "
            f"[{row['delta_bootstrap_ci95_pp'][0]:+.3f}, {row['delta_bootstrap_ci95_pp'][1]:+.3f}] | "
            f"{row['wins']}/{row['losses']} | "
            f"{row['mean_optimistic_endpoint_switch_gain_pp']:+.3f} "
            f"([{row['endpoint_switch_bootstrap_ci95_pp'][0]:+.3f}, "
            f"{row['endpoint_switch_bootstrap_ci95_pp'][1]:+.3f}]) | "
            f"{row['max_optimistic_endpoint_switch_gain_pp']:+.3f} | "
            f"{row['n_cells_endpoint_switch_at_least_1pp']} |"
        )
    context = decision["historical_full_angle_context"]
    lines.extend([
        "",
        f"The historical split-half full-angle sweep was `{context['mean_delta_pp']:+.3f}`",
        f"points (95% interval `[{context['ci95_pp'][0]:+.3f}, {context['ci95_pp'][1]:+.3f}]`).",
        "It is contextual only because it used the historical full feature pool,",
        "per-split sign(rho) orientation, and the historical deployed U-PCR configuration.",
        "It cannot close every angle under the current fixed-stable schema.",
        "",
        "## Decision",
        "",
        f"`stop_tested_family = {str(decision['stop_tested_family']).lower()}`.",
        "",
        "The frozen stop rule is literal at cell level. A false flag does not promote",
        "a method: it means at least one cell-specific endpoint switch reached the",
        "one-point threshold, even if the method's overall mean was negative. A true",
        "flag would stop further variants of the tested family. Neither outcome proves",
        "that every U2 angle or every future U2 estimator is closed. The next",
        "branch remains a user decision after comparing few-label subset-adaptation",
        "headroom with a current-schema, recycling-guarded FUSE pseudo-target probe.",
        "",
        "## Plots",
        "",
        "- `synthetic_method_deltas.png`",
        "- `basis_geometry.png`",
        "- `synthetic_optimistic_controls.png`",
        "- `saved_real_reconciliation.png`",
        "",
        "No confirmation experiment was run.",
    ])
    with open(os.path.join(out, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def provenance():
    relative_sources = (
        SPEC,
        "spectral_utils/u2_prior_reconciliation.py",
        "scripts/u2_prior_reconciliation.py",
        "scripts/test_u2_prior_reconciliation.py",
        "spectral_utils/semi_supervised_fusion.py",
        "spectral_utils/feature_contract.py",
        "spectral_utils/target_anchored_laplacian.py",
        "spectral_utils/laplacian_upcr.py",
        "spectral_utils/upcr.py",
        "spectral_utils/fusion_utils.py",
        "scripts/target_anchored_laplacian_synthetic.py",
    )
    return {
        "version": VERSION,
        "scope": "consumed development synthetic plus saved real artifacts",
        "confirmation_generated": False,
        "raw_real_features_opened": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "source_hashes": {
            path: sha256_file(os.path.join(REPO, path)) for path in relative_sources
        },
        "input_hashes": {
            os.path.relpath(path, REPO): sha256_file(path)
            for path in (REAL_REPLICATES, MIX_ROWS, MIX_SUMMARY)
        },
        "config": {
            "development_seed_offset": DEVELOPMENT_SEED_OFFSET,
            "confirmation_seed_offset": CONFIRMATION_SEED_OFFSET,
            "replicates": REPLICATES,
            "n": N,
            "calibration_draws": CALIBRATION_DRAWS,
            "budgets": list(BUDGETS),
            "current_upcr": CURRENT_UPCR_KWARGS,
            "historical_upcr": HISTORICAL_UPCR_KWARGS,
            "geometry_tolerances": GEOMETRY_TOLERANCES,
            "fit_tolerances": FIT_TOLERANCES,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_DEFAULT)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    metrics, geometry, fit_equality, invariants, row_scores = run_synthetic()
    datasets, summaries = aggregate_synthetic(metrics)
    replay, decision = reconcile_saved_real()

    write_csv(os.path.join(args.out, "synthetic_metrics.csv"), metrics)
    write_csv(os.path.join(args.out, "synthetic_datasets.csv"), datasets)
    write_csv(os.path.join(args.out, "synthetic_summary.csv"), summaries)
    write_csv(os.path.join(args.out, "basis_geometry.csv"), geometry)
    write_csv(os.path.join(args.out, "fit_equivalence.csv"), fit_equality)
    write_csv(os.path.join(args.out, "paired_invariants.csv"), invariants)
    write_csv(os.path.join(args.out, "real_cell_replay.csv"), replay)
    np.savez_compressed(os.path.join(args.out, "row_scores.npz"), **row_scores)
    write_json(os.path.join(args.out, "decision.json"), decision)
    write_json(os.path.join(args.out, "provenance.json"), provenance())
    make_plots(args.out, summaries, geometry, decision)
    write_report(args.out, summaries, geometry, fit_equality, decision)
    print(json.dumps({
        "out": args.out,
        "synthetic_metric_rows": len(metrics),
        "synthetic_dataset_rows": len(datasets),
        "row_score_rows": len(row_scores["label"]),
        "stop_tested_family": decision["stop_tested_family"],
        "confirmation_generated": False,
    }, indent=2))


if __name__ == "__main__":
    main()
