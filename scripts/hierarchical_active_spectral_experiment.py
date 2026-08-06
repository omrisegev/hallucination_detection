#!/usr/bin/env python3
"""LOFO hierarchical transfer and active-label experiment.

The protocol and gates are frozen in
``SPEC_HIERARCHICAL_ACTIVE_SPECTRAL_V2.md``.  The script reuses the exact v1
real-data split namespace so the local controlled arm is a paired reproduction.

Usage:
    python scripts/hierarchical_active_spectral_experiment.py
    python scripts/hierarchical_active_spectral_experiment.py --quick
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import types

import numpy as np
from sklearn.metrics import roc_auc_score


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from scripts import semi_supervised_spectral_experiment as v1          # noqa: E402
from spectral_utils.feature_contract import SCHEMA_VERSION             # noqa: E402
from spectral_utils.hierarchical_spectral_fusion import (              # noqa: E402
    fisher_d_optimal_order,
    fit_grouped_logistic_head,
    fit_score_head,
    fit_shared_representation,
)
from spectral_utils.semi_supervised_fusion import (                    # noqa: E402
    fit_logistic_head,
    orient_weight,
    spectral_score_basis,
)
from spectral_utils.feature_contract import consensus_anchor           # noqa: E402
from spectral_utils.upcr import upcr_fit                                # noqa: E402


VERSION = "hierarchical-active-spectral-v2-2026-08-06"
DEFAULT_OUT = os.path.join(REPO, "results", "hierarchical_active_spectral_v2")
BUDGETS = (0, 5, 10, 20, 40, 80)
DONOR_LABELS_PER_CELL = 20
REAL_REPEATS = 20
SYNTHETIC_REPEATS = 20
N_BOOT = 10000
LOCAL_PRIOR_STRENGTH = 10.0
SHARED_PRIOR_STRENGTH = 20.0
GROUP_INTERCEPT_STRENGTH = 1.0

METHODS = (
    "upcr",
    "local_controlled2",
    "local_uniform2",
    "local_active2",
    "pooled_domain_lofo",
    "pooled_all_lofo",
    "hybrid_domain_uniform",
    "hybrid_domain_active",
)

SYNTHETIC_WORLDS = (
    "upcr_sufficient",
    "shared_correction",
    "family_shift",
)


def stable_seed(*parts):
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2 ** 32)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def synthetic_cells(world, repetition, n=1200, m=18):
    """Return a 12-cell, three-family meta-learning world."""
    cells = {}
    feature_names = tuple(f"f{index:02d}" for index in range(m))
    for cell_index in range(12):
        family_index = cell_index // 4
        rng = np.random.default_rng(stable_seed(
            VERSION, "synthetic", world, repetition, cell_index,
        ))
        prevalence = 0.35 + 0.15 * ((cell_index % 4) / 3.0)
        labels = rng.binomial(1, prevalence, size=n).astype(int)
        target = 2.0 * labels - 1.0
        noise = rng.normal(size=(n, m))

        if world == "upcr_sufficient":
            strength = np.linspace(0.18, 0.55, m)
            matrix = target[:, None] * strength + noise
        elif world == "shared_correction":
            split = m // 2
            strength = np.r_[np.full(split, 0.09), np.linspace(0.38, 0.62, m - split)]
            shared = rng.normal(size=n)
            matrix = target[:, None] * strength + 0.65 * noise
            matrix[:, :split] += 1.35 * shared[:, None]
        elif world == "family_shift":
            reliable = np.arange(6 * family_index, 6 * (family_index + 1))
            strength = np.zeros(m)
            strength[reliable] = np.linspace(0.42, 0.62, len(reliable))
            matrix = target[:, None] * strength + 0.80 * noise
        else:
            raise ValueError(world)

        name = f"s{cell_index:02d}"
        cells[name] = {
            "matrix": matrix,
            "labels": labels,
            "features": feature_names,
            "domain": "synthetic",
            "family": f"family{family_index}",
        }
    return cells


def common_feature_vocabulary(cells):
    common = set.intersection(*(set(cell["features"]) for cell in cells.values()))
    if not common:
        raise RuntimeError("cells have no common feature vocabulary")
    return tuple(sorted(common))


def prepare_cell(cell, unit, repetition, common_features):
    train, test, y_train, y_test = v1.split_and_standardize(
        cell["matrix"], cell["labels"], unit, repetition,
    )
    upcr = upcr_fit(train.T, **v1.UPCR_FIT)
    upcr_weight = orient_weight(upcr.w, train, consensus_anchor(train))
    local_basis = spectral_score_basis(train, upcr_weight, rank=2)
    representation = fit_shared_representation(
        train, cell["features"], common_features, upcr_weight,
    )
    shared_train = representation.transform(train)
    shared_test = representation.transform(test)
    return {
        "train": train,
        "test": test,
        "y_train": y_train,
        "y_test": y_test,
        "upcr_weight": upcr_weight,
        "upcr_train": shared_train[:, 0],
        "upcr_test": shared_test[:, 0],
        "local_basis": local_basis,
        "local_train": train @ local_basis,
        "local_test": test @ local_basis,
        "shared_train": shared_train,
        "shared_test": shared_test,
        "family": cell["family"],
        "domain": cell["domain"],
        "n_features": train.shape[1],
        "upcr_n_kept": int(upcr.keep.sum()),
    }


def uniform_order(n_rows, unit, repetition, namespace):
    rng = np.random.default_rng(stable_seed(VERSION, namespace, unit, repetition))
    return rng.permutation(int(n_rows))


def fit_donor_head(prepared, target, scope, repetition):
    target_cell = prepared[target]
    donors = []
    for name, cell in prepared.items():
        if cell["family"] == target_cell["family"]:
            continue
        if scope == "domain" and cell["domain"] != target_cell["domain"]:
            continue
        donors.append(name)
    if not donors:
        dimension = prepared[target]["shared_train"].shape[1]
        return np.r_[1.0, np.zeros(dimension - 1)], [], 0, True

    matrices = []
    labels = []
    groups = []
    for donor in sorted(donors):
        cell = prepared[donor]
        order = uniform_order(
            len(cell["y_train"]), donor, repetition,
            f"donor-{scope}-for-{prepared[target]['family']}",
        )
        chosen = order[: min(DONOR_LABELS_PER_CELL, len(order))]
        matrices.append(cell["shared_train"][chosen])
        labels.append(cell["y_train"][chosen])
        groups.extend([donor] * len(chosen))
    matrix = np.vstack(matrices)
    y = np.concatenate(labels)
    prior = np.r_[1.0, np.zeros(matrix.shape[1] - 1)]
    fit = fit_grouped_logistic_head(
        matrix,
        y,
        groups,
        prior,
        prior_strength=SHARED_PRIOR_STRENGTH,
        intercept_strength=GROUP_INTERCEPT_STRENGTH,
    )
    return fit.coefficients, donors, len(y), fit.converged


def hybrid_scores(cell, pooled_coefficients):
    correction_train = cell["shared_train"][:, 1:] @ pooled_coefficients[1:]
    correction_test = cell["shared_test"][:, 1:] @ pooled_coefficients[1:]
    center = float(correction_train.mean())
    scale = float(correction_train.std())
    if scale <= 1e-9:
        scale = 1.0
        correction_train = np.zeros_like(correction_train)
        correction_test = np.zeros_like(correction_test)
        center = 0.0
    train = np.column_stack([
        cell["upcr_train"], (correction_train - center) / scale,
    ])
    test = np.column_stack([
        cell["upcr_test"], (correction_test - center) / scale,
    ])
    prior = np.asarray([pooled_coefficients[0], scale], dtype=float)
    return train, test, prior


def score_auc(labels, score):
    score = np.asarray(score, dtype=float)
    if not np.isfinite(score).all() or score.std() < 1e-12:
        raise RuntimeError("non-finite or constant held-out score")
    return float(roc_auc_score(labels, score))


def run_target(source, world, target, prepared, repetition):
    cell = prepared[target]
    baseline_auc = score_auc(cell["y_test"], cell["upcr_test"])
    domain_beta, domain_donors, domain_labels, domain_converged = fit_donor_head(
        prepared, target, "domain", repetition,
    )
    all_beta, all_donors, all_labels, all_converged = fit_donor_head(
        prepared, target, "all", repetition,
    )
    pooled_domain_auc = score_auc(
        cell["y_test"], cell["shared_test"] @ domain_beta,
    )
    pooled_all_auc = score_auc(cell["y_test"], cell["shared_test"] @ all_beta)
    hybrid_train, hybrid_test, hybrid_prior = hybrid_scores(cell, domain_beta)

    max_budget = min(max(BUDGETS), len(cell["y_train"]))
    local_uniform = uniform_order(len(cell["y_train"]), target, repetition, "target-uniform")
    hybrid_uniform = local_uniform
    local_active = fisher_d_optimal_order(
        cell["local_train"], np.asarray([1.0, 0.0]), max_budget,
        prior_strength=LOCAL_PRIOR_STRENGTH,
    )
    hybrid_active = fisher_d_optimal_order(
        hybrid_train, hybrid_prior, max_budget,
        prior_strength=LOCAL_PRIOR_STRENGTH,
    )

    base = {
        "source": source,
        "world": world,
        "unit": target,
        "group": cell["domain"],
        "family": cell["family"],
        "repetition": repetition,
        "n_train": len(cell["y_train"]),
        "n_test": len(cell["y_test"]),
        "n_features": cell["n_features"],
        "upcr_n_kept": cell["upcr_n_kept"],
        "n_domain_donors": len(domain_donors),
        "n_domain_donor_labels": domain_labels,
        "n_all_donors": len(all_donors),
        "n_all_donor_labels": all_labels,
        "domain_shared_upcr_coefficient": float(domain_beta[0]),
        "domain_shared_residual_norm": float(np.linalg.norm(domain_beta[1:])),
    }
    rows = []

    def add(
        budget, method, auc, selected, donor_labels=0, converged=True,
        coefficients=None, prior_coefficients=None,
    ):
        selected = np.asarray(selected, dtype=int)
        coefficients = np.asarray(
            [] if coefficients is None else coefficients, dtype=float,
        )
        prior_coefficients = np.asarray(
            [] if prior_coefficients is None else prior_coefficients, dtype=float,
        )
        positive_rate = float(cell["y_train"][selected].mean()) if len(selected) else float("nan")
        rows.append({
            **base,
            "budget": int(budget),
            "method": method,
            "auc": float(auc),
            "delta_vs_upcr": float(auc - baseline_auc),
            "n_target_labels": int(len(selected)),
            "n_donor_labels_used": int(donor_labels),
            "selected_positive_rate": positive_rate,
            "selected_both_classes": int(
                len(selected) > 0 and len(np.unique(cell["y_train"][selected])) == 2
            ),
            "converged": int(bool(converged)),
            "head_coefficient_upcr": (
                float(coefficients[0]) if len(coefficients) >= 1 else float("nan")
            ),
            "head_coefficient_correction": (
                float(coefficients[1]) if len(coefficients) >= 2 else float("nan")
            ),
            "prior_coefficient_upcr": (
                float(prior_coefficients[0])
                if len(prior_coefficients) >= 1 else float("nan")
            ),
            "prior_coefficient_correction": (
                float(prior_coefficients[1])
                if len(prior_coefficients) >= 2 else float("nan")
            ),
        })

    for budget in BUDGETS:
        actual = min(int(budget), len(cell["y_train"]))
        controlled = v1.stratified_label_indices(
            cell["y_train"], actual, target, repetition,
        )
        uniform = local_uniform[:actual]
        local_selected = local_active[:actual]
        hybrid_selected = hybrid_active[:actual]

        add(budget, "upcr", baseline_auc, [])
        add(
            budget, "pooled_domain_lofo", pooled_domain_auc, [],
            donor_labels=domain_labels, converged=domain_converged,
        )
        add(
            budget, "pooled_all_lofo", pooled_all_auc, [],
            donor_labels=all_labels, converged=all_converged,
        )

        if budget == 0:
            for method in ("local_controlled2", "local_uniform2", "local_active2"):
                add(
                    budget, method, baseline_auc, [],
                    coefficients=np.asarray([1.0, 0.0]),
                    prior_coefficients=np.asarray([1.0, 0.0]),
                )
            for method in ("hybrid_domain_uniform", "hybrid_domain_active"):
                add(
                    budget, method, pooled_domain_auc, [],
                    donor_labels=domain_labels, converged=domain_converged,
                    coefficients=hybrid_prior, prior_coefficients=hybrid_prior,
                )
            continue

        local_prior = np.asarray([1.0, 0.0])
        local_fits = {
            "local_controlled2": (controlled, fit_score_head(
                cell["local_train"][controlled], cell["y_train"][controlled], local_prior,
                prior_strength=LOCAL_PRIOR_STRENGTH,
            )),
            "local_uniform2": (uniform, fit_score_head(
                cell["local_train"][uniform], cell["y_train"][uniform], local_prior,
                prior_strength=LOCAL_PRIOR_STRENGTH,
            )),
            "local_active2": (local_selected, fit_score_head(
                cell["local_train"][local_selected], cell["y_train"][local_selected], local_prior,
                prior_strength=LOCAL_PRIOR_STRENGTH,
            )),
        }
        for method, (selected, fit) in local_fits.items():
            add(
                budget, method,
                score_auc(cell["y_test"], fit.intercept + cell["local_test"] @ fit.weight),
                selected, converged=fit.converged,
                coefficients=fit.weight, prior_coefficients=local_prior,
            )

        hybrid_fits = {
            "hybrid_domain_uniform": (hybrid_uniform[:actual], fit_score_head(
                hybrid_train[hybrid_uniform[:actual]],
                cell["y_train"][hybrid_uniform[:actual]],
                hybrid_prior,
                prior_strength=LOCAL_PRIOR_STRENGTH,
            )),
            "hybrid_domain_active": (hybrid_selected, fit_score_head(
                hybrid_train[hybrid_selected], cell["y_train"][hybrid_selected],
                hybrid_prior,
                prior_strength=LOCAL_PRIOR_STRENGTH,
            )),
        }
        for method, (selected, fit) in hybrid_fits.items():
            add(
                budget, method,
                score_auc(cell["y_test"], fit.intercept + hybrid_test @ fit.weight),
                selected, donor_labels=domain_labels,
                converged=fit.converged and domain_converged,
                coefficients=fit.weight, prior_coefficients=hybrid_prior,
            )
    return rows


def run_collection(source, world, cells, repetitions):
    rows = []
    common_features = common_feature_vocabulary(cells)
    for repetition in range(repetitions):
        prepared = {
            name: prepare_cell(
                cell,
                name if source == "real" else f"{world}:{name}",
                repetition,
                common_features,
            )
            for name, cell in cells.items()
        }
        for target in sorted(prepared):
            rows.extend(run_target(source, world, target, prepared, repetition))
    return rows, common_features


def bootstrap_ci(values, name, n_boot=N_BOOT):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(stable_seed(VERSION, "bootstrap", name))
    means = np.empty(int(n_boot), dtype=float)
    for start in range(0, int(n_boot), 1000):
        size = min(1000, int(n_boot) - start)
        picks = rng.integers(0, len(values), size=(size, len(values)))
        means[start:start + size] = values[picks].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def unit_means(rows, source, world, budget, method, group=None):
    selected = [
        row for row in rows
        if row["source"] == source and row["world"] == world
        and row["budget"] == budget and row["method"] == method
        and (group is None or row["group"] == group)
    ]
    units = sorted({row["unit"] for row in selected})
    return {
        unit: float(np.mean([row["auc"] for row in selected if row["unit"] == unit]))
        for unit in units
    }


def paired_contrast(rows, source, world, budget, reference, candidate, group=None):
    ref = unit_means(rows, source, world, budget, reference, group)
    cand = unit_means(rows, source, world, budget, candidate, group)
    units = sorted(set(ref) & set(cand))
    delta = np.asarray([cand[unit] - ref[unit] for unit in units], dtype=float)
    lo, hi = bootstrap_ci(
        delta, f"contrast-{source}-{world}-{group}-{budget}-{reference}-{candidate}",
    )
    return {
        "source": source,
        "world": world,
        "group": "all" if group is None else group,
        "budget": budget,
        "reference": reference,
        "candidate": candidate,
        "n_units": len(units),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "ci95_low": lo,
        "ci95_high": hi,
        "wins": int(np.sum(delta > 0)),
        "losses": int(np.sum(delta < 0)),
        "catastrophic_losses_5pp": int(np.sum(delta <= -0.05)),
    }


def learning_curves(rows):
    output = []
    panels = [("real", "real", None), ("real", "real", "QA"), ("real", "real", "math")]
    panels += [("synthetic", world, None) for world in SYNTHETIC_WORLDS]
    for source, world, group in panels:
        for budget in BUDGETS:
            for method in METHODS:
                means = unit_means(rows, source, world, budget, method, group)
                if not means:
                    continue
                values = np.asarray(list(means.values()), dtype=float)
                panel = "all" if group is None else group
                lo, hi = bootstrap_ci(values, f"curve-{source}-{world}-{panel}-{budget}-{method}")
                output.append({
                    "source": source,
                    "world": world,
                    "panel": panel,
                    "budget": budget,
                    "method": method,
                    "n_units": len(values),
                    "mean_auc": float(values.mean()),
                    "ci95_low": lo,
                    "ci95_high": hi,
                })
    return output


def all_contrasts(rows):
    comparisons = (
        ("upcr", "local_controlled2"),
        ("upcr", "local_uniform2"),
        ("upcr", "local_active2"),
        ("upcr", "pooled_domain_lofo"),
        ("upcr", "pooled_all_lofo"),
        ("upcr", "hybrid_domain_uniform"),
        ("upcr", "hybrid_domain_active"),
        ("local_uniform2", "local_active2"),
        ("local_uniform2", "hybrid_domain_uniform"),
        ("local_uniform2", "hybrid_domain_active"),
        ("hybrid_domain_uniform", "hybrid_domain_active"),
        ("pooled_domain_lofo", "hybrid_domain_active"),
    )
    output = []
    for budget in BUDGETS:
        for reference, candidate in comparisons:
            output.append(paired_contrast(
                rows, "real", "real", budget, reference, candidate,
            ))
            for domain in ("QA", "math"):
                output.append(paired_contrast(
                    rows, "real", "real", budget, reference, candidate, domain,
                ))
            for world in SYNTHETIC_WORLDS:
                output.append(paired_contrast(
                    rows, "synthetic", world, budget, reference, candidate,
                ))
    return output


def lookup(contrasts, source, world, group, budget, reference, candidate):
    matches = [
        row for row in contrasts
        if row["source"] == source and row["world"] == world
        and row["group"] == group and row["budget"] == budget
        and row["reference"] == reference and row["candidate"] == candidate
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one contrast for {source}/{world}/{group}/{budget}/"
            f"{reference}/{candidate}; got {len(matches)}"
        )
    return matches[0]


def build_decision(contrasts, eligible):
    primary = lookup(
        contrasts, "real", "real", "all", 20, "upcr", "hybrid_domain_active",
    )
    local = lookup(
        contrasts, "real", "real", "all", 20,
        "local_uniform2", "hybrid_domain_active",
    )
    acquisition = lookup(
        contrasts, "real", "real", "all", 20,
        "hybrid_domain_uniform", "hybrid_domain_active",
    )
    qa = lookup(
        contrasts, "real", "real", "QA", 20, "upcr", "hybrid_domain_active",
    )
    math = lookup(
        contrasts, "real", "real", "math", 20, "upcr", "hybrid_domain_active",
    )
    shared = lookup(
        contrasts, "synthetic", "shared_correction", "all", 20,
        "upcr", "hybrid_domain_active",
    )
    shared_active = lookup(
        contrasts, "synthetic", "shared_correction", "all", 20,
        "hybrid_domain_uniform", "hybrid_domain_active",
    )
    sufficient = lookup(
        contrasts, "synthetic", "upcr_sufficient", "all", 20,
        "upcr", "hybrid_domain_active",
    )
    shift = lookup(
        contrasts, "synthetic", "family_shift", "all", 20,
        "upcr", "pooled_domain_lofo",
    )
    gates = []

    def gate(name, observed, operator, threshold):
        if operator == ">=":
            passed = observed >= threshold
        elif operator == ">":
            passed = observed > threshold
        elif operator == "<=":
            passed = observed <= threshold
        else:
            raise ValueError(operator)
        gates.append({
            "gate": name,
            "observed": float(observed),
            "operator": operator,
            "threshold": float(threshold),
            "pass": bool(passed),
        })

    gate("real_mean_vs_upcr", primary["mean_delta"], ">=", 0.010)
    gate("real_ci_low_vs_upcr", primary["ci95_low"], ">", 0.0)
    gate("real_vs_local_uniform", local["mean_delta"], ">=", 0.0)
    gate("real_active_vs_hybrid_uniform", acquisition["mean_delta"], ">=", 0.0)
    gate("real_qa_vs_upcr", qa["mean_delta"], ">=", -0.005)
    gate("real_math_vs_upcr", math["mean_delta"], ">=", -0.005)
    gate("real_catastrophic_losses", primary["catastrophic_losses_5pp"], "<=", 2)
    gate("synthetic_shared_mean", shared["mean_delta"], ">=", 0.010)
    gate("synthetic_shared_ci_low", shared["ci95_low"], ">", 0.0)
    gate("synthetic_sufficient_no_harm", sufficient["mean_delta"], ">=", -0.005)
    gate("synthetic_family_shift_no_harm", shift["mean_delta"], ">=", -0.010)
    gate("synthetic_shared_active_vs_uniform", shared_active["mean_delta"], ">=", 0.0)

    passed = bool(eligible and all(row["pass"] for row in gates))
    return {
        "eligible_confirmatory_run": bool(eligible),
        "all_gates_pass": passed,
        "decision": (
            "PROMOTE_FOR_PROSPECTIVE_FAMILY_REPLAY" if passed else
            ("NOT_A_CONFIRMATORY_RUN" if not eligible else "STOP_AND_REVISE")
        ),
        "gates": gates,
    }


def convergence(rows, max_repeats):
    output = []
    methods = (
        "local_uniform2", "local_active2", "pooled_domain_lofo",
        "hybrid_domain_uniform", "hybrid_domain_active",
    )
    for upto in range(1, max_repeats + 1):
        subset = [
            row for row in rows
            if row["source"] == "real" and row["budget"] == 20
            and row["repetition"] < upto
        ]
        for method in methods:
            contrast = paired_contrast(
                subset, "real", "real", 20, "upcr", method,
            )
            output.append({
                "repeat_count": upto,
                "budget": 20,
                "method": method,
                "mean_delta_vs_upcr": contrast["mean_delta"],
            })
    return output


def acquisition_summary(rows):
    output = []
    for source, world in [("real", "real")] + [
        ("synthetic", name) for name in SYNTHETIC_WORLDS
    ]:
        for budget in BUDGETS[1:]:
            for method in (
                "local_controlled2", "local_uniform2", "local_active2",
                "hybrid_domain_uniform", "hybrid_domain_active",
            ):
                selected = [
                    row for row in rows
                    if row["source"] == source and row["world"] == world
                    and row["budget"] == budget and row["method"] == method
                ]
                if not selected:
                    continue
                rates = np.asarray([
                    row["selected_positive_rate"] for row in selected
                    if np.isfinite(row["selected_positive_rate"])
                ])
                output.append({
                    "source": source,
                    "world": world,
                    "budget": budget,
                    "method": method,
                    "mean_selected_positive_rate": float(rates.mean()),
                    "both_classes_fraction": float(np.mean([
                        row["selected_both_classes"] for row in selected
                    ])),
                })
    return output


def curve_value(curves, source, world, budget, method, panel="all"):
    matches = [
        row for row in curves
        if row["source"] == source and row["world"] == world
        and row["panel"] == panel and row["budget"] == budget
        and row["method"] == method
    ]
    return float("nan") if not matches else matches[0]["mean_auc"]


def render_report(summary, curves, contrasts, acquisition):
    lines = [
        "# Hierarchical and active spectral correction v2", "",
        f"Decision: **{summary['decision']['decision']}**.", "",
        "The real replay is leave-one-family-out: no target-family labels enter the donor "
        "head. Active and uniform arms use identical target-label budgets.", "",
        "## Registered gates", "",
        "| gate | observed | rule | result |", "|---|---:|---:|:---:|",
    ]
    for row in summary["decision"]["gates"]:
        lines.append(
            f"| `{row['gate']}` | {row['observed']:.6f} | "
            f"{row['operator']} {row['threshold']:.6f} | "
            f"**{'PASS' if row['pass'] else 'FAIL'}** |"
        )

    lines.extend([
        "", "## Real-cell learning curves", "",
        "Cell-macro held-out AUROC after averaging split repetitions.", "",
        "| target labels | U-PCR | local uniform | local active | pooled LOFO | "
        "hybrid uniform | hybrid active |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    shown = (
        "upcr", "local_uniform2", "local_active2", "pooled_domain_lofo",
        "hybrid_domain_uniform", "hybrid_domain_active",
    )
    for budget in BUDGETS:
        values = [curve_value(curves, "real", "real", budget, method) for method in shown]
        lines.append(
            f"| {budget} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
        )

    lines.extend([
        "", "## Mechanism decomposition at 20 target labels", "",
        "| mechanism | contrast | mean [95% CI] | W/L |", "|---|---|---:|---:|",
    ])
    mechanism_pairs = (
        ("local acquisition", "local_uniform2", "local_active2"),
        ("LOFO transfer", "local_uniform2", "hybrid_domain_uniform"),
        ("hybrid acquisition", "hybrid_domain_uniform", "hybrid_domain_active"),
        ("combined candidate", "upcr", "hybrid_domain_active"),
        ("pooled, no target labels", "upcr", "pooled_domain_lofo"),
        ("all-domain pooling", "upcr", "pooled_all_lofo"),
    )
    for label, reference, candidate in mechanism_pairs:
        row = lookup(contrasts, "real", "real", "all", 20, reference, candidate)
        lines.append(
            f"| {label} | `{reference}` -> `{candidate}` | "
            f"{100*row['mean_delta']:+.2f}pp "
            f"[{100*row['ci95_low']:+.2f}, {100*row['ci95_high']:+.2f}] | "
            f"{row['wins']}/{row['losses']} |"
        )

    lines.extend([
        "", "## Synthetic transfer boundary at 20 target labels", "",
        "| world | pooled LOFO vs U-PCR | hybrid active vs U-PCR | active vs uniform |",
        "|---|---:|---:|---:|",
    ])
    for world in SYNTHETIC_WORLDS:
        pooled = lookup(
            contrasts, "synthetic", world, "all", 20,
            "upcr", "pooled_domain_lofo",
        )
        hybrid = lookup(
            contrasts, "synthetic", world, "all", 20,
            "upcr", "hybrid_domain_active",
        )
        active = lookup(
            contrasts, "synthetic", world, "all", 20,
            "hybrid_domain_uniform", "hybrid_domain_active",
        )
        lines.append(
            f"| `{world}` | {100*pooled['mean_delta']:+.2f}pp | "
            f"{100*hybrid['mean_delta']:+.2f}pp | {100*active['mean_delta']:+.2f}pp |"
        )

    acquisition20 = [
        row for row in acquisition
        if row["source"] == "real" and row["budget"] == 20
    ]
    lines.extend([
        "", "## Acquisition validity at 20 labels", "",
        "| policy | selected sets containing both classes |", "|---|---:|",
    ])
    for row in acquisition20:
        lines.append(
            f"| `{row['method']}` | {100*row['both_classes_fraction']:.1f}% |"
        )

    lines.extend([
        "", "## Protocol notes", "",
        f"- Feature schema: `{summary['feature_schema']}`; common named features: "
        f"{summary['n_common_real_features']}.",
        f"- Real bundle validity macro: `{summary['real_bundle_validity']:.12f}`.",
        f"- Repetitions: {summary['real_repeats']} real and "
        f"{summary['synthetic_repeats']} per synthetic meta-world.",
        f"- Donor acquisition: {DONOR_LABELS_PER_CELL} uniform labels per eligible donor cell.",
        "- Donor cost is historical supervision and is not equal to the target-only label "
        "budget; only active-vs-uniform contrasts are equal-cost acquisition comparisons.",
        "- Test labels are read only after every target score is frozen.", "",
    ])
    return "\n".join(lines)


def run(args):
    started = time.time()
    os.makedirs(args.out, exist_ok=True)
    real_cells = v1.load_real_cells(args.bundle, args.manifest)
    validity = v1.validate_real_bundle(real_cells)
    real_repeats = 3 if args.quick else REAL_REPEATS
    synthetic_repeats = 3 if args.quick else SYNTHETIC_REPEATS

    real_rows, common_real = run_collection(
        "real", "real", real_cells, real_repeats,
    )
    rows = list(real_rows)
    synthetic_common = {}
    for world in SYNTHETIC_WORLDS:
        world_rows = []
        for repetition in range(synthetic_repeats):
            cells = synthetic_cells(world, repetition)
            # One collection repetition is enough because data generation itself
            # is indexed by the outer repetition.  Rewrite its internal index so
            # independent worlds remain visible to aggregation.
            generated, common = run_collection(
                "synthetic", world, cells, 1,
            )
            for row in generated:
                row["repetition"] = repetition
            world_rows.extend(generated)
            synthetic_common[world] = common
        rows.extend(world_rows)

    curves = learning_curves(rows)
    contrasts = all_contrasts(rows)
    acquisition = acquisition_summary(rows)
    convergence_rows = convergence(real_rows, real_repeats)
    eligible = not args.quick and real_repeats == REAL_REPEATS \
        and synthetic_repeats == SYNTHETIC_REPEATS
    decision = build_decision(contrasts, eligible)
    summary = {
        "version": VERSION,
        "feature_schema": SCHEMA_VERSION,
        "real_bundle_validity": validity,
        "n_real_cells": len(real_cells),
        "n_common_real_features": len(common_real),
        "common_real_features": list(common_real),
        "synthetic_worlds": list(SYNTHETIC_WORLDS),
        "budgets": list(BUDGETS),
        "methods": list(METHODS),
        "donor_labels_per_cell": DONOR_LABELS_PER_CELL,
        "real_repeats": real_repeats,
        "synthetic_repeats": synthetic_repeats,
        "runtime_seconds": float(time.time() - started),
        "decision": decision,
    }
    write_csv(os.path.join(args.out, "replicates.csv"), rows)
    write_csv(os.path.join(args.out, "learning_curves.csv"), curves)
    write_csv(os.path.join(args.out, "contrasts.csv"), contrasts)
    write_csv(os.path.join(args.out, "acquisition.csv"), acquisition)
    write_csv(os.path.join(args.out, "convergence.csv"), convergence_rows)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, indent=2, sort_keys=True)
    report = render_report(summary, curves, contrasts, acquisition)
    with open(os.path.join(args.out, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)
    print(f"\nRuntime: {summary['runtime_seconds']:.1f}s")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=v1.DEFAULT_BUNDLE)
    parser.add_argument("--manifest", default=v1.DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
