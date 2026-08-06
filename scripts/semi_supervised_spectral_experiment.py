#!/usr/bin/env python3
"""Label-budget study for U-PCR-anchored semi-supervised fusion.

The confirmatory protocol is frozen in ``SPEC_SEMISUPERVISED_SPECTRAL_V1.md``.
Synthetic mechanism worlds are run first; the existing 24-cell bundle is then
used for a retrospective real-artifact replay.  Every split is inductive: test
features and labels are withheld from standardisation, U-PCR, PCA, pseudo-label
construction, and labelled-head fitting.

Usage:
    python scripts/semi_supervised_spectral_experiment.py
    python scripts/semi_supervised_spectral_experiment.py --quick
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
from sklearn.model_selection import train_test_split


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
# Avoid importing the heavyweight package facade; the experiment needs only
# NumPy/SciPy/scikit-learn modules.
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.feature_contract import (                         # noqa: E402
    LEGACY_FEATURE_SIGNS,
    SCHEMA_VERSION,
    confidence_oriented_matrix,
    consensus_anchor,
)
from spectral_utils.semi_supervised_fusion import (                   # noqa: E402
    fit_logistic_head,
    fit_soft_logistic_head,
    orient_weight,
    pca_score_basis,
    spectral_score_basis,
    standardize_train_test,
)
from spectral_utils.upcr import upcr_fit                               # noqa: E402


VERSION = "semi-supervised-spectral-v1-2026-08-06"
DEFAULT_BUNDLE = os.path.join(REPO, "results", "dependency_fusion_raw", "cells.npz")
DEFAULT_MANIFEST = os.path.join(
    REPO, "results", "dependency_fusion_raw", "cells_manifest.csv",
)
DEFAULT_OUT = os.path.join(REPO, "results", "semi_supervised_spectral_v1")
BUDGETS = (0, 5, 10, 20, 40, 80)
REAL_REPEATS = 30
SYNTHETIC_REPEATS = 40
N_BOOT = 10000
EXPECTED_FIXED_STABLE_UPCR = 0.7735279028911624

UPCR_FIT = dict(
    loss="l2", exclusion=True, difficulty_gate=False,
    simple_avg_fallback=True, recompute_after_exclusion=True,
    g2_projection_k=1, scale_ratio=0.25,
)

METHODS = (
    "upcr",
    "platt_upcr",
    "gold_pcr2",
    "gold_pcr6",
    "gold_ridge_all",
    "anchored_pcr2",
    "anchored_pcr6",
    "pseudo_gold_pcr6",
)

SYNTHETIC_WORLDS = (
    "independent",
    "grouped",
    "sparse_pairs",
    "correlated_weak_block",
)

KNOWN_FAMILIES = (
    "triviaqa", "coqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
    "truthfulqa", "gsm8k", "math500", "gpqa", "webq", "humaneval",
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


def family(cell):
    return next((name for name in KNOWN_FAMILIES if name in cell), cell)


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["cell"]: row for row in csv.DictReader(handle)}


def load_real_cells(bundle_path, manifest_path):
    data = np.load(bundle_path, allow_pickle=True)
    manifest = load_manifest(manifest_path)
    keys = sorted({name.rsplit("__", 1)[0] for name in data.files})
    cells = {}
    for key in keys:
        names = [str(value) for value in data[f"{key}__pool"]]
        legacy = np.asarray(data[f"{key}__hand_signs"], dtype=float)
        expected = np.asarray([LEGACY_FEATURE_SIGNS[name] for name in names], dtype=float)
        if not np.array_equal(legacy, expected):
            raise RuntimeError(f"{key}: legacy sign vector no longer reconstructs raw views")
        raw = np.asarray(data[f"{key}__V"], dtype=float) * legacy
        matrix, kept, _ = confidence_oriented_matrix(raw, names, stable=True)
        labels = np.asarray(data[f"{key}__labels"], dtype=int)
        cells[key] = {
            "matrix": matrix,
            "labels": labels,
            "features": kept,
            "domain": manifest[key]["domain"],
            "family": family(key),
        }
    return cells


def validate_real_bundle(cells):
    aucs = []
    for cell in cells.values():
        matrix = np.asarray(cell["matrix"], dtype=float)
        weight = upcr_fit(matrix.T, **UPCR_FIT).w
        weight = orient_weight(weight, matrix, consensus_anchor(matrix))
        aucs.append(roc_auc_score(cell["labels"], matrix @ weight))
    macro = float(np.mean(aucs))
    if abs(macro - EXPECTED_FIXED_STABLE_UPCR) > 1e-9:
        raise RuntimeError(
            f"real bundle validity failed: fixed-stable U-PCR={macro:.12f}, "
            f"expected {EXPECTED_FIXED_STABLE_UPCR:.12f}"
        )
    return macro


def synthetic_matrix(world, repetition, n=1800, m=18):
    rng = np.random.default_rng(stable_seed(VERSION, "synthetic", world, repetition))
    labels = rng.binomial(1, 0.5, size=n).astype(int)
    target = 2.0 * labels - 1.0
    noise = rng.normal(size=(n, m))

    if world == "independent":
        strength = np.linspace(0.18, 0.52, m)
        matrix = target[:, None] * strength + noise
    elif world == "grouped":
        strength = np.repeat(np.array([0.16, 0.34, 0.56]), 6)
        groups = rng.normal(size=(n, 3))
        matrix = target[:, None] * strength
        for group_index in range(3):
            cols = slice(6 * group_index, 6 * (group_index + 1))
            matrix[:, cols] += 0.95 * groups[:, [group_index]] + 0.55 * noise[:, cols]
    elif world == "sparse_pairs":
        strength = np.linspace(0.18, 0.52, m)
        matrix = target[:, None] * strength + noise
        for left, right, magnitude in (
            (0, 9, 1.2), (2, 12, -1.1), (5, 15, 1.3), (7, 17, -1.2),
        ):
            shared = rng.normal(size=n)
            matrix[:, left] += magnitude * shared
            matrix[:, right] += shared
    elif world == "correlated_weak_block":
        strength = np.concatenate([
            np.full(10, 0.12), np.full(4, 0.42), np.full(4, 0.62),
        ])
        shared = rng.normal(size=n)
        matrix = target[:, None] * strength + 0.65 * noise
        matrix[:, :10] += 1.35 * shared[:, None]
    else:
        raise ValueError(world)
    return matrix, labels


def split_and_standardize(matrix, labels, unit, repetition):
    seed = stable_seed(VERSION, "split", unit, repetition)
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.40, random_state=seed, stratify=labels,
    )
    train, test, _, _ = standardize_train_test(matrix[train_idx], matrix[test_idx])
    return train, test, labels[train_idx], labels[test_idx]


def stratified_label_indices(labels, budget, unit, repetition):
    labels = np.asarray(labels, dtype=int)
    if budget <= 0:
        return np.asarray([], dtype=int)
    budget = min(int(budget), len(labels))
    rng = np.random.default_rng(stable_seed(VERSION, "trusted", unit, repetition, budget))
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(positive) == 0 or len(negative) == 0:
        return np.sort(rng.choice(len(labels), size=budget, replace=False))

    target_positive = int(round(budget * len(positive) / len(labels)))
    target_positive = min(max(1, target_positive), len(positive), budget - 1)
    target_negative = min(budget - target_positive, len(negative))
    if target_positive + target_negative < budget:
        target_positive = min(len(positive), budget - target_negative)
    chosen = np.concatenate([
        rng.choice(positive, size=target_positive, replace=False),
        rng.choice(negative, size=target_negative, replace=False),
    ])
    return np.sort(chosen)


def _score_auc(labels, matrix, weight, intercept=0.0):
    score = float(intercept) + np.asarray(matrix, dtype=float) @ np.asarray(weight, dtype=float)
    if not np.isfinite(score).all() or score.std() < 1e-12:
        raise RuntimeError("non-finite or constant held-out score")
    return float(roc_auc_score(labels, score))


def run_split(source, unit, group, domain, matrix, labels, repetition):
    train, test, y_train, y_test = split_and_standardize(matrix, labels, unit, repetition)
    anchor = consensus_anchor(train)
    upcr = upcr_fit(train.T, **UPCR_FIT)
    upcr_weight = orient_weight(upcr.w, train, anchor)
    anchored2 = spectral_score_basis(train, upcr_weight, rank=2)
    anchored6 = spectral_score_basis(train, upcr_weight, rank=6)
    pca2 = pca_score_basis(train, rank=2)
    pca6 = pca_score_basis(train, rank=6)
    identity = np.eye(train.shape[1])
    baseline_auc = _score_auc(y_test, test, upcr_weight)

    base = {
        "source": source,
        "unit": unit,
        "group": group,
        "domain": domain,
        "repetition": repetition,
        "n_train": len(train),
        "n_test": len(test),
        "n_features": train.shape[1],
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "upcr_n_kept": int(upcr.keep.sum()),
    }
    rows = []

    def add(budget, method, auc, n_trusted, converged=True, n_iter=0):
        rows.append({
            **base,
            "budget": int(budget),
            "n_trusted": int(n_trusted),
            "method": method,
            "auc": float(auc),
            "delta_vs_upcr": float(auc - baseline_auc),
            "converged": int(bool(converged)),
            "n_iter": int(n_iter),
        })

    for budget in BUDGETS:
        trusted = stratified_label_indices(y_train, budget, unit, repetition)
        if budget == 0:
            for method in ("upcr", "platt_upcr", "anchored_pcr2", "anchored_pcr6",
                           "pseudo_gold_pcr6"):
                add(budget, method, baseline_auc, 0)
            continue

        labelled_x, labelled_y = train[trusted], y_train[trusted]
        add(budget, "upcr", baseline_auc, len(trusted))
        # Positive one-dimensional calibration cannot change AUROC.  Keeping the
        # score exactly equal makes this a strict harness invariant.
        add(budget, "platt_upcr", baseline_auc, len(trusted))

        fits = {
            "gold_pcr2": fit_logistic_head(
                labelled_x, labelled_y, pca2, np.zeros(pca2.shape[1]),
                prior_strength=1.0,
            ),
            "gold_pcr6": fit_logistic_head(
                labelled_x, labelled_y, pca6, np.zeros(pca6.shape[1]),
                prior_strength=1.0,
            ),
            "gold_ridge_all": fit_logistic_head(
                labelled_x, labelled_y, identity, np.zeros(identity.shape[1]),
                prior_strength=1.0,
            ),
            "anchored_pcr2": fit_logistic_head(
                labelled_x, labelled_y, anchored2,
                np.r_[1.0, np.zeros(anchored2.shape[1] - 1)],
                prior_strength=10.0,
            ),
            "anchored_pcr6": fit_logistic_head(
                labelled_x, labelled_y, anchored6,
                np.r_[1.0, np.zeros(anchored6.shape[1] - 1)],
                prior_strength=10.0,
            ),
        }
        for method, fit in fits.items():
            add(
                budget, method,
                _score_auc(y_test, test, fit.weight, fit.intercept),
                len(trusted), fit.converged, fit.n_iter,
            )

        remaining_mask = np.ones(len(train), dtype=bool)
        remaining_mask[trusted] = False
        pseudo_matrix = train[remaining_mask]
        teacher_score = pseudo_matrix @ anchored6[:, 0]
        teacher_probability = np.empty_like(teacher_score)
        positive = teacher_score >= 0
        teacher_probability[positive] = 1.0 / (1.0 + np.exp(-teacher_score[positive]))
        exp_score = np.exp(teacher_score[~positive])
        teacher_probability[~positive] = exp_score / (1.0 + exp_score)
        pseudo_fit = fit_soft_logistic_head(
            labelled_x, labelled_y, pseudo_matrix, teacher_probability,
            anchored6, np.zeros(anchored6.shape[1]),
            prior_strength=0.1, soft_total_weight=10.0,
        )
        add(
            budget, "pseudo_gold_pcr6",
            _score_auc(y_test, test, pseudo_fit.weight, pseudo_fit.intercept),
            len(trusted), pseudo_fit.converged, pseudo_fit.n_iter,
        )
    return rows


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


def unit_means(rows, source, budget, method, group=None):
    selected = [row for row in rows if row["source"] == source
                and row["budget"] == budget and row["method"] == method]
    if group is not None:
        selected = [row for row in selected if row["group"] == group]
    # Real uncertainty is across cells after averaging split repetitions.
    # Synthetic uncertainty is across independent repetitions within a planted
    # world; treating the world name itself as the unit would collapse its CI
    # to a single point.
    unit_key = (lambda row: row["repetition"]) if source == "synthetic" \
        else (lambda row: row["unit"])
    units = sorted({unit_key(row) for row in selected})
    return {
        unit: float(np.mean([row["auc"] for row in selected if unit_key(row) == unit]))
        for unit in units
    }


def learning_curves(rows):
    output = []
    panels = [("real", "all"), ("real", "QA"), ("real", "math")]
    panels += [("synthetic", world) for world in SYNTHETIC_WORLDS]
    for source, panel in panels:
        group = None if panel == "all" else panel
        for budget in BUDGETS:
            for method in METHODS:
                means = unit_means(rows, source, budget, method, group=group)
                if not means:
                    continue
                values = np.asarray(list(means.values()))
                lo, hi = bootstrap_ci(values, f"curve_{source}_{panel}_{budget}_{method}")
                output.append({
                    "source": source,
                    "panel": panel,
                    "budget": budget,
                    "method": method,
                    "n_units": len(values),
                    "mean_auc": float(values.mean()),
                    "ci95_low": lo,
                    "ci95_high": hi,
                })
    return output


def paired_contrast(rows, source, budget, reference, candidate, group=None):
    ref = unit_means(rows, source, budget, reference, group=group)
    cand = unit_means(rows, source, budget, candidate, group=group)
    units = sorted(set(ref) & set(cand))
    delta = np.asarray([cand[unit] - ref[unit] for unit in units], dtype=float)
    lo, hi = bootstrap_ci(
        delta, f"contrast_{source}_{group}_{budget}_{candidate}_{reference}",
    )
    return {
        "source": source,
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


def all_contrasts(rows):
    output = []
    comparisons = (
        ("upcr", "anchored_pcr2"),
        ("upcr", "anchored_pcr6"),
        ("gold_ridge_all", "anchored_pcr6"),
        ("anchored_pcr6", "pseudo_gold_pcr6"),
        ("gold_pcr6", "anchored_pcr6"),
    )
    for budget in BUDGETS[1:]:
        for reference, candidate in comparisons:
            output.append(paired_contrast(rows, "real", budget, reference, candidate))
            for domain in ("QA", "math"):
                output.append(paired_contrast(
                    rows, "real", budget, reference, candidate, group=domain,
                ))
            for world in SYNTHETIC_WORLDS:
                output.append(paired_contrast(
                    rows, "synthetic", budget, reference, candidate, group=world,
                ))
    return output


def lookup_contrast(contrasts, source, group, budget, reference, candidate):
    matches = [row for row in contrasts if row["source"] == source
               and row["group"] == group and row["budget"] == budget
               and row["reference"] == reference and row["candidate"] == candidate]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one contrast {source}/{group}/{budget}/{reference}/{candidate}, "
            f"got {len(matches)}"
        )
    return matches[0]


def build_decision(rows, contrasts, eligible):
    platt_error = max(abs(row["delta_vs_upcr"]) for row in rows
                      if row["method"] == "platt_upcr")
    real_upcr = lookup_contrast(
        contrasts, "real", "all", 20, "upcr", "anchored_pcr6",
    )
    real_ridge = lookup_contrast(
        contrasts, "real", "all", 20, "gold_ridge_all", "anchored_pcr6",
    )
    qa = lookup_contrast(contrasts, "real", "QA", 20, "upcr", "anchored_pcr6")
    math = lookup_contrast(contrasts, "real", "math", 20, "upcr", "anchored_pcr6")
    synthetic = {
        world: {
            "vs_upcr": lookup_contrast(
                contrasts, "synthetic", world, 20, "upcr", "anchored_pcr6",
            ),
            "vs_ridge": lookup_contrast(
                contrasts, "synthetic", world, 20, "gold_ridge_all", "anchored_pcr6",
            ),
        } for world in SYNTHETIC_WORLDS
    }

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
            "gate": name, "observed": float(observed), "operator": operator,
            "threshold": float(threshold), "pass": bool(passed),
        })

    gate("platt_ranking_invariant", platt_error, "<=", 1e-10)
    gate("real_mean_vs_upcr", real_upcr["mean_delta"], ">=", 0.010)
    gate("real_ci_low_vs_upcr", real_upcr["ci95_low"], ">", 0.0)
    gate("real_mean_vs_gold_ridge", real_ridge["mean_delta"], ">=", 0.0)
    gate("real_qa_vs_upcr", qa["mean_delta"], ">=", -0.005)
    gate("real_math_vs_upcr", math["mean_delta"], ">=", -0.005)
    gate(
        "real_catastrophic_losses",
        real_upcr["catastrophic_losses_5pp"], "<=", 2,
    )
    gate(
        "synthetic_grouped_vs_upcr",
        synthetic["grouped"]["vs_upcr"]["mean_delta"], ">", 0.0,
    )
    gate(
        "synthetic_weak_block_vs_upcr",
        synthetic["correlated_weak_block"]["vs_upcr"]["mean_delta"], ">", 0.0,
    )
    ridge_wins = sum(
        synthetic[world]["vs_ridge"]["mean_delta"] > 0 for world in SYNTHETIC_WORLDS
    )
    gate("synthetic_worlds_beating_ridge", ridge_wins, ">=", 3)
    gate(
        "synthetic_independent_not_harmful",
        synthetic["independent"]["vs_upcr"]["mean_delta"], ">=", -0.005,
    )
    passed = bool(eligible and all(row["pass"] for row in gates))
    return {
        "eligible_confirmatory_run": bool(eligible),
        "decision": (
            "PROMOTE_FOR_PROSPECTIVE_REPLAY" if passed else
            ("NOT_A_CONFIRMATORY_RUN" if not eligible else "STOP_AND_REVISE")
        ),
        "all_gates_pass": passed,
        "gates": gates,
    }


def convergence(rows, max_repeats):
    output = []
    selected_methods = ("upcr", "gold_ridge_all", "anchored_pcr6", "pseudo_gold_pcr6")
    for budget in (5, 20, 80):
        for upto in range(1, max_repeats + 1):
            subset = [row for row in rows if row["source"] == "real"
                      and row["budget"] == budget and row["repetition"] < upto]
            for method in selected_methods:
                means = unit_means(subset, "real", budget, method)
                if not means:
                    continue
                upcr = unit_means(subset, "real", budget, "upcr")
                common = sorted(set(means) & set(upcr))
                delta = np.mean([means[unit] - upcr[unit] for unit in common])
                output.append({
                    "budget": budget,
                    "repeat_count": upto,
                    "method": method,
                    "mean_auc": float(np.mean(list(means.values()))),
                    "mean_delta_vs_upcr": float(delta),
                })
    return output


def render_report(summary, curves, contrasts):
    decision = summary["decision"]["decision"]
    lines = [
        "# Semi-supervised spectral fusion v1", "",
        f"Decision: **{decision}**.", "",
        "This is a retrospective label-budget replay on the existing 24-cell feature "
        "bundle plus a disjoint synthetic mechanism study. A pass would require a "
        "prospective dataset/model-family confirmation.", "",
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
        "", "## Real-cell learning curve", "",
        "Cell-macro held-out AUROC after averaging registered repetitions.", "",
        "| labels | U-PCR | ridge-all | anchored-2 | anchored-6 | pseudo+gold-6 |",
        "|---:|---:|---:|---:|---:|---:|",
    ])

    def curve_value(source, panel, budget, method):
        matches = [row for row in curves if row["source"] == source
                   and row["panel"] == panel and row["budget"] == budget
                   and row["method"] == method]
        return float("nan") if not matches else matches[0]["mean_auc"]

    for budget in BUDGETS:
        values = [curve_value("real", "all", budget, method) for method in (
            "upcr", "gold_ridge_all", "anchored_pcr2", "anchored_pcr6",
            "pseudo_gold_pcr6",
        )]
        lines.append(
            f"| {budget} | " + " | ".join(
                "—" if not np.isfinite(value) else f"{value:.4f}" for value in values
            ) + " |"
        )

    lines.extend([
        "", "## Primary 20-label contrasts", "",
        "| source/group | reference -> candidate | mean [95% CI] | W/L | <= -5pp |",
        "|---|---|---:|---:|---:|",
    ])
    selected = [row for row in contrasts if row["budget"] == 20
                and row["candidate"] == "anchored_pcr6"
                and row["reference"] in {"upcr", "gold_ridge_all"}]
    for row in selected:
        lines.append(
            f"| `{row['source']}/{row['group']}` | `{row['reference']}` -> "
            f"`anchored_pcr6` | {100*row['mean_delta']:+.2f}pp "
            f"[{100*row['ci95_low']:+.2f}, {100*row['ci95_high']:+.2f}] | "
            f"{row['wins']}/{row['losses']} | {row['catastrophic_losses_5pp']} |"
        )
    lines.extend([
        "", "## Protocol notes", "",
        f"- Feature schema: `{summary['feature_schema']}`.",
        f"- Real bundle validity macro: `{summary['real_bundle_validity']:.12f}`.",
        f"- Repetitions: {summary['real_repeats']} real, "
        f"{summary['synthetic_repeats']} synthetic.",
        "- Acquisition is controlled stratification, approximately preserving cell "
        "prevalence and forcing both classes. It is optimistic and is not an active "
        "label-acquisition result.",
        "- Test labels are read only after every score is frozen.", "",
    ])
    return "\n".join(lines)


def run(args):
    started = time.time()
    os.makedirs(args.out, exist_ok=True)
    real_cells = load_real_cells(args.bundle, args.manifest)
    validity = validate_real_bundle(real_cells)
    real_repeats = 3 if args.quick else REAL_REPEATS
    synthetic_repeats = 4 if args.quick else SYNTHETIC_REPEATS
    rows = []

    for world in SYNTHETIC_WORLDS:
        for repetition in range(synthetic_repeats):
            matrix, labels = synthetic_matrix(world, repetition)
            rows.extend(run_split(
                "synthetic", world, world, "synthetic", matrix, labels, repetition,
            ))

    for cell_name, cell in real_cells.items():
        for repetition in range(real_repeats):
            rows.extend(run_split(
                "real", cell_name, cell["domain"], cell["domain"],
                cell["matrix"], cell["labels"], repetition,
            ))

    curves = learning_curves(rows)
    contrasts = all_contrasts(rows)
    eligible = not args.quick and real_repeats == REAL_REPEATS \
        and synthetic_repeats == SYNTHETIC_REPEATS
    decision = build_decision(rows, contrasts, eligible)
    convergence_rows = convergence(rows, real_repeats)
    summary = {
        "version": VERSION,
        "feature_schema": SCHEMA_VERSION,
        "real_bundle_validity": validity,
        "n_real_cells": len(real_cells),
        "synthetic_worlds": list(SYNTHETIC_WORLDS),
        "budgets": list(BUDGETS),
        "methods": list(METHODS),
        "real_repeats": real_repeats,
        "synthetic_repeats": synthetic_repeats,
        "runtime_seconds": float(time.time() - started),
        "decision": decision,
    }
    write_csv(os.path.join(args.out, "replicates.csv"), rows)
    write_csv(os.path.join(args.out, "learning_curves.csv"), curves)
    write_csv(os.path.join(args.out, "contrasts.csv"), contrasts)
    write_csv(os.path.join(args.out, "convergence.csv"), convergence_rows)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, indent=2, sort_keys=True)
    report = render_report(summary, curves, contrasts)
    with open(os.path.join(args.out, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)
    print(f"\nRuntime: {summary['runtime_seconds']:.1f}s")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
