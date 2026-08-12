#!/usr/bin/env python3
"""Retrospective supervised PoC for IU-PCR contribution-space correction.

The script uses the existing 24-cell mixed-v2 bundle.  It never changes the
feature pool and never uses a second model pass.  Correctness labels are used
only inside each split's training partition and for held-out evaluation.

This is an exploratory proof of feasibility, not a label-free result and not a
prospective generalization claim.  Its output defines the supervised teacher
that a later unsupervised or self-supervised fusion component would need to
approximate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    DEFAULT_BUNDLE,
    family,
    load_contract,
)
from scripts.inscope_cells import INSCOPE  # noqa: E402
from spectral_utils.contribution_subspace import (  # noqa: E402
    fit_anchored_contribution_head,
    fit_contribution_transform,
    iu_family_contributions,
)
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402


VERSION = "harp-contribution-subspace-poc-v1-2026-08-12"
DEFAULT_OUT = REPO / "results" / "harp_contribution_subspace_poc_v1"
N_SPLITS = 30
BUDGETS = (20, 40, 80, "all")
PRIOR_STRENGTHS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
PRIMARY_BUDGET = "all"
PRIMARY_PRIOR = 0.3
MIN_POSITIVES = 20
RIDGE_C = 0.1


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def standardize(values, training_indices):
    values = np.asarray(values, dtype=float)
    mean = np.mean(values[training_indices], axis=0)
    scale = np.std(values[training_indices], axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (values - mean) / scale


def controlled_budget(training_indices, labels, budget, seed):
    """Approximately prevalence-preserving labelled subset.

    This intentionally uses labels to keep both classes represented.  It is an
    optimistic label-efficiency diagnostic, not a deployable acquisition rule.
    The full-training primary does not use this helper.
    """
    indices = np.asarray(training_indices, dtype=int)
    if budget == "all" or int(budget) >= len(indices):
        return indices
    budget = int(budget)
    rng = np.random.default_rng(7919 * budget + int(seed))
    chosen = []
    for class_value in (0, 1):
        available = indices[labels[indices] == class_value]
        count = max(1, int(round(budget * len(available) / len(indices))))
        chosen.extend(
            rng.choice(available, min(count, len(available)), replace=False)
        )
    chosen = np.asarray(sorted(set(chosen)), dtype=int)
    if len(chosen) > budget:
        chosen = rng.choice(chosen, budget, replace=False)
    elif len(chosen) < budget:
        remaining = np.setdiff1d(indices, chosen)
        extra = rng.choice(
            remaining, min(budget - len(chosen), len(remaining)), replace=False
        )
        chosen = np.concatenate([chosen, extra])
    return np.sort(chosen)


def fixed_ridge_scores(X, labels, training_indices):
    if len(np.unique(labels[training_indices])) < 2:
        return np.zeros(len(labels), dtype=float), True
    model = LogisticRegression(
        C=RIDGE_C,
        penalty="l2",
        fit_intercept=True,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        tol=1e-8,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(X[training_indices], labels[training_indices])
    return model.decision_function(X), False


def fit_cell(F, names, labels, cell):
    baseline = upcr_fit(F, **IU_FIT_DEFAULTS)
    space = iu_family_contributions(F, names, baseline.w)
    rows = []
    for seed in range(N_SPLITS):
        training, evaluation = train_test_split(
            np.arange(len(labels)),
            test_size=0.40,
            random_state=seed,
            stratify=labels,
        )
        training = np.sort(training)
        evaluation = np.sort(evaluation)
        transform = fit_contribution_transform(space, training)
        iu_score, residuals = transform.apply(
            space.baseline_score, space.contributions
        )
        full_features = standardize(F.T, training)
        family_design = np.column_stack([iu_score, residuals])
        iu_auc = float(roc_auc_score(labels[evaluation], iu_score[evaluation]))

        for budget in BUDGETS:
            labelled = controlled_budget(training, labels, budget, seed)
            common = {
                "version": VERSION,
                "cell": cell,
                "family": family(cell),
                "seed": seed,
                "budget": budget,
                "n": len(labels),
                "n_positive": int(labels.sum()),
                "n_training": len(training),
                "n_labelled": len(labelled),
                "n_evaluation": len(evaluation),
                "n_features": F.shape[0],
                "n_families": len(space.families),
                "iu_auroc": iu_auc,
            }
            family_score, family_fallback = fixed_ridge_scores(
                family_design, labels, labelled
            )
            full_score, full_fallback = fixed_ridge_scores(
                full_features, labels, labelled
            )
            row = {
                **common,
                "family_ridge_auroc": float(roc_auc_score(
                    labels[evaluation], family_score[evaluation]
                )),
                "full_ridge_auroc": float(roc_auc_score(
                    labels[evaluation], full_score[evaluation]
                )),
                "family_ridge_fallback": family_fallback,
                "full_ridge_fallback": full_fallback,
            }
            for strength in PRIOR_STRENGTHS:
                head = fit_anchored_contribution_head(
                    space,
                    labels,
                    labelled,
                    transform_indices=training,
                    prior_strength=strength,
                )
                score = head.score(space.baseline_score, space.contributions)
                key = f"anchored_{strength:g}"
                row[f"{key}_auroc"] = float(roc_auc_score(
                    labels[evaluation], score[evaluation]
                ))
                row[f"{key}_delta_norm"] = head.diagnostics["delta_norm"]
                for family_name, coefficient in zip(
                    space.families, head.delta
                ):
                    row[f"{key}_delta__{family_name}"] = float(coefficient)
            rows.append(row)
    return rows, space


def aggregate_cell_means(rows):
    methods = ["iu", "family_ridge", "full_ridge"] + [
        f"anchored_{strength:g}" for strength in PRIOR_STRENGTHS
    ]
    output = []
    keys = sorted({(row["cell"], str(row["budget"])) for row in rows})
    for cell, budget in keys:
        selected = [
            row for row in rows
            if row["cell"] == cell and str(row["budget"]) == budget
        ]
        item = {
            "version": VERSION,
            "cell": cell,
            "family": selected[0]["family"],
            "budget": budget,
            "n_splits": len(selected),
        }
        for method in methods:
            item[f"{method}_auroc"] = float(np.mean([
                row[f"{method}_auroc"] for row in selected
            ]))
        output.append(item)
    return output


def equal_family_bootstrap(cell_rows, method, namespace, count=20000):
    families = sorted({row["family"] for row in cell_rows})
    family_delta = np.asarray([
        np.mean([
            row[f"{method}_auroc"] - row["iu_auroc"]
            for row in cell_rows if row["family"] == item
        ])
        for item in families
    ])
    seed = int(hashlib.sha256(namespace.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    draws = family_delta[
        rng.integers(0, len(family_delta), size=(count, len(family_delta)))
    ].mean(axis=1)
    return (
        float(np.mean(family_delta)),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def summarize(cell_means):
    methods = ["family_ridge", "full_ridge"] + [
        f"anchored_{strength:g}" for strength in PRIOR_STRENGTHS
    ]
    rows = []
    for budget in BUDGETS:
        selected = [
            row for row in cell_means if str(row["budget"]) == str(budget)
        ]
        baseline = np.asarray([row["iu_auroc"] for row in selected])
        for method in methods:
            values = np.asarray([row[f"{method}_auroc"] for row in selected])
            delta, low, high = equal_family_bootstrap(
                selected, method, f"{VERSION}:{budget}:{method}"
            )
            rows.append({
                "version": VERSION,
                "budget": budget,
                "method": method,
                "n_cells": len(selected),
                "n_families": len({row["family"] for row in selected}),
                "cell_macro_auroc": float(np.mean(values)),
                "cell_macro_iu_auroc": float(np.mean(baseline)),
                "cell_macro_delta_pp": float(100 * np.mean(values - baseline)),
                "equal_family_delta_pp": float(100 * delta),
                "equal_family_ci_low_pp": float(100 * low),
                "equal_family_ci_high_pp": float(100 * high),
                "wins": int(np.sum(values > baseline + 1e-12)),
                "losses": int(np.sum(values < baseline - 1e-12)),
                "ties": int(np.sum(np.abs(values - baseline) <= 1e-12)),
                "worst_delta_pp": float(100 * np.min(values - baseline)),
            })
    return rows


def summarize_teacher_targets(rows, spaces):
    """Persist the cross-fitted supervised target for later premise tests."""
    method = f"anchored_{PRIMARY_PRIOR:g}"
    output = []
    for cell, space in spaces.items():
        selected = [
            row for row in rows
            if row["cell"] == cell
            and str(row["budget"]) == str(PRIMARY_BUDGET)
        ]
        deltas = np.asarray([
            [row[f"{method}_delta__{name}"] for name in space.families]
            for row in selected
        ], dtype=float)
        mean_delta = np.mean(deltas, axis=0)
        denominator = np.linalg.norm(deltas, axis=1) * np.linalg.norm(mean_delta)
        cosine = np.divide(
            deltas @ mean_delta,
            denominator,
            out=np.zeros(len(deltas), dtype=float),
            where=denominator > 1e-12,
        )
        for index, name in enumerate(space.families):
            values = deltas[:, index]
            output.append({
                "version": VERSION,
                "cell": cell,
                "family": family(cell),
                "contribution_family": name,
                "budget": PRIMARY_BUDGET,
                "prior_strength": PRIMARY_PRIOR,
                "n_splits": len(selected),
                "mean_delta": float(np.mean(values)),
                "std_delta": float(np.std(values)),
                "positive_fraction": float(np.mean(values > 0)),
                "mean_delta_norm": float(np.linalg.norm(mean_delta)),
                "mean_cosine_to_cell_mean": float(np.mean(cosine)),
            })
    return output


def render_report(summary, teacher_targets, exclusions, spaces):
    lookup = {(str(row["budget"]), row["method"]): row for row in summary}
    cell_teacher_cosine = {
        row["cell"]: row["mean_cosine_to_cell_mean"]
        for row in teacher_targets
    }
    primary = lookup[str(PRIMARY_BUDGET), f"anchored_{PRIMARY_PRIOR:g}"]
    unrestricted = lookup[str(PRIMARY_BUDGET), "family_ridge"]
    full = lookup[str(PRIMARY_BUDGET), "full_ridge"]
    lines = [
        "# HARP-inspired IU contribution-subspace PoC",
        "",
        "**Status:** retrospective supervised feasibility study; not a label-free method.",
        "",
        "## Main result",
        "",
        (
            f"The primary anchored family head (`prior={PRIMARY_PRIOR:g}`) "
            f"reached **{primary['cell_macro_auroc']:.4f}** cell-macro AUROC, "
            f"a **{primary['cell_macro_delta_pp']:+.3f}pp** change from IU-PCR "
            f"({primary['wins']}W/{primary['losses']}L; worst "
            f"{primary['worst_delta_pp']:+.3f}pp). Its equal-family interval is "
            f"[{primary['equal_family_ci_low_pp']:+.3f}, "
            f"{primary['equal_family_ci_high_pp']:+.3f}]pp."
        ),
        "",
        (
            f"The unrestricted family-space ridge changed IU-PCR by "
            f"{unrestricted['cell_macro_delta_pp']:+.3f}pp. Full-feature ridge "
            f"changed it by {full['cell_macro_delta_pp']:+.3f}pp; this control "
            "tests whether labels alone explain the teacher's gain."
        ),
        "",
        "## Method boundary",
        "",
        "For every sample, ordinary IU-PCR contributions are summed inside the frozen probability-telemetry provenance families. These family contributions reconstruct the IU score exactly. They are residualized against that score on the training partition. The supervised head learns only a small residual correction; zero correction returns the IU ranking exactly.",
        "",
        "No new feature, generation, hidden state, model weight, attention map, or white-box quantity enters the method. Correctness labels train the proof-of-concept teacher, so the result is supervised.",
        "",
        "## Results by label budget",
        "",
        "| budget | method | AUROC | delta vs IU | equal-family 95% interval | W/L | worst |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    shown = ["family_ridge", "full_ridge"] + [
        f"anchored_{strength:g}" for strength in PRIOR_STRENGTHS
    ]
    for budget in BUDGETS:
        for method in shown:
            row = lookup[str(budget), method]
            lines.append(
                f"| {budget} | `{method}` | {row['cell_macro_auroc']:.4f} | "
                f"{row['cell_macro_delta_pp']:+.3f}pp | "
                f"[{row['equal_family_ci_low_pp']:+.3f}, {row['equal_family_ci_high_pp']:+.3f}] | "
                f"{row['wins']}/{row['losses']} | {row['worst_delta_pp']:+.3f}pp |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A low-dimensional supervised correction exists in IU-PCR's own contribution space and generalizes to held-out samples from the same cell. This does not establish transfer to unseen dataset families. The next premise test must ask whether unlabeled, cell-local statistics predict the supervised correction under leave-one-family-out evaluation. A fixed cross-family correction and another graph regularizer are not justified by the existing evidence.",
        "",
        "The 20/40/80 budgets use a label-aware prevalence-preserving acquisition diagnostic. They are optimistic and may not be described as deployable active or semi-supervised results.",
        "",
        (
            "Across the primary cross-fitted teachers, the median per-cell "
            "cosine to that cell's mean correction was "
            f"{np.median(list(cell_teacher_cosine.values())):.3f}. "
            "This is a within-cell stability diagnostic, not evidence that one "
            "correction transfers across cells."
        ),
        "",
        "## Exclusions and audit",
        "",
        f"Cells below {MIN_POSITIVES} positive examples were excluded from the PoC: `{', '.join(exclusions)}`.",
        f"All {len(spaces)} evaluated cells reconstructed their IU score from family contributions; maximum reconstruction error was {max(space.diagnostics['reconstruction_error'] for space in spaces.values()):.3e}.",
        "",
        "Primary artifacts: `replicates.csv`, `cell_means.csv`, `summary.csv`, `teacher_targets.csv`, and `RUN_DEFINITION.json`.",
    ])
    return "\n".join(lines) + "\n"


def main():
    bundle = Path(os.environ.get("HARP_POC_BUNDLE", str(DEFAULT_BUNDLE)))
    out = Path(os.environ.get("HARP_POC_OUT", str(DEFAULT_OUT)))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    exclusions = []
    spaces = {}
    with np.load(bundle, allow_pickle=True) as data:
        for index, cell in enumerate(INSCOPE, start=1):
            F, names = load_contract(data, cell, "mixed_v2")
            labels = np.asarray(data[f"{cell}__labels"], dtype=int)
            if int(labels.sum()) < MIN_POSITIVES:
                exclusions.append(cell)
                print(f"[{index:02d}/24] {cell}: excluded ({labels.sum()} positives)")
                continue
            cell_rows, space = fit_cell(F, names, labels, cell)
            rows.extend(cell_rows)
            spaces[cell] = space
            print(f"[{index:02d}/24] {cell}: complete", flush=True)

    cell_means = aggregate_cell_means(rows)
    summary = summarize(cell_means)
    teacher_targets = summarize_teacher_targets(rows, spaces)
    write_csv(out / "replicates.csv", rows)
    write_csv(out / "cell_means.csv", cell_means)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "teacher_targets.csv", teacher_targets)
    definition = {
        "version": VERSION,
        "status": "retrospective_supervised_proof_of_concept",
        "bundle": str(bundle.relative_to(REPO)),
        "bundle_sha256": sha256_file(bundle),
        "cells": list(spaces),
        "excluded_cells": exclusions,
        "exclusion_rule": f"n_positive < {MIN_POSITIVES}",
        "feature_contract": "dufs-liu-mixed-v2-development-2026-08-07",
        "n_splits": N_SPLITS,
        "split": "60/40 stratified within cell",
        "budgets": list(BUDGETS),
        "budget_acquisition": "label-aware prevalence-preserving diagnostic",
        "prior_strengths": list(PRIOR_STRENGTHS),
        "primary_budget": PRIMARY_BUDGET,
        "primary_prior": PRIMARY_PRIOR,
        "ridge_C": RIDGE_C,
        "allowed_inputs": "same one-pass mixed-v2 feature matrix only",
        "labels_used": True,
        "claim_boundary": "supervised within-cell held-out-sample feasibility only",
        "source_sha256": {
            "script": sha256_file(Path(__file__)),
            "module": sha256_file(REPO / "spectral_utils" / "contribution_subspace.py"),
            "test": sha256_file(REPO / "scripts" / "test_contribution_subspace.py"),
        },
    }
    write_json(out / "RUN_DEFINITION.json", definition)
    (out / "REPORT.md").write_text(
        render_report(summary, teacher_targets, exclusions, spaces), encoding="utf-8"
    )
    primary = next(
        row for row in summary
        if str(row["budget"]) == str(PRIMARY_BUDGET)
        and row["method"] == f"anchored_{PRIMARY_PRIOR:g}"
    )
    print(json.dumps(primary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
