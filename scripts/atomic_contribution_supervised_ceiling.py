#!/usr/bin/env python3
"""Supervised diagnostic ceiling for atomic versus family contributions.

This is label-using research instrumentation, never a deployable candidate.
Metrics are averaged per split within cell and then equally across dataset
groups; held-out predictions are never concatenated into a global AUROC.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.atomic_nrm_retrospective_controls import load_original  # noqa: E402
from scripts.hard_filter_dufs_liu_benchmark import DEFAULT_BUNDLE  # noqa: E402
from spectral_utils.contribution_subspace import (  # noqa: E402
    fit_anchored_contribution_head,
)


VERSION = "atomic-contribution-supervised-ceiling-v1-2026-08-13"
DEFAULT_OUT = REPO / "results" / "atomic_contribution_supervised_ceiling_v1"
N_SPLITS = 30
PRIOR_STRENGTHS = (0.3, 1.0, 3.0, 10.0)
BOOTSTRAP_DRAWS = 20000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def replicate_rows(cells):
    rows = []
    for cell in cells:
        labels = cell["correctness"]
        indices = np.arange(len(labels))
        spaces = {
            "family": cell["space"],
            "atomic": cell["atomic_space"],
        }
        for seed in range(N_SPLITS):
            training, evaluation = train_test_split(
                indices,
                test_size=0.40,
                random_state=seed,
                stratify=labels,
            )
            training = np.sort(training)
            evaluation = np.sort(evaluation)
            for representation, space in spaces.items():
                for prior in PRIOR_STRENGTHS:
                    head = fit_anchored_contribution_head(
                        space,
                        labels,
                        training,
                        transform_indices=training,
                        prior_strength=prior,
                        class_balanced=True,
                    )
                    score = head.score(
                        space.baseline_score, space.contributions
                    )
                    baseline, _ = head.transform.apply(
                        space.baseline_score, space.contributions
                    )
                    rows.append({
                        "version": VERSION,
                        "cell": cell["cell"],
                        "group": cell["group"],
                        "seed": seed,
                        "representation": representation,
                        "prior_strength": prior,
                        "n_coordinates": len(space.families),
                        "n_training": len(training),
                        "n_evaluation": len(evaluation),
                        "iu_auroc": float(roc_auc_score(
                            labels[evaluation], baseline[evaluation]
                        )),
                        "head_auroc": float(roc_auc_score(
                            labels[evaluation], score[evaluation]
                        )),
                        "delta_norm": head.diagnostics["delta_norm"],
                        "class_balanced": head.diagnostics["class_balanced"],
                    })
    return rows


def cell_means(rows):
    output = []
    keys = sorted({
        (row["cell"], row["representation"], row["prior_strength"])
        for row in rows
    })
    for cell, representation, prior in keys:
        selected = [
            row for row in rows
            if row["cell"] == cell
            and row["representation"] == representation
            and row["prior_strength"] == prior
        ]
        output.append({
            "version": VERSION,
            "cell": cell,
            "group": selected[0]["group"],
            "representation": representation,
            "prior_strength": prior,
            "n_splits": len(selected),
            "n_coordinates": selected[0]["n_coordinates"],
            "mean_iu_auroc": float(np.mean([
                row["iu_auroc"] for row in selected
            ])),
            "mean_head_auroc": float(np.mean([
                row["head_auroc"] for row in selected
            ])),
            "mean_delta_pp": float(100 * np.mean([
                row["head_auroc"] - row["iu_auroc"] for row in selected
            ])),
        })
    return output


def summaries(means):
    output = []
    for representation in ("family", "atomic"):
        for prior in PRIOR_STRENGTHS:
            selected = [
                row for row in means
                if row["representation"] == representation
                and row["prior_strength"] == prior
            ]
            groups = sorted({row["group"] for row in selected})
            group_deltas = np.asarray([
                np.mean([
                    row["mean_head_auroc"] - row["mean_iu_auroc"]
                    for row in selected if row["group"] == group
                ])
                for group in groups
            ])
            seed = int(hashlib.sha256(
                f"{VERSION}:{representation}:{prior}".encode()
            ).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            draws = group_deltas[rng.integers(
                0, len(group_deltas),
                size=(BOOTSTRAP_DRAWS, len(group_deltas)),
            )].mean(axis=1)
            cell_deltas = np.asarray([
                row["mean_head_auroc"] - row["mean_iu_auroc"]
                for row in selected
            ])
            output.append({
                "version": VERSION,
                "representation": representation,
                "prior_strength": prior,
                "n_cells": len(selected),
                "n_groups": len(groups),
                "equal_group_delta_pp": float(100 * np.mean(group_deltas)),
                "ci_low_pp": float(100 * np.quantile(draws, 0.025)),
                "ci_high_pp": float(100 * np.quantile(draws, 0.975)),
                "wins": int(np.sum(cell_deltas > 0)),
                "losses": int(np.sum(cell_deltas < 0)),
                "worst_cell_delta_pp": float(100 * np.min(cell_deltas)),
            })
    return output


def representation_contrasts(means):
    output = []
    for prior in PRIOR_STRENGTHS:
        lookup = {
            (row["cell"], row["representation"]): row
            for row in means if row["prior_strength"] == prior
        }
        cells = sorted({cell for cell, _ in lookup})
        groups = sorted({lookup[cell, "family"]["group"] for cell in cells})
        group_contrasts = np.asarray([
            np.mean([
                lookup[cell, "atomic"]["mean_head_auroc"]
                - lookup[cell, "family"]["mean_head_auroc"]
                for cell in cells if lookup[cell, "family"]["group"] == group
            ])
            for group in groups
        ])
        seed = int(hashlib.sha256(
            f"{VERSION}:atomic-minus-family:{prior}".encode()
        ).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        draws = group_contrasts[rng.integers(
            0, len(group_contrasts),
            size=(BOOTSTRAP_DRAWS, len(group_contrasts)),
        )].mean(axis=1)
        output.append({
            "version": VERSION,
            "prior_strength": prior,
            "atomic_minus_family_pp": float(100 * np.mean(group_contrasts)),
            "ci_low_pp": float(100 * np.quantile(draws, 0.025)),
            "ci_high_pp": float(100 * np.quantile(draws, 0.975)),
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = replicate_rows(load_original(args.bundle))
    means = cell_means(rows)
    summary = summaries(means)
    contrasts = representation_contrasts(means)
    write_csv(args.out / "replicates.csv", rows)
    write_csv(args.out / "cell_means.csv", means)
    write_csv(args.out / "summary.csv", summary)
    write_csv(args.out / "atomic_minus_family.csv", contrasts)
    write_json(args.out / "RESULT.json", {
        "version": VERSION,
        "status": "supervised_diagnostic_only",
        "n_splits": N_SPLITS,
        "prior_strengths": list(PRIOR_STRENGTHS),
        "class_balanced": True,
        "aggregation": "split mean within cell, then equal dataset-group mean",
        "no_global_concatenated_auroc": True,
    })
    write_json(args.out / "RUN_DEFINITION.json", {
        "version": VERSION,
        "sources": {
            "script": sha256_file(Path(__file__)),
            "bundle": sha256_file(args.bundle),
            "atomic_module": sha256_file(
                REPO / "spectral_utils" / "atomic_neutral_residual.py"
            ),
            "contribution_module": sha256_file(
                REPO / "spectral_utils" / "contribution_subspace.py"
            ),
        },
        "n_splits": N_SPLITS,
        "prior_strengths": list(PRIOR_STRENGTHS),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    })
    lines = [
        "# Atomic contribution supervised ceiling",
        "",
        "**Status:** supervised diagnostic only; not a label-free method.",
        "",
        "| representation | prior | equal-group delta vs IU | 95% interval | W/L | worst |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['representation']} | {row['prior_strength']:g} "
            f"| {row['equal_group_delta_pp']:+.3f}pp "
            f"| [{row['ci_low_pp']:+.3f}, {row['ci_high_pp']:+.3f}] "
            f"| {row['wins']}/{row['losses']} "
            f"| {row['worst_cell_delta_pp']:+.3f}pp |"
        )
    lines.extend([
        "",
        "## Direct atomic-minus-family contrast",
        "",
        "| prior | atomic minus family | 95% interval |",
        "|---:|---:|---:|",
    ])
    for row in contrasts:
        lines.append(
            f"| {row['prior_strength']:g} "
            f"| {row['atomic_minus_family_pp']:+.3f}pp "
            f"| [{row['ci_low_pp']:+.3f}, {row['ci_high_pp']:+.3f}] |"
        )
    lines.extend([
        "",
        "Each AUROC is computed on one held-out split, then averaged within "
        "cell. No out-of-fold predictions are concatenated across cells. All "
        "heads use class-balanced loss.",
        "",
    ])
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
