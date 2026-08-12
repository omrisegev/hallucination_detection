#!/usr/bin/env python3
"""Retrospective audit and frozen calibration for NRM-CS-IU v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.harp_global_contribution_teacher import (  # noqa: E402
    DEFAULT_BUNDLE,
    load_llama_processbench_cells,
    load_original_cells,
    load_qwen_processbench_cells,
    load_semgrad_cells,
)
from spectral_utils.contribution_subspace import (  # noqa: E402
    fit_neutral_residual_mode_calibration,
    neutral_residual_mode_score,
)
from spectral_utils.specrage_views import VIEW_ORDER  # noqa: E402


VERSION = "neutral-residual-mode-cs-iu-v1-2026-08-13"
DEFAULT_OUT = REPO / "results" / "neutral_residual_mode_cs_iu_v1"
BOOTSTRAP_DRAWS = 20000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def calibration_payload(calibration):
    return {
        "version": VERSION,
        "families": list(calibration.families),
        "direction": calibration.direction.tolist(),
        "residual_covariance": calibration.residual_covariance.tolist(),
        "pair_counts": calibration.pair_counts.tolist(),
        "eigenvalues": calibration.eigenvalues.tolist(),
        "selected_index": calibration.selected_index,
        "diagnostics": calibration.diagnostics,
        "uses_labels": False,
    }


def evaluate(cell, calibration, regime):
    fitted = neutral_residual_mode_score(
        cell["space"], cell["weights"], calibration
    )
    if not np.allclose(fitted.baseline_score, cell["baseline"], atol=1e-12):
        raise RuntimeError(f"IU coordinate mismatch: {cell['cell']}")
    y = cell["correctness"]
    iu_auc = float(roc_auc_score(y, fitted.baseline_score))
    nrm_auc = float(roc_auc_score(y, fitted.score))
    return {
        "version": VERSION,
        "regime": regime,
        "domain": cell["domain"],
        "group": cell["group"],
        "cell": cell["cell"],
        "n": cell["n"],
        "n_correct": cell["n_correct"],
        "families_present": "|".join(cell["families"]),
        "iu_auroc": iu_auc,
        "nrm_auroc": nrm_auc,
        "nrm_delta_pp": 100 * (nrm_auc - iu_auc),
        "correction_scale": fitted.diagnostics["correction_scale"],
        "baseline_correction_covariance": fitted.diagnostics[
            "baseline_correction_covariance"
        ],
        "weight_reconstruction_error": fitted.diagnostics[
            "weight_reconstruction_error"
        ],
    }


def grouped_summary(rows, domain):
    selected = [row for row in rows if row["domain"] == domain]
    groups = sorted({row["group"] for row in selected})
    group_deltas = np.asarray([
        np.mean([
            row["nrm_auroc"] - row["iu_auroc"]
            for row in selected if row["group"] == group
        ])
        for group in groups
    ])
    seed = int(hashlib.sha256(
        f"{VERSION}:{domain}".encode()
    ).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    draws = group_deltas[
        rng.integers(
            0,
            len(group_deltas),
            size=(BOOTSTRAP_DRAWS, len(group_deltas)),
        )
    ].mean(axis=1)
    cell_deltas = np.asarray([
        row["nrm_auroc"] - row["iu_auroc"] for row in selected
    ])
    return {
        "version": VERSION,
        "domain": domain,
        "n_cells": len(selected),
        "n_groups": len(groups),
        "cell_macro_delta_pp": float(100 * np.mean(cell_deltas)),
        "equal_group_delta_pp": float(100 * np.mean(group_deltas)),
        "equal_group_ci_low_pp": float(100 * np.quantile(draws, 0.025)),
        "equal_group_ci_high_pp": float(100 * np.quantile(draws, 0.975)),
        "wins": int(np.sum(cell_deltas > 0)),
        "losses": int(np.sum(cell_deltas < 0)),
        "ties": int(np.sum(cell_deltas == 0)),
        "worst_cell_delta_pp": float(100 * np.min(cell_deltas)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    original = load_original_cells(args.bundle)
    rows = []
    for heldout_group in sorted({cell["group"] for cell in original}):
        source = [cell for cell in original if cell["group"] != heldout_group]
        calibration = fit_neutral_residual_mode_calibration(
            cell["space"] for cell in source
        )
        for cell in original:
            if cell["group"] == heldout_group:
                rows.append(evaluate(cell, calibration, "original_lofo"))

    source_calibration = fit_neutral_residual_mode_calibration(
        cell["space"] for cell in original
    )
    write_json(
        args.out / "FROZEN_CALIBRATION.json",
        calibration_payload(source_calibration),
    )
    external = (
        load_qwen_processbench_cells()
        + load_llama_processbench_cells()
        + load_semgrad_cells()
    )
    for cell in external:
        rows.append(evaluate(cell, source_calibration, "source23_transfer"))

    summaries = [
        grouped_summary(rows, domain)
        for domain in sorted({row["domain"] for row in rows})
    ]
    write_csv(args.out / "cell_results.csv", rows)
    write_csv(args.out / "summary.csv", summaries)
    numerical_pass = bool(all(
        np.isfinite(row["nrm_auroc"])
        and abs(row["baseline_correction_covariance"]) < 1e-10
        and row["weight_reconstruction_error"] < 1e-10
        for row in rows
    ))
    write_json(args.out / "RESULT.json", {
        "version": VERSION,
        "status": "frozen_label_free_candidate",
        "source_calibration": calibration_payload(source_calibration),
        "numerical_invariants_pass": numerical_pass,
        "retrospective_only": True,
        "hle_not_evaluated_here": True,
    })
    sources = {
        "script": sha256_file(Path(__file__)),
        "spec": sha256_file(REPO / "SPEC_NEUTRAL_RESIDUAL_MODE_CS_IU_V1.md"),
        "module": sha256_file(
            REPO / "spectral_utils" / "contribution_subspace.py"
        ),
        "bundle": sha256_file(args.bundle),
    }
    write_json(args.out / "RUN_DEFINITION.json", {
        "version": VERSION,
        "sources": sources,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "calibration_scope": "23 original cells; contribution spaces only",
        "labels_used_for_calibration": False,
        "target_labels_used_for_fit": False,
    })

    def signed(value):
        return f"{float(value):+.3f}"

    lines = [
        "# Neutral residual mode CS-IU v1",
        "",
        "**Status:** frozen label-free candidate; all results below are "
        "retrospective.",
        "",
        "| domain | equal-group delta vs IU | 95% interval | W/L | worst |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['domain']} | {signed(row['equal_group_delta_pp'])}pp "
            f"| [{signed(row['equal_group_ci_low_pp'])}, "
            f"{signed(row['equal_group_ci_high_pp'])}] "
            f"| {row['wins']}/{row['losses']} "
            f"| {signed(row['worst_cell_delta_pp'])}pp |"
        )
    lines.extend([
        "",
        "## Frozen source calibration",
        "",
        f"Selected eigenvalue: "
        f"`{source_calibration.diagnostics['selected_eigenvalue']:.6f}`.",
        "",
        "| family | direction |",
        "|---|---:|",
    ])
    for family, value in zip(VIEW_ORDER, source_calibration.direction):
        lines.append(f"| `{family}` | {value:+.6f} |")
    lines.extend([
        "",
        "No HLE label or score is read by this audit.  The calibration above is "
        "the immutable input to the separate frozen HLE confirmation.",
        "",
    ])
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "version": VERSION,
        "direction": source_calibration.direction.tolist(),
        "selected_eigenvalue": source_calibration.diagnostics[
            "selected_eigenvalue"
        ],
        "numerical_invariants_pass": numerical_pass,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
