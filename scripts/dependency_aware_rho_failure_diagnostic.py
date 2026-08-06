#!/usr/bin/env python3
"""Post-hoc, label-free diagnosis of the rejected v5 synthetic hypothesis.

This script does not promote or tune a candidate.  It replays only the methods
that development had promoted, never reads real artifacts, and decomposes known-
truth rho error into the two PCR-retained coordinates and the discarded tail.
"""

import csv
import json
import os
import sys
import types

import numpy as np
from scipy.linalg import eigh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from dependency_aware_rho_autoresearch import (                       # noqa: E402
    PRIMARY_WORLDS,
    VERSION,
    WORLDS,
    bootstrap_ci,
    draw_world,
    fit_candidates,
    write_csv,
)
from spectral_utils.dependency_fusion import _pcr_weights              # noqa: E402


DEFAULT_RESULT = os.path.join(REPO, "results", "dependency_aware_rho_v5")


def safe_relative(error, target):
    return float(np.linalg.norm(error) / (np.linalg.norm(target) + 1e-12))


def run(result_dir=DEFAULT_RESULT):
    with open(os.path.join(result_dir, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    promoted = tuple(
        row["candidate"] for row in summary["development"] if row["promoted"]
    )
    methods = ("ols",) + promoted
    repetitions = int(summary["config"]["validation_repeats"])
    rows = []
    errors = {}

    for world in WORLDS:
        for repetition in range(repetitions):
            X_train, X_test, _labels_not_used, rho_true = draw_world(
                world, repetition, "sealed_validation",
            )
            fit, results = fit_candidates(X_train.T, methods)
            m = len(rho_true)
            vectors = eigh(fit.covariance, subset_by_index=[m - 2, m - 1])[1][:, ::-1]
            head_true = vectors.T @ rho_true
            tail_projector = np.eye(m) - vectors @ vectors.T
            oracle_weight, _ = _pcr_weights(fit.covariance, rho_true, n_components=2)
            oracle_score = X_test @ oracle_weight
            for method, result in results.items():
                error = result.rho_hat - rho_true
                centered_error = (
                    result.rho_hat - result.rho_hat.mean()
                    - (rho_true - rho_true.mean())
                )
                head_error = vectors.T @ error
                tail_error = tail_projector @ error
                weight_error = result.w_pcr - oracle_weight
                score = X_test @ result.w_pcr
                score_corr = float(np.corrcoef(score, oracle_score)[0, 1])
                key = (world, method)
                errors.setdefault(key, []).append(error / (np.linalg.norm(rho_true) + 1e-12))
                rows.append({
                    "world": world, "repetition": repetition, "method": method,
                    "full_rho_nrmse": safe_relative(error, rho_true),
                    "shape_rho_nrmse": safe_relative(
                        centered_error, rho_true - rho_true.mean(),
                    ),
                    "head_rho_nrmse": safe_relative(head_error, head_true),
                    "tail_error_norm": float(np.linalg.norm(tail_error)),
                    "pcr_weight_nrmse": safe_relative(weight_error, oracle_weight),
                    "oracle_score_correlation": score_corr,
                    "g2_hat": result.g2_hat,
                })

    baseline = {
        (row["world"], row["repetition"]): row
        for row in rows if row["method"] == "ols"
    }
    metrics = (
        "full_rho_nrmse", "shape_rho_nrmse", "head_rho_nrmse", "pcr_weight_nrmse",
    )
    contrasts = []
    for method in promoted:
        for metric in metrics:
            values = []
            for row in rows:
                if row["method"] != method or row["world"] not in PRIMARY_WORLDS:
                    continue
                control = baseline[(row["world"], row["repetition"])][metric]
                values.append((control - row[metric]) / (control + 1e-12))
            lo, hi = bootstrap_ci(
                values, f"v5_failure_diagnostic_{method}_{metric}",
            )
            contrasts.append({
                "method": method, "metric": metric,
                "relative_error_reduction": float(np.mean(values)),
                "ci95_low": lo, "ci95_high": hi,
            })

    bias_variance = []
    for (world, method), values in sorted(errors.items()):
        values = np.asarray(values)
        mean_error = values.mean(axis=0)
        bias_squared = float(np.dot(mean_error, mean_error))
        variance = float(np.mean(np.sum((values - mean_error) ** 2, axis=1)))
        bias_variance.append({
            "world": world, "method": method,
            "normalized_bias_squared": bias_squared,
            "normalized_variance": variance,
            "normalized_mse": bias_squared + variance,
        })

    output = {
        "version": f"{VERSION}-failure-diagnostic-1",
        "scope": "sealed synthetic known truth only; no correctness labels used",
        "methods": methods,
        "contrasts": contrasts,
        "bias_variance": bias_variance,
    }
    write_csv(os.path.join(result_dir, "failure_diagnostic_replicates.csv"), rows)
    write_csv(os.path.join(result_dir, "failure_diagnostic_contrasts.csv"), contrasts)
    write_csv(os.path.join(result_dir, "failure_diagnostic_bias_variance.csv"), bias_variance)
    with open(os.path.join(result_dir, "failure_diagnostic.json"), "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)

    lines = [
        "# v5 failure diagnosis — retained PCR coordinates", "",
        "This is a post-hoc mechanism diagnosis, not a promotion experiment. It "
        "replayed only development-promoted methods on sealed synthetic known truth; "
        "correctness labels and real artifacts were not read.", "",
        "Positive values mean lower error than OLS SU-PCR.", "",
        "| candidate | full rho | centered shape | retained PCR head | PCR weight |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in promoted:
        found = {row["metric"]: row for row in contrasts if row["method"] == method}
        cells = []
        for metric in metrics:
            row = found[metric]
            cells.append(
                f"{100*row['relative_error_reduction']:+.2f}% "
                f"[{100*row['ci95_low']:+.2f}, {100*row['ci95_high']:+.2f}]"
            )
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")

    with open(os.path.join(result_dir, "FAILURE_DIAGNOSTIC.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return output


if __name__ == "__main__":
    run()

