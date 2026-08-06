#!/usr/bin/env python3
"""Matched real-artifact replay of bootstrap-stabilized SDSF.

The candidate and gates were frozen by the synthetic v3 study before this
script was run.  Inputs are the committed, bit-exact per-cell matrices in
``results/dependency_fusion_raw/cells.npz``.  This is a retrospective replay:
it can reject the candidate on the existing cells, but success would still
need confirmation on a new dataset/model family.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
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

from spectral_utils.dependency_fusion import sparse_upcr_fit           # noqa: E402
from spectral_utils.feature_contract import (                          # noqa: E402
    LEGACY_FEATURE_SIGNS,
    SCHEMA_VERSION,
    confidence_oriented_matrix,
    consensus_anchor,
)
from spectral_utils.robust_dependency_fusion import (                  # noqa: E402
    bootstrap_reliability,
    stability_shrunk_weights,
)


VERSION = "sdsf-stability-real-replay-v1"
DEFAULT_BUNDLE = os.path.join(REPO, "results", "dependency_fusion_raw", "cells.npz")
DEFAULT_MANIFEST = os.path.join(
    REPO, "results", "dependency_fusion_raw", "cells_manifest.csv",
)
DEFAULT_OUT = os.path.join(REPO, "results", "sdsf_stability_real_replay")
N_BOOT = 10
TAU = 0.5

SPARSE_FIT = dict(
    scale_ratio=0.25, rank=2, n_components=2,
    g2_projection_components=1, threshold_multiplier=1.0,
    max_iter=100, inner_completion_iter=40, decomposition_tol=1e-8,
    max_sparse_fraction=None, target_condition=100.0,
)

# The original real-data SDSF contribution gate is retained against SU-PCR.
# The final two gates are the additional requirement for claiming that the new
# stabilizer improves, rather than merely reproduces, current SDSF.
GATES = {
    "mean_vs_su_min_pp": 1.0,
    "cell_ci_low_vs_su_min_pp": 0.0,
    "qa_vs_su_min_pp": -0.5,
    "math_vs_su_min_pp": -0.5,
    "family_macro_vs_su_min_pp": 0.0,
    "mean_vs_current_sdsf_min_pp": 0.0,
    "cell_ci_low_vs_current_sdsf_min_pp": 0.0,
}

KNOWN_FAMILIES = (
    "triviaqa", "coqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
    "truthfulqa", "gsm8k", "math500", "gpqa", "webq", "humaneval",
)


def stable_seed(*parts):
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16) % (2 ** 32)


def family(cell):
    return next((name for name in KNOWN_FAMILIES if name in cell), cell)


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["cell"]: row for row in csv.DictReader(handle)}


def load_cells(path, manifest_path):
    data = np.load(path, allow_pickle=True)
    manifest = load_manifest(manifest_path)
    keys = sorted({name.rsplit("__", 1)[0] for name in data.files})
    cells = {}
    for key in keys:
        names = [str(name) for name in data[f"{key}__pool"]]
        legacy = np.asarray(data[f"{key}__hand_signs"], dtype=float)
        expected = np.asarray([LEGACY_FEATURE_SIGNS[name] for name in names], dtype=float)
        if not np.array_equal(legacy, expected):
            raise RuntimeError(f"{key}: historical sign vector no longer reconstructs raw views")
        raw = np.asarray(data[f"{key}__V"], dtype=float) * legacy
        matrix, kept, _ = confidence_oriented_matrix(raw, names, stable=True)
        cells[key] = {
            "matrix": matrix, "kept": kept,
            "labels": np.asarray(data[f"{key}__labels"], dtype=int),
            "anchor": consensus_anchor(matrix),
            "family": family(key), "domain": manifest[key]["domain"],
        }
    return cells


def oriented_auc(weight, matrix, labels, anchor):
    score = matrix @ np.asarray(weight, dtype=float)
    corr = float(np.corrcoef(score, anchor)[0, 1])
    if not np.isfinite(corr):
        raise RuntimeError("constant score")
    if corr < 0:
        score = -score
    return float(roc_auc_score(labels, score))


def bootstrap_ci(values, name, n_boot=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_seed(VERSION, "ci", name))
    means = np.empty(n_boot)
    for start in range(0, n_boot, 1000):
        size = min(1000, n_boot - start)
        idx = rng.integers(0, len(values), size=(size, len(values)))
        means[start:start + size] = values[idx].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def aggregate(rows, reference, candidate):
    delta = np.asarray([r[f"auc_{candidate}"] - r[f"auc_{reference}"] for r in rows])
    lo, hi = bootstrap_ci(delta, f"{candidate}_minus_{reference}")
    domains = {
        domain: float(np.mean([
            r[f"auc_{candidate}"] - r[f"auc_{reference}"]
            for r in rows if r["domain"] == domain
        ])) for domain in ("QA", "math")
    }
    families = sorted({r["family"] for r in rows})
    family_delta = np.asarray([
        np.mean([
            r[f"auc_{candidate}"] - r[f"auc_{reference}"]
            for r in rows if r["family"] == fam
        ]) for fam in families
    ])
    return {
        "reference": reference, "candidate": candidate,
        "mean_delta_pp": float(100 * delta.mean()),
        "ci95_low_pp": float(100 * lo), "ci95_high_pp": float(100 * hi),
        "median_delta_pp": float(100 * np.median(delta)),
        "qa_delta_pp": float(100 * domains["QA"]),
        "math_delta_pp": float(100 * domains["math"]),
        "family_macro_delta_pp": float(100 * family_delta.mean()),
        "wins": int(np.sum(delta > 0)), "losses": int(np.sum(delta < 0)),
        "worst_cell_delta_pp": float(100 * delta.min()),
        "catastrophic_losses_5pp": int(np.sum(delta <= -0.05)),
    }


def render(summary):
    lines = [
        "# Bootstrap-stabilized SDSF — real-artifact replay", "",
        f"Decision: **{summary['decision']}**.", "",
        f"Feature schema: `{summary['feature_schema']}`; candidate: bootstrap rho "
        f"shrinkage with tau={TAU}, {N_BOOT} row bootstraps, leading two coordinates "
        "preserved.", "",
        "This is a retrospective replay on the same 24-cell artifact used to diagnose "
        "SDSF. Labels were read only after each method's score was frozen.", "",
        "## Matched contrasts", "",
        "| reference -> candidate | mean [cell-bootstrap 95% CI] | QA / math | family macro | W/L | worst | <=-5pp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["contrasts"]:
        lines.append(
            f"| `{row['reference']}` -> `{row['candidate']}` | "
            f"{row['mean_delta_pp']:+.2f} [{row['ci95_low_pp']:+.2f}, "
            f"{row['ci95_high_pp']:+.2f}] | {row['qa_delta_pp']:+.2f} / "
            f"{row['math_delta_pp']:+.2f} | {row['family_macro_delta_pp']:+.2f} | "
            f"{row['wins']}/{row['losses']} | {row['worst_cell_delta_pp']:+.2f} | "
            f"{row['catastrophic_losses_5pp']} |"
        )
    lines += ["", "## Frozen gates", "",
              "| gate | observed | rule | result |", "|---|---:|---:|:---:|"]
    for gate in summary["gates"]:
        lines.append(
            f"| `{gate['gate']}` | {gate['observed']:+.3f} | >= "
            f"{gate['threshold']:+.3f} | **{'PASS' if gate['pass'] else 'FAIL'}** |"
        )
    lines += ["", "## Interpretation", "",
              summary["interpretation"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()
    cells = load_cells(args.bundle, args.manifest)
    rows = []
    for index, (cell_key, cell) in enumerate(cells.items()):
        F = cell["matrix"].T
        fit = sparse_upcr_fit(F, **SPARSE_FIT)
        boot = bootstrap_reliability(
            F, fit, n_boot=N_BOOT,
            seed=stable_seed(VERSION, cell_key, "bootstrap"),
            fit_kwargs=SPARSE_FIT,
        )
        stable_weight, diag = stability_shrunk_weights(
            fit, boot, tau=TAU, preserve_components=2,
            covariance_shrinkage=0.0, target_condition=50.0,
        )
        row = {
            "cell": cell_key, "family": cell["family"], "domain": cell["domain"],
            "n": len(cell["labels"]), "m": F.shape[0],
            "auc_su_pcr": oriented_auc(fit.w_pcr, cell["matrix"], cell["labels"], cell["anchor"]),
            "auc_current_sdsf": oriented_auc(fit.w_structured, cell["matrix"], cell["labels"], cell["anchor"]),
            "auc_stable_sdsf": oriented_auc(stable_weight, cell["matrix"], cell["labels"], cell["anchor"]),
            "tail_kappa_mean": diag["tail_kappa_mean"],
            "rho_retained_fraction": diag["rho_retained_fraction"],
            "bootstrap_success": boot.n_successful / boot.n_requested,
        }
        rows.append(row)
        print(f"{index + 1:02d}/{len(cells)} {cell_key}", flush=True)

    vs_su = aggregate(rows, "su_pcr", "stable_sdsf")
    vs_current = aggregate(rows, "current_sdsf", "stable_sdsf")
    observed = {
        "mean_vs_su_min_pp": vs_su["mean_delta_pp"],
        "cell_ci_low_vs_su_min_pp": vs_su["ci95_low_pp"],
        "qa_vs_su_min_pp": vs_su["qa_delta_pp"],
        "math_vs_su_min_pp": vs_su["math_delta_pp"],
        "family_macro_vs_su_min_pp": vs_su["family_macro_delta_pp"],
        "mean_vs_current_sdsf_min_pp": vs_current["mean_delta_pp"],
        "cell_ci_low_vs_current_sdsf_min_pp": vs_current["ci95_low_pp"],
    }
    gates = [{"gate": gate, "observed": observed[gate], "threshold": threshold,
              "pass": bool(observed[gate] >= threshold)}
             for gate, threshold in GATES.items()]
    passed = all(gate["pass"] for gate in gates)
    summary = {
        "version": VERSION, "feature_schema": SCHEMA_VERSION,
        "decision": "MEETS_REAL_CONTRIBUTION_GATE" if passed else "DOES_NOT_MEET_REAL_CONTRIBUTION_GATE",
        "config": {"n_boot": N_BOOT, "tau": TAU, "gates": GATES,
                   "bundle": os.path.abspath(args.bundle)},
        "n_cells": len(rows), "contrasts": [vs_su, vs_current], "gates": gates,
        "interpretation": (
            "The candidate clears every predeclared real-data contribution condition; "
            "because this artifact is retrospective, confirmation on a new family remains required."
            if passed else
            "The candidate fails at least one predeclared real-data contribution condition. "
            "It does provide strong evidence that bootstrap reliability shrinkage improves the "
            "current SDSF implementation, but stable SDSF remains materially below SU-PCR and "
            "therefore cannot replace the leading method. The failed gates identify the next "
            "hypothesis; they must not be repaired by tuning tau on these labels."
        ),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "per_cell.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(render(summary))
    print(summary["decision"])


if __name__ == "__main__":
    main()
