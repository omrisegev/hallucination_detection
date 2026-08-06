#!/usr/bin/env python3
"""Apples-to-apples comparison of U-PCR variants under fixed feature contracts.

The experiment removes both data-derived orientation seams used by the historical
deployment:

* no per-cell ``sign(rho)`` feature orientation; and
* no post-hoc anchor flip of the fused score.

Every input column is instead assigned the frozen semantic direction from
``confidence-orientation-v1``: larger means more likely correct.  The raw fused
score is evaluated as returned.  An AUROC below 0.5 is therefore an orientation
failure, not something that is silently repaired.

Three solvers receive exactly the same matrix for each feature contract:

* ``deployed_upcr``: the maintained deployment configuration of U-PCR, based on
  Dror et al. (2017), *Unsupervised Ensemble Regression*;
* ``iu_pcr``: the independent-error IU-PCR formulation in Tenzer et al. (2022),
  *Crowdsourcing Regression: A Spectral Approach*;
* ``su_pcr``: the sparse-error SU-PCR formulation from the same 2022 paper.

The transformation arms are deliberately small and frozen.  They replace (never
duplicate) the four quarantined views.  ``replace_squared`` implements the user's
mean-centred fold and ``replace_mode`` implements the Step-218 transform of record.
Both encode the explicit semantic assumption that values near the feature centre
mean greater confidence, hence the leading minus sign.  No correctness labels are
used by construction or fitting.
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
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, rankdata, wilcoxon
from sklearn.metrics import roc_auc_score


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Import the lightweight submodules without executing spectral_utils/__init__.py,
# whose model-loading facade is irrelevant to this CPU-only artifact replay.
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.dependency_fusion import sparse_upcr_fit          # noqa: E402
from spectral_utils.feature_contract import (                         # noqa: E402
    CONFIDENCE_FEATURE_SIGNS_V1,
    FIXED_STABLE_EXCLUDED_V1,
    LEGACY_FEATURE_SIGNS,
    SCHEMA_VERSION,
)
from spectral_utils.upcr import upcr_fit                               # noqa: E402


VERSION = "feature-contract-solver-comparison-v1-2026-08-06"
DEFAULT_BUNDLE = os.path.join(REPO, "results", "dependency_fusion_raw", "cells.npz")
DEFAULT_OUT = os.path.join(REPO, "results", "feature_contract_solver_comparison")

METHODS = ("deployed_upcr", "iu_pcr", "su_pcr")
CONTRACTS = (
    "fixed_all",
    "remove_unstable",
    "replace_squared",
    "replace_mode",
)

# Ties in the LOFO diagnostic prefer the simplest predeclared contract.
CONTRACT_PRIORITY = {
    "remove_unstable": 0,
    "fixed_all": 1,
    "replace_mode": 2,
    "replace_squared": 3,
}

PAPERS = {
    "deployed_upcr": {
        "label": "Deployed U-PCR",
        "paper": "Dror, Nadler, Bilal & Kluger (2017), Unsupervised Ensemble Regression",
        "url": "https://arxiv.org/abs/1703.02965",
        "note": "maintained deployment configuration with exclusion and recomputation",
    },
    "iu_pcr": {
        "label": "IU-PCR",
        "paper": "Tenzer, Dror, Nadler, Bilal & Kluger (2022), Crowdsourcing Regression: A Spectral Approach",
        "url": "https://proceedings.mlr.press/v151/tenzer22a.html",
        "note": "independent/uncorrelated-error variant, two-component PCR",
    },
    "su_pcr": {
        "label": "SU-PCR reproduction",
        "paper": "Tenzer, Dror, Nadler, Bilal & Kluger (2022), Crowdsourcing Regression: A Spectral Approach",
        "url": "https://proceedings.mlr.press/v151/tenzer22a.html",
        "note": "sparse correlated-error variant, two-component PCR",
    },
}

DEPLOYED_FIT = dict(
    loss="l2",
    exclusion=True,
    difficulty_gate=False,
    simple_avg_fallback=True,
    recompute_after_exclusion=True,
    g2_projection_k=1,
    scale_ratio=0.25,
)

IU_FIT = dict(
    loss="l2",
    exclusion=False,
    difficulty_gate=False,
    simple_avg_fallback=False,
    recompute_after_exclusion=False,
    g2_projection_k=1,
    scale_ratio=0.25,
    n_components=2,
    auto_components=False,
)

SU_FIT = dict(
    scale_ratio=0.25,
    rank=2,
    n_components=2,
    g2_projection_components=1,
    threshold_multiplier=1.0,
    max_iter=100,
    inner_completion_iter=40,
    decomposition_tol=1e-8,
    max_sparse_fraction=None,
    target_condition=100.0,
)

KNOWN_FAMILIES = (
    "triviaqa", "coqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
    "truthfulqa", "gsm8k", "math500", "gpqa", "webq", "humaneval",
)


def family(cell_key):
    return next((name for name in KNOWN_FAMILIES if name in cell_key), cell_key)


def domain(cell_key):
    math_tokens = ("gsm8k", "math500", "gpqa", "humaneval")
    return "math" if any(token in cell_key for token in math_tokens) else "QA"


def zscore_columns(matrix):
    matrix = np.asarray(matrix, dtype=float)
    mean = matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return (matrix - mean) / scale


def percentile_rank(values):
    values = np.asarray(values, dtype=float)
    return (rankdata(values) - 0.5) / len(values)


def mode_percentile(values, grid_size=512, min_prominence=0.05):
    """Label-free KDE mode location, matching the Step-218 definition."""
    values = np.asarray(values, dtype=float)
    if len(values) < 50 or np.std(values) < 1e-12:
        return 0.5
    try:
        kde = gaussian_kde(values)
        grid = np.linspace(values.min(), values.max(), int(grid_size))
        density = kde(grid)
    except Exception:
        return 0.5
    if not np.isfinite(density).all() or density.max() <= 0:
        return 0.5
    peaks, props = find_peaks(density, prominence=min_prominence * density.max())
    if len(peaks):
        peak = int(peaks[np.argmax(props["prominences"])])
    else:
        peak = int(np.argmax(density))
    return float(np.mean(values < grid[peak]))


def reconstruct_raw(data, key):
    names = [str(value) for value in data[f"{key}__pool"]]
    legacy = np.asarray(data[f"{key}__hand_signs"], dtype=float)
    expected = np.asarray([LEGACY_FEATURE_SIGNS[name] for name in names], dtype=float)
    if not np.array_equal(legacy, expected):
        raise RuntimeError(f"{key}: stored legacy signs do not match the registered mapping")
    # Stored V already contains the legacy sign; multiplying it again recovers
    # the raw standardized columns because every sign is +/-1.
    raw = np.asarray(data[f"{key}__V"], dtype=float) * legacy
    return raw, names


def build_contract(raw, names, contract):
    """Return one fixed, higher-means-correct matrix without labels or rho signs."""
    unknown = sorted(set(names) - set(CONFIDENCE_FEATURE_SIGNS_V1))
    if unknown:
        raise KeyError("unregistered feature(s): " + ", ".join(unknown))
    signs = np.asarray([CONFIDENCE_FEATURE_SIGNS_V1[name] for name in names], dtype=float)
    oriented = zscore_columns(np.asarray(raw, dtype=float) * signs)

    if contract == "fixed_all":
        return oriented, list(names), {}

    if contract == "remove_unstable":
        keep = np.asarray([name not in FIXED_STABLE_EXCLUDED_V1 for name in names])
        return oriented[:, keep], [name for name, flag in zip(names, keep) if flag], {
            "removed": sorted(set(names) & set(FIXED_STABLE_EXCLUDED_V1)),
        }

    if contract not in ("replace_squared", "replace_mode"):
        raise ValueError(f"unknown feature contract: {contract}")

    transformed = oriented.copy()
    details = {}
    for index, name in enumerate(names):
        if name not in FIXED_STABLE_EXCLUDED_V1:
            continue
        values = oriented[:, index]
        if contract == "replace_squared":
            replacement = -(values ** 2)
            details[name] = {"transform": "-z^2", "centre": 0.0}
        else:
            centre = mode_percentile(values)
            replacement = -np.abs(percentile_rank(values) - centre)
            details[name] = {"transform": "-|rank(x)-mode_rank|", "centre": centre}
        transformed[:, index] = replacement
    return zscore_columns(transformed), list(names), {"replaced": details}


def score_method(method, matrix):
    """Return the raw score. No sign(rho), consensus, EPR, or label flip occurs."""
    features_by_samples = np.asarray(matrix, dtype=float).T
    if method == "deployed_upcr":
        fit = upcr_fit(features_by_samples, **DEPLOYED_FIT)
        return fit.w @ features_by_samples, {
            "n_kept": int(fit.keep.sum()),
            "fit_residual": float(fit.proj_residual),
        }
    if method == "iu_pcr":
        fit = upcr_fit(features_by_samples, **IU_FIT)
        return fit.w @ features_by_samples, {
            "n_kept": int(fit.keep.sum()),
            "fit_residual": float(fit.proj_residual),
        }
    if method == "su_pcr":
        fit = sparse_upcr_fit(features_by_samples, **SU_FIT)
        return fit.w_pcr @ features_by_samples, {
            "n_kept": int(features_by_samples.shape[0]),
            "fit_residual": float(fit.projection_residual),
            "sparse_fraction": float(fit.decomposition.sparse_fraction),
            "decomposition_converged": bool(fit.decomposition.converged),
        }
    raise ValueError(method)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values, namespace, n_boot=20000):
    values = np.asarray(values, dtype=float)
    seed = int(hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_boot), dtype=float)
    for start in range(0, int(n_boot), 1000):
        count = min(1000, int(n_boot) - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def summarize(rows):
    output = []
    for method in METHODS:
        for contract in CONTRACTS:
            selected = [row for row in rows if row["method"] == method
                        and row["contract"] == contract]
            auc = np.asarray([row["auroc"] for row in selected], dtype=float)
            family_macro = np.mean([
                np.mean([row["auroc"] for row in selected if row["family"] == fam])
                for fam in sorted({row["family"] for row in selected})
            ])
            output.append({
                "method": method,
                "contract": contract,
                "n_cells": len(selected),
                "macro_auroc": float(auc.mean()),
                "qa_auroc": float(np.mean([row["auroc"] for row in selected
                                            if row["domain"] == "QA"])),
                "math_auroc": float(np.mean([row["auroc"] for row in selected
                                              if row["domain"] == "math"])),
                "equal_family_auroc": float(family_macro),
                "orientation_failures": int(np.sum(auc < 0.5)),
                "mean_n_features": float(np.mean([row["n_features"] for row in selected])),
                "mean_n_kept": float(np.mean([row["n_kept"] for row in selected])),
            })
    return output


def contract_contrasts(rows):
    by_key = {(row["method"], row["contract"], row["cell"]): row for row in rows}
    output = []
    for method in METHODS:
        for reference in ("fixed_all", "remove_unstable"):
            for candidate in CONTRACTS:
                if candidate == reference:
                    continue
                cells = sorted({row["cell"] for row in rows if row["method"] == method})
                delta = np.asarray([
                    by_key[method, candidate, cell]["auroc"]
                    - by_key[method, reference, cell]["auroc"]
                    for cell in cells
                ])
                lo, hi = bootstrap_ci(delta, f"{method}-{reference}-{candidate}")
                try:
                    p_value = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
                except ValueError:
                    p_value = float("nan")
                output.append({
                    "method": method,
                    "reference": reference,
                    "candidate": candidate,
                    "n_cells": len(cells),
                    "mean_delta_pp": float(100 * delta.mean()),
                    "ci95_low_pp": float(100 * lo),
                    "ci95_high_pp": float(100 * hi),
                    "wins": int(np.sum(delta > 0)),
                    "losses": int(np.sum(delta < 0)),
                    "ties": int(np.sum(delta == 0)),
                    "p_wilcoxon": p_value,
                })
    return output


def method_contrasts(rows):
    by_key = {(row["method"], row["contract"], row["cell"]): row for row in rows}
    output = []
    for contract in CONTRACTS:
        for candidate in ("iu_pcr", "su_pcr"):
            cells = sorted({row["cell"] for row in rows if row["contract"] == contract})
            delta = np.asarray([
                by_key[candidate, contract, cell]["auroc"]
                - by_key["deployed_upcr", contract, cell]["auroc"]
                for cell in cells
            ])
            lo, hi = bootstrap_ci(delta, f"method-{contract}-{candidate}")
            output.append({
                "contract": contract,
                "reference": "deployed_upcr",
                "candidate": candidate,
                "n_cells": len(cells),
                "mean_delta_pp": float(100 * delta.mean()),
                "ci95_low_pp": float(100 * lo),
                "ci95_high_pp": float(100 * hi),
                "wins": int(np.sum(delta > 0)),
                "losses": int(np.sum(delta < 0)),
                "ties": int(np.sum(delta == 0)),
            })
    return output


def lofo_contract_selection(rows):
    """Select one whole feature contract per method on other dataset families."""
    output = []
    families = sorted({row["family"] for row in rows})
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        for heldout in families:
            candidates = []
            for contract in CONTRACTS:
                train = [row["auroc"] for row in method_rows
                         if row["contract"] == contract and row["family"] != heldout]
                candidates.append((float(np.mean(train)), -CONTRACT_PRIORITY[contract], contract))
            train_macro, _, chosen = max(candidates)
            for row in method_rows:
                if row["family"] == heldout and row["contract"] == chosen:
                    output.append({
                        "method": method,
                        "heldout_family": heldout,
                        "chosen_contract": chosen,
                        "training_macro": train_macro,
                        "cell": row["cell"],
                        "domain": row["domain"],
                        "auroc": row["auroc"],
                    })
    return output


def make_plot(summary_rows, output_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    lookup = {(row["method"], row["contract"]): row for row in summary_rows}
    x = np.arange(len(CONTRACTS))
    width = 0.24
    fig, axis = plt.subplots(figsize=(11, 5.5))
    colours = ("#1f77b4", "#ff7f0e", "#2ca02c")
    for index, (method, colour) in enumerate(zip(METHODS, colours)):
        values = [lookup[method, contract]["macro_auroc"] for contract in CONTRACTS]
        axis.bar(x + (index - 1) * width, values, width, label=PAPERS[method]["label"],
                 color=colour)
        for xpos, value in zip(x + (index - 1) * width, values):
            axis.text(xpos, value + 0.002, f"{value:.3f}", ha="center", va="bottom",
                      fontsize=8, rotation=90)
    axis.set_ylim(min(row["macro_auroc"] for row in summary_rows) - 0.012,
                  max(row["macro_auroc"] for row in summary_rows) + 0.025)
    axis.set_ylabel("Cell-macro AUROC (raw score; no orientation flip)")
    axis.set_xticks(x, [name.replace("_", "\n") for name in CONTRACTS])
    axis.set_title("Common feature contracts × fusion solvers")
    axis.legend(loc="upper center", ncol=3, frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_report(summary_rows, contract_rows, method_rows, lofo_rows, metadata):
    lookup = {(row["method"], row["contract"]): row for row in summary_rows}
    lines = [
        "# Fixed feature-contract × solver comparison",
        "",
        f"Version: `{VERSION}`. Feature orientation schema: `{SCHEMA_VERSION}`.",
        "",
        "This is the apples-to-apples replay: every solver receives the same matrix for a given "
        "feature contract. There is **no per-cell `sign(rho)` orientation and no global anchor "
        "flip**. Larger input values are defined to mean greater confidence, and the raw fused "
        "score is evaluated directly. Labels enter only after all scores are frozen.",
        "",
        "![Common feature contracts by solver](comparison.png)",
        "",
        "## Solver provenance",
        "",
        "| method | source | implementation scope |",
        "|---|---|---|",
    ]
    for method in METHODS:
        paper = PAPERS[method]
        lines.append(
            f"| `{method}` | [{paper['paper']}]({paper['url']}) | {paper['note']} |"
        )
    lines.extend([
        "",
        "## Common-contract results",
        "",
        "Cell-macro AUROC. `inv` is the number of cells whose unflipped score has AUROC below "
        "0.5, which is an orientation-assumption failure.",
        "",
        "| feature contract | deployed U-PCR | IU-PCR | SU-PCR | inv (U/I/S) |",
        "|---|---:|---:|---:|---:|",
    ])
    for contract in CONTRACTS:
        values = [lookup[method, contract] for method in METHODS]
        lines.append(
            f"| `{contract}` | {values[0]['macro_auroc']:.4f} | "
            f"{values[1]['macro_auroc']:.4f} | {values[2]['macro_auroc']:.4f} | "
            f"{values[0]['orientation_failures']}/{values[1]['orientation_failures']}/"
            f"{values[2]['orientation_failures']} |"
        )
    lines.extend([
        "",
        "### Primary common baseline: remove unstable views",
        "",
        "| method | overall | QA | math | mean input / retained views |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        row = lookup[method, "remove_unstable"]
        lines.append(
            f"| `{method}` | {row['macro_auroc']:.4f} | {row['qa_auroc']:.4f} | "
            f"{row['math_auroc']:.4f} | {row['mean_n_features']:.1f} / "
            f"{row['mean_n_kept']:.1f} |"
        )
    lines.extend([
        "",
        "Contracts:",
        "",
        "- `fixed_all`: all raw views with frozen higher-means-correct directions.",
        "- `remove_unstable`: removes `pe_mean`, `stft_spectral_entropy`, "
        "  `cusum_shift_idx`, and `rpdi`.",
        "- `replace_squared`: replaces those four columns with `-z²`; higher means closer to "
        "  the mean and therefore greater confidence under the declared central-confidence assumption.",
        "- `replace_mode`: replaces them with `-|rank(x)-mode_rank|`, using a label-free KDE mode.",
        "",
        "## Feature-contract effects",
        "",
        "Paired cell deltas against the removal baseline:",
        "",
        "| method | candidate | mean [95% CI] | W/L/T |",
        "|---|---|---:|---:|",
    ])
    selected_contract_rows = [row for row in contract_rows
                              if row["reference"] == "remove_unstable"]
    for row in selected_contract_rows:
        lines.append(
            f"| `{row['method']}` | `{row['candidate']}` | "
            f"{row['mean_delta_pp']:+.2f}pp "
            f"[{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}] | "
            f"{row['wins']}/{row['losses']}/{row['ties']} |"
        )
    lines.extend([
        "",
        "## Solver effects within each identical contract",
        "",
        "| contract | candidate vs deployed U-PCR | mean [95% CI] | W/L/T |",
        "|---|---|---:|---:|",
    ])
    for row in method_rows:
        lines.append(
            f"| `{row['contract']}` | `{row['candidate']}` | "
            f"{row['mean_delta_pp']:+.2f}pp "
            f"[{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}] | "
            f"{row['wins']}/{row['losses']}/{row['ties']} |"
        )

    lines.extend([
        "",
        "## Leave-one-family-out contract diagnostic",
        "",
        "This diagnostic asks whether each solver should use a different whole feature contract. "
        "For each held-out dataset family, the contract is selected only on the other families. "
        "It is not the primary solver comparison because the methods may receive different inputs.",
        "",
        "| method | LOFO-selected macro | contract choices by held-out family |",
        "|---|---:|---|",
    ])
    for method in METHODS:
        selected = [row for row in lofo_rows if row["method"] == method]
        choices = {}
        for row in selected:
            choices.setdefault(row["chosen_contract"], set()).add(row["heldout_family"])
        choice_text = "; ".join(
            f"`{contract}`: {len(families)}"
            for contract, families in sorted(choices.items())
        )
        lines.append(
            f"| `{method}` | {np.mean([row['auroc'] for row in selected]):.4f} | {choice_text} |"
        )

    best_common = max(
        CONTRACTS,
        key=lambda contract: np.mean([lookup[method, contract]["macro_auroc"]
                                      for method in METHODS]),
    )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        f"The highest average common contract on this retrospective artifact is "
        f"`{best_common}`. This identifies the clean baseline for subsequent solver work; it is "
        "not prospective validation. Method-specific contract choices are reported only through "
        "the leave-one-family-out diagnostic, and no per-cell label-selected transformation is allowed.",
        "",
        "The fixed directions themselves were frozen after examining earlier cells, so a new "
        "dataset/model family remains necessary to validate the complete orientation-free pipeline.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/feature_contract_solver_comparison.py",
        "```",
        "",
        f"Runtime: {metadata['runtime_seconds']:.1f}s; cells: {metadata['n_cells']}.",
    ])
    return "\n".join(lines) + "\n"


def self_test():
    names = ["pe_mean", "epr", "rpdi", "trace_length", "spectral_entropy"]
    raw = np.column_stack([
        np.linspace(-2, 2, 101),
        np.linspace(2, -2, 101),
        np.sin(np.linspace(0, 4 * np.pi, 101)),
        np.linspace(-1, 1, 101),
        np.cos(np.linspace(0, 2 * np.pi, 101)),
    ])
    for contract in CONTRACTS:
        matrix, kept, _ = build_contract(raw, names, contract)
        assert matrix.shape[0] == len(raw)
        assert np.isfinite(matrix).all()
        assert matrix.shape[1] == len(kept)
    removed, kept, _ = build_contract(raw, names, "remove_unstable")
    assert removed.shape[1] == 3
    assert set(kept).isdisjoint(FIXED_STABLE_EXCLUDED_V1)
    squared, _, _ = build_contract(raw, names, "replace_squared")
    pe = names.index("pe_mean")
    assert squared[np.argmin(np.abs(raw[:, pe])), pe] > squared[0, pe]
    print("SELF-TEST PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    started = time.time()
    data = np.load(args.bundle, allow_pickle=True)
    keys = sorted({name.rsplit("__", 1)[0] for name in data.files})
    rows = []
    contract_metadata = {}
    for cell_key in keys:
        raw, names = reconstruct_raw(data, cell_key)
        labels = np.asarray(data[f"{cell_key}__labels"], dtype=int)
        for contract in CONTRACTS:
            matrix, contract_names, details = build_contract(raw, names, contract)
            contract_metadata[f"{cell_key}:{contract}"] = details
            for method in METHODS:
                score, diagnostics = score_method(method, matrix)
                rows.append({
                    "cell": cell_key,
                    "family": family(cell_key),
                    "domain": domain(cell_key),
                    "method": method,
                    "contract": contract,
                    "n": len(labels),
                    "n_features": len(contract_names),
                    "n_kept": diagnostics["n_kept"],
                    "auroc": float(roc_auc_score(labels, score)),
                    "score_mean": float(np.mean(score)),
                    "score_std": float(np.std(score)),
                    "fit_residual": diagnostics["fit_residual"],
                    "sparse_fraction": diagnostics.get("sparse_fraction", ""),
                    "decomposition_converged": diagnostics.get("decomposition_converged", ""),
                })

    summary_rows = summarize(rows)
    contract_rows = contract_contrasts(rows)
    method_rows = method_contrasts(rows)
    lofo_rows = lofo_contract_selection(rows)
    metadata = {
        "version": VERSION,
        "feature_schema": SCHEMA_VERSION,
        "bundle": os.path.abspath(args.bundle),
        "n_cells": len(keys),
        "runtime_seconds": time.time() - started,
        "per_cell_rho_orientation": False,
        "global_anchor_flip": False,
        "transformed_features": sorted(FIXED_STABLE_EXCLUDED_V1),
        "contract_metadata": contract_metadata,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "per_cell.csv"), rows)
    write_csv(os.path.join(args.out_dir, "summary.csv"), summary_rows)
    write_csv(os.path.join(args.out_dir, "contract_contrasts.csv"), contract_rows)
    write_csv(os.path.join(args.out_dir, "method_contrasts.csv"), method_rows)
    write_csv(os.path.join(args.out_dir, "lofo_contract_selection.csv"), lofo_rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    make_plot(summary_rows, os.path.join(args.out_dir, "comparison.png"))
    report = render_report(summary_rows, contract_rows, method_rows, lofo_rows, metadata)
    with open(os.path.join(args.out_dir, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
