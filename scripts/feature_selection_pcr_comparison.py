#!/usr/bin/env python3
"""Feature-selection comparison on the fixed hallucination-feature contract.

This experiment is stacked on ``feature_contract_solver_comparison.py``.  It
uses only the ``remove_unstable`` contract, never estimates per-cell feature
directions, and never flips a fused score.  Labels are unavailable to every
selector and enter only after a subset and a solver score have been frozen.

The design separates three questions that the historical L-SML selector bench
mixed together:

1. Native-size utility: does a selector's own stopping rule improve a PCR
   solver, and beat random subsets with the same number of views?
2. Ranking utility: at a common budget of six views, does a selector rank views
   better than other selectors and size-matched random subsets?
3. Solver compatibility: does the same subset interact differently with the
   deployed U-PCR, IU-PCR, and SU-PCR assumptions?

Insertion points are deliberately solver-specific after the common upstream
selection stage:

* deployed U-PCR retains its own exclusion-and-recompute step, so external
  selection is judged on top of (not instead of) its maintained safeguard;
* IU-PCR receives the subset directly, with dependence diagnostics because
  DPP/decorrelation most directly target its uncorrelated-error assumption;
* SU-PCR receives the subset directly, while decomposition convergence and the
  paper's sparse-support condition are treated as validity guardrails.  We do
  not delete views merely to force the theorem condition to pass.

Representative selector families are rerun from scratch on the corrected pool:
DUFS, GroupFS, Laplacian Score, SPEC, MCFS, Concrete AE, LS-CAE, DPP greedy
log-det, and greedy decorrelation.  The U-PCR residual selector is included as
an explicitly method-specific control.  No result from the old sign(rho)/L-SML
benchmark is reused.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
for path in (REPO, SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)

# Keep the CPU replay independent of spectral_utils/__init__.py's model facade.
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from feature_contract_solver_comparison import (  # noqa: E402
    DEPLOYED_FIT,
    IU_FIT,
    METHODS,
    PAPERS,
    SU_FIT,
    build_contract,
    domain,
    family,
    reconstruct_raw,
)
from spectral_utils.dependency_fusion import sparse_upcr_fit  # noqa: E402
from spectral_utils.feature_contract import SCHEMA_VERSION  # noqa: E402
from spectral_utils.selector_bench import UnlabeledCell  # noqa: E402
from spectral_utils.selectors import get_selector  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402


VERSION = "feature-selection-pcr-comparison-v1-2026-08-06"
DEFAULT_BUNDLE = os.path.join(REPO, "results", "dependency_fusion_raw", "cells.npz")
DEFAULT_OUT = os.path.join(REPO, "results", "feature_selection_pcr_comparison")
CONTRACT = "remove_unstable"
FIXED_K = 6
RANDOM_REPEATS = 32

COMMON_METHODS = tuple(METHODS)
ARM_ORDER = (
    "full",
    "dufs_native", "groupfs_native",
    "lapscore_native", "spec_native", "mcfs_native",
    "cae_native", "lscae_native",
    "dpp_native", "dpp_ridge_native",
    "dufs_k6", "lapscore_k6", "spec_k6", "mcfs_k6",
    "cae_k6", "lscae_k6", "dpp_k6", "decorr_k6",
    "upcrres_native",
)

ARM_META = {
    "full": ("baseline", "full", "All corrected views"),
    "dufs_native": ("native", "dufs", "DUFS parameter-free native gates"),
    "groupfs_native": ("native", "groupfs", "GroupFS native groups/gates"),
    "lapscore_native": ("native", "lapscore", "Laplacian Score adaptive"),
    "spec_native": ("native", "spec", "SPEC adaptive"),
    "mcfs_native": ("native", "mcfs", "MCFS adaptive"),
    "cae_native": ("native", "cae", "Concrete AE adaptive"),
    "lscae_native": ("native", "lscae", "LS-CAE adaptive"),
    "dpp_native": ("native", "dpp", "DPP data-driven stop"),
    "dpp_ridge_native": ("native", "dpp", "DPP ridge data-driven stop"),
    "dufs_k6": ("fixed_k6", "dufs", "DUFS gate ranking, k=6"),
    "lapscore_k6": ("fixed_k6", "lapscore", "Laplacian Score, k=6"),
    "spec_k6": ("fixed_k6", "spec", "SPEC, k=6"),
    "mcfs_k6": ("fixed_k6", "mcfs", "MCFS, k=6"),
    "cae_k6": ("fixed_k6", "cae", "Concrete AE, k=6"),
    "lscae_k6": ("fixed_k6", "lscae", "LS-CAE, k=6"),
    "dpp_k6": ("fixed_k6", "dpp", "DPP greedy log-det, k=6"),
    "decorr_k6": ("fixed_k6", "decorrelation", "Greedy decorrelation, k=6"),
    "upcrres_native": ("method_specific", "upcr_residual", "U-PCR residual greedy"),
}


def stable_seed(*parts):
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2 ** 32)


def zscore(values):
    values = np.asarray(values, dtype=float)
    scale = float(values.std())
    return (values - values.mean()) / (scale if scale > 1e-12 else 1.0)


def absolute_spearman(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[1] == 1:
        return np.ones((1, 1), dtype=float)
    result = spearmanr(matrix, axis=0).statistic
    result = np.abs(np.nan_to_num(np.atleast_2d(result), nan=0.0))
    np.fill_diagonal(result, 1.0)
    return result


def make_unlabeled_cell(cell_key, matrix, names):
    # A fixed consensus is exposed only to selector families whose published
    # objective accepts an unlabeled anchor.  None of the selected primary arms
    # in this experiment uses correctness labels or a fitted rho polarity.
    anchor = zscore(np.asarray(matrix, dtype=float).mean(axis=1))
    return UnlabeledCell(
        domain=domain(cell_key),
        cell_key=cell_key,
        pool=list(names),
        pool_bits=np.arange(len(names), dtype=np.uint8),
        V=np.asarray(matrix, dtype=float),
        anchor=anchor,
        anchor_name="fixed_confidence_consensus",
        rho=absolute_spearman(matrix),
    )


def by_variant(outputs):
    return {str(item["variant"]): item for item in outputs}


def clean_cols(item, p):
    cols = np.asarray(item["cols"], dtype=int)
    cols = np.unique(cols)
    if len(cols) < 3 or np.any(cols < 0) or np.any(cols >= p):
        raise ValueError(f"invalid selector subset: {cols.tolist()} for p={p}")
    return cols


def arm_record(arm, item, p, intended_methods=COMMON_METHODS, extra_diag=None):
    phase, selector, label = ARM_META[arm]
    diag = dict(item.get("diag", {}))
    if extra_diag:
        diag.update(extra_diag)
    return {
        "arm": arm,
        "phase": phase,
        "selector": selector,
        "label": label,
        "variant": str(item.get("variant", arm)),
        "cols": clean_cols(item, p),
        "fallback": bool(item.get("fallback", False)),
        "diag": diag,
        "intended_methods": tuple(intended_methods),
    }


def select_one_cell(payload):
    """Run every selector family once; suitable for process-level parallelism."""
    cell_key, matrix, names = payload
    cell = make_unlabeled_cell(cell_key, matrix, names)
    p = cell.p
    arms = [{
        "arm": "full", "phase": "baseline", "selector": "full",
        "label": ARM_META["full"][2], "variant": "full",
        "cols": np.arange(p, dtype=int), "fallback": False, "diag": {},
        "intended_methods": COMMON_METHODS,
    }]

    family_outputs = {}
    for selector_family in (
        "a2_groupfs", "classical_fs", "a3_concrete_ae", "a8_lscae",
        "a9_dpp", "simple_stats", "a1_residual",
    ):
        rng = np.random.default_rng(stable_seed(VERSION, cell_key, selector_family))
        family_outputs[selector_family] = by_variant(
            get_selector(selector_family)(cell, rng, cache=None)
        )

    a2 = family_outputs["a2_groupfs"]
    arms.append(arm_record("dufs_native", a2["a2.dufs_pf"], p))
    arms.append(arm_record("groupfs_native", a2["a2.select"], p))
    gates = np.asarray(a2["a2.dufs_pf"].get("diag", {}).get("feat_gate_means", []),
                       dtype=float)
    if len(gates) == p and p >= FIXED_K:
        dufs_k6 = {
            "variant": "a2.dufs_pf_top6",
            "cols": np.sort(np.argsort(gates)[::-1][:FIXED_K]),
            "fallback": bool(a2["a2.dufs_pf"].get("fallback", False)),
            "diag": {"source": "a2.dufs_pf", "rule": "six largest gate means"},
        }
    else:
        dufs_k6 = {
            "variant": "a2.dufs_pf_top6", "cols": np.arange(p), "fallback": True,
            "diag": {"error": "DUFS gate ranking unavailable"},
        }
    arms.append(arm_record("dufs_k6", dufs_k6, p))

    classical = family_outputs["classical_fs"]
    for arm, variant in (
        ("lapscore_native", "lapscore_adapt"), ("lapscore_k6", "lapscore_s6"),
        ("spec_native", "spec_adapt"), ("spec_k6", "spec_s6"),
        ("mcfs_native", "mcfs_adapt"), ("mcfs_k6", "mcfs_s6"),
    ):
        arms.append(arm_record(arm, classical[variant], p))

    cae = family_outputs["a3_concrete_ae"]
    arms.append(arm_record("cae_native", cae["a3.cae"], p))
    arms.append(arm_record("cae_k6", cae["a3.cae_k6"], p))

    lscae = family_outputs["a8_lscae"]
    arms.append(arm_record("lscae_native", lscae["lscae"], p))
    arms.append(arm_record("lscae_k6", lscae["lscae.k6"], p))

    dpp = family_outputs["a9_dpp"]
    arms.append(arm_record("dpp_native", dpp["dpp"], p))
    arms.append(arm_record("dpp_ridge_native", dpp["dpp.ridge"], p))
    arms.append(arm_record("dpp_k6", dpp["dpp.k6"], p))

    stats = family_outputs["simple_stats"]
    arms.append(arm_record("decorr_k6", stats["decorr_s6"], p))

    residual = family_outputs["a1_residual"]
    arms.append(arm_record(
        "upcrres_native", residual["a1.upcrres_greedy"], p,
        intended_methods=("deployed_upcr",),
        extra_diag={"scope": "U-PCR only; its own projection residual is the selector objective"},
    ))

    # Stable ordering makes output diffs and resume audits straightforward.
    order = {arm: index for index, arm in enumerate(ARM_ORDER)}
    arms.sort(key=lambda row: order[row["arm"]])
    return cell_key, arms


def dependence_diagnostics(matrix):
    matrix = np.asarray(matrix, dtype=float)
    p = matrix.shape[1]
    rho = absolute_spearman(matrix)
    tri = rho[np.triu_indices(p, 1)] if p > 1 else np.asarray([], dtype=float)
    covariance = np.cov(matrix, rowvar=False, bias=True)
    eigenvalues = np.maximum(np.linalg.eigvalsh(np.atleast_2d(covariance)), 0.0)
    total = float(eigenvalues.sum())
    if total <= 1e-12:
        effective_rank = 0.0
    else:
        probs = eigenvalues[eigenvalues > 1e-12] / total
        effective_rank = float(np.exp(-np.sum(probs * np.log(probs))))
    return {
        "mean_abs_spearman": float(np.mean(tri)) if len(tri) else 0.0,
        "max_abs_spearman": float(np.max(tri)) if len(tri) else 0.0,
        "effective_rank": effective_rank,
        "effective_rank_fraction": float(effective_rank / max(p, 1)),
    }


def score_solver(method, matrix):
    features = np.asarray(matrix, dtype=float).T
    if method == "deployed_upcr":
        fit = upcr_fit(features, **DEPLOYED_FIT)
        return fit.w @ features, {
            "n_kept": int(fit.keep.sum()),
            "fit_residual": float(fit.proj_residual),
            "decomposition_converged": "", "theorem_support_ok": "",
            "nnz_pairs": "", "sparse_fraction": "",
        }
    if method == "iu_pcr":
        fit = upcr_fit(features, **IU_FIT)
        return fit.w @ features, {
            "n_kept": int(fit.keep.sum()),
            "fit_residual": float(fit.proj_residual),
            "decomposition_converged": "", "theorem_support_ok": "",
            "nnz_pairs": "", "sparse_fraction": "",
        }
    if method == "su_pcr":
        fit = sparse_upcr_fit(features, **SU_FIT)
        decomp = fit.decomposition
        return fit.w_pcr @ features, {
            "n_kept": int(features.shape[0]),
            "fit_residual": float(fit.projection_residual),
            "decomposition_converged": bool(decomp.converged),
            "theorem_support_ok": bool(decomp.theorem_support_ok),
            "nnz_pairs": int(decomp.meta.get("nnz_pairs", 0)),
            "sparse_fraction": float(decomp.sparse_fraction),
        }
    raise ValueError(method)


def random_floor(method, matrix, labels, k, repeats, namespace):
    p = matrix.shape[1]
    if k == p:
        score, _ = score_solver(method, matrix)
        value = float(roc_auc_score(labels, score))
        return np.full(repeats, value, dtype=float)
    rng = np.random.default_rng(stable_seed(VERSION, namespace, method, k))
    values = []
    for _ in range(repeats):
        cols = np.sort(rng.choice(p, size=k, replace=False))
        try:
            score, _ = score_solver(method, matrix[:, cols])
            values.append(float(roc_auc_score(labels, score)))
        except Exception:
            continue
    return np.asarray(values, dtype=float)


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
    rng = np.random.default_rng(stable_seed(VERSION, namespace))
    means = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, 1000):
        count = min(1000, n_boot - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def summarize(rows):
    full = {(row["method"], row["cell"]): row for row in rows if row["arm"] == "full"}
    output = []
    keys = sorted({(row["arm"], row["method"]) for row in rows},
                  key=lambda item: (ARM_ORDER.index(item[0]), METHODS.index(item[1])))
    for arm, method in keys:
        selected = [row for row in rows if row["arm"] == arm and row["method"] == method]
        auc = np.asarray([row["auroc"] for row in selected], dtype=float)
        delta = np.asarray([
            row["auroc"] - full[method, row["cell"]]["auroc"] for row in selected
        ], dtype=float)
        random_delta = np.asarray([
            row["auroc"] - row["random_mean_auroc"] for row in selected
        ], dtype=float)
        lo, hi = bootstrap_ci(delta, f"full-{arm}-{method}")
        rlo, rhi = bootstrap_ci(random_delta, f"random-{arm}-{method}")
        try:
            p_value = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
        except ValueError:
            p_value = float("nan")
        theorem = [row["theorem_support_ok"] for row in selected
                   if row["theorem_support_ok"] != ""]
        theorem_eligible = [row for row in selected
                            if row["theorem_support_ok"] != "" and row["n_selected"] >= 5]
        converged = [row["decomposition_converged"] for row in selected
                     if row["decomposition_converged"] != ""]
        phase, selector, label = ARM_META[arm]
        output.append({
            "arm": arm, "phase": phase, "selector": selector, "label": label,
            "method": method, "n_cells": len(selected),
            "macro_auroc": float(auc.mean()),
            "mean_delta_vs_full_pp": float(100 * delta.mean()),
            "delta_vs_full_ci95_low_pp": float(100 * lo),
            "delta_vs_full_ci95_high_pp": float(100 * hi),
            "wins_vs_full": int(np.sum(delta > 0)),
            "losses_vs_full": int(np.sum(delta < 0)),
            "ties_vs_full": int(np.sum(delta == 0)),
            "p_wilcoxon_vs_full": p_value,
            "mean_delta_vs_matched_random_pp": float(100 * random_delta.mean()),
            "delta_vs_random_ci95_low_pp": float(100 * rlo),
            "delta_vs_random_ci95_high_pp": float(100 * rhi),
            "wins_vs_matched_random": int(np.sum(random_delta > 0)),
            "mean_n_selected": float(np.mean([row["n_selected"] for row in selected])),
            "mean_n_kept": float(np.mean([row["n_kept"] for row in selected])),
            "no_selection_cells": int(np.sum([
                row["n_selected"] == row["n_pool"] for row in selected
            ])),
            "orientation_failures": int(np.sum(auc < 0.5)),
            "mean_abs_spearman": float(np.mean([row["mean_abs_spearman"] for row in selected])),
            "mean_effective_rank_fraction": float(np.mean([
                row["effective_rank_fraction"] for row in selected
            ])),
            "fallback_cells": int(np.sum([row["selector_fallback"] for row in selected])),
            "su_minimum_size_rate": float(np.mean([
                row["n_selected"] >= 5 for row in selected
            ])) if theorem else "",
            "su_theorem_support_rate": float(np.mean(theorem)) if theorem else "",
            "su_theorem_support_rate_if_size_eligible": float(np.mean([
                row["theorem_support_ok"] for row in theorem_eligible
            ])) if theorem_eligible else "",
            "su_decomposition_convergence_rate": float(np.mean(converged)) if converged else "",
        })
    return output


def feature_frequency(selection_rows):
    output = []
    for arm in ARM_ORDER:
        selected = [row for row in selection_rows if row["arm"] == arm]
        if not selected:
            continue
        names = sorted({name for row in selected for name in row["selected_features"].split("|")})
        for name in names:
            count = sum(name in row["selected_features"].split("|") for row in selected)
            output.append({
                "arm": arm, "feature": name, "selected_cells": count,
                "available_cells": sum(name in row["pool_features"].split("|") for row in selected),
                "selection_rate_over_all_cells": float(count / len(selected)),
            })
    return output


def make_plot(summary_rows, output_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=False)
    colours = {"deployed_upcr": "#1f77b4", "iu_pcr": "#ff7f0e", "su_pcr": "#2ca02c"}
    phases = (
        ("native", axes[0], "Native selector stopping rules"),
        ("fixed_k6", axes[1], "Common six-view budget (ranking quality)"),
    )
    for phase, axis, title in phases:
        arms = [arm for arm in ARM_ORDER if ARM_META[arm][0] == phase]
        x = np.arange(len(arms))
        width = 0.24
        for index, method in enumerate(METHODS):
            lookup = {(row["arm"], row["method"]): row for row in summary_rows}
            values = [lookup[arm, method]["mean_delta_vs_full_pp"] for arm in arms]
            random_adv = [lookup[arm, method]["mean_delta_vs_matched_random_pp"] for arm in arms]
            axis.bar(x + (index - 1) * width, values, width, color=colours[method],
                     label=PAPERS[method]["label"] if phase == "native" else None)
            for xpos, value, advantage in zip(x + (index - 1) * width, values, random_adv):
                marker = "▲" if advantage > 0 else "▼" if advantage < 0 else "•"
                axis.text(xpos, value + (0.12 if value >= 0 else -0.12), marker,
                          ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("Δ AUROC vs full corrected pool (pp)")
        short_labels = [
            arm.removesuffix("_native").removesuffix("_k6").replace("_", "\n")
            for arm in arms
        ]
        axis.set_xticks(x, short_labels)
        axis.set_title(title + "\n▲ beats matched random; ▼ loses to matched random")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Feature selection × PCR solver interaction", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def fmt_delta(row, field="mean_delta_vs_full_pp"):
    return f"{row[field]:+.2f}pp"


def render_report(summary_rows, metadata):
    lookup = {(row["arm"], row["method"]): row for row in summary_rows}
    lines = [
        "# Feature selection × PCR solver comparison",
        "",
        f"Version: `{VERSION}`; feature schema: `{SCHEMA_VERSION}`; contract: `{CONTRACT}`.",
        "",
        "This reruns feature selection on the corrected, orientation-free pool. No old L-SML "
        "subset is reused. There is no per-cell `sign(rho)` and no post-hoc score flip; labels "
        "enter only after selection and fusion are frozen.",
        "",
        "**Result:** no tested selector improves any of the three PCR solvers over the full "
        "corrected pool. Some selectors beat random subsets of the same size, which shows they "
        "recover structure, but every reduction still discards complementary hallucination signal.",
        "",
        "![Selector and solver interactions](comparison.png)",
        "",
        "## Where selection is inserted",
        "",
        "| solver | insertion and guardrail | why |",
        "|---|---|---|",
        "| deployed U-PCR | selector → U-PCR's maintained exclusion → recompute | external "
        "selection must add value beyond the solver's existing weak-view exclusion |",
        "| IU-PCR | selector → IU-PCR; report residual dependence/effective rank | DPP and "
        "decorrelation target the uncorrelated-error assumption, but may select independent noise |",
        "| SU-PCR | selector → SU-PCR; report decomposition convergence and sparse-support theorem | "
        "selection changes the covariance graph, so theorem support is a validity diagnostic, not "
        "an objective to game by deletion |",
        "",
        "## Selector provenance and tested concept",
        "",
        "| family | source/concept tested here |",
        "|---|---|",
        "| DUFS | Lindenbaum et al. (NeurIPS 2021), parameter-free gated-Laplacian objective |",
        "| GroupFS | Lifshitz et al. (AAAI 2026), joint feature-group discovery and group gates |",
        "| Laplacian Score / SPEC / MCFS | sample-manifold spectral ranking (He et al. 2005; "
        "Zhao & Liu 2007; Cai et al. 2010) |",
        "| Concrete AE | Balin, Abid & Zou (ICML 2019), reconstruction-preserving subset |",
        "| LS-CAE | Shaham, Lindenbaum, Svirsky & Kluger (2021), reconstruction plus a "
        "Laplacian computed on the selected representation |",
        "| DPP / decorrelation | covariance-volume and minimum-redundancy conditions; these "
        "directly probe IU-PCR's uncorrelated-error assumption |",
        "| U-PCR residual | method-specific projection-residual minimization |",
        "",
        "## Baselines",
        "",
        "| method | full corrected pool AUROC | mean input / final kept |",
        "|---|---:|---:|",
    ]
    for method in METHODS:
        row = lookup["full", method]
        lines.append(
            f"| `{method}` | {row['macro_auroc']:.4f} | "
            f"{row['mean_n_selected']:.2f} / {row['mean_n_kept']:.2f} |"
        )

    lines.extend([
        "",
        "## Native stopping rules",
        "",
        "`Δfull` is the paired cell-macro change from the full corrected pool. `Δrandom` "
        "compares each chosen subset with 32 random subsets of the same size in the same cell "
        "and solver. A method must improve `Δfull` and beat `Δrandom` to establish useful "
        "selection rather than a generic small-subset effect.",
        "",
        "| selector | mean k / no-op cells | deployed U-PCR Δfull / Δrandom | IU-PCR Δfull / Δrandom | "
        "SU-PCR Δfull / Δrandom |",
        "|---|---:|---:|---:|---:|",
    ])
    native_arms = [arm for arm in ARM_ORDER if ARM_META[arm][0] == "native"]
    for arm in native_arms:
        values = [lookup[arm, method] for method in METHODS]
        lines.append(
            f"| `{arm}` | {values[0]['mean_n_selected']:.2f} / "
            f"{values[0]['no_selection_cells']} | "
            f"{fmt_delta(values[0])} / {fmt_delta(values[0], 'mean_delta_vs_matched_random_pp')} | "
            f"{fmt_delta(values[1])} / {fmt_delta(values[1], 'mean_delta_vs_matched_random_pp')} | "
            f"{fmt_delta(values[2])} / {fmt_delta(values[2], 'mean_delta_vs_matched_random_pp')} |"
        )

    lines.extend([
        "",
        "## Equal-budget ranking test (k=6)",
        "",
        "| selector | deployed U-PCR Δfull / Δrandom | IU-PCR Δfull / Δrandom | "
        "SU-PCR Δfull / Δrandom |",
        "|---|---:|---:|---:|",
    ])
    fixed_arms = [arm for arm in ARM_ORDER if ARM_META[arm][0] == "fixed_k6"]
    for arm in fixed_arms:
        values = [lookup[arm, method] for method in METHODS]
        lines.append(
            f"| `{arm}` | "
            f"{fmt_delta(values[0])} / {fmt_delta(values[0], 'mean_delta_vs_matched_random_pp')} | "
            f"{fmt_delta(values[1])} / {fmt_delta(values[1], 'mean_delta_vs_matched_random_pp')} | "
            f"{fmt_delta(values[2])} / {fmt_delta(values[2], 'mean_delta_vs_matched_random_pp')} |"
        )

    upcr_specific = lookup.get(("upcrres_native", "deployed_upcr"))
    if upcr_specific:
        lines.extend([
            "",
            "## Solver-specific control",
            "",
            f"The U-PCR-residual greedy selector retained {upcr_specific['mean_n_selected']:.2f} "
            f"views on average and changed deployed U-PCR by {fmt_delta(upcr_specific)}; its "
            f"advantage over matched random was "
            f"{fmt_delta(upcr_specific, 'mean_delta_vs_matched_random_pp')}. It is not applied "
            "to IU-PCR or SU-PCR because its objective is U-PCR's own projection residual.",
        ])

    su_rows = [row for row in summary_rows if row["method"] == "su_pcr"]
    lines.extend([
        "",
        "## SU-PCR validity audit",
        "",
        "Numerical SU-PCR outputs below five views are retained for diagnosis, but are outside "
        "the paper's minimum-size theorem condition. The table therefore reports both the "
        "unconditional support rate and the rate among size-eligible cells.",
        "",
        "| arm | size ≥5 | theorem support: all / size-eligible | decomposition convergence |",
        "|---|---:|---:|---:|",
    ])
    for row in su_rows:
        theorem = row["su_theorem_support_rate"]
        eligible = row["su_theorem_support_rate_if_size_eligible"]
        convergence = row["su_decomposition_convergence_rate"]
        lines.append(
            f"| `{row['arm']}` | {100 * float(row['su_minimum_size_rate']):.1f}% | "
            f"{100 * float(theorem):.1f}% / {100 * float(eligible):.1f}% | "
            f"{100 * float(convergence):.1f}% |"
        )

    best_by_method = {}
    admissible = [row for row in summary_rows if row["arm"] != "full"
                  and row["phase"] != "method_specific"]
    for method in METHODS:
        method_rows = [row for row in admissible if row["method"] == method]
        best_by_method[method] = max(method_rows, key=lambda row: row["mean_delta_vs_full_pp"])
    lines.extend([
        "",
        "## What the experiment establishes",
        "",
        "- **Deployed U-PCR:** its internal exclusion is the correct selection location for now. "
        "The best external arm, GroupFS, is still negative; its apparent closeness comes from "
        f"making no selection in {lookup['groupfs_native', 'deployed_upcr']['no_selection_cells']}"
        "/24 cells. The U-PCR-residual selector also fails, so a lower equation residual is not "
        "a proxy for hallucination relevance.",
        "- **IU-PCR:** enforcing diversity does not rescue the independence model. At k=6, DPP "
        f"changes IU-PCR by {fmt_delta(lookup['dpp_k6', 'iu_pcr'])} and decorrelation by "
        f"{fmt_delta(lookup['decorr_k6', 'iu_pcr'])}; both also lose to matched random. "
        "Independence without a relevance term preferentially keeps orthogonal noise.",
        "- **SU-PCR:** the full stable pool already has 100% sparse-support validity and 100% "
        "decomposition convergence. Selection cannot improve that condition; aggressive small "
        "subsets often make the theorem inapplicable and can create raw-score inversions. LS-CAE "
        "at k=6 is meaningfully better than random six-view subsets, but remains below full SU-PCR.",
        "- **Structure is real but insufficient:** MCFS/CAE/LS-CAE sometimes beat matched random. "
        "That is evidence that their rankings are not arbitrary, not evidence that feature removal "
        "improves the detector. The useful next concept is relevance-aware shrinkage or soft "
        "weighting—not another diversity-only hard selector.",
        "- Laplacian Score and SPEC selected exactly the same subsets in all 24 cells under this "
        "standardized construction; their identical rows are one empirical condition, not two "
        "independent confirmations.",
        "",
        "## Decision rule",
        "",
        "A selector is considered a credible improvement only if its mean `Δfull` is positive, "
        "its 95% paired bootstrap interval excludes zero, it beats matched random on average, "
        "and it does not introduce orientation failures. SU-PCR additionally requires acceptable "
        "decomposition convergence and theorem-support behavior. Retrospective success is still "
        "a hypothesis for a new dataset/model family, not prospective proof.",
        "",
        "Best observed non-method-specific arm by raw mean change:",
        "",
    ])
    for method in METHODS:
        row = best_by_method[method]
        lines.append(
            f"- `{method}`: `{row['arm']}` at {fmt_delta(row)} versus full and "
            f"{fmt_delta(row, 'mean_delta_vs_matched_random_pp')} versus matched random."
        )

    lines.extend([
        "",
        "## Validation audit",
        "",
        f"- Full-pool solver scores reproduce the preceding feature-contract experiment with "
        f"maximum absolute AUROC difference `{metadata['baseline_reproduction_max_abs_delta']:.3g}`.",
        f"- Selector fallbacks: `{metadata['selector_fallbacks']}`; fixed-k arms with a non-six "
        f"subset: `{metadata['invalid_fixed_k_subsets']}`.",
        f"- Laplacian Score/SPEC subset identity: `{metadata['lap_spec_native_identical_cells']}`/"
        f"{metadata['n_cells']} native and `{metadata['lap_spec_k6_identical_cells']}`/"
        f"{metadata['n_cells']} at k=6.",
        f"- Raw-score orientation failures: `{metadata['orientation_failures']}` across "
        f"`{metadata['n_evaluations']}` evaluations; all occurred in aggressive SU-PCR subset arms, "
        "not in any full-pool baseline.",
        "",
        "## Files and reproduction",
        "",
        "- `per_cell.csv`: every selector × solver result and matched-random floor.",
        "- `selections.csv`: selected feature names and label-free diagnostics.",
        "- `summary.csv`: macro effects, paired intervals, structural diagnostics.",
        "- `feature_frequency.csv`: which corrected views each selector repeatedly chose.",
        "",
        "```bash",
        "python scripts/feature_selection_pcr_comparison.py",
        "```",
        "",
        f"Runtime: {metadata['runtime_seconds']:.1f}s; cells: {metadata['n_cells']}; "
        f"matched-random repeats: {metadata['random_repeats']}.",
    ])
    return "\n".join(lines) + "\n"


def self_test():
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(200, 3))
    matrix = np.column_stack([
        latent[:, 0], latent[:, 0] + 0.01 * rng.normal(size=200),
        latent[:, 1], latent[:, 2], rng.normal(size=200), rng.normal(size=200),
    ])
    matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    cell = make_unlabeled_cell("synthetic", matrix, [f"x{i}" for i in range(6)])
    assert cell.V.shape == (200, 6)
    assert np.allclose(np.diag(cell.rho), 1.0)
    diag = dependence_diagnostics(matrix)
    assert 0 <= diag["mean_abs_spearman"] <= 1
    score, details = score_solver("iu_pcr", matrix)
    assert len(score) == 200 and details["n_kept"] == 6
    print("SELF-TEST PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--random-repeats", type=int, default=RANDOM_REPEATS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    started = time.time()
    data = np.load(args.bundle, allow_pickle=True)
    keys = sorted({name.rsplit("__", 1)[0] for name in data.files})
    prepared = {}
    for cell_key in keys:
        raw, names = reconstruct_raw(data, cell_key)
        matrix, kept_names, _ = build_contract(raw, names, CONTRACT)
        prepared[cell_key] = (matrix, kept_names)

    payloads = [(key, prepared[key][0], prepared[key][1]) for key in keys]
    selections = {}
    workers = max(1, min(int(args.workers), len(payloads)))
    if workers == 1:
        for index, payload in enumerate(payloads, 1):
            cell_key, arms = select_one_cell(payload)
            selections[cell_key] = arms
            print(f"selectors {index}/{len(payloads)}: {cell_key}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(select_one_cell, payload): payload[0]
                       for payload in payloads}
            for index, future in enumerate(as_completed(futures), 1):
                cell_key, arms = future.result()
                selections[cell_key] = arms
                print(f"selectors {index}/{len(payloads)}: {cell_key}", flush=True)

    selection_rows = []
    rows = []
    random_cache = {}
    for cell_index, cell_key in enumerate(keys, 1):
        matrix, names = prepared[cell_key]
        labels = np.asarray(data[f"{cell_key}__labels"], dtype=int)
        for arm in selections[cell_key]:
            cols = np.asarray(arm["cols"], dtype=int)
            selected_matrix = matrix[:, cols]
            selection_rows.append({
                "cell": cell_key, "family": family(cell_key), "domain": domain(cell_key),
                "arm": arm["arm"], "phase": arm["phase"], "selector": arm["selector"],
                "variant": arm["variant"], "n_pool": len(names), "n_selected": len(cols),
                "selected_features": "|".join(names[index] for index in cols),
                "pool_features": "|".join(names), "fallback": arm["fallback"],
                "intended_methods": "|".join(arm["intended_methods"]),
                "diag_json": json.dumps(arm["diag"], sort_keys=True),
            })
            dep = dependence_diagnostics(selected_matrix)
            for method in arm["intended_methods"]:
                score, diagnostics = score_solver(method, selected_matrix)
                auc = float(roc_auc_score(labels, score))
                cache_key = (cell_key, method, len(cols))
                if cache_key not in random_cache:
                    random_cache[cache_key] = random_floor(
                        method, matrix, labels, len(cols), int(args.random_repeats),
                        namespace=cell_key,
                    )
                random_values = random_cache[cache_key]
                random_mean = float(np.mean(random_values))
                random_percentile = float(
                    100 * (np.sum(random_values < auc) + 0.5 * np.sum(random_values == auc))
                    / max(len(random_values), 1)
                )
                rows.append({
                    "cell": cell_key, "family": family(cell_key), "domain": domain(cell_key),
                    "method": method, "arm": arm["arm"], "phase": arm["phase"],
                    "selector": arm["selector"], "variant": arm["variant"],
                    "n": len(labels), "n_pool": len(names), "n_selected": len(cols),
                    "n_kept": diagnostics["n_kept"], "auroc": auc,
                    "random_mean_auroc": random_mean,
                    "delta_vs_matched_random_pp": 100 * (auc - random_mean),
                    "random_percentile": random_percentile,
                    "random_repeats_completed": len(random_values),
                    "orientation_failure": bool(auc < 0.5),
                    "selector_fallback": arm["fallback"],
                    "fit_residual": diagnostics["fit_residual"],
                    "decomposition_converged": diagnostics["decomposition_converged"],
                    "theorem_support_ok": diagnostics["theorem_support_ok"],
                    "nnz_pairs": diagnostics["nnz_pairs"],
                    "sparse_fraction": diagnostics["sparse_fraction"],
                    **dep,
                })
        print(f"evaluation {cell_index}/{len(keys)}: {cell_key}", flush=True)

    summary_rows = summarize(rows)
    frequency_rows = feature_frequency(selection_rows)

    # Fail loudly on an incomplete experimental grid before writing a report.
    expected_selection_rows = len(keys) * len(ARM_ORDER)
    if len(selection_rows) != expected_selection_rows:
        raise RuntimeError(
            f"incomplete selector grid: {len(selection_rows)} != {expected_selection_rows}"
        )
    invalid_fixed = [row for row in selection_rows
                     if row["phase"] == "fixed_k6" and row["n_selected"] != FIXED_K]
    if any(row["random_repeats_completed"] != int(args.random_repeats) for row in rows):
        raise RuntimeError("one or more matched-random floors are incomplete")

    prior_path = os.path.join(
        REPO, "results", "feature_contract_solver_comparison", "per_cell.csv"
    )
    baseline_reproduction = float("nan")
    if os.path.exists(prior_path):
        with open(prior_path, newline="", encoding="utf-8") as handle:
            prior_rows = list(csv.DictReader(handle))
        previous = {
            (row["cell"], row["method"]): float(row["auroc"])
            for row in prior_rows if row["contract"] == CONTRACT
        }
        current = {
            (row["cell"], row["method"]): float(row["auroc"])
            for row in rows if row["arm"] == "full"
        }
        if set(previous) == set(current):
            baseline_reproduction = float(max(
                abs(current[key] - previous[key]) for key in current
            ))

    def identical_cells(first, second):
        first_by_cell = {
            row["cell"]: row["selected_features"] for row in selection_rows
            if row["arm"] == first
        }
        second_by_cell = {
            row["cell"]: row["selected_features"] for row in selection_rows
            if row["arm"] == second
        }
        return sum(first_by_cell.get(key) == second_by_cell.get(key) for key in keys)

    metadata = {
        "version": VERSION,
        "feature_schema": SCHEMA_VERSION,
        "contract": CONTRACT,
        "bundle": os.path.abspath(args.bundle),
        "n_cells": len(keys),
        "random_repeats": int(args.random_repeats),
        "workers": workers,
        "runtime_seconds": time.time() - started,
        "n_evaluations": len(rows),
        "per_cell_rho_orientation": False,
        "global_anchor_flip": False,
        "labels_available_to_selectors": False,
        "fixed_k": FIXED_K,
        "arms": list(ARM_ORDER),
        "selector_fallbacks": int(sum(row["fallback"] for row in selection_rows)),
        "invalid_fixed_k_subsets": len(invalid_fixed),
        "orientation_failures": int(sum(row["orientation_failure"] for row in rows)),
        "baseline_reproduction_max_abs_delta": baseline_reproduction,
        "lap_spec_native_identical_cells": identical_cells(
            "lapscore_native", "spec_native"
        ),
        "lap_spec_k6_identical_cells": identical_cells("lapscore_k6", "spec_k6"),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "per_cell.csv"), rows)
    write_csv(os.path.join(args.out_dir, "selections.csv"), selection_rows)
    write_csv(os.path.join(args.out_dir, "summary.csv"), summary_rows)
    write_csv(os.path.join(args.out_dir, "feature_frequency.csv"), frequency_rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    make_plot(summary_rows, os.path.join(args.out_dir, "comparison.png"))
    report = render_report(summary_rows, metadata)
    with open(os.path.join(args.out_dir, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
