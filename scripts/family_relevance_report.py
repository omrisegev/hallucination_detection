#!/usr/bin/env python3
"""Verify the score freeze, then evaluate graph-coupled family relevance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
import types

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "hallucination_detection_mpl")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from scripts.family_relevance_fit import (  # noqa: E402
    BETAS,
    BLENDS,
    CONTEXTS,
    CONTEXT_BINS,
    CONTEXT_PERMUTATIONS,
    DEFAULT_BUNDLE,
    DEFAULT_OUT,
    PRIMARY_BETA,
    PRIMARY_BLEND,
    VERSION,
    dependency_version,
    sha256_file,
)
from scripts.inscope_cells import GROUP, INSCOPE  # noqa: E402


FAMILY_NAMES = (
    "triviaqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
    "truthfulqa", "gsm8k", "math500",
)
PRIMARY = f"manual_graph__beta_{PRIMARY_BETA:g}__blend_{PRIMARY_BLEND:g}"
NO_GRAPH = f"manual_graph__beta_0__blend_{PRIMARY_BLEND:g}"
PERMUTED_GRAPH = f"permuted_graph__beta_{PRIMARY_BETA:g}__blend_{PRIMARY_BLEND:g}"
GLOBAL_GATE = f"global_gate__beta_{PRIMARY_BETA:g}__blend_{PRIMARY_BLEND:g}"
SAMPLE_PERMUTED = (
    f"sample_permuted_gate__beta_{PRIMARY_BETA:g}__blend_{PRIMARY_BLEND:g}"
)
HEADLINE = (
    "deployed_upcr", "iu_pcr", "dufs_liu", NO_GRAPH, PERMUTED_GRAPH,
    GLOBAL_GATE, SAMPLE_PERMUTED, PRIMARY,
)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dataset_family(cell):
    return next((name for name in FAMILY_NAMES if name in cell), cell)


def stable_seed(namespace):
    return int(hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:8], 16)


def metric(labels, scores, function):
    try:
        return float(function(labels, scores))
    except ValueError:
        return float("nan")


def verify_freeze(out_dir, bundle):
    with open(os.path.join(out_dir, "RUN_DEFINITION.json"), encoding="utf-8") as handle:
        definition = json.load(handle)
    with open(os.path.join(out_dir, "FIT_COMPLETE.json"), encoding="utf-8") as handle:
        complete = json.load(handle)
    if not definition.get("scientific_run") or not complete.get("scientific_run"):
        raise RuntimeError("debug output cannot be evaluated")
    if definition.get("version") != VERSION or complete.get("version") != VERSION:
        raise RuntimeError("fit/report version mismatch")
    if definition.get("run_fingerprint") != complete.get("run_fingerprint"):
        raise RuntimeError("run fingerprints disagree")
    if tuple(definition.get("cells", ())) != tuple(INSCOPE) or complete.get("n_cells") != 24:
        raise RuntimeError("fit does not contain the exact 24-cell roster")
    if sha256_file(bundle) != definition.get("bundle_sha256"):
        raise RuntimeError("input bundle changed")
    stripped = os.path.join(REPO, definition["label_free_fit_bundle"])
    if sha256_file(stripped) != definition.get("label_free_fit_bundle_sha256"):
        raise RuntimeError("label-free fit bundle changed")
    synthetic = os.path.join(REPO, definition["synthetic_decision"])
    if sha256_file(synthetic) != definition.get("synthetic_decision_sha256"):
        raise RuntimeError("synthetic calibration decision changed")
    if platform.python_version() != definition.get("python"):
        raise RuntimeError("Python version differs from fit")
    if np.__version__ != definition.get("numpy"):
        raise RuntimeError("NumPy version differs from fit")
    for name, expected in definition.get("dependencies", {}).items():
        if dependency_version(name) != expected:
            raise RuntimeError(f"dependency version differs: {name}")
    for relative, expected in definition.get("source_sha256", {}).items():
        path = os.path.join(REPO, relative)
        if not os.path.exists(path) or sha256_file(path) != expected:
            raise RuntimeError(f"registered source changed after fit: {relative}")
    manifest = complete.get("artifact_manifest", [])
    if [row.get("cell") for row in manifest] != list(INSCOPE):
        raise RuntimeError("artifact roster/order mismatch")
    reference_dir = os.path.join(REPO, definition["reference_score_dir"])
    for row in manifest:
        cell = row["cell"]
        reference_path = os.path.join(reference_dir, f"{cell}.npz")
        if sha256_file(reference_path) != definition["reference_score_sha256"][cell]:
            raise RuntimeError(f"frozen reference changed: {cell}")
        for key, hash_key in (
            ("score_file", "score_sha256"),
            ("diagnostic_file", "diagnostic_sha256"),
        ):
            path = os.path.join(out_dir, row[key])
            if sha256_file(path) != row[hash_key]:
                raise RuntimeError(f"artifact changed: {cell}/{key}")
    freeze = {
        "version": VERSION,
        "run_fingerprint": definition["run_fingerprint"],
        "bundle_sha256": definition["bundle_sha256"],
        "artifact_manifest": manifest,
    }
    freeze_path = os.path.join(out_dir, "SCORE_FREEZE_MANIFEST.json")
    if os.path.exists(freeze_path):
        with open(freeze_path, encoding="utf-8") as handle:
            if json.load(handle) != freeze:
                raise RuntimeError("existing score-freeze manifest differs")
    else:
        write_json(freeze_path, freeze)
    return definition


def equal_family_values(rows, method, baseline="iu_pcr"):
    local = [row for row in rows if row["method"] == method]
    baseline_map = {
        row["cell"]: row["auroc"] for row in rows if row["method"] == baseline
    }
    cell_delta = {
        row["cell"]: 100.0 * (row["auroc"] - baseline_map[row["cell"]])
        for row in local
    }
    values = np.asarray([
        np.mean([value for cell, value in cell_delta.items()
                 if dataset_family(cell) == family])
        for family in FAMILY_NAMES
    ], dtype=float)
    return values, cell_delta


def bootstrap_ci(values, namespace, count=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_seed(namespace))
    draws = values[rng.integers(0, len(values), size=(count, len(values)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))


def context_headroom(labels, experts, context, *, bins=4):
    labels = np.asarray(labels, dtype=int)
    experts = np.asarray(experts, dtype=float)
    ranks = (rankdata(np.asarray(context, dtype=float), method="average") - 1.0)
    ranks /= max(len(ranks) - 1, 1)
    assignment = np.minimum((bins * ranks).astype(int), bins - 1)
    auc_rows = []
    winners = []
    for bucket in range(bins):
        keep = assignment == bucket
        if np.sum(labels[keep] == 0) < 3 or np.sum(labels[keep] == 1) < 3:
            continue
        aucs = np.asarray([
            roc_auc_score(labels[keep], score[keep]) for score in experts
        ], dtype=float)
        auc_rows.append(aucs)
        winners.append(int(np.argmax(aucs)))
    if len(auc_rows) < 2:
        return float("nan"), 0, []
    matrix = np.asarray(auc_rows)
    conditional = float(np.mean(np.max(matrix, axis=1)))
    fixed = float(np.max(np.mean(matrix, axis=0)))
    return 100.0 * (conditional - fixed), len(set(winners)), winners


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def make_plots(out_dir, metric_rows, summary_rows, sensitivity_rows, context_rows):
    figure_dir = os.path.join(out_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    display = {
        "deployed_upcr": "deployed U-PCR",
        "iu_pcr": "IU-PCR",
        "dufs_liu": "DUFS-LIU",
        NO_GRAPH: "family gate, no graph",
        PERMUTED_GRAPH: "permuted family graph",
        GLOBAL_GATE: "global family gate",
        SAMPLE_PERMUTED: "permuted local gate",
        PRIMARY: "GCFR-U-PCR",
    }
    selected = [next(row for row in summary_rows if row["method"] == method) for method in HEADLINE]
    plt.figure(figsize=(11, 5))
    plt.bar(range(len(selected)), [row["mean_delta_pp"] for row in selected])
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(range(len(selected)), [display[row["method"]] for row in selected], rotation=28, ha="right")
    plt.ylabel("Mean AUROC change vs IU-PCR (pp)")
    plt.title("Does graph-coupled local family relevance help?")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "headline_methods.png"), dpi=180)
    plt.close()

    primary_rows = [row for row in metric_rows if row["method"] == PRIMARY]
    primary_rows.sort(key=lambda row: row["delta_pp"])
    colors = ["#e76f51" if row["delta_pp"] < 0 else "#2a9d8f" for row in primary_rows]
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(primary_rows)), [row["delta_pp"] for row in primary_rows], color=colors)
    plt.yticks(range(len(primary_rows)), [row["cell"] for row in primary_rows], fontsize=8)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("GCFR-U-PCR change vs IU-PCR (pp)")
    plt.title("Primary path in each dataset/model cell")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "primary_per_cell.png"), dpi=180)
    plt.close()

    grid = np.full((len(BETAS), len(BLENDS)), np.nan)
    for row in sensitivity_rows:
        grid[list(BETAS).index(row["beta"]), list(BLENDS).index(row["blend"])] = row["mean_delta_pp"]
    plt.figure(figsize=(7, 5))
    image = plt.imshow(grid, cmap="coolwarm", vmin=-max(abs(np.nanmin(grid)), abs(np.nanmax(grid))),
                       vmax=max(abs(np.nanmin(grid)), abs(np.nanmax(grid))))
    plt.colorbar(image, label="Mean change vs IU-PCR (pp)")
    plt.xticks(range(len(BLENDS)), BLENDS)
    plt.yticks(range(len(BETAS)), BETAS)
    plt.xlabel("local-gate blend alpha")
    plt.ylabel("family-graph strength beta")
    plt.title("Frozen GCFR sensitivity path")
    for i in range(len(BETAS)):
        for j in range(len(BLENDS)):
            plt.text(j, i, f"{grid[i,j]:+.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "beta_blend_sensitivity.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    labels = [row["context"].replace("context_", "") for row in context_rows]
    plt.bar(range(len(context_rows)), [row["headroom_pp"] for row in context_rows])
    plt.axhline(0.5, color="black", linestyle="--", label="registered 0.5pp gate")
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel("Conditional family-oracle headroom (pp)")
    plt.title("Do frozen contexts organize family specialization?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "context_headroom.png"), dpi=180)
    plt.close()

    synthetic_path = os.path.join(REPO, "results", "family_relevance_synthetic", "summary.csv")
    if os.path.exists(synthetic_path):
        with open(synthetic_path, encoding="utf-8") as handle:
            synthetic = list(csv.DictReader(handle))
        methods = [PRIMARY.replace("manual_graph", "manual_graph"),
                   f"permuted_graph__beta_{PRIMARY_BETA:g}__blend_{PRIMARY_BLEND:g}",
                   f"global_gate__beta_{PRIMARY_BETA:g}__blend_{PRIMARY_BLEND:g}",
                   f"sample_permuted_gate__beta_{PRIMARY_BETA:g}__blend_{PRIMARY_BLEND:g}"]
        labels_method = ["correct graph", "permuted graph", "global", "sample-permuted"]
        x = np.arange(len(methods)); width = 0.36
        plt.figure(figsize=(9, 4.8))
        for offset, scenario in ((-width/2, "independent_noise"), (width/2, "correlated_nuisance")):
            values = [float(next(row for row in synthetic if row["scenario"] == scenario and row["method"] == method)["mean_delta_pp"]) for method in methods]
            plt.bar(x + offset, values, width, label=scenario.replace("_", " "))
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(x, labels_method, rotation=18, ha="right")
        plt.ylabel("Mean AUROC change vs IU-PCR (pp)")
        plt.title("Synthetic mechanism and explicit failure world")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, "synthetic_boundary.png"), dpi=180)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    bundle = os.path.abspath(args.bundle)
    definition = verify_freeze(out_dir, bundle)

    # Correctness is opened only after verify_freeze created/checked the score manifest.
    data = np.load(bundle, allow_pickle=True)
    metric_rows = []
    context_cell = {context: [] for context in CONTEXTS}
    diagnostic_rows = []
    for cell in INSCOPE:
        labels = np.asarray(data[f"{cell}__labels"], dtype=int)
        with np.load(os.path.join(out_dir, "scores", f"{cell}.npz"), allow_pickle=False) as scores:
            iu_auc = roc_auc_score(labels, scores["iu_pcr"])
            methods = list(HEADLINE) + [
                f"manual_graph__beta_{beta:g}__blend_{blend:g}"
                for beta in BETAS for blend in BLENDS
                if f"manual_graph__beta_{beta:g}__blend_{blend:g}" not in HEADLINE
            ]
            for method in methods:
                auc = metric(labels, scores[method], roc_auc_score)
                auprc = metric(labels, scores[method], average_precision_score)
                metric_rows.append({
                    "cell": cell,
                    "domain": GROUP[cell],
                    "family": dataset_family(cell),
                    "method": method,
                    "auroc": auc,
                    "auprc": auprc,
                    "delta_pp": 100.0 * (auc - iu_auc),
                })
            primary_score = np.asarray(scores[PRIMARY], dtype=float)
            primary_gates = np.asarray(
                scores[f"family_gates__beta_{PRIMARY_BETA:g}"], dtype=float
            )
            diagnostic_rows.append({
                "cell": cell,
                "domain": GROUP[cell],
                "family": dataset_family(cell),
                "mean_gate_std_across_families": float(np.mean(np.std(primary_gates, axis=1))),
                "mean_gate_std_across_samples": float(np.mean(np.std(primary_gates, axis=0))),
                "mean_abs_rank_change": float(np.mean(np.abs(
                    rankdata(primary_score) - rankdata(scores["iu_pcr"])
                )) / len(labels)),
            })
            experts = np.asarray(scores["family_experts"], dtype=float)
            family_names = tuple(map(str, scores["family_names"]))
            for context in CONTEXTS:
                observed, winner_count, winners = context_headroom(
                    labels, experts, scores[context], bins=CONTEXT_BINS
                )
                null = []
                rng = np.random.default_rng(stable_seed(f"context:{context}:{cell}"))
                context_values = np.asarray(scores[context], dtype=float)
                for _ in range(CONTEXT_PERMUTATIONS):
                    value, _, _ = context_headroom(
                        labels, experts, context_values[rng.permutation(len(labels))],
                        bins=CONTEXT_BINS,
                    )
                    null.append(value)
                context_cell[context].append({
                    "cell": cell,
                    "family": dataset_family(cell),
                    "headroom_pp": observed,
                    "winner_count": winner_count,
                    "winners": [family_names[index] for index in winners],
                    "null": np.asarray(null, dtype=float),
                })

    summary_rows = []
    for method in HEADLINE:
        values, cell_delta = equal_family_values(metric_rows, method)
        local = [row for row in metric_rows if row["method"] == method]
        cell_values = np.asarray([row["delta_pp"] for row in local])
        ci = bootstrap_ci(values, f"family-relevance:{method}")
        summary_rows.append({
            "method": method,
            "cell_macro_auroc": float(np.mean([row["auroc"] for row in local])),
            "mean_delta_pp": float(np.mean(cell_values)),
            "family_delta_pp": float(np.mean(values)),
            "family_ci_low": ci[0],
            "family_ci_high": ci[1],
            "wins": int(np.sum(cell_values > 1e-12)),
            "losses": int(np.sum(cell_values < -1e-12)),
            "worst_delta_pp": float(np.min(cell_values)),
        })

    sensitivity_rows = []
    for beta in BETAS:
        for blend in BLENDS:
            method = f"manual_graph__beta_{beta:g}__blend_{blend:g}"
            local = [row for row in metric_rows if row["method"] == method]
            values = np.asarray([row["delta_pp"] for row in local])
            sensitivity_rows.append({
                "beta": float(beta),
                "blend": float(blend),
                "mean_delta_pp": float(np.mean(values)),
                "wins": int(np.sum(values > 1e-12)),
                "losses": int(np.sum(values < -1e-12)),
                "worst_delta_pp": float(np.min(values)),
            })

    context_rows = []
    raw_p = []
    for context in CONTEXTS:
        valid = [row for row in context_cell[context] if np.isfinite(row["headroom_pp"])]
        family_observed = []
        family_null = []
        for family in FAMILY_NAMES:
            local = [row for row in valid if row["family"] == family]
            if not local:
                continue
            family_observed.append(float(np.mean([row["headroom_pp"] for row in local])))
            family_null.append(np.nanmean(np.asarray([row["null"] for row in local]), axis=0))
        observed = float(np.mean(family_observed)) if family_observed else 0.0
        null = np.nanmean(np.asarray(family_null), axis=0) if family_null else np.zeros(CONTEXT_PERMUTATIONS)
        null = np.where(np.isfinite(null), null, np.inf)
        pvalue = float((1 + np.sum(null >= observed - 1e-12)) / (len(null) + 1))
        raw_p.append(pvalue)
        context_rows.append({
            "context": context,
            "valid_cells": len(valid),
            "headroom_pp": observed,
            "permutation_p": pvalue,
            "mean_distinct_winners": float(np.mean([row["winner_count"] for row in valid])) if valid else 0.0,
        })
    adjusted = holm_adjust(raw_p)
    for row, value in zip(context_rows, adjusted):
        row["holm_p"] = float(value)
        row["supports_specialization"] = bool(
            row["headroom_pp"] >= 0.5 and row["holm_p"] <= 0.05
        )

    summary_map = {row["method"]: row for row in summary_rows}
    primary = summary_map[PRIMARY]
    gates = [
        {"gate": "primary mean change versus IU-PCR > 0", "observed": primary["mean_delta_pp"], "passed": primary["mean_delta_pp"] > 0},
        {"gate": "primary equal-family lower bound > 0", "observed": primary["family_ci_low"], "passed": primary["family_ci_low"] > 0},
        {"gate": "primary improves at least 14 of 24 cells", "observed": primary["wins"], "passed": primary["wins"] >= 14},
        {"gate": "primary worst loss no worse than -2pp", "observed": primary["worst_delta_pp"], "passed": primary["worst_delta_pp"] >= -2.0},
        {"gate": "primary beats beta=0 family gate", "observed": primary["mean_delta_pp"] - summary_map[NO_GRAPH]["mean_delta_pp"], "passed": primary["mean_delta_pp"] > summary_map[NO_GRAPH]["mean_delta_pp"]},
        {"gate": "primary beats permuted family graph", "observed": primary["mean_delta_pp"] - summary_map[PERMUTED_GRAPH]["mean_delta_pp"], "passed": primary["mean_delta_pp"] > summary_map[PERMUTED_GRAPH]["mean_delta_pp"]},
        {"gate": "primary beats global family gate", "observed": primary["mean_delta_pp"] - summary_map[GLOBAL_GATE]["mean_delta_pp"], "passed": primary["mean_delta_pp"] > summary_map[GLOBAL_GATE]["mean_delta_pp"]},
        {"gate": "primary beats sample-permuted local gate", "observed": primary["mean_delta_pp"] - summary_map[SAMPLE_PERMUTED]["mean_delta_pp"], "passed": primary["mean_delta_pp"] > summary_map[SAMPLE_PERMUTED]["mean_delta_pp"]},
        {"gate": "primary beats frozen DUFS-LIU", "observed": primary["mean_delta_pp"] - summary_map["dufs_liu"]["mean_delta_pp"], "passed": primary["mean_delta_pp"] > summary_map["dufs_liu"]["mean_delta_pp"]},
        {"gate": "at least one frozen context supports specialization", "observed": int(sum(row["supports_specialization"] for row in context_rows)), "passed": any(row["supports_specialization"] for row in context_rows)},
    ]
    all_passed = bool(all(row["passed"] for row in gates))

    write_csv(os.path.join(out_dir, "metrics.csv"), metric_rows)
    write_csv(os.path.join(out_dir, "headline_summary.csv"), summary_rows)
    write_csv(os.path.join(out_dir, "sensitivity.csv"), sensitivity_rows)
    write_csv(os.path.join(out_dir, "context_specialization.csv"), context_rows)
    write_csv(os.path.join(out_dir, "gate_diagnostics.csv"), diagnostic_rows)
    write_json(os.path.join(out_dir, "CONTINUATION_GATES.json"), {
        "all_gates_passed": all_passed,
        "gates": gates,
    })
    make_plots(out_dir, metric_rows, summary_rows, sensitivity_rows, context_rows)

    decision = "CONTINUE TO LEARNED MIXTURE" if all_passed else "STOP BEFORE LEARNED MIXTURE"
    lines = [
        "# Graph-coupled family relevance diagnostic", "",
        f"**Decision: {decision}.**", "",
        "## Terms", "",
        "- **Family gate:** a sample-specific weight shared by related features.",
        "- **Family graph:** prior knowledge about which measurement families are related.",
        "- **Conditional headroom:** the optimistic gain from choosing a different fixed family expert in each frozen context stratum.",
        "- **pp:** AUROC percentage points.", "",
        "## Synthetic boundary", "",
        "The selected path improved IU-PCR by +0.773pp with 20/20 wins when inactive family members had independent noise. It lost 9.272pp with 0/20 wins when inactive members shared a coherent nuisance. The gate can detect inconsistency; it cannot detect a consistently wrong family.", "",
        "## Real-data headline", "",
        "| method | macro AUROC | change vs IU | family change | wins/losses | worst |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['cell_macro_auroc']:.4f} | {row['mean_delta_pp']:+.3f}pp | "
            f"{row['family_delta_pp']:+.3f}pp | {row['wins']}/{row['losses']} | {row['worst_delta_pp']:+.3f}pp |"
        )
    lines += ["", "## Conditional specialization", "", "| context | valid cells | headroom | permutation p | Holm p | support |", "|---|---:|---:|---:|---:|:---:|"]
    for row in context_rows:
        lines.append(
            f"| {row['context']} | {row['valid_cells']} | {row['headroom_pp']:+.3f}pp | "
            f"{row['permutation_p']:.4f} | {row['holm_p']:.4f} | {'yes' if row['supports_specialization'] else 'no'} |"
        )
    lines += ["", "## Continuation gates", "", "| gate | observed | pass |", "|---|---:|:---:|"]
    for row in gates:
        lines.append(f"| {row['gate']} | {row['observed']:.4f} | {'yes' if row['passed'] else 'no'} |")
    best = max(sensitivity_rows, key=lambda row: row["mean_delta_pp"])
    lines += [
        "", "## Parameter diagnosis", "",
        f"The best descriptive frozen path was beta={best['beta']:g}, alpha={best['blend']:g}, with {best['mean_delta_pp']:+.3f}pp. The registered primary remains beta=3, alpha=1 regardless of this result.",
        "`beta` controls how strongly related family gates are smoothed. `alpha` controls how much the local gate replaces ordinary IU-PCR. A better post-label grid point is not a promoted method.",
        "", "## Interpretation", "",
    ]
    if all_passed:
        lines.append("The family prior, local gate, and frozen contexts all show the registered mechanism. A small learned mixture may be designed, but these reused cells cannot confirm it.")
    else:
        lines.append("The complete mechanism was not shown. Do not build a more flexible learned family mixture from these labels. Use the failed controls to decide whether the missing part is family prior, local routing, or target information.")
    lines += [
        "", "## Audit", "",
        f"Run fingerprint: `{definition['run_fingerprint']}`. All sources, inputs, frozen reference scores, and new score artifacts were verified before correctness labels were read. These 24 cells are retrospective development evidence.",
    ]
    report = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
