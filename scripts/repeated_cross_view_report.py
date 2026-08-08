#!/usr/bin/env python3
"""Verify the score freeze, then evaluate repeated cross-view diffusion."""

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
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from scripts.inscope_cells import GROUP, INSCOPE  # noqa: E402
from scripts.repeated_cross_view_fit import (  # noqa: E402
    DEFAULT_BUNDLE,
    DEFAULT_OUT,
    LAMBDAS,
    PARTITION_COUNT,
    PREFIX_COUNTS,
    PRIMARY_K,
    PRIMARY_LAMBDA,
    VERSION,
    dependency_version,
    sha256_file,
)
from spectral_utils.repeated_cross_view_diffusion import lambda_token  # noqa: E402


FAMILY_NAMES = (
    "triviaqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
    "truthfulqa", "gsm8k", "math500",
)
TOKEN = lambda_token(PRIMARY_LAMBDA)
PRIMARY = f"dependency_blocked__lambda_{TOKEN}"
ATOMIC = f"atomic_random__lambda_{TOKEN}"
FAMILY = f"family_blocked__lambda_{TOKEN}"
DIRECT = f"dependency_direct__lambda_{TOKEN}"
NODE_PERMUTED = f"dependency_node_permuted__lambda_{TOKEN}"
T4 = f"dependency_blocked_t4__lambda_{TOKEN}"
T8 = f"dependency_blocked_t8__lambda_{TOKEN}"
K5 = f"dependency_blocked_k5__lambda_{TOKEN}"
K11 = f"dependency_blocked_k11__lambda_{TOKEN}"
HEADLINE = (
    "deployed_upcr", "iu_pcr", "dufs_liu", "raw_uniform_liu",
    ATOMIC, FAMILY, T4, T8, DIRECT, NODE_PERMUTED, K5, K11, PRIMARY,
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
        if sha256_file(os.path.join(reference_dir, f"{cell}.npz")) != definition[
            "reference_score_sha256"
        ][cell]:
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
    path = os.path.join(out_dir, "SCORE_FREEZE_MANIFEST.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            if json.load(handle) != freeze:
                raise RuntimeError("existing score-freeze manifest differs")
    else:
        write_json(path, freeze)
    return definition


def equal_family_values(rows, method, baseline="iu_pcr"):
    baseline_map = {
        row["cell"]: row["auroc"] for row in rows if row["method"] == baseline
    }
    cell_delta = {
        row["cell"]: 100.0 * (row["auroc"] - baseline_map[row["cell"]])
        for row in rows if row["method"] == method
    }
    values = np.asarray([
        np.mean([value for cell, value in cell_delta.items()
                 if dataset_family(cell) == family])
        for family in FAMILY_NAMES
    ], dtype=float)
    return values, cell_delta


def bootstrap_ci(values, namespace, count=20_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_seed(namespace))
    draws = values[rng.integers(0, len(values), size=(count, len(values)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))


def summarize_methods(metric_rows, methods):
    output = []
    for method in methods:
        values, _ = equal_family_values(metric_rows, method)
        local = [row for row in metric_rows if row["method"] == method]
        deltas = np.asarray([row["delta_pp"] for row in local])
        ci = bootstrap_ci(values, f"rcv-ad:{method}")
        output.append({
            "method": method,
            "cell_macro_auroc": float(np.mean([row["auroc"] for row in local])),
            "mean_delta_pp": float(np.mean(deltas)),
            "family_delta_pp": float(np.mean(values)),
            "family_ci_low": ci[0],
            "family_ci_high": ci[1],
            "wins": int(np.sum(deltas > 1e-12)),
            "losses": int(np.sum(deltas < -1e-12)),
            "worst_delta_pp": float(np.min(deltas)),
        })
    return output


def make_plots(out_dir, metric_rows, summary_rows, lambda_rows, diagnostic_rows):
    figure_dir = os.path.join(out_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    labels = {
        "deployed_upcr": "deployed U-PCR",
        "iu_pcr": "IU-PCR",
        "dufs_liu": "DUFS-LIU",
        "raw_uniform_liu": "full-feature LIU",
        ATOMIC: "random AD",
        FAMILY: "family-blocked AD",
        T4: "dependency AD, T=4",
        T8: "dependency AD, T=8",
        DIRECT: "direct-average",
        NODE_PERMUTED: "node-permuted",
        K5: "dependency AD, k=5",
        K11: "dependency AD, k=11",
        PRIMARY: "dependency AD, T=16",
    }
    selected = [next(row for row in summary_rows if row["method"] == method)
                for method in HEADLINE]
    plt.figure(figsize=(13, 5.5))
    plt.bar(range(len(selected)), [row["mean_delta_pp"] for row in selected])
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(range(len(selected)), [labels[row["method"]] for row in selected],
               rotation=31, ha="right")
    plt.ylabel("Mean AUROC change vs IU-PCR (pp)")
    plt.title("Repeated cross-view alternating diffusion: frozen comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "headline_methods.png"), dpi=180)
    plt.close()

    primary_rows = sorted(
        (row for row in metric_rows if row["method"] == PRIMARY),
        key=lambda row: row["delta_pp"],
    )
    colors = ["#e76f51" if row["delta_pp"] < 0 else "#2a9d8f"
              for row in primary_rows]
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(primary_rows)), [row["delta_pp"] for row in primary_rows],
             color=colors)
    plt.yticks(range(len(primary_rows)), [row["cell"] for row in primary_rows],
               fontsize=8)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Primary RCV-AD-IU-PCR change vs IU-PCR (pp)")
    plt.title("Registered primary in every dataset/model cell")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "primary_per_cell.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for schema, label in (
        ("atomic_random", "atomic random"),
        ("dependency_blocked", "dependency blocked"),
        ("family_blocked", "family blocked"),
    ):
        local = sorted((row for row in lambda_rows if row["schema"] == schema),
                       key=lambda row: row["lambda"])
        plt.plot([row["lambda"] for row in local],
                 [row["mean_delta_pp"] for row in local], marker="o", label=label)
    plt.xscale("symlog", linthresh=0.03)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Laplacian strength lambda")
    plt.ylabel("Mean AUROC change vs IU-PCR (pp)")
    plt.title("Frozen lambda paths")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "lambda_paths.png"), dpi=180)
    plt.close()

    schemas = ("atomic_random", "dependency_blocked", "family_blocked")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for position, schema in enumerate(schemas):
        local = [row for row in diagnostic_rows if row["schema"] == schema]
        axes[0].boxplot([row["graph_cka"] for row in local], positions=[position])
        axes[1].boxplot([row["score_spearman"] for row in local], positions=[position])
    for axis in axes:
        axis.set_xticks(range(len(schemas)))
        axis.set_xticklabels(["random", "dependency", "family"], rotation=20)
        axis.set_ylim(-0.05, 1.05)
    axes[0].axhline(0.5, color="black", linestyle="--")
    axes[0].set_title("Partition graph vs consensus")
    axes[0].set_ylabel("Median centered-kernel alignment")
    axes[1].set_title("Partition score vs consensus score")
    axes[1].set_ylabel("Median Spearman")
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "partition_consistency.png"), dpi=180)
    plt.close(fig)

    dependency_diag = [row for row in diagnostic_rows
                       if row["schema"] == "dependency_blocked"]
    delta_map = {row["cell"]: row["delta_pp"] for row in primary_rows}
    plt.figure(figsize=(7, 5))
    plt.scatter([row["graph_cka"] for row in dependency_diag],
                [delta_map[row["cell"]] for row in dependency_diag], alpha=0.8)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0.5, color="black", linestyle="--")
    plt.xlabel("Label-free partition-to-consensus graph CKA")
    plt.ylabel("Primary AUROC change vs IU-PCR (pp)")
    plt.title("Does convergence imply correctness relevance?")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "stability_vs_utility.png"), dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    bundle = os.path.abspath(args.bundle)
    definition = verify_freeze(out_dir, bundle)

    # Correctness is opened only after the score-freeze manifest exists.
    data = np.load(bundle, allow_pickle=True)
    lambda_methods = [
        f"{schema}__lambda_{lambda_token(value)}"
        for schema in ("atomic_random", "dependency_blocked", "family_blocked")
        for value in LAMBDAS
    ]
    methods = list(dict.fromkeys((*HEADLINE, *lambda_methods)))
    metric_rows = []
    diagnostic_rows = []
    for cell in INSCOPE:
        labels = np.asarray(data[f"{cell}__labels"], dtype=int)
        with np.load(os.path.join(out_dir, "scores", f"{cell}.npz"),
                     allow_pickle=False) as scores:
            iu_auc = roc_auc_score(labels, scores["iu_pcr"])
            for method in methods:
                auc = metric(labels, scores[method], roc_auc_score)
                metric_rows.append({
                    "cell": cell,
                    "domain": GROUP[cell],
                    "family": dataset_family(cell),
                    "method": method,
                    "auroc": auc,
                    "auprc": metric(labels, scores[method], average_precision_score),
                    "delta_pp": 100.0 * (auc - iu_auc),
                })
        with open(os.path.join(out_dir, "diagnostics", f"{cell}.json"),
                  encoding="utf-8") as handle:
            diagnostic = json.load(handle)
        for schema, item in diagnostic["schemas"].items():
            diagnostic_rows.append({
                "cell": cell,
                "domain": GROUP[cell],
                "family": dataset_family(cell),
                "schema": schema,
                "block_count": item["block_count"],
                "max_block_size": max(item["block_sizes"]),
                "partition_count": item["partition_count_used"],
                "graph_cka": item["partition_consensus_cka_median"],
                "graph_cka_min": item["partition_consensus_cka_min"],
                "edge_jaccard": item["partition_consensus_jaccard_median"],
                "score_spearman": item["partition_score_spearman_median"],
                "score_spearman_min": item["partition_score_spearman_min"],
                "rank_change": item["mean_abs_rank_change_vs_iu"],
                "n_components": item["graph"]["n_components"],
                "degree_min": item["graph"]["degree_min"],
                "t4_spearman_vs_final": item.get("prefixes", {}).get("4", {}).get(
                    "score_spearman_vs_final", float("nan")
                ),
                "t8_spearman_vs_final": item.get("prefixes", {}).get("8", {}).get(
                    "score_spearman_vs_final", float("nan")
                ),
            })

    summary_rows = summarize_methods(metric_rows, HEADLINE)
    summary_map = {row["method"]: row for row in summary_rows}
    lambda_rows = []
    for schema in ("atomic_random", "dependency_blocked", "family_blocked"):
        for value in LAMBDAS:
            method = f"{schema}__lambda_{lambda_token(value)}"
            local = [row for row in metric_rows if row["method"] == method]
            deltas = np.asarray([row["delta_pp"] for row in local])
            lambda_rows.append({
                "schema": schema,
                "lambda": float(value),
                "mean_delta_pp": float(np.mean(deltas)),
                "wins": int(np.sum(deltas > 1e-12)),
                "losses": int(np.sum(deltas < -1e-12)),
                "worst_delta_pp": float(np.min(deltas)),
            })

    primary = summary_map[PRIMARY]
    dependency_diag = [row for row in diagnostic_rows
                       if row["schema"] == "dependency_blocked"]
    median_cka = float(np.median([row["graph_cka"] for row in dependency_diag]))
    median_t8 = float(np.nanmedian([
        row["t8_spearman_vs_final"] for row in dependency_diag
    ]))
    gates = [
        {"gate": "primary mean improvement is at least +0.20pp",
         "observed": primary["mean_delta_pp"], "passed": primary["mean_delta_pp"] >= 0.20},
        {"gate": "primary equal-family lower bound is above zero",
         "observed": primary["family_ci_low"], "passed": primary["family_ci_low"] > 0},
        {"gate": "primary improves at least 14 of 24 cells",
         "observed": primary["wins"], "passed": primary["wins"] >= 14},
        {"gate": "primary worst loss is no worse than -2pp",
         "observed": primary["worst_delta_pp"], "passed": primary["worst_delta_pp"] >= -2},
        {"gate": "primary beats atomic-random splitting",
         "observed": primary["mean_delta_pp"] - summary_map[ATOMIC]["mean_delta_pp"],
         "passed": primary["mean_delta_pp"] > summary_map[ATOMIC]["mean_delta_pp"]},
        {"gate": "primary beats family-blocked splitting",
         "observed": primary["mean_delta_pp"] - summary_map[FAMILY]["mean_delta_pp"],
         "passed": primary["mean_delta_pp"] > summary_map[FAMILY]["mean_delta_pp"]},
        {"gate": "primary beats direct arithmetic view averaging",
         "observed": primary["mean_delta_pp"] - summary_map[DIRECT]["mean_delta_pp"],
         "passed": primary["mean_delta_pp"] > summary_map[DIRECT]["mean_delta_pp"]},
        {"gate": "primary beats the node-permuted graph",
         "observed": primary["mean_delta_pp"] - summary_map[NODE_PERMUTED]["mean_delta_pp"],
         "passed": primary["mean_delta_pp"] > summary_map[NODE_PERMUTED]["mean_delta_pp"]},
        {"gate": "primary beats frozen DUFS-LIU",
         "observed": primary["mean_delta_pp"] - summary_map["dufs_liu"]["mean_delta_pp"],
         "passed": primary["mean_delta_pp"] > summary_map["dufs_liu"]["mean_delta_pp"]},
        {"gate": "median partition-to-consensus graph CKA is at least 0.50",
         "observed": median_cka, "passed": median_cka >= 0.50},
        {"gate": "median T=8 versus T=16 score Spearman is at least 0.95",
         "observed": median_t8, "passed": median_t8 >= 0.95},
    ]
    all_passed = bool(all(row["passed"] for row in gates))

    write_csv(os.path.join(out_dir, "metrics.csv"), metric_rows)
    write_csv(os.path.join(out_dir, "headline_summary.csv"), summary_rows)
    write_csv(os.path.join(out_dir, "lambda_sensitivity.csv"), lambda_rows)
    write_csv(os.path.join(out_dir, "diagnostic_summary.csv"), diagnostic_rows)
    write_json(os.path.join(out_dir, "CONTINUATION_GATES.json"), {
        "all_gates_passed": all_passed,
        "gates": gates,
    })
    make_plots(out_dir, metric_rows, summary_rows, lambda_rows, diagnostic_rows)

    decision = "CONTINUE CROSS-VIEW DIFFUSION" if all_passed else "STOP OR REVISE CROSS-VIEW DIFFUSION"
    lines = [
        "# Repeated cross-view alternating-diffusion experiment", "",
        f"**Decision: {decision}.**", "",
        "## Terms", "",
        "- **Partition:** two disjoint feature sets that together contain the full pool.",
        "- **Dependency block:** near-duplicate rank-correlated features that cannot be split across views.",
        "- **Alternating diffusion:** a two-step sample transition through both view graphs.",
        "- **Consensus graph:** the average alternating graph across frozen partitions, reduced back to k neighbours.",
        "- **pp:** AUROC percentage points.", "",
        "## Headline", "",
        "| method | macro AUROC | change vs IU | family change | family interval | wins/losses | worst |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['cell_macro_auroc']:.4f} | "
            f"{row['mean_delta_pp']:+.3f}pp | {row['family_delta_pp']:+.3f}pp | "
            f"[{row['family_ci_low']:+.3f},{row['family_ci_high']:+.3f}] | "
            f"{row['wins']}/{row['losses']} | {row['worst_delta_pp']:+.3f}pp |"
        )
    lines += ["", "## Label-free convergence", ""]
    for schema in ("atomic_random", "dependency_blocked", "family_blocked"):
        local = [row for row in diagnostic_rows if row["schema"] == schema]
        lines.append(
            f"- **{schema}:** median graph CKA {np.median([r['graph_cka'] for r in local]):.3f}; "
            f"median partition-score Spearman {np.median([r['score_spearman'] for r in local]):.3f}; "
            f"median edge Jaccard {np.median([r['edge_jaccard'] for r in local]):.3f}."
        )
    lines += ["", "## Continuation gates", "",
              "| gate | observed | pass |", "|---|---:|:---:|"]
    for row in gates:
        lines.append(
            f"| {row['gate']} | {row['observed']:.4f} | "
            f"{'yes' if row['passed'] else 'no'} |"
        )
    lines += ["", "## Interpretation", ""]
    if all_passed:
        lines.append(
            "The registered common-manifold mechanism passed every effect, control, safety, and convergence gate. These reused cells justify a new-data confirmation, not a final claim."
        )
    else:
        lines.append(
            "The registered common-manifold mechanism did not pass every gate. Stable partitions alone must not be interpreted as correctness evidence. Use the random, family, direct-average, and permuted controls to identify whether the failure is duplicate leakage, missing cross-view signal, ordinary shrinkage, or stable target-irrelevant geometry."
        )
    lines += ["", "## Audit", "",
              f"Run fingerprint: `{definition['run_fingerprint']}`. All registered sources, inputs, reference scores, and new score artifacts were verified before correctness labels were opened. The 24 cells are retrospective development evidence."]
    report = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
