#!/usr/bin/env python3
"""Evaluate already-frozen scores from the 24-cell fusion benchmark.

The fit program never reads labels.  This program first verifies the complete
score manifest and its SHA-256 hashes, writes a second immutable freeze manifest,
and only then opens labels to compute tables, uncertainty intervals, and plots.
It does not choose hyperparameters from the observed results.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import hashlib
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.frozen_24cell_benchmark import (                   # noqa: E402
    ALL_GRAPH_ARMS,
    DEFAULT_BUNDLE,
    DEFAULT_OUT,
    FROZEN_LAMBDA,
    LAMBDAS,
    VIEW_SCHEMAS,
    VERSION,
    lambda_token,
    score_key,
    schema_arm,
    sha256_file,
    validate_bundle,
)
from scripts.inscope_cells import GROUP, INSCOPE                 # noqa: E402


FAMILY_NAMES = (
    "triviaqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
    "truthfulqa", "gsm8k", "math500",
)

HEADLINE_METHODS = OrderedDict((
    ("deployed_upcr", "Deployed U-PCR"),
    ("iu_pcr", "IU-PCR"),
    (score_key("dufs_liu", FROZEN_LAMBDA["dufs_liu"]), "DUFS-LIU (lambda=0.1)"),
    (score_key(schema_arm("manual", "ca_specrage_alpha_liu"), 10.0),
     "CA-alpha, manual views (lambda=10)"),
    (score_key(schema_arm("atomic", "ca_specrage_alpha_liu"), 10.0),
     "CA-alpha, balanced atomic views (lambda=10)"),
    (score_key(schema_arm("micro", "ca_specrage_alpha_liu"), 10.0),
     "CA-alpha, LOCO micro-views (lambda=10)"),
))

SECONDARY_METHODS = OrderedDict()
for _schema in VIEW_SCHEMAS:
    for _interface, _label in (
        ("adapted_specrage_y_liu", "adapted plain-loss Y"),
        ("ca_specrage_y_liu", "CA-trained Y"),
        ("uniform_y_liu", "prior-only uniform Y"),
        ("ca_uniform_alpha_control", "CA prior-alpha graph control"),
        ("ca_global_alpha_control", "global alpha control"),
        ("ca_permuted_alpha_control", "permuted alpha control"),
    ):
        SECONDARY_METHODS[
            score_key(schema_arm(_schema, _interface), 10.0)
        ] = f"{_schema}: {_label} (lambda=10)"
SECONDARY_METHODS[score_key("raw_uniform_liu", 10.0)] = \
    "Raw-uniform graph control (lambda=10)"

ARM_LABELS = {
    "dufs_liu": "DUFS-LIU",
    "raw_uniform_liu": "Raw uniform",
}
for _schema in VIEW_SCHEMAS:
    ARM_LABELS.update({
        schema_arm(_schema, "adapted_specrage_y_liu"):
            f"{_schema}: adapted Y",
        schema_arm(_schema, "ca_specrage_alpha_liu"):
            f"{_schema}: CA alpha",
        schema_arm(_schema, "ca_specrage_y_liu"):
            f"{_schema}: CA Y",
        schema_arm(_schema, "uniform_y_liu"):
            f"{_schema}: uniform Y",
        schema_arm(_schema, "ca_uniform_alpha_control"):
            f"{_schema}: prior alpha",
        schema_arm(_schema, "ca_global_alpha_control"):
            f"{_schema}: global alpha",
        schema_arm(_schema, "ca_permuted_alpha_control"):
            f"{_schema}: permuted alpha",
    })


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def family(cell: str) -> str:
    return next((name for name in FAMILY_NAMES if name in cell), cell)


def safe_metric(metric, labels, scores) -> float:
    try:
        return float(metric(labels, scores))
    except ValueError:
        return float("nan")


def bootstrap_mean_ci(values, namespace: str, count: int = 20000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    seed = int(hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(count, len(values)))
    estimates = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.975)))


def grouped_values(cells: list[str], values: np.ndarray) -> np.ndarray:
    groups = sorted({family(cell) for cell in cells})
    return np.asarray([
        np.mean([value for cell, value in zip(cells, values) if family(cell) == group])
        for group in groups
    ])


def diagnostic_availability_fields(item: dict) -> dict:
    paths = list(item.get("nonfinite_diagnostic_paths", []))
    return {
        "nonfinite_diagnostic_count": len(paths),
        "nonfinite_diagnostic_paths": json.dumps(paths, sort_keys=True),
    }


def verify_score_freeze(out_dir: str, bundle: str) -> tuple[dict, dict, dict]:
    definition_path = os.path.join(out_dir, "RUN_DEFINITION.json")
    complete_path = os.path.join(out_dir, "FIT_COMPLETE.json")
    if not os.path.exists(definition_path) or not os.path.exists(complete_path):
        raise RuntimeError("fit is incomplete: RUN_DEFINITION.json or FIT_COMPLETE.json is missing")
    with open(definition_path, encoding="utf-8") as handle:
        definition = json.load(handle)
    with open(complete_path, encoding="utf-8") as handle:
        complete = json.load(handle)
    if not definition.get("scientific_run") or not complete.get("scientific_run"):
        raise RuntimeError("debug output cannot be evaluated as the scientific 24-cell run")
    if definition.get("version") != VERSION or complete.get("version") != VERSION:
        raise RuntimeError("runner/report version mismatch")
    if definition.get("run_fingerprint") != complete.get("run_fingerprint"):
        raise RuntimeError("run fingerprints disagree")
    if tuple(definition.get("cells", ())) != tuple(INSCOPE) or complete.get("n_cells") != 24:
        raise RuntimeError("score run does not contain the exact registered 24-cell roster")
    if sha256_file(bundle) != definition.get("bundle_sha256"):
        raise RuntimeError("evaluation bundle differs from the bundle used for fitting")
    for relative, expected_hash in definition.get("source_sha256", {}).items():
        current_path = os.path.join(REPO, relative)
        if not os.path.exists(current_path) or sha256_file(current_path) != expected_hash:
            raise RuntimeError(
                f"registered source changed after score fitting: {relative}. "
                "Use a new output directory and refit before opening labels."
            )

    observed = {}
    manifest = complete.get("score_manifest", [])
    if [row.get("cell") for row in manifest] != list(INSCOPE):
        raise RuntimeError("score manifest order/roster differs from scripts.inscope_cells")
    for row in manifest:
        score_path = os.path.join(out_dir, row["score_file"])
        diagnostic_path = os.path.join(out_dir, row["diagnostic_file"])
        if sha256_file(score_path) != row["score_sha256"]:
            raise RuntimeError(f"score checkpoint changed after fitting: {score_path}")
        if sha256_file(diagnostic_path) != row["diagnostic_sha256"]:
            raise RuntimeError(f"diagnostic checkpoint changed after fitting: {diagnostic_path}")
        with np.load(score_path, allow_pickle=False) as checkpoint:
            if any("label" in key.lower() for key in checkpoint.files):
                raise RuntimeError(f"labels found in frozen score checkpoint: {score_path}")
            observed[row["cell"]] = {
                key: np.asarray(checkpoint[key]) for key in checkpoint.files
            }
    freeze = {
        "version": VERSION,
        "freeze_protocol": "immutable-score-freeze-v2",
        "run_fingerprint": definition["run_fingerprint"],
        "bundle_sha256": definition["bundle_sha256"],
        "run_definition_sha256": sha256_file(definition_path),
        "fit_complete_sha256": sha256_file(complete_path),
        "report_source_sha256": sha256_file(os.path.abspath(__file__)),
        "registered_source_sha256": definition.get("source_sha256", {}),
        "score_files_verified_before_labels": True,
        "score_manifest": manifest,
    }
    freeze_path = os.path.join(out_dir, "SCORE_FREEZE_MANIFEST.json")
    if os.path.exists(freeze_path):
        with open(freeze_path, encoding="utf-8") as handle:
            previous_freeze = json.load(handle)
        if previous_freeze != freeze:
            raise RuntimeError(
                "immutable score-freeze manifest disagrees with current artifacts; "
                "use a new output directory and refit"
            )
    else:
        # Exclusive creation prevents a later report run from silently replacing
        # the evidence that was frozen before labels were first opened.
        with open(freeze_path, "x", encoding="utf-8") as handle:
            json.dump(freeze, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    return definition, complete, observed


def evaluate_scores(scores_by_cell: dict, data) -> tuple[list[dict], dict]:
    rows = []
    raw_scores = {}
    expected_keys = set(HEADLINE_METHODS) | set(SECONDARY_METHODS)
    expected_keys |= {
        score_key(arm, lambda_) for arm in ALL_GRAPH_ARMS for lambda_ in LAMBDAS
    }
    for cell in INSCOPE:
        labels = np.asarray(data[f"{cell}__labels"], dtype=int)
        frozen = scores_by_cell[cell]
        missing = expected_keys - set(frozen)
        if missing:
            raise RuntimeError(f"{cell} is missing score arrays: {sorted(missing)}")
        if not np.array_equal(frozen["sample_index"], np.arange(len(labels))):
            raise RuntimeError(f"sample order mismatch for {cell}")
        raw_scores[cell] = {key: frozen[key] for key in expected_keys}
        for key in sorted(expected_keys):
            values = np.asarray(frozen[key], dtype=float)
            if values.shape != labels.shape or not np.isfinite(values).all():
                raise RuntimeError(f"invalid score array {cell}/{key}: {values.shape}")
            rows.append({
                "cell": cell,
                "domain": GROUP[cell],
                "family": family(cell),
                "n": len(labels),
                "n_positive": int(labels.sum()),
                "positive_rate": float(labels.mean()),
                "method_key": key,
                "auroc": safe_metric(roc_auc_score, labels, values),
                "auprc": safe_metric(average_precision_score, labels, values),
                "score_variance": float(np.var(values)),
            })
    return rows, raw_scores


def lookup_metric(rows: list[dict], metric: str) -> dict[tuple[str, str], float]:
    return {(row["cell"], row["method_key"]): float(row[metric]) for row in rows}


def headline_summary(rows: list[dict]) -> list[dict]:
    output = []
    for metric in ("auroc", "auprc"):
        lookup = lookup_metric(rows, metric)
        for key, name in HEADLINE_METHODS.items():
            values = np.asarray([lookup[cell, key] for cell in INSCOPE])
            qa = values[[GROUP[cell] == "QA" for cell in INSCOPE]]
            math = values[[GROUP[cell] == "math" for cell in INSCOPE]]
            lo, hi = bootstrap_mean_ci(values, f"headline-{metric}-{key}")
            family_values = grouped_values(list(INSCOPE), values)
            family_lo, family_hi = bootstrap_mean_ci(
                family_values, f"headline-family-{metric}-{key}"
            )
            output.append({
                "metric": metric,
                "method_key": key,
                "method": name,
                "cell_macro": float(np.mean(values)),
                "cell_ci_low": lo,
                "cell_ci_high": hi,
                "qa_macro": float(np.mean(qa)),
                "math_macro": float(np.mean(math)),
                "family_macro": float(np.mean(family_values)),
                "family_ci_low": family_lo,
                "family_ci_high": family_hi,
            })
    return output


def secondary_summary(rows: list[dict]) -> list[dict]:
    auroc_lookup = lookup_metric(rows, "auroc")
    auprc_lookup = lookup_metric(rows, "auprc")
    output = []
    for key, name in SECONDARY_METHODS.items():
        roc = np.asarray([auroc_lookup[cell, key] for cell in INSCOPE])
        pr = np.asarray([auprc_lookup[cell, key] for cell in INSCOPE])
        output.append({
            "method_key": key,
            "method": name,
            "cell_macro_auroc": float(np.mean(roc)),
            "qa_macro_auroc": float(np.mean([
                value for cell, value in zip(INSCOPE, roc) if GROUP[cell] == "QA"
            ])),
            "math_macro_auroc": float(np.mean([
                value for cell, value in zip(INSCOPE, roc) if GROUP[cell] == "math"
            ])),
            "cell_macro_auprc": float(np.mean(pr)),
            "delta_vs_iu_pp": float(100 * np.mean(np.asarray([
                auroc_lookup[cell, key] - auroc_lookup[cell, "iu_pcr"]
                for cell in INSCOPE
            ]))),
        })
    return output


def holm_adjust(p_values: list[float]) -> list[float]:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_comparisons(rows: list[dict]) -> list[dict]:
    lookup = lookup_metric(rows, "auroc")
    references = OrderedDict((
        ("deployed_upcr", "Deployed U-PCR"),
        ("iu_pcr", "IU-PCR"),
        (score_key("dufs_liu", 0.1), "DUFS-LIU (lambda=0.1)"),
        (score_key(schema_arm("manual", "ca_specrage_alpha_liu"), 10.0),
         "CA-alpha, manual views (lambda=10)"),
        (score_key(schema_arm("atomic", "ca_specrage_alpha_liu"), 10.0),
         "CA-alpha, balanced atomic views (lambda=10)"),
    ))
    comparisons = []
    for candidate_key, candidate_name in list(HEADLINE_METHODS.items())[3:]:
        for reference_key, reference_name in references.items():
            if candidate_key == reference_key:
                continue
            delta = np.asarray([
                lookup[cell, candidate_key] - lookup[cell, reference_key]
                for cell in INSCOPE
            ])
            family_delta = grouped_values(list(INSCOPE), delta)
            cell_lo, cell_hi = bootstrap_mean_ci(
                delta, f"pair-cell-{candidate_key}-{reference_key}"
            )
            family_lo, family_hi = bootstrap_mean_ci(
                family_delta, f"pair-family-{candidate_key}-{reference_key}"
            )
            try:
                p_value = float(wilcoxon(delta, zero_method="pratt").pvalue)
            except ValueError:
                p_value = 1.0
            comparisons.append({
                "candidate_key": candidate_key,
                "candidate": candidate_name,
                "reference_key": reference_key,
                "reference": reference_name,
                "mean_delta_pp": float(100 * np.mean(delta)),
                "median_delta_pp": float(100 * np.median(delta)),
                "cell_ci_low_pp": float(100 * cell_lo),
                "cell_ci_high_pp": float(100 * cell_hi),
                "family_delta_pp": float(100 * np.mean(family_delta)),
                "family_ci_low_pp": float(100 * family_lo),
                "family_ci_high_pp": float(100 * family_hi),
                "wins": int(np.sum(delta > 1e-12)),
                "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                "losses": int(np.sum(delta < -1e-12)),
                "worst_delta_pp": float(100 * np.min(delta)),
                "best_delta_pp": float(100 * np.max(delta)),
                "wilcoxon_p": p_value,
            })
    adjusted = holm_adjust([row["wilcoxon_p"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["holm_p"] = value
    return comparisons


def lambda_summary(rows: list[dict]) -> list[dict]:
    lookup = lookup_metric(rows, "auroc")
    output = []
    for arm in ALL_GRAPH_ARMS:
        for lambda_ in LAMBDAS:
            key = score_key(arm, lambda_)
            values = np.asarray([lookup[cell, key] for cell in INSCOPE])
            iu = np.asarray([lookup[cell, "iu_pcr"] for cell in INSCOPE])
            delta = values - iu
            output.append({
                "arm": arm,
                "method": ARM_LABELS[arm],
                "lambda": lambda_,
                "all_auroc": float(np.mean(values)),
                "qa_auroc": float(np.mean([
                    value for cell, value in zip(INSCOPE, values) if GROUP[cell] == "QA"
                ])),
                "math_auroc": float(np.mean([
                    value for cell, value in zip(INSCOPE, values) if GROUP[cell] == "math"
                ])),
                "delta_vs_iu_pp": float(100 * np.mean(delta)),
                "wins_vs_iu": int(np.sum(delta > 1e-12)),
                "losses_vs_iu": int(np.sum(delta < -1e-12)),
                "worst_delta_vs_iu_pp": float(100 * np.min(delta)),
            })
    return output


def flatten_diagnostics(out_dir: str) -> tuple[list[dict], list[dict]]:
    rows, histories = [], []
    for cell in INSCOPE:
        path = os.path.join(out_dir, "diagnostics", f"{cell}.json")
        with open(path, encoding="utf-8") as handle:
            item = json.load(handle)
        dufs = item["dufs"]
        partition = item["micro_partition_diagnostics"]
        availability = diagnostic_availability_fields(item)
        for schema in VIEW_SCHEMAS:
            schema_item = item["schemas"][schema]
            ca = schema_item["ca_specrage"]
            plain = schema_item["adapted_specrage"]
            ca_arm = schema_arm(schema, "ca_specrage_alpha_liu")
            ca_l10 = item["liu"][ca_arm][lambda_token(10.0)]
            rows.append({
                "cell": cell,
                "domain": GROUP[cell],
                "schema": schema,
                "n": item["n_samples"],
                "m": item["n_features_stable"],
                "n_views": len(schema_item["view_dimensions"]),
                "view_dimensions": json.dumps(
                    schema_item["view_dimensions"], sort_keys=True
                ),
                "view_prior": json.dumps(schema_item["view_prior"], sort_keys=True),
                "micro_chosen_k": partition["chosen_k"],
                "micro_silhouette": partition["chosen_silhouette"],
                "micro_bootstrap_ari": partition["chosen_bootstrap_ari"],
                "micro_singleton_fraction": partition["chosen_singleton_fraction"],
                **availability,
                "deployed_n_kept": item["deployed"]["n_kept"],
                "deployed_simple_average": item["deployed"]["used_simple_average"],
                "dufs_effective_features": dufs["effective_feature_count"],
                "dufs_seed_std": dufs["mean_seed_std"],
                "plain_alpha_entropy": plain["alpha_entropy_normalized"],
                "plain_alpha_seed_mad": plain["alpha_seed_mad"],
                "ca_alpha_entropy": ca["alpha_entropy_normalized"],
                "ca_alpha_seed_mad": ca["alpha_seed_mad"],
                "ca_graph_seed_distance": ca["graph_seed_relative_distance_mean"],
                "ca_target_entropy": ca.get("agreement_target", {}).get(
                    "entropy_normalized", float("nan")
                ),
                "ca_alpha_target_mad": ca.get("agreement_target", {}).get(
                    "alpha_target_mad", float("nan")
                ),
                "cross_view_agreement_median": ca.get("agreement_target", {}).get(
                    "score_median", float("nan")
                ),
                "base_edge_jaccard_median": ca["base_edge_jaccard_median"],
                "view_effective_rank": json.dumps(
                    ca["view_effective_rank"], sort_keys=True
                ),
                "alpha_effective_views": ca["alpha_effective_views_mean"],
                "alpha_kl_from_prior": ca["alpha_kl_from_prior_mean"],
                "projected_roughness_distance_median": schema_item[
                    "fusion_geometry"
                ]["projected_roughness_distance_median"],
                "ca_graph_components": ca["n_components"],
                "ca_graph_degree_p05_over_mean": ca["degree_p05_over_mean"],
                "ca_projected_condition": ca_l10["projected_condition_number"],
                "ca_weight_cosine_vs_iu": ca_l10["weight_cosine_vs_iu"],
                "ca_score_energy": ca_l10["score_laplacian_energy"],
                "total_runtime_seconds": item["runtime"]["total_seconds"],
                "dufs_runtime_seconds": item["runtime"]["dufs_seconds"],
                "plain_runtime_seconds": item["runtime"][
                    f"{schema}_adapted_specrage_seconds"
                ],
                "ca_runtime_seconds": item["runtime"][
                    f"{schema}_ca_specrage_seconds"
                ],
            })
        for record in item.get("histories", []):
            histories.append({"cell": cell, "domain": GROUP[cell], **record})
    return rows, histories


def add_rank_diagnostics(diagnostics: list[dict], raw_scores: dict) -> None:
    for row in diagnostics:
        cell = row["cell"]
        iu = raw_scores[cell]["iu_pcr"]
        ca = raw_scores[cell][score_key(
            schema_arm(row["schema"], "ca_specrage_alpha_liu"), 10.0
        )]
        correlation = spearmanr(iu, ca).statistic
        row["ca_score_spearman_vs_iu"] = float(correlation)
        iu_rank = np.argsort(np.argsort(iu))
        ca_rank = np.argsort(np.argsort(ca))
        row["ca_mean_absolute_rank_shift"] = float(np.mean(np.abs(iu_rank - ca_rank)))
        row["ca_mean_absolute_rank_shift_fraction"] = float(
            np.mean(np.abs(iu_rank - ca_rank)) / len(iu)
        )


def make_plots(
    rows: list[dict], summary: list[dict], comparisons: list[dict],
    paths: list[dict], diagnostics: list[dict], histories: list[dict], out_dir: str
) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(out_dir, "figures")
    os.makedirs(plot_dir, exist_ok=True)

    auroc = [row for row in summary if row["metric"] == "auroc"]
    x = np.arange(len(auroc))
    means = np.asarray([row["cell_macro"] for row in auroc])
    lower = means - np.asarray([row["cell_ci_low"] for row in auroc])
    upper = np.asarray([row["cell_ci_high"] for row in auroc]) - means
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.errorbar(x, means, yerr=np.vstack([lower, upper]), fmt="o", capsize=4)
    axis.set_xticks(x, [row["method"] for row in auroc], rotation=22, ha="right")
    axis.set_ylabel("Cell-macro AUROC")
    axis.set_title("Frozen 24-cell comparison (95% cell-bootstrap intervals)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "headline_auroc.png"), dpi=190)
    plt.close(fig)

    lookup = lookup_metric(rows, "auroc")
    candidates = list(HEADLINE_METHODS.items())[2:]
    fig, axes = plt.subplots(len(candidates), 1, figsize=(12, 3.2 * len(candidates)), sharex=True)
    if len(candidates) == 1:
        axes = [axes]
    for axis, (key, label) in zip(axes, candidates):
        delta = np.asarray([
            100 * (lookup[cell, key] - lookup[cell, "iu_pcr"]) for cell in INSCOPE
        ])
        colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in delta]
        axis.bar(np.arange(len(INSCOPE)), delta, color=colors)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("AUROC change (pp)")
        axis.set_title(f"{label} versus IU-PCR")
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(
        np.arange(len(INSCOPE)), INSCOPE, rotation=78, ha="right", fontsize=7
    )
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "paired_cell_deltas.png"), dpi=190)
    plt.close(fig)

    method_keys = list(HEADLINE_METHODS)
    matrix = np.asarray([
        [lookup[cell, key] for key in method_keys] for cell in INSCOPE
    ])
    fig, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    axis.set_xticks(range(len(method_keys)), HEADLINE_METHODS.values(), rotation=28, ha="right")
    axis.set_yticks(range(len(INSCOPE)), INSCOPE, fontsize=7)
    axis.set_title("Per-cell AUROC (no orientation flipping)")
    fig.colorbar(image, ax=axis, label="AUROC")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "per_cell_heatmap.png"), dpi=190)
    plt.close(fig)

    selected_arms = (
        "dufs_liu",
        schema_arm("manual", "ca_specrage_alpha_liu"),
        schema_arm("atomic", "ca_specrage_alpha_liu"),
        schema_arm("micro", "ca_specrage_alpha_liu"),
        schema_arm("micro", "adapted_specrage_y_liu"),
        "raw_uniform_liu",
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, field, title in zip(
        axes, ("all_auroc", "qa_auroc", "math_auroc"), ("All 24", "QA", "Math")
    ):
        for arm in selected_arms:
            selected = [row for row in paths if row["arm"] == arm]
            axis.plot(
                [row["lambda"] for row in selected],
                [row[field] for row in selected],
                marker="o", label=ARM_LABELS[arm],
            )
        axis.set_xscale("symlog", linthresh=0.1)
        axis.set_xlabel("Laplacian strength lambda")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Macro AUROC")
    axes[-1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Sensitivity path; it is not used to choose the headline setting")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "lambda_paths.png"), dpi=190)
    plt.close(fig)

    ca_key = score_key(schema_arm("micro", "ca_specrage_alpha_liu"), 10.0)
    deltas = np.asarray([
        100 * (lookup[cell, ca_key] - lookup[cell, "iu_pcr"]) for cell in INSCOPE
    ])
    micro_diagnostics = [row for row in diagnostics if row["schema"] == "micro"]
    entropy = np.asarray([row["ca_alpha_entropy"] for row in micro_diagnostics])
    shift = np.asarray([
        row["ca_mean_absolute_rank_shift_fraction"] for row in micro_diagnostics
    ])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(entropy, deltas, c=[GROUP[cell] == "QA" for cell in INSCOPE], cmap="coolwarm")
    axes[0].set_xlabel("Normalized CA view-weight entropy")
    axes[0].set_ylabel("CA minus IU-PCR AUROC (pp)")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].grid(alpha=0.2)
    axes[1].scatter(shift, deltas, c=[GROUP[cell] == "QA" for cell in INSCOPE], cmap="coolwarm")
    axes[1].set_xlabel("Mean rank displacement / n")
    axes[1].set_ylabel("CA minus IU-PCR AUROC (pp)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].grid(alpha=0.2)
    fig.suptitle("Does the CA mechanism activate, and is activation useful?")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "ca_mechanism.png"), dpi=190)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for fit in (
        "manual_ca_specrage", "atomic_ca_specrage", "micro_ca_specrage",
    ):
        selected = [row for row in histories if row.get("fit") == fit]
        epochs = sorted({int(row["epoch"]) for row in selected})
        means = [
            np.mean([
                float(row["validation_loss"])
                for row in selected if int(row["epoch"]) == epoch
            ])
            for epoch in epochs
        ]
        axis.plot(epochs, means, label=fit.replace("_", " "))
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean unlabeled validation objective")
    axis.set_title("Optimization convergence across cells and seeds")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "training_convergence.png"), dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    schema_positions = np.arange(len(VIEW_SCHEMAS))
    roughness = [
        [row["projected_roughness_distance_median"] for row in diagnostics
         if row["schema"] == schema]
        for schema in VIEW_SCHEMAS
    ]
    axes[0].boxplot(roughness, tick_labels=VIEW_SCHEMAS, showfliers=True)
    axes[0].set_ylabel("Median projected-roughness distance")
    axes[0].set_title("Do views actuate IU-PCR differently?")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].scatter(
        np.arange(len(micro_diagnostics)),
        [row["micro_bootstrap_ari"] for row in micro_diagnostics],
        c=[row["micro_chosen_k"] for row in micro_diagnostics], cmap="viridis",
    )
    axes[1].axhline(0.75, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(
        np.arange(len(micro_diagnostics)),
        [row["cell"] for row in micro_diagnostics], rotation=75, ha="right", fontsize=6,
    )
    axes[1].set_ylabel("LOCO partition bootstrap ARI")
    axes[1].set_title("Micro-view stability (color = chosen K)")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "view_schema_diagnostics.png"), dpi=190)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(
        [row["n"] for row in micro_diagnostics],
        [row["total_runtime_seconds"] for row in micro_diagnostics],
        c=[GROUP[row["cell"]] == "QA" for row in micro_diagnostics], cmap="coolwarm",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Samples in cell (log scale)")
    axis.set_ylabel("Fit time in seconds (log scale)")
    axis.set_title("Observed runtime scaling")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "runtime_scaling.png"), dpi=190)
    plt.close(fig)


def promotion_gates(comparisons: list[dict]) -> list[dict]:
    candidate = score_key(schema_arm("micro", "ca_specrage_alpha_liu"), 10.0)
    lookup = {
        row["reference_key"]: row
        for row in comparisons if row["candidate_key"] == candidate
    }
    gates = []
    for reference_key, description in (
        ("deployed_upcr", "Mean improvement over deployed U-PCR is at least 0.5 pp"),
        ("iu_pcr", "Mean improvement over IU-PCR is at least 0.5 pp"),
        (score_key("dufs_liu", 0.1), "Mean improvement over DUFS-LIU is at least 0.5 pp"),
    ):
        row = lookup[reference_key]
        gates.append({
            "gate": description,
            "observed": row["mean_delta_pp"],
            "passed": bool(row["mean_delta_pp"] >= 0.5),
        })
    row = lookup["iu_pcr"]
    gates.extend((
        {
            "gate": "Family-bootstrap lower bound versus IU-PCR is above 0 pp",
            "observed": row["family_ci_low_pp"],
            "passed": bool(row["family_ci_low_pp"] > 0.0),
        },
        {
            "gate": "At least 14 of 24 cells improve versus IU-PCR",
            "observed": row["wins"],
            "passed": bool(row["wins"] >= 14),
        },
        {
            "gate": "Worst loss versus IU-PCR is no worse than -2 pp",
            "observed": row["worst_delta_pp"],
            "passed": bool(row["worst_delta_pp"] >= -2.0),
        },
    ))
    for reference_key, description in (
        (score_key(schema_arm("manual", "ca_specrage_alpha_liu"), 10.0),
         "LOCO micro-views improve over manual views"),
        (score_key(schema_arm("atomic", "ca_specrage_alpha_liu"), 10.0),
         "LOCO micro-views do not lose to balanced atomic views"),
    ):
        # These comparisons are not part of the generic baseline table, so the
        # caller appends them before this function is used.
        if reference_key in lookup:
            comparison = lookup[reference_key]
            gates.append({
                "gate": description,
                "observed": comparison["mean_delta_pp"],
                "passed": bool(
                    comparison["mean_delta_pp"] >= 0.0
                    if "do not lose" in description
                    else comparison["mean_delta_pp"] > 0.0
                ),
            })
    return gates


def fmt(value: float, digits: int = 4) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def render_report(
    definition: dict, summary: list[dict], secondary: list[dict], comparisons: list[dict],
    per_cell: list[dict], gates: list[dict], diagnostics: list[dict]
) -> str:
    auroc = [row for row in summary if row["metric"] == "auroc"]
    auprc = [row for row in summary if row["metric"] == "auprc"]
    lookup = lookup_metric(per_cell, "auroc")
    gate_pass = sum(row["passed"] for row in gates)
    unavailable_by_cell = {
        row["cell"]: int(row.get("nonfinite_diagnostic_count", 0))
        for row in diagnostics
    }
    unavailable_count = sum(unavailable_by_cell.values())
    unavailable_cells = sum(value > 0 for value in unavailable_by_cell.values())
    lines = [
        "# Frozen 24-cell unsupervised fusion benchmark",
        "",
        f"Version: `{VERSION}`.",
        "",
        "## Read this first: terms and metrics",
        "",
        "A **cell** is one dataset/model pair. This benchmark contains 24 cells: "
        "9 question-answering cells and 15 mathematics cells. A **feature** is one "
        "continuous hallucination signal. U-PCR treats each feature as an expert that "
        "tries to rank correct answers above incorrect answers.",
        "",
        "A **graph** connects samples that look similar. Its **Laplacian** is a matrix "
        "that measures how quickly a score changes between connected samples. LIU adds "
        "a penalty when the fused score is rough on the graph.",
        "",
        "**AUROC** is the probability that a random correct answer receives a higher "
        "score than a random incorrect answer. 0.5 is random ranking and 1.0 is perfect. "
        "**AUPRC** summarizes precision and recall; its random reference is the positive "
        "rate, so it is especially useful for imbalanced cells. No method is allowed to "
        "flip a score after seeing AUROC.",
        "",
        "A **cell-macro average** gives every one of the 24 cells equal weight. A "
        "**family-macro average** first averages repeated cells from the same dataset "
        "family, then gives each family equal weight. A **95% bootstrap interval** "
        "shows the uncertainty obtained by resampling cells or families.",
        "",
        "## Experimental status",
        "",
        "All model settings, seeds, feature directions, feature exclusions, view-building "
        "rules, and headline lambda values were fixed before this report opened labels. Score "
        "files were hashed first. The fit program never read labels.",
        "",
        "This is still **retrospective development evidence**, not a clean confirmation "
        "set. The same 24 cells influenced earlier feature-contract work, and ten cells "
        "were inspected during the SpecRaGE execution pilot. A second reviewer can reduce "
        "interpretation bias, but cannot make previously seen data statistically unseen.",
        "",
        "![Headline AUROC](figures/headline_auroc.png)",
        "",
        "## Headline results",
        "",
        "| method | cell-macro AUROC [95% CI] | QA | math | family macro | cell-macro AUPRC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    auprc_by_key = {row["method_key"]: row for row in auprc}
    for row in auroc:
        ap = auprc_by_key[row["method_key"]]
        lines.append(
            f"| {row['method']} | {fmt(row['cell_macro'])} "
            f"[{fmt(row['cell_ci_low'])}, {fmt(row['cell_ci_high'])}] | "
            f"{fmt(row['qa_macro'])} | {fmt(row['math_macro'])} | "
            f"{fmt(row['family_macro'])} | {fmt(ap['cell_macro'])} |"
        )
    lines.extend((
        "",
        "The SpecRaGE headline uses `lambda=10`, selected on the earlier synthetic "
        "mechanism study. The full lambda path is a sensitivity analysis only; this "
        "report does not replace the headline with the best observed real-data value.",
        "",
        "## Paired changes",
        "",
        "A positive change means the candidate ranks answers better than the reference. "
        "The cell estimate gives all 24 cells equal weight and its interval resamples "
        "cells. The family estimate first averages within the eight dataset families "
        "and its interval resamples families. Both intervals are paired: candidate and "
        "reference stay together. `Holm p` corrects the paired Wilcoxon tests for "
        "multiple comparisons.",
        "",
        "| candidate | reference | cell mean (pp) [cell 95% CI] | family mean (pp) [family 95% CI] | W/T/L | worst (pp) | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ))
    for row in comparisons:
        lines.append(
            f"| {row['candidate']} | {row['reference']} | "
            f"{row['mean_delta_pp']:+.3f} [{row['cell_ci_low_pp']:+.3f}, "
            f"{row['cell_ci_high_pp']:+.3f}] | "
            f"{row['family_delta_pp']:+.3f} [{row['family_ci_low_pp']:+.3f}, "
            f"{row['family_ci_high_pp']:+.3f}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | "
            f"{row['worst_delta_pp']:+.3f} | {row['holm_p']:.4g} |"
        )
    lines.extend((
        "",
        "## Interface and control results",
        "",
        "These methods test whether any change comes from sample-specific alpha, the "
        "learned embedding Y, or a simpler graph. They use the same frozen lambda 10.",
        "",
        "| method | AUROC | QA | math | AUPRC | change vs IU-PCR (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for row in secondary:
        lines.append(
            f"| {row['method']} | {row['cell_macro_auroc']:.4f} | "
            f"{row['qa_macro_auroc']:.4f} | {row['math_macro_auroc']:.4f} | "
            f"{row['cell_macro_auprc']:.4f} | {row['delta_vs_iu_pp']:+.3f} |"
        )
    lines.extend((
        "",
        "![Paired cell changes](figures/paired_cell_deltas.png)",
        "",
        "![Per-cell heatmap](figures/per_cell_heatmap.png)",
        "",
        "## Predeclared CA-SpecRaGE promotion gates",
        "",
        "These gates prevent a small mean gain from hiding unstable failures. Passing "
        "every gate would justify a new unseen-data confirmation run; it would not by "
        "itself prove generalization.",
        "",
        "| gate | observed | pass |",
        "|---|---:|---:|",
    ))
    for row in gates:
        lines.append(
            f"| {row['gate']} | {row['observed']:.3f} | "
            f"{'yes' if row['passed'] else 'no'} |"
        )
    lines.extend((
        "",
        f"Overall gate result: **{gate_pass}/{len(gates)} passed**. "
        + ("Proceed to unseen confirmation." if gate_pass == len(gates)
           else "Do not promote CA-SpecRaGE from this benchmark."),
        "",
        "## Mechanism checks",
        "",
        "The plots below separate two questions: did the learner actually change its "
        "view reliance and sample ranking, and were those changes useful? High weight "
        "entropy means near-uniform view weights. Rank displacement near zero means the "
        "Laplacian hardly changed IU-PCR.",
        "",
        "![CA mechanism](figures/ca_mechanism.png)",
        "",
        "![Lambda paths](figures/lambda_paths.png)",
        "",
        "![Training convergence](figures/training_convergence.png)",
        "",
        "![View-schema diagnostics](figures/view_schema_diagnostics.png)",
        "",
        "![Runtime scaling](figures/runtime_scaling.png)",
        "",
        f"Unavailable numerical diagnostics: **{unavailable_count} values across "
        f"{unavailable_cells} cells**. These are written as JSON `null`, and their "
        "full paths are listed in `diagnostics.csv`; they are not replaced by zero.",
        "",
        "## View construction experiment",
        "",
        "This run compares three definitions. `manual` uses the old provenance groups. "
        "`atomic` uses one feature per view, but divides equal micro-cluster mass among "
        "near-duplicate features. `micro` clusters features that have a similar and "
        "bootstrap-stable effect on the two-dimensional IU-PCR subspace.",
        "",
        "For each held cell, the micro partition is learned from the other 23 cells only. "
        "Raw projected matrices are not compared across cells: pairwise Frobenius distances "
        "inside each cell are used so eigenvector sign or basis changes do not alter the "
        "distance. Cluster count is selected by a fixed label-free combination of distance "
        "silhouette, bootstrap adjusted Rand stability, singleton fraction, and size "
        "imbalance. Every partition and candidate score is stored for review.",
        "",
        "## Reproducibility files",
        "",
        "- `RUN_DEFINITION.json`: all fixed settings and source hashes.",
        "- `FIT_COMPLETE.json`: per-cell score and diagnostic hashes.",
        "- `SCORE_FREEZE_MANIFEST.json`: verification performed before labels were read.",
        "- `per_cell_metrics.csv`: every cell and method result.",
        "- `headline_summary.csv`, `paired_comparisons.csv`, `lambda_paths.csv`.",
        "- `diagnostics.csv` and `training_history.csv`.",
        "- `REVIEWER_GUIDE.md`: instructions for an independent model or researcher.",
        "",
        "The raw score files contain sample indices, feature names, and scores, but no "
        "labels. The labels remain in the input bundle used only by this report step.",
    ))
    return "\n".join(lines) + "\n"


def render_reviewer_guide(definition: dict) -> str:
    return """# Independent review guide for the frozen 24-cell benchmark

## Purpose

Review the experiment without accepting the generated conclusion. The reviewer
should inspect the registered configuration, recompute the tables from frozen
scores, and look for leakage, unfair comparisons, silent failures, and claims
that exceed the evidence.

## Order of review

1. Read `RUN_DEFINITION.json`. Confirm that it lists exactly 24 cells, 9 QA and
   15 math, and that `scientific_run` is true.
2. Read `FIT_COMPLETE.json` and `SCORE_FREEZE_MANIFEST.json`. Recompute SHA-256
   for every score file. Confirm that no score checkpoint contains labels.
3. Inspect the method documents under `docs/methods/`. For every equation, mark
   whether it comes from a paper or is a project extension.
4. Re-run `python3 scripts/frozen_24cell_report.py`. The CSV files and figures
   should reproduce without fitting a model again.
5. Check that no method receives a per-cell sign flip, a different feature pool,
   or more favorable rows. Confirm that all LIU arms equal IU-PCR exactly at
   lambda zero.
6. Check AUROC and AUPRC independently from the raw scores and bundle labels.
7. Inspect the per-cell deltas, lower tail, view weights, graph diagnostics,
   rank displacement, convergence, and runtime. A mean-only review is incomplete.
8. State explicitly that the 24 cells are retrospective development data. An
   independent reviewer reduces analysis bias; it does not create an unseen
   confirmation set.

## Questions the review must answer

- Does CA-SpecRaGE beat deployed U-PCR, IU-PCR, and DUFS-LIU at its predeclared
  synthetic-transfer setting (`lambda=10`), or only somewhere on the sensitivity
  path?
- If performance changes, did sample-specific alpha matter relative to the
  global, uniform, and permuted controls?
- Did the learned graph materially change IU-PCR ranks, or are methods tied
  because LIU is almost inactive?
- Are gains concentrated in one domain, dataset family, or class prevalence?
- Did any graph collapse, optimizer fail, seed disagree, or projected system
  become ill-conditioned?
- Do LOCO micro-views improve over both manual provenance families and
  duplicate-balanced atomic views at the frozen setting?
- Are the selected micro partitions stable enough to interpret, and are any
  gains preserved when the Y interface replaces the alpha interface?

## Claim boundary

Do not call a result externally validated, unbiased confirmation, or proof of
generalization. The feature contract and method development previously used
information from these cells. A positive result can justify a new-data test.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    definition, complete, frozen = verify_score_freeze(args.out_dir, args.bundle)
    data = np.load(args.bundle, allow_pickle=True)
    validate_bundle(data)
    per_cell, raw_scores = evaluate_scores(frozen, data)
    summary = headline_summary(per_cell)
    secondary = secondary_summary(per_cell)
    comparisons = paired_comparisons(per_cell)
    paths = lambda_summary(per_cell)
    diagnostics, histories = flatten_diagnostics(args.out_dir)
    add_rank_diagnostics(diagnostics, raw_scores)
    gates = promotion_gates(comparisons)

    write_csv(os.path.join(args.out_dir, "per_cell_metrics.csv"), per_cell)
    write_csv(os.path.join(args.out_dir, "headline_summary.csv"), summary)
    write_csv(os.path.join(args.out_dir, "secondary_summary.csv"), secondary)
    write_csv(os.path.join(args.out_dir, "paired_comparisons.csv"), comparisons)
    write_csv(os.path.join(args.out_dir, "lambda_paths.csv"), paths)
    write_csv(os.path.join(args.out_dir, "diagnostics.csv"), diagnostics)
    write_csv(os.path.join(args.out_dir, "training_history.csv"), histories)
    write_json(os.path.join(args.out_dir, "promotion_gates.json"), gates)
    make_plots(
        per_cell, summary, comparisons, paths, diagnostics, histories, args.out_dir
    )
    report = render_report(
        definition, summary, secondary, comparisons, per_cell, gates, diagnostics
    )
    with open(os.path.join(args.out_dir, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(report)
    with open(os.path.join(args.out_dir, "REVIEWER_GUIDE.md"), "w", encoding="utf-8") as handle:
        handle.write(render_reviewer_guide(definition))
    print(report)


if __name__ == "__main__":
    main()
