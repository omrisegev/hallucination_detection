#!/usr/bin/env python3
"""Conservative protocol-repair report for the frozen SU pooled-graph fit bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    DEFAULT_BUNDLE,
    family as dataset_family,
)
from scripts.inscope_cells import INSCOPE  # noqa: E402
from scripts.su_pooled_graph_adaptation_sidecar import (  # noqa: E402
    ARM_SPECS,
    arm_variants,
    bootstrap_ci,
    candidate_score,
    canonical_auc,
    direction_for,
    mean_group_delta,
    parse_alpha,
    sha256_file,
    variant_id,
    write_csv,
    write_json,
)


warnings.filterwarnings("ignore", category=DeprecationWarning)
VERSION = "su-pooled-graph-adaptation-conservative-v2-2026-08-23"
FIT_ROOT = REPO / "results" / "su_pooled_graph_adaptation_sidecar_v1"
DEFAULT_OUT = REPO / "results" / "su_pooled_graph_adaptation_conservative_v2"
PROTOCOL = REPO / "docs" / "experiments" / "SU_POOLED_GRAPH_ADAPTATION_CONSERVATIVE_V2.md"
FIT_SOURCE = REPO / "scripts" / "su_pooled_graph_adaptation_sidecar.py"
SETTING = "union_k7"
LAMBDAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
TRUSTS = (0.5, 1.0, 2.0)
TAIL_GUARD = -0.005


def candidates(rho_mode: str, clean_mode: str):
    return tuple(
        (variant, lambda_, trust)
        for variant in arm_variants(rho_mode, clean_mode)
        for lambda_ in LAMBDAS
        for trust in TRUSTS
    )


def candidate_inner_values(records, training_groups, items, pooling):
    output = {}
    for item in items:
        variant, lambda_, trust = item
        held_values = []
        for held in training_groups:
            source_groups = [group for group in training_groups if group != held]
            direction, _ = direction_for(
                records, source_groups, variant, SETTING, lambda_, pooling
            )
            held_values.append(mean_group_delta(
                records,
                held,
                lambda row, v=variant, d=direction, t=trust: candidate_score(row, v, d, t),
            ))
        output[item] = np.asarray(held_values, dtype=float)
    return output


def conservative_choice(values):
    means = {item: float(np.mean(scores)) for item, scores in values.items()}
    best = max(values, key=lambda item: means[item])
    best_scores = values[best]
    standard_error = float(np.std(best_scores, ddof=1) / np.sqrt(len(best_scores))) if len(best_scores) > 1 else 0.0
    threshold = means[best] - standard_error
    eligible = [item for item in values if means[item] >= threshold - 1e-15]
    guarded = [item for item in eligible if float(np.min(values[item])) >= TAIL_GUARD]
    pool = guarded if guarded else eligible
    selected = min(
        pool,
        key=lambda item: (item[2], item[1], parse_alpha(item[0]), -means[item]),
    )
    return selected, {
        "best": best,
        "best_mean": means[best],
        "best_standard_error": standard_error,
        "one_se_threshold": threshold,
        "eligible_count": len(eligible),
        "guarded_count": len(guarded),
        "selected_mean": means[selected],
        "selected_worst": float(np.min(values[selected])),
    }


def conservative_upstream_choice(records, training_groups, variants):
    values = {}
    for variant in variants:
        values[variant] = np.asarray([
            mean_group_delta(
                records,
                held,
                lambda row, v=variant: row["payload"][f"baseline__{v}"],
            )
            for held in training_groups
        ], dtype=float)
    means = {variant: float(np.mean(scores)) for variant, scores in values.items()}
    best = max(variants, key=lambda variant: means[variant])
    se = float(np.std(values[best], ddof=1) / np.sqrt(len(values[best]))) if len(values[best]) > 1 else 0.0
    eligible = [variant for variant in variants if means[variant] >= means[best] - se - 1e-15]
    selected = min(eligible, key=lambda variant: (parse_alpha(variant), -means[variant]))
    return selected, {"best_mean": means[best], "one_se": se, "selected_mean": means[selected]}


def paired_bootstrap(left, right, seed_text, draws=50000):
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return bootstrap_ci(differences, seed_text, draws), float(np.mean(differences))


def load_records(bundle_path: Path, fit_root: Path):
    definition = json.loads((fit_root / "RUN_DEFINITION.json").read_text(encoding="utf-8"))
    manifest = json.loads((fit_root / "FIT_MANIFEST.json").read_text(encoding="utf-8"))
    if definition["source_sha256"] != sha256_file(FIT_SOURCE):
        raise RuntimeError("frozen fit source hash mismatch")
    records = []
    with np.load(bundle_path, allow_pickle=True) as bundle:
        for cell in INSCOPE:
            path = fit_root / "cells" / f"{cell}.npz"
            if sha256_file(path) != manifest["cells"][cell]["npz_sha256"]:
                raise RuntimeError(f"frozen fit artifact mismatch: {cell}")
            labels = np.asarray(bundle[f"{cell}__labels"], dtype=int)
            if int(np.sum(labels)) < 20:
                continue
            records.append({
                "cell": cell,
                "group": dataset_family(cell),
                "labels": labels,
                "payload": np.load(path, allow_pickle=False),
            })
    return records, definition, manifest


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--fit-root", type=Path, default=FIT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    records, fit_definition, fit_manifest = load_records(args.bundle.resolve(), args.fit_root.resolve())
    groups = sorted({row["group"] for row in records})

    outer_rows = []
    full_rows = []
    for arm_index, (arm, rho_mode, clean_mode, pooling) in enumerate(ARM_SPECS, 1):
        print(f"[{arm_index:02d}/{len(ARM_SPECS)}] conservative nested {arm}", flush=True)
        items = candidates(rho_mode, clean_mode)
        variants = arm_variants(rho_mode, clean_mode)
        for held in groups:
            training = [group for group in groups if group != held]
            selected, selection_diag = conservative_choice(
                candidate_inner_values(records, training, items, pooling)
            )
            upstream_variant, upstream_diag = conservative_upstream_choice(records, training, variants)
            selected_variant, lambda_, trust = selected
            direction, source_weights = direction_for(
                records, training, selected_variant, SETTING, lambda_, pooling
            )
            held_records = [row for row in records if row["group"] == held]
            graph_delta, matched_delta, increment, independent_delta = [], [], [], []
            for row in held_records:
                iu_auc = canonical_auc(row)
                matched_auc = __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
                    row["labels"], row["payload"][f"baseline__{selected_variant}"]
                )
                graph_auc = __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
                    row["labels"], candidate_score(row, selected_variant, direction, trust)
                )
                upstream_auc = __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
                    row["labels"], row["payload"][f"baseline__{upstream_variant}"]
                )
                graph_delta.append(float(graph_auc - iu_auc))
                matched_delta.append(float(matched_auc - iu_auc))
                increment.append(float(graph_auc - matched_auc))
                independent_delta.append(float(upstream_auc - iu_auc))
            outer_rows.append({
                "arm": arm,
                "held_group": held,
                "rho_mode": rho_mode,
                "clean_mode": clean_mode,
                "pooling": pooling,
                "selected_variant": selected_variant,
                "selected_alpha": parse_alpha(selected_variant),
                "selected_lambda": lambda_,
                "selected_trust": trust,
                "selection_best_mean_pp": 100 * selection_diag["best_mean"],
                "selection_one_se_pp": 100 * selection_diag["best_standard_error"],
                "selection_selected_mean_pp": 100 * selection_diag["selected_mean"],
                "selection_selected_worst_pp": 100 * selection_diag["selected_worst"],
                "selection_eligible_count": selection_diag["eligible_count"],
                "selection_guarded_count": selection_diag["guarded_count"],
                "independent_upstream_variant": upstream_variant,
                "graph_delta_vs_iu_pp": 100 * float(np.mean(graph_delta)),
                "matched_upstream_delta_vs_iu_pp": 100 * float(np.mean(matched_delta)),
                "graph_increment_pp": 100 * float(np.mean(increment)),
                "independent_upstream_delta_vs_iu_pp": 100 * float(np.mean(independent_delta)),
                "direction": json.dumps(direction.tolist()),
                "source_group_weights": json.dumps(dict(zip(training, map(float, source_weights)))),
            })
        full_selected, full_diag = conservative_choice(
            candidate_inner_values(records, groups, items, pooling)
        )
        full_upstream, full_upstream_diag = conservative_upstream_choice(records, groups, variants)
        full_direction, full_weights = direction_for(
            records, groups, full_selected[0], SETTING, full_selected[1], pooling
        )
        full_rows.append({
            "arm": arm,
            "selected_variant": full_selected[0],
            "selected_alpha": parse_alpha(full_selected[0]),
            "selected_lambda": full_selected[1],
            "selected_trust": full_selected[2],
            "selected_cross_validated_delta_vs_iu_pp": 100 * full_diag["selected_mean"],
            "best_cross_validated_delta_vs_iu_pp": 100 * full_diag["best_mean"],
            "one_se_pp": 100 * full_diag["best_standard_error"],
            "selected_worst_pp": 100 * full_diag["selected_worst"],
            "independent_upstream_variant": full_upstream,
            "independent_upstream_cross_validated_delta_vs_iu_pp": 100 * full_upstream_diag["selected_mean"],
            "direction": json.dumps(full_direction.tolist()),
            "source_group_weights": json.dumps(dict(zip(groups, map(float, full_weights)))),
        })

    write_csv(out / "NESTED_OUTER.csv", outer_rows)
    write_csv(out / "FULL_SELECTION.csv", full_rows)
    summary = []
    for arm, _, _, _ in ARM_SPECS:
        selected = [row for row in outer_rows if row["arm"] == arm]
        graph = np.asarray([row["graph_delta_vs_iu_pp"] for row in selected])
        matched = np.asarray([row["matched_upstream_delta_vs_iu_pp"] for row in selected])
        increment = np.asarray([row["graph_increment_pp"] for row in selected])
        independent = np.asarray([row["independent_upstream_delta_vs_iu_pp"] for row in selected])
        graph_ci = bootstrap_ci(graph, VERSION + arm + "graph")
        increment_ci = bootstrap_ci(increment, VERSION + arm + "increment")
        summary.append({
            "arm": arm,
            "nested_graph_delta_vs_iu_pp": float(np.mean(graph)),
            "graph_ci_low_pp": graph_ci[0],
            "graph_ci_high_pp": graph_ci[1],
            "positive_groups": int(np.sum(graph > 0)),
            "worst_group_pp": float(np.min(graph)),
            "matched_upstream_delta_vs_iu_pp": float(np.mean(matched)),
            "graph_increment_pp": float(np.mean(increment)),
            "increment_ci_low_pp": increment_ci[0],
            "increment_ci_high_pp": increment_ci[1],
            "independent_upstream_delta_vs_iu_pp": float(np.mean(independent)),
            "nrm_gain_recovery_fraction": float(np.mean(graph) / 0.277),
        })
    write_csv(out / "SUMMARY.csv", summary)

    by_arm = {row["arm"]: row for row in summary}
    current_group = [row["graph_delta_vs_iu_pp"] for row in outer_rows if row["arm"] == "iu_observed_mean"]
    primary_group = [row["graph_delta_vs_iu_pp"] for row in outer_rows if row["arm"] == "iu_cross_sparse_mean"]
    direct_ci, direct_mean = paired_bootstrap(primary_group, current_group, VERSION + "primary-current")
    direct = {
        "contrast": "iu_cross_sparse_mean - iu_observed_mean",
        "mean_pp": direct_mean,
        "ci_low_pp": direct_ci[0],
        "ci_high_pp": direct_ci[1],
        "positive_groups": int(np.sum((np.asarray(primary_group) - np.asarray(current_group)) > 0)),
    }
    write_json(out / "DIRECT_CONTRAST.json", direct)

    labels = [row[0] for row in ARM_SPECS]
    means = np.asarray([by_arm[arm]["nested_graph_delta_vs_iu_pp"] for arm in labels])
    lows = np.asarray([by_arm[arm]["graph_ci_low_pp"] for arm in labels])
    highs = np.asarray([by_arm[arm]["graph_ci_high_pp"] for arm in labels])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.errorbar(means, y, xerr=np.vstack([means - lows, highs - means]), fmt="o", capsize=3)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(0.277, color="#888888", linestyle="--", label="Family-NRM +0.277pp")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Conservative nested AUROC delta vs IU-PCR (pp)")
    ax.set_title("SU-aware cleaning with canonical k=7 and one-SE selection")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "CONSERVATIVE_METHOD_COMPARISON.png", dpi=180)
    plt.close(fig)

    matched = np.asarray([by_arm[arm]["matched_upstream_delta_vs_iu_pp"] for arm in labels])
    increments = np.asarray([by_arm[arm]["graph_increment_pp"] for arm in labels])
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.barh(y, matched, label="clean covariance / rho", color="#4C78A8")
    ax.barh(y, increments, left=matched, label="pooled graph increment", color="#F58518")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC delta vs canonical IU-PCR (pp)")
    ax.set_title("Conservative outer-fold mechanism decomposition")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "CONSERVATIVE_MECHANISM_DECOMPOSITION.png", dpi=180)
    plt.close(fig)

    heat = np.asarray([
        [next(row for row in outer_rows if row["arm"] == arm and row["held_group"] == group)["graph_delta_vs_iu_pp"] for group in groups]
        for arm in labels
    ])
    fig, ax = plt.subplots(figsize=(12, 7))
    limit = max(0.25, float(np.max(np.abs(heat))))
    image = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(groups)), groups, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Conservative outer-family AUROC delta vs IU-PCR (pp)")
    fig.colorbar(image, ax=ax, label="pp")
    fig.tight_layout()
    fig.savefig(out / "CONSERVATIVE_OUTER_FAMILY_HEATMAP.png", dpi=180)
    plt.close(fig)

    current, primary = by_arm["iu_observed_mean"], by_arm["iu_cross_sparse_mean"]
    lines = [
        "# SU-aware pooled graph adaptation — conservative V2",
        "",
        "V2 repairs the failed V1 reproduction by fixing union k=7 and using the canonical one-SE/tail-guard selector. V1 is an optimistic sensitivity only.",
        "",
        "## Headline",
        "",
        f"- Current observed-IU pooled graph: **{current['nested_graph_delta_vs_iu_pp']:+.3f}pp** "
        f"[{current['graph_ci_low_pp']:+.3f},{current['graph_ci_high_pp']:+.3f}], {current['positive_groups']}/8 wins.",
        f"- Prespecified IU + cross-family sparse cleaning: **{primary['nested_graph_delta_vs_iu_pp']:+.3f}pp** "
        f"[{primary['graph_ci_low_pp']:+.3f},{primary['graph_ci_high_pp']:+.3f}], graph increment {primary['graph_increment_pp']:+.3f}pp.",
        f"- Direct primary minus current: **{direct['mean_pp']:+.3f}pp** "
        f"[{direct['ci_low_pp']:+.3f},{direct['ci_high_pp']:+.3f}], {direct['positive_groups']}/8 families.",
        "",
        "## Arms",
        "",
        "| arm | graph vs IU | 95% CI | wins | clean/rho alone | graph increment | independently selected no-graph |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['arm']}` | {row['nested_graph_delta_vs_iu_pp']:+.3f} | "
            f"[{row['graph_ci_low_pp']:+.3f},{row['graph_ci_high_pp']:+.3f}] | "
            f"{row['positive_groups']}/8 | {row['matched_upstream_delta_vs_iu_pp']:+.3f} | "
            f"{row['graph_increment_pp']:+.3f} | {row['independent_upstream_delta_vs_iu_pp']:+.3f} |"
        )
    lines.extend([
        "",
        "## Claim boundary",
        "",
        "All directions and fit artifacts are label-free; lambda, trust, and alpha are retrospectively meta-selected in nested folds. "
        "The primary comparison is the paired clean-minus-current contrast, not whether each arm separately exceeds IU. "
        "No arm is confirmation until frozen transfer is run.",
        "",
    ])
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(out / "RUN_DEFINITION.json", {
        "version": VERSION,
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_sha256": sha256_file(Path(__file__)),
        "fit_root": str(args.fit_root.resolve()),
        "fit_definition_sha256": sha256_file(args.fit_root.resolve() / "RUN_DEFINITION.json"),
        "fit_manifest_sha256": sha256_file(args.fit_root.resolve() / "FIT_MANIFEST.json"),
        "bundle_sha256": sha256_file(args.bundle.resolve()),
        "setting": SETTING,
        "lambdas": list(LAMBDAS),
        "trusts": list(TRUSTS),
        "tail_guard_auroc": TAIL_GUARD,
        "retrospective": True,
    })
    write_json(out / "REPORT_COMPLETE.json", {
        "version": VERSION,
        "current_pp": current["nested_graph_delta_vs_iu_pp"],
        "primary_pp": primary["nested_graph_delta_vs_iu_pp"],
        "direct_primary_minus_current": direct,
        "n_cells": len(records),
        "n_groups": len(groups),
    })
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
