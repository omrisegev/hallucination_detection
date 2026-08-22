#!/usr/bin/env python3
"""Evaluate the immutable B0/B1/B2/B3 score freeze; this is the target-access boundary."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.residual_graph_deem import (  # noqa: E402
    ResidualGraphDeemError, atomic_write_json, canonical_sha256, sha256_file,
)
from spectral_utils.residual_graph_deem_data import load_registry, load_target_free_bundle  # noqa: E402
from spectral_utils.residual_graph_deem_labels import join_labels_by_id, load_label_sidecar  # noqa: E402
from scripts.evaluate_residual_graph_deem_24cell_v1 import (  # noqa: E402
    conditional_draws, ensemble, verify_score_freeze,
)
from scripts.run_deem_vs_iupcr_24cell_v1 import load_experiment_config  # noqa: E402

ARMS = ("B0", "B1", "B2", "B3")
CONTRASTS = (("B0", "B3"), ("B1", "B3"), ("B2", "B3"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def metrics(y, score) -> dict[str, float]:
    target = np.asarray(y, dtype=np.int8)
    values = np.asarray(score, dtype=np.float64)
    if values.shape != target.shape or len(np.unique(target)) != 2 or not np.isfinite(values).all():
        raise ResidualGraphDeemError("invalid target/score pair")
    return {"auroc": float(roc_auc_score(target, values)),
            "auprc": float(average_precision_score(target, values))}


def summary(per_cell: list[dict], arm: str, metric: str) -> dict:
    rows = [row for row in per_cell if row["arm_id"] == arm]
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["dataset_family"]].append(float(row[metric]))
    family_means = {family: float(np.mean(values)) for family, values in by_family.items()}
    return {
        "arm_id": arm, "metric": metric,
        "equal_family_macro": float(np.mean(list(family_means.values()))),
        "qa_macro": float(np.mean([row[metric] for row in rows if row["task_type"] == "QA"])),
        "math_macro": float(np.mean([row[metric] for row in rows if row["task_type"] == "math"])),
        "worst_cell": float(min(row[metric] for row in rows)),
        "worst_family": float(min(family_means.values())),
        "family_means": family_means,
    }


def bootstrap(per_cell: list[dict], reference: str, candidate: str, metric: str,
              *, draws: int, seed: int) -> dict:
    paired = defaultdict(list)
    lookup = {(row["cell_id"], row["arm_id"]): row for row in per_cell}
    for cell in sorted({row["cell_id"] for row in per_cell}):
        family = lookup[(cell, reference)]["dataset_family"]
        paired[family].append(float(lookup[(cell, candidate)][metric] - lookup[(cell, reference)][metric]))
    families = sorted(paired)
    family_delta = {family: float(np.mean(paired[family])) for family in families}
    generator = np.random.Generator(np.random.PCG64(seed))
    distribution = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = generator.choice(families, len(families), replace=True)
        values = []
        for family in selected:
            array = np.asarray(paired[family], dtype=np.float64)
            values.append(float(np.mean(array[generator.integers(0, len(array), len(array))])))
        distribution[draw] = float(np.mean(values))
    observed = float(np.mean(list(family_delta.values())))
    return {
        "reference": reference, "candidate": candidate, "metric": metric,
        "observed": observed, "lower": float(np.quantile(distribution, .025)),
        "upper": float(np.quantile(distribution, .975)),
        "one_sided_p": float((1 + np.sum(distribution <= 0.0)) / (draws + 1)),
        "family_delta": family_delta,
        "leave_one_family_out": {omit: float(np.mean([v for f, v in family_delta.items() if f != omit]))
                                  for omit in families},
        "distribution": distribution,
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    m = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def _delta_summary(targets, bundles, scores, reference, candidate, metric, scope) -> float:
    values = defaultdict(list)
    flat = []
    for cell, y in targets.items():
        if scope in {"QA", "math"} and bundles[cell].task_type != scope:
            continue
        ref = metrics(y, scores[cell][reference])[metric]
        cand = metrics(y, scores[cell][candidate])[metric]
        delta = cand - ref
        if scope == "equal_family":
            values[bundles[cell].dataset_family].append(delta)
        else:
            flat.append(delta)
    if scope == "equal_family":
        return float(np.mean([np.mean(rows) for rows in values.values()]))
    return float(np.mean(flat))


def whole_search_null(targets, bundles, scores, draws, *, B: int) -> dict:
    keys = [(ref, cand, metric, scope) for ref, cand in CONTRASTS
            for metric in ("auroc", "auprc") for scope in ("equal_family", "QA", "math")]
    observed = {"|".join(key): _delta_summary(targets, bundles, scores, *key) for key in keys}
    observed_max = float(max(observed.values()))
    output = {"observed": observed, "observed_max": observed_max, "B": int(B)}
    for null_name, cell_draws in draws.items():
        null_matrix = np.empty((B, len(keys)), dtype=np.float64)
        for draw in range(B):
            target_draw = {cell: cell_draws[cell][:, draw] for cell in targets}
            null_matrix[draw] = [
                _delta_summary(target_draw, bundles, scores, *key) for key in keys
            ]
        null_max = np.max(null_matrix, axis=1)
        output[null_name] = {
            "p_value": float((1 + np.sum(null_max >= observed_max)) / (B + 1)),
            "p_by_statistic": {
                "|".join(key): float(
                    (1 + np.sum(null_max >= observed["|".join(key)])) / (B + 1)
                )
                for key in keys
            },
            "null_max_mean": float(np.mean(null_max)),
            "null_max_q95": float(np.quantile(null_max, .95)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--B", type=int, choices=(199, 999), default=199)
    parser.add_argument("--promotion-decision", type=Path)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    registry = load_registry(args.registry)
    freeze = verify_score_freeze(args.run_dir)
    definition = json.loads((args.run_dir / "RUN_DEFINITION.json").read_text(encoding="utf-8"))
    if (
        definition.get("config_sha256") != sha256_file(args.config)
        or definition.get("registry_content_sha256") != registry["registry_content_sha256"]
        or registry["registry_content_sha256"] != config["source_registry_content_sha256"]
    ):
        raise SystemExit("evaluator config/registry drift from immutable Stage A")
    if freeze.get("arms") != list(ARMS) or freeze.get("expected_fit_artifacts") != 480:
        raise SystemExit("score freeze is not the frozen four-arm benchmark")
    if args.B == 999:
        if args.promotion_decision is None:
            parser.error("B=999 requires --promotion-decision")
        promoted = json.loads(args.promotion_decision.read_text(encoding="utf-8"))
        if not promoted.get("eligible_for_B999"):
            raise SystemExit("B=999 is not eligible under the frozen B=199 decision")
    bundles, targets, scores = {}, {}, {}
    per_fit, per_cell, stability_rows = [], [], []
    for cell_row in registry["cells"]:
        cell = cell_row["cell_id"]
        bundle = load_target_free_bundle(args.bundle_dir / f"{cell}.npz")
        sidecar = load_label_sidecar(args.sidecar_dir / f"{cell}.npz")
        y = join_labels_by_id(bundle, sidecar)
        bundles[cell], targets[cell], scores[cell] = bundle, y, {}
        for arm in ARMS:
            ensemble_score, stability = ensemble(args.run_dir, cell, arm)
            scores[cell][arm] = ensemble_score
            values = metrics(y, ensemble_score)
            per_cell.append({"cell_id": cell, "dataset_family": bundle.dataset_family,
                             "task_type": bundle.task_type, "arm_id": arm, **values})
            stability_rows.append({"cell_id": cell, "arm_id": arm, **stability})
            for seed in config["seeds"]:
                with np.load(args.run_dir / "fits" / cell / f"{arm}__seed{seed}.npz",
                             allow_pickle=False) as data:
                    current = metrics(y, np.asarray(data["score"], dtype=np.float64))
                per_fit.append({"cell_id": cell, "dataset_family": bundle.dataset_family,
                                "task_type": bundle.task_type, "arm_id": arm,
                                "seed": seed, **current})
    summaries = [summary(per_cell, arm, metric) for arm in ARMS for metric in ("auroc", "auprc")]
    boots = []
    for index, (reference, candidate) in enumerate(CONTRASTS):
        for metric_offset, metric in enumerate(("auroc", "auprc")):
            boots.append(bootstrap(per_cell, reference, candidate, metric,
                                   draws=int(config["evaluation"]["bootstrap_draws"]),
                                   seed=int(config["evaluation"]["base_seed"]) + 10 * index + metric_offset))
    auroc_p = {row["reference"]: row["one_sided_p"] for row in boots if row["metric"] == "auroc"}
    adjusted = holm(auroc_p)
    lookup = {(row["arm_id"], row["metric"]): row for row in summaries}
    comparisons = []
    for reference, candidate in CONTRASTS:
        boot = next(row for row in boots if row["reference"] == reference and row["metric"] == "auroc")
        ref = lookup[(reference, "auroc")]; cand = lookup[(candidate, "auroc")]
        deltas = []
        for cell in targets:
            c = next(row for row in per_cell if row["cell_id"] == cell and row["arm_id"] == candidate)
            r = next(row for row in per_cell if row["cell_id"] == cell and row["arm_id"] == reference)
            deltas.append(c["auroc"] - r["auroc"])
        tolerance = float(config["evaluation"]["tie_tolerance"])
        comparisons.append({
            "reference": reference, "candidate": candidate,
            "equal_family_auroc_delta": boot["observed"], "lower": boot["lower"], "upper": boot["upper"],
            "holm_p": adjusted[reference],
            "qa_delta": cand["qa_macro"] - ref["qa_macro"],
            "math_delta": cand["math_macro"] - ref["math_macro"],
            "wins": int(sum(value > tolerance for value in deltas)),
            "ties": int(sum(abs(value) <= tolerance for value in deltas)),
            "losses": int(sum(value < -tolerance for value in deltas)),
            "worst_cell_delta": float(min(deltas)),
        })
    draws, null_diagnostics = conditional_draws(targets, bundles, B=args.B,
                                                 seed=int(config["evaluation"]["base_seed"]))
    null = whole_search_null(targets, bundles, scores, draws, B=args.B)
    b3_stability = [row for row in stability_rows if row["arm_id"] == "B3"]
    stable = len(b3_stability) == 24 and all(
        row["median_abs_spearman"] >= config["health"]["minimum_median_seed_spearman"]
        for row in b3_stability
    )
    primary = next(row for row in comparisons if row["reference"] == "B0")
    primary_null_key = "B0|B3|auroc|equal_family"
    null_pass = all(null[name]["p_by_statistic"][primary_null_key] <= config["evaluation"]["alpha"]
                    for name in ("exact", "crt", "family_group"))
    superiority = (
        primary["equal_family_auroc_delta"] >= config["evaluation"]["primary_delta"]
        and primary["lower"] > 0 and primary["holm_p"] <= config["evaluation"]["alpha"]
        and primary["qa_delta"] >= -config["evaluation"]["task_degradation_margin"]
        and primary["math_delta"] >= -config["evaluation"]["task_degradation_margin"]
        and primary["wins"] + primary["ties"] >= config["evaluation"]["minimum_wins_or_ties"]
        and primary["worst_cell_delta"] >= config["evaluation"]["worst_cell_delta"] and null_pass
    )
    noninferior = (primary["lower"] > -config["evaluation"]["noninferiority_margin"]
                   and primary["qa_delta"] >= -config["evaluation"]["task_degradation_margin"]
                   and primary["math_delta"] >= -config["evaluation"]["task_degradation_margin"])
    adapter_superiority = all(
        row["equal_family_auroc_delta"] >= config["evaluation"]["adapter_delta"]
        and row["lower"] > 0 and row["holm_p"] <= config["evaluation"]["alpha"]
        and all(
            null[name]["p_by_statistic"][f"{row['reference']}|B3|auroc|equal_family"]
            <= config["evaluation"]["alpha"]
            for name in ("exact", "crt", "family_group")
        )
        for row in comparisons if row["reference"] in {"B1", "B2"}
    )
    if not stable:
        verdict = "CONTINUOUS_DEEM_UNSTABLE"
    elif superiority and adapter_superiority:
        verdict = "CONTINUOUS_DEEM_SUPERIOR_TO_IUPCR_AND_ADAPTERS"
    elif superiority:
        verdict = "CONTINUOUS_DEEM_SUPERIOR_TO_IUPCR"
    elif primary["upper"] < -config["evaluation"]["noninferiority_margin"]:
        verdict = "CONTINUOUS_DEEM_INFERIOR_TO_IUPCR"
    elif noninferior:
        verdict = "CONTINUOUS_DEEM_NONINFERIOR_TO_IUPCR"
    else:
        verdict = "CONTINUOUS_DEEM_NO_ADVANTAGE"
    decision = {
        "schema": "deem_vs_iupcr_decision_v1", "decision": verdict, "B": args.B,
        "eligible_for_B999": bool(args.B == 199 and superiority and adapter_superiority and stable),
        "graph_direction_remains_closed": True, "primary": primary,
        "adapter_superiority": adapter_superiority, "b3_stability_pass": stable,
        "conditional_max_null": null,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "PER_FIT_METRICS.csv", per_fit)
    write_csv(args.out_dir / "PER_CELL_METRICS.csv", per_cell)
    write_csv(args.out_dir / "SEED_STABILITY.csv", stability_rows)
    write_csv(args.out_dir / "PAIRWISE_COMPARISONS.csv", comparisons)
    atomic_write_json(args.out_dir / "FAMILY_SUMMARY.json", summaries)
    atomic_write_json(args.out_dir / "BOOTSTRAP.json", [{**row, "distribution": row["distribution"].tolist()} for row in boots])
    atomic_write_json(args.out_dir / "WHOLE_SEARCH_NULL.json", {**null, "diagnostics": null_diagnostics})
    atomic_write_json(args.out_dir / "DECISION.json", decision)
    complete = {"schema": "deem_vs_iupcr_evaluation_complete_v1", "status": "complete",
                "score_freeze_sha256": sha256_file(args.run_dir / "SCORE_FREEZE_MANIFEST.json"),
                "sidecar_manifest_sha256": sha256_file(args.sidecar_dir / "LABEL_SIDECARS.json"),
                "decision_sha256": sha256_file(args.out_dir / "DECISION.json")}
    complete["content_sha256"] = canonical_sha256(complete)
    atomic_write_json(args.out_dir / "EVALUATION_COMPLETE.json", complete)


if __name__ == "__main__":
    main()
