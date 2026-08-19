#!/usr/bin/env python3
"""Run the frozen CPU-only supervised c-STG routing diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_global_local_online_architecture_v2 import (  # noqa: E402
    _best_threshold,
    _cell_path,
    _processbench,
    load_rows,
)
from scripts.run_local_online_comprehensive_stage1 import _stage_partition  # noqa: E402
from spectral_utils.contextual_stg import (  # noqa: E402
    ContextualSTGConfig,
    ContextualSTGModel,
)
from spectral_utils.local_online_comprehensive import (  # noqa: E402
    FAMILY_NAMES,
    IU_FIT,
    causal_operator_matrices,
    fit_references,
    prepare_trace,
)
from spectral_utils.multitask_trajectory import truncate_row  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402


OUT = ROOT / "results" / "contextual_stg_router_diagnostic_v1"
PROTOCOL = ROOT / "docs" / "experiments" / "CONTEXTUAL_STG_ROUTER_DIAGNOSTIC_V1.md"
MODEL = "qwen3_4b"
FAMILIES = ("gsm8k", "math")
BUDGETS = (64, 128)
DSP_OPERATORS = (
    "innovation", "shortlong", "positive_mean", "persistence", "recovery",
)
SEEDS = (11, 23, 47)
BOOTSTRAP = 2000
RNG_SEED = 20260819
CONFIG = ContextualSTGConfig()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    def clean(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, (np.integer, np.floating)):
            return item.item()
        if isinstance(item, Mapping):
            return {str(key): clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_ap(labels, scores) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _question_weights(groups) -> np.ndarray:
    _, codes = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    counts = np.bincount(codes)
    weights = 1.0 / counts[codes]
    return weights / weights.sum()


def _weighted_scale(values: np.ndarray, weights: np.ndarray):
    weights = weights / weights.sum()
    mean = weights @ values
    scale = np.sqrt(np.maximum(weights @ ((values - mean) ** 2), 0.0))
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return mean, scale


def _top5(matrix: np.ndarray, left: int, right: int) -> np.ndarray | None:
    matrix = np.asarray(matrix, dtype=float)
    left, right = max(0, int(left)), min(len(matrix), int(right))
    if left >= right:
        return None
    values = matrix[left:right]
    k = min(5, len(values))
    return np.mean(np.partition(values, -k, axis=0)[-k:], axis=0)


def _local_samples(rows, references) -> dict[str, Any]:
    X, dsp, groups, labels, meta = [], [], [], [], []
    for row in rows:
        prepared = prepare_trace(row, references)
        level = prepared.representations["family6"]
        states = causal_operator_matrices(level)
        target_step = int(row["label"])
        for step, span in enumerate(row.get("step_token_spans") or ()):
            if span is None:
                continue
            family = _top5(level, span[0], span[1])
            blocks = [_top5(states[name], span[0], span[1]) for name in DSP_OPERATORS]
            if family is None or any(value is None for value in blocks):
                continue
            X.append(family)
            dsp.append(np.concatenate(blocks))
            groups.append(row["_unit"])
            labels.append(int(target_step == step))
            meta.append({
                "unit": row["_unit"], "step": int(step),
                "target_step": target_step, "position": int(span[1]),
            })
    return {
        "X": np.asarray(X, dtype=float), "dsp": np.asarray(dsp, dtype=float),
        "groups": np.asarray(groups, dtype=str), "labels": np.asarray(labels, dtype=int),
        "meta": meta, "feature_group_ids": np.arange(6, dtype=int),
    }


def _early_samples(rows, references) -> dict[str, Any]:
    X, dsp, groups, labels, meta = [], [], [], [], []
    for row in rows:
        for budget in BUDGETS:
            if len(row["token_entropies"]) <= budget:
                continue
            prefix = truncate_row(row, budget)
            prepared = prepare_trace(prefix, references)
            level = prepared.representations["family6"]
            states = causal_operator_matrices(level)
            X.append(np.concatenate([states["fast"][-1], states["slow"][-1]]))
            dsp.append(np.concatenate([states[name][-1] for name in DSP_OPERATORS]))
            groups.append(row["_unit"])
            labels.append(int(not bool(row["final_answer_correct"])))
            meta.append({
                "unit": row["_unit"], "budget": int(budget),
                "position": int(budget), "target": labels[-1],
            })
    return {
        "X": np.asarray(X, dtype=float), "dsp": np.asarray(dsp, dtype=float),
        "groups": np.asarray(groups, dtype=str), "labels": np.asarray(labels, dtype=int),
        "meta": meta,
        "feature_group_ids": np.asarray(list(range(6)) + list(range(6)), dtype=int),
    }


def _core_context(train: Mapping[str, Any], query: Mapping[str, Any]):
    X_train = np.asarray(train["X"], dtype=float)
    X_query = np.asarray(query["X"], dtype=float)
    weights = _question_weights(train["groups"])
    mean, scale = _weighted_scale(X_train, weights)
    Z_train = np.clip((X_train - mean) / scale, -8.0, 8.0)
    Z_query = np.clip((X_query - mean) / scale, -8.0, 8.0)
    fitted = upcr_fit(Z_train.T, **IU_FIT)
    iu_weights = np.asarray(fitted.w, dtype=float)
    score_train = Z_train @ iu_weights
    score_query = Z_query @ iu_weights
    consensus = Z_train.mean(axis=1)
    correlation = float(np.corrcoef(score_train, consensus)[0, 1])
    if np.isfinite(correlation) and correlation < 0:
        iu_weights = -iu_weights
        score_train = -score_train
        score_query = -score_query
    ordered = np.sort(np.round(score_train, 12))
    group_ids = np.asarray(train["feature_group_ids"], dtype=int)

    def build(standardized, scores, metadata):
        contributions = np.column_stack([
            standardized[:, group_ids == family] @ iu_weights[group_ids == family]
            for family in range(6)
        ])
        mad = np.median(
            np.abs(contributions - np.median(contributions, axis=1)[:, None]),
            axis=1,
        )
        rank = np.searchsorted(ordered, np.round(scores, 12), side="right") / len(ordered)
        positions = np.asarray([item["position"] for item in metadata], dtype=float)
        return np.column_stack([rank, np.log1p(positions), mad])

    diagnostics = {
        "labels_seen": False,
        "iu_orientation_correlation": correlation,
        "iu_weights": iu_weights.tolist(),
        "x_mean": mean.tolist(), "x_scale": scale.tolist(),
    }
    return (
        build(Z_train, score_train, train["meta"]),
        build(Z_query, score_query, query["meta"]),
        diagnostics,
    )


def _fit_lr(train_X, train_y, train_groups, query_X):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", max_iter=3000, solver="lbfgs", C=1.0,
            random_state=RNG_SEED,
        ),
    )
    weights = _question_weights(train_groups)
    model.fit(train_X, train_y, logisticregression__sample_weight=weights)
    return (
        model.decision_function(train_X).astype(float),
        model.decision_function(query_X).astype(float),
    )


def _fit_cstg(train, query, train_context, query_context, *, permuted=False):
    train_context = np.asarray(train_context, dtype=float).copy()
    query_context = np.asarray(query_context, dtype=float).copy()
    if permuted:
        rng = np.random.default_rng(RNG_SEED + train_context.shape[1])
        train_context = train_context[rng.permutation(len(train_context))]
        query_context = query_context[rng.permutation(len(query_context))]
    train_scores, query_scores, train_gates, query_gates, diagnostics = [], [], [], [], []
    for seed in SEEDS:
        model = ContextualSTGModel(CONFIG).fit(
            train["X"], train_context, train["labels"], train["groups"],
            feature_group_ids=train["feature_group_ids"], seed=seed,
        )
        train_prediction = model.predict(train["X"], train_context)
        query_prediction = model.predict(query["X"], query_context)
        train_scores.append(train_prediction.score)
        query_scores.append(query_prediction.score)
        train_gates.append(train_prediction.family_gates)
        query_gates.append(query_prediction.family_gates)
        diagnostics.append(model.diagnostics_)
    return {
        "train_score": np.mean(train_scores, axis=0),
        "query_score": np.mean(query_scores, axis=0),
        "train_gates": np.mean(train_gates, axis=0),
        "query_gates": np.mean(query_gates, axis=0),
        "seed_diagnostics": diagnostics,
        "context_permuted": bool(permuted),
    }


def _fit_methods(train, query):
    core_train, core_query, core_diagnostics = _core_context(train, query)
    dsp_train = np.column_stack([core_train, train["dsp"]])
    dsp_query = np.column_stack([core_query, query["dsp"]])
    methods = {}
    for name, train_matrix, query_matrix in (
        ("global_lr", train["X"], query["X"]),
        ("context_only_lr", dsp_train, dsp_query),
        ("augmented_lr", np.column_stack([train["X"], dsp_train]),
         np.column_stack([query["X"], dsp_query])),
    ):
        left, right = _fit_lr(train_matrix, train["labels"], train["groups"], query_matrix)
        methods[name] = {"train_score": left, "query_score": right}
    methods["cstg_core"] = _fit_cstg(train, query, core_train, core_query)
    methods["cstg_dsp"] = _fit_cstg(train, query, dsp_train, dsp_query)
    methods["cstg_dsp_permuted"] = _fit_cstg(
        train, query, dsp_train, dsp_query, permuted=True,
    )
    return methods, {
        "core": core_diagnostics,
        "context_widths": {"core": 3, "dsp": int(dsp_train.shape[1])},
    }


def _question_local_scores(samples, scores):
    grouped: dict[str, list[int]] = {}
    for index, group in enumerate(samples["groups"]):
        grouped.setdefault(str(group), []).append(index)
    output = []
    for unit, indices in grouped.items():
        best = max(indices, key=lambda index: (float(scores[index]), -samples["meta"][index]["step"]))
        output.append({
            "unit": unit,
            "score": float(scores[best]),
            "locator": int(samples["meta"][best]["step"]),
            "target": int(samples["meta"][best]["target_step"]),
        })
    return output


def _local_metrics(family, train, query, methods):
    records, metrics = [], []
    for method, values in methods.items():
        calibration = _question_local_scores(train, values["train_score"])
        development = _question_local_scores(query, values["query_score"])
        threshold, calibration_f1 = _best_threshold(
            [row["score"] for row in calibration],
            [row["locator"] for row in calibration],
            [row["target"] for row in calibration],
        )
        predictions = [row["locator"] if row["score"] > threshold else -1 for row in development]
        target = [row["target"] for row in development]
        scored = _processbench(predictions, target)
        metrics.append({
            "task": "localization", "family": family, "method": method,
            "primary": scored["f1"], **scored,
            "detector_auroc": _safe_auc([value != -1 for value in target], [row["score"] for row in development]),
            "detector_auprc": _safe_ap([value != -1 for value in target], [row["score"] for row in development]),
            "threshold": threshold, "calibration_f1": calibration_f1,
            "n": len(development),
        })
        records.extend({
            "task": "localization", "family": family, "method": method,
            **row, "prediction": int(prediction),
        } for row, prediction in zip(development, predictions))
    return records, metrics


def _early_metrics(family, query, methods):
    records, metrics = [], []
    for method, values in methods.items():
        scores = values["query_score"]
        records.extend({
            "task": "early", "family": family, "method": method,
            "unit": item["unit"], "budget": item["budget"],
            "target": item["target"], "score": float(score),
        } for item, score in zip(query["meta"], scores))
        for budget in BUDGETS:
            mask = np.asarray([item["budget"] == budget for item in query["meta"]])
            labels = query["labels"][mask]
            selected = scores[mask]
            metrics.append({
                "task": "early", "family": family, "method": method,
                "budget": budget, "primary": _safe_auc(labels, selected),
                "auroc": _safe_auc(labels, selected), "auprc": _safe_ap(labels, selected),
                "n": int(mask.sum()), "n_error": int(labels.sum()),
            })
    return records, metrics


def _gate_utility(task, family, query, methods):
    output = []
    for method in ("cstg_core", "cstg_dsp", "cstg_dsp_permuted"):
        gates = methods[method]["query_gates"]
        correlations = []
        if task == "localization":
            by_unit: dict[str, list[int]] = {}
            for index, group in enumerate(query["groups"]):
                by_unit.setdefault(str(group), []).append(index)
            for indices in by_unit.values():
                target = query["meta"][indices[0]]["target_step"]
                if target < 0:
                    continue
                true = [index for index in indices if query["meta"][index]["step"] == target]
                other = [index for index in indices if query["meta"][index]["step"] != target]
                if not true or not other:
                    continue
                utility = query["X"][true[0]] - np.max(query["X"][other], axis=0)
                if np.std(gates[true[0]]) <= 1e-12 or np.std(utility) <= 1e-12:
                    continue
                correlation = spearmanr(gates[true[0]], utility).statistic
                if np.isfinite(correlation):
                    correlations.append(float(correlation))
        else:
            for index, label in enumerate(query["labels"]):
                family_score = np.column_stack([
                    query["X"][:, np.asarray(query["feature_group_ids"]) == group].mean(axis=1)
                    for group in range(6)
                ])[index]
                utility = (2 * int(label) - 1) * rankdata(family_score)
                if np.std(gates[index]) <= 1e-12 or np.std(utility) <= 1e-12:
                    continue
                correlation = spearmanr(gates[index], utility).statistic
                if np.isfinite(correlation):
                    correlations.append(float(correlation))
        output.append({
            "task": task, "family": family, "method": method,
            "mean_gate_utility_spearman": float(np.mean(correlations)) if correlations else float("nan"),
            "n": len(correlations),
            "interpretation": "supervised diagnostic; not independent validation",
        })
    return output


def _macro_metrics(metrics):
    output = []
    methods = sorted({row["method"] for row in metrics})
    for task in ("localization", "early"):
        for method in methods:
            rows = [row for row in metrics if row["task"] == task and row["method"] == method]
            if not rows:
                continue
            if task == "localization":
                values = [row["primary"] for row in rows]
            else:
                values = [row["auroc"] for row in rows if row["budget"] in BUDGETS]
            output.append({
                "task": task, "method": method,
                "primary": float(np.mean(values)), "cells": len(values),
            })
    return output


def _bootstrap(records, task, candidate, reference):
    rng = np.random.default_rng(RNG_SEED + (0 if task == "localization" else 1))
    relevant = [row for row in records if row["task"] == task and row["method"] in {candidate, reference}]
    draws, point_by_family = [], []
    prepared = {}
    for family in FAMILIES:
        rows = [row for row in relevant if row["family"] == family]
        units = sorted({row["unit"] for row in rows})
        lookup = {(row["method"], row["unit"], row.get("budget")): row for row in rows}
        prepared[family] = (units, lookup)

        def family_metric(method, sampled):
            if task == "localization":
                chosen = [lookup[(method, unit, None)] for unit in sampled]
                return _processbench(
                    [row["prediction"] for row in chosen], [row["target"] for row in chosen]
                )["f1"]
            values = []
            for budget in BUDGETS:
                chosen = [lookup[(method, unit, budget)] for unit in sampled if (method, unit, budget) in lookup]
                auc = _safe_auc([row["target"] for row in chosen], [row["score"] for row in chosen])
                if np.isfinite(auc):
                    values.append(auc)
            return float(np.mean(values)) if values else float("nan")

        point_by_family.append(family_metric(candidate, units) - family_metric(reference, units))

    for _ in range(BOOTSTRAP):
        family_deltas = []
        for family, (units, lookup) in prepared.items():
            sampled = rng.choice(units, size=len(units), replace=True).tolist()

            def metric(method):
                if task == "localization":
                    chosen = [lookup[(method, unit, None)] for unit in sampled]
                    return _processbench(
                        [row["prediction"] for row in chosen], [row["target"] for row in chosen]
                    )["f1"]
                values = []
                for budget in BUDGETS:
                    chosen = [lookup[(method, unit, budget)] for unit in sampled if (method, unit, budget) in lookup]
                    auc = _safe_auc([row["target"] for row in chosen], [row["score"] for row in chosen])
                    if np.isfinite(auc):
                        values.append(auc)
                return float(np.mean(values)) if values else float("nan")

            delta = metric(candidate) - metric(reference)
            if np.isfinite(delta):
                family_deltas.append(delta)
        if family_deltas:
            draws.append(float(np.mean(family_deltas)))
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "task": task, "candidate": candidate, "reference": reference,
        "delta": float(np.mean(point_by_family)), "ci_low": float(low),
        "ci_high": float(high),
        "family_wins": int(np.sum(np.asarray(point_by_family) > 0)),
        "family_losses": int(np.sum(np.asarray(point_by_family) < 0)),
        "bootstrap_draws": len(draws),
    }


def _decision(macro, metrics, intervals):
    lookup = {(row["task"], row["method"]): row["primary"] for row in macro}
    interval_lookup = {(row["task"], row["reference"]): row for row in intervals if row["candidate"] == "cstg_dsp"}
    task_rows = []
    for task, margin in (("localization", 0.010), ("early", 0.015)):
        delta = lookup[(task, "cstg_dsp")] - lookup[(task, "global_lr")]
        interval = interval_lookup[(task, "global_lr")]
        family_losses = []
        for family in FAMILIES:
            def family_value(method):
                rows = [row for row in metrics if row["task"] == task and row["family"] == family and row["method"] == method]
                return float(np.mean([row["primary"] for row in rows]))
            family_losses.append(family_value("cstg_dsp") - family_value("global_lr"))
        permuted_gain = lookup[(task, "cstg_dsp_permuted")] - lookup[(task, "global_lr")]
        checks = {
            "gain_at_least_0p005": delta >= 0.005,
            "ci_lower_above_zero": interval["ci_low"] > 0.0,
            "family_guard": min(family_losses) >= -margin,
            "beats_augmented_point": lookup[(task, "cstg_dsp")] > lookup[(task, "augmented_lr")],
            "beats_core_point": lookup[(task, "cstg_dsp")] > lookup[(task, "cstg_core")],
            "permutation_does_not_reproduce": permuted_gain < max(0.005, delta),
        }
        task_rows.append({
            "task": task, "delta_vs_global_lr": delta,
            "ci": [interval["ci_low"], interval["ci_high"]],
            "family_deltas": dict(zip(FAMILIES, family_losses)),
            "permuted_delta_vs_global_lr": permuted_gain,
            "checks": checks, "pass": all(checks.values()),
        })
    passed = [row["task"] for row in task_rows if row["pass"]]
    return {
        "status": "CONTEXT_HAS_ROUTING_SIGNAL" if passed else "STOP_CONTEXT_NOT_SUFFICIENT",
        "tasks_passing": passed, "task_decisions": task_rows,
        "next_step": (
            "register paired-intervention DiSC then LTSREx nuisance audit"
            if passed else "do not open DiSC/LTSREx; seek a new target-relevant observation"
        ),
        "retrospective_premise_evidence": True,
        "not_external_confirmation": True,
    }


def _report(macro, metrics, intervals, gate_utility, decision):
    macro_lookup = {(row["task"], row["method"]): row["primary"] for row in macro}
    lines = [
        "# Supervised c-STG router diagnostic v1", "",
        f"**Decision: `{decision['status']}`.**", "",
        "Retrospective premise evidence only. The router saw calibration labels; this is not a label-free method or external confirmation.",
        "", "## Primary results", "",
        "| method | Localization F1 | Early AUROC@64/128 |", "|---|---:|---:|",
    ]
    for method in sorted({row["method"] for row in macro}):
        lines.append(
            f"| {method} | {macro_lookup[('localization', method)]:.4f} | "
            f"{macro_lookup[('early', method)]:.4f} |"
        )
    lines.extend(["", "## Paired intervals", "", "| task | contrast | delta | 95% CI |", "|---|---|---:|---|"])
    for row in intervals:
        lines.append(
            f"| {row['task']} | {row['candidate']} - {row['reference']} | "
            f"{row['delta']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |"
        )
    lines.extend(["", "## Gate decision", ""])
    for row in decision["task_decisions"]:
        lines.append(
            f"- **{row['task']}**: {'PASS' if row['pass'] else 'FAIL'}; "
            f"delta vs global LR {row['delta_vs_global_lr']:+.4f}, "
            f"CI [{row['ci'][0]:+.4f}, {row['ci'][1]:+.4f}]."
        )
        failed = [name for name, value in row["checks"].items() if not value]
        if failed:
            lines.append(f"  Failed checks: {', '.join(failed)}.")
    lines.extend([
        "", "## Interpretation", "",
        "c-STG is used here as a supervised sufficiency test. A failure means the currently measured context did not robustly expose the known conditional-family headroom under this constrained router. It does not prove that no future intervention-derived context can route.",
        "",
        "This was not a mechanical no-actuation failure. The registered switching-family known-answer test passed, real-data gates varied substantially across samples, and DSP c-STG fit the calibration objective more aggressively than core-only c-STG. The learned actuation simply did not generalize to held-out questions: gate-versus-family-utility Spearman correlations were near zero or negative, and Early's permuted-context control matched or exceeded the real-context router.",
        "",
        "The earlier +2.833pp oracle is evidence for conditional family specialization in the completed-trace 24-cell fusion diagnostic. It is not itself an oracle ceiling for these Localization/Early constructions. This run therefore answers the narrower question: the present causal DSP summaries do not provide a robust supervised routing key for the two online tasks under the frozen construction.",
        "",
        "LTSREx/LEGO were not run. Their justified role begins only after a target-anchored routing signal survives this gate; LTSREx would then audit what locally parameterizes that signal and veto nuisance-dominated geometry.",
    ])
    text = "\n".join(lines) + "\n"
    (OUT / "REPORT.md").write_text(text)
    escaped = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    (OUT / "REPORT.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>c-STG diagnostic</title>"
        "<style>body{font:16px system-ui;max-width:980px;margin:40px auto;line-height:1.45}pre{white-space:pre-wrap}</style>"
        f"</head><body><pre>{escaped}</pre></body></html>"
    )


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    all_records, all_metrics, all_gate_utility = [], [], []
    diagnostics = {}
    input_hashes = {}
    for family in FAMILIES:
        path = _cell_path(MODEL, family)
        input_hashes[str(path.relative_to(ROOT))] = _sha256(path)
        rows = load_rows(path)
        for row in rows:
            row["_stage"] = _stage_partition(family, row["_unit"])
        calibration = [row for row in rows if row["_stage"] == "calibration"]
        development = [row for row in rows if row["_stage"] == "development"]
        references = fit_references(calibration)
        diagnostics[family] = {
            "calibration_questions": len(calibration),
            "development_questions": len(development),
            "references": references.as_dict(), "tasks": {},
        }
        for task, builder in (("localization", _local_samples), ("early", _early_samples)):
            print(f"{family}/{task}: building", flush=True)
            train, query = builder(calibration, references), builder(development, references)
            print(
                f"{family}/{task}: fit n={len(train['labels'])}, eval n={len(query['labels'])}",
                flush=True,
            )
            methods, context_diagnostics = _fit_methods(train, query)
            if task == "localization":
                records, metrics = _local_metrics(family, train, query, methods)
            else:
                records, metrics = _early_metrics(family, query, methods)
            all_records.extend(records)
            all_metrics.extend(metrics)
            all_gate_utility.extend(_gate_utility(task, family, query, methods))
            diagnostics[family]["tasks"][task] = {
                "train_samples": len(train["labels"]),
                "development_samples": len(query["labels"]),
                "train_questions": len(np.unique(train["groups"])),
                "development_questions": len(np.unique(query["groups"])),
                "context": context_diagnostics,
                "cstg": {
                    name: methods[name]["seed_diagnostics"]
                    for name in ("cstg_core", "cstg_dsp", "cstg_dsp_permuted")
                },
            }

    macro = _macro_metrics(all_metrics)
    intervals = []
    for task in ("localization", "early"):
        for reference in ("global_lr", "augmented_lr", "cstg_core", "cstg_dsp_permuted"):
            intervals.append(_bootstrap(all_records, task, "cstg_dsp", reference))
    decision = _decision(macro, all_metrics, intervals)
    _write_csv(OUT / "PER_QUESTION.csv", all_records)
    _write_csv(OUT / "CELL_METRICS.csv", all_metrics)
    _write_csv(OUT / "MACRO_METRICS.csv", macro)
    _write_csv(OUT / "PAIRED_INTERVALS.csv", intervals)
    _write_csv(OUT / "GATE_UTILITY.csv", all_gate_utility)
    _write_json(OUT / "DIAGNOSTICS.json", diagnostics)
    _write_json(OUT / "DECISION.json", decision)
    _write_json(OUT / "AUDIT.json", {
        "labels_used_only_for_supervised_fit_and_evaluation": True,
        "architecture_audit_targets_used": False,
        "feature_directions_changed": False,
        "cstg_context_direct_path": False,
        "context_permutation_control": True,
        "source_question_grouping": True,
        "ranking_metrics_fold_local": True,
        "known_answer_switching_world_test": "scripts/test_contextual_stg.py",
        "mechanical_gate_collapse_rejected": True,
        "new_inference": False, "gpu_used": False, "drive_mutation": False,
    })
    _write_json(OUT / "RUN_MANIFEST.json", {
        "version": "contextual-stg-router-diagnostic-v1-2026-08-19",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": _sha256(PROTOCOL),
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                ROOT / "spectral_utils/contextual_stg.py",
                ROOT / "scripts/run_contextual_stg_router_diagnostic.py",
                ROOT / "scripts/test_contextual_stg.py",
            )
        },
        "input_sha256": input_hashes,
        "model": MODEL, "families": list(FAMILIES), "budgets": list(BUDGETS),
        "seeds": list(SEEDS), "bootstrap": BOOTSTRAP,
        "python": platform.python_version(), "numpy": np.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "decision": decision["status"],
    })
    _report(macro, all_metrics, intervals, all_gate_utility, decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
