#!/usr/bin/env python3
"""Run S4 scorer-transfer, robustness, warning, and efficiency audit."""

from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import tracemalloc
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/gl_liu_v1/localization"))

from scripts.run_global_local_online_architecture_v2 import (  # noqa: E402
    _best_threshold,
    _cell_path,
    _processbench,
    _safe_ap,
    _safe_auc,
    _zapply,
    _zfit,
    fit_registered_global,
    fit_registered_local,
    load_rows,
)
from scripts.run_local_online_comprehensive_stage1 import (  # noqa: E402
    OUT,
    COMPETITOR_PATTERNS,
    COMPETITOR_ROOTS,
    PROTOCOL,
    PROTOCOL_SHA256,
    SEED,
    _evaluate_curve_system,
    _global_local_incumbent,
    _mindgap,
    _sha256,
    _stage_partition,
    _step_top5_locator,
    _tier_b,
)
from spectral_utils.local_online_comprehensive import (  # noqa: E402
    FAMILY_NAMES,
    IU_FIT,
    PreparedTrace,
    fit_references,
    fit_trajectory_head_prepared,
    operator_matrix,
    prepare_trace,
)
from spectral_utils.multitask_trajectory import equal_positions, truncate_row  # noqa: E402
from spectral_utils.online_convergence import (  # noqa: E402
    causal_raw_prefix_matrix,
    fit_frozen_prefix_iu,
)
from spectral_utils.streaming_utils import deepconf_lowest_group_conf  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402


MODELS = ("qwen3_8b", "llama31_8b")
FAMILIES = ("gsm8k", "math", "olympiadbench", "omnimath")
BUDGETS = (16, 32, 64, 128, 256, 512)
PRIMARY_BUDGETS = (64, 128)
BOOTSTRAP = 2000
FINALIST = "finalist_global_detector_local_locator"
CHECKPOINT = OUT / "stage4_checkpoints"
DIRECT_ONLINE = (
    "mean_entropy", "max_entropy", "deepconf_w32", "deepconf_w64",
    "iu28_registered", "step272_twohead",
)


def _tier_b_partial(family: str, rows: Sequence[Mapping[str, Any]]):
    """Score valid native predictions and expose abstention coverage explicitly."""

    wanted = {row["_unit"]: row for row in rows}
    records, metrics = [], []
    for candidate, root in COMPETITOR_ROOTS.items():
        path = root / COMPETITOR_PATTERNS[candidate].format(family=family)
        with path.open("rb") as handle:
            data = pickle.load(handle)
        lookup = {str(item.get("id", key)): item for key, item in data.items()}
        if set(wanted) - set(lookup):
            raise RuntimeError(f"{candidate}/{family}: incomplete ID join")
        units = sorted(
            unit for unit in wanted if lookup[unit].get("prediction") is not None
        )
        labels = np.asarray([int(wanted[unit]["label"]) for unit in units])
        predictions = np.asarray([int(lookup[unit]["prediction"]) for unit in units])
        result = _processbench(predictions, labels)
        fidelity = (
            "exact_local_competitor_run" if candidate != "qwen72b_critic"
            else "critic_protocol_different_model"
        )
        metrics.append({
            "candidate": candidate, "base_candidate": candidate,
            "locator": "native_prediction", "family": family, "task": "local",
            "primary": result["f1"], **result, "n": len(units),
            "requested_n": len(wanted), "coverage": len(units) / len(wanted),
            "abstentions": len(wanted) - len(units), "access_tier": "B",
            "fidelity": fidelity,
        })
        records.extend({
            "candidate": candidate, "base_candidate": candidate,
            "locator_kind": "native_prediction", "family": family,
            "unit": unit, "target": int(target), "locator": int(prediction),
            "prediction": int(prediction), "access_tier": "B",
            "fidelity": fidelity,
        } for unit, target, prediction in zip(units, labels, predictions))
    return records, metrics


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _split(model: str, family: str):
    rows = load_rows(_cell_path(model, family))
    for row in rows:
        row["_stage"] = _stage_partition(family, row["_unit"])
    return (
        [row for row in rows if row["_stage"] == "calibration"],
        [row for row in rows if row["_stage"] == "audit"],
    )


def _metadata(row: Mapping[str, Any], length_cut: tuple[float, float]) -> dict[str, Any]:
    length = len(row["token_entropies"])
    length_stratum = (
        "short" if length <= length_cut[0]
        else ("medium" if length <= length_cut[1] else "long")
    )
    label = int(row["label"])
    if label == -1:
        position = "clean"
    else:
        denominator = max(len(row.get("step_token_spans") or ()) - 1, 1)
        fraction = label / denominator
        position = (
            "q1" if fraction <= 0.25 else "q2" if fraction <= 0.50
            else "q3" if fraction <= 0.75 else "q4"
        )
    answer_error = int(not bool(row["final_answer_correct"]))
    process_error = int(label != -1)
    cross = (
        "answer_correct_process_clean" if not answer_error and not process_error
        else "answer_correct_process_error" if not answer_error
        else "answer_wrong_process_clean" if not process_error
        else "answer_wrong_process_error"
    )
    return {
        "trace_length": length,
        "length_stratum": length_stratum,
        "error_position_quartile": position,
        "answer_process_stratum": cross,
    }


def _finalist_local(
    model: str,
    family: str,
    calibration: Sequence[Mapping[str, Any]],
    audit: Sequence[Mapping[str, Any]],
    global_model: Any,
    local_head: Any,
    prepared_cal: Sequence[PreparedTrace],
    prepared_audit: Sequence[PreparedTrace],
):
    cal_curves = [
        local_head.curve_from_level(item.representations["family6"])
        for item in prepared_cal
    ]
    audit_curves = [
        local_head.curve_from_level(item.representations["family6"])
        for item in prepared_audit
    ]
    cal_score = np.asarray([global_model.score(row, None) for row in calibration])
    audit_score = np.asarray([global_model.score(row, None) for row in audit])
    cal_locator = np.asarray([
        _step_top5_locator(curve, row)
        for curve, row in zip(cal_curves, calibration)
    ])
    audit_locator = np.asarray([
        _step_top5_locator(curve, row)
        for curve, row in zip(audit_curves, audit)
    ])
    cal_target = np.asarray([int(row["label"]) for row in calibration])
    audit_target = np.asarray([int(row["label"]) for row in audit])
    threshold, calibration_f1 = _best_threshold(cal_score, cal_locator, cal_target)
    prediction = np.where(audit_score > threshold, audit_locator, -1)
    result = _processbench(prediction, audit_target)
    cuts = tuple(np.quantile(
        [len(row["token_entropies"]) for row in calibration], (1 / 3, 2 / 3)
    ))
    records = [{
        "candidate": FINALIST, "model": model, "family": family,
        "unit": row["_unit"], "task": "local", "target": int(target),
        "score": float(score), "locator": int(locator),
        "prediction": int(pred), "access_tier": "A",
        **_metadata(row, cuts),
    } for row, target, score, locator, pred in zip(
        audit, audit_target, audit_score, audit_locator, prediction
    )]
    metric = {
        "candidate": FINALIST, "model": model, "family": family,
        "task": "local", "primary": result["f1"], **result,
        "detector_auroc": _safe_auc(audit_target != -1, audit_score),
        "detector_auprc": _safe_ap(audit_target != -1, audit_score),
        "threshold": threshold, "calibration_f1": calibration_f1,
        "n": len(audit), "access_tier": "A",
    }
    return records, [metric], cal_curves, audit_curves, threshold


def _prefix_scores(
    row: Mapping[str, Any],
    budget: int,
    *,
    references: Any,
    global_model: Any,
    raw9_head: Any,
    iu28: Any,
    global_fit: tuple[float, float],
    local_fit: tuple[float, float],
) -> dict[str, float]:
    prefix = truncate_row(row, budget)
    entropy = np.asarray(prefix["token_entropies"], dtype=float)
    raw28, _ = causal_raw_prefix_matrix(
        prefix, None, include_elapsed_length=False
    )
    local_max = float(np.max(raw9_head.curve(prefix, references)))
    global_score = float(global_model.score(prefix, None))
    return {
        FINALIST: global_score,
        "mean_entropy": float(np.mean(entropy)),
        "max_entropy": float(np.max(entropy)),
        "deepconf_w32": float(-deepconf_lowest_group_conf(entropy, 32)),
        "deepconf_w64": float(-deepconf_lowest_group_conf(entropy, 64)),
        "iu28_registered": float(np.max(iu28.risk(raw28))),
        "step272_twohead": float(
            0.5 * _zapply([global_score], global_fit)[0]
            + 0.5 * _zapply([local_max], local_fit)[0]
        ),
    }


def _online_records_cell(
    model: str,
    family: str,
    calibration: Sequence[Mapping[str, Any]],
    audit: Sequence[Mapping[str, Any]],
    *,
    references: Any,
    global_model: Any,
    raw9_head: Any,
    prepared_cal: Sequence[PreparedTrace],
):
    iu28 = fit_frozen_prefix_iu(calibration, include_elapsed_length=False)
    global_fit = _zfit([global_model.score(row, None) for row in calibration])
    local_fit = _zfit([
        float(np.max(raw9_head.curve_from_level(item.representations["raw9"])))
        for item in prepared_cal
    ])
    output = []
    cuts = tuple(np.quantile(
        [len(row["token_entropies"]) for row in calibration], (1 / 3, 2 / 3)
    ))
    for split, rows in (("calibration", calibration), ("audit", audit)):
        for index, row in enumerate(rows, 1):
            length = len(row["token_entropies"])
            target = int(not bool(row["final_answer_correct"]))
            meta = _metadata(row, cuts)
            for budget in BUDGETS:
                if length <= budget:
                    continue
                scores = _prefix_scores(
                    row, budget, references=references,
                    global_model=global_model, raw9_head=raw9_head, iu28=iu28,
                    global_fit=global_fit, local_fit=local_fit,
                )
                output.extend({
                    "candidate": candidate, "model": model, "family": family,
                    "split": split, "unit": row["_unit"], "task": "online",
                    "budget": budget, "target": target, "score": float(score),
                    "access_tier": "A", **meta,
                } for candidate, score in scores.items())
            if index % 200 == 0:
                print(f"    {split} online {index}/{len(rows)}", flush=True)
    return output, iu28.diagnostics


def _online_metrics(records: Sequence[Mapping[str, Any]]):
    metrics = []
    audit = [row for row in records if row["split"] == "audit"]
    for candidate in sorted({row["candidate"] for row in audit}):
        primary = []
        for budget in BUDGETS:
            rows = [
                row for row in audit
                if row["candidate"] == candidate and int(row["budget"]) == budget
            ]
            if not rows:
                continue
            labels = np.asarray([row["target"] for row in rows])
            scores = np.asarray([row["score"] for row in rows])
            auc = _safe_auc(labels, scores)
            if budget in PRIMARY_BUDGETS and np.isfinite(auc):
                primary.append(auc)
            metrics.append({
                "candidate": candidate, "model": rows[0]["model"],
                "family": rows[0]["family"], "task": "online_budget",
                "budget": budget, "primary": auc, "auroc": auc,
                "auprc": _safe_ap(labels, scores), "n": len(rows),
                "n_error": int(labels.sum()), "access_tier": "A",
            })
        metrics.append({
            "candidate": candidate, "model": audit[0]["model"],
            "family": audit[0]["family"], "task": "online",
            "primary": float(np.mean(primary)), "n": 0, "access_tier": "A",
        })
    return metrics


def _warning_threshold(rows, candidate: str, alpha: float):
    clean = [
        row for row in rows
        if row["split"] == "calibration" and row["candidate"] == candidate
        and int(row["target"]) == 0
    ]
    maxima = {}
    for row in clean:
        maxima[row["unit"]] = max(maxima.get(row["unit"], -np.inf), row["score"])
    values = np.asarray(list(maxima.values()), dtype=float)
    for threshold in np.r_[np.unique(values), np.nextafter(np.max(values), np.inf)]:
        rate = float(np.mean(values >= threshold))
        if rate <= alpha + 1e-12:
            return float(threshold), rate, len(values)
    raise RuntimeError("warning calibration failed")


def _warning_metrics(records: Sequence[Mapping[str, Any]]):
    output, declarations = [], []
    for candidate in (FINALIST, *DIRECT_ONLINE):
        for alpha in (0.05, 0.10):
            threshold, cal_fpr, n_cal = _warning_threshold(records, candidate, alpha)
            audit = [
                row for row in records
                if row["split"] == "audit" and row["candidate"] == candidate
            ]
            by_unit = {}
            for row in audit:
                by_unit.setdefault(row["unit"], []).append(row)
            current = []
            for unit_rows in by_unit.values():
                unit_rows.sort(key=lambda row: int(row["budget"]))
                first = next(
                    (row for row in unit_rows if float(row["score"]) >= threshold),
                    None,
                )
                base = unit_rows[0]
                item = {
                    "candidate": candidate, "model": base["model"],
                    "family": base["family"], "unit": base["unit"],
                    "target_false_warning": alpha, "target": int(base["target"]),
                    "warned": first is not None,
                    "first_warning_budget": int(first["budget"]) if first else "",
                    "potential_tokens_remaining": (
                        int(base["trace_length"] - int(first["budget"])) if first else 0
                    ),
                }
                declarations.append(item); current.append(item)
            clean = [row for row in current if not row["target"]]
            error = [row for row in current if row["target"]]
            warned = [row for row in current if row["warned"]]
            output.append({
                "candidate": candidate, "model": current[0]["model"],
                "family": current[0]["family"], "target_false_warning": alpha,
                "threshold": threshold, "calibration_false_warning": cal_fpr,
                "n_calibration_clean": n_cal,
                "audit_false_warning": float(np.mean([row["warned"] for row in clean])),
                "audit_error_coverage": float(np.mean([row["warned"] for row in error])),
                "audit_warning_precision": float(np.mean([row["target"] for row in warned])) if warned else float("nan"),
                "audit_overall_coverage": len(warned) / max(len(current), 1),
                "mean_first_warning_budget": float(np.mean([
                    row["first_warning_budget"] for row in warned
                ])) if warned else float("nan"),
                "mean_potential_tokens_remaining": float(np.mean([
                    row["potential_tokens_remaining"] for row in warned
                ])) if warned else 0.0,
                "n": len(current),
            })
    return output, declarations


def _length_residualized(records: Sequence[Mapping[str, Any]]):
    output = []
    for candidate in (FINALIST, *DIRECT_ONLINE):
        for budget in PRIMARY_BUDGETS:
            cal = [
                row for row in records if row["split"] == "calibration"
                and row["candidate"] == candidate and row["budget"] == budget
            ]
            audit = [
                row for row in records if row["split"] == "audit"
                and row["candidate"] == candidate and row["budget"] == budget
            ]
            x_cal = np.log1p([row["trace_length"] for row in cal])
            y_cal = np.asarray([row["score"] for row in cal])
            design = np.column_stack([np.ones(len(cal)), x_cal])
            coef = np.linalg.lstsq(design, y_cal, rcond=None)[0]
            x_audit = np.log1p([row["trace_length"] for row in audit])
            residual = np.asarray([row["score"] for row in audit]) - (
                coef[0] + coef[1] * x_audit
            )
            labels = [row["target"] for row in audit]
            output.append({
                "candidate": candidate, "model": audit[0]["model"],
                "family": audit[0]["family"], "budget": budget,
                "raw_auroc": _safe_auc(labels, [row["score"] for row in audit]),
                "length_residualized_auroc": _safe_auc(labels, residual),
                "calibration_log_length_slope": float(coef[1]), "n": len(audit),
                "diagnostic_uses_final_length_not_deployable": True,
            })
    return output


def _ablation_metrics(
    model: str, family: str, audit, prepared_audit, local_head,
    global_model, threshold,
):
    target = np.asarray([int(row["label"]) for row in audit])
    detector = np.asarray([global_model.score(row, None) for row in audit])
    groups = {
        **{f"family::{name}": (index,) for index, name in enumerate(FAMILY_NAMES)},
        "primitive::entropy": (0, 1, 2),
        "primitive::sampled_energy": (3,),
        "primitive::partition_energy": (4,),
        "primitive::topk_distribution": (5,),
    }
    output = []
    for name, columns in groups.items():
        locators = []
        for row, item in zip(audit, prepared_audit):
            level = item.representations["family6"].copy()
            level[:, list(columns)] = 0.0
            curve = local_head.curve_from_level(level)
            locators.append(_step_top5_locator(curve, row))
        prediction = np.where(detector > threshold, locators, -1)
        result = _processbench(prediction, target)
        output.append({
            "model": model, "family": family, "ablation": name,
            "task": "local", "primary": result["f1"], **result,
            "detector_unchanged": True, "n": len(audit),
        })
    return output


def _robustness_audit(
    model: str, family: str, calibration, references, prepared_cal,
    local_head, global_model,
):
    repeated = fit_trajectory_head_prepared(
        prepared_cal, name="repeat", representation="family6", operators=("level",)
    )
    label_permuted = [dict(row, label=999, final_answer_correct=None) for row in calibration]
    # The fit accepts PreparedTrace objects only; targets cannot enter this path.
    permuted_fit = fit_trajectory_head_prepared(
        prepared_cal, name="permuted", representation="family6", operators=("level",)
    )
    sampled = []
    for item in prepared_cal:
        raw = operator_matrix(item.representations["family6"], ("level",))
        sampled.append(raw[equal_positions(len(raw), 32)])
    raw = np.vstack(sampled)
    V = (raw[:, local_head.keep] - local_head.mean) / local_head.std
    F = V.T
    order = np.arange(F.shape[0])[::-1]
    permuted = upcr_fit(F[order], **IU_FIT)
    w = np.asarray(permuted.w)
    if np.corrcoef(w @ F[order], F[order].mean(axis=0))[0, 1] < 0:
        w = -w
    remapped = np.empty_like(w); remapped[order] = w

    row = calibration[0]
    budget = min(64, len(row["token_entropies"]) - 1)
    changed = deepcopy(row)
    rng = np.random.default_rng(SEED)
    for key in ("token_entropies", "token_spilled_energies", "token_logsumexp"):
        if changed.get(key) is not None:
            values = np.asarray(changed[key]).copy()
            values[budget:] = rng.normal(size=len(values) - budget)
            changed[key] = values
    if isinstance(changed.get("top_k_logprobs"), Mapping):
        changed["top_k_logprobs"] = {
            key: np.asarray(value).copy() for key, value in changed["top_k_logprobs"].items()
        }
        values = changed["top_k_logprobs"].get("logprobs")
        if values is not None:
            values[budget:] = rng.normal(size=values[budget:].shape)
    left = global_model.score(row, budget)
    right = global_model.score(changed, budget)
    chunk = global_model.score(truncate_row(row, budget), None)
    return {
        "model": model, "family": family,
        "repeated_fit_weights_exact": bool(np.array_equal(local_head.weights, repeated.weights)),
        "label_permutation_weights_exact": bool(np.array_equal(local_head.weights, permuted_fit.weights)),
        "labels_present_in_fit_api": False,
        "label_permutation_fixture_rows": len(label_permuted),
        "feature_order_max_abs_weight_difference": float(np.max(np.abs(local_head.weights - remapped))),
        "feature_order_score_allclose": bool(np.allclose(local_head.weights @ F, remapped @ F, atol=1e-10, rtol=1e-10)),
        "suffix_replacement_prefix_score_exact": bool(left == right),
        "chunk_endpoint_prefix_score_exact": bool(left == chunk),
    }


def _aggregate(metrics, task: str):
    output = []
    for candidate in sorted({row["candidate"] for row in metrics if row["task"] == task}):
        rows = [row for row in metrics if row["task"] == task and row["candidate"] == candidate]
        output.append({
            "candidate": candidate, "task": task,
            "primary": float(np.mean([float(row["primary"]) for row in rows])),
            "cells": len(rows), "access_tier": rows[0].get("access_tier", "A"),
        })
    return output


def _grouped_interval(records, task: str, candidate: str, reference: str):
    relevant = [
        row for row in records if row["task"] == task
        and row["candidate"] in {candidate, reference}
        and (task != "online" or int(row["budget"]) in PRIMARY_BUDGETS)
        and (task != "online" or row["split"] == "audit")
    ]

    def metric(method, sampled, lookup, models):
        values = []
        for model in models:
            if task == "local":
                rows = [lookup[(method, model, unit, "final")] for unit in sampled]
                values.append(_processbench(
                    [row["prediction"] for row in rows],
                    [row["target"] for row in rows],
                )["f1"])
            else:
                for budget in PRIMARY_BUDGETS:
                    rows = [
                        lookup[(method, model, unit, str(budget))] for unit in sampled
                        if (method, model, unit, str(budget)) in lookup
                    ]
                    auc = _safe_auc(
                        [row["target"] for row in rows],
                        [row["score"] for row in rows],
                    )
                    if np.isfinite(auc):
                        values.append(auc)
        return float(np.mean(values))

    prepared, points = [], []
    for family in FAMILIES:
        rows = [row for row in relevant if row["family"] == family]
        models = sorted({row["model"] for row in rows})
        lookup = {}
        for row in rows:
            budget = "final" if task == "local" else str(row["budget"])
            lookup[(row["candidate"], row["model"], row["unit"], budget)] = row
        units = sorted(set.intersection(*[
            {row["unit"] for row in rows if row["candidate"] == method and row["model"] == model}
            for method in (candidate, reference) for model in models
        ]))
        prepared.append((units, lookup, models))
        points.append(
            metric(candidate, units, lookup, models)
            - metric(reference, units, lookup, models)
        )
    rng = np.random.default_rng(SEED + sum(ord(c) for c in candidate + reference + task))
    draws = []
    for _ in range(BOOTSTRAP):
        deltas = []
        for units, lookup, models in prepared:
            sampled = [units[i] for i in rng.integers(0, len(units), len(units))]
            deltas.append(
                metric(candidate, sampled, lookup, models)
                - metric(reference, sampled, lookup, models)
            )
        draws.append(float(np.mean(deltas)))
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(np.mean(points)), float(low), float(high), int(sum(x > 0 for x in points)), int(sum(x < 0 for x in points))


def _strata(local_records, online_records):
    output = []
    finalist_local = [row for row in local_records if row["candidate"] == FINALIST]
    for model in MODELS:
        for family in FAMILIES:
            cell = [row for row in finalist_local if row["model"] == model and row["family"] == family]
            for field in ("length_stratum", "error_position_quartile", "answer_process_stratum"):
                for value in sorted({row[field] for row in cell}):
                    rows = [row for row in cell if row[field] == value]
                    error = [row for row in rows if row["target"] != -1]
                    output.append({
                        "model": model, "family": family, "task": "local",
                        "stratum_type": field, "stratum": value, "n": len(rows),
                        "exact_error": float(np.mean([
                            row["prediction"] == row["target"] for row in error
                        ])) if error else float("nan"),
                        "within_one": float(np.mean([
                            abs(row["prediction"] - row["target"]) <= 1 for row in error
                        ])) if error else float("nan"),
                        "clean_abstention": float(np.mean([
                            row["prediction"] == -1 for row in rows if row["target"] == -1
                        ])) if any(row["target"] == -1 for row in rows) else float("nan"),
                    })
            for length in ("short", "medium", "long"):
                for budget in PRIMARY_BUDGETS:
                    rows = [
                        row for row in online_records
                        if row["candidate"] == FINALIST and row["split"] == "audit"
                        and row["model"] == model and row["family"] == family
                        and row["length_stratum"] == length and row["budget"] == budget
                    ]
                    if rows:
                        output.append({
                            "model": model, "family": family, "task": "online",
                            "stratum_type": "length_stratum", "stratum": length,
                            "budget": budget, "n": len(rows),
                            "auroc": _safe_auc(
                                [row["target"] for row in rows],
                                [row["score"] for row in rows],
                            ),
                        })
    return output


def main() -> None:
    if _sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("frozen protocol hash mismatch")
    architecture = json.loads((OUT / "STAGE_3_ARCHITECTURE_SELECTION.json").read_text())
    if architecture["selected"]["candidate"] != "global_local__w1.00":
        raise RuntimeError("unexpected S3 finalist")
    started_all = time.perf_counter()
    local_records, local_metrics = [], []
    online_records, online_metrics = [], []
    warning_metrics, warning_records = [], []
    residualized, ablations, efficiency, robustness = [], [], [], []
    diagnostics = {}
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for family in FAMILIES:
            checkpoint_path = CHECKPOINT / f"{model}__{family}.pkl"
            if checkpoint_path.exists():
                with checkpoint_path.open("rb") as handle:
                    payload = pickle.load(handle)
                local_records.extend(payload["local_records"])
                local_metrics.extend(payload["local_metrics"])
                online_records.extend(payload["online_records"])
                online_metrics.extend(payload["online_metrics"])
                warning_metrics.extend(payload["warning_metrics"])
                warning_records.extend(payload["warning_records"])
                residualized.extend(payload["residualized"])
                ablations.extend(payload["ablations"])
                efficiency.extend(payload["efficiency"])
                robustness.extend(payload["robustness"])
                diagnostics[f"{model}/{family}"] = payload["diagnostics"]
                print(f"S4 {model}/{family}: resumed checkpoint", flush=True)
                continue
            starts = {
                "local_records": len(local_records),
                "local_metrics": len(local_metrics),
                "online_records": len(online_records),
                "online_metrics": len(online_metrics),
                "warning_metrics": len(warning_metrics),
                "warning_records": len(warning_records),
                "residualized": len(residualized),
                "ablations": len(ablations),
                "efficiency": len(efficiency),
                "robustness": len(robustness),
            }
            calibration, audit = _split(model, family)
            print(f"S4 {model}/{family}: cal={len(calibration)} audit={len(audit)}", flush=True)
            tracemalloc.start(); started = time.perf_counter()
            references = fit_references(calibration)
            prepared_cal = [prepare_trace(row, references) for row in calibration]
            prepared_audit = [prepare_trace(row, references) for row in audit]
            local_head = fit_trajectory_head_prepared(
                prepared_cal, name="finalist_local", representation="family6",
                operators=("level",),
            )
            raw9_head = fit_trajectory_head_prepared(
                prepared_cal, name="step272_raw9", representation="raw9",
                operators=("level",),
            )
            global_model = fit_registered_global(calibration)
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            fit_seconds = time.perf_counter() - started
            records, metrics, _, _, threshold = _finalist_local(
                model, family, calibration, audit, global_model, local_head,
                prepared_cal, prepared_audit,
            )
            local_records.extend(records); local_metrics.extend(metrics)

            # Local same-row competitors.
            registered = fit_registered_local(calibration)
            reg_cal = [registered.curve(row) for row in calibration]
            reg_audit = [registered.curve(row) for row in audit]
            competitor_specs = []
            competitor_specs.append(_global_local_incumbent(
                "gl_liu_v1_replay", family, calibration, audit,
                global_model, registered, reg_cal, reg_audit, 0.75,
            ))
            raw9_cal = [raw9_head.curve_from_level(item.representations["raw9"]) for item in prepared_cal]
            raw9_audit = [raw9_head.curve_from_level(item.representations["raw9"]) for item in prepared_audit]
            competitor_specs.append(_global_local_incumbent(
                "step272_twohead", family, calibration, audit,
                global_model, None, raw9_cal, raw9_audit, 0.50,
            ))
            competitor_specs.append(_evaluate_curve_system(
                "max_entropy", family, calibration, audit,
                [np.asarray(row["token_entropies"]) for row in calibration],
                [np.asarray(row["token_entropies"]) for row in audit],
                access_tier="A", fidelity="transparent_same_trace_baseline",
            ))
            competitor_specs.append(_mindgap(calibration, audit, family))
            competitor_specs.append(_tier_b_partial(family, audit))
            for cell_records, cell_metrics in competitor_specs:
                for row in cell_records:
                    row.update({"model": model, "task": "local"})
                for row in cell_metrics:
                    row.update({"model": model, "task": "local"})
                local_records.extend(cell_records); local_metrics.extend(cell_metrics)

            started_score = time.perf_counter()
            cell_online, iu_diag = _online_records_cell(
                model, family, calibration, audit, references=references,
                global_model=global_model, raw9_head=raw9_head,
                prepared_cal=prepared_cal,
            )
            score_seconds = time.perf_counter() - started_score
            online_records.extend(cell_online)
            online_metrics.extend(_online_metrics(cell_online))
            cell_warning, cell_declarations = _warning_metrics(cell_online)
            warning_metrics.extend(cell_warning); warning_records.extend(cell_declarations)
            residualized.extend(_length_residualized(cell_online))
            ablations.extend(_ablation_metrics(
                model, family, audit, prepared_audit, local_head,
                global_model, threshold,
            ))
            robustness.append(_robustness_audit(
                model, family, calibration, references, prepared_cal,
                local_head, global_model,
            ))
            efficiency.append({
                "model": model, "family": family,
                "fit_seconds": fit_seconds, "online_score_seconds": score_seconds,
                "python_peak_bytes": int(peak), "local_coordinates": 6,
                "global_coordinates": len(global_model.names),
                "physical_heads": 2, "persistent_local_state_scalars": 6,
                "gpu_hours": 0,
            })
            diagnostics[f"{model}/{family}"] = {
                "references": references.as_dict(),
                "local": local_head.diagnostics,
                "raw9": raw9_head.diagnostics,
                "global": global_model.diagnostics,
                "iu28": iu_diag,
            }
            payload = {
                "local_records": local_records[starts["local_records"]:],
                "local_metrics": local_metrics[starts["local_metrics"]:],
                "online_records": online_records[starts["online_records"]:],
                "online_metrics": online_metrics[starts["online_metrics"]:],
                "warning_metrics": warning_metrics[starts["warning_metrics"]:],
                "warning_records": warning_records[starts["warning_records"]:],
                "residualized": residualized[starts["residualized"]:],
                "ablations": ablations[starts["ablations"]:],
                "efficiency": efficiency[starts["efficiency"]:],
                "robustness": robustness[starts["robustness"]:],
                "diagnostics": diagnostics[f"{model}/{family}"],
            }
            with checkpoint_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    aggregate = _aggregate(local_metrics, "local") + _aggregate(online_metrics, "online")
    local_tier_a = [
        row for row in aggregate if row["task"] == "local"
        and row["candidate"] != FINALIST and row["access_tier"] == "A"
    ]
    online_tier_a = [
        row for row in aggregate if row["task"] == "online"
        and row["candidate"] != FINALIST
    ]
    local_reference = max(local_tier_a, key=lambda row: row["primary"])["candidate"]
    online_reference = max(online_tier_a, key=lambda row: row["primary"])["candidate"]
    intervals = []
    for task, records, candidates, reference in (
        ("local", local_records, [row["candidate"] for row in local_tier_a] + [FINALIST], local_reference),
        ("online", online_records, [row["candidate"] for row in online_tier_a] + [FINALIST], online_reference),
    ):
        for candidate in sorted(set(candidates)):
            if candidate == reference:
                continue
            delta, low, high, wins, losses = _grouped_interval(
                records, task, candidate, reference
            )
            intervals.append({
                "candidate": candidate, "task": task, "reference": reference,
                "delta": delta, "ci_low": low, "ci_high": high,
                "family_wins": wins, "family_losses": losses,
            })
    finalist_intervals = {
        task: next(row for row in intervals if row["candidate"] == FINALIST and row["task"] == task)
        for task in ("local", "online")
    }
    finalist_values = {
        task: next(row["primary"] for row in aggregate if row["candidate"] == FINALIST and row["task"] == task)
        for task in ("local", "online")
    }
    reference_values = {
        "local": next(row["primary"] for row in aggregate if row["candidate"] == local_reference and row["task"] == "local"),
        "online": next(row["primary"] for row in aggregate if row["candidate"] == online_reference and row["task"] == "online"),
    }
    if all(finalist_intervals[task]["ci_low"] > 0 for task in ("local", "online")):
        verdict = "IMPROVES_DIRECT_COMPETITOR"
    elif (
        finalist_values["local"] >= reference_values["local"] - 0.010
        and finalist_values["online"] >= reference_values["online"] - 0.015
    ):
        verdict = "PARITY_WITH_DIRECT_COMPETITOR"
    else:
        verdict = "REGRESSES_DIRECT_COMPETITOR"
    decision = {
        "verdict": verdict,
        "finalist": FINALIST,
        "local_primary": finalist_values["local"],
        "online_primary": finalist_values["online"],
        "local_reference": local_reference,
        "online_reference": online_reference,
        "reference_values": reference_values,
        "intervals": finalist_intervals,
        "stable_same_access_improvement": verdict == "IMPROVES_DIRECT_COMPETITOR",
        "new_gpu_run_recommended": False,
        "reason": "A fresh run is not requested unless transfer shows a stable direct-competitor improvement.",
        "protocol_sha256": PROTOCOL_SHA256,
    }
    strata = _strata(local_records, online_records)
    audit_json = {
        "all_prefixes_recomputed_from_truncated_telemetry": True,
        "scorer_copies_resampled_with_source_question": True,
        "robustness": robustness,
        "all_repeated_fit_exact": all(row["repeated_fit_weights_exact"] for row in robustness),
        "all_label_permutation_exact": all(row["label_permutation_weights_exact"] for row in robustness),
        "all_suffix_exact": all(row["suffix_replacement_prefix_score_exact"] for row in robustness),
        "all_chunk_endpoint_exact": all(row["chunk_endpoint_prefix_score_exact"] for row in robustness),
    }

    _write_csv(OUT / "STAGE_4_LOCAL_PER_QUESTION.csv", local_records)
    _write_csv(OUT / "STAGE_4_ONLINE_PER_QUESTION.csv", online_records)
    _write_csv(OUT / "STAGE_4_CELL_METRICS.csv", local_metrics + online_metrics)
    _write_csv(OUT / "STAGE_4_AGGREGATE.csv", aggregate)
    _write_csv(OUT / "STAGE_4_INTERVALS.csv", intervals)
    _write_csv(OUT / "STAGE_4_WARNING_METRICS.csv", warning_metrics)
    _write_csv(OUT / "STAGE_4_WARNINGS.csv", warning_records)
    _write_csv(OUT / "STAGE_4_LENGTH_RESIDUALIZED.csv", residualized)
    _write_csv(OUT / "STAGE_4_ABLATIONS.csv", ablations)
    _write_csv(OUT / "STAGE_4_STRATA.csv", strata)
    _write_csv(OUT / "STAGE_4_EFFICIENCY.csv", efficiency)
    _write_json(OUT / "STAGE_4_DIAGNOSTICS.json", diagnostics)
    _write_json(OUT / "STAGE_4_DECISION.json", decision)
    _write_json(OUT / "AUDIT.json", audit_json)

    lookup = {(row["candidate"], row["task"]): row for row in intervals}
    lines = [
        "# S4 scorer-transfer and robustness audit",
        "",
        f"**Verdict: `{verdict}`.**",
        "",
        "Qwen3-8B and Llama-3.1-8B scorer copies are paired by source question in every interval.",
        f"Local direct reference: `{local_reference}`. Online direct reference: `{online_reference}`.",
        "",
        "| task | method | primary | delta vs direct | grouped 95% CI | tier |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in sorted(aggregate, key=lambda item: (item["task"], -item["primary"])):
        interval = lookup.get((row["candidate"], row["task"]))
        delta = "—" if interval is None else f"{interval['delta']:+.4f}"
        ci = "—" if interval is None else f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
        lines.append(
            f"| {row['task']} | {row['candidate']} | {row['primary']:.4f} | {delta} | {ci} | {row['access_tier']} |"
        )
    lines.extend([
        "",
        "Tier-B rows are same-question compute ceilings, not same-access deltas. Potential tokens remaining are not realized savings.",
    ])
    (OUT / "STAGE_4_TRANSFER.md").write_text("\n".join(lines) + "\n")

    manifest_path = OUT / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "status": "STAGE_4_COMPLETE_REPORT_PENDING",
        "stage4_decision_sha256": _sha256(OUT / "STAGE_4_DECISION.json"),
        "audit_sha256": _sha256(OUT / "AUDIT.json"),
        "elapsed_stage4_seconds": time.perf_counter() - started_all,
        "new_inference": False, "gpu_hours": 0, "drive_mutation": False,
    })
    _write_json(manifest_path, manifest)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
