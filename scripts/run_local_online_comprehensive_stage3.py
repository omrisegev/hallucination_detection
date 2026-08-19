#!/usr/bin/env python3
"""Run S3 same-matrix fusion and joint Local/Online architecture selection."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import time
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
    PROTOCOL,
    PROTOCOL_SHA256,
    _bootstrap_interval,
    _evaluate_curve_system,
    _global_local_incumbent,
    _mindgap,
    _stage_partition,
    _step_top5_locator,
    _tier_b,
    _sha256,
)
from scripts.run_local_online_comprehensive_stage2 import (  # noqa: E402
    DIRECT_METHODS,
    _direct_scores,
    _paired_interval,
)
from spectral_utils.comprehensive_fusion import fit_fusion_panel  # noqa: E402
from spectral_utils.local_online_comprehensive import (  # noqa: E402
    fit_references,
    fit_trajectory_head_prepared,
    prepare_trace,
)
from spectral_utils.multitask_trajectory import truncate_row  # noqa: E402
from spectral_utils.online_convergence import fit_frozen_prefix_iu  # noqa: E402


CELLS = (("qwen3_4b", "olympiadbench"), ("qwen3_4b", "omnimath"))
BUDGETS = (64, 128)


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


def _load_cell(model_name: str, family: str):
    rows = load_rows(_cell_path(model_name, family))
    for row in rows:
        row["_stage"] = _stage_partition(family, row["_unit"])
    return (
        [row for row in rows if row["_stage"] == "calibration"],
        [row for row in rows if row["_stage"] == "architecture"],
    )


def _fusion_screen_cell(model_name: str, family: str):
    calibration, evaluation = _load_cell(model_name, family)
    references = fit_references(calibration)
    prepared_cal = [prepare_trace(row, references) for row in calibration]
    prepared_eval = [prepare_trace(row, references) for row in evaluation]
    local_panel = fit_fusion_panel(
        prepared_cal,
        representation="family6",
        operators=("level",),
        include_temporal=True,
    )
    online_panel = fit_fusion_panel(
        prepared_cal,
        representation="family6",
        operators=("fast", "slow"),
        include_temporal=False,
    )
    records, metrics = [], []
    cal_local_target = np.asarray([int(row["label"]) for row in calibration])
    eval_local_target = np.asarray([int(row["label"]) for row in evaluation])
    for variant in local_panel.weights:
        cal_curves = [local_panel.curve(item, variant) for item in prepared_cal]
        eval_curves = [local_panel.curve(item, variant) for item in prepared_eval]
        cal_detector = np.asarray([float(np.max(curve)) for curve in cal_curves])
        eval_detector = np.asarray([float(np.max(curve)) for curve in eval_curves])
        cal_locator = np.asarray([
            _step_top5_locator(curve, row)
            for curve, row in zip(cal_curves, calibration)
        ])
        eval_locator = np.asarray([
            _step_top5_locator(curve, row)
            for curve, row in zip(eval_curves, evaluation)
        ])
        threshold, calibration_f1 = _best_threshold(
            cal_detector, cal_locator, cal_local_target
        )
        prediction = np.where(eval_detector > threshold, eval_locator, -1)
        result = _processbench(prediction, eval_local_target)
        candidate = f"local_fusion__{variant}"
        metrics.append({
            "candidate": candidate, "variant": variant, "task": "local",
            "family": family, "primary": result["f1"], **result,
            "calibration_f1": calibration_f1, "threshold": threshold,
            "n": len(evaluation),
        })
        records.extend({
            "candidate": candidate, "family": family, "unit": row["_unit"],
            "target": int(target), "score": float(score),
            "locator": int(locator), "prediction": int(prediction_value),
            "task": "local",
        } for row, target, score, locator, prediction_value in zip(
            evaluation, eval_local_target, eval_detector, eval_locator, prediction
        ))

    prefix_prepared = {budget: [] for budget in BUDGETS}
    prefix_rows = {budget: [] for budget in BUDGETS}
    for budget in BUDGETS:
        for row in evaluation:
            if len(row["token_entropies"]) <= budget:
                continue
            prefix_rows[budget].append(row)
            prefix_prepared[budget].append(
                prepare_trace(truncate_row(row, budget), references)
            )
    for variant in online_panel.weights:
        candidate = f"online_fusion__{variant}"
        values = []
        for budget in BUDGETS:
            rows = prefix_rows[budget]
            scores = np.asarray([
                float(np.max(online_panel.curve(item, variant)))
                for item in prefix_prepared[budget]
            ])
            labels = np.asarray([
                int(not bool(row["final_answer_correct"])) for row in rows
            ])
            auc = _safe_auc(labels, scores)
            values.append(auc)
            metrics.append({
                "candidate": candidate, "variant": variant, "task": "online_budget",
                "family": family, "budget": budget, "primary": auc,
                "auroc": auc, "auprc": _safe_ap(labels, scores), "n": len(rows),
            })
            records.extend({
                "candidate": candidate, "family": family, "unit": row["_unit"],
                "target": int(target), "score": float(score), "task": "online",
                "budget": budget, "split": "development", "is_final": False,
            } for row, target, score in zip(rows, labels, scores))
        metrics.append({
            "candidate": candidate, "variant": variant, "task": "online",
            "family": family, "primary": float(np.mean(values)), "n": 0,
        })
    diagnostics = {
        "local": dict(local_panel.diagnostics),
        "online": dict(online_panel.diagnostics),
        "calibration": len(calibration),
        "architecture": len(evaluation),
    }
    return records, metrics, diagnostics


def _select_fusion(
    records: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    task: str,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    task_metrics = [row for row in metrics if row["task"] == task]
    families = {row["family"] for row in task_metrics}
    aggregate = []
    for candidate in sorted({row["candidate"] for row in task_metrics}):
        rows = [row for row in task_metrics if row["candidate"] == candidate]
        if {row["family"] for row in rows} != families:
            continue
        aggregate.append({
            "task": task,
            "candidate": candidate,
            "variant": rows[0]["variant"],
            "primary": float(np.mean([float(row["primary"]) for row in rows])),
        })
    reference = f"{task}_fusion__ordinary"
    intervals = []
    for row in aggregate:
        if row["candidate"] == reference:
            continue
        if task == "local":
            delta, low, high, wins, losses = _bootstrap_interval(
                [item for item in records if item["task"] == "local"],
                row["candidate"], reference
            )
        else:
            delta, low, high, wins, losses = _paired_interval(
                [item for item in records if item["task"] == "online"],
                row["candidate"], reference
            )
        intervals.append({
            "task": task, "candidate": row["candidate"], "reference": reference,
            "delta": delta, "ci_low": low, "ci_high": high,
            "family_wins": wins, "family_losses": losses,
        })
    ordinary = next(row for row in aggregate if row["candidate"] == reference)
    improving = [
        row for row in aggregate
        if row["candidate"] != reference
        and next(item for item in intervals if item["candidate"] == row["candidate"])["ci_low"] > 0
    ]
    selected = max(improving, key=lambda row: row["primary"]) if improving else ordinary
    return str(selected["variant"]), aggregate, intervals


def _architecture_configs() -> list[dict[str, Any]]:
    output = [{
        "architecture": "shared_local", "local_weights": (0.0, 1.0, 0.0),
        "online_weights": (0.0, 1.0, 0.0),
    }, {
        "architecture": "independent_heads", "local_weights": (0.0, 1.0, 0.0),
        "online_weights": (0.0, 0.0, 1.0),
    }]
    for weight in (0.0, 0.25, 0.50, 0.75, 1.0):
        output.append({
            "architecture": f"global_local__w{weight:.2f}",
            "local_weights": (weight, 1.0 - weight, 0.0),
            "online_weights": (weight, 1.0 - weight, 0.0),
        })
        output.append({
            "architecture": f"global_online__w{weight:.2f}",
            "local_weights": (0.0, 1.0, 0.0),
            "online_weights": (weight, 0.0, 1.0 - weight),
        })
    for g in (0.0, 0.25, 0.50, 0.75, 1.0):
        for l in (0.0, 0.25, 0.50, 0.75, 1.0):
            o = 1.0 - g - l
            if o < -1e-12 or abs(o * 4 - round(o * 4)) > 1e-12:
                continue
            output.append({
                "architecture": f"three_signal__g{g:.2f}__l{l:.2f}__o{o:.2f}",
                "local_weights": (g, l, o), "online_weights": (g, l, o),
            })
    unique = {}
    for item in output:
        unique[item["architecture"]] = item
    return list(unique.values())


def _weighted(signals: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    return sum(float(weight) * np.asarray(signal) for weight, signal in zip(weights, signals))


def _architecture_cell(
    model_name: str,
    family: str,
    local_variant: str,
    online_variant: str,
    configs: Sequence[Mapping[str, Any]],
):
    calibration, evaluation = _load_cell(model_name, family)
    references = fit_references(calibration)
    prepared_cal = [prepare_trace(row, references) for row in calibration]
    prepared_eval = [prepare_trace(row, references) for row in evaluation]
    local_panel = fit_fusion_panel(
        prepared_cal, representation="family6", operators=("level",),
        include_temporal=True,
    )
    online_panel = fit_fusion_panel(
        prepared_cal, representation="family6", operators=("fast", "slow"),
        include_temporal=False,
    )
    global_model = fit_registered_global(calibration)
    local_cal_curves = [local_panel.curve(item, local_variant) for item in prepared_cal]
    local_eval_curves = [local_panel.curve(item, local_variant) for item in prepared_eval]
    online_cal_curves = [online_panel.curve(item, online_variant) for item in prepared_cal]
    online_eval_curves = [online_panel.curve(item, online_variant) for item in prepared_eval]
    cal_g = np.asarray([global_model.score(row, None) for row in calibration])
    eval_g = np.asarray([global_model.score(row, None) for row in evaluation])
    cal_l = np.asarray([float(np.max(curve)) for curve in local_cal_curves])
    eval_l = np.asarray([float(np.max(curve)) for curve in local_eval_curves])
    cal_o = np.asarray([float(np.max(curve)) for curve in online_cal_curves])
    eval_o = np.asarray([float(np.max(curve)) for curve in online_eval_curves])
    fits = (_zfit(cal_g), _zfit(cal_l), _zfit(cal_o))
    cal_signals = tuple(_zapply(values, fit) for values, fit in zip((cal_g, cal_l, cal_o), fits))
    eval_signals = tuple(_zapply(values, fit) for values, fit in zip((eval_g, eval_l, eval_o), fits))
    cal_local_target = np.asarray([int(row["label"]) for row in calibration])
    eval_local_target = np.asarray([int(row["label"]) for row in evaluation])
    cal_locator = np.asarray([
        _step_top5_locator(curve, row)
        for curve, row in zip(local_cal_curves, calibration)
    ])
    eval_locator = np.asarray([
        _step_top5_locator(curve, row)
        for curve, row in zip(local_eval_curves, evaluation)
    ])

    prefix = {}
    for budget in BUDGETS:
        indexes, g_values, l_values, o_values, labels = [], [], [], [], []
        for index, row in enumerate(evaluation):
            if len(row["token_entropies"]) <= budget:
                continue
            item = prepare_trace(truncate_row(row, budget), references)
            indexes.append(index)
            labels.append(int(not bool(row["final_answer_correct"])))
            g_values.append(global_model.score(row, budget))
            l_values.append(float(np.max(local_panel.curve(item, local_variant))))
            o_values.append(float(np.max(online_panel.curve(item, online_variant))))
        prefix[budget] = {
            "indexes": np.asarray(indexes, dtype=int),
            "labels": np.asarray(labels, dtype=int),
            "signals": (
                _zapply(g_values, fits[0]), _zapply(l_values, fits[1]),
                _zapply(o_values, fits[2]),
            ),
        }

    records, metrics = [], []
    for config in configs:
        candidate = str(config["architecture"])
        local_weights = tuple(config["local_weights"])
        online_weights = tuple(config["online_weights"])
        cal_detector = _weighted(cal_signals, local_weights)
        eval_detector = _weighted(eval_signals, local_weights)
        threshold, calibration_f1 = _best_threshold(
            cal_detector, cal_locator, cal_local_target
        )
        prediction = np.where(eval_detector > threshold, eval_locator, -1)
        local_result = _processbench(prediction, eval_local_target)
        metrics.append({
            "candidate": candidate, "family": family, "task": "local",
            "primary": local_result["f1"], **local_result,
            "threshold": threshold, "calibration_f1": calibration_f1,
            "local_weights": str(local_weights),
            "online_weights": str(online_weights), "n": len(evaluation),
        })
        records.extend({
            "candidate": candidate, "family": family, "unit": row["_unit"],
            "task": "local", "target": int(target), "score": float(score),
            "locator": int(locator), "prediction": int(pred),
        } for row, target, score, locator, pred in zip(
            evaluation, eval_local_target, eval_detector, eval_locator, prediction
        ))
        online_values = []
        for budget in BUDGETS:
            value = prefix[budget]
            scores = _weighted(value["signals"], online_weights)
            auc = _safe_auc(value["labels"], scores)
            online_values.append(auc)
            metrics.append({
                "candidate": candidate, "family": family,
                "task": "online_budget", "budget": budget,
                "primary": auc, "auroc": auc,
                "auprc": _safe_ap(value["labels"], scores),
                "local_weights": str(local_weights),
                "online_weights": str(online_weights), "n": len(scores),
            })
            records.extend({
                "candidate": candidate, "family": family,
                "unit": evaluation[index]["_unit"], "task": "online",
                "budget": budget, "target": int(target), "score": float(score),
                "split": "development", "is_final": False,
            } for index, target, score in zip(
                value["indexes"], value["labels"], scores
            ))
        metrics.append({
            "candidate": candidate, "family": family, "task": "online",
            "primary": float(np.mean(online_values)),
            "local_weights": str(local_weights),
            "online_weights": str(online_weights), "n": 0,
        })

    # Same-row direct Local competitors.
    registered_local = fit_registered_local(calibration)
    reg_cal = [registered_local.curve(row) for row in calibration]
    reg_eval = [registered_local.curve(row) for row in evaluation]
    direct_records, direct_metrics = _global_local_incumbent(
        "gl_liu_v1_replay", family, calibration, evaluation,
        global_model, registered_local, reg_cal, reg_eval, 0.75,
    )
    records.extend(direct_records); metrics.extend(direct_metrics)
    raw9_head = fit_trajectory_head_prepared(
        prepared_cal, name="step272_raw9_level", representation="raw9",
        operators=("level",),
    )
    raw9_cal = [raw9_head.curve_from_level(item.representations["raw9"]) for item in prepared_cal]
    raw9_eval = [raw9_head.curve_from_level(item.representations["raw9"]) for item in prepared_eval]
    direct_records, direct_metrics = _global_local_incumbent(
        "step272_twohead", family, calibration, evaluation,
        global_model, None, raw9_cal, raw9_eval, 0.50,
    )
    records.extend(direct_records); metrics.extend(direct_metrics)
    entropy_cal = [np.asarray(row["token_entropies"], dtype=float) for row in calibration]
    entropy_eval = [np.asarray(row["token_entropies"], dtype=float) for row in evaluation]
    direct_records, direct_metrics = _evaluate_curve_system(
        "max_entropy", family, calibration, evaluation, entropy_cal, entropy_eval,
        access_tier="A", fidelity="transparent_same_trace_baseline",
    )
    records.extend(direct_records); metrics.extend(direct_metrics)
    direct_records, direct_metrics = _mindgap(calibration, evaluation, family)
    records.extend(direct_records); metrics.extend(direct_metrics)
    tier_records, tier_metrics = _tier_b(family, evaluation)
    records.extend(tier_records); metrics.extend(tier_metrics)

    # Same-row direct Online competitors using exact causal prefix replay.
    iu28 = fit_frozen_prefix_iu(calibration, include_elapsed_length=False)
    global_fit, local_fit = _zfit(cal_g), _zfit([
        float(np.max(curve)) for curve in raw9_cal
    ])
    for budget in BUDGETS:
        direct_by_method = {name: [] for name in DIRECT_METHODS}
        direct_rows = []
        for row in evaluation:
            if len(row["token_entropies"]) <= budget:
                continue
            prefix_row = truncate_row(row, budget)
            item = prepare_trace(prefix_row, references)
            score = _direct_scores(
                prefix_row, budget, item, iu28=iu28, global_model=global_model,
                raw9_head=raw9_head, global_fit=global_fit, local_fit=local_fit,
            )
            direct_rows.append(row)
            for name in DIRECT_METHODS:
                direct_by_method[name].append(score[name])
        labels = np.asarray([
            int(not bool(row["final_answer_correct"])) for row in direct_rows
        ])
        for name, scores in direct_by_method.items():
            metrics.append({
                "candidate": name, "family": family, "task": "online_budget",
                "budget": budget, "primary": _safe_auc(labels, scores),
                "auroc": _safe_auc(labels, scores),
                "auprc": _safe_ap(labels, scores), "n": len(scores),
            })
            records.extend({
                "candidate": name, "family": family, "unit": row["_unit"],
                "task": "online", "budget": budget, "target": int(target),
                "score": float(score), "split": "development", "is_final": False,
            } for row, target, score in zip(direct_rows, labels, scores))
    for name in DIRECT_METHODS:
        values = [
            row["primary"] for row in metrics
            if row["candidate"] == name and row["family"] == family
            and row["task"] == "online_budget"
        ]
        metrics.append({
            "candidate": name, "family": family, "task": "online",
            "primary": float(np.mean(values)), "n": 0,
        })
    for row in records:
        row.setdefault("task", "local")
    return records, metrics, {
        "calibration": len(calibration), "architecture": len(evaluation),
        "local_panel": dict(local_panel.diagnostics),
        "online_panel": dict(online_panel.diagnostics),
        "global": global_model.diagnostics,
    }


def _aggregate_architecture(metrics: Sequence[Mapping[str, Any]], configs):
    config_names = {row["architecture"] for row in configs}
    output = []
    for candidate in sorted(config_names):
        item = {"candidate": candidate}
        example = next(
            row for row in metrics
            if row["candidate"] == candidate and row["task"] == "local"
        )
        item["local_weights"] = example["local_weights"]
        item["online_weights"] = example["online_weights"]
        for task in ("local", "online"):
            values = [
                float(row["primary"]) for row in metrics
                if row["candidate"] == candidate and row["task"] == task
            ]
            item[task] = float(np.mean(values))
        weights = (
            tuple(float(x) for x in example["local_weights"].strip("()").split(","))
            + tuple(float(x) for x in example["online_weights"].strip("()").split(","))
        )
        # A Local head is always required to emit the step locator, even when
        # its detector weight is zero.  Count physical heads, not just nonzero
        # detector coefficients.
        active = {1}
        active.update(
            index % 3 for index, value in enumerate(weights)
            if abs(value) > 1e-12
        )
        item["head_count"] = len(active)
        item["coordinate_count"] = sum(
            {0: 30, 1: 6, 2: 12}[index] for index in active
        )
        item["fusion_terms"] = sum(abs(value) > 1e-12 for value in weights)
        output.append(item)
    return output


def main() -> None:
    if _sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("frozen protocol hash mismatch")
    stage1 = json.loads((OUT / "STAGE_1_LOCAL_SELECTION.json").read_text())
    stage2 = json.loads((OUT / "STAGE_2_ONLINE_SELECTION.json").read_text())
    if stage1["selected"]["base_candidate"] != "l_family6__level":
        raise RuntimeError("unexpected S1 frozen identity")
    if stage2["selected"]["candidate"] != "o_family6__fast_slow":
        raise RuntimeError("unexpected S2 frozen identity")

    started = time.perf_counter()
    fusion_path = OUT / "STAGE_3_FUSION_SELECTION.json"
    if fusion_path.exists():
        fusion_selection = json.loads(fusion_path.read_text())
        if fusion_selection.get("protocol_sha256") != PROTOCOL_SHA256:
            raise RuntimeError("saved S3 fusion selection has a different protocol")
        local_variant = str(fusion_selection["local_variant"])
        online_variant = str(fusion_selection["online_variant"])
        print(
            f"S3 fusion resume Local={local_variant} Online={online_variant}",
            flush=True,
        )
    else:
        fusion_records, fusion_metrics, fusion_diagnostics = [], [], {}
        for model_name, family in CELLS:
            print(f"S3 fusion {model_name}/{family}", flush=True)
            records, metrics, diagnostics = _fusion_screen_cell(model_name, family)
            fusion_records.extend(records); fusion_metrics.extend(metrics)
            fusion_diagnostics[f"{model_name}/{family}"] = diagnostics
        local_variant, local_aggregate, local_intervals = _select_fusion(
            fusion_records, fusion_metrics, "local"
        )
        online_variant, online_aggregate, online_intervals = _select_fusion(
            fusion_records, fusion_metrics, "online"
        )
        fusion_selection = {
            "local_variant": local_variant, "online_variant": online_variant,
            "rule": "retain a non-ordinary fusion only when its grouped 95% interval versus ordinary is wholly positive",
            "local_aggregate": local_aggregate,
            "online_aggregate": online_aggregate,
            "protocol_sha256": PROTOCOL_SHA256,
        }
        _write_csv(OUT / "STAGE_3_FUSION_PER_QUESTION.csv", fusion_records)
        _write_csv(OUT / "STAGE_3_FUSION_METRICS.csv", fusion_metrics)
        _write_csv(OUT / "STAGE_3_FUSION_AGGREGATE.csv", local_aggregate + online_aggregate)
        _write_csv(OUT / "STAGE_3_FUSION_INTERVALS.csv", local_intervals + online_intervals)
        _write_json(OUT / "STAGE_3_FUSION_DIAGNOSTICS.json", fusion_diagnostics)
        _write_json(fusion_path, fusion_selection)
        print(f"  fusion selected Local={local_variant} Online={online_variant}", flush=True)

    configs = _architecture_configs()
    records, metrics, diagnostics = [], [], {}
    for model_name, family in CELLS:
        print(f"S3 architecture {model_name}/{family}", flush=True)
        cell_records, cell_metrics, cell_diagnostics = _architecture_cell(
            model_name, family, local_variant, online_variant, configs
        )
        records.extend(cell_records); metrics.extend(cell_metrics)
        diagnostics[f"{model_name}/{family}"] = cell_diagnostics
    aggregate = _aggregate_architecture(metrics, configs)
    config_names = {row["candidate"] for row in aggregate}
    local_direct = []
    for candidate in sorted({row["candidate"] for row in metrics} - config_names):
        rows = [
            row for row in metrics
            if row["candidate"] == candidate and row["task"] == "local"
        ]
        if rows:
            local_direct.append({
                "candidate": candidate,
                "primary": float(np.mean([float(row["primary"]) for row in rows])),
                "tier": rows[0].get("access_tier", "A"),
            })
    online_direct = []
    for candidate in DIRECT_METHODS:
        rows = [
            row for row in metrics
            if row["candidate"] == candidate and row["task"] == "online"
        ]
        online_direct.append({
            "candidate": candidate,
            "primary": float(np.mean([float(row["primary"]) for row in rows])),
            "tier": "A",
        })
    local_reference = max(
        (row for row in local_direct if row["tier"] == "A"),
        key=lambda row: row["primary"],
    )["candidate"]
    online_reference = max(online_direct, key=lambda row: row["primary"])["candidate"]

    intervals = []
    for row in aggregate:
        for task, reference in (("local", local_reference), ("online", online_reference)):
            if task == "local":
                delta, low, high, wins, losses = _bootstrap_interval(
                    [item for item in records if item["task"] == "local"],
                    row["candidate"], reference
                )
            else:
                delta, low, high, wins, losses = _paired_interval(
                    [item for item in records if item["task"] == "online"],
                    row["candidate"], reference
                )
            intervals.append({
                "candidate": row["candidate"], "task": task,
                "reference": reference, "delta": delta, "ci_low": low,
                "ci_high": high, "family_wins": wins, "family_losses": losses,
            })

    best_local = max(row["local"] for row in aggregate)
    best_online = max(row["online"] for row in aggregate)
    survivors = [
        row for row in aggregate
        if row["local"] >= best_local - 0.010
        and row["online"] >= best_online - 0.015
    ]
    simplest = min(
        survivors,
        key=lambda row: (
            row["head_count"], row["coordinate_count"],
            row["fusion_terms"], row["candidate"],
        ),
    )
    survivor_intervals = []
    jointly_positive = []
    for row in survivors:
        task_intervals = {}
        for task in ("local", "online"):
            if row["candidate"] == simplest["candidate"]:
                delta, low, high, wins, losses = 0.0, 0.0, 0.0, 0, 0
            elif task == "local":
                delta, low, high, wins, losses = _bootstrap_interval(
                    [item for item in records if item["task"] == "local"],
                    row["candidate"], simplest["candidate"]
                )
            else:
                delta, low, high, wins, losses = _paired_interval(
                    [item for item in records if item["task"] == "online"],
                    row["candidate"], simplest["candidate"]
                )
            item = {
                "candidate": row["candidate"], "task": task,
                "reference": simplest["candidate"], "delta": delta,
                "ci_low": low, "ci_high": high,
                "family_wins": wins, "family_losses": losses,
            }
            survivor_intervals.append(item)
            task_intervals[task] = item
        if (
            row["candidate"] != simplest["candidate"]
            and task_intervals["local"]["ci_low"] > 0
            and task_intervals["online"]["ci_low"] > 0
        ):
            jointly_positive.append(row)
    selected = max(
        jointly_positive, key=lambda row: row["local"] + row["online"]
    ) if jointly_positive else simplest
    selected_direct = {
        task: next(
            row for row in intervals
            if row["candidate"] == selected["candidate"] and row["task"] == task
        ) for task in ("local", "online")
    }
    if all(selected_direct[task]["ci_low"] > 0 for task in ("local", "online")):
        verdict = "IMPROVES_DIRECT_COMPETITOR"
    elif (
        selected["local"] >= next(row["primary"] for row in local_direct if row["candidate"] == local_reference) - 0.010
        and selected["online"] >= next(row["primary"] for row in online_direct if row["candidate"] == online_reference) - 0.015
    ):
        verdict = "PARITY_WITH_DIRECT_COMPETITOR"
    else:
        verdict = "REGRESSES_DIRECT_COMPETITOR"
    selection = {
        "verdict": verdict,
        "selected": selected,
        "simplest_survivor": simplest,
        "best_local": best_local,
        "best_online": best_online,
        "local_reference": local_reference,
        "online_reference": online_reference,
        "local_fusion": local_variant,
        "online_fusion": online_variant,
        "survivors": [row["candidate"] for row in survivors],
        "rule": "within 0.010 Local and 0.015 Online of panel best; require wholly positive paired improvement on both tasks to displace the simplest survivor",
        "protocol_sha256": PROTOCOL_SHA256,
    }

    _write_csv(OUT / "STAGE_3_ARCHITECTURE_PER_QUESTION.csv", records)
    _write_csv(OUT / "STAGE_3_ARCHITECTURE_METRICS.csv", metrics)
    _write_csv(OUT / "STAGE_3_ARCHITECTURE_AGGREGATE.csv", aggregate)
    _write_csv(OUT / "STAGE_3_ARCHITECTURE_INTERVALS.csv", intervals)
    _write_csv(OUT / "STAGE_3_ARCHITECTURE_SURVIVOR_INTERVALS.csv", survivor_intervals)
    _write_json(OUT / "STAGE_3_ARCHITECTURE_DIAGNOSTICS.json", diagnostics)
    _write_json(OUT / "STAGE_3_ARCHITECTURE_SELECTION.json", selection)

    interval_lookup = {
        (row["candidate"], row["task"]): row for row in intervals
    }
    lines = [
        "# S3 fusion and joint architecture",
        "",
        f"**Verdict: `{verdict}`.**",
        "",
        f"Same-matrix fusion selection: Local `{local_variant}`, Online `{online_variant}`.",
        f"Joint architecture selection: `{selected['candidate']}`.",
        f"Direct references: Local `{local_reference}`, Online `{online_reference}`.",
        "",
        "| architecture | Local F1 | delta | Local 95% CI | Online AUROC | delta | Online 95% CI | heads |",
        "|---|---:|---:|---|---:|---:|---|---:|",
    ]
    for row in sorted(aggregate, key=lambda item: (item["local"] + item["online"]), reverse=True):
        li = interval_lookup[(row["candidate"], "local")]
        oi = interval_lookup[(row["candidate"], "online")]
        lines.append(
            f"| {row['candidate']} | {row['local']:.4f} | {li['delta']:+.4f} | "
            f"[{li['ci_low']:+.4f}, {li['ci_high']:+.4f}] | {row['online']:.4f} | "
            f"{oi['delta']:+.4f} | [{oi['ci_low']:+.4f}, {oi['ci_high']:+.4f}] | {row['head_count']} |"
        )
    lines.extend([
        "",
        "Tier-B critic and PRM metrics remain in the machine-readable Local table and are not treated as same-access deltas.",
    ])
    (OUT / "STAGE_3_ARCHITECTURE.md").write_text("\n".join(lines) + "\n")

    manifest_path = OUT / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "status": "STAGE_3_COMPLETE_STAGE_4_PENDING",
        "stage3_fusion_selection_sha256": _sha256(OUT / "STAGE_3_FUSION_SELECTION.json"),
        "stage3_architecture_selection_sha256": _sha256(OUT / "STAGE_3_ARCHITECTURE_SELECTION.json"),
        "elapsed_stage3_seconds": time.perf_counter() - started,
        "new_inference": False, "gpu_hours": 0, "drive_mutation": False,
    })
    _write_json(manifest_path, manifest)
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
