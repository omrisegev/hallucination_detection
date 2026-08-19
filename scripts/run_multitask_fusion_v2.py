#!/usr/bin/env python3
"""Same-matrix ordinary/uniform/DUFS/temporal controls for architecture v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_global_local_online_architecture_v2 as run  # noqa: E402
from spectral_utils.adapted_dufs import adapted_dufs_soft_gates  # noqa: E402
from spectral_utils.laplacian_upcr import (  # noqa: E402
    build_graph_from_features,
    laplacian_iu_path,
)
from spectral_utils.multitask_trajectory import (  # noqa: E402
    FIT_POSITIONS,
    causal_states,
    feature_matrix_for_head,
)


OUT = run.OUT
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
K = 7
LAMBDA = 0.1


def _orient(weights, matrix, *, risk_coordinates: bool):
    weights = np.asarray(weights, dtype=float).copy()
    score = matrix.T @ weights
    anchor = matrix.mean(axis=0)
    correlation = float(np.corrcoef(score, anchor)[0, 1])
    if np.isfinite(correlation) and correlation < 0:
        weights = -weights
    # Global mixed-v2 coordinates are confidence-oriented; Local coordinates
    # are risk-oriented.
    return (-weights if not risk_coordinates else weights), correlation


def _temporal_graph(n_traces: int, rows_per_trace: int):
    rows, cols, values = [], [], []
    for trace in range(int(n_traces)):
        start = trace * int(rows_per_trace)
        left = np.arange(start, start + int(rows_per_trace) - 1)
        right = left + 1
        rows.extend(left.tolist()); cols.extend(right.tolist()); values.extend([1.0] * len(left))
        rows.extend(right.tolist()); cols.extend(left.tolist()); values.extend([1.0] * len(left))
    n = int(n_traces) * int(rows_per_trace)
    return coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()


def _global_matrix(model, rows):
    raw = []
    for row in rows:
        features = run.causal_trace_features(row, None)
        raw.append([features.get(name, np.nan) for name in model.names])
    return np.asarray(model.transformer.transform(np.asarray(raw, dtype=float)), dtype=float)


def _global_prefix_matrix(model, rows, budget):
    raw, keep = [], []
    for index, row in enumerate(rows):
        if len(row["token_entropies"]) <= budget:
            continue
        features = run.causal_trace_features(row, budget)
        raw.append([features.get(name, np.nan) for name in model.names])
        keep.append(index)
    return np.asarray(model.transformer.transform(np.asarray(raw, dtype=float))), np.asarray(keep, dtype=int)


def _local_standardized(model, reference, row, budget=None):
    if budget is not None:
        row = run.truncate_row(row, budget)
    states = causal_states(row, reference)
    raw = np.column_stack([states[name] for name in model.feature_names])
    selected = raw[:, model.keep]
    clean = np.where(np.isfinite(selected), selected, model.median[None, :])
    return (clean - model.mean[None, :]) / model.std[None, :]


def _fit_graph_weights(calibration, models, selection):
    global_model = models.global_heads[selection["global"]]
    local_model = models.local_heads[selection["local"]]
    if not isinstance(global_model, run.RegisteredGlobal):
        raise RuntimeError("fusion runner is frozen for the selected registered Global head")
    if not isinstance(local_model, run.FrozenIUHead):
        raise RuntimeError("fusion runner is frozen for the selected token-native Local head")

    global_values = _global_matrix(global_model, calibration)
    global_F = global_values.T
    local_raw = feature_matrix_for_head(calibration, models.reference, selection["local"])
    local_selected = local_raw[:, local_model.keep]
    local_clean = np.where(np.isfinite(local_selected), local_selected, local_model.median[None, :])
    local_values = (local_clean - local_model.mean[None, :]) / local_model.std[None, :]
    local_F = local_values.T

    diagnostics, weights = {}, {"global": {}, "local": {}}
    for task, F, risk in (("global", global_F, False), ("local", local_F, True)):
        tracemalloc.start(); started = time.perf_counter()
        uniform_graph = build_graph_from_features(F, k=K)
        uniform_path = laplacian_iu_path(F, (0.0, LAMBDA), graph=uniform_graph)
        uniform_seconds = time.perf_counter() - started
        _, uniform_peak = tracemalloc.get_traced_memory(); tracemalloc.stop()

        # Exact same-matrix lambda-zero identity.
        ordinary = uniform_path[0.0].baseline.w
        if not np.array_equal(ordinary, uniform_path[0.0].w):
            raise AssertionError(f"{task} lambda=0 weights are not bit-identical")
        oriented, corr = _orient(ordinary, F, risk_coordinates=risk)
        weights[task]["ordinary"] = oriented
        weights[task]["uniform"] = _orient(
            uniform_path[LAMBDA].w, F, risk_coordinates=risk
        )[0]

        tracemalloc.start(); started = time.perf_counter()
        gates, gate_diag = adapted_dufs_soft_gates(
            F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
        )
        dufs_graph = build_graph_from_features(F, gates=gates, k=K)
        dufs_path = laplacian_iu_path(F, (0.0, LAMBDA), graph=dufs_graph)
        dufs_seconds = time.perf_counter() - started
        _, dufs_peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        if not np.array_equal(ordinary, dufs_path[0.0].w):
            raise AssertionError(f"{task} DUFS lambda=0 differs from ordinary")
        weights[task]["dufs"] = _orient(
            dufs_path[LAMBDA].w, F, risk_coordinates=risk
        )[0]

        diagnostics[task] = {
            "n_features": int(F.shape[0]), "n_fit_rows": int(F.shape[1]),
            "ordinary_orientation_correlation": corr,
            "uniform_seconds": uniform_seconds, "uniform_peak_bytes": uniform_peak,
            "dufs_seconds": dufs_seconds, "dufs_peak_bytes": dufs_peak,
            "dufs_effective_features": gate_diag.get("effective_feature_count"),
            "lambda_zero_exact": True,
        }

    temporal = _temporal_graph(len(calibration), FIT_POSITIONS)
    temporal_path = laplacian_iu_path(local_F, (0.0, LAMBDA), graph=temporal)
    if not np.array_equal(weights["local"]["ordinary"], _orient(
        temporal_path[0.0].w, local_F, risk_coordinates=True
    )[0]):
        raise AssertionError("temporal lambda=0 differs from oriented ordinary")
    weights["local"]["temporal"] = _orient(
        temporal_path[LAMBDA].w, local_F, risk_coordinates=True
    )[0]
    return weights, diagnostics


def _local_curves(rows, model, reference, weights):
    output = {name: [] for name in weights}
    for row in rows:
        standardized = _local_standardized(model, reference, row)
        for name, vector in weights.items():
            output[name].append(standardized @ vector)
    return output


def _global_scores(rows, model, weights):
    values = _global_matrix(model, rows)
    return {name: values @ vector for name, vector in weights.items()}


def main() -> None:
    selection = json.load(open(OUT / "HEAD_SELECTION.json", encoding="utf-8"))["selected"]
    records, metrics, diagnostics = [], [], {}
    for model_name, family in sorted(run.DEV_CELLS):
        rows = run.load_rows(run._cell_path(model_name, family))
        calibration, evaluation = run._split(rows)
        print(f"[fusion] {model_name}/{family}", flush=True)
        models = run._fit_selected_cell(calibration, selection)
        weights, diag = _fit_graph_weights(calibration, models, selection)
        diagnostics[f"{model_name}/{family}"] = diag
        global_model = models.global_heads[selection["global"]]
        local_model = models.local_heads[selection["local"]]
        cal_global = _global_scores(calibration, global_model, weights["global"])
        eval_global = _global_scores(evaluation, global_model, weights["global"])
        cal_local = _local_curves(calibration, local_model, models.reference, weights["local"])
        eval_local = _local_curves(evaluation, local_model, models.reference, weights["local"])
        cal_global_target = np.asarray([int(not bool(row["final_answer_correct"])) for row in calibration])
        eval_global_target = np.asarray([int(not bool(row["final_answer_correct"])) for row in evaluation])
        cal_local_target = np.asarray([int(row["label"]) for row in calibration])
        eval_local_target = np.asarray([int(row["label"]) for row in evaluation])

        global_prefix = {name: {} for name in weights["global"]}
        local_prefix = {name: {} for name in weights["local"]}
        for budget in (64, 128):
            values, keep = _global_prefix_matrix(global_model, evaluation, budget)
            for name, vector in weights["global"].items():
                global_prefix[name][budget] = (keep, values @ vector)
            for name, vector in weights["local"].items():
                selected = []
                for index, row in enumerate(evaluation):
                    if len(row["token_entropies"]) <= budget:
                        continue
                    standardized = _local_standardized(local_model, models.reference, row, budget)
                    selected.append(float(np.max(standardized @ vector)))
                local_prefix[name][budget] = np.asarray(selected)

        for global_name in weights["global"]:
            for local_name in weights["local"]:
                method = f"global_{global_name}__local_{local_name}"
                cal_g_fit = run._zfit(cal_global[global_name])
                cal_l_max = np.asarray([np.max(curve) for curve in cal_local[local_name]])
                eval_l_max = np.asarray([np.max(curve) for curve in eval_local[local_name]])
                cal_l_fit = run._zfit(cal_l_max)
                cal_detector = 0.5 * run._zapply(cal_global[global_name], cal_g_fit) + 0.5 * run._zapply(cal_l_max, cal_l_fit)
                eval_detector = 0.5 * run._zapply(eval_global[global_name], cal_g_fit) + 0.5 * run._zapply(eval_l_max, cal_l_fit)
                cal_locator = np.asarray([run._peak_locator(curve, row) for curve, row in zip(cal_local[local_name], calibration)])
                eval_locator = np.asarray([run._peak_locator(curve, row) for curve, row in zip(eval_local[local_name], evaluation)])
                threshold, _ = run._best_threshold(cal_detector, cal_locator, cal_local_target)
                prediction = np.where(eval_detector > threshold, eval_locator, -1)
                local_metric = run._processbench(prediction, eval_local_target)
                online_auc = []
                for budget in (64, 128):
                    keep, global_score = global_prefix[global_name][budget]
                    local_score = local_prefix[local_name][budget]
                    score = 0.5 * run._zapply(global_score, cal_g_fit) + 0.5 * run._zapply(local_score, cal_l_fit)
                    label = eval_global_target[keep]
                    auc = run._safe_auc(label, score)
                    online_auc.append(auc)
                    records.extend({
                        "model": model_name, "family": family, "unit": evaluation[index]["_unit"],
                        "method": method, "task": "online", "budget": budget,
                        "target": int(target), "score": float(value),
                    } for index, target, value in zip(keep, label, score))
                metrics.extend([
                    {"model": model_name, "family": family, "method": method, "global_fusion": global_name, "local_fusion": local_name, "task": "global", "primary": run._safe_auc(eval_global_target, eval_global[global_name])},
                    {"model": model_name, "family": family, "method": method, "global_fusion": global_name, "local_fusion": local_name, "task": "local", "primary": local_metric["f1"], **local_metric},
                    {"model": model_name, "family": family, "method": method, "global_fusion": global_name, "local_fusion": local_name, "task": "online", "primary": float(np.mean(online_auc))},
                ])
                records.extend({
                    "model": model_name, "family": family, "unit": row["_unit"],
                    "method": method, "task": "global", "budget": "final",
                    "target": int(target), "score": float(score),
                } for row, target, score in zip(evaluation, eval_global_target, eval_global[global_name]))
                records.extend({
                    "model": model_name, "family": family, "unit": row["_unit"],
                    "method": method, "task": "local", "budget": "final",
                    "target": int(target), "score": float(score), "prediction": int(pred),
                } for row, target, score, pred in zip(evaluation, eval_local_target, eval_detector, prediction))

    aggregate = []
    for method in sorted({row["method"] for row in metrics}):
        item = {"method": method}
        example = next(row for row in metrics if row["method"] == method)
        item.update({"global_fusion": example["global_fusion"], "local_fusion": example["local_fusion"]})
        for task in ("global", "local", "online"):
            item[task] = float(np.mean([row["primary"] for row in metrics if row["method"] == method and row["task"] == task]))
        aggregate.append(item)
    run._write_csv(OUT / "FUSION_PER_QUESTION.csv", records)
    run._write_csv(OUT / "FUSION_METRICS.csv", metrics)
    run._write_csv(OUT / "FUSION_AGGREGATE.csv", aggregate)
    run._write_json(OUT / "FUSION_DIAGNOSTICS.json", diagnostics)
    run._write_json(OUT / "FUSION_MANIFEST.json", {
        "status": "COMPLETE", "lambda": LAMBDA, "k": K,
        "dufs_seeds": DUFS_SEEDS, "dufs_epochs": DUFS_EPOCHS,
        "lambda_zero_exact": True, "selection_cells_only": True,
        "labels_seen_during_fit": False,
        "score_hash": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
    })
    print(f"[done] wrote fusion controls to {OUT}", flush=True)


if __name__ == "__main__":
    main()
