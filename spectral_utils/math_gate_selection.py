"""Selection utilities for the 15-cell math answer-gate development panel."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


Q_GRID = tuple(float(value) for value in np.arange(0.05, 1.0, 0.05))
C_GRID = (0.01, 0.1, 1.0, 10.0)
CORRELATION_CUTOFF = 0.995
SIMPLEX_EPSILON = 0.05
MAX_FEATURES = 4


def percentile_by_cell(values: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Map each column to mid-rank empirical percentiles inside each cell."""
    values = np.asarray(values, dtype=np.float64)
    cells = np.asarray(cells).astype(str)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[:, None]
    if values.ndim != 2 or len(values) != len(cells) or not np.isfinite(values).all():
        raise ValueError("percentile calibration requires a finite aligned matrix")
    output = np.empty_like(values)
    for cell in sorted(set(cells)):
        mask = cells == cell
        for column in range(values.shape[1]):
            output[mask, column] = (
                rankdata(values[mask, column], method="average") - 0.5
            ) / int(mask.sum())
    return output[:, 0] if one_dimensional else output


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def _safe_auprc(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def _binary_summary(y: np.ndarray, prediction: np.ndarray, score: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    tp = int(np.sum((y == 1) & (prediction == 1)))
    return {
        "n": int(len(y)),
        "errors": int(y.sum()),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "accuracy": float(np.mean(y == prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "auroc": _safe_auc(y, score),
        "auprc": _safe_auprc(y, score),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_score(
    y: np.ndarray,
    score: np.ndarray,
    cells: np.ndarray,
    families: np.ndarray,
    q: float,
) -> dict:
    """Evaluate one uniform percentile threshold with family-first macros."""
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    cells = np.asarray(cells).astype(str)
    families = np.asarray(families).astype(str)
    if not (len(y) == len(score) == len(cells) == len(families)):
        raise ValueError("score evaluation arrays are not aligned")
    prediction = (score >= float(q)).astype(np.int8)
    output = evaluate_binary_prediction(y, prediction, score, cells, families)
    output["q"] = float(q)
    return output


def evaluate_binary_prediction(
    y: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray,
    cells: np.ndarray,
    families: np.ndarray,
) -> dict:
    """Evaluate an already-thresholded binary gate and its continuous score."""
    y = np.asarray(y, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    cells = np.asarray(cells).astype(str)
    families = np.asarray(families).astype(str)
    if not (len(y) == len(prediction) == len(score) == len(cells) == len(families)):
        raise ValueError("binary gate evaluation arrays are not aligned")
    if not np.isin(prediction, (0, 1)).all() or not np.isfinite(score).all():
        raise ValueError("binary gate evaluation requires finite scores and binary predictions")
    per_cell = {}
    metrics = ("macro_f1", "accuracy", "sensitivity", "specificity", "auroc", "auprc")
    for cell in sorted(set(cells)):
        mask = cells == cell
        family_values = set(families[mask])
        if len(family_values) != 1:
            raise ValueError("cell spans multiple dataset families")
        per_cell[cell] = {
            "family": next(iter(family_values)),
            **_binary_summary(y[mask], prediction[mask], score[mask]),
        }
    per_family = {}
    for family in sorted(set(families)):
        members = [value for value in per_cell.values() if value["family"] == family]
        per_family[family] = {
            "cells": len(members),
            **{
                key: float(np.nanmean([value[key] for value in members]))
                for key in metrics
            },
        }
    family_macro = {
        key: float(np.nanmean([value[key] for value in per_family.values()]))
        for key in metrics
    }
    cell_macro = {
        key: float(np.nanmean([value[key] for value in per_cell.values()]))
        for key in metrics
    }
    return {
        "family_macro": family_macro,
        "cell_macro": cell_macro,
        "worst_family_macro_f1": float(min(value["macro_f1"] for value in per_family.values())),
        "worst_cell_macro_f1": float(min(value["macro_f1"] for value in per_cell.values())),
        "families": per_family,
        "cells": per_cell,
        "pooled": _binary_summary(y, prediction, score),
    }


def _q_key(result: dict) -> tuple:
    return (
        -result["family_macro"]["macro_f1"],
        -result["family_macro"]["auroc"],
        -result["worst_family_macro_f1"],
        -result["worst_cell_macro_f1"],
        abs(result["q"] - 0.5),
        result["q"],
    )


def select_q(y, score, cells, families) -> dict:
    curve = [evaluate_score(y, score, cells, families, q) for q in Q_GRID]
    selected = min(curve, key=_q_key)
    return {
        "selected_q": selected["q"],
        "selected": selected,
        "curve": [
            {
                "q": row["q"],
                "family_macro_f1": row["family_macro"]["macro_f1"],
                "family_macro_auroc": row["family_macro"]["auroc"],
                "family_macro_auprc": row["family_macro"]["auprc"],
                "cell_macro_f1": row["cell_macro"]["macro_f1"],
                "worst_family_macro_f1": row["worst_family_macro_f1"],
                "worst_cell_macro_f1": row["worst_cell_macro_f1"],
            }
            for row in curve
        ],
    }


def result_key(result: dict, *, n_features: int, name: str) -> tuple:
    selected = result["selected"]
    return (
        -selected["family_macro"]["macro_f1"],
        -selected["family_macro"]["auroc"],
        -selected["worst_family_macro_f1"],
        -selected["worst_cell_macro_f1"],
        int(n_features),
        str(name),
    )


def rank_single_candidates(y, matrix, names, cells, families) -> list[dict]:
    output = []
    for column, name in enumerate(names):
        result = select_q(y, matrix[:, column], cells, families)
        output.append({"method": str(name), **result})
    output.sort(key=lambda row: result_key(row, n_features=1, name=row["method"]))
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def remove_near_duplicates(matrix, names, ranked, cutoff=CORRELATION_CUTOFF) -> dict:
    index = {name: position for position, name in enumerate(names)}
    kept, dropped = [], []
    for row in ranked:
        name = row["method"]
        column = matrix[:, index[name]]
        duplicate = None
        for earlier in kept:
            corr = float(np.corrcoef(column, matrix[:, index[earlier]])[0, 1])
            if np.isfinite(corr) and abs(corr) >= cutoff:
                duplicate = {"method": name, "kept": earlier, "correlation": corr}
                break
        if duplicate is None:
            kept.append(name)
        else:
            dropped.append(duplicate)
    return {"cutoff": float(cutoff), "kept": kept, "dropped": dropped}


def greedy_equal_selection(y, matrix, names, cells, families, ranked, eligible) -> dict:
    index = {name: position for position, name in enumerate(names)}
    selected = [next(row["method"] for row in ranked if row["method"] in eligible)]
    current_score = matrix[:, index[selected[0]]]
    current = select_q(y, current_score, cells, families)
    trace = [{"step": 1, "added": selected[0], "result": current}]
    while len(selected) < MAX_FEATURES:
        trials = []
        for candidate in eligible:
            if candidate in selected:
                continue
            proposed = selected + [candidate]
            score = percentile_by_cell(
                matrix[:, [index[name] for name in proposed]].mean(axis=1), cells
            )
            result = select_q(y, score, cells, families)
            trials.append((result_key(result, n_features=len(proposed), name=candidate), candidate, result))
        if not trials:
            break
        _, candidate, result = min(trials, key=lambda item: item[0])
        previous = current["selected"]["family_macro"]["macro_f1"]
        proposed = result["selected"]["family_macro"]["macro_f1"]
        trace.append({
            "step": len(selected) + 1,
            "candidate": candidate,
            "previous_family_macro_f1": previous,
            "proposed_family_macro_f1": proposed,
            "accepted": bool(proposed > previous + 1e-12),
            "result": result,
        })
        if proposed <= previous + 1e-12:
            break
        selected.append(candidate)
        current = result
    score = percentile_by_cell(
        matrix[:, [index[name] for name in selected]].mean(axis=1), cells
    )
    return {"selected": selected, "trace": trace, "score": score, "result": current}


def balanced_sample_weight(cells: np.ndarray, families: np.ndarray) -> np.ndarray:
    cells = np.asarray(cells).astype(str)
    families = np.asarray(families).astype(str)
    output = np.zeros(len(cells), dtype=np.float64)
    unique_families = sorted(set(families))
    for family in unique_families:
        family_cells = sorted(set(cells[families == family]))
        for cell in family_cells:
            mask = cells == cell
            output[mask] = 1.0 / (len(unique_families) * len(family_cells) * int(mask.sum()))
    output *= len(output) / output.sum()
    return output


def fit_simplex(matrix, y, cells, families, epsilon=SIMPLEX_EPSILON) -> dict:
    matrix = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sample_weight = balanced_sample_weight(cells, families)
    prevalence = float(np.average(y, weights=sample_weight))
    intercept = float(np.log((prevalence + 1e-8) / (1.0 - prevalence + 1e-8)))
    initial = np.r_[intercept, np.full(matrix.shape[1], 1.0 / matrix.shape[1])]

    def objective(parameters):
        z = parameters[0] + matrix @ parameters[1:]
        loss = np.logaddexp(0.0, z) - y * z
        return float(np.average(loss, weights=sample_weight) + 1e-4 * np.sum(parameters[1:] ** 2))

    fitted = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(None, None)] + [(0.0, 1.0)] * matrix.shape[1],
        constraints={"type": "eq", "fun": lambda parameters: parameters[1:].sum() - 1.0},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError("simplex fit failed: " + str(fitted.message))
    weights = np.maximum(fitted.x[1:], 0.0)
    weights[weights < float(epsilon)] = 0.0
    if weights.sum() <= 0:
        weights[int(np.argmax(fitted.x[1:]))] = 1.0
    weights /= weights.sum()
    return {
        "intercept": float(fitted.x[0]),
        "weights": weights,
        "raw_weights": np.asarray(fitted.x[1:], dtype=np.float64),
        "epsilon": float(epsilon),
        "objective": float(fitted.fun),
        "iterations": int(fitted.nit),
    }


def crossfit_simplex(matrix, y, cells, families) -> tuple[np.ndarray, list[dict]]:
    output = np.full(len(y), np.nan, dtype=np.float64)
    fits = []
    for held_out in sorted(set(cells)):
        test = cells == held_out
        train = ~test
        fitted = fit_simplex(matrix[train], y[train], cells[train], families[train])
        output[test] = fitted["intercept"] + matrix[test] @ fitted["weights"]
        fits.append({
            "held_out": str(held_out),
            "intercept": fitted["intercept"],
            "weights": fitted["weights"].tolist(),
            "raw_weights": fitted["raw_weights"].tolist(),
            "iterations": fitted["iterations"],
        })
    if not np.isfinite(output).all():
        raise ValueError("simplex cross-fit produced nonfinite scores")
    return output, fits


def _fit_logistic(matrix, y, cells, families, c_value) -> LogisticRegression:
    model = LogisticRegression(
        l1_ratio=1.0,
        solver="liblinear",
        C=float(c_value),
        random_state=0,
        max_iter=5000,
    )
    model.fit(matrix, y, sample_weight=balanced_sample_weight(cells, families))
    return model


def choose_logistic_c(matrix, y, cells, families) -> dict:
    rows = []
    for c_value in C_GRID:
        score = np.full(len(y), np.nan, dtype=np.float64)
        for held_out in sorted(set(cells)):
            test = cells == held_out
            train = ~test
            fitted = _fit_logistic(matrix[train], y[train], cells[train], families[train], c_value)
            score[test] = fitted.decision_function(matrix[test])
        ranked = percentile_by_cell(score, cells)
        result = select_q(y, ranked, cells, families)
        rows.append({"C": float(c_value), "result": result})
    selected = min(
        rows,
        key=lambda row: result_key(row["result"], n_features=matrix.shape[1], name=str(row["C"])),
    )
    return {"selected_C": selected["C"], "candidates": rows}


def crossfit_logistic(matrix, y, cells, families) -> tuple[np.ndarray, list[dict]]:
    output = np.full(len(y), np.nan, dtype=np.float64)
    fits = []
    for held_out in sorted(set(cells)):
        test = cells == held_out
        train = ~test
        c_selection = choose_logistic_c(matrix[train], y[train], cells[train], families[train])
        fitted = _fit_logistic(
            matrix[train], y[train], cells[train], families[train], c_selection["selected_C"]
        )
        output[test] = fitted.decision_function(matrix[test])
        fits.append({
            "held_out": str(held_out),
            "C": c_selection["selected_C"],
            "intercept": float(fitted.intercept_[0]),
            "coefficients": fitted.coef_[0].astype(float).tolist(),
        })
    if not np.isfinite(output).all():
        raise ValueError("logistic cross-fit produced nonfinite scores")
    return output, fits


def develop_fusions(y, raw_matrix, names, cells, families) -> tuple[dict, dict[str, np.ndarray]]:
    """Run single screening, common forward selection, and four fusion arms."""
    matrix = percentile_by_cell(raw_matrix, cells)
    singles = rank_single_candidates(y, matrix, names, cells, families)
    duplicate_audit = remove_near_duplicates(matrix, names, singles)
    forward = greedy_equal_selection(
        y, matrix, names, cells, families, singles, duplicate_audit["kept"]
    )
    index = {name: position for position, name in enumerate(names)}
    selected_names = forward["selected"]
    selected_matrix = matrix[:, [index[name] for name in selected_names]]

    best_single = singles[0]
    scores = {
        "best_single": matrix[:, index[best_single["method"]]],
        "equal_mean": forward["score"],
    }
    simplex_raw, simplex_folds = crossfit_simplex(selected_matrix, y, cells, families)
    logistic_raw, logistic_folds = crossfit_logistic(selected_matrix, y, cells, families)
    scores["simplex"] = percentile_by_cell(simplex_raw, cells)
    scores["sparse_logistic"] = percentile_by_cell(logistic_raw, cells)

    arm_results = {
        "best_single": {
            "features": [best_single["method"]],
            "result": select_q(y, scores["best_single"], cells, families),
        },
        "equal_mean": {
            "features": selected_names,
            "result": select_q(y, scores["equal_mean"], cells, families),
        },
        "simplex": {
            "features": selected_names,
            "result": select_q(y, scores["simplex"], cells, families),
            "crossfit": simplex_folds,
        },
        "sparse_logistic": {
            "features": selected_names,
            "result": select_q(y, scores["sparse_logistic"], cells, families),
            "crossfit": logistic_folds,
        },
    }
    winner = min(
        arm_results,
        key=lambda name: result_key(
            arm_results[name]["result"],
            n_features=len(arm_results[name]["features"]),
            name=name,
        ),
    )

    simplex_final = fit_simplex(selected_matrix, y, cells, families)
    logistic_c = choose_logistic_c(selected_matrix, y, cells, families)
    logistic_final = _fit_logistic(
        selected_matrix, y, cells, families, logistic_c["selected_C"]
    )
    frozen = {
        "arm": winner,
        "features": arm_results[winner]["features"],
        "q": arm_results[winner]["result"]["selected_q"],
        "percentile_calibration": "midrank within cell before and after fusion",
        "simplex": {
            "features": selected_names,
            "intercept": simplex_final["intercept"],
            "weights": simplex_final["weights"].tolist(),
            "raw_weights": simplex_final["raw_weights"].tolist(),
            "epsilon": simplex_final["epsilon"],
        },
        "sparse_logistic": {
            "features": selected_names,
            "C": logistic_c["selected_C"],
            "intercept": float(logistic_final.intercept_[0]),
            "coefficients": logistic_final.coef_[0].astype(float).tolist(),
        },
    }
    report = {
        "schema": "math-gate-development-selection-v1",
        "status": "DEVELOPMENT_SELECTION",
        "q_grid": list(Q_GRID),
        "single_ranking": singles,
        "duplicate_audit": duplicate_audit,
        "forward_selection": {key: value for key, value in forward.items() if key != "score"},
        "arms": arm_results,
        "winner": winner,
        "frozen_gate": frozen,
        "selection_uses_all_math_development_labels": True,
        "not_external_confirmation": True,
    }
    return report, scores


def apply_frozen_fusion(matrix, names, cells, frozen) -> np.ndarray:
    """Apply one math-frozen arm to another unlabeled panel."""
    calibrated = percentile_by_cell(matrix, cells)
    index = {name: position for position, name in enumerate(names)}
    features = frozen["features"]
    selected = calibrated[:, [index[name] for name in features]]
    arm = frozen["arm"]
    if arm == "best_single":
        raw = selected[:, 0]
    elif arm == "equal_mean":
        raw = selected.mean(axis=1)
    elif arm == "simplex":
        model = frozen["simplex"]
        selected = calibrated[:, [index[name] for name in model["features"]]]
        raw = float(model["intercept"]) + selected @ np.asarray(model["weights"], dtype=float)
    elif arm == "sparse_logistic":
        model = frozen["sparse_logistic"]
        selected = calibrated[:, [index[name] for name in model["features"]]]
        raw = float(model["intercept"]) + selected @ np.asarray(model["coefficients"], dtype=float)
    else:
        raise ValueError("unknown frozen gate arm: " + str(arm))
    return percentile_by_cell(raw, cells)


__all__ = [
    "C_GRID", "CORRELATION_CUTOFF", "MAX_FEATURES", "Q_GRID", "SIMPLEX_EPSILON",
    "apply_frozen_fusion", "balanced_sample_weight", "develop_fusions",
    "evaluate_binary_prediction", "evaluate_score", "fit_simplex", "percentile_by_cell",
    "select_q",
]
