#!/usr/bin/env python3
"""Develop and confirm GL-LIU v1 on ProcessBench caches.

The method never partitions text into steps while constructing scores.  It
computes full-trace answer features and continuous per-token moving-window
views.  ProcessBench step spans are read only after all scores freeze, to map a
predicted token position to the benchmark's step label and evaluate it.

Development cells: Qwen3-4B GSM8K and MATH.
Confirmation cells: Qwen3-4B OlympiadBench/OmniMath and all Qwen3-8B cells.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
SUPPORT = Path(__file__).resolve().parent
LOC = SUPPORT / "localization"

package = types.ModuleType("spectral_utils")
package.__path__ = [str(ROOT / "spectral_utils")]
sys.modules["spectral_utils"] = package
sys.path[:0] = [str(ROOT), str(SUPPORT), str(LOC)]

from evaluate_answer_level import fit_scores as fit_answer_stable, trace_features
from optimize_localization import (
    RAW_CORE, RAW_FULL, apply_arm, build_channels, fit_arm,
)
from two_stage_localization import evaluate_two_stage
from spectral_utils.adapted_dufs import adapted_dufs_soft_gates
from spectral_utils.dufs_liu_feature_contract import dufs_liu_mixed_v2_matrix
from spectral_utils.feature_contract import CONFIDENCE_FEATURE_SIGNS_V1
from spectral_utils.laplacian_upcr import build_graph_from_features, laplacian_iu_path
from spectral_utils.streaming_utils import anchor_orient
from spectral_utils.upcr import upcr_fit
from evidence_drop import EVIDENCE_FNS, evidence_drop_risk
from localization_metrics import NO_ERROR, step_drop_scores


DEV = {("qwen3_4b", "gsm8k"), ("qwen3_4b", "math")}
LAMBDAS = (0.03, 0.1, 0.3)
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
K = 7
MAX_GRAPH_TOKENS = 60_000
POSITIONAL_FIT = dict(
    loss="l2", exclusion=True, difficulty_gate=False,
    simple_avg_fallback=True, recompute_after_exclusion=True,
    g2_projection_k=1, scale_ratio=0.25,
)


def load_rows(path):
    import pickle
    with open(path, "rb") as handle:
        cache = pickle.load(handle)
    return [cache[key] for key in sorted(cache)
            if not cache[key]["align_diag"]["problems"]]


def _hash_rows(values):
    array = np.concatenate([np.asarray(value, dtype=float) for value in values])
    return hashlib.sha256(array.astype("<f8").tobytes()).hexdigest()


def fit_answer_mixed(rows):
    features = [trace_features(row) for row in rows]
    names, columns, availability = [], [], {}
    for name in CONFIDENCE_FEATURE_SIGNS_V1:
        raw = np.asarray([item.get(name, np.nan) for item in features], dtype=float)
        finite = np.isfinite(raw)
        availability[name] = float(finite.mean())
        if finite.mean() < 0.70 or not finite.any():
            continue
        raw = np.where(finite, raw, np.median(raw[finite]))
        if raw.std() < 1e-8 or np.mean(raw == np.median(raw)) > 0.40:
            continue
        names.append(name)
        columns.append(raw)
    raw = np.column_stack(columns)
    transformed, names, details = dufs_liu_mixed_v2_matrix(raw, names)
    F = transformed.T
    gates, gate_diag = adapted_dufs_soft_gates(
        F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
    )
    dufs_graph = build_graph_from_features(F, gates=gates, k=K)
    dufs_path = laplacian_iu_path(F, (0.0, 0.1), graph=dufs_graph)
    uniform_graph = build_graph_from_features(F, k=K)
    uniform = laplacian_iu_path(F, (0.1,), graph=uniform_graph)[0.1]
    return {
        "answer_iu_mixed": -(dufs_path[0.0].baseline.w @ F),
        "answer_dufs_liu_mixed": -(dufs_path[0.1].w @ F),
        "answer_uniform_liu_mixed": -(uniform.w @ F),
    }, {
        "feature_names": list(names), "n_features": len(names),
        "availability": availability, "contract": details,
        "gate_effective_count": gate_diag.get("effective_feature_count"),
    }


def answer_detectors(rows):
    confidence, stable_diag = fit_answer_stable(rows)
    keys = {
        "deployed_upcr_fixed_stable": "answer_deployed_upcr_stable",
        "iu_pcr_fixed_stable": "answer_iu_stable",
        "dufs_liu_fixed_stable_l0p1": "answer_dufs_liu_stable",
        "uniform_liu_fixed_stable_l0p1": "answer_uniform_liu_stable",
    }
    output = {new: -np.asarray(confidence[old], dtype=float)
              for old, new in keys.items()}
    mixed, mixed_diag = fit_answer_mixed(rows)
    output.update(mixed)
    return output, {"stable": stable_diag, "mixed": mixed_diag}


def _prepare_token_matrix(channels, row_indices, names, max_tokens=MAX_GRAPH_TOKENS):
    chunks, lengths, total = [], [], 0
    for index in row_indices:
        length = len(channels[names[0]][index])
        if length <= 0:
            continue
        take = min(length, max_tokens - total)
        if take <= 0:
            break
        chunks.append((int(index), int(take)))
        lengths.append(int(take))
        total += int(take)
        if total >= max_tokens:
            break
    raw_columns, mu, sd, kept = [], [], [], []
    for name in names:
        values = np.concatenate([
            np.asarray(channels[name][index], dtype=float)[:take]
            for index, take in chunks
        ])
        finite = np.isfinite(values)
        if not finite.any():
            continue
        median = float(np.median(values[finite]))
        values = np.where(finite, values, median)
        scale = float(values.std())
        if scale < 1e-8:
            continue
        kept.append(name); mu.append(float(values.mean())); sd.append(scale)
        raw_columns.append((values - values.mean()) / scale)
    V = np.column_stack(raw_columns)
    first = upcr_fit(V.T, **POSITIONAL_FIT)
    derived = np.sign(first.rho_hat_full)
    derived[derived == 0] = 1.0
    F = (V * derived).T
    return F, V, kept, np.asarray(mu), np.asarray(sd), derived, chunks, lengths


def _temporal_graph(lengths):
    rows, cols, values, start = [], [], [], 0
    for length in lengths:
        if length > 1:
            left = np.arange(start, start + length - 1)
            right = left + 1
            rows.extend(left.tolist()); cols.extend(right.tolist()); values.extend([1.0] * len(left))
            rows.extend(right.tolist()); cols.extend(left.tolist()); values.extend([1.0] * len(left))
        start += length
    return coo_matrix((values, (rows, cols)), shape=(start, start)).tocsr()


def fit_token_laplacian(channels, fit_rows):
    F, V, names, mu, sd, derived, chunks, lengths = _prepare_token_matrix(
        channels, fit_rows, RAW_CORE
    )
    graphs = {"uniform": build_graph_from_features(F, k=K)}
    gates, gate_diag = adapted_dufs_soft_gates(
        F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
    )
    graphs["dufs"] = build_graph_from_features(F, gates=gates, k=K)
    graphs["temporal"] = _temporal_graph(lengths)

    fits = {}
    baseline = laplacian_iu_path(F, (0.0,), graph=graphs["uniform"])[0.0]
    fits["token_iu"] = baseline.baseline.w
    diagnostics = {"token_names": names, "n_fit_tokens": F.shape[1],
                   "gate_effective_count": gate_diag.get("effective_feature_count"),
                   "paths": {}}
    anchor = V[:, names.index("entropy_series")]
    for graph_name, graph in graphs.items():
        path = laplacian_iu_path(F, LAMBDAS, graph=graph)
        diagnostics["paths"][graph_name] = {}
        for lambda_, fitted in path.items():
            key = f"token_{graph_name}_liu_l{str(lambda_).replace('.', 'p')}"
            fits[key] = fitted.w
            diagnostics["paths"][graph_name][str(lambda_)] = fitted.diagnostics

    arms = {}
    for name, weights in fits.items():
        _, flipped = anchor_orient(weights @ F, anchor)
        arms[name] = {
            "names": names, "mu": mu, "sd": sd, "derived": derived,
            "weights": np.asarray(weights), "flipped": bool(flipped),
        }
    return arms, diagnostics


def apply_token_arm(arm, channels):
    output = []
    n_rows = len(next(iter(channels.values())))
    for row_index in range(n_rows):
        columns = []
        for j, name in enumerate(arm["names"]):
            values = np.asarray(channels[name][row_index], dtype=float)
            values = np.where(np.isfinite(values), values, arm["mu"][j])
            columns.append((values - arm["mu"][j]) / arm["sd"][j] * arm["derived"][j])
        risk = arm["weights"] @ np.vstack(columns)
        if arm["flipped"]:
            risk = -risk
        output.append(np.asarray(risk, dtype=float))
    return output


def token_scores(rows):
    channels, fit_rows, mode_centres = build_channels(rows)
    curves = {}
    for name, pool in {"token_upcr_core": RAW_CORE, "token_upcr_full": RAW_FULL}.items():
        arm = fit_arm(channels, fit_rows, pool)
        curves[name] = apply_arm(arm, channels)
    lap_arms, lap_diag = fit_token_laplacian(channels, fit_rows)
    for name, arm in lap_arms.items():
        curves[name] = apply_token_arm(arm, channels)
    return curves, {
        "mode_centres": mode_centres, "fit_rows": fit_rows,
        "laplacian": lap_diag,
    }


def aggregate_curve(values, kind):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0
    if kind == "max":
        return float(np.max(values))
    if kind == "top5mean":
        count = max(1, int(math.ceil(0.05 * len(values))))
        return float(np.mean(np.partition(values, len(values) - count)[-count:]))
    raise ValueError(kind)


def token_detectors(curves):
    return {
        f"{name}__{kind}": np.asarray([aggregate_curve(row, kind) for row in values])
        for name, values in curves.items() for kind in ("max", "top5mean")
    }


def token_to_step(curve, row):
    curve = np.asarray(curve, dtype=float)
    finite = np.isfinite(curve)
    if not finite.any():
        return NO_ERROR
    index = int(np.nanargmax(curve))
    for step, span in enumerate(row["step_token_spans"]):
        if span is not None and span[0] <= index < span[1]:
            return int(step)
    return NO_ERROR


def token_locators(curves, rows):
    return {
        name: np.asarray([token_to_step(curve, row)
                          for curve, row in zip(values, rows)], dtype=int)
        for name, values in curves.items()
    }


def mindgap_control(rows):
    detector, locator = [], []
    for row in rows:
        evidence = EVIDENCE_FNS["shannon"](row, 20)
        detector.append(evidence_drop_risk(evidence, M=5, ema_span=5))
        step = step_drop_scores(evidence, row["step_token_spans"], ema_span=5)
        locator.append(int(np.nanargmax(step)) if np.isfinite(step).any() else NO_ERROR)
    return np.asarray(detector), np.asarray(locator)


def component_metrics(detectors, locators, labels):
    error = labels != NO_ERROR
    detector_rows = []
    for name, risk in detectors.items():
        detector_rows.append({
            "candidate": name,
            "auroc": float(roc_auc_score(error, risk)),
            "auprc": float(average_precision_score(error, risk)),
        })
    locator_rows = []
    for name, predicted in locators.items():
        exact = float(np.mean(predicted[error] == labels[error]))
        tol1 = float(np.mean(np.abs(predicted[error] - labels[error]) <= 1))
        locator_rows.append({"candidate": name, "exact": exact, "tol1": tol1})
    return detector_rows, locator_rows


def evaluate_cell(model, subset, path):
    rows = load_rows(path)
    answer, answer_diag = answer_detectors(rows)
    curves, token_diag = token_scores(rows)
    detectors = dict(answer)
    detectors.update(token_detectors(curves))
    locators = token_locators(curves, rows)
    hashes = {
        "detectors": {name: hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()
                      for name, value in detectors.items()},
        "token_curves": {name: _hash_rows(value) for name, value in curves.items()},
    }
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    detector_rows, locator_rows = component_metrics(detectors, locators, labels)
    return {
        "model": model, "subset": subset, "rows": rows, "labels": labels,
        "detectors": detectors, "locators": locators, "curves": curves,
        "detector_metrics": detector_rows, "locator_metrics": locator_rows,
        "diag": {"answer": answer_diag, "token": token_diag,
                 "hashes_before_labels": hashes, "labels_seen_during_fit": False},
    }


def macro_select(cells, metric, primary, secondary):
    values = {}
    for cell in cells:
        for row in cell[metric]:
            values.setdefault(row["candidate"], []).append(row)
    summary = []
    for candidate, rows in values.items():
        if len(rows) != len(cells):
            continue
        summary.append({
            "candidate": candidate,
            primary: float(np.mean([row[primary] for row in rows])),
            secondary: float(np.mean([row[secondary] for row in rows])),
        })
    summary.sort(key=lambda row: (row[primary], row[secondary]), reverse=True)
    return summary


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(row.keys() for row in rows))))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cells = []
    for model in ("qwen3_4b", "qwen3_8b"):
        folder = os.path.join(args.cache_root, "pb_" + model)
        for subset in ("gsm8k", "math", "olympiadbench", "omnimath"):
            path = os.path.join(folder, f"processbench_{subset}.pkl")
            print(f"FIT {model} {subset}", flush=True)
            cell = evaluate_cell(model, subset, path)
            cells.append(cell)
            with open(os.path.join(args.out_dir, f"{model}__{subset}__diagnostics.json"), "w") as handle:
                json.dump(cell["diag"], handle, indent=2, default=str)

    development = [cell for cell in cells if (cell["model"], cell["subset"]) in DEV]
    detector_summary = macro_select(development, "detector_metrics", "auroc", "auprc")
    locator_summary = macro_select(development, "locator_metrics", "exact", "tol1")
    selected_detector = detector_summary[0]["candidate"]
    selected_locator = locator_summary[0]["candidate"]
    print("SELECTED", selected_detector, selected_locator, flush=True)

    per_cell = []
    component_per_cell = []
    for cell in cells:
        labels = cell["labels"]
        ours = evaluate_two_stage(
            cell["detectors"][selected_detector],
            cell["locators"][selected_locator], labels,
        )
        paper_detector, paper_locator = mindgap_control(cell["rows"])
        paper = evaluate_two_stage(paper_detector, paper_locator, labels)
        hybrid = evaluate_two_stage(
            paper_detector, cell["locators"][selected_locator], labels,
        )
        for system, metrics in (("ours_only", ours), ("mindgap_control", paper),
                                ("mindgap_detector_ours_locator", hybrid)):
            per_cell.append({"model": cell["model"], "subset": cell["subset"],
                             "split": "development" if (cell["model"], cell["subset"]) in DEV else "confirmation",
                             "system": system, **metrics})
        for row in cell["detector_metrics"]:
            component_per_cell.append({"model": cell["model"], "subset": cell["subset"],
                                       "component": "detector", **row})
        for row in cell["locator_metrics"]:
            component_per_cell.append({"model": cell["model"], "subset": cell["subset"],
                                       "component": "locator", **row})

    write_csv(os.path.join(args.out_dir, "development_detector_ranking.csv"), detector_summary)
    write_csv(os.path.join(args.out_dir, "development_locator_ranking.csv"), locator_summary)
    write_csv(os.path.join(args.out_dir, "component_metrics_per_cell.csv"), component_per_cell)
    write_csv(os.path.join(args.out_dir, "final_systems_per_cell.csv"), per_cell)
    with open(os.path.join(args.out_dir, "selection.json"), "w") as handle:
        json.dump({
            "selected_detector": selected_detector,
            "selected_locator": selected_locator,
            "development_cells": sorted([list(item) for item in DEV]),
            "lambdas": LAMBDAS, "k": K, "dufs_seeds": DUFS_SEEDS,
            "labels_used": "development selection, split-local threshold calibration, and evaluation only",
            "score_construction_uses_steps": False,
            "step_spans_used_only_for_evaluation_mapping": True,
        }, handle, indent=2)
    print(os.path.join(args.out_dir, "final_systems_per_cell.csv"))


if __name__ == "__main__":
    main()
