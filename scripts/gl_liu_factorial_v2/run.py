#!/usr/bin/env python3
"""Run the preregistered GL-LIU two-axis factorial diagnostic.

The run contains two clean 2x2 comparisons:

1. global IU-PCR vs global DUFS-LIU, crossed with temporal-LIU vs
   DUFS-LIU localization on the same five token views;
2. the same global comparison, crossed with five-view vs broad-view local
   DUFS-LIU.

All score constructors are label-free.  Labels are opened only after every
detector and locator has been fitted, applied, and hashed.  Labels are then used
for component evaluation and the already-declared split-local F1 threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.gl_liu_v1.run import (  # noqa: E402
    DUFS_EPOCHS,
    DUFS_SEEDS,
    K,
    MAX_GRAPH_TOKENS,
    POSITIONAL_FIT,
    _temporal_graph,
    answer_detectors,
    load_rows,
    mindgap_control,
    token_to_step,
)
from scripts.gl_liu_v1.two_stage_localization import evaluate_two_stage  # noqa: E402
from spectral_utils.adapted_dufs import adapted_dufs_soft_gates  # noqa: E402
from spectral_utils.laplacian_upcr import (  # noqa: E402
    build_graph_from_features,
    laplacian_iu_path,
)
from spectral_utils.streaming_utils import anchor_orient  # noqa: E402
from spectral_utils.token_feature_views import (  # noqa: E402
    BROAD_TOKEN_VIEWS,
    CORE_TOKEN_VIEWS,
    TOKEN_TO_GLOBAL_FEATURES,
    build_token_channels,
)
from spectral_utils.upcr import upcr_fit  # noqa: E402


DEV = {("qwen3_4b", "gsm8k"), ("qwen3_4b", "math")}
MODELS = ("qwen3_4b", "qwen3_8b")
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
GLOBAL_METHODS = {
    "global_iu": "answer_iu_mixed",
    "global_dufs": "answer_dufs_liu_mixed",
}
LOCAL_METHODS = (
    "local_temporal_core",
    "local_dufs_core",
    "local_dufs_broad",
)
LOCAL_LAMBDA = 0.3
N_SPLITS = 100
SPLIT_SEED = 0
FIT_ROW_SEED = 0
MAX_FIT_POPULATION_TOKENS = 200_000


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _score_hash(values):
    if isinstance(values, np.ndarray):
        packed = np.asarray(values, dtype="<f8")
    else:
        packed = np.concatenate([np.asarray(row, dtype=float) for row in values]).astype("<f8")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _fit_row_order(channels):
    rng = np.random.default_rng(FIT_ROW_SEED)
    n_rows = len(next(iter(channels.values())))
    chosen, total = [], 0
    for index in rng.permutation(n_rows):
        chosen.append(int(index))
        total += len(channels["entropy_series"][int(index)])
        if total >= MAX_FIT_POPULATION_TOKENS:
            break
    return chosen


def _prepare_token_matrix(channels, row_indices, names):
    chunks, lengths, total = [], [], 0
    for index in row_indices:
        length = len(channels[names[0]][index])
        if length <= 0:
            continue
        take = min(length, MAX_GRAPH_TOKENS - total)
        if take <= 0:
            break
        chunks.append((int(index), int(take)))
        lengths.append(int(take))
        total += int(take)
        if total >= MAX_GRAPH_TOKENS:
            break

    kept, means, scales, standardized = [], [], [], []
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
        if scale < 1e-8 or float(np.mean(values == np.median(values))) > 0.95:
            continue
        kept.append(name)
        means.append(float(values.mean()))
        scales.append(scale)
        standardized.append((values - values.mean()) / scale)

    raw_standardized = np.column_stack(standardized)
    first = upcr_fit(raw_standardized.T, **POSITIONAL_FIT)
    derived = np.sign(first.rho_hat_full)
    derived[derived == 0] = 1.0
    feature_matrix = (raw_standardized * derived).T
    return {
        "F": feature_matrix,
        "V": raw_standardized,
        "names": kept,
        "mu": np.asarray(means),
        "sd": np.asarray(scales),
        "derived": np.asarray(derived),
        "chunks": chunks,
        "lengths": lengths,
    }


def _effective_rank(matrix):
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    weights = singular ** 2
    return float((weights.sum() ** 2) / (np.sum(weights ** 2) + 1e-12))


def _fit_one_arm(prepared, graph_kind):
    F = prepared["F"]
    gate_diag = None
    if graph_kind == "temporal":
        graph = _temporal_graph(prepared["lengths"])
    elif graph_kind == "dufs":
        gates, gate_diag = adapted_dufs_soft_gates(
            F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
        )
        graph = build_graph_from_features(F, gates=gates, k=K)
    else:
        raise ValueError(graph_kind)

    fit = laplacian_iu_path(F, (LOCAL_LAMBDA,), graph=graph)[LOCAL_LAMBDA]
    anchor_index = prepared["names"].index("entropy_series")
    anchor = prepared["V"][:, anchor_index]
    _, flipped = anchor_orient(fit.w @ F, anchor)
    arm = {
        "names": prepared["names"],
        "mu": prepared["mu"],
        "sd": prepared["sd"],
        "derived": prepared["derived"],
        "weights": np.asarray(fit.w),
        "flipped": bool(flipped),
    }
    diagnostics = {
        "graph_kind": graph_kind,
        "n_features": len(prepared["names"]),
        "n_fit_tokens": int(F.shape[1]),
        "feature_names": prepared["names"],
        "feature_effective_rank": _effective_rank(F),
        "laplacian": fit.diagnostics,
    }
    if gate_diag is not None:
        raw = np.asarray(gate_diag["raw_probabilities"], dtype=float)
        diagnostics["dufs_gate"] = {
            **gate_diag,
            "ranked_features": [
                {"feature": prepared["names"][int(index)], "probability": float(raw[index])}
                for index in np.argsort(-raw)
            ],
        }
    return arm, diagnostics


def _apply_arm(arm, channels):
    output = []
    n_rows = len(next(iter(channels.values())))
    for row_index in range(n_rows):
        columns = []
        for j, name in enumerate(arm["names"]):
            values = np.asarray(channels[name][row_index], dtype=float)
            values = np.where(np.isfinite(values), values, arm["mu"][j])
            columns.append(
                (values - arm["mu"][j]) / arm["sd"][j] * arm["derived"][j]
            )
        score = arm["weights"] @ np.vstack(columns)
        if arm["flipped"]:
            score = -score
        output.append(np.asarray(score, dtype=float))
    return output


def _curve_rank_displacement(left, right):
    values = []
    for a, b in zip(left, right):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if len(a) < 2:
            continue
        ra, rb = rankdata(a), rankdata(b)
        values.append(float(np.mean(np.abs(ra - rb)) / len(a)))
    return float(np.mean(values)) if values else float("nan")


def fit_local_scores(rows):
    channels = build_token_channels(rows)
    fit_rows = _fit_row_order(channels)
    core = _prepare_token_matrix(channels, fit_rows, CORE_TOKEN_VIEWS)
    broad = _prepare_token_matrix(channels, fit_rows, BROAD_TOKEN_VIEWS)

    arms, diagnostics = {}, {
        "fit_rows": fit_rows,
        "contracts": {
            "core": list(CORE_TOKEN_VIEWS),
            "broad_requested": list(BROAD_TOKEN_VIEWS),
            "token_to_global_features": TOKEN_TO_GLOBAL_FEATURES,
        },
    }
    for name, prepared, graph_kind in (
        ("local_temporal_core", core, "temporal"),
        ("local_dufs_core", core, "dufs"),
        ("local_dufs_broad", broad, "dufs"),
    ):
        arms[name], diagnostics[name] = _fit_one_arm(prepared, graph_kind)

    curves = {name: _apply_arm(arm, channels) for name, arm in arms.items()}
    locators = {
        name: np.asarray([
            token_to_step(curve, row) for curve, row in zip(curves[name], rows)
        ], dtype=int)
        for name in curves
    }
    diagnostics["rank_displacement"] = {
        "temporal_vs_dufs_core": _curve_rank_displacement(
            curves["local_temporal_core"], curves["local_dufs_core"]
        ),
        "dufs_core_vs_dufs_broad": _curve_rank_displacement(
            curves["local_dufs_core"], curves["local_dufs_broad"]
        ),
    }
    return curves, locators, diagnostics


def _component_metrics(detectors, locators, labels):
    error = labels != -1
    detector_rows = []
    for name, values in detectors.items():
        detector_rows.append({
            "component": "detector",
            "candidate": name,
            "auroc": float(roc_auc_score(error, values)),
            "auprc": float(average_precision_score(error, values)),
        })
    locator_rows = []
    for name, values in locators.items():
        locator_rows.append({
            "component": "locator",
            "candidate": name,
            "exact": float(np.mean(values[error] == labels[error])),
            "tol1": float(np.mean(np.abs(values[error] - labels[error]) <= 1)),
        })
    return detector_rows + locator_rows


def evaluate_cell(model, subset, path):
    rows = load_rows(path)
    answer, answer_diag = answer_detectors(rows)
    detectors = {alias: np.asarray(answer[source], dtype=float)
                 for alias, source in GLOBAL_METHODS.items()}
    curves, locators, local_diag = fit_local_scores(rows)
    paper_detector, paper_locator = mindgap_control(rows)

    hashes = {
        "detectors": {name: _score_hash(score) for name, score in detectors.items()},
        "locators": {name: _score_hash(score) for name, score in locators.items()},
        "token_curves": {name: _score_hash(score) for name, score in curves.items()},
        "mindgap_detector": _score_hash(paper_detector),
        "mindgap_locator": _score_hash(paper_locator),
    }

    # Evaluation boundary: labels are not read above this line.
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    component_rows = _component_metrics(detectors, locators, labels)
    system_rows = []
    for global_name, risk in detectors.items():
        for local_name, locator in locators.items():
            system_rows.append({
                "system": f"{global_name}__{local_name}",
                "global": global_name,
                "local": local_name,
                **evaluate_two_stage(
                    risk, locator, labels, n_splits=N_SPLITS, seed=SPLIT_SEED
                ),
            })
    system_rows.append({
        "system": "mindgap_control",
        "global": "mindgap",
        "local": "mindgap",
        **evaluate_two_stage(
            paper_detector, paper_locator, labels,
            n_splits=N_SPLITS, seed=SPLIT_SEED,
        ),
    })
    return {
        "model": model,
        "subset": subset,
        "n_rows": len(rows),
        "component_rows": component_rows,
        "system_rows": system_rows,
        "diagnostics": {
            "labels_seen_during_score_fit": False,
            "scores_hashed_before_labels": True,
            "hashes_before_labels": hashes,
            "global": answer_diag,
            "local": local_diag,
        },
    }


def _aggregate(records, group_name, included):
    selected = [row for row in records if included(row)]
    systems = sorted({row["system"] for row in selected})
    output = []
    for system in systems:
        rows = [row for row in selected if row["system"] == system]
        record = {
            "group": group_name,
            "system": system,
            "global": rows[0]["global"],
            "local": rows[0]["local"],
            "n_cells": len(rows),
        }
        for metric in ("f1", "acc_erroneous", "acc_correct", "sla", "sla_tol1"):
            values = np.asarray([row[metric] for row in rows], dtype=float)
            record[metric] = float(np.mean(values))
            record[metric + "_cell_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        output.append(record)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    diagnostics_dir = out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    system_rows, component_rows = [], []
    for model in MODELS:
        folder = Path(args.cache_root) / ("pb_" + model)
        for subset in SUBSETS:
            print(f"FIT {model} {subset}", flush=True)
            cell = evaluate_cell(model, subset, folder / f"processbench_{subset}.pkl")
            split = "development" if (model, subset) in DEV else "nonselection"
            prefix = {"model": model, "subset": subset, "split": split}
            system_rows.extend([{**prefix, **row} for row in cell["system_rows"]])
            component_rows.extend([{**prefix, **row} for row in cell["component_rows"]])
            with (diagnostics_dir / f"{model}__{subset}.json").open("w") as handle:
                json.dump(_jsonable(cell["diagnostics"]), handle, indent=2)

    macro_rows = []
    macro_rows.extend(_aggregate(system_rows, "all_8_cells", lambda row: True))
    macro_rows.extend(_aggregate(system_rows, "development_2_cells", lambda row: row["split"] == "development"))
    macro_rows.extend(_aggregate(system_rows, "nonselection_6_cells", lambda row: row["split"] == "nonselection"))

    _write_csv(out_dir / "systems_per_cell.csv", system_rows)
    _write_csv(out_dir / "components_per_cell.csv", component_rows)
    _write_csv(out_dir / "system_macros.csv", macro_rows)

    run_definition = {
        "version": "gl-liu-factorial-v2-2026-08-08",
        "purpose": "test unified DUFS-LIU and broad token-resolved feature counterparts",
        "factorial_1": {
            "rows": ["global_iu", "global_dufs"],
            "columns": ["local_temporal_core", "local_dufs_core"],
            "question": "does using DUFS-LIU in both heads help when the local feature pool is fixed?",
        },
        "factorial_2": {
            "rows": ["global_iu", "global_dufs"],
            "columns": ["local_dufs_core", "local_dufs_broad"],
            "question": "does the broad token-resolved pool help when the local graph is fixed to DUFS-LIU?",
        },
        "fixed_parameters": {
            "global_lambda": 0.1,
            "local_lambda": LOCAL_LAMBDA,
            "k": K,
            "dufs_seeds": list(DUFS_SEEDS),
            "dufs_epochs": DUFS_EPOCHS,
            "max_graph_tokens": MAX_GRAPH_TOKENS,
            "n_threshold_splits": N_SPLITS,
            "threshold_split_seed": SPLIT_SEED,
        },
        "feature_contract": {
            "core": list(CORE_TOKEN_VIEWS),
            "broad": list(BROAD_TOKEN_VIEWS),
            "broad_count": len(BROAD_TOKEN_VIEWS),
            "mapping": TOKEN_TO_GLOBAL_FEATURES,
            "omitted": {
                "trace_length": "constant within each trace and cannot change a token argmax",
                "cusum_shift_idx_duplicate": "shares the cusum_max absolute-CUSUM curve and is represented once",
            },
        },
        "label_policy": {
            "score_fit": "no labels",
            "score_hash": "before labels are read",
            "threshold_calibration": "labels in calibration half only",
            "evaluation": "labels after scores freeze",
            "candidate_selection": "none; every preregistered cell of both matrices is reported",
        },
    }
    with (out_dir / "RUN_DEFINITION.json").open("w") as handle:
        json.dump(_jsonable(run_definition), handle, indent=2)
    print(out_dir / "system_macros.csv", flush=True)


if __name__ == "__main__":
    main()
