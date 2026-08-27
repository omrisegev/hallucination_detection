#!/usr/bin/env python3
"""Nested proof-of-concept for a pooled graph-roughness family direction.

This is a retrospective research instrument.  Graphs and their roughness
operators use no correctness labels.  Labels are used only by nested
leave-dataset-family-out hyperparameter selection and evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.harp_global_contribution_teacher import (  # noqa: E402
    DEFAULT_BUNDLE,
    load_original_cells,
)
from spectral_utils.family_residual_graph import graphs_from_coordinates  # noqa: E402
from spectral_utils.laplacian_upcr import (  # noqa: E402
    symmetric_normalized_laplacian,
)
from spectral_utils.specrage_views import VIEW_ORDER  # noqa: E402


VERSION = "pooled-graph-roughness-direction-poc-v1-2026-08-23"
DEFAULT_OUT = REPO / "results" / "pooled_graph_roughness_direction_poc_v1"
GRAPH_SETTINGS = (("union", 5), ("union", 7), ("union", 15), ("adaptive", 7))
CALIBRATION_LAMBDAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)
TRUST_FACTORS = (0.25, 0.5, 1.0, 1.5, 2.0)
EPS = 1e-12


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def graph_operator(cell, topology, k):
    residuals = np.asarray(cell["residuals"], dtype=float)
    presence = np.asarray(cell["presence"], dtype=bool)
    coordinates = residuals[:, presence]
    graph = graphs_from_coordinates(
        coordinates, (k,), topology=topology
    )[k]
    laplacian = symmetric_normalized_laplacian(graph)
    n = len(residuals)
    local_r = residuals[:, presence]
    roughness = np.asarray(local_r.T @ (laplacian @ local_r) / n)
    roughness = 0.5 * (roughness + roughness.T)
    cross = np.asarray(local_r.T @ (laplacian @ cell["baseline"]) / n)
    trace = float(np.trace(roughness))
    scale = float(np.sum(presence) / trace) if trace > EPS else 0.0
    aligned_a = np.zeros((len(VIEW_ORDER), len(VIEW_ORDER)), dtype=float)
    aligned_c = np.zeros(len(VIEW_ORDER), dtype=float)
    present = np.flatnonzero(presence)
    aligned_a[np.ix_(present, present)] = scale * roughness
    aligned_c[present] = scale * cross
    return {
        "A": aligned_a,
        "c": aligned_c,
        "presence": presence,
        "n_edges": int(graph.nnz // 2),
    }


def pool_direction(cells, operators, setting, calibration_lambda):
    a_sum = np.zeros((len(VIEW_ORDER), len(VIEW_ORDER)), dtype=float)
    a_count = np.zeros_like(a_sum, dtype=int)
    c_sum = np.zeros(len(VIEW_ORDER), dtype=float)
    c_count = np.zeros(len(VIEW_ORDER), dtype=int)
    for cell in cells:
        operator = operators[(cell["cell"], setting)]
        presence = operator["presence"]
        present = np.flatnonzero(presence)
        a_sum[np.ix_(present, present)] += operator["A"][
            np.ix_(present, present)
        ]
        a_count[np.ix_(present, present)] += 1
        c_sum[present] += operator["c"][present]
        c_count[present] += 1
    if np.any(c_count == 0) or np.any(a_count == 0):
        raise RuntimeError("calibration subset does not cover every family pair")
    pooled_a = a_sum / a_count
    pooled_a = 0.5 * (pooled_a + pooled_a.T)
    pooled_c = c_sum / c_count
    ridge = np.eye(len(VIEW_ORDER)) + float(calibration_lambda) * pooled_a
    direction = -float(calibration_lambda) * np.linalg.solve(ridge, pooled_c)
    return direction, pooled_a, pooled_c


def score_cell(cell, direction, trust_factor):
    presence = np.asarray(cell["presence"], dtype=bool)
    raw = np.asarray(cell["residuals"], dtype=float) @ direction
    scale = float(np.std(raw))
    if scale <= EPS or not np.any(direction[presence]):
        correction = np.zeros_like(raw)
    else:
        correction = (
            float(trust_factor) / int(np.sum(presence))
        ) * raw / scale
    return np.asarray(cell["baseline"], dtype=float) + correction


def candidate_key(setting, calibration_lambda, trust_factor):
    topology, k = setting
    return f"{topology}:k{k}:lambda{calibration_lambda:g}:trust{trust_factor:g}"


def evaluate_group(cells, direction, trust_factor):
    deltas = []
    for cell in cells:
        y = cell["correctness"]
        baseline = roc_auc_score(y, cell["baseline"])
        candidate = roc_auc_score(y, score_cell(cell, direction, trust_factor))
        deltas.append(candidate - baseline)
    return float(np.mean(deltas))


def fit_direction(source, operators, candidate):
    setting, calibration_lambda, _ = candidate
    return pool_direction(
        source, operators, setting, calibration_lambda
    )[0]


def cross_validated_candidate_values(cells, groups, operators, candidates):
    values = {candidate: {} for candidate in candidates}
    for held in groups:
        source = [cell for cell in cells if cell["group"] != held]
        target = [cell for cell in cells if cell["group"] == held]
        for candidate in candidates:
            direction = fit_direction(source, operators, candidate)
            values[candidate][held] = evaluate_group(
                target, direction, candidate[2]
            )
    return values


def choose_max_mean(values, groups):
    return max(
        values,
        key=lambda candidate: (
            np.mean([values[candidate][group] for group in groups]),
            -candidate[2],
            -candidate[1],
            -candidate[0][1],
            candidate[0][0],
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    cells = load_original_cells(args.bundle)
    groups = sorted({cell["group"] for cell in cells})
    operators = {}
    for index, cell in enumerate(cells, start=1):
        print(f"[{index}/{len(cells)}] operators {cell['cell']}", flush=True)
        for setting in GRAPH_SETTINGS:
            operators[(cell["cell"], setting)] = graph_operator(cell, *setting)
    candidates = tuple(
        (setting, calibration_lambda, trust_factor)
        for setting in GRAPH_SETTINGS
        for calibration_lambda in CALIBRATION_LAMBDAS
        for trust_factor in TRUST_FACTORS
    )

    outer_rows = []
    for held in groups:
        training = [group for group in groups if group != held]
        outer_source = [cell for cell in cells if cell["group"] != held]
        inner_values = cross_validated_candidate_values(
            outer_source, training, operators, candidates
        )
        selected = choose_max_mean(inner_values, training)
        direction = fit_direction(outer_source, operators, selected)
        target = [cell for cell in cells if cell["group"] == held]
        held_delta = evaluate_group(target, direction, selected[2])
        outer_rows.append({
            "held_group": held,
            "selected_key": candidate_key(*selected),
            "held_delta_pp": 100 * held_delta,
            "direction": direction.tolist(),
        })

    full_values = cross_validated_candidate_values(
        cells, groups, operators, candidates
    )
    final = choose_max_mean(full_values, groups)
    final_direction, pooled_a, pooled_c = pool_direction(
        cells, operators, final[0], final[1]
    )
    outer = np.asarray([row["held_delta_pp"] for row in outer_rows])
    nrm_direction = np.asarray(json.loads((
        REPO / "results" / "neutral_residual_mode_cs_iu_v1"
        / "FROZEN_CALIBRATION.json"
    ).read_text())["direction"], dtype=float)
    teacher_direction = np.asarray(json.loads((
        REPO / "results" / "harp_global_contribution_teacher_v1"
        / "RESULT.json"
    ).read_text())["source23_delta"], dtype=float)

    def cosine(left, right):
        return float(np.dot(left, right) / (
            np.linalg.norm(left) * np.linalg.norm(right)
        ))

    result = {
        "version": VERSION,
        "status": "retrospective_proof_of_concept",
        "n_cells": len(cells),
        "n_groups": len(groups),
        "candidate_count": len(candidates),
        "nested_delta_vs_iu_pp": float(np.mean(outer)),
        "nested_positive_groups": int(np.sum(outer > 0)),
        "nested_worst_group_pp": float(np.min(outer)),
        "outer_rows": outer_rows,
        "final_key": candidate_key(*final),
        "final_config": {
            "topology": final[0][0],
            "k": final[0][1],
            "calibration_lambda": final[1],
            "trust_factor": final[2],
        },
        "final_cross_validated_delta_pp": 100 * float(np.mean([
            full_values[final][group] for group in groups
        ])),
        "final_direction": final_direction.tolist(),
        "direction_cosine_nrm": cosine(final_direction, nrm_direction),
        "direction_cosine_supervised_teacher": cosine(
            final_direction, teacher_direction
        ),
        "pooled_A": pooled_a.tolist(),
        "pooled_c": pooled_c.tolist(),
        "uses_labels_for_direction": False,
        "uses_labels_for_hyperparameter_selection": True,
    }
    args.out.mkdir(parents=True)
    write_json(args.out / "RESULT.json", result)
    write_json(args.out / "RUN_DEFINITION.json", {
        "version": VERSION,
        "bundle": str(args.bundle.resolve()),
        "bundle_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        "graph_settings": GRAPH_SETTINGS,
        "calibration_lambdas": CALIBRATION_LAMBDAS,
        "trust_factors": TRUST_FACTORS,
        "retrospective": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
