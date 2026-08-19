#!/usr/bin/env python3
"""Deterministic causal/no-label tests for GLOBAL_LOCAL_ONLINE_IU_V1."""

from __future__ import annotations

import copy
import random

import numpy as np

from spectral_utils.global_local_online import (
    CANDIDATE_FEATURES,
    fit_dynamic_online_head,
)
from spectral_utils.laplacian_upcr import (
    build_graph_from_features,
    laplacian_iu_path,
)


def synthetic_rows(n_traces: int = 24) -> list[dict]:
    rows = []
    budgets = (16, 32, 64, 128)
    for unit in range(n_traces):
        phase = (unit % 7) / 7.0
        length = 180 + unit
        label = unit % 2
        for j, budget in enumerate(budgets):
            cusum = 0.4 * phase + 0.07 * j + 0.03 * np.sin(unit + j)
            swvar = 0.3 * (1.0 - phase) + 0.05 * j + 0.02 * np.cos(unit - j)
            for method, score in (("cusum_max", cusum), ("sw_var_peak", swvar)):
                rows.append({
                    "unit_index": unit,
                    "trace_id": f"trace-{unit}",
                    "group": f"question-{unit // 2}",
                    "label_error": label,
                    "trace_length": length,
                    "budget": budget,
                    "is_final": False,
                    "method": method,
                    "score": float(score),
                })
        for method, score in (
            ("cusum_max", 0.4 * phase + 0.38 + 0.02 * np.sin(unit)),
            ("sw_var_peak", 0.3 * (1.0 - phase) + 0.28 + 0.02 * np.cos(unit)),
        ):
            rows.append({
                "unit_index": unit,
                "trace_id": f"trace-{unit}",
                "group": f"question-{unit // 2}",
                "label_error": label,
                "trace_length": length,
                "budget": length,
                "is_final": True,
                "method": method,
                "score": float(score),
            })
    return rows


def parameter_vector(model) -> np.ndarray:
    return np.concatenate([
        np.asarray([model.component_centres[name] for name in sorted(model.component_centres)]),
        np.asarray([model.component_scales[name] for name in sorted(model.component_scales)]),
        model.feature_keep.astype(float),
        model.feature_mean,
        model.feature_std,
        model.weights,
    ])


def keyed_scores(rows: list[dict]) -> dict[tuple, float]:
    return {
        (row["group"], row["unit_index"], row["budget"], row["is_final"]): row["score"]
        for row in rows
    }


def test_order_repeat_and_no_label() -> None:
    source = synthetic_rows()
    shuffled = source.copy()
    random.Random(123).shuffle(shuffled)
    relabeled = copy.deepcopy(source)
    for row in relabeled:
        row["label_error"] = 1 - int(row["label_error"])
    for candidate in CANDIDATE_FEATURES:
        first = fit_dynamic_online_head(source, candidate)
        repeat = fit_dynamic_online_head(source, candidate)
        reordered = fit_dynamic_online_head(shuffled, candidate)
        no_label = fit_dynamic_online_head(relabeled, candidate)
        assert np.array_equal(parameter_vector(first), parameter_vector(repeat))
        assert np.array_equal(parameter_vector(first), parameter_vector(reordered))
        assert np.array_equal(parameter_vector(first), parameter_vector(no_label))
        assert first.diagnostics["labels_seen_during_fit"] is False
        assert keyed_scores(first.score_rows(source)) == keyed_scores(first.score_rows(shuffled))


def test_suffix_invariance_and_missing_component() -> None:
    fit_rows = synthetic_rows()
    model = fit_dynamic_online_head(fit_rows, "dyn_persist6_iu")
    left = synthetic_rows(4)
    right = copy.deepcopy(left)
    for row in right:
        if int(row["budget"]) > 64:
            row["score"] = float(row["score"]) * -19.0 + 7.0
    left_scores = keyed_scores(model.score_rows(left))
    right_scores = keyed_scores(model.score_rows(right))
    for key, value in left_scores.items():
        if int(key[2]) <= 64 and not bool(key[3]):
            assert value == right_scores[key]

    missing = [
        row for row in left
        if not (
            row["method"] == "sw_var_peak"
            and row["unit_index"] == 0
            and row["budget"] == 32
        )
    ]
    scored = model.score_rows(missing)
    assert len(scored) == len({
        (row["unit_index"], row["budget"], row["is_final"])
        for row in missing if row["method"] == "cusum_max"
    })
    assert np.isfinite([row["score"] for row in scored]).all()


def test_exact_lambda_zero() -> None:
    rng = np.random.default_rng(20260816)
    F = rng.normal(size=(6, 80))
    graph = build_graph_from_features(F, k=7)
    path = laplacian_iu_path(F, (0.0, 0.1), graph=graph)
    assert np.array_equal(path[0.0].w, path[0.0].baseline.w)
    assert np.array_equal(path[0.0].w @ F, path[0.0].baseline.w @ F)


def main() -> None:
    test_order_repeat_and_no_label()
    test_suffix_invariance_and_missing_component()
    test_exact_lambda_zero()
    print("global-local-online tests: PASS")


if __name__ == "__main__":
    main()
