#!/usr/bin/env python3
"""Focused tests for the frozen external-to-discovery manifold audit."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from scripts.build_supervised_conditional_manifold_external_cells_v1 import make_matrix
from scripts.validate_supervised_conditional_manifold_external_v2 import (
    PERMUTATIONS,
    decide,
    residualize_against_score,
)


FEATURES = ("epr", "cusum_max", "sw_var_peak")


def test_matrix_contract_is_label_independent_and_standardized() -> None:
    rows = [
        {"epr": 1.0 + index, "cusum_max": 3.0 * index, "sw_var_peak": index ** 2}
        for index in range(8)
    ]
    matrix = make_matrix(rows, FEATURES)
    assert matrix.shape == (8, 3)
    np.testing.assert_allclose(matrix.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(matrix.std(axis=0), 1.0, atol=1e-12)


def test_linear_residual_is_orthogonal_to_score() -> None:
    rng = np.random.default_rng(4)
    score = rng.normal(size=100)
    matrix = np.column_stack((2 * score + rng.normal(scale=.1, size=100), rng.normal(size=100)))
    residual = residualize_against_score(matrix, score)
    correlations = residual.T @ (score - score.mean())
    np.testing.assert_allclose(correlations, 0.0, atol=1e-9)


def _rows(*, family_count: int, geometry: bool, residual: bool, distinct: bool) -> list[dict]:
    rows = []
    for family_index in range(family_count):
        family = f"family{family_index}"
        cell = f"{family}_cell"
        for seed in (101, 211, 307):
            base = {
                "cell": cell, "dataset_family": family, "model_family": f"model{family_index}",
                "dataset_new": True, "model_new": family_index == 0, "tie_seed": seed,
                "graph_eligible": True, "exact_eligible": True, "crt_eligible": True,
                "selected_k": 3, "largest_component_fraction": 1.0, "isolated_fraction": 0.0,
                "n": 200, "hallucination_rate": .5,
            }
            rows.append({
                **base, "graph_role": "metric_graph",
                "exact_effect": .12 if geometry else .0, "crt_effect": .11 if geometry else .0,
                "min_conditional_effect": .11 if geometry else .0,
                "exact_p_holm": .01 if geometry else 1.0, "crt_p_holm": .01 if geometry else 1.0,
                "advantage_vs_linear_min": .04 if distinct else .0,
                "advantage_vs_linear_exact_p_holm": .01 if distinct else 1.0,
                "advantage_vs_linear_crt_p_holm": .01 if distinct else 1.0,
                "advantage_vs_equal_min": -.01, "liu_delta_auroc": .006,
            })
            rows.append({
                **base, "graph_role": "linear_residual_graph",
                "exact_effect": .08 if residual else .0, "crt_effect": .07 if residual else .0,
                "min_conditional_effect": .07 if residual else .0,
                "exact_p_holm": .01 if residual else 1.0, "crt_p_holm": .01 if residual else 1.0,
                "liu_delta_auroc": float("nan"),
            })
    return rows


def test_coverage_fails_closed_below_three_families() -> None:
    decision, _ = decide(
        _rows(family_count=2, geometry=True, residual=True, distinct=True),
        {"minimum_independent_dataset_families": 3,
         "claim_status": "retrospective_external_to_discovery_not_prospective_confirmation"},
    )
    assert decision["decision"] == "INSUFFICIENT_EXTERNAL_COVERAGE"


def test_distinct_decision_requires_residual_geometry() -> None:
    decision, _ = decide(
        _rows(family_count=3, geometry=True, residual=False, distinct=True),
        {"minimum_independent_dataset_families": 3,
         "claim_status": "retrospective_external_to_discovery_not_prospective_confirmation"},
    )
    assert decision["decision"] == "RETROSPECTIVE_EXTERNAL_SHARED_DIRECTION_ONLY"
    assert not decision["distinct_vs_linear_pass"]


def test_conditional_null_ineligibility_does_not_become_transfer_failure() -> None:
    rows = _rows(family_count=3, geometry=False, residual=False, distinct=False)
    for row in rows:
        row["exact_eligible"] = False
    decision, _ = decide(
        rows,
        {"minimum_independent_dataset_families": 3,
         "claim_status": "retrospective_external_to_discovery_not_prospective_confirmation"},
    )
    assert decision["decision"] == "CONDITIONAL_NULL_INELIGIBILITY_INVALIDATES_EXTERNAL_AUDIT"


def test_permutation_budget_is_frozen() -> None:
    assert PERMUTATIONS == 999


def main() -> None:
    test_matrix_contract_is_label_independent_and_standardized()
    test_linear_residual_is_orthogonal_to_score()
    test_coverage_fails_closed_below_three_families()
    test_distinct_decision_requires_residual_geometry()
    test_conditional_null_ineligibility_does_not_become_transfer_failure()
    test_permutation_budget_is_frozen()
    print("external manifold v2 focused tests: PASS")


if __name__ == "__main__":
    main()
