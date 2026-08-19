#!/usr/bin/env python3
"""Known-answer tests for the existing-cache online convergence protocol."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spectral_utils.online_convergence import (  # noqa: E402
    apply_declaration_policy,
    causal_raw_prefix_matrix,
    declaration_summary,
    fit_frozen_prefix_iu,
    grouped_calibration_split,
    normalize_cache_records,
)


def synthetic_row(seed: int, length: int, label: int, group: str) -> dict:
    rng = np.random.default_rng(seed)
    entropy = rng.uniform(0.1, 1.8, length)
    spilled = rng.uniform(0.05, 2.2, length)
    logsumexp = rng.uniform(2.0, 6.0, length)
    logprobs = rng.normal(-2.0, 0.4, size=(length, 8))
    logprobs = -np.sort(-logprobs, axis=1)
    return {
        "token_entropies": entropy,
        "token_spilled_energies": spilled,
        "token_logsumexp": logsumexp,
        "top_k_logprobs": {
            "ids": np.tile(np.arange(8), (length, 1)),
            "logprobs": logprobs,
        },
        "label": int(label),
        "question_id": group,
    }


def test_prefix_suffix_invariance() -> None:
    prefix = synthetic_row(1, 64, 1, "a")
    longer = synthetic_row(2, 120, 1, "a")
    for key in ("token_entropies", "token_spilled_energies", "token_logsumexp"):
        longer[key][:64] = prefix[key]
    for key in ("ids", "logprobs"):
        longer["top_k_logprobs"][key][:64] = prefix["top_k_logprobs"][key]
    for include_length in (False, True):
        left, names_left = causal_raw_prefix_matrix(
            prefix, 64, include_elapsed_length=include_length
        )
        right, names_right = causal_raw_prefix_matrix(
            longer, 64, include_elapsed_length=include_length
        )
        assert names_left == names_right
        assert np.array_equal(left, right, equal_nan=True)


def test_model_is_frozen_across_suffixes() -> None:
    records = [synthetic_row(i, 80 + i, i % 2, f"q{i}") for i in range(8)]
    normalized = normalize_cache_records(records)
    model = fit_frozen_prefix_iu(
        normalized[:6], include_elapsed_length=False, rows_per_trace=8
    )
    base = normalized[6]
    extended = synthetic_row(99, 160, base["label"], "q6")
    for key in ("token_entropies", "token_spilled_energies", "token_logsumexp"):
        extended[key][:64] = np.asarray(base[key])[:64]
    for key in ("ids", "logprobs"):
        extended["top_k_logprobs"][key][:64] = np.asarray(
            base["top_k_logprobs"][key]
        )[:64]
    left, _ = causal_raw_prefix_matrix(base, 64, include_elapsed_length=False)
    right, _ = causal_raw_prefix_matrix(extended, 64, include_elapsed_length=False)
    assert np.allclose(model.risk(left), model.risk(right), equal_nan=True)


def test_group_split() -> None:
    records = [synthetic_row(i, 80, i % 2, f"q{i // 2}") for i in range(20)]
    normalized = normalize_cache_records(records)
    calibration, evaluation, _ = grouped_calibration_split(normalized, seed=11)
    left = {normalized[index]["_group"] for index in calibration}
    right = {normalized[index]["_group"] for index in evaluation}
    assert left.isdisjoint(right)
    assert {1 - normalized[index]["label"] for index in calibration} == {0, 1}
    assert {1 - normalized[index]["label"] for index in evaluation} == {0, 1}


def test_answer_label_precedence() -> None:
    row = synthetic_row(4, 80, 0, "p")
    row["label"] = 0
    row["final_answer_correct"] = True
    normalized = normalize_cache_records([row])
    assert normalized[0]["label"] == 1


def test_two_observation_declaration() -> None:
    rows = []
    for unit, (truth, scores) in enumerate(((1, [0.8, 0.9]), (0, [0.2, 0.1]))):
        for budget, score in zip((16, 32), scores):
            rows.append({
                "unit_index": unit,
                "group": f"q{unit}",
                "label_error": truth,
                "trace_length": 100,
                "budget": budget,
                "is_final": False,
                "method": "m",
                "score": score,
            })
        rows.append({
            "unit_index": unit,
            "group": f"q{unit}",
            "label_error": truth,
            "trace_length": 100,
            "budget": 100,
            "is_final": True,
            "method": "m",
            "score": scores[-1],
        })
    declared = apply_declaration_policy(rows, "m", low=0.3, high=0.7)
    summary = declaration_summary(declared)
    assert summary["coverage"] == 1.0
    assert summary["ever_wrong_rate_all"] == 0.0
    assert all(row["declaration_budget"] == 32 for row in declared)


def main() -> None:
    test_prefix_suffix_invariance()
    test_model_is_frozen_across_suffixes()
    test_group_split()
    test_answer_label_precedence()
    test_two_observation_declaration()
    print("test_online_convergence: PASS")


if __name__ == "__main__":
    main()
