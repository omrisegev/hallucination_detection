#!/usr/bin/env python3
"""Mechanical tests for comprehensive Local/Online feature construction."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from spectral_utils.local_online_comprehensive import (
    BROAD_FAMILIES,
    FAMILY_NAMES,
    RAW7_NAMES,
    causal_operator_matrices,
    fit_references,
    fit_trajectory_head,
    local_candidate_roster,
    online_candidate_roster,
    representation_matrix,
)
from spectral_utils.multitask_trajectory import CHANNEL_NAMES
from spectral_utils.token_feature_views import BROAD_TOKEN_VIEWS


def _row(seed: int, n: int = 96) -> dict:
    rng = np.random.default_rng(seed)
    logits = np.sort(rng.normal(-3.0, 1.0, size=(n, 20)), axis=1)[:, ::-1]
    return {
        "id": f"fixture-{seed}",
        "token_entropies": rng.gamma(1.5, 0.4, n),
        "token_spilled_energies": rng.gamma(1.2, 0.5, n),
        "token_logsumexp": rng.normal(25.0, 2.0, n),
        "top_k_logprobs": {
            "ids": np.tile(np.arange(20), (n, 1)),
            "logprobs": logits,
        },
        "label": int(seed % 2),
        "final_answer_correct": bool(seed % 2),
        "step_token_spans": [(0, 32), (32, 64), (64, n)],
    }


def _truncate(row: dict, n: int) -> dict:
    out = deepcopy(row)
    for key in ("token_entropies", "token_spilled_energies", "token_logsumexp"):
        out[key] = np.asarray(out[key])[:n]
    out["top_k_logprobs"] = {
        key: np.asarray(value)[:n] for key, value in out["top_k_logprobs"].items()
    }
    return out


def test_rosters_and_families() -> None:
    flattened = [name for family in FAMILY_NAMES for name in BROAD_FAMILIES[family]]
    assert len(flattened) == len(set(flattened)) == len(BROAD_TOKEN_VIEWS)
    assert set(flattened) == set(BROAD_TOKEN_VIEWS)
    assert len(CHANNEL_NAMES) == 9 and len(RAW7_NAMES) == 7
    assert len(local_candidate_roster()) == 19
    assert len(online_candidate_roster()) == 15


def test_shapes_and_suffix_replacement() -> None:
    rows = [_row(seed) for seed in range(6)]
    refs = fit_references(rows[:4])
    original = rows[4]
    changed = deepcopy(original)
    rng = np.random.default_rng(100)
    changed["token_entropies"][48:] = rng.normal(10.0, 1.0, 48)
    changed["token_spilled_energies"][48:] = rng.normal(10.0, 1.0, 48)
    changed["token_logsumexp"][48:] = rng.normal(-20.0, 1.0, 48)
    changed["top_k_logprobs"]["logprobs"][48:] = rng.normal(-20.0, 1.0, (48, 20))
    for representation, width in (("raw9", 9), ("raw7", 7), ("broad28", 28), ("family6", 6)):
        matrix, names = representation_matrix(original, refs, representation)
        assert matrix.shape == (96, width)
        assert len(names) == width
        left, _ = representation_matrix(_truncate(original, 48), refs, representation)
        right, _ = representation_matrix(_truncate(changed, 48), refs, representation)
        np.testing.assert_array_equal(left, right)


def test_operator_contract() -> None:
    rng = np.random.default_rng(12)
    level = rng.normal(size=(80, 6))
    states = causal_operator_matrices(level)
    assert set(states) == {
        "level", "fast", "slow", "innovation", "shortlong",
        "positive_mean", "persistence", "recovery",
    }
    assert all(value.shape == level.shape for value in states.values())
    assert np.all(states["innovation"] >= 0.0)
    assert np.all(states["persistence"] >= 0.0)
    assert np.all(states["persistence"] <= 1.0)
    assert np.all(states["recovery"] <= 1e-12)
    boosted = level.copy()
    boosted[40, 2] += 3.0
    boosted_states = causal_operator_matrices(boosted)
    assert boosted_states["level"][40, 2] > states["level"][40, 2]
    assert boosted_states["innovation"][40, 2] >= states["innovation"][40, 2]
    assert boosted_states["positive_mean"][40, 2] >= states["positive_mean"][40, 2]


def test_label_blind_repeat_and_order() -> None:
    rows = [_row(seed) for seed in range(8)]
    refs = fit_references(rows[:5])
    head1 = fit_trajectory_head(
        rows[:5], refs, name="fixture", representation="family6",
        operators=("level", "innovation", "shortlong"),
    )
    stripped = [
        {key: value for key, value in row.items() if key not in {"label", "final_answer_correct", "step_token_spans"}}
        for row in rows[:5]
    ]
    reversed_rows = [{key: row[key] for key in reversed(tuple(row))} for row in stripped]
    refs2 = fit_references(reversed_rows)
    head2 = fit_trajectory_head(
        reversed_rows, refs2, name="fixture", representation="family6",
        operators=("level", "innovation", "shortlong"),
    )
    np.testing.assert_array_equal(head1.weights, head2.weights)
    np.testing.assert_array_equal(head1.keep, head2.keep)
    test1 = head1.curve(rows[6], refs)
    test2 = head2.curve({key: rows[6][key] for key in reversed(tuple(rows[6]))}, refs2)
    np.testing.assert_array_equal(test1, test2)


def main() -> None:
    tests = [
        test_rosters_and_families,
        test_shapes_and_suffix_replacement,
        test_operator_contract,
        test_label_blind_repeat_and_order,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all ({len(tests)} tests)")


if __name__ == "__main__":
    main()
