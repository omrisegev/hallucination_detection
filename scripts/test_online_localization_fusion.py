#!/usr/bin/env python3
"""Known-answer tests for the causal GL-LIU online adapter."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spectral_utils.online_localization_fusion import (  # noqa: E402
    causal_local_core_matrix,
    causal_trace_features,
    fit_frozen_online_gl_liu,
)


def synthetic_row(seed: int, length: int, label: int, group: str) -> dict:
    rng = np.random.default_rng(seed)
    entropy = rng.uniform(0.1, 2.0, length)
    spilled = rng.uniform(0.05, 2.5, length)
    logsumexp = rng.uniform(2.0, 7.0, length)
    logprobs = rng.normal(-2.0, 0.5, size=(length, 10))
    logprobs = -np.sort(-logprobs, axis=1)
    return {
        "token_entropies": entropy,
        "token_spilled_energies": spilled,
        "token_logsumexp": logsumexp,
        "top_k_logprobs": {
            "ids": np.tile(np.arange(10), (length, 1)),
            "logprobs": logprobs,
        },
        "label": int(label),
        "_group": group,
        "_trace_id": group,
        "_length": int(length),
    }


def same_prefix_pair() -> tuple[dict, dict]:
    left = synthetic_row(100, 96, 1, "same")
    right = synthetic_row(200, 150, 1, "same")
    for name in ("token_entropies", "token_spilled_energies", "token_logsumexp"):
        right[name][:64] = left[name][:64]
    for name in ("ids", "logprobs"):
        right["top_k_logprobs"][name][:64] = left["top_k_logprobs"][name][:64]
    return left, right


def test_raw_prefix_suffix_invariance() -> None:
    left, right = same_prefix_pair()
    left_features = causal_trace_features(left, 64)
    right_features = causal_trace_features(right, 64)
    assert left_features.keys() == right_features.keys()
    for name in left_features:
        assert np.isclose(left_features[name], right_features[name], equal_nan=True), name
    left_local, left_names = causal_local_core_matrix(left, 64)
    right_local, right_names = causal_local_core_matrix(right, 64)
    assert left_names == right_names
    assert np.array_equal(left_local, right_local, equal_nan=True)


def test_frozen_ensemble_and_label_blind_fit() -> None:
    rows = [synthetic_row(i, 72 + 3 * i, i % 2, f"q{i}") for i in range(10)]
    model = fit_frozen_online_gl_liu(
        rows, max_fit_tokens=700, dufs_epochs=1
    )
    left, right = same_prefix_pair()
    left_scores = model.scores(left, 64)
    right_scores = model.scores(right, 64)
    required = {
        "global_gl_liu_no_length",
        "global_gl_liu_elapsed_length",
        "local_temporal_gl_liu_max",
        "local_dufs_gl_liu_top5",
        "fused_gl_liu",
        "cusum_max",
        "sw_var_peak",
        "cusum_swvar_equal",
    }
    assert required <= set(left_scores)
    for name in required:
        assert np.isclose(left_scores[name], right_scores[name]), name

    relabelled = deepcopy(rows)
    for row in relabelled:
        row["label"] = 1 - row["label"]
    other = fit_frozen_online_gl_liu(
        relabelled, max_fit_tokens=700, dufs_epochs=1
    )
    other_scores = other.scores(left, 64)
    for name in required:
        assert np.isclose(left_scores[name], other_scores[name]), name


def main() -> None:
    test_raw_prefix_suffix_invariance()
    test_frozen_ensemble_and_label_blind_fit()
    print("test_online_localization_fusion: PASS")


if __name__ == "__main__":
    main()
