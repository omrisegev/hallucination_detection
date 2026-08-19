#!/usr/bin/env python3
"""Deterministic pre-outcome tests for the v2 token-native architecture."""

from __future__ import annotations

from copy import deepcopy
import json

import numpy as np

from spectral_utils.multitask_trajectory import (
    CHANNEL_NAMES,
    ChannelReference,
    causal_states,
    fit_channel_reference,
    fit_iu_head,
    global_feature_row,
    raw_risk_channels,
    stable_partition,
    truncate_row,
)
from spectral_utils.repgrid_scoring import logprob_features, logprob_features_extended


def _row(seed: int, n: int = 72) -> dict:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n).cumsum() / np.sqrt(np.arange(1, n + 1))
    entropy = np.clip(1.2 + 0.25 * latent + rng.normal(0, 0.2, n), 1e-4, None)
    spilled = np.clip(0.7 + 0.20 * latent + rng.normal(0, 0.2, n), 1e-4, None)
    logsumexp = 30.0 - 0.3 * latent + rng.normal(0, 0.3, n)
    raw = rng.normal(-3.0, 0.8, size=(n, 8))
    raw[:, 0] = -0.2 - 0.2 * latent + rng.normal(0, 0.1, n)
    raw[:, 1] = raw[:, 0] - np.abs(0.5 - 0.1 * latent + rng.normal(0, 0.1, n))
    raw = -np.sort(-raw, axis=1)
    return {
        "id": f"item-{seed}",
        "token_entropies": entropy,
        "token_spilled_energies": spilled,
        "token_logsumexp": logsumexp,
        "top_k_logprobs": {
            "ids": np.tile(np.arange(raw.shape[1]), (n, 1)),
            "logprobs": raw,
        },
        "label": -1 if seed % 2 else 1,
        "final_answer_correct": bool(seed % 2),
        "step_token_spans": [(0, n // 2), (n // 2, n)],
    }


def _head_signature(head) -> str:
    value = head.as_dict()
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_raw_formula_identity() -> None:
    row = _row(101)
    raw = raw_risk_channels(row)
    basic = logprob_features(row["top_k_logprobs"])
    extended = logprob_features_extended(row["top_k_logprobs"])
    np.testing.assert_allclose(-raw["neg_top1"].mean(), basic["mean_top1_logprob"])
    np.testing.assert_allclose(-raw["neg_margin"].mean(), basic["logprob_margin"])
    np.testing.assert_allclose(raw["topk_entropy"].mean(), basic["mean_logprob_entropy"])
    np.testing.assert_allclose(raw["topk_varentropy"].mean(), extended["varentropy"])
    np.testing.assert_allclose(raw["topk_renyi2"].mean(), extended["renyi_entropy_2"])
    np.testing.assert_allclose(raw["topk_tail_mass"].mean(), extended["topk_tail_mass"])


def test_suffix_and_chunk_identity() -> None:
    rows = [_row(i) for i in range(12)]
    reference = fit_channel_reference(rows[:6])
    row = rows[8]
    changed = deepcopy(row)
    cut = 29
    changed["token_entropies"][cut:] += 100.0
    changed["token_spilled_energies"][cut:] += 100.0
    changed["token_logsumexp"][cut:] -= 100.0
    changed["top_k_logprobs"]["logprobs"][cut:] -= 7.0
    left, right = causal_states(row, reference), causal_states(changed, reference)
    for name in left:
        np.testing.assert_array_equal(left[name][:cut], right[name][:cut])
    np.testing.assert_equal(
        global_feature_row(row, reference, upto=cut),
        global_feature_row(changed, reference, upto=cut),
    )

    # The final state obtained by independent prefix replay must match the
    # corresponding state of one tokenwise full replay.
    for stop in (1, 2, 7, 16, 31, len(row["token_entropies"])):
        prefix = truncate_row(row, stop)
        replay = causal_states(prefix, reference)
        for name in left:
            np.testing.assert_array_equal(replay[name], left[name][:stop])
        np.testing.assert_equal(
            global_feature_row(prefix, reference),
            global_feature_row(row, reference, upto=stop),
        )
    global_head = fit_iu_head(rows[:6], reference, "g_mean_q90_18")
    np.testing.assert_array_equal(
        global_head.score_global(row, reference, upto=cut),
        global_head.score_global(changed, reference, upto=cut),
    )


def test_label_blind_and_repeat_identity() -> None:
    rows = [_row(i) for i in range(30)]
    reference = fit_channel_reference(rows[:15])
    first = fit_iu_head(rows[:15], reference, "o_level_ewma_onset27")
    second = fit_iu_head(rows[:15], reference, "o_level_ewma_onset27")
    assert _head_signature(first) == _head_signature(second)

    altered = deepcopy(rows[:15])
    for index, row in enumerate(altered):
        row["label"] = 99 - index
        row["final_answer_correct"] = "removed"
        row["step_token_spans"] = [(0, 1)]
    reference_alt = fit_channel_reference(altered)
    head_alt = fit_iu_head(altered, reference_alt, "o_level_ewma_onset27")
    np.testing.assert_array_equal(reference.centres, reference_alt.centres)
    np.testing.assert_array_equal(reference.scales, reference_alt.scales)
    assert _head_signature(first) == _head_signature(head_alt)


def test_feature_order_invariance() -> None:
    rows = [_row(i) for i in range(30)]
    reference = fit_channel_reference(rows[:15])
    first = fit_iu_head(rows[:15], reference, "l_level_onset18")
    reverse = fit_iu_head(
        rows[:15], reference, "l_level_onset18",
        feature_order=tuple(reversed(first.feature_names)),
    )
    for row in rows[15:]:
        np.testing.assert_allclose(
            first.score_curve(row, reference),
            reverse.score_curve(row, reference),
            rtol=1e-10,
            atol=1e-10,
        )


def test_missing_and_operator_contract() -> None:
    rows = [_row(i) for i in range(18)]
    reference = fit_channel_reference(rows[:9])
    missing = deepcopy(rows[12])
    missing.pop("token_logsumexp")
    missing.pop("top_k_logprobs")
    states = causal_states(missing, reference)
    assert all(np.isfinite(values).all() for values in states.values())

    neutral = ChannelReference(
        np.zeros(len(CHANNEL_NAMES)), np.ones(len(CHANNEL_NAMES)),
        np.ones(len(CHANNEL_NAMES)), 1, 32,
    )
    base = _row(91, n=8)
    base["token_entropies"] = np.asarray([-1.0, -0.5, 0.0, 0.5, 0.0, 1.0, 0.2, 0.3])
    higher = deepcopy(base)
    higher["token_entropies"][5] += 0.5
    left = causal_states(base, neutral)
    right = causal_states(higher, neutral)
    # Fixed-time evidence monotonicity for the declared monotone operators.
    for state in ("level", "ewma", "positive_mean", "persistence", "running_max"):
        assert right[f"entropy__{state}"][5] >= left[f"entropy__{state}"][5]
    assert np.all(np.diff(left["entropy__running_max"]) >= 0.0)
    assert np.all(left["entropy__onset"] >= 0.0)
    # Onset is intentionally not history-monotone: increasing a previous value
    # raises the reference EWMA and can lower a later event magnitude.
    previous_higher = deepcopy(base)
    previous_higher["token_entropies"][4] += 1.0
    altered = causal_states(previous_higher, neutral)
    assert altered["entropy__onset"][5] < left["entropy__onset"][5]


def test_shared_partition() -> None:
    identities = [f"gsm8k-{index}" for index in range(200)]
    first = [stable_partition(identity) for identity in identities]
    second = [stable_partition(identity) for identity in identities]
    assert first == second
    assert 70 < first.count("calibration") < 130


def main() -> None:
    tests = [
        test_raw_formula_identity,
        test_suffix_and_chunk_identity,
        test_label_blind_and_repeat_identity,
        test_feature_order_invariance,
        test_missing_and_operator_contract,
        test_shared_partition,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all ({len(tests)} tests)")


if __name__ == "__main__":
    main()
