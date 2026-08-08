#!/usr/bin/env python3
"""Known-answer checks for the GL-LIU factorial feature contract."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spectral_utils.token_feature_views import (  # noqa: E402
    BROAD_TOKEN_VIEWS,
    CORE_TOKEN_VIEWS,
    TOKEN_TO_GLOBAL_FEATURES,
    token_feature_views,
)


def _row(n=96):
    t = np.linspace(0.0, 6.0, n)
    entropy = 0.8 + 0.2 * np.sin(t) + 0.05 * np.cos(4.0 * t)
    spilled = 1.2 + 0.3 * np.cos(0.7 * t)
    energy = 5.0 + 0.4 * np.sin(0.4 * t)
    base = np.linspace(-0.1, -8.0, 50)
    logprobs = np.vstack([base - 0.05 * np.sin(value) for value in t]).astype(np.float32)
    return {
        "token_entropies": entropy.tolist(),
        "token_spilled_energies": spilled.tolist(),
        "token_logsumexp": energy.tolist(),
        "top_k_logprobs": {"logprobs": logprobs},
        "label": 3,
        "step_token_spans": [(0, n // 2), (n // 2, n)],
    }


def main():
    row = _row()
    curves = token_feature_views(row)
    assert tuple(curves) == BROAD_TOKEN_VIEWS
    assert len(BROAD_TOKEN_VIEWS) == 28
    assert set(CORE_TOKEN_VIEWS).issubset(curves)
    assert all(value.shape == (96,) for value in curves.values())
    assert all(np.isfinite(value).all() for value in curves.values())

    # Labels and benchmark step spans are outside the score-construction API.
    changed = copy.deepcopy(row)
    changed["label"] = -1
    changed["step_token_spans"] = [(0, 96)]
    changed_curves = token_feature_views(changed)
    for name in curves:
        assert np.array_equal(curves[name], changed_curves[name]), name

    # No local curve name occurs twice, and the two global CUSUM summaries are
    # deliberately represented by one curve rather than duplicated.
    assert len(TOKEN_TO_GLOBAL_FEATURES) == len(set(TOKEN_TO_GLOBAL_FEATURES))
    assert TOKEN_TO_GLOBAL_FEATURES["entropy_cusum_abs_series"] == (
        "cusum_max", "cusum_shift_idx"
    )
    assert not any("trace_length" in parents for parents in TOKEN_TO_GLOBAL_FEATURES.values())

    # A planted entropy change must affect multiple local views near/after the
    # change; otherwise the broad contract would only be a set of constants.
    changed = _row()
    changed["token_entropies"][64:] = np.asarray(changed["token_entropies"][64:]) + 1.5
    changed_curves = token_feature_views(changed)
    affected = [
        name for name in BROAD_TOKEN_VIEWS
        if not np.allclose(curves[name], changed_curves[name])
    ]
    assert len(affected) >= 10, affected

    # Short traces still produce aligned finite curves.
    short = token_feature_views(_row(6))
    assert all(value.shape == (6,) for value in short.values())
    assert all(np.isfinite(value).all() for value in short.values())
    print("test_gl_liu_factorial_v2: PASS")


if __name__ == "__main__":
    main()
