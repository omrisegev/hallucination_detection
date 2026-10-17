#!/usr/bin/env python3
"""Unit checks for the factorial Renyi locator feature bank."""
from __future__ import annotations

import numpy as np

from spectral_utils import renyi_locator_feature_bank as model


def run() -> dict:
    rng = np.random.default_rng(2026091406)
    logits = np.sort(rng.normal(size=(37, 50)), axis=1)[:, ::-1]
    logprobs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True) + 3.0)
    entropy = rng.uniform(0.2, 3.0, size=37)
    spans = np.asarray([[0, 7], [7, 19], [19, 37]])
    features = model.feature_matrix(logprobs, entropy)
    assert features["matrix"].shape == (37, 7)
    assert len(model.BANKS) == 8 and len(model.METHODS) == 24
    assert len({bank.name for bank in model.BANKS}) == 8
    baseline = model.BANK_BY_NAME["ve1q15__h10__hinf0"]
    assert baseline.indices == (0, 1, 2, 3)
    q50 = model.BANK_BY_NAME["ve1q50__h10__hinf0"]
    assert q50.indices == (0, 1, 2, 4)
    for bank in model.BANKS:
        for solver in model.SOLVERS:
            score, info = model.score_bank(features, spans, bank, solver)
            assert score.shape == (3,) and np.isfinite(score).all()
            assert info["features"] == [model.ALL_FEATURES[index] for index in bank.indices]
    return {"schema": "renyi-locator-feature-bank-unit-v1", "status": "PASS", "banks": 8, "methods": 24}


if __name__ == "__main__":
    print(run())
