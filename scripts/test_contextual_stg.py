#!/usr/bin/env python3
"""Known-answer tests for the oriented c-STG diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spectral_utils.contextual_stg import (  # noqa: E402
    ContextualSTGConfig,
    ContextualSTGModel,
    balanced_group_weights,
)


def switching_world(seed: int = 7, n: int = 900):
    rng = np.random.default_rng(seed)
    regime = rng.integers(0, 2, size=n)
    labels = rng.integers(0, 2, size=n)
    signed = 2.0 * labels - 1.0
    X = rng.normal(0.0, 1.0, size=(n, 6))
    X[np.arange(n), regime] += 2.5 * signed
    # A coherent distractor is globally useful but weaker than the routed pair.
    X[:, 5] += 0.45 * signed
    Z = np.column_stack([regime, rng.normal(size=n)])
    groups = np.asarray([f"q{index}" for index in range(n)])
    return X, Z, labels, groups, regime


def main() -> None:
    X, Z, labels, groups, regime = switching_world()
    weights = balanced_group_weights(labels, groups)
    assert np.isclose(weights[labels == 0].sum(), 0.5)
    assert np.isclose(weights[labels == 1].sum(), 0.5)

    config = ContextualSTGConfig(
        hidden_dim=12, epochs=450, minimum_epochs=180, patience=80,
        learning_rate=0.008, sparsity=0.005,
    )
    model = ContextualSTGModel(config).fit(X, Z, labels, groups, seed=11)
    prediction = model.predict(X, Z)
    assert prediction.score.shape == (len(X),)
    assert prediction.family_gates.shape == (len(X), X.shape[1])
    assert np.isfinite(prediction.score).all()
    assert np.min(prediction.family_gates) >= 0.0
    assert np.max(prediction.family_gates) <= 1.0
    active = prediction.family_gates[np.arange(len(X)), regime]
    inactive = prediction.family_gates[np.arange(len(X)), 1 - regime]
    assert float(np.mean(active - inactive)) > 0.15
    assert roc_auc_score(labels, prediction.score) > 0.90
    assert min(model.diagnostics_["feature_weights"]) >= 0.0
    assert model.diagnostics_["context_direct_prediction_path"] is False

    # Determinism and feature-to-family grouping.
    second = ContextualSTGModel(config).fit(X, Z, labels, groups, seed=11)
    assert np.allclose(prediction.score, second.predict(X, Z).score)
    grouped = ContextualSTGModel(config).fit(
        np.column_stack([X[:, :2], X[:, :2]]), Z, labels, groups,
        feature_group_ids=(0, 1, 0, 1), seed=12,
    )
    grouped_prediction = grouped.predict(
        np.column_stack([X[:, :2], X[:, :2]]), Z
    )
    assert np.allclose(
        grouped_prediction.feature_gates[:, :2],
        grouped_prediction.feature_gates[:, 2:],
    )

    try:
        ContextualSTGModel(config).fit(X, Z, labels, groups, feature_group_ids=(0, 2, 2, 2, 2, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("non-contiguous family ids must fail")
    print("CONTEXTUAL STG TEST PASS")


if __name__ == "__main__":
    main()
