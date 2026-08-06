#!/usr/bin/env python3
"""CPU smoke tests for hierarchical spectral correction."""

import os
import sys
import types

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.hierarchical_spectral_fusion import (          # noqa: E402
    fisher_d_optimal_order,
    fit_grouped_logistic_head,
    fit_score_head,
    fit_shared_representation,
)


def main():
    rng = np.random.default_rng(20260806)
    names = tuple(f"f{index}" for index in range(7))
    common = names[:5]
    matrices = []
    labels = []
    group_ids = []
    representations = []

    for group in range(3):
        y = rng.binomial(1, 0.35 + 0.1 * group, size=160)
        target = 2 * y - 1
        matrix = target[:, None] * np.linspace(0.1, 0.5, 7) + rng.normal(size=(160, 7))
        matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
        weight = np.linspace(0.5, 1.2, 7)
        representation = fit_shared_representation(matrix, names, common, weight)
        design = representation.transform(matrix)
        assert design.shape == (160, 1 + len(common))
        assert abs(design[:, 0].mean()) < 1e-10
        assert np.max(abs(design[:, 1:].T @ design[:, 0] / len(design))) < 1e-8
        matrices.append(design[:30])
        labels.append(y[:30])
        group_ids.extend([f"g{group}"] * 30)
        representations.append(representation)

    matrix = np.vstack(matrices)
    y = np.concatenate(labels)
    prior = np.r_[1.0, np.zeros(matrix.shape[1] - 1)]
    grouped = fit_grouped_logistic_head(matrix, y, group_ids, prior)
    assert grouped.converged
    assert grouped.coefficients.shape == prior.shape
    assert len(grouped.group_intercepts) == 3
    assert np.isfinite(grouped.coefficients).all()

    two_scores = matrix[:, :2]
    order = fisher_d_optimal_order(two_scores, np.array([1.0, 0.0]), 40)
    assert len(order) == 40 and len(set(order.tolist())) == 40
    assert np.all((order >= 0) & (order < len(two_scores)))
    assert np.array_equal(
        order[:10],
        fisher_d_optimal_order(two_scores, np.array([1.0, 0.0]), 10),
    )
    head = fit_score_head(two_scores[order[:20]], y[order[:20]], np.array([1.0, 0.0]))
    assert head.converged and isinstance(head.weight, np.ndarray)
    assert np.isfinite(head.weight).all()
    print("hierarchical spectral fusion smoke: PASS")


if __name__ == "__main__":
    main()
