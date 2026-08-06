#!/usr/bin/env python3
"""CPU smoke tests for the semi-supervised spectral head."""

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

from spectral_utils.semi_supervised_fusion import (                 # noqa: E402
    fit_logistic_head,
    fit_soft_logistic_head,
    orient_weight,
    spectral_score_basis,
    standardize_train_test,
)


def main():
    rng = np.random.default_rng(20260806)
    y = rng.binomial(1, 0.5, size=300)
    target = 2 * y - 1
    X = target[:, None] * np.linspace(0.1, 0.7, 8) + rng.normal(size=(300, 8))
    train, test, center, scale = standardize_train_test(X[:220], X[220:])
    assert train.shape == (220, 8) and test.shape == (80, 8)
    assert np.max(np.abs(train.mean(axis=0))) < 1e-12
    assert np.all(scale > 0) and np.isfinite(center).all()

    base = orient_weight(np.ones(8), train, train.mean(axis=1))
    basis = spectral_score_basis(train, base, rank=6)
    covariance = (train.T @ train) / len(train)
    gram = basis.T @ covariance @ basis
    assert np.max(np.abs(gram - np.eye(gram.shape[0]))) < 1e-8, gram
    assert np.corrcoef(train @ basis[:, 0], train @ base)[0, 1] > 1 - 1e-10

    trusted = np.r_[np.flatnonzero(y[:220] == 0)[:10], np.flatnonzero(y[:220] == 1)[:10]]
    prior = np.r_[1.0, np.zeros(basis.shape[1] - 1)]
    head = fit_logistic_head(train[trusted], y[trusted], basis, prior, prior_strength=10.0)
    assert head.converged and np.isfinite(head.weight).all()
    assert np.std(test @ head.weight) > 0

    teacher_score = train @ basis[:, 0]
    teacher_probability = 1.0 / (1.0 + np.exp(-teacher_score))
    soft = fit_soft_logistic_head(
        train[trusted], y[trusted], train, teacher_probability,
        basis, np.zeros(basis.shape[1]), soft_total_weight=10.0,
    )
    assert soft.converged and np.isfinite(soft.weight).all()

    # A positive affine change of a one-dimensional score cannot alter order.
    original_order = np.argsort(test @ basis[:, 0])
    transformed_order = np.argsort(2.5 * (test @ basis[:, 0]) - 1.7)
    assert np.array_equal(original_order, transformed_order)
    print("semi-supervised fusion smoke: PASS")


if __name__ == "__main__":
    main()
