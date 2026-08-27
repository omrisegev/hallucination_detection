#!/usr/bin/env python3
"""Focused mechanical tests for the SU-aware pooled-graph sidecar."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.su_pooled_graph_adaptation_sidecar import (
    feature_cross_mask,
    geomedian_weights,
    pcr_weights,
    psd_projection,
)


def main() -> None:
    rng = np.random.default_rng(20260823)
    X = rng.normal(size=(7, 400))
    C = X @ X.T / X.shape[1]
    rho = rng.normal(size=7)
    w = pcr_weights(C, rho, 2)
    values, vectors = np.linalg.eigh(C)
    U = vectors[:, np.argsort(values)[::-1][:2]]
    expected = U @ np.linalg.solve(U.T @ C @ U, U.T @ rho)
    assert np.max(np.abs(w - expected)) < 1e-10

    indefinite = C.copy()
    indefinite[0, 0] = -1.0
    projected, diag = psd_projection(indefinite)
    assert np.min(np.linalg.eigvalsh(projected)) > -1e-10
    assert diag["n_negative_eigenvalues"] >= 1

    names = ("epr", "spectral_entropy", "epr_spilled", "min_spilled")
    mask = feature_cross_mask(names)
    assert not mask[2, 3]
    assert mask[0, 1] and mask[1, 2]
    assert not np.any(np.diag(mask))

    cloud = rng.normal(scale=0.05, size=(7, 12))
    cloud[-1] += 20.0
    weights = geomedian_weights(cloud)
    assert np.isclose(np.sum(weights), 1.0)
    assert weights[-1] < np.mean(weights[:-1])
    print("su pooled graph adaptation mechanical tests: PASS")


if __name__ == "__main__":
    main()
