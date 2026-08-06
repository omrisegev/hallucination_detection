#!/usr/bin/env python3
"""Unit checks for the SDSF stabilization operators."""

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

from spectral_utils.dependency_fusion import sparse_upcr_fit            # noqa: E402
from spectral_utils.robust_dependency_fusion import (                   # noqa: E402
    bootstrap_reliability,
    diagonal_shrinkage,
    stability_shrunk_weights,
)


def main():
    C = np.array([[2.0, 0.8], [0.8, 1.0]])
    assert np.allclose(diagonal_shrinkage(C, 0.0), C)
    assert np.allclose(diagonal_shrinkage(C, 1.0), np.diag(np.diag(C)))

    rng = np.random.default_rng(7)
    latent = rng.normal(size=500)
    F = np.vstack([latent + 0.5 * rng.normal(size=500) for _ in range(8)])
    fit_kwargs = dict(max_iter=20, inner_completion_iter=10, g2_grid=50)
    fit = sparse_upcr_fit(F, **fit_kwargs)
    boot = bootstrap_reliability(F, fit, n_boot=4, seed=9, fit_kwargs=fit_kwargs)
    weight, diag = stability_shrunk_weights(fit, boot, tau=1.0)
    assert weight.shape == (8,)
    assert np.isfinite(weight).all()
    assert 0.0 <= diag["tail_kappa_min"] <= diag["tail_kappa_mean"] <= 1.0
    assert boot.n_successful == 4
    print("robust dependency-fusion tests passed")


if __name__ == "__main__":
    main()
