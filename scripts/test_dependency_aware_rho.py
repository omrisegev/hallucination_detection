#!/usr/bin/env python3
"""Mechanism and no-label unit gates for covariance-aware U-PCR rho."""

import inspect
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

from spectral_utils.dependency_aware_rho import (                    # noqa: E402
    CANDIDATE_METHODS,
    estimate_dependency_aware_rho,
    pair_moment_covariance,
    pair_product_samples,
)
from spectral_utils.dependency_fusion import sparse_upcr_fit          # noqa: E402


def check(name, condition, detail=""):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(name)


def planted_sample(seed=17, m=8, n=2000):
    g2 = 0.18
    a = np.linspace(-0.02, 0.02, m)
    rho = g2 + a
    C = g2 * np.ones((m, m)) + a[:, None] + a[None, :]
    np.fill_diagonal(C, 1.0)
    joint = np.block([[C, rho[:, None]], [rho[None, :], np.ones((1, 1))]])
    sample = np.random.default_rng(seed).multivariate_normal(
        np.zeros(m + 1), joint, size=n,
    )
    F = sample[:, :-1].T
    F = (F - F.mean(axis=1, keepdims=True)) / F.std(axis=1, keepdims=True)
    return F


def main():
    F = planted_sample()
    Z, A, pairs = pair_product_samples(F)
    check("pair sample dimensions", Z.shape == (F.shape[1], len(pairs)))
    check("additive design dimensions", A.shape == (len(pairs), F.shape[0]))
    check("shared-feature pair moments are dependent", abs(np.corrcoef(Z[:, 0], Z[:, 1])[0, 1]) > 0.02)

    fit = sparse_upcr_fit(F)
    results = {}
    for method in CANDIDATE_METHODS:
        precision, _, _, diag = pair_moment_covariance(F, fit.covariance, method)
        check(f"{method} precision finite", np.isfinite(precision).all())
        check(f"{method} moment condition capped", diag["moment_condition_regularized"] <= 100.0001)
        results[method] = estimate_dependency_aware_rho(
            F, fit.covariance, fit.decomposition.low_rank, fit.var_y,
            method=method,
        )
        check(f"{method} rho finite", np.isfinite(results[method].rho_hat).all())
        check(f"{method} PCR weights finite", np.isfinite(results[method].w_pcr).all())

    check("OLS rho exactly reproduces SU-PCR", np.allclose(results["ols"].rho_hat, fit.rho_hat, atol=1e-9))
    check("OLS PCR exactly reproduces SU-PCR", np.allclose(results["ols"].w_pcr, fit.w_pcr, atol=1e-9))
    check("full GLS is not diagonal WLS", not np.allclose(results["lw_gls"].rho_hat, results["diag_var"].rho_hat))

    forbidden = {"label", "labels", "y", "target", "targets"}
    for function in (pair_moment_covariance, estimate_dependency_aware_rho):
        check(
            f"{function.__name__} has no label seam",
            forbidden.isdisjoint(inspect.signature(function).parameters),
        )
    print("ALL PASSED")


if __name__ == "__main__":
    main()

