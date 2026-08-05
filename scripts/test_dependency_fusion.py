#!/usr/bin/env python3
"""Dataset-free gates for the dependency-aware fusion experiment.

These are planted-world mechanism tests, not evidence that the method improves
hallucination AUROC.  Run them before the real cache:

    python3 scripts/test_dependency_fusion.py
"""

import inspect
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.deem_adapter import (                         # noqa: E402
    _aligned_probabilities,
    continuous_to_deem_hard, continuous_to_deem_soft, fit_deem_score,
)
from spectral_utils.dependency_fusion import (                     # noqa: E402
    projected_sparse_decomposition,
    regularized_covariance_weights,
    sparse_upcr_fit,
)


FAILED = []


def check(name, condition, detail=""):
    state = "PASS" if condition else "FAIL"
    print(f"  [{state}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILED.append(name)


def planted_covariance(m=14, with_sparse=True):
    g2 = 0.22
    a = np.linspace(-0.035, 0.035, m)
    low = g2 * np.ones((m, m)) + a[:, None] + a[None, :]
    sparse = np.zeros((m, m))
    edges = [(0, 7, 0.48), (2, 10, -0.70), (5, 12, 0.40)] if with_sparse else []
    for i, j, value in edges:
        sparse[i, j] = sparse[j, i] = value
    C = low + sparse
    # Individual error variances live on the diagonal and are not sparse.
    np.fill_diagonal(C, 1.20)
    # The planted covariance must be valid for the sample-level test below.
    eigmin = np.linalg.eigvalsh(C).min()
    if eigmin <= 0:
        C += (abs(eigmin) + 0.05) * np.eye(m)
    return C, low, sparse, edges


def test_deem_transforms():
    print("\n1. continuous -> DEEM input transforms")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(101, 8))
    X[:, 1] = np.round(X[:, 1], 1)  # ties must remain deterministic
    hard = continuous_to_deem_hard(X)
    soft = continuous_to_deem_soft(X)
    check("hard shape is (N,M)", hard.shape == (101, 8), str(hard.shape))
    check("hard labels are binary", set(np.unique(hard)) <= {0, 1})
    check("soft shape is (N,2,M)", soft.shape == (101, 2, 8), str(soft.shape))
    check("soft distributions sum to one",
          np.allclose(soft.sum(axis=1), 1.0, atol=1e-12))
    order = np.argsort(X[:, 0])
    check("soft class-1 probability preserves rank",
          np.all(np.diff(soft[order, 1, 0]) >= 0))

    # DEEM 0.2.0 deliberately does not align return_probs=True.  This fake
    # captures that API contract without requiring a DEEM installation or a
    # stochastic neural fit in the unit gate.
    class FakeDeem:
        def __init__(self):
            self.mapping = None

        def predict(self, predictions, return_probs=False, align_to=None):
            if return_probs:
                return np.array([[0.8, 0.2], [0.1, 0.9]])
            self.mapping = {0: 1, 1: 0}
            return np.array([1, 0])

        def get_class_mapping(self):
            return self.mapping

    aligned, mapping = _aligned_probabilities(FakeDeem(), hard[:2])
    check("DEEM probability columns receive the unsupervised class map",
          mapping == {0: 1, 1: 0} and np.allclose(
              aligned, [[0.2, 0.8], [0.9, 0.1]]
          ))


def test_sparse_support():
    print("\n2. projected low-rank + sparse support")
    C, low, sparse, edges = planted_covariance()
    fit = projected_sparse_decomposition(C)
    planted = {(min(i, j), max(i, j)) for i, j, _ in edges}
    recovered = set()
    iu = np.triu_indices_from(C, 1)
    for i, j, value in zip(iu[0], iu[1], fit.sparse[iu]):
        if abs(value) > 0:
            recovered.add((int(i), int(j)))
    overlap = len(planted & recovered) / len(planted)
    off = ~np.eye(len(C), dtype=bool)
    low_error = np.linalg.norm((fit.low_rank - low)[off]) / np.linalg.norm(low[off])
    check("decomposition converges", fit.converged, f"iterations={fit.n_iter}")
    check("recovers most planted dependency edges", overlap >= 2 / 3,
          f"overlap={overlap:.3f}, recovered={sorted(recovered)}")
    recovered_values = [fit.sparse[i, j] for i, j in planted & recovered]
    check("support recovery handles both dependency signs",
          any(v > 0 for v in recovered_values) and any(v < 0 for v in recovered_values),
          f"recovered values={np.round(recovered_values, 3).tolist()}")
    check("recovers the rank-two off-diagonal component", low_error < 0.25,
          f"relative error={low_error:.4f}")
    check("structured covariance preserves observed diagonal",
          np.allclose(np.diag(fit.structured_cov), np.diag(C)))


def test_clean_world():
    print("\n3. clean-world degeneration")
    C, _, _, _ = planted_covariance(with_sparse=False)
    fit = projected_sparse_decomposition(C)
    check("clean world does not invent a dense sparse matrix",
          fit.sparse_fraction <= 0.05,
          f"sparse fraction={fit.sparse_fraction:.4f}")
    check("clean-world residual is small", fit.relative_residual < 0.10,
          f"relative residual={fit.relative_residual:.4f}")


def test_condition_control():
    print("\n4. dependency-aware ridge condition control")
    C = np.array([[1.0, 0.9999, 0.2],
                  [0.9999, 1.0, 0.2],
                  [0.2, 0.2, 1.0]])
    rho = np.array([0.5, 0.5, 0.3])
    w, diag = regularized_covariance_weights(C, rho, target_condition=50)
    check("weights are finite", np.isfinite(w).all())
    check("condition cap is respected", diag["condition_after"] <= 50.0001,
          f"condition={diag['condition_after']:.4f}")
    check("a positive ridge is applied to a near-duplicate pair", diag["ridge"] > 0,
          f"ridge={diag['ridge']:.6g}")


def test_joint_fit():
    print("\n5. one decomposition feeds both SU-PCR and SDSF")
    C, _, _, _ = planted_covariance(m=14)
    rng = np.random.default_rng(19)
    F = rng.multivariate_normal(np.zeros(len(C)), C, size=6000).T
    F = (F - F.mean(axis=1, keepdims=True)) / F.std(axis=1, keepdims=True)
    result = sparse_upcr_fit(F)
    check("rho is finite", np.isfinite(result.rho_hat).all())
    check("SU-PCR weights are finite", np.isfinite(result.w_pcr).all())
    check("SDSF weights are finite", np.isfinite(result.w_structured).all())
    check("the two registered weight rules are genuinely different",
          not np.allclose(result.w_pcr, result.w_structured, atol=1e-6))

    # Recovered rho must satisfy the cleaned pair equations at the selected g2.
    errors = []
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            predicted = result.rho_hat[i] + result.rho_hat[j] - result.g2_hat
            errors.append(predicted - result.decomposition.low_rank[i, j])
    check("rho solves the cleaned additive system",
          np.sqrt(np.mean(np.square(errors))) < 0.05,
          f"pair RMSE={np.sqrt(np.mean(np.square(errors))):.5f}")


def test_no_label_seam():
    print("\n6. estimator API cannot receive labels")
    parameters = inspect.signature(sparse_upcr_fit).parameters
    forbidden = {"label", "labels", "y", "target", "targets"}
    check("no label-like estimator parameter", forbidden.isdisjoint(parameters),
          f"parameters={list(parameters)}")
    deem_parameters = inspect.signature(fit_deem_score).parameters
    check("DEEM adapter has no label-like estimator parameter",
          forbidden.isdisjoint(deem_parameters),
          f"parameters={list(deem_parameters)}")


def main():
    print("=" * 76)
    print("UNIT GATES — dependency-aware fusion")
    print("=" * 76)
    test_deem_transforms()
    test_sparse_support()
    test_clean_world()
    test_condition_control()
    test_joint_fit()
    test_no_label_seam()
    print("\n" + "=" * 76)
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
        raise SystemExit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
