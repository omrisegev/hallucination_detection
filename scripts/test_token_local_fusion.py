#!/usr/bin/env python3
"""Mechanical and target-free synthetic checks for token-local Phase 1."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.dependency_fusion import sparse_upcr_fit  # noqa: E402
from spectral_utils.laplacian_upcr import (  # noqa: E402
    build_graph_from_features,
    laplacian_iu_fit,
)
from spectral_utils.reconstruction_benchmark.localization_fit import (  # noqa: E402
    _fit_token_iu,
)
from spectral_utils.token_local_fusion import (  # noqa: E402
    CONTROL_METHOD_IDS,
    IU_CONFIG,
    LOCAL_EQUAL_FAMILY,
    LOCAL_IU29,
    PRIMARY_METHOD_IDS,
    SU_CONFIG,
    fit_local_equal29,
    fit_local_equal_family,
    fit_local_iu29,
    fit_phase1_ladder,
    learn_stg_sparse_support,
    prepare_token_fusion,
    step_maxima,
)


def fixture(
    *, seed: int = 20260828, n_rows: int = 50, tokens_per_row: int = 24
):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_rows, 29))
    parts = []
    starts, ends = [], []
    for row in range(n_rows):
        base = row * tokens_per_row
        trend = np.linspace(-0.5, 0.5, tokens_per_row)[:, None]
        parts.append(
            latent[row] + 0.20 * trend
            + 0.35 * rng.normal(size=(tokens_per_row, 29))
        )
        for step in range(3):
            lo = base + step * (tokens_per_row // 3)
            hi = base + (step + 1) * (tokens_per_row // 3)
            starts.append(lo)
            ends.append(hi)
    return (
        np.vstack(parts),
        np.arange(
            0, (n_rows + 1) * tokens_per_row, tokens_per_row, dtype=np.int64
        ),
        tuple(f"opaque_row_{index:03d}" for index in range(n_rows)),
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
    )


def sparse_world(*, correlated: bool):
    rng = np.random.default_rng(120)
    n_rows, tokens_per_row, n_features = 80, 40, 29
    n_tokens = n_rows * tokens_per_row
    latent = rng.normal(size=(n_tokens, 1))
    values = latent + 0.8 * rng.normal(size=(n_tokens, n_features))
    planted = ((0, 1), (2, 3), (4, 5))
    if correlated:
        for first, second in planted:
            shared_error = 1.8 * rng.normal(size=n_tokens)
            values[:, first] += shared_error
            values[:, second] += shared_error
    preparation = prepare_token_fusion(
        values,
        np.arange(
            0, (n_rows + 1) * tokens_per_row,
            tokens_per_row, dtype=np.int64,
        ),
        tuple(f"synthetic_group_{index:03d}" for index in range(n_rows)),
    )
    return preparation, planted


class TokenLocalFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        values, offsets, rows, starts, ends = fixture()
        cls.values = values
        cls.starts = starts
        cls.ends = ends
        cls.preparation = prepare_token_fusion(values, offsets, rows)

    def test_equal_and_incumbent_aliases_are_exact(self) -> None:
        equal = fit_local_equal29(self.preparation)
        expected_equal = -self.preparation.standardized_fit.mean(axis=1)
        # atol covers the BLAS-vs-pairwise-mean accumulation gap: the deployed
        # path computes -(S @ uniform_weights) via dgemv while this reference
        # uses S.mean(axis=1). Measured max gap is 3.34e-16 (1.5 ULP, 1 of
        # 1200 fit tokens) on both the AIRCC pytorch:25.01 container
        # (numpy 1.26.4) and Windows numpy 2.2.4; 3e-16 was one ULP too tight
        # on every available platform. 1e-15 still fails for any real weight
        # or standardization regression, which moves scores by >=1e-2.
        self.assertTrue(np.allclose(
            equal.token_risk[self.preparation.fit_indices], expected_equal,
            rtol=0.0, atol=1e-15,
        ))

        incumbent, _ = _fit_token_iu(
            SimpleNamespace(token_confidence=self.values)
        )
        alias = fit_local_iu29(self.preparation)
        self.assertEqual(alias.method_id, LOCAL_IU29)
        self.assertTrue(np.array_equal(alias.token_risk, incumbent))

    def test_equal_family_excludes_structural_context_from_local_mass(self) -> None:
        result = fit_local_equal_family(self.preparation)
        self.assertEqual(result.method_id, LOCAL_EQUAL_FAMILY)
        family_names = self.preparation.kept_family_names
        structural = [i for i, name in enumerate(family_names) if name == "structural"]
        self.assertEqual(len(structural), 1)
        self.assertEqual(float(result.weights[structural[0]]), 0.0)
        nonstructural = sorted(set(family_names) - {"structural"})
        masses = [
            float(sum(
                result.weights[index]
                for index, name in enumerate(family_names) if name == family
            ))
            for family in nonstructural
        ]
        self.assertTrue(np.allclose(masses, np.full(len(masses), 1.0 / len(masses))))
        self.assertAlmostEqual(float(result.weights.sum()), 1.0)

    def test_zero_laplacian_gate_is_exact_iu_identity(self) -> None:
        graph = build_graph_from_features(self.preparation.F, k=7)
        zero = laplacian_iu_fit(
            self.preparation.F,
            lambda_=0.0,
            graph=graph,
            baseline_kwargs=dict(IU_CONFIG),
        )
        ordinary = fit_local_iu29(self.preparation)
        oriented = np.asarray(zero.w, dtype=float)
        anchor = self.preparation.standardized_fit.mean(axis=1)
        if np.corrcoef(self.preparation.standardized_fit @ oriented, anchor)[0, 1] < 0:
            oriented = -oriented
        self.assertTrue(np.array_equal(oriented, ordinary.weights))

    def test_fixed_sparse_support_is_respected(self) -> None:
        support = np.zeros(
            (self.preparation.n_features, self.preparation.n_features), dtype=bool
        )
        support[0, 1] = support[1, 0] = True
        support[3, 7] = support[7, 3] = True
        fitted = sparse_upcr_fit(
            self.preparation.F, **dict(SU_CONFIG), fixed_support=support
        )
        self.assertTrue(np.array_equal(fitted.decomposition.support, support))
        self.assertTrue(fitted.decomposition.meta["fixed_support"])

    def test_stg_synthetic_recovery_and_null(self) -> None:
        null_preparation, _ = sparse_world(correlated=False)
        null = learn_stg_sparse_support(null_preparation)
        self.assertEqual(int(np.triu(null.support, 1).sum()), 0)

        planted_preparation, planted = sparse_world(correlated=True)
        first = learn_stg_sparse_support(planted_preparation)
        second = learn_stg_sparse_support(planted_preparation)
        self.assertTrue(np.array_equal(first.support, second.support))
        self.assertTrue(np.array_equal(
            first.pair_probability, second.pair_probability
        ))
        recovered = sum(bool(first.support[i, j]) for i, j in planted)
        self.assertGreaterEqual(recovered, 2)
        self.assertTrue(first.diagnostics["labels_seen_during_fit"] is False)

    def test_complete_ladder_is_deterministic_and_reconstructable(self) -> None:
        first = fit_phase1_ladder(self.preparation)
        second = fit_phase1_ladder(self.preparation)
        self.assertEqual(tuple(first), PRIMARY_METHOD_IDS + CONTROL_METHOD_IDS)
        for method_id in first:
            self.assertTrue(np.array_equal(
                first[method_id].weights, second[method_id].weights
            ), method_id)
            self.assertTrue(np.array_equal(
                first[method_id].token_risk, second[method_id].token_risk
            ), method_id)
            reconstructed = self.preparation.token_risk(first[method_id].weights)
            self.assertTrue(np.array_equal(
                reconstructed, first[method_id].token_risk
            ), method_id)
            step = step_maxima(
                first[method_id].token_risk, self.starts, self.ends
            )
            self.assertEqual(step.shape, self.starts.shape)
            self.assertTrue(np.isfinite(step).all())
            self.assertTrue(
                first[method_id].diagnostics["labels_seen_during_fit"] is False
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
