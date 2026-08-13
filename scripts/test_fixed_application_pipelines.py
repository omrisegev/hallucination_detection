#!/usr/bin/env python3
"""Regression tests for the fixed shared-feature application pipelines."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from spectral_utils.fixed_application_pipelines import (
    SHARED_TOKEN_VIEWS,
    contract_audit,
    fit_rag_evidence_head,
    fit_shared_mixed_transformer,
    fit_shared_token_iu,
    rag_evidence_matrix,
    raw_token_feature_matrix,
)


def synthetic_rows(count=7):
    rng = np.random.default_rng(19)
    rows = []
    for index in range(count):
        n = 72 + 3 * index
        lp = -np.sort(-rng.normal(-4.0, 0.6, size=(n, 10)), axis=1)
        rows.append({
            "token_entropies": rng.uniform(0.05, 2.1, n),
            "token_spilled_energies": rng.uniform(0.1, 4.5, n),
            "token_logsumexp": rng.uniform(1.0, 7.0, n),
            "top_k_logprobs": {"logprobs": lp},
        })
    return rows


class FixedApplicationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            (f"row-{index}", raw_token_feature_matrix(row))
            for index, row in enumerate(synthetic_rows())
        ]

    def test_contract_covers_original_30(self):
        audit = contract_audit()
        self.assertEqual(audit["token_stream_count"], 29)
        self.assertEqual(audit["covered_global_feature_count"], 30)
        self.assertEqual(set(audit["covered_global_features"]), set(sum(
            audit["token_to_global_features"].values(), []
        )))

    def test_trace_length_is_constant_within_response(self):
        matrix = self.records[0][1]
        self.assertEqual(matrix.shape[1], len(SHARED_TOKEN_VIEWS))
        self.assertTrue(np.all(matrix[:, 0] == len(matrix)))

    def test_fit_is_invariant_to_record_order(self):
        left = fit_shared_token_iu(self.records, max_fit_tokens=400)
        right = fit_shared_token_iu(list(reversed(self.records)), max_fit_tokens=400)
        np.testing.assert_allclose(left.weights, right.weights, atol=0, rtol=0)
        np.testing.assert_allclose(left.risk(self.records[0][1]), right.risk(self.records[0][1]))

    def test_rag_profiles_keep_same_base_contract(self):
        transformer = fit_shared_mixed_transformer(self.records, max_fit_tokens=400)
        base = self.records[0][1]
        conditions = {"full": base, "noctx": base + 0.01,
                      "loo_0": base + 0.02, "loo_1": base - 0.01}
        noctx, noctx_names = rag_evidence_matrix(conditions, transformer, profile="noctx")
        loo, loo_names = rag_evidence_matrix(conditions, transformer, profile="loo")
        self.assertEqual(noctx.shape[1], 2 * len(SHARED_TOKEN_VIEWS))
        self.assertEqual(loo.shape[1], 6 * len(SHARED_TOKEN_VIEWS))
        self.assertEqual(len(noctx_names), noctx.shape[1])
        self.assertEqual(len(loo_names), loo.shape[1])

    def test_lambda_free_final_head_is_finite(self):
        transformer = fit_shared_mixed_transformer(self.records, max_fit_tokens=400)
        evidence = []
        for sample_id, base in self.records:
            matrix, names = rag_evidence_matrix(
                {"full": base, "noctx": base + 0.01}, transformer, profile="noctx"
            )
            evidence.append((sample_id, matrix, names))
        head = fit_rag_evidence_head(evidence, profile="noctx", max_fit_tokens=400)
        self.assertTrue(np.isfinite(head.risk(evidence[0][1])).all())

    def test_fit_api_has_no_label_argument(self):
        for function in (fit_shared_token_iu, fit_shared_mixed_transformer, fit_rag_evidence_head):
            parameters = set(inspect.signature(function).parameters)
            self.assertFalse(any("label" in name for name in parameters))


if __name__ == "__main__":
    unittest.main()
