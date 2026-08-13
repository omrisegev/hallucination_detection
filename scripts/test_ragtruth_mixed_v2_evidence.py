#!/usr/bin/env python3
"""Dataset-free tests for the original-30 RAG evidence tensor."""

from __future__ import annotations

import unittest
from types import MappingProxyType

import numpy as np

from spectral_utils.ragtruth_evidence_contrast import (
    FULL_FEATURES,
    ConditionTrace,
    FeatureTable,
    RagDataset,
    RagResponse,
)
from spectral_utils.ragtruth_mixed_v2_evidence import (
    ORIGINAL_FEATURES,
    build_mixed_v2_evidence_tensor,
    build_variant_matrices,
    feature_availability,
    permute_evidence_blocks,
    trace_original30,
)


def _trace(condition: str, seed: int, length: int = 48) -> ConditionTrace:
    rng = np.random.default_rng(seed)
    entropy = 1.0 + rng.uniform(0.0, 1.0, size=length)
    nll = 0.5 + rng.uniform(0.0, 1.0, size=length)
    logsumexp = 8.0 + rng.normal(scale=0.2, size=length)
    probabilities = np.linspace(0.06, 0.001, 50)
    probabilities *= 0.92 / probabilities.sum()
    top_logprobs = np.log(np.tile(probabilities, (length, 1)))
    top_ids = np.tile(np.arange(50), (length, 1))
    return ConditionTrace(
        condition=condition,
        prompt_len=100 + seed,
        token_ids=np.arange(length),
        target_logprob=-nll,
        entropy=entropy,
        logsumexp=logsumexp,
        top_ids=top_ids,
        top_logprobs=top_logprobs,
    )


def _dataset(length: int = 48) -> RagDataset:
    responses = []
    for index in range(6):
        conditions = {
            "full": _trace("full", 100 + index, length),
            "noctx": _trace("noctx", 200 + index, length),
            "loo_0": _trace("loo_0", 300 + index, length),
            "loo_1": _trace("loo_1", 400 + index, length),
        }
        responses.append(RagResponse(
            response_id=str(index + 1),
            source_id=f"source-{index // 2}",
            task_type="QA" if index < 3 else "Data2txt",
            source="unit-test",
            generator_model="generator",
            quality="good",
            response_text="x" * length,
            conditions=MappingProxyType(conditions),
            sentences=(),
        ))
    return RagDataset(tuple(responses), "hash", "tokenizer")


def _ec_table(tensor) -> FeatureTable:
    rng = np.random.default_rng(91)
    n = len(tensor.response_ids)
    return FeatureTable(
        name="full_response",
        contract="EC-full-v1",
        feature_names=FULL_FEATURES,
        values=rng.normal(size=(n, len(FULL_FEATURES))),
        sample_ids=tensor.response_ids,
        response_ids=tensor.response_ids,
        source_ids=tensor.source_ids,
        task_types=tensor.task_types,
        sources=tensor.sources,
        generator_models=tuple("generator" for _ in range(n)),
        response_lengths=tensor.response_lengths,
        unit_lengths=tensor.response_lengths,
        chunk_counts=tensor.chunk_counts,
        context_lengths=tensor.context_lengths,
        supporting_chunks=np.full(n, -1),
    )


class MixedV2EvidenceTests(unittest.TestCase):
    def test_one_condition_extracts_exactly_the_original_30(self):
        values = trace_original30(_trace("full", 17))
        self.assertEqual(tuple(values), ORIGINAL_FEATURES)
        self.assertEqual(len(values), 30)
        self.assertTrue(np.isfinite(list(values.values())).all())

    def test_full_fitted_transform_reproduces_mixed_v2(self):
        tensor = build_mixed_v2_evidence_tensor(_dataset())
        self.assertLessEqual(tensor.exact_full_contract_error, 1e-10)
        self.assertEqual(tensor.mixed_full.shape, (6, 30))
        self.assertEqual(tuple(tensor.feature_names), ORIGINAL_FEATURES)

    def test_all_four_matrices_keep_original_feature_provenance(self):
        tensor = build_mixed_v2_evidence_tensor(_dataset())
        variants = build_variant_matrices(tensor, _ec_table(tensor))
        self.assertEqual(variants["original30_full"].values.shape, (6, 30))
        self.assertEqual(variants["original30_noctx"].values.shape, (6, 60))
        self.assertEqual(variants["original30_loo"].values.shape, (6, 180))
        self.assertEqual(variants["hybrid"].values.shape, (6, 194))
        for name in ("original30_full", "original30_noctx", "original30_loo"):
            self.assertTrue(set(variants[name].base_features) <= set(ORIGINAL_FEATURES))
        trace_index = ORIGINAL_FEATURES.index("trace_length")
        self.assertTrue(np.allclose(
            variants["original30_noctx"].values[:, 30 + trace_index], 0.0
        ))

    def test_condition_permutation_preserves_full_and_block_marginals(self):
        tensor = build_mixed_v2_evidence_tensor(_dataset())
        variant = build_variant_matrices(tensor, _ec_table(tensor))["original30_loo"]
        tasks = np.asarray(tensor.task_types)
        permuted = permute_evidence_blocks(variant, tasks, seed=7)
        full = np.asarray(variant.block_names) == "full"
        np.testing.assert_array_equal(permuted[:, full], variant.values[:, full])
        for block in sorted(set(variant.block_names) - {"full"}):
            columns = np.asarray(variant.block_names) == block
            for task in sorted(set(tasks)):
                rows = tasks == task
                np.testing.assert_allclose(
                    np.sort(permuted[np.ix_(rows, columns)], axis=0),
                    np.sort(variant.values[np.ix_(rows, columns)], axis=0),
                )

    def test_missing_original_feature_fails_closed(self):
        data = _dataset(length=3)
        audit = feature_availability(data)
        self.assertTrue(any(not row["fully_available"] for row in audit))
        with self.assertRaisesRegex(ValueError, "no imputation"):
            build_mixed_v2_evidence_tensor(data)


if __name__ == "__main__":
    unittest.main()

