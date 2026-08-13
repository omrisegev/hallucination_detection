#!/usr/bin/env python3
"""Known-answer tests for the RAGTruth evidence-contrast experiment."""

from __future__ import annotations

import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spectral_utils.laplacian_upcr import dufs_soft_gates, laplacian_iu_path
from spectral_utils.ragtruth_evidence_contrast import (
    ConditionTrace,
    RagDataset,
    RagResponse,
    adapt_cache,
    approximate_topk_jsd,
    build_feature_tables,
    load_cache,
    topk_plus_tail_entropy,
)
from ragtruth_ec_experiment import (
    _bootstrap_indices,
    _metric_bundle,
    final_decision,
)


class CharacterTokenizer:
    """Tiny deterministic tokenizer used only for adapter tests."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        output = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            output["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return output


def trace(condition, target_lp, entropy, ids, logprobs):
    return ConditionTrace(
        condition=condition,
        prompt_len=10,
        token_ids=np.arange(len(target_lp)),
        target_logprob=np.asarray(target_lp, dtype=float),
        entropy=np.asarray(entropy, dtype=float),
        logsumexp=np.ones(len(target_lp), dtype=float),
        top_ids=np.asarray(ids, dtype=int),
        top_logprobs=np.asarray(logprobs, dtype=float),
    )


def cache_row(response_id, condition, text, *, response_label=False, spans=None):
    n = len(text)
    ids = np.tile(np.asarray([1, 2, 3]), (n, 1))
    logprobs = np.tile(np.log(np.asarray([0.6, 0.25, 0.1])), (n, 1))
    return {
        "response_id": str(response_id),
        "source_id": "source-1",
        "condition": condition,
        "task_type": "QA",
        "source": "unit-test",
        "generator_model": "generator",
        "quality": "good",
        "response_label": bool(response_label),
        "span_labels": [{"label_type": "Evident Baseless Info"}] if spans else [],
        "span_token_spans": list(spans or []),
        "align_diag": {"n_labels": len(spans or []), "n_unmapped": 0, "n_tokens": n},
        "prompt_len": 10,
        "gen_token_ids": [ord(char) for char in text],
        "token_entropies": [1.0] * n,
        "token_spilled_energies": [1.0 if condition == "full" else 2.0] * n,
        "token_logsumexp": [3.0] * n,
        "top_k_logprobs": {"ids": ids, "logprobs": logprobs},
    }


class EvidenceContrastTests(unittest.TestCase):
    def test_approximate_jsd_is_symmetric_bounded_and_zero_on_identity(self):
        ids_a = [1, 2, 3]
        lp_a = np.log([0.6, 0.2, 0.1])
        ids_b = [1, 4, 3]
        lp_b = np.log([0.2, 0.5, 0.1])
        self.assertEqual(approximate_topk_jsd(ids_a, lp_a, ids_a, lp_a), 0.0)
        ab = approximate_topk_jsd(ids_a, lp_a, ids_b, lp_b)
        ba = approximate_topk_jsd(ids_b, lp_b, ids_a, lp_a)
        self.assertAlmostEqual(ab, ba, places=14)
        self.assertGreaterEqual(ab, 0.0)
        self.assertLessEqual(ab, math.log(2.0))

    def test_feature_formulas_have_registered_direction(self):
        ids = [[1, 2], [1, 2], [1, 2]]
        lp = np.log([[0.7, 0.2], [0.6, 0.3], [0.8, 0.1]])
        full = trace("full", [-1.0, -2.0, -3.0], [1.0, 1.0, 1.0], ids, lp)
        noctx = trace("noctx", [-2.0, -4.0, -6.0], [2.0, 2.0, 2.0], ids, lp)
        loo = trace("loo_0", [-1.5, -3.0, -4.5], [1.5, 1.5, 1.5], ids, lp)
        response = RagResponse(
            "1", "s", "QA", "test", "g", "good", "abc",
            {"full": full, "noctx": noctx, "loo_0": loo}, (),
        )
        dataset = RagDataset((response,), "hash", "tokenizer")
        table = build_feature_tables(dataset)["full_response"]
        row = dict(zip(table.feature_names, table.values[0]))
        self.assertAlmostEqual(row["mean_context_gap"], 2.0)
        self.assertAlmostEqual(row["max_loo_mean_drop"], 1.0)
        self.assertAlmostEqual(row["top2_loo_mean_drop"], 1.0)
        self.assertAlmostEqual(row["mean_positive_loo_drop"], 1.0)
        self.assertAlmostEqual(row["fraction_tokens_positive_best_drop"], 1.0)
        # The registered entropy is reconstructed from top-k plus one tail
        # category, so identical saved distributions have zero change even
        # when the synthetic full-vocabulary telemetry differs.
        self.assertAlmostEqual(row["mean_top50_tail_entropy_increase_noctx"], 0.0)

    def test_topk_plus_tail_entropy_has_a_hand_computed_value(self):
        logprobs = np.log(np.asarray([[0.5, 0.25]]))
        observed = topk_plus_tail_entropy(logprobs)[0]
        expected = -sum(value * math.log(value) for value in (0.5, 0.25, 0.25))
        self.assertAlmostEqual(observed, expected, places=14)

    def test_adapter_splits_labels_and_is_order_invariant(self):
        text = "Abc. Def!"
        rows = {}
        for condition in ("full", "noctx", "loo_0"):
            rows[f"7::{condition}"] = cache_row(
                "7", condition, text, response_label=True, spans=[(5, 8)]
            )
        official = {"id": "7", "source_id": "source-1", "response": text}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response_path = root / "response.jsonl"
            response_path.write_text(json.dumps(official) + "\n", encoding="utf-8")
            paths = []
            for name, items in (("a", list(rows.items())), ("b", list(reversed(rows.items())))):
                path = root / f"{name}.pkl"
                with path.open("wb") as handle:
                    pickle.dump(dict(items), handle)
                paths.append(path)
            (root / "manifest.json").write_text(json.dumps({
                "split": "test", "logprob_top_k": 3,
                "stats": {
                    "n_items": 3, "n_responses": 1,
                    "by_condition": {"full": 1, "noctx": 1, "loo_0": 1},
                    "by_task_type": {"QA": 3},
                },
            }), encoding="utf-8")
            data_a, labels_a, audit_a = adapt_cache(paths[0], response_path, CharacterTokenizer())
            data_b, labels_b, _ = adapt_cache(paths[1], response_path, CharacterTokenizer())
            self.assertFalse(hasattr(data_a, "labels"))
            self.assertNotIn("response_label", data_a.responses[0].__dataclass_fields__)
            self.assertEqual(
                data_a.responses[0].token_offsets,
                tuple((index, index + 1) for index in range(len(text))),
            )
            self.assertTrue(labels_a.response["7"].hallucinated)
            self.assertEqual(sum(x.hallucinated for x in labels_a.sentence.values()), 1)
            table_a = build_feature_tables(data_a)["full_sentence"]
            table_b = build_feature_tables(data_b)["full_sentence"]
            self.assertEqual(table_a.sample_ids, table_b.sample_ids)
            np.testing.assert_array_equal(table_a.values, table_b.values)
            self.assertEqual(labels_a, labels_b)
            self.assertTrue(audit_a["sidecar_manifest_validated"])

    def test_adapter_rejects_token_misalignment(self):
        text = "abcdef"
        rows = {
            "7::full": cache_row("7", "full", text),
            "7::noctx": cache_row("7", "noctx", text),
        }
        rows["7::noctx"]["gen_token_ids"][2] = 999
        official = {"id": "7", "source_id": "source-1", "response": text}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "cache.pkl"
            response_path = root / "response.jsonl"
            with cache_path.open("wb") as handle:
                pickle.dump(rows, handle)
            response_path.write_text(json.dumps(official) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tokens differ"):
                adapt_cache(cache_path, response_path, CharacterTokenizer())

    def test_restricted_loader_blocks_globals(self):
        class Unsafe:
            pass
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.pkl"
            with path.open("wb") as handle:
                with self.assertRaises(AttributeError):
                    pickle.dump({"x": Unsafe()}, handle)
            # A top-level pathlib object is picklable but not allow-listed.
            with path.open("wb") as handle:
                pickle.dump({"x": Path("secret")}, handle)
            with self.assertRaises(pickle.UnpicklingError):
                load_cache(path)

    def test_lambda_zero_is_exact_iu_anchor(self):
        rng = np.random.default_rng(4)
        F = rng.normal(size=(6, 80))
        path = laplacian_iu_path(F, [0.0, 0.1], k=5)
        np.testing.assert_array_equal(path[0.0].w, path[0.0].baseline.w)
        self.assertLess(path[0.0].diagnostics["zero_equation_weight_error"], 1e-10)

    def test_group_bootstrap_never_splits_a_source(self):
        groups = np.asarray(["a", "a", "b", "b", "b", "c"])
        group_sizes = {group: int(np.sum(groups == group)) for group in set(groups)}
        for indexes in _bootstrap_indices(groups, 30, 17):
            counts = {group: int(np.sum(groups[indexes] == group)) for group in set(groups)}
            for group, count in counts.items():
                self.assertEqual(count % group_sizes[group], 0)

    def test_metrics_are_identical_when_evaluation_rows_are_reordered(self):
        y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
        score = np.asarray([0.1, 0.9, 0.4, 0.6, 0.2, 0.8, 0.3, 0.7])
        groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
        original, _ = _metric_bundle(y, {"m": score}, groups, 40, 9)
        order = np.asarray([6, 2, 0, 7, 3, 1, 5, 4])
        reordered, _ = _metric_bundle(y[order], {"m": score[order]}, groups[order], 40, 9)
        self.assertEqual(original, reordered)

    def test_final_decision_requires_a_dufs_gain_over_iu(self):
        rows = []
        for table, task, reference, delta, low, high in (
            ("full_sentence", "ALL", "gasp_top50", 0.03, 0.02, 0.04),
            ("full_sentence", "ALL", "ec_iu_pcr", -0.001, -0.002, -0.0001),
            ("full_sentence", "QA", "gasp_top50", 0.02, 0.01, 0.03),
            ("full_sentence", "Data2txt", "gasp_top50", 0.01, 0.001, 0.02),
            ("full_response", "ALL", "gasp_top50", 0.02, -0.005, 0.04),
        ):
            rows.append({
                "table": table, "task": task, "challenger": "ec_dufs_liu",
                "reference": reference, "delta_auroc": delta,
                "ci_low": low, "ci_high": high,
            })
        decision = final_decision([], rows)
        self.assertFalse(decision["success"])
        self.assertFalse(decision["checks"][
            "dufs_laplacian_improvement_over_iu_interval_excludes_zero"
        ])

    def test_optional_dufs_history_does_not_change_gates(self):
        rng = np.random.default_rng(31)
        F = rng.normal(size=(4, 24))
        plain, plain_diag = dufs_soft_gates(F, seeds=(5,), epochs=2)
        recorded, recorded_diag = dufs_soft_gates(
            F, seeds=(5,), epochs=2, return_history=True
        )
        np.testing.assert_array_equal(plain, recorded)
        self.assertNotIn("training_history", plain_diag)
        self.assertEqual(np.asarray(recorded_diag["training_history"]).shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
