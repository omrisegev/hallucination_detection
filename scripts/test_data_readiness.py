#!/usr/bin/env python3
"""Known-answer tests for the dataset-only readiness layer."""

from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spectral_utils.data_readiness import (
    READY,
    Audit,
    audit_candidate_cache,
    audit_rag_conditions,
    registry_payload,
    read_jsonl,
    resolve_lfs_path,
    restricted_pickle,
    sha256_file,
    validate_hle_judge_rows,
)


def trace_row(response_id: str, condition: str, tokens=(1, 2, 3)) -> dict:
    n = len(tokens)
    return {
        "response_id": response_id,
        "source_id": "source-1",
        "task_type": "QA",
        "condition": condition,
        "response_label": True,
        "gen_token_ids": list(tokens),
        "token_entropies": [1.0] * n,
        "token_spilled_energies": [2.0] * n,
        "token_logsumexp": [3.0] * n,
        "top_k_logprobs": {
            "ids": np.tile(np.asarray([1, 2]), (n, 1)),
            "logprobs": np.tile(np.log(np.asarray([0.7, 0.2])), (n, 1)),
        },
    }


class DataReadinessTests(unittest.TestCase):
    def test_lfs_pointer_is_resolved_without_replacing_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            payload = b"immutable cache bytes"
            oid = hashlib.sha256(payload).hexdigest()
            obj = repo / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
            obj.parent.mkdir(parents=True)
            obj.write_bytes(payload)
            pointer = repo / "cache.pkl"
            pointer_text = (
                "version https://git-lfs.github.com/spec/v1\n"
                f"oid sha256:{oid}\nsize {len(payload)}\n"
            )
            pointer.write_text(pointer_text)
            resolved, observed_oid = resolve_lfs_path(repo, pointer)
            self.assertEqual(resolved, obj)
            self.assertEqual(observed_oid, oid)
            self.assertEqual(pointer.read_text(), pointer_text)

    def test_restricted_loader_blocks_arbitrary_globals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.pkl"
            with path.open("wb") as handle:
                pickle.dump({"path": Path("not-allowed")}, handle)
            with self.assertRaises(pickle.UnpicklingError):
                restricted_pickle(path)

    def test_rag_audit_checks_condition_tokens_and_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            cache = repo / "cache.pkl"
            manifest = repo / "manifest.json"
            rows = {
                "7::full": trace_row("7", "full"),
                "7::noctx": trace_row("7", "noctx"),
                "7::loo_0": trace_row("7", "loo_0"),
            }
            with cache.open("wb") as handle:
                pickle.dump(rows, handle)
            manifest.write_text(json.dumps({"model": "tiny"}))
            before = sha256_file(cache)
            audit = audit_rag_conditions(
                repo, "tiny_rag", "Tiny RAG", "cache.pkl", "manifest.json", 1, 3
            )
            self.assertEqual(audit.status, READY)
            self.assertTrue(all(audit.checks.values()))
            self.assertEqual(sha256_file(cache), before)

    def test_rag_audit_rejects_changed_answer_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            rows = {
                "7::full": trace_row("7", "full"),
                "7::noctx": trace_row("7", "noctx", tokens=(1, 9, 3)),
            }
            with (repo / "cache.pkl").open("wb") as handle:
                pickle.dump(rows, handle)
            (repo / "manifest.json").write_text("{}")
            audit = audit_rag_conditions(
                repo, "tiny_rag", "Tiny RAG", "cache.pkl", "manifest.json", 1, 2
            )
            self.assertFalse(audit.checks["condition_token_identity"])
            self.assertEqual(audit.status, "BLOCKED")

    def test_candidate_audit_preserves_bem_label_and_is_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            candidate = trace_row("", "")
            candidate.update({"bem_correct": True, "bem_score": 0.9})
            rows = {0: {"question": "q", "gold_row": {}, "candidates": [candidate]}}
            with (repo / "cache.pkl").open("wb") as handle:
                pickle.dump(rows, handle)
            (repo / "manifest.json").write_text(json.dumps({
                "model": "BEM", "threshold": 0.8,
            }))
            audit = audit_candidate_cache(
                repo, "tiny_bem", "Tiny BEM", "cache.pkl", "manifest.json", 1
            )
            self.assertEqual(audit.status, READY)
            self.assertEqual(audit.balance["positive"], 1)
            self.assertTrue(all(audit.checks.values()))

    def test_registry_fingerprint_is_order_sensitive_to_dataset_identity_not_dict_order(self):
        a = Audit("a", "A", "kind", READY, file_hashes={"x": "1", "y": "2"})
        b = Audit("a", "A", "kind", READY, file_hashes={"y": "2", "x": "1"})
        left = registry_payload(Path("/repo"), [a])
        right = registry_payload(Path("/repo"), [b])
        self.assertEqual(left["registry_fingerprint"], right["registry_fingerprint"])

    def test_hle_judgments_require_complete_aligned_provenance(self):
        queue = [{
            "row_key": index,
            "id": f"id-{index}",
            "answer_type": "multipleChoice" if index % 2 else "exactMatch",
            "provisional_rouge_label": index == 0,
        } for index in range(2158)]
        judged = [{
            "row_key": index,
            "id": f"id-{index}",
            "extracted_final_answer": "A",
            "reasoning": "Equivalent to the reference.",
            "correct": "yes" if index == 0 else "no",
            "confidence": 90,
            "judge_model": "gpt-5.6-sol",
            "judge_reasoning_effort": "xhigh",
            "judge_protocol": "official HLE criteria",
        } for index in range(2158)]
        counts, provenance, checks = validate_hle_judge_rows(queue, judged)
        self.assertTrue(all(checks.values()))
        self.assertEqual(counts["correct"], 1)
        self.assertEqual(counts["agreement_with_provisional_rouge"], 1.0)
        self.assertEqual(provenance["judge_models"], ["gpt-5.6-sol"])
        judged[4]["id"] = "wrong"
        _, _, broken = validate_hle_judge_rows(queue, judged)
        self.assertFalse(broken["source_alignment"])

    def test_jsonl_reader_preserves_unicode_line_separator_inside_string(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps({"text": "before\u2028after"}) + "\n", encoding="utf-8")
            rows = read_jsonl(path)
            self.assertEqual(rows, [{"text": "before\u2028after"}])


if __name__ == "__main__":
    unittest.main()
