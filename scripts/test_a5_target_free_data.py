#!/usr/bin/env python3
"""Development tests for the Phase A5 target firewall and content grouping."""

from __future__ import annotations

import unittest

import numpy as np

from spectral_utils.a5_target_free_data import (
    A0_SOURCE_ENVIRONMENTS,
    CORE_FEATURES,
    RawSourceArtifact,
    SourceExpectation,
    assert_no_target_fields,
    canonical_item_id,
    canonical_revision,
    connected_content_groups,
    dataset_family,
    normalize_question,
    question_content_hash,
    sanitize_source_row,
    select_primary_responses,
    build_target_free_boundary,
    FROZEN_A0_SOURCE_SPECS,
    load_frozen_raw_sources,
)
import spectral_utils.a5_target_free_data as target_free_module


def candidate(**extra):
    length = 20
    lp = np.tile(np.asarray([-0.1, -0.8, -1.4, -2.0, -2.7, -3.1]), (length, 1))
    value = {
        "token_entropies": np.linspace(0.2, 1.0, length),
        "token_spilled_energies": np.linspace(0.1, 0.7, length),
        "token_logsumexp": np.linspace(3.0, 4.0, length),
        "top_k_logprobs": {"logprobs": lp, "ids": np.zeros_like(lp, dtype=int)},
    }
    value.update(extra)
    return value


class Poison:
    def __bool__(self):
        raise AssertionError("target value was accessed")

    def __array__(self, *args, **kwargs):
        raise AssertionError("target value was converted")


class TrackingMapping(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed = []

    def get(self, key, default=None):
        self.accessed.append(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.accessed.append(key)
        return super().__getitem__(key)


class FirewallTests(unittest.TestCase):
    def test_sanitizer_never_indexes_candidate_targets(self):
        clean = {"question": "What is 2 + 2?", "candidates": [candidate()]}
        poisoned = {
            "question": "What is 2 + 2?",
            "gold_row": {"answer": Poison()},
            "label": Poison(),
            "candidates": [candidate(label=Poison(), full_text=Poison(), answer=Poison())],
        }
        left = sanitize_source_row(
            environment_id="e1", dataset="gsm8k", split="test",
            item_group_id=4, row=clean,
        )[0]
        right = sanitize_source_row(
            environment_id="e1", dataset="gsm8k", split="test",
            item_group_id=4, row=poisoned,
        )[0]
        np.testing.assert_array_equal(left.features, right.features)
        self.assertEqual(left.trace_length, right.trace_length)
        self.assertIn("label", right.unexpected_candidate_keys)
        self.assertIn("answer", right.unexpected_candidate_keys)

    def test_mismatched_or_nonfinite_telemetry_fails_closed(self):
        bad_length = candidate()
        bad_length["token_logsumexp"] = bad_length["token_logsumexp"][:-1]
        with self.assertRaisesRegex(ValueError, "lengths disagree"):
            sanitize_source_row(
                environment_id="e1", dataset="gsm8k", split="test",
                item_group_id=1,
                row={"question": "valid", "candidates": [bad_length]},
            )
        bad_value = candidate()
        bad_value["token_entropies"][2] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            sanitize_source_row(
                environment_id="e1", dataset="gsm8k", split="test",
                item_group_id=1,
                row={"question": "valid", "candidates": [bad_value]},
            )

    def test_empty_question_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            sanitize_source_row(
                environment_id="e1", dataset="gsm8k", split="test",
                item_group_id=1,
                row={"question": " \n ", "candidates": [candidate()]},
            )

    def test_exact_roster_and_no_trace_length_fit_coordinate(self):
        self.assertEqual(len(CORE_FEATURES), 17)
        self.assertEqual(len(set(CORE_FEATURES)), 17)
        self.assertNotIn("trace_length", CORE_FEATURES)

    def test_public_payload_rejects_target_like_keys(self):
        assert_no_target_fields({"X": np.ones((3, 2)), "environment_id": ["a"]})
        with self.assertRaises(ValueError):
            assert_no_target_fields({"X": np.ones((3, 2)), "final_answer_correct": [1]})

    def test_isolated_crop_outputs_only_telemetry_and_never_reads_target(self):
        import spectral_utils.a5_target_free_data as module
        value = TrackingMapping(candidate(
            full_text="answer\nQuestion: fabricated",
            token_offsets=[(i, i + 1) for i in range(20)],
            label=Poison(), correct=Poison(), gold=Poison(),
        ))
        cropped = module._cropped_telemetry_only(value)
        self.assertEqual(set(cropped), set(module.TELEMETRY_KEYS))
        self.assertTrue(set(value.accessed).issubset(
            set(module.TELEMETRY_KEYS) | {"full_text", "token_offsets"}
        ))
        self.assertFalse({"label", "correct", "gold"} & set(value.accessed))


class GroupBoundaryTests(unittest.TestCase):
    def test_normalization_preserves_punctuation_and_case(self):
        self.assertEqual(normalize_question(" x + 1\n  = ? "), "x + 1 = ?")
        self.assertNotEqual(question_content_hash("x + 1"), question_content_hash("x - 1"))
        self.assertNotEqual(question_content_hash("X + 1"), question_content_hash("x + 1"))

    def test_gsm8k_revision_is_shared_across_sources(self):
        self.assertEqual(canonical_revision("gsm8k", "test"), "gsm8k")
        self.assertEqual(
            canonical_item_id("gsm8k", "test", "7"),
            canonical_item_id(canonical_revision("gsm8k", "test"), "test", "7"),
        )

    def test_trivia_revisions_are_distinct_but_family_is_shared(self):
        from spectral_utils.a5_target_free_data import dataset_family
        revisions = {
            canonical_revision(name, "validation")
            for name in ("trivia_qa", "trivia_qa_wiki", "trivia_qa_rougel")
        }
        self.assertEqual(len(revisions), 3)
        self.assertEqual(
            {dataset_family(name) for name in revisions},
            {"triviaqa"},
        )

    def test_connected_components_join_id_or_content_transitively(self):
        rows = []
        for environment, item, question in (
            ("a", 0, "same question"),
            ("b", 0, "same question"),
            ("c", 4, "same question"),
        ):
            rows.extend(sanitize_source_row(
                environment_id=environment, dataset="gsm8k", split="test",
                item_group_id=item,
                row={"question": question, "candidates": [candidate()]},
            ))
        groups = connected_content_groups(rows)
        self.assertEqual(len(set(groups.values())), 1)

    def test_conflicting_question_for_canonical_id_fails_closed(self):
        rows = []
        for environment, question in (("a", "question A"), ("b", "question B")):
            rows.extend(sanitize_source_row(
                environment_id=environment, dataset="gsm8k", split="test",
                item_group_id=0,
                row={"question": question, "candidates": [candidate()]},
            ))
        with self.assertRaisesRegex(ValueError, "CLOSE_INVALID_GLOBAL_ITEM_BOUNDARY"):
            connected_content_groups(rows)

    def test_primary_selects_one_per_environment_content_component(self):
        rows = []
        for ordinal in range(10):
            rows.extend(sanitize_source_row(
                environment_id="k10", dataset="gsm8k", split="test",
                item_group_id=12,
                row={"question": "shared", "candidates": [candidate()]},
            ))
            # Give each one a unique source ordinal while preserving content.
            rows[-1] = type(rows[-1])(**{
                **rows[-1].__dict__, "candidate_ordinal": ordinal,
            })
        groups = connected_content_groups(rows)
        primary = select_primary_responses(rows, groups)
        self.assertEqual(len(primary), 1)

    def test_duplicate_boundary_key_fails_closed(self):
        row = sanitize_source_row(
            environment_id="e1", dataset="gsm8k", split="test",
            item_group_id=2,
            row={"question": "unique", "candidates": [candidate()]},
        )[0]
        with self.assertRaisesRegex(ValueError, "duplicate A5 boundary key"):
            connected_content_groups([row, row])


class FrozenBoundaryBuilderTests(unittest.TestCase):
    @staticmethod
    def boundary(*, count_override=None, conflict=False):
        raw_sources, expectations = {}, []
        for index, environment in enumerate(A0_SOURCE_ENVIRONMENTS):
            dataset = "gsm8k" if index < 10 else (
                "math500" if 11 <= index <= 15 else "truthfulqa"
            )
            split = "test" if dataset in {"gsm8k", "math500"} else "validation"
            # Same canonical key across a revision must carry the same question.
            question = f"shared {dataset} question"
            if conflict and index == 1:
                question = "conflicting GSM question"
            rows = {0: {"question": question, "candidates": [candidate()]}}
            metadata = dict(
                source_sha256=f"source-{index}", source_size=100 + index,
                source_mtime="2026-08-13T00:00:00Z",
                manifest_sha256=f"manifest-{index}",
            )
            raw_sources[environment] = RawSourceArtifact(rows=rows, **metadata)
            expectations.append(SourceExpectation(
                environment_id=environment, dataset=dataset, split=split,
                dataset_family=(dataset if dataset != "truthfulqa" else "truthfulqa"),
                expected_admitted_count=(
                    count_override if count_override is not None and index == 0 else 1
                ),
                admission_mode="complete_h16", **metadata,
            ))
        return raw_sources, expectations

    def test_exact_roster_counts_hashes_and_overlap_audit(self):
        raw_sources, expectations = self.boundary()
        boundary = target_free_module._build_target_free_boundary(
            raw_sources, expectations, enforce_frozen=False
        )
        self.assertEqual(boundary.audit["source_count"], 23)
        self.assertEqual(boundary.audit["all_admitted_count"], 23)
        self.assertEqual(len(boundary.audit["all_admitted_key_sha256"]), 64)
        gsm_overlap = [
            row for row in boundary.audit["overlap_rows"]
            if row["source_a"] == A0_SOURCE_ENVIRONMENTS[0]
            and row["source_b"] == A0_SOURCE_ENVIRONMENTS[1]
        ][0]
        self.assertEqual(gsm_overlap["expected_canonical_id_overlap"], 1)
        self.assertEqual(gsm_overlap["observed_shared_components"], 1)

    def test_population_count_mismatch_closes(self):
        raw_sources, expectations = self.boundary(count_override=2)
        with self.assertRaisesRegex(ValueError, "CLOSE_A0_POPULATION_MISMATCH"):
            target_free_module._build_target_free_boundary(
                raw_sources, expectations, enforce_frozen=False
            )

    def test_same_revision_question_conflict_closes(self):
        raw_sources, expectations = self.boundary(conflict=True)
        with self.assertRaisesRegex(ValueError, "CLOSE_INVALID_GLOBAL_ITEM_BOUNDARY"):
            target_free_module._build_target_free_boundary(
                raw_sources, expectations, enforce_frozen=False
            )

    def test_source_hash_mismatch_closes_before_rows(self):
        raw_sources, expectations = self.boundary()
        first = A0_SOURCE_ENVIRONMENTS[0]
        artifact = raw_sources[first]
        raw_sources[first] = RawSourceArtifact(
            **{**artifact.__dict__, "source_sha256": "tampered"}
        )
        with self.assertRaisesRegex(ValueError, "CLOSE_SOURCE_ARTIFACT_MISMATCH"):
            target_free_module._build_target_free_boundary(
                raw_sources, expectations, enforce_frozen=False
            )

    def test_cropped_mode_keeps_short_answer_population(self):
        raw_sources, expectations = self.boundary()
        environment = A0_SOURCE_ENVIRONMENTS[20]
        short = candidate()
        for key in ("token_entropies", "token_spilled_energies", "token_logsumexp"):
            short[key] = short[key][:3]
        short["top_k_logprobs"] = {
            "logprobs": short["top_k_logprobs"]["logprobs"][:3],
            "ids": short["top_k_logprobs"]["ids"][:3],
        }
        raw_sources[environment] = RawSourceArtifact(
            **{
                **raw_sources[environment].__dict__,
                "rows": {0: {"question": "shared truthfulqa question", "candidates": [short]}},
            }
        )
        value = expectations[20]
        expectations[20] = SourceExpectation(
            **{**value.__dict__, "admission_mode": "cropped_all_rows"}
        )
        boundary = target_free_module._build_target_free_boundary(
            raw_sources, expectations, enforce_frozen=False
        )
        kept = [row for row in boundary.all_admitted_rows if row.environment_id == environment]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].trace_length, 3)

    def test_production_builder_rejects_caller_controlled_registry(self):
        raw_sources, expectations = self.boundary()
        with self.assertRaisesRegex(ValueError, "frozen registry mismatch"):
            build_target_free_boundary(raw_sources, expectations)

    def test_frozen_registry_has_exact_metadata_and_valid_hashes(self):
        self.assertEqual(
            tuple(value.environment_id for value in FROZEN_A0_SOURCE_SPECS),
            A0_SOURCE_ENVIRONMENTS,
        )
        self.assertEqual(len(FROZEN_A0_SOURCE_SPECS), 23)
        for value in FROZEN_A0_SOURCE_SPECS:
            self.assertRegex(value.source_sha256, r"^[0-9a-f]{64}$")
            self.assertRegex(value.manifest_sha256, r"^[0-9a-f]{64}$")
            self.assertGreater(value.source_size, 0)
            self.assertEqual(value.dataset_family, dataset_family(value.dataset))

    def test_loader_fails_before_unpickle_on_manifest_or_source_tamper(self):
        import hashlib
        import json
        from pathlib import Path
        import pickle
        import tempfile

        # Exercise the verifier with a temporary one-source registry by
        # replacing the module constant; the payload would raise if unpickled.
        import spectral_utils.a5_target_free_data as module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "dataset_cache" / "repgrid" / "fake"
            cell.mkdir(parents=True)
            manifest = {"dataset": "gsm8k", "split": "test",
                        "cells": [{"pkl": "raw.pkl"}]}
            manifest_bytes = json.dumps(manifest).encode()
            (cell / "manifest.json").write_bytes(manifest_bytes)
            (cell / "raw.pkl").write_bytes(b"not a pickle")
            spec = module.FrozenSourceSpec(
                "fake", "gsm8k", "test", "gsm8k", 1, "complete_h16",
                "dataset_cache/repgrid/fake/raw.pkl",
                hashlib.sha256(b"different").hexdigest(), len(b"not a pickle"),
                hashlib.sha256(manifest_bytes).hexdigest(),
            )
            original = module.FROZEN_A0_SOURCE_SPECS
            try:
                module.FROZEN_A0_SOURCE_SPECS = (spec,)
                with self.assertRaisesRegex(ValueError, "sha256"):
                    load_frozen_raw_sources(root)
            finally:
                module.FROZEN_A0_SOURCE_SPECS = original


if __name__ == "__main__":
    unittest.main()
