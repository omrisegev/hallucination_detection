"""Fail-closed tests for the frozen A6-S0a execution primitives."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from spectral_utils.a6_s0a import (
    MODEL_IDENTITIES,
    NATURAL_COHORTS,
    POPQA_ROWS,
    QUARTET_POPULATIONS,
    SCORER_IDS,
    SnapshotManifest,
    assert_no_a6_llama_payloads,
    build_contextual_input_evidence,
    contextual_quartet_evidence,
    build_null_strata,
    build_s0a_natural_prompts,
    build_s0a_quartets,
    canonical_json_bytes,
    first64,
    frozen_natural_slots,
    frozen_quartet_slots,
    future_llama_sidecar_schema,
    inner_fold_manifest,
    popqa_opaque_reservation,
    prepare_content_addressed_tokenizer_snapshot,
    public_natural_prompt_record,
    public_quartet_record,
    natural_record_from_public,
    quartet_record_from_public,
    sanitize_natural_prompt_row,
    sha256_lsb,
    sha256_file,
    verify_content_addressed_snapshot,
    verify_null_strata,
)


class FakeContextTokenizer:
    """Subword-like fast tokenizer with deterministic reversible offsets."""

    is_fast = True
    eos_token_id = 0
    pad_token_id = 0

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def apply_chat_template(
        self, messages, *, tokenize=False, add_generation_prompt=False, **kwargs,
    ):
        if self.kind == "qwen":
            if kwargs != {"enable_thinking": False}:
                raise AssertionError("Qwen template kwargs drift")
        elif kwargs:
            raise AssertionError("Llama received nondefault template kwargs")
        text = f"<user>{messages[0]['content']}</user><assistant>"
        if len(messages) == 2:
            text += f"{messages[1]['content']}</assistant>"
        elif not add_generation_prompt:
            text += "</assistant>"
        return self(text)["input_ids"] if tokenize else text

    def __call__(self, text, **kwargs):
        matches = list(re.finditer(
            r"\s*(?:[A-Za-z_]+|[-+]?\d+(?:/[1-9]\d*)?|[^\w\s])", text,
        ))
        output = {
            "input_ids": [
                sum(map(ord, match.group())) % 65_521 for match in matches
            ]
        }
        if kwargs.get("return_offsets_mapping"):
            output["offset_mapping"] = [match.span() for match in matches]
        return output


class ResponseDependentBoundaryTokenizer(FakeContextTokenizer):
    """Simulate a BPE merge that consumes one prefix token for one response."""

    def __init__(self, kind: str, response_marker: str) -> None:
        super().__init__(kind)
        self.response_marker = response_marker

    def __call__(self, text, **kwargs):
        output = super().__call__(text, **kwargs)
        if "</assistant>" not in text or self.response_marker not in text:
            return output
        start = text.index("<assistant>") + len("<assistant>")
        offsets = output.get("offset_mapping")
        if offsets is None:
            # ``apply_chat_template(..., tokenize=True)`` must follow the same
            # token sequence as the offset-bearing path.
            offsets = super().__call__(text, return_offsets_mapping=True)["offset_mapping"]
        prefix_index = max(index for index, (_, right) in enumerate(offsets) if right <= start)
        response_index = next(index for index, (left, right) in enumerate(offsets)
                              if right > start and left < start + len(self.response_marker))
        left, _ = offsets[prefix_index]
        _, right = offsets[response_index]
        output["input_ids"].pop(prefix_index)
        if kwargs.get("return_offsets_mapping"):
            output["offset_mapping"].pop(prefix_index)
            adjusted_index = response_index - (prefix_index < response_index)
            output["offset_mapping"][adjusted_index] = (left, right)
        return output


TOKENIZERS = {
    "qwen3-4b": FakeContextTokenizer("qwen"),
    "qwen3-8b": FakeContextTokenizer("qwen"),
    "llama31-8b": FakeContextTokenizer("llama"),
}


class PoisonMapping(dict):
    def __getitem__(self, key):
        if key == "label":
            raise AssertionError("forbidden target property was touched")
        return super().__getitem__(key)


class A6S0aTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.quartets = build_s0a_quartets(TOKENIZERS)
        cls.inner = inner_fold_manifest(cls.quartets)
        cls.null_strata = build_null_strata(cls.quartets, cls.inner)

    def test_exact_quartet_schedule_balance_seed_and_derived_quota(self) -> None:
        slots = frozen_quartet_slots()
        self.assertEqual(len(slots), 1_800)
        self.assertEqual(tuple(dict.fromkeys(value.population_id for value in slots)),
                         QUARTET_POPULATIONS)
        first = slots[0]
        self.assertEqual(
            first.seed,
            first64(b"a6-s0-slot-v2\0" + first.slot_id.encode("utf-8")),
        )
        for population in QUARTET_POPULATIONS:
            selected = [value for value in slots if value.population_id == population]
            self.assertEqual(len(selected), 900)
            self.assertEqual(sum(value.subdomain == "derived-answer" for value in selected), 60)
            for domain in ("arithmetic", "relational", "finite_logic"):
                for mutation in ("value_leaf", "relation_operator", "constraint_condition"):
                    for grammar in ("short", "certificate"):
                        cell = [
                            value for value in selected if value.domain == domain
                            and value.mutation_family == mutation
                            and value.response_grammar == grammar
                        ]
                        self.assertEqual(len(cell), 50)
                        self.assertEqual([sum(value.outer_fold == fold for value in cell)
                                          for fold in range(5)], [10] * 5)

    def test_contextual_quartets_and_inner_folds_are_complete(self) -> None:
        self.assertEqual(len(self.quartets), 1_800)
        self.assertTrue(all(value.attempt_ledger[-1].status == "ACCEPTED"
                            for value in self.quartets))
        self.assertEqual(len(self.inner), 7_200)
        for outer in range(5):
            for inner in range(5):
                self.assertEqual(sum(
                    row_outer == outer and row_inner == inner
                    for row_outer, _, row_inner in self.inner
                ), 2 * 18 * 8)

    def test_construction_callback_checkpoints_each_accepted_unit_immediately(self) -> None:
        observed = []
        records = build_s0a_quartets(
            TOKENIZERS, slots=frozen_quartet_slots()[:3],
            on_record=lambda index, record: observed.append((index, record.slot.slot_id)),
        )
        self.assertEqual(observed, [
            (index, record.slot.slot_id) for index, record in enumerate(records)
        ])

    def test_null_strata_replay_and_every_partition_count_at_least_four(self) -> None:
        self.assertEqual(len(self.null_strata), 36)
        self.assertTrue(all(
            count >= 4 for cell in self.null_strata
            for _, _, count in cell.final_partition_counts
        ))
        self.assertTrue(all(
            len(row) == 7 and 0 <= row[3] < 50 and 0 <= row[4] < 50
            for cell in self.null_strata for row in cell.group_ranks
        ))
        verify_null_strata(self.quartets, self.inner, self.null_strata)
        tampered = list(self.null_strata)
        tampered[0] = replace(tampered[0], merges=tampered[0].merges[:-1])
        with self.assertRaisesRegex(ValueError, "replay"):
            verify_null_strata(self.quartets, self.inner, tampered)

    def test_natural_schedule_is_three_exact_2000_prompt_manifests(self) -> None:
        slots = frozen_natural_slots()
        self.assertEqual(len(slots), 6_000)
        for cohort in NATURAL_COHORTS:
            selected = [value for value in slots if value.cohort_id == cohort]
            self.assertEqual(len(selected), 2_000)
            self.assertEqual(sum(value.subdomain == "derived-answer" for value in selected), 150)
            self.assertEqual([sum(value.outer_fold == fold for value in selected)
                              for fold in range(5)], [400] * 5)
        record = build_s0a_natural_prompts(TOKENIZERS, slots=slots[:1])[0]
        expected_side = hashlib.sha256(
            ("a6-natural-side-v1\0" + slots[0].slot_id).encode("utf-8")
        ).digest()[-1] & 1
        self.assertEqual(
            sha256_lsb(("a6-natural-side-v1\0" + slots[0].slot_id).encode("utf-8")),
            expected_side,
        )
        public = asdict(record.row)
        self.assertEqual(sanitize_natural_prompt_row(public), record.row)
        self.assertFalse(any(
            fragment in key.casefold() for key in public
            for fragment in ("answer", "response", "correct", "label", "target")
        ))

    def test_qwen_firewall_rejects_target_keys_without_touching_value(self) -> None:
        record = build_s0a_natural_prompts(
            TOKENIZERS, slots=frozen_natural_slots()[:1]
        )[0]
        poisoned = PoisonMapping(asdict(record.row))
        poisoned["label"] = object()
        with self.assertRaisesRegex(ValueError, "allowlist"):
            sanitize_natural_prompt_row(poisoned)
        tampered = asdict(record.row)
        tampered["tokenizer_evidence"]["generation_seed"] += 1
        with self.assertRaisesRegex(ValueError, "generation seed"):
            sanitize_natural_prompt_row(tampered)
        nested = asdict(record.row)
        nested["tokenizer_evidence"]["prefix_text_sha256"] = PoisonMapping(
            {"label": object()}
        )
        with self.assertRaisesRegex(ValueError, "nested key"):
            sanitize_natural_prompt_row(nested)
        string_ids = asdict(record.row)
        string_ids["tokenizer_evidence"]["input_ids"] = [
            str(value) for value in string_ids["tokenizer_evidence"]["input_ids"]
        ]
        with self.assertRaisesRegex(ValueError, "input IDs"):
            sanitize_natural_prompt_row(string_ids)
        wrong_subdomain = asdict(record.row)
        wrong_subdomain["subdomain"] = "gold_truth_partition"
        with self.assertRaisesRegex(ValueError, "frozen slot"):
            sanitize_natural_prompt_row(wrong_subdomain)

    def test_crossed_prefix_lengths_must_be_response_world_invariant(self) -> None:
        group = self.quartets[0].group
        tokenizer = ResponseDependentBoundaryTokenizer(
            "qwen", group.response_text_b,
        )
        with self.assertRaisesRegex(ValueError, "crossed contextual-prefix"):
            contextual_quartet_evidence(
                group,
                {scorer_id: tokenizer for scorer_id in SCORER_IDS},
            )

    def test_resume_prefix_computes_only_the_missing_quartet_unit(self) -> None:
        slots = frozen_quartet_slots()[:3]
        prefix = build_s0a_quartets(TOKENIZERS, slots=slots[:2])
        emitted = []
        resumed = build_s0a_quartets(
            TOKENIZERS, slots=slots, existing_records=prefix,
            on_record=lambda index, record: emitted.append((index, record.slot.slot_id)),
        )
        self.assertEqual(resumed[:2], prefix)
        self.assertEqual(emitted, [(2, slots[2].slot_id)])

    def test_public_checkpoint_records_roundtrip_into_strict_resume_types(self) -> None:
        quartet_payload = json.loads(
            canonical_json_bytes(public_quartet_record(self.quartets[0]))
        )
        restored_quartet = quartet_record_from_public(quartet_payload)
        self.assertEqual(
            json.loads(canonical_json_bytes(public_quartet_record(restored_quartet))),
            quartet_payload,
        )
        natural = build_s0a_natural_prompts(
            TOKENIZERS, slots=frozen_natural_slots()[:1],
        )[0]
        natural_payload = json.loads(
            canonical_json_bytes(public_natural_prompt_record(natural))
        )
        restored_natural = natural_record_from_public(natural_payload)
        self.assertEqual(
            json.loads(canonical_json_bytes(public_natural_prompt_record(restored_natural))),
            natural_payload,
        )

    def test_resume_rejects_forged_terminal_and_rejection_ledger_hashes(self) -> None:
        base = json.loads(canonical_json_bytes(public_quartet_record(self.quartets[0])))
        for field in ("semantic_task_sha256_a", "contextual_evidence_sha256"):
            forged = json.loads(json.dumps(base))
            forged["attempt_ledger"][-1][field] = "f" * 64
            with self.assertRaisesRegex(ValueError, "terminal ledger"):
                quartet_record_from_public(forged)
        natural = build_s0a_natural_prompts(
            TOKENIZERS, slots=frozen_natural_slots()[:1],
        )[0]
        forged_natural = json.loads(
            canonical_json_bytes(public_natural_prompt_record(natural))
        )
        forged_natural["attempt_ledger"][-1]["semantic_task_sha256_a"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "terminal ledger"):
            natural_record_from_public(forged_natural)

        index = next(
            index for index, record in enumerate(self.quartets)
            if len(record.attempt_ledger) > 1
        )
        restored = []
        for offset, record in enumerate(self.quartets[:index + 1]):
            payload = json.loads(canonical_json_bytes(public_quartet_record(record)))
            if offset == index:
                payload["attempt_ledger"][0]["reason"] = "forged_rejection"
                payload["attempt_ledger"][0]["attempt_seed"] += 1
            restored.append(quartet_record_from_public(payload))
        with self.assertRaisesRegex(ValueError, "ledger does not replay"):
            build_s0a_quartets(
                TOKENIZERS, slots=frozen_quartet_slots()[:index + 1],
                existing_records=restored,
            )
        multi_payload = json.loads(
            canonical_json_bytes(public_quartet_record(self.quartets[index]))
        )
        for substituted in (False, 0.0):
            forged_type = json.loads(json.dumps(multi_payload))
            forged_type["attempt_ledger"][0]["attempt_index"] = substituted
            with self.assertRaisesRegex(ValueError, "integer fields"):
                quartet_record_from_public(forged_type)

    def test_resume_rejects_json_bool_and_float_substitutions(self) -> None:
        for location, field, value in (
            ("slot", "within_fold", False),
            ("slot", "within_fold", 0.0),
            ("group", "outer_fold", False),
            ("group", "outer_fold", 0.0),
        ):
            payload = json.loads(
                canonical_json_bytes(public_quartet_record(self.quartets[0]))
            )
            payload[location][field] = value
            restored = quartet_record_from_public(payload)
            with self.assertRaisesRegex(ValueError, "bytes do not match frozen replay"):
                build_s0a_quartets(
                    TOKENIZERS, slots=frozen_quartet_slots()[:1],
                    existing_records=(restored,),
                )

    def test_contextual_builder_assigns_boundary_token_to_response(self) -> None:
        identity = MODEL_IDENTITIES[0]
        evidence = build_contextual_input_evidence(
            TOKENIZERS["qwen3-4b"], identity, "Compute 2+2.", "The final answer is 4.",
        )
        self.assertTrue(evidence.response_ids)
        self.assertEqual(evidence.scorer_id, "qwen3-4b")
        self.assertEqual(len(evidence.full_ids_sha256), 64)

    def test_snapshot_copy_is_content_addressed_regular_and_tamper_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "tokenizer.json").write_text("{}", encoding="utf-8")
            (source / "tokenizer_config.json").write_text(
                '{"chat_template":"x"}', encoding="utf-8"
            )
            ignored = source / "weights.bin"
            ignored.write_bytes(b"not selected")
            destination, manifest = prepare_content_addressed_tokenizer_snapshot(
                source, root / "boundary", MODEL_IDENTITIES[0],
                resolved_revision=MODEL_IDENTITIES[0].revision,
            )
            self.assertEqual([value.path for value in manifest.files],
                             ["tokenizer.json", "tokenizer_config.json"])
            self.assertTrue(all(not path.is_symlink() for path in destination.rglob("*")))
            verify_content_addressed_snapshot(destination, manifest)
            (destination / "tokenizer.json").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_content_addressed_snapshot(destination, manifest)

    def test_snapshot_revision_mismatch_and_extra_file_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "tokenizer.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "revision"):
                prepare_content_addressed_tokenizer_snapshot(
                    source, root / "boundary", MODEL_IDENTITIES[0],
                    resolved_revision="wrong",
                )
            destination, manifest = prepare_content_addressed_tokenizer_snapshot(
                source, root / "boundary2", MODEL_IDENTITIES[0],
                resolved_revision=MODEL_IDENTITIES[0].revision,
            )
            (destination / "extra.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "path set"):
                verify_content_addressed_snapshot(destination, manifest)

    def test_popqa_reservation_is_opaque_and_llama_schema_is_data_free(self) -> None:
        reservation = popqa_opaque_reservation()
        self.assertEqual(len(reservation["opaque_row_ids"]), POPQA_ROWS)
        self.assertFalse(reservation["dataset_content_accessed"])
        blob = str(reservation).casefold()
        self.assertNotIn("question_text", blob)
        schema = future_llama_sidecar_schema()
        self.assertEqual(schema.key_fields, ("cohort_id", "item_id", "response_sha256"))

    def test_a6_llama_namespace_payload_check_is_scoped_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prompt_manifest.json").write_text("{}", encoding="utf-8")
            assert_no_a6_llama_payloads((root,))
            (root / "future_response.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "future payload"):
                assert_no_a6_llama_payloads((root,))


if __name__ == "__main__":
    unittest.main()
