"""Tests for A6-S0's combined two-population global ledger."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import re
import unittest

from spectral_utils.a6_s0_population import (
    PopulationConstruction,
    PopulationSlot,
    audit_population,
    build_population,
    frozen_combined_slots,
    frozen_population_slots,
    population_rarity_sidecars,
)
from spectral_utils.a6_interventions import semantic_task_sha256


def _tokens(text: str) -> list[int]:
    pieces = re.findall(r"[A-Za-z_]+|[-+]?\d+(?:/[1-9]\d*)?|[^\w\s]", text)
    return [sum(map(ord, piece)) % 65_521 for piece in pieces]


TOKENIZERS = {"llama": _tokens, "qwen": _tokens}


class A6S0PopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.slots = frozen_combined_slots()
        cls.population = build_population(
            cls.slots, TOKENIZERS, max_attempts_per_slot=1_000
        )

    def test_full_frozen_allocation_and_replay(self) -> None:
        audit = audit_population(
            self.population, TOKENIZERS, require_frozen_allocation=True
        )
        self.assertTrue(audit["pass"], audit)
        self.assertEqual(audit["n_groups"], 1_800)
        self.assertGreater(audit["n_local_rejections"], 0)
        self.assertGreater(audit["n_global_rejections"], 0)
        self.assertEqual(len(population_rarity_sidecars(self.population)), 1_800)

    def test_numeric_rarity_counts_keys_and_values_from_canonical_ast(self) -> None:
        rarity = population_rarity_sidecars(self.population)
        logic_group = next(
            group for group in self.population.groups if group.domain == "finite_logic"
        )
        expected = sum(
            len(re.findall(r"[-+]?\d+(?:/[1-9]\d*)?", f"{key}\0{value}"))
            for task in (logic_group.task_a, logic_group.task_b)
            for key, value, _ in task.records
        )
        self.assertEqual(rarity[logic_group.group_id]["numeric_atom_count"], expected)

    def test_another_balanced_schedule_is_not_the_frozen_manifest(self) -> None:
        alternate_slots = (
            frozen_population_slots("qwen-source", 640_001)
            + frozen_population_slots("llama-audit", 650_001)
        )
        alternate = build_population(
            alternate_slots, TOKENIZERS, max_attempts_per_slot=1_000
        )
        self.assertFalse(
            audit_population(alternate, TOKENIZERS, require_frozen_allocation=True)[
                "allocation_pass"
            ]
        )

    def test_ast_prompt_source_donor_template_boundaries_are_disjoint(self) -> None:
        asts, prompts, semantic_asts, prompt_contents, group_ids = (
            set(), set(), set(), set(), set()
        )
        ownership = {}
        for slot, group in zip(self.population.slots, self.population.groups):
            self.assertTrue(asts.isdisjoint((group.ast_sha256_a, group.ast_sha256_b)))
            asts.update((group.ast_sha256_a, group.ast_sha256_b))
            group_semantic = {
                semantic_task_sha256(group.task_a), semantic_task_sha256(group.task_b)
            }
            self.assertTrue(semantic_asts.isdisjoint(group_semantic))
            semantic_asts.update(group_semantic)
            group_prompts = set((*group.complete_prompt_ids_a, *group.complete_prompt_ids_b))
            self.assertTrue(prompts.isdisjoint(group_prompts))
            prompts.update(group_prompts)
            content_hashes = {
                hashlib.sha256(text.encode("utf-8")).hexdigest()
                for text in (*group.prompts_a, *group.prompts_b)
            }
            self.assertTrue(prompt_contents.isdisjoint(content_hashes))
            prompt_contents.update(content_hashes)
            self.assertNotIn(group.group_id, group_ids)
            group_ids.add(group.group_id)
            owner = (slot.population_id, slot.outer_fold)
            for identifier in (slot.source_record_id, slot.donor_id, slot.template_id):
                self.assertEqual(ownership.setdefault(identifier, owner), owner)
        self.assertEqual(len(semantic_asts), 3_600)
        self.assertEqual(len(prompt_contents), 14_400)

    def test_ledger_tampering_is_rejected(self) -> None:
        ledgers = list(self.population.attempt_ledgers)
        first = list(ledgers[0])
        first[-1] = replace(first[-1], reason="tampered")
        ledgers[0] = tuple(first)
        tampered = PopulationConstruction(
            self.population.slots, self.population.groups, tuple(ledgers)
        )
        self.assertFalse(
            audit_population(tampered, TOKENIZERS, require_frozen_allocation=True)["pass"]
        )

    def test_identity_crossing_outer_fold_fails_before_construction(self) -> None:
        base = self.slots[0]
        crossed = PopulationSlot(
            "crossed-slot", base.population_id, base.seed + 1,
            (base.outer_fold + 1) % 5, base.source_record_id,
            "new-donor", "new-template", base.domain, base.mutation_family,
            base.response_grammar,
        )
        with self.assertRaisesRegex(ValueError, "crosses population or outer fold"):
            build_population((base, crossed), TOKENIZERS, max_attempts_per_slot=2)


if __name__ == "__main__":
    unittest.main()
