"""Development-only tests for the frozen A6 reciprocal construction."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import re
import unittest

from spectral_utils.a6_interventions import (
    DOMAINS,
    MUTATIONS,
    RENDERINGS,
    RESPONSE_GRAMMARS,
    ReciprocalConstruction,
    ResponseAST,
    TaskAST,
    audit_reciprocal_construction,
    audit_reciprocal_group,
    build_reciprocal_group,
    canonical_answer,
    contains_answer_atom,
    construct_reciprocal_attempt,
    evaluate_generator,
    evaluate_verifier,
    parse_answer_atom,
    parse_closed_answer,
    parse_closed_natural_answer,
    parse_response,
    public_group_record,
    verify_response,
)


def _word_token_ids(text: str) -> list[int]:
    pieces = re.findall(r"[A-Za-z_]+|[-+]?\d+(?:/[1-9]\d*)?|[^\w\s]", text)
    return [sum(map(ord, piece)) % 65_521 for piece in pieces]


TOKENIZERS = {"llama": _word_token_ids, "qwen": _word_token_ids}


def _kwargs(
    *, seed: int = 17, domain: str = "arithmetic",
    mutation: str = "value_leaf", grammar: str = "short",
) -> dict:
    identity = f"{domain}:{mutation}:{grammar}:{seed}"
    return {
        "seed": seed,
        "population_id": "development-qwen",
        "outer_fold": seed % 5,
        "source_record_id": f"source:{identity}",
        "donor_id": f"donor:{identity}",
        "template_id": f"template:{identity}",
        "domain": domain,
        "mutation_family": mutation,
        "response_grammar": grammar,
        "tokenizers": TOKENIZERS,
    }


class A6InterventionTests(unittest.TestCase):
    def test_all_registered_cells_construct_and_audit(self) -> None:
        for seed in (17, 31, 43):
            for domain in DOMAINS:
                for mutation in MUTATIONS:
                    for grammar in RESPONSE_GRAMMARS:
                        with self.subTest(
                            seed=seed, domain=domain, mutation=mutation, grammar=grammar
                        ):
                            construction = build_reciprocal_group(
                                **_kwargs(
                                    seed=seed, domain=domain, mutation=mutation,
                                    grammar=grammar,
                                ),
                                max_attempts=1_000,
                            )
                            group = construction.group
                            audit = audit_reciprocal_group(group, TOKENIZERS)
                            self.assertTrue(audit["pass"], audit)
                            self.assertTrue(
                                audit_reciprocal_construction(construction, TOKENIZERS)["pass"]
                            )
                            self.assertEqual(len(group.prompts_a), len(RENDERINGS))
                            self.assertEqual(
                                group.prompt_token_counts_a,
                                group.prompt_token_counts_b,
                            )
                            self.assertEqual(
                                [record.status for record in construction.attempts].count(
                                    "ACCEPTED"
                                ),
                                1,
                            )

    def test_integer_and_rational_arithmetic_are_covered(self) -> None:
        output_kinds = set()
        for seed in range(40):
            construction = build_reciprocal_group(
                **_kwargs(seed=seed, domain="arithmetic", mutation="value_leaf"),
                max_attempts=100,
            )
            output_kinds.add(construction.group.task_a.output_kind)
        self.assertEqual(output_kinds, {"exact_integer", "exact_rational"})

    def test_relational_aggregation_and_lookup_subtypes_are_covered(self) -> None:
        seen = set()
        for seed in range(80):
            group = build_reciprocal_group(
                **_kwargs(
                    seed=seed, domain="relational", mutation="relation_operator",
                    grammar="certificate",
                ),
                max_attempts=100,
            ).group
            operators = {group.task_a.operator, group.task_b.operator}
            seen.add("lookup" if any(value.startswith("lookup_") for value in operators)
                     else "aggregation")
            self.assertTrue(audit_reciprocal_group(group, TOKENIZERS)["pass"])
        self.assertEqual(seen, {"aggregation", "lookup"})

    def test_actual_fractional_certificate_roundtrips(self) -> None:
        fractional = None
        for seed in range(100):
            construction = build_reciprocal_group(
                **_kwargs(
                    seed=seed, domain="arithmetic", mutation="value_leaf",
                    grammar="certificate",
                ),
                max_attempts=100,
            )
            if any("/" in row[1] for row in construction.group.task_a.records):
                fractional = construction.group
                break
        self.assertIsNotNone(fractional)
        self.assertTrue(audit_reciprocal_group(fractional, TOKENIZERS)["pass"])
        self.assertTrue(any("/" in fact for fact in fractional.response_a.source_facts))

    def test_attempt_api_and_ledger_are_byte_reproducible(self) -> None:
        kwargs = _kwargs(seed=29, domain="finite_logic", mutation="value_leaf")
        first = build_reciprocal_group(**kwargs, max_attempts=100)
        second = build_reciprocal_group(**kwargs, max_attempts=100)
        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(public_group_record(first.group), sort_keys=True),
            json.dumps(public_group_record(second.group), sort_keys=True),
        )
        record, group = construct_reciprocal_attempt(
            **kwargs, attempt_index=first.group.attempt_index
        )
        self.assertEqual(record, first.attempts[-1])
        self.assertEqual(group, first.group)

    def test_construction_audit_rejects_skipping_an_earlier_legal_attempt(self) -> None:
        found = None
        for seed in range(200):
            kwargs = _kwargs(seed=seed, domain="finite_logic", mutation="constraint_condition")
            first_record, first_group = construct_reciprocal_attempt(**kwargs, attempt_index=0)
            second_record, second_group = construct_reciprocal_attempt(**kwargs, attempt_index=1)
            if first_group is not None and second_group is not None:
                found = (first_record, second_record, second_group)
                break
        self.assertIsNotNone(found)
        first_record, second_record, second_group = found
        fake = ReciprocalConstruction(second_group, (first_record, second_record))
        self.assertFalse(audit_reciprocal_construction(fake, TOKENIZERS)["pass"])

    def test_finite_logic_constraint_space_has_two_population_capacity(self) -> None:
        unordered_pairs = set()
        for population_index, population in enumerate(("qwen", "llama")):
            for slot in range(150):
                kwargs = _kwargs(
                    seed=10_000 * population_index + slot,
                    domain="finite_logic", mutation="constraint_condition",
                )
                kwargs.update({
                    "population_id": population,
                    "source_record_id": f"{population}:source:{slot}",
                    "donor_id": f"{population}:donor:{slot}",
                    "template_id": f"{population}:template:{slot}",
                })
                group = build_reciprocal_group(**kwargs, max_attempts=1_000).group
                unordered_pairs.add(tuple(sorted((group.ast_sha256_a, group.ast_sha256_b))))
        self.assertGreaterEqual(len(unordered_pairs), 200)

    def test_tokenizer_evidence_is_mandatory_and_complete(self) -> None:
        kwargs = _kwargs(grammar="certificate")
        with self.assertRaisesRegex(ValueError, "exactly the frozen families"):
            build_reciprocal_group(**{**kwargs, "tokenizers": {}}, max_attempts=1)
        with self.assertRaisesRegex(TypeError, "token-ID sequence"):
            build_reciprocal_group(
                **{**kwargs, "tokenizers": {"llama": lambda _: 1, "qwen": lambda _: 1}},
                max_attempts=1,
            )
        construction = build_reciprocal_group(**kwargs, max_attempts=100)
        group = construction.group
        self.assertEqual(tuple(name for name, _ in group.prompt_token_counts_a), ("llama", "qwen"))
        self.assertTrue(all(len(counts) == 4 for _, counts in group.prompt_token_counts_a))
        self.assertTrue(
            all(40 <= len(ids) <= 80 for _, ids in group.response_token_ids_a)
        )

    def test_certificate_band_rejects_short_tokenizer_output(self) -> None:
        kwargs = _kwargs(grammar="certificate")
        tiny = {"llama": lambda _: [1], "qwen": lambda _: [1]}
        record, group = construct_reciprocal_attempt(
            **{**kwargs, "tokenizers": tiny}, attempt_index=0
        )
        self.assertIsNone(group)
        self.assertEqual(record.reason, "certificate_token_band")

    def test_full_certificate_verifier_rejects_fault_injections(self) -> None:
        group = build_reciprocal_group(
            **_kwargs(grammar="certificate", domain="relational", mutation="value_leaf"),
            max_attempts=100,
        ).group
        parsed = parse_response(group.response_text_a, "certificate")
        self.assertTrue(verify_response(group.task_a, parsed))
        self.assertFalse(
            verify_response(group.task_a, replace(parsed, answer=str(int(parsed.answer) + 1)))
        )
        self.assertFalse(
            verify_response(group.task_a, replace(parsed, source_facts=("poison/1/red",)))
        )
        selected = list(parsed.selected_values)
        selected[0] = canonical_answer(int(selected[0]) + 1)
        self.assertFalse(
            verify_response(group.task_a, replace(parsed, selected_values=tuple(selected)))
        )
        self.assertFalse(verify_response(group.task_b, parsed))

    def test_generator_and_independent_verifier_agree(self) -> None:
        for domain in DOMAINS:
            for mutation in MUTATIONS:
                group = build_reciprocal_group(
                    **_kwargs(domain=domain, mutation=mutation, grammar="certificate"),
                    max_attempts=100,
                ).group
                for task in (group.task_a, group.task_b):
                    self.assertEqual(evaluate_generator(task), evaluate_verifier(task))

    def test_hash_and_text_tampering_fails_closed(self) -> None:
        group = build_reciprocal_group(**_kwargs(), max_attempts=100).group
        self.assertTrue(audit_reciprocal_group(group, TOKENIZERS)["pass"])
        bad_hash = replace(group, ast_sha256_a="0" * 64)
        self.assertFalse(audit_reciprocal_group(bad_hash, TOKENIZERS)["pass"])
        wrong_answer = canonical_answer(parse_answer_atom(group.response_a.answer) + 1)
        bad_text = replace(
            group, response_text_a=f"The final answer is {wrong_answer}."
        )
        self.assertFalse(audit_reciprocal_group(bad_text, TOKENIZERS)["pass"])
        malformed_prompt = replace(
            group, prompts_a=("malformed", *group.prompts_a[1:])
        )
        with self.assertRaisesRegex(ValueError, "outside|schema"):
            audit_reciprocal_group(malformed_prompt, TOKENIZERS)

    def test_short_construction_rejects_alternate_on_policy_wrapper(self) -> None:
        group = build_reciprocal_group(**_kwargs(), max_attempts=100).group
        alternate = f"Answer: {group.response_a.answer}"
        alternate_ids = tuple(
            (name, tuple(callback(alternate))) for name, callback in sorted(TOKENIZERS.items())
        )
        altered = replace(
            group, response_text_a=alternate,
            response_sha256_a=hashlib.sha256(alternate.encode("utf-8")).hexdigest(),
            response_token_ids_a=alternate_ids,
        )
        audit = audit_reciprocal_group(altered, TOKENIZERS)
        self.assertFalse(audit["deterministic_response_construction_pass"])
        self.assertFalse(audit["pass"])

    def test_seed_domain_mutation_and_short_response_metadata_are_bound(self) -> None:
        group = build_reciprocal_group(**_kwargs(), max_attempts=100).group
        self.assertFalse(
            audit_reciprocal_group(replace(group, seed=group.seed + 1), TOKENIZERS)["pass"]
        )
        self.assertFalse(
            audit_reciprocal_group(replace(group, domain="relational"), TOKENIZERS)["pass"]
        )
        relabeled_a = replace(group.task_a, mutation_family="relation_operator")
        relabeled_b = replace(group.task_b, mutation_family="relation_operator")
        relabeled = replace(
            group, mutation_family="relation_operator", task_a=relabeled_a,
            task_b=relabeled_b,
        )
        self.assertFalse(audit_reciprocal_group(relabeled, TOKENIZERS)["pass"])
        bad_response = replace(
            group, response_a=replace(group.response_a, domain="relational")
        )
        self.assertFalse(audit_reciprocal_group(bad_response, TOKENIZERS)["pass"])

    def test_closed_answer_parser_nfkc_and_rejects_extra_prose(self) -> None:
        self.assertEqual(parse_closed_answer("  The final answer is １２.  "), 12)
        self.assertEqual(parse_closed_answer("Answer: -3/4"), -3 / 4)
        for invalid in (
            "Reasoning. The final answer is 12.",
            "The final answer is 12. Extra",
            "Answer: 12 or 13",
            "",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_closed_answer(invalid)

    def test_full_natural_answer_parser_covers_registered_atom_productions(self) -> None:
        cases = (
            ("integer", "12", (), ""),
            ("rational", "-3/4", (), ""),
            ("finite_set", "{Ada,Bela}", ("Ada", "Bela"), ""),
            ("entity", "New York City", ("New York City",), ""),
            ("relation", "Ada>Bela", ("Ada>Bela",), ""),
            ("integer", "12 kg", (), "kg"),
        )
        wrappers = (
            "{}", "Answer: {}", "The answer is {}.", "The final answer is {}.",
        )
        for kind, atom, registered, unit in cases:
            for wrapper in wrappers:
                with self.subTest(kind=kind, wrapper=wrapper):
                    self.assertEqual(
                        parse_closed_natural_answer(
                            wrapper.format(atom), kind=kind,
                            registered_atoms=registered, required_unit=unit,
                        ),
                        atom,
                    )
        for invalid, kwargs in (
            ("Reasoning. The final answer is 12.", {"kind": "integer"}),
            ("I refuse", {"kind": "integer"}),
            ("Answer: 12 lb", {"kind": "integer", "required_unit": "kg"}),
            ("Answer: Paris", {"kind": "entity", "registered_atoms": ("Rome",)}),
            ("Answer: {Ada,Ada}", {"kind": "finite_set", "registered_atoms": ("Ada",)}),
            ("Answer: {999}", {"kind": "finite_set", "registered_atoms": ("1", "2")}),
            ("Answer: {Bela,Ada}", {
                "kind": "finite_set", "registered_atoms": ("Ada", "Bela"),
            }),
            ("Answer: Ada > Bela", {"kind": "relation", "registered_atoms": ("Ada>Bela",)}),
            ("Answer: +001", {"kind": "integer"}),
            ("Answer: 2/4", {"kind": "rational"}),
            ("Answer: Ada", {"kind": "entity", "registered_atoms": ("Answer: Ada",)}),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_closed_natural_answer(invalid, **kwargs)
        self.assertEqual(
            parse_closed_natural_answer(
                "The final answer is １２ kg.", kind="integer", required_unit="kg"
            ),
            "12 kg",
        )
        self.assertEqual(
            parse_closed_natural_answer(
                "Answer: {}", kind="finite_set", registered_atoms=("Ada",)
            ),
            "{}",
        )

    def test_answer_atom_boundary_avoids_numeric_substring_shortcut(self) -> None:
        self.assertFalse(contains_answer_atom("Records contain 12 and 20.", "2"))
        self.assertTrue(contains_answer_atom("Records contain 12 and 2.", "2"))
        self.assertFalse(contains_answer_atom("Ratio 3/4.", "3"))
        self.assertTrue(contains_answer_atom("Ratio is -3/4.", "-3/4"))

    def test_ids_are_construction_inputs_not_retrofitted_metadata(self) -> None:
        base = _kwargs(seed=71)
        left = build_reciprocal_group(**base, max_attempts=100).group
        changed = build_reciprocal_group(
            **{**base, "donor_id": "donor:alternate", "template_id": "template:alternate"},
            max_attempts=100,
        ).group
        self.assertNotEqual(left.group_id, changed.group_id)
        self.assertNotEqual(left.prompts_a, changed.prompts_a)
        self.assertNotEqual(left.ast_sha256_a, changed.ast_sha256_a)
        self.assertTrue(set(left.complete_prompt_ids_a).isdisjoint(changed.complete_prompt_ids_a))

    def test_public_record_has_only_mechanical_truth(self) -> None:
        group = build_reciprocal_group(**_kwargs(), max_attempts=100).group
        record = public_group_record(group)
        self.assertEqual(record["mechanical_truth"], [[1, 0], [0, 1]])
        serialized_keys = " ".join(record).casefold()
        for forbidden in ("classification", "benchmark_label", "human_label"):
            self.assertNotIn(forbidden, serialized_keys)
        self.assertIn("shortcut_sidecar", record)
        self.assertEqual(record["shortcut_sidecar"]["donor_id"], group.donor_id)

    def test_every_prompt_contains_the_explicit_domain_schema(self) -> None:
        for domain in DOMAINS:
            group = build_reciprocal_group(
                **_kwargs(domain=domain, mutation="constraint_condition"),
                max_attempts=100,
            ).group
            self.assertTrue(all("\nSchema: " in prompt for prompt in group.prompts_a))
            if domain == "finite_logic":
                self.assertTrue(all("0 means neither" in prompt for prompt in group.prompts_a))
                self.assertTrue(all("cardinality" in prompt for prompt in group.prompts_a))

    def test_response_ast_short_contains_one_assertion(self) -> None:
        response = ResponseAST("short", "4", "arithmetic")
        self.assertEqual(response.source_facts, ())
        with self.assertRaises(ValueError):
            ResponseAST("short", "4", "arithmetic", ("fact",))

    def test_task_ast_rejects_reserved_certificate_and_prompt_delimiters(self) -> None:
        for poisoned in ("1~2", "1|2", "1;2", "1,2", "1\n2"):
            with self.subTest(poisoned=poisoned):
                with self.assertRaisesRegex(ValueError, "reserved delimiter"):
                    TaskAST(
                        "arithmetic", "value_leaf", (("v0", poisoned, "number"),),
                        "sum", "all",
                    )

    def test_task_ast_rejects_invalid_domain_record_schemas_and_units(self) -> None:
        invalid = (
            ("arithmetic", (("v0", "+01", "number"),), "sum", "all", ""),
            ("arithmetic", (("v0", "1", "wrong"),), "sum", "all", ""),
            ("relational", (("Ada", "not-an-int", "red"),), "sum", "red", ""),
            ("relational", (("Ada", "1", "green"),), "sum", "red", ""),
            ("finite_logic", (("1", "4", "membership"),), "union", "all", ""),
            ("finite_logic", (("1", "1", "membership"), ("1", "2", "membership")),
             "union", "all", ""),
            ("arithmetic", (("v0", "1", "number"),), "sum", "all", "kg"),
        )
        for domain, records, operator, constraint, unit in invalid:
            with self.subTest(domain=domain, records=records, unit=unit):
                with self.assertRaises(ValueError):
                    TaskAST(
                        domain, "value_leaf", records, operator, constraint,
                        "exact_integer", unit,
                    )


if __name__ == "__main__":
    unittest.main()
