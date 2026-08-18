#!/usr/bin/env python3
"""CPU-only tests for fair-comparison registries and report primitives."""

from __future__ import annotations

import json
import hashlib
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_utils.fair_comparisons import registry as R  # noqa: E402
from spectral_utils.fair_comparisons import reporting as P  # noqa: E402


def _hash(label: str) -> str:
    return R.canonical_sha256({"fixture": label})


class FairRegistryFixtures(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.json"
        self.source.write_text('{"source":"fixture"}\n', encoding="utf-8")
        self.asset = R.make_asset_record(self.source, artifact_id="source", root=self.root)
        self.population_entry = R.make_population_entry(
            population_id="pb-global",
            lane="global",
            dataset_revision="processbench@fixture",
            ordered_ids=["pb::gsm8k::0", "pb::math::0", "pb::math::1"],
            group_ids=["q0", "q1", "q2"],
            cell_ids=["gsm8k", "math", "math"],
            families=["math", "math", "math"],
            label_definition={"positive_class": "wrong final answer", "positive_value": 1},
            eligibility_rules=["official rows", "no positional fallback"],
        )
        self.populations = R.build_population_registry([self.population_entry])
        self.unified = R.make_method_entry(
            method_id="unified-28",
            display_name="Unified-28",
            fidelity="adapted-common-protocol",
            source_artifacts=[self.asset],
            access={
                "input_type": "gray-box token probabilities",
                "supervision": "none",
                "model_passes_per_question": 1,
                "traces_per_question": 1,
            },
            training_label_use={"score_fit": "none", "threshold_fit": "cross-fitted"},
            checkpoint_revision="frozen-unified-28",
            prompt_sha256=_hash("prompt"),
            decoding_sha256=_hash("decoding"),
            evaluator_sha256=_hash("evaluator"),
            run_commit="d3ca3a4",
            deviations=[{"field": "population", "reason": "common-protocol replay"}],
        )
        self.incumbent = R.make_method_entry(
            method_id="classic-mixed-v2-no-length",
            display_name="classic_mixed_v2_no_length",
            fidelity="official-exact",
            source_artifacts=[self.asset],
            access={
                "input_type": "gray-box token probabilities",
                "supervision": "none",
                "model_passes_per_question": 1,
                "traces_per_question": 1,
            },
            training_label_use="no labels for score construction",
            checkpoint_revision="registered-fit-v1",
            prompt_sha256=_hash("prompt"),
            decoding_sha256=_hash("decoding"),
            evaluator_sha256=_hash("evaluator"),
            run_commit="e51450d",
            deviations=[],
        )
        self.methods = R.build_method_registry([self.unified, self.incumbent])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def records(self) -> list[dict]:
        rows = []
        population = self.population_entry
        for index, row_id in enumerate(population["ordered_ids"]):
            for method, offset in ((self.unified, 0.1), (self.incumbent, 0.2)):
                rows.append(
                    R.make_comparison_record(
                        lane="global",
                        population_id="pb-global",
                        row_id=row_id,
                        group_id=population["group_ids"][index],
                        cell_id=population["cell_ids"][index],
                        method_id=method["method_id"],
                        continuous_score=offset + index,
                        discrete_prediction=int(index > 0),
                        label=int(index > 0),
                        budget=None,
                        fold=index % 5,
                        calibration_hash=_hash(f"calibration-{method['method_id']}"),
                        source_artifact_hash=method["source_artifacts_sha256"],
                    )
                )
        return rows


class PopulationRegistryTests(FairRegistryFixtures):
    def test_fidelity_vocabulary_is_exact(self) -> None:
        self.assertEqual(
            R.FIDELITY_LABELS,
            (
                "official-exact",
                "paper-specified",
                "paper-specified-partial",
                "adapted-common-protocol",
                "published-context-only",
                "blocked-assets",
            ),
        )

    def test_order_hash_is_deterministic_and_order_sensitive(self) -> None:
        ids = ["a", "b", "c"]
        self.assertEqual(R.ordered_id_sha256(ids), R.ordered_id_sha256(list(ids)))
        self.assertNotEqual(R.ordered_id_sha256(ids), R.ordered_id_sha256(["b", "a", "c"]))
        # Canonical JSON prevents delimiter ambiguity.
        self.assertNotEqual(R.ordered_id_sha256(["a\nb", "c"]), R.ordered_id_sha256(["a", "b\nc"]))

    def test_population_rejects_duplicate_ids_and_hash_drift(self) -> None:
        with self.assertRaises(R.RegistryError):
            R.make_population_entry(
                population_id="bad",
                lane="global",
                dataset_revision="r",
                ordered_ids=["x", "x"],
                group_ids=["g", "g"],
                cell_ids=["c", "c"],
                families=["f", "f"],
                label_definition="wrong is positive",
                eligibility_rules=["all"],
            )
        tampered = json.loads(json.dumps(self.populations))
        tampered["populations"][0]["ordered_ids"].reverse()
        with self.assertRaises(R.RegistryError):
            R.validate_population_registry(tampered)

    def test_population_metadata_vectors_must_align(self) -> None:
        with self.assertRaisesRegex(R.RegistryError, "group_ids"):
            R.make_population_entry(
                population_id="bad",
                lane="global",
                dataset_revision="r",
                ordered_ids=["x", "y"],
                group_ids=["g"],
                cell_ids=["c", "c"],
                families=["f", "f"],
                label_definition="wrong is positive",
                eligibility_rules=["all"],
            )

    def test_empty_population_is_not_a_vacuous_complete_join(self) -> None:
        with self.assertRaisesRegex(R.RegistryError, "at least one row"):
            R.make_population_entry(
                population_id="empty",
                lane="global",
                dataset_revision="r",
                ordered_ids=[],
                group_ids=[],
                cell_ids=[],
                families=[],
                label_definition="wrong is positive",
                eligibility_rules=["all"],
            )

    def test_repeated_group_cannot_cross_family(self) -> None:
        with self.assertRaisesRegex(R.RegistryError, "conflicting families"):
            R.make_population_entry(
                population_id="bad-group",
                lane="prefix",
                dataset_revision="r",
                ordered_ids=["copy-a", "copy-b"],
                group_ids=["question", "question"],
                cell_ids=["model-a", "model-b"],
                families=["math", "qa"],
                label_definition="wrong is positive",
                eligibility_rules=["unfinished only"],
            )

    def test_registry_bytes_are_repeatable(self) -> None:
        first = R.canonical_json_bytes(self.populations)
        second = R.canonical_json_bytes(R.build_population_registry([self.population_entry]))
        self.assertEqual(first, second)
        with self.assertRaises(R.RegistryError):
            R.canonical_json_bytes({"bad": math.nan})

    def test_exact_eligible_population_hash_is_bound_to_registered_order(self) -> None:
        descriptor = R.make_eligible_population(
            ["pb::gsm8k::0", "pb::math::1"], rule="fixture strict eligibility"
        )
        entry = R.make_population_entry(
            population_id="eligible-fixture",
            lane="prefix",
            dataset_revision="r",
            ordered_ids=["pb::gsm8k::0", "pb::math::0", "pb::math::1"],
            group_ids=["g0", "g1", "g2"],
            cell_ids=["c", "c", "c"],
            families=["f", "f", "f"],
            label_definition="wrong is positive",
            eligibility_rules="strict unfinished trace",
            extra={"eligible_populations": {"budget_64": descriptor}},
        )
        self.assertEqual(
            entry["eligible_populations"]["budget_64"]["ordered_id_sha256"],
            R.ordered_id_sha256(["pb::gsm8k::0", "pb::math::1"]),
        )
        tampered = dict(descriptor)
        tampered["ordered_ids"] = list(reversed(tampered["ordered_ids"]))
        tampered["ordered_id_sha256"] = R.ordered_id_sha256(tampered["ordered_ids"])
        with self.assertRaisesRegex(R.RegistryError, "preserve"):
            R.make_population_entry(
                population_id="bad-eligible-fixture",
                lane="prefix",
                dataset_revision="r",
                ordered_ids=["pb::gsm8k::0", "pb::math::0", "pb::math::1"],
                group_ids=["g0", "g1", "g2"],
                cell_ids=["c", "c", "c"],
                families=["f", "f", "f"],
                label_definition="wrong is positive",
                eligibility_rules="strict unfinished trace",
                extra={"eligible_populations": {"budget_64": tampered}},
            )


class MethodRegistryTests(FairRegistryFixtures):
    def test_asset_and_method_hashes_bind_content(self) -> None:
        self.assertEqual(self.asset["sha256"], R.sha256_file(self.source))
        asset_registry = R.build_asset_registry([self.asset])
        self.assertEqual(asset_registry["schema"], "asset_registry_v1")
        self.assertEqual(R.validate_asset_registry(asset_registry), asset_registry)
        self.assertEqual(self.unified["method_hash"], R.method_definition_sha256(self.unified))
        tampered = json.loads(json.dumps(self.methods))
        tampered["methods"][1]["display_name"] = "changed"
        with self.assertRaises(R.RegistryError):
            R.validate_method_registry(tampered)

    def test_derived_asset_hashes_its_declared_projection(self) -> None:
        projection = {
            "schema": "fixture_derived_ledger_v1",
            "members": [{"path": "a", "size_bytes": 3, "sha256": _hash("a")}],
        }
        encoded = R.canonical_json_bytes(projection)
        record = {
            "schema": "asset_record_v1",
            "artifact_kind": "composite-ledger",
            "artifact_id": "fixture/composite",
            "uri": "canonical-json:fixture/composite",
            "size_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "projection": projection,
        }
        self.assertEqual(R.validate_asset_record(record), record)
        with self.assertRaisesRegex(R.RegistryError, "canonical projection"):
            R.validate_asset_record({**record, "sha256": _hash("wrong")})
        with self.assertRaisesRegex(R.RegistryError, "size_bytes"):
            R.validate_asset_record({**record, "size_bytes": len(encoded) + 1})

    def test_file_assets_hash_raw_bytes_not_canonical_json(self) -> None:
        source = self.root / "noncanonical.json"
        raw = b'{ "z": 2, "a": 1 }\n'
        source.write_bytes(raw)
        record = R.make_asset_record(source, artifact_id="raw-json", root=self.root)
        self.assertEqual(record["artifact_kind"], "file")
        self.assertEqual(record["size_bytes"], len(raw))
        self.assertEqual(record["sha256"], hashlib.sha256(raw).hexdigest())
        canonical = R.canonical_json_bytes(json.loads(raw))
        self.assertNotEqual(raw, canonical)
        self.assertNotEqual(record["sha256"], hashlib.sha256(canonical).hexdigest())

    def test_both_derived_kinds_hash_exact_canonical_projection_bytes(self) -> None:
        projection = {
            "z": [{"sha256": _hash("member"), "path": "relative/member"}],
            "a": "ledger",
        }
        expected_bytes = R.canonical_json_bytes(projection)
        for kind in ("derived-ledger", "composite-ledger"):
            with self.subTest(kind=kind):
                record = R.make_derived_asset_record(
                    projection,
                    artifact_kind=kind,
                    artifact_id=f"fixture/{kind}",
                    uri=f"canonical-json:fixture/{kind}",
                )
                self.assertEqual(record["size_bytes"], len(expected_bytes))
                self.assertEqual(
                    record["sha256"], hashlib.sha256(expected_bytes).hexdigest()
                )
                reordered = {"a": projection["a"], "z": projection["z"]}
                self.assertEqual(
                    R.make_derived_asset_record(
                        reordered,
                        artifact_kind=kind,
                        artifact_id=f"fixture/{kind}",
                        uri=f"canonical-json:fixture/{kind}",
                    )["sha256"],
                    record["sha256"],
                )

    def test_projection_cannot_bypass_derived_asset_hashing(self) -> None:
        projection = {"schema": "fixture", "value": 1}
        raw = R.canonical_json_bytes(projection)
        base = {
            "schema": "asset_record_v1",
            "artifact_id": "fixture/bypass",
            "uri": "canonical-json:fixture/bypass",
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "projection": projection,
        }
        with self.assertRaisesRegex(R.RegistryError, "must declare"):
            R.validate_asset_record(base)
        with self.assertRaisesRegex(R.RegistryError, "must not claim"):
            R.validate_asset_record({**base, "artifact_kind": "file"})

    def test_derived_source_alias_is_projection_bound_and_joinable(self) -> None:
        alias = _hash("upstream-package-fingerprint")
        preimage = {"fixture": "upstream-package-fingerprint"}
        asset = R.make_derived_asset_record(
            {"schema": "fixture_projection_v1", "members": ["a", "b"]},
            artifact_kind="composite-ledger",
            artifact_id="fixture/composite",
            uri="derived:fixture/composite",
            source_fingerprint_aliases=[alias],
            source_fingerprint_preimages=[preimage],
        )
        self.assertEqual(asset["source_fingerprint_aliases"], [alias])
        self.assertEqual(asset["projection"]["source_fingerprint_aliases"], [alias])
        self.assertEqual(
            asset["projection"]["source_fingerprint_preimages"],
            [{"sha256": alias, "preimage": preimage}],
        )
        method = R.make_method_entry(
            method_id="composite-source-method",
            display_name="Composite source method",
            fidelity="adapted-common-protocol",
            source_artifacts=[asset],
            access=self.unified["access"],
            training_label_use="none",
            checkpoint_revision="fixture",
            prompt_sha256=_hash("p"),
            decoding_sha256=_hash("d"),
            evaluator_sha256=_hash("e"),
            run_commit="fixture",
            deviations=["fixture"],
        )
        row_id = self.population_entry["ordered_ids"][0]
        record = R.make_comparison_record(
            lane="global",
            population_id="pb-global",
            row_id=row_id,
            group_id=self.population_entry["group_ids"][0],
            cell_id=self.population_entry["cell_ids"][0],
            method_id=method["method_id"],
            continuous_score=0.5,
            discrete_prediction=None,
            label=0,
            budget="final",
            fold=0,
            calibration_hash=None,
            source_artifact_hash=alias,
        )
        audit = R.audit_comparison_records(
            [record],
            self.populations,
            R.build_method_registry([method]),
            expectations=[],
        )
        self.assertEqual(audit["artifact_hash_conflicts"], [])
        tampered = dict(asset)
        tampered["source_fingerprint_aliases"] = [_hash("different")]
        with self.assertRaisesRegex(R.RegistryError, "declared in projection"):
            R.validate_asset_record(tampered)
        with self.assertRaisesRegex(R.RegistryError, "do not hash"):
            R.make_derived_asset_record(
                {"schema": "fixture_projection_v1"},
                artifact_kind="derived-ledger",
                artifact_id="fixture/bad-alias",
                uri="derived:fixture/bad-alias",
                source_fingerprint_aliases=[_hash("wrong")],
                source_fingerprint_preimages=[preimage],
            )

    def test_access_fields_are_orthogonal_and_required(self) -> None:
        access = dict(self.unified["access"])
        del access["traces_per_question"]
        with self.assertRaisesRegex(R.RegistryError, "traces_per_question"):
            R.make_method_entry(
                method_id="bad",
                display_name="bad",
                fidelity="adapted-common-protocol",
                source_artifacts=[self.asset],
                access=access,
                training_label_use="none",
                checkpoint_revision="r",
                prompt_sha256=_hash("p"),
                decoding_sha256=_hash("d"),
                evaluator_sha256=_hash("e"),
                run_commit="c",
                deviations=["adapted"],
            )

    def test_fidelity_and_deviation_contract(self) -> None:
        with self.assertRaisesRegex(R.RegistryError, "incompatible"):
            R.make_method_entry(
                method_id="not-exact",
                display_name="not exact",
                fidelity="official-exact",
                source_artifacts=[self.asset],
                access=self.unified["access"],
                training_label_use="none",
                checkpoint_revision="r",
                prompt_sha256=_hash("p"),
                decoding_sha256=_hash("d"),
                evaluator_sha256=_hash("e"),
                run_commit="c",
                deviations=["changed prompt"],
            )
        with self.assertRaisesRegex(R.RegistryError, "requires"):
            R.make_method_entry(
                method_id="partial-without-deviation",
                display_name="partial",
                fidelity="paper-specified-partial",
                source_artifacts=[self.asset],
                access=self.unified["access"],
                training_label_use="none",
                checkpoint_revision="r",
                prompt_sha256=_hash("p"),
                decoding_sha256=_hash("d"),
                evaluator_sha256=_hash("e"),
                run_commit="c",
                deviations=[],
            )

    def test_blocked_method_records_unknown_provenance_honestly(self) -> None:
        blocked = R.make_method_entry(
            method_id="full-uprm",
            display_name="Full trained uPRM",
            fidelity="blocked-assets",
            source_artifacts=[],
            access={
                "input_type": "hidden states",
                "supervision": "process labels",
                "model_passes_per_question": None,
                "traces_per_question": None,
            },
            training_label_use="required but unavailable",
            checkpoint_revision=None,
            prompt_sha256=None,
            decoding_sha256=None,
            evaluator_sha256=None,
            run_commit=None,
            deviations=["official code/checkpoint unavailable"],
        )
        self.assertEqual(blocked["fidelity"], "blocked-assets")
        self.assertTrue(R.is_sha256(blocked["method_hash"]))


class ComparisonJoinTests(FairRegistryFixtures):
    def expectations(self) -> list[dict]:
        return [
            {"table_id": "pb", "population_id": "pb-global", "method_id": method, "budget": None}
            for method in ("unified-28", "classic-mixed-v2-no-length")
        ]

    def test_clean_identical_row_join_passes(self) -> None:
        records = R.canonicalize_comparison_records(self.records(), self.populations)
        report = R.audit_comparison_records(
            records, self.populations, self.methods, expectations=self.expectations()
        )
        self.assertTrue(report["ok"])
        self.assertTrue(report["headline_ok"])
        self.assertTrue(all(row["coverage"] == 1.0 for row in report["coverage"]))
        self.assertTrue(all(row["order_matches"] for row in report["coverage"]))
        R.require_clean_join(report)

    def test_missing_row_fails_full_coverage(self) -> None:
        records = [
            row
            for row in self.records()
            if not (row["method_id"] == "unified-28" and row["row_id"] == "pb::math::1")
        ]
        records = R.canonicalize_comparison_records(records, self.populations)
        report = R.audit_comparison_records(
            records, self.populations, self.methods, expectations=self.expectations()
        )
        self.assertFalse(report["headline_ok"])
        unified = next(row for row in report["coverage"] if row["method_id"] == "unified-28")
        self.assertEqual(unified["coverage"], 2 / 3)
        self.assertEqual(unified["missing_row_ids"], ["pb::math::1"])
        with self.assertRaises(R.JoinAuditError):
            R.require_clean_join(report, headline_only=True)

    def test_duplicate_and_label_conflict_are_both_reported(self) -> None:
        records = self.records()
        duplicate = dict(records[0])
        duplicate["label"] = 1
        records.append(duplicate)
        records = R.canonicalize_comparison_records(records, self.populations)
        report = R.audit_comparison_records(
            records, self.populations, self.methods, expectations=self.expectations()
        )
        self.assertFalse(report["ok"])
        self.assertEqual(len(report["duplicates"]), 1)
        self.assertEqual(len(report["label_conflicts"]), 1)

    def test_metadata_and_order_mismatch_fail(self) -> None:
        records = self.records()
        records[0] = {**records[0], "group_id": "wrong-group"}
        # Reverse every method's row sequence without canonicalizing it back.
        records = list(reversed(records))
        report = R.audit_comparison_records(
            records, self.populations, self.methods, expectations=self.expectations()
        )
        self.assertFalse(report["ok"])
        self.assertEqual(len(report["metadata_conflicts"]), 1)
        self.assertTrue(any(not row["order_matches"] for row in report["coverage"]))

    def test_eligibility_subset_must_preserve_registered_order(self) -> None:
        bad_expectation = [{
            "population_id": "pb-global",
            "method_id": "unified-28",
            "budget": None,
            "eligible_row_ids": ["pb::math::1", "pb::gsm8k::0"],
        }]
        report = R.audit_comparison_records(
            self.records(), self.populations, self.methods, expectations=bad_expectation
        )
        self.assertFalse(report["ok"])
        self.assertIn("preserve", report["expectation_problems"][0]["problem"])

    def test_unparsed_prediction_stays_a_row(self) -> None:
        record = dict(self.records()[0])
        record["continuous_score"] = None
        record["discrete_prediction"] = None
        validated = R.validate_comparison_record(record)
        self.assertIsNone(validated["discrete_prediction"])
        self.assertEqual(validated["label"], 0)

    def test_fixed_output_record_allows_final_budget_without_calibration_fit(self) -> None:
        record = dict(self.records()[0])
        record.update({"budget": "final", "fold": None, "calibration_hash": None})
        validated = R.validate_comparison_record(record)
        self.assertEqual(validated["budget"], "final")
        self.assertIsNone(validated["fold"])
        self.assertIsNone(validated["calibration_hash"])

    def test_record_artifact_hash_must_resolve_to_registered_method(self) -> None:
        records = self.records()
        records[0] = {**records[0], "source_artifact_hash": _hash("unregistered")}
        records = R.canonicalize_comparison_records(records, self.populations)
        report = R.audit_comparison_records(
            records, self.populations, self.methods, expectations=self.expectations()
        )
        self.assertFalse(report["ok"])
        self.assertEqual(len(report["artifact_hash_conflicts"]), 1)

    def test_expectation_binds_population_method_to_one_exact_source_hash(self) -> None:
        expectations = self.expectations()
        expectations[0] = {
            **expectations[0],
            "expected_source_artifact_hash": self.unified[
                "source_artifacts_sha256"
            ],
        }
        clean = R.audit_comparison_records(
            self.records(), self.populations, self.methods, expectations=expectations
        )
        self.assertTrue(clean["coverage"][0]["source_artifact_hash_matches"])

        records = self.records()
        for record in records:
            if record["method_id"] == "unified-28":
                # This raw asset hash belongs to the method, so the generic membership
                # gate accepts it; the population-specific expectation must reject it.
                record["source_artifact_hash"] = self.asset["sha256"]
        report = R.audit_comparison_records(
            records, self.populations, self.methods, expectations=expectations
        )
        self.assertEqual(report["artifact_hash_conflicts"], [])
        unified = next(
            row for row in report["coverage"] if row["method_id"] == "unified-28"
        )
        self.assertFalse(unified["source_artifact_hash_matches"])
        self.assertFalse(unified["passes"])
        self.assertFalse(report["headline_ok"])

    def test_realized_budget_expectation_uses_registered_ids_not_budget_bins(self) -> None:
        records = self.records()
        for index, record in enumerate(records):
            record["budget"] = 10 + index
        expectation = {
            "population_id": "pb-global",
            "method_id": "unified-28",
            "budget": "registered-realized-tokens",
            "match_any_budget": True,
            "eligible_row_ids": self.population_entry["ordered_ids"],
            "eligible_ordered_id_sha256": self.population_entry["ordered_id_sha256"],
        }
        # Restrict to one method so the other method does not correctly appear as
        # an unexpected unregistered expectation group.
        method_records = [row for row in records if row["method_id"] == "unified-28"]
        report = R.audit_comparison_records(
            method_records,
            self.populations,
            self.methods,
            expectations=[expectation],
        )
        self.assertTrue(report["ok"])
        self.assertTrue(report["coverage"][0]["match_any_budget"])
        self.assertEqual(report["coverage"][0]["n_expected"], 3)
        duplicate_budget = dict(method_records[0])
        duplicate_budget["budget"] = 999
        bad = R.audit_comparison_records(
            [*method_records, duplicate_budget],
            self.populations,
            self.methods,
            expectations=[expectation],
        )
        self.assertFalse(bad["headline_ok"])


class HashManifestTests(FairRegistryFixtures):
    def test_manifest_is_deterministic_and_detects_mutation(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "b.txt").write_text("b\n", encoding="utf-8")
        (nested / "a.txt").write_text("a\n", encoding="utf-8")
        first = R.build_hash_manifest(self.root)
        second = R.build_hash_manifest(self.root)
        self.assertEqual(first, second)
        self.assertEqual([row["path"] for row in first["files"]], sorted(row["path"] for row in first["files"]))
        R.write_canonical_json(self.root / "HASH_MANIFEST.json", first)
        self.assertTrue(R.verify_hash_manifest(self.root, first)["ok"])
        (nested / "a.txt").write_text("changed\n", encoding="utf-8")
        verification = R.verify_hash_manifest(self.root, first)
        self.assertFalse(verification["ok"])
        self.assertTrue(any(problem.get("path") == "nested/a.txt" for problem in verification["problems"]))

    def test_manifest_rejects_path_outside_root(self) -> None:
        outside = self.root.parent / "outside-fixture.txt"
        outside.write_text("outside", encoding="utf-8")
        try:
            with self.assertRaises(R.RegistryError):
                R.build_hash_manifest(self.root, include=[outside])
        finally:
            outside.unlink()


class ReportingTests(FairRegistryFixtures):
    def report_row(self, method: dict, fidelity: str | None = None) -> dict:
        row = {
            "lane": "global",
            "method_id": method["method_id"],
            "method": method["display_name"],
            "auroc": 0.7,
            "eligible population hash": _hash("eligible-population"),
            "evaluator hash": _hash("shared-evaluator"),
            "fidelity": fidelity or method["fidelity"],
            **method["access"],
        }
        if method["method_id"] == "unified-28":
            row["paired delta vs incumbent"] = "+0.0100 [-0.0100, +0.0300]"
        return row

    def direct_table(self) -> dict:
        return {
            "table_id": "pb-global",
            "title": "Llama ProcessBench 3,400",
            "lane": "global",
            "required_method_ids": ["unified-28", "classic-mixed-v2-no-length"],
            "direct_claim_contract": {
                "eligible_population_hash_fields": ["eligible population hash"],
                "evaluator_hash_field": "evaluator hash",
                "paired_intervals": [
                    {
                        "left_method_id": "unified-28",
                        "right_method_id": "classic-mixed-v2-no-length",
                        "field": "paired delta vs incumbent",
                    }
                ],
            },
            "columns": [
                ("method", "Method"),
                ("auroc", "AUROC"),
                ("fidelity", "Fidelity"),
                ("input_type", "Input"),
            ],
            "rows": [self.report_row(self.unified), self.report_row(self.incumbent)],
        }

    def context_table(self) -> dict:
        row = self.report_row(self.incumbent, fidelity="published-context-only")
        return {"title": "Published context", "lane": "global", "columns": ["method", "fidelity"], "rows": [row]}

    def blocked_table(self) -> dict:
        row = self.report_row(self.incumbent, fidelity="blocked-assets")
        row["headline_eligible"] = False
        row["status"] = "blocked-assets"
        return {"title": "Missing assets", "lane": "global", "columns": ["method", "status", "fidelity"], "rows": [row]}

    def test_markdown_orders_direct_context_then_blocked(self) -> None:
        report = P.render_markdown_report(
            title="Fair comparisons",
            direct_tables=[self.direct_table()],
            native_context_tables=[self.context_table()],
            partial_blocked_tables=[self.blocked_table()],
            provenance={"evaluator_sha256": _hash("eval")},
        )
        self.assertLess(report.index("## Direct comparisons"), report.index("## Native-paper and context"))
        self.assertLess(report.index("## Native-paper and context"), report.index("## Partial and blocked coverage"))
        self.assertIn("Unified-28", report)

    def test_html_defaults_to_direct_only_and_is_repeatable(self) -> None:
        kwargs = dict(
            title="Fair comparisons",
            direct_tables=[self.direct_table()],
            native_context_tables=[self.context_table()],
            partial_blocked_tables=[self.blocked_table()],
        )
        first = P.render_advisor_html(**kwargs)
        second = P.render_advisor_html(**kwargs)
        self.assertEqual(first, second)
        self.assertIn('<html lang="en" data-default-view="direct">', first)
        self.assertIn('id="panel-direct" class="panel" role="tabpanel" aria-labelledby="tab-direct">', first)
        self.assertIn('id="panel-context" class="panel" role="tabpanel" aria-labelledby="tab-context" hidden>', first)
        self.assertIn('id="panel-partial" class="panel" role="tabpanel" aria-labelledby="tab-partial" hidden>', first)
        self.assertLess(first.index("panel-direct"), first.index("panel-context"))
        self.assertLess(first.index("panel-context"), first.index("panel-partial"))

    def test_direct_table_rejects_context_or_blocked_rows(self) -> None:
        table = self.direct_table()
        table["rows"][0]["fidelity"] = "blocked-assets"
        with self.assertRaisesRegex(R.RegistryError, "cannot enter"):
            P.render_markdown_report(title="bad", direct_tables=[table])

    def test_required_unified_and_incumbent_gate(self) -> None:
        table = self.direct_table()
        table["rows"] = [table["rows"][0]]
        with self.assertRaisesRegex(R.RegistryError, "missing required methods"):
            P.render_markdown_report(title="bad", direct_tables=[table])

    def test_direct_claim_rejects_population_evaluator_or_interval_drift(self) -> None:
        table = self.direct_table()
        table["rows"][1]["eligible population hash"] = _hash("different-population")
        with self.assertRaisesRegex(R.RegistryError, "does not share eligible population hash"):
            P.render_markdown_report(title="bad", direct_tables=[table])

        table = self.direct_table()
        table["rows"][1]["evaluator hash"] = _hash("different-evaluator")
        with self.assertRaisesRegex(R.RegistryError, "does not share one evaluator hash"):
            P.render_markdown_report(title="bad", direct_tables=[table])

        table = self.direct_table()
        del table["rows"][0]["paired delta vs incumbent"]
        with self.assertRaisesRegex(R.RegistryError, "lacks a valid paired interval"):
            P.render_markdown_report(title="bad", direct_tables=[table])

    def test_direct_claim_contract_is_mandatory(self) -> None:
        table = self.direct_table()
        del table["direct_claim_contract"]
        with self.assertRaisesRegex(R.RegistryError, "must declare direct_claim_contract"):
            P.render_markdown_report(title="bad", direct_tables=[table])

    def test_partition_keeps_complete_partial_fidelity_direct(self) -> None:
        complete = self.report_row(self.unified, fidelity="paper-specified-partial")
        incomplete = {**complete, "method_id": "partial", "status": "incomplete 512/1000"}
        published = {**complete, "method_id": "paper", "fidelity": "published-context-only"}
        partitioned = P.partition_rows([complete, incomplete, published])
        self.assertEqual([row["method_id"] for row in partitioned[P.DIRECT_TIER]], ["unified-28"])
        self.assertEqual([row["method_id"] for row in partitioned[P.PARTIAL_TIER]], ["partial"])
        self.assertEqual([row["method_id"] for row in partitioned[P.CONTEXT_TIER]], ["paper"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
