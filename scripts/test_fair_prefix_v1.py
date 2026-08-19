#!/usr/bin/env python3
"""CPU-only regression and read-only real-asset audits for Prefix lane v1."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.fair_comparisons import prefix as P  # noqa: E402
from spectral_utils.fair_comparisons import processbench as PB  # noqa: E402
from spectral_utils.fair_comparisons import stopping as S  # noqa: E402
from spectral_utils.fair_comparisons.registry import (  # noqa: E402
    make_comparison_record,
    validate_comparison_record,
)


SOURCE_HASH = "a" * 64


@dataclass(frozen=True)
class _PopulationRow:
    row_id: str
    wrong_label: int
    trace_length: int
    model: str = "llama31_8b"


@dataclass(frozen=True)
class _Population:
    rows: dict[str, _PopulationRow]
    ordered_ids: tuple[str, ...]


def _population(*specs: tuple[str, str, int, int, str]) -> _Population:
    rows: dict[str, _PopulationRow] = {}
    ordered: list[str] = []
    for subset, official_id, wrong, length, model in specs:
        row_id = PB.canonical_processbench_id(subset, official_id)
        rows[row_id] = _PopulationRow(row_id, wrong, length, model)
        ordered.append(row_id)
    return _Population(rows, tuple(ordered))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _comparison(
    *,
    row_id: str,
    cell_id: str,
    method_id: str,
    budget: int | str,
    label: int,
    score: float,
    final_length: int = 600,
    group_id: str | None = None,
    family: str = "gsm8k",
) -> dict[str, Any]:
    return make_comparison_record(
        lane="prefix",
        population_id="fixture-source",
        row_id=row_id,
        group_id=group_id or row_id,
        cell_id=cell_id,
        method_id=method_id,
        continuous_score=score,
        discrete_prediction=None,
        label=label,
        budget=budget,
        fold=None,
        calibration_hash=None,
        source_artifact_hash=SOURCE_HASH,
        extra={
            "family": family,
            "model": "llama31_8b",
            "final_length": final_length,
            "source_question_id": row_id.rsplit("::", 1)[-1],
            "source_kind": "test_fixture",
        },
    )


def _complete_method_records(
    method_id: str,
    row_id: str,
    cell_id: str,
    *,
    label: int,
    final_length: int = 600,
) -> list[dict[str, Any]]:
    budgets: list[int | str] = [
        budget for budget in P.PREFIX_BUDGETS if final_length > budget
    ] + ["final"]
    return [
        _comparison(
            row_id=row_id,
            cell_id=cell_id,
            method_id=method_id,
            budget=budget,
            label=label,
            score=float(label) + (0.001 * index),
            final_length=final_length,
        )
        for index, budget in enumerate(budgets)
    ]


class FrozenConstantsTests(unittest.TestCase):
    def test_budget_grid_and_step272_identity_are_exact(self):
        self.assertEqual(P.PREFIX_BUDGETS, (16, 32, 64, 128, 256, 512))
        self.assertEqual(
            P.SELECTED_STEP272_ARCHITECTURE,
            "a_two_global_local__w0.50__peak",
        )
        self.assertEqual(
            P.DIRECT_REQUIRED_METHODS,
            (P.UNIFIED28_METHOD_ID, P.STEP272_METHOD_ID),
        )
        self.assertEqual(
            P.FROZEN_PREFIX_REPLAY_REVISION,
            "frozen_prefix_incumbent_replay_v1.0.0",
        )


class SourceAdapterTests(unittest.TestCase):
    HISTORICAL_FIELDS = [
        "budget",
        "cell_id",
        "family",
        "group",
        "is_final",
        "label_error",
        "length_band",
        "method",
        "score",
        "trace_id",
        "trace_length",
        "unit_index",
    ]
    STEP_FIELDS = [
        "architecture",
        "budget",
        "family",
        "locator",
        "model",
        "prediction",
        "score",
        "target",
        "task",
        "unit",
    ]

    def _historical_rows(self, *, trace_length: int = 600) -> list[dict[str, Any]]:
        rows = []
        for method in (P.IU28_METHOD_ID, "deepconf_entropy_w64"):
            for budget in P.PREFIX_BUDGETS:
                if trace_length > budget:
                    rows.append(
                        {
                            "budget": budget,
                            "cell_id": "processbench_gsm8k__llama31_8b",
                            "family": "gsm8k",
                            "group": "gsm8k-0",
                            "is_final": False,
                            "label_error": 1,
                            "length_band": "fixture",
                            "method": method,
                            "score": budget / 1000,
                            "trace_id": "gsm8k-0",
                            "trace_length": trace_length,
                            "unit_index": 0,
                        }
                    )
            rows.append(
                {
                    "budget": trace_length,
                    "cell_id": "processbench_gsm8k__llama31_8b",
                    "family": "gsm8k",
                    "group": "gsm8k-0",
                    "is_final": True,
                    "label_error": 1,
                    "length_band": "fixture",
                    "method": method,
                    "score": 0.9,
                    "trace_id": "gsm8k-0",
                    "trace_length": trace_length,
                    "unit_index": 0,
                }
            )
        return rows

    def _step_rows(self, *, target: int = 1) -> list[dict[str, Any]]:
        rows = [
            {
                "architecture": P.SELECTED_STEP272_ARCHITECTURE,
                "budget": budget,
                "family": "gsm8k",
                "locator": "",
                "model": "llama31_8b",
                "prediction": "",
                "score": budget / 1000,
                "target": target,
                "task": "online",
                "unit": "gsm8k-0",
            }
            for budget in P.PREFIX_BUDGETS
        ]
        rows.append(
            {
                "architecture": P.SELECTED_STEP272_ARCHITECTURE,
                "budget": "final",
                "family": "gsm8k",
                "locator": "",
                "model": "llama31_8b",
                "prediction": "",
                "score": 0.95,
                "target": target,
                "task": "global",
                "unit": "gsm8k-0",
            }
        )
        return rows

    def test_historical_loader_canonicalizes_ids_cells_and_final_budget(self):
        population = _population(("gsm8k", "gsm8k-0", 1, 600, "llama31_8b"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.csv"
            _write_csv(path, self.HISTORICAL_FIELDS, self._historical_rows())
            loaded = P.load_historical_prefix_scores(path, population=population)

        self.assertEqual(len(loaded["records"]), 14)
        self.assertEqual(
            {row["row_id"] for row in loaded["records"]},
            {PB.canonical_processbench_id("gsm8k", "gsm8k-0")},
        )
        self.assertEqual(
            {row["cell_id"] for row in loaded["records"]},
            {"processbench@e8024636bcab::llama31_8b::gsm8k"},
        )
        self.assertEqual(
            {row["budget"] for row in loaded["records"]},
            {*P.PREFIX_BUDGETS, "final"},
        )
        self.assertIn(P.HISTORICAL_DEEPCONF_METHOD_ID, loaded["audit"]["method_ids"])
        self.assertTrue(loaded["audit"]["context_only"])
        self.assertTrue(all(not row["direct_eligible"] for row in loaded["records"]))
        self.assertFalse(loaded["audit"]["positional_fallback"])

    def test_historical_loader_rejects_length_equal_to_budget(self):
        population = _population(("gsm8k", "gsm8k-0", 1, 16, "llama31_8b"))
        bad = self._historical_rows(trace_length=16)
        # The fixture generator correctly omits b16; inject an invalid legacy row.
        bad.insert(0, {**bad[0], "budget": 16, "is_final": False})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.csv"
            _write_csv(path, self.HISTORICAL_FIELDS, bad)
            with self.assertRaisesRegex(P.PrefixIntegrityError, "strict causal gate"):
                P.load_historical_prefix_scores(path, population=population)

    def test_historical_question_id_match_does_not_imply_trace_compatibility(self):
        population = _population(("gsm8k", "gsm8k-0", 1, 601, "llama31_8b"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.csv"
            _write_csv(path, self.HISTORICAL_FIELDS, self._historical_rows())
            loaded = P.load_historical_prefix_scores(path, population=population)
        self.assertEqual(loaded["audit"]["registered_comparable_traces"], 1)
        self.assertEqual(loaded["audit"]["registered_length_match_traces"], 0)
        self.assertEqual(loaded["audit"]["registered_label_match_traces"], 1)
        self.assertTrue(
            all(record["registered_length_match"] is False for record in loaded["records"])
        )
        self.assertTrue(all(not record["direct_eligible"] for record in loaded["records"]))

    def test_unified28_loader_is_id_joined_complete_and_causally_gated(self):
        population = _population(("gsm8k", "gsm8k-0", 1, 600, "llama31_8b"))
        source = {
            "candidate": "base7_full28",
            "family": "gsm8k",
            "unit": "gsm8k-0",
            "source_group": "gsm8k::gsm8k-0",
            "model": "llama31_8b",
            "wrong": 1,
            "global_score": 0.9,
            **{f"risk_at_{budget}": budget / 1000 for budget in P.PREFIX_BUDGETS},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unified.jsonl"
            path.write_text(json.dumps(source) + "\n", encoding="utf-8")
            loaded = P.load_unified28_prefix_records(path, population)

        self.assertEqual(len(loaded["records"]), 7)
        self.assertEqual(loaded["audit"]["population_coverage"], 1.0)
        self.assertTrue(all(row["method_id"] == P.UNIFIED28_METHOD_ID for row in loaded["records"]))
        self.assertTrue(all(validate_comparison_record(row) == row for row in loaded["records"]))

    def test_unified28_loader_rejects_label_disagreement(self):
        population = _population(("gsm8k", "gsm8k-0", 0, 600, "llama31_8b"))
        source = {
            "candidate": "base7_full28",
            "family": "gsm8k",
            "unit": "gsm8k-0",
            "source_group": "gsm8k::gsm8k-0",
            "model": "llama31_8b",
            "wrong": 1,
            "global_score": 0.9,
            **{f"risk_at_{budget}": 0.1 for budget in P.PREFIX_BUDGETS},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unified.jsonl"
            path.write_text(json.dumps(source) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(P.PrefixIntegrityError, "label disagreement"):
                P.load_unified28_prefix_records(path, population)

    def test_step272_loader_selects_only_exact_frozen_architecture(self):
        population = _population(("gsm8k", "gsm8k-0", 1, 600, "llama31_8b"))
        rows = self._step_rows()
        rows.append({**rows[0], "architecture": "a_two_global_local__w0.40__peak"})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "architecture.csv"
            _write_csv(path, self.STEP_FIELDS, rows)
            loaded = P.load_step272_prefix_records(path, population)

        self.assertEqual(len(loaded["records"]), 7)
        self.assertEqual(loaded["audit"]["architecture"], P.SELECTED_STEP272_ARCHITECTURE)
        self.assertEqual({row["budget"] for row in loaded["records"]}, {*P.PREFIX_BUDGETS, "final"})

    def test_step272_loader_rejects_emitted_completed_prefix(self):
        population = _population(("gsm8k", "gsm8k-0", 1, 512, "llama31_8b"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "architecture.csv"
            _write_csv(path, self.STEP_FIELDS, self._step_rows())
            with self.assertRaisesRegex(P.PrefixIntegrityError, "ineligible prefix"):
                P.load_step272_prefix_records(path, population)

    def test_strict_provenance_mode_requires_registered_manifests(self):
        population = _population(("gsm8k", "gsm8k-0", 1, 600, "llama31_8b"))
        unified = {
            "candidate": "base7_full28",
            "family": "gsm8k",
            "unit": "gsm8k-0",
            "source_group": "gsm8k::gsm8k-0",
            "model": "llama31_8b",
            "wrong": 1,
            "global_score": 0.9,
            **{f"risk_at_{budget}": 0.1 for budget in P.PREFIX_BUDGETS},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unified_path = root / "unified.jsonl"
            unified_path.write_text(json.dumps(unified) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(P.PrefixIntegrityError, "provenance is unavailable"):
                P.load_unified28_prefix_records(
                    unified_path,
                    population,
                    require_registered_telemetry_provenance=True,
                )
            step_path = root / "architecture.csv"
            _write_csv(step_path, self.STEP_FIELDS, self._step_rows())
            with self.assertRaisesRegex(P.PrefixIntegrityError, "provenance is unavailable"):
                P.load_step272_prefix_records(
                    step_path,
                    population,
                    require_registered_telemetry_provenance=True,
                )

    def test_entropy_controls_are_prefix_only_and_require_explicit_ids(self):
        population = _population(("gsm8k", "gsm8k-0", 1, 600, "llama31_8b"))
        row = {
            "id": "gsm8k-0",
            "gen_token_ids": list(range(600)),
            "token_entropies": [float(value) for value in range(600)],
            "final_answer_correct": False,
        }
        rows = {subset: [] for subset in PB.PROCESSBENCH_SUBSETS}
        rows["gsm8k"] = [row]
        hashes = {subset: SOURCE_HASH for subset in PB.PROCESSBENCH_SUBSETS}
        loaded = P.build_entropy_prefix_records(
            rows,
            population,
            source_artifact_hashes=hashes,
            include_row_ids=set(population.ordered_ids),
        )
        keyed = {(record["method_id"], record["budget"]): record for record in loaded["records"]}
        self.assertEqual(keyed[(P.MEAN_ENTROPY_METHOD_ID, 16)]["continuous_score"], 7.5)
        self.assertEqual(keyed[(P.MAX_ENTROPY_METHOD_ID, 16)]["continuous_score"], 15.0)

        changed = dict(row)
        changed["token_entropies"] = row["token_entropies"][:16] + [10_000.0] * 584
        changed_rows = {subset: [] for subset in PB.PROCESSBENCH_SUBSETS}
        changed_rows["gsm8k"] = [changed]
        rebuilt = P.build_entropy_prefix_records(
            changed_rows,
            population,
            source_artifact_hashes=hashes,
            include_row_ids=set(population.ordered_ids),
        )
        rebuilt_keyed = {
            (record["method_id"], record["budget"]): record
            for record in rebuilt["records"]
        }
        self.assertEqual(
            keyed[(P.MEAN_ENTROPY_METHOD_ID, 16)]["continuous_score"],
            rebuilt_keyed[(P.MEAN_ENTROPY_METHOD_ID, 16)]["continuous_score"],
        )

        del row["id"]
        with self.assertRaisesRegex(P.PrefixIntegrityError, "positional fallback forbidden"):
            P.build_entropy_prefix_records(
                rows,
                population,
                source_artifact_hashes=hashes,
                include_row_ids=set(population.ordered_ids),
            )

    def test_frozen_incumbent_replay_fails_closed_without_original_fit_assets(self):
        population = _population(("gsm8k", "gsm8k-0", 1, 600, "llama31_8b"))
        rows = {subset: [] for subset in PB.PROCESSBENCH_SUBSETS}
        hashes = {subset: SOURCE_HASH for subset in PB.PROCESSBENCH_SUBSETS}
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "absent"
            with self.assertRaisesRegex(P.PrefixIntegrityError, "replay root is missing"):
                P.replay_frozen_prefix_incumbents(
                    rows,
                    population,
                    historical_results_root=missing,
                    source_artifact_hashes=hashes,
                )


class IdenticalPopulationAssemblyTests(unittest.TestCase):
    def test_direct_population_is_label_blind_intersection_and_never_fabricates(self):
        cell = "processbench@e8024636bcab::llama31_8b::gsm8k"
        row_a = PB.canonical_processbench_id("gsm8k", "gsm8k-0")
        row_b = PB.canonical_processbench_id("gsm8k", "gsm8k-1")
        other_cell = "historical::other-cell"
        other_row = "historical::math500::0:0"
        records: list[dict[str, Any]] = []
        for row_id, label in ((row_a, 0), (row_b, 1)):
            records.extend(_complete_method_records(P.IU28_METHOD_ID, row_id, cell, label=label))
        for method in (P.UNIFIED28_METHOD_ID, P.STEP272_METHOD_ID, P.MEAN_ENTROPY_METHOD_ID):
            records.extend(_complete_method_records(method, row_a, cell, label=0))
        records.extend(
            _complete_method_records(P.IU28_METHOD_ID, other_row, other_cell, label=1)
        )

        assembled = P.assemble_historical_common_panel(records)
        coverage = {row["cell_id"]: row for row in assembled["coverage"]}
        self.assertEqual(coverage[cell]["reference_traces"], 2)
        self.assertEqual(coverage[cell]["direct_common_traces"], 1)
        self.assertTrue(coverage[cell]["direct_table_eligible"])
        self.assertFalse(coverage[other_cell]["direct_table_eligible"])
        self.assertEqual(assembled["audit"]["n_reference_cells"], 2)
        self.assertEqual(assembled["audit"]["n_direct_cells"], 1)

        direct = assembled["records"]
        self.assertEqual({record["row_id"] for record in direct}, {row_a})
        self.assertNotIn(row_b, {record["row_id"] for record in direct})
        self.assertEqual(
            {record["method_id"] for record in direct},
            {
                P.UNIFIED28_METHOD_ID,
                P.IU28_METHOD_ID,
                P.STEP272_METHOD_ID,
                P.MEAN_ENTROPY_METHOD_ID,
            },
        )
        keys_by_method: dict[str, set[tuple[str, int | str]]] = {}
        for method in {record["method_id"] for record in direct}:
            keys_by_method[method] = {
                (record["row_id"], record["budget"])
                for record in direct
                if record["method_id"] == method
            }
        self.assertEqual(len({frozenset(keys) for keys in keys_by_method.values()}), 1)
        self.assertEqual(assembled["populations"][0]["n_rows"], 1)
        self.assertFalse(assembled["populations"][0]["outcome_filtering"])

    def test_context_iu_is_pending_replay_not_silently_direct_joined(self):
        cell = "processbench@e8024636bcab::llama31_8b::gsm8k"
        row_id = PB.canonical_processbench_id("gsm8k", "gsm8k-0")
        direct = []
        for method in (P.UNIFIED28_METHOD_ID, P.STEP272_METHOD_ID):
            direct.extend(_complete_method_records(method, row_id, cell, label=1))
        context = _complete_method_records(
            P.IU28_METHOD_ID,
            row_id,
            cell,
            label=1,
            final_length=500,
        )
        for record in context:
            record["direct_eligible"] = False
            record["direct_ineligibility_reason"] = "different telemetry realization"
            record["input_telemetry_revision"] = P.HISTORICAL_PREFIX_TELEMETRY
        assembled = P.assemble_historical_common_panel([*direct, *context])
        self.assertEqual(
            {record["method_id"] for record in assembled["records"]},
            {P.UNIFIED28_METHOD_ID, P.STEP272_METHOD_ID},
        )
        coverage = assembled["coverage"][0]
        self.assertIn(P.IU28_METHOD_ID, coverage["context_only_method_ids"])
        self.assertIn(P.IU28_METHOD_ID, coverage["pending_registered_cpu_replay"])
        self.assertEqual(coverage["direct_common_traces"], 1)
        self.assertFalse(assembled["audit"]["question_id_equality_implies_trace_equality"])

    def test_population_registers_metric_specific_prefix_eligibility_hashes(self):
        cell = "processbench@e8024636bcab::llama31_8b::gsm8k"
        short_id = PB.canonical_processbench_id("gsm8k", "gsm8k-short")
        long_id = PB.canonical_processbench_id("gsm8k", "gsm8k-long")
        records = []
        for method in (P.UNIFIED28_METHOD_ID, P.STEP272_METHOD_ID):
            records.extend(
                _complete_method_records(
                    method, short_id, cell, label=0, final_length=100
                )
            )
            records.extend(
                _complete_method_records(
                    method, long_id, cell, label=1, final_length=600
                )
            )
        population = P.assemble_historical_common_panel(records)["populations"][0]
        ordered = population["ordered_row_ids"]
        eligible = population["eligible_populations"]
        self.assertEqual(
            eligible["budget_64"]["ordered_ids"], ordered
        )
        self.assertEqual(
            eligible["budget_128"]["ordered_ids"], [long_id]
        )
        self.assertEqual(
            eligible["complete_six_budget_warning"]["ordered_ids"], [long_id]
        )
        for descriptor in eligible.values():
            self.assertEqual(
                descriptor["ordered_id_sha256"],
                P.ordered_id_sha256(descriptor["ordered_ids"]),
            )

    def test_metadata_or_duplicate_disagreement_fails_closed(self):
        cell = "cell"
        row_id = "row"
        records = _complete_method_records(P.IU28_METHOD_ID, row_id, cell, label=0)
        records.append(dict(records[0]))
        with self.assertRaisesRegex(P.PrefixIntegrityError, "duplicate"):
            P.assemble_historical_common_panel(records)


class MetricsAndWarningTests(unittest.TestCase):
    def _metric_records(self) -> list[dict[str, Any]]:
        cell = "processbench@e8024636bcab::llama31_8b::gsm8k"
        records = []
        labels = [0, 0, 1, 1]
        for index, label in enumerate(labels):
            row_id = PB.canonical_processbench_id("gsm8k", f"gsm8k-{index}")
            for budget in P.PREFIX_BUDGETS:
                if budget == 16:
                    score = 0.5
                elif budget == 32:
                    score = [0.0, 0.8, 0.4, 1.0][index]
                else:
                    score = [0.0, 0.1, 0.9, 1.0][index]
                records.append(
                    _comparison(
                        row_id=row_id,
                        cell_id=cell,
                        method_id=P.UNIFIED28_METHOD_ID,
                        budget=budget,
                        label=label,
                        score=score,
                    )
                )
            records.append(
                _comparison(
                    row_id=row_id,
                    cell_id=cell,
                    method_id=P.UNIFIED28_METHOD_ID,
                    budget="final",
                    label=label,
                    score=[0.0, 0.1, 0.9, 1.0][index],
                )
            )
        return records

    def test_per_budget_metrics_include_ap_normalized_ap_and_recovery(self):
        result = P.summarize_prefix_metrics(self._metric_records())
        per_budget = {row["budget"]: row for row in result["per_budget"]}
        self.assertEqual(per_budget[16]["auroc"], 0.5)
        self.assertEqual(per_budget[16]["error_auprc"], 0.5)
        self.assertEqual(per_budget[16]["prevalence_normalized_ap"], 0.0)
        self.assertEqual(per_budget[64]["auroc"], 1.0)
        self.assertEqual(per_budget[64]["recovered_above_chance_signal"], 1.0)
        summary = result["per_cell_method"][0]
        self.assertEqual(summary["primary_mean_auroc_64_128"], 1.0)
        self.assertEqual(summary["earliest_budget_reaching_95pct_final_signal"], 64)

    def test_warning_inputs_require_complete_six_budget_unfinished_paths(self):
        cell = "cell"
        complete = _complete_method_records(
            P.UNIFIED28_METHOD_ID, "complete", cell, label=0, final_length=600
        )
        short = _complete_method_records(
            P.UNIFIED28_METHOD_ID, "short", cell, label=1, final_length=500
        )
        result = P.build_warning_inputs(complete + short)
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["row_id"], "complete")
        self.assertEqual(list(result["rows"][0]["score_path"]), list(P.PREFIX_BUDGETS))
        self.assertEqual(result["audit"]["incomplete_or_short_paths"], 1)
        self.assertFalse(result["audit"]["thresholds_fitted"])
        self.assertEqual(
            result["audit"]["required_calibration"],
            "maximum_over_complete_six_budget_correct_trace_path",
        )


class S2SchemaAuditTests(unittest.TestCase):
    def _row(self) -> dict[str, Any]:
        return {
            "question_id": "gsm8k:0",
            "arm": "cot",
            "setting_label": "central",
            "gen_token_ids": [1, 2],
            "channels": {
                "raw_entropy": [0.1, 0.2],
                "raw_logsumexp": [4.0, 4.1],
                "raw_margin": [0.8, 0.7],
            },
            "raw_top_k_logprobs": {"logprobs": [[-0.1], [-0.2]]},
        }

    def test_raw_driver_aliases_pass_schema_but_do_not_authorize_scoring(self):
        audit = P.audit_s2_cot_telemetry(
            [self._row()],
            dataset_revision="test",
            dataset="gsm8k",
            model="meta-llama/Llama-3.1-8B-Instruct",
            dedicated_required_fields=("channels.raw_margin",),
        )
        self.assertTrue(audit["raw_telemetry_gate_passed"])
        self.assertEqual(audit["raw_telemetry_coverage"], 1.0)
        self.assertEqual(
            audit["resolved_alias_counts"]["token_entropies"],
            {"channels.raw_entropy": 1},
        )
        self.assertFalse(audit["frozen_model_join_gate_passed"])
        self.assertFalse(audit["prefix_scoring_eligible"])
        self.assertFalse(audit["global_scoring_eligible"])
        self.assertEqual(
            audit["strict_budget_eligible_rows"],
            {"16": 0, "32": 0, "64": 0, "128": 0, "256": 0, "512": 0},
        )
        unified_gate = audit["method_gates"][P.UNIFIED28_METHOD_ID]
        self.assertFalse(unified_gate["input_contract_passed"])
        self.assertEqual(
            {
                blocker["field"]
                for blocker in unified_gate["blockers"]
                if isinstance(blocker, dict)
            },
            {"token_entropies", "top_k_logprobs"},
        )
        self.assertFalse(audit["scores_materialized"])

    def test_exact_sidecar_and_verified_bindings_can_pass_dynamic_gate(self):
        row = self._row()
        row.update(
            {
                "gen_token_ids": list(range(600)),
                "token_entropies": [0.1] * 600,
                "token_spilled_energies": [0.2] * 600,
                "token_logsumexp": [10.0] * 600,
                "top_k_logprobs": {
                    "ids": [[1, 2]] * 600,
                    "logprobs": [[-0.1, -2.0]] * 600,
                },
            }
        )
        bindings = {
            method: {
                "anchor_verified": True,
                "target_binding_verified": True,
                "binding_id": f"fixture::{method}",
            }
            for method in P.S2_PREFIX_METHOD_INPUTS
        }
        audit = P.audit_s2_cot_telemetry(
            [row],
            dataset_revision="test",
            dataset="gsm8k",
            model="model",
            method_bindings=bindings,
            ordered_question_ids=["gsm8k:0"],
        )
        self.assertTrue(audit["frozen_model_join_gate_passed"])
        self.assertTrue(audit["prefix_scoring_eligible"])
        self.assertTrue(audit["global_model_join_gate_passed"])
        self.assertTrue(audit["global_scoring_eligible"])
        self.assertTrue(
            all(gate["input_contract_passed"] for gate in audit["method_gates"].values())
        )
        self.assertEqual(
            audit["strict_budget_eligible_rows"],
            {"16": 1, "32": 1, "64": 1, "128": 1, "256": 1, "512": 1},
        )
        self.assertEqual(audit["ordered_ids_source"], "declared_manifest_question_order")

    def test_missing_stream_duplicate_or_missing_identity_fails_closed(self):
        missing = self._row()
        del missing["raw_top_k_logprobs"]
        audit = P.audit_s2_cot_telemetry(
            [missing], dataset_revision="test", dataset="gsm8k", model="model"
        )
        self.assertFalse(audit["raw_telemetry_gate_passed"])
        self.assertIn("top_k_logprobs", audit["missing_rows"][0]["missing_fields"])

        row = self._row()
        with self.assertRaisesRegex(P.PrefixIntegrityError, "duplicate S2 COT ID"):
            P.audit_s2_cot_telemetry(
                [row, dict(row)], dataset_revision="test", dataset="gsm8k", model="model"
            )
        invalid = self._row()
        del invalid["question_id"]
        with self.assertRaisesRegex(P.PrefixIntegrityError, "positional fallback forbidden"):
            P.audit_s2_cot_telemetry(
                [invalid], dataset_revision="test", dataset="gsm8k", model="model"
            )


def run_real_asset_audit(root: Path) -> dict[str, Any]:
    """Read-only end-to-end audit over the already acquired local Prefix artifacts."""

    telemetry_paths = {
        subset: root
        / "dataset_cache"
        / "repgrid"
        / "pb_llama31_8b"
        / f"processbench_{subset}.pkl"
        for subset in PB.PROCESSBENCH_SUBSETS
    }

    telemetry, telemetry_hashes = PB.load_pickle_bundle(telemetry_paths)
    population_result = PB.build_processbench_population(
        telemetry, source_hashes=telemetry_hashes
    )
    population = population_result.population

    unified_path = (
        root
        / "results/unified_causal_subset_validation_base7_dufs_llama31_v1"
        / "VALIDATION_RECORDS.jsonl"
    )
    historical_path = root / "results/global_local_online_iu_v1/PER_QUESTION_SCORES.csv"
    architecture_path = (
        root
        / "results/global_local_online_architecture_v2/ARCHITECTURE_PER_QUESTION.csv"
    )

    unified = P.load_unified28_prefix_records(
        unified_path,
        population,
        require_registered_telemetry_provenance=True,
    )
    historical = P.load_historical_prefix_scores(historical_path, population=population)
    step272 = P.load_step272_prefix_records(
        architecture_path,
        population,
        require_registered_telemetry_provenance=True,
    )
    entropy = P.build_entropy_prefix_records(
        telemetry,
        population,
        source_artifact_hashes=telemetry_hashes,
    )
    incumbent_replay = P.replay_frozen_prefix_incumbents(
        telemetry,
        population,
        historical_results_root=(
            root / "results/early_online_localization_models_v1"
        ),
        source_artifact_hashes=telemetry_hashes,
    )
    panel = P.assemble_historical_common_panel(
        [
            *unified["records"],
            *historical["records"],
            *step272["records"],
            *entropy["records"],
            *incumbent_replay["records"],
        ]
    )
    metrics = P.summarize_prefix_metrics(panel["records"])
    warnings = P.build_warning_inputs(panel["records"])

    direct_populations = panel["populations"]
    for population_entry in direct_populations:
        methods = set(population_entry["included_methods"])
        required = set(P.DIRECT_REQUIRED_METHODS)
        if not required.issubset(methods):
            raise RuntimeError(
                f"real direct population lacks frozen required methods: {population_entry}"
            )
        expected_methods = {
            P.UNIFIED28_METHOD_ID,
            P.IU28_METHOD_ID,
            P.STEP272_METHOD_ID,
            P.MEAN_ENTROPY_METHOD_ID,
            P.MAX_ENTROPY_METHOD_ID,
            P.HISTORICAL_DEEPCONF_METHOD_ID,
        }
        if methods != expected_methods:
            raise RuntimeError(
                f"real direct population method roster drift: {population_entry}"
            )
    expected_direct_counts = {
        "processbench@e8024636bcab::llama31_8b::gsm8k": 177,
        "processbench@e8024636bcab::llama31_8b::math": 514,
        "processbench@e8024636bcab::llama31_8b::olympiadbench": 509,
        "processbench@e8024636bcab::llama31_8b::omnimath": 517,
    }
    observed_direct_counts = {
        population_entry["cell_id"]: population_entry["n_rows"]
        for population_entry in direct_populations
    }
    if observed_direct_counts != expected_direct_counts:
        raise RuntimeError(
            f"registered direct population drift: {observed_direct_counts}"
        )
    if panel["audit"]["n_reference_cells"] != 11:
        raise RuntimeError("historical coverage panel no longer has 11 cells")
    if historical["audit"]["registered_length_match_traces"] != 48:
        raise RuntimeError("historical/registered telemetry mismatch audit drifted")
    if not incumbent_replay["audit"]["all_anchor_scores_exact"]:
        raise RuntimeError("frozen incumbent replay did not reproduce every anchor")
    registered_cells = set(expected_direct_counts)
    for coverage_entry in panel["coverage"]:
        if (
            coverage_entry["cell_id"] in registered_cells
            and coverage_entry["pending_registered_cpu_replay"]
        ):
            raise RuntimeError(
                f"registered CPU replay still pending: {coverage_entry}"
            )
    for source_name, source_audit in (("unified28", unified), ("step272", step272)):
        if not source_audit["audit"]["registered_telemetry_provenance"]["verified"]:
            raise RuntimeError(f"{source_name} telemetry provenance did not verify")
    direct_population_summaries = [
        {
            key: population_entry[key]
            for key in (
                "population_id",
                "cell_id",
                "ordered_id_sha256",
                "n_rows",
                "required_methods",
                "included_methods",
                "population_construction",
            )
        }
        for population_entry in direct_populations
    ]
    return {
        "schema": "fair_prefix_real_asset_audit_v1",
        "population": population_result.audit.to_dict(),
        "population_ordered_id_sha256": population.ordered_id_sha256,
        "sources": {
            "unified28": unified["audit"],
            "historical": historical["audit"],
            "step272": step272["audit"],
            "entropy": entropy["audit"],
            "incumbent_replay": incumbent_replay["audit"],
        },
        "panel": panel["audit"],
        "coverage": panel["coverage"],
        "direct_populations": direct_population_summaries,
        "metric_rows": len(metrics["per_budget"]),
        "metric_summaries": len(metrics["per_cell_method"]),
        "warning_complete_paths": warnings["audit"]["complete_paths"],
        "warning_incomplete_or_short_paths": warnings["audit"][
            "incomplete_or_short_paths"
        ],
        "gpu_used": False,
        "artifacts_mutated": False,
    }


def run_real_s2_gate_audit(root: Path) -> dict[str, Any]:
    """Prove the six acquired COT cells cannot satisfy the frozen input contract."""

    cache_root = root / "local_cache/fair_paper_exact_comparisons_v1"
    suite = S.load_s2_suite(
        cache_root,
        verify_hashes=True,
        require_six_complete_cells=True,
    )
    # Grant every method a counterfactual perfect model/target binding.  A failed gate
    # under this stronger assumption proves the acquired input itself is sufficient to
    # block scoring, independently of any secondary model-binding issue.
    input_isolation_bindings = {
        method: {
            "anchor_verified": True,
            "target_binding_verified": True,
            "evidence_scope": "counterfactual_input_isolation_only",
        }
        for method in P.S2_PREFIX_METHOD_INPUTS
    }
    expected_counts = {"aqua": 254, "gsm8k": 300}
    cell_audits: list[dict[str, Any]] = []
    for run_dir in sorted(cache_root.glob("s2_leash_*")):
        manifest = json.loads((run_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        dataset = (
            "aqua"
            if manifest["dataset_source"] == "deepmind/aqua_rat"
            else "gsm8k"
        )
        raw_cot: list[dict[str, Any]] = []
        for line in (run_dir / "INDEX.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            with (run_dir / entry["path"]).open("rb") as handle:
                shard = pickle.load(handle)
            raw_cot.extend(
                row
                for row in shard
                if row.get("arm") == "cot" and row.get("setting_label") == "central"
            )
        audit = P.audit_s2_cot_telemetry(
            raw_cot,
            dataset_revision=str(manifest["dataset_revision"]),
            dataset=dataset,
            model=str(manifest["model_id"]),
            method_bindings=input_isolation_bindings,
            ordered_question_ids=manifest["dataset_example_ids"],
        )
        if audit["cot_rows"] != expected_counts[dataset]:
            raise RuntimeError(f"S2 COT population drift: {run_dir}")
        if not audit["raw_telemetry_gate_passed"]:
            raise RuntimeError(f"S2 raw schema unexpectedly incomplete: {run_dir}")
        if audit["prefix_scoring_eligible"] or audit["global_scoring_eligible"]:
            raise RuntimeError(f"S2 frozen input gate unexpectedly passed: {run_dir}")
        for method, gate in audit["method_gates"].items():
            if gate["input_contract_passed"]:
                raise RuntimeError(f"S2 frozen method input unexpectedly passed: {method}")
        if audit["strict_budget_eligible_rows"]["512"] != 0:
            raise RuntimeError(f"S2 acquired cap no longer excludes budget 512: {run_dir}")
        cell_audits.append(
            {
                "dataset": dataset,
                "model": manifest["model_id"],
                "cot_rows": audit["cot_rows"],
                "ordered_id_sha256": audit["ordered_id_sha256"],
                "ordered_group_id_sha256": audit["ordered_group_id_sha256"],
                "strict_budget_eligible_rows": audit["strict_budget_eligible_rows"],
                "frozen_input_contract": audit["frozen_input_contract"],
                "method_input_contract_passed": {
                    method: gate["input_contract_passed"]
                    for method, gate in audit["method_gates"].items()
                },
                "prefix_scoring_eligible_even_if_bindings_verified": audit[
                    "prefix_scoring_eligible"
                ],
                "global_scoring_eligible_even_if_bindings_verified": audit[
                    "global_scoring_eligible"
                ],
            }
        )
    return {
        "schema": "fair_s2_frozen_transfer_gate_audit_v1",
        "suite": suite["suite_audit"],
        "cells": cell_audits,
        "n_cells": len(cell_audits),
        "scores_materialized": False,
        "outcomes_used_for_gate": False,
        "gpu_used": False,
        "artifacts_mutated": False,
    }


if __name__ == "__main__":
    if "--real-s2-assets" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--real-s2-assets", action="store_true")
        parser.add_argument("--root", type=Path, default=ROOT)
        arguments = parser.parse_args()
        print(
            json.dumps(
                run_real_s2_gate_audit(arguments.root.resolve()),
                indent=2,
                sort_keys=True,
            )
        )
    elif "--real-assets" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--real-assets", action="store_true")
        parser.add_argument("--root", type=Path, default=ROOT)
        arguments = parser.parse_args()
        print(json.dumps(run_real_asset_audit(arguments.root.resolve()), indent=2, sort_keys=True))
    else:
        unittest.main(verbosity=2)
