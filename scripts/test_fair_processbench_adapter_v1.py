#!/usr/bin/env python3
"""Focused stdlib-only tests for the Fair v1 ProcessBench join adapters."""

from __future__ import annotations

import copy
import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_adapter_module():
    """Load only the adapter and folds modules, avoiding optional ML dependencies."""

    spectral = types.ModuleType("spectral_utils")
    spectral.__path__ = [str(ROOT / "spectral_utils")]
    fair = types.ModuleType("spectral_utils.fair_comparisons")
    fair.__path__ = [str(ROOT / "spectral_utils" / "fair_comparisons")]
    sys.modules["spectral_utils"] = spectral
    sys.modules["spectral_utils.fair_comparisons"] = fair
    for name in ("folds", "processbench"):
        qualified = f"spectral_utils.fair_comparisons.{name}"
        path = ROOT / "spectral_utils" / "fair_comparisons" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(qualified, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    return sys.modules["spectral_utils.fair_comparisons.processbench"]


PB = _load_adapter_module()
HASH_A = "a" * 64
HASH_B = "b" * 64


def telemetry_fixture():
    rows = {}
    definitions = {
        "gsm8k": (0, False, 16),
        "math": (-1, True, 20),
        "olympiadbench": (1, False, 20),
        # Process-correct and final-answer-wrong are distinct valid targets.
        "omnimath": (-1, False, 20),
    }
    for subset, (label, correct, length) in definitions.items():
        rows[subset] = {
            927: {
                "id": f"{subset}-0",
                "label": label,
                "final_answer_correct": correct,
                "steps": ["first", "second"],
                "step_token_spans": [(0, length // 2), (length // 2, length)],
                "gen_token_ids": list(range(length)),
                "align_diag": {"ok": True},
            }
        }
    return rows


def build_population():
    return PB.build_processbench_population(
        telemetry_fixture(),
        expected_counts={subset: 1 for subset in PB.PROCESSBENCH_SUBSETS},
    ).population


def unified_fixture(population, candidate="base7_full28"):
    rows = []
    for index, row_id in enumerate(population.ordered_ids):
        pop = population.rows[row_id]
        rows.append(
            {
                "candidate": candidate,
                "family": pop.subset,
                "unit": pop.official_id,
                "source_group": f"{pop.subset}::{pop.official_id}",
                "wrong": pop.wrong_label,
                "target_step": pop.localization_label,
                "global_score": index / 10 + 0.1,
                "localization_score": index / 10 + 0.2,
                "prediction": pop.localization_label,
                "risk_at_16": index / 10 + 0.3,
                "risk_at_32": index / 10 + 0.4,
                "risk_at_64": index / 10 + 0.5,
                "risk_at_128": index / 10 + 0.6,
                "risk_at_256": index / 10 + 0.7,
                "risk_at_512": index / 10 + 0.8,
            }
        )
    return rows


class CanonicalPopulationTests(unittest.TestCase):
    def test_canonical_key(self):
        self.assertEqual(
            PB.canonical_processbench_id("gsm8k", "gsm8k-17"),
            "processbench@e8024636bcab::gsm8k::gsm8k-17",
        )
        with self.assertRaises(ValueError):
            PB.canonical_processbench_id("gsm8k", "math-17")
        with self.assertRaises(ValueError):
            PB.canonical_processbench_id("gsm8k", "")
        with self.assertRaises(ValueError):
            PB.canonical_processbench_id("gsm8k", 17)

    def test_population_is_complete_and_hash_stable(self):
        first = PB.build_processbench_population(
            telemetry_fixture(),
            expected_counts={subset: 1 for subset in PB.PROCESSBENCH_SUBSETS},
        )
        second = PB.build_processbench_population(
            telemetry_fixture(),
            expected_counts={subset: 1 for subset in PB.PROCESSBENCH_SUBSETS},
        )
        self.assertTrue(first.audit.ok)
        self.assertEqual(first.population.ordered_ids, second.population.ordered_ids)
        self.assertEqual(first.population.ordered_id_sha256, second.population.ordered_id_sha256)
        self.assertEqual(len(first.population.ordered_id_sha256), 64)
        omni = first.population.rows[first.population.ordered_ids[-1]]
        self.assertEqual(omni.localization_label, -1)
        self.assertEqual(omni.wrong_label, 1)

    def test_mapping_key_is_never_an_id_fallback(self):
        rows = telemetry_fixture()
        broken = copy.deepcopy(rows)
        del broken["gsm8k"][927]["id"]
        result = PB.build_processbench_population(
            broken,
            expected_counts={subset: 1 for subset in PB.PROCESSBENCH_SUBSETS},
            strict=False,
        )
        self.assertFalse(result.audit.ok)
        self.assertEqual(result.audit.missing_fields[0]["fields"], ["id"])
        self.assertNotIn("gsm8k-0", "".join(result.population.ordered_ids))
        with self.assertRaises(PB.ProcessBenchJoinError):
            PB.build_processbench_population(
                broken,
                expected_counts={subset: 1 for subset in PB.PROCESSBENCH_SUBSETS},
            )

    def test_duplicate_official_id_fails(self):
        rows = telemetry_fixture()
        rows["gsm8k"] = [rows["gsm8k"][927], copy.deepcopy(rows["gsm8k"][927])]
        result = PB.build_processbench_population(
            rows,
            expected_counts={"gsm8k": 2, "math": 1, "olympiadbench": 1, "omnimath": 1},
            strict=False,
        )
        self.assertFalse(result.audit.ok)
        self.assertEqual(len(result.audit.duplicate_source_ids), 1)


class UnifiedAdapterTests(unittest.TestCase):
    def setUp(self):
        self.population = build_population()

    def test_localization_and_global_records(self):
        local = PB.adapt_unified_validation_records(
            unified_fixture(self.population),
            self.population,
            lane="localization",
            source_artifact_hash=HASH_A,
        )
        self.assertTrue(local.audit.ok)
        self.assertEqual(len(local.records), 4)
        self.assertEqual(
            [record["row_id"] for record in local.records],
            list(self.population.ordered_ids),
        )
        self.assertTrue(all(record["method_id"] == "unified28" for record in local.records))

        paired = []
        for index, pop_id in enumerate(self.population.ordered_ids):
            pop = self.population.rows[pop_id]
            paired.append(
                {
                    "family": pop.subset,
                    "unit": pop.official_id,
                    "source_group": f"{pop.subset}::{pop.official_id}",
                    "wrong": pop.wrong_label,
                    "base7_full28": index + 0.1,
                    "classic_mixed_v2_no_length": index + 0.2,
                }
            )
        global_result = PB.adapt_unified_global_records(
            paired,
            self.population,
            source_artifact_hash=HASH_A,
        )
        self.assertTrue(global_result.audit.ok)
        self.assertEqual(len(global_result.records), 8)
        self.assertEqual(
            {record["method_id"] for record in global_result.records},
            {"unified28", "classic_mixed_v2_no_length"},
        )

    def test_prefix_uses_strict_length_greater_than_budget(self):
        result = PB.adapt_unified_validation_records(
            unified_fixture(self.population),
            self.population,
            lane="prefix",
            budgets=(16,),
            source_artifact_hash=HASH_A,
        )
        self.assertTrue(result.audit.ok)
        self.assertEqual(len(result.records), 3)
        self.assertNotIn(
            PB.canonical_processbench_id("gsm8k", "gsm8k-0"),
            {record["row_id"] for record in result.records},
        )
        self.assertTrue(all(record["budget"] == 16 for record in result.records))

    def test_label_conflict_and_duplicate_fail_closed(self):
        rows = unified_fixture(self.population)
        rows[0]["wrong"] = 1 - rows[0]["wrong"]
        result = PB.adapt_unified_validation_records(
            rows,
            self.population,
            lane="global",
            source_artifact_hash=HASH_A,
            strict=False,
        )
        self.assertFalse(result.audit.ok)
        self.assertEqual(len(result.audit.label_conflicts), 1)
        with self.assertRaises(PB.ProcessBenchJoinError):
            PB.adapt_unified_validation_records(
                rows,
                self.population,
                lane="global",
                source_artifact_hash=HASH_A,
            )

        duplicate = unified_fixture(self.population)
        duplicate.append(copy.deepcopy(duplicate[0]))
        duplicate_result = PB.adapt_unified_validation_records(
            duplicate,
            self.population,
            lane="global",
            source_artifact_hash=HASH_A,
            strict=False,
        )
        self.assertFalse(duplicate_result.audit.ok)
        self.assertEqual(len(duplicate_result.audit.duplicate_record_keys), 1)


class ExternalPredictionTests(unittest.TestCase):
    def setUp(self):
        self.population = build_population()

    def external_rows(self):
        rows = {}
        for index, subset in enumerate(PB.PROCESSBENCH_SUBSETS):
            pop = self.population.rows[self.population.ids_for_subset(subset)[0]]
            rows[subset] = {
                index: {
                    "id": pop.official_id,
                    "label": pop.localization_label,
                    "prediction": None if subset == "omnimath" else pop.localization_label,
                }
            }
        return rows

    def test_unparsed_prediction_remains_a_row(self):
        result = PB.adapt_external_localization_records(
            self.external_rows(),
            self.population,
            method_id="critic_qwen25_72b_single_greedy",
            source_artifact_hashes={subset: HASH_B for subset in PB.PROCESSBENCH_SUBSETS},
        )
        self.assertTrue(result.audit.ok)
        self.assertEqual(len(result.records), 4)
        self.assertEqual(result.audit.n_unparsed, 1)
        unparsed = [r for r in result.records if r["prediction_status"] == "unparsed"]
        self.assertEqual(len(unparsed), 1)
        self.assertIsNone(unparsed[0]["discrete_prediction"])

    def test_external_mapping_key_is_not_fallback(self):
        rows = self.external_rows()
        del rows["gsm8k"][0]["id"]
        result = PB.adapt_external_localization_records(
            rows,
            self.population,
            method_id="prm",
            source_artifact_hashes=HASH_B,
            strict=False,
        )
        self.assertFalse(result.audit.ok)
        self.assertEqual(result.audit.missing_fields[0]["fields"], ["id"])

    def test_external_label_conflict_fails(self):
        rows = self.external_rows()
        rows["math"][1]["label"] = 0
        result = PB.adapt_external_localization_records(
            rows,
            self.population,
            method_id="prm",
            source_artifact_hashes=HASH_B,
            strict=False,
        )
        self.assertEqual(len(result.audit.label_conflicts), 1)
        self.assertLess(result.audit.coverage_by_method["prm"], 1.0)

    def test_eq6_question_id_is_explicit_not_positional(self):
        shards = []
        for subset in PB.PROCESSBENCH_SUBSETS:
            pop = self.population.rows[self.population.ids_for_subset(subset)[0]]
            shards.append(
                {
                    "subset": subset,
                    "question_id": f"{subset}:{pop.official_id}",
                    "label": pop.localization_label,
                    "prediction": pop.localization_label,
                }
            )
        result = PB.adapt_eq6_shard_records(
            shards,
            self.population,
            source_artifact_hash=HASH_B,
        )
        self.assertTrue(result.audit.ok)
        self.assertEqual(len(result.records), 4)

        shards[0]["question_id"] = "0"
        broken = PB.adapt_eq6_shard_records(
            shards,
            self.population,
            source_artifact_hash=HASH_B,
            strict=False,
        )
        self.assertFalse(broken.audit.ok)
        self.assertTrue(broken.audit.schema_errors)


class ComparisonRecordValidationTests(unittest.TestCase):
    def test_rejects_nonfinite_score_and_cross_question_group(self):
        population = build_population()
        record = PB.adapt_unified_validation_records(
            unified_fixture(population),
            population,
            lane="global",
            source_artifact_hash=HASH_A,
        ).records[0]
        PB.validate_comparison_record(record)

        bad_score = dict(record, continuous_score=math.nan)
        with self.assertRaises(ValueError):
            PB.validate_comparison_record(bad_score)

        bad_group = dict(record, group_id=population.ordered_ids[1])
        with self.assertRaises(ValueError):
            PB.validate_comparison_record(bad_group)

        missing_hash = dict(record)
        del missing_hash["source_artifact_hash"]
        with self.assertRaises(ValueError):
            PB.validate_comparison_record(missing_hash)


def _indexed_shard_rows(run_dir: Path):
    """Read an L1 run's indexed shards and verify every declared shard hash."""

    indexes = sorted(run_dir.glob("**/INDEX.jsonl"))
    if not indexes:
        raise FileNotFoundError(f"no INDEX.jsonl below {run_dir}")
    rows = []
    provenance = []
    for index_path in indexes:
        entries = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
        index_rows = 0
        for entry in entries:
            shard_path = index_path.parent / entry["path"]
            actual_hash = PB.sha256_file(shard_path)
            if actual_hash != entry["sha256"]:
                raise RuntimeError(f"L1 shard hash mismatch: {shard_path}")
            with shard_path.open("rb") as handle:
                shard_rows = pickle.load(handle)
            if len(shard_rows) != entry["n_traces"]:
                raise RuntimeError(f"L1 shard count mismatch: {shard_path}")
            rows.extend(shard_rows)
            index_rows += len(shard_rows)
        provenance.append(
            {
                "index": str(index_path.relative_to(run_dir)),
                "index_sha256": PB.sha256_file(index_path),
                "n_shards": len(entries),
                "n_rows": index_rows,
            }
        )
    package_hash = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return rows, package_hash, provenance


def run_real_asset_audit(root: Path, l1_dir: Path | None = None) -> dict:
    """Read-only 3,400-row join audit used by the integration package."""

    telemetry_paths = {
        subset: root / "dataset_cache" / "repgrid" / "pb_llama31_8b" / f"processbench_{subset}.pkl"
        for subset in PB.PROCESSBENCH_SUBSETS
    }
    telemetry, telemetry_hashes = PB.load_pickle_bundle(telemetry_paths)
    population_result = PB.build_processbench_population(
        telemetry,
        source_hashes=telemetry_hashes,
    )
    population = population_result.population
    report = {
        "population": population_result.audit.to_dict(),
        "population_ordered_id_sha256": population.ordered_id_sha256,
        "adapters": {},
    }

    unified_root = Path("/private/tmp/hallucination-unified-causal-iu-v1/results")
    global_path = (
        unified_root
        / "unified_causal_subset_classic30_v1"
        / "LLAMA_GLOBAL_RECORDS.jsonl"
    )
    validation_path = (
        unified_root
        / "unified_causal_subset_validation_base7_dufs_llama31_v1"
        / "VALIDATION_RECORDS.jsonl"
    )
    report["adapters"]["unified_global"] = PB.adapt_unified_global_records(
        global_path, population
    ).audit.to_dict()
    report["adapters"]["unified_localization"] = PB.adapt_unified_validation_records(
        validation_path, population, lane="localization"
    ).audit.to_dict()
    report["adapters"]["unified_prefix"] = PB.adapt_unified_validation_records(
        validation_path, population, lane="prefix"
    ).audit.to_dict()
    registered_validation_methods = {
        "base7_full28": "unified28",
        "base7_full28__dufs_l0p1": "unified28_dufs_l0p1",
        "base7_full28__dufs_l0p3": "unified28_dufs_l0p3",
        "base7_full28__dufs_l1": "unified28_dufs_l1",
        "base7_full28__dufs_l3": "unified28_dufs_l3",
        "base7_full28__rw_a0p5": "unified28_task_reweighted_a0p5_historical",
        "raw9_full36": "ordinary36_historical_control",
    }
    report["adapters"]["unified_registered_candidates_global"] = (
        PB.adapt_unified_validation_records(
            validation_path,
            population,
            lane="global",
            candidate_methods=registered_validation_methods,
        ).audit.to_dict()
    )

    external_specs = {
        "prm_qwen25math7b": (
            root / "dataset_cache" / "four_localization" / "pb_prm_qwen25math7b_full",
            "pb_prm_{subset}.pkl",
        ),
        "critic_qwen72b_single_greedy": (
            root / "dataset_cache" / "four_localization" / "pb_critic_qwen72b_full",
            "pb_critic_{subset}.pkl",
        ),
        "uprm_eq6_qwen3_8b_precontract": (
            root / "dataset_cache" / "four_localization" / "pb_uprm_baseline_qwen3_8b_full",
            "pb_uprm_base_{subset}.pkl",
        ),
    }
    for method, (directory, template) in external_specs.items():
        paths = {
            subset: directory / template.format(subset=subset)
            for subset in PB.PROCESSBENCH_SUBSETS
        }
        rows, hashes = PB.load_pickle_bundle(paths)
        report["adapters"][method] = PB.adapt_external_localization_records(
            rows,
            population,
            method_id=method,
            source_artifact_hashes=hashes,
        ).audit.to_dict()

    if l1_dir is not None:
        l1_rows, l1_hash, provenance = _indexed_shard_rows(l1_dir)
        report["l1_provenance"] = provenance
        report["adapters"]["uprm_eq6_qwen25_14b_control"] = PB.adapt_eq6_shard_records(
            l1_rows,
            population,
            source_artifact_hash=l1_hash,
        ).audit.to_dict()
    return report


if __name__ == "__main__":
    if "--real-assets" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--real-assets", action="store_true")
        parser.add_argument("--root", type=Path, default=ROOT)
        parser.add_argument("--l1-dir", type=Path)
        args = parser.parse_args()
        print(
            json.dumps(
                run_real_asset_audit(args.root.resolve(), args.l1_dir),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        unittest.main(verbosity=2)
