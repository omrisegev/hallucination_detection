#!/usr/bin/env python3
"""Deterministic synthetic tests for reconstruction reporting contracts."""

from __future__ import annotations

import csv
import importlib.util
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_utils.reconstruction_reporting.io import (  # noqa: E402
    ReleaseLayout,
    materialize_plot_data,
    read_tidy_csv,
    validate_plot_data_sources,
    write_canonical_json,
    write_parquet,
    write_tidy_csv,
)
from spectral_utils.reconstruction_reporting.query import (  # noqa: E402
    LEADERBOARD_EXPORT_SPECS,
    VIEW_NAMES,
    build_duckdb,
    export_leaderboard_csvs,
    query_results,
    query_view_sql,
)
from spectral_utils.reconstruction_reporting.registry import (  # noqa: E402
    build_registry,
    expected_coverage_rows,
    make_system_id,
    validate_registry,
    validate_result_references,
)
from spectral_utils.reconstruction_reporting.report import (  # noqa: E402
    GRAPH_DISPLAY_EDGE_LIMIT,
    _alignment_scatter,
    _continuous_lsml_panel,
    _display_edges,
    _diagnostic_summary_panels,
    _diagnostic_selector_key,
    _embedded_diagnostics,
    _family_nrm_panel,
    _marker_svg,
    _pgrd_panel,
    _su_pcr_panel,
    default_plot_manifest,
    render_report,
)
from spectral_utils.reconstruction_reporting.schemas import (  # noqa: E402
    MissingOptionalDependency,
    SchemaError,
    canonical_sha256,
    derive_aggregate_cohort_id,
    derive_cohort_id,
    derive_comparison_group_id,
    make_plot_spec,
    record_sort_key,
    rank_metric_group,
    validate_comparison_groups,
    validate_equal_unit_aggregates,
    validate_expected_coverage,
    validate_metric_record,
    validate_plot_manifest,
)


def _hash(label: str) -> str:
    return canonical_sha256({"synthetic_fixture": label})


def _method(
    method_id: str,
    display_name: str,
    color: str,
    marker: str,
    *,
    role: str = "primary",
) -> dict:
    return {
        "method_id": method_id,
        "display_name": display_name,
        "acronym_expansion": f"{display_name} synthetic expansion",
        "family_id": method_id,
        "plain_summary": "Combines frozen synthetic measurements without using outcome labels.",
        "input_operation_output": "prepared measurements → fixed fusion → one continuous score",
        "formula": "s_i = w^T x_i",
        "formula_terms": {"s_i": "score for example i", "w": "fixed label-free weight vector", "x_i": "prepared measurement vector"},
        "origin": {
            "kind": "project control",
            "title": "Synthetic test fixture",
            "year": 2026,
            "relationship": "Test-only definition; not an experimental result or paper reproduction.",
        },
        "development_history": "Created only to test reporting mechanics.",
        "inputs": "Frozen gray-box synthetic measurements.",
        "access_tier": "gray_box_single_pass",
        "supervision": "none",
        "donor_regime": "within_cell_fully_unsupervised",
        "model_passes": 1,
        "assumptions": ["The prepared measurements share a useful direction."],
        "fallbacks": ["Return the registered equal mean when the fit is undefined."],
        "limitations": ["Synthetic fixture only."],
        "role": role,
        "research_stage": "canonical",
        "references": [],
        "style": {"color": color, "marker": marker},
    }


def fixture_registry() -> dict:
    release_id = "synthetic-reporting-fixture-v1"
    feature_contract_id = "prepared-feature-fixture-v1"
    adapter_id = "localization-fixture-adapter-v1"
    methods = [
        _method("equal-mean", "Equal mean", "#315ea8", "circle", role="control"),
        _method("graph-fusion", "Graph fusion", "#b14d32", "diamond"),
    ]
    versions = []
    systems = []
    for method in methods:
        version_id = f"{method['method_id']}-v1"
        versions.append(
            {
                "method_version_id": version_id,
                "method_id": method["method_id"],
                "version_label": "synthetic fixed version",
                "definition_sha256": _hash(version_id),
                "formula": method["formula"],
                "fixed_parameters": {"synthetic": True},
                "source_paths": ["scripts/reconstruction_benchmark/test_reconstruction_reporting.py"],
                "feature_contract_id": feature_contract_id,
                "research_stage": "canonical",
            }
        )
        systems.append(
            {
                "system_id": make_system_id(version_id, adapter_id),
                "method_version_id": version_id,
                "adapter_id": adapter_id,
                "access_contract_id": "gray-single-pass-v1",
                "display_name": method["display_name"],
                "enabled": True,
            }
        )
    return build_registry(
        release_id=release_id,
        tasks=[
            {
                "task_id": "localization",
                "display_name": "First-error localization",
                "description": "Find the first erroneous reasoning step.",
                "prediction_unit": "reasoning trace",
                "primary_metric_id": "macro_f1",
                "positive_class": "first erroneous step",
                "bootstrap_unit": "source question",
            }
        ],
        datasets=[
            {
                "dataset_id": "processbench",
                "task_id": "localization",
                "display_name": "ProcessBench",
                "description": "Synthetic ProcessBench-shaped fixture; no benchmark value is copied.",
                "prediction_unit": "one reasoning trace",
                "label_definition": "Synthetic first-error step or clean trace.",
                "positive_class": "trace contains an error",
                "inclusion_reason": "Tests dataset/cell drill-down.",
                "dataset_family": "reasoning",
                "revision": "fixture-only",
                "limitations": ["Contains four invented rows for software tests only."],
                "source": {"title": "Synthetic fixture", "citation": "Not a scientific source", "url": ""},
            }
        ],
        methods=methods,
        method_versions=versions,
        adapters=[
            {
                "adapter_id": adapter_id,
                "display_name": "Synthetic localization adapter",
                "task_id": "localization",
                "plain_summary": "Maps one response score to a synthetic first-error decision.",
                "input_unit": "response",
                "output_unit": "reasoning trace",
                "definition_sha256": _hash(adapter_id),
                "source_paths": ["scripts/reconstruction_benchmark/test_reconstruction_reporting.py"],
                "limitations": ["Test-only adapter."],
            }
        ],
        systems=systems,
        access_contracts=[
            {
                "access_contract_id": "gray-single-pass-v1",
                "access_tier": "gray_box_single_pass",
                "input_type": "saved token probabilities",
                "supervision": "none",
                "model_passes_per_question": 1,
                "traces_per_question": 1,
                "donor_regime": "within_cell_fully_unsupervised",
            }
        ],
        feature_contracts=[
            {
                "feature_contract_id": feature_contract_id,
                "display_name": "Synthetic prepared features",
                "definition": "Two deterministic fixture columns; not scientific data.",
                "sha256": _hash(feature_contract_id),
            }
        ],
        evaluators=[
            {
                "evaluator_id": "localization-evaluator-fixture-v1",
                "display_name": "Synthetic macro-F1 evaluator",
                "definition": "Test-only deterministic evaluator definition.",
                "sha256": _hash("evaluator"),
            }
        ],
        populations=[
            {
                "population_id": "processbench-fixture-population",
                "task_id": "localization",
                "dataset_id": "processbench",
                "display_name": "Synthetic ProcessBench population",
                "population_sha256": _hash("population"),
                "expected_n": 4,
                "group_unit": "source question",
                "eligibility_rule": "All four synthetic rows.",
            }
        ],
        cells=[
            {
                "cell_id": "processbench-gsm8k-llama-fixture",
                "population_id": "processbench-fixture-population",
                "task_id": "localization",
                "dataset_id": "processbench",
                "generation_model_id": "fixture-generator",
                "scorer_model_id": "llama-fixture",
                "split_id": "fixture-eval",
                "decoding_id": "greedy-fixture",
                "dataset_family": "gsm8k",
                "expected_n": 4,
                "status": "fixture",
            }
        ],
        slices=[
            {
                "slice_id": "processbench-gsm8k-all",
                "population_id": "processbench-fixture-population",
                "cell_id": "processbench-gsm8k-llama-fixture",
                "slice_dimension": "subset",
                "slice_value": "gsm8k",
                "display_name": "GSM8K synthetic slice",
                "expected_n": 4,
            }
        ],
        aggregations=[
            {
                "aggregation_id": "processbench-gsm8k-cell",
                "display_name": "Synthetic cell metric",
                "rule": "native_metric",
                "unit_field": "native",
                "component_ids": ["processbench-gsm8k-llama-fixture"],
                "bootstrap_unit": "source question",
                "weighting": "native official-style fixture",
            }
        ],
    )


def _fixture_cohort_id() -> str:
    return derive_cohort_id(
        {
            "row_id": f"fixture-row-{row_index}",
            "group_id": f"fixture-question-{row_index}",
            "eligible": True,
            "status": "OK",
            "continuous_score": 0.5,
        }
        for row_index in range(4)
    )


def _common(registry: dict, system_index: int) -> dict:
    system = registry["systems"][system_index]
    version = next(item for item in registry["method_versions"] if item["method_version_id"] == system["method_version_id"])
    return {
        "release_id": registry["release_id"],
        "run_id": "synthetic-run-a",
        "lane_id": "localization",
        "task_id": "localization",
        "dataset_id": "processbench",
        "population_id": "processbench-fixture-population",
        "cell_id": "processbench-gsm8k-llama-fixture",
        "slice_id": "processbench-gsm8k-all",
        "cohort_id": _fixture_cohort_id(),
        "method_id": version["method_id"],
        "method_version_id": version["method_version_id"],
        "adapter_id": system["adapter_id"],
        "system_id": system["system_id"],
        "comparison_group_id": "pending",
        "feature_contract_id": "prepared-feature-fixture-v1",
        "access_contract_id": "gray-single-pass-v1",
        "evaluator_id": "localization-evaluator-fixture-v1",
        "evidence_grade": "D0",
        "status": "OK",
        "status_detail": "",
    }


def fixture_rows(registry: dict) -> dict[str, list[dict]]:
    metrics = []
    for system_index, value in enumerate((0.61, 0.64)):
        row = {
            **_common(registry, system_index),
            "aggregation_id": "processbench-gsm8k-cell",
            "aggregation_level": "cell",
            "metric_id": "macro_f1",
            "metric_label": "Official-style macro F1 (synthetic)",
            "metric_unit": "fraction",
            "positive_class": "first erroneous step",
            "better_direction": "higher",
            "value": value,
            "ci_low": value - 0.04,
            "ci_high": value + 0.04,
            "n_rows": 4,
            "n_groups": 4,
            "n_positive": 2,
            "n_negative": 2,
            "bootstrap_unit": "source question",
            "bootstrap_draws": 20,
            "is_primary": True,
            "fidelity": "adapted-common-protocol",
            "component_ids": ["processbench-gsm8k-llama-fixture"],
        }
        row["comparison_group_id"] = derive_comparison_group_id(row)
        metrics.append(row)
    group_id = metrics[0]["comparison_group_id"]
    predictions = []
    labels = (False, True, False, True)
    scores = ((0.1, 0.7, 0.3, 0.8), (0.2, 0.8, 0.4, 0.9))
    for system_index in range(2):
        for row_index in range(4):
            predictions.append(
                {
                    **_common(registry, system_index),
                    "comparison_group_id": group_id,
                    "row_id": f"fixture-row-{row_index}",
                    "group_id": f"fixture-question-{row_index}",
                    "continuous_score": scores[system_index][row_index],
                    "discrete_prediction": scores[system_index][row_index] >= 0.5,
                    "label": labels[row_index],
                    "eligible": True,
                    "fallback_used": False,
                    "score_hash": _hash(f"score-{system_index}"),
                }
            )
    coverage = []
    for system_index in range(2):
        coverage.append(
            {
                **_common(registry, system_index),
                "comparison_group_id": group_id,
                "expected_n": 4,
                "eligible_n": 4,
                "scored_n": 4,
                "fallback_n": 0,
                "excluded_n": 0,
                "failed_n": 0,
                "coverage_fraction": 1.0,
            }
        )
    contrast = {
        **_common(registry, 1),
        "comparison_group_id": group_id,
        "aggregation_id": "processbench-gsm8k-cell",
        "aggregation_level": "cell",
        "metric_id": "macro_f1",
        "metric_unit": "fraction",
        "positive_class": "first erroneous step",
        "better_direction": "higher",
        "left_system_id": registry["systems"][1]["system_id"],
        "right_system_id": registry["systems"][0]["system_id"],
        "delta": 0.03,
        "ci_low": -0.01,
        "ci_high": 0.07,
        "wins": 1,
        "ties": 0,
        "losses": 0,
        "n_pairs": 1,
        "bootstrap_unit": "source question",
        "bootstrap_draws": 20,
        "paired": True,
        "fidelity": "adapted-common-protocol",
    }
    diagnostic = {
        **_common(registry, 1),
        "comparison_group_id": group_id,
        "graph_id": "synthetic-knn-graph",
        "graph_variant": "real",
        "graph_hash": _hash("graph"),
        "matrix_hash": _hash("matrix"),
        "diagnostic_id": "target_minus_nuisance_smoothness",
        "diagnostic_label": "Target minus nuisance smoothness",
        "diagnostic_unit": "normalized roughness difference",
        "value": 0.4,
        "null_value": 0.1,
        "effect": 0.3,
        "p_value": 0.02,
        "permutation_count": 199,
        "label_stage": "post_freeze_labels",
        "n_nodes": 4,
        "n_edges": 4,
        "notes": "Synthetic assumption check only.",
    }
    return {
        "predictions": predictions,
        "metrics": metrics,
        "contrasts": [contrast],
        "coverage": coverage,
        "graph_diagnostics": [diagnostic],
        "graph_examples": [],
    }


class RegistryAndSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = fixture_registry()
        self.rows = fixture_rows(self.registry)

    def test_registry_is_deterministic_and_method_guide_is_complete(self) -> None:
        rebuilt = validate_registry(json.loads(json.dumps(self.registry)))
        self.assertEqual(rebuilt, self.registry)
        self.assertEqual(len(self.registry["registry_sha256"]), 64)
        for method in self.registry["methods"]:
            self.assertTrue(method["plain_summary"])
            self.assertTrue(method["formula"])
            self.assertTrue(method["formula_terms"])
            self.assertTrue(method["assumptions"])

    def test_system_id_separates_version_and_adapter(self) -> None:
        system = self.registry["systems"][0]
        self.assertEqual(
            system["system_id"],
            make_system_id(system["method_version_id"], system["adapter_id"]),
        )

    def test_result_references_and_exact_comparison_group(self) -> None:
        validate_result_references(self.registry, self.rows)
        validate_comparison_groups(self.rows["metrics"])
        bad = json.loads(json.dumps(self.rows["metrics"]))
        bad[1]["access_contract_id"] = "different-contract"
        with self.assertRaisesRegex(SchemaError, "mixes incompatible"):
            validate_comparison_groups(bad)

    def test_prediction_cohort_is_derived_from_full_row_and_group_ids(self) -> None:
        bad = json.loads(json.dumps(self.rows))
        bad["predictions"][0]["row_id"] = "different-row-identity"
        with self.assertRaisesRegex(SchemaError, "not derived from its full row/group"):
            validate_result_references(self.registry, bad)

    def test_contrast_must_resolve_both_metric_sides_and_numeric_delta(self) -> None:
        missing_side = json.loads(json.dumps(self.rows))
        missing_side["metrics"] = missing_side["metrics"][1:]
        with self.assertRaisesRegex(SchemaError, "requires exactly one registered metric row"):
            validate_result_references(self.registry, missing_side)

        wrong_delta = json.loads(json.dumps(self.rows))
        wrong_delta["contrasts"][0]["delta"] = 0.02
        with self.assertRaisesRegex(SchemaError, "is not left minus right"):
            validate_result_references(self.registry, wrong_delta)

    def test_system_access_contract_is_bound_to_method_and_result_rows(self) -> None:
        registry = json.loads(json.dumps(self.registry))
        registry["access_contracts"][0]["supervision"] = "uses labels"
        registry.pop("registry_sha256")
        with self.assertRaisesRegex(SchemaError, "access contract disagrees"):
            validate_registry(registry)

        rows = json.loads(json.dumps(self.rows))
        rows["metrics"][0]["access_contract_id"] = "unregistered-access"
        with self.assertRaisesRegex(SchemaError, "unknown access_contract_id"):
            validate_result_references(self.registry, rows)

    def test_adapter_task_is_checked_on_every_result_row(self) -> None:
        registry = json.loads(json.dumps(self.registry))
        registry["tasks"].append(
            {
                "task_id": "other-task",
                "display_name": "Other task",
                "description": "Synthetic incompatible task.",
                "prediction_unit": "other unit",
                "primary_metric_id": "other_metric",
                "positive_class": "other positive",
                "bootstrap_unit": "other group",
            }
        )
        registry.pop("registry_sha256")
        registry = validate_registry(registry)
        rows = json.loads(json.dumps(self.rows))
        rows["metrics"][0]["task_id"] = "other-task"
        with self.assertRaisesRegex(SchemaError, "adapter/task mismatch"):
            validate_result_references(registry, rows)

    def test_comparison_group_id_is_content_addressed(self) -> None:
        bad = json.loads(json.dumps(self.rows["metrics"]))
        for row in bad:
            row["comparison_group_id"] = "hand-written-group-name"
        with self.assertRaisesRegex(SchemaError, "not content-addressed"):
            validate_comparison_groups(bad)

    def test_evidence_and_fidelity_are_part_of_comparison_identity(self) -> None:
        original = self.rows["metrics"][0]
        changed_evidence = dict(original, evidence_grade="D1")
        changed_fidelity = dict(original, fidelity="paper-exact")
        self.assertNotEqual(
            derive_comparison_group_id(original),
            derive_comparison_group_id(changed_evidence),
        )
        self.assertNotEqual(
            derive_comparison_group_id(original),
            derive_comparison_group_id(changed_fidelity),
        )

    def test_equal_unit_aggregate_requires_one_exact_component_per_unit(self) -> None:
        first = dict(
            self.rows["metrics"][0],
            cell_id="cell-a",
            cohort_id="cohort-a",
            value=0.6,
            component_ids=["cell-a"],
        )
        second = dict(
            self.rows["metrics"][0],
            cell_id="cell-b",
            cohort_id="cohort-b",
            value=0.8,
            component_ids=["cell-b"],
        )
        aggregate_definition = {
            "aggregation_id": "macro-two-cells",
            "rule": "equal_unit_mean",
            "unit_field": "cell_id",
            "component_ids": ["cell-a", "cell-b"],
        }
        aggregate = dict(
            first,
            aggregation_id="macro-two-cells",
            aggregation_level="dataset",
            cell_id="aggregate-cell-placeholder",
            value=0.7,
            component_ids=["cell-a", "cell-b"],
            cohort_id=derive_aggregate_cohort_id("cell_id", [first, second]),
        )
        validate_equal_unit_aggregates(
            [first, second, aggregate],
            [aggregate_definition],
        )
        duplicate = dict(first, aggregation_id="duplicate-component")
        with self.assertRaisesRegex(SchemaError, "exactly one compatible row per component"):
            validate_equal_unit_aggregates(
                [first, duplicate, second, aggregate],
                [aggregate_definition],
            )

    def test_missing_status_is_not_silently_numeric(self) -> None:
        bad = dict(self.rows["metrics"][0])
        bad["status"] = "BLOCKED_ASSET"
        with self.assertRaisesRegex(SchemaError, "cannot carry a metric value"):
            validate_metric_record(bad)

    def test_undefined_single_class_stays_visible(self) -> None:
        row = dict(self.rows["metrics"][0])
        row.update(
            status="METRIC_UNDEFINED_SINGLE_CLASS",
            status_detail="Only the negative class is present.",
            value=None,
            ci_low=None,
            ci_high=None,
            n_positive=0,
            n_negative=4,
        )
        validated = validate_metric_record(row)
        self.assertEqual(validated["status"], "METRIC_UNDEFINED_SINGLE_CLASS")

    def test_coverage_requires_every_system_slice_combination(self) -> None:
        expected = expected_coverage_rows(self.registry)
        validate_expected_coverage(expected, self.rows["coverage"])
        with self.assertRaisesRegex(SchemaError, "missing"):
            validate_expected_coverage(expected, self.rows["coverage"][:-1])

    def test_point_leader_and_uncertainty_set_are_separate(self) -> None:
        ranked = rank_metric_group(self.rows["metrics"])
        self.assertEqual([row["point_leader"] for row in ranked], [True, False])
        self.assertEqual([row["uncertainty_tie"] for row in ranked], [True, True])

    def test_exact_numeric_ties_share_first_rank(self) -> None:
        template = self.rows["metrics"][0]
        tied_rows = []
        for system_id, value, interval in (
            ("system-a", 0.64, (0.60, 0.68)),
            ("system-b", 0.64, (0.62, 0.66)),
            ("system-c", 0.61, (0.59, 0.63)),
        ):
            row = dict(template)
            row.update(
                system_id=system_id,
                value=value,
                ci_low=interval[0],
                ci_high=interval[1],
            )
            tied_rows.append(row)
        ranked = rank_metric_group(tied_rows)
        self.assertEqual([row["point_rank"] for row in ranked], [1, 1, 3])
        self.assertEqual([row["point_leader"] for row in ranked], [True, True, False])


class IOAndReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = fixture_registry()
        self.rows = fixture_rows(self.registry)

    def test_tidy_csv_is_deterministic_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.csv"
            second = Path(temporary) / "second.csv"
            staged = Path(temporary) / "staged.csv"
            write_tidy_csv(first, "metrics", reversed(self.rows["metrics"]))
            write_tidy_csv(second, "metrics", self.rows["metrics"])
            write_tidy_csv(staged, "metrics", self.rows["metrics"], atomic=False)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first.read_bytes(), staged.read_bytes())
            self.assertEqual(read_tidy_csv(first, "metrics"), self.rows["metrics"])
            with self.assertRaises(FileExistsError):
                write_tidy_csv(staged, "metrics", self.rows["metrics"], atomic=False)

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow") is not None,
        "PyArrow is required for the staged hash test",
    )
    def test_staged_csv_and_parquet_hash_without_reopening_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "metrics.csv"
            parquet_path = Path(temporary) / "metrics.parquet"
            with mock.patch(
                "spectral_utils.reconstruction_reporting.io.sha256_file",
                side_effect=AssertionError("output was reopened for hashing"),
            ):
                csv_record = write_tidy_csv(
                    csv_path,
                    "metrics",
                    self.rows["metrics"],
                    atomic=False,
                )
                parquet_record = write_parquet(
                    parquet_path,
                    "metrics",
                    self.rows["metrics"],
                    atomic=False,
                )
            self.assertEqual(
                csv_record["file_sha256"],
                hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                parquet_record["file_sha256"],
                hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
            )

    def test_report_starts_with_method_guide_and_is_self_contained(self) -> None:
        manifest = default_plot_manifest(self.registry["release_id"], self.rows)
        rendered = render_report(
            registry=self.registry,
            rows_by_table=self.rows,
            plot_manifest=manifest,
            title="Synthetic <unsafe> report",
        )
        self.assertLess(rendered.index('id="method-guide"'), rendered.index('id="results"'))
        self.assertIn("Input → operation → output", rendered)
        self.assertIn("Graph assumption checks", rendered)
        self.assertIn("Color/marker = method card", rendered)
        self.assertIn("registered 95% interval", rendered)
        self.assertIn('aria-label="Complete method legend"', rendered)
        self.assertIn('id="status-guide"', rendered)
        self.assertIn('id="diagnostic-plot"', rendered)
        self.assertIn('id="diagnostic-unit"', rendered)
        self.assertIn('id="diagnostic-direction"', rendered)
        self.assertIn('id="diagnostic-null"', rendered)
        self.assertIn('id="diagnostic-stage"', rendered)
        self.assertIn('id="view-preset"', rendered)
        self.assertIn('value="headline_24cell_auroc"', rendered)
        self.assertIn('id="filter-generation_model_id"', rendered)
        self.assertIn('id="filter-scorer_model_id"', rendered)
        self.assertIn('id="heatmap-colorbar"', rendered)
        self.assertIn('id="heatmap-scale-values"', rendered)
        for prefix in ("forest", "heatmap", "contrast", "diagnostic"):
            self.assertIn(f'id="{prefix}-source-link"', rendered)
            self.assertIn(f'id="{prefix}-source-hash"', rendered)
        self.assertIn("node-permutation median is a descriptive reference", rendered)
        self.assertIn("weight-sensitivity checks, not graph-refit stability", rendered)
        self.assertIn("arbitrary two-dimensional spectral coordinates", rendered)
        self.assertIn("Method-specific mechanism panels", rendered)
        for status in ("BLOCKED_ASSET", "METRIC_UNDEFINED_SINGLE_CLASS", "CONTEXT_ONLY"):
            self.assertIn(status, rendered)
        self.assertNotIn("<script src=", rendered)
        self.assertNotIn("<link rel=", rendered)
        self.assertNotIn('id="graph-examples-data"', rendered)
        self.assertIn("Synthetic &lt;unsafe&gt; report", rendered)
        embedded_payloads = re.findall(
            r'<script type="application/json" id="[^"]+">(.*?)</script>',
            rendered,
            flags=re.DOTALL,
        )
        self.assertTrue(embedded_payloads)
        for payload in embedded_payloads:
            json.loads(
                payload,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number: {value}")
                ),
            )

    def test_dense_graph_display_uses_deterministic_label_free_edge_sample(self) -> None:
        edges = [
            {
                "edge_source_index": index % 131,
                "edge_target_index": (index * 17 + 3) % 131,
                "edge_weight": 1.0 + index / 10000.0,
            }
            for index in range(GRAPH_DISPLAY_EDGE_LIMIT + 137)
        ]
        selected_a = _display_edges(edges, graph_hash=_hash("dense-graph"))
        selected_b = _display_edges(list(reversed(edges)), graph_hash=_hash("dense-graph"))
        self.assertEqual(len(selected_a), GRAPH_DISPLAY_EDGE_LIMIT)
        self.assertEqual(selected_a, selected_b)
        self.assertTrue(all("y_error" not in row for row in selected_a))

    def test_embedded_diagnostics_keep_only_browser_fields(self) -> None:
        source = dict(self.rows["graph_diagnostics"][0])
        source["notes"] = "large signed provenance payload"
        projected = _embedded_diagnostics([source])
        self.assertEqual(len(projected), 1)
        self.assertNotIn("notes", projected[0])
        self.assertEqual(
            set(projected[0]),
            {
                "task_id",
                "dataset_id",
                "cell_id",
                "slice_id",
                "comparison_group_id",
                "method_id",
                "system_id",
                "graph_variant",
                "diagnostic_label",
                "diagnostic_unit",
                "value",
                "null_value",
                "effect",
                "p_value",
                "permutation_count",
                "label_stage",
                "status",
            },
        )
        self.assertEqual(
            projected[0]["diagnostic_label"], source["diagnostic_label"]
        )

    def test_generic_diagnostic_selector_excludes_dense_dedicated_sweeps(self) -> None:
        source = dict(self.rows["graph_diagnostics"][0])
        curated = []
        for method_index in range(2):
            row = dict(source)
            row["method_id"] = source["method_id"]
            row["system_id"] = source["system_id"]
            row["diagnostic_label"] = (
                "Correctness and trace-length smoothness — error label roughness"
            )
            row["graph_variant"] = (
                f"panel=target_vs_nuisance_roughness;series=observed;x={method_index}"
            )
            curated.append(row)
        dense = []
        for index in range(4_000):
            row = dict(source)
            row["diagnostic_label"] = "Random feature-family graph control"
            row["graph_variant"] = (
                f"panel=random_family_graph_control;series=family;x={index}"
            )
            dense.append(row)
        projected = _embedded_diagnostics([*dense, *curated])
        keys = {
            (row["diagnostic_label"], row["graph_variant"])
            for row in projected
        }
        self.assertEqual(len(projected), 2)
        self.assertLessEqual(len(keys), 64)
        self.assertEqual(
            projected[0]["comparison_group_id"],
            source["comparison_group_id"],
        )

    def test_each_bounded_diagnostic_key_has_its_own_exact_plot_contract(self) -> None:
        manifest = default_plot_manifest(self.registry["release_id"], self.rows)
        projected = _embedded_diagnostics(self.rows["graph_diagnostics"])
        keys = {_diagnostic_selector_key(row) for row in projected}
        selector_plots = [
            plot for plot in manifest["plots"]
            if plot["plot_id"].startswith("diagnostic_selector_")
        ]
        self.assertEqual(len(selector_plots), len(keys))
        for plot in selector_plots:
            matching = [
                row
                for row in self.rows["graph_diagnostics"]
                if all(
                    row[field] in expected
                    if isinstance(expected, list)
                    else row[field] == expected
                    for field, expected in plot["filters"].items()
                )
            ]
            self.assertEqual(len(matching), plot["n_source_rows"])
            self.assertEqual(
                len({_diagnostic_selector_key(row) for row in matching}), 1
            )

    def test_alignment_scatter_prints_signed_descriptive_associations(self) -> None:
        method_id = "graph-fusion"
        point_notes = {
            "source_panel_id": "alignment_vs_improvement",
            "source_metric_id": "published_cell_auroc_delta_vs_iu_pcr",
        }
        rows = []
        for cell_id, alignment, delta in (("cell-a", 0.2, 0.01), ("cell-b", 0.4, -0.02)):
            rows.append(
                {
                    "method_id": method_id,
                    "cell_id": cell_id,
                    "value": delta,
                    "notes": json.dumps({**point_notes, "x_value": alignment}),
                    "status": "OK",
                    "graph_variant": "panel=alignment_vs_improvement;series=cell_relation;x=0",
                    "diagnostic_label": "Cell alignment versus AUROC change",
                }
            )
        for label, value, x_index in (
            ("Across-cell association — spearman", 0.25, 0),
            ("Across-cell association — pearson", -0.5, 1),
        ):
            rows.append(
                {
                    "method_id": method_id,
                    "cell_id": "aggregate::all",
                    "value": value,
                    "notes": "{}",
                    "status": "OK",
                    "graph_variant": f"panel=alignment_vs_improvement_summary;series=descriptive_relation;x={x_index}",
                    "diagnostic_label": label,
                }
            )
        rendered = _alignment_scatter(
            rows,
            self.registry,
            {
                "plots": [
                    {
                        "plot_id": "alignment-fixture",
                        "kind": "diagnostic_scatter",
                        "filters": {},
                    }
                ]
            },
        )
        self.assertIn("Spearman ρ=0.250", rendered)
        self.assertIn("Pearson r=-0.500", rendered)
        self.assertIn("no reporting-layer confidence interval or p-value", rendered)

    def test_method_specific_mechanism_panels_are_visible_and_claim_bounded(self) -> None:
        def row(
            method_id: str,
            panel: str,
            label: str,
            value: float,
            *,
            series: str = "observed",
            x_index: int = 0,
            source_note: str | None = None,
        ) -> dict:
            return {
                "method_id": method_id,
                "cell_id": "fixture-cell",
                "value": value,
                "status": "OK",
                "graph_variant": f"panel={panel};series={series};x={x_index}",
                "diagnostic_label": label,
                "notes": json.dumps(
                    {
                        "source_note": source_note,
                        "x_index": x_index,
                    }
                ),
            }

        diagnostics = [
            row("continuous_lsml", "continuous_lsml_cluster_boundaries", "Continuous L-SML cluster boundaries — cluster id", 0, series="a"),
            row("continuous_lsml", "continuous_lsml_cluster_boundaries", "Continuous L-SML cluster boundaries — cluster id", 0, series="b", x_index=1),
            row("continuous_lsml", "continuous_lsml_cluster_boundaries", "Continuous L-SML cluster boundaries — cluster id", 1, series="c", x_index=2),
            row("continuous_lsml", "continuous_lsml_correlation_clusters", "Continuous L-SML feature-correlation clusters — feature correlation", 0.8, series="a", source_note="column_feature=b;same_cluster=true"),
            row("continuous_lsml", "continuous_lsml_correlation_clusters", "Continuous L-SML feature-correlation clusters — feature correlation", 0.2, series="a", x_index=2, source_note="column_feature=c;same_cluster=false"),
            row("family_nrm_a", "family_nrm_residual_covariance", "Family-NRM residual covariance — residual covariance", 1.0, series="family_a", source_note="column_family=family_a"),
            row("family_nrm_a", "family_nrm_residual_covariance", "Family-NRM residual covariance — residual covariance", -0.4, series="family_a", x_index=1, source_note="column_family=family_b"),
            row("family_nrm_a", "family_nrm_residual_covariance", "Family-NRM residual covariance — residual covariance", -0.4, series="family_b", source_note="column_family=family_a"),
            row("family_nrm_a", "family_nrm_residual_covariance", "Family-NRM residual covariance — residual covariance", 0.8, series="family_b", x_index=1, source_note="column_family=family_b"),
            row("family_nrm_a", "family_nrm_residual_eigenspectrum", "Family-NRM residual eigenspectrum — residual eigenvalue", 1.5, series="eigenspectrum"),
            row("family_nrm_a", "family_nrm_residual_eigenspectrum", "Family-NRM residual eigenspectrum — residual eigenvalue", 0.5, series="eigenspectrum", x_index=1),
            row("su_pcr", "su_pcr_decomposition", "SU-PCR low-rank plus sparse decomposition — sparse support fraction off diagonal", 0.05),
            row("su_pcr", "su_pcr_low_rank_eigenspectrum", "SU-PCR low-rank eigenspectrum — low rank eigenvalue", -0.2, series="eigenspectrum"),
            row("su_pcr", "su_pcr_low_rank_eigenspectrum", "SU-PCR low-rank eigenspectrum — low rank eigenvalue", 1.2, series="eigenspectrum", x_index=1),
            row("pgrd_a", "pgrd_cross_gradient", "PGRD residual-gradient decomposition — cross term at registered direction", -0.4),
            row("pgrd_a", "pgrd_cross_gradient", "PGRD residual-gradient decomposition — quadratic term at registered direction", 0.1, x_index=1),
            row("pgrd_a", "pgrd_cross_gradient", "PGRD residual-gradient decomposition — predicted energy change at unit step", -0.3, x_index=2),
        ]
        plot_manifest = {
            "plots": [
                {"plot_id": plot_id, "kind": "diagnostic_summary", "filters": {}}
                for plot_id in (
                    "continuous_lsml_correlation_clusters",
                    "family_nrm_residual_structure",
                    "su_pcr_support_and_eigenspectrum",
                    "pgrd_energy_decomposition",
                )
            ]
        }
        lsml = _continuous_lsml_panel(diagnostics, plot_manifest)
        family = _family_nrm_panel(diagnostics, plot_manifest)
        su_pcr = _su_pcr_panel(diagnostics, plot_manifest)
        pgrd = _pgrd_panel(diagnostics, plot_manifest)
        self.assertIn("Mean absolute Pearson correlation", lsml)
        self.assertIn("does not show that those blocks track correctness", lsml)
        self.assertIn("Median residual covariance", family)
        self.assertIn("does not identify its correctness orientation", family)
        self.assertIn("0 = empty support", su_pcr)
        self.assertIn("does not establish that it caused any performance change", su_pcr)
        self.assertIn("1/1 cells have negative predicted unit-step change", pgrd)
        self.assertIn("does not mean that the direction follows correctness", pgrd)

    def test_bounded_assumption_summaries_cover_every_promised_control(self) -> None:
        def row(
            panel: str,
            label: str,
            value: float,
            *,
            method_id: str = "graph-fusion",
            series: str = "observed",
            effect: float | None = None,
            cell_id: str = "fixture-cell",
        ) -> dict:
            return {
                "method_id": method_id,
                "cell_id": cell_id,
                "value": value,
                "effect": value if effect is None else effect,
                "status": "OK",
                "graph_variant": f"panel={panel};series={series};x=0",
                "diagnostic_label": label,
            }

        diagnostics = [
            row("random_family_graph_control", "Random feature-family graph control — error label roughness", 0.8, series="entropy_level"),
            row("ca_alpha_controls", "CA-SpecRaGE registered controls — error label roughness", 0.7, series="equal_view", effect=0.1),
            row("ca_view_weights", "CA-SpecRaGE view weights — mean learned alpha", 0.3, series="epr"),
            row("ca_view_weights", "CA-SpecRaGE view weights — frozen view prior", 0.2, series="epr"),
            row("dufs_gate_weights", "DUFS feature weights — rms normalized gate weight", 1.2, series="epr"),
            row("dufs_gate_weights_per_seed", "DUFS feature weights by seed — gate survival probability", 0.9, series="epr"),
            row("fixed_graph_group_bootstrap_stability", "Fixed fitted-graph weight sensitivity under source-group resampling — edge support jaccard", 0.6),
            row("family_nrm_family_contributions", "Family-NRM family contributions — absolute direction share", 0.4, method_id="family_nrm_a", series="entropy_level"),
            row("family_nrm_family_contributions", "Family-NRM family contributions — direction coefficient", -0.2, method_id="family_nrm_a", series="entropy_level"),
            row("su_pcr_sparse_support_stability", "SU-PCR support stability — support jaccard vs frozen", 0.75, method_id="su_pcr"),
        ]
        plot_ids = (
            "random_family_graph_control_summary",
            "ca_registered_control_summary",
            "ca_view_weight_summary",
            "dufs_gate_weight_summary",
            "fixed_graph_weight_sensitivity_summary",
            "family_nrm_family_contribution_summary",
            "su_pcr_support_stability_summary",
        )
        manifest = {
            "plots": [
                {
                    "plot_id": plot_id,
                    "kind": "diagnostic_summary",
                    "filters": {},
                    "data_sha256": _hash(plot_id),
                }
                for plot_id in plot_ids
            ]
        }
        rendered = _diagnostic_summary_panels(
            diagnostics,
            self.registry,
            manifest,
        )
        for figure_id in (
            "random-family-control-panel",
            "ca-control-panel",
            "ca-view-weight-panel",
            "dufs-gate-weight-panel",
            "fixed-graph-bootstrap-panel",
            "family-nrm-contribution-panel",
            "su-pcr-support-stability-panel",
        ):
            self.assertIn(f'id="{figure_id}"', rendered)
        for plot_id in plot_ids:
            self.assertIn(f'plot_data/{plot_id}.csv', rendered)
            self.assertIn(f'data_sha256={_hash(plot_id)}', rendered)
        self.assertIn("Dot = median", rendered)
        self.assertIn("Line = full", rendered)

    def test_all_registered_marker_shapes_have_explicit_svg_legends(self) -> None:
        for marker in (
            "circle",
            "square",
            "triangle",
            "diamond",
            "cross",
            "plus",
            "star",
            "hexagon",
        ):
            rendered = _marker_svg(marker, "#123456")
            self.assertIn(f'aria-label="{marker} marker"', rendered)

    def test_only_faceted_heatmaps_may_span_compatible_cell_groups(self) -> None:
        rows = json.loads(json.dumps(self.rows))
        second_cell = []
        for source in rows["metrics"]:
            clone = dict(source)
            clone.update(
                cell_id="processbench-math-llama-fixture",
                population_id="processbench-math-fixture-population",
                slice_id="processbench-math-all",
                cohort_id="cohort::" + _hash("second-cell-cohort"),
                aggregation_id="processbench-math-cell",
                component_ids=["processbench-math-llama-fixture"],
            )
            clone["comparison_group_id"] = derive_comparison_group_id(clone)
            second_cell.append(clone)
        rows["metrics"].extend(second_cell)
        manifest = default_plot_manifest(self.registry["release_id"], rows)
        _, selected = validate_plot_data_sources(manifest, rows)
        for plot in manifest["plots"]:
            groups = {
                row["comparison_group_id"]
                for row in selected[plot["plot_id"]]
            }
            if plot["kind"] == "faceted_heatmap":
                self.assertEqual(len(groups), 2)
            else:
                self.assertLessEqual(len(groups), 1)
        with self.assertRaisesRegex(SchemaError, "mixes comparison groups"):
            make_plot_spec(
                plot_id="unsafe-mixed-plot",
                title="Unsafe mixed plot",
                kind="forest",
                source_table="metrics",
                rows=rows["metrics"],
                filters={},
                encodings={"x": "value", "y": "system_id"},
                legend=["Synthetic legend."],
                caption="Synthetic caption.",
                better_direction="higher",
                ci_definition="Synthetic interval.",
                selection_rule="All rows.",
            )

        faceted = make_plot_spec(
            plot_id="safe-faceted-cell-heatmap",
            title="Safe faceted cell heatmap",
            kind="faceted_heatmap",
            source_table="metrics",
            rows=rows["metrics"],
            filters={"aggregation_level": "cell", "metric_id": "macro_f1"},
            encodings={
                "x": "cell_id",
                "y": "system_id",
                "fill": "value",
                "facet_group": "comparison_group_id",
            },
            legend=["Each cell retains its exact comparison group."],
            caption="Two compatible cell cohorts are shown without pooling them.",
            better_direction="higher",
            ci_definition="Intervals stay in the exact result rows.",
            selection_rule="All compatible fixture cells.",
        )
        self.assertEqual(faceted["n_source_rows"], 4)

    def test_report_rechecks_plot_hashes_and_fails_closed_on_unknown_renderer(self) -> None:
        manifest = default_plot_manifest(self.registry["release_id"], self.rows)
        changed = json.loads(json.dumps(self.rows))
        for metric in changed["metrics"]:
            metric["value"] += 0.01
            metric["ci_low"] += 0.01
            metric["ci_high"] += 0.01
        with self.assertRaisesRegex(SchemaError, "source rows do not match"):
            render_report(
                registry=self.registry,
                rows_by_table=changed,
                plot_manifest=manifest,
            )

        line_plot = make_plot_spec(
            plot_id="unsupported-line",
            title="Unsupported line",
            kind="line",
            source_table="metrics",
            rows=self.rows["metrics"],
            filters={
                "comparison_group_id": self.rows["metrics"][0]["comparison_group_id"]
            },
            encodings={"x": "cell_id", "y": "value"},
            legend=["Synthetic line legend."],
            caption="Synthetic line caption.",
            better_direction="higher",
            ci_definition="Synthetic interval.",
            selection_rule="One exact group.",
        )
        unsupported = validate_plot_manifest(
            {
                "schema": "reconstruction_plot_manifest_v1",
                "release_id": self.registry["release_id"],
                "plots": [line_plot],
            }
        )
        with self.assertRaisesRegex(SchemaError, "no renderer"):
            render_report(
                registry=self.registry,
                rows_by_table=self.rows,
                plot_manifest=unsupported,
            )

    def test_manifest_and_graph_rows_have_total_canonical_order(self) -> None:
        manifest = default_plot_manifest(self.registry["release_id"], self.rows)
        reversed_manifest = {
            "schema": manifest["schema"],
            "release_id": manifest["release_id"],
            "plots": list(reversed(manifest["plots"])),
        }
        self.assertEqual(validate_plot_manifest(reversed_manifest), manifest)

        first = dict(self.rows["graph_diagnostics"][0], notes="alpha")
        second = dict(self.rows["graph_diagnostics"][0], notes="omega")
        ordered_a = sorted([first, second], key=lambda row: record_sort_key("graph_diagnostics", row))
        ordered_b = sorted([second, first], key=lambda row: record_sort_key("graph_diagnostics", row))
        self.assertEqual(ordered_a, ordered_b)

    def test_every_plot_materializes_exact_data_csv(self) -> None:
        manifest = default_plot_manifest(self.registry["release_id"], self.rows)
        rendered = render_report(
            registry=self.registry,
            rows_by_table=self.rows,
            plot_manifest=manifest,
        )
        with tempfile.TemporaryDirectory() as temporary:
            layout = ReleaseLayout.from_root(Path(temporary) / "release")
            layout.create_directories()
            outputs = materialize_plot_data(layout, manifest, self.rows)
            self.assertEqual(len(outputs), len(manifest["plots"]))
            for plot in manifest["plots"]:
                self.assertTrue(plot["legend"])
                self.assertTrue(plot["caption"])
                self.assertTrue((layout.plot_data / f"{plot['plot_id']}.csv").exists())
                self.assertIn(
                    f'href="plot_data/{plot["plot_id"]}.csv"', rendered
                )
                self.assertIn(f'data_sha256={plot["data_sha256"]}', rendered)

    def test_plot_hash_mismatch_fails_before_materialization(self) -> None:
        manifest = default_plot_manifest(self.registry["release_id"], self.rows)
        bad_rows = {table: list(rows) for table, rows in self.rows.items()}
        bad_rows["metrics"] = [dict(row) for row in bad_rows["metrics"]]
        bad_rows["metrics"][0]["value"] = 0.62
        with self.assertRaisesRegex(SchemaError, "source rows do not match"):
            validate_plot_data_sources(manifest, bad_rows)

    def test_embedded_script_terminator_is_escaped(self) -> None:
        registry = json.loads(json.dumps(self.registry))
        registry["methods"][0]["plain_summary"] = "safe </script><script>alert(1)</script> text"
        registry.pop("registry_sha256")
        registry = validate_registry(registry)
        rendered = render_report(registry=registry, rows_by_table=self.rows)
        self.assertNotIn("</script><script>alert(1)</script>", rendered)
        self.assertIn("&lt;/script&gt;&lt;script&gt;alert(1)&lt;/script&gt;", rendered)


class QueryLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = fixture_registry()
        self.rows = fixture_rows(self.registry)

    def test_all_required_views_have_a_definition(self) -> None:
        sql = "\n".join(query_view_sql())
        for view in VIEW_NAMES:
            self.assertIn(f"CREATE VIEW {view}", sql)

    def test_missing_duckdb_fails_with_install_instruction(self) -> None:
        if importlib.util.find_spec("duckdb") is not None:
            self.skipTest("DuckDB is installed; integration test covers the build path")
        with self.assertRaisesRegex(MissingOptionalDependency, "requirements-reporting"):
            build_duckdb("/path/that/is/not/opened-before-import")

    @unittest.skipUnless(importlib.util.find_spec("duckdb") is not None, "DuckDB not installed")
    def test_query_results_has_a_total_order_across_metric_ties_and_graph_rows(self) -> None:
        import duckdb

        leaderboard_rows = [
            ("group-b", "metric-a", 1, "system-a", "group-b"),
            ("group-a", "metric-b", 1, "system-a", "metric-b"),
            ("group-a", "metric-a", 2, "system-b", "rank-2"),
            ("group-a", "metric-a", 1, "system-z", "system-z"),
            ("group-a", "metric-a", 1, "system-a", "system-a"),
        ]
        graph_rows = [
            ("group-b", "diagnostic-a", "graph-a", "system-a", "group-b"),
            ("group-a", "diagnostic-b", "graph-a", "system-a", "diagnostic-b"),
            ("group-a", "diagnostic-a", "graph-b", "system-a", "graph-b"),
            ("group-a", "diagnostic-a", "graph-a", "system-z", "system-z"),
            ("group-a", "diagnostic-a", "graph-a", "system-a", "system-a"),
        ]

        with tempfile.TemporaryDirectory() as temporary:
            outputs = []
            graph_outputs = []
            for name, physical_order in (
                ("forward", leaderboard_rows),
                ("reverse", list(reversed(leaderboard_rows))),
            ):
                database = Path(temporary) / f"{name}.duckdb"
                connection = duckdb.connect(str(database))
                connection.execute(
                    """
                    CREATE TABLE v_atomic_leaderboard (
                        task_id VARCHAR,
                        dataset_id VARCHAR,
                        cell_id VARCHAR,
                        slice_id VARCHAR,
                        comparison_group_id VARCHAR,
                        metric_id VARCHAR,
                        point_rank BIGINT,
                        system_id VARCHAR,
                        payload VARCHAR
                    )
                    """
                )
                connection.executemany(
                    "INSERT INTO v_atomic_leaderboard VALUES "
                    "('task', 'dataset', 'cell', 'slice', ?, ?, ?, ?, ?)",
                    physical_order,
                )
                connection.execute(
                    """
                    CREATE TABLE v_graph_assumption_checks (
                        task_id VARCHAR,
                        dataset_id VARCHAR,
                        cell_id VARCHAR,
                        slice_id VARCHAR,
                        comparison_group_id VARCHAR,
                        diagnostic_id VARCHAR,
                        graph_id VARCHAR,
                        graph_variant VARCHAR,
                        label_stage VARCHAR,
                        graph_hash VARCHAR,
                        matrix_hash VARCHAR,
                        system_id VARCHAR,
                        payload VARCHAR
                    )
                    """
                )
                ordered_graph_rows = (
                    graph_rows if name == "forward" else list(reversed(graph_rows))
                )
                connection.executemany(
                    "INSERT INTO v_graph_assumption_checks VALUES "
                    "('task', 'dataset', 'cell', 'slice', ?, ?, ?, 'real', "
                    "'post_freeze_labels', 'graph-hash', 'matrix-hash', ?, ?)",
                    ordered_graph_rows,
                )
                connection.close()

                columns, rows = query_results(database, view="v_atomic_leaderboard")
                outputs.append([row[columns.index("payload")] for row in rows])
                graph_columns, queried_graph_rows = query_results(
                    database,
                    view="v_graph_assumption_checks",
                )
                graph_outputs.append(
                    [
                        row[graph_columns.index("payload")]
                        for row in queried_graph_rows
                    ]
                )

            expected = ["system-a", "system-z", "rank-2", "metric-b", "group-b"]
            self.assertEqual(outputs, [expected, expected])
            expected_graph = [
                "system-a",
                "system-z",
                "graph-b",
                "diagnostic-b",
                "group-b",
            ]
            self.assertEqual(graph_outputs, [expected_graph, expected_graph])

    @unittest.skipUnless(importlib.util.find_spec("duckdb") is not None, "DuckDB not installed")
    def test_duckdb_drilldown_returns_all_compatible_systems(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = ReleaseLayout.from_root(Path(temporary) / "release")
            layout.create_directories()
            write_canonical_json(layout.registry_json, self.registry)
            write_tidy_csv(layout.metrics_csv, "metrics", self.rows["metrics"])
            write_tidy_csv(layout.contrasts_csv, "contrasts", self.rows["contrasts"])
            write_tidy_csv(layout.coverage_csv, "coverage", self.rows["coverage"])
            write_tidy_csv(
                layout.graph_diagnostics_csv,
                "graph_diagnostics",
                self.rows["graph_diagnostics"],
            )
            write_tidy_csv(layout.graph_examples_csv, "graph_examples", self.rows["graph_examples"])
            build_duckdb(layout.root)
            columns, values = query_results(
                layout.database,
                view="v_processbench_localization",
                filters={
                    "task_id": "localization",
                    "dataset_id": "processbench",
                    "cell_id": "processbench-gsm8k-llama-fixture",
                    "fidelity": "adapted-common-protocol",
                },
            )
            systems = {row[columns.index("system_id")] for row in values}
            self.assertEqual(systems, {item["system_id"] for item in self.registry["systems"]})
            example_columns, example_rows = query_results(
                layout.database,
                view="v_graph_examples",
                filters={"task_id": "localization", "dataset_id": "processbench"},
            )
            self.assertIn("row_kind", example_columns)
            self.assertEqual(len(example_rows), len(self.rows["graph_examples"]))

    @unittest.skipUnless(importlib.util.find_spec("duckdb") is not None, "DuckDB not installed")
    def test_leaderboard_exports_are_deterministic_and_rankable_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            roots = []
            records_by_root = []
            for independent_run in ("a", "b"):
                layout = ReleaseLayout.from_root(Path(temporary) / independent_run)
                layout.create_directories()
                write_canonical_json(layout.registry_json, self.registry)
                write_tidy_csv(layout.metrics_csv, "metrics", self.rows["metrics"])
                write_tidy_csv(layout.contrasts_csv, "contrasts", self.rows["contrasts"])
                write_tidy_csv(layout.coverage_csv, "coverage", self.rows["coverage"])
                write_tidy_csv(
                    layout.graph_diagnostics_csv,
                    "graph_diagnostics",
                    self.rows["graph_diagnostics"],
                )
                write_tidy_csv(
                    layout.graph_examples_csv,
                    "graph_examples",
                    self.rows["graph_examples"],
                )
                build_duckdb(layout.root)
                records_by_root.append(export_leaderboard_csvs(layout.root))
                roots.append(layout.root)

            expected_levels = [spec[0] for spec in LEADERBOARD_EXPORT_SPECS]
            self.assertEqual(
                [record["leaderboard_level"] for record in records_by_root[0]],
                expected_levels,
            )
            for level, view, filename in LEADERBOARD_EXPORT_SPECS:
                relative = Path("05_evaluation") / "leaderboards" / filename
                first = roots[0] / relative
                second = roots[1] / relative
                self.assertEqual(first.read_bytes(), second.read_bytes())
                columns, query_rows = query_results(
                    roots[0] / "05_evaluation" / "benchmark.duckdb",
                    view=view,
                )
                with first.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    self.assertEqual(reader.fieldnames, columns)
                    exported = list(reader)
                self.assertEqual(len(exported), len(query_rows))
                expected_source_level = "task" if level == "slice" else level
                self.assertTrue(
                    all(
                        row["aggregation_level"] == expected_source_level
                        for row in exported
                    )
                )
                self.assertTrue(all(row["status"] in {"OK", "OK_FALLBACK"} for row in exported))
                self.assertTrue(all(row["value"] != "" for row in exported))
                self.assertTrue(all(row["aggregation_id"] != "" for row in exported))
                if level == "task":
                    self.assertTrue(
                        all(row["slice_dimension"] == "macro24" for row in exported)
                    )
                    self.assertTrue(
                        all(row["slice_value"] == "all_24_cells" for row in exported)
                    )
                if level == "slice":
                    self.assertTrue(
                        all(
                            row["slice_dimension"] in {"domain", "model_family"}
                            for row in exported
                        )
                    )

    @unittest.skipUnless(importlib.util.find_spec("duckdb") is not None, "DuckDB not installed")
    def test_task_leaderboard_selects_only_registered_macro24_aggregation(self) -> None:
        registry = json.loads(json.dumps(self.registry))
        rows = json.loads(json.dumps(self.rows))
        slice_specs = (
            (
                "task-macro24-all",
                "macro24",
                "all_24_cells",
                "frozen24::equal-cell::macro24-all-24-cells",
                "task",
            ),
            (
                "task-domain-math",
                "domain",
                "math",
                "frozen24::equal-cell::domain-math",
                "task",
            ),
            (
                "task-model-family-fixture",
                "model_family",
                "fixture-model",
                "frozen24::equal-cell::model-family-fixture-model",
                "task",
            ),
            (
                "task-macro24-partial",
                "macro24",
                "not_all_24_cells",
                "frozen24::equal-cell::macro24-not-all-24-cells",
                "task",
            ),
            (
                "release-all-tasks",
                "across_task",
                "all_tasks",
                "release::all-tasks",
                "release",
            ),
        )
        for slice_id, dimension, value, aggregation_id, _level in slice_specs:
            registry["slices"].append(
                {
                    "slice_id": slice_id,
                    "population_id": "processbench-fixture-population",
                    "cell_id": "processbench-gsm8k-llama-fixture",
                    "slice_dimension": dimension,
                    "slice_value": value,
                    "display_name": f"Synthetic {dimension}={value} slice",
                    "expected_n": 4,
                }
            )
            registry["aggregations"].append(
                {
                    "aggregation_id": aggregation_id,
                    "display_name": f"Synthetic aggregate for {slice_id}",
                    "rule": "equal_unit_mean",
                    "unit_field": "cell_id",
                    "component_ids": ["processbench-gsm8k-llama-fixture"],
                    "bootstrap_unit": "source question",
                    "weighting": "synthetic equal-cell fixture",
                }
            )
            for source in self.rows["metrics"]:
                metric = dict(source)
                metric.update(
                    slice_id=slice_id,
                    aggregation_id=aggregation_id,
                    aggregation_level=_level,
                )
                metric["comparison_group_id"] = derive_comparison_group_id(metric)
                rows["metrics"].append(metric)

        registry.pop("registry_sha256")
        registry = validate_registry(registry)
        with tempfile.TemporaryDirectory() as temporary:
            layout = ReleaseLayout.from_root(Path(temporary) / "release")
            layout.create_directories()
            write_canonical_json(layout.registry_json, registry)
            write_tidy_csv(layout.metrics_csv, "metrics", rows["metrics"])
            write_tidy_csv(layout.contrasts_csv, "contrasts", rows["contrasts"])
            write_tidy_csv(layout.coverage_csv, "coverage", rows["coverage"])
            write_tidy_csv(
                layout.graph_diagnostics_csv,
                "graph_diagnostics",
                rows["graph_diagnostics"],
            )
            write_tidy_csv(layout.graph_examples_csv, "graph_examples", rows["graph_examples"])
            build_duckdb(layout.root)

            task_columns, task_rows = query_results(
                layout.database,
                view="v_task_leaderboard",
            )
            self.assertEqual(len(task_rows), len(self.registry["systems"]))
            task_records = [dict(zip(task_columns, row)) for row in task_rows]
            self.assertEqual(
                {row["aggregation_id"] for row in task_records},
                {"frozen24::equal-cell::macro24-all-24-cells"},
            )
            self.assertEqual({row["aggregation_level"] for row in task_records}, {"task"})
            self.assertEqual({row["slice_dimension"] for row in task_records}, {"macro24"})
            self.assertEqual({row["slice_value"] for row in task_records}, {"all_24_cells"})

            slice_columns, slice_rows = query_results(
                layout.database,
                view="v_slice_leaderboard",
            )
            slice_records = [dict(zip(slice_columns, row)) for row in slice_rows]
            self.assertEqual(
                {row["aggregation_id"] for row in slice_records},
                {
                    "frozen24::equal-cell::domain-math",
                    "frozen24::equal-cell::model-family-fixture-model",
                },
            )
            self.assertEqual(
                {row["slice_dimension"] for row in slice_records},
                {"domain", "model_family"},
            )
            self.assertEqual(
                {row["aggregation_level"] for row in slice_records}, {"task"}
            )

            release_columns, release_rows = query_results(
                layout.database,
                view="v_release_leaderboard",
            )
            release_records = [dict(zip(release_columns, row)) for row in release_rows]
            self.assertEqual(
                {row["aggregation_id"] for row in release_records},
                {"release::all-tasks"},
            )
            self.assertEqual(
                {row["aggregation_level"] for row in release_records},
                {"release"},
            )

            artifacts = export_leaderboard_csvs(layout.root)
            task_artifact = next(
                artifact for artifact in artifacts if artifact["leaderboard_level"] == "task"
            )
            self.assertEqual(
                task_artifact["selection_semantics"],
                "aggregation_level=task; slice_dimension=macro24; "
                "slice_value=all_24_cells",
            )
            task_csv = layout.root / task_artifact["relative_path"]
            with task_csv.open("r", encoding="utf-8", newline="") as handle:
                exported_task = list(csv.DictReader(handle))
            self.assertEqual(
                {row["aggregation_id"] for row in exported_task},
                {"frozen24::equal-cell::macro24-all-24-cells"},
            )
            self.assertEqual({row["slice_dimension"] for row in exported_task}, {"macro24"})
            self.assertEqual({row["slice_value"] for row in exported_task}, {"all_24_cells"})

            slice_artifact = next(
                artifact
                for artifact in artifacts
                if artifact["leaderboard_level"] == "slice"
            )
            self.assertEqual(
                slice_artifact["selection_semantics"],
                "aggregation_level=task; slice_dimension in (domain, model_family)",
            )
            slice_csv = layout.root / slice_artifact["relative_path"]
            with slice_csv.open("r", encoding="utf-8", newline="") as handle:
                exported_slices = list(csv.DictReader(handle))
            self.assertEqual(
                {row["aggregation_id"] for row in exported_slices},
                {
                    "frozen24::equal-cell::domain-math",
                    "frozen24::equal-cell::model-family-fixture-model",
                },
            )

            release_artifact = next(
                artifact for artifact in artifacts if artifact["leaderboard_level"] == "release"
            )
            release_csv = layout.root / release_artifact["relative_path"]
            with release_csv.open("r", encoding="utf-8", newline="") as handle:
                exported_release = list(csv.DictReader(handle))
            self.assertEqual(
                {row["aggregation_id"] for row in exported_release},
                {"release::all-tasks"},
            )


class FullReleaseTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("duckdb") is not None
        and importlib.util.find_spec("pyarrow") is not None,
        "DuckDB and PyArrow are required for the complete release test",
    )
    def test_failed_staged_release_is_never_published(self) -> None:
        script = REPO_ROOT / "scripts" / "reconstruction_benchmark" / "build_reporting_release.py"
        spec = importlib.util.spec_from_file_location("failing_reporting_builder", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        registry = fixture_registry()
        rows = fixture_rows(registry)
        plots = default_plot_manifest(registry["release_id"], rows)
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "reporting"
            root = parent / registry["release_id"]
            with mock.patch.object(
                module,
                "write_report",
                side_effect=RuntimeError("synthetic publication failure"),
            ):
                bridge_manifest = {
                    "schema": "reconstruction-reporting-bridge-v3",
                    "release_id": registry["release_id"],
                    "scientific_publication_eligible": True,
                    "graph_diagnostics_status": "VERIFIED_SIGNED_SOURCE_CONVERTED",
                }
                bridge_manifest["payload_sha256"] = canonical_sha256(bridge_manifest)
                with self.assertRaisesRegex(RuntimeError, "synthetic publication failure"):
                    module.build_release(
                        root,
                        registry=registry,
                        rows=rows,
                        plot_manifest=plots,
                        title="Synthetic failed release",
                        bridge_manifest=bridge_manifest,
                    )
            self.assertFalse(root.exists())
            self.assertEqual(
                list(parent.glob(f".{registry['release_id']}.building-*")),
                [],
            )

    @unittest.skipUnless(
        importlib.util.find_spec("duckdb") is not None
        and importlib.util.find_spec("pyarrow") is not None,
        "DuckDB and PyArrow are required for the complete release test",
    )
    def test_full_staged_release_never_reopens_large_outputs_for_hashing(self) -> None:
        script = REPO_ROOT / "scripts" / "reconstruction_benchmark" / "build_reporting_release.py"
        spec = importlib.util.spec_from_file_location("single_descriptor_reporting_builder", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        registry = fixture_registry()
        rows = fixture_rows(registry)
        plots = default_plot_manifest(registry["release_id"], rows)
        bridge_manifest = {
            "schema": "reconstruction-reporting-bridge-v3",
            "release_id": registry["release_id"],
            "scientific_publication_eligible": True,
            "graph_diagnostics_status": "VERIFIED_SIGNED_SOURCE_CONVERTED",
        }
        bridge_manifest["payload_sha256"] = canonical_sha256(bridge_manifest)
        real_builder_hash = module.sha256_file

        def reject_large_output_reopen(path: object) -> str:
            suffix = Path(path).suffix.lower()
            if suffix in {".csv", ".parquet", ".html"}:
                raise AssertionError(f"large staged output was reopened for hashing: {path}")
            return real_builder_hash(path)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "reporting" / registry["release_id"]
            with mock.patch.object(
                module,
                "sha256_file",
                side_effect=reject_large_output_reopen,
            ), mock.patch(
                "spectral_utils.reconstruction_reporting.query.sha256_file",
                side_effect=AssertionError("DuckDB reopened a staged source only to hash it"),
            ):
                module.build_release(
                    root,
                    registry=registry,
                    rows=rows,
                    plot_manifest=plots,
                    title="Synthetic single-descriptor release",
                    bridge_manifest=bridge_manifest,
                )

            manifest = json.loads((root / "REPORTING_MANIFEST.json").read_text())
            database_artifact = next(
                artifact for artifact in manifest["artifacts"]
                if artifact.get("kind") == "duckdb"
            )
            expected_sources = {}
            for artifact in manifest["artifacts"]:
                kind = artifact.get("kind")
                if kind == "registry":
                    expected_sources["registry"] = artifact["file_sha256"]
                elif kind == "predictions":
                    expected_sources["predictions"] = artifact["file_sha256"]
                elif kind == "tidy_csv" and artifact.get("table") in {
                    "metrics",
                    "contrasts",
                    "coverage",
                    "graph_diagnostics",
                    "graph_examples",
                }:
                    expected_sources[artifact["table"]] = artifact["file_sha256"]
            self.assertEqual(database_artifact["source_sha256"], expected_sources)

    @unittest.skipUnless(
        importlib.util.find_spec("duckdb") is not None
        and importlib.util.find_spec("pyarrow") is not None,
        "DuckDB and PyArrow are required for the complete release test",
    )
    def test_two_independent_release_builds_are_canonically_identical(self) -> None:
        script = REPO_ROOT / "scripts" / "reconstruction_benchmark" / "build_reporting_release.py"
        spec = importlib.util.spec_from_file_location("synthetic_reporting_builder", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        registry = fixture_registry()
        rows = fixture_rows(registry)
        plots = default_plot_manifest(registry["release_id"], rows)
        bridge_manifest = {
            "schema": "reconstruction-reporting-bridge-v3",
            "release_id": registry["release_id"],
            "scientific_publication_eligible": True,
            "graph_diagnostics_status": "VERIFIED_SIGNED_SOURCE_CONVERTED",
        }
        bridge_manifest["payload_sha256"] = canonical_sha256(bridge_manifest)
        with tempfile.TemporaryDirectory() as temporary:
            roots = []
            for independent_run in ("a", "b"):
                root = Path(temporary) / independent_run / registry["release_id"]
                module.build_release(
                    root,
                    registry=registry,
                    rows=rows,
                    plot_manifest=plots,
                    title="Synthetic deterministic release",
                    bridge_manifest=bridge_manifest,
                )
                roots.append(root)

            relative_files = [
                sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())
                for root in roots
            ]
            self.assertEqual(relative_files[0], relative_files[1])
            for relative_path in relative_files[0]:
                # DuckDB's physical container contains engine-owned bytes that
                # are not canonical across independent builds.  Its source
                # recipe and logical query results are checked below; every
                # canonical release artifact remains byte-identical.
                if relative_path == Path("05_evaluation/benchmark.duckdb"):
                    continue
                self.assertEqual(
                    (roots[0] / relative_path).read_bytes(),
                    (roots[1] / relative_path).read_bytes(),
                    msg=f"non-deterministic release artifact: {relative_path}",
                )

            manifest = json.loads((roots[0] / "REPORTING_MANIFEST.json").read_text())
            self.assertNotIn(".building-", json.dumps(manifest, sort_keys=True))
            database_artifact = next(
                artifact for artifact in manifest["artifacts"]
                if artifact.get("kind") == "duckdb"
            )
            self.assertFalse(database_artifact["physical_bytes_canonical"])
            self.assertEqual(len(database_artifact["logical_sha256"]), 64)
            self.assertNotIn("file_sha256", database_artifact)
            self.assertNotIn("size_bytes", database_artifact)
            self.assertEqual(
                manifest["source_bridge"]["payload_sha256"],
                bridge_manifest["payload_sha256"],
            )
            self.assertTrue(manifest["source_bridge"]["scientific_publication_eligible"])
            self.assertTrue((roots[0] / "01_registries" / "BRIDGE_MANIFEST.json").exists())
            leaderboard_artifacts = {
                artifact["leaderboard_level"]: artifact
                for artifact in manifest["artifacts"]
                if artifact.get("kind") == "leaderboard_csv"
            }
            self.assertEqual(
                set(leaderboard_artifacts),
                {"cell", "dataset", "task", "slice", "release"},
            )
            for level, view, filename in LEADERBOARD_EXPORT_SPECS:
                artifact = leaderboard_artifacts[level]
                self.assertEqual(artifact["source_view"], view)
                self.assertEqual(
                    artifact["relative_path"],
                    f"05_evaluation/leaderboards/{filename}",
                )
                self.assertEqual(
                    artifact["source_database_logical_sha256"],
                    database_artifact["logical_sha256"],
                )

            logical_queries = [
                query_results(root / "05_evaluation" / "benchmark.duckdb")
                for root in roots
            ]
            self.assertEqual(logical_queries[0], logical_queries[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
