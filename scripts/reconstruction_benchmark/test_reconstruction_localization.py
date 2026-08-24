#!/usr/bin/env python3
"""Focused contract tests for reconstruction localization v1 (no data run)."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark import localization_fit as LF  # noqa: E402
from spectral_utils.reconstruction_benchmark import localization_evaluation as LE  # noqa: E402
from spectral_utils.reconstruction_benchmark.localization_ab import (  # noqa: E402
    _external_response_bindings,
)
from spectral_utils.reconstruction_benchmark.localization_comparators import _project_rows  # noqa: E402
from spectral_utils.reconstruction_benchmark.localization_contract import (  # noqa: E402
    FIT_ROW_IDENTITY_SCHEMA_VERSION,
    ID_CONTRACT_VERSION,
    ID_DIGEST_ALGORITHM,
    IDENTITY_KEY_BYTES,
    IDENTITY_KEY_CONTRACT_VERSION,
    OPAQUE_ROW_ID_PREFIX,
    PreparedLocalizationCell,
    assert_no_target_named_members,
    empirical_midrank,
    load_localization_registry,
    payload_sha256,
    primary_system_roster,
)
from spectral_utils.reconstruction_benchmark.localization_evaluation import (  # noqa: E402
    LOCALIZATION_DECISION_FIELDS,
    METRIC_FIELDS,
    PRMBENCH_ERROR_FAMILIES,
    PROCESSBENCH_SUBSETS,
    UNDEFINED_SINGLE_CLASS,
    assign_processbench_folds,
    bootstrap_prmbench_steps,
    crossfit_processbench_threshold,
    grouped_bootstrap_metric_map,
    prmbench_panel_metrics,
    processbench_prediction,
    processbench_trace_metrics,
)
from spectral_utils.reconstruction_benchmark.localization_postfreeze import (  # noqa: E402
    PBCell,
    PRMPanel,
    _evaluate_prmbench,
    _evaluate_processbench,
    _paired_contrast,
    _pb_bootstrap_statistic,
    _require_bootstrap_execution,
    _shared_prm_bootstrap_results,
    _shared_pb_bootstrap_results,
    _validate_evaluation_tables,
    verify_localization_evaluation_ab,
    write_localization_evaluation_build,
)
from spectral_utils.reconstruction_benchmark.methods import PRIMARY_METHOD_IDS  # noqa: E402


def _identity_contract() -> dict:
    value = {
        "schema_version": FIT_ROW_IDENTITY_SCHEMA_VERSION,
        "version": ID_CONTRACT_VERSION,
        "digest_algorithm": ID_DIGEST_ALGORITHM,
        "identity_key_contract_version": IDENTITY_KEY_CONTRACT_VERSION,
        "identity_key_bytes": IDENTITY_KEY_BYTES,
        "opaque_row_id_prefix": OPAQUE_ROW_ID_PREFIX,
        "row_namespace_scope": "cell",
        "canonical_row_order": "lexicographic_opaque_row_id",
        "key_id": "xkidv1_" + "1" * 64,
        "private_group_linkage_commitment": "xglcv1_" + "2" * 64,
    }
    value["contract_sha256"] = payload_sha256(value)
    return value


class LocalizationContractTests(unittest.TestCase):
    def test_registry_is_exact_12_plus_9_and_system_roster_is_27(self) -> None:
        registry = load_localization_registry(
            REPO / "configs/reconstruction_benchmark_v1/localization.json"
        )
        self.assertEqual(len(registry["processbench"]["source_cells"]), 12)
        self.assertEqual(tuple(registry["prmbench"]["error_families"]), PRMBENCH_ERROR_FAMILIES)
        roster = primary_system_roster([f"method_{index}" for index in range(13)])
        self.assertEqual(len(roster), 27)
        self.assertEqual(sum(row["role"] == "primary_localization_adapter" for row in roster), 13)
        self.assertEqual(sum(row["role"] == "adapter_ablation" for row in roster), 14)

    def test_empirical_midrank_ties_and_geometric_adapter(self) -> None:
        self.assertTrue(np.array_equal(
            empirical_midrank([1.0, 1.0, 3.0, 5.0]),
            np.asarray([0.25, 0.25, 0.625, 0.875]),
        ))
        row_ids = tuple(f"xridv2_{index:064x}" for index in range(4))
        response = np.vstack([
            np.asarray([3.0, 0.0, 2.0, 1.0]) + method_index
            for method_index in range(13)
        ])
        cell = PreparedLocalizationCell(
            cell_id="processbench_gsm8k_qwen3_4b",
            population_id="pb",
            dataset_id="processbench",
            model_id="qwen3_4b",
            slice_id="gsm8k",
            row_ids=row_ids,
            token_confidence=np.zeros((8, 29), dtype=np.float64),
            token_offsets=np.asarray([0, 2, 4, 6, 8]),
            segment_offsets=np.asarray([0, 1, 2, 3, 4]),
            segment_starts=np.asarray([0, 2, 4, 6]),
            segment_ends=np.asarray([2, 4, 6, 8]),
            response_scores=response,
            method_ids=tuple(f"method_{index}" for index in range(13)),
            identity_contract=_identity_contract(),
            external_certificate_sha256="3" * 64,
            external_score_bindings_sha256="4" * 64,
            token_transform_sha256="5" * 64,
            artifact_sha256="6" * 64,
        )
        token_risk = np.asarray([0.0, 0.1, 0.4, 0.3, 0.2, 0.7, 0.5, 0.9])
        with mock.patch.object(
            LF, "_fit_token_iu",
            return_value=(token_risk, {"fit_sha256": "7" * 64}),
        ):
            bundle = LF.fit_localization_cell(cell)
        step_rank = empirical_midrank([0.1, 0.4, 0.7, 0.9])
        response_rank = empirical_midrank(response[0])
        self.assertTrue(np.allclose(bundle.system_scores[0], np.sqrt(step_rank * response_rank)))
        self.assertTrue(np.array_equal(bundle.system_scores[13], response_rank))
        self.assertTrue(np.array_equal(bundle.system_scores[-1], step_rank))
        legacy = 0.75 * response_rank + 0.25 * step_rank
        self.assertFalse(np.allclose(bundle.system_scores[0], legacy))

    def test_fit_boundary_rejects_target_or_group_members(self) -> None:
        for name in ("label", "source_group", "first_error_step", "correctness"):
            with self.assertRaises(RuntimeError):
                assert_no_target_named_members(("token_confidence", name))

    def test_comparator_projection_selects_scores_not_colocated_labels(self) -> None:
        first = [
            {"rewards": [0.9, 0.2], "prediction": 1, "label": 777},
            {"rewards": [0.8], "prediction": -1, "first_error": 22},
        ]
        second = [
            {"rewards": [0.9, 0.2], "prediction": 1, "label": -999},
            {"rewards": [0.8], "prediction": -1, "first_error": -1},
        ]
        projected_a = _project_rows(kind="processbench_prm", rows=first)
        projected_b = _project_rows(kind="processbench_prm", rows=second)
        for left, right in zip(projected_a, projected_b):
            self.assertTrue(np.array_equal(left, right))

    def test_external_response_binding_is_recomputed_from_signed_comparisons(self) -> None:
        cell_id = "processbench_gsm8k_qwen3_4b"
        records = []
        for index, method_id in enumerate(PRIMARY_METHOD_IDS):
            records.append({
                "cell_id": cell_id, "method_id": method_id,
                "method_version_id": f"v{index}", "config_sha256": f"c{index}",
                "record_sha256": f"r{index}", "score_sha256": f"s{index}",
                "row_roster_sha256": "rows", "status": "OK",
            })
        bindings, digest = _external_response_bindings(
            {"comparison_records": records}, expected_cells={cell_id}
        )[cell_id]
        self.assertEqual(digest, payload_sha256(bindings))
        changed = [dict(row) for row in records]
        changed[0]["score_sha256"] = "foreign"
        changed_digest = _external_response_bindings(
            {"comparison_records": changed}, expected_cells={cell_id}
        )[cell_id][1]
        self.assertNotEqual(digest, changed_digest)


class ProcessBenchEvaluationTests(unittest.TestCase):
    def _rows(self) -> list[dict]:
        rows = []
        for subset in PROCESSBENCH_SUBSETS:
            for index in range(10):
                rows.append({
                    "row_id": f"{subset}:clean:{index}",
                    "group_id": f"{subset}:clean:{index}",
                    "slice_id": subset,
                    "first_error": -1,
                    "step_scores": [0.05, 0.10],
                })
                rows.append({
                    "row_id": f"{subset}:error:{index}",
                    "group_id": f"{subset}:error:{index}",
                    "slice_id": subset,
                    "first_error": 1,
                    "step_scores": [0.05, 0.90],
                })
        return rows

    def test_argmax_rule_is_strict_and_ties_choose_earliest(self) -> None:
        self.assertEqual(processbench_prediction([0.1, 0.9, 0.9], 0.9), -1)
        self.assertEqual(processbench_prediction([0.1, 0.9, 0.9], 0.89), 1)

    def test_five_fold_source_crossfit_is_perfect_and_model_independent(self) -> None:
        rows = self._rows()
        folds = assign_processbench_folds(rows)
        model_copy = [{**row, "model_id": "another_model", "step_scores": [9.0]} for row in rows]
        self.assertEqual(folds, assign_processbench_folds(model_copy))
        self.assertEqual(set(folds.values()), set(range(5)))
        result = crossfit_processbench_threshold(rows)
        self.assertEqual(result["metrics"]["aggregate"]["official_macro_f1"], 1.0)
        self.assertEqual(result["metrics"]["aggregate"]["first_error_exact"], 1.0)
        self.assertEqual(result["metrics"]["aggregate"]["first_error_within_one"], 1.0)
        self.assertEqual(result["metrics"]["aggregate"]["clean_abstention_accuracy"], 1.0)
        self.assertEqual(len(result["calibration_ledgers"]), 5)
        self.assertTrue(all(row["threshold_fit_stage"] == "post_score_freeze" for row in [result]))

    def test_abstention_is_not_a_within_one_localization_for_step_zero(self) -> None:
        metrics = processbench_trace_metrics([0, -1], [-1, -1])
        self.assertEqual(metrics["first_error_within_one"], 0.0)
        self.assertEqual(metrics["clean_abstention_accuracy"], 1.0)

    def test_official_f1_is_zero_not_undefined_when_both_accuracies_are_zero(self) -> None:
        metrics = processbench_trace_metrics([0, -1], [-1, 0])
        self.assertEqual(metrics["first_error_exact"], 0.0)
        self.assertEqual(metrics["clean_abstention_accuracy"], 0.0)
        self.assertEqual(metrics["official_macro_f1"], 0.0)
        self.assertEqual(metrics["status"], "OK")

    def test_vectorized_shared_pb_bootstrap_matches_literal_resampling(self) -> None:
        rows = []
        predictions = {"a": [], "b": []}
        for subset in PROCESSBENCH_SUBSETS:
            for index in range(3):
                rows.append({
                    "row_id": f"{subset}:c:{index}", "group_id": f"{subset}:c:{index}",
                    "slice_id": subset, "first_error": -1,
                    "bootstrap_stratum": f"{subset}::clean", "prediction_step": -1,
                })
                predictions["a"].append(-1)
                predictions["b"].append(None if index == 0 else 0)
                rows.append({
                    "row_id": f"{subset}:e:{index}", "group_id": f"{subset}:e:{index}",
                    "slice_id": subset, "first_error": index % 2,
                    "bootstrap_stratum": f"{subset}::error", "prediction_step": index % 2,
                })
                predictions["a"].append(index % 2)
                predictions["b"].append(-1)
        shared = _shared_pb_bootstrap_results(
            rows=rows, predictions_by_system=predictions, draws=41, seed=29,
        )
        for system_id in ("a", "b"):
            literal_rows = [
                {**row, "prediction_step": prediction}
                for row, prediction in zip(rows, predictions[system_id])
            ]
            literal = grouped_bootstrap_metric_map(
                literal_rows, _pb_bootstrap_statistic,
                stratum_key="bootstrap_stratum", draws=41, seed=29,
                bootstrap_unit="source_question", include_samples=True,
            )
            self.assertEqual(
                shared[system_id]["draw_stream_sha256"], literal["draw_stream_sha256"]
            )
            for metric_id, values in shared[system_id]["samples"].items():
                self.assertTrue(np.allclose(
                    values,
                    np.asarray(literal["samples"][metric_id], dtype=np.float64),
                    equal_nan=True,
                ), metric_id)


class PRMBenchEvaluationTests(unittest.TestCase):
    def _panel_rows(self) -> list[dict]:
        rows = []
        for family in PRMBENCH_ERROR_FAMILIES:
            labels = [0, 0] if family == "multi_solutions" else [0, 1]
            for index, label in enumerate(labels):
                rows.append({
                    "group_id": f"{family}:{index}",
                    "error_family": family,
                    "step_label": label,
                    "step_score": 0.9 if label else 0.1,
                })
        return rows

    def test_all_nine_families_are_visible_and_multi_solutions_is_explicit(self) -> None:
        panel = prmbench_panel_metrics(self._panel_rows())
        self.assertTrue(panel["all_nine_families_visible"])
        multi = panel["per_family"]["multi_solutions"]
        self.assertEqual(multi["status"], UNDEFINED_SINGLE_CLASS)
        self.assertIsNone(multi["auroc"])
        self.assertIsNone(multi["auprc"])
        self.assertEqual(multi["coverage"], 1.0)
        self.assertEqual(panel["overall"]["status"], "OK")

    def test_grouped_bootstrap_is_deterministic(self) -> None:
        rows = [row for row in self._panel_rows() if row["error_family"] != "multi_solutions"]
        first = bootstrap_prmbench_steps(rows, draws=100, seed=19)
        second = bootstrap_prmbench_steps(rows, draws=100, seed=19)
        self.assertEqual(first, second)
        self.assertEqual(first["draws"], 100)
        self.assertEqual(first["draws_executed"], 100)
        self.assertEqual(first["bootstrap_unit"], "source_idx")

    def test_actual_draw_stream_is_enforced_and_pairable(self) -> None:
        rows = [
            {"group_id": f"g{i}", "value": float(i)} for i in range(6)
        ]

        def statistic(sample: list[dict]) -> dict[str, float]:
            return {"mean": float(np.mean([row["value"] for row in sample]))}

        first = grouped_bootstrap_metric_map(
            rows, statistic, draws=31, seed=7, bootstrap_unit="source_idx",
            include_samples=True,
        )
        second = grouped_bootstrap_metric_map(
            [{**row, "value": row["value"] * 2} for row in rows],
            statistic, draws=31, seed=7, bootstrap_unit="source_idx",
            include_samples=True,
        )
        _require_bootstrap_execution(first, draws=31, unit="source_idx")
        self.assertEqual(first["draw_stream_sha256"], second["draw_stream_sha256"])
        self.assertNotEqual(first["sample_stream_sha256"], second["sample_stream_sha256"])
        candidate = {"value": first["statistics"]["mean"]["point"]}
        reference = {"value": second["statistics"]["mean"]["point"]}
        delta = _paired_contrast(
            candidate=candidate, reference=reference,
            candidate_samples=first["samples"]["mean"],
            reference_samples=second["samples"]["mean"], draws=31,
        )
        self.assertEqual(delta[3], 31)
        tampered = {**first, "draws_executed": 30}
        with self.assertRaisesRegex(RuntimeError, "exact registered draws"):
            _require_bootstrap_execution(tampered, draws=31, unit="source_idx")

    def test_vectorized_shared_prm_bootstrap_matches_literal_resampling(self) -> None:
        group_ids = ("g0", "g0", "g1", "g1", "g2", "g2", "g2")
        families = ("f",) * len(group_ids)
        labels = np.asarray([0, 1, 0, 1, 0, 0, 1], dtype=np.int8)
        score_matrix = np.asarray([
            [0.1, 0.9, 0.2, 0.8, 0.2, 0.4, 0.8],
            [0.8, 0.2, 0.7, 0.3, 0.7, 0.5, 0.3],
        ], dtype=np.float64)
        shared = _shared_prm_bootstrap_results(
            labels=labels, score_matrix=score_matrix, group_ids=group_ids,
            strata=families, system_ids=("a", "b"), draws=37, seed=23,
        )
        for system_index, system_id in enumerate(("a", "b")):
            rows = [
                {
                    "group_id": group_id, "error_family": "f",
                    "step_label": int(label),
                    "step_score": float(score_matrix[system_index, index]),
                }
                for index, (group_id, label) in enumerate(zip(group_ids, labels))
            ]

            def statistic(sample: list[dict]) -> dict[str, float | None]:
                value = LE.prmbench_step_metrics(
                    [row["step_label"] for row in sample],
                    [row["step_score"] for row in sample],
                )
                return {
                    metric: value[metric]
                    for metric in ("auroc", "auprc", "mean_risk", "risk_q90", "coverage")
                }

            literal = grouped_bootstrap_metric_map(
                rows, statistic, stratum_key="error_family", draws=37, seed=23,
                bootstrap_unit="source_idx", include_samples=True,
            )
            self.assertEqual(
                shared[system_id]["draw_stream_sha256"], literal["draw_stream_sha256"]
            )
            for metric_id in ("auroc", "auprc", "mean_risk", "risk_q90", "coverage"):
                self.assertTrue(np.allclose(
                    shared[system_id]["samples"][metric_id],
                    np.asarray([
                        np.nan if value is None else value
                        for value in literal["samples"][metric_id]
                    ]),
                    equal_nan=True,
                ), metric_id)


class EvaluationArtifactTests(unittest.TestCase):
    def _decision(self) -> dict:
        return dict(zip(LOCALIZATION_DECISION_FIELDS, (
            "first_error_localization", "processbench", "processbench_qwen3_fixed_first_error_v1",
            "processbench_gsm8k_qwen3_4b", "gsm8k", "qwen3_4b",
            "iu_pcr__loc_geomean_v1", "xridv2_" + "a" * 64,
            "cohort", "group", 0, 1, 1, "OK", "black_box", "exact_common_rows",
            "pb_core", "runhash",
        )))

    def _metric(self) -> dict:
        return dict(zip(METRIC_FIELDS, (
            "first_error_localization", "processbench", "processbench_qwen3_fixed_first_error_v1",
            "processbench_gsm8k_qwen3_4b", "gsm8k", "qwen3_4b",
            "iu_pcr__loc_geomean_v1", "official_macro_f1", 1.0, 1.0,
            1.0, 1, 1, 0, "OK", "black_box", "exact_common_rows", "pb_core",
            "source_question", 20000, "cohort", "runhash",
        )))

    def test_caller_supplied_writer_is_removed_and_verifier_has_no_table_arguments(self) -> None:
        self.assertFalse(hasattr(LE, "write_localization_evaluation_bundle"))
        parameters = inspect.signature(verify_localization_evaluation_ab).parameters
        for forbidden in ("decisions", "metrics", "coverage", "calibration_ledgers"):
            self.assertNotIn(forbidden, parameters)

    def test_integer_decision_schema_and_fabricated_roster_fail_closed(self) -> None:
        bad = self._decision()
        bad["prediction_step"] = True
        config = load_localization_registry(
            REPO / "configs/reconstruction_benchmark_v1/localization.json"
        )
        with self.assertRaisesRegex(RuntimeError, "coerced to boolean"):
            _validate_evaluation_tables(
                config=config, decisions=[bad], metrics=[], contrasts=[], coverage=[],
                calibration=[], executions=[], draws=20000,
            )
        with self.assertRaisesRegex(RuntimeError, "12 cells x 30 systems"):
            _validate_evaluation_tables(
                config=config, decisions=[], metrics=[self._metric()], contrasts=[],
                coverage=[], calibration=[], executions=[], draws=20000,
            )

    def test_scientific_writer_rejects_declared_short_bootstrap_before_derivation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exactly 20,000"):
            write_localization_evaluation_build(
                release_id="r", build_id="A", release_root=REPO,
                scientific_full=True, bootstrap_draws=19_999,
            )


class StrictPostFreezeDerivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_localization_registry(
            REPO / "configs/reconstruction_benchmark_v1/localization.json"
        )
        cls.core_ids = tuple(
            row["system_id"] for row in primary_system_roster(PRIMARY_METHOD_IDS)
        )

    def test_complete_synthetic_pb_and_prm_rosters_are_derived(self) -> None:
        pb_context = tuple(
            row["system_id"] for row in self.config["comparators"]
            if row["dataset_id"] == "processbench"
        )
        cells = {}
        for model_id in self.config["processbench"]["models"]:
            for subset in PROCESSBENCH_SUBSETS:
                cell_id = f"processbench_{subset}_{model_id}"
                n_rows = 20
                labels = np.asarray([-1] * 10 + [0] * 10, dtype=np.int64)
                row_ids = tuple(f"{model_id}:{subset}:row:{index}" for index in range(n_rows))
                groups = tuple(f"{subset}:source:{index}" for index in range(n_rows))
                values = np.asarray([0.1] * 10 + [0.9] * 10, dtype=np.float64)
                all_systems = (*self.core_ids, *pb_context)
                cells[cell_id] = PBCell(
                    cell_id=cell_id,
                    population_id=self.config["processbench"]["population_id_by_model"][model_id],
                    model_id=model_id,
                    slice_id=subset,
                    row_ids=row_ids,
                    group_ids=groups,
                    first_error=labels,
                    segment_offsets=np.arange(n_rows + 1, dtype=np.int64),
                    core_system_ids=self.core_ids,
                    core_scores=np.tile(values, (27, 1)),
                    comparator_predictions={
                        system_id: tuple(map(int, labels)) for system_id in pb_context
                    },
                    comparator_coverage={
                        system_id: np.ones(n_rows, dtype=np.int8) for system_id in pb_context
                    },
                    run_hashes={system_id: f"run::{cell_id}::{system_id}" for system_id in all_systems},
                    source_records=(),
                )
        pb = _evaluate_processbench(config=self.config, cells_by_id=cells, draws=3)
        decisions, metrics, contrasts, coverage, calibration, executions = pb
        self.assertEqual(len(decisions), 3 * 4 * 20 * 30)
        self.assertEqual(len(metrics), 3 * 5 * 30 * 5)
        self.assertEqual(len(contrasts), 3 * 5 * 27 * 5)
        self.assertEqual(len(coverage), 3 * 5 * 30)
        self.assertEqual(len(calibration), 3 * 27 * 5)
        self.assertEqual(len(executions), 3 * 30)
        self.assertTrue(all(row["draws_executed"] == 3 for row in executions))

        prm_context = next(
            row["system_id"] for row in self.config["comparators"]
            if row["dataset_id"] == "prmbench"
        )
        families = tuple(
            family for family in PRMBENCH_ERROR_FAMILIES for _ in range(2)
        )
        labels = np.asarray([
            label
            for family in PRMBENCH_ERROR_FAMILIES
            for label in ((0, 0) if family == "multi_solutions" else (0, 1))
        ], dtype=np.int8)
        prm_systems = (*self.core_ids, prm_context)
        prm_panel = PRMPanel(
            cell_id=self.config["prmbench"]["source_cell"],
            population_id=self.config["prmbench"]["population_id"],
            model_id=self.config["prmbench"]["model_id"],
            response_row_ids=tuple(f"prm:row:{index}" for index in range(len(labels))),
            group_ids=tuple(f"prm:source_idx:{index}" for index in range(len(labels))),
            error_families=families,
            step_offsets=np.arange(len(labels) + 1, dtype=np.int64),
            step_labels=labels,
            system_ids=prm_systems,
            system_scores=np.tile(np.where(labels == 1, 0.9, 0.1), (28, 1)),
            run_hashes={system_id: f"run::prm::{system_id}" for system_id in prm_systems},
            source_records=(),
        )
        prm_metrics, prm_contrasts, prm_coverage, prm_executions, prm_npz = (
            _evaluate_prmbench(config=self.config, panel=prm_panel, draws=3)
        )
        self.assertEqual(len(prm_metrics), 10 * 28 * 5)
        self.assertEqual(len(prm_contrasts), 10 * 27 * 5)
        self.assertEqual(len(prm_coverage), 10 * 28)
        self.assertEqual(len(prm_executions), 10 * 28)
        self.assertGreater(len(prm_npz), 0)
        undefined = [
            row for row in prm_metrics
            if row["slice_id"] == "multi_solutions"
            and row["metric_id"] in ("auroc", "auprc")
        ]
        self.assertEqual(len(undefined), 28 * 2)
        self.assertTrue(all(row["status"] == UNDEFINED_SINGLE_CLASS for row in undefined))


if __name__ == "__main__":
    unittest.main()
