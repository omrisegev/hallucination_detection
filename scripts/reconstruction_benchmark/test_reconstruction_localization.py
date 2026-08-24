#!/usr/bin/env python3
"""Focused contract tests for reconstruction localization v1 (no data run)."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark import localization_fit as LF  # noqa: E402
from spectral_utils.reconstruction_benchmark import localization_evaluation as LE  # noqa: E402
from spectral_utils.reconstruction_benchmark import localization_postfreeze as LP  # noqa: E402
from spectral_utils.reconstruction_benchmark import (  # noqa: E402
    localization_evaluation_ab_verifier as LEAV,
)
from spectral_utils.reconstruction_benchmark import (  # noqa: E402
    localization_postfreeze_amendment as LPA,
)
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
    fit_processbench_threshold,
    grouped_bootstrap_metric_map,
    prmbench_panel_metrics,
    processbench_panel_metrics,
    processbench_prediction,
    processbench_trace_metrics,
)
from spectral_utils.reconstruction_benchmark.localization_postfreeze import (  # noqa: E402
    DerivedLocalizationEvaluation,
    PBCell,
    PRMPanel,
    _partition_prmbench_error_steps,
    _score_verifier_repo_snapshot,
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
from spectral_utils.reconstruction_benchmark.localization_postfreeze_amendment import (  # noqa: E402
    EXPECTED_AMENDMENT_FILE_SHA256,
    EXPECTED_AMENDMENT_PAYLOAD_SHA256,
    EXPECTED_LOCALIZATION_REGISTRY_SHA256,
    EXPECTED_OOB_RECORDS_SHA256,
    EXPECTED_RELEASE_ID,
    EXPECTED_SCORE_AB_CERTIFICATE_FILE_SHA256,
    EXPECTED_SCORE_AB_CERTIFICATE_SHA256,
    EXPECTED_SCORE_VERIFIER_GIT_HEAD,
    EXPECTED_TELEMETRY_MANIFEST_SHA256,
    EXPECTED_TELEMETRY_SHA256,
    apply_localization_postfreeze_amendment,
    load_localization_postfreeze_amendment,
    validate_observed_prmbench_oob_audit,
)
from spectral_utils.reconstruction_benchmark.localization_evaluation_ab_verifier import (  # noqa: E402
    EVALUATION_AB_RELEASE_SCHEMA_VERSION,
    verify_localization_evaluation_ab_release,
)
from spectral_utils.reconstruction_benchmark.io import sha256_file  # noqa: E402
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


def _literal_fit_processbench_threshold(
    rows: list[dict], *, expected_subsets=PROCESSBENCH_SUBSETS,
) -> dict:
    """Frozen O(n^2) oracle used only to prove optimized sweep equivalence."""

    rows = list(rows)
    if not rows:
        raise ValueError("cannot fit a ProcessBench threshold on zero rows")
    maxima = [float(np.max(LE._finite_step_scores(row))) for row in rows]
    below_minimum = float(np.nextafter(min(maxima), -np.inf))
    if not np.isfinite(below_minimum):
        raise ValueError("ProcessBench score range cannot form a finite threshold sweep")
    candidates = sorted(set(maxima) | {below_minimum})
    best = None
    for threshold in candidates:
        predictions = [
            processbench_prediction(row["step_scores"], threshold) for row in rows
        ]
        metrics = processbench_panel_metrics(
            rows, predictions, expected_subsets=expected_subsets
        )
        objective = metrics["aggregate"]["official_macro_f1"]
        candidate = {
            "threshold": float(threshold),
            "objective_equal_subset_official_macro_f1": float(objective),
            "n_calibration_rows": len(rows),
            "n_threshold_candidates": len(candidates),
            "decision_rule": "argmax_if_max_strictly_greater_than_threshold",
            "tie_break": "largest_numeric_threshold",
        }
        order = (
            float(objective) if np.isfinite(objective) else -float("inf"),
            float(threshold),
        )
        if best is None or order > best[0]:
            best = (order, candidate)
    assert best is not None
    if not np.isfinite(best[1]["objective_equal_subset_official_macro_f1"]):
        raise ValueError("ProcessBench calibration lacks clean/error support in every subset")
    best[1]["calibration_sha256"] = payload_sha256(best[1])
    return best[1]


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

    def test_incremental_threshold_fit_matches_literal_randomized_tied_sweeps(self) -> None:
        for seed in range(20):
            rng = np.random.default_rng(seed)
            rows = []
            for subset in PROCESSBENCH_SUBSETS:
                for label_kind in ("clean", "error"):
                    for index in range(7):
                        n_steps = int(rng.integers(1, 7))
                        # A small discrete support deliberately creates both
                        # within-row argmax ties and across-row maximum ties.
                        scores = (
                            rng.integers(-4, 5, size=n_steps).astype(np.float64) / 4.0
                        )
                        label = (
                            -1 if label_kind == "clean"
                            else int(rng.integers(0, n_steps))
                        )
                        rows.append({
                            "row_id": f"{seed}:{subset}:{label_kind}:{index}",
                            "group_id": f"{seed}:{subset}:{label_kind}:{index}",
                            "slice_id": subset,
                            "first_error": label,
                            "step_scores": scores,
                        })
            self.assertEqual(
                fit_processbench_threshold(rows),
                _literal_fit_processbench_threshold(rows),
            )

    def test_incremental_threshold_fit_preserves_strict_rule_and_tie_break_edges(self) -> None:
        rows = []
        for subset in PROCESSBENCH_SUBSETS:
            rows.extend((
                {
                    "row_id": f"{subset}:clean", "group_id": f"{subset}:clean",
                    "slice_id": subset, "first_error": -1,
                    "step_scores": [0.8, 0.8],
                },
                {
                    "row_id": f"{subset}:error", "group_id": f"{subset}:error",
                    "slice_id": subset, "first_error": 0,
                    "step_scores": [0.8, 0.8],
                },
            ))
        observed = fit_processbench_threshold(rows)
        self.assertEqual(observed, _literal_fit_processbench_threshold(rows))
        self.assertEqual(observed["threshold"], 0.8)
        self.assertEqual(observed["objective_equal_subset_official_macro_f1"], 0.0)
        self.assertEqual(observed["tie_break"], "largest_numeric_threshold")

        unsupported = [row for row in rows if row["first_error"] == -1]
        with self.assertRaisesRegex(ValueError, "exact registered subsets"):
            fit_processbench_threshold(unsupported[:-1])
        with self.assertRaisesRegex(ValueError, "lacks clean/error support"):
            fit_processbench_threshold(unsupported)

        extreme = deepcopy(rows)
        for row in extreme:
            row["step_scores"] = [-np.finfo(np.float64).max]
        with np.errstate(over="ignore"):
            with self.assertRaisesRegex(ValueError, "finite threshold sweep"):
                fit_processbench_threshold(extreme)

    def test_crossfit_output_is_byte_semantic_equal_to_literal_threshold_oracle(self) -> None:
        rng = np.random.default_rng(991)
        rows = self._rows()
        for row in rows:
            row["step_scores"] = (
                rng.integers(-8, 9, size=5).astype(np.float64) / 8.0
            )
            if row["first_error"] != -1:
                row["first_error"] = int(rng.integers(0, 5))
        optimized = crossfit_processbench_threshold(rows)
        with mock.patch.object(
            LE, "fit_processbench_threshold",
            side_effect=_literal_fit_processbench_threshold,
        ):
            literal = crossfit_processbench_threshold(rows)
        self.assertEqual(optimized, literal)

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


class PostFreezeAmendmentTests(unittest.TestCase):
    amendment_path = (
        REPO
        / "configs/reconstruction_benchmark_v1/localization_postfreeze_amendment_v1.json"
    )

    def _amendment_document(self) -> dict:
        return json.loads(self.amendment_path.read_text(encoding="utf-8"))

    def test_amendment_is_exactly_100_rows_151_inert_and_13144_effective(self) -> None:
        value = self._amendment_document()
        payload = dict(value)
        recorded = payload.pop("payload_sha256")
        self.assertEqual(sha256_file(self.amendment_path), EXPECTED_AMENDMENT_FILE_SHA256)
        self.assertEqual(recorded, EXPECTED_AMENDMENT_PAYLOAD_SHA256)
        self.assertEqual(recorded, payload_sha256(payload))
        records = value["oob_audit"]["records"]
        self.assertEqual(len(records), 100)
        self.assertEqual(sum(len(row["invalid"]) for row in records), 151)
        self.assertEqual(payload_sha256(records), EXPECTED_OOB_RECORDS_SHA256)
        self.assertEqual(
            value["effective_prmbench_counts"]["expected_positive_steps"], 13_144
        )
        self.assertEqual(
            value["original_prmbench_counts"]["expected_positive_steps"] - 151,
            value["effective_prmbench_counts"]["expected_positive_steps"],
        )

    def test_real_shaped_oob_is_inert_without_shift_clamp_or_row_drop(self) -> None:
        valid, invalid = _partition_prmbench_error_steps([52, 54], n_steps=53)
        self.assertEqual(valid, (52,))
        self.assertEqual(invalid, (54,))
        labels = np.asarray([
            int(step_index + 1 in set(valid)) for step_index in range(53)
        ])
        self.assertEqual(int(labels.sum()), 1)
        self.assertEqual(int(labels[51]), 1)
        self.assertEqual(int(labels[52]), 0)
        with self.assertRaisesRegex(RuntimeError, "zero/negative"):
            _partition_prmbench_error_steps([0, 1], n_steps=3)
        with self.assertRaisesRegex(RuntimeError, "exact one-based integers"):
            _partition_prmbench_error_steps([True, 1], n_steps=3)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            _partition_prmbench_error_steps([1, 1], n_steps=3)

    def test_amendment_application_changes_only_disclosed_prm_counts(self) -> None:
        config = load_localization_registry(
            REPO / "configs/reconstruction_benchmark_v1/localization.json"
        )
        original = deepcopy(config)
        amendment = self._amendment_document()
        effective = apply_localization_postfreeze_amendment(config, amendment)
        self.assertEqual(config, original)
        expected = deepcopy(original)
        for field in (
            "expected_error_responses", "expected_steps", "expected_positive_steps",
            "expected_by_family",
        ):
            expected["prmbench"][field] = deepcopy(
                amendment["effective_prmbench_counts"][field]
            )
        self.assertEqual(effective, expected)

    def test_observed_audit_must_match_all_100_records_byte_semantically(self) -> None:
        amendment = self._amendment_document()
        amendment["file_sha256"] = EXPECTED_AMENDMENT_FILE_SHA256
        audit = validate_observed_prmbench_oob_audit(
            amendment["oob_audit"]["records"], amendment,
            all_annotation_count=13_295,
            minimum_annotation=1,
            zero_count=0,
            negative_count=0,
            duplicate_annotation_rows=0,
        )
        self.assertEqual(audit["row_count"], 100)
        self.assertEqual(audit["annotation_count"], 151)
        tampered = deepcopy(amendment["oob_audit"]["records"])
        tampered[0]["invalid"][0] += 1
        with self.assertRaisesRegex(RuntimeError, "differs from amendment"):
            validate_observed_prmbench_oob_audit(
                tampered, amendment,
                all_annotation_count=13_295,
                minimum_annotation=1,
                zero_count=0,
                negative_count=0,
                duplicate_annotation_rows=0,
            )

    def test_wrong_tampered_and_malformed_amendments_fail_closed(self) -> None:
        exact = self._amendment_document()
        certificate = {
            "schema_version": "reconstruction-localization-ab-certificate-v2",
            "certificate_sha256": EXPECTED_SCORE_AB_CERTIFICATE_SHA256,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            amendment_path = root / "amendment.json"
            amendment_path.write_bytes(self.amendment_path.read_bytes())
            certificate_path = (
                root / EXPECTED_RELEASE_ID / "localization/AB_VERIFICATION.json"
            )
            certificate_path.parent.mkdir(parents=True)
            certificate_path.write_text("{}\n", encoding="utf-8")
            source_root = root / "source"
            source_root.mkdir()

            def fixed_sha(path: str | Path) -> str:
                candidate = Path(path).resolve()
                if candidate == amendment_path.resolve():
                    return EXPECTED_AMENDMENT_FILE_SHA256
                if candidate == (
                    REPO / "configs/reconstruction_benchmark_v1/localization.json"
                ).resolve():
                    return EXPECTED_LOCALIZATION_REGISTRY_SHA256
                if candidate == certificate_path.resolve():
                    return EXPECTED_SCORE_AB_CERTIFICATE_FILE_SHA256
                if candidate.name == "prmbench_telemetry.pkl":
                    return EXPECTED_TELEMETRY_SHA256
                if candidate.name == "manifest.json":
                    return EXPECTED_TELEMETRY_MANIFEST_SHA256
                raise AssertionError(f"unexpected hash target: {candidate}")

            kwargs = {
                "release_id": EXPECTED_RELEASE_ID,
                "localization_registry_path": (
                    REPO / "configs/reconstruction_benchmark_v1/localization.json"
                ),
                "score_ab_certificate_path": certificate_path,
                "score_ab_certificate": certificate,
                "source_root": source_root,
            }
            with mock.patch.object(LPA, "sha256_file", side_effect=fixed_sha):
                loaded = load_localization_postfreeze_amendment(amendment_path, **kwargs)
                self.assertEqual(loaded["amendment_id"], exact["amendment_id"])
                with self.assertRaisesRegex(RuntimeError, "does not apply"):
                    load_localization_postfreeze_amendment(
                        amendment_path, **{**kwargs, "release_id": "wrong_release"}
                    )

                tampered = deepcopy(exact)
                tampered["reason"] += " tampered"
                payload = dict(tampered)
                payload.pop("payload_sha256")
                tampered["payload_sha256"] = payload_sha256(payload)
                amendment_path.write_text(json.dumps(tampered), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "payload hash"):
                    load_localization_postfreeze_amendment(amendment_path, **kwargs)

                malformed = deepcopy(exact)
                malformed["oob_audit"] = []
                payload = dict(malformed)
                payload.pop("payload_sha256")
                malformed["payload_sha256"] = payload_sha256(payload)
                amendment_path.write_text(json.dumps(malformed), encoding="utf-8")
                with mock.patch.object(
                    LPA, "EXPECTED_AMENDMENT_PAYLOAD_SHA256",
                    malformed["payload_sha256"],
                ):
                    with self.assertRaisesRegex(RuntimeError, "OOB audit is malformed"):
                        load_localization_postfreeze_amendment(amendment_path, **kwargs)

    def test_score_verifier_repo_head_is_enforced_not_decorative(self) -> None:
        clean = {
            "git_head": EXPECTED_SCORE_VERIFIER_GIT_HEAD,
            "git_clean": True,
            "git_status_sha256": "0" * 64,
            "snapshot_sha256": "1" * 64,
        }
        with mock.patch.object(LP, "_repo_state", return_value=clean):
            snapshot = _score_verifier_repo_snapshot(
                REPO, required_git_head=EXPECTED_SCORE_VERIFIER_GIT_HEAD,
            )
        self.assertEqual(snapshot["git_head"], EXPECTED_SCORE_VERIFIER_GIT_HEAD)
        wrong = {**clean, "git_head": "f" * 40}
        with mock.patch.object(LP, "_repo_state", return_value=wrong):
            with self.assertRaisesRegex(RuntimeError, "frozen HEAD"):
                _score_verifier_repo_snapshot(
                    REPO, required_git_head=EXPECTED_SCORE_VERIFIER_GIT_HEAD,
                )


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

    def test_score_ab_failure_precedes_amendment_or_target_access(self) -> None:
        score_repo = REPO
        snapshot = {
            "repo_role": "score_ab_verifier",
            "required_git_head": EXPECTED_SCORE_VERIFIER_GIT_HEAD,
            "git_head": EXPECTED_SCORE_VERIFIER_GIT_HEAD,
            "git_clean": True,
            "git_status_sha256": "0" * 64,
            "snapshot_sha256": "1" * 64,
        }
        with (
            mock.patch.object(LP, "_score_verifier_repo_snapshot", return_value=snapshot),
            mock.patch.object(
                LP, "assert_localization_ab_certificate",
                side_effect=RuntimeError("score certificate rejected"),
            ) as score_gate,
            mock.patch.object(LP, "load_localization_postfreeze_amendment") as amendment,
        ):
            with self.assertRaisesRegex(RuntimeError, "score certificate rejected"):
                LP.derive_localization_evaluation(
                    release_id=EXPECTED_RELEASE_ID,
                    build_id="A",
                    release_root=REPO,
                    score_verifier_repo=score_repo,
                    bootstrap_draws=1,
                )
        amendment.assert_not_called()
        self.assertEqual(score_gate.call_args.kwargs["repo"], score_repo.resolve())

    def test_evaluation_writer_is_atomic_cleans_failure_and_never_clobbers(self) -> None:
        derived = DerivedLocalizationEvaluation(
            files={"first.bin": b"first", "second.bin": b"second"},
            manifest_core={"bootstrap_draws": 3},
        )
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "evaluation"
            with mock.patch.object(LP, "derive_localization_evaluation", return_value=derived):
                manifest = write_localization_evaluation_build(
                    release_id="r", build_id="A", release_root=parent,
                    output_root=output, scientific_full=False,
                    score_verifier_repo=REPO, bootstrap_draws=3,
                )
            self.assertEqual(manifest["status"], "PASS")
            self.assertEqual((output / "first.bin").read_bytes(), b"first")
            self.assertTrue((output / "MANIFEST.json").is_file())
            with mock.patch.object(LP, "derive_localization_evaluation", return_value=derived):
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    write_localization_evaluation_build(
                        release_id="r", build_id="A", release_root=parent,
                        output_root=output, scientific_full=False,
                        score_verifier_repo=REPO, bootstrap_draws=3,
                    )

            failed = parent / "failed_evaluation"
            with (
                mock.patch.object(LP, "derive_localization_evaluation", return_value=derived),
                mock.patch.object(
                    LP, "atomic_write_bytes",
                    side_effect=[None, RuntimeError("serialization failed")],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "serialization failed"):
                    write_localization_evaluation_build(
                        release_id="r", build_id="A", release_root=parent,
                        output_root=failed, scientific_full=False,
                        score_verifier_repo=REPO, bootstrap_draws=3,
                    )
            self.assertFalse(failed.exists())
            self.assertEqual(list(parent.glob(".failed_evaluation.staging-*")), [])

    def test_directory_publish_primitive_preserves_raced_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            target = root / "target"
            staging.mkdir()
            target.mkdir()
            (staging / "new").write_bytes(b"new")
            (target / "incumbent").write_bytes(b"incumbent")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                LP._rename_directory_noreplace(staging, target)
            self.assertEqual((staging / "new").read_bytes(), b"new")
            self.assertEqual((target / "incumbent").read_bytes(), b"incumbent")

    def test_evaluation_ab_requires_byte_identity_and_binds_both_repos(self) -> None:
        manifest_core = {
            "score_ab_certificate_sha256": "a" * 64,
            "score_ab_certificate_file_sha256": "b" * 64,
            "postfreeze_amendment": {"amendment_id": "amendment"},
            "score_verifier_repo_snapshot": {"git_head": EXPECTED_SCORE_VERIFIER_GIT_HEAD},
            "evaluation_source_snapshot": {"git_head": "e" * 40},
            "completeness": {"completeness_sha256": "c" * 64},
        }
        same = DerivedLocalizationEvaluation(files={"artifact": b"same"}, manifest_core=manifest_core)
        different = DerivedLocalizationEvaluation(
            files={"artifact": b"different"}, manifest_core=manifest_core,
        )
        clean = {
            "git_head": "e" * 40, "git_clean": True,
            "git_status_sha256": "0" * 64, "snapshot_sha256": "1" * 64,
        }
        validated = {
            "manifest_file_sha256": "d" * 64,
            "tree_sha256": "f" * 64,
            "artifact_sha256": {"artifact": "9" * 64},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certificate_path = root / "EVALUATION_AB_VERIFICATION.json"
            with (
                mock.patch.object(LP, "_repo_state", return_value=clean),
                mock.patch.object(LP, "derive_localization_evaluation", return_value=same),
                mock.patch.object(
                    LP, "_validate_evaluation_build_against_derivation",
                    return_value=validated,
                ),
            ):
                certificate = verify_localization_evaluation_ab(
                    release_id="r", release_root=root,
                    score_verifier_repo=REPO,
                    evaluation_repo=REPO,
                    output_path=certificate_path,
                )
            self.assertEqual(certificate["schema_version"], LP.EVALUATION_AB_SCHEMA_VERSION)
            self.assertEqual(
                certificate["score_verifier_repo_snapshot"]["git_head"],
                EXPECTED_SCORE_VERIFIER_GIT_HEAD,
            )
            self.assertEqual(
                certificate["evaluation_source_snapshot"]["git_head"], "e" * 40
            )
            self.assertTrue(certificate_path.is_file())

            with (
                mock.patch.object(LP, "_repo_state", return_value=clean),
                mock.patch.object(
                    LP, "derive_localization_evaluation", side_effect=[same, different],
                ),
                mock.patch.object(
                    LP, "_validate_evaluation_build_against_derivation",
                    return_value=validated,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "derivations differ"):
                    verify_localization_evaluation_ab(
                        release_id="r", release_root=root,
                        score_verifier_repo=REPO,
                        evaluation_repo=REPO,
                        output_path=root / "different-certificate.json",
                    )

    def test_release_verifier_accepts_only_legitimate_build_specific_freezes(self) -> None:
        def manifest_core(freeze_hash: str) -> dict:
            return {
                "score_ab_certificate_sha256": "a" * 64,
                "score_ab_certificate_file_sha256": "b" * 64,
                "score_freeze_payload_sha256": freeze_hash,
                "postfreeze_amendment": {"amendment_id": "amendment"},
                "score_verifier_repo_snapshot": {
                    "git_head": EXPECTED_SCORE_VERIFIER_GIT_HEAD,
                },
                "evaluation_source_snapshot": {"git_head": "6" * 40},
                "completeness": {"completeness_sha256": "c" * 64},
            }

        freeze_a = "1" * 64
        freeze_b = "2" * 64
        derived_a = DerivedLocalizationEvaluation(
            files={"artifact": b"same"}, manifest_core=manifest_core(freeze_a),
        )
        derived_b = DerivedLocalizationEvaluation(
            files={"artifact": b"same"}, manifest_core=manifest_core(freeze_b),
        )
        validated = {
            "manifest_file_sha256": "d" * 64,
            "tree_sha256": "f" * 64,
            "artifact_sha256": {"artifact": "9" * 64},
        }
        boundary = {
            "evaluation_producer_snapshot": {"git_head": "6" * 40},
            "evaluation_ab_verifier_source_snapshot": {"git_head": "7" * 40},
        }
        bindings = {
            "A": {
                "score_freeze_file_sha256": "3" * 64,
                "score_freeze_payload_sha256": freeze_a,
            },
            "B": {
                "score_freeze_file_sha256": "4" * 64,
                "score_freeze_payload_sha256": freeze_b,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "certificate.json"
            with (
                mock.patch.object(
                    LEAV, "_recorded_producer_snapshot",
                    return_value=boundary["evaluation_producer_snapshot"],
                ),
                mock.patch.object(
                    LEAV, "_attest_repository_boundary",
                    side_effect=[boundary, boundary],
                ),
                mock.patch.object(
                    LEAV, "derive_localization_evaluation",
                    side_effect=[derived_a, derived_b],
                ),
                mock.patch.object(
                    LEAV, "_validate_evaluation_build_against_derivation",
                    return_value=validated,
                ),
                mock.patch.object(
                    LEAV, "_verify_score_freeze_payload_bindings",
                    return_value=bindings,
                ),
            ):
                certificate = verify_localization_evaluation_ab_release(
                    release_id="r",
                    release_root=root,
                    output_path=output,
                    score_verifier_repo=REPO,
                    evaluation_producer_repo=root / "producer",
                    verification_repo=REPO,
                    localization_registry_path=REPO / "registry.json",
                    external_registry_path=REPO / "external.json",
                    population_registry_path=REPO / "populations.json",
                    source_root=REPO,
                    localization_postfreeze_amendment_path=REPO / "amendment.json",
                )
            self.assertEqual(
                certificate["schema_version"], EVALUATION_AB_RELEASE_SCHEMA_VERSION
            )
            self.assertEqual(
                certificate["build_specific_manifest_core_fields"],
                ["score_freeze_payload_sha256"],
            )
            self.assertEqual(
                certificate["builds"]["A"]["score_freeze_payload_sha256"],
                freeze_a,
            )
            self.assertEqual(
                certificate["builds"]["B"]["score_freeze_payload_sha256"],
                freeze_b,
            )
            payload = dict(certificate)
            claimed = payload.pop("certificate_sha256")
            self.assertEqual(claimed, payload_sha256(payload))

    def test_release_verifier_normalizes_exactly_one_required_field(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "lacks the sole build-specific"):
            LEAV._split_manifest_core({"shared": "same"})
        shared, build_specific = LEAV._split_manifest_core({
            "score_freeze_payload_sha256": "1" * 64,
            "shared": "same",
            "unexpected_build_field": "still-shared",
        })
        self.assertEqual(
            build_specific, {"score_freeze_payload_sha256": "1" * 64}
        )
        self.assertEqual(shared, {
            "shared": "same", "unexpected_build_field": "still-shared",
        })

    def test_release_verifier_rejects_unanchored_producer_or_dirty_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_id = "r"
            snapshot = {
                "repo_role": "postfreeze_evaluator",
                "git_head": "0" * 40,
                "git_clean": True,
                "git_status_sha256": "e3b0" + "0" * 60,
                "files": [],
            }
            snapshot["snapshot_sha256"] = payload_sha256(snapshot)
            for build_id in ("A", "B"):
                path = (
                    root / release_id / f"build_{build_id}"
                    / "localization/evaluation/MANIFEST.json"
                )
                path.parent.mkdir(parents=True)
                manifest = {
                    "release_id": release_id,
                    "build_id": build_id,
                    "status": "PASS",
                    "scientific_full": True,
                    "bootstrap_draws": 20000,
                    "evaluation_source_snapshot": snapshot,
                }
                manifest["payload_sha256"] = payload_sha256(manifest)
                path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "snapshot anchor is invalid"):
                LEAV._recorded_producer_snapshot(
                    release_root=root, release_id=release_id,
                )

        dirty = {
            "git_head": "1" * 40,
            "git_clean": False,
            "git_status_sha256": "2" * 64,
            "snapshot_sha256": "3" * 64,
        }
        with mock.patch.object(LEAV, "_repo_state", return_value=dirty):
            with self.assertRaisesRegex(RuntimeError, "verifier repo must be clean"):
                LEAV._verifier_source_snapshot(REPO)

    def test_release_verifier_rejects_extra_build_specific_divergence(self) -> None:
        base = {
            "score_ab_certificate_sha256": "a" * 64,
            "score_ab_certificate_file_sha256": "b" * 64,
            "score_freeze_payload_sha256": "1" * 64,
            "postfreeze_amendment": {"amendment_id": "amendment"},
            "score_verifier_repo_snapshot": {"git_head": "d" * 40},
            "evaluation_source_snapshot": {"git_head": "6" * 40},
            "completeness": {"completeness_sha256": "c" * 64},
            "unexpected_build_field": "A",
        }
        derived_a = DerivedLocalizationEvaluation(
            files={"artifact": b"same"}, manifest_core=base,
        )
        derived_b = DerivedLocalizationEvaluation(
            files={"artifact": b"same"},
            manifest_core={**base, "unexpected_build_field": "B"},
        )
        validated = {
            "manifest_file_sha256": "d" * 64,
            "tree_sha256": "f" * 64,
            "artifact_sha256": {"artifact": "9" * 64},
        }
        boundary = {
            "evaluation_producer_snapshot": {"git_head": "6" * 40},
            "evaluation_ab_verifier_source_snapshot": {"git_head": "7" * 40},
        }
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(
                    LEAV, "_recorded_producer_snapshot",
                    return_value=boundary["evaluation_producer_snapshot"],
                ),
                mock.patch.object(
                    LEAV, "_attest_repository_boundary", return_value=boundary,
                ),
                mock.patch.object(
                    LEAV, "derive_localization_evaluation",
                    side_effect=[derived_a, derived_b],
                ),
                mock.patch.object(
                    LEAV, "_validate_evaluation_build_against_derivation",
                    return_value=validated,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "shared manifest cores differ"
                ):
                    verify_localization_evaluation_ab_release(
                        release_id="r",
                        release_root=directory,
                        score_verifier_repo=REPO,
                        evaluation_producer_repo=Path(directory) / "producer",
                        verification_repo=REPO,
                        localization_registry_path=REPO / "registry.json",
                        external_registry_path=REPO / "external.json",
                        population_registry_path=REPO / "populations.json",
                        source_root=REPO,
                        localization_postfreeze_amendment_path=(
                            REPO / "amendment.json"
                        ),
                    )

    def test_release_verifier_rejects_score_freeze_tamper_and_missing_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_id = "r"
            release = root / release_id
            freeze_payloads = {}
            freeze_paths = {}
            for build_id in ("A", "B"):
                freeze_path = (
                    release / f"build_{build_id}"
                    / "localization/fit/SCORE_FREEZE_MANIFEST.json"
                )
                freeze_path.parent.mkdir(parents=True)
                value = {"release_id": release_id, "build_id": build_id}
                value["payload_sha256"] = payload_sha256(value)
                freeze_path.write_text(json.dumps(value), encoding="utf-8")
                freeze_payloads[build_id] = value["payload_sha256"]
                freeze_paths[build_id] = freeze_path

            def write_certificate(path: Path, builds: dict) -> None:
                value = {
                    "release_id": release_id, "status": "PASS", "builds": builds,
                }
                value["certificate_sha256"] = payload_sha256(value)
                path.write_text(json.dumps(value), encoding="utf-8")

            builds = {
                build_id: {
                    "score_freeze_sha256": sha256_file(freeze_paths[build_id]),
                }
                for build_id in ("A", "B")
            }
            certificate_path = release / "score.json"
            write_certificate(certificate_path, builds)
            score_certificate = json.loads(certificate_path.read_text())
            derived = {
                build_id: DerivedLocalizationEvaluation(
                    files={}, manifest_core={
                        "score_freeze_payload_sha256": freeze_payloads[build_id],
                        "score_ab_certificate_sha256": score_certificate[
                            "certificate_sha256"
                        ],
                        "score_ab_certificate_file_sha256": sha256_file(
                            certificate_path
                        ),
                    },
                )
                for build_id in ("A", "B")
            }
            bindings = LEAV._verify_score_freeze_payload_bindings(
                release_root=root,
                release_id=release_id,
                score_ab_certificate_path=certificate_path,
                derived=derived,
            )
            self.assertNotEqual(
                bindings["A"]["score_freeze_payload_sha256"],
                bindings["B"]["score_freeze_payload_sha256"],
            )

            tampered = {"release_id": release_id, "build_id": "A", "tamper": True}
            tampered["payload_sha256"] = payload_sha256(tampered)
            freeze_paths["A"].write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not certificate-bound"):
                LEAV._verify_score_freeze_payload_bindings(
                    release_root=root,
                    release_id=release_id,
                    score_ab_certificate_path=certificate_path,
                    derived=derived,
                )

            freeze_paths["A"].write_text(json.dumps({
                "release_id": release_id,
                "build_id": "A",
                "payload_sha256": freeze_payloads["A"],
            }), encoding="utf-8")
            missing_path = release / "score-missing.json"
            write_certificate(missing_path, {"A": builds["A"]})
            missing_certificate = json.loads(missing_path.read_text())
            missing_derived = {
                build_id: DerivedLocalizationEvaluation(
                    files={}, manifest_core={
                        "score_freeze_payload_sha256": freeze_payloads[build_id],
                        "score_ab_certificate_sha256": missing_certificate[
                            "certificate_sha256"
                        ],
                        "score_ab_certificate_file_sha256": sha256_file(
                            missing_path
                        ),
                    },
                )
                for build_id in ("A", "B")
            }
            with self.assertRaisesRegex(RuntimeError, "lacks build B"):
                LEAV._verify_score_freeze_payload_bindings(
                    release_root=root,
                    release_id=release_id,
                    score_ab_certificate_path=missing_path,
                    derived=missing_derived,
                )

    def test_evaluation_ab_certificate_is_immutable_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "certificate.json"
            LP._write_immutable_certificate(path, b"one\n")
            LP._write_immutable_certificate(path, b"one\n")
            with self.assertRaisesRegex(FileExistsError, "already differs"):
                LP._write_immutable_certificate(path, b"two\n")
            self.assertEqual(path.read_bytes(), b"one\n")


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
