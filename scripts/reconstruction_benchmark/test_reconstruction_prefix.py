#!/usr/bin/env python3
"""Synthetic contract tests for the causal-prefix reconstruction lane."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.prefix_ab import (  # noqa: E402
    verify_prefix_preparation_ab,
    verify_prefix_score_ab,
)
from spectral_utils.reconstruction_benchmark.prefix_contract import (  # noqa: E402
    AtomicPrefixDirectory,
    BUDGETS,
    METHOD_IDS,
    PREPARATION_SCHEMA,
    PRIVATE_LABEL_SCHEMA,
    SCORE_AB_SCHEMA,
    SUBSETS,
    PrefixContractError,
    add_payload_sha256,
    load_registry,
    payload_sha256,
    validate_observation_arrays,
    write_json_noreplace,
)
from spectral_utils.reconstruction_benchmark.prefix_evaluation import (  # noqa: E402
    BOOTSTRAP_FILENAME,
    CONTRASTS_FILENAME,
    EVALUATION_MANIFEST_FILENAME,
    LABELED_SCORES_FILENAME,
    METRICS_FILENAME,
    _weighted_metric_draws,
    evaluate_prefix_build,
    evaluate_prefix_arrays,
    verify_prefix_evaluation_ab,
)
from spectral_utils.reconstruction_benchmark.prefix_fit import (  # noqa: E402
    SCORES_FILENAME,
    SCORE_MANIFEST_FILENAME,
    SCORE_SOURCE_FILES,
    run_prefix_methods,
)
from spectral_utils.reconstruction_benchmark.prefix_preparation import (  # noqa: E402
    EXPECTED_SCORE_FILENAME,
    FIT_INPUT_FILENAME,
    PREPARATION_MANIFEST_FILENAME,
    PRIVATE_LABEL_FILENAME,
    sanitize_source_row,
)


REGISTRY_PATH = REPO / "configs/reconstruction_benchmark_v1/prefix.json"


def _small_registry(*, bootstrap_draws: int = 20) -> dict:
    registry = deepcopy(load_registry(REGISTRY_PATH))
    population = registry["population"]
    population.update(
        {
            "expected_source_rows": 16,
            "expected_evaluation_traces": 16,
            "expected_incorrect": 8,
            "expected_correct": 8,
            "expected_prefix_observations": 96,
            "expected_complete_all_budgets": 16,
            "expected_evaluation_traces_by_subset": {family: 4 for family in SUBSETS},
            "expected_prefix_observations_by_budget": {str(budget): 16 for budget in BUDGETS},
        }
    )
    registry["evaluation"]["bootstrap"]["draws"] = bootstrap_draws
    registry["evaluation"]["bootstrap"]["seed"] = 41
    return registry


def _synthetic_payload() -> tuple[dict[str, np.ndarray], dict]:
    row_ids, families, budgets, labels = [], [], [], {}
    method_values = {method_id: [] for method_id in METHOD_IDS}
    for family_index, family in enumerate(SUBSETS):
        for trace_index in range(4):
            row_id = f"processbench@test::{family}::{family}-{trace_index}"
            label = trace_index % 2
            labels[row_id] = {
                "row_id": row_id,
                "group_id": row_id,
                "family": family,
                "label": label,
                "final_length": 700,
            }
            for budget in BUDGETS:
                row_ids.append(row_id)
                families.append(family)
                budgets.append(budget)
                # All methods preserve the same ordering but differ monotonically.
                method_values[METHOD_IDS[0]].append(label + trace_index * 0.01 + budget * 1e-6)
                method_values[METHOD_IDS[1]].append(label * 0.8 + trace_index * 0.01 + budget * 1e-6)
                method_values[METHOD_IDS[2]].append(label * 0.6 + trace_index * 0.01 + budget * 1e-6)
    arrays = {
        "row_id": np.asarray(row_ids),
        "family": np.asarray(families),
        "budget": np.asarray(budgets, dtype=np.int16),
        **{name: np.asarray(values, dtype=np.float64) for name, values in method_values.items()},
    }
    label_bundle = add_payload_sha256(
        {
            "schema_version": PRIVATE_LABEL_SCHEMA,
            "population_id": "processbench_llama31_prefix_v1",
            "positive_class": "completed trace has an incorrect final answer",
            "grouping_unit": "source question",
            "rows": sorted(labels.values(), key=lambda row: row["row_id"]),
        }
    )
    return arrays, label_bundle


class PrefixContractTests(unittest.TestCase):
    def test_atomic_directory_never_publishes_partial_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "fit"
            stage = AtomicPrefixDirectory(target)
            try:
                (stage.path / "partial.txt").write_text("partial", encoding="utf-8")
            finally:
                stage.cleanup()
            self.assertFalse(target.exists())
            self.assertFalse(stage.path.exists())

    def test_atomic_directory_preserves_raced_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "fit"
            stage = AtomicPrefixDirectory(target)
            (stage.path / "candidate").write_bytes(b"candidate")
            target.mkdir()
            (target / "incumbent").write_bytes(b"incumbent")
            try:
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    stage.commit()
                self.assertEqual((stage.path / "candidate").read_bytes(), b"candidate")
                self.assertEqual((target / "incumbent").read_bytes(), b"incumbent")
            finally:
                stage.cleanup()

    def test_certificate_publication_preserves_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "CERTIFICATE.json"
            target.write_bytes(b"incumbent\n")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_json_noreplace(target, {"candidate": True})
            self.assertEqual(target.read_bytes(), b"incumbent\n")

    def test_registry_forbids_signed_score_rebind_as_rerun(self) -> None:
        registry = deepcopy(load_registry(REGISTRY_PATH))
        registry["method_roster"][0]["execution_mode"] = "SIGNED_HISTORICAL_SCORE_REBIND"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prefix.json"
            path.write_text(json.dumps(registry), encoding="utf-8")
            with self.assertRaisesRegex(PrefixContractError, "rebinding"):
                load_registry(path)

    def test_sanitizer_drops_every_target_and_text_field(self) -> None:
        n = 24
        row = {
            "id": "gsm8k-7",
            "token_entropies": np.linspace(0.1, 1.0, n),
            "token_spilled_energies": np.linspace(1.0, 2.0, n),
            "token_logsumexp": np.linspace(2.0, 3.0, n),
            "top_k_logprobs": {
                "ids": np.tile(np.asarray([1, 2], dtype=np.int32), (n, 1)),
                "logprobs": np.column_stack(
                    (
                        np.linspace(-0.1, -0.2, n),
                        np.linspace(-1.0, -1.2, n),
                    )
                ),
            },
            "label": 3,
            "final_answer_correct": False,
            "first_error": 2,
            "problem": "secret question",
            "response": "secret answer",
            "steps": ["future"],
        }
        clean = sanitize_source_row(row, family="gsm8k")
        self.assertEqual(set(clean), set(load_registry(REGISTRY_PATH)["fit_visibility"]["allowed_fields"]))
        self.assertFalse(set(clean) & set(load_registry(REGISTRY_PATH)["fit_visibility"]["forbidden_fields"]))
        self.assertEqual(len(clean["token_entropies"]), n)

    def test_evaluation_is_budget_sliced_and_question_grouped(self) -> None:
        registry = _small_registry()
        arrays, labels = _synthetic_payload()
        validate_observation_arrays(arrays, registry=registry, include_scores=True)
        result = evaluate_prefix_arrays(
            score_arrays=arrays,
            label_bundle=labels,
            registry=registry,
        )
        self.assertEqual(len(result["metrics"]), 3 * 6 * 2)
        self.assertEqual(len(result["per_subset"]), 3 * 6 * 2 * 4)
        self.assertEqual(len(result["contrasts"]), 3 * 6 * 2)
        self.assertEqual(len(result["bootstrap_arrays"]), 80)
        for family in SUBSETS:
            counts = result["bootstrap_arrays"][f"group_count__{family}"]
            self.assertEqual(counts.shape, (20, 4))
            np.testing.assert_array_equal(counts.sum(axis=1), np.full(20, 4))
        self.assertTrue(all(row["budget"] in BUDGETS for row in result["metrics"]))
        self.assertTrue(all(row["aggregation"] == "equal_subset_macro" for row in result["metrics"]))
        self.assertTrue(all(row["bootstrap_draws"] == 20 for row in result["metrics"]))
        self.assertTrue(all(row["point"] == 1.0 for row in result["metrics"]))

    def test_evaluation_rejects_completed_trace_at_budget(self) -> None:
        registry = _small_registry()
        arrays, labels = _synthetic_payload()
        victim = arrays["row_id"][0]
        for row in labels["rows"]:
            if row["row_id"] == victim:
                row["final_length"] = int(arrays["budget"][0])
        labels.pop("payload_sha256")
        labels = add_payload_sha256(labels)
        with self.assertRaisesRegex(PrefixContractError, "strict unfinished"):
            evaluate_prefix_arrays(score_arrays=arrays, label_bundle=labels, registry=registry)

    def test_equal_subset_macro_blocks_single_class_registered_subset(self) -> None:
        registry = _small_registry()
        arrays, labels = _synthetic_payload()
        keep = np.ones(len(arrays["row_id"]), dtype=bool)
        gsm_at_512 = np.flatnonzero(
            (arrays["family"] == "gsm8k") & (arrays["budget"] == 512)
        )
        # Keep exactly the positive GSM8K source question at budget 512.
        positive_id = next(
            row["row_id"]
            for row in labels["rows"]
            if row["family"] == "gsm8k" and row["label"] == 1
        )
        for index in gsm_at_512:
            keep[index] = arrays["row_id"][index] == positive_id
        arrays = {name: np.asarray(values)[keep] for name, values in arrays.items()}
        registry["population"].update(
            {
                "expected_prefix_observations": 93,
                "expected_complete_all_budgets": 13,
                "expected_prefix_observations_by_budget": {
                    **registry["population"]["expected_prefix_observations_by_budget"],
                    "512": 13,
                },
            }
        )
        result = evaluate_prefix_arrays(
            score_arrays=arrays, label_bundle=labels, registry=registry
        )
        macro = [row for row in result["metrics"] if row["budget"] == 512]
        self.assertTrue(all(row["point"] is None for row in macro))
        self.assertTrue(all(row["families_used"] == 3 for row in macro))
        self.assertTrue(all(row["families_excluded"] == ["gsm8k"] for row in macro))
        self.assertTrue(
            all(
                row["status"] == "METRIC_UNDEFINED_MISSING_REGISTERED_SUBSET"
                for row in macro
            )
        )
        gsm = [
            row
            for row in result["per_subset"]
            if row["budget"] == 512 and row["family"] == "gsm8k"
        ]
        self.assertTrue(all(row["n_traces"] == row["n_positive"] == 1 for row in gsm))
        self.assertTrue(all(row["n_negative"] == 0 for row in gsm))
        self.assertTrue(
            all(
                np.isnan(values).all()
                for name, values in result["bootstrap_arrays"].items()
                if name.startswith("metric__") and "__b512__" in name
            )
        )

    def test_weighted_bootstrap_metrics_equal_duplicate_row_expansion(self) -> None:
        from sklearn.metrics import average_precision_score, roc_auc_score

        labels = np.asarray([0, 1, 0, 1, 1], dtype=np.int8)
        scores = np.asarray([0.1, 0.8, 0.1, 0.4, 0.8], dtype=np.float64)
        weights = np.asarray(
            [
                [1, 1, 1, 1, 1],
                [2, 0, 1, 3, 1],
                [0, 2, 4, 1, 0],
            ],
            dtype=np.int64,
        )
        for metric, scorer in (
            ("auroc", roc_auc_score),
            ("auprc", average_precision_score),
        ):
            observed = _weighted_metric_draws(
                metric=metric,
                labels=labels,
                scores=scores,
                weights=weights,
            )
            expected = []
            for draw in weights:
                expanded_labels = np.repeat(labels, draw)
                expanded_scores = np.repeat(scores, draw)
                expected.append(float(scorer(expanded_labels, expanded_scores)))
            np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-15)


class PrefixPreparationABTests(unittest.TestCase):
    def _write_registry(self, root: Path, registry: dict) -> Path:
        path = root / "prefix.json"
        path.write_text(json.dumps(registry, sort_keys=True), encoding="utf-8")
        return path

    def _replace_fit_payload(
        self,
        *,
        release_root: Path,
        release_id: str,
        build_id: str,
        payload: bytes,
    ) -> None:
        root = release_root / release_id / "prefix" / build_id
        fit_path = root / "inputs" / FIT_INPUT_FILENAME
        fit_sha = atomic_write_bytes(fit_path, payload)
        manifest_path = root / PREPARATION_MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["fit_input"]["sha256"] = fit_sha
        manifest["fit_input"]["size_bytes"] = len(payload)
        manifest.pop("payload_sha256")
        atomic_write_json(manifest_path, add_payload_sha256(manifest))

    def _write_build(
        self,
        *,
        release_root: Path,
        private_root: Path,
        release_id: str,
        build_id: str,
        registry_path: Path,
        arrays: dict[str, np.ndarray],
        labels: dict,
    ) -> dict:
        root = release_root / release_id / "prefix" / build_id
        inputs = root / "inputs"
        private = private_root / release_id / "prefix" / build_id
        inputs.mkdir(parents=True)
        private.mkdir(parents=True)
        rows_by_family = {}
        for family in SUBSETS:
            rows_by_family[family] = []
            for label_row in labels["rows"]:
                if label_row["family"] != family:
                    continue
                n = int(label_row["final_length"])
                rows_by_family[family].append(
                    {
                        "row_id": label_row["row_id"],
                        "source_question_id": label_row["row_id"],
                        "family": family,
                        "partition": "evaluation",
                        "token_entropies": np.linspace(0.1, 1.0, n),
                        "token_spilled_energies": np.linspace(0.2, 1.2, n),
                        "token_logsumexp": np.linspace(1.0, 2.0, n),
                        "top_k_logprobs": {
                            "ids": np.tile(np.asarray([1, 2], dtype=np.int32), (n, 1)),
                            "logprobs": np.column_stack(
                                (
                                    np.linspace(-0.1, -0.2, n),
                                    np.linspace(-1.0, -1.2, n),
                                )
                            ),
                        },
                    }
                )
        model_audit = {"synthetic_target_free_model": True}
        fit_input = {
            "schema_version": "reconstruction-prefix-fit-input-v1",
            "lane_id": "processbench_llama31_prefix_v1",
            "task_id": "causal_early_final_answer_error_detection",
            "population_id": "processbench_llama31_prefix_v1",
            "budgets": BUDGETS,
            "method_ids": METHOD_IDS,
            "rows_by_family": rows_by_family,
            "frozen_models": {
                "unified28": "synthetic-unified-model",
                "iu28_no_length": "synthetic-iu-models",
            },
            "model_audit": model_audit,
            "target_fields_present": False,
            "claim_boundary": load_registry(REGISTRY_PATH)["claim_boundary"],
        }
        fit_payload = pickle.dumps(fit_input, protocol=5)
        fit_sha = atomic_write_bytes(inputs / FIT_INPUT_FILENAME, fit_payload)
        score_sha = atomic_write_npz(inputs / EXPECTED_SCORE_FILENAME, arrays)
        label_sha = atomic_write_json(private / PRIVATE_LABEL_FILENAME, labels)
        source_binding = {
            "source_root": "/read-only/source",
            "registry": {
                "path": str(registry_path),
                "sha256": sha256_file(registry_path),
            },
            "assets": [],
            "asset_roster_sha256": payload_sha256([]),
        }
        manifest = add_payload_sha256(
            {
                "schema_version": PREPARATION_SCHEMA,
                "release_id": release_id,
                "build_id": build_id,
                "scientific_full_build": True,
                "lane_id": "processbench_llama31_prefix_v1",
                "task_id": "causal_early_final_answer_error_detection",
                "population_id": "processbench_llama31_prefix_v1",
                "source_binding": source_binding,
                "source_binding_sha256": payload_sha256(source_binding),
                "fit_input": {
                    "path": f"inputs/{FIT_INPUT_FILENAME}",
                    "sha256": fit_sha,
                    "size_bytes": len(fit_payload),
                    "target_fields_present": False,
                },
                "expected_scores": {
                    "path": f"inputs/{EXPECTED_SCORE_FILENAME}",
                    "sha256": score_sha,
                    "observations": len(arrays["row_id"]),
                    "labels_present": False,
                    "use": "post-recomputation score anchor only",
                },
                "private_labels": {
                    "path": str(private / PRIVATE_LABEL_FILENAME),
                    "sha256": label_sha,
                    "rows": len(labels["rows"]),
                    "fit_visible": False,
                },
                "fit_model_audit_sha256": payload_sha256(model_audit),
                "execution_modes": {
                    row["method_id"]: row["execution_mode"]
                    for row in load_registry(REGISTRY_PATH)["method_roster"]
                },
                "labels_opened_by_preparation_controller": True,
                "labels_exposed_to_fit": False,
                "historical_scores_are_execution_substitute": False,
                "claim_boundary": load_registry(REGISTRY_PATH)["claim_boundary"],
            }
        )
        atomic_write_json(root / PREPARATION_MANIFEST_FILENAME, manifest)
        reconstruction = {
            "registry": json.loads(registry_path.read_text(encoding="utf-8")),
            "source_binding": source_binding,
            "fit_input": fit_input,
            "fit_input_bytes": (inputs / FIT_INPUT_FILENAME).read_bytes(),
            "fit_input_sha256": fit_sha,
            "expected_scores": arrays,
            "expected_scores_bytes": (inputs / EXPECTED_SCORE_FILENAME).read_bytes(),
            "expected_scores_sha256": score_sha,
            "private_labels": labels,
            "private_labels_bytes": (private / PRIVATE_LABEL_FILENAME).read_bytes(),
            "private_labels_sha256": label_sha,
            "model_audit_sha256": payload_sha256(model_audit),
        }
        self._synthetic_reconstruction = reconstruction
        return reconstruction

    def _verify_preparation(self, **kwargs: object) -> dict:
        with self._reconstruction_patch():
            return verify_prefix_preparation_ab(
                repo=REPO,
                source_root=Path("/registered/source"),
                **kwargs,
            )

    def _reconstruction_patch(self):
        return patch(
            "spectral_utils.reconstruction_benchmark.prefix_ab."
            "reconstruct_prefix_preparation",
            return_value=self._synthetic_reconstruction,
        )

    def _write_score_build(
        self,
        *,
        release_root: Path,
        release_id: str,
        build_id: str,
        arrays: dict[str, np.ndarray],
    ) -> dict:
        lane_root = release_root / release_id / "prefix"
        root = lane_root / build_id
        preparation_path = root / PREPARATION_MANIFEST_FILENAME
        preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
        prep_certificate_path = lane_root / "PREPARATION_AB_VERIFICATION.json"
        prep_certificate = json.loads(prep_certificate_path.read_text(encoding="utf-8"))
        fit_root = root / "fit"
        fit_root.mkdir()
        score_sha = atomic_write_npz(fit_root / SCORES_FILENAME, arrays)
        source_snapshot = [
            {"path": relative, "sha256": sha256_file(REPO / relative)}
            for relative in SCORE_SOURCE_FILES
        ]
        anchors = {
            method_id: {
                "execution_mode": load_registry(REGISTRY_PATH)["method_roster"][index]["execution_mode"],
                "observations": len(arrays["row_id"]),
                "max_abs_score_difference": 0.0,
                "exact_float_identity": True,
                "absolute_tolerance": float(
                    load_registry(REGISTRY_PATH)["score_anchor"]["absolute_tolerance"]
                ),
                "status": "CPU_RECOMPUTED_AND_ANCHOR_VERIFIED",
                "historical_score_rebind": False,
            }
            for index, method_id in enumerate(METHOD_IDS)
        }
        recomputation_audit = {"anchors": anchors, "step272_fit": {}}
        manifest = add_payload_sha256(
            {
                "schema_version": "reconstruction-prefix-score-freeze-v1",
                "release_id": release_id,
                "build_id": build_id,
                "scientific_full_build": True,
                "lane_id": "processbench_llama31_prefix_v1",
                "task_id": "causal_early_final_answer_error_detection",
                "preparation_manifest_sha256": sha256_file(preparation_path),
                "preparation_ab_certificate_sha256": sha256_file(prep_certificate_path),
                "preparation_ab_certificate_payload_sha256": prep_certificate["payload_sha256"],
                "fit_input_sha256": preparation["fit_input"]["sha256"],
                "expected_score_anchor_sha256": preparation["expected_scores"]["sha256"],
                "score_artifact": {
                    "path": SCORES_FILENAME,
                    "sha256": score_sha,
                    "observations": len(arrays["row_id"]),
                    "method_scores": len(arrays["row_id"]) * len(METHOD_IDS),
                },
                "recomputation_audit": recomputation_audit,
                "fit_visible_targets": False,
                "future_tokens_used_for_scored_trace": False,
                "historical_scores_are_execution_substitute": False,
                "execution_status": "CPU_RECOMPUTED_AND_ANCHOR_VERIFIED",
                "claim_boundary": load_registry(REGISTRY_PATH)["claim_boundary"],
                "source_snapshot": source_snapshot,
                "source_snapshot_sha256": payload_sha256(source_snapshot),
            }
        )
        atomic_write_json(fit_root / SCORE_MANIFEST_FILENAME, manifest)
        return recomputation_audit

    def _forge_coordinated_label_swap_chain(
        self,
        *,
        release_root: Path,
        private_root: Path,
        release_id: str,
    ) -> dict:
        """Reproduce the concrete reviewer attack against self-hashed prep certs."""

        lane_root = release_root / release_id / "prefix"
        certificate_path = lane_root / "PREPARATION_AB_VERIFICATION.json"
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        swapped_sha = None
        for build_id in ("A", "B"):
            private_path = (
                private_root / release_id / "prefix" / build_id / PRIVATE_LABEL_FILENAME
            )
            labels = json.loads(private_path.read_text(encoding="utf-8"))
            first = int(labels["rows"][0]["label"])
            opposite = next(
                index
                for index, row in enumerate(labels["rows"][1:], start=1)
                if int(row["label"]) != first
            )
            labels["rows"][0]["label"], labels["rows"][opposite]["label"] = (
                labels["rows"][opposite]["label"],
                labels["rows"][0]["label"],
            )
            labels.pop("payload_sha256")
            swapped_sha = atomic_write_json(private_path, add_payload_sha256(labels))
            manifest_path = lane_root / build_id / PREPARATION_MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["private_labels"]["sha256"] = swapped_sha
            manifest.pop("payload_sha256")
            manifest = add_payload_sha256(manifest)
            atomic_write_json(manifest_path, manifest)
            certificate["builds"][build_id] = {
                "preparation_manifest_sha256": sha256_file(manifest_path),
                "preparation_manifest_payload_sha256": manifest["payload_sha256"],
            }
        assert swapped_sha is not None
        certificate["private_label_sha256"] = swapped_sha
        certificate.pop("payload_sha256")
        certificate = add_payload_sha256(certificate)
        atomic_write_json(certificate_path, certificate)
        return certificate

    def _write_evaluation_build(
        self,
        *,
        release_root: Path,
        private_root: Path,
        release_id: str,
        build_id: str,
        arrays: dict[str, np.ndarray],
    ) -> None:
        lane_root = release_root / release_id / "prefix"
        build_root = lane_root / build_id
        evaluation_root = build_root / "evaluation"
        evaluation_root.mkdir()
        labels = json.loads(
            (private_root / release_id / "prefix" / build_id / PRIVATE_LABEL_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        label_by_id = {row["row_id"]: row["label"] for row in labels["rows"]}
        labeled = {
            **arrays,
            "label": np.asarray([label_by_id[str(value)] for value in arrays["row_id"]], dtype=np.int8),
        }
        labeled_sha = atomic_write_npz(evaluation_root / LABELED_SCORES_FILENAME, labeled)
        draws_sha = atomic_write_npz(
            evaluation_root / BOOTSTRAP_FILENAME,
            {"synthetic_linked_draw": np.zeros(2000, dtype=np.float64)},
        )
        metric_rows = [
            {"method_id": METHOD_IDS[0], "budget": budget, "metric": "auroc", "point": 1.0}
            for budget in BUDGETS
        ]
        per_subset_rows = [
            {
                "method_id": METHOD_IDS[0],
                "budget": budget,
                "metric": "auroc",
                "family": family,
                "point": 1.0,
            }
            for budget in BUDGETS
            for family in SUBSETS
        ]
        contrast_rows = [
            {
                "left_method_id": METHOD_IDS[0],
                "right_method_id": METHOD_IDS[1],
                "budget": budget,
                "metric": "auroc",
                "point_delta": 0.0,
            }
            for budget in BUDGETS
        ]
        metrics = add_payload_sha256(
            {
                "schema_version": "reconstruction-prefix-metrics-v1",
                "rows": metric_rows,
                "per_subset_rows": per_subset_rows,
            }
        )
        contrasts = add_payload_sha256(
            {
                "schema_version": "reconstruction-prefix-contrasts-v1",
                "rows": contrast_rows,
            }
        )
        metrics_sha = atomic_write_json(evaluation_root / METRICS_FILENAME, metrics)
        contrasts_sha = atomic_write_json(evaluation_root / CONTRASTS_FILENAME, contrasts)
        score_certificate_path = lane_root / "SCORE_AB_VERIFICATION.json"
        score_manifest_path = build_root / "fit" / SCORE_MANIFEST_FILENAME
        private_path = private_root / release_id / "prefix" / build_id / PRIVATE_LABEL_FILENAME
        manifest = add_payload_sha256(
            {
                "schema_version": "reconstruction-prefix-evaluation-v1",
                "release_id": release_id,
                "build_id": build_id,
                "scientific_full_build": True,
                "lane_id": "processbench_llama31_prefix_v1",
                "task_id": "causal_early_final_answer_error_detection",
                "score_ab_certificate_sha256": sha256_file(score_certificate_path),
                "score_manifest_sha256": sha256_file(score_manifest_path),
                "private_label_sha256": sha256_file(private_path),
                "artifacts": {
                    LABELED_SCORES_FILENAME: labeled_sha,
                    METRICS_FILENAME: metrics_sha,
                    CONTRASTS_FILENAME: contrasts_sha,
                    BOOTSTRAP_FILENAME: draws_sha,
                },
                "metric_rows": len(metric_rows),
                "per_subset_rows": len(per_subset_rows),
                "contrast_rows": len(contrast_rows),
                "bootstrap_draws": 2000,
                "labels_opened_after_score_ab": True,
                "causal_early_scoring_only": True,
                "stopping_claim_allowed": False,
                "cross_budget_macro_allowed": False,
                "cross_task_macro_allowed": False,
                "claim_boundary": load_registry(REGISTRY_PATH)["claim_boundary"],
            }
        )
        atomic_write_json(evaluation_root / EVALUATION_MANIFEST_FILENAME, manifest)

    def _certify_score_chain(
        self,
        *,
        registry_path: Path,
        release_root: Path,
        private_root: Path,
        release_id: str,
        arrays: dict[str, np.ndarray],
        labels: dict,
    ) -> tuple[dict, dict]:
        for build_id in ("A", "B"):
            self._write_build(
                release_root=release_root,
                private_root=private_root,
                release_id=release_id,
                build_id=build_id,
                registry_path=registry_path,
                arrays=arrays,
                labels=labels,
            )
        preparation_certificate = self._verify_preparation(
            registry_path=registry_path,
            release_root=release_root,
            private_root=private_root,
            release_id=release_id,
            require_scientific_full=True,
        )
        audit = None
        for build_id in ("A", "B"):
            audit = self._write_score_build(
                release_root=release_root,
                release_id=release_id,
                build_id=build_id,
                arrays=arrays,
            )
        assert audit is not None
        self._synthetic_score_audit = audit
        with self._reconstruction_patch(), patch(
            "spectral_utils.reconstruction_benchmark.prefix_ab.recompute_prefix_scores",
            return_value=(arrays, audit),
        ):
            score_certificate = verify_prefix_score_ab(
                repo=REPO,
                registry_path=registry_path,
                release_root=release_root,
                private_root=private_root,
                release_id=release_id,
                source_root=Path("/registered/source"),
                require_scientific_full=True,
            )
        return preparation_certificate, score_certificate

    def test_preparation_ab_accepts_identical_core_and_rejects_private_drift(self) -> None:
        registry = _small_registry(bootstrap_draws=2000)
        arrays, labels = _synthetic_payload()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = self._write_registry(root, registry)
            release_root, private_root = root / "releases", root / "private"
            for build_id in ("A", "B"):
                self._write_build(
                    release_root=release_root,
                    private_root=private_root,
                    release_id="ok",
                    build_id=build_id,
                    registry_path=registry_path,
                    arrays=arrays,
                    labels=labels,
                )
            certificate = self._verify_preparation(
                registry_path=registry_path,
                release_root=release_root,
                private_root=private_root,
                release_id="ok",
                require_scientific_full=True,
            )
            self.assertEqual(certificate["status"], "PASS")
            self.assertFalse(certificate["labels_exposed_to_fit"])

            audit = None
            for build_id in ("A", "B"):
                audit = self._write_score_build(
                    release_root=release_root,
                    release_id="ok",
                    build_id=build_id,
                    arrays=arrays,
                )
            assert audit is not None
            with self._reconstruction_patch(), patch(
                "spectral_utils.reconstruction_benchmark.prefix_ab.recompute_prefix_scores",
                return_value=(arrays, audit),
            ):
                score_certificate = verify_prefix_score_ab(
                    repo=REPO,
                    registry_path=registry_path,
                    release_root=release_root,
                    private_root=private_root,
                    release_id="ok",
                    source_root=Path("/registered/source"),
                    require_scientific_full=True,
                )
            self.assertEqual(score_certificate["status"], "PASS")
            self.assertFalse(score_certificate["stopping_claim_allowed"])
            with self._reconstruction_patch(), patch(
                "spectral_utils.reconstruction_benchmark.prefix_ab.recompute_prefix_scores",
                return_value=(arrays, audit),
            ):
                for build_id in ("A", "B"):
                    evaluate_prefix_build(
                        repo=REPO,
                        registry_path=registry_path,
                        release_root=release_root,
                        private_root=private_root,
                        release_id="ok",
                        build_id=build_id,
                        source_root=Path("/registered/source"),
                        scientific_full=True,
                    )
                evaluation_certificate = verify_prefix_evaluation_ab(
                    repo=REPO,
                    registry_path=registry_path,
                    release_root=release_root,
                    private_root=private_root,
                    release_id="ok",
                    source_root=Path("/registered/source"),
                    require_scientific_full=True,
                )
            self.assertEqual(evaluation_certificate["status"], "PASS")
            self.assertEqual(evaluation_certificate["bootstrap_draws"], 2000)
            self.assertFalse(evaluation_certificate["cross_budget_macro_allowed"])

            drift = deepcopy(labels)
            first = drift["rows"][0]["label"]
            swap_index = next(
                index for index, row in enumerate(drift["rows"][1:], start=1)
                if row["label"] != first
            )
            drift["rows"][0]["label"], drift["rows"][swap_index]["label"] = (
                drift["rows"][swap_index]["label"],
                drift["rows"][0]["label"],
            )
            drift.pop("payload_sha256")
            drift = add_payload_sha256(drift)
            self._write_build(
                release_root=release_root,
                private_root=private_root,
                release_id="drift",
                build_id="A",
                registry_path=registry_path,
                arrays=arrays,
                labels=labels,
            )
            self._write_build(
                release_root=release_root,
                private_root=private_root,
                release_id="drift",
                build_id="B",
                registry_path=registry_path,
                arrays=arrays,
                labels=drift,
            )
            with self.assertRaises(PrefixContractError):
                self._verify_preparation(
                    registry_path=registry_path,
                    release_root=release_root,
                    private_root=private_root,
                    release_id="drift",
                    require_scientific_full=True,
                )

    def test_preparation_ab_rejects_coordinated_invalid_pickle(self) -> None:
        registry = _small_registry(bootstrap_draws=2000)
        arrays, labels = _synthetic_payload()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = self._write_registry(root, registry)
            release_root, private_root = root / "releases", root / "private"
            for build_id in ("A", "B"):
                self._write_build(
                    release_root=release_root,
                    private_root=private_root,
                    release_id="invalid-pickle",
                    build_id=build_id,
                    registry_path=registry_path,
                    arrays=arrays,
                    labels=labels,
                )
                self._replace_fit_payload(
                    release_root=release_root,
                    release_id="invalid-pickle",
                    build_id=build_id,
                    payload=b"coordinated-but-not-a-pickle",
                )
            with self.assertRaises(PrefixContractError):
                self._verify_preparation(
                    registry_path=registry_path,
                    release_root=release_root,
                    private_root=private_root,
                    release_id="invalid-pickle",
                    require_scientific_full=True,
                )

    def test_preparation_ab_rejects_coordinated_synthetic_source(self) -> None:
        registry = _small_registry(bootstrap_draws=2000)
        arrays, labels = _synthetic_payload()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = self._write_registry(root, registry)
            release_root, private_root = root / "releases", root / "private"
            for build_id in ("A", "B"):
                self._write_build(
                    release_root=release_root,
                    private_root=private_root,
                    release_id="synthetic-source",
                    build_id=build_id,
                    registry_path=registry_path,
                    arrays=arrays,
                    labels=labels,
                )
            with self.assertRaisesRegex(PrefixContractError, "canonical frozen registry"):
                verify_prefix_preparation_ab(
                    repo=REPO,
                    source_root=root / "coordinated-fake-source",
                    registry_path=registry_path,
                    release_root=release_root,
                    private_root=private_root,
                    release_id="synthetic-source",
                    require_scientific_full=True,
                )

    def test_preparation_ab_rejects_forbidden_target_inside_fit_pickle(self) -> None:
        registry = _small_registry(bootstrap_draws=2000)
        arrays, labels = _synthetic_payload()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = self._write_registry(root, registry)
            release_root, private_root = root / "releases", root / "private"
            for build_id in ("A", "B"):
                self._write_build(
                    release_root=release_root,
                    private_root=private_root,
                    release_id="target-leak",
                    build_id=build_id,
                    registry_path=registry_path,
                    arrays=arrays,
                    labels=labels,
                )
                fit_path = (
                    release_root
                    / "target-leak"
                    / "prefix"
                    / build_id
                    / "inputs"
                    / FIT_INPUT_FILENAME
                )
                with fit_path.open("rb") as handle:
                    fit_input = pickle.load(handle)
                fit_input["rows_by_family"]["gsm8k"][0]["label"] = 1
                self._replace_fit_payload(
                    release_root=release_root,
                    release_id="target-leak",
                    build_id=build_id,
                    payload=pickle.dumps(fit_input, protocol=5),
                )
            with self.assertRaises(PrefixContractError):
                self._verify_preparation(
                    registry_path=registry_path,
                    release_root=release_root,
                    private_root=private_root,
                    release_id="target-leak",
                    require_scientific_full=True,
                )

    def test_executor_rejects_coordinated_post_certificate_preparation_replacement(self) -> None:
        registry = _small_registry(bootstrap_draws=2000)
        arrays, labels = _synthetic_payload()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = self._write_registry(root, registry)
            release_root, private_root = root / "releases", root / "private"
            for build_id in ("A", "B"):
                self._write_build(
                    release_root=release_root,
                    private_root=private_root,
                    release_id="prep-replaced",
                    build_id=build_id,
                    registry_path=registry_path,
                    arrays=arrays,
                    labels=labels,
                )
            self._verify_preparation(
                registry_path=registry_path,
                release_root=release_root,
                private_root=private_root,
                release_id="prep-replaced",
                require_scientific_full=True,
            )
            manifest_path = (
                release_root
                / "prep-replaced/prefix/A"
                / PREPARATION_MANIFEST_FILENAME
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["fit_model_audit_sha256"] = "f" * 64
            manifest.pop("payload_sha256")
            atomic_write_json(manifest_path, add_payload_sha256(manifest))
            with self._reconstruction_patch(), self.assertRaises(PrefixContractError):
                run_prefix_methods(
                    repo=REPO,
                    registry_path=registry_path,
                    release_root=release_root,
                    private_root=private_root,
                    release_id="prep-replaced",
                    build_id="A",
                    source_root=Path("/registered/source"),
                    scientific_full=True,
                )

    def test_transitive_chain_rejects_self_hashed_opposite_class_label_swap(self) -> None:
        registry = _small_registry(bootstrap_draws=2000)
        arrays, labels = _synthetic_payload()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = self._write_registry(root, registry)
            release_root, private_root = root / "releases", root / "private"
            release_id = "forged-label-chain"
            for build_id in ("A", "B"):
                self._write_build(
                    release_root=release_root,
                    private_root=private_root,
                    release_id=release_id,
                    build_id=build_id,
                    registry_path=registry_path,
                    arrays=arrays,
                    labels=labels,
                )
            self._verify_preparation(
                registry_path=registry_path,
                release_root=release_root,
                private_root=private_root,
                release_id=release_id,
                require_scientific_full=True,
            )
            self._forge_coordinated_label_swap_chain(
                release_root=release_root,
                private_root=private_root,
                release_id=release_id,
            )

            with self._reconstruction_patch(), patch(
                "spectral_utils.reconstruction_benchmark.prefix_fit.recompute_prefix_scores"
            ) as executor_recompute, self.assertRaisesRegex(
                PrefixContractError, "provenance binding"
            ):
                run_prefix_methods(
                    repo=REPO,
                    registry_path=registry_path,
                    release_root=release_root,
                    private_root=private_root,
                    release_id=release_id,
                    build_id="A",
                    source_root=Path("/registered/source"),
                    scientific_full=True,
                )
            executor_recompute.assert_not_called()

            audit = None
            for build_id in ("A", "B"):
                audit = self._write_score_build(
                    release_root=release_root,
                    release_id=release_id,
                    build_id=build_id,
                    arrays=arrays,
                )
            assert audit is not None
            lane_root = release_root / release_id / "prefix"
            atomic_write_json(
                lane_root / "SCORE_AB_VERIFICATION.json",
                add_payload_sha256(
                    {
                        "schema_version": SCORE_AB_SCHEMA,
                        "release_id": release_id,
                        "status": "PASS",
                        "self_attested_forgery": True,
                    }
                ),
            )

            with self._reconstruction_patch(), patch(
                "spectral_utils.reconstruction_benchmark.prefix_ab.recompute_prefix_scores"
            ) as verifier_recompute, self.assertRaisesRegex(
                PrefixContractError, "provenance binding"
            ):
                verify_prefix_score_ab(
                    repo=REPO,
                    registry_path=registry_path,
                    release_root=release_root,
                    private_root=private_root,
                    release_id=release_id,
                    source_root=Path("/registered/source"),
                    require_scientific_full=True,
                )
            verifier_recompute.assert_not_called()

            for gate in ("executor", "verifier"):
                with self.subTest(evaluation_gate=gate), self._reconstruction_patch(), patch(
                    "spectral_utils.reconstruction_benchmark.prefix_ab.recompute_prefix_scores"
                ) as evaluation_recompute, patch(
                    "spectral_utils.reconstruction_benchmark.prefix_evaluation.load_private_labels"
                ) as private_loader, self.assertRaisesRegex(
                    PrefixContractError, "provenance binding"
                ):
                    if gate == "executor":
                        evaluate_prefix_build(
                            repo=REPO,
                            registry_path=registry_path,
                            release_root=release_root,
                            private_root=private_root,
                            release_id=release_id,
                            build_id="A",
                            source_root=Path("/registered/source"),
                            scientific_full=True,
                        )
                    else:
                        verify_prefix_evaluation_ab(
                            repo=REPO,
                            registry_path=registry_path,
                            release_root=release_root,
                            private_root=private_root,
                            release_id=release_id,
                            source_root=Path("/registered/source"),
                            require_scientific_full=True,
                        )
                evaluation_recompute.assert_not_called()
                private_loader.assert_not_called()

    def test_score_ab_rejects_fabricated_identical_outputs(self) -> None:
        registry = _small_registry(bootstrap_draws=2000)
        arrays, labels = _synthetic_payload()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = self._write_registry(root, registry)
            release_root, private_root = root / "releases", root / "private"
            for build_id in ("A", "B"):
                self._write_build(
                    release_root=release_root,
                    private_root=private_root,
                    release_id="fabricated-score",
                    build_id=build_id,
                    registry_path=registry_path,
                    arrays=arrays,
                    labels=labels,
                )
            self._verify_preparation(
                registry_path=registry_path,
                release_root=release_root,
                private_root=private_root,
                release_id="fabricated-score",
                require_scientific_full=True,
            )
            audit = None
            for build_id in ("A", "B"):
                audit = self._write_score_build(
                    release_root=release_root,
                    release_id="fabricated-score",
                    build_id=build_id,
                    arrays=arrays,
                )
            assert audit is not None
            recomputed = {name: np.array(values, copy=True) for name, values in arrays.items()}
            recomputed[METHOD_IDS[0]] = recomputed[METHOD_IDS[0]] + 0.25
            with self._reconstruction_patch(), patch(
                "spectral_utils.reconstruction_benchmark.prefix_ab.recompute_prefix_scores",
                return_value=(recomputed, audit),
            ):
                with self.assertRaisesRegex(PrefixContractError, "independent CPU recomputation"):
                    verify_prefix_score_ab(
                        repo=REPO,
                        registry_path=registry_path,
                        release_root=release_root,
                        private_root=private_root,
                        release_id="fabricated-score",
                        source_root=Path("/registered/source"),
                        require_scientific_full=True,
                    )

    def test_evaluation_ab_rejects_minimal_identical_fabrication(self) -> None:
        registry = _small_registry(bootstrap_draws=2000)
        arrays, labels = _synthetic_payload()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = self._write_registry(root, registry)
            release_root, private_root = root / "releases", root / "private"
            self._certify_score_chain(
                registry_path=registry_path,
                release_root=release_root,
                private_root=private_root,
                release_id="fabricated-evaluation",
                arrays=arrays,
                labels=labels,
            )
            for build_id in ("A", "B"):
                self._write_evaluation_build(
                    release_root=release_root,
                    private_root=private_root,
                    release_id="fabricated-evaluation",
                    build_id=build_id,
                    arrays=arrays,
                )
            with self._reconstruction_patch(), patch(
                "spectral_utils.reconstruction_benchmark.prefix_ab.recompute_prefix_scores",
                return_value=(arrays, self._synthetic_score_audit),
            ), self.assertRaises(PrefixContractError):
                verify_prefix_evaluation_ab(
                    repo=REPO,
                    registry_path=registry_path,
                    release_root=release_root,
                    private_root=private_root,
                    release_id="fabricated-evaluation",
                    source_root=Path("/registered/source"),
                    require_scientific_full=True,
                )

    def test_evaluation_does_not_open_labels_after_score_manifest_replacement(self) -> None:
        registry = _small_registry(bootstrap_draws=2000)
        arrays, labels = _synthetic_payload()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = self._write_registry(root, registry)
            release_root, private_root = root / "releases", root / "private"
            self._certify_score_chain(
                registry_path=registry_path,
                release_root=release_root,
                private_root=private_root,
                release_id="score-replaced",
                arrays=arrays,
                labels=labels,
            )
            manifest_path = (
                release_root
                / "score-replaced/prefix/A/fit"
                / SCORE_MANIFEST_FILENAME
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["scientific_full_build"] = False
            manifest.pop("payload_sha256")
            atomic_write_json(manifest_path, add_payload_sha256(manifest))
            with self._reconstruction_patch(), patch(
                "spectral_utils.reconstruction_benchmark.prefix_ab.recompute_prefix_scores",
                return_value=(arrays, self._synthetic_score_audit),
            ), patch(
                "spectral_utils.reconstruction_benchmark.prefix_evaluation.load_private_labels"
            ) as private_loader:
                with self.assertRaisesRegex(PrefixContractError, "scientific-full"):
                    evaluate_prefix_build(
                        repo=REPO,
                        registry_path=registry_path,
                        release_root=release_root,
                        private_root=private_root,
                        release_id="score-replaced",
                        build_id="A",
                        source_root=Path("/registered/source"),
                        scientific_full=True,
                    )
                private_loader.assert_not_called()

    def test_evaluation_ab_rejects_stale_evaluator_source_snapshot(self) -> None:
        registry = _small_registry(bootstrap_draws=2000)
        arrays, labels = _synthetic_payload()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = self._write_registry(root, registry)
            release_root, private_root = root / "releases", root / "private"
            self._certify_score_chain(
                registry_path=registry_path,
                release_root=release_root,
                private_root=private_root,
                release_id="stale-evaluator",
                arrays=arrays,
                labels=labels,
            )
            stale_snapshot = [
                {"role": "registry", "path": "stale", "sha256": "0" * 64}
            ]
            with self._reconstruction_patch(), patch(
                "spectral_utils.reconstruction_benchmark.prefix_ab.recompute_prefix_scores",
                return_value=(arrays, self._synthetic_score_audit),
            ), patch(
                "spectral_utils.reconstruction_benchmark.prefix_evaluation."
                "_evaluation_source_snapshot",
                return_value=stale_snapshot,
            ):
                for build_id in ("A", "B"):
                    evaluate_prefix_build(
                        repo=REPO,
                        registry_path=registry_path,
                        release_root=release_root,
                        private_root=private_root,
                        release_id="stale-evaluator",
                        build_id=build_id,
                        source_root=Path("/registered/source"),
                        scientific_full=True,
                    )
            with self._reconstruction_patch(), patch(
                "spectral_utils.reconstruction_benchmark.prefix_ab.recompute_prefix_scores",
                return_value=(arrays, self._synthetic_score_audit),
            ), self.assertRaisesRegex(PrefixContractError, "roster binding"):
                verify_prefix_evaluation_ab(
                    repo=REPO,
                    registry_path=registry_path,
                    release_root=release_root,
                    private_root=private_root,
                    release_id="stale-evaluator",
                    source_root=Path("/registered/source"),
                    require_scientific_full=True,
                )


if __name__ == "__main__":
    unittest.main()
