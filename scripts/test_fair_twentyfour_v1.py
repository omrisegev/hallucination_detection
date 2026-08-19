#!/usr/bin/env python3
"""Focused tests and an opt-in real preflight for the fair 24-cell adapter.

The normal invocation uses only synthetic fixtures::

    python scripts/test_fair_twentyfour_v1.py

The real preflight is intentionally explicit because materialized sources total
multiple GiB.  It never scores Unified-28 unless ``--anchor`` is supplied and it
never writes an artifact::

    python scripts/test_fair_twentyfour_v1.py --real-preflight \
      --source-root /private/tmp/unified_24cell_raw --cells sciq_llama8b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.a5_target_free_data import (  # noqa: E402
    CORE_FEATURES,
    TELEMETRY_KEYS,
    FrozenSourceSpec,
)
from spectral_utils.fair_comparisons import twentyfour as T24  # noqa: E402
from spectral_utils.fair_comparisons.twentyfour import (  # noqa: E402
    BLOCKED_CELL,
    DIRECT_METHOD_IDS,
    ELIGIBLE_CELLS,
    ELIGIBLE_ROWS,
    EXPECTED_CELLS,
    EXPECTED_ROWS,
    IdentityAlignment,
    MODEL_ARTIFACT_SHA256,
    TwentyFourError,
    admit_source_rows,
    eligible_cell_ids,
    freeze_identity_alignment,
    incumbent_risk_scores,
    load_unified28_model,
    open_and_verify_labels,
    partial_identity_audit,
    paired_cell_intervals,
    per_cell_metrics,
    real_identity_preflight,
    static_preflight,
    unified28_parameter_sha256,
    unified28_replay_source_artifact_sha256,
    verify_processbench_anchor,
)


def _candidate(seed: int, label: int) -> dict:
    rng = np.random.default_rng(seed)
    n = 32 + seed % 5
    entropy = np.linspace(0.15, 1.25, n) + rng.normal(0.0, 0.04, n)
    spilled = np.linspace(0.25, 0.95, n) + rng.normal(0.0, 0.03, n)
    logsum = np.linspace(3.1, 4.4, n) + rng.normal(0.0, 0.05, n)
    topk = np.sort(rng.normal(-2.0, 0.7, (n, 8)), axis=1)[:, ::-1]
    return {
        "token_entropies": entropy,
        "token_spilled_energies": spilled,
        "token_logsumexp": logsum,
        "top_k_logprobs": {
            "logprobs": topk,
            "ids": np.tile(np.arange(8), (n, 1)),
        },
        "label": bool(label),
    }


def _fixture(n: int = 15, *, cell_id: str = "fixture_cell"):
    source = {}
    # Insertion order is deliberately different from canonical lexicographic order.
    for index in [10, 2, 1, 11, 3, 20, 4, 12, 5, 13, 6, 14, 7, 15, 8][:n]:
        source[str(index)] = {
            "question": f"Fixture question {index}?",
            "candidates": [_candidate(index, index % 2)],
        }
    spec = FrozenSourceSpec(
        environment_id=cell_id,
        dataset="gsm8k",
        split="test",
        dataset_family="gsm8k",
        expected_admitted_count=n,
        admission_mode="complete_h16",
        raw_relative_path="unused.pkl",
        source_sha256="0" * 64,
        source_size=0,
        manifest_sha256="1" * 64,
    )
    identities = admit_source_rows(source, spec)
    core = np.vstack([row.core_features for row in identities])
    signs = np.asarray([(-1.0 if index % 3 else 1.0) for index in range(len(CORE_FEATURES))])
    columns = []
    for index in range(len(CORE_FEATURES)):
        values = core[:, index] * signs[index]
        columns.append((values - values.mean()) / values.std())
    canonical = np.column_stack(columns)
    bundle_order = np.asarray(list(range(1, n, 2)) + list(range(0, n, 2)), dtype=int)
    raw_labels = np.asarray(
        [int(row.source_candidate["label"]) for row in identities], dtype=np.int8
    )
    bundle = {
        f"{cell_id}__V": canonical[bundle_order],
        f"{cell_id}__pool": np.asarray(CORE_FEATURES, dtype=object),
        f"{cell_id}__hand_signs": signs,
        f"{cell_id}__labels": raw_labels[bundle_order],
    }
    return identities, bundle, bundle_order


class LabelAccessGuard(dict):
    def __init__(self, values):
        super().__init__(values)
        self.allow_labels = False
        self.labels_accessed = False

    def __getitem__(self, key):
        if str(key).endswith("__labels"):
            self.labels_accessed = True
            if not self.allow_labels:
                raise AssertionError("labels opened before identity freeze")
        return super().__getitem__(key)


class TwentyFourFixtureTests(unittest.TestCase):
    def test_frozen_coverage_and_blocker_are_exact(self):
        self.assertEqual(len(eligible_cell_ids()), ELIGIBLE_CELLS)
        self.assertNotIn(BLOCKED_CELL, eligible_cell_ids())
        self.assertEqual(sum(spec.expected_admitted_count for spec in __import__(
            "spectral_utils.a5_target_free_data", fromlist=["FROZEN_A0_SOURCE_SPECS"]
        ).FROZEN_A0_SOURCE_SPECS), ELIGIBLE_ROWS)
        self.assertEqual(EXPECTED_CELLS, 24)
        self.assertEqual(EXPECTED_ROWS - ELIGIBLE_ROWS, 256)

    def test_identity_is_canonical_sorted_not_positional(self):
        identities, bundle, bundle_order = _fixture()
        self.assertTrue(all(set(row.telemetry) == set(TELEMETRY_KEYS) for row in identities))
        self.assertTrue(all("label" not in row.telemetry for row in identities))
        alignment = freeze_identity_alignment(identities, bundle)
        inverse = np.argsort(bundle_order)
        self.assertEqual(alignment.bundle_position_by_row, tuple(inverse.tolist()))
        self.assertNotEqual(alignment.bundle_position_by_row, tuple(range(len(identities))))
        self.assertTrue(alignment.identity_frozen)
        self.assertLessEqual(alignment.max_abs_feature_error, 1e-10)
        self.assertEqual(len(set(alignment.row_ids)), len(identities))
        self.assertTrue(all("::candidate0" in row_id for row_id in alignment.row_ids))

    def test_source_question_id_preserves_raw_candidate_ordinal(self):
        short = _candidate(91, 0)
        short["token_entropies"] = short["token_entropies"][:3]
        short["token_spilled_energies"] = short["token_spilled_energies"][:3]
        short["token_logsumexp"] = short["token_logsumexp"][:3]
        short["top_k_logprobs"] = {
            key: value[:3] for key, value in short["top_k_logprobs"].items()
        }
        source = {
            "problem-7": {
                "question": "Which admitted candidate keeps its raw ordinal?",
                "candidates": [short, _candidate(92, 1)],
            }
        }
        spec = FrozenSourceSpec(
            environment_id="fixture_cell",
            dataset="gsm8k",
            split="test",
            dataset_family="gsm8k",
            expected_admitted_count=1,
            admission_mode="complete_h16",
            raw_relative_path="unused.pkl",
            source_sha256="0" * 64,
            source_size=0,
            manifest_sha256="1" * 64,
        )
        (identity,) = admit_source_rows(source, spec)
        self.assertEqual(identity.candidate_ordinal, 1)
        self.assertEqual(identity.source_question_id, "problem-7::candidate1")
        self.assertEqual(identity.row_id, "fixture_cell::problem-7::candidate1")
        self.assertEqual(identity.group_id, "fixture_cell::problem-7")

    def test_unified28_replay_hash_binds_model_and_raw_source(self):
        model_hash = "a" * 64
        raw_hash = "b" * 64
        observed = unified28_replay_source_artifact_sha256(model_hash, raw_hash)
        self.assertEqual(
            observed,
            T24.canonical_sha256(
                {
                    "schema": "unified28_24cell_replay_source_v1",
                    "model_artifact_sha256": model_hash,
                    "raw_source_sha256": raw_hash,
                }
            ),
        )
        self.assertNotEqual(
            observed,
            unified28_replay_source_artifact_sha256("c" * 64, raw_hash),
        )
        self.assertNotEqual(
            observed,
            unified28_replay_source_artifact_sha256(model_hash, "d" * 64),
        )

    def test_static_preflight_reports_only_size_readiness(self):
        audit = static_preflight(
            REPO,
            verify_score_hashes=False,
            verify_raw_hashes=False,
        )
        self.assertNotIn("currently_materialized_cells", audit)
        self.assertNotIn("currently_materialized_rows", audit)
        self.assertEqual(audit["source_file_size_ready_cells"], len(audit["sources"]))
        self.assertEqual(
            audit["source_file_size_ready_rows"],
            sum(row["expected_admitted_count"] for row in audit["sources"]),
        )
        self.assertEqual(audit["identity_aligned_cells"], 0)
        self.assertEqual(audit["identity_aligned_rows"], 0)
        self.assertEqual(audit["scored_cells"], 0)
        self.assertEqual(audit["scored_rows"], 0)
        self.assertFalse(
            audit["row_identity_contract"]["positional_fallback_allowed"]
        )

    def test_partial_identity_audit_continues_and_never_scores(self):
        pass_cell = "z_pass_cell"
        fail_cell = "a_fail_cell"
        identities, bundle, _ = _fixture(n=8, cell_id=pass_cell)
        specs = {
            pass_cell: FrozenSourceSpec(
                environment_id=pass_cell,
                dataset="gsm8k",
                split="test",
                dataset_family="gsm8k",
                expected_admitted_count=8,
                admission_mode="complete_h16",
                raw_relative_path="unused-pass.pkl",
                source_sha256="a" * 64,
                source_size=1,
                manifest_sha256="b" * 64,
            ),
            fail_cell: FrozenSourceSpec(
                environment_id=fail_cell,
                dataset="gsm8k",
                split="test",
                dataset_family="gsm8k",
                expected_admitted_count=3,
                admission_mode="complete_h16",
                raw_relative_path="unused-fail.pkl",
                source_sha256="c" * 64,
                source_size=1,
                manifest_sha256="d" * 64,
            ),
        }

        class BundleContext:
            def __enter__(self):
                return bundle

            def __exit__(self, exc_type, exc, traceback):
                return False

        def load_cell(repo, cell_id, *, source_root, verify_sha256):
            self.assertTrue(verify_sha256)
            self.assertEqual(source_root, "/size-ready")
            if cell_id == fail_cell:
                raise TwentyFourError(
                    "raw source SHA-256 mismatch: a_fail_cell"
                )
            return identities, {
                "cell_id": pass_cell,
                "source_sha256": "a" * 64,
                "source_sha256_verified": True,
                "expected_admitted_count": 8,
            }

        with (
            mock.patch.object(T24, "sha256_file", return_value=T24.BUNDLE_SHA256),
            mock.patch.object(T24.np, "load", return_value=BundleContext()),
            mock.patch.object(T24, "_source_spec", side_effect=lambda cell: specs[cell]),
            mock.patch.object(T24, "load_admitted_cell", side_effect=load_cell) as loader,
            mock.patch.object(
                T24,
                "materialize_cell_records",
                side_effect=AssertionError("partial audit must never score"),
            ) as scorer,
        ):
            audit = partial_identity_audit(
                REPO,
                source_root="/size-ready",
                cells=[pass_cell, fail_cell],
            )

        self.assertEqual(audit["requested_cells"], [fail_cell, pass_cell])
        self.assertEqual(loader.call_count, 2)
        scorer.assert_not_called()
        self.assertEqual(audit["identity_proven_cells"], 1)
        self.assertEqual(audit["identity_proven_rows"], 8)
        self.assertEqual(audit["failed_cells"], 1)
        self.assertFalse(audit["all_ok"])
        self.assertFalse(audit["scoring_performed"])
        self.assertTrue(audit["raw_sha256_required"])
        failed, passed = audit["audits"]
        self.assertEqual(failed["cell_id"], fail_cell)
        self.assertEqual(failed["failure_stage"], "raw_sha256_and_admission")
        self.assertEqual(failed["failure_type"], "TwentyFourError")
        self.assertEqual(
            failed["failure_reason"],
            "raw source SHA-256 mismatch: a_fail_cell",
        )
        self.assertEqual(passed["status"], "identity-proven")
        self.assertTrue(passed["alignment"]["label_alignment_ok"])
        payload = {key: value for key, value in audit.items() if key != "audit_sha256"}
        self.assertEqual(audit["audit_sha256"], T24.canonical_sha256(payload))

    def test_labels_are_not_opened_until_identity_freezes(self):
        identities, values, _ = _fixture()
        guarded = LabelAccessGuard(values)
        alignment = freeze_identity_alignment(identities, guarded)
        self.assertFalse(guarded.labels_accessed)
        guarded.allow_labels = True
        labels = open_and_verify_labels(identities, alignment, guarded)
        self.assertTrue(guarded.labels_accessed)
        self.assertTrue(labels.label_alignment_ok)
        self.assertEqual(
            labels.error_labels,
            tuple(1 - int(row.source_candidate["label"]) for row in identities),
        )

    def test_label_permutation_cannot_change_unified_score_inputs(self):
        identities, _, _ = _fixture()
        model, _ = load_unified28_model(REPO)
        before = np.asarray(
            [model.score_row(row.telemetry).global_score for row in identities]
        )
        for row in identities:
            row.source_candidate["label"] = not row.source_candidate["label"]
        after = np.asarray(
            [model.score_row(row.telemetry).global_score for row in identities]
        )
        np.testing.assert_array_equal(before, after)
        for row in identities:
            del row.source_candidate["label"]
        after_removal = np.asarray(
            [model.score_row(row.telemetry).global_score for row in identities]
        )
        np.testing.assert_array_equal(before, after_removal)

    def test_label_disagreement_fails_after_identity(self):
        identities, bundle, _ = _fixture()
        alignment = freeze_identity_alignment(identities, bundle)
        bundle["fixture_cell__labels"] = bundle["fixture_cell__labels"].copy()
        bundle["fixture_cell__labels"][0] ^= 1
        with self.assertRaisesRegex(TwentyFourError, "label disagreement"):
            open_and_verify_labels(identities, alignment, bundle)

    def test_duplicate_feature_signature_fails_closed(self):
        identities, bundle, _ = _fixture()
        bundle["fixture_cell__V"] = bundle["fixture_cell__V"].copy()
        bundle["fixture_cell__V"][1] = bundle["fixture_cell__V"][0]
        with self.assertRaisesRegex(TwentyFourError, "identity join failed"):
            freeze_identity_alignment(identities, bundle)

    def test_unified28_deserializes_with_exact_frozen_roster(self):
        model, artifact_hash = load_unified28_model(REPO)
        self.assertEqual(len(model.feature_names), 28)
        self.assertEqual(model.accumulator.kind, "identity")
        self.assertEqual(model.diagnostics["components"], 2)
        self.assertEqual(artifact_hash, MODEL_ARTIFACT_SHA256)
        self.assertEqual(
            unified28_parameter_sha256(model),
            "13d9d302b00e21b69a1af1a8f025157483069251240e39c314a1e73d9bc41d7a",
        )
        with self.assertRaisesRegex(TwentyFourError, "anchor"):
            T24._require_processbench_anchor({}, model, artifact_hash)

    def test_real_score_checkpoint_is_bound_by_freeze_manifest(self):
        path, observed_hash = T24._verified_score_path(REPO, "sciq_llama8b")
        self.assertEqual(path.name, "sciq_llama8b.npz")
        self.assertEqual(path.parent.parent.name, "hard_filter_dufs_liu_24cell")
        self.assertEqual(len(observed_hash), 64)

    def test_direct_roster_is_only_approved_mixed_v2_rows(self):
        self.assertEqual(
            DIRECT_METHOD_IDS,
            (
                "unified28",
                "mixed_v2_iu_pcr_24cell",
                "mixed_v2_dufs_liu_l0p1_24cell",
                "max_entropy_24cell",
            ),
        )
        self.assertEqual(T24.PRIMARY_INCUMBENT_METHOD_ID, T24.IU_METHOD_ID)
        self.assertNotIn("deployed_upcr", DIRECT_METHOD_IDS)
        self.assertEqual(
            T24.population_id_for_cell("sciq_llama8b"),
            f"{T24.DATASET_REVISION}::sciq_llama8b::identity-proven",
        )

    def test_incumbent_scores_are_reordered_and_risk_oriented(self):
        alignment = IdentityAlignment(
            cell_id="cell",
            row_ids=("a", "b", "c"),
            group_ids=("ga", "gb", "gc"),
            bundle_position_by_row=(2, 0, 1),
            ordered_id_sha256="0" * 64,
            identity_feature_names=("a",) * 8,
            identity_feature_sha256="1" * 64,
            max_abs_feature_error=0.0,
            signature_decimals=12,
            identity_frozen=True,
        )
        checkpoint = {
            "sample_index": np.arange(3),
            "mixed_v2__full__iu_pcr": np.asarray([1.0, 2.0, 3.0]),
            "mixed_v2__full__dufs_liu": np.asarray([7.0, 8.0, 9.0]),
        }
        scores = incumbent_risk_scores(checkpoint, alignment)
        np.testing.assert_array_equal(
            scores["mixed_v2_iu_pcr_24cell"], [-3.0, -1.0, -2.0]
        )
        np.testing.assert_array_equal(
            scores["mixed_v2_dufs_liu_l0p1_24cell"], [-9.0, -7.0, -8.0]
        )

    def test_per_cell_metrics_require_identical_direct_rows_and_crossfit(self):
        rows = []
        for method_index, method_id in enumerate(DIRECT_METHOD_IDS):
            for index in range(100):
                label = index % 2
                rows.append(
                    {
                        "cell_id": "c",
                        "row_id": f"r{index}",
                        "method_id": method_id,
                        "label": label,
                        "continuous_score": float(label + method_index * 1e-3),
                        "fold": index % 5,
                    }
                )
        metrics = per_cell_metrics(rows)
        self.assertEqual(set(metrics), set(DIRECT_METHOD_IDS))
        self.assertTrue(all(value["auroc"] == 1.0 for value in metrics.values()))
        self.assertTrue(all(
            len(value["operating_fpr_05"]["calibration_ledgers"]) == 5
            for value in metrics.values()
        ))
        rows.pop()
        with self.assertRaisesRegex(TwentyFourError, "row IDs differ"):
            per_cell_metrics(rows)

    def test_grouped_bootstrap_carries_candidate_siblings_without_overwrite(self):
        rows = []
        n_candidate_rows = 0
        for group_index in range(100):
            candidate_count = 2 if group_index % 7 == 0 else 1
            n_candidate_rows += candidate_count
            for candidate in range(candidate_count):
                label = (group_index + candidate) % 2
                for method_index, method_id in enumerate(DIRECT_METHOD_IDS):
                    rows.append(
                        {
                            "cell_id": "sciq_llama8b",
                            "population_id": T24.population_id_for_cell(
                                "sciq_llama8b"
                            ),
                            "group_id": f"sciq_llama8b::q{group_index}",
                            "row_id": (
                                f"sciq_llama8b::q{group_index}::candidate{candidate}"
                            ),
                            "family": "sciq",
                            "method_id": method_id,
                            "label": label,
                            "continuous_score": float(
                                label + method_index * 1e-3 + candidate * 1e-5
                            ),
                            "fold": group_index % 5,
                        }
                    )
        result = paired_cell_intervals(rows, n_boot=4, seed=20260818)
        self.assertEqual(result["source_question_groups"], 100)
        self.assertEqual(result["candidate_rows"], n_candidate_rows)
        self.assertTrue(result["candidate_siblings_carried_together"])
        self.assertEqual(
            result["population_ordered_id_sha256"],
            T24.ordered_id_sha256(
                [
                    row["row_id"]
                    for row in rows
                    if row["method_id"] == T24.U28_METHOD_ID
                ]
            ),
        )
        self.assertEqual(
            result["primary_contrast"]["right_method_id"], T24.IU_METHOD_ID
        )
        self.assertEqual(result["n_boot"], 4)

        split = [dict(row) for row in rows]
        first = next(
            row
            for row in split
            if row["method_id"] == T24.U28_METHOD_ID
            and row["group_id"] == "sciq_llama8b::q0"
            and row["row_id"].endswith("candidate0")
        )
        first["fold"] = 1
        with self.assertRaisesRegex(TwentyFourError, "split across folds"):
            paired_cell_intervals(split, n_boot=2, seed=20260818)


def _real_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-preflight", action="store_true", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--cells", nargs="*", default=[])
    parser.add_argument("--verify-raw-hashes", action="store_true")
    parser.add_argument("--anchor", action="store_true")
    args = parser.parse_args(argv)
    static = static_preflight(
        REPO,
        source_root=args.source_root,
        verify_score_hashes=True,
        verify_raw_hashes=False,
    )
    result = {"static": static}
    if args.cells:
        result["identity"] = real_identity_preflight(
            REPO,
            source_root=args.source_root,
            cells=args.cells,
            verify_raw_hashes=args.verify_raw_hashes,
        )
    if args.anchor:
        model, _ = load_unified28_model(REPO)
        result["processbench_anchor"] = verify_processbench_anchor(REPO, model)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    if "--real-preflight" in sys.argv:
        raise SystemExit(_real_main(sys.argv[1:]))
    unittest.main()
