#!/usr/bin/env python3
"""Target-free synthetic and mechanical gates for Phase-2 temporal B3."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.reconstruction_benchmark.io import atomic_write_npz, sha256_file  # noqa: E402
from spectral_utils.reconstruction_benchmark.localization_contract import (  # noqa: E402
    FIT_SAFE_CELL_FIELDS,
    ID_CONTRACT_VERSION,
    ID_DIGEST_ALGORITHM,
    IDENTITY_KEY_BYTES,
    IDENTITY_KEY_CONTRACT_VERSION,
    OPAQUE_ROW_ID_PREFIX,
    PREPARED_SCHEMA_VERSION,
    load_prepared_localization_token_cell,
)
from spectral_utils.token_local_fusion import prepare_token_fusion  # noqa: E402
from spectral_utils.token_temporal_innovation_b3 import (  # noqa: E402
    CORE_INDICES,
    LOCAL_TOKEN_B3_SELF_INNOV,
    ROOK_PEERS,
    apply_innovation_map,
    fit_innovation_b3,
    fit_innovation_map,
    fixed_support,
    fit_token_b3_ladder,
    select_projected_stg_support,
)
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ContinuousDeemConfig,
    fit_continuous_deem,
    predict_continuous_deem,
)
from spectral_utils.fixed_application_pipelines import SHARED_GLOBAL_FEATURES  # noqa: E402
from spectral_utils.reconstruction_benchmark.localization_contract import payload_sha256  # noqa: E402
from scripts.reconstruction_benchmark.evaluate_token_local_temporal_innovation_b3 import (  # noqa: E402
    _final_selection,
)


def _temporal_world(*, n_rows: int = 30, tokens_per_row: int = 32):
    """Independent predictors with three planted sparse rook edges."""

    rng = np.random.default_rng(20260828)
    parts = []
    offsets = [0]
    truth = {(0, 1), (3, 4), (6, 7)}
    for _row in range(n_rows):
        x = rng.normal(size=(tokens_per_row, 9))
        for target, source in sorted(truth):
            x[:, target] = 4.0 * np.r_[0.0, x[:-1, source]] + 0.1 * rng.normal(size=tokens_per_row)
            x[0, target] = rng.normal()
        parts.append(x)
        offsets.append(offsets[-1] + tokens_per_row)
    return np.vstack(parts), np.asarray(offsets, dtype=np.int64), truth


class TemporalInnovationTests(unittest.TestCase):
    def test_final_selection_applies_pstg_tie_break(self):
        promotion = [
            {"method_id": "LOCAL_TOKEN_B3", "promote": True},
            {"method_id": "LOCAL_TOKEN_B3_ROOK_ALL_INNOV", "promote": True},
            {"method_id": "LOCAL_TOKEN_B3_ROOK_PSTG_INNOV", "promote": True},
        ]
        bootstrap = {"comparisons": {
            "LOCAL_TOKEN_B3_ROOK_PSTG_INNOV": {"delta_vs_rook_all": 0.0005},
        }}
        decision = _final_selection(promotion, bootstrap)
        self.assertEqual(
            decision["selected_method"], "LOCAL_TOKEN_B3_ROOK_PSTG_INNOV"
        )
        bootstrap["comparisons"]["LOCAL_TOKEN_B3_ROOK_PSTG_INNOV"]["delta_vs_rook_all"] = -0.0011
        decision = _final_selection(promotion, bootstrap)
        self.assertEqual(
            decision["selected_method"], "LOCAL_TOKEN_B3_ROOK_ALL_INNOV"
        )

    def test_rook_cardinality_and_nonrook_control(self):
        self.assertTrue(all(len(peers) == 4 for peers in ROOK_PEERS))
        for target, peers in enumerate(ROOK_PEERS):
            self.assertNotIn(target, peers)
        control = fixed_support("LOCAL_TOKEN_B3_NONROOK_INNOV_CONTROL")
        self.assertEqual(int(control.sum()), 36)
        for target in range(9):
            self.assertTrue(all(source not in ROOK_PEERS[target] for source in np.flatnonzero(control[target])))

    def test_projected_stg_sparse_rook_recovery(self):
        core, offsets, truth = _temporal_world()
        rows = tuple(f"row-{i:03d}" for i in range(len(offsets) - 1))
        result = select_projected_stg_support(
            core, offsets, rows, np.arange(len(rows)), np.arange(len(core))
        )
        selected = {tuple(map(int, pair)) for pair in np.argwhere(result.support)}
        true_positives = len(selected & truth)
        false_discoveries = len(selected - truth)
        self.assertGreaterEqual(true_positives / len(truth), 0.80)
        self.assertLessEqual(false_discoveries / max(1, len(selected)), 0.20)
        self.assertTrue(result.diagnostics["exact_subset_audit_passed"])

    def test_self_only_null_has_no_cross_edges(self):
        core, offsets, _truth = _temporal_world()
        support = fixed_support(LOCAL_TOKEN_B3_SELF_INNOV)
        fitted = fit_innovation_map(
            LOCAL_TOKEN_B3_SELF_INNOV, core, offsets,
            np.arange(len(offsets) - 1), np.arange(len(core)), support,
        )
        self.assertEqual(int(fitted.support.sum()), 0)
        self.assertTrue(np.array_equal(fitted.cross_coefficients, np.zeros((9, 9))))

    def test_row_boundaries_and_future_perturbation(self):
        core, offsets, _truth = _temporal_world(n_rows=8, tokens_per_row=20)
        support = fixed_support("LOCAL_TOKEN_B3_ROOK_ALL_INNOV")
        fitted = fit_innovation_map(
            "LOCAL_TOKEN_B3_ROOK_ALL_INNOV", core, offsets,
            np.arange(len(offsets) - 1), np.arange(len(core)), support,
        )
        innovation, mask = apply_innovation_map(core, offsets, fitted)
        first_tokens = offsets[:-1]
        self.assertTrue(np.array_equal(mask[first_tokens], np.zeros(len(first_tokens))))
        self.assertTrue(np.array_equal(innovation[first_tokens], np.zeros((len(first_tokens), 9))))
        cut = int(offsets[3] + 4)
        changed = core.copy()
        changed[cut + 1:] += 123.0
        perturbed, _ = apply_innovation_map(changed, offsets, fitted)
        self.assertTrue(np.array_equal(innovation[:cut + 1], perturbed[:cut + 1]))

    def test_time_shift_control_loses_causal_prediction_advantage(self):
        """A within-question time shift must break the planted lag signal."""

        core, offsets, _truth = _temporal_world(n_rows=24, tokens_per_row=24)
        support = fixed_support("LOCAL_TOKEN_B3_ROOK_ALL_INNOV")
        donor_rows = np.arange(18)
        held_rows = np.arange(18, 24)
        donor_indices = np.concatenate([
            np.arange(int(offsets[row]), int(offsets[row + 1])) for row in donor_rows
        ])
        fitted = fit_innovation_map(
            "LOCAL_TOKEN_B3_ROOK_ALL_INNOV", core, offsets,
            donor_rows, donor_indices, support,
        )
        causal, mask = apply_innovation_map(core, offsets, fitted)
        shifted = core.copy()
        for row in held_rows:
            lo, hi = offsets[row:row + 2]
            lo_i, hi_i = int(lo), int(hi)
            # Shift only the source streams on held questions.  Shifting every
            # stream together would preserve the same lag relation.
            for source in (1, 4, 7):
                shifted[lo_i:hi_i, source] = np.roll(
                    shifted[lo_i:hi_i, source], 1
                )
        broken, _ = apply_innovation_map(shifted, offsets, fitted)
        held_indices = np.concatenate([
            np.arange(int(offsets[row]), int(offsets[row + 1])) for row in held_rows
        ])
        valid = mask[held_indices].astype(bool)
        causal_mse = float(np.mean(np.square(causal[held_indices][valid])))
        shifted_mse = float(np.mean(np.square(broken[held_indices][valid])))
        self.assertGreater(shifted_mse, causal_mse * 1.25)

    def test_b3_zero_extension_is_exact(self):
        rng = np.random.default_rng(9)
        original = rng.normal(size=(48, 29))
        innovations = rng.normal(size=(48, 9))
        mask = np.ones(48)
        config = ContinuousDeemConfig(epochs=3, mala_steps=1, family_width=4)
        baseline = fit_continuous_deem(original, SHARED_GLOBAL_FEATURES, seed=2, config=config)
        zero = fit_innovation_b3(original, innovations, mask, seed=2, config=config, gain=0.0)
        self.assertIsInstance(zero, type(baseline))
        left = predict_continuous_deem(baseline, original)
        right = predict_continuous_deem(zero, original)
        self.assertTrue(np.array_equal(left["score"], right["score"]))
        self.assertTrue(np.array_equal(left["logit"], right["logit"]))

    def test_innovation_additive_reconstruction_and_first_mask(self):
        rng = np.random.default_rng(11)
        original = rng.normal(size=(64, 29))
        innovations = rng.normal(size=(64, 9))
        mask = np.ones(64)
        mask[0] = 0.0
        config = ContinuousDeemConfig(epochs=3, mala_steps=1, family_width=4)
        fitted = fit_innovation_b3(
            original, innovations, mask, seed=3, config=config, gain=1.0
        )
        prediction = fitted  # fit-time diagnostics include the full reconstruction gate
        self.assertTrue(prediction.health["healthy"])
        self.assertLessEqual(prediction.health["additive_logit_reconstruction_max_abs"], 1e-8)
        self.assertEqual(prediction.health["masked_first_token_innovation_contribution_max_abs"], 0.0)

    def test_full_ladder_has_five_folds_and_five_seed_records(self):
        core, offsets, _truth = _temporal_world(n_rows=50, tokens_per_row=12)
        values = np.random.default_rng(17).normal(size=(len(core), 29))
        values[:, CORE_INDICES] = core
        rows = tuple(f"row-{i:03d}" for i in range(len(offsets) - 1))
        preparation = prepare_token_fusion(values, offsets, rows)
        result = fit_token_b3_ladder(
            preparation,
            config=ContinuousDeemConfig(epochs=2, mala_steps=1, family_width=2),
        )
        self.assertEqual(tuple(result), (
            "LOCAL_TOKEN_B3", "LOCAL_TOKEN_B3_SELF_INNOV",
            "LOCAL_TOKEN_B3_ROOK_ALL_INNOV", "LOCAL_TOKEN_B3_ROOK_PSTG_INNOV",
            "LOCAL_TOKEN_B3_NONROOK_INNOV_CONTROL",
        ))
        for value in result.values():
            self.assertEqual(len(value.fold_diagnostics), 5)
            self.assertEqual(len(value.per_seed_model_records), 25)
            self.assertTrue(value.health["targets_accessed_during_fit"] is False)

    def test_parallel_fit_schedule_is_byte_exact(self):
        core, offsets, _truth = _temporal_world(n_rows=40, tokens_per_row=10)
        values = np.random.default_rng(1701).normal(size=(len(core), 29))
        values[:, CORE_INDICES] = core
        rows = tuple(f"row-{i:03d}" for i in range(len(offsets) - 1))
        preparation = prepare_token_fusion(values, offsets, rows)
        config = ContinuousDeemConfig(epochs=1, mala_steps=1, family_width=2)
        serial = fit_token_b3_ladder(
            preparation, config=config, execution_workers=1,
        )
        parallel = fit_token_b3_ladder(
            preparation, config=config, execution_workers=4,
        )
        for method_id in serial:
            self.assertTrue(np.array_equal(
                serial[method_id].token_risk, parallel[method_id].token_risk
            ))
            for left, right in zip(
                serial[method_id].per_seed_model_records,
                parallel[method_id].per_seed_model_records,
            ):
                self.assertEqual(set(left["state"]), set(right["state"]))
                for name in left["state"]:
                    self.assertTrue(np.array_equal(
                        left["state"][name], right["state"][name]
                    ))

    def test_token_only_loader_never_indexes_response_scores(self):
        rows = tuple(
            f"{OPAQUE_ROW_ID_PREFIX}{index:064x}" for index in range(3)
        )
        methods = tuple(f"method_{index:02d}" for index in range(13))
        contract_payload = {
            "schema_version": "reconstruction-external-fit-row-identity-v1",
            "version": ID_CONTRACT_VERSION,
            "digest_algorithm": ID_DIGEST_ALGORITHM,
            "identity_key_contract_version": IDENTITY_KEY_CONTRACT_VERSION,
            "identity_key_bytes": IDENTITY_KEY_BYTES,
            "opaque_row_id_prefix": OPAQUE_ROW_ID_PREFIX,
            "row_namespace_scope": "cell",
            "canonical_row_order": "lexicographic_opaque_row_id",
            "key_id": f"xkidv1_{'a' * 64}",
            "private_group_linkage_commitment": f"xglcv1_{'b' * 64}",
        }
        contract = dict(contract_payload)
        contract["contract_sha256"] = payload_sha256(contract_payload)
        arrays = {
            "token_confidence": np.arange(3 * 29, dtype="<f8").reshape(3, 29) + 1.0,
            "token_offsets": np.asarray([0, 1, 2, 3], dtype="<i8"),
            "segment_offsets": np.asarray([0, 1, 2, 3], dtype="<i8"),
            "segment_starts": np.asarray([0, 1, 2], dtype="<i8"),
            "segment_ends": np.asarray([1, 2, 3], dtype="<i8"),
            "row_ids": np.asarray(rows, dtype="<U80"),
            "method_ids": np.asarray(methods, dtype="<U48"),
            "id_contract_version": np.asarray([ID_CONTRACT_VERSION], dtype="<U64"),
            "id_contract_sha256": np.asarray([contract["contract_sha256"]], dtype="<U64"),
            "identity_key_id": np.asarray([contract["key_id"]], dtype="<U80"),
            "row_namespace_sha256": np.asarray(['c' * 64], dtype="<U64"),
            "external_certificate_sha256": np.asarray(['d' * 64], dtype="<U64"),
            "external_score_bindings_sha256": np.asarray(['e' * 64], dtype="<U64"),
            "token_transform_sha256": np.asarray(['f' * 64], dtype="<U64"),
            "response_scores": np.ones((13, 3), dtype="<f8"),
        }
        record = {
            "schema_version": PREPARED_SCHEMA_VERSION,
            "cell_id": "synthetic_cell",
            "population_id": "synthetic_population",
            "dataset_id": "processbench",
            "model_id": "qwen3_4b",
            "slice_id": "gsm8k",
            "status": "ELIGIBLE",
            "n_rows": 3,
            "n_tokens": 3,
            "n_segments": 3,
            "n_token_streams": 29,
            "method_ids": list(methods),
            "token_contract_id": "localization-token-iu29-mixed-v2-v1",
            "token_mixed_v2_applied_count": 1,
            "token_matrix_semantics": "higher_is_confidence",
            "identity_contract": contract,
            "id_contract_version": ID_CONTRACT_VERSION,
            "id_contract_sha256": contract["contract_sha256"],
            "identity_key_id": contract["key_id"],
            "row_namespace_sha256": 'c' * 64,
            "row_roster_sha256": '1' * 64,
            "external_certificate_sha256": 'd' * 64,
            "external_score_bindings_sha256": 'e' * 64,
            "token_transform_sha256": 'f' * 64,
            "artifact_path": "cell.npz",
            "artifact_sha256": "",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cell.npz"
            record["artifact_sha256"] = atomic_write_npz(path, arrays)
            def guarded_getitem(instance, key):
                if str(key) == "response_scores":
                    raise AssertionError("response_scores was materialized on token-only path")
                return original_getitem(instance, key)
            original_getitem = np.lib.npyio.NpzFile.__getitem__
            with mock.patch.object(np.lib.npyio.NpzFile, "__getitem__", guarded_getitem):
                value = load_prepared_localization_token_cell(path, record)
            self.assertEqual(value.token_confidence.shape, (3, 29))


if __name__ == "__main__":
    unittest.main()
