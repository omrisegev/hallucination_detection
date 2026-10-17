from __future__ import annotations

import hashlib
import inspect
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import run_fusion_independence_atlas_v1 as atlas
from spectral_utils import fusion_signal_registry as registry


class FusionIndependenceAtlasContractTests(unittest.TestCase):
    def test_causal_backgrounds_never_observe_current_or_future(self):
        rng = np.random.default_rng(4)
        original = rng.normal(size=(24, 4))
        changed = original.copy()
        changed[12:] = rng.normal(loc=1_000, size=(12, 4))
        left = atlas.causal_backgrounds(original)
        right = atlas.causal_backgrounds(changed)
        self.assertEqual(set(left), {"prefix", "mean16", "noreset", "bocpd"})
        for name in left:
            np.testing.assert_allclose(left[name][:13], right[name][:13], atol=1e-12, rtol=0,
                                       err_msg=f"{name} observed a future token")
            np.testing.assert_array_equal(left[name][0], np.zeros(4))

    def test_bundle_contract_has_exactly_four_primitive_targets(self):
        self.assertEqual(atlas.PRIMITIVE_TARGETS, ("H0lim", "VE0", "VE0.75", "VE1"))
        self.assertEqual(atlas.EXPECTED_TOKENS, 6_968_779)
        metadata = atlas.sanitized_metadata(
            {"uid": "u", "cell": "pb_x", "group_id": "g", "tokens": 2},
            fold=1, token_offset=0, step_start=0, step_stop=1,
            mean=[1, 2, 3, 4], scale=[1, 1, 1, 1], signs=[1, 1, 1, 1],
        )
        self.assertEqual(len(metadata["mean"]), 4)
        self.assertFalse({"label", "labels", "target", "correctness"} & set(metadata))
        self.assertNotIn("H0lim innovation", atlas.PRIMITIVE_TARGETS)

    def test_immutable_manifest_and_completion_resume_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "MANIFEST.json"
            digest = atlas.bind_immutable_manifest(manifest, {"schema": "fixture", "shape": [3, 4]})
            bitmap = atlas.CompletionBitmap(root / "completion.npy", 3, digest)
            bitmap.mark(1)
            bitmap.close()
            resumed = atlas.CompletionBitmap(root / "completion.npy", 3, digest)
            self.assertEqual(resumed.count(), 1)
            self.assertTrue(resumed.done(1))
            resumed.close()
            with self.assertRaises(atlas.AtlasContractError):
                atlas.bind_immutable_manifest(manifest, {"schema": "fixture", "shape": [4, 4]})
            with self.assertRaises(atlas.AtlasContractError):
                atlas.CompletionBitmap(root / "completion.npy", 4, digest)

    def test_signal_predictor_and_fusion_fit_apis_are_label_free(self):
        callables = (
            atlas.extract_atomic_answer,
            atlas.causal_backgrounds,
            registry.equal_rank_fusion,
            registry.family_equal_fusion,
            registry.nonnegative_shrunk_simplex_fusion,
        )
        forbidden = {"label", "labels", "correctness", "annotations", "y"}
        for function in callables:
            self.assertFalse(forbidden & set(inspect.signature(function).parameters), function.__name__)
            atlas.assert_label_free_callable(function)

        def bad_fit(scores, labels):
            return scores, labels

        with self.assertRaises(atlas.AtlasContractError):
            atlas.assert_label_free_callable(bad_fit)

    def test_fifteen_exclusion_fits_and_oof_application(self):
        import torch
        from spectral_utils.temporal_context_models import TelemetryTCN

        self.assertEqual(len(atlas.predictor_exclusions()), 15)
        self.assertEqual(atlas.predictor_exclusions()[:5], tuple((fold,) for fold in range(5)))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, output = root / "bundle", root / "predictors"
            bundle.mkdir(); (output / "ridge").mkdir(parents=True); (output / "tcn").mkdir()
            values = np.arange(60, dtype=np.float32).reshape(15, 4) / 10
            np.save(bundle / "primitive_levels.npy", values)
            metadata = []
            for fold in range(5):
                metadata.append({"uid": f"u{fold}", "fold": fold, "offset": 3 * fold, "tokens": 3})
                dimension = atlas._ridge_design(values[:3])[0].shape[1]
                coefficient = np.zeros((dimension, 4), dtype=float)
                coefficient[-1] = fold + 1
                np.savez(output / f"ridge/exclude_{fold}.npz", coefficient=coefficient)
                model = TelemetryTCN(dimensions=4, width=32)
                torch.save({"state_dict": model.state_dict(), "targets": atlas.PRIMITIVE_TARGETS},
                           output / f"tcn/exclude_{fold}.pt")
            completed, done = atlas._apply_outer_predictors(
                bundle, output, metadata, device="cpu", contract_sha256="a" * 64,
            )
            self.assertTrue(done)
            self.assertEqual(completed, 5)
            background = np.load(output / "learned_oof/backgrounds.npy")
            active = np.load(output / "learned_oof/active.npy")
            self.assertEqual(background.shape, (15, 2, 4))
            for fold in range(5):
                start = 3 * fold
                self.assertFalse(active[start])
                np.testing.assert_array_equal(background[start, :, :], 0)
                np.testing.assert_allclose(background[start + 1:start + 3, 0], fold + 1)

    def test_pair_excluded_inner_oof_axes_resume_and_drift(self):
        import torch
        from spectral_utils.temporal_context_models import TelemetryTCN

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, output = root / "bundle", root / "predictors"
            bundle.mkdir(); (output / "ridge").mkdir(parents=True); (output / "tcn").mkdir()
            values = np.arange(36, dtype=np.float32).reshape(9, 4) / 10
            np.save(bundle / "primitive_levels.npy", values)
            metadata = [
                {"uid": "u0", "fold": 0, "offset": 0, "tokens": 3},
                {"uid": "u1", "fold": 1, "offset": 3, "tokens": 3},
                {"uid": "u4", "fold": 4, "offset": 6, "tokens": 3},
            ]
            ridge_dimension = atlas._ridge_design(values[:3])[0].shape[1]
            for left, right in itertools.combinations(range(5), 2):
                coefficient = np.zeros((ridge_dimension, 4), dtype=float)
                coefficient[-1] = 10 * left + right + 1
                np.savez(
                    output / f"ridge/exclude_{left}_{right}.npz",
                    coefficient=coefficient, targets=np.asarray(atlas.PRIMITIVE_TARGETS),
                )
                model = TelemetryTCN(dimensions=4, width=32)
                for parameter in model.parameters():
                    parameter.data.zero_()
                torch.save(
                    {"state_dict": model.state_dict(), "targets": atlas.PRIMITIVE_TARGETS},
                    output / f"tcn/exclude_{left}_{right}.pt",
                )

            completed, done = atlas._apply_inner_predictors(
                bundle, output, metadata, device="cpu", contract_sha256="b" * 64,
                max_answers=1,
            )
            self.assertEqual(completed, 1)
            self.assertFalse(done)
            partial = np.load(output / "learned_inner/backgrounds.npy").copy()
            active = np.load(output / "learned_inner/active.npy")
            self.assertEqual(partial.shape, (9, 5, 2, 4))
            self.assertEqual(active.shape, (9, 5))
            np.testing.assert_array_equal(active[0], False)
            np.testing.assert_array_equal(active[1:3, 0], False)  # f == g diagonal
            np.testing.assert_array_equal(active[1:3, 1:], True)
            np.testing.assert_array_equal(partial[:3, 0], 0)
            np.testing.assert_allclose(partial[1:3, 1, 0], 2.0)  # exclude sorted(0, 1)

            completed, done = atlas._apply_inner_predictors(
                bundle, output, metadata, device="cpu", contract_sha256="b" * 64,
            )
            self.assertEqual(completed, 3)
            self.assertTrue(done)
            resumed = np.load(output / "learned_inner/backgrounds.npy")
            np.testing.assert_array_equal(resumed[:3], partial[:3])
            np.testing.assert_allclose(resumed[4:6, 0, 0], 2.0)  # g=1, f=0: same pair
            active = np.load(output / "learned_inner/active.npy")
            np.testing.assert_array_equal(active[3], False)
            np.testing.assert_array_equal(active[4:6, 1], False)
            np.testing.assert_array_equal(active[4:6, [0, 2, 3, 4]], True)
            manifest = json.loads((output / "learned_inner/MANIFEST.json").read_text())
            self.assertEqual(manifest["axes"]["1"], "inner_fold_f")
            self.assertEqual(manifest["axes"]["2"], ["ridge", "tcn"])
            self.assertEqual(len(manifest["implementation_sha256"]), 64)
            with self.assertRaisesRegex(atlas.AtlasContractError, "immutable manifest drift"):
                atlas._apply_inner_predictors(
                    bundle, output, metadata, device="cpu", contract_sha256="c" * 64,
                )

    def test_learned_predictor_design_does_not_observe_final_answer_length(self):
        prefix = np.arange(24, dtype=float).reshape(6, 4)
        extended = np.vstack((prefix, np.full((5, 4), 10_000.0)))
        short, _ = atlas._ridge_design(prefix)
        long, _ = atlas._ridge_design(extended)
        np.testing.assert_allclose(short, long[:len(prefix)], atol=0, rtol=0)
        # 16*4 primitive-history values, 16 observed-mask values, and a bias;
        # there is deliberately no normalized full-answer-position column.
        self.assertEqual(short.shape[1], 16 * 4 + 16 + 1)

    def test_atomic_extraction_covers_registry_and_short_steps(self):
        rng = np.random.default_rng(8)
        n = 12
        raw = np.sort(rng.uniform(-8, -2, size=(n, 50)), axis=1)[:, ::-1]
        probability = np.exp(raw)
        probability /= 1.2 * probability.sum(axis=1, keepdims=True)
        logprobs = np.log(probability)
        ids = np.tile(np.arange(100, 150), (n, 1))
        generated = ids[:, 0].copy()
        # Three real digit opportunities with a different digit at rank one.
        generated[::4] = 15
        ids[::4, 0] = 16
        selected = np.where(generated == ids[:, 0], -logprobs[:, 0], 5.0)
        entropy = np.ones(n)
        from spectral_utils.renyi_locator_feature_bank import feature_matrix
        levels = feature_matrix(logprobs, entropy)["matrix"][:, :4]
        spans = np.array([[0, 1], [1, 4], [4, 12]])
        output = atlas.extract_atomic_answer(
            levels, logprobs, ids, generated, selected, entropy, spans, "fixture",
        )
        self.assertIn("answer__tail15.answer_prominence", output)
        self.assertIn("token__digit.opportunity_clock_innovation__active", output)
        self.assertEqual(output["step__q15.H0lim__readouts"].shape, (3, 15))
        self.assertEqual(output["decision__q15.H0lim__readouts"].shape, (15, 3))
        self.assertTrue(np.isfinite(output["step__q15.H0lim__readouts"]).all())

    def test_gate_ranks_are_recomputed_inside_each_bootstrap_draw(self):
        scores = np.array([[0., 3.], [1., 2.], [2., 1.], [3., 0.]])
        callback = atlas.gate_reranking_callback(scores, ["pb_x"] * 4)
        first = callback([0, 0, 3, 3])
        second = callback([1, 1, 2, 2])
        self.assertEqual(callback.calls["count"], 2)
        np.testing.assert_allclose(first["rank_fused"], [.5, .5, .5, .5])
        np.testing.assert_allclose(second["rank_fused"], [.5, .5, .5, .5])
        # Per-view midranks for the duplicated draw are 1/6 and 5/6, not the
        # original full-sample percentiles 0 and 1.
        one_view = atlas.rerank_gate_draw(scores[:, 0], ["pb_x"] * 4, [0, 0, 3, 3])
        np.testing.assert_allclose(one_view["rank_fused"], [1/6, 1/6, 5/6, 5/6])

    def test_baseline_replay_is_fail_closed(self):
        valid = json.loads(json.dumps(atlas.BASELINE_EXPECTED))
        result = atlas.verify_baseline_replay(valid)
        self.assertEqual(result["status"], "PASS")
        valid["current"]["pb"] += 1e-3
        with self.assertRaises(atlas.AtlasContractError):
            atlas.verify_baseline_replay(valid)

    def test_four_arm_baseline_replay_uses_scores_extract_and_joined_labels(self):
        from spectral_utils.fusion_signal_registry import READOUT_NAMES

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = root / "contract"
            joined_root = contract / "results/localization_full_benchmark_v3/evaluation"
            baseline = root / "baseline_replay"
            extract = root / "extract"
            answers = extract / "answers"
            joined_root.mkdir(parents=True); baseline.mkdir(); answers.mkdir(parents=True)

            cells = [f"pb_family{index}_{panel}" for index in range(4)
                     for panel in ("q4", "q8")]
            records, targets, label_rows, score_rows, tail = [], [], [], [], []
            packed_rows = []
            for cell in cells:
                for clean in (True, False):
                    index = len(records)
                    records.append({"uid": f"u{index}", "cell": cell, "steps": 2})
                    targets.append(-1 if clean else 0)
                    label_rows.append(np.array([-2, -2], dtype=np.int8))
                    score_rows.append(np.array([1.0, 0.0]))
                    tail.append(0.0 if clean else 1.0)
                    packed_rows.append((np.array([1.0, 1.0]) if clean else np.zeros(2),
                                        np.ones(2)))
            for prm_index, labels in enumerate(
                    (np.array([0, 1], dtype=np.int8), np.array([1, 0], dtype=np.int8))):
                index = len(records)
                records.append({"uid": f"u{index}", "cell": "prmbench_fixture", "steps": 2})
                targets.append(-2); label_rows.append(labels); score_rows.append(np.array([0.0, 1.0]))
                tail.append(0.0)
                packed_rows.append((np.array([0.0, 1.0]) if prm_index == 0 else np.zeros(2),
                                    np.ones(2)))

            offsets = np.arange(0, 2 * len(records) + 1, 2, dtype=np.int64)
            labels = np.concatenate(label_rows)
            score = np.concatenate(score_rows)
            target = np.asarray(targets, dtype=np.int32)
            (joined_root / "JOINED.json").write_text(
                json.dumps({"records": records}), encoding="utf8",
            )
            np.savez(joined_root / "JOINED.npz", offsets=offsets, target=target, labels=labels)
            pb = np.asarray([row["cell"].startswith("pb_") for row in records])
            cell_array = np.asarray([row["cell"] for row in records])
            tail_rank = atlas._baseline_gate_percentiles(np.asarray(tail), cell_array, pb)
            np.savez_compressed(
                baseline / "SCORES_FROZEN.npz",
                steps__mean__H0lim_VE0_VE075_VE1=score,
                steps__append_innovation__H0lim=score,
                gate_raw=np.asarray(tail), gate_percentile=tail_rank,
            )

            top10 = READOUT_NAMES.index("top10")
            recorded = np.empty(len(records), dtype="S64")
            for index, ((digit, opportunity), local_score) in enumerate(zip(packed_rows, score_rows)):
                matrix = np.zeros((2, len(READOUT_NAMES)), dtype=np.float32)
                matrix[:, top10] = local_score
                digit_matrix = np.zeros_like(matrix); digit_matrix[:, top10] = digit
                payload = {
                    **{f"step__{name}__readouts": matrix for name in
                       ("q15.H0lim", "q15.VE0", "q15.VE0.75", "q15.VE1")},
                    "step__q15.H0lim.prefix_mean_innovation__readouts": matrix,
                    "step__digit.disagreement__readouts": digit_matrix,
                    "token__digit.disagreement__values": digit.astype(np.float32),
                    "token__digit.opportunity__values": opportunity.astype(np.float32),
                }
                path = answers / f"{index:05d}.npz"
                np.savez_compressed(path, **payload)
                recorded[index] = atlas.sha256_file(path).encode("ascii")
            np.save(extract / "completion.npy", np.ones(len(records), dtype=bool))
            np.save(extract / "answer_sha256.npy", recorded)
            (extract / "MANIFEST.json").write_text('{"schema":"fixture"}\n', encoding="utf8")
            (extract / "STATUS.json").write_text('{"status":"COMPLETE"}\n', encoding="utf8")

            replay = atlas.replay_frozen_baselines(
                contract, baseline, extract, strict_roster=False,
            )
            self.assertEqual(set(replay["metrics"]), set(atlas.BASELINE_EXPECTED))
            self.assertEqual(replay["metrics"]["original4"]["pb"], 1.0)
            self.assertEqual(replay["metrics"]["innovation5"]["pb"], 1.0)
            self.assertEqual(replay["metrics"]["digit025"]["pb"], 1.0)
            self.assertEqual(replay["metrics"]["current"]["pb"], 0.0)
            self.assertEqual(replay["metrics"]["digit025"]["within"], 0.5)
            self.assertEqual(replay["metrics"]["current"]["within"], 0.5)
            from spectral_utils.temporal_context_models import residual_step_score
            prm_start = offsets[16]
            np.testing.assert_allclose(
                replay["per_answer"]["steps__digit025"][prm_start:prm_start + 2],
                residual_step_score(np.array([0.0, 1.0]), np.array([0.0, 1.0]), .25),
            )
            self.assertEqual(replay["audit"]["pb_cells"], sorted(cells))
            self.assertEqual(replay["audit"]["tail_gate_open"], 8)
            self.assertEqual(replay["audit"]["current_gate_open"], 16)
            self.assertTrue(replay["provenance"]["development_only"])
            self.assertEqual(len(replay["provenance"]["implementation_sha256"]), 64)
            self.assertEqual(len(replay["provenance"]["score_inputs"]), 5)
            self.assertEqual(len(replay["provenance"]["label_inputs"]), 2)

            replay_path = root / "PER_ANSWER.npz"
            atlas._bind_baseline_replay_arrays(replay_path, replay["per_answer"])
            atlas._bind_baseline_replay_arrays(replay_path, replay["per_answer"])
            changed = dict(replay["per_answer"])
            changed["digit_rate"] = changed["digit_rate"].copy()
            changed["digit_rate"][0] = -1
            with self.assertRaisesRegex(atlas.AtlasContractError, "array drift"):
                atlas._bind_baseline_replay_arrays(replay_path, changed)

    def test_baseline_prmb_orientation_matches_frozen_positive_class(self):
        records = [
            {"cell": "pb_fixture_q4"},
            {"cell": "pb_fixture_q4"},
            {"cell": "prmbench_fixture"},
        ]
        metric, _ = atlas._baseline_arm_metrics(
            records,
            np.asarray([0, 2, 4, 6], dtype=np.int64),
            np.asarray([-1, 1, -1], dtype=np.int64),
            np.asarray([-1, -1, -1, -1, 0, 1], dtype=np.int8),
            np.asarray([0.1, 0.2, 0.1, 0.9, 0.1, 0.9]),
            np.asarray([False, True, False]),
        )
        self.assertEqual(metric["within"], 1.0)

    def test_historical_score_and_peak_identities_match_inventory_contract(self):
        values = np.asarray([0.1, 0.9, 0.2, 0.3, 0.8], dtype=np.float32)
        offsets = np.asarray([0, 2, 5], dtype=np.int64)
        expected_score = hashlib.sha256(
            np.asarray(values, dtype="<f8").tobytes()
        ).hexdigest()
        expected_peak = hashlib.sha256(
            np.asarray([1, 2], dtype="<i4").tobytes()
        ).hexdigest()
        self.assertEqual(atlas._historical_score_identity(values), expected_score)
        self.assertEqual(atlas._historical_peak_identity(values, offsets), expected_peak)

    def test_factorial_has_all_32_unique_arms(self):
        self.assertEqual(
            atlas.FUSION_POINTS,
            (
                "background", "token_pre_readout", "step_post_readout",
                "decoder", "answer_gate",
            ),
        )
        arms = atlas.factorial_arms()
        self.assertEqual(len(arms), 32)
        self.assertEqual(len({row["arm"] for row in arms}), 32)
        self.assertEqual(arms[0]["arm"], "00000")
        self.assertEqual(arms[-1]["arm"], "11111")
        self.assertTrue(all(set(row["selection"]) == set(atlas.FUSION_POINTS) for row in arms))

    def test_complete_factorial_marks_report_development_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for stage in ("reconcile", "fusion", "ablation"):
                (output / stage).mkdir(parents=True)
            (output / "reconcile/LEDGER.json").write_text(json.dumps({
                "historical_unique_score_arrays": 178,
                "historical_unique_peak_vectors": 173,
                "report_only_archives": 10,
            }), encoding="utf8")
            (output / "fusion/SUMMARY.json").write_text(
                json.dumps({"pareto": []}), encoding="utf8",
            )
            (output / "ablation/FACTORIAL.json").write_text(json.dumps({
                "status": "COMPLETE", "arm_count": 32,
            }), encoding="utf8")
            args = atlas.parser().parse_args(["report", "--output-root", str(output)])
            args.output_root = output
            atlas.stage_report(args)
            report = (output / "report/REPORT.md").read_text(encoding="utf8")
            self.assertIn("DEVELOPMENT_COMPLETE", report)

    def test_unresolved_composition_writes_blocked_factorial_and_partial_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for stage in ("reconcile", "fusion"):
                (output / stage).mkdir(parents=True)
            (output / "reconcile/LEDGER.json").write_text(json.dumps({
                "historical_unique_score_arrays": 178,
                "historical_unique_peak_vectors": 173,
                "report_only_archives": 9,
            }), encoding="utf8")
            (output / "fusion/COMPOSITION_CONTRACT.json").write_text(
                json.dumps({"status": "UNRESOLVED"}), encoding="utf8",
            )
            finalists = {point: {"name": point + "-finalist"} for point in atlas.FUSION_POINTS}
            (output / "fusion/SUMMARY.json").write_text(json.dumps({
                "pareto": [], "finalists": finalists,
                "factorial_results": [],
                "factorial_status": "UNRESOLVED_PIPELINE_COMPOSITION_NOT_FABRICATED",
            }), encoding="utf8")
            args = atlas.parser().parse_args(["ablation", "--output-root", str(output)])
            args.output_root = output
            atlas.stage_ablation(args)
            factorial = json.loads((output / "ablation/FACTORIAL.json").read_text())
            self.assertEqual(factorial["status"], "BLOCKED_COMPOSITION_CONTRACT")
            self.assertEqual(factorial["factorial_results"], [])
            self.assertFalse(factorial["fabricated_arms"])
            self.assertEqual(
                json.loads((output / "ablation/STATUS.json").read_text())["status"],
                "BLOCKED_COMPOSITION_CONTRACT",
            )

            args = atlas.parser().parse_args(["report", "--output-root", str(output)])
            args.output_root = output
            atlas.stage_report(args)
            report = (output / "report/REPORT.md").read_text(encoding="utf8")
            self.assertIn("DEVELOPMENT_PARTIAL", report)
            self.assertIn("Factorial arms evaluated: 0", report)
            self.assertIn("blocked rather than fabricated", report)

    def test_cli_stage_dependencies_fail_before_stage_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaisesRegex(atlas.AtlasContractError, "requires completed stages: preflight"):
                atlas.main(["registry", "--output-root", str(output)])
            (output / "preflight").mkdir()
            (output / "preflight/STATUS.json").write_text('{"status":"COMPLETE"}\n', encoding="utf8")
            self.assertEqual(atlas.main(["registry", "--output-root", str(output)]), 0)
            self.assertEqual(json.loads((output / "registry/STATUS.json").read_text())["status"], "COMPLETE")
            with self.assertRaisesRegex(atlas.AtlasContractError, "bundle"):
                atlas.main(["extract", "--output-root", str(output)])

    def test_dependence_failure_does_not_write_complete_marker(self):
        from spectral_utils.error_dependence import AtlasBackendError

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for stage in ("extract", "predictors", "reconcile"):
                (output / stage).mkdir(parents=True)
                (output / stage / "STATUS.json").write_text('{"status":"COMPLETE"}\n', encoding="utf8")
            with self.assertRaises((atlas.AtlasContractError, AtlasBackendError)):
                atlas.main(["dependence", "--output-root", str(output), "--draws", "2"])
            self.assertEqual(json.loads((output / "dependence/STATUS.json").read_text())["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
