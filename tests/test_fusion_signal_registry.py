from dataclasses import FrozenInstanceError
import inspect
import unittest

import numpy as np

import spectral_utils.fusion_signal_registry as registry_module
from spectral_utils.fusion_signal_registry import (
    BUILTIN_REGISTRY,
    BACKGROUND_KINDS,
    BACKGROUND_SIGNAL_NAMES,
    DECODER_NAMES,
    DIRECT_PROBABILITY_NAMES,
    DIGIT_INNOVATION_NAMES,
    DIGIT_SIGNAL_NAMES,
    FusionSetSpec,
    FusionSignalRegistry,
    Q15_PRIMITIVE_NAMES,
    Q15_PREFIX_INNOVATION_NAMES,
    Q50_H1_HINF_NAMES,
    READOUT_NAMES,
    RENYI_ESCORT_NAMES,
    RegistryValidationError,
    STEP395_SIGNAL_NAMES,
    TAIL15_ROLE_NAMES,
    SignalSpec,
    answer_top10_minus_mean,
    array_sha256,
    build_builtin_registry,
    canonical_json,
    causal_prefix_topk_innovation,
    decode_argmax,
    decode_earlier_ve_peak,
    decode_first_near_max,
    decode_persistent_q90_3,
    decode_step_top5,
    digit_clock_innovations,
    equal_rank_fusion,
    family_equal_fusion,
    make_fusion_set_spec,
    nonnegative_shrunk_simplex_fusion,
    pending_expansion_signal_spec,
    prefix_mean_innovation,
    readout_steps,
    report_only_signal_spec,
    tail15_causal_innovations,
    trailing_mean_innovation,
    validate_fusion_members,
)


def _spec(name="signal.a", *, point="token_pre_readout", resolution="token", target="risk"):
    digest = array_sha256(np.frombuffer(name.encode(), dtype=np.uint8))
    return SignalSpec(
        name=name,
        provenance_family="fixture",
        insertion_point=point,
        resolution=resolution,
        target=target,
        transform="fixture identity",
        access_scope="answer_only",
        orientation="high_is_risk",
        mask_semantics="all_active",
        status="ELIGIBLE",
        provenance=("tests/test_fusion_signal_registry.py",),
        implementation_hash=digest,
        semantic_hash=digest,
    )


class FusionSignalRegistryTests(unittest.TestCase):
    def test_all_exports_exist_and_import_star_succeeds(self):
        namespace = {}
        exec("from spectral_utils.fusion_signal_registry import *", namespace)
        for name in registry_module.__all__:
            self.assertTrue(hasattr(registry_module, name), name)
            self.assertIn(name, namespace)

    def test_specs_are_frozen_canonical_and_hash_bound(self):
        left = _spec()
        right = _spec()
        self.assertEqual(left, right)
        self.assertEqual(left.spec_hash, right.spec_hash)
        self.assertEqual(
            canonical_json({"b": 2, "a": 1}),
            canonical_json({"a": 1, "b": 2}),
        )
        with self.assertRaises(FrozenInstanceError):
            left.name = "changed"
        float32 = np.asarray([1.0, 2.0], dtype=np.float32)
        bound = left.bind_score(float32)
        self.assertEqual(bound.score_hash, array_sha256(float32))
        self.assertNotEqual(
            bound.score_hash,
            array_sha256(np.asarray([1.0, 2.0], dtype=np.float64)),
        )

    def test_registry_roster_cardinality_status_and_exact_aliases(self):
        self.assertEqual(len(Q15_PRIMITIVE_NAMES), 4)
        self.assertEqual(len(Q15_PREFIX_INNOVATION_NAMES), 4)
        self.assertEqual(len(BACKGROUND_KINDS), 6)
        self.assertEqual(len(BACKGROUND_SIGNAL_NAMES), 24)
        self.assertEqual(len(RENYI_ESCORT_NAMES), 31)
        self.assertEqual(len(Q50_H1_HINF_NAMES), 3)
        self.assertEqual(len(STEP395_SIGNAL_NAMES), 9)
        self.assertEqual(len(DIRECT_PROBABILITY_NAMES), 17)
        self.assertEqual(len(DIGIT_SIGNAL_NAMES), 6)
        self.assertEqual(len(DIGIT_INNOVATION_NAMES), 2)
        self.assertEqual(len(TAIL15_ROLE_NAMES), 4)
        self.assertEqual(len(READOUT_NAMES), 15)
        self.assertEqual(len(DECODER_NAMES), 5)
        self.assertEqual(BUILTIN_REGISTRY.signal("renyi_escort.H1").name, "q15.H1_native")
        self.assertEqual(
            BUILTIN_REGISTRY.signal("step395.surprisal").name,
            "direct_probability.selected_token_surprisal",
        )
        self.assertEqual(BUILTIN_REGISTRY.signal("step395.digit").name, "digit.disagreement")
        canonical_names = {spec.name for spec in BUILTIN_REGISTRY.signals}
        self.assertTrue(set(Q15_PRIMITIVE_NAMES).issubset(canonical_names))
        self.assertTrue(set(Q15_PREFIX_INNOVATION_NAMES).issubset(canonical_names))
        self.assertTrue(set(BACKGROUND_SIGNAL_NAMES).issubset(canonical_names))
        self.assertEqual(
            BUILTIN_REGISTRY.signal("tail15.level").name,
            "direct_probability.residual_tail_mass",
        )
        self.assertEqual(BUILTIN_REGISTRY.signal("tail15.answer_prominence").resolution, "answer")
        self.assertEqual(
            BUILTIN_REGISTRY.signal("background.H0lim.source_excluded_tcn").access_scope,
            "source_excluded",
        )
        self.assertEqual(BUILTIN_REGISTRY.signal("pending.fm").status, "PENDING_EXPANSION")
        self.assertEqual(BUILTIN_REGISTRY.signal("pending.diflo").access_scope, "not_available")
        self.assertEqual(build_builtin_registry().registry_hash, BUILTIN_REGISTRY.registry_hash)

    def test_report_only_and_pending_hooks_fail_closed_for_fusion(self):
        report = report_only_signal_spec(
            "history.unexplained", "history", transform="saved opaque scores",
            provenance=("artifact.json",),
        )
        pending = pending_expansion_signal_spec("pending.fixture", "fixture")
        self.assertEqual(report.status, "REPORT_ONLY")
        self.assertEqual(pending.status, "PENDING_EXPANSION")
        with self.assertRaisesRegex(RegistryValidationError, "not eligible"):
            validate_fusion_members((report,))
        with self.assertRaisesRegex(RegistryValidationError, "not eligible"):
            validate_fusion_members((pending,))

    def test_score_hash_dedup_is_order_stable_and_prefers_eligible(self):
        scores = np.asarray([0.1, 0.4, 0.2], dtype=np.float32)
        left = _spec("signal.left").bind_score(scores)
        right = _spec("signal.right").bind_score(scores)
        first = FusionSignalRegistry.build((right, left), include_registered_aliases=False)
        second = FusionSignalRegistry.build((left, right), include_registered_aliases=False)
        self.assertEqual(first.registry_hash, second.registry_hash)
        self.assertEqual(first.signal("signal.right").name, "signal.left")

        opaque = report_only_signal_spec(
            "history.opaque", "fixture", transform="opaque historical score",
            provenance=("history.json",), insertion_point="token_pre_readout",
            resolution="token", mask_semantics="all_active",
            access_scope="answer_only", target="risk", score_hash=left.score_hash,
        )
        preferred = FusionSignalRegistry.build(
            (opaque, right), include_registered_aliases=False,
        )
        self.assertEqual(preferred.signal("history.opaque").name, "signal.right")
        self.assertEqual(preferred.signal("history.opaque").status, "ELIGIBLE")

    def test_fusion_rejects_mixed_points_resolutions_and_targets(self):
        cases = (
            (_spec("signal.step", point="step_post_readout", resolution="step"), "insertion"),
            (_spec("signal.target", target="other"), "targets"),
        )
        for other, message in cases:
            with self.subTest(other=other.name):
                with self.assertRaisesRegex(RegistryValidationError, message):
                    validate_fusion_members((_spec(), other))

    def test_registry_revalidates_fusion_metadata(self):
        members = (_spec(), _spec("signal.b"))
        fusion = make_fusion_set_spec("set.ab", members, head="equal_rank")
        registry = FusionSignalRegistry.build(members, fusion_sets=(fusion,))
        self.assertEqual(registry.fusion_sets[0].members, ("signal.a", "signal.b"))
        bad = FusionSetSpec(
            name="set.bad",
            members=fusion.members,
            provenance_families=fusion.provenance_families,
            insertion_point=fusion.insertion_point,
            resolution=fusion.resolution,
            target="wrong",
            head=fusion.head,
            access_scope=fusion.access_scope,
            orientation=fusion.orientation,
            mask_semantics=fusion.mask_semantics,
            status=fusion.status,
            provenance=fusion.provenance,
            implementation_hash=fusion.implementation_hash,
            semantic_hash=fusion.semantic_hash,
        )
        with self.assertRaisesRegex(RegistryValidationError, "metadata drift"):
            FusionSignalRegistry.build(members, fusion_sets=(bad,))

    def test_prefix_and_trailing_innovations_are_strictly_causal(self):
        values = np.asarray([1.0, 3.0, 7.0, 15.0])
        prefix, prefix_active = prefix_mean_innovation(values)
        np.testing.assert_allclose(prefix, [0.0, 2.0, 5.0, 15.0 - 11.0 / 3.0])
        np.testing.assert_array_equal(prefix_active, [False, True, True, True])
        trailing, trailing_active = trailing_mean_innovation(values, window=2)
        np.testing.assert_allclose(trailing, [0.0, 2.0, 5.0, 10.0])
        np.testing.assert_array_equal(trailing_active, [False, True, True, True])
        changed = values.copy()
        changed[2:] = 999.0
        np.testing.assert_array_equal(prefix_mean_innovation(changed)[0][:2], prefix[:2])
        np.testing.assert_array_equal(trailing_mean_innovation(changed, 2)[0][:2], trailing[:2])

    def test_masked_prefix_and_tail15_top10_do_not_use_future_or_inactive_values(self):
        values = np.arange(1.0, 14.0)
        mask = np.ones(len(values), dtype=bool)
        mask[2] = False
        masked = values.copy()
        masked[2] = np.nan
        innovation, active = causal_prefix_topk_innovation(masked, 10, mask)
        self.assertFalse(active[0])
        self.assertFalse(active[2])
        expected = np.mean([2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        self.assertAlmostEqual(innovation[-1], 13.0 - expected)
        tail = tail15_causal_innovations(masked, mask)
        np.testing.assert_array_equal(tail["prefix_top10"][1], active)
        altered = masked.copy()
        altered[-1] = -1000.0
        np.testing.assert_allclose(
            tail15_causal_innovations(altered, mask)["prefix_top10"][0][:-1],
            tail["prefix_top10"][0][:-1],
        )

    def test_digit_token_and_opportunity_clocks_have_distinct_masks(self):
        disagreement = np.asarray([0, 0, 1, 0, 0, 1], dtype=float)
        opportunity = np.asarray([False, True, True, False, True, True])
        result = digit_clock_innovations(disagreement, opportunity)
        token, token_active = result["token_clock"]
        chance, chance_active = result["opportunity_clock"]
        np.testing.assert_array_equal(token_active, [False, True, True, False, True, True])
        np.testing.assert_array_equal(chance_active, [False, False, True, False, True, True])
        self.assertAlmostEqual(token[2], 1.0)
        self.assertAlmostEqual(chance[2], 1.0)
        self.assertAlmostEqual(chance[4], -0.5)
        with self.assertRaisesRegex(ValueError, "without an opportunity"):
            digit_clock_innovations([1, 0], [False, True])

    def test_all_fifteen_readouts_and_short_step_rules(self):
        values = np.asarray([
            1.0, 2.0, 3.0, 4.0, 5.0, 50.0,
            6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        ])
        spans = np.asarray([[0, 3], [3, 12]])
        outputs = {name: readout_steps(values, spans, name)[0] for name in READOUT_NAMES}
        self.assertEqual(set(outputs), set(READOUT_NAMES))
        self.assertAlmostEqual(outputs["top10"][0], 2.0)
        self.assertAlmostEqual(outputs["first4"][0], 2.0)
        self.assertAlmostEqual(outputs["last4"][0], 2.0)
        self.assertAlmostEqual(outputs["best_contiguous10"][0], 2.0)
        self.assertAlmostEqual(outputs["top1"][1], 50.0)
        self.assertAlmostEqual(outputs["top25pct"][1], np.mean([50.0, 11.0, 10.0]))
        self.assertAlmostEqual(outputs["top50pct"][1], np.mean([50.0, 11.0, 10.0, 9.0, 8.0]))
        np.testing.assert_allclose(readout_steps(values, spans, "top25%")[0], outputs["top25pct"])

    def test_readout_masks_constants_and_no_future(self):
        values = np.arange(10.0)
        spans = np.asarray([[0, 5], [5, 10]])
        mask = np.asarray([False] * 5 + [True] * 5)
        score, active = readout_steps(values, spans, "mean", active_mask=mask)
        np.testing.assert_array_equal(active, [False, True])
        np.testing.assert_allclose(score, [0.0, 7.0])
        changed = values.copy()
        changed[5:] = 999.0
        for name in READOUT_NAMES:
            with self.subTest(readout=name):
                self.assertAlmostEqual(
                    readout_steps(changed, spans, name)[0][0],
                    readout_steps(values, spans, name)[0][0],
                )

    def test_overlapping_monotone_logical_steps_are_read_independently(self):
        # Frozen PRMBench contains three answers where an otherwise empty
        # logical step is represented by a one-token span sharing the prior
        # step's terminal token.  Keep both step rows aligned to their labels.
        values = np.arange(12.0)
        spans = np.asarray([[0, 9], [8, 9], [8, 9], [9, 12]])
        score, active = readout_steps(values, spans, "mean")
        np.testing.assert_allclose(score, [4.0, 8.0, 8.0, 10.0])
        np.testing.assert_array_equal(active, [True, True, True, True])
        with self.assertRaisesRegex(ValueError, "monotone chronological"):
            readout_steps(values, np.asarray([[4, 8], [3, 9]]), "mean")

    def test_answer_prominence_is_constant_shift_invariant_and_cannot_move_peak(self):
        values = np.arange(20.0)
        prominence = answer_top10_minus_mean(values)
        self.assertAlmostEqual(prominence, answer_top10_minus_mean(values + 123.0))
        step_scores = np.asarray([0.2, 1.0, 0.7])
        self.assertEqual(decode_argmax(step_scores + prominence), decode_argmax(step_scores))
        self.assertEqual(answer_top10_minus_mean(np.ones(20)), 0.0)

    def test_decoders_are_stable_and_short_safe(self):
        scores = np.asarray([0.0, 0.8, 1.0, 0.95])
        self.assertEqual(decode_argmax(scores), 2)
        self.assertEqual(decode_first_near_max(scores), 2)
        persistent = np.asarray([0.0, 1.0, 1.0, 1.0, 0.0])
        self.assertEqual(decode_persistent_q90_3(persistent), 1)
        self.assertEqual(decode_persistent_q90_3([0.0, 2.0]), 1)
        tokens = np.asarray([1.0, 2.0, 20.0, 19.0, 18.0])
        self.assertEqual(decode_step_top5(tokens, [[0, 2], [2, 5]]), 1)
        self.assertEqual(decode_earlier_ve_peak([0, 3, 1], [0, 1, 4]), 1)

    def test_positive_fusions_mask_constants_and_are_identity_on_one_live_view(self):
        varying = np.asarray([3.0, 1.0, 2.0, 4.0])
        scores = np.column_stack((varying, np.ones(4), varying[::-1]))
        equal = equal_rank_fusion(scores[:, :2])
        self.assertEqual(equal.active_members.tolist(), [True, False])
        np.testing.assert_allclose(equal.score, [0.75, 0.25, 0.5, 1.0])
        np.testing.assert_allclose(equal.weights, [1.0, 0.0])
        family = family_equal_fusion(scores, ["a", "a", "b"])
        self.assertTrue(np.all(family.weights >= 0))
        self.assertAlmostEqual(float(family.weights.sum()), 1.0)
        simplex = nonnegative_shrunk_simplex_fusion(scores)
        self.assertEqual(simplex.active_members.tolist(), [True, False, True])
        self.assertTrue(np.all(simplex.weights >= -1e-14))
        self.assertAlmostEqual(float(simplex.weights.sum()), 1.0)
        self.assertTrue(np.isfinite(simplex.score).all())

    def test_fusions_renormalize_per_row_masks_without_zero_evidence(self):
        scores = np.asarray([[1.0, 4.0], [2.0, 3.0], [3.0, np.nan], [4.0, np.nan]])
        mask = np.isfinite(scores)
        result = equal_rank_fusion(scores, active_mask=mask)
        self.assertTrue(result.active.all())
        self.assertTrue(np.isfinite(result.score).all())
        with self.assertRaisesRegex(ValueError, "explicit inactive mask"):
            equal_rank_fusion(scores)

    def test_label_firewall_rejects_unexpected_labels(self):
        public = (
            prefix_mean_innovation, trailing_mean_innovation, digit_clock_innovations,
            tail15_causal_innovations, readout_steps, answer_top10_minus_mean,
            equal_rank_fusion, family_equal_fusion, nonnegative_shrunk_simplex_fusion,
            make_fusion_set_spec,
        )
        self.assertTrue(all("label" not in inspect.signature(function).parameters for function in public))
        with self.assertRaises(TypeError):
            prefix_mean_innovation([1.0, 2.0], labels=[0, 1])
        with self.assertRaises(TypeError):
            equal_rank_fusion([[1.0], [2.0]], labels=[0, 1])


if __name__ == "__main__":
    unittest.main()
