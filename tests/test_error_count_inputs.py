"""Known-answer fixtures for the donor-only absolute evidence bridge."""
import copy
from decimal import Decimal, localcontext
import json
import unittest

import numpy as np

from spectral_utils.error_count_inputs import (
    answer_standardize, apply_vote_representation, fit_vote_representation,
)
import spectral_utils.error_count_inputs as bridge


class VoteRepresentationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(118)
        self.off = np.arange(0, 145, 6)
        self.raw = rng.normal(size=(144, 4)) + np.repeat(rng.normal(size=(24, 4)), 6, axis=0)
        self.active = np.ones(self.raw.shape, dtype=bool)
        self.active[::3, 3] = False
        self.raw[~self.active] = 0.
        self.V = answer_standardize(self.raw, self.off, self.active)
        self.groups = [
            {'columns': [0, 1], 'weights': [.8, -.6]},
            {'columns': [2], 'weights': [1.]},
            {'columns': [3], 'weights': [1.]},
        ]
        self.cross = [.6, -.4, .2]
        self.model = fit_vote_representation(self.raw, self.V, self.active, self.off,
                                             self.groups, cross_weights=self.cross)

    def apply(self, raw=None, V=None, active=None, off=None, kind='absolute'):
        return apply_vote_representation(self.model,
            self.raw if raw is None else raw, self.V if V is None else V,
            self.active if active is None else active, self.off if off is None else off,
            kind=kind)

    def test_inverse_and_affine_reconstruction_and_json_roundtrip(self):
        model = json.loads(json.dumps(self.model, allow_nan=False))
        scores, votes, diag = apply_vote_representation(model, self.raw, self.V,
                                                       self.active, self.off)
        self.assertEqual(votes.dtype, np.int8)
        self.assertEqual(scores.shape, (144, 3))
        for name in ['canonical_V_max_abs', 'inverse_V_max_abs', 'virtual_group_max_abs',
                     'masked_affine_max_abs', 'cross_fused_max_abs']:
            self.assertLessEqual(self.model['diagnostics'][name], 1e-9)
            self.assertLessEqual(diag[name], 1e-9)
        np.testing.assert_array_equal(votes, np.where(scores > model['absolute']['threshold'], 1, -1))
        self.assertEqual(model['group_weights'], self.groups)

    def test_local_scores_equal_historical_answer_z(self):
        virtual = np.column_stack([self.V[:, g['columns']] @ np.array(g['weights'])
                                   for g in self.groups])
        expected = np.zeros_like(virtual)
        for a, b in zip(self.off[:-1], self.off[1:]):
            block = virtual[a:b]
            expected[a:b] = (block - block.mean(0)) / block.std(0)
        scores, _, _ = self.apply(kind='local')
        np.testing.assert_allclose(scores, expected, atol=1e-14, rtol=0)
        np.testing.assert_allclose(self.model['local']['threshold'],
                                   np.quantile(expected, .8, axis=0, method='linear'), atol=1e-14, rtol=0)

    def test_answer_shift_is_retained_only_by_absolute_representation(self):
        shifted = self.raw.copy()
        shifted[:6] += np.where(self.active[:6], 50., 0.)
        shifted_v = answer_standardize(shifted, self.off, self.active)
        np.testing.assert_allclose(shifted_v, self.V, atol=2e-14, rtol=0)
        before, _, _ = self.apply()
        after, _, _ = self.apply(raw=shifted, V=shifted_v)
        self.assertGreater(float(np.max(np.abs(before[:6] - after[:6]))), 10.)
        local_before, _, _ = self.apply(kind='local')
        local_after, _, _ = self.apply(raw=shifted, V=shifted_v, kind='local')
        np.testing.assert_allclose(local_before, local_after, atol=3e-14, rtol=0)

    def test_digit_inactivity_is_neutral_and_not_used_for_moments(self):
        changed = self.raw.copy()
        changed[~self.active] = 1e6
        changed_v = answer_standardize(changed, self.off, self.active)
        np.testing.assert_array_equal(changed_v, self.V)
        changed_model = fit_vote_representation(changed, changed_v, self.active, self.off,
                                               self.groups, cross_weights=self.cross)
        self.assertEqual(changed_model, self.model)
        before = self.apply()
        after = self.apply(raw=changed, V=changed_v)
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        for a, b in zip(self.off[:-1], self.off[1:]):
            use = self.active[a:b, 3]
            np.testing.assert_allclose(self.V[a:b, 3][use].mean(), 0., atol=1e-15)
            np.testing.assert_array_equal(self.V[a:b, 3][~use], 0.)

    def test_donor_scales_have_population_within_answer_denominator(self):
        expected = np.zeros(4)
        for a, b in zip(self.off[:-1], self.off[1:]):
            for j in range(4):
                value = self.raw[a:b, j][self.active[a:b, j]]
                expected[j] += np.square(value - value.mean()).sum()
        expected = np.sqrt(expected / self.active.sum(0))
        np.testing.assert_allclose(self.model['r'], expected, atol=1e-15, rtol=0)

    def test_holdout_change_does_not_mutate_donor_state(self):
        before = json.dumps(self.model, sort_keys=True)
        for offset in [-12., 0., 37.]:
            raw = self.raw[:12].copy() + np.where(self.active[:12], offset, 0.)
            V = answer_standardize(raw, [0, 6, 12], self.active[:12])
            self.apply(raw=raw, V=V, active=self.active[:12], off=[0, 6, 12])
        self.assertEqual(before, json.dumps(self.model, sort_keys=True))

    def test_zero_and_all_votes_are_possible_without_answer_quota(self):
        for level, expected in [(-100., -1), (100., 1)]:
            raw = np.full((6, 4), level)
            active = np.ones_like(raw, dtype=bool)
            V = answer_standardize(raw, [0, 6], active)
            _, votes, diag = self.apply(raw=raw, V=V, active=active, off=[0, 6])
            np.testing.assert_array_equal(votes, expected)
            self.assertEqual(diag['zero_positive_vote_answers'], int(expected < 0))
            self.assertEqual(diag['all_positive_vote_answers'], int(expected > 0))

    def test_boundary_ties_are_negative(self):
        model = copy.deepcopy(self.model)
        scores, _, _ = self.apply()
        model['absolute']['threshold'] = scores[0].tolist()
        _, votes, diag = apply_vote_representation(model, self.raw, self.V, self.active, self.off)
        np.testing.assert_array_equal(votes[0], -1)
        self.assertTrue(all(n >= 1 for n in diag['threshold_tie_counts']))

    def test_nonfinite_and_bad_shapes_reject_without_fallback(self):
        raw = self.raw.copy()
        raw[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            self.apply(raw=raw)
        with self.assertRaisesRegex(ValueError, 'bool matrix'):
            self.apply(active=self.active.astype(int))
        for offsets in [[1, 144], [0, 0, 144], [0, 143], [0., 144.]]:
            with self.assertRaisesRegex(ValueError, 'offsets'):
                self.apply(off=offsets)
        with self.assertRaisesRegex(ValueError, 'Unknown vote representation'):
            self.apply(kind='unknown')

    def test_wrong_V_and_bad_group_partition_reject(self):
        wrong = self.V.copy()
        wrong[0, 0] += .1
        with self.assertRaisesRegex(RuntimeError, 'Canonical V reconstruction'):
            self.apply(V=wrong)
        groups = copy.deepcopy(self.groups)
        groups[1]['columns'] = [0]
        with self.assertRaisesRegex(ValueError, 'disjoint complete'):
            fit_vote_representation(self.raw, self.V, self.active, self.off, groups)

    def test_zero_required_scale_and_constant_group_reject(self):
        raw = self.raw.copy()
        raw[:, 2] = np.repeat(np.arange(24), 6)  # Between-answer variation cannot identify its old local weight.
        V = answer_standardize(raw, self.off, self.active)
        with self.assertRaisesRegex(ValueError, 'zero donor within-answer scale'):
            fit_vote_representation(raw, V, self.active, self.off, self.groups)
        raw = self.raw.copy()
        raw[:, 1] = raw[:, 0]
        groups = copy.deepcopy(self.groups)
        groups[0]['weights'] = [1., -1.]
        V = answer_standardize(raw, self.off, self.active)
        with self.assertRaisesRegex(ValueError, 'Constant donor group'):
            fit_vote_representation(raw, V, self.active, self.off, groups)

    def test_optional_cross_weights_and_no_labels_in_fit_interface(self):
        model = fit_vote_representation(self.raw, self.V, self.active, self.off, self.groups)
        self.assertIsNone(model['diagnostics']['cross_fused_max_abs'])
        with self.assertRaisesRegex(ValueError, 'cross_weights'):
            fit_vote_representation(self.raw, self.V, self.active, self.off, self.groups,
                                    cross_weights=[1.])
        with self.assertRaises(TypeError):
            fit_vote_representation(self.raw, self.V, self.active, self.off, self.groups,
                                    labels=np.zeros(144))

    def test_tiny_digit_variation_survives_larger_donor_centering(self):
        raw = self.raw.copy()
        raw[6:, 3] = np.where(self.active[6:, 3], .03 + .05 * raw[6:, 3], 0.)
        raw[:6, 3] = np.where(self.active[:6, 3],
                              [0., 2.4236843531351278e-11, 7e-11, 0., 2e-10, 5e-10], 0.)
        V = answer_standardize(raw, self.off, self.active)
        model = fit_vote_representation(raw, V, self.active, self.off, self.groups,
                                        cross_weights=self.cross)
        model = json.loads(json.dumps(model, allow_nan=False))
        _, _, diagnostics = apply_vote_representation(model, raw, V, self.active, self.off)
        self.assertEqual(model['version'], 2)
        self.assertEqual(diagnostics['coordinate_precision'], 'compensated_float64_high_plus_residual')
        for name in ['inverse_V_max_abs', 'virtual_group_max_abs', 'masked_affine_max_abs', 'cross_fused_max_abs']:
            self.assertLessEqual(model['diagnostics'][name], 1e-9)
            self.assertLessEqual(diagnostics[name], 1e-9)
        # The old single-value coordinate loses the tiny variation even though
        # direct raw-to-local normalization is well defined. Keep this witness
        # so a future audit change cannot hide cancellation by relaxing its bar.
        mu, r = np.asarray(model['mu']), np.asarray(model['r'])
        active = self.active[:6, 3]
        mean = raw[:6, 3][active].mean()
        sd = raw[:6, 3][active].std()
        old = (r[3] * ((raw[:6, 3][active] - mu[3]) / r[3]) + mu[3] - mean) / sd
        self.assertGreater(float(np.max(np.abs(old - V[:6, 3][active]))), 1e-9)
        high, residual = bridge._absolute_coordinates(raw, self.active, mu, r)
        with localcontext() as context:
            context.prec = 60
            exact = (Decimal.from_float(raw[1, 3]) - Decimal.from_float(mu[3])) / Decimal.from_float(r[3])
            actual = Decimal.from_float(high[1, 3]) + Decimal.from_float(residual[1, 3])
            self.assertLess(abs(exact - actual), Decimal('1e-30'))


if __name__ == '__main__':
    unittest.main()
