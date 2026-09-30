import unittest
from unittest.mock import patch
import numpy as np
from spectral_utils import fusion_sampling_replication as s
from spectral_utils.answer_localization_v2 import moment_plan, moment_matrix, prepare_local, json_safe
from spectral_utils.fusion_window_sampling import choose_all, top_indices


class SamplingReplicationTests(unittest.TestCase):
    def fixture(self, tokens):
        raw = np.random.default_rng(19).normal(size=(tokens, 29))
        plan = moment_plan(tokens, 8); values, names = moment_matrix(raw, plan)
        ss, ee = np.array([0, tokens//2]), np.array([tokens//2, tokens])
        detail = dict(valid=True, decision_valid=True, fixed_iu_valid=True,
                      prediction=-1, fixed_iu_prediction=-1, peak=0)
        meta = dict(routing={'routes': {'dual': 'moment_joint'}},
                    methods={'moment__joint0': detail, 'moment__iu': detail})
        anchors = {}; anchor_meta = {'methods': {}}
        for core, source in s.ANCHORS.items():
            anchors[source+'__window'] = np.arange(len(plan.starts), dtype=float)
            anchors[source+'__risk'] = np.array([1., 2.])
            anchor_meta['methods'][source] = dict(detail)
        return raw, ss, ee, {'moment__features': values}, meta, anchors, anchor_meta, 'fixture'

    def test_actual_selector_axes_budget_and_determinism(self):
        x = np.random.default_rng(23).normal(size=(40, 9))
        first, detail = choose_all(x, 'axis-fixture'); second, _ = choose_all(x, 'axis-fixture')
        self.assertEqual(set(first), set(s.SELECTORS))
        self.assertEqual(len(detail['dufs_transposed']['probabilities']), 40)
        self.assertEqual(detail['dufs_transposed']['graph_nodes'], 9)
        for selector in s.SELECTORS:
            indices = first[selector]
            self.assertEqual(len(indices), 40 if selector == 'full' else 32)
            self.assertTrue(np.all(np.diff(indices) > 0))
            np.testing.assert_array_equal(indices, second[selector])
        np.testing.assert_array_equal(top_indices(np.ones(7), 3), [0, 1, 2])

    def test_scoring_end_window_never_fits_and_bad_choices_fail(self):
        plan = moment_plan(267, 8)
        self.assertEqual(len(plan.starts), 34); self.assertEqual(len(plan.fit_indices), 33)
        selected = s.check_selection(np.arange(32), plan.fit_indices)
        self.assertNotIn(len(plan.starts)-1, selected)
        for bad in [np.array([0, 0]), np.array([2, 1]), np.array([-1]), np.array([33])]:
            with self.assertRaises(ValueError): s.check_selection(bad, plan.fit_indices)

    def test_fit_normalization_ignores_unselected_values(self):
        raw, _, _, _, _, _, _, _ = self.fixture(400)
        values, names = moment_matrix(raw, moment_plan(400, 8)); fi = np.arange(0, 50, 2)
        z, anchor, detail = prepare_local(values, names, fi)
        changed = values.copy(); changed[1::2] = 100*np.random.default_rng(8).normal(size=changed[1::2].shape)
        z2, anchor2, detail2 = prepare_local(changed, names, fi)
        np.testing.assert_array_equal(z[fi], z2[fi]); self.assertEqual(anchor, anchor2)
        self.assertEqual(json_safe(detail), json_safe(detail2))

    def test_all_row_outputs_are_exact_anchors(self):
        fixture = self.fixture(67)
        with patch.object(s, 'fit_bank', side_effect=AssertionError('unexpected refit')):
            arrays, methods, diagnostic = s.score_sampling(*fixture)
        self.assertFalse(diagnostic['eligible'])
        for selector in s.SELECTORS:
            for core, source in s.ANCHORS.items():
                arm = s.arm_name(selector, core)
                np.testing.assert_array_equal(arrays[arm+'__risk'], fixture[5][source+'__risk'])
                self.assertTrue(methods[arm]['anchor_replay'])

    def test_support_is_token_overlap_not_step_count(self):
        covered, fractions = s.selection_support(24, np.array([0, 8, 16]), np.array([8, 16, 24]),
            np.array([0, 2]), np.array([4, 8, 14]), np.array([12, 14, 20]))
        self.assertEqual(int(covered.sum()), 16)
        np.testing.assert_allclose(fractions, [.5, 0., 4/6])

    def test_dense_gate_support_and_model_only_fallback(self):
        fixture = self.fixture(267)
        def choices(values, identity):
            n = len(values)
            return ({k: np.arange(n if k == 'full' else 32) for k in s.SELECTORS},
                    {k: {'status': 'OK', 'diagnostics': {'per_seed_probabilities': [np.ones(n)]*3}} for k in s.SELECTORS})
        for joint_valid in (True, False):
            def fit(values, names, fi, identity):
                self.assertEqual(len(fi), 32)
                risks = {c: np.linspace(-1, 1, len(values)) for c in s.CORES}
                detail = {c: {'valid': c != 'graph010'} for c in s.CORES}
                detail['graph010']['reason'] = 'GRAPH_NUMERIC_FAILED'
                return {}, risks, detail, {'joint_valid': joint_valid, 'joint_failure': 'FIT_FAILED'}
            with patch.object(s, 'choose_all', side_effect=choices), patch.object(s, 'fit_bank', side_effect=fit), \
                 patch('spectral_utils.fusion_token_gap.mixture_readout', return_value={'prediction': 0}) as gate:
                arrays, methods, _ = s.score_sampling(*fixture)
            self.assertTrue(all(len(call.args[0]) == 33 for call in gate.call_args_list))
            graph = methods[s.arm_name('uniform', 'graph010')]
            self.assertEqual(graph['fallback_to_sample_iu'], not joint_valid)
            self.assertEqual(graph['valid'], not joint_valid)


if __name__ == '__main__': unittest.main()
