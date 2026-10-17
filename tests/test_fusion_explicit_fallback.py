"""Scientific failure/selection contracts for answer-only composite fusion."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fallback', ROOT / 'spectral_utils/fusion_explicit_fallback.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def fixture(moment=True, context=True):
    arrays, methods = {}, {}
    for i, arm in enumerate(f.PARENT_ARMS):
        valid = moment if arm.startswith('moment__') and arm.split('__')[1] in f.CORES else (
            context if arm.startswith('context__') and arm.split('__')[1] in f.CORES else True)
        methods[arm] = {'valid': valid, 'decision_valid': valid, 'fixed_iu_valid': valid,
                        'prediction': -1 if valid else None, 'fixed_iu_prediction': -1 if valid else None,
                        'peak': 1 if valid else None}
        if valid:
            arrays[arm + '__window'] = np.array([i + .1, i + .5, i - .2])
            arrays[arm + '__risk'] = np.array([i + .1, i + .5])
    return arrays, methods


class RoutingContract(unittest.TestCase):
    def test_all_eligibility_states_and_unknown_policy(self):
        expected = {(True, True): ('moment_joint', 'moment_joint'),
                    (True, False): ('moment_joint', 'moment_joint'),
                    (False, True): ('moment_iu', 'context_joint'),
                    (False, False): ('moment_iu', 'moment_iu')}
        for (m, c), routes in expected.items():
            self.assertEqual(tuple(f.choose_route(m, c, p) for p in ('single', 'dual')), routes)
        with self.assertRaises(ValueError): f.choose_route(True, True, 'best_score')
        with self.assertRaises(TypeError): f.choose_route(.8, True, 'dual')

    def test_no_error_and_failed_readout_do_not_change_fit_route(self):
        arrays, methods = fixture()
        _, out, _ = f.compose_fallback(arrays, methods)
        self.assertEqual(out['dual__joint0']['source_arm'], 'moment__joint0')
        self.assertEqual(out['dual__joint0']['prediction'], -1)
        methods['moment__joint0'].update(decision_valid=False, prediction=None)
        _, out, _ = f.compose_fallback(arrays, methods)
        self.assertEqual(out['dual__joint0']['source_arm'], 'moment__joint0')
        self.assertFalse(out['dual__joint0']['decision_valid'])
        self.assertTrue(out['dual__joint0']['valid'])

    def test_shared_bank_simple_controls_and_exact_source_inheritance(self):
        arrays, methods = fixture(False, True)
        original = deepcopy(methods)
        new_arrays, out, route = f.compose_fallback(arrays, methods)
        self.assertEqual(route['routes']['dual'], 'context_joint')
        for core in f.CORES + ('equal', 'iu'):
            arm = 'dual__' + core
            self.assertEqual(out[arm]['source_arm'], 'context__' + core)
            np.testing.assert_array_equal(new_arrays[arm + '__risk'], arrays['context__' + core + '__risk'])
        for core in f.CORES:
            self.assertEqual(out['single__' + core]['source_arm'], 'moment__iu')
        self.assertEqual(methods, original)
        new_arrays['dual__joint0__risk'][0] = -999
        self.assertNotEqual(arrays['context__joint0__risk'][0], -999)

    def test_exhausted_fallback_remains_failure(self):
        arrays, methods = fixture(False, False)
        methods['moment__iu'].update(valid=False, decision_valid=False, fixed_iu_valid=False,
                                     prediction=None, fixed_iu_prediction=None)
        _, out, _ = f.compose_fallback(arrays, methods)
        for policy in ('single', 'dual'):
            for core in f.CORES:
                self.assertFalse(out[policy + '__' + core]['valid'])
                self.assertFalse(out[policy + '__' + core]['decision_valid'])
                self.assertIsNone(out[policy + '__' + core]['prediction'])

    def test_variant_mismatch_and_nonfinite_scores_fail_closed(self):
        arrays, methods = fixture()
        methods['moment__graph010']['valid'] = False
        with self.assertRaisesRegex(ValueError, 'ELIGIBILITY_MISMATCH'): f.compose_fallback(arrays, methods)
        arrays, methods = fixture()
        arrays['moment__joint0__risk'][0] = np.nan
        with self.assertRaisesRegex(ValueError, 'INVALID_SOURCE_SCORE'): f.compose_fallback(arrays, methods)


if __name__ == '__main__': unittest.main()
