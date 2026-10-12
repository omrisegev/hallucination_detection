"""Scientific invariants for a supporting prediction-error fusion view."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prediction_view_tested', ROOT/'spectral_utils/fusion_prediction_view.py')
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)


class PredictionViewTests(unittest.TestCase):
    def test_prefix_batch_fit_and_prediction_before_update(self):
        x = np.random.default_rng(31).normal(size=(90, 4)).cumsum(axis=0)
        result = core.prediction_views(x)
        for t in range(2, len(x)):
            a, b = x[:t-1], x[1:t]
            ac = a - a.mean(axis=0); bc = b - b.mean(axis=0)
            denominator = np.sum(ac * ac, axis=0)
            slope = np.ones(4)
            np.divide(np.sum(ac * bc, axis=0), denominator, out=slope, where=denominator > 0)
            slope = np.clip(slope, -1., 1.)
            eta = (t-1) / (t-1 + 16.)
            expected = x[t-1] + eta * (b.mean(axis=0) - a.mean(axis=0)
                        + (slope-1.) * (x[t-1] - a.mean(axis=0)))
            np.testing.assert_allclose(result['predictions']['ar1'][t], expected, atol=2e-13, rtol=2e-13)
        changed = x.copy(); changed[41:] += 300.
        perturbed = core.prediction_views(changed)
        for kind in core.KINDS:
            # The prediction AT the changed token also cannot see its value.
            np.testing.assert_array_equal(result['predictions'][kind][:42], perturbed['predictions'][kind][:42])
            np.testing.assert_array_equal(result['residuals'][kind][:41], perturbed['residuals'][kind][:41])
        np.testing.assert_array_equal(result['fit_pair_counts'], np.maximum(np.arange(90)-1, 0))

    def test_affine_constant_and_answer_reset(self):
        x = np.random.default_rng(18).normal(size=(65, 3))
        a = core.prediction_views(x); scale = np.array([2., .5, 4.]); offset = np.array([8., -3., 11.])
        b = core.prediction_views(x * scale + offset)
        for kind in core.KINDS:
            np.testing.assert_allclose(b['predictions'][kind], a['predictions'][kind]*scale + offset, atol=2e-13)
            np.testing.assert_allclose(b['residuals'][kind], a['residuals'][kind]*scale, atol=2e-13)
            np.testing.assert_array_equal(core.prediction_views(np.full((64, 3), 7.))['residuals'][kind], 0.)
        core.prediction_views(x + 1000.)
        np.testing.assert_array_equal(core.prediction_views(x)['predictions']['ar1'], a['predictions']['ar1'])

    def test_mask_support_and_original_columns_preserved(self):
        residual = np.arange(17*9, dtype=float).reshape(17, 9)
        residual[0] = 1e9
        mask = np.arange(17) > 0
        values, counts = core.residual_window_features(residual, mask, [0, 8, 9], [8, 16, 17])
        np.testing.assert_array_equal(counts, [7, 8, 8])
        np.testing.assert_array_equal(values[0], residual[1:8].mean(axis=0))
        base = np.random.default_rng(5).normal(size=(3, 27))
        augmented, names = core.augment_bank(base, [str(i) for i in range(27)], values, [str(i) for i in range(9)])
        np.testing.assert_array_equal(augmented[:, :27], base)
        np.testing.assert_array_equal(augmented[:, 27:], values)
        self.assertEqual(augmented.shape, (3, 36)); self.assertEqual(len(names), 36)
        with self.assertRaisesRegex(ValueError, 'NO_PREDICTED_TOKEN'):
            core.residual_window_features(residual, mask, [0], [1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
