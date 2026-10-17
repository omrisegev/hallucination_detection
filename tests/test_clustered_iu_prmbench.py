import unittest
import numpy as np
from spectral_utils.clustered_iu_prmbench import fit_filtered_level, fit_clustered_iu


class ScientificInvariants(unittest.TestCase):
    def test_small_group_exceptions(self):
        rng = np.random.default_rng(13)
        X = rng.normal(size=(500, 2))
        for n, mode in [(1, 'singleton_passthrough'), (2, 'continuous_lsml_pair')]:
            m = fit_filtered_level(X[:, :n])
            self.assertEqual(m.diagnostic['refit_mode'], mode)
            self.assertIsNone(m.diagnostic['rho_hat_full'])
            self.assertTrue(np.isfinite(m.predict(X[:, :n])).all())

    def test_survivors_follow_rule_without_restoration(self):
        rng = np.random.default_rng(8)
        signal = rng.normal(size=3000)
        X = np.column_stack([signal + .15*rng.normal(size=3000) for _ in range(3)] +
                            [rng.normal(size=3000) for _ in range(5)])
        m = fit_filtered_level(X); d = m.diagnostic
        expected = np.flatnonzero(np.asarray(d['rho_hat_full']) >= d['threshold'])
        self.assertEqual(d['retained_columns'], expected.tolist())
        self.assertFalse(d['forced_restoration'])
        self.assertLess(len(expected), X.shape[1])
        dropped = np.setdiff1d(np.arange(X.shape[1]), expected)
        self.assertTrue(np.all(m.weight[dropped] == 0))

    def test_hierarchy_affine_on_unseen_shifted_data(self):
        rng = np.random.default_rng(61)
        X = rng.normal(size=(1000, 9))
        X[:, 1:3] += X[:, [0]]
        groups = np.array([0,0,0,1,1,1,2,2,2])
        m = fit_clustered_iu(X, groups)
        test = rng.normal(size=(120,9))*2 + 7
        d = m.diagnostic
        virtual = np.column_stack([test[:, g['columns']] @ np.asarray(g['model']['weight']) + g['model']['intercept'] for g in d['inner']])
        explicit = virtual @ np.asarray(d['outer']['weight']) + d['outer']['intercept']
        np.testing.assert_allclose(m.predict(test), explicit, atol=1e-9, rtol=1e-12)
        np.testing.assert_array_equal(groups, np.array(d['group']))

    def test_filtering_can_reduce_to_pair_without_forcing_third(self):
        rng = np.random.default_rng(81)
        signal = rng.normal(size=3000)
        X = np.column_stack([signal + .1*rng.normal(size=3000) for _ in range(2)] +
                            [rng.normal(size=3000) for _ in range(5)])
        m = fit_filtered_level(X)
        self.assertEqual(m.diagnostic['retained_columns'], [0, 1])
        self.assertEqual(m.diagnostic['refit_mode'], 'continuous_lsml_pair')
        self.assertTrue(np.all(m.weight[2:] == 0))

    def test_constant_input_fails_explicitly(self):
        with self.assertRaises(ValueError):
            fit_filtered_level(np.ones((30,4)))


if __name__ == '__main__':
    unittest.main()
