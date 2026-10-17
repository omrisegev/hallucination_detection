import unittest
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
from spectral_utils.residual_moment_fusion import (fit_pair, fit_head, stream_top10,
                                                   context_design, residuals, paired_moments)
from spectral_utils.contextual_iu import DEFAULT_IU_FIT
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.temporal_context_models import RidgePredictor


class ResidualFusionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(31)
        self.x = rng.normal(size=(320, 5)) @ rng.normal(size=(5, 5))
        self.r = self.x - .4 * np.roll(self.x, 1, axis=0)

    def test_canonical_and_independent_qp(self):
        f = fit_pair(self.x, self.r)
        for rep in ('L', 'R'):
            C = f['C_' + rep]; h = f[rep]
            ref = upcr_fit_covariance(C, var_y=f['var_y'], **DEFAULT_IU_FIT)
            np.testing.assert_allclose(h['rho'], ref.rho_hat_full, atol=2e-12)
            np.testing.assert_allclose(h['native_a'], ref.w, atol=2e-10)
            B = f['sd'] / f['sd'].mean(); Q = C * B[:, None] * B; r = B * h['rho']
            opt = minimize(lambda w: .5*w@Q@w-w@r, np.ones(5)/5,
                jac=lambda w: Q@w-r, bounds=[(0, 1)]*5,
                constraints={'type': 'eq', 'fun': lambda w: w.sum()-1, 'jac': lambda w: np.ones(5)},
                method='SLSQP', options={'ftol': 1e-12, 'maxiter': 1000})
            self.assertTrue(opt.success)
            np.testing.assert_allclose(h['raw_simplex'], opt.x, atol=3e-6)

    def test_zero_prediction_identity(self):
        f = fit_pair(self.x, self.x)
        for key in ('rho', 'native', 'simplex'):
            np.testing.assert_array_equal(f['L'][key], f['R'][key])

    def test_common_coordinate_scale(self):
        sd, CL, CR, ceiling = paired_moments(self.x, .1*self.x)
        np.testing.assert_allclose(CR-1e-6*np.eye(5), .01*(CL-1e-6*np.eye(5)), atol=1e-12)
        self.assertAlmostEqual(ceiling, .25000025)

    def test_negative_weight_keeps_top_tokens(self):
        x = np.column_stack((np.arange(20), np.arange(20)[::-1]))
        s = stream_top10(x, [[0, 20]])
        np.testing.assert_array_equal(s, [[14.5, 14.5]])
        self.assertEqual(float((s @ [-1., 0.])[0]), -14.5)

    def test_mask_and_scalar_design(self):
        x = self.x[:40]; b = SimpleNamespace(features=x, columns=np.arange(5),
            length=np.array([20,20]), offset=np.array([0,20]),
            mean=np.array([x[:20].mean(0),x[20:].mean(0)]), scale=np.ones((2,5)))
        ids=np.array([0,0,1,1]); pos=np.array([0,19,1,16])
        from scripts.run_temporal_linear_context import design
        expected,_=design(b,ids,pos)
        np.testing.assert_array_equal(context_design(b,ids,pos),expected)
        model=RidgePredictor(np.zeros((98,5)),np.zeros(97),np.ones(97))
        np.testing.assert_allclose(residuals(b,ids,pos,model), x[b.offset[ids]+pos]-b.mean[ids])
        changed=x.copy();changed[20:]+=1000;b.features=changed
        np.testing.assert_array_equal(context_design(b,ids[:2],pos[:2]),expected[:2])

    def test_constant_and_invalid(self):
        f=fit_pair(np.ones((20,5)),np.zeros((20,5)))
        self.assertTrue(np.isfinite(f['L']['simplex']).all())
        with self.assertRaises(ValueError):fit_pair(self.x,self.r[:-1])


if __name__ == '__main__':unittest.main()
