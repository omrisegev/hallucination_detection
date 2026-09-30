import unittest
import numpy as np
from spectral_utils import fusion_gate_calibration as c


class GateCalibrationTests(unittest.TestCase):
    def test_seed_roles_are_disjoint(self):
        self.assertNotEqual(c.seed('evaluation','same'),c.seed('calibration','same'))
        self.assertEqual(c.seed('evaluation','same'),c.seed('evaluation','same'))

    def test_rho_affine_invariance_and_direct_least_squares(self):
        x=np.random.default_rng(42).normal(size=30);x=np.cumsum(x)
        a=c.fit_rho(x);b=c.fit_rho(x*7-10)
        expected=np.linalg.lstsq(np.column_stack((np.ones(len(x)-1),x[:-1])),x[1:],rcond=None)[0][1]
        self.assertAlmostEqual(a['raw'],expected);self.assertAlmostEqual(a['raw'],b['raw']);self.assertLessEqual(abs(a['value']),.95)

    def test_rank_ties_and_reopening_negative_bic(self):
        obj=lambda t:dict(valid=True,gate=dict(bic_gain=t))
        result=c.calibrated(obj(-1),[obj(-2)]*c.B)
        self.assertEqual(result['pvalue'],.025);self.assertTrue(result['open'])
        tied=c.calibrated(obj(-1),[obj(-1)]*c.B);self.assertEqual(tied['pvalue'],1);self.assertFalse(tied['open'])
        almost=c.calibrated(obj(1),[obj(1-1e-15)]*c.B);self.assertEqual(almost['pvalue'],1)

    def test_missing_draw_is_not_silently_discarded(self):
        obj=dict(valid=True,gate=dict(bic_gain=3));nulls=[dict(valid=True,gate=dict(bic_gain=0))]*c.B
        self.assertFalse(c.calibrated(obj,nulls[:-1])['valid'])
        nulls[1]=dict(valid=False);self.assertFalse(c.calibrated(obj,nulls)['valid'])


if __name__=='__main__':unittest.main()
