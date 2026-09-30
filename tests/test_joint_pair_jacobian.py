import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.joint_pair_jacobian import profiled_pair_jacobian,fit_joint_pairs_checked
from spectral_utils.joint_lsml import _profiled_jacobian_audit


class PairJacobianTests(unittest.TestCase):
    def test_zero_pair_products_cannot_hide_global_scale_ambiguity(self):
        groups=np.repeat(np.arange(3),2);v=np.array([.3,.4,.5,.6,0.,0.]);u=np.zeros(6)
        mask=(groups[:,None]==groups[None,:]).astype(float)-np.eye(6)
        legacy=_profiled_jacobian_audit(v,u,mask);correct=profiled_pair_jacobian(v,u,groups)
        self.assertTrue(legacy['full_global_rank'])
        self.assertFalse(correct['full_global_rank']);self.assertEqual(correct['rank'],5)
        # Explicit equally fitting nearby global factors: reciprocal group scaling.
        changed=v.copy();changed[:2]*=1.01;changed[2:4]/=1.01
        cross=groups[:,None]!=groups[None,:]
        np.testing.assert_allclose(np.outer(changed,changed)[cross],np.outer(v,v)[cross],atol=1e-14)
        # Pair residuals absorb the changed within-pair global products.
        for i in (0,2,4):
            residual=v[i]*v[i+1]-changed[i]*changed[i+1]
            self.assertLess(abs(residual),np.sqrt((1-changed[i]**2)*(1-changed[i+1]**2)))

    def test_nonzero_pair_products_match_legacy_profiled_global_rank(self):
        groups=np.repeat(np.arange(3),2);v=np.linspace(.3,.55,6);u=np.linspace(.2,.45,6)
        mask=(groups[:,None]==groups[None,:]).astype(float)-np.eye(6)
        old=_profiled_jacobian_audit(v,u,mask);new=profiled_pair_jacobian(v,u,groups)
        self.assertTrue(new['full_global_rank'])
        self.assertEqual(new['rank'],old['rank'])
        self.assertAlmostEqual(new['condition_number'],old['condition_number'],places=12)

    def test_checked_fit_replays_nonsingular_pair_covariance(self):
        groups=np.repeat(np.arange(3),2);v=np.linspace(.3,.55,6);u=np.linspace(.2,.45,6)
        c=np.outer(v,v)+(groups[:,None]==groups[None,:])*np.outer(u,u);np.fill_diagonal(c,1.)
        fit=fit_joint_pairs_checked(c,groups,anchor_index=0)
        self.assertTrue(fit.joint.jacobian_audit['full_global_rank'])
        self.assertEqual(fit.joint.jacobian_audit['pair_product_columns'],3)
        np.testing.assert_allclose(fit.joint.model_covariance,c,atol=1e-8)


if __name__=='__main__':unittest.main()
