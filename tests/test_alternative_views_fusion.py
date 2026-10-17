import unittest
import numpy as np
from spectral_utils.alternative_views_fusion import extract,shrink_covariance,score
from spectral_utils.step_evidence_fusion import standardize
from spectral_utils.predictor_subset_fusion import fast_iu
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS


class AlternativeTests(unittest.TestCase):
    def test_tail_not_double_normalized_and_rank(self):
        p=np.linspace(.01,.001,50);lp=np.log(np.tile(p,(2,1)));ids=np.tile(np.arange(15,65),(2,1))
        x,d=extract(np.array([16,99]),ids,lp,np.array([-lp[0,1],9.]))
        np.testing.assert_allclose(x[:,7],1-p[:15].sum());np.testing.assert_allclose(x[:,8],1-p.sum())
        np.testing.assert_allclose(x[:,1],[1,50]);self.assertEqual(x[0,6],1)
        self.assertEqual(d['absent_top50'],1)

    def test_block_shrink_identity_psd_and_canonical(self):
        z=standardize(np.random.default_rng(5).normal(size=(47,7)))[0];C=z.T@z/len(z);g=np.array([0]*4+[1]*2+[2])
        for kind in ('diag','block'):
            c,a,n=shrink_covariance(z,C,g,kind);self.assertTrue(0<=a<=1);self.assertEqual(n,3)
            self.assertGreater(np.linalg.eigvalsh(c)[0],-1e-10)
            np.testing.assert_array_equal(shrink_covariance(z,C,g,kind,0)[0],C)
            np.testing.assert_allclose(fast_iu(c)[0],upcr_fit_covariance(c,**IU_FIT_DEFAULTS).w,atol=2e-7,rtol=2e-7)

    def test_one_block_explicit_target(self):
        z=standardize(np.array([[1.,2.,0.],[0.,1.,1.],[2.,0.,3.]]))[0];C=z.T@z/3
        c,a,_=shrink_covariance(z,C,np.arange(3),'diag')
        self.assertEqual(a,1);np.testing.assert_array_equal(c,np.diag(np.diag(C)))

    def test_all_constant_is_explicit_and_finite(self):
        out,d=score(np.ones((20,5)),np.zeros((20,9)),np.array([[0,10],[10,20]]),np.array([1.,2.]))
        self.assertTrue(np.isfinite(out).all());self.assertEqual(out.shape,(2,28))
        self.assertEqual(d['new7']['heads']['iu']['fallback'],'fewer_than_three_live_views')

    def test_no_labels_in_fit_interface(self):
        with self.assertRaises(TypeError):
            score(np.ones((20,5)),np.ones((20,9)),np.array([[0,20]]),np.array([1.]),labels=[1])


if __name__=='__main__':unittest.main()
