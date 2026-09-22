import unittest
import numpy as np
from spectral_utils.digit_fusion import digit_streams,summarize,add_correction,direct_bank
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS


class DigitTests(unittest.TestCase):
    def test_event_definition_and_null_preserves_opportunities(self):
        g=np.array([15,16,40,18,19]);p=np.array([16,16,15,40,15])
        d,o,n=digit_streams(g,p,range(15,25),'x')
        np.testing.assert_array_equal(d,[1,0,0,0,1])
        self.assertEqual(n.sum(),2);self.assertTrue(np.all(n[o==0]==0))
        np.testing.assert_array_equal(n,digit_streams(g,p,range(15,25),'x')[2])

    def test_binary_top10_and_rate(self):
        d=np.r_[np.ones(12),np.zeros(8)];o=np.ones(20);spans=np.array([[0,3],[3,20]])
        a,c,p=summarize(d,o,d,spans)
        np.testing.assert_allclose(a[:,0],[1,.9])
        np.testing.assert_allclose(a[:,3],[1,9/17])

    def test_zero_auxiliary_is_identity(self):
        b=np.array([1.,3.,2.])
        np.testing.assert_array_equal(add_correction(b,np.zeros(3)),b)

    def test_no_digit_bank_equals_original_and_canonical_weights(self):
        x=np.random.default_rng(6).normal(size=(40,5));spans=np.array([[0,10],[10,20],[20,40]])
        s,ds=direct_bank(x,np.zeros(40),spans,np.array([1.,2.,4.]))
        np.testing.assert_array_equal(s[:,0],s[:,2]);np.testing.assert_array_equal(s[:,1],s[:,3])
        for d in ds:
            C=np.array(d['covariance']);w=upcr_fit_covariance(C,**IU_FIT_DEFAULTS).w
            if w@C@np.ones(len(w))<0:w=-w
            np.testing.assert_allclose(np.asarray(d['weights'])[d['live']],w,atol=2e-7,rtol=2e-7)

    def test_short_constant_bank_explicit_fallback(self):
        s,d=direct_bank(np.ones((2,5)),np.zeros(2),np.array([[0,2]]),np.array([4.]))
        np.testing.assert_array_equal(s,np.full((1,4),4.))
        self.assertFalse(d[0]['native'])


if __name__=='__main__':unittest.main()
