import numpy as np
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectral_utils.predictor_subset_fusion import fast_iu,fuse,top10,correction,METHODS,SUBSETS
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS

class PredictorSubsetTests(unittest.TestCase):
    def test_vectorized_grid_matches_canonical(self):
        rng=np.random.default_rng(389)
        for m in (3,4,5):
            for noise in (1,.1,.001):
                for seed in range(5):
                    x=rng.normal(size=(160,m))*noise+rng.normal(size=(160,1))
                    x=(x-x.mean(0))/x.std(0);C=x.T@x/len(x)
                    w,d=fast_iu(C);ref=upcr_fit_covariance(C,**IU_FIT_DEFAULTS)
                    np.testing.assert_allclose(w,ref.w,rtol=2e-7,atol=2e-7)
                    np.testing.assert_allclose(d['g2'],ref.g2_hat,atol=1e-14)

    def test_all_subsets_and_affine_invariance(self):
        rng=np.random.default_rng(50);r=rng.normal(size=(70,5));spans=np.array([[0,20],[20,45],[45,70]])
        base=np.array([2.,1.,4.]);scores,diag,single=fuse(r,spans,base)
        assert len(SUBSETS)==16 and len(METHODS)==32 and scores.shape==(3,32)
        other,_,_=fuse(r*np.arange(1,6)+np.arange(5),spans,base)
        np.testing.assert_allclose(scores,other,atol=2e-12)
        Z=(r-r.mean(0))/r.std(0)
        for j,s in enumerate(SUBSETS):
            signal=Z[:,s].mean(1)
            aux=np.array([np.sort(signal[a:b])[-min(10,b-a):].mean() for a,b in spans])
            expected=base+.25*base.std()*(aux-aux.mean())/aux.std()
            np.testing.assert_allclose(scores[:,2*j+1],expected,atol=2e-12)
        np.testing.assert_allclose(scores.mean(0),base.mean(),atol=1e-12)

    def test_constant_failure_and_zero_correction_are_explicit(self):
        with self.assertRaisesRegex(ValueError,'Constant'):fuse(np.ones((20,5)),[[0,10],[10,20]],[0.,1.])
        out,constant=correction([0.,1.],np.ones((2,3)))
        assert constant.all()
        np.testing.assert_array_equal(out,np.array([[0.,0.,0.],[1.,1.,1.]]))

if __name__=='__main__':unittest.main()
