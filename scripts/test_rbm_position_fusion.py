import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.special import logsumexp,expit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectral_utils.rbm_position_fusion import contexts,objective,fit_correction
from spectral_utils.moment_rbm_fusion import rbm_objective


class ContractTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(711);self.X=rng.normal(size=(31,4));self.a=rng.normal(size=4);self.w=rng.normal(size=4);self.b=.3
        self.c=np.r_[-np.ones(13),np.ones(18)];self.d=rng.normal(size=4)*.2

    def test_gradient_and_normalized_density(self):
        for c in (self.c,np.ones(31)):
            f,g=objective(self.d,self.X,c,self.a,self.w,self.b,.4)
            num=approx_derivative(lambda v:objective(v,self.X,c,self.a,self.w,self.b,.4)[0],self.d).ravel()
            np.testing.assert_allclose(g,num,atol=1e-7,rtol=1e-6)
            losses=[]
            for x,sign in zip(self.X,c):
                w=self.w+sign*self.d;prior=expit(self.b+self.a@w+.5*w@w)
                losses.append(-logsumexp([np.log1p(-prior)-.5*np.sum((x-self.a)**2),np.log(prior)-.5*np.sum((x-self.a-w)**2)]))
            self.assertAlmostEqual(f,np.mean(losses)+.2*self.d@self.d,places=12)

    def test_zero_is_saved_rbm(self):
        expected,_=rbm_objective(np.r_[self.a,self.w,self.b],self.X)
        self.assertAlmostEqual(objective(np.zeros(4),self.X,self.c,self.a,self.w,self.b,.4)[0],expected,places=12)

    def test_partition_gaps_and_short(self):
        c,d=contexts(12,[[1,3],[4,7],[8,11]],'a')
        np.testing.assert_array_equal(c['position'],[-1]*8+[1]*4)
        self.assertEqual(d['outside_scored_spans'],4)
        self.assertEqual(sorted(d['step_signs']),sorted(d['permuted_step_signs']))
        with self.assertRaises(ValueError):contexts(12,[[1,5],[4,8]],'a')
        score,delta,diag=fit_correction(self.X,self.c,self.a,self.w,self.b,.4,-1,single_step=True)
        np.testing.assert_array_equal(delta,np.zeros(4));np.testing.assert_allclose(score,-(self.b+self.X@self.w))

    def test_fit_and_ridge(self):
        _,d,a=fit_correction(self.X,self.c,self.a,self.w,self.b,.4,1)
        _,strong,b=fit_correction(self.X,self.c,self.a,self.w,self.b,1e6,1)
        self.assertLessEqual(a['objective_final'],a['objective_initial']+1e-8)
        self.assertLess(np.linalg.norm(strong),np.linalg.norm(d))
        with self.assertRaises(ValueError):fit_correction(self.X*np.nan,self.c,self.a,self.w,self.b,.4,1)

if __name__=='__main__':unittest.main()
