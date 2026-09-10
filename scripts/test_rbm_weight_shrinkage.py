"""Mathematical and compatibility checks for the one fixed RBM penalty."""
import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils import rbm_weight_shrinkage as new, moment_rbm_fusion as old
from spectral_utils.direct_probability_fusion import zscore_columns


class Tests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(590)
        self.lp=np.log(np.sort(rng.dirichlet(np.ones(50),size=90),axis=1)[:,::-1])
        self.chosen=rng.uniform(.1,6,90)
        self.Z=zscore_columns(old.representation(self.lp,self.chosen))[0]

    def test_penalty_gradient_and_scope(self):
        p=self.Z.shape[1];theta=np.linspace(-.7,1.3,2*p+1)
        loss,grad=new.penalized_objective(theta,self.Z)
        numerical=approx_derivative(lambda t:new.penalized_objective(t,self.Z)[0],theta).ravel()
        np.testing.assert_allclose(grad,numerical,rtol=1e-6,atol=1e-7)
        oldloss,oldgrad=old.rbm_objective(theta,self.Z)
        self.assertAlmostEqual(loss-oldloss,.1*np.sum((theta[p:2*p]-2/p)**2))
        np.testing.assert_array_equal(grad[:p],oldgrad[:p])
        self.assertEqual(grad[-1],oldgrad[-1])

    def test_zero_penalty_exact_replay(self):
        p=self.Z.shape[1];theta=np.linspace(-.7,1.3,2*p+1)
        a=old.rbm_objective(theta,self.Z);b=new.penalized_objective(theta,self.Z,0)
        self.assertEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        oldscore,oldstate,_=old.fit_rbm(self.Z)
        score,state,_=new.fit_regularized(self.Z,strength=0)
        np.testing.assert_array_equal(score,oldscore)
        for k in state:np.testing.assert_array_equal(state[k],oldstate[k])

    def test_controls_and_posterior_replay(self):
        fits,fail,_=new.fit_all(self.lp,self.chosen);self.assertFalse(fail)
        with patch.object(old,'METHODS',('rbm','rbm_initial')):
            previous,fail,_=old.fit_all(self.lp,self.chosen,'test');self.assertFalse(fail)
        for m in previous:np.testing.assert_array_equal(fits[m]['score'],previous[m]['score'])
        for f in fits.values():
            state=f['state'];q=expit(state['b']+self.Z@state['w'])
            if f['diagnostics']['orientation']<0:q=1-q
            np.testing.assert_array_equal(q,f['score'])
        self.assertEqual(set(inspect.signature(new.fit_all).parameters),{'logprobs','chosen','entropy'})
        # The compatibility entropy input cannot set direction or fit weights.
        other,_,_=new.fit_all(self.lp,self.chosen,np.arange(90)[::-1])
        for m in fits:np.testing.assert_array_equal(fits[m]['score'],other[m]['score'])

    def test_constant_and_short_data(self):
        # Selected-token powers become constants; remaining H,V,m3 are retained.
        fits,fail,_=new.fit_all(self.lp,np.ones(90));self.assertFalse(fail)
        for f in fits.values():
            self.assertEqual(f['diagnostics']['columns'],[0,1,2])
            np.testing.assert_array_equal(f['weights'][3:],0)
        for lp,a in ((self.lp[:2],self.chosen[:2]),(np.tile(self.lp[0],(90,1)),np.ones(90))):
            fits,fail,_=new.fit_all(lp,a)
            self.assertFalse(fits);self.assertEqual(set(fail),set(new.METHODS))


if __name__=='__main__':unittest.main()
