"""Mechanism tests; synthetic cases provide no research rankings."""
import unittest
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spectral_utils import higher_moment_fusion as new
from spectral_utils import moment_rbm_fusion as old
from spectral_utils.direct_probability_fusion import zscore_columns


class HigherMomentTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(941)
        p=np.sort(rng.dirichlet(np.ones(50),size=90),axis=1)[:,::-1]
        self.lp=np.log(p)
        self.a=rng.uniform(.1,9.,size=90)

    def test_formula_nesting_and_finiteness(self):
        X=new.representation_order(self.lp,self.a,6)
        self.assertEqual(X.shape,(90,12))
        np.testing.assert_array_equal(X[:,:6],old.representation(self.lp,self.a))
        p=np.exp(self.lp[:,:15]);q=p/(p.sum(axis=1,keepdims=True)+1e-12)
        y=-np.log(q+1e-12)
        for d in new.ORDERS:
            np.testing.assert_array_equal(X[:,:2*d],new.representation_order(self.lp,self.a,d))
        for d in (4,5,6):
            col=6+2*(d-4)
            np.testing.assert_allclose(X[:,col],np.sum(q*y**d,axis=1),rtol=1e-14)
            np.testing.assert_array_equal(X[:,col+1],X[:,3]**d)
        self.assertTrue(np.isfinite(X).all())

    def test_original_four_arms_and_saved_state(self):
        # Restrict historical solver list only for this test, avoiding unused B3 fits.
        saved=old.METHODS;old.METHODS=new.SOLVERS
        try:reference,errors,_=old.fit_all(self.lp,self.a,'synthetic')
        finally:old.METHODS=saved
        fits,failures,_=new.fit_all(self.lp,self.a)
        self.assertEqual(errors,{})
        self.assertEqual(failures,{})
        for solver in new.SOLVERS:
            np.testing.assert_array_equal(reference[solver]['score'],fits['d3__'+solver]['score'])
        for name,f in fits.items():
            Z,_,_,_=zscore_columns(new.representation_order(self.lp,self.a,int(name[1])))
            state=f['state'];score=Z@state['w']
            if name.split('__')[1].startswith('rbm'):score=expit(score+state['b'])
            if f['diagnostics']['orientation']<0:
                score=1-score if name.split('__')[1].startswith('rbm') else -score
            np.testing.assert_allclose(score,f['score'],atol=2e-15,rtol=0)

    def test_constant_short_and_extreme_inputs(self):
        fits,failures,_=new.fit_all(self.lp[:2],self.a[:2])
        self.assertFalse(fits);self.assertEqual(len(failures),16)
        fits,failures,_=new.fit_all(np.tile(self.lp[:1],(10,1)),np.ones(10))
        self.assertFalse(fits);self.assertEqual(len(failures),16)
        X=new.representation_order(np.tile(np.linspace(-.01,-750,50),(3,1)),np.array([0.,100.,1000.]),6)
        self.assertTrue(np.isfinite(X).all())


if __name__=='__main__': unittest.main()
