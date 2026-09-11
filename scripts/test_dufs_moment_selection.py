"""Mechanics tests, no development labels or quality-based tuning."""
from pathlib import Path
import sys, unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.dufs_moment_selection import fit_all, METHODS, correlation_six, top_six
from spectral_utils.higher_moment_fusion import representation_order
from spectral_utils.direct_probability_fusion import zscore_columns, _orient
from spectral_utils.moment_rbm_fusion import fit_rbm
from scipy.special import expit


class SelectionTests(unittest.TestCase):
    def test_ties_and_invalid(self):
        np.testing.assert_array_equal(top_six(np.ones(12)),np.arange(6))
        with self.assertRaises(ValueError):top_six([1,np.nan,2,3,4,5])

    def test_greedy_independent_replay(self):
        rng=np.random.default_rng(91);Z=rng.normal(size=(300,12));Z[:,6]=Z[:,0]
        c=np.corrcoef(Z,rowvar=False)**2;np.fill_diagonal(c,0)
        chosen=[min(range(12),key=lambda i:(sum(c[:,i]),i))]
        while len(chosen)<6:
            chosen.append(min(set(range(12))-set(chosen),key=lambda i:(sum(c[i,j] for j in chosen),i)))
        np.testing.assert_array_equal(correlation_six(Z),sorted(chosen))
        self.assertEqual(len(set(correlation_six(Z))),6)
        self.assertFalse({0,6}.issubset(set(correlation_six(Z))))

    def test_models_determinism_anchor_and_reference(self):
        rng=np.random.default_rng(551)
        p=rng.dirichlet(np.ones(50)*.13,size=55)
        lp=np.log(np.sort(p,axis=1)[:,::-1]);chosen=-lp[:,0]+rng.exponential(.5,size=55)
        fits,fail,_=fit_all(lp,chosen);self.assertEqual(fail,{})
        fits2,fail2,_=fit_all(lp,chosen);self.assertEqual(fail2,{})
        anchor=zscore_columns(representation_order(lp,chosen,3))[0].mean(axis=1)
        for m in METHODS:
            np.testing.assert_array_equal(fits[m]['score'],fits2[m]['score'])
            d=fits[m]['diagnostics'];cols=d['columns'];self.assertEqual(len(cols),12 if m.startswith('all12') else 6)
            if m.startswith('dufs6') or m.startswith('correlation6'):
                bank=representation_order(lp,chosen,6)[:,cols];Z=zscore_columns(bank)[0]
                q=expit(Z@fits[m]['state']['w']+fits[m]['state']['b'])
                if d['orientation']<0:q=1-q
                np.testing.assert_allclose(q,fits[m]['score'],atol=1e-12)
            self.assertTrue(np.all(fits[m]['weights'][np.setdiff1d(np.arange(12),cols)]==0))
        for degree,bank in ((3,'original6'),(6,'all12')):
            Z=zscore_columns(representation_order(lp,chosen,degree))[0]
            q,_,_=fit_rbm(Z,maxiter=100);q,flip,_=_orient(q,anchor)
            if flip:q=1+q
            np.testing.assert_array_equal(q,fits[bank+'__rbm']['score'])
        for bank in ('dufs6','correlation6'):
            self.assertEqual(fits[bank+'__rbm']['diagnostics']['columns'],fits[bank+'__rbm_initial']['diagnostics']['columns'])

    def test_short_and_constant_fail_explicitly(self):
        for n in (2,30):
            lp=np.full((n,50),-np.log(50.));chosen=np.full(n,1.)
            fits,fail,_=fit_all(lp,chosen)
            self.assertEqual(fits,{});self.assertEqual(set(fail),set(METHODS))


if __name__=='__main__':unittest.main()
