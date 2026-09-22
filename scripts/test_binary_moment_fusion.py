"""Mechanism tests independent of benchmark labels."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.linalg import eigh
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.binary_moment_fusion import representation,prepare,strict_signed,fit_all,METHODS
from spectral_utils.varentropy_contribution_fusion import contributions


class Tests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(341)
        logits=np.sort(rng.normal(size=(300,50))*rng.uniform(.2,4,(300,1)),axis=1)[:,::-1]
        p=np.exp(logits-logits.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
        self.lp=np.log(p);self.a=rng.uniform(.1,8,300)

    def test_decomposition_and_chosen(self):
        banks,H=representation(self.lp,self.a);X=banks['both33']
        np.testing.assert_allclose(X[:,:15].sum(axis=1),H)
        np.testing.assert_array_equal(X[:,15:30],contributions(self.lp,15))
        np.testing.assert_allclose(X[:,15:30].sum(axis=1),X[:,31])
        np.testing.assert_array_equal(X[:,32],self.a)
        np.testing.assert_array_equal(banks['var18'],X[:,15:])

    def test_median_power_duplicates_and_constant(self):
        a=np.arange(1.,102.)
        X=np.column_stack([a,a*a,a**3,np.ones(len(a))])
        Z,B,indices,retained,signs,thr=prepare(X,a)
        self.assertEqual(B.shape,(101,1));self.assertEqual(retained.tolist(),[0])
        self.assertEqual(B[50,0],-1);self.assertEqual(B[51,0],1)
        self.assertEqual(indices.tolist(),[0,1,2])

    def test_orientation_and_token_permutation(self):
        banks,H=representation(self.lp,self.a);X=banks['both33']
        a=prepare(X,H);perm=np.random.default_rng(4).permutation(len(H))
        b=prepare(X[perm],H[perm])
        np.testing.assert_array_equal(a[1][perm],b[1])
        np.testing.assert_array_equal(a[4],b[4])
        self.assertTrue(np.all(a[0].T@(H-H.mean())>=-1e-9))

    def test_sml_explicit_covariance_eigenpair(self):
        rng=np.random.default_rng(35);latent=rng.choice([-1.,1.],500)
        B=np.column_stack([latent*np.where(rng.random(500)<p,-1.,1.) for p in (.1,.2,.3,.35)])
        score,w,gap=strict_signed(B)
        C=np.cov(B.T);O=C-np.diag(np.diag(C));ev,_=eigh(O)
        np.testing.assert_allclose(O@w,ev[-1]*w,atol=1e-12)
        np.testing.assert_allclose(score,B@w);self.assertGreater(gap,0)
        with self.assertRaises(ValueError):strict_signed(B[:,:2])

    def test_full_solver_and_short_input(self):
        fits,failures,_=fit_all(self.lp,self.a)
        self.assertFalse(failures);self.assertEqual(set(fits),set(METHODS))
        for f in fits.values():
            self.assertEqual(f['score'].shape,(300,));self.assertTrue(np.isfinite(f['score']).all())
        fits,failures,_=fit_all(self.lp[:2],self.a[:2])
        self.assertFalse(fits);self.assertEqual(len(failures),8)

    def test_reconstruct_each_solver_without_fusion_calls(self):
        banks,H=representation(self.lp,self.a)
        fits,failures,_=fit_all(self.lp,self.a)
        self.assertFalse(failures)
        for bank,X in banks.items():
            Z,B,indices,retained,signs,thresholds=prepare(X,H)
            np.testing.assert_allclose(fits[bank+'__continuous_equal']['score'],Z.mean(axis=1))
            np.testing.assert_allclose(fits[bank+'__binary_equal']['score'],B.mean(axis=1))
            f=fits[bank+'__sml'];w=f['weights'][indices[retained]]
            np.testing.assert_allclose(f['score'],B@w);self.assertAlmostEqual(float(np.abs(w).sum()),1.)
            f=fits[bank+'__lsml'];d=f['diagnostics'];virtual=[]
            for g in d['within_weights']:
                lookup=[int(np.flatnonzero(indices[retained]==col)[0]) for col in g['columns']]
                raw=B[:,lookup]@np.array(g['weights'])
                virtual.append(np.where(raw>=0,1.,-1.))
            reconstructed=np.column_stack(virtual)@np.array(d['cross_weights'])
            np.testing.assert_allclose(f['score'],reconstructed,atol=1e-12)


if __name__=='__main__':unittest.main()
