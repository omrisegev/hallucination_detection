"""Mechanical tests: temporal alignment, fusion equivalence and explicit failures."""
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.sparse import diags
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral_utils.direct_probability_temporal import (
    representation, fit_linear, hierarchical, chain_fit, lw_alpha_memory_bounded, fit_all,
)
from spectral_utils.direct_probability_fusion import fit_rank_fusion, zscore_columns, step_top_mean
from spectral_utils.shrinkage_iu import ledoit_wolf_alpha, target_matrix
from spectral_utils.laplacian_upcr import laplacian_iu_path


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(42).normal(size=(160,17))
        self.anchor = self.X[:,0] + self.X[:,1]*.4

    def test_lag_and_delta_alignment(self):
        X = np.arange(12*17).reshape(12,17)
        L = representation(X,'lag8').reshape(12,8,17)
        for t in range(12):
            for lag in range(8):
                np.testing.assert_array_equal(L[t,lag],X[max(t-lag,0)])
        D = representation(X,'delta')
        np.testing.assert_array_equal(D[0,17:],0)
        np.testing.assert_array_equal(D[1:,17:],X[1:]-X[:-1])
        np.testing.assert_array_equal(representation(X[:1],'lag8'),np.tile(X[:1],8))

    def test_future_cannot_change_earlier_representation(self):
        for kind in ('current','lag8','delta'):
            altered=self.X.copy();altered[70:]+=999
            np.testing.assert_array_equal(representation(self.X,kind)[:70],representation(altered,kind)[:70])
        # This is input causality only: answer-local fitting itself uses all tokens.

    def test_shuffled_control_preserves_current_and_history_marginals(self):
        L=representation(self.X,'lag8');S=representation(self.X,'shuffled_lag8',seed=9)
        np.testing.assert_array_equal(L[:,:17],S[:,:17])
        np.testing.assert_array_equal(np.sort(L[:,17:],axis=0),np.sort(S[:,17:],axis=0))
        self.assertFalse(np.array_equal(L[:,17:],S[:,17:]))
        np.testing.assert_array_equal(S,representation(self.X,'shuffled_lag8',seed=9))

    def test_existing_v2_continuity(self):
        for method in ('equal','iu','joint_lw'):
            a=fit_rank_fusion(self.X,method=method,anchor=self.anchor)
            b=fit_linear(self.X,self.anchor,method)
            np.testing.assert_allclose(a.score,b.score,atol=1e-10,rtol=1e-10)

    def test_lw_equivalence(self):
        Z,_,_,_=zscore_columns(self.X);C=Z.T@Z/len(Z)
        for kind in ('joint','diag'):
            T=target_matrix(C,np.arange(17),kind)
            self.assertAlmostEqual(lw_alpha_memory_bounded(Z,C,T),ledoit_wolf_alpha(Z,C,T),places=12)

    def test_hierarchy_reconstruction(self):
        L=representation(self.X,'lag8');Z,k,_,_=zscore_columns(L)
        for first in (True,False):
            fit=hierarchical(L,self.anchor,time_first=first)
            np.testing.assert_allclose(Z@fit.weights[k],fit.score,atol=1e-9)

    def test_chain_matches_existing_solve_and_zero(self):
        X=self.X[:30];a=self.anchor[:30];Z,k,_,_=zscore_columns(X)
        W=diags((np.ones(29),np.ones(29)),(-1,1),shape=(30,30),format='csr')
        for lam in (0.,.1):
            old=laplacian_iu_path(Z.T,(lam,),graph=W)[lam]
            fit=chain_fit(X,a,lambda_=lam)
            sign=-1 if fit.diagnostics['orientation_flipped'] else 1
            np.testing.assert_allclose(fit.weights[k],old.w*sign,atol=1e-10,rtol=1e-10)
        np.testing.assert_array_equal(chain_fit(X,a,lambda_=0).score,fit_linear(X,a,'iu').score)

    def test_spike_is_not_averaged_in_representation(self):
        X=np.zeros((20,17));X[10,2]=1
        L=representation(X,'lag8')
        self.assertEqual(L[10,2],1)
        self.assertEqual(L[11,17+2],1)
        self.assertEqual(L[11,2],0)
        D=representation(X,'delta')
        self.assertEqual(D[10,17+2],1);self.assertEqual(D[11,17+2],-1)
        np.testing.assert_array_equal(step_top_mean(X[:,2],np.array([0,10]),np.array([10,20])),[0,.1])

    def test_short_and_constant_fail_without_fallback(self):
        fits,failures,_=fit_all(np.ones((1,17)),np.ones(1),uid='short')
        self.assertFalse(fits);self.assertEqual(len(failures),18)
        with self.assertRaises(ValueError):
            representation(np.full((3,17),np.nan),'lag8')


if __name__=='__main__':
    with threadpool_limits(limits=1):
        unittest.main()
