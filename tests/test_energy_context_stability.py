import unittest
from pathlib import Path
from scipy.optimize import minimize
import numpy as np
from scipy.spatial.distance import cdist
from spectral_utils.energy_context_stability import (history_landmarks,group_weights,
    ContextFit,GroupNeighbors,heads,conditional_moments,FLOOR,gaussian_nll,simplex_qp)
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.contextual_iu import DEFAULT_IU_FIT


class EnergyTests(unittest.TestCase):
    def test_numerical_boundary_regression(self):
        path=Path(__file__).resolve().parents[1]/'results/energy_context_stability_v1/QP_REGRESSION.npz'
        with np.load(path) as f:Q,r,expected=f['Q'],f['r'],f['expected']
        w=simplex_qp(Q,r)[0]
        np.testing.assert_allclose(w,expected,atol=1e-12)
        grad=Q@w-r;level=w@grad
        self.assertLess(np.max(np.where(w>1e-8,abs(grad-level),np.maximum(level-grad,0))),1e-10)
        self.assertGreater(w[-1],0.)

    def test_simplex_independent_optimizer(self):
        rng=np.random.default_rng(85)
        for _ in range(24):
            A=rng.normal(size=(5,5));Q=A.T@A+.02*np.eye(5);r=rng.normal(size=5)
            w=simplex_qp(Q,r)[0]
            solution=minimize(lambda x:.5*x@Q@x-x@r,np.ones(5)/5,jac=lambda x:Q@x-r,
                method='SLSQP',bounds=[(0,1)]*5,
                constraints=[dict(type='eq',fun=lambda x:x.sum()-1,jac=lambda x:np.ones(5))],
                options=dict(ftol=1e-13,maxiter=1000))
            self.assertTrue(solution.success)
            np.testing.assert_allclose(w,solution.x,atol=2e-6)

    def test_history_scalar_and_causal(self):
        rng=np.random.default_rng(80);x=rng.normal(size=(600,5));p,m,s=history_landmarks(x)
        for j,t in enumerate(p):
            np.testing.assert_allclose(m[j],np.mean(x[t-16:t],axis=0),atol=1e-15)
            np.testing.assert_allclose(s[j],np.mean(x[t-16:t]**2,axis=0),atol=1e-15)
            y=x.copy();y[t:]=1000
            _,m2,s2=history_landmarks(y)
            np.testing.assert_array_equal(m[j],m2[j]);np.testing.assert_array_equal(s[j],s2[j])

    def test_group_answer_balancing(self):
        g=np.array(['a','a','a','b']);a=np.array([1,1,2,3]);w=group_weights(g,a)
        np.testing.assert_allclose(w,[.125,.125,.25,.5])
        idx=np.repeat(np.arange(4),2);w2=group_weights(g[idx],a[idx])
        np.testing.assert_allclose(w2.reshape(4,2).sum(axis=1),w)

    def test_group_knn_brute_force(self):
        rng=np.random.default_rng(81);z=rng.normal(size=(400,7));g=np.repeat(np.arange(100),4);q=rng.normal(size=(15,7))
        model=GroupNeighbors(z,g);ids,kernel,d=model.nearest(q)
        dist=cdist(q,z)
        for j in range(len(q)):
            ordered=np.argsort(dist[j]);chosen=[];seen=set()
            for i in ordered:
                if g[i] not in seen:chosen.append(i);seen.add(g[i])
                if len(chosen)==64:break
            np.testing.assert_array_equal(ids[j],chosen)
            self.assertEqual(len(set(g[ids[j]])),64)
        self.assertGreater(float(np.min(1/(kernel**2).sum(axis=1))),60.)

    def test_heads_match_canonical_and_constraints(self):
        rng=np.random.default_rng(82)
        for _ in range(12):
            A=rng.normal(size=(5,5));C=A.T@A+.1*np.eye(5);sd=np.exp(rng.normal(size=5))
            h=heads(C,sd,.25);ref=upcr_fit_covariance(C,var_y=.25,**DEFAULT_IU_FIT)
            np.testing.assert_allclose(h['rho'][0],ref.rho_hat_full,atol=1e-10)
            np.testing.assert_allclose(h['native_a'][0],ref.w,atol=1e-10)
            for key in ('qp_w','group_w'):
                self.assertAlmostEqual(h[key].sum(),1.);self.assertGreaterEqual(h[key].min(),.15-1e-10)
            np.testing.assert_allclose(h['group_w'][0,[0,1,4]],h['group_w'][0,0])
            np.testing.assert_allclose(h['qp_a'],h['qp_w']*sd/sd.mean())

    def test_conditional_moments_and_nll_scalar(self):
        rng=np.random.default_rng(83);x=rng.normal(size=(4,64,5));w=rng.uniform(size=(4,64));w/=w.sum(axis=1,keepdims=True)
        Cg=np.eye(5)*(1+FLOOR);mu,C=conditional_moments(x,w,Cg);y=rng.normal(size=(4,5));nll=gaussian_nll(y,mu,C)
        for j in range(4):
            mean=np.average(x[j],axis=0,weights=w[j]);z=x[j]-mean
            expected=.5*z.T@np.diag(w[j])@z+.5*Cg+.5*FLOOR*np.eye(5)
            np.testing.assert_allclose(mu[j],.5*mean);np.testing.assert_allclose(C[j],expected)
            e=y[j]-.5*mean
            scalar=.5*(np.log(np.linalg.det(expected))+e@np.linalg.inv(expected)@e+5*np.log(2*np.pi))
            self.assertAlmostEqual(scalar,nll[j],places=10)

    def test_profile_held_transform_does_not_refit(self):
        rng=np.random.default_rng(84);x=rng.normal(size=(100,5));hm=x*.2;hs=hm**2+1;p=np.linspace(.1,.9,100);length=np.full(100,400)
        fit=ContextFit().fit(x,hm,hs,p,length,np.arange(100));before=fit.profile.copy()
        fit.transform(hm*100,hs*10000,p,length)
        np.testing.assert_array_equal(before,fit.profile)


if __name__=='__main__':unittest.main()
