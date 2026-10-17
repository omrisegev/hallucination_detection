"""Independent optimizer/canonical identities for the bounded gate."""
import unittest
import numpy as np
from scipy.optimize import minimize
from spectral_utils.cca_iu_isolation import (simplex_qp, iu_moments, fusion_weights,
    local_moments, CCAContext, covariance)
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.contextual_iu import DEFAULT_IU_FIT


class IsolationTests(unittest.TestCase):
    def test_qp_against_independent_slsqp(self):
        rng=np.random.default_rng(20)
        for m in (4,5,6):
            for _ in range(8):
                A=rng.normal(size=(m,m));Q=A.T@A+.1*np.eye(m);r=rng.normal(size=m)
                w=simplex_qp(Q,r)[0]
                opt=minimize(lambda v:.5*v@Q@v-v@r,np.ones(m)/m,
                    jac=lambda v:Q@v-r,bounds=[(0,None)]*m,
                    constraints={'type':'eq','fun':lambda v:v.sum()-1,'jac':lambda v:np.ones(m)},
                    method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
                self.assertTrue(opt.success);np.testing.assert_allclose(w,opt.x,atol=3e-6)

    def test_canonical_moments(self):
        rng=np.random.default_rng(21)
        for m in (4,5,6):
            C=np.array([covariance(rng.normal(size=(100,m))@rng.normal(size=(m,m)))+.01*np.eye(m) for _ in range(12)])
            rho,g2,_=iu_moments(C,.25)
            for i,c in enumerate(C):
                fit=upcr_fit_covariance(c,var_y=.25,**DEFAULT_IU_FIT)
                np.testing.assert_allclose(rho[i],fit.rho_hat_full,atol=1e-11)
                self.assertAlmostEqual(g2[i],fit.g2_hat,places=12)

    def test_eta_identity_and_units(self):
        C=np.eye(5);sd=np.arange(1.,6.)
        w,t=fusion_weights(C,sd,.25,eta=0)
        np.testing.assert_array_equal(w,np.ones((1,5))*.2)
        np.testing.assert_allclose(t['a'],w*sd/sd.mean())

    def test_group_restriction(self):
        C=np.diag([1.,2.,3.,4.,5.,6.]);w,_=fusion_weights(C,np.ones(6),.25,'group',eta=1)
        self.assertAlmostEqual(w.sum(),1.);np.testing.assert_allclose(w[0,:3],w[0,0])
        np.testing.assert_allclose(w[0,3:],w[0,3])

    def test_random_null_preserves_kernel_and_no_self(self):
        rng=np.random.default_rng(22);x=rng.normal(size=(100,5));z=rng.normal(size=(100,2));q=rng.normal(size=(20,2))
        _,_,a=local_moments(x,z,q);_,_,b=local_moments(x,z,q,random_seed=12)
        np.testing.assert_array_equal(a,b);self.assertGreater(a.min(),60.)

    def test_cca_current_input_firewall(self):
        rng=np.random.default_rng(23);H=rng.normal(size=(320,16,6));X=rng.normal(size=(320,6))
        for mode in ('linear','history_square_only','second_moment'):
            fit=CCAContext().fit(H,X,mode);q=rng.normal(size=(10,16,6))
            np.testing.assert_array_equal(fit.transform(q,np.ones((10,6))),fit.transform(q,np.zeros((10,6))))
            L,Y=fit.views(q,np.full((10,6),-2.))
            np.testing.assert_array_equal(Y,np.full((10,6),4. if mode=='second_moment' else -2.))

    def test_no_labels_in_fit_interfaces(self):
        import inspect
        for f in (iu_moments,simplex_qp,fusion_weights,local_moments,CCAContext.fit):
            self.assertNotIn('labels',inspect.signature(f).parameters)

    def test_population_moments_against_frozen_generator(self):
        from scripts.run_cca_iu_isolation_gate import old,population,WORLDS
        for world in WORLDS:
            sample=old._world(9286,world,100000)
            for active in (np.r_[np.ones(3),np.zeros(3)],np.r_[np.zeros(3),np.ones(3)]):
                mask=sample[7][:,0]==bool(active[0]);x=sample[0][mask];target=sample[5][mask]
                C,rho=population(world,active)
                np.testing.assert_allclose(covariance(x),C,atol=.055)
                observed=(x-x.mean(axis=0)).T@(target-target.mean())/len(x)
                np.testing.assert_allclose(observed,rho,atol=.035)


if __name__=='__main__': unittest.main()
