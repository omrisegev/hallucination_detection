"""Independent algebra and sampling checks; no benchmark-label tuning."""
import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit, logsumexp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spectral_utils.rbm_literature_completion import (
    ExactRBM, initial, unpack, pack, TwoStateVariance, shared_to_mixture,
    markov_inference, markov_fit, mean_unit_scores, cd_fit, exact_fit)
from spectral_utils.moment_rbm_fusion import rbm_objective


class AlgebraTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(12)
        self.x=self.rng.normal(size=(31,5))

    def test_exact_gradients_one_and_four(self):
        for h in (1,4):
            f=ExactRBM(self.x,h);theta=initial(5,h,2)
            theta+=self.rng.normal(0,.1,len(theta))
            num=approx_derivative(lambda t:f(t)[0],theta,method='3-point').ravel()
            np.testing.assert_allclose(f(theta)[1],num,atol=2e-8,rtol=1e-6)

    def test_one_unit_historical_equivalence(self):
        theta=initial(5,1,2)
        loss,g=ExactRBM(self.x,1)(theta);old,og=rbm_objective(theta,self.x)
        np.testing.assert_allclose(loss,old,atol=1e-14)
        np.testing.assert_allclose(g,og,atol=1e-14)

    def test_four_unit_explicit_mixture(self):
        theta=initial(5,4,2);a,w,b=unpack(theta,5,4)
        hs=np.array(list(itertools.product([0.,1.],repeat=4)))
        means=a+hs@w.T
        lp=hs@b+(hs@w.T)@a+.5*np.square(hs@w.T).sum(axis=1)
        logpi=lp-logsumexp(lp)
        comp=-.5*np.square(self.x[:,None,:]-means).sum(axis=2)+logpi
        expected=-logsumexp(comp,axis=1).mean()
        self.assertAlmostEqual(expected,ExactRBM(self.x,4)(theta)[0],places=12)

    def test_variance_gradients_and_shared_identity(self):
        mu=self.rng.normal(size=(2,5));v=self.rng.normal(size=5)*.2
        ts=np.r_[mu.ravel(),v,.3];td=np.r_[mu.ravel(),v,v,.3]
        self.assertAlmostEqual(TwoStateVariance(self.x,False)(ts)[0],
                               TwoStateVariance(self.x,True)(td)[0],places=12)
        for sep,t in ((False,ts),(True,td)):
            f=TwoStateVariance(self.x,sep)
            numerical=approx_derivative(lambda u:f(u)[0],t).ravel()
            np.testing.assert_allclose(f(t)[1],numerical,atol=2e-8,rtol=1e-6)

    def test_mixture_mapping(self):
        theta=initial(5,1,2);a,w,b=unpack(theta,5,1);w=w[:,0];b=b[0]
        mu,prior=shared_to_mixture(a,w,b)
        t=np.r_[mu.ravel(),np.zeros(5),prior]
        f=TwoStateVariance(self.x,False);parts,_=f.component_logp(t)
        np.testing.assert_allclose(parts[:,1]-parts[:,0],self.x@w+b,atol=1e-14)
        self.assertAlmostEqual(f(t)[0],ExactRBM(self.x,1)(theta)[0],places=12)

    def test_stable_hidden_scores(self):
        ell=np.array([[-1000.],[-2.],[0.],[1000.]])
        logit,post=mean_unit_scores(ell)
        np.testing.assert_allclose(logit,ell[:,0],atol=1e-13)
        np.testing.assert_allclose(post,expit(logit))

    def test_actual_gibbs_moments(self):
        a=np.array([.2,-.4]);w=np.array([.6,-.3]);b=.1
        prior=expit(b+a@w+.5*w@w)
        h=self.rng.random(200000)<prior
        x=a+h[:,None]*w+self.rng.normal(size=(len(h),2))
        np.testing.assert_allclose(x.mean(axis=0),a+prior*w,atol=.009)
        np.testing.assert_allclose(np.cov(x.T),np.eye(2)+prior*(1-prior)*np.outer(w,w),atol=.015)
        # Gibbs h|x then x|h leaves the same joint-model marginal invariant.
        h2=self.rng.random(len(x))<expit(x@w+b)
        x2=a+h2[:,None]*w+self.rng.normal(size=x.shape)
        np.testing.assert_allclose(x2.mean(axis=0),a+prior*w,atol=.009)

    def test_cd_seed_and_data_dependence(self):
        t1,d=cd_fit(self.x,1,seed=9,epochs=3)
        t2,_=cd_fit(self.x,1,seed=9,epochs=3)
        t3,_=cd_fit(self.x,1,seed=10,epochs=3)
        np.testing.assert_array_equal(t1,t2)
        self.assertGreater(np.linalg.norm(t1-t3),1e-6)
        self.assertTrue(np.isfinite(d['nll_final']))

    def test_markov_against_enumerated_sequences(self):
        ll=np.array([.4,-2.,1.3,3.1,-.7]);A=np.array([[.85,.15],[.25,.75]])
        prior=.3;starts=np.array([True,False,False,False,False])
        score,counts=markov_inference(ll,A,prior,starts)
        paths=np.array(list(itertools.product([0,1],repeat=len(ll))))
        logp=[]
        for h in paths:
            l=np.log(expit(prior) if h[0] else expit(-prior))+h@ll
            l+=sum(np.log(A[i,j]) for i,j in zip(h[:-1],h[1:]))
            logp.append(l)
        p=np.exp(logp-logsumexp(logp));marg=p@paths
        expected=np.zeros((2,2))
        for h,q in zip(paths,p):
            for a,b in zip(h[:-1],h[1:]):expected[a,b]+=q
        np.testing.assert_allclose(expit(score),marg,atol=1e-13)
        np.testing.assert_allclose(counts,expected,atol=1e-13)

    def test_markov_independence_and_resets(self):
        ell=np.array([-1000.,.2,5.,-3.,1000.]);prior=.4
        A=np.tile([expit(-prior),expit(prior)],(2,1))
        starts=np.array([True,False,True,False,False])
        score,c=markov_inference(ell-prior,A,prior,starts)
        np.testing.assert_allclose(score,ell,atol=1e-12)
        self.assertAlmostEqual(c.sum(),3.)
        A=np.array([[.8,.2],[.3,.7]])
        full,_=markov_inference(ell-prior,A,prior,starts)
        l,_=markov_inference(ell[:2]-prior,A,prior,np.array([True,False]))
        r,_=markov_inference(ell[2:]-prior,A,prior,np.array([True,False,False]))
        np.testing.assert_array_equal(full,np.r_[l,r])


if __name__=='__main__':unittest.main()
