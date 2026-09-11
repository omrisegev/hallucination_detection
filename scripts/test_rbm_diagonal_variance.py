"""Mathematical equivalence, gradient, and bounded-fit checks."""
from pathlib import Path
import sys,unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.rbm_diagonal_variance import diagonal_objective,fit_from_original,covariance_diagnostics,VARIANCE_FLOOR
from spectral_utils.moment_rbm_fusion import rbm_objective,fit_rbm


class DiagonalTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(441)
        self.X=rng.normal(size=(151,4))
        self.theta=rng.normal(scale=.3,size=13)

    def test_gradient(self):
        for strength in (0.,.1):
            fun=lambda t:diagonal_objective(t,self.X,strength)
            numeric=approx_derivative(lambda t:fun(t)[0],self.theta,method='3-point').ravel()
            np.testing.assert_allclose(fun(self.theta)[1],numeric,atol=3e-8,rtol=2e-6)

    def test_unit_variance_bridge(self):
        t=self.theta.copy();t[9:]=0
        loss,grad=diagonal_objective(t,self.X)
        old,g=rbm_objective(t[:9],self.X)
        np.testing.assert_allclose(loss,old,atol=2e-15,rtol=0)
        np.testing.assert_allclose(grad[:9],g,atol=2e-15,rtol=0)

    def test_independent_gaussian_mixture_likelihood(self):
        a,w,b,logs=self.theta[:4],self.theta[4:8],self.theta[8],self.theta[9:]
        var=np.exp(logs);s=b+np.sum((a*w+.5*w*w)/var)
        component0=-.5*logs.sum()-.5*np.sum((self.X-a)**2/var,axis=1)-np.logaddexp(0,s)
        component1=-.5*logs.sum()-.5*np.sum((self.X-a-w)**2/var,axis=1)-np.logaddexp(0,-s)
        independent=-np.mean(np.logaddexp(component0,component1))+.1*np.sum(logs**2)
        np.testing.assert_allclose(diagonal_objective(self.theta,self.X)[0],independent,atol=2e-15,rtol=0)

    def test_bounded_fit_and_posterior_coefficients(self):
        _,original,_=fit_rbm(self.X)
        for diagonal in (False,True):
            score,state,diag=fit_from_original(self.X,original,diagonal)
            self.assertTrue(np.all(state['variance']>=VARIANCE_FLOOR-1e-12))
            self.assertLessEqual(diag['objective_final'],diag['objective_initial']+1e-10)
            np.testing.assert_array_equal(score,expit(state['b']+self.X@(state['w']/state['variance'])))

    def test_total_covariance_formula(self):
        a=np.array([.1,.3]);w=np.array([1.,-1.]);var=np.array([.2,.5]);b=-.7
        pi=expit(b+np.sum((a*w+.5*w*w)/var))
        rng=np.random.default_rng(112)
        X=a+rng.binomial(1,pi,size=(100000,1))*w+rng.normal(size=(100000,2))*np.sqrt(var)
        diag=covariance_diagnostics(X,dict(a=a,w=w,b=b,variance=var))
        self.assertLess(diag['variance_rmse'],.01)
        self.assertLess(diag['covariance_relative_error'],.02)


if __name__=='__main__':unittest.main()
