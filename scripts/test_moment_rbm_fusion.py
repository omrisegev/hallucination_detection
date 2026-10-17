"""Independent gradient/mixture checks and fitted-score replay, no benchmark labels."""
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize import check_grad
from scipy.special import expit, logsumexp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.moment_rbm_fusion import representation,rbm_objective,fit_rbm,fit_all,METHODS
from spectral_utils.direct_probability_fusion import zscore_columns,step_top_mean
from spectral_utils.varentropy_contribution_fusion import contributions
from spectral_utils.deem_b3_contract_ablation import GenericEnergy
from spectral_utils.residual_graph_deem import ContinuousDeemConfig

class Tests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(574)
        self.X=rng.normal(size=(70,6));self.theta=rng.normal(size=13)*.3
        self.lp=np.log(np.sort(rng.dirichlet(np.ones(50),size=70),axis=1)[:,::-1])
        self.a=rng.uniform(.01,7,70)

    def test_exact_gradient(self):
        error=check_grad(lambda t:rbm_objective(t,self.X)[0],lambda t:rbm_objective(t,self.X)[1],self.theta)
        self.assertLess(error,2e-6)

    def test_independent_gaussian_mixture_likelihood(self):
        a,w,b=self.theta[:6],self.theta[6:12],self.theta[-1]
        s=b+a@w+.5*w@w
        logs=np.column_stack((-.5*((self.X-a)**2).sum(axis=1)-np.logaddexp(0,s),
            -.5*((self.X-a-w)**2).sum(axis=1)-np.logaddexp(0,-s)))
        np.testing.assert_allclose(rbm_objective(self.theta,self.X)[0],-logsumexp(logs,axis=1).mean(),atol=1e-12)

    def test_features_and_constants(self):
        X=representation(self.lp,self.a)
        self.assertEqual(X.shape,(70,6))
        np.testing.assert_array_equal(X[:,1],contributions(self.lp,15).sum(axis=1))
        np.testing.assert_allclose(X[:,4:],np.column_stack((self.a**2,self.a**3)))
        const=np.tile(self.lp[0],(5,1))
        _,fail,_=fit_all(const,np.ones(5),'constant',epochs=1)
        self.assertEqual(set(fail),set(METHODS))
        np.testing.assert_allclose(step_top_mean(np.array([1.,3.,2.]),np.array([0,1]),np.array([1,3])),[1,2.5])

    def test_fit_state_replay_and_determinism(self):
        import torch
        torch.set_num_threads(1)
        fits,fail,_=fit_all(self.lp,self.a,'test',epochs=2)
        self.assertFalse(fail);self.assertEqual(set(fits),set(METHODS))
        Z,keep,_,_=zscore_columns(representation(self.lp,self.a))
        for m in METHODS:
            f=fits[m];s=f['state'];sign=f['diagnostics']['orientation']
            if m in ('equal','iu'):pred=sign*(Z@s['w'])
            elif m.startswith('rbm'):
                pred=expit(s['b']+Z@s['w']);pred=pred if sign>0 else 1-pred
            else:
                p=Z.shape[1];model=GenericEnergy(tuple(str(i) for i in range(p)),{'moments':tuple(range(p))},ContinuousDeemConfig(),1)
                with torch.no_grad():
                    model.a.copy_(torch.as_tensor(s['a']));model.b.copy_(torch.as_tensor(s['b']))
                    for key in ('w','W','d','V','e'):getattr(model,key)['moments'].copy_(torch.as_tensor(s[key+'::moments']))
                    ell,_,_=model.logit(torch.as_tensor(Z,dtype=torch.float64))
                pred=expit(ell.numpy());pred=pred if sign>0 else 1-pred
            np.testing.assert_allclose(pred,f['score'],atol=1e-12)
        again,_,_=fit_all(self.lp,self.a,'test',epochs=2)
        for m in METHODS:np.testing.assert_array_equal(fits[m]['score'],again[m]['score'])
        d=fits['rbm']['diagnostics'];self.assertLessEqual(d['nll_final'],d['nll_initial']+1e-10)

if __name__=='__main__':unittest.main()
