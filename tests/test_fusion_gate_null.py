import unittest
import numpy as np
from spectral_utils import fusion_gate_null as f


class GateNullTests(unittest.TestCase):
    def test_stationary_covariance_exact(self):
        n=12
        for rho in (0.,.6,.9):
            A=np.column_stack([f.stationary_ar(np.eye(n)[:,j],rho) for j in range(n)])
            expected=rho**np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
            np.testing.assert_allclose(A@A.T,expected,atol=1e-14,rtol=0)

    def test_affine_normalization_and_warm_alignment(self):
        _,_,tail,context,meta=f.generate(32,.6,42,False)
        np.testing.assert_allclose(f.normalize(tail*3+10)[0],tail,atol=1e-14,rtol=0)
        out,raw,model=f.curves(context,32);r=model['observation_variance'];mean=0.;variance=1.;history=[]
        for value in context:
            prediction=variance+.01*r;gain=prediction/(prediction+r);mean+=gain*(value-mean)
            variance=(1-gain)*prediction;history.append(mean)
        np.testing.assert_allclose(raw['kalman_warm'],history[-32:],atol=1e-13,rtol=1e-13)

    def test_bic_likelihood_identity(self):
        x=np.random.default_rng(44).normal(size=64);g=f.inspect_mixture(x,x)
        np.testing.assert_allclose(g['bic'],[2*np.log(64)-2*g['log_likelihood'][0],5*np.log(64)-2*g['log_likelihood'][1]],atol=1e-12,rtol=0)
        self.assertEqual(g['two_components_selected'],g['bic_gain']>0)


if __name__=='__main__':unittest.main()
