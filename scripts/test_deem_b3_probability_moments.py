"""Data-contract and estimator replay tests; no benchmark labels."""
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.deem_b3_probability_moments import inputs,fit_all,METHODS
from spectral_utils.deem_b3_contract_ablation import GenericEnergy
from spectral_utils.residual_graph_deem import ContinuousDeemConfig

class Tests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(342)
        self.p=np.sort(rng.dirichlet(np.ones(50),size=60),axis=1)[:,::-1]
        self.lp=np.log(self.p);self.a=rng.uniform(.1,5,60)

    def test_probability_semantics(self):
        d=inputs(self.lp,self.a)['prob17'];r=d['soft'][:,1,:]
        full=np.column_stack([1-self.p[:,0],self.p[:,1:15],-np.expm1(-self.a),1-self.p[:,:15].sum(axis=1)])
        np.testing.assert_allclose(r,full[:,d['keep']])
        np.testing.assert_allclose(d['soft'].sum(axis=1),1.)

    def test_moment_adapter_and_selected_scope(self):
        d=inputs(self.lp,self.a);e=inputs(self.lp,self.a+1)
        np.testing.assert_array_equal(d['var15']['soft'],e['var15']['soft'])
        self.assertEqual(d['var15']['soft'].shape,(60,2,15))
        self.assertTrue(np.all((d['var15']['soft']>0)&(d['var15']['soft']<1)))

    def test_native_soft_and_b3_short_fit_replay(self):
        import torch
        torch.set_num_threads(1)
        fits,failures,_=fit_all(self.lp,self.a,'test',epochs=2)
        self.assertFalse(failures);self.assertEqual(set(fits),set(METHODS))
        banks=inputs(self.lp,self.a)
        for bank in banks:
            f=fits[bank+'__b3'];Z=banks[bank]['Z'];k=Z.shape[1]
            model=GenericEnergy(tuple(f'coordinate_{i}' for i in range(k)),{'input_bank':tuple(range(k))},ContinuousDeemConfig(),2)
            with torch.no_grad():
                for key,param in [('a',model.a),('b',model.b)]:param.copy_(torch.as_tensor(f['state'][key]))
                for prefix in ['w','W','d','V','e']:
                    getattr(model,prefix)['input_bank'].copy_(torch.as_tensor(f['state'][prefix+'::input_bank']))
                ell,_,_=model.logit(torch.as_tensor(Z,dtype=torch.float64))
                q=torch.sigmoid(ell).numpy()
            if f['diagnostics']['orientation']<0:q=1-q
            np.testing.assert_allclose(q,f['score'],atol=1e-12)
            self.assertEqual(fits[bank+'__deem']['score'].shape,(60,))

if __name__=='__main__':unittest.main()
