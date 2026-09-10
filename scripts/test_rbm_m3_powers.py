"""Test ablation isolation, old scorer replay and raw-power RBM state replay."""
import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils import rbm_m3_powers as new,moment_rbm_fusion as old
from spectral_utils.direct_probability_fusion import zscore_columns

class Tests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(938)
        self.lp=np.log(np.sort(rng.dirichlet(np.ones(50),size=90),axis=1)[:,::-1]);self.a=rng.uniform(.1,6,90)
        self.h=old.representation(self.lp,self.a)[:,0]

    def test_banks_and_chosen_powers_retained(self):
        b=new.banks(self.lp,self.a)
        self.assertEqual(b['mom5'].shape,(90,5));self.assertEqual(b['power48'].shape,(90,48))
        np.testing.assert_array_equal(b['mom5'],b['mom6'][:,[0,1,3,4,5]])
        np.testing.assert_array_equal(b['power48'][:,[15,31,47]],np.column_stack([self.a,self.a**2,self.a**3]))

    def test_original_replay_and_latent_state(self):
        with patch.object(old,'METHODS',('rbm',)):
            prior,fail,_=old.fit_all(self.lp,self.a,'test');self.assertFalse(fail)
        fits,fail,_=new.fit_all(self.lp,self.a,self.h);self.assertFalse(fail)
        np.testing.assert_array_equal(prior['rbm']['score'],fits['mom6_rbm']['score'])
        b=new.banks(self.lp,self.a)
        for method,f in fits.items():
            bank='mom6' if method.startswith('mom6') else 'mom5' if method.startswith('mom5') else 'power48'
            Z,_,_,_=zscore_columns(b[bank]);s=f['state'];q=expit(s['b']+Z@s['w'])
            if f['diagnostics']['orientation']<0:q=1-q
            np.testing.assert_array_equal(q,f['score'])

    def test_removed_m3_cannot_influence_reduced_fit(self):
        before,fail,_=new.fit_all(self.lp,self.a,self.h);self.assertFalse(fail)
        original=new.moments
        def changed(lp,chosen):
            X=original(lp,chosen);X[:,2]=np.arange(len(X))**2;return X
        with patch.object(new,'moments',changed):after,fail,_=new.fit_all(self.lp,self.a,self.h)
        self.assertFalse(fail)
        np.testing.assert_array_equal(before['mom5_rbm']['score'],after['mom5_rbm']['score'])
        np.testing.assert_array_equal(before['power48_rbm']['score'],after['power48_rbm']['score'])

if __name__=='__main__':unittest.main()
