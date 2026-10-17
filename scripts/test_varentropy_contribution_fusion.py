"""Mechanism checks, not benchmark performance tests."""
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.varentropy_contribution_fusion import contributions,fit_all,METHODS
from spectral_utils.token_feature_views import _logprob_token_series
from spectral_utils.direct_probability_fusion import zscore_columns


class ContributionTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(339)
        logits=np.sort(rng.normal(size=(160,50))*rng.uniform(.3,3,size=(160,1)),axis=1)[:,::-1]
        p=np.exp(logits-logits.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
        self.lp=np.log(p)

    def test_exact_frozen_raw(self):
        frozen=_logprob_token_series({'logprobs':self.lp},len(self.lp))['topk_varentropy_series']
        np.testing.assert_array_equal(contributions(self.lp,50).sum(axis=1),frozen)

    def test_rank_trim_and_scalar(self):
        C=contributions(self.lp,15)
        np.testing.assert_array_equal(C,contributions(self.lp[:,:15],15))
        for row,c in zip(self.lp[:5],C[:5]):
            p=np.exp(row[:15]);p=p/(sum(p)+1e-12);s=-np.log(p+1e-12);h=sum(p*s)
            np.testing.assert_allclose(sum(float(q*(v-h)**2) for q,v in zip(p,s)),c.sum(),rtol=1e-13)

    def test_uniform_and_known_binary(self):
        self.assertLess(contributions(np.log(np.full((4,50),.02)),50).max(),1e-20)
        p=np.array([[.9,.1]])
        expected=.09*np.log(9)**2
        self.assertAlmostEqual(contributions(np.log(p),2).sum(),expected,places=10)

    def test_all_fits_reconstruct_and_equal(self):
        fits,failures,_=fit_all(self.lp);self.assertFalse(failures);self.assertEqual(set(fits),set(METHODS))
        for name,fit in fits.items():
            k=int(name.split('__')[0][1:]);C=contributions(self.lp,k)
            np.testing.assert_allclose(C@fit['effective']+fit['intercept'],fit['score'],atol=1e-8)
            if name.endswith('__equal'):
                np.testing.assert_allclose(fit['score'],zscore_columns(C)[0].mean(axis=1),atol=1e-14)

    def test_short_constant_and_nonfinite(self):
        fits,failures,_=fit_all(self.lp[:2]);self.assertEqual(len(failures),4)
        self.assertEqual(set(fits),{'k15__raw','k50__raw'})
        fits,failures,_=fit_all(np.repeat(self.lp[:1],30,axis=0));self.assertEqual(len(failures),4)
        bad=self.lp.copy();bad[0,0]=np.nan
        with self.assertRaises(ValueError):contributions(bad,50)


if __name__=='__main__':unittest.main()
