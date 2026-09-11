"""Numerical contract tests for saved-parameter score replay."""
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.special import expit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectral_utils.rbm_logit_readout import saved_scores


class ReplayTests(unittest.TestCase):
    def test_orientations_top10_and_extremes(self):
        x=np.array([-1000,-100,-4,-1,0,1,2,3,4,5,6,7,40,100,1000.])
        spans=np.array([[0,15],[0,3],[3,4]])
        for orientation in (-1,1):
            scores,t=saved_scores(x[:,None],np.ones(1),0.,orientation,spans)
            p=expit(x) if orientation==1 else 1-expit(x)
            for name,v in [('posterior',p),('logit',orientation*x)]:
                expected=np.array([np.mean(sorted(v[a:b])[-10:]) for a,b in spans])
                np.testing.assert_allclose(scores[name],expected,rtol=1e-14,atol=1e-14)
            self.assertGreater(t['collapsed_token_pairs'],0)

    def test_constant_and_first_tie(self):
        s,_=saved_scores(np.zeros((12,1)),np.ones(1),0.,1,np.array([[0,6],[6,12]]))
        self.assertEqual(np.argmax(s['logit_near']),0)
        self.assertGreater(s['logit_near'][0],s['logit_near'][1])
        self.assertTrue(np.all(s['logit_near']>s['logit'].max()))

    def test_invalid(self):
        for x,o in [(np.array([[np.nan]]),1),(np.zeros((1,1)),0)]:
            with self.assertRaises(ValueError):saved_scores(x,np.ones(1),0.,o,np.array([[0,1]]))

if __name__=='__main__':unittest.main()
