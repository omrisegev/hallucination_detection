import unittest
from unittest.mock import patch
import numpy as np
from spectral_utils.fusion_token_gap import provided_token_gap,gap_matrix,chosen_core,apply_readout
from spectral_utils.answer_localization_v2 import moment_plan


class TokenGapTest(unittest.TestCase):
    def test_log_normalizer_cancels(self):
        logits=np.array([[1.,2.,4.],[-5.,0.,3.],[2.,2.,0.]])
        logz=np.log(np.exp(logits).sum(axis=1));raw=np.zeros((3,29));raw[:,15]=logz-logits[:,1];raw[:,23]=logits.max(axis=1)-logz
        np.testing.assert_allclose(provided_token_gap(raw),logits.max(axis=1)-logits[:,1],atol=1e-14)
        self.assertEqual(provided_token_gap(raw)[2],0.)

    def test_invalid_distributions_rejected(self):
        raw=np.zeros((16,29));raw[:,15]=.1;raw[:,23]=-1.
        with self.assertRaisesRegex(ValueError,'DISTRIBUTION_MISMATCH'):provided_token_gap(raw)
        raw[0,23]=np.nan
        with self.assertRaisesRegex(ValueError,'NONFINITE'):provided_token_gap(raw)

    def test_dimension_and_untouched_columns(self):
        rng=np.random.default_rng(13);raw=rng.normal(size=(64,29));raw[:,15]=2+np.abs(raw[:,15]);raw[:,23]=-.2
        for bank in ('moment','context'):
            plan,x,names,base,_,gap=gap_matrix(raw,bank)
            self.assertEqual(x.shape,(8,27));keep=[i for i in range(27) if i not in (3,4,5)]
            np.testing.assert_array_equal(x[:,keep],base[:,keep]);self.assertTrue(all('provided_token_gap' in names[i] for i in (3,4,5)))

    def test_readout_failure_never_chooses_iu(self):
        self.assertEqual(chosen_core('graph010',True),'graph010');self.assertEqual(chosen_core('graph010',False),'iu')
        plan=moment_plan(16,8);ref={'valid':True,'decision_valid':True,'prediction':-1}
        with patch('spectral_utils.fusion_token_gap.mixture_readout',side_effect=ValueError('readout failed')):
            step,detail=apply_readout(np.array([0.,1.]),{'valid':True},plan,np.array([0,8]),np.array([8,16]),ref)
        self.assertFalse(detail['decision_valid']);self.assertEqual(chosen_core('graph010',True),'graph010')


if __name__=='__main__':unittest.main()
