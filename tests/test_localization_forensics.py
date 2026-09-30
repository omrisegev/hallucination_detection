import unittest
import numpy as np
from spectral_utils.localization_forensics import character_alignment, direct_token_projection, peak_geometry, pb_outcome


class ForensicsTest(unittest.TestCase):
    def test_shared_boundary_plateau_is_not_two_independent_peaks(self):
        d = peak_geometry([0,8],[8,16],[3.,0.],[0,4],[4,16],16,[3.,3.])
        self.assertEqual(d['peak'],0)
        self.assertEqual(d['shared_peak_plateaus'],[{'windows':[0],'steps':[0,1]}])
        self.assertTrue(pb_outcome(1,0,0,d['numerical_top_steps'])['target_in_top_tie'])

    def test_disjoint_equal_scores_have_no_shared_plateau(self):
        d = peak_geometry([0,8],[8,16],[3.,3.],[0,8],[8,16],16,[3.,3.])
        self.assertEqual(d['exact_top_steps'],[0,1])
        self.assertEqual(d['shared_peak_plateaus'],[])

    def test_end_window_overlap_is_averaged(self):
        t = direct_token_projection([0,8,10],[8,16,18],[0.,2.,4.],18)
        np.testing.assert_array_equal(t,np.r_[np.zeros(8),np.full(2,2.),np.full(6,3.),np.full(2,4.)])

    def test_gap_rejected(self):
        with self.assertRaisesRegex(ValueError,'UNSCORED_TOKEN_GAP'):
            direct_token_projection([0,8],[4,12],[0.,1.],12)

    def test_character_overlap_and_separator(self):
        chars, spans = character_alignment(['abc','de'],[(0,2),(2,3),(3,5),(5,7)])
        self.assertEqual(chars,[(0,3),(5,7)])
        self.assertEqual(spans,[[0,2],[3,4]])

    def test_gate_can_hide_an_exact_peak(self):
        d = pb_outcome(2,-1,2,[2])
        self.assertEqual(d['category'],'error_gate_closed')
        self.assertTrue(d['exact_peak_hidden'])


if __name__ == '__main__': unittest.main()
