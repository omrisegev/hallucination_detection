import unittest
import numpy as np
from spectral_utils.digit_alternative_probability import (
    second_digit_probability, disagreement, prefix_innovation, step_top_mean, digit_innovation_step_max)


class DigitAlternativeTests(unittest.TestCase):
    def test_hand_examples(self):
        ids = np.array([[14,17,99], [14,99,17], [99,14,17]])
        p = np.array([[.45,.40,.10], [.8,.1,.02], [.9,.002,.001]])
        lo, hi, seen = second_digit_probability(ids, np.log(p), range(10,20))
        np.testing.assert_allclose(lo, [.4,.02,.001])
        np.testing.assert_array_equal(lo, hi)
        np.testing.assert_array_equal(seen, [2,2,2])

    def test_censor_bounds_against_complete_distribution(self):
        p = np.array([[.7,.15,.07,.04,.02,.01,.005,.003,.001,.001]])
        ids = np.array([[99,98,14,97,17,16,15,13,12,11]])
        full = second_digit_probability(ids, np.log(p), range(10,20))[0]
        for k in [2,3,4,5,10]:
            lo, hi, _ = second_digit_probability(ids[:,:k], np.log(p[:,:k]), range(10,20))
            self.assertTrue(np.all(lo <= full+1e-15))
            self.assertTrue(np.all(full <= hi+1e-15))

    def test_selected_token_invariance_and_greedy_degeneracy(self):
        ids = np.array([[14,17,99],[17,14,99]])
        lp = np.log([[.45,.4,.1],[.6,.3,.05]])
        expected = second_digit_probability(ids,lp,range(10,20))[0]
        np.testing.assert_array_equal(disagreement(ids[:,0],ids[:,0],range(10,20)), [0,0])
        np.testing.assert_array_equal(disagreement(ids[:,1],ids[:,0],range(10,20)), [1,1])
        np.testing.assert_array_equal(expected, second_digit_probability(ids,lp,range(10,20))[0])
        self.assertTrue(np.all(expected > 0))

    def test_causal_innovation_and_pooling(self):
        x=np.array([0.,1.,0.,1.])
        np.testing.assert_allclose(prefix_innovation(x), [0,1,-.5,2/3])
        np.testing.assert_allclose(prefix_innovation(x)[:3],prefix_innovation(x[:3]))
        np.testing.assert_allclose(step_top_mean(x,[[0,4],[0,1]],2),[1,0])

    def test_reject_invalid(self):
        for ids, p in [([[14,14]], [[.5,.2]]), ([[14,17]],[[.2,.5]]), ([[14,17]],[[.8,.6]])]:
            with self.assertRaises(ValueError):
                second_digit_probability(ids,np.log(p),range(10,20))
        with self.assertRaises(ValueError):
            step_top_mean([1,2],[[1,1]],2)

    def test_historical_innovation_mask_is_not_all_tokens(self):
        scores, available = digit_innovation_step_max(
            np.array([1.,0.,0.,0.]), np.array([14,14,99,99]), range(10,20), [[1,3],[2,4]])
        np.testing.assert_array_equal(scores, [-1.,0.])
        np.testing.assert_array_equal(available, [True,False])


if __name__ == '__main__':
    unittest.main()
