import unittest
import numpy as np
from spectral_utils.fusion_entropy_sampling import choose
from spectral_utils.fusion_window_sampling import budget


class EntropySelectors(unittest.TestCase):
    def test_two_tails_and_ties(self):
        np.testing.assert_array_equal(choose(np.arange(100.),'entropy_tails'),np.r_[np.arange(25),np.arange(75,100)])
        np.testing.assert_array_equal(choose(np.ones(100),'entropy_tails'),np.arange(50))
        # Odd budget33:16 low,17 high; no repeated windows.
        np.testing.assert_array_equal(choose(np.arange(65.),'entropy_tails'),np.r_[np.arange(16),np.arange(48,65)])

    def test_quantiles_cover_all_deciles(self):
        picked=choose(np.arange(100.),'entropy_quantiles')
        np.testing.assert_array_equal(picked,np.arange(1,100,2))
        np.testing.assert_array_equal(np.bincount(picked//10,minlength=10),np.repeat(5,10))
        np.testing.assert_array_equal(choose(-np.arange(100.),'entropy_quantiles'),np.arange(0,100,2))

    def test_every_size_and_temporal_order(self):
        rng=np.random.default_rng(8)
        for n in range(1,501):
            x=rng.integers(0,7,n)
            for selector in ('entropy_tails','entropy_quantiles'):
                pick=choose(x,selector)
                self.assertEqual(len(pick),budget(n));self.assertTrue(np.all(np.diff(pick)>0))
                self.assertTrue(np.all((pick>=0)&(pick<n)))
                if n<=32:np.testing.assert_array_equal(pick,np.arange(n))

    def test_invalid_input(self):
        for values in ([],[np.nan],[np.inf],[[1,2]]):
            with self.assertRaises(ValueError):choose(values,'entropy_tails')
        with self.assertRaises(ValueError):choose([1,2],'unknown')
