"""Check polynomial inputs, chosen-token scope and unchanged fusion contracts."""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.surprisal_power_fusion import representation,fit_all,METHODS
from spectral_utils.direct_probability_fusion import zscore_columns


class PowerTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(340)
        self.lp=-np.sort(rng.uniform(.1,20,(100,50)),axis=1)
        self.chosen=rng.uniform(.1,25,100);self.entropy=rng.uniform(.1,3,100)

    def test_layout_and_chosen_powers(self):
        S=np.column_stack((-self.lp[:,:15],self.chosen))
        for d in (1,2,3):
            X=representation(self.lp,self.chosen,d);self.assertEqual(X.shape,(100,16*d))
            for power in range(1,d+1):np.testing.assert_array_equal(X[:,16*(power-1):16*power],S**power)

    def test_chosen_only_changes_its_columns(self):
        a=representation(self.lp,self.chosen,3);b=representation(self.lp,self.chosen+1,3)
        changed=np.any(a!=b,axis=0)
        np.testing.assert_array_equal(np.flatnonzero(changed),[15,31,47])

    def test_truncated_ranks_and_no_row_mixing(self):
        np.testing.assert_array_equal(representation(self.lp,self.chosen,3),representation(self.lp[:,:15],self.chosen,3))
        lp=self.lp.copy();lp[70:]*=2
        np.testing.assert_array_equal(representation(self.lp,self.chosen,3)[:70],representation(lp,self.chosen,3)[:70])

    def test_equal_after_powers_and_full_reconstruction(self):
        fits,failures,_=fit_all(self.lp,self.chosen,self.entropy)
        self.assertFalse(failures);self.assertEqual(set(fits),set(METHODS))
        for m,f in fits.items():
            X=representation(self.lp,self.chosen,int(m[1]))
            np.testing.assert_allclose(X@f['effective']+f['intercept'],f['score'],atol=1e-8)
            if m.endswith('equal'):np.testing.assert_allclose(zscore_columns(X)[0].mean(axis=1),f['score'],atol=1e-14)

    def test_short_constant_invalid(self):
        fits,failures,_=fit_all(self.lp[:2],self.chosen[:2],self.entropy[:2]);self.assertFalse(fits);self.assertEqual(len(failures),6)
        fits,failures,_=fit_all(np.repeat(self.lp[:1],20,axis=0),np.ones(20),np.ones(20));self.assertEqual(len(failures),6)
        with self.assertRaises(ValueError):representation(self.lp,np.full(100,np.nan),3)
        with self.assertRaises(ValueError):representation(self.lp,self.chosen,4)


if __name__=='__main__':unittest.main()
