import copy
import unittest
import numpy as np
from spectral_utils.temporal_position_profiles import answer_moments,fit_profiles,transform,CENTERS
from spectral_utils.context_training import METADATA_KEYS

class PositionTests(unittest.TestCase):
    def test_constant_profile_identity_and_first_token(self):
        x=np.array([0.,2.,-3.,7.]);p=dict(mean=[3.]*16,std=[2.]*16,global_mean=3.,global_std=2.)
        for key in ('mean_detrended','location_scale_detrended'):
            np.testing.assert_allclose(transform(x,p)[key],x)
        self.assertTrue(all(v[0]==0 for v in transform(x,p).values()))

    def test_nonlinear_profile_removal(self):
        profile=np.sin(CENTERS*5);p=dict(mean=profile.tolist(),std=[2.]*16,global_mean=0.,global_std=2.)
        x=np.interp((np.arange(160)+.5)/160,CENTERS,profile);x[0]=0
        np.testing.assert_allclose(transform(x,p)['mean_detrended'],0,atol=1e-14)

    def test_excluded_observations_and_label_firewall(self):
        meta=[];moments=[]
        for i in range(12):
            m={k:None for k in METADATA_KEYS};m.update(uid=str(i),group_id=str(i),fold=i%3,cell='cell',tokens=64)
            meta.append(m);moments.append(answer_moments(np.arange(64,dtype=float)+i))
        a=fit_profiles(meta,moments,[0],min_groups=2)
        changed=np.array(moments);changed[::3,:,1:]*=1000
        self.assertEqual(a,fit_profiles(meta,changed,[0],min_groups=2))
        labeled=copy.deepcopy(meta);labeled[0]['correct']=True
        with self.assertRaises(ValueError):fit_profiles(labeled,moments,[0])

    def test_short_and_constant(self):
        np.testing.assert_array_equal(answer_moments([0]),np.zeros((16,3)))
        m=answer_moments(np.full(64,3.));self.assertAlmostEqual(m[:,0].sum(),1)
        self.assertAlmostEqual(m[:,1].sum(),3)

if __name__=='__main__':unittest.main()
