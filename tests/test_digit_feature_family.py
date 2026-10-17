import unittest
import numpy as np
from spectral_utils.digit_feature_family import token_family,step_family,answer_standardize


class FamilyTests(unittest.TestCase):
    def test_entropy_mass_weighting(self):
        ids=np.array([[15,16,99],[99,15,16]])
        p=np.array([[.4,.4,.1],[.9,.04,.04]])
        x,_,_=token_family(ids,np.log(p),range(15,25))
        np.testing.assert_allclose(x[:,1],[.8*np.log(2)/np.log(10),.08*np.log(2)/np.log(10)])

    def test_breadth_changes_while_second_probability_same(self):
        ids=np.array([[15,16,99,98],[15,16,17,99]])
        p=np.array([[.4,.2,.1,.05],[.4,.2,.1,.05]])
        x,_,_=token_family(ids,np.log(p),range(15,25))
        self.assertEqual(x[0,0],x[1,0]);self.assertGreater(x[1,1],x[0,1])

    def test_causal_and_choice_independent_api(self):
        ids=np.tile([15,16,99],(4,1));p=np.array([[.8,.05,.01],[.8,.05,.01],[.45,.4,.01],[.5,.3,.01]])
        x,active,_=token_family(ids,np.log(p),range(15,25))
        np.testing.assert_allclose(x[:3],token_family(ids[:3],np.log(p[:3]),range(15,25))[0])
        np.testing.assert_allclose(x[2,2],.35);self.assertFalse(active[0,2])
        # No provided/generated-token argument exists; greedy selection leaves x nonzero.
        self.assertGreater(x[:,0].sum(),0)

    def test_missing_innovation_does_not_set_background(self):
        x=np.array([[.1,.2,0],[.2,.4,.1],[.3,.6,.2]])
        active=np.ones(x.shape,bool);active[0,2]=False
        steps,available=step_family(x,active,[[0,1],[1,2],[2,3]])
        z=answer_standardize(steps,[0,3],available)
        np.testing.assert_allclose(z[:,2],[0,-1,1])


if __name__=='__main__':unittest.main()
