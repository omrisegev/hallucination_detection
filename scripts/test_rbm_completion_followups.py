"""End-to-end worker checks for the dependent stability/depth stages."""
import io
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_rbm_literature_completion as run
from scripts.review_rbm_literature_completion import replay_one
from spectral_utils import rbm_literature_completion as model


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(19)
        x=rng.normal(size=(83,12))+(rng.random(83)>.65)[:,None]*1.7
        x=(x-x.mean(axis=0))/x.std(axis=0)
        cls.spans=np.array([[0,5],[5,29],[28,57],[57,83]])  # short step plus shared boundary token
        cls.banks={};states={}
        for bank in (6,12):
            z=np.ascontiguousarray(x[:,:bank]);theta,_=model.exact_fit(z,1)
            cls.banks[bank]=dict(x=z,theta=theta,orientation=1,columns=np.arange(bank))
            for h in (1,4):
                t,_=model.exact_fit(z,h,seed=12 if h==4 else 0)
                _,sign=model.oriented_units(z,t,h,x[:,:6].mean(axis=1))
                states[f'b{bank}_exact{h}::theta']=t
                states[f'b{bank}_exact{h}::signs']=sign
        cls.data=(0,'synthetic-followup-no-labels',cls.spans,x[:,:6].mean(axis=1),cls.banks)
        cls.capacity=(states,{})

    def test_starts_choose_only_lowest_likelihood(self):
        _,blob,info=run.worker(('stability',self.data,self.capacity));info=json.loads(info)
        self.assertFalse(info['failures'])
        with np.load(io.BytesIO(blob)) as z:
            arrays={k:z[k] for k in z.files}
            for bank in (6,12):
                for h in (1,4):
                    key=f'b{bank}_best_exact{h}';obj=model.ExactRBM(self.banks[bank]['x'],h)
                    starts=arrays[key+'::starts'];loss=np.array([obj(t)[0] for t in starts])
                    selected=info['models'][key]['chosen_start']
                    self.assertEqual(selected,int(np.argmin(loss)))
                    np.testing.assert_array_equal(arrays[key+'::theta'],starts[selected])
            self.assertEqual(replay_one(self.data,arrays,info),8)

    def test_second_layer_retains_identical_first_representation(self):
        _,blob,info=run.worker(('depth',self.data,self.capacity));info=json.loads(info)
        self.assertFalse(info['failures'])
        with np.load(io.BytesIO(blob)) as z:
            arrays={k:z[k] for k in z.files}
            for bank in (6,12):
                for variant in ('layer2_exact','layer2_cd'):
                    key=f'b{bank}_{variant}'
                    np.testing.assert_array_equal(arrays[key+'::first'],self.capacity[0][f'b{bank}_exact4::theta'])
                    np.testing.assert_array_equal(arrays[key+'::firstsign'],self.capacity[0][f'b{bank}_exact4::signs'])
            self.assertEqual(replay_one(self.data,arrays,info),8)


if __name__=='__main__':
    with threadpool_limits(limits=1):unittest.main()
