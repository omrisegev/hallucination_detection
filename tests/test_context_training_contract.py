import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from spectral_utils.context_training import FeatureBundle
from spectral_utils.context_readout import prediction_readouts
from spectral_utils.temporal_context_models import probability_features,flow_condition
from spectral_utils.direct_probability_fusion import step_top_mean


class TrainingContract(unittest.TestCase):
    def bundle(self,path):
        rng=np.random.default_rng(1);lp=rng.normal(size=(30,15))
        original=probability_features(torch.tensor(lp),torch.ones(4)).numpy()
        prefix=np.r_[0.,np.cumsum(original[:-1,0])]
        innovation=original[:,0]-np.divide(prefix,np.arange(30),out=np.zeros(30),where=np.arange(30)>0)
        innovation[0]=0
        features=np.column_stack((original,innovation))
        np.save(path/'features.npy',features.astype(np.float32));np.save(path/'logprobs15.npy',lp.astype(np.float32))
        np.save(path/'h0_prefix_sum.npy',prefix);np.save(path/'step_spans.npy',[[0,10],[10,30]])
        meta=dict(uid='test',cell='prm',group_id='g',fold=0,offset=0,tokens=30,step_start=0,step_stop=2,
                  mean=features.mean(0).tolist(),scale=features.std(0).tolist(),signs=[1.]*4)
        (path/'METADATA.json').write_text(json.dumps([meta]))
        (path/'MANIFEST.json').write_text(json.dumps({'banks':{'original4':[0,1,2,3],'innovation5':[0,1,2,3,4]}}))
        return FeatureBundle(path,'innovation5')

    def test_probability_condition_and_future_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=self.bundle(Path(tmp));batch=b.batch([0,0,0],[0,9,29],flow=True,with_target=False)
            original=flow_condition(batch['history'],batch['mask'],batch['position'])
            reconstructed=b.condition_from_probabilities(batch,batch['source_logits'])
            np.testing.assert_allclose(original.numpy(),reconstructed.numpy(),atol=2e-6)
            past=b.batch([0],[9],flow=False)
            # The TCN's latest visible token is 8, even across the step boundary.
            self.assertEqual(int(past['local'][0,-1]),8)
            self.assertNotIn('target',batch)
            metadata=json.loads((Path(tmp)/'METADATA.json').read_text());metadata[0]['label']=1
            (Path(tmp)/'METADATA.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):FeatureBundle(tmp)
            del b  # Release Windows memory maps before TemporaryDirectory cleanup.

    def test_variance_fusion_and_constant_residual_identities(self):
        rng=np.random.default_rng(2);raw=rng.normal(size=(30,4));spans=np.array([[0,8],[8,19],[19,30]])
        base=np.mean([step_top_mean(raw[:,j],spans[:,0],spans[:,1],10) for j in range(4)],axis=0)
        out=prediction_readouts(raw,raw,raw,spans,base,np.ones_like(raw))
        np.testing.assert_allclose(out['variance_fusion'],base,rtol=1e-14,atol=1e-14)
        np.testing.assert_array_equal(out['squared_residual_0.25'],base)


if __name__=='__main__':unittest.main()
