import sys,unittest,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_global_moment_fusion_24 import aggregate_tokens,old
from scripts.verify_global_moment_fusion_24 import rank_auc
from spectral_utils import moment_rbm_fusion as core

class Tests(unittest.TestCase):
    def test_summaries_use_each_coordinate_top_tokens(self):
        X=np.column_stack((np.arange(13),np.arange(13)[::-1]))
        np.testing.assert_allclose(aggregate_tokens(X),[7.5,7.5])
        np.testing.assert_allclose(aggregate_tokens(X[:2]),[.5,11.5])

    def test_refactor_replays_frozen_localization_estimator(self):
        import torch
        torch.set_num_threads(1)
        original=ROOT.parent/'moment-rbm-fusion-v1/spectral_utils/moment_rbm_fusion.py'
        spec=importlib.util.spec_from_file_location('spectral_utils._frozen_rbm_reference',original)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rng=np.random.default_rng(713);lp=np.log(np.sort(rng.dirichlet(np.ones(50),size=55),axis=1)[:,::-1]);a=rng.uniform(.1,5,55)
        ref,fail,_=module.fit_all(lp,a,'replay',epochs=2);self.assertFalse(fail)
        new,fail,_=core.fit_matrix(core.representation(lp,a),'replay',epochs=2);self.assertFalse(fail)
        for m in core.METHODS:
            np.testing.assert_array_equal(ref[m]['score'],new[m]['score'])
            for k,v in ref[m]['state'].items():np.testing.assert_array_equal(v,new[m]['state'][k])

    def test_group_auc_ties_matches_repeated_samples(self):
        y=np.array([0,1,0,1],bool);s=np.array([0.,0.,1.,2.]);groups=np.array([0,0,1,2])
        counts=np.array([[1,1,1],[2,1,3]])
        actual=old._weighted_auc_batch(y,s,groups,counts)
        for i,c in enumerate(counts):
            index=np.repeat(np.arange(4),c[groups]);yp=y[index];sp=s[index]
            d=sp[yp][:,None]-sp[~yp][None,:]
            expected=np.mean((d>0)+.5*(d==0))
            np.testing.assert_allclose(actual[i],expected)
            np.testing.assert_allclose(rank_auc(yp,sp),expected)

if __name__=='__main__':unittest.main()
