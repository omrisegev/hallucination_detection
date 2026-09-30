"""Scientific identities for the evaluation-only gate audit."""
import os
for option in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[option]='1'
from pathlib import Path
import sys
import unittest
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.answer_localization_v2 import prepare_local
from spectral_utils.fusion_gate_interface_audit import (
    projection, inspect_mixture, comparison_parts, oracle_predictions, pb_from_predictions)


class AuditIdentities(unittest.TestCase):
    def test_feature_affine_normalization_and_projection(self):
        rng=np.random.default_rng(7); x=rng.normal(size=(48,5))
        x[:,1] += .7*x[:,0]; names=['entropy_series__level','a','b','c','d']; fit=np.arange(47)
        z,anchor,normal=prepare_local(x,names,fit)
        changed=x*np.array([.5,1.,2.,3.,4.])+np.array([2.,-1.,.5,1.,2.])
        zz,anchor2,normal2=prepare_local(changed,names,fit)
        np.testing.assert_allclose(z,zz,atol=1e-12); self.assertEqual(anchor,anchor2)
        np.testing.assert_array_equal(normal['feature_signs'],normal2['feature_signs'])
        w=rng.normal(size=z.shape[1]); centered,uncentered,offset=projection(x,names,fit,normal,w)
        np.testing.assert_allclose(centered,-z@w,atol=1e-12)
        np.testing.assert_allclose(uncentered,centered+offset,atol=1e-12)

    def test_actual_gmm_translation(self):
        rng=np.random.default_rng(8); x=np.r_[rng.normal(-1,.3,50),rng.normal(1,.3,50)]
        a=inspect_mixture(x,np.array([-1.,1.])); b=inspect_mixture(x+10,np.array([9.,11.]))
        self.assertTrue(a['two_components_selected']); self.assertEqual(a['prediction'],b['prediction'])
        self.assertAlmostEqual(a['bic_gain'],b['bic_gain'],places=8)
        self.assertAlmostEqual(a['threshold']+10,b['threshold'],places=10)

    def test_auc_decomposition_including_pure_answers(self):
        y=[np.array([0,1,1]),np.array([0,0]),np.array([1,0])]
        x=[np.array([.4,.3,.8]),np.array([.2,.4]),np.array([.4,.1])]
        result=comparison_parts(y,x)
        self.assertAlmostEqual(result['pooled_auc'],roc_auc_score(np.concatenate(y),np.concatenate(x)))
        self.assertEqual(result['total_pairs'],12); self.assertEqual(result['within_pairs'],3)
        self.assertAlmostEqual(result['pooled_auc'],
            .25*result['pair_weighted_within_auc']+.75*result['cross_answer_auc'])

    def test_oracles_preserve_failure_and_wrong_location(self):
        rows=[]
        for target,valid,opened,peak in [(-1,True,True,0),(-1,False,False,0),(1,True,False,0),(1,True,True,1)]:
            rows.append({'cell':'a','target':target,'predictions':oracle_predictions(target,valid,opened,peak)})
        self.assertEqual(pb_from_predictions(rows,'actual')['macro_f1'],0.)
        self.assertAlmostEqual(pb_from_predictions(rows,'perfect_gate')['macro_f1'],.5)
        self.assertAlmostEqual(pb_from_predictions(rows,'both_perfect')['macro_f1'],2/3)
        self.assertIsNone(rows[1]['predictions']['both_perfect'])

    def test_fixed_parameter_replication_bic_identity(self):
        rng=np.random.default_rng(12); x=rng.normal(size=80); a=inspect_mixture(x,x)
        l1,l2=a['log_likelihood']; delta=2*(l2-l1)-3*np.log(len(x))
        self.assertAlmostEqual(a['bic_gain'],delta,places=9)
        # k=1 has mean+variance=2 params; k=2 has 2 means+2 variances+one mixing probability=5.
        bic1=-2*(2*l1)+2*np.log(160); bic2=-2*(2*l2)+5*np.log(160)
        self.assertAlmostEqual(a['duplicated_fixed_parameter_bic_gain'],bic1-bic2,places=9)


if __name__=='__main__': unittest.main()
