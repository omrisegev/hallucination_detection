"""Policy tests: new fit eligibility must not reroute banks or hide failures."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils import fusion_prediction_quality as core
from spectral_utils.answer_localization_v2 import moment_plan


def fixture():
    plan=moment_plan(64,8)
    arrays={'window_starts':plan.starts,'window_ends':plan.ends,'fit_indices':plan.fit_indices}
    meta={'diagnostics':{'banks':{}}}
    for bank,v in [('moment',1.),('context',2.)]:
        for kind in core.KINDS:
            arrays[bank+'__'+kind+'__features']=np.full((8,36),v)
            meta['diagnostics']['banks'][bank+'__'+kind]={'names':[str(i) for i in range(36)]}
    original={'routing':{'routes':{'dual':'context_joint'}},'official_step_starts':[0,16,32,48],
        'official_step_ends':[16,32,48,64], 'methods':{
            'context__joint0':{'valid':True},'moment__iu':{'valid':True,'decision_valid':True,'prediction':0}}}
    return arrays,meta,original


def fake_fit(valid=True,numerical_failure=False):
    def fit(values,names,indices,identity):
        np.testing.assert_array_equal(values,2.)  # original bank stays context
        risks={c:np.arange(8,dtype=float)+(i*.1) for i,c in enumerate(core.CORES)}
        methods={c:{'valid':True} for c in core.CORES}
        if not valid:
            for c in core.JOINT_CORES:methods[c]={'valid':False,'reason':'PLANTED_FIT_FAILURE'};risks.pop(c)
        if numerical_failure:methods['graph010']={'valid':False,'reason':'PLANTED_INVERSE_FAILURE'};risks.pop('graph010')
        return {},risks,methods,{'joint_valid':valid,'joint_failure':'PLANTED_FIT_FAILURE'}
    return fit


class QualityPolicyTests(unittest.TestCase):
    def test_fit_failure_falls_back_in_original_bank(self):
        a,m,o=fixture()
        with patch.object(core,'fit_bank',side_effect=fake_fit(False)), patch.object(core,'mixture_readout',return_value={'prediction':0}):
            arrays,methods,diagnostics=core.score_augmented(a,m,o,64,'fixture')
        self.assertEqual(diagnostics['bank'],'context')
        for kind in core.KINDS:
            for c in core.JOINT_CORES:
                d=methods[kind+'__'+c]
                self.assertTrue(d['fallback_to_augmented_iu']);self.assertTrue(d['valid'])
                self.assertEqual(d['source_arm'],kind+'__native_iu')
                np.testing.assert_array_equal(arrays[kind+'__'+c+'__risk'],arrays[kind+'__iu__risk'])

    def test_readout_and_numeric_failures_do_not_trigger_fallback(self):
        a,m,o=fixture()
        with patch.object(core,'fit_bank',side_effect=fake_fit(True,True)), patch.object(core,'mixture_readout',side_effect=ValueError('PLANTED_READOUT_FAILURE')):
            arrays,methods,_=core.score_augmented(a,m,o,64,'fixture')
        for kind in core.KINDS:
            for c in core.JOINT_CORES:self.assertFalse(methods[kind+'__'+c]['fallback_to_augmented_iu'])
            self.assertFalse(methods[kind+'__graph010']['valid'])
            self.assertNotIn(kind+'__graph010__risk',arrays)
            self.assertTrue(methods[kind+'__joint0']['valid'])
            self.assertFalse(methods[kind+'__joint0']['decision_valid'])
            self.assertTrue(methods[kind+'__joint0']['fixed_iu_valid'])
            self.assertEqual(methods[kind+'__joint0']['source_arm'],kind+'__native_joint0')

    def test_equal_graph_matches_explicit_laplacian(self):
        rng=np.random.default_rng(63);z=rng.normal(size=(40,9));z-=z.mean(0);z/=z.std(0)
        graph=rng.uniform(.1,1.,size=(40,40));graph=(graph+graph.T)/2;np.fill_diagonal(graph,0.)
        fit_indices=np.arange(40)
        for lam in (0.,.1):
            risk,detail=core.graph_head(z,fit_indices,0,np.eye(9),np.ones(9)/9,csr_matrix(graph),100,lam)
            degree=graph.sum(1);lap=np.eye(40)-graph/np.sqrt(np.outer(degree,degree))
            rough=z.T@lap@z/40;rough=(rough+rough.T)/2;rough*=9/np.trace(rough)
            system=np.eye(9)+lam*rough
            ev,q=np.linalg.eigh(system);psd=(q*np.maximum(ev,0.))@q.T
            ridge=np.linalg.eigvalsh(psd).max()*1e-10
            w=np.linalg.solve(psd+ridge*np.eye(9),np.ones(9)/9);w/=np.std(z@w)
            if np.corrcoef(z@w,z.mean(1))[0,1]<0:w=-w
            np.testing.assert_allclose(risk,-z@w,atol=1e-10,rtol=1e-10)
            self.assertLessEqual(detail['inverse']['condition_after'],100.)


if __name__=='__main__':unittest.main(verbosity=2)
