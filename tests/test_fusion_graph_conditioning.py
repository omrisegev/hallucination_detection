import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT/'results/fusion_native_conditioning_v1';ORIGINAL=ROOT/'results/fusion_replication_v1'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.fusion_graph_conditioning import score_graph_conditioning,graph_head,compose_routes,CONDITIONS,GRAPHS

def load(p):return json.loads(p.read_text(encoding='utf-8'))


class GraphConditionTests(unittest.TestCase):
    def test_laplacian_inverse_and_equal_control_cap_invariance(self):
        rng=np.random.default_rng(621);z=rng.normal(size=(30,6));z-=z.mean(0);z/=z.std(0)
        wgraph=rng.uniform(.1,1.,size=(30,30));wgraph=(wgraph+wgraph.T)/2;np.fill_diagonal(wgraph,0.)
        degree=wgraph.sum(1);lap=np.eye(30)-wgraph/np.sqrt(np.outer(degree,degree))
        rough=z.T@lap@z/30;rough=(rough+rough.T)/2;rough*=6/np.trace(rough)
        system=np.eye(6)+.1*rough;evals=np.linalg.eigvalsh(system);ridge=evals[-1]*1e-10
        expected=np.linalg.solve(system+ridge*np.eye(6),np.ones(6)/6);score=z@expected
        self.assertGreater(np.corrcoef(score,z.mean(1))[0,1],.02)
        expected/=score.std();results=[]
        for cap in (*CONDITIONS,1000):
            risk,meta=graph_head(z,np.arange(30),0,np.eye(6),np.ones(6)/6,csr_matrix(wgraph),cap)
            np.testing.assert_allclose(meta['standardized_weights'],expected,atol=1e-12,rtol=1e-12)
            self.assertLessEqual(meta['inverse']['condition_after'],1.6+1e-10);results.append(risk)
        for risk in results:np.testing.assert_array_equal(risk,results[0])
        zero,_=graph_head(z,np.arange(30),0,np.eye(6),np.ones(6)/6,csr_matrix(wgraph),30,0.)
        np.testing.assert_allclose(zero,-z.mean(1)/z.mean(1).std(),atol=1e-12)

    def test_real_graph_replay_and_original_routes_without_joint_refitting(self):
        selected={}
        for rec in load(PARENT/'MANIFEST.json')['selected']:
            old=load(ORIGINAL/'scores'/(rec['uid']+'.json'));selected.setdefault(old['routing']['routes']['dual'],(rec,old))
        self.assertEqual(set(selected),{'moment_joint','context_joint','moment_iu'})
        for route,(rec,old) in selected.items():
            parent=load(PARENT/'scores'/(rec['uid']+'.json'))
            with np.load(PARENT/'scores'/(rec['uid']+'.npz'),allow_pickle=False) as file:pa={k:file[k] for k in file.files}
            saved=deepcopy(parent)
            with patch('spectral_utils.joint_lsml.fit_joint_lsml',side_effect=AssertionError('Unexpected Joint refit')) as fitter:
                ar,methods,routing,diag=score_graph_conditioning(pa,parent,old,rec['tokens']);fitter.assert_not_called()
            self.assertEqual(diag['joint_refits'],0);self.assertEqual(parent,saved);self.assertEqual(routing,old['routing'])
            for key,x in pa.items():np.testing.assert_array_equal(ar[key],x)
            for arm,d in parent['methods'].items():self.assertEqual(methods[arm],d)
            for bank in ('moment','context'):
                self.assertLess(diag['banks'][bank]['equal_zero_replay_maximum_difference'],1e-10)
                for kind in GRAPHS:
                    self.assertTrue(methods[bank+'__equal_'+kind]['valid'])
                    if old['methods'][bank+'__joint0']['valid']:self.assertLess(diag['banks'][bank][kind+'_condition1000_replay_maximum_difference'],1e-10)
                    for cap in CONDITIONS:self.assertEqual(methods[f'{bank}__cond{cap}_{kind}']['valid'],old['methods'][bank+'__joint0']['valid'])

    def test_failed_head_readout_and_distinct_equal_fallback(self):
        arrays={};methods={}
        for bank in ('moment','context'):
            for suffix in [f'cond{k}_{g}' for k in CONDITIONS for g in GRAPHS]+['equal_'+g for g in GRAPHS]:
                arm=bank+'__'+suffix;methods[arm]={'valid':True,'decision_valid':True,'fixed_iu_valid':False,'prediction':-1}
                for end in ('window','risk'):arrays[arm+'__'+end]=np.array([1.,2.])
        methods['moment__iu']={'valid':True,'decision_valid':True,'fixed_iu_valid':True,'prediction':-1}
        for end in ('window','risk'):arrays['moment__iu__'+end]=np.array([3.,4.])
        methods['moment__cond30_graph010'].update(decision_valid=False,prediction=None)
        route={'routes':{'single':'moment_joint','dual':'moment_joint'}}
        _,out=compose_routes(arrays,methods,route)
        self.assertFalse(out['dual__cond30_graph010']['decision_valid']);self.assertEqual(out['dual__cond30_graph010']['source_arm'],'moment__cond30_graph010')
        methods['moment__cond30_graph010']['valid']=False
        ar,out=compose_routes(arrays,methods,route);self.assertFalse(out['dual__cond30_graph010']['valid']);self.assertNotIn('dual__cond30_graph010__risk',ar)
        route={'routes':{'single':'moment_iu','dual':'moment_iu'}};ar,out=compose_routes(arrays,methods,route)
        self.assertEqual(out['dual__cond30_graph010']['source_arm'],'moment__iu')
        self.assertEqual(out['dual__equal_graph010']['source_arm'],'moment__equal_graph010')
        np.testing.assert_array_equal(ar['dual__cond30_graph010__risk'],[3.,4.]);np.testing.assert_array_equal(ar['dual__equal_graph010__risk'],[1.,2.])


if __name__=='__main__':unittest.main()
