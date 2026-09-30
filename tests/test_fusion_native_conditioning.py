import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT/'results/fusion_pair_quality_v1';ORIGINAL=ROOT/'results/fusion_replication_v1'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.fusion_native_conditioning import score_conditioning,compose_fixed_routes,CONDITIONS
from spectral_utils.joint_lsml import regularized_joint_map_weights

def load(p):return json.loads(p.read_text(encoding='utf-8'))


class ConditioningTests(unittest.TestCase):
    def test_known_spectrum_and_ridge_not_graph_lambda(self):
        rng=np.random.default_rng(214);q,_=np.linalg.qr(rng.normal(size=(4,4)))
        eig=np.array([0.,.01,1.,12.]);cov=(q*eig)@q.T;v=q@np.array([.3,-.1,.2,.4])
        previous=-1.
        for condition in (1000,300,100,30):
            w,d=regularized_joint_map_weights(np.eye(4),cov,v,mode='liu',lam=0.,target_condition=condition)
            ridge=12./(condition-1)
            np.testing.assert_allclose(w,q@((q.T@v)/(eig+ridge)),atol=2e-12,rtol=2e-12)
            self.assertAlmostEqual(d['ridge'],ridge,places=12)
            self.assertAlmostEqual(d['condition_after'],condition,places=7)
            self.assertEqual(d['lambda'],0.);self.assertGreater(d['ridge'],previous);previous=d['ridge']

    def test_original_scores_and_all_three_routes_replay(self):
        manifest=load(ORIGINAL/'MANIFEST.json');selected={}
        for rec in manifest['selected']:
            original=load(ORIGINAL/'scores'/(rec['uid']+'.json'))
            selected.setdefault(original['routing']['routes']['dual'],(rec,original))
        self.assertEqual(set(selected),{'moment_joint','context_joint','moment_iu'})
        for route,(rec,original) in selected.items():
            parent=load(PARENT/'scores'/(rec['uid']+'.json'))
            with np.load(PARENT/'scores'/(rec['uid']+'.npz'),allow_pickle=False) as file:pa={k:file[k] for k in file.files}
            oldmeta=deepcopy(parent);oldorig=deepcopy(original)
            ar,methods,routing,diag=score_conditioning(pa,parent,original,rec['tokens'])
            self.assertEqual(parent,oldmeta);self.assertEqual(original,oldorig);self.assertEqual(routing,original['routing'])
            for k,x in pa.items():np.testing.assert_array_equal(ar[k],x)
            for k,d in parent['methods'].items():self.assertEqual(methods[k],d)
            for bank in ('moment','context'):
                for condition in CONDITIONS:self.assertEqual(methods[f'{bank}__cond{condition}']['valid'],original['methods'][bank+'__joint0']['valid'])
                if original['methods'][bank+'__joint0']['valid']:
                    self.assertEqual(diag['banks'][bank]['status'],'ORIGINAL_FIT_REPLAY_PASS')
                    self.assertLess(diag['banks'][bank]['maximum_parent_risk_difference'],1e-10)
            for condition in CONDITIONS:
                self.assertEqual(methods[f'dual__cond{condition}']['route'],route)
                if route=='moment_iu':
                    np.testing.assert_array_equal(ar[f'dual__cond{condition}__risk'],pa['moment__iu__risk'])

    def test_failed_new_head_or_readout_cannot_change_original_route(self):
        arrays={};methods={}
        for bank in ('moment','context'):
            for condition in CONDITIONS:
                a=f'{bank}__cond{condition}';methods[a]={'valid':True,'decision_valid':bank=='context','fixed_iu_valid':False,'prediction':-1 if bank=='context' else None}
                for suffix in ('window','risk'):arrays[a+'__'+suffix]=np.array([1.,2.])
        methods['moment__iu']={'valid':True,'decision_valid':True,'fixed_iu_valid':True,'prediction':-1}
        for suffix in ('window','risk'):arrays['moment__iu__'+suffix]=np.array([3.,4.])
        routing={'routes':{'single':'moment_joint','dual':'moment_joint'}}
        _,out=compose_fixed_routes(arrays,methods,routing)
        self.assertFalse(out['dual__cond30']['decision_valid']);self.assertEqual(out['dual__cond30']['source_arm'],'moment__cond30')
        methods['moment__cond30']['valid']=False
        ar,out=compose_fixed_routes(arrays,methods,routing)
        self.assertFalse(out['dual__cond30']['valid']);self.assertNotIn('dual__cond30__risk',ar)
        self.assertEqual(out['dual__cond30']['source_arm'],'moment__cond30')


if __name__=='__main__':unittest.main()
