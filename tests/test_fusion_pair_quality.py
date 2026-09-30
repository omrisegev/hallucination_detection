import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT/'results/fusion_replication_v1';AUDIT=ROOT/'results/joint_pair_identifiability_audit_v1'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.fusion_pair_quality import score_pair_banks,compose_pair_routes,CORES
from spectral_utils.joint_pair_jacobian import profiled_pair_jacobian


def load(p):return json.loads(p.read_text(encoding='utf-8'))


class PairQualityTests(unittest.TestCase):
    def test_real_replay_pair_refit_and_infeasible_pair_case(self):
        manifest=load(PARENT/'MANIFEST.json');records=[]
        allrows=[load(AUDIT/'rows'/(r['uid']+'.json')) for r in manifest['selected']]
        for predicate in [lambda r:r['banks']['moment'].get('same_partition_as_parent') and r['banks']['moment']['old_valid'],
                          lambda r:r['banks']['moment']['valid'] and r['banks']['moment'].get('pairs',{}).get('pair_count',0)>0,
                          lambda r:r['banks']['moment']['status'].startswith('PAIR_')]:
            records.append(next(r for r in allrows if predicate(r)))
        for audit in records:
            uid=audit['uid'];parent=load(PARENT/'scores'/(uid+'.json'))
            with np.load(PARENT/'scores'/(uid+'.npz'),allow_pickle=False) as file:
                pa={k:file[k] for k in file.files}
            with np.load(PARENT/'inputs'/(uid+'.npz'),allow_pickle=False) as inp:
                ar,methods,routes,_=score_pair_banks(inp['raw'],inp['step_starts'],inp['step_ends'],
                    manifest['scoring_namespace']+'/'+audit['cell']+'/'+audit['row_id'],pa,parent)
            with np.load(AUDIT/'rows'/(uid+'.npz'),allow_pickle=False) as expected:
                for bank in ('moment','context'):
                    detail=audit['banks'][bank]
                    self.assertEqual(methods['pair_'+bank+'__joint0']['valid'],detail['valid'])
                    if bank+'__covariance' in expected.files:
                        np.testing.assert_allclose(ar['pair_'+bank+'__covariance'],expected[bank+'__covariance'],atol=1e-10,rtol=1e-10)
                    if detail.get('same_partition_as_parent') and detail['old_valid']:
                        for core in CORES:
                            for suffix in ('window','risk'):
                                np.testing.assert_allclose(ar['pair_'+bank+'__'+core+'__'+suffix],pa[bank+'__'+core+'__'+suffix],atol=1e-9,rtol=1e-9)
                            self.assertEqual(methods['pair_'+bank+'__'+core]['prediction'],parent['methods'][bank+'__'+core]['prediction'])
            for arm,detail in parent['methods'].items():
                self.assertEqual(methods[arm],detail)
                if detail['valid']:
                    for suffix in ('window','risk'):np.testing.assert_array_equal(ar[arm+'__'+suffix],pa[arm+'__'+suffix])
            if audit['banks']['moment']['status'].startswith('PAIR_'):
                self.assertEqual(routes['routes']['single'],'moment_iu')
                self.assertEqual(routes['routes']['dual'],'context_joint')

    def test_selected_readout_failure_does_not_route_elsewhere(self):
        methods={};arrays={}
        for bank in ('moment','context'):
            for core in CORES:
                a='pair_'+bank+'__'+core
                methods[a]={'valid':True,'decision_valid':bank=='context','fixed_iu_valid':False,'prediction':None if bank=='moment' else -1}
                arrays[a+'__window']=np.array([1.,2.]);arrays[a+'__risk']=np.array([2.])
            for core in ('equal','iu'):
                a=bank+'__'+core;methods[a]={'valid':True,'decision_valid':True,'fixed_iu_valid':True,'prediction':-1}
                arrays[a+'__window']=np.array([3.,4.]);arrays[a+'__risk']=np.array([4.])
        _,out,routes=compose_pair_routes(arrays,methods)
        self.assertEqual(routes['routes']['dual'],'moment_joint')
        self.assertFalse(out['pair_dual__graph010']['decision_valid'])
        self.assertIsNone(out['pair_dual__graph010']['prediction'])
        for bank in ('moment','context'):
            for core in CORES:methods['pair_'+bank+'__'+core].update(valid=False,decision_valid=False,fixed_iu_valid=False)
        _,out,routes=compose_pair_routes(arrays,methods)
        self.assertEqual(routes['routes']['dual'],'moment_iu')
        self.assertEqual(out['pair_dual__graph010']['source_arm'],'moment__iu')
        self.assertEqual(out['pair_dual__graph010']['prediction'],-1)


if __name__=='__main__':unittest.main()
