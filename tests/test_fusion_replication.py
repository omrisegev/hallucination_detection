"""Fixed-recipe replay before any new replication scores are evaluated."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.fusion_replication import score_fixed_banks,ARMS


class ReplicationReplay(unittest.TestCase):
    def test_three_distinct_existing_routing_cases(self):
        parent=ROOT/'results/fusion_explicit_fallback_pilot_v1'
        raw_root=ROOT/'results/answer_localization_representation_pilot_v1'
        manifest=json.loads((parent/'MANIFEST.json').read_text());cases={}
        for rec in manifest['selected']:
            meta=json.loads((parent/'scores'/(rec['uid']+'.json')).read_text())
            route=meta['routing']['routes']['dual']
            cases.setdefault(route,(rec,meta))
        self.assertEqual(set(cases),{'moment_joint','context_joint','moment_iu'})
        for route,(rec,meta) in cases.items():
            with self.subTest(route=route), np.load(raw_root/'inputs'/(rec['uid']+'.npz'),allow_pickle=False) as inp:
                arrays,methods,routing,_=score_fixed_banks(inp['raw'],inp['step_starts'],inp['step_ends'],
                    manifest['release_id']+'/'+rec['cell']+'/'+rec['row_id'])
            self.assertEqual(routing,meta['routing'])
            with np.load(parent/'scores'/(rec['uid']+'.npz'),allow_pickle=False) as old:
                for arm in ARMS:
                    for key in ('valid','decision_valid','fixed_iu_valid','prediction','fixed_iu_prediction'):
                        self.assertEqual(methods[arm].get(key),meta['methods'][arm].get(key),(route,arm,key))
                    if methods[arm]['valid']:
                        for suffix in ('window','risk'):
                            np.testing.assert_allclose(arrays[arm+'__'+suffix],old[arm+'__'+suffix],atol=1e-10,rtol=1e-10)

    def test_degenerate_answer_retains_explicit_failures(self):
        x=np.ones((64,29));arrays,methods,routing,_=score_fixed_banks(x,[0],[64],'constant')
        self.assertTrue(all(not v['valid'] and not v['decision_valid'] for v in methods.values()))
        self.assertEqual(routing['routes']['dual'],'moment_iu')
        self.assertFalse(any(k.endswith('__risk') for k in arrays))


if __name__=='__main__':unittest.main()
