import importlib.util
import os
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prm_transfer', ROOT/'results/prm_boosting_generalization_v1/evaluate.py')
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)


class Evaluation(unittest.TestCase):
    def test_historical_all_6969_answers(self):
        path = os.environ.get('PRM_V1_METADATA')
        if not path: self.skipTest('PRM_V1_METADATA must point at existing historical PRM metadata')
        report = E.replay_v1(Path(path))
        self.assertEqual(report['n_checked'], 6969)
        self.assertAlmostEqual(report['metrics']['S3_own']['within_auc'], .8508269739045268, places=8)
        self.assertAlmostEqual(report['metrics']['S3_own']['prmscore'], 69.38457404510251, places=8)
        self.assertAlmostEqual(report['metrics']['EQ2_own']['within_auc'], .8404734467946724, places=8)
        self.assertAlmostEqual(report['metrics']['EQ2_own']['prmscore'], 68.53382311808627, places=8)

    def test_gates_fail_closed(self):
        good = dict(token_surprisal_median=.2,n_rows_step_text_mismatch=0,n_rows=6969,n_failed_rows=0)
        parity = dict(max_abs_error=0.,digit_active_mismatch=0)
        self.assertTrue(E.check_gates(68.2,good,parity,6969,68.2)['publish_fused'])
        for score, manifest, p, n in [(65.,good,parity,6969),(68.2,dict(good,token_surprisal_median=1),parity,6969),
                                       (68.2,dict(good,n_rows_step_text_mismatch=1),parity,6969),
                                       (68.2,good,dict(parity,max_abs_error=1e-8),6969),
                                       (68.2,good,parity,40),(68.2,dict(good,n_failed_rows=1),parity,6969),
                                       (68.2,dict(good,status='INCOMPLETE'),parity,6969)]:
            self.assertFalse(E.check_gates(score,manifest,p,n,68.2)['publish_fused'])

    def test_frozen_definitions(self):
        o=np.array([0,3,6]); prm=np.array([-.3,1.,2.,0.,2.,1.]); n16=prm[::-1]
        own={c: E.az(prm*(k+1),o) for k,c in enumerate(E.S3+('hist_topk_varentropy_series',))}
        q3={c:E.az(n16*(k+1),o) for k,c in enumerate(E.S3+('hist_energy_cusum_abs_series',))}
        arms=E.frozen_arms(prm,n16,own,q3,o)
        np.testing.assert_array_equal(arms['S3_own'],.5*E.az(prm,o)+.5*np.column_stack([own[c] for c in E.S3]).mean(1))
        np.testing.assert_array_equal(arms['EQ2_own'],.5*E.az(prm,o)+.5*E.az(n16,o))
        self.assertEqual(set(arms),{'PRM_native','PRM_Q80','N16_own','EQ2_own','S3_own','S3_q3','S6_own'})

    def test_paired_group_bootstrap_is_repeatable(self):
        ids=np.array([str(i) for i in range(10)]);o=np.arange(0,41,4)
        y=np.tile([False,True,False,True],10);s=np.tile([0.,2.,1.,3.],10)
        meta=[dict(idx=i,classification='confidence',error_steps=[2,4]) for i in ids]
        arms={k:s.copy() for k in ['PRM_native','PRM_Q80','N16_own','S3_own','EQ2_own']}
        flags={k:np.tile([False,True,False,True],10) for k in arms}
        groups=np.array(['a']*5+['b']*5)
        report=E.bootstrap(arms,flags,y,o,ids,meta,groups)
        self.assertEqual(report,E.bootstrap(arms,flags,y,o,ids,meta,groups))
        self.assertEqual(len(report),8)
        self.assertTrue(all(r['delta']==r['ci_low']==r['ci_high']==0 for r in report))
        self.assertTrue(all(r['n_source_groups']==2 and r['draws']==2000 and r['seed']==20261013 for r in report))


if __name__ == '__main__': unittest.main()
