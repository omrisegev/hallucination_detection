"""Scientific invariants for the corrected historical comparison, no real labels."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    output=importlib.util.module_from_spec(spec);spec.loader.exec_module(output)
    return output


class HistoricalRefitContract(unittest.TestCase):
    def test_nested_source_exclusion(self):
        core=module('refit_contract','spectral_utils/historical_fusion_refit.py')
        groups=['held','held','a','a','b','c','d']
        outer=dict(held=0,a=1,b=2,c=3,d=4);inner=dict(a=0,b=1,c=2,d=3)
        for stage in (None,0,1,2,3):
            train,evaluate=core.fold_masks(groups,outer,inner,0,stage)
            self.assertFalse(train[:2].any())
            self.assertEqual(bool(train[2]),bool(train[3]))
            if stage is not None:self.assertFalse(evaluate[:2].any())

    def test_metric_edge_cases(self):
        core=module('metric_contract','spectral_utils/historical_fusion_evaluation.py')
        self.assertEqual(core.review_fixtures()['status'],'PASS')

    def test_paired_common_rows_and_zero_difference(self):
        evaluator=module('bootstrap_contract','scripts/evaluate_historical_fusion_refit_v3.py')
        records=[];target=[];labels=[];scores=[];outer=[]
        for fold in range(5):
            for task in ('prmbench_qwen3_8b','pb_fixture_q4','pb_fixture_q8'):
                for j in range(4):
                    records.append(dict(group_id=f'{fold}:{task}:{j//2}',cell=task,steps=2))
                    outer.append(fold)
                    target.append(-2 if task.startswith('prm') else (-1 if j<2 else j%2))
                    labels.extend([0,1] if task.startswith('prm') else [-2,-2])
                    scores.extend([[.1,.1,.1],[.9,.9,.9]])
        n=len(records);arms=['dual__iu','equal_all23','iu_test']
        arrays=dict(scores=np.array(scores),labels=np.array(labels),target=np.array(target),
            offsets=np.arange(0,2*n+1,2),valid=np.ones((n,3),bool),decision=np.ones((n,3),bool),
            predictions=np.tile(target,(3,1)).T,within=np.full((n,3),np.nan),peaks=np.zeros((n,3),int))
        for i,r in enumerate(records):
            if r['cell'].startswith('prm'):arrays['within'][i,:]=1.
        arrays['valid'][0,0]=False;arrays['within'][0,0]=np.nan;arrays['decision'][5,0]=False
        evaluator.DRAWS=12
        result=evaluator.bootstrap(records,arms,arrays,np.array(outer),['equal_all23','iu_test'])
        self.assertEqual(result['contrasts'][0]['prm_common_answers'],19)
        for contrast in result['contrasts']:
            self.assertEqual(contrast['prm_common_point_difference'],[0.,0.])
            for endpoint in ('prm_fold_mean_auc','prm_within_auc'):
                self.assertEqual(contrast['intervals'][endpoint]['low'],0.)
                self.assertEqual(contrast['intervals'][endpoint]['high'],0.)


if __name__=='__main__':unittest.main()
