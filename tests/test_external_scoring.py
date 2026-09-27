"""Scientific invariants for the locked external comparison."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT'):os.environ[k]='1'
import ast,copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.metrics import f1_score,recall_score
from spectral_utils.external_generalization.evaluation import confusion,metric
from spectral_utils.external_generalization.scoring import score_answer,ALL_ARMS
from spectral_utils.external_generalization.fusion import fit_weights
ROOT=Path(__file__).resolve().parents[1]

class ExternalScoringTests(unittest.TestCase):
    def test_official_metrics(self):
        path=ROOT/'scratch/external_generalization_private/sources/hard2verify/utils.py'
        if not path.exists():self.skipTest('pinned official source unavailable')
        nodes=[n for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='calculate_metrics']
        ns={'recall_score':recall_score};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
        rng=np.random.default_rng(101)
        for _ in range(20):
            y=rng.integers(2,size=101);p=rng.integers(2,size=101);c=confusion(y,p)
            self.assertEqual(round(float(metric(c,'hard2verify'))*100,2),ns['calculate_metrics'](p,y)['balanced_f1_score'])
            self.assertAlmostEqual(float(metric(c,'socratic')),f1_score(y,p,average='macro'))
            tp=sum(a==b==1 for a,b in zip(y,p));fp=sum(a==0 and b==1 for a,b in zip(y,p))
            tn=sum(a==b==0 for a,b in zip(y,p));fn=sum(a==1 and b==0 for a,b in zip(y,p))
            np.testing.assert_array_equal(c,[tp,fp,tn,fn])

    def test_pinned_socratic_aggregation(self):
        path=ROOT/'scratch/external_generalization_private/sources/prmeval_classified_task.py'
        if not path.exists():self.skipTest('pinned official source unavailable')
        names={'evaluate_function','eval_on_hallucination_step'}
        nodes=[n for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
        ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
        rng=np.random.default_rng(11)
        for _ in range(20):
            results=[];meta=[];total=np.zeros(4,dtype=int)
            for i in range(15):
                y=rng.integers(2,size=41);pred=rng.integers(2,size=41)
                errors=(np.flatnonzero(y==0)+1).tolist()+[1000]  # official inert OOB index
                meta.append({'idx':str(i),'classification':'fixture','error_steps':errors})
                results.append({'idx':str(i),'scores':{'step_level_validity_labels':pred.tolist()}})
                row=ns['eval_on_hallucination_step'](errors,pred.tolist())['f1_matrix']
                actual=np.array([row[k] for k in ('TP','FP','TN','FN')])
                np.testing.assert_array_equal(actual,confusion(y,pred));total+=actual
            official=ns['evaluate_function'](results,meta)['total_hallucination_results']
            self.assertAlmostEqual(float(metric(total,'socratic')),(official['f1']+official['negative_f1'])/2)

    def test_numerical_failure_is_not_native_equal(self):
        from spectral_utils.external_generalization import fusion
        rng=np.random.default_rng(19)
        with patch.object(fusion.numerical_backend,'eigh',side_effect=np.linalg.LinAlgError('injected failure')):
            with self.assertRaises(ValueError):fit_weights(rng.normal(size=(100,11)))

    def test_shared_bootstrap_matches_pairwise(self):
        from spectral_utils.external_bootstrap import bootstrap_all,contrast_from_draws
        from spectral_utils.external_generalization.evaluation import paired_bootstrap
        rng=np.random.default_rng(12)
        counts={k:rng.integers(1,8,size=(20,4)) for k in ('a','b')}
        groups=[str(i//2) for i in range(20)]
        for benchmark in ('hard2verify','socratic'):
            names,samples=bootstrap_all(counts,groups,benchmark,draws=1000)
            new=contrast_from_draws(counts,names,samples,'a','b',groups,benchmark)
            old=paired_bootstrap(counts['a'],counts['b'],groups,benchmark,18,draws=1000)
            self.assertEqual(new,old)

    def test_empty_and_label_isolation(self):
        paths=sorted((ROOT/'scratch/external_generalization_private/cpu_full_preflight').glob('*.record.json'))
        if not paths:self.skipTest('real forward fixture unavailable')
        row=json.loads(paths[0].read_text())['payload']['telemetry']
        row['step_token_spans'].insert(1,[0,0])
        rng=np.random.default_rng(1)
        bundle={'fit':fit_weights(rng.normal(size=(100,11))),'thresholds':{k:.8 for k in ALL_ARMS}}
        a=score_answer(row,bundle)
        changed=copy.deepcopy(row);changed['correct']=[True]*len(row['step_token_spans'])
        b=score_answer(changed,bundle)
        self.assertEqual(a['scores'],b['scores']);self.assertEqual(a['predictions'],b['predictions'])
        for arm in ALL_ARMS:
            self.assertIsNone(a['scores'][arm][1]);self.assertEqual(a['predictions'][arm][1],0)
        self.assertFalse(a['local']['native'])
        self.assertEqual(a['scores']['local_lsml'],a['scores']['local_equal'])
        self.assertEqual(a['scores']['local_lsml'],a['scores']['local_partition_equal'])

if __name__=='__main__':unittest.main()
