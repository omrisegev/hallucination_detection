"""Signal identities and an explicit-resampling reference for the new pilot."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.fusion_context_bank import prefix_ema,context_matrix,expanded_k,score_context_bank
from spectral_utils.answer_localization_v2 import moment_plan,moment_matrix
from spectral_utils.fusion_benchmark_bootstrap import paired_source_group_intervals


class ContextContracts(unittest.TestCase):
    def test_ema_closed_form_and_first_sample(self):
        x=np.arange(1,17,dtype=float).reshape(8,2)
        for span in (8,32):
            alpha=2/(span+1);y=prefix_ema(x,span)
            for t in range(len(x)):
                expected=(1-alpha)**t*x[0]+sum(alpha*(1-alpha)**(t-j)*x[j] for j in range(1,t+1))
                np.testing.assert_allclose(y[t],expected,atol=1e-12)

    def test_constant_affine_and_prefix(self):
        x=np.random.default_rng(4).normal(size=(99,9))
        for span in (8,32):
            np.testing.assert_allclose(prefix_ema(np.ones_like(x)*7,span),7.)
            np.testing.assert_allclose(prefix_ema(2*x+3,span),2*prefix_ema(x,span)+3,atol=1e-12)
            np.testing.assert_array_equal(prefix_ema(x[:40],span),prefix_ema(x,span)[:40])

    def test_window_levels_match_parent_and_prefix_features(self):
        raw=np.random.default_rng(6).normal(size=(139,29));plan=moment_plan(len(raw),8)
        context,names=context_matrix(raw,plan);old,_=moment_matrix(raw,plan)
        np.testing.assert_array_equal(context[:,::3],old[:,::3])
        self.assertEqual(context.shape,(len(plan.starts),27));self.assertEqual(names[0],'entropy_series__level')
        changed=raw.copy();changed[80:]+=30
        cc,_=context_matrix(changed,plan)
        np.testing.assert_array_equal(context[plan.ends<=80],cc[plan.ends<=80])

    def test_expanded_group_roster_does_not_force_small_groups(self):
        self.assertEqual(expanded_k(27),(3,4,5,6,7,8,9))
        self.assertEqual(expanded_k(26),(3,4,5,6,7,8))
        self.assertEqual(expanded_k(8),())
        with self.assertRaises(ValueError):expanded_k(3.5)

    def test_cached_joint_and_permuted_graph_replay(self):
        p=ROOT/'results/answer_localization_representation_pilot_v1'
        prepared=json.loads((p/'PREPARED.json').read_text());rec=prepared['selected'][0];uid=rec['uid']
        metadata=json.loads((p/'scores'/f'{uid}.json').read_text())
        with np.load(p/'inputs'/f'{uid}.npz',allow_pickle=False) as raw,np.load(p/'scores'/f'{uid}.npz',allow_pickle=False) as old:
            arrays,methods,diagnostics=score_context_bank(raw['raw'],old,metadata,
                prepared['release_id']+'/'+rec['cell']+'/'+rec['row_id'])
        self.assertTrue(diagnostics['joint_fits']['moment_allk']['same_parent_partition'])
        for core in ('joint0','graph010','graph_perm'):
            self.assertTrue(methods['moment_allk__'+core]['valid'])
            np.testing.assert_allclose(arrays['moment_allk__'+core+'__window'],arrays['moment__'+core+'__window'],atol=1e-10,rtol=1e-10)

    def test_bootstrap_matches_explicit_parent_group_sampling(self):
        spec=importlib.util.spec_from_file_location('metric_parent',ROOT/'scripts/run_answer_localization_v2.py')
        parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
        rows=[]
        for i,y in enumerate(([0,1,0],[1,1],[0,1],[0,0])):
            rows.append({'cell':'prm_x','group_id':str(i//2),'target':y,'scores':{'a':[.2+i/10,.7,.1][:len(y)],'b':[.4,.3,.6][:len(y)]},
                'valid':{'a':True,'b':True},'decision_valid':{},'predictions':{},'fixed_iu_valid':{},'fixed_iu_predictions':{}})
        for cell in ('pb_x','pb_y'):
            for i,y in enumerate((-1,0,-1,1)):
                pred={'a':y if i!=3 else -1,'b':y if i!=0 else 0}
                rows.append({'cell':cell,'group_id':str(i),'target':y,'scores':{},'valid':{'a':True,'b':i!=2},
                    'decision_valid':{'a':True,'b':i!=2},'predictions':pred,
                    'fixed_iu_valid':{'a':True,'b':i!=2},'fixed_iu_predictions':pred})
        old=parent.paired_intervals(rows,'a','b',draws=43)
        new=paired_source_group_intervals(rows,'a','b',draws=43)
        for key in ('prm_common_valid_ci95','pb_all_population_ci95'):
            np.testing.assert_allclose(new[key],old[key],atol=1e-12)
        self.assertEqual(new['prm_common_valid_valid_draws'],old['prm_valid_draws'])
        self.assertEqual(new['pb_all_population_valid_draws'],old['pb_valid_draws'])
        np.testing.assert_allclose(new['pb_common_iu_gate_all_population_ci95'],new['pb_all_population_ci95'])
        # Independently materialize repeated rows for the additional within-answer endpoint.
        def strata(records):
            out={}
            for r in records:out.setdefault(r['cell'],{}).setdefault(r['group_id'],[]).append(r)
            return out
        def sample(groups,rng):
            out=[]
            for cell in groups.values():
                ids=sorted(cell)
                for index in rng.integers(len(ids),size=len(ids)):out.extend(cell[ids[index]])
            return out
        pp=strata([r for r in rows if r['cell'].startswith('prm')]);bb=strata([r for r in rows if r['cell'].startswith('pb_')])
        rng=np.random.default_rng(2026090706);deltas=[]
        for _ in range(43):
            chosen=sample(pp,rng);sample(bb,rng)
            a,b=[parent.prm_metric(chosen,k)['within_answer_auc'] for k in ('a','b')]
            deltas.append(a-b)
        np.testing.assert_allclose(new['prm_within_answer_common_valid_ci95'],np.quantile(deltas,[.025,.975]),atol=1e-12)


if __name__=='__main__':unittest.main()
