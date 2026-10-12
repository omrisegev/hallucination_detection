"""Scientific checks for sensitivity-driven fusion weight regularization."""
import os
for option in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[option]='1'
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils import fusion_reliability_regularization as reg
from spectral_utils.answer_localization_v2 import moment_matrix, moment_plan, prepare_local


class ReliabilityContract(unittest.TestCase):
    def test_second_moment_includes_bias_and_matches_direct_score_error(self):
        rng=np.random.default_rng(425); delta=rng.normal(size=(16,32,7))+1.2; w=rng.normal(size=7)
        q=reg.second_moment(delta)
        self.assertGreaterEqual(np.linalg.eigvalsh(q).min(),-1e-12)
        self.assertAlmostEqual(float(w@q@w),float(np.mean((delta@w)**2)),places=11)
        np.testing.assert_array_equal(reg.second_moment(np.ones((10,3))),np.ones((3,3)))

    def test_trace_matching_is_scale_invariant(self):
        q=np.diag([.2,1.,8.]); matrix=np.diag([1.,2.,4.])
        scaled=reg.trace_match(q,matrix)
        self.assertAlmostEqual(np.trace(scaled),np.trace(matrix))
        np.testing.assert_allclose(scaled,reg.trace_match(100*q,matrix))

    def test_selection_tolerance_ties_and_invalid_baseline(self):
        self.assertEqual(reg.choose_lambda([1.,.5,.499,.498]),1)
        self.assertEqual(reg.choose_lambda([1.,1.,1.,1.]),0)
        self.assertEqual(reg.choose_lambda([1.,np.inf,.3,.31]),2)
        with self.assertRaises(ValueError): reg.choose_lambda([np.inf,.1,.2,.3])

    def test_identity_head_isotropic_penalty_preserves_score(self):
        rng=np.random.default_rng(377); fit=rng.normal(size=(100,6)); rhs=np.ones(6)
        weights,losses,selected,_,_=reg.penalty_grid(np.eye(6),rhs,np.eye(6),fit,0,np.eye(6))
        for w in weights: np.testing.assert_allclose(fit@w,fit@weights[0],atol=1e-12)
        self.assertEqual(selected,0)
        np.testing.assert_allclose(losses,losses[0],atol=1e-12)

    def test_held_perturbations_do_not_enter_training_penalty(self):
        rng=np.random.default_rng(476); raw=rng.normal(size=(64,29));plan=moment_plan(64,8)
        values,names=moment_matrix(raw,plan); _,_,norm=prepare_local(values,names,plan.fit_indices)
        def perturb_a(raw,identity,replicate): return raw + (1 if replicate<16 else 2)
        def perturb_b(raw,identity,replicate): return raw + (1 if replicate<16 else 200)
        with patch.object(reg,'perturb_windows',perturb_a): a,b,per=reg.perturbation_moments(raw,plan,values,norm,'test')
        with patch.object(reg,'perturb_windows',perturb_b): a2,b2,_=reg.perturbation_moments(raw,plan,values,norm,'test')
        np.testing.assert_array_equal(a,a2); self.assertGreater(np.trace(b2),100*np.trace(b))
        np.testing.assert_allclose(a,per[:16].mean(axis=0)); np.testing.assert_allclose(b,per[16:].mean(axis=0))

    def test_actual_cached_joint_graph_and_all_parent_replays(self):
        parent=ROOT/'results/answer_localization_representation_pilot_v1'
        records=json.loads((parent/'PREPARED.json').read_text())['selected']
        for row in records:
            meta=json.loads((parent/'scores'/f"{row['uid']}.json").read_text())
            if meta['report']['methods'][reg.PARENTS['joint_parent']].get('valid'): break
        with np.load(parent/'inputs'/f"{row['uid']}.npz") as data: raw=data['raw']
        with np.load(parent/'scores'/f"{row['uid']}.npz") as old:
            arrays,methods,detail=reg.score_reliability(old,meta,raw,row['cell']+'/'+row['row_id'])
            self.assertLess(detail['replay_errors']['joint_graph010'],1e-10)
            self.assertEqual(set(methods),set(reg.ARMS))
            for arm,source in reg.PARENTS.items():
                if methods[arm]['valid']: np.testing.assert_array_equal(arrays[arm+'__risk'],old[source+'__step'])
            for core in reg.CORES:
                for family in reg.FAMILIES:
                    self.assertLess(detail['families'][core+'__'+family]['lambda_zero_error'],1e-10)


if __name__=='__main__': unittest.main()
