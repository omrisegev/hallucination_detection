"""Contract tests: quota preservation, tie keys, removal and orientation anchor."""
from pathlib import Path
import sys
import types
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('spectral_utils'); pkg.__path__=[str(ROOT/'spectral_utils')]
sys.modules['spectral_utils']=pkg
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/experiments')]
from spectral_utils.monday_fusion_core import LEVEL,removal_names,quota_marks,choose_anchor,between_weights
from spectral_utils.monday_fusion_position import half_groups
from spectral_utils.monday_fusion_lr import training_labels,nested_lr
from spectral_utils.monday_fusion_decision import rules
from spectral_utils.monday_fusion_secondary import paired_intervals,endpoint
from spectral_utils.monday_fusion_nulls import nulls

class Contracts(unittest.TestCase):
    def test_paired_bootstrap_preserves_identity_and_source_alignment(self):
        groups=np.array(['q1','q2','q3']); local=np.array(['q3','q1','q2'])
        sums=np.array([[.3,1],[1.8,2],[.8,1]])
        arrays={('a','within_answer_auc'):sums,('b','within_answer_auc'):sums.copy()}
        result=paired_intervals(groups,local,arrays,[('a','b','within_answer_auc')],draws=100)[0]
        self.assertEqual(result['delta'],0.); self.assertEqual(result['lo95'],0.); self.assertEqual(result['hi95'],0.)
        perm=np.array([1,2,0]); arrays2={k:v[perm] for k,v in arrays.items()}
        self.assertEqual(result,paired_intervals(groups,local[perm],arrays2,[('a','b','within_answer_auc')],draws=100)[0])
        altered=sums.copy(); altered[:,0]-=.2*altered[:,1]
        arrays[('b','within_answer_auc')]=altered
        actual=paired_intervals(groups,local,arrays,[('a','b','within_answer_auc')],draws=100)[0]
        for k in ['delta','lo95','hi95']: self.assertAlmostEqual(actual[k],.2)

    def test_pb_null_uses_valid_targets_without_full_step_labels(self):
        ns=np.array([2,2,3,3,4,5]); off=np.r_[0,np.cumsum(ns)]
        pop=types.SimpleNamespace(prm=np.zeros(6,bool),target=np.array([-1,1,0,2,3,-1]))
        bank={'off':off,'answers':np.arange(6)}; score=np.arange(off[-1],dtype=float)
        a=nulls(pop,bank,{'recipe':score},permutations=10)
        b=nulls(pop,bank,{'recipe':score},permutations=10)
        self.assertEqual(a,b); self.assertEqual(a['recipe']['same_length_singletons'],2)
        self.assertAlmostEqual(a['recipe']['observed_first_error_accuracy'],.75)

    def test_entropy_gate_keeps_frozen_flags_inside_open_answers(self):
        off=np.array([0,3,6]); score=np.array([-1.,0.,1.,-1.,0.,1.])
        flags,_,diagnostic=rules(score,{'off':off},score[:,None],np.array([0.,1.]),np.array(['a','b']))
        self.assertEqual(diagnostic['entropy_closed_count'],1)
        self.assertFalse(flags['ENTROPY_GATE_Q80'][:3].any())
        np.testing.assert_array_equal(flags['ENTROPY_GATE_Q80'][3:],flags['ANSWER_Z_Q80'][3:])

    def test_secondary_f1_uses_counts_and_requires_both_answer_classes(self):
        self.assertAlmostEqual(float(endpoint(np.array([3,5,4,5,1,2.]),'pb_official_f1')),2*.6*.8/(.6+.8))
        self.assertTrue(np.isnan(endpoint(np.array([3,5,0,0,0,2.]),'pb_official_f1')))
        self.assertTrue(np.isnan(endpoint(np.zeros(4),'prmscore')))

    def test_less_than_three_groups_is_failure_even_for_equal_control(self):
        z=np.ones((20,2)); votes=np.ones((20,2),np.int8)
        for method in ['EQ','DS','SML','MOM']:
            with self.assertRaisesRegex(ValueError,'minimum is 3'):
                between_weights(method,z,votes,z[:,0],0)

    def test_three_group_sml_guard_is_disclosed(self):
        rng=np.random.default_rng(2); z=rng.normal(size=(100,3)); votes=np.where(z>0,1,-1)
        weights,diagnostic=between_weights('SML',z,votes,z[:,0],0)
        self.assertTrue(diagnostic['small_m_guard'])
        self.assertNotIn('psi',diagnostic); self.assertNotIn('eta',diagnostic)
        self.assertAlmostEqual(float(np.abs(weights).sum()),1.)

    def test_nested_lr_never_uses_outer_fold_labels(self):
        rng=np.random.default_rng(23); x=rng.normal(size=(60,3)); off=np.arange(0,61,3)
        y=np.tile([0,1,0],20).astype(bool); folds=np.repeat(np.arange(5),4)
        pop=types.SimpleNamespace(prm=np.ones(20,bool),target=np.full(20,-1),off=off,labels=y.copy(),fold=folds)
        bank={'V':x,'off':off,'answers':np.arange(20)}
        before=nested_lr(pop,bank,np.arange(3))
        pop.labels[:12]=~pop.labels[:12]
        after=nested_lr(pop,bank,np.arange(3))
        np.testing.assert_array_equal(before['oof'][:12],after['oof'][:12])
        np.testing.assert_allclose(before['diagnostics'][0]['mean'],x[12:].mean(0),rtol=0,atol=1e-15)
        self.assertEqual(before['diagnostics'][0]['C'],after['diagnostics'][0]['C'])

    def test_r2_allocates_raw_global_count_and_keeps_earliest_ties(self):
        off=np.array([0,3,6]); bank={'off':off}
        score=np.array([1.,1.,0.,0.,2.,1.]); raw=np.array([[10.],[9.],[8.],[0.],[-1.],[-2.]])
        flags,_,diagnostic=rules(score,bank,raw,np.array([.2,.8]),np.array(['a','b']))
        # All cell-global high values belong to the first answer, so R2 gives it both flags.
        np.testing.assert_array_equal(flags['R2_COUNT'],[True,True,False,False,False,False])
        self.assertEqual(flags['R2_COUNT'].sum(),2)

    def test_crossfit_does_not_split_small_half_after_failure(self):
        with self.assertRaisesRegex(ValueError,'no alternate split'):
            half_groups(np.array([0,0,0,0,1,2]))
        a,b=half_groups(np.array([0,0,0,1,1,2]))
        np.testing.assert_array_equal(a,[0]); np.testing.assert_array_equal(b,[1,2])

    def test_pb_post_error_steps_are_unknown_not_negative(self):
        pop=types.SimpleNamespace(prm=np.array([False,False,True]),target=np.array([1,-1,-1]),
                                  off=np.array([0,4,7,10]),labels=np.array([True]*7+[False,True,False]))
        bank={'V':np.zeros((10,2)),'off':pop.off,'answers':np.array([0,1,2])}
        np.testing.assert_array_equal(training_labels(pop,bank),[0,1,-1,-1,0,0,0,0,1,0])

    def test_cell_quota_keeps_rounded_total(self):
        # Unequal lengths: sum ceil(.2*n)=3, whereas ceil(.2*sum n)=2.
        off=np.array([0,1,3,7]); v=np.array([[4,1],[3,2],[2,3],[1,4],[0,5],[-1,6],[-2,7]],float)
        key=np.random.default_rng(9).random(v.shape)
        old=quota_marks(v,off,key); new=quota_marks(v,off,key,True)
        np.testing.assert_array_equal((old>0).sum(0),[3,3])
        np.testing.assert_array_equal((new>0).sum(0),(old>0).sum(0))
        self.assertFalse(np.array_equal(old,new))

    def test_cell_ties_use_existing_key(self):
        off=np.array([0,2,4]); v=np.zeros((4,1)); key=np.array([[.9],[.1],[.8],[.2]])
        result=quota_marks(v,off,key,True)
        np.testing.assert_array_equal(result[:,0],[-1,1,-1,1])

    def test_no_removal_and_lexical_ties(self):
        v=np.tile(np.arange(20)[:,None],(1,5))
        self.assertEqual(removal_names(v,0),[])
        self.assertEqual(removal_names(v,2),sorted(LEVEL)[:2])

    def test_random_control_is_nested_and_from_level(self):
        v=np.zeros((10,5))
        two=removal_names(v,2,True); three=removal_names(v,3,True)
        self.assertEqual(two,three[:2]); self.assertTrue(set(three)<=set(LEVEL))

    def test_anchor_prefers_retained_q15_then_level(self):
        names=['hist_entropy_series','energy_level','q15_H1','q15_VE1']
        pi=np.array([.99,.8,.6,.8])
        self.assertEqual(choose_anchor(names,np.array([0,1,2,3]),pi),(2,'original'))
        self.assertEqual(choose_anchor(names,np.array([0,1,3]),pi),(1,'highest_DS_pi_surviving_LEVEL'))
        self.assertEqual(choose_anchor(names,np.array([0]),pi),(0,'highest_DS_pi_any_survivor'))
        with self.assertRaises(ValueError): choose_anchor(names,np.array([],int),pi)

if __name__=='__main__': unittest.main(verbosity=2)
