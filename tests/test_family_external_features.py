import unittest
import numpy as np

from spectral_utils.family_external_features import step_features, ct7_streams
from spectral_utils.family_tail_transfer import score_locked, load_lock


def fixture(n=35):
    rng=np.random.default_rng(451)
    logits=rng.normal(size=(n,80))*np.linspace(.2,3.,n)[:,None]
    exp=np.exp(logits-logits.max(1,keepdims=True));p=exp/exp.sum(1,keepdims=True)
    ids=np.argsort(-p,axis=1)[:,:50];lp=np.log(np.take_along_axis(p,ids,axis=1))
    chosen=ids[np.arange(n),np.arange(n)%20]
    q=np.exp(lp[:,:15]);q/=q.sum(1,keepdims=True)
    return {'gen_token_ids':chosen,'top_k_logprobs':{'ids':ids,'logprobs':lp},
            'token_entropies':-(q*np.log(q)).sum(1),
            'token_spilled_energies':-np.log(p[np.arange(n),chosen]),
            'token_logsumexp':logits.max(1)+np.log(exp.sum(1)),
            'step_token_spans':np.array([[0,n//2],[n//2,n]]) if n>1 else np.array([[0,1]])}


class FamilyExternalFeaturesTest(unittest.TestCase):
    def test_labels_cannot_change_features_or_predictions(self):
        row=fixture();a,names=step_features(row)
        row.update(correct=[False,False],labels=[1,0],error_steps=[1,2],category='arbitrary')
        b,other=step_features(row)
        np.testing.assert_array_equal(a,b);self.assertEqual(names,other)
        methods=tuple(k for k in load_lock()['rows'] if k!='ct7')
        sa=score_locked(a,names,np.array([0,2]),methods=methods)
        sb=score_locked(b,other,np.array([0,2]),methods=methods)
        self.assertEqual(len(sa),9)
        for name in sa:
            np.testing.assert_array_equal(sa[name]['scores'],sb[name]['scores'])
            np.testing.assert_array_equal(sa[name]['pred_valid'],sb[name]['pred_valid'])

    def test_empty_steps_require_explicit_caller_mask(self):
        row=fixture();row['step_token_spans']=np.array([[0,0],[0,17],[17,35]])
        with self.assertRaisesRegex(ValueError,'remove empty'):
            step_features(row)
        x,names=step_features(row,row['step_token_spans'][1:])
        self.assertEqual(x.shape,(2,48));self.assertTrue(np.isfinite(x).all())

    def test_short_answers_and_missing_innovation_are_finite(self):
        for n in (1,2,15,16,17):
            row=fixture(n);x,names=step_features(row)
            self.assertTrue(np.isfinite(x).all());self.assertEqual(x.shape[1],48)
            if n==1:
                out=score_locked(x,names,np.array([0,1]))
                for arm in out.values():
                    np.testing.assert_array_equal(arm['scores'],[0.])

    def test_chosen_readout_keeps_first_step(self):
        row=fixture();x,names=step_features(row)
        tokens,valid=ct7_streams(row);chosen=tokens[:17,6]
        expected=np.sort(chosen)[-10:].mean()
        self.assertAlmostEqual(x[0,names.index('ct7_chosen_std_excess')],expected,places=12)
        self.assertFalse(valid[0,4]);self.assertTrue(valid[:,6].all())


if __name__=='__main__':unittest.main()
