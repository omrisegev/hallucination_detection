"""ProcessBench metrics for PRM boosting (results/prm_boost_processbench_v1/local): tiny hand-computed fixtures."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'results/prm_boost_processbench_v1/local'
sys.path.insert(0, str(LOCAL)); sys.path.insert(0, str(ROOT))
import pb_metrics as PM  # noqa: E402
from spectral_utils.processbench import first_error_f1  # noqa: E402
spec = importlib.util.spec_from_file_location('pb_eval', LOCAL / 'eval_processbench.py')
EV = importlib.util.module_from_spec(spec); spec.loader.exec_module(EV)

# A: gsm8k label 1 (3 steps); B: gsm8k correct (2); C: math label 0 (2); D: math correct (3)
LABELS = np.array([1, -1, 0, -1]); LENS = np.array([3, 2, 2, 3]); SUBS = np.array(['gsm8k', 'gsm8k', 'math', 'math'])
RISK = np.array([.2, .9, .1,   .3, .4,   .8, .7,   .1, .6, .2])


class Official(unittest.TestCase):
    def test_hand_values(self):
        o = PM.official(LABELS, np.array([1, -1, 1, -1]), SUBS)
        self.assertEqual((o['gsm8k']['error_acc'], o['gsm8k']['correct_acc'], o['gsm8k']['f1']), (100., 100., 100.))
        self.assertEqual((o['math']['error_acc'], o['math']['correct_acc'], o['math']['f1']), (0., 100., 0.))
        self.assertEqual(o['average']['f1'], 50.)

    def test_matches_repo_first_error_f1(self):
        rng = np.random.default_rng(0)
        lab = rng.integers(-1, 5, 200); pred = np.where(rng.random(200) < .6, lab, rng.integers(-1, 5, 200))
        ref = first_error_f1({i: {'label': int(l), 'match': bool(p == l)} for i, (l, p) in enumerate(zip(lab, pred))})
        o = PM.official(lab, pred, np.array(['math'] * 200))['math']
        for k in ('error_acc', 'correct_acc', 'f1'):
            self.assertAlmostEqual(o[k], ref[k], places=10)

    def test_first_flag(self):
        self.assertEqual(PM.first_flag([0, 0, 1, 1]), 2); self.assertEqual(PM.first_flag([0, 0]), -1)


class StepMetrics(unittest.TestCase):
    def test_targets(self):
        y, inc = PM.step_arrays(LABELS, LENS)['prefix']
        self.assertEqual(y.tolist(), [0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        self.assertEqual(inc.tolist(), [1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
        self.assertEqual(PM.step_arrays(LABELS, LENS)['cumulative'][0].tolist(), [0, 1, 1, 0, 0, 1, 1, 0, 0, 0])
        self.assertEqual(PM.step_arrays(LABELS, LENS)['first_only'][1].all(), True)

    def test_pooled_auroc_prefix(self):
        # positives .9 and .8; negatives .2 .3 .4 .1 .6 .2 -> both positives above every negative
        self.assertEqual(PM.pooled_auroc(LABELS, LENS, RISK, 'prefix'), 1.0)
        # cumulative adds the post-error steps .1 (positive) and .7 (positive) -> .1 ties/loses
        y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0], bool)
        self.assertAlmostEqual(PM.pooled_auroc(LABELS, LENS, RISK, 'cumulative'), PM.auc(y, RISK))
        r = RISK.copy(); r[1] = np.nan          # empty step on the first error of A is excluded
        self.assertEqual(PM.pooled_auroc(LABELS, LENS, r, 'prefix'), 1.0)

    def test_within(self):
        w = PM.within_auc_per_answer(LABELS, LENS, RISK)
        self.assertEqual(w[0], 1.0); self.assertTrue(np.isnan(w[1:]).all())
        w = PM.within_auc_per_answer(np.array([2]), np.array([3]), np.array([.5, .5, .5]))
        self.assertEqual(w[0], .5)
        w = PM.within_auc_per_answer(np.array([2]), np.array([4]), np.array([.9, .1, .5, 9.]))
        self.assertEqual(w[0], .5)   # step 3 is after the first error: unlabelled, ignored

    def test_step_confusion(self):
        flags = RISK >= .6     # flagged: .9 .8 .7 .6
        c = PM.step_confusion(LABELS, LENS, flags)
        # prefix domain: pos .9 .8 (both flagged); neg .2 .3 .4 .1 .6 .2 (one flagged: .6)
        self.assertEqual((c['tp'], c['fn'], c['tn'], c['fp']), (2, 0, 5, 1))
        self.assertAlmostEqual(c['f1_error'], 4 / 5)


class Bootstrap(unittest.TestCase):
    def test_weighted_auc_identity(self):
        rng = np.random.default_rng(1); n = 50; lens = rng.integers(1, 6, n); lab = rng.integers(-1, 4, n)
        lab = np.minimum(lab, lens - 1); risk = rng.random(lens.sum()).round(1)   # many ties
        y, inc = PM.step_arrays(lab, lens)['prefix']; sa = np.repeat(np.arange(n), lens)
        got = EV.weighted_auc_draws(y[inc], risk[inc], sa[inc], np.ones((1, n)))[0]
        self.assertAlmostEqual(got, PM.pooled_auroc(lab, lens, risk, 'prefix'), places=12)
        counts = np.zeros((1, n)); idx = rng.integers(0, n, n); np.add.at(counts[0], idx, 1)
        rep = np.repeat(idx, 1)   # explicit duplication gives the same weighted AUC
        yy = np.concatenate([y[sa == i] for i in rep]); ii = np.concatenate([inc[sa == i] for i in rep])
        rr = np.concatenate([risk[sa == i] for i in rep])
        self.assertAlmostEqual(EV.weighted_auc_draws(y[inc], risk[inc], sa[inc], counts)[0], PM.auc(yy[ii], rr[ii]), places=12)

    def test_avg_f1_identity(self):
        rng = np.random.default_rng(2); subs = np.array(sum([[s] * 30 for s in PM.SUBSETS], []))
        lab = rng.integers(-1, 3, 120); pred = np.where(rng.random(120) < .5, lab, -1)
        W, cols = EV.stratified_draws(subs, rng, draws=3)
        self.assertAlmostEqual(EV.avg_f1_draws(cols[None, :], lab, pred, subs)[0],
                               PM.official(lab, pred, subs)['average']['f1'], places=10)


class Census(unittest.TestCase):
    def test_counts_overflow_without_truncation(self):
        spec = importlib.util.spec_from_file_location('pb_census', ROOT / 'results/prm_boost_processbench_v1/aircc/length_census.py')
        C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)

        class Fake:
            def build(self, tok, q, steps):
                if q == 'boom': raise ValueError('template')
                return {'input_ids': list(range(len(q) + sum(map(len, steps)))), 'n_step_text_mismatch': int('?' in q)}
        rows = [{'uid': 'gsm8k-0', 'benchmark': 'gsm8k', 'question': 'ab', 'steps': ['c']},
                {'uid': 'math-0', 'benchmark': 'math', 'question': 'abc?', 'steps': ['defgh']},
                {'uid': 'math-1', 'benchmark': 'math', 'question': 'boom', 'steps': ['x']}]
        r = C.census(Fake(), None, 5, rows)
        self.assertEqual(r['n_context_overflow'], 1); self.assertEqual(r['overflow_rows'], [{'uid': 'math-0', 'n_input_tokens': 9}])
        self.assertEqual(r['overflow_by_subset'], {'gsm8k': 0, 'math': 1})
        self.assertEqual(r['n_rows_step_text_mismatch'], 1); self.assertEqual(list(r['build_errors']), ['math-1'])


class EndToEnd(unittest.TestCase):
    def test_fake_scores(self):
        """Two fake PRMs, one of which failed on one answer: merge, parity, coverage, bootstrap run."""
        rng = np.random.default_rng(3); subs = sum([[s] * 6 for s in PM.SUBSETS], [])
        gold = []
        for i, s in enumerate(subs):
            n = int(rng.integers(2, 6)); lab = [-1, 1][i % 6] if i % 6 < 2 else int(rng.integers(-1, n))
            gold.append({'uid': f'{s}-{i}', 'subset': s, 'label': lab, 'n_steps': n})
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); (t / 'gold.json').write_text(json.dumps(gold))
            for m, drop in (('a', None), ('b', gold[0]['uid'])):
                G = sorted([g for g in gold if g['uid'] != drop], key=lambda g: g['uid'])
                lens = np.array([g['n_steps'] for g in G]); ne = np.ones(lens.sum(), bool); ne[0] = False
                prm = rng.normal(size=ne.sum()); nat = (rng.random(lens.sum()) > .3).astype(int)
                common = dict(uids=np.array([g['uid'] for g in G]), nonempty=ne, nonempty_lengths=lens, native_valid=nat)
                np.savez(t / f'SCORES_{m}.npz', **common, score__PRM=prm, score__HALF_N16=prm + rng.normal(size=prm.size))
                np.savez(t / f'SCORES_CHAIN_{m}.npz', **common, score__PRM=prm, score__CHAIN_N13=rng.normal(size=prm.size))
            sys.argv = ['x', str(t), str(t / 'out'), '--gold', str(t / 'gold.json')]
            EV.main()
            import csv
            rows = list(csv.DictReader(open(t / 'out/RESULTS_processbench.csv')))
            arms = {(r['model'], r['population'], r['arm']) for r in rows}
            self.assertIn(('a', 'common', 'CHAIN_N13_Q80'), arms); self.assertIn(('b', 'scored', 'PRM_native'), arms)
            self.assertNotIn(('b', 'common', 'PRM_native'), arms)   # b already is the common population
            b_rows = [r for r in rows if r['model'] == 'b']
            self.assertTrue(all(r['n_unscored'] == '1' for r in b_rows))
            self.assertTrue(list(csv.DictReader(open(t / 'out/BOOTSTRAP_processbench.csv'))))
            # parity failure is loud
            S = dict(np.load(t / 'SCORES_CHAIN_a.npz')); S['score__PRM'] = S['score__PRM'] + 1
            np.savez(t / 'SCORES_CHAIN_a.npz', **S)
            with self.assertRaises(SystemExit):
                EV.main()


if __name__ == '__main__':
    unittest.main()
