"""ProcessBench pre-registration (results/prm_boost_processbench_v1/PREREGISTRATION_HE.md, section 8): the NC first-error
rule, the question-clustered stratified bootstrap, E1 / E2 / verdict, the bank-parity comparer and the new control
arms of score_chain_pb.py. Hand-computed fixtures."""
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PB = ROOT / 'results/prm_boost_processbench_v1'
sys.path.insert(0, str(PB / 'local')); sys.path.insert(0, str(ROOT))
import pb_metrics as PM  # noqa: E402
spec = importlib.util.spec_from_file_location('pb_eval_prereg', PB / 'local/eval_processbench.py')
EV = importlib.util.module_from_spec(spec); spec.loader.exec_module(EV)
spec = importlib.util.spec_from_file_location('pb_bank_parity', PB / 'aircc/bank_parity.py')
BP = importlib.util.module_from_spec(spec); spec.loader.exec_module(BP)


def nc(lens, ne, nat, score_ne):
    f, info = PM.nc_flags(np.array(lens), np.array(ne, bool), np.array(nat, bool), np.array(score_ne, float))
    starts = np.r_[0, np.cumsum(lens)]
    return [PM.first_flag(f[a:b]) for a, b in zip(starts[:-1], starts[1:])], f.astype(int).tolist(), info


class NCRule(unittest.TestCase):
    def test_count_from_prm_order_from_fused(self):
        # native flags steps 1 and 3 (k = 2); fused top-2 are steps 0 (.9) and 2 (.5) -> flags {0, 2}, prediction 0
        p, f, info = nc([4], [1, 1, 1, 1], [0, 1, 0, 1], [.9, .1, .5, .3])
        self.assertEqual((p, f, info), ([0], [1, 0, 1, 0], {'n_gate': 1, 'n_k0': 0}))

    def test_gate_closed_predicts_no_error(self):
        p, f, info = nc([3], [1, 1, 1], [0, 0, 0], [9., 8., 7.])
        self.assertEqual((p, f, info), ([-1], [0, 0, 0], {'n_gate': 0, 'n_k0': 0}))

    def test_ties_go_to_the_earlier_step(self):
        # k = 2; .7 three times -> steps 1 and 2 (earlier first), prediction 1
        self.assertEqual(nc([4], [1] * 4, [1, 1, 0, 0], [.2, .7, .7, .7])[:2], ([1], [0, 1, 1, 0]))
        # k = 1; tie of .5 between steps 1 and 2 -> step 1
        self.assertEqual(nc([3], [1] * 3, [0, 0, 1], [.1, .5, .5])[:2], ([1], [0, 1, 0]))
        # all equal, k = 1 -> the first step
        self.assertEqual(nc([3], [1] * 3, [0, 1, 0], [0., 0., 0.])[0], [0])

    def test_k0_falls_back_to_the_native_first_error(self):
        # only the empty step 1 is flagged by the PRM: k = 0 but the gate fired -> native flags, prediction 1
        p, f, info = nc([3], [1, 0, 1], [0, 1, 0], [5., 6.])
        self.assertEqual((p, f, info), ([1], [0, 1, 0], {'n_gate': 1, 'n_k0': 1}))

    def test_empty_step_flag_before_a_counted_one(self):
        # native flags the empty step 0 and the non-empty step 2 (k = 1); fused top-1 is step 3 -> prediction 3,
        # not the empty step 0 (PRM_native would predict 0)
        p, f, info = nc([4], [0, 1, 1, 1], [1, 0, 1, 0], [.1, .2, .9])
        self.assertEqual((p, f, info), ([3], [0, 0, 0, 1], {'n_gate': 1, 'n_k0': 0}))

    def test_several_answers_in_the_non_empty_layout(self):
        lens = [3, 2, 4]
        ne = [1, 1, 1,   1, 0,   1, 1, 1, 1]
        nat = [0, 1, 0,  0, 1,   0, 0, 0, 0]
        s = [.3, .2, .9,  .4,   .8, .1, .2, .3]          # non-empty rows only
        p, f, info = nc(lens, ne, nat, s)
        self.assertEqual(p, [2, 1, -1])                    # answer 2: k = 1 -> step 2; answer 1: k = 0 -> step 1
        self.assertEqual(info, {'n_gate': 2, 'n_k0': 1})

    def test_prm_order_reproduces_native_without_empty_steps(self):
        rng = np.random.default_rng(0); lens = rng.integers(1, 8, 200); n = lens.sum()
        risk = rng.normal(size=n); nat = risk > 0
        f, _ = PM.nc_flags(lens, np.ones(n, bool), nat, risk)
        self.assertTrue(np.array_equal(f, nat))

    def test_non_finite_score_is_refused(self):
        with self.assertRaises(ValueError):
            PM.nc_flags([2], [1, 1], [1, 0], [np.nan, 1.])

    def test_decisions_add_nc_for_every_arm_and_keep_q80_native(self):
        lay = {'nonempty': np.array([1, 0, 1, 1], bool), 'nonempty_lengths': np.array([2, 2]),
               'native_valid': np.array([1, 0, 1, 1])}
        arms = {'PRM': np.array([-1., 2., -3.]), 'X': np.array([5., 1., 2.])}
        d = EV.decisions(lay, arms)
        self.assertEqual(set(d), {'PRM_Q80', 'X_Q80', 'PRM_native', 'PRM_native_adapter', 'PRM_NC', 'X_NC'})
        self.assertEqual(d['PRM_native'][1].tolist(), [False, True, True, False])   # empty step via native_valid
        # answer 0: native flags only its empty step 1 -> k = 0 -> step 1; answer 1: k = 1 -> X's top step (3: 2. > 1.)
        self.assertEqual(d['X_NC'][1].astype(int).tolist(), [0, 1, 0, 1])
        self.assertEqual(d['X_NC'][2], {'n_gate': 2, 'n_k0': 1})


class FakeRng:
    """rng.choice returning preset draws, one array per call (one call per subset)."""
    def __init__(self, picks): self.picks = list(picks)
    def choice(self, C, size): return np.asarray(self.picks.pop(0)).reshape(size)


class ClusteredBootstrap(unittest.TestCase):
    def test_hand_counts(self):
        # gsm8k: answers 0, 1 share question A, answer 2 is B; math: answer 3 is C. Draw 1: A, A; draw 2: A, B.
        subs = np.array(['gsm8k', 'gsm8k', 'gsm8k', 'math']); cl = np.array(['A', 'A', 'B', 'C'], dtype=object)
        c = PM.clustered_counts(subs, cl, FakeRng([[[0, 0], [0, 1]], [[0], [0]]]), 2)
        self.assertEqual(c.tolist(), [[2, 2, 0, 1], [1, 1, 1, 1]])

    def test_clusters_numbered_by_first_appearance(self):
        subs = np.array(['math'] * 3); cl = np.array(['z', 'a', 'z'], dtype=object)   # 'z' is cluster 0
        c = PM.clustered_counts(subs, cl, FakeRng([[[0, 0]]]), 1)
        self.assertEqual(c.tolist(), [[2, 0, 2]])

    def test_singletons_equal_the_answer_level_stratified_draws(self):
        subs = np.array(sum([[s] * k for s, k in zip(PM.SUBSETS, (5, 3, 4, 2))], []))
        perm = np.random.default_rng(9).permutation(len(subs)); subs = subs[perm]
        W, _ = EV.stratified_draws(subs, np.random.default_rng(7), draws=50)
        ref = np.array([np.bincount(w, minlength=len(subs)) for w in W])
        got = PM.clustered_counts(subs, np.array([f'u{i}' for i in range(len(subs))], dtype=object),
                                  np.random.default_rng(7), 50)
        self.assertTrue(np.array_equal(got, ref))

    def test_draw_mass_per_subset_is_the_cluster_count(self):
        rng = np.random.default_rng(1); subs = np.array(['gsm8k'] * 6 + ['omnimath'] * 4)
        cl = np.array(['a', 'a', 'b', 'c', 'c', 'c', 'd', 'e', 'e', 'f'], dtype=object)
        c = PM.clustered_counts(subs, cl, rng, 200)
        self.assertTrue((c[:, 0] == c[:, 1]).all() and (c[:, 3] == c[:, 5]).all() and (c[:, 7] == c[:, 8]).all())
        self.assertTrue((c[:, 0] + c[:, 2] + c[:, 3] == 3).all())     # 3 gsm8k clusters, one answer counted each
        self.assertTrue((c[:, 6] + c[:, 7] + c[:, 9] == 3).all())

    def test_cluster_across_subsets_is_an_error(self):
        with self.assertRaises(ValueError):
            PM.clustered_counts(np.array(['gsm8k', 'math']), np.array(['q', 'q'], dtype=object), np.random.default_rng(0), 2)

    def test_weighted_f1_by_hand(self):
        # gsm8k: 0 error label 1 pred 1 (hit), 1 error label 0 pred 1 (miss), 2 correct pred -1 (hit); clusters {0,1}, {2}
        lab = np.array([1, 0, -1]); pred = np.array([1, 1, -1]); subs = np.array(['gsm8k'] * 3)
        counts = np.array([[1, 1, 1], [2, 2, 0], [0, 0, 2]], float)
        f1 = EV.avg_f1_counts(counts, lab, pred, subs)
        np.testing.assert_allclose(f1, [2 * 50 * 100 / 150, 0., 0.])   # ea 50 / ca 100; ca 0; ea 0

    def test_weighted_metrics_equal_the_index_versions(self):
        rng = np.random.default_rng(2); subs = np.array(sum([[s] * 30 for s in PM.SUBSETS], []))
        lab = rng.integers(-1, 3, 120); pred = np.where(rng.random(120) < .5, lab, -1)
        W, _ = EV.stratified_draws(subs, rng, draws=20)
        counts = np.array([np.bincount(w, minlength=120) for w in W], float)
        np.testing.assert_allclose(EV.avg_f1_counts(counts, lab, pred, subs), EV.avg_f1_draws(W, lab, pred, subs))
        v = np.where(rng.random(120) < .3, np.nan, rng.random(120))
        np.testing.assert_allclose(EV.weighted_nanmean(counts, v), np.nanmean(v[W], axis=1))


def boot_row(model, metric, lo, contrast='CHAIN_N16_NC - PRM_native'):
    return {'model': model, 'population': 'scored', 'contrast': contrast, 'metric': metric,
            'delta': lo + .1, 'ci_low': lo, 'ci_high': lo + .2}


class Verdict(unittest.TestCase):
    def test_e1_e2_bounds(self):
        B = [boot_row('m', 'within_auc', .001), boot_row('m', 'avg_f1_failures_wrong', -.99)]
        self.assertEqual({k: v for k, v in PM.prm_success(B, 'm').items() if k in ('E1', 'E2', 'success')},
                         {'E1': True, 'E2': True, 'success': True})
        B = [boot_row('m', 'within_auc', .001), boot_row('m', 'avg_f1_failures_wrong', -1.0)]   # strict: -1.0 fails
        self.assertEqual(PM.prm_success(B, 'm')['E2'], False)
        B = [boot_row('m', 'within_auc', 0.), boot_row('m', 'avg_f1_failures_wrong', 3.)]       # strict: 0 fails E1
        self.assertFalse(PM.prm_success(B, 'm')['success'])
        self.assertEqual(PM.prm_success([], 'm'), {'E1': None, 'E2': None, 'success': False,
                                                   'within_auc': None, 'avg_f1_failures_wrong': None})
        # other contrasts and populations are ignored
        B = [boot_row('m', 'within_auc', .5, 'CHAIN_N16_Q80 - PRM_native'),
             dict(boot_row('m', 'avg_f1_failures_wrong', 5.), population='common')]
        self.assertEqual(PM.prm_success(B, 'm')['E1'], None)

    def test_three_of_five_with_a_strong_prm(self):
        ok, no = {'success': True}, {'success': False}
        v = PM.verdict({'qwen_prm': no, 'qwen_prm_72b': no, 'pure_prm': ok, 'rlhflow_prm': ok, 'math_shepherd': ok})
        self.assertEqual((v['verdict'], v['n_success']), ('FAIL', 3))           # 3 of 5, but no Qwen PRM
        v = PM.verdict({'qwen_prm': no, 'qwen_prm_72b': ok, 'pure_prm': ok, 'rlhflow_prm': ok, 'math_shepherd': no})
        self.assertEqual((v['verdict'], v['strong_success']), ('SUCCESS', ['qwen_prm_72b']))
        v = PM.verdict({'qwen_prm': ok, 'qwen_prm_72b': ok, 'pure_prm': no, 'rlhflow_prm': no, 'math_shepherd': no})
        self.assertEqual(v['verdict'], 'FAIL')                                  # 2 of 5
        v = PM.verdict({'qwen_prm': ok, 'qwen_prm_72b': ok, 'pure_prm': ok, 'rlhflow_prm': ok})
        self.assertEqual((v['verdict'], v['missing_prms']), ('INCOMPLETE', ['math_shepherd']))


class EndToEndPrereg(unittest.TestCase):
    def test_clusters_phantoms_and_verdict(self):
        rng = np.random.default_rng(5); gold, answers = [], []
        for s in PM.SUBSETS:
            for i in range(8):
                n = int(rng.integers(2, 6)); lab = -1 if i % 3 == 0 else int(rng.integers(0, n))
                u = f'{s}-{i}'; gold.append({'uid': u, 'subset': s, 'label': lab, 'n_steps': n})
                q = f'{s}{"  " if i % 2 else " "}question\n{i // 2} '   # pairs share a question up to whitespace
                answers.append({'uid': u, 'question': q, 'steps': ['x'] * n})
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); (t / 'gold.json').write_text(json.dumps(gold)); (t / 'answers.json').write_text(json.dumps(answers))
            for m, drop in (('qwen_prm', None), ('pure_prm', 'math-3')):
                G = sorted([g for g in gold if g['uid'] != drop], key=lambda g: g['uid'])
                lens = np.array([g['n_steps'] for g in G]); ne = np.ones(lens.sum(), bool)
                prm = rng.normal(size=ne.sum()); nat = (prm <= 0).astype(int)
                lay = dict(uids=np.array([g['uid'] for g in G]), nonempty=ne, nonempty_lengths=lens, native_valid=nat)
                np.savez(t / f'SCORES_{m}.npz', **lay, score__PRM=prm, score__HALF_N16=prm + rng.normal(size=prm.size))
                np.savez(t / f'SCORES_CHAIN_{m}.npz', **lay, score__PRM=prm, score__CHAIN_N16=prm + rng.normal(size=prm.size),
                         score__EQ_N16=rng.normal(size=prm.size))
            sys.argv = ['x', str(t), str(t / 'out'), '--gold', str(t / 'gold.json'), '--answers', str(t / 'answers.json')]
            EV.main()
            man = json.loads((t / 'out/EVAL_MANIFEST.json').read_text())
            self.assertEqual((man['clusters']['mode'], man['clusters']['n_clusters']), ('question', 16))
            with open(t / 'out/BOOTSTRAP_processbench.csv') as fh: B = list(csv.DictReader(fh))
            with open(t / 'out/RESULTS_processbench.csv') as fh: R = list(csv.DictReader(fh))
            fw = [r for r in B if r['model'] == 'pure_prm' and r['population'] == 'scored'
                  and r['contrast'] == 'CHAIN_N16_NC - PRM_native' and r['metric'] == 'avg_f1_failures_wrong']
            self.assertEqual(len(fw), 1); self.assertEqual(fw[0]['n_unscored_as_miss'], '1')
            pt = {r['arm']: r for r in R if r['model'] == 'pure_prm' and r['population'] == 'scored'}
            self.assertAlmostEqual(float(fw[0]['delta']), float(pt['CHAIN_N16_NC']['avg_f1_failures_wrong'])
                                   - float(pt['PRM_native']['avg_f1_failures_wrong']), places=9)
            self.assertFalse([r for r in B if r['population'] == 'common' and r['metric'] == 'avg_f1_failures_wrong'])
            self.assertTrue({'CHAIN_N16_NC - HALF_N16_NC', 'CHAIN_N16_NC - EQ_N16_NC', 'CHAIN_N16_Q80 - HALF_N16_Q80'}
                            <= {r['contrast'] for r in B})
            self.assertEqual(pt['PRM_NC']['avg_f1'], pt['PRM_native']['avg_f1'])   # no empty steps: NC = native
            self.assertEqual(pt['PRM_NC']['n_pred_diff_vs_native'], '0')
            V = json.loads((t / 'out/VERDICT_processbench.json').read_text())
            self.assertEqual(V['overall']['verdict'], 'INCOMPLETE')
            self.assertEqual(set(V['per_prm']), {'qwen_prm', 'pure_prm'})
            self.assertEqual(V['per_prm']['pure_prm']['avg_f1_failures_wrong']['ci_low'], float(fw[0]['ci_low']))
            # every gold uid needs a question once the answers file matches any of them
            (t / 'answers.json').write_text(json.dumps(answers[1:]))
            with self.assertRaises(SystemExit):
                EV.main()


def write_bank(d, values, digit_ids=tuple(range(15, 25))):
    d.mkdir(parents=True)
    np.savez_compressed(d / 'FEATURES.npz', names=np.array(['a', 'b']), values=values, digit_active=np.ones((3, 3), bool),
                        offsets=np.array([0, 3]), uids=np.array(['u']), nonempty=np.ones(3, bool), nonempty_lengths=np.array([3]))
    np.savez_compressed(d / 'REWARDS.npz', uids=np.array(['u']), rewards=np.array([.1, .2, .3]), native_labels=np.array([0, 0, 0]),
                        nonempty_lengths=np.array([3]))
    (d / 'MANIFEST.json').write_text(json.dumps({'raw_sha256': 'r', 'n_records': 1, 'n_failed_rows': 0, 'failed_rows': {},
                                                 'pre_empty_steps': {}, 'steps_total': 3, 'steps_nonempty': 3, 'seconds': 1.,
                                                 'digit_ids': {'digit_ids': list(digit_ids),
                                                               'matches_qwen_15_24': tuple(digit_ids) == tuple(range(15, 25))}}))


class BankParity(unittest.TestCase):
    def test_exact_or_fail(self):
        v = np.arange(6.).reshape(3, 2)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); write_bank(t / 'old', v); write_bank(t / 'new', v); write_bank(t / 'off', v + np.r_[0, 0, 0, 0, 0, 1e-12].reshape(3, 2))
            write_bank(t / 'mistral', v, digit_ids=range(100, 110))
            self.assertTrue(BP.compare(t / 'old', t / 'new')['ok'])
            r = BP.compare(t / 'old', t / 'off')
            self.assertFalse(r['ok']); self.assertEqual(r['diff'], ['FEATURES.npz:values float64(3, 2) vs float64(3, 2)'])
            self.assertGreater(r['max_abs'], 0)
            self.assertFalse(BP.compare(t / 'old', t / 'mistral')['ok'])
            self.assertEqual(BP.main([str(t / 'S.json'), 'x', str(t / 'old'), str(t / 'new')]), 0)
            self.assertEqual(BP.main([str(t / 'S.json'), 'x', str(t / 'old'), str(t / 'new'), 'y', str(t / 'old'), str(t / 'off')]), 1)
            S = json.loads((t / 'S.json').read_text())
            self.assertEqual((S['ok'], S['pairs']['x']['ok'], S['pairs']['y']['ok']), (False, True, False))


@unittest.skipUnless((ROOT / 'results/prm_boosting_v1/fuse.py').exists(), 'frozen PRM-boosting modules not in this checkout')
class ScoreChainControls(unittest.TestCase):
    def test_new_arms(self):
        import pickle
        rng = np.random.default_rng(0)
        N16 = json.loads((ROOT / 'results/monday_nested_banks_v2/BANKS.json').read_text())['N16']
        names = N16 + [f'x{i}' for i in range(56 - len(N16))]
        lens = np.array([4, 3, 5] * 10); tot = lens.sum(); starts = np.r_[0, np.cumsum(lens)]
        ne = np.ones(tot, bool); ne[[1, 9]] = False
        nel = np.array([ne[a:b].sum() for a, b in zip(starts[:-1], starts[1:])]); off = np.r_[0, np.cumsum(nel)]
        vals = rng.normal(size=(ne.sum(), 1)) @ rng.normal(size=(1, 56)) + rng.normal(size=(ne.sum(), 56))
        uids = np.array([f'u{i:02d}' for i in range(len(lens))]); rew = rng.uniform(.01, .99, tot)
        raw, toks = {}, []
        for j, u in enumerate(uids):
            sp, at = [], 0
            for k in range(lens[j]):
                L = int(rng.integers(1, 30)) if ne[starts[j] + k] else 0; sp.append((at, at + L)); at += L
                if L: toks.append(L)
            raw[j] = {'idx': u, 'step_token_spans': sp}
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); (t / 'bank').mkdir()
            np.savez(t / 'bank/FEATURES.npz', names=np.array(names), values=vals, digit_active=np.ones((ne.sum(), 3), bool),
                     offsets=off, uids=uids, nonempty=ne, nonempty_lengths=lens)
            np.savez(t / 'bank/REWARDS.npz', uids=uids, rewards=rew, native_labels=(rew > .5).astype(int), nonempty_lengths=lens)
            pickle.dump(raw, open(t / 'raw.pkl', 'wb'))
            env = dict(os.environ, PRM_MIRROR=str(ROOT))
            subprocess.run([sys.executable, '-B', str(PB / 'aircc/score_chain_pb.py'), str(t / 'bank'), str(t / 'S.npz'),
                            str(t / 'raw.pkl')], check=True, env=env, cwd=ROOT, capture_output=True)
            S = dict(np.load(t / 'S.npz')); D = json.loads((t / 'S.json').read_text())
        import types   # as score_chain_pb.py: the repo copy executed unchanged with the mirror __file__
        src = ROOT / 'results/prm_boosting_generalization_v1/run_v1/evaluate_amended.py'
        E = types.ModuleType('ev_am'); E.__file__ = str(ROOT / 'results/prm_boosting_generalization_v1/evaluate_amended.py')
        exec(compile(src.read_text(), str(src), 'exec'), E.__dict__)
        zp = E.az(E.risk(np.clip(rew[ne], 1e-6, 1 - 1e-6)), off)
        np.testing.assert_allclose(S['score__HALF_N16'], .5 * zp + .5 * S['score__MEAN_N16'])
        n16 = len(N16); n13 = 13
        np.testing.assert_allclose(S['score__EQ_N16'], (zp + n16 * S['score__MEAN_N16']) / (n16 + 1))
        np.testing.assert_allclose(S['score__EQ_N13'], (zp + n13 * S['score__MEAN_N13']) / (n13 + 1))
        # answer 0 has non-empty steps 0, 2, 3 -> positions 1/6, 1/2, 5/6 -> answer-z -sqrt(1.5), 0, sqrt(1.5)
        np.testing.assert_allclose(S['score__POS_ONLY'][:3], [-np.sqrt(1.5), 0, np.sqrt(1.5)], atol=1e-6)
        np.testing.assert_allclose(S['score__LEN_ONLY'], E.az(np.log1p(toks), off))
        self.assertEqual(D['lsml'], 'lsml_continuous_guarded')
        self.assertTrue({'score__MEAN_N13', 'score__HALF_N13', 'score__CHAIN_N16', 'score__LSML_N16_PRMG'} <= set(S))


if __name__ == '__main__':
    unittest.main()
