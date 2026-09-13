"""Mechanism tests for the supervised diagnostic correction (2026-09-13); not benchmark performance tests.

Covers: the smoke PASS rule (an all-failed run cannot PASS, an unexpected exception cannot
PASS, an expected-limitation-only run is INCONCLUSIVE, an iteration-limit fit is never
reported converged); extraction verification (a changed boundary with the same number of
steps is detected, token/logprob misalignment is detected); manifest refusal naming the
changed raw-source hash; and held-fold-blind calibration (flipping the held fold's labels
leaves every calibration model and its threshold bit-identical, while flipping a
training fold's labels does change the model).
"""
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from spectral_utils import varentropy_expansion_supervised as sup
from spectral_utils.varentropy_expansion_supervised import (
    BANKS, calibration_plan, calibration_quantile, classify_fit, fit, fit_health, held_fold_blind_check, manifest_differences,
    score_steps, smoke_status, spans_sha256, training_matrix, verify_answer)


def fit_record(bank, status='FIT', converged=True, message='CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH', reason=None, cell='c', fold=0):
    r = dict(status=status, bank=bank, cell=cell, fold=fold)
    if status == 'FIT': r.update(converged=converged, message=message)
    if status == 'STALLED': r.update(converged=False, stalled=True, message=message)
    if status == 'FAILED': r.update(reason=reason)
    return r


class SmokeRuleTests(unittest.TestCase):
    def test_all_failed_cannot_pass(self):
        fits = [fit_record(b, 'FAILED', reason='ValueError: empty training or held fold (declared fold failure)', fold=f) for b in BANKS for f in range(5)]
        status, health = smoke_status(fits, BANKS)
        self.assertEqual(status, 'INCONCLUSIVE'); self.assertNotEqual(status, 'PASS'); self.assertEqual(health['n_expected_limitations'], 10)
        fits = [fit_record(b, 'FAILED', reason='LinAlgError: singular matrix', fold=f) for b in BANKS for f in range(5)]
        self.assertEqual(smoke_status(fits, BANKS)[0], 'FAIL')
        self.assertEqual(smoke_status([], BANKS)[0], 'FAIL')

    def test_unexpected_exception_cannot_pass(self):
        fits = [fit_record('B2_sel'), fit_record('B2d_sel'), fit_record('B2d_sel', 'FAILED', reason='KeyError: token_spilled_energies', fold=1)]
        status, health = smoke_status(fits, BANKS)
        self.assertEqual(status, 'FAIL'); self.assertEqual(health['n_unexpected_failures'], 1)
        fits = [fit_record('B2_sel'), fit_record('B2d_sel'), fit_record('B2d_sel', 'STALLED', fold=1)]
        self.assertEqual(smoke_status(fits, BANKS)[0], 'FAIL')

    def test_expected_limitation_only_is_inconclusive(self):
        fits = [fit_record('B2_sel'), fit_record('B2d_sel', 'FAILED', reason='ValueError: training set has no step of class 1 (declared fold failure)')]
        status, health = smoke_status(fits, BANKS)
        self.assertEqual(status, 'INCONCLUSIVE'); self.assertEqual(health['n_converged'], 1); self.assertEqual(health['n_expected_limitations'], 1)

    def test_pass_and_iteration_limit(self):
        limit = fit_record('B2_sel', converged=False, message='STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT')
        self.assertEqual(classify_fit(limit), 'FIT_ITERATION_LIMIT')
        self.assertEqual(classify_fit(fit_record('B2_sel', converged=False, message='ABNORMAL_TERMINATION_IN_LNSRCH')), 'UNEXPECTED_FAILURE')
        status, health = smoke_status([limit, fit_record('B2d_sel'), fit_record('B2d_sel', 'FAILED', reason='ValueError: empty training or held fold (declared fold failure)', fold=1)], BANKS)
        self.assertEqual(status, 'PASS'); self.assertEqual(health['n_iteration_limit'], 1); self.assertEqual(health['n_converged'], 1)
        self.assertEqual(fit_health([limit])['n_converged'], 0)


def synthetic_row(T=60, steps=((0, 20), (20, 41), (41, 60)), seed=1):
    rng = np.random.default_rng(seed)
    lp = np.log(rng.dirichlet(np.ones(50), size=T)).astype(np.float32)
    lp = -np.sort(-lp, axis=1)
    return dict(id='q1', idx=7, token_entropies=rng.uniform(.1, 2, size=T).tolist(), token_spilled_energies=rng.uniform(0, 3, size=T).tolist(),
                top_k_logprobs=dict(ids=np.zeros((T, 50), np.int32), logprobs=lp), step_token_spans=[list(s) for s in steps])


class VerifyAnswerTests(unittest.TestCase):
    def setUp(self):
        self.row = synthetic_row(); self.record = dict(uid='pb_gsm8k_q4__x', row_id='gsm8k::q1', tokens=60, steps=3, group_id='g')
        self.starts = np.array([0, 20, 41]); self.ends = np.array([20, 41, 60])
        self.detector = float(np.mean(self.row['token_entropies']))

    def ok(self, **kw):
        args = dict(kind='pb', dataset='gsm8k', bench_starts=self.starts, bench_ends=self.ends, detector=self.detector); args.update(kw)
        return verify_answer(self.record, self.row, **args)

    def test_passes_and_records_provenance(self):
        p = self.ok()
        self.assertEqual(p['n_tokens'], 60); self.assertEqual(p['n_steps'], 3); self.assertEqual(p['spans_sha256'], spans_sha256(self.row['step_token_spans']))
        self.assertEqual(p['source_key'], 'gsm8k::q1'); self.assertTrue(p['topk_finite'])
        prm = dict(self.record, uid='prmbench_qwen3_8b__x', row_id='7')
        self.assertEqual(verify_answer(prm, self.row, kind='prm', bench_starts=self.starts, bench_ends=self.ends)['source_key'], '7')

    def test_changed_boundary_same_step_count_detected(self):
        with self.assertRaisesRegex(ValueError, 'boundaries differ'): self.ok(bench_starts=np.array([0, 21, 41]), bench_ends=np.array([21, 41, 60]))
        with self.assertRaisesRegex(ValueError, 'boundaries differ'): self.ok(bench_ends=np.array([20, 41, 59]))
        with self.assertRaisesRegex(ValueError, 'boundary count'): self.ok(bench_starts=np.array([0, 20]), bench_ends=np.array([20, 60]))

    def test_alignment_and_mapping_failures_detected(self):
        with self.assertRaisesRegex(ValueError, 'row_id'): verify_answer(dict(self.record, row_id='gsm8k::q2'), self.row, kind='pb', dataset='gsm8k', bench_starts=self.starts, bench_ends=self.ends, detector=self.detector)
        row = dict(self.row, token_spilled_energies=self.row['token_spilled_energies'][:-1])
        with self.assertRaisesRegex(ValueError, 'token_spilled_energies'): verify_answer(self.record, row, kind='pb', dataset='gsm8k', bench_starts=self.starts, bench_ends=self.ends, detector=self.detector)
        row = dict(self.row, top_k_logprobs=dict(logprobs=self.row['top_k_logprobs']['logprobs'][:-1]))
        with self.assertRaisesRegex(ValueError, 'top-K rows'): verify_answer(self.record, row, kind='pb', dataset='gsm8k', bench_starts=self.starts, bench_ends=self.ends, detector=self.detector)
        chosen = list(self.row['token_spilled_energies']); chosen[3] = float('inf')
        with self.assertRaisesRegex(ValueError, 'non-finite selected surprisal'): verify_answer(self.record, dict(self.row, token_spilled_energies=chosen), kind='pb', dataset='gsm8k', bench_starts=self.starts, bench_ends=self.ends, detector=self.detector)
        with self.assertRaisesRegex(ValueError, 'detector'): self.ok(detector=self.detector + 1e-6)
        with self.assertRaisesRegex(ValueError, 'token count'): verify_answer(dict(self.record, tokens=61), self.row, kind='pb', dataset='gsm8k', bench_starts=self.starts, bench_ends=self.ends, detector=self.detector)


class ManifestTests(unittest.TestCase):
    def test_changed_source_hash_detected_and_named(self):
        a = dict(schema='s', hashes={'x/processbench_gsm8k.pkl': 'aaa', 'y/code.py': 'bbb'}, raw_source_hashes={'x/processbench_gsm8k.pkl': 'aaa'})
        b = json.loads(json.dumps(a)); b['hashes']['x/processbench_gsm8k.pkl'] = 'ccc'; b['raw_source_hashes']['x/processbench_gsm8k.pkl'] = 'ccc'
        self.assertEqual(manifest_differences(a, a), [])
        self.assertEqual(manifest_differences(a, b), ['hashes/x/processbench_gsm8k.pkl', 'raw_source_hashes/x/processbench_gsm8k.pkl'])
        from scripts import run_varentropy_expansion_supervised_v1 as driver
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'CHECKPOINT.sqlite'
            driver.connect(path, a).close()
            driver.connect(path, a).close()                      # same manifest: accepted
            with self.assertRaisesRegex(ValueError, r'hashes/x/processbench_gsm8k\.pkl'): driver.connect(path, b)
            con = sqlite3.connect(path)                          # the frozen checkpoint is untouched
            stored = json.loads(con.execute('SELECT payload FROM manifest WHERE id=1').fetchone()[0]); con.close()
            self.assertEqual(stored, a)


def synthetic_cell(n_answers=30, n_folds=5, seed=3):
    """Tiny PRMB-like cell: answers dict {idx: (z float32, spans)}, records, outer folds, flat labels/offsets."""
    rng = np.random.default_rng(seed); answers = {}; records = []; labels = []; offsets = [0]
    for i in range(n_answers):
        n_steps = int(rng.integers(3, 6)); lengths = rng.integers(4, 9, size=n_steps); ends = np.cumsum(lengths); starts = ends - lengths
        T = int(ends[-1]); z = rng.normal(size=(T, 6)).astype(np.float32); spans = np.stack([starts, ends], axis=1)
        y = rng.integers(0, 2, size=n_steps); y[rng.random(n_steps) < .1] = -1
        z[:, 0] += np.repeat(y == 1, lengths) * 1.5          # a learnable signal
        answers[i] = (z, spans); records.append(dict(uid=f'u{i}', group_id=f'g{i % 12}', cell='prm')); labels.append(y); offsets.append(offsets[-1] + n_steps)
    labels = np.concatenate(labels); offsets = np.array(offsets); outer = np.array([(i % 12) % n_folds for i in range(n_answers)])
    return answers, records, labels, offsets, outer


def calibrate(answers, records, labels, offsets, outer, held, cols=(0, 1, 2, 3, 4, 5)):
    prm = list(range(len(records))); flat = np.full(int(offsets[-1]), np.nan); thetas = {}
    target = np.full(len(records), -1)
    for h, train, score in calibration_plan(prm, outer, held):
        held_fold_blind_check(records, outer, held, train, score)
        x, spans, y = training_matrix(answers, train, list(cols), kind='prm', target=target, labels=labels, offsets=offsets)
        theta, info = fit(x, spans, y); thetas[h] = theta
        for i in score: flat[offsets[i]:offsets[i + 1]] = score_steps(answers[i][0][:, list(cols)], answers[i][1], theta)
    return flat, thetas


class HeldFoldBlindCalibrationTests(unittest.TestCase):
    def test_plan_and_blind_check(self):
        answers, records, labels, offsets, outer = synthetic_cell()
        plan = calibration_plan(list(range(len(records))), outer, 2)
        self.assertEqual([h for h, _, _ in plan], [0, 1, 3, 4])
        for h, train, score in plan:
            self.assertFalse(any(outer[i] in (2, h) for i in train)); self.assertTrue(all(outer[i] == h for i in score))
        with self.assertRaises(AssertionError): held_fold_blind_check(records, outer, 2, [i for i in range(len(records)) if outer[i] != 1], [])

    def test_held_fold_labels_never_matter_but_training_labels_do(self):
        answers, records, labels, offsets, outer = synthetic_cell(); held = 0
        flat, thetas = calibrate(answers, records, labels, offsets, outer, held)
        train = [i for i in range(len(records)) if outer[i] != held]
        self.assertTrue(all(np.isfinite(flat[offsets[i]:offsets[i + 1]]).all() for i in train))
        self.assertTrue(all(np.isnan(flat[offsets[i]:offsets[i + 1]]).all() for i in range(len(records)) if outer[i] == held))
        q = calibration_quantile(flat, train, offsets, .8)
        flipped = labels.copy(); n = 0
        for i in range(len(records)):
            if outer[i] == held:
                sl = slice(offsets[i], offsets[i + 1]); known = flipped[sl] >= 0; block = flipped[sl]; block[known] = 1 - block[known]; flipped[sl] = block; n += int(known.sum())
        self.assertGreater(n, 0); self.assertFalse(np.array_equal(flipped, labels))
        flat2, thetas2 = calibrate(answers, records, flipped, offsets, outer, held)
        for h in thetas: self.assertEqual(thetas[h].tobytes(), thetas2[h].tobytes())
        self.assertEqual(q, calibration_quantile(flat2, train, offsets, .8))
        np.testing.assert_array_equal(np.nan_to_num(flat, nan=-9), np.nan_to_num(flat2, nan=-9))
        # the same perturbation on a TRAINING fold changes at least one calibration model (the test has teeth)
        flipped_train = labels.copy()
        for i in range(len(records)):
            if outer[i] == 1:
                sl = slice(offsets[i], offsets[i + 1]); known = flipped_train[sl] >= 0; block = flipped_train[sl]; block[known] = 1 - block[known]; flipped_train[sl] = block
        _, thetas3 = calibrate(answers, records, flipped_train, offsets, outer, held)
        self.assertTrue(any(thetas[h].tobytes() != thetas3[h].tobytes() for h in thetas if h != 1))

    def test_quantile_mirrors_evaluate_arrays_convention(self):
        offsets = np.array([0, 3, 5, 9]); flat = np.array([1., 2, 3, np.nan, 5, 6, 7, 8, 9])
        expected = float(np.quantile(np.concatenate([flat[0:3], flat[5:9]]), .8))
        self.assertEqual(calibration_quantile(flat, [0, 1, 2], offsets, .8), expected)
        with self.assertRaises(ValueError): calibration_quantile(flat, [1], offsets, .8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
