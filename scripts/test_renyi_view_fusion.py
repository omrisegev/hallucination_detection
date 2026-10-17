"""Mechanism checks for Renyi-view fusion v1; not benchmark performance tests.

Synthetic checks always run.  The real-row checks load ONLY the smallest
benchmark pickle (ProcessBench gsm8k, Qwen3-4B, ~50 MB) and are skipped with
an explicit message when the free-memory guard (3,000,000 KB) is not met or
the source root is unavailable; run with --source-root to bind the source.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.renyi_view_fusion import (ALPHAS, COLUMN_NAMES, K, METHODS, NOT_APPLICABLE, VIEW_NAMES,
    anchor_stream, bank_matrix, fit_all, head_distribution, renyi_entropy, renyi_views, view_diagnostics)
from spectral_utils.moment_rbm_fusion import representation
from spectral_utils.token_feature_views import _logprob_token_series
from spectral_utils.direct_probability_fusion import zscore_columns
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS
from spectral_utils.upcr import upcr_fit

SOURCE_ROOT = None
REAL_ROWS = None


def synthetic(n=160, seed=339):
    rng = np.random.default_rng(seed)
    logits = np.sort(rng.normal(size=(n, 50)) * rng.uniform(.3, 3, size=(n, 1)), axis=1)[:, ::-1]
    p = np.exp(logits - logits.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
    lp = np.log(p)
    chosen = rng.uniform(0, 4, size=n)
    return lp, chosen


def free_memory_kb():
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command',
                              '(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out)
    except Exception:
        return -1


def load_real_rows(limit=40):
    global REAL_ROWS
    if REAL_ROWS is not None:
        return REAL_ROWS
    if SOURCE_ROOT is None:
        raise unittest.SkipTest('no --source-root given; real-row checks skipped')
    free = free_memory_kb()
    if free < 3_000_000:
        raise unittest.SkipTest(f'free memory {free} KB below the 3,000,000 KB guard; real-row checks skipped')
    from scripts import run_direct_probability_fusion_v2 as old
    old.configure_source_root(Path(SOURCE_ROOT))
    rows = old._source_row_map(old.load_pickle(old.PB_DIRS['q4'] / 'processbench_gsm8k.pkl'), kind='pb', dataset='gsm8k')
    out = []
    for key in sorted(rows)[:limit]:
        row = rows[key]
        lp = np.asarray(old._topk_payload(row)['logprobs'], float)
        out.append((key, lp, np.asarray(row['token_spilled_energies'], float)))
    REAL_ROWS = out
    return out


class ViewDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.lp, self.chosen = synthetic()

    def test_h1_equals_frozen_entropy15_synthetic(self):
        V, names = renyi_views(self.lp)
        h1 = representation(self.lp, self.chosen)[:, 0]
        np.testing.assert_allclose(V[:, names.index('H1')], h1, atol=1e-12, rtol=0)

    def test_h1_equals_frozen_entropy15_real_rows(self):
        for key, lp, chosen in load_real_rows():
            V, names = renyi_views(lp)
            h1 = representation(lp, chosen)[:, 0]
            np.testing.assert_allclose(V[:, names.index('H1')], h1, atol=1e-12, rtol=0, err_msg=key)

    def test_hinf_equals_s1(self):
        for lp in [self.lp] + [r[1] for r in (REAL_ROWS or [])]:
            q, s = head_distribution(lp)
            V, names = renyi_views(lp)
            np.testing.assert_array_equal(V[:, names.index('Hinf')], s.min(axis=1))
            np.testing.assert_allclose(V[:, names.index('Hinf')], s[:, 0], atol=1e-6, rtol=0)

    def test_hartley_constant(self):
        q, _ = head_distribution(self.lp)
        h0 = renyi_entropy(q, 0.0)
        self.assertEqual(float(h0.std()), 0.0)
        np.testing.assert_allclose(h0, np.log(K), rtol=0, atol=1e-15)
        self.assertNotIn(0.0, ALPHAS)

    def test_alpha_to_one_limit(self):
        q, _ = head_distribution(self.lp)
        h1 = renyi_entropy(q, 1.0)
        below, above = renyi_entropy(q, 0.999), renyi_entropy(q, 1.001)
        self.assertLess(np.max(np.abs(below - h1)), 5e-3)
        self.assertLess(np.max(np.abs(above - h1)), 5e-3)
        self.assertTrue(np.all(below >= h1 - 1e-9) and np.all(above <= h1 + 1e-9))
        # Symmetric bracket: H_1 lies between the two one-sided values.
        self.assertLess(np.max(np.abs((below + above) / 2 - h1)), 1e-5)

    def test_monotone_nonincreasing_in_alpha(self):
        for lp in [self.lp] + [r[1] for r in (REAL_ROWS or [])]:
            V, _ = renyi_views(lp)
            self.assertTrue(np.all(np.diff(V, axis=1) <= 1e-9), 'H_alpha must be non-increasing in alpha')
            q, _ = head_distribution(lp)
            dense = np.column_stack([renyi_entropy(q, a) for a in (0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4, 8, 16, np.inf)])
            self.assertTrue(np.all(np.diff(dense, axis=1) <= 1e-9))

    def test_uniform_and_peaked(self):
        q = np.full((3, K), 1.0 / K)
        for a in ALPHAS:
            np.testing.assert_allclose(renyi_entropy(q, a), np.log(K), atol=1e-9)
        peaked = np.zeros((2, K)); peaked[:, 0] = 1.0
        for a in ALPHAS:
            np.testing.assert_allclose(renyi_entropy(peaked, a), 0.0, atol=1e-9)

    def test_k50_renyi2_replay_and_k15_difference(self):
        frozen = _logprob_token_series({'logprobs': self.lp}, len(self.lp))['topk_renyi2_series']
        d = view_diagnostics(self.lp, self.chosen)
        p50 = np.exp(self.lp); p50 /= (p50.sum(axis=1, keepdims=True) + 1e-12)
        np.testing.assert_array_equal(-np.log((p50 ** 2).sum(axis=1) + 1e-12), frozen)
        V, names = renyi_views(self.lp)
        self.assertFalse(np.allclose(V[:, names.index('H2')], frozen), 'K=15 H2 must not be treated as the K=50 series')
        self.assertIsNotNone(d['h2_k15_vs_renyi2_k50_pearson'])
        self.assertTrue(d['hartley_constant'])
        self.assertEqual(len(d['pearson']), len(d['columns']))


class FitTests(unittest.TestCase):
    def setUp(self):
        self.lp, self.chosen = synthetic()

    def test_all_fits_reconstruct(self):
        fits, failures, seconds = fit_all(self.lp, self.chosen)
        self.assertFalse(failures, failures)
        self.assertEqual(set(fits), set(METHODS))
        X, names = bank_matrix(self.lp, self.chosen, 'R5_sel')
        for name, fit in fits.items():
            self.assertEqual(fit['weights'].shape, (len(COLUMN_NAMES),))
            np.testing.assert_allclose(X @ fit['effective'] + fit['intercept'], fit['score'], atol=1e-8, rtol=1e-8)
            for key in ('score', 'weights', 'state', 'diagnostics', 'effective', 'intercept'):
                self.assertIn(key, fit)
        np.testing.assert_array_equal(fits['view__H1']['score'], X[:, 1])
        for name in ('R5__equal', 'R5__iu'):
            self.assertTrue(np.all(fits[name]['weights'][5:] == 0))

    def test_iu_transposition(self):
        """upcr_fit expects features x samples: passing Z (not Z.T) must differ."""
        X, _ = bank_matrix(self.lp, self.chosen, 'R5_sel')
        Z, keep, _, _ = zscore_columns(X)
        anchor = anchor_stream(self.lp)
        signs = np.where(np.array([np.corrcoef(Z[:, j], anchor)[0, 1] for j in range(Z.shape[1])]) < 0, -1., 1.)
        Zo = Z * signs
        right = upcr_fit(Zo.T, **dict(IU_FIT_DEFAULTS))
        self.assertEqual(len(right.w), Zo.shape[1])
        fits, _, _ = fit_all(self.lp, self.chosen, methods=('R5_sel__iu',))
        w = fits['R5_sel__iu']['state']['w']
        np.testing.assert_allclose(w, right.w, atol=1e-12)
        try:
            wrong = upcr_fit(Zo, **dict(IU_FIT_DEFAULTS))
        except Exception:
            return
        self.assertNotEqual(len(wrong.w), Zo.shape[1], 'transposed input silently accepted')

    def test_failure_paths(self):
        fits, failures, _ = fit_all(self.lp[:2], self.chosen[:2])
        for m in ('R5__equal', 'R5__iu', 'R5_sel__equal', 'R5_sel__iu', 'R5_sel__shrink'):
            self.assertIn(m, failures)
        constant = np.repeat(self.lp[:1], 30, axis=0)
        fits, failures, _ = fit_all(constant, np.full(30, 1.0))
        self.assertEqual(set(failures), set(METHODS))
        self.assertEqual(fits, {})
        bad = self.lp.copy(); bad[0, 0] = np.nan
        fits, failures, _ = fit_all(bad, self.chosen)
        self.assertEqual(set(failures), set(METHODS))
        # SEL constant but Renyi varying: R5 arms succeed, shrink fails explicitly (single group).
        fits, failures, _ = fit_all(self.lp, np.full(len(self.lp), 2.0))
        self.assertIn('R5__iu', fits); self.assertIn('R5_sel__shrink', failures); self.assertIn('view__sel1', failures)
        self.assertIn('R5_sel__iu', fits)

    def test_joint_not_applicable(self):
        self.assertIn('R5_sel__joint', NOT_APPLICABLE); self.assertIn('R5__joint', NOT_APPLICABLE)
        for name in NOT_APPLICABLE:
            self.assertNotIn(name, METHODS)
            with self.assertRaises(ValueError):
                fit_all(self.lp, self.chosen, methods=(name,))

    def test_equal_is_oriented_mean(self):
        fits, _, _ = fit_all(self.lp, self.chosen, methods=('R5__equal',))
        f = fits['R5__equal']; X, _ = bank_matrix(self.lp, self.chosen, 'R5')
        Z, keep, _, _ = zscore_columns(X)
        expected = (Z * f['state']['column_signs']).mean(axis=1) * (-1 if f['diagnostics']['orientation_flipped'] else 1)
        np.testing.assert_allclose(f['score'], expected, atol=1e-12)

    def test_real_rows_fit(self):
        for key, lp, chosen in load_real_rows(limit=5):
            fits, failures, _ = fit_all(lp, chosen)
            X, _ = bank_matrix(lp, chosen, 'R5_sel')
            for name, fit in fits.items():
                np.testing.assert_allclose(X @ fit['effective'] + fit['intercept'], fit['score'], atol=1e-8, rtol=1e-8)
            d = view_diagnostics(lp, chosen)
            self.assertTrue(d['hartley_constant'], key)


def main():
    global SOURCE_ROOT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=None)
    args, rest = p.parse_known_args()
    SOURCE_ROOT = args.source_root.resolve() if args.source_root else None
    if SOURCE_ROOT is not None:
        try:
            load_real_rows()
        except unittest.SkipTest as reason:
            print('[real rows]', reason, flush=True)
    unittest.main(argv=[sys.argv[0]] + rest)


if __name__ == '__main__':
    main()
