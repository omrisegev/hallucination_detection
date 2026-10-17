"""Mechanism checks for the varentropy expansion fusion; not benchmark performance tests."""
import os
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from spectral_utils import varentropy_expansion_fusion as vef
from spectral_utils.varentropy_expansion_fusion import (
    BANKS, BANK_COLUMNS, EXPECTED_GROUP_COUNT, GROUPS, HISTORICAL, IDENTITY_COEFFICIENTS, IDENTITY_SIGNS, METHODS, SOLVERS,
    TERM, WIDTH, _solve, anchor_series, expansion_columns, fit_all, group_admissibility, identity_discrepancy, identity_fixed_fusion)
from spectral_utils.varentropy_contribution_fusion import contributions
from spectral_utils.varentropy_contribution_fusion import fit_all as historical_fit_all
from spectral_utils.direct_probability_fusion import step_top_mean, zscore_columns
from spectral_utils.upcr import upcr_fit
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS
from spectral_utils import varentropy_expansion_supervised as sup
from spectral_utils.rbm_matched_top10 import top10_value_gradient

SOURCE = Path(os.environ.get('HD_SOURCE_ROOT', r'C:\Users\omris\TAU\hallucination_detection'))
REAL_PICKLE = SOURCE / 'dataset_cache/repgrid/pb_qwen3_4b/processbench_gsm8k.pkl'


def synthetic(seed=340, T=160):
    rng = np.random.default_rng(seed)
    logits = np.sort(rng.normal(size=(T, 50)) * rng.uniform(.3, 3, size=(T, 1)), axis=1)[:, ::-1]
    p = np.exp(logits - logits.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
    lp = np.log(p).astype(np.float32).astype(float)          # saved logprobs are float32
    chosen = rng.uniform(0, 4, size=T)
    spans = np.array([[0, 20], [20, 55], [55, 90], [90, T]])
    return lp, chosen, spans


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.lp, self.chosen, self.spans = synthetic()

    def test_layout(self):
        X, names, coef = expansion_columns(self.lp, self.chosen)
        self.assertEqual(X.shape, (len(self.lp), WIDTH)); self.assertEqual(len(names), WIDTH)
        self.assertEqual(list(coef[:15]), [1.] * 15); self.assertEqual(list(coef[15:30]), [-1.] * 15)
        self.assertEqual(list(coef[30:135]), [-2.] * 105); self.assertEqual(list(coef[135:]), [0.] * 3)
        self.assertEqual(TERM.count('D'), 15); self.assertEqual(TERM.count('Pii'), 15); self.assertEqual(TERM.count('Pij'), 105); self.assertEqual(TERM.count('SEL'), 3)
        self.assertEqual(expansion_columns(self.lp, None)[0].shape, (len(self.lp), 135))
        self.assertEqual({b: len(c) for b, c in BANK_COLUMNS.items()}, {'B2d_sel': 33, 'B2_sel': 138, 'B2d': 30, 'B2': 135})

    def test_identity_synthetic_and_manual(self):
        X, _, _ = expansion_columns(self.lp, self.chosen)
        self.assertLess(identity_discrepancy(X, self.lp), 1e-9)
        # manual per-token expansion from the frozen q/s convention
        lp15 = self.lp[:, :15]; p = np.exp(lp15); q = p / (p.sum(axis=1, keepdims=True) + 1e-12); s = -np.log(q + 1e-12)
        for t in range(3):
            D = [q[t, i] * s[t, i] ** 2 for i in range(15)]; Pii = [(q[t, i] * s[t, i]) ** 2 for i in range(15)]
            Pij = [q[t, i] * s[t, i] * q[t, j] * s[t, j] for i in range(15) for j in range(i + 1, 15)]
            np.testing.assert_allclose(X[t, :15], D, rtol=1e-12); np.testing.assert_allclose(X[t, 15:30], Pii, rtol=1e-12)
            np.testing.assert_allclose(X[t, 30:135], Pij, rtol=1e-12)
            V = sum(D) - sum(Pii) - 2 * sum(Pij)
            self.assertAlmostEqual(V, float(contributions(self.lp, 15).sum(axis=1)[t]), delta=1e-9)
        np.testing.assert_allclose(X[:, 135], np.maximum(self.chosen, 0)); np.testing.assert_allclose(X[:, 137], np.maximum(self.chosen, 0) ** 3)

    @unittest.skipUnless(REAL_PICKLE.exists() and os.environ.get('HD_SKIP_REAL_ROWS') is None, 'benchmark pickle unavailable')
    def test_identity_real_rows(self):
        import pickle
        with REAL_PICKLE.open('rb') as f: payload = pickle.load(f)
        rows = [r for r in (payload.values() if isinstance(payload, dict) else payload) if isinstance(r, dict)][:6]
        self.assertTrue(rows)
        for row in rows:
            lp = np.asarray(row['top_k_logprobs']['logprobs'], float); chosen = np.asarray(row['token_spilled_energies'], float)
            X, _, _ = expansion_columns(lp, chosen)
            self.assertLess(identity_discrepancy(X, lp), 1e-9)
            np.testing.assert_allclose(identity_fixed_fusion(X), contributions(lp, 15).sum(axis=1), atol=1e-9, rtol=0)

    def test_identity_fusion_equals_k15_raw(self):
        X, _, _ = expansion_columns(self.lp, self.chosen)
        fits, failures, _ = fit_all(self.lp, self.chosen, uid='u')
        raw = fits['B1_hist__raw']['score']
        np.testing.assert_array_equal(raw, contributions(self.lp, 15).sum(axis=1))
        np.testing.assert_allclose(identity_fixed_fusion(X), raw, atol=1e-9, rtol=0)
        np.testing.assert_allclose(step_top_mean(identity_fixed_fusion(X), self.spans[:, 0], self.spans[:, 1], 10),
                                   step_top_mean(raw, self.spans[:, 0], self.spans[:, 1], 10), atol=1e-9, rtol=0)
        np.testing.assert_allclose(X @ IDENTITY_COEFFICIENTS, raw, atol=1e-9, rtol=0)   # SEL coefficient is zero

    def test_group_admissibility(self):
        for bank in BANKS:
            g = GROUPS[BANK_COLUMNS[bank]]
            self.assertEqual(len(np.unique(g)), EXPECTED_GROUP_COUNT[bank])
        sizes, short = group_admissibility(GROUPS[BANK_COLUMNS['B2_sel']])
        self.assertEqual(sorted(sizes.values()), sorted([5, 5, 5, 5, 5, 5, 10, 25, 25, 10, 25, 10, 3])); self.assertEqual(short, [])
        self.assertEqual(sorted(group_admissibility(GROUPS[BANK_COLUMNS['B2d']])[0].values()), [5] * 6)
        # a group with fewer than three varying columns is a declared joint failure naming the group
        cols = BANK_COLUMNS['B2d']; Z0, keep, _, _ = zscore_columns(expansion_columns(self.lp, self.chosen)[0][:, cols])
        keep = keep.copy(); keep[10:15] = False; keep[10:12] = True        # D block 2 keeps two columns
        Z = Z0[:, keep]
        with self.assertRaises(ValueError) as ctx:
            _solve('joint', Z, keep, GROUPS[cols], IDENTITY_SIGNS[cols], anchor_series(self.lp), 'u')
        self.assertIn('group(s) [2]', str(ctx.exception)); self.assertIn('declared failure', str(ctx.exception))

    def test_iu_transposition(self):
        cols = BANK_COLUMNS['B2d_sel']; Xb = expansion_columns(self.lp, self.chosen)[0][:, cols]
        Z, keep, _, _ = zscore_columns(Xb)
        w, diag = _solve('iu', Z, keep, GROUPS[cols], IDENTITY_SIGNS[cols], anchor_series(self.lp), 'u')
        self.assertEqual(w.shape, (Z.shape[1],)); self.assertNotEqual(Z.shape[0], Z.shape[1])
        np.testing.assert_allclose(w, upcr_fit(Z.T, **dict(IU_FIT_DEFAULTS)).w)
        self.assertFalse(diag['analytic_pair_path'])
        cols = BANK_COLUMNS['B2_sel']; Z, keep, _, _ = zscore_columns(expansion_columns(self.lp, self.chosen)[0][:, cols])
        if Z.shape[1] >= 64:
            self.assertTrue(_solve('iu', Z, keep, GROUPS[cols], IDENTITY_SIGNS[cols], anchor_series(self.lp), 'u')[1]['analytic_pair_path'])

    def test_equal_identity_signs(self):
        fits, _, _ = fit_all(self.lp, self.chosen, uid='u')
        for bank in BANKS:
            f = fits[f'{bank}__equal_identity']; cols = BANK_COLUMNS[bank]; keep = f['state']['keep']; p = keep.sum()
            sign = -1 if f['diagnostics']['orientation_flipped'] else 1
            np.testing.assert_allclose(f['state']['w'], sign * IDENTITY_SIGNS[cols][keep] / p)
            np.testing.assert_allclose(f['weights'][keep], sign * IDENTITY_SIGNS[cols][keep] / p); np.testing.assert_array_equal(f['weights'][~keep], 0)
        np.testing.assert_array_equal(IDENTITY_SIGNS[:15], 1); np.testing.assert_array_equal(IDENTITY_SIGNS[15:135], -1); np.testing.assert_array_equal(IDENTITY_SIGNS[135:], 1)

    def test_all_fits_reconstruct_orient_and_historical(self):
        fits, failures, seconds = fit_all(self.lp, self.chosen, uid='u')
        self.assertEqual(failures, {}); self.assertEqual(set(fits), set(METHODS))
        X, _, _ = expansion_columns(self.lp, self.chosen); C15 = contributions(self.lp, 15); anchor = anchor_series(self.lp)
        hist = historical_fit_all(self.lp)[0]
        for m, f in fits.items():
            Xb = C15 if m in HISTORICAL else X[:, BANK_COLUMNS[m.split('__')[0]]]
            self.assertEqual(len(f['weights']), Xb.shape[1])
            np.testing.assert_allclose(Xb @ f['effective'] + f['intercept'], f['score'], atol=1e-8, rtol=1e-8)
            for key in ('score', 'weights', 'state', 'diagnostics', 'effective', 'intercept'): self.assertIn(key, f)
            if m != 'B1_hist__raw' and np.std(f['score']) > 1e-12:
                self.assertGreaterEqual(np.corrcoef(f['score'], anchor)[0, 1], 0.0)
            if m in HISTORICAL:
                old = hist['k15__' + m.split('__')[1]]
                np.testing.assert_array_equal(f['score'], old['score']); np.testing.assert_array_equal(f['weights'], old['weights'])
        self.assertIn('bank_build', seconds)
        for m in METHODS: self.assertIn(m, seconds)
        j = fits['B2_sel__joint']['diagnostics']
        for key in ('converged', 'multistart_status', 'joint_relative_offdiag_misfit', 'hard_relative_offdiag_misfit', 'model_covariance_condition', 'global_loading_cosine_min'):
            self.assertIn(key, j)
        self.assertEqual(j['n_groups'], 13); self.assertIn('alpha', fits['B2_sel__shrink']['diagnostics'])

    def test_failure_declaration(self):
        fits, failures, _ = fit_all(self.lp[:2], self.chosen[:2], uid='u')
        self.assertEqual(set(fits), {'B1_hist__raw'}); self.assertEqual(len(failures), len(METHODS) - 1)
        self.assertTrue(all(vef.is_declared_failure(r) for r in failures.values()))
        fits, failures, _ = fit_all(np.repeat(self.lp[:1], 30, axis=0), np.repeat(self.chosen[:1], 30), uid='u')
        self.assertEqual(set(fits), {'B1_hist__raw'}); self.assertTrue(all(vef.is_declared_failure(r) for r in failures.values()))
        abstain = mock.Mock(abstained=True, used_simple_average=False, w=np.ones(33), g2_hat=0., n_components_used=2, proj_residual=0.)
        with mock.patch.object(vef, 'upcr_fit', return_value=abstain):
            fits, failures, _ = fit_all(self.lp, self.chosen, uid='u')
        for bank in BANKS:
            self.assertIn(f'{bank}__iu', failures); self.assertIn('IU abstention', failures[f'{bank}__iu']); self.assertTrue(vef.is_declared_failure(failures[f'{bank}__iu']))
            self.assertIn(f'{bank}__shrink', fits)
        with self.assertRaises(ValueError): expansion_columns(self.lp, self.chosen[:-1])
        self.assertFalse(vef.is_declared_failure('AssertionError: reconstruction'))

    def test_seed_and_padding_contract(self):
        self.assertEqual(vef.seed_for('a'), vef.seed_for('a')); self.assertNotEqual(vef.seed_for('a'), vef.seed_for('b'))
        f1 = fit_all(self.lp, self.chosen, uid='same')[0]['B2d__joint']; f2 = fit_all(self.lp, self.chosen, uid='same')[0]['B2d__joint']
        np.testing.assert_array_equal(f1['score'], f2['score'])


class SupervisedTests(unittest.TestCase):
    def setUp(self):
        self.lp, self.chosen, self.spans = synthetic(seed=7, T=120)
        self.z, self.keep = sup.standardized_bank(self.lp, self.chosen)

    def test_standardized_bank(self):
        self.assertEqual(self.z.shape, (120, WIDTH)); np.testing.assert_allclose(self.z[:, self.keep].mean(axis=0), 0, atol=1e-10)
        np.testing.assert_array_equal(self.z[:, ~self.keep], 0)

    def test_top10_matches_reference(self):
        rng = np.random.default_rng(3); w = rng.normal(size=WIDTH); b = .3
        v_ref, g_ref = top10_value_gradient(self.z, self.spans, w, b)
        v, g = sup.StepTop10(self.z, self.spans).evaluate(w, b, derivative=True)
        np.testing.assert_allclose(v, v_ref, atol=1e-12); np.testing.assert_allclose(g, g_ref, atol=1e-12)
        v32, g32 = sup.StepTop10(self.z.astype(np.float32), self.spans).evaluate(w, b, derivative=True)
        np.testing.assert_allclose(v32, v_ref, rtol=1e-4, atol=1e-4)

    def test_labels_and_weights(self):
        np.testing.assert_array_equal(sup.step_labels('pb', 5, target=2), [0, 0, 1, -1, -1])
        np.testing.assert_array_equal(sup.step_labels('pb', 4, target=-1), [0, 0, 0, 0])
        np.testing.assert_array_equal(sup.step_labels('prm', 4, labels=[0, -2, 1, 1]), [0, -1, 1, 1])
        w = sup.class_balanced_weights(np.array([0, 0, 1, -1])); self.assertAlmostEqual(w[:2].sum(), .5); self.assertAlmostEqual(w[2], .5); self.assertEqual(w[3], 0)
        with self.assertRaises(ValueError): sup.class_balanced_weights(np.array([0, 0, -1]))

    def test_objective_gradient_and_fit(self):
        y = np.array([0, 1, -1, 0]); weight = sup.class_balanced_weights(y); top = sup.StepTop10(self.z, self.spans)
        rng = np.random.default_rng(5); theta = rng.normal(size=WIDTH + 1) * .1
        loss, grad = sup.objective(theta, top, y, weight)
        for k in rng.choice(WIDTH + 1, 5, replace=False):
            e = np.zeros(WIDTH + 1); e[k] = 1e-6
            fd = (sup.objective(theta + e, top, y, weight)[0] - sup.objective(theta - e, top, y, weight)[0]) / 2e-6
            self.assertAlmostEqual(fd, grad[k], delta=1e-4 * max(1, abs(grad[k])))
        theta, info = sup.fit(self.z[:, sup.bank_columns('B2d_sel')], self.spans, y)
        self.assertLessEqual(info['final_loss'], info['initial_loss']); self.assertEqual(theta.shape, (34,))
        s = sup.score_steps(self.z[:, sup.bank_columns('B2d_sel')], self.spans, theta); self.assertEqual(s.shape, (4,)); self.assertTrue(np.isfinite(s).all())


if __name__ == '__main__':
    unittest.main()
