"""Synthetic proof of each v1b fix (docs/experiments/RAW_NULL_GATE_V1B_BUGFIX_2026-10-31.md). No real data.

Each test shows the v1 behaviour (the bug) next to the v1b behaviour (the fix) on the same input."""
import importlib.util
import os
import unittest
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import numpy as np

from lever_imports import ensure_spectral_package

ensure_spectral_package()
from spectral_utils import raw_null_gate as G1  # noqa: E402
from spectral_utils import raw_null_gate_v1b as G  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def answers(n, rng, lo=6, hi=13):
    lens = rng.integers(lo, hi, n)
    return np.r_[0, np.cumsum(lens)]


class F2Characterization(unittest.TestCase):
    """F2 was dropped: these tests pin down WHY (a strict pi0_raw <= 1 rule rejects error-free data)."""
    def test_pure_null_has_pi0_raw_above_one(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(16_000)
        old, new = G1.empirical_null(x), G.empirical_null(x)
        self.assertTrue(old['ok'] and new['ok'])
        self.assertGreater(new['pi0_raw'], 1)
        self.assertEqual(old['pi0_raw'], new['pi0_raw'])
        self.assertAlmostEqual(new['sd'], x.std())

    def test_separated_mixture_passes(self):
        rng = np.random.default_rng(2)
        x = np.r_[rng.normal(0, 1, 51_000), rng.normal(4, 1, 9_000)]
        f = G.empirical_null(x)
        self.assertTrue(f['ok'], f.get('reason'))
        self.assertLessEqual(f['pi0_raw'], 1)
        self.assertLess(f['sigma0'], x.std())
        self.assertAlmostEqual(f['pi0'], .85, delta=.05)


def _two_group_null_data(rng, n_ans=3000):
    off = answers(n_ans, rng); n = off[-1]
    y = np.zeros(n, bool)
    for j in np.flatnonzero(rng.random(n_ans) < .4):
        y[rng.integers(off[j], off[j + 1])] = True
    return off, y


class F1OwnPi0(unittest.TestCase):
    def test_single_group_gives_back_its_own_perr(self):
        """With one group, the product posterior must be that group's own 1 - lfdr (outside the LR clip)."""
        rng = np.random.default_rng(3)
        off, y = _two_group_null_data(rng)
        raw = (3.5 * y + rng.standard_normal(len(y)))[:, None]
        out = G.gate_g2(raw, np.array([1.]), off)
        u, f = G._group_null(raw[:, 0], 1., np.ones(len(y), bool))
        pe = G.p_error(u, f)
        lr = (f['pi0'] / (1 - f['pi0'])) * (1 / np.maximum(1 - pe, 1e-12) - 1)
        m = (u > f['mu0']) & (lr > G.LR_CLIP[0]) & (lr < G.LR_CLIP[1])
        self.assertGreater(m.sum(), 100)
        np.testing.assert_allclose(out['p_err_step'][m], pe[m], atol=1e-9)

    def test_no_error_groups_overcall_less(self):
        """No errors at all, three groups with different shapes (review B probe): the shared-pi0 LR over-calls;
        own-pi0 LR calls fewer answers erroneous on the same data."""
        rng = np.random.default_rng(4)
        off = answers(3000, rng); n = off[-1]
        raw = np.column_stack([rng.standard_normal(n), rng.standard_normal(n), rng.standard_t(6, n)])
        o = np.ones(3)
        old = G1.gate_g2(raw, o, off); new = G.gate_g2(raw, o, off)
        share_old, share_new = 1 - old['clean'].mean(), 1 - new['clean'].mean()
        self.assertLess(share_new, share_old)


class F3Monotone(unittest.TestCase):
    def test_monotone_above_mu0(self):
        rng = np.random.default_rng(5)
        x = np.r_[rng.normal(0, 1, 45_000), rng.normal(2.5, 1.3, 5_000)]
        f = G.empirical_null(x)
        self.assertTrue(f['ok'], f.get('reason'))
        p = G.p_error(x, f, monotone=True)
        up = x > f['mu0']; o = np.argsort(x[up], kind='stable')
        self.assertTrue(np.all(np.diff(p[up][o]) >= 0))
        self.assertTrue(np.all(p[~up] == 0))
        q = G.p_error(x, f, monotone=False)
        self.assertTrue(np.all(p >= q - 1e-15))                              # the running max never lowers P(err)


class F4ConstantVoters(unittest.TestCase):
    def _cell(self, k_inf, rng):
        off, y = _two_group_null_data(rng, 2000)
        cols = [2.5 * y + rng.standard_normal(len(y)) for _ in range(k_inf)] + [rng.standard_normal(len(y))]
        return np.column_stack(cols), off

    def _with_constant_last_group(self, raw, off):
        last = raw.shape[1] - 1
        real = G._group_perr

        def perr(u, f, d, monotone):                                          # the last group never votes "error"
            return np.zeros(len(u)) if perr.calls % raw.shape[1] == last else real(u, f, d, monotone)

        def counted(u, f, d, monotone):
            out = perr(u, f, d, monotone); perr.calls += 1
            return out
        perr.calls = 0
        G._group_perr = counted
        try:
            return G.gate_g3(raw, np.ones(raw.shape[1]), off)
        finally:
            G._group_perr = real

    def test_constant_voter_is_dropped(self):
        raw, off = self._cell(3, np.random.default_rng(6))
        out = self._with_constant_last_group(raw, off)
        self.assertTrue(out['ok'], out.get('reason'))
        self.assertEqual(out['constant_voters'], [3])
        self.assertEqual(out['groups_used'], [0, 1, 2])

    def test_too_few_varying_voters_is_invalid(self):
        raw, off = self._cell(2, np.random.default_rng(7))
        out = self._with_constant_last_group(raw, off)
        self.assertFalse(out['ok'])
        self.assertEqual(out['constant_voters'], [2])


class F5F6Runner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.V1 = _load('results/raw_null_gate_v1/run_raw_null_gate.py', 'run_v1_t')
        cls.V1B = _load('results/raw_null_gate_v1b/run_raw_null_gate_v1b.py', 'run_v1b_t')

    def test_invalid_gate_is_not_no_harm(self):
        rows = []
        for b in self.V1.BENCHES:
            for p in self.V1.PRMS:
                rows.append(dict(arm='SUB5', mode='sa', bench=b, prm=p, gate='GAP_ONLY', metric=50.))
                rows.append(dict(arm='SUB5', mode='sa', bench=b, prm=p, gate='G3', ok=False, closed_clean=0.,
                                 kept_error=1., metric=50.))
        old = self.V1.verdict(rows)['SUB5|sa|G3']['prmbench_no_harm']
        new = self.V1B.verdict(rows)['SUB5|sa|G3']['prmbench_no_harm']
        self.assertEqual(old, '4/4')
        self.assertEqual(new, '0/4')

    def test_ece_ignores_float_noise_on_ties(self):
        rng = np.random.default_rng(8)
        n = 20_000
        p = np.where(rng.random(n) < .35, .0585847, rng.random(n))
        y = (rng.random(n) < p).astype(float)
        q = p + rng.normal(0, 1e-13, n)
        self.assertNotEqual(self.V1.ece(p, y), self.V1.ece(q, y))           # v1: tie order moves steps across bins
        self.assertAlmostEqual(self.V1B.ece(p, y), self.V1B.ece(q, y), places=10)


if __name__ == '__main__':
    unittest.main()
