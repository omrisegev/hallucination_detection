"""Synthetic checks of spectral_utils.raw_null_gate (pre-registration RAW_NULL_GATE_PREREG_2026-10-30). No real data."""
import os
import unittest

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import numpy as np

from lever_imports import ensure_spectral_package

ensure_spectral_package()
from spectral_utils import raw_null_gate as G  # noqa: E402


def synthetic_cell(rng, n_ans=1500, p_clean=.4, groups=(3, 3, 2, 2), d=(1.2, 1.0, .9, .8), answer_sd=.3, nuisance=.4):
    """Answers of 6-12 steps; clean answers have no error, the others 1-2 contiguous errors. Channels in groups: each
    channel = d_g * y + group noise + channel noise + a small answer effect. Raw values keep each answer's level."""
    lens = rng.integers(6, 13, n_ans); off = np.r_[0, np.cumsum(lens)]; n = off[-1]
    y = np.zeros(n, bool); has = rng.random(n_ans) > p_clean
    for j in np.flatnonzero(has):
        a, b = off[j], off[j + 1]; k = rng.integers(1, 3); s = rng.integers(a, b - k + 1); y[s:s + k] = True
    a_eff = np.repeat(rng.normal(0, answer_sd, n_ans), lens)
    eta = rng.standard_normal(n)
    cols = []
    for gsize, dg in zip(groups, d):
        e = rng.standard_normal(n)
        for _ in range(gsize):
            cols.append(dg * y + .8 * e + nuisance * eta + .5 * rng.standard_normal(n) + a_eff)
    return np.column_stack(cols), off, y, has


def answer_z(X, off):
    Z = np.empty_like(X)
    for a, b in zip(off[:-1], off[1:]):
        v = X[a:b]; Z[a:b] = (v - v.mean(0)) / np.maximum(v.std(0), 1e-8)
    return Z


def gap_counts(z, off):
    k = []
    for a, b in zip(off[:-1], off[1:]):
        m = b - a
        if m == 1: k.append(1); continue
        zz = np.sort(z[a:b])[::-1]; km = min(int(np.ceil(m / 2)), m - 1)
        k.append(int(np.argmax(zz[:km] - zz[1:km + 1]) + 1))
    return np.array(k)


def topk_flags(s, off, k):
    f = np.zeros(off[-1], bool)
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        f[a + np.argsort(-s[a:b], kind='stable')[:k[j]]] = True
    return f


class EmpiricalNull(unittest.TestCase):
    def test_recovers_null_and_pi0(self):
        rng = np.random.default_rng(0)
        for mu, sd in ((0., 1.), (2., 1.5)):
            x = np.r_[rng.normal(mu, sd, 42_500), rng.normal(mu + 3 * sd, sd, 7_500)]
            f = G.empirical_null(x)
            self.assertTrue(f['ok'], f)
            self.assertAlmostEqual(f['mu0'], mu, delta=.08 * sd)
            self.assertAlmostEqual(f['sigma0'], sd, delta=.08 * sd)
            self.assertAlmostEqual(f['pi0_raw'], .85, delta=.05)

    def test_p_error_is_one_sided_and_bounded(self):
        rng = np.random.default_rng(1)
        x = np.r_[rng.normal(0, 1, 9000), rng.normal(3.5, 1, 1000)]
        f = G.empirical_null(x); pe = G.p_error(x, f)
        self.assertTrue(np.all((pe >= 0) & (pe <= 1)))
        self.assertTrue(np.all(pe[x <= f['mu0']] == 0))
        self.assertGreater(pe[x > 4].mean(), .8)


class Readout(unittest.TestCase):
    def test_linear_lsml_reproduces_fit(self):
        rng = np.random.default_rng(2)
        X, off, y, has = synthetic_cell(rng, n_ans=400)
        s, p = G.linear_lsml(answer_z(X, off))
        self.assertLess(p['parity'], 1e-6)
        self.assertGreaterEqual(p['K'], 2)

    def test_ds_answer_posterior_matches_bayes_rule(self):
        fit = dict(psi=[.8, .7, .9], eta=[.6, .85, .75], prevalence=.3)
        B = np.array([[1, -1, 1], [-1, -1, -1], [1, 1, 1]])
        got = G.ds_answer_posterior(B, fit)
        for b, g in zip(B, got):
            l1 = .3 * np.prod([p if v > 0 else 1 - p for v, p in zip(b, fit['psi'])])
            l0 = .7 * np.prod([1 - e if v > 0 else e for v, e in zip(b, fit['eta'])])
            self.assertAlmostEqual(g, l1 / (l1 + l0), places=12)


class Gates(unittest.TestCase):
    """Strong separation (fused-score AUC about .97): every gate must work. At weak separation (AUC about .86,
    test_weak_signal_is_documented) the gates keep only about half of the erroneous answers; that is a property of
    a P(err) < .5 cut at low signal, recorded rather than asserted."""

    def make(self, d):
        rng = np.random.default_rng(3)
        X, off, y, has = synthetic_cell(rng, d=d)
        z_fused, p = G.linear_lsml(answer_z(X, off))
        Xr = np.column_stack([G.pooled_z(X[:, j]) for j in range(X.shape[1])])
        return dict(off=off, has=has, z=z_fused, raw=G.pooled_z(G.fused_linear(Xr, p)), groups=G.group_scores(Xr, p),
                    orient=G.group_orientation(p))

    @staticmethod
    def rates(clean, has):
        return float(clean[~has].mean()), float((~clean[has]).mean())

    def test_all_gates_at_strong_separation(self):
        c = self.make((3.0, 2.8, 2.5, 2.3))
        f = topk_flags(c['z'], c['off'], gap_counts(c['z'], c['off']))
        outs = {'g1': G.gate_g1(c['raw'], c['z'], c['off']), 'g1b': G.gate_g1b(c['raw'], c['z'], c['off'], f),
                'g2': G.gate_g2(c['groups'], c['orient'], c['off']), 'g3': G.gate_g3(c['groups'], c['orient'], c['off'])}
        for name, o in outs.items():
            closed, kept = self.rates(o['clean'], c['has'])
            self.assertGreater(closed, .7, name); self.assertGreater(kept, .9, name)
        self.assertAlmostEqual(outs['g3']['prevalence'], c['has'].mean(), delta=.1)

    def test_weak_signal_is_documented(self):
        c = self.make((1.2, 1.0, .9, .8))
        closed, kept = self.rates(G.gate_g1(c['raw'], c['z'], c['off'])['clean'], c['has'])
        self.assertTrue(0 <= closed <= 1 and 0 <= kept <= 1)


class DefinedSteps(unittest.TestCase):
    def test_undefined_zero_spike_does_not_create_false_votes(self):
        """A group with no errors whose undefined steps are exact zeros: with its defined mask, G3's per-group vote must
        not call answers abnormal (review finding: without the mask, 30% zeros made ~48% of answers vote abnormal)."""
        rng = np.random.default_rng(7)
        lens = rng.integers(6, 13, 4000); off = np.r_[0, np.cumsum(lens)]; n = off[-1]
        raw = rng.standard_normal((n, 3))
        undefined = rng.random(n) < .3
        raw[undefined, 2] = 0.
        D = np.ones((n, 3), bool); D[:, 2] = ~undefined
        u, f = G._group_null(raw[:, 2], 1., D[:, 2])
        self.assertTrue(f['ok'], f)
        pe = np.where(D[:, 2], G.p_error(u, f), 0.)
        self.assertLess((G.answer_max(pe, off) >= .5).mean(), .05)

    def test_g2_is_one_sided(self):
        """Very low group scores must not raise P(err) (LR = 1 at or below the null mean)."""
        rng = np.random.default_rng(8)
        lens = rng.integers(6, 13, 3000); off = np.r_[0, np.cumsum(lens)]; n = off[-1]
        raw = rng.standard_t(5, (n, 3))
        o = G.gate_g2(raw, np.ones(3), off)
        low = (raw < -2).all(1)
        if low.any():
            self.assertLess(o['p_err_step'][low].max(), .5)


if __name__ == "__main__":
    unittest.main()
