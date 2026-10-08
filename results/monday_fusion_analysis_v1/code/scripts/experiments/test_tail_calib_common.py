"""Known-result tests for tail_calib_common (replay of lsml_fit, scale invariance after
standardization, group search from R == detect_dependent_groups, exact latent-group residual = 0,
nested label guard, value marks, coordinate descent)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calfix_common import lsml_fit  # noqa: E402
from calfix_evaluate import prmscore  # noqa: E402
import tail_calib_common as TC  # noqa: E402


def _blocks(n=4000, sizes=(4, 4, 4, 3), seed=0):
    rng = np.random.default_rng(seed); y = rng.normal(size=n); cols = []
    for g, m in enumerate(sizes):
        f = rng.normal(size=n)
        for _ in range(m):
            cols.append(.5 * y + .6 * f + rng.normal(size=n))
    return np.column_stack(cols)


def test_replay_lsml_fit():
    X = _blocks()
    a = lsml_fit(X, 0); b = TC.lsml_fit_scaled(X, 0, standardize=False, loading_scale='unit')
    assert np.array_equal(a['weights'], b['weights']) and np.array_equal(a['groups'], b['groups']) and a['K'] == b['K']
    return 'lsml_fit_scaled(standardize=False, unit) == lsml_fit'


def test_scale_invariance():
    X = _blocks(seed=1)
    for scale in ('unit', 'complete'):
        a = TC.lsml_fit_scaled(X, 0, standardize=True, loading_scale=scale); b = TC.lsml_fit_scaled(.4 * X, 0, .4 * X, standardize=True, loading_scale=scale)
        assert a['K'] == b['K'] and np.array_equal(a['groups'], b['groups']) and np.allclose(a['weights'], b['weights'], atol=1e-10), scale
    return 'standardized fit invariant to input scale'


def test_groups_from_R():
    X = _blocks(sizes=(5, 5, 5), seed=2); F = TC.fu()
    for scale in ('unit', 'complete'):
        K, c, r, _s, _curve = F.detect_dependent_groups(list(X.T), loading_scale=scale, return_curve=True)
        K2, c2, r2, _ = TC.groups_from_R(np.cov(X.T), scale)
        assert K == K2 and np.array_equal(c, c2) and abs(r - r2) < 1e-12, scale
    return 'groups_from_R == detect_dependent_groups'


def test_exact_residual_zero():
    rng = np.random.default_rng(3); c = np.repeat([0, 1, 2], 5)
    von = rng.uniform(.4, .7, 15); voff = rng.uniform(.2, .4, 15)
    R = np.where(c[:, None] == c[None, :], np.outer(von, von), np.outer(voff, voff)); np.fill_diagonal(R, 1.0)
    assert np.linalg.eigvalsh(R).min() > 0
    r = TC.residual_at(R, c, 'complete'); assert r < 1e-10, r
    rel, K, cc = TC.rel_residual(R, 'complete'); assert rel < 1e-8 and K == 3, (rel, K)
    return 'exact latent-group correlation: complete residual = 0 and K=3 recovered'


def test_fit_prmscore_guard():
    rng = np.random.default_rng(4); s = rng.normal(size=200); y = (rng.random(200) < .7).astype(float)
    sel = np.arange(100); cal = np.arange(100, 200)
    got = TC.fit_prmscore(s, y, sel, cal)
    tau = np.quantile(s[cal], .8); p = s[sel] < tau; yy = y[sel].astype(bool)
    ref = float(prmscore(np.array([np.sum(p & yy), np.sum(p & ~yy), np.sum(~p & ~yy), np.sum(~p & yy)], float)))
    assert abs(got - ref) < 1e-15
    ym = y.copy(); ym[50] = np.nan; caught = False
    try:
        TC.fit_prmscore(s, ym, sel, cal)
    except AssertionError as e:
        caught = 'withheld labels' in str(e)
    assert caught, 'masked label read without error'
    return 'fit_prmscore == official formula; withheld labels raise'


def test_value_marks():
    rng = np.random.default_rng(5); off = np.array([0, 7, 12, 30]); V = rng.normal(size=(30, 2))
    T, tau = TC.value_marks(V, off, .3, np.arange(30))
    assert np.allclose(np.add.reduceat(T, off[:-1], axis=0), 0)
    raw = (V > tau).mean(0); assert np.all(np.abs(raw - .3) <= 1 / 30 + 1e-12), raw
    M = {q: np.full((4, 3), q) for q in (.1, .2)}; assert np.array_equal(TC.mixed_marks(M, (.1, .2, .1))[0], [.1, .2, .1])
    return 'value marks centred, rate q; mixed marks pick per-column fraction'


def test_coordinate_descent():
    target = (.05, .3, .5, .2)
    qs, v, tr = TC.coordinate_descent(4, lambda q: -sum((a - b) ** 2 for a, b in zip(q, target)))
    assert qs == target and abs(v) < 1e-15
    qs2, _, _ = TC.coordinate_descent(3, lambda q: 1.0)          # flat objective: never moves
    assert qs2 == (.2, .2, .2)
    return 'coordinate descent reaches a separable optimum; flat objective keeps the start'


def test_fast_answer_z():
    rng = np.random.default_rng(6); off = np.array([0, 4, 5, 12]); s = rng.normal(size=12); s[0:4] = 2.0
    z = TC.fast_answer_z(s, off)
    assert np.allclose(z[0:5], 0) and abs(z[5:12].mean()) < 1e-12 and abs(z[5:12].std() - 1) < 1e-12
    return 'fast answer-z: constant and one-step answers -> 0'


if __name__ == '__main__':
    for t in (test_replay_lsml_fit, test_scale_invariance, test_groups_from_R, test_exact_residual_zero, test_fit_prmscore_guard,
              test_value_marks, test_coordinate_descent, test_fast_answer_z):
        print(t())
    print('ALL PASS')
