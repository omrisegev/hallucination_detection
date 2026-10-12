import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np

spec = importlib.util.spec_from_file_location('tested_groups', Path(__file__).resolve().parents[1] / 'spectral_utils/group_contrast_oracle.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def test_compression_loses_internal_difference():
    x = np.array([[1., 3., 5.], [3., 1., 5.]])
    groups = [0, 0, 1]
    a = g.group_means_matrix(groups)
    np.testing.assert_array_equal(x @ a, [[2., 5.], [2., 5.]])
    d, _ = g.cs.differences(3, groups)
    assert (x @ d)[0, 0] != (x @ d)[1, 0]
    assert g.cs.span(np.column_stack([a, d])).shape[1] == 3


def test_nonnegative_constraint_and_unrestricted_objective():
    rng = np.random.default_rng(918)
    x = rng.normal(size=(400, 3))
    y = (2 * x[:, 0] - 2 * x[:, 1] + rng.normal(size=400) > 0).astype(int)
    theta, grad = g.nonnegative_lr(x, y)
    assert theta[:-1].min() >= 0 and grad < 1e-7
    signed = g.cs.lr_fit(x, y)
    assert signed.coef_[0, 1] < 0
    weight = len(y) / (2. * np.bincount(y)[y])
    def objective(coef, intercept):
        z = x @ coef + intercept
        return weight @ (np.logaddexp(0, z) - y * z) + .5 * (coef @ coef)
    assert objective(signed.coef_[0], signed.intercept_[0]) <= objective(theta[:-1], theta[-1])


def test_fold_isolation_and_full_span():
    rng = np.random.default_rng(919)
    x = rng.normal(size=(200, 5))
    groups = [0, 0, 1, 1, 2]
    z = x @ g.group_means_matrix(groups)
    y = (x[:, 0] - x[:, 1] + rng.normal(size=200) > 0).astype(int)
    folds = np.repeat(np.arange(5), 40)
    te, scores, info = g.fold_fit(x, z, y, folds, 0, groups)
    changed = y.copy(); changed[te] = 1 - changed[te]
    _, other, other_info = g.fold_fit(x, z, changed, folds, 0, groups)
    assert info == other_info
    for key in scores:
        np.testing.assert_array_equal(scores[key], other[key])
    xx = x.copy(); zz = z.copy(); xx[te] += 30; zz[te] -= 20
    _, _, shifted = g.fold_fit(xx, zz, y, folds, 0, groups)
    for key in ['fits', 'x_mean', 'x_scale', 'z_mean', 'z_scale', 'group_basis', 'internal_basis']:
        assert info[key] == shifted[key]
    scaler = g.StandardScaler().fit(x[~te])
    model = g.cs.lr_fit(scaler.transform(x[~te]), y[~te])
    np.testing.assert_allclose(scores['LR_RAW_GROUPS_PLUS_WITHIN'], model.decision_function(scaler.transform(x[te])), atol=1e-5)


def test_independent_tie_metrics():
    pop = SimpleNamespace(prm=np.array([True, False]), eligible=np.array([True, False]),
                          labels=np.array([0, 1, 0, 1, 0, 0]), off=np.array([0, 4, 6]), target=np.array([-1, 0]))
    bank = dict(answers=np.arange(2), off=pop.off)
    score = np.array([0., 1., 1., 2., 3., 3. + np.finfo(float).eps])
    auc, hit = g.independent_metrics(score, bank, pop)
    assert auc[0] == .875 and hit[1] == 1.
    for a, b in zip(g.cs.per_answer(score, bank, pop), (auc, hit)):
        np.testing.assert_allclose(a, b, equal_nan=True)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    assert len(tests) == 4
    for test in tests:
        test(); print(test.__name__, 'PASS', flush=True)
    print('4/4 tests PASS', flush=True)
