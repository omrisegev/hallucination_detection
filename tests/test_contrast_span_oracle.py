"""Boundary and leakage checks for the contrast-span diagnostic."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cs_tested', ROOT / 'spectral_utils/contrast_span_oracle.py')
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)


def test_complete_and_restricted_spans():
    d, _ = cs.differences(6, [0, 0, 0, 1, 1, 2])
    assert cs.span(d).shape == (6, 3)
    b = cs.span(d)
    duplicate = cs.span(np.column_stack([d, d, -d]))
    np.testing.assert_allclose(b @ b.T, duplicate @ duplicate.T, atol=1e-12)
    all_d, _ = cs.differences(6)
    full = cs.span(np.column_stack([np.ones(6), all_d]))
    np.testing.assert_allclose(full @ full.T, np.eye(6), atol=1e-12)


def test_labels_in_test_fold_cannot_change_fits():
    rng = np.random.default_rng(177)
    x = rng.normal(size=(200, 4))
    y = (x[:, 0] - x[:, 1] + rng.normal(size=200) > 0).astype(int)
    folds = np.repeat(np.arange(5), 40)
    names = ['realized_drv', 'chosen_surprisal', 'c', 'd']
    args = (folds, 0, [0, 0, 1, 1], names, np.arange(0, 201, 10))
    te, a, fit_a = cs.fold_fit(x, y, *args)
    changed = y.copy()
    changed[te] = 1 - changed[te]
    _, b, fit_b = cs.fold_fit(x, changed, *args)
    assert fit_a == fit_b
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_test_features_cannot_change_fitted_parameters():
    rng = np.random.default_rng(178)
    x = rng.normal(size=(200, 4))
    y = (x[:, 0] + rng.normal(size=200) > 0).astype(int)
    folds = np.repeat(np.arange(5), 40)
    names = ['realized_drv', 'chosen_surprisal', 'c', 'd']
    args = (folds, 0, [0, 0, 1, 1], names, np.arange(0, 201, 10))
    _, _, a = cs.fold_fit(x, y, *args)
    x[folds == 0] += 100
    _, _, b = cs.fold_fit(x, y, *args)
    for key in ['scale_mean', 'scale_sd', 'pc1', 'fits']:
        assert a[key] == b[key]


def test_metric_ties_and_eligibility():
    pop = SimpleNamespace(prm=np.array([True, False, False]), eligible=np.array([True, False, False]),
                          labels=np.array([0, 1, 0, 0, 0, 0]), off=np.array([0, 2, 4, 6]),
                          target=np.array([-1, 0, -1]))
    bank = dict(answers=np.arange(3), off=pop.off)
    scores = np.array([1., 1., 2., 2. + np.finfo(float).eps, 0., 1.])
    auc, hit = cs.per_answer(scores, bank, pop)
    assert auc[0] == .5 and hit[1] == 1.
    assert np.isnan(hit[2])


def test_missing_named_feature_is_unavailable_not_substituted():
    rng = np.random.default_rng(179)
    x = rng.normal(size=(200, 4))
    y = (x[:, 0] + rng.normal(size=200) > 0).astype(int)
    folds = np.repeat(np.arange(5), 40)
    _, scores, info = cs.fold_fit(x, y, folds, 0, [0, 0, 1, 1],
                                 ['different', 'chosen_surprisal', 'c', 'd'], np.arange(0, 201, 10))
    assert not info['single_contrast_available']
    assert 'LR_PC1_single_linear' not in scores
    assert 'LR_PC1_single_answer_std' not in scores
    assert 'LR_PC1_within_group_linear' in scores


def test_bootstrap_excludes_ineligible_and_absent_arms():
    import pandas as pd
    rows = []
    for panel in ['PRMB', 'PB']:
        for answer in range(8):
            for method, value in [('a', .8), ('b', .5)]:
                rows.append(dict(bank='B', panel=panel, cell=panel, source_group=str(answer),
                                 answer=answer, method=method, value=value if answer < 7 else np.nan))
    out = cs.bootstrap_differences(pd.DataFrame(rows), [('a', 'b'), ('missing', 'b')], draws=30)
    assert len(out) == 2
    np.testing.assert_allclose(out['delta'], .3)
    np.testing.assert_allclose(out['lo95'], .3)
    assert set(out.n_source_groups) == {7}


def test_rotation_precision_with_collinear_features():
    rng = np.random.default_rng(181)
    x = rng.normal(size=(400, 8))
    x[:, 4:] = x[:, :4] + 1e-3 * x[:, 4:]
    y = (x[:, 0] - x[:, 1] + rng.normal(size=400) > 0).astype(int)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    a, b = cs.lr_fit(x, y), cs.lr_fit(x @ q, y)
    np.testing.assert_allclose(a.decision_function(x), b.decision_function(x @ q), atol=1e-8, rtol=0)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    assert len(tests) == 7
    for test in tests:
        test()
        print(test.__name__, 'PASS', flush=True)
    print('7/7 tests PASS', flush=True)
