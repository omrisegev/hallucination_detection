import importlib.util
import inspect
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'results/monday_fusion_analysis_v1/code'
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / 'scripts/experiments'))
spec = importlib.util.spec_from_file_location('orth_test', ROOT / 'spectral_utils/orthogonal_family_fusion.py')
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def test_basis_reconstruction_and_common_direction():
    for m in range(1, 33):
        b = f.helmert_basis(m)
        np.testing.assert_allclose(b @ b.T, np.eye(m), atol=1e-12)
        np.testing.assert_allclose(b[:, 1:].sum(0), 0., atol=1e-12)
        x = np.random.default_rng(m).normal(size=(20, m))
        np.testing.assert_allclose(x @ b @ b.T, x, atol=1e-12)


def test_fixed_name_order_and_group_separation():
    x = np.random.default_rng(22).normal(size=(30, 5))
    groups = np.array([0, 1, 0, 1, 0]); names = ['z', 'y', 'b', 'c', 'a']; off = np.arange(0, 31, 10)
    a, ga, ca, da = f.representation(x, groups, names, off)
    order = [4, 3, 1, 0, 2]
    b, gb, cb, db = f.representation(x[:, order], groups[order], [names[i] for i in order], off)
    np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_array_equal(ga, gb)
    np.testing.assert_allclose(ca, cb, atol=1e-12)
    original_mean = f.GW.group_matrix(x, off, groups, None, f.answer_standardize)
    np.testing.assert_allclose(ca, original_mean, atol=1e-12)


def test_tail_ties_quota_and_determinism():
    values = np.zeros((17, 3)); off = np.array([0, 1, 6, 17]); names = ['common', 'c1', 'c2']
    a = f.projected_marks(values, off, names, 'test')
    b = f.projected_marks(values, off, names, 'test')
    np.testing.assert_array_equal(a, b)
    for lo, hi in zip(off[:-1], off[1:]):
        assert np.all((a[lo:hi] > 0).sum(0) == int(np.ceil(.2 * (hi - lo))))


def test_label_free_api_and_small_group_policy():
    assert list(inspect.signature(f.fit_cell).parameters) == ['x', 'marks', 'groups', 'names', 'off', 'group_key', 'anchor_index', 'namespace']
    x = np.random.default_rng(23).normal(size=(50, 3)); marks = np.where(x > 0, 1, -1)
    for method in ['MOM', 'DS']:
        w, info = f.within_weights(method, x, marks, np.array([0, 1, 1]))
        assert w == [None, None]
        assert all(d['source'] == 'equal_small_group' for d in info['groups'])


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    assert len(tests) == 4
    for test in tests:
        test(); print(test.__name__, 'PASS', flush=True)
    print('4/4 tests PASS', flush=True)
