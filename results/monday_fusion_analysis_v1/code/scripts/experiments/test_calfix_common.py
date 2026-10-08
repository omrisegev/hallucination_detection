"""Known-result tests for calfix_common / calfix_evaluate (constant, short, tied inputs; write-once
bundle; L-SML wrapper == frozen extgen fit; PRMScore formula == official evaluator)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calfix_common import MAIN, ScoreBundle, load_extgen, lsml_fit, tail_marks  # noqa: E402
from calfix_evaluate import prmscore  # noqa: E402


def test_tail_marks():
    off = np.array([0, 5, 6, 16])
    v = np.zeros((16, 3))
    v[0:5, 0] = [3, 3, 1, 0, 0]              # tie at the top, k=1 (ceil(.2*5))
    v[0:5, 1] = 7.0                          # constant column
    v[0:5, 2] = [0, 1, 2, 3, 4]
    v[5, :] = [1, 2, 3]                      # one-step answer
    v[6:16, 0] = [5, 4, 4, 1, 0, 0, 0, 0, 0, 0]   # k=2: 5 marked, the two 4s share one mark
    v[6:16, 1] = np.arange(10); v[6:16, 2] = 1.0
    T, d = tail_marks(v, off, .2, tie_aware=True, centred=False)
    assert np.allclose(T[0:5, 0], [.5, .5, 0, 0, 0]) and np.allclose(T[0:5, 1], .2) and np.allclose(T[0:5, 2], [0, 0, 0, 0, 1])
    assert np.allclose(T[5], 1.0) and np.allclose(T[6:16, 0], [1, .5, .5, 0, 0, 0, 0, 0, 0, 0]) and np.allclose(T[6:16, 2], .2)
    Tc, _ = tail_marks(v, off, .2, tie_aware=True, centred=True)
    assert np.allclose(Tc[0:5, 1], 0) and np.allclose(Tc[5], 0) and np.allclose(Tc[6:16, 2], 0)   # uninformative -> zero
    H, _ = tail_marks(v, off, .2, tie_aware=False, centred=False)
    assert np.allclose(H[0:5, 1], [1, 0, 0, 0, 0])            # historical: constant column marks the FIRST step
    assert d['answer_columns'] == 9 and abs(d['constant_rate'] - 5 / 9) < 1e-12   # a0 col1, all 3 columns of the one-step answer, a2 col2
    return 'tail_marks ok'


class _Pop:
    def __init__(self):
        self.n = 10; self.ns = np.full(10, 2); self.off = np.arange(0, 21, 2); self.total = 20
        self.fold = np.arange(10) % 5; self.groups = np.arange(10)
    def rows(self, m): return np.flatnonzero(np.repeat(m, self.ns))


def test_bundle():
    b = ScoreBundle(_Pop()); s = np.arange(20.0)
    b.put_full('m', 0, s, {'w': [1]})
    caught = False
    try:
        b.put_full('m', 0, s, {'w': [1]})
    except AssertionError as e:
        caught = 'overwrite attempt' in str(e)
    assert caught, 'overwrite not caught'
    for k in range(1, 5):                                   # a complete bundle: every answer written once per role
        b.put_full('m', k, s + k, {'w': [k]})
    assert b.written['m', 'eval'].all() and b.written['m', 'cal'].all()
    assert np.array_equal(b.scores['m', 'eval'], s + np.repeat(np.arange(10) % 5, 2))          # eval score of fold f from model f
    assert np.array_equal(b.scores['m', 'cal'], s + np.repeat((np.arange(10) % 5 - 1) % 5, 2))  # cal score of fold f from model f-1
    return 'bundle write-once and role mapping ok'


def test_lsml_equivalence():
    fz = load_extgen(); rng = np.random.default_rng(0)
    y = rng.standard_normal(6000); X = np.column_stack([y + rng.standard_normal(6000) * s for s in (.5, .8, 1, 1.2, .7, 2.0)])
    X[:, 4] = -X[:, 4]
    a = np.array(fz.fit_weights(X)['weights']); b = lsml_fit(X, 0)['weights']
    assert np.abs(a - b).max() < 1e-12
    c = lsml_fit(X[:, [2, 0, 1, 3, 4, 5]], 1)['weights']      # anchor moved to index 1: same weights, permuted
    assert np.abs(c - b[[2, 0, 1, 3, 4, 5]]).max() < 1e-9
    assert np.allclose(X @ b, sum(X[:, j] * b[j] for j in range(6)))
    return 'lsml wrapper == extgen fit_weights ok'


def test_prmscore():
    sys.path.insert(0, str(MAIN / '.worktrees/depth-feature-fusion-v1'))
    from spectral_utils.prmbench import prmbench_evaluate
    rng = np.random.default_rng(1); preds, metas, c = [], [], np.zeros(4)
    for i in range(300):
        n = rng.integers(1, 12); err = sorted(set(rng.integers(1, n + 1, size=rng.integers(0, 3)).tolist())); lab = rng.integers(0, 2, n)
        preds.append({'idx': f'x_{i}', 'labels': lab.tolist()}); metas.append({'idx': f'x_{i}', 'classification': 'x', 'error_steps': err})
        yv = np.array([j + 1 not in err for j in range(n)]); p = lab.astype(bool)
        c += [np.sum(p & yv), np.sum(p & ~yv), np.sum(~p & ~yv), np.sum(~p & yv)]
    t = prmbench_evaluate(preds, metas)['total']
    assert abs(prmscore(c) - .5 * (t['f1'] + t['negative_f1'])) < 1e-12
    return 'prmscore == official ok'


if __name__ == '__main__':
    for f in (test_tail_marks, test_bundle, test_lsml_equivalence, test_prmscore):
        print(f())
