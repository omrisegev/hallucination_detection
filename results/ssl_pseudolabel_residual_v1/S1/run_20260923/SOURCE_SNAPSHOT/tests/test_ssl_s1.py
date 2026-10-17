"""Plan section 17.2 tests for S1, run BEFORE real-data scoring."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.special import softmax, expit
from spectral_utils import ssl_s1 as S

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection')
W = MAIN / '.worktrees/readout-quickest-detection-v1'
rng = np.random.default_rng(0)


def test_pb_target_is_a_distribution_and_prm_target_is_per_step():
    Z = S.answer_z(rng.normal(size=(6, 11)))
    for arm in ['P_HARD', 'P_SOFT', 'P_RANDOM', 'P_AGREE']:
        t, w, keep, _ = S.targets_pb(Z, arm, rng=np.random.default_rng(1))
        assert t.shape == (6,) and abs(t.sum() - 1) < 1e-12 and (t >= 0).all()
        tp, m, sel, _ = S.targets_prm(Z, arm, rng=np.random.default_rng(1))
        assert tp.shape == (6,) and (tp >= 0).all() and (tp <= 1).all() and m.shape == (6,) and sel.dtype == bool
    # PRMB hard targets may carry several positives; PB hard target exactly one
    Zm = S.answer_z(np.column_stack([np.array([3, 3, -3, 3, -3, -3.])] * 11) + rng.normal(scale=.01, size=(6, 11)))
    assert S.targets_prm(Zm, 'P_HARD')[0].sum() >= 2 and S.targets_pb(Zm, 'P_HARD')[0].sum() == 1


def test_softmax_is_per_channel_and_P_differs_from_Z():
    P = rng.normal(size=(5, 11)); Z = S.answer_z(P)
    q = S.teacher_pb(Z)
    assert np.allclose(q, softmax(Z, axis=0).mean(1)) and not np.allclose(q, softmax(Z.mean(1)))
    assert np.allclose(S.teacher_prm(Z), expit(Z).mean(1))
    assert not np.allclose(S.teacher_pb(P), S.teacher_pb(Z))          # P vs Z distinguished
    assert np.allclose(S.answer_z(P * 3 + 7), Z)                       # affine per channel invariance


def test_top5_reproduces_frozen_profiles():
    spec = importlib.util.spec_from_file_location('cvf_core', W / 'results/step_evidence_v1/source_snapshot/core.py')
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    z = np.load(W.parent / 'token-probability-fusion-v1/results/token_probability_fusion_v1/TOKEN_MATRICES.npz')
    prof = np.load(W / 'results/step_evidence_v1/profiles_full.npy', mmap_mode='r')
    off = np.load(W / 'results/step_evidence_v1/OOF_STEP_SCORES.npz')['offsets']; toff = z['token_offsets']; spans = z['step_spans']; tokens = z['tokens']
    for i in np.random.default_rng(5).choice(len(off) - 1, 6, replace=False):
        a, b = off[i:i+2]; ta, tb = toff[i:i+2]
        got = core.profiles(tokens[ta:tb].astype(float), spans[a:b])[:, :, 0]
        assert np.array_equal(got, np.asarray(prof[a:b, :, 0])), i


def test_one_step_and_constant_channel_edge_cases():
    Z1 = S.answer_z(rng.normal(size=(1, 11)))
    assert np.all(Z1 == 0) and S.entropy_conf(S.teacher_pb(Z1)) == 0.
    t, w, keep, info = S.targets_pb(Z1, 'P_AGREE'); assert not keep and w == 0.
    P = rng.normal(size=(7, 11)); P[:, 3] = 2.5
    assert np.all(S.answer_z(P)[:, 3] == 0)
    tp, m, sel, _ = S.targets_prm(S.answer_z(P), 'P_AGREE'); assert sel.dtype == bool and m.shape == (7,)


def test_fit_rejects_labels_and_gradients_match_finite_differences():
    X = [S.answer_z(rng.normal(size=(s, 11))) for s in [4, 7, 1, 9]]
    T = [S.teacher_pb(x) for x in X]; w = S.answer_weights(['a', 'a', 'b', 'b'], ['g1', 'g2', 'g3', 'g3'], 'pb_q4')
    with pytest.raises(ValueError): S.fit_head(X, T, w, 'pb', labels=[1, 2, 3, 4])
    with pytest.raises(ValueError): S.fit_head(X, T, w, 'prm', error_steps=[[1]])
    for task in ['pb', 'prm']:
        Tt = T if task == 'pb' else [S.teacher_prm(x) for x in X]
        M = None if task == 'pb' else [np.abs(2 * t - 1) for t in Tt]
        # finite-difference gradient check via the closure used inside fit_head
        d = 11 + (1 if task == 'prm' else 0); theta = rng.normal(scale=.3, size=d)
        import types
        cap = {}
        orig = S.minimize
        def fake_minimize(fun, x0, **kw):
            cap['fun'] = fun; return types.SimpleNamespace(x=x0, success=True, status=0, message='captured', nit=0, fun=0., jac=np.zeros_like(x0))
        S.minimize = fake_minimize
        try: S.fit_head(X, Tt, w, task, M_list=M)
        finally: S.minimize = orig
        f0, g = cap['fun'](theta); eps = 1e-6; num = np.zeros(d)
        for j in range(d):
            e = np.zeros(d); e[j] = eps; num[j] = (cap['fun'](theta + e)[0] - cap['fun'](theta - e)[0]) / (2 * eps)
        assert np.allclose(num, g, atol=1e-6), (task, num, g)


def test_fit_converges_on_separable_fixture_and_one_step_answer_is_inert():
    Xs = [S.answer_z(rng.normal(size=(s, 11))) for s in [6, 8, 5, 1, 9, 7]]
    T = []
    for x in Xs:
        t = np.zeros(len(x)); t[S.first_argmax(x[:, 0])] = 1; T.append(t)   # channel 0 peak = target
    w = S.answer_weights(['a'] * 6, ['g%d' % i for i in range(6)], 'pb_q4')
    m = S.fit_head(Xs, T, w, 'pb')
    assert m['w'][0] > 0 and m['contributing_answers'] == 5 and m['b'] == 0.
    pred = [S.first_argmax(S.predict_head(x, m)) for x in Xs]
    assert sum(p == S.first_argmax(t) for p, t in zip(pred, T)) >= 5


def test_metric_fixtures():
    assert S.within_auc([0, 1, 1, 0], [1, 3, 2, 0]) == 1.
    assert S.within_auc([0, 1, 0], [1, 1, 0]) == .75                      # tie credited 0.5
    assert np.isnan(S.within_auc([0, 0], [1, 2]))
    assert S.first_argmax([1, 3, 3, 2]) == 1                               # earliest tie
    assert S.unique_peak([1, 3, 3, 2]) is None and S.unique_peak([1, 3, 2]) == 1


def test_coverage_match_is_deterministic_and_bin_exact():
    from collections import Counter
    cands = [(('cellA', '2-5', '[0,.2)'), S.coverage_key(f'u{i}', 'answer'), f'u{i}') for i in range(10)] + [(('cellB', '11+', '[.8,1]'), S.coverage_key(f'v{i}', 'answer'), f'v{i}') for i in range(4)]
    kept = Counter({('cellA', '2-5', '[0,.2)'): 3, ('cellB', '11+', '[.8,1]'): 9})
    sel = S.coverage_match(kept, cands); sel2 = S.coverage_match(kept, list(reversed(cands)))
    assert sel == sel2 and sum(s.startswith('u') for s in sel) == 3 and sum(s.startswith('v') for s in sel) == 4


def test_answer_weights_equal_mass():
    w = S.answer_weights(['a', 'a', 'a', 'b'], ['g1', 'g1', 'g2', 'g3'], 'pb_q4')
    assert abs(w.sum() - 1) < 1e-12 and abs(w[:3].sum() - .5) < 1e-12 and abs(w[0] - w[1]) < 1e-12 and abs(w[2] - .25) < 1e-12
