"""Plan section 17.2 tests for S2, run BEFORE real-data scoring."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from spectral_utils import ssl_s2 as S2
from spectral_utils import ssl_s1 as S1

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); W = MAIN / '.worktrees/readout-quickest-detection-v1'
rng = np.random.default_rng(0)


def test_lag_features_use_only_earlier_tokens():
    z = rng.normal(size=(40, 3)); X = S2.lag_features(z)
    z2 = z.copy(); z2[20] += 5.; X2 = S2.lag_features(z2)
    assert np.array_equal(X[:21], X2[:21])                      # rows <= 20 untouched (row 20 predicts from < 20)
    assert not np.array_equal(X[21:37], X2[21:37]) and np.array_equal(X[37:], X2[37:])
    assert X[0, :-1].sum() == 0 and X[0, -1] == 0 and X[-1, -1] == 1
    assert X[5, 16 * 3:16 * 3 + 5].sum() == 5 and X[5, 16 * 3 + 5:16 * 3 + 16].sum() == 0


def test_ridge_closed_form_matches_augmented_lstsq_and_rejects_labels():
    X = rng.normal(size=(200, 7)); Y = rng.normal(size=(200, 3)); m = S2.fit_ridge(X, Y, alpha=1.0)
    # augmented least squares with unpenalized intercept: [X 1; sqrt(a) I 0]
    A = np.vstack([np.hstack([X, np.ones((200, 1))]), np.hstack([np.eye(7), np.zeros((7, 1))])]); B = np.vstack([Y, np.zeros((7, 3))])
    sol = np.linalg.lstsq(A, B, rcond=None)[0]
    assert np.allclose(sol[:7], m['W'], atol=1e-8) and np.allclose(sol[7], m['b'], atol=1e-8)
    with pytest.raises(ValueError): S2.fit_ridge(X, Y, labels=[1])


def test_dose_zero_returns_base_and_zero_predictor_equals_mean_top5_profile():
    base = rng.normal(size=9); aux = rng.normal(size=9)
    assert np.array_equal(S2.corrected(base, aux, dose=0.), base)
    assert np.allclose(S2.corrected(base, aux), base + .25 * base.std() * (aux - aux.mean()) / aux.std())
    assert np.array_equal(S2.corrected(np.full(9, 2.), aux), np.full(9, 2.))       # constant base -> zero correction
    assert np.all(S2.corrected(base, np.full(9, 1.)) == base)                        # constant aux -> zero correction
    # zero predictor: residual = z_token, so aux = mean over channels of the frozen top5 profile
    spec = importlib.util.spec_from_file_location('cvf_core', W / 'results/step_evidence_v1/source_snapshot/core.py')
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    z = np.load(W.parent / 'token-probability-fusion-v1/results/token_probability_fusion_v1/TOKEN_MATRICES.npz')
    off = np.load(W / 'results/step_evidence_v1/OOF_STEP_SCORES.npz')['offsets']; toff = z['token_offsets']; spans = z['step_spans']; tokens = z['tokens']
    for i in np.random.default_rng(3).choice(len(off) - 1, 5, replace=False):
        a, b = off[i:i+2]; ta, tb = toff[i:i+2]; x = tokens[ta:tb].astype(float)
        zt = S2.robust_standardize_tokens(x); frozen = core.profiles(x, spans[a:b])[:, :, 0]
        assert np.allclose(S2.step_aux(zt, spans[a:b]), frozen.mean(1), atol=1e-12)   # spans are answer-relative


def test_noreset_and_edge_cases():
    z = rng.normal(size=(5, 2)); p = S2.noreset_prediction(z)
    assert np.all(p[0] == 0) and np.allclose(p[1], z[0] / 2) and np.allclose(p[3], z[:3].sum(0) / 4)
    one = S2.robust_standardize_tokens(rng.normal(size=(1, 3))); assert np.all(one == 0)
    const = rng.normal(size=(20, 3)); const[:, 1] = 4.; assert np.all(S2.robust_standardize_tokens(const)[:, 1] == 0)
    assert np.all(S2.std_answer(np.ones(4)) == 0)
    X = S2.lag_features(rng.normal(size=(1, 3))); assert X.shape == (1, 16 * 3 + 17) and X.sum() == 0


def test_hierarchical_sampling_respects_hierarchy_and_is_seeded():
    cells = ['a', 'a', 'a', 'b']; groups = ['g1', 'g1', 'g2', 'g3']; ids = [10, 11, 12, 13]; counts = [100, 1, 50, 5]
    ai, ti = S2.hierarchical_token_sample(np.random.default_rng(1), cells, groups, ids, counts, 4000)
    ai2, _ = S2.hierarchical_token_sample(np.random.default_rng(1), cells, groups, ids, counts, 4000)
    assert np.array_equal(ai, ai2)
    frac_b = (ai == 13).mean(); assert .45 < frac_b < .55                       # cell mass equal, not token mass
    assert (ti[ai == 11] == 0).all() and (ti[ai == 10] < 100).all()
    assert .2 < (ai == 12).mean() < .3                                          # a: 1/2 * 1/2 for g2
