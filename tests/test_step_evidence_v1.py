"""Tests for spectral_utils.step_evidence_v1 (Step 432)."""
import numpy as np
import pytest

from spectral_utils.step_evidence_v1 import (position_bins, fit_tables_flat, evidence_flat, quantile_edges, fit_tables, evidence, evidence_matrix,
                                             pseudo_positive_weights, seed_mass_equal_softmax, seed_agreement,
                                             effective_channels, position_prior_only)


def synthetic(n_answers=400, C=3, seed=0, ramp=0., shift=2.):
    """Answers with a planted first-error step: channel 0 shifts by `shift` at the target and
    carries a linear position ramp of size `ramp` (signal plus drift), channel 1 carries the
    ramp only, channel 2 is pure noise."""
    rng = np.random.default_rng(seed)
    profiles, targets, weights = [], [], []
    for _ in range(n_answers):
        S = int(rng.integers(3, 15))
        t = int(rng.integers(0, S))
        x = rng.normal(size=(S, C))
        x[t, 0] += shift
        x[:, 0] += ramp * np.arange(S) / (S - 1)
        x[:, 1] += ramp * np.arange(S) / (S - 1)
        w = np.zeros(S); w[t] = 1.
        profiles.append(x); targets.append(t); weights.append(w)
    return profiles, np.array(targets), weights


def test_position_bins_edges():
    assert position_bins(1).tolist() == [0]
    b = position_bins(9, B=8)
    assert b[0] == 0 and b[-1] == 7 and np.all(np.diff(b) >= 0)


def test_constant_channel_gives_zero_evidence():
    profiles, targets, weights = synthetic(C=2)
    for p in profiles:
        p[:, 1] = 3.
    tables = fit_tables(profiles, weights)
    assert tables['n_bins'][1] == 1
    m = evidence_matrix(profiles[0], tables)
    assert np.all(m[:, 1] == 0.)
    assert effective_channels(m) == 1


def test_planted_shift_recovered_with_true_positives():
    profiles, targets, weights = synthetic()
    tables = fit_tables(profiles, weights)
    pred = np.array([int(np.argmax(evidence(p, tables))) for p in profiles])
    assert (pred == targets).mean() > 0.7
    # the shifted channel carries a positive log LR in its top bin
    assert tables['log_lr'][0, -1] > 0.5 and abs(tables['log_lr'][2]).mean() < 0.3


def test_position_null_removes_ramp():
    profiles, targets, weights = synthetic(ramp=3., shift=1.5, n_answers=800)
    tables = fit_tables(profiles, weights, B=8)
    plain = np.array([int(np.argmax(evidence(p, tables, conditional=False))) for p in profiles])
    cond = np.array([int(np.argmax(evidence(p, tables, conditional=True))) for p in profiles])
    # The ramp channel's plain null makes late steps look unusual; the conditional null does not.
    late_plain = (plain > targets).mean(); late_cond = (cond > targets).mean()
    assert late_cond < late_plain
    assert (cond == targets).mean() >= (plain == targets).mean()


def test_pseudo_positive_weights():
    w = pseudo_positive_weights(np.array([0.1, 0.5, 0.5, 0.2]), 'argmax')
    assert w.tolist() == [0, 1, 0, 0]
    w = pseudo_positive_weights(np.array([1., 2., 3.]), 'mass')
    assert abs(w.sum() - 1) < 1e-12 and w[0] == 0.
    with pytest.raises(ValueError):
        pseudo_positive_weights(np.zeros(3), 'other')


def test_seed_mass_handles_nonfinite():
    p = np.array([[0., 1.], [1., -np.inf], [2., 0.]])
    m = seed_mass_equal_softmax(p)
    assert np.isfinite(m).all() and abs(m.sum() - 1) < 1e-12 and int(np.argmax(m)) == 2


def test_seed_agreement_and_guard():
    profiles, targets, weights = synthetic(shift=0.)
    seeds = [pseudo_positive_weights(seed_mass_equal_softmax(p)) for p in profiles]
    tables = fit_tables(profiles, seeds)
    seed_pred = np.array([int(np.argmax(s)) for s in seeds])
    ev_pred = np.array([int(np.argmax(evidence(p, tables))) for p in profiles])
    a = seed_agreement(seed_pred, ev_pred)
    assert 0. <= a <= 1.
    assert seed_agreement([1, 2], [1, 3]) == 0.5


def test_position_prior_only_shape():
    profiles, targets, weights = synthetic(ramp=2.)
    tables = fit_tables(profiles, weights)
    v = position_prior_only(7, tables)
    assert v.shape == (7,) and np.isfinite(v).all()


def test_quantile_edges_constant():
    e = quantile_edges(np.full(10, 2.))
    assert len(e) == 2 and e[0] == -np.inf


def test_flat_path_matches_per_answer_path():
    profiles, targets, weights = synthetic(ramp=1., n_answers=120)
    tables = fit_tables(profiles, weights, B=8)
    X = np.concatenate(profiles); w = np.concatenate(weights)
    pbin = np.concatenate([position_bins(len(p), 8) for p in profiles])
    flat = fit_tables_flat(X, w, pbin, np.arange(len(X)), B=8)
    assert np.allclose(flat['log_lr'], tables['log_lr']) and np.allclose(flat['log_lr_pos'], tables['log_lr_pos'])
    ev = np.concatenate([evidence_matrix(p, tables, conditional=True) for p in profiles])
    assert np.allclose(evidence_flat(X, flat, pbin, conditional=True), ev)
    # edges from a training subset only: held-out steps still get a finite evidence
    train = np.arange(len(X) // 2)
    sub = fit_tables_flat(X, w, pbin, train, B=8)
    assert np.isfinite(evidence_flat(X, sub, pbin)).all()
