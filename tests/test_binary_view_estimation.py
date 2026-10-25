import numpy as np
from spectral_utils.binary_view_estimation import (spectral_binary, dawid_skene_binary, rank_one_misfit,
                                                   tci_residuals, triad_lambda, factor_information,
                                                   anchored_two_factor)


def _votes(n=60000, prev=.2, psi=(.9, .7, .6, .65, .8, .55), eta=(.85, .75, .7, .6, .9, .65), seed=0):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < prev
    psi, eta = np.array(psi), np.array(eta)
    p_flag = np.where(y[:, None], psi, 1 - eta)
    V = np.where(rng.random((n, len(psi))) < p_flag, 1., -1.)
    return V, y, psi, eta


def test_spectral_recovers_ci_parameters():
    V, y, psi, eta = _votes()
    r = spectral_binary(V)
    assert abs(r['prevalence'] - .2) < .02
    assert np.max(np.abs(r['psi'] - psi)) < .04 and np.max(np.abs(r['eta'] - eta)) < .03


def test_dawid_skene_recovers_ci_parameters():
    V, y, psi, eta = _votes(seed=1)
    r = dawid_skene_binary(V)
    assert abs(r['prevalence'] - .2) < .02
    assert np.max(np.abs(r['psi'] - psi)) < .03 and np.max(np.abs(r['eta'] - eta)) < .03


def test_ds_label_switch_resolved_by_anchor():
    V, y, psi, eta = _votes(seed=2)
    r = dawid_skene_binary(-V[:, ::-1] * 1.0 * -1)     # same votes, columns reversed: anchor is now the weakest-psi view
    assert r['psi'][0] + r['eta'][0] > 1


def test_ci_diagnostics_detect_a_copy():
    V, *_ = _votes(seed=3)
    ok = spectral_binary(V)
    Vc = np.column_stack([V, V[:, 2]])                  # an exact copy of view 2 breaks CI
    bad = spectral_binary(Vc)
    assert rank_one_misfit(ok['S'], ok['u'])['overall'] < .01
    assert rank_one_misfit(bad['S'], bad['u'])['overall'] > 5 * rank_one_misfit(ok['S'], ok['u'])['overall']
    assert np.nanmax(tci_residuals(ok['S'], ok['T'])) < .25
    assert np.nanmax(tci_residuals(bad['S'], bad['T'])) > np.nanmax(tci_residuals(ok['S'], ok['T']))


def test_two_factor_anchor_and_information():
    rng = np.random.default_rng(4); n, m = 80000, 6
    f = rng.standard_normal(n); z = rng.standard_normal(n)
    u = np.array([.7, .5, .4, .3, .2, .1]); lam = np.array([0., .6, .6, .5, .5, .4])
    X = f[:, None] * u + z[:, None] * lam + rng.standard_normal((n, m)) * .6
    uh, lh, psih, info = anchored_two_factor(np.cov(X.T), anchor=0)
    assert np.max(np.abs(uh - u)) < .03 and np.max(np.abs(lh - lam)) < .03
    assert np.max(np.abs(psih - .36)) < .03 and info['rel_misfit'] < 1e-3
    psi = np.full(m, .36)
    I_all = factor_information(u, lam, psi); I_p = u[0] ** 2 / psi[0]
    assert I_all > I_p
    med, iqr = triad_lambda(np.cov(X.T), 0)
    assert np.all(iqr[1:] > 0)                          # nuisance makes one-factor triads disagree
