import itertools
import numpy as np
from scipy.special import expit

from spectral_utils.lsml_group_confidence import (
    binary_tail, collapse_duplicates, fit_binary_tree, group_contributions,
    linearized_weights, score,
)


def fixture_model():
    return {'pi': .3, 'a': [[.15, .8], [.25, .75], [.1, .85]],
            'theta': [[.2, .85], [.15, .8], [0, 1], [0, 1]],
            'groups': [0, 0, 1, 2], 'mu': [.4]*4, 'sd': [.49]*4}


def sample(model, n, seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < model['pi']).astype(int)
    a, theta, groups = map(np.asarray, (model['a'], model['theta'], model['groups']))
    alpha = rng.random((n, len(a))) < a[:, y].T
    b = rng.random((n, len(groups))) < theta[np.arange(len(groups)), alpha[:, groups].astype(int)]
    return b.astype(np.uint8), y.astype(int)


def test_exact_latent_enumeration_and_binary_continuation():
    m = fixture_model()
    b = np.array(list(itertools.product([0, 1], repeat=4)))
    a, theta = np.array(m['a']), np.array(m['theta'])
    likelihood = np.zeros((len(b), 2))
    for j, row in enumerate(b):
        for y in (0, 1):
            for alpha in itertools.product([0, 1], repeat=3):
                prob = np.prod([a[g, y] if alpha[g] else 1-a[g, y] for g in range(3)])
                prob *= np.prod([theta[i, alpha[g]] if row[i] else 1-theta[i, alpha[g]]
                                 for i, g in enumerate(m['groups'])])
                likelihood[j, y] += prob
    np.testing.assert_allclose(likelihood.sum(0), 1)
    odds = np.log(likelihood[:, 1]/likelihood[:, 0]) + np.log(m['pi']/(1-m['pi']))
    np.testing.assert_allclose(score(b, m, continuous=False), odds, atol=1e-12)
    z = (b-np.array(m['mu']))/np.array(m['sd'])
    np.testing.assert_allclose(score(z, m), odds, atol=1e-12)


def test_linearization_and_saturation():
    m = fixture_model()
    h = 1e-6
    finite_difference = (score(np.eye(4)*h, m)-score(-np.eye(4)*h, m))/(2*h)
    np.testing.assert_allclose(linearized_weights(m), finite_difference, atol=1e-8)
    c = group_contributions(np.full((2, 4), [[-1e6], [1e6]]), m)
    assert np.isfinite(c).all()
    np.testing.assert_allclose(c[:, 0], [np.log(.2/.85), np.log(.8/.15)], atol=1e-8)


def test_ties_and_constant_do_not_create_votes():
    x = np.array([[1., 0.], [1., 0.], [1., 2.], [1., 2.]])
    b = binary_tail(x, [0, 4])
    assert not b.any()  # Strict empirical quantile threshold, no temporal tie break.


def test_duplicate_invariance_including_binary_equivalent_continuous_columns():
    m = fixture_model()
    b, _ = sample(m, 2000, 8)
    x = b.astype(float)
    x[:, 1] += np.linspace(0, .1, len(x))
    bg, e, g, _ = collapse_duplicates(b, x, m['groups'])
    bb = np.column_stack([b, b[:, 0], b[:, 0]])
    xx = np.column_stack([x, x[:, 0], x[:, 0]])
    bd, ed, gd, _ = collapse_duplicates(bb, xx, m['groups']+[0, 0])
    np.testing.assert_array_equal(bg, bd)
    np.testing.assert_array_equal(g, gd)
    np.testing.assert_allclose(x@e, xx@ed, atol=1e-15)
    fitted = fit_binary_tree(bg, g)
    repeated = fit_binary_tree(bd, gd)
    assert fitted['converged'] and repeated['converged']
    np.testing.assert_allclose(score(x@e, fitted), score(xx@ed, repeated), atol=1e-10)


def test_extra_independent_measurement_improves_heldout_log_loss():
    # Four groups; first group contains three independent noisy measurements of
    # its latent state. The added column is NOT a clone and shares the same alpha.
    m = {'pi': .35, 'a': [[.15, .85]]*4,
         'theta': [[.2, .8]]*3 + [[0, 1]]*3,
         'groups': [0, 0, 0, 1, 2, 3], 'mu': [.4]*6, 'sd': [.49]*6}
    train, _ = sample(m, 30000, 40)
    test, y = sample(m, 30000, 41)
    full = fit_binary_tree(train, m['groups'])
    subset = [0, 3, 4, 5]
    small = fit_binary_tree(train[:, subset], [0, 1, 2, 3])
    assert full['converged'] and small['converged']
    assert np.min(np.diff(full['objective'])) > -1e-4
    def loss(logodds):
        return np.mean(np.logaddexp(0, logodds)-y*logodds)
    assert loss(score(test, full, continuous=False)) < loss(score(test[:, subset], small, continuous=False))-.005
    # Same learned parameters; evidence averaging is a genuinely different arm.
    assert not np.allclose(score(test, full, continuous=False),
                           score(test, full, continuous=False, average_evidence=True))
