"""Model-inverse head rejects a factor-independent block without renormalizing it."""
import numpy as np
from spectral_utils.joint_lsml import regularized_joint_map_weights


def test_independent_noise_block_has_zero_weight():
    v = np.array([.6, .5, .4, .3, .2, .1])
    c = np.outer(v, v) + np.diag(np.linspace(.4, .8, len(v)))
    w, _ = regularized_joint_map_weights(None, c, v, mode='liu', lam=0., target_condition=1000.)
    augmented = np.zeros((9, 9)); augmented[:6, :6] = c; augmented[6:, 6:] = np.eye(3)
    wa, _ = regularized_joint_map_weights(None, augmented, np.r_[v, np.zeros(3)],
        mode='liu', lam=0., target_condition=1000.)
    np.testing.assert_allclose(wa[:6], w, atol=1e-12)
    np.testing.assert_array_equal(wa[6:], np.zeros(3))
    np.testing.assert_allclose(c @ w, v, atol=1e-12)


def test_nonzero_noise_loading_is_not_normalized_to_unit_influence():
    # Exact factor model, no finite-sample fitting claim. A weak additional
    # global measurement should have influence proportional to its loading.
    v = np.array([.6, .5, .4, 1e-5])
    c = np.outer(v, v) + np.eye(4)
    w, _ = regularized_joint_map_weights(None, c, v, mode='liu', lam=0., target_condition=1000.)
    np.testing.assert_allclose(w, v/(1+v@v), atol=1e-12)
    assert abs(w[-1])/np.abs(w).sum() < 1e-4
