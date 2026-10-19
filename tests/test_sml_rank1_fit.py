"""Step 467: the zero-diagonal eigenvector used by sml_fuse / sml_fuse_signed is not the rank-one fit of the
off-diagonal covariance; rank1_offdiag_fit is."""
import numpy as np
import pytest
from scipy.linalg import eigh

from spectral_utils.fusion_utils import rank1_offdiag_fit, sml_fuse, sml_fuse_rank1


def offdiag(v):
    R = np.outer(v, v); np.fill_diagonal(R, 0.); return R


def zero_diag_eigvec(R):
    return np.abs(eigh(R)[1][:, -1])


@pytest.mark.parametrize('v', [[0.9, 0.3, 0.3, 0.3], [0.8, 0.6, 0.4, 0.2], [1.0, 0.5, 0.25, 0.1, 0.05]])
def test_exact_recovery_and_eigenvector_shrinkage(v):
    v = np.array(v); R = offdiag(v)
    fit = rank1_offdiag_fit(R)
    assert np.allclose(fit, v, atol=1e-6)
    e = zero_diag_eigvec(R); e /= e.sum(); t = v / v.sum()
    assert e[0] < t[0] - 0.05 and e[-1] > t[-1]          # strongest underweighted, weakest overweighted
    assert np.ptp(e) < np.ptp(t)                          # pulled toward equal weights


def test_equal_accuracies_agree():
    R = offdiag(np.full(5, 0.6))
    e = zero_diag_eigvec(R)
    assert np.allclose(rank1_offdiag_fit(R) / 0.6, 1.) and np.allclose(e / e.sum(), 0.2)


def test_diagonal_is_ignored():
    v = np.array([0.9, 0.5, 0.3, 0.2]); R = offdiag(v)
    noisy = R + np.diag([5., 1., 0.3, 2.])
    assert np.allclose(rank1_offdiag_fit(noisy), rank1_offdiag_fit(R))


def test_nonneg_turns_a_negative_covariance_into_a_zero_weight():
    R = offdiag(np.array([0.8, 0.6, 0.5, 0.4]))
    R[0, 1:] = R[1:, 0] = -np.abs(R[0, 1:])               # view 0 anti-correlated with all the others
    w = rank1_offdiag_fit(R)
    assert np.all(w >= 0) and w[0] == 0. and np.all(w[1:] > 0)


def test_sml_fuse_rank1_on_samples():
    """Conditionally independent views of one label, unequal accuracies: the fit's weights track the truth."""
    rng = np.random.default_rng(0); n = 40000
    y = rng.choice([-1., 1.], n); a = np.array([1.2, 0.6, 0.4, 0.3, 0.2])
    X = [a_i * y + rng.normal(size=n) for a_i in a]
    _, w_fit = sml_fuse_rank1(*X)
    _, w_eig = sml_fuse(*X)
    t = a / a.sum()
    assert np.abs(w_fit - t).max() < 0.03 < np.abs(w_eig - t).max()
