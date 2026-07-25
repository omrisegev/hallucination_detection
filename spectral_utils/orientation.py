"""
orientation.py — Extension H (Phase 1): Prior-Free Orientation Module

Eliminates the `epr` anchor prior from L-SML fusion by replacing it with:
1. `z2_sign_recovery(V)`: Spectral relaxation of pairwise-sign consensus (Z2 synchronization)
   recovering per-feature relative signs s ∈ {±1}^p from the sign pattern of feature correlations.
2. `distributional_orient(score)`: Feature-free global ±1 orientation tiebreaker using score skewness
   to place the minority (hallucination) mode on the low side without any external anchor view.
"""

import numpy as np
from scipy import stats

_EPS = 1e-12


def z2_sign_recovery(V):
    """Recover relative feature signs s ∈ {±1}^p via Z2 synchronization.

    Computes the leading eigenvector of the pairwise-sign correlation matrix:
      S_ij = sign(corr(V_i, V_j))
    The leading eigenvector v1 gives the continuous relaxation of s_i * s_j ≈ S_ij.
    Taking sign(v1) recovers the discrete relative sign assignment s ∈ {±1}^p.

    Parameters
    ----------
    V : array-like of shape (n_samples, n_features)

    Returns
    -------
    signs : ndarray of shape (n_features,) with entries in {+1.0, -1.0}
    """
    V = np.asarray(V, dtype=np.float64)
    n, p = V.shape
    if p < 2:
        return np.ones(p, dtype=np.float64)

    # Compute correlation matrix
    cov = np.cov(V, rowvar=False)
    std = np.sqrt(np.diag(cov)) + _EPS
    corr = cov / np.outer(std, std)

    # Pairwise sign matrix
    S = np.sign(corr)
    # Ensure diagonal is 1.0
    np.fill_diagonal(S, 1.0)

    # Spectral relaxation: leading eigenvector of S
    w, v = np.linalg.eigh(S)
    leading_v = v[:, -1]

    # Convert to discrete signs in {+1, -1}
    signs = np.sign(leading_v)
    signs[signs == 0] = 1.0
    return signs


def distributional_orient(score):
    """Orient a continuous consensus score without using any external anchor.

    In hallucination detection benchmark datasets, hallucinations represent the
    minority mode (lower frequency than correct outputs). Hallucinations should
    produce low consensus scores. 

    If skewness is positive (right-skewed), the long tail contains the minority mode
    on the high end, so we flip the sign (-score) to place the minority mode on the
    low end. If skewness is negative (left-skewed), the minority mode is already on
    the low end.

    Parameters
    ----------
    score : array-like of shape (n_samples,)

    Returns
    -------
    oriented_score : ndarray of shape (n_samples,)
    flipped : bool
        True if the score vector was negated to enforce low minority mode.
    """
    score = np.asarray(score, dtype=np.float64)
    if len(score) < 3:
        return score.copy(), False

    sk = float(stats.skew(score))
    # If right-skewed (sk > 0), flip so the minority tail is at the bottom
    flipped = bool(sk > 0.0)
    oriented_score = -score if flipped else score
    return oriented_score, flipped
