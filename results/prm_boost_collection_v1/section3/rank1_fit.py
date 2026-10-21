"""Verbatim copy of spectral_utils.fusion_utils.rank1_offdiag_fit (repo commit 3927d562c); the AIRCC mirror's fusion_utils predates it."""
import numpy as np
from scipy.linalg import eigh


def rank1_offdiag_fit(R, nonneg: bool = True, iters: int = 500, tol: float = 1e-10):
    """Least-squares rank-one fit of the OFF-diagonal covariance: argmin_v sum_{i!=j} (r_ij - v_i v_j)^2.

    Under the SML / L-SML model (Parisi et al. 2014, Lemma 1; Jaffe et al. 2016, Lemma 1) only the
    off-diagonal entries satisfy r_ij = v_i v_j; the diagonal holds the views' own variance and carries
    no information about v. ``sml_fuse`` / ``sml_fuse_signed`` instead zero the diagonal and take the
    leading eigenvector of R_off = v v^T - diag(v^2). That eigenvector equals v only when all v_i are
    equal; otherwise it is pulled toward equal weights (Step 467 [SML weight estimator]):

        true v [0.9, 0.3, 0.3, 0.3] -> zero-diagonal eigenvector [0.32, 0.23, 0.23, 0.23] (L1-normalized),
        this fit [0.50, 0.17, 0.17, 0.17], which is exact.

    EMPIRICAL RESULT, read before using it: on the development benchmarks the unbiased fit LOSES
    (results/sml_weight_estimator_v1/, Step 467). Telemetry-only L-SML on the nested banks, old template:
    N16 ProcessBench argmax hit 41.61 -> 38.61, PRMBench within-AUC .7958 -> .7762, and worse on almost every
    bank. Residual dependence between L-SML groups inflates one loading; the exact fit then puts up to 0.89
    of the weight on one group, while the eigenvector's shrinkage toward equal weights acts as a regularizer.

    Projected coordinate descent from |leading eigenvector| * sqrt(eigenvalue). With ``nonneg`` the fit is
    constrained to v >= 0, the L-SML paper's assumption that every (pre-oriented) view beats random
    (Jaffe et al. 2016, Lemma 4). A negative r_ij between two pre-oriented views then cannot be fitted
    and is a model-violation signal, not an inverted view.

    Args:
        R: square covariance matrix (its diagonal is ignored).
        nonneg: constrain v >= 0 (default). False gives the unconstrained fit, sign fixed by majority.
        iters, tol: coordinate-descent budget and convergence tolerance on max |change|.

    Returns:
        v (np.ndarray, length k), not normalized.
    """
    R = np.atleast_2d(np.asarray(R, dtype=float))
    k = R.shape[0]
    R_off = R - np.diag(np.diag(R))
    lam, vecs = eigh(R_off)
    v = vecs[:, -1] * np.sqrt(max(float(lam[-1]), 1e-12))
    if np.sum(v > 0) < k / 2:
        v = -v
    if nonneg:
        v = np.abs(v)
    off = ~np.eye(k, dtype=bool)
    for _ in range(iters):
        old = v.copy()
        for i in range(k):
            m = off[i]
            den = float(v[m] @ v[m])
            vi = float(R_off[i, m] @ v[m]) / den if den > 1e-15 else 0.
            v[i] = max(0., vi) if nonneg else vi
        if np.max(np.abs(v - old)) < tol:
            break
    return v
