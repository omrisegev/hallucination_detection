"""Dependency-aware extensions of U-PCR for the sparse-error experiment.

This module implements the *weights-channel* experiment described in
``SPEC_DEPENDENCY_FUSION_EXPERIMENT.md``.  It deliberately keeps two objects
separate:

``w_pcr``
    A reproduction of the sparse-error U-PCR (SU-PCR) mechanism in Tenzer et
    al., *Crowdsourcing Regression: A Spectral Approach* (AISTATS 2022): split
    the observed covariance into a rank-two part and a sparse error part,
    recover ``rho`` from the cleaned rank-two matrix, and use two-component
    PCR weights.

``w_structured``
    Our tailored SDSF / dependency-weighted arm: use the *same* decomposition
    and the *same* ``rho`` estimate, but solve a condition-controlled ridge
    system with the denoised structured covariance.  Therefore
    ``w_structured - w_pcr`` isolates the proposed weighting change instead of
    mixing it with a new feature selector or a different reliability estimate.

The paper describes using the projected-gradient low-rank-plus-sparse algorithm
of Cherapanamjeri, Gupta & Jain (2017), on the observed off-diagonal entries.
No authors' implementation was published with the U-PCR paper.  The routine
below is an auditable small-matrix reproduction: it uses alternating rank-two
completion and sparse hard thresholding with the two parameter rules specified
in the U-PCR supplement.  It does not claim byte-for-byte equivalence to the
authors' unreported PG-RMC implementation.  Experiment outputs therefore call
this arm ``su_pcr_reproduction`` rather than ``official_su_pcr``.

References
----------
Tenzer et al. (2022):
https://proceedings.mlr.press/v151/tenzer22a/tenzer22a.pdf

Cherapanamjeri et al. (2017):
https://arxiv.org/abs/1606.07315
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

from .upcr import additive_design, solve_additive

__all__ = [
    "SparseDecomposition",
    "SparseUPCRResult",
    "projected_sparse_decomposition",
    "regularized_covariance_weights",
    "sparse_upcr_fit",
]


def _symmetrize(a):
    a = np.asarray(a, dtype=float)
    return 0.5 * (a + a.T)


def _rank_projection_symmetric(a, rank):
    """Best symmetric rank-``rank`` approximation in Frobenius norm.

    The U-PCR low-rank component can be indefinite (rank at most two), so the
    eigenvalues are selected by *magnitude*, not by keeping only the largest
    positive eigenvalues as ordinary PCA would do.
    """
    a = _symmetrize(a)
    vals, vecs = eigh(a)
    take = np.argsort(np.abs(vals))[-min(int(rank), len(vals)):]
    return _symmetrize((vecs[:, take] * vals[take]) @ vecs[:, take].T)


def _complete_rank_from_mask(target, observed_mask, rank=2, n_inner=40,
                             initial=None):
    """Complete arbitrary missing entries while projecting to symmetric rank r."""
    target = _symmetrize(target)
    m = target.shape[0]
    mask = np.asarray(observed_mask, dtype=bool)
    if mask.shape != target.shape or not np.array_equal(mask, mask.T):
        raise ValueError("observed_mask must be symmetric and match target")
    x = np.zeros_like(target)
    if initial is None:
        for i in range(m):
            row = target[i, mask[i]]
            x[i, ~mask[i]] = float(np.median(row)) if row.size else 0.0
    else:
        x[~mask] = np.asarray(initial, dtype=float)[~mask]
    x[mask] = target[mask]
    x = _symmetrize(x)

    low = _rank_projection_symmetric(x, rank)
    for _ in range(max(1, int(n_inner))):
        old = low.copy()
        x = low.copy()
        x[mask] = target[mask]
        low = _rank_projection_symmetric(_symmetrize(x), rank)
        if (np.linalg.norm(low - old, ord="fro")
                / (np.linalg.norm(old, ord="fro") + 1e-12)) < 1e-10:
            break
    return _symmetrize(low)


def _complete_rank_from_offdiag(target, rank=2, n_inner=40, initial=None):
    """Complete the missing diagonal while projecting onto symmetric rank r."""
    mask = ~np.eye(np.asarray(target).shape[0], dtype=bool)
    return _complete_rank_from_mask(
        target, mask, rank=rank, n_inner=n_inner, initial=initial,
    )


def _global_support_cap(sparse, max_fraction):
    """Optionally keep only the largest off-diagonal entries.

    The published reproduction leaves ``max_fraction=None`` and relies on the
    projected-gradient threshold.  A cap exists only for planted-world stress
    tests and must not be selected using answer labels.
    """
    if max_fraction is None:
        return sparse
    m = sparse.shape[0]
    iu = np.triu_indices(m, 1)
    total = len(iu[0])
    k = int(np.ceil(float(max_fraction) * total))
    if k <= 0:
        return np.zeros_like(sparse)
    if k >= total:
        return sparse
    values = np.abs(sparse[iu])
    keep = np.argpartition(values, -k)[-k:]
    out = np.zeros_like(sparse)
    out[iu[0][keep], iu[1][keep]] = sparse[iu[0][keep], iu[1][keep]]
    return out + out.T


@dataclass
class SparseDecomposition:
    """Low-rank-plus-sparse decomposition of an observed covariance."""

    low_rank: np.ndarray
    sparse: np.ndarray
    structured_cov: np.ndarray
    residual: np.ndarray
    support: np.ndarray
    converged: bool
    n_iter: int
    mu: float
    threshold: float
    sparse_fraction: float
    relative_residual: float
    theorem_support_ok: bool
    meta: dict = field(default_factory=dict)


def projected_sparse_decomposition(
    C,
    *,
    rank=2,
    threshold_multiplier=1.0,
    max_iter=100,
    inner_completion_iter=40,
    tol=1e-8,
    max_sparse_fraction=None,
    fixed_support=None,
):
    """Decompose the *off-diagonal* covariance as ``C = L + S + R``.

    ``L`` is symmetric rank at most two and ``S`` is selected by the threshold
    specified in Appendix B of Tenzer et al.  The observed diagonal is never
    treated as sparse: after the off-diagonal fit it is copied verbatim into the
    structured covariance.  This matches the fact that SU-PCR's sparsity
    condition concerns correlated *pairs* of errors, not individual variances.

    Parameters
    ----------
    C:
        Square empirical covariance matrix.
    threshold_multiplier:
        Multiplies the paper threshold.  The registered primary is exactly 1.0.
    max_sparse_fraction:
        Optional planted-world safety cap.  It is ``None`` in the registered
        real-data experiment and must not be tuned with correctness labels.
    fixed_support:
        Optional symmetric boolean off-diagonal support.  When supplied, the
        sparse-error topology is held fixed and only its values and the
        complementary rank-two completion are estimated.  This is used by
        separately preregistered, target-free support learners; the default
        ``None`` preserves the published SU-PCR reproduction exactly.
    """
    C = _symmetrize(C)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix")
    if not np.isfinite(C).all():
        raise ValueError("C contains non-finite values")
    m = C.shape[0]
    if m < 3:
        raise ValueError("low-rank+sparse U-PCR needs at least 3 features")

    prescribed_support = None
    if fixed_support is not None:
        prescribed_support = np.asarray(fixed_support, dtype=bool)
        if (
            prescribed_support.shape != C.shape
            or not np.array_equal(prescribed_support, prescribed_support.T)
            or np.any(np.diag(prescribed_support))
        ):
            raise ValueError(
                "fixed_support must be a symmetric zero-diagonal boolean mask"
            )
        if max_sparse_fraction is not None:
            raise ValueError(
                "max_sparse_fraction cannot be combined with fixed_support"
            )

    observed = C.copy()
    np.fill_diagonal(observed, 0.0)
    mask = ~np.eye(m, dtype=bool)

    # Appendix B: mu approximated from the leading rank-r eigenspace of C.
    # C is a covariance matrix, so the paper's leading eigenvectors are the
    # largest algebraic eigenvectors (not magnitude-sorted eigenvectors of an
    # intermediate, potentially indefinite residual).
    _, vecs = eigh(C, subset_by_index=[m - min(int(rank), m), m - 1])
    row_norm = np.linalg.norm(vecs, axis=1)
    mu = float(np.sqrt(m / max(int(rank), 1)) * np.max(row_norm))

    sparse = np.zeros_like(C)
    # PG-RMC updates the sparse residual from the *current* low-rank iterate,
    # then takes the projected low-rank step.  Starting from L=0 is important:
    # reversing these two operations lets the first rank projection absorb the
    # very corruptions that hard thresholding is meant to expose.
    low = np.zeros_like(C)
    converged = False
    threshold = float("nan")
    n_done = 0
    for it in range(1, max(1, int(max_iter)) + 1):
        work = observed - sparse
        residual = observed - low
        np.fill_diagonal(residual, 0.0)
        if prescribed_support is None:
            threshold = (float(threshold_multiplier) * mu
                         * float(np.linalg.norm(work, ord=2)) / m)
            support_new = (np.abs(residual) >= threshold) & mask
        else:
            threshold = float("nan")
            support_new = prescribed_support

        # Once an entry is classified as corrupted it becomes missing for the
        # low-rank completion step.  Merely subtracting the current residual and
        # then treating the resulting value as observed creates a feedback drift
        # on stable supports; PG-RMC's sparse cleanup is meant to remove those
        # entries from the low-rank gradient objective.
        low_new = _complete_rank_from_mask(
            observed,
            mask & ~support_new,
            rank=rank,
            n_inner=inner_completion_iter,
            initial=None,
        )
        sparse_new = np.where(support_new, observed - low_new, 0.0)
        np.fill_diagonal(sparse_new, 0.0)
        sparse_new = _global_support_cap(_symmetrize(sparse_new), max_sparse_fraction)
        capped_support = np.abs(sparse_new) > 0
        if not np.array_equal(capped_support, support_new):
            low_new = _complete_rank_from_mask(
                observed, mask & ~capped_support, rank=rank,
                n_inner=inner_completion_iter, initial=None,
            )
            sparse_new = np.where(capped_support, observed - low_new, 0.0)

        scale = max(float(np.linalg.norm(observed, ord="fro")), 1e-12)
        shift = float(np.linalg.norm(low_new - low, ord="fro"))
        shift += float(np.linalg.norm(sparse_new - sparse, ord="fro"))
        low, sparse = low_new, sparse_new
        n_done = it
        if shift / scale < tol:
            converged = True
            break

    residual = observed - low - sparse
    np.fill_diagonal(residual, 0.0)
    structured = low + sparse
    np.fill_diagonal(structured, np.diag(C))
    structured = _symmetrize(structured)

    iu = np.triu_indices(m, 1)
    support = np.abs(sparse) > 0
    nnz = int(np.sum(support[iu]))
    n_pairs = len(iu[0])
    relative_residual = float(
        np.linalg.norm(residual[iu]) / (np.linalg.norm(observed[iu]) + 1e-12)
    )
    return SparseDecomposition(
        low_rank=low,
        sparse=sparse,
        structured_cov=structured,
        residual=residual,
        support=support,
        converged=converged,
        n_iter=n_done,
        mu=mu,
        threshold=float(threshold),
        sparse_fraction=float(nnz / max(n_pairs, 1)),
        relative_residual=relative_residual,
        theorem_support_ok=bool(m >= 5 and nnz < (m - 1) / 2),
        meta={"rank": int(rank), "nnz_pairs": nnz, "n_pairs": n_pairs,
              "threshold_multiplier": float(threshold_multiplier),
              "max_sparse_fraction": max_sparse_fraction,
              "fixed_support": prescribed_support is not None},
    )


def _nearest_psd(a):
    """Symmetric positive-semidefinite projection by eigenvalue clipping."""
    a = _symmetrize(a)
    vals, vecs = eigh(a)
    vals_clipped = np.maximum(vals, 0.0)
    psd = (vecs * vals_clipped) @ vecs.T
    return _symmetrize(psd), vals


def regularized_covariance_weights(C, rho, *, target_condition=100.0):
    """Solve a label-free, condition-controlled approximation to ``C w=rho``.

    The smallest ridge value that makes the PSD-projected covariance's condition
    number no larger than ``target_condition`` is chosen analytically.  No label,
    AUROC, or per-cell grid search is used.

    Returns ``(w, diagnostics)``.
    """
    C = _symmetrize(C)
    rho = np.asarray(rho, dtype=float)
    if C.shape != (rho.size, rho.size):
        raise ValueError("C and rho dimensions disagree")
    if target_condition <= 1:
        raise ValueError("target_condition must be > 1")

    psd, raw_evals = _nearest_psd(C)
    evals = eigh(psd, eigvals_only=True)
    lo, hi = float(evals[0]), float(evals[-1])
    if hi <= 1e-14:
        ridge = 1.0
    else:
        ridge = max(0.0, (hi - float(target_condition) * lo)
                    / (float(target_condition) - 1.0))
        ridge = max(ridge, hi * 1e-10)
    system = psd + ridge * np.eye(len(rho))
    w = np.linalg.solve(system, rho)
    cond_after = float(np.linalg.cond(system))
    return w, {
        "ridge": float(ridge),
        "target_condition": float(target_condition),
        "condition_after": cond_after,
        "raw_min_eigenvalue": float(raw_evals[0]),
        "raw_max_eigenvalue": float(raw_evals[-1]),
        "n_negative_eigenvalues_clipped": int(np.sum(raw_evals < -1e-12)),
    }


def _pcr_weights(C, rho, n_components=2):
    C = _symmetrize(C)
    m = C.shape[0]
    k = min(max(1, int(n_components)), m)
    vals, vecs = eigh(C, subset_by_index=[m - k, m - 1])
    vals, vecs = vals[::-1], vecs[:, ::-1]
    w = np.zeros(m, dtype=float)
    for j in range(k):
        if vals[j] > 1e-12:
            w += (vecs[:, j] @ rho) / vals[j] * vecs[:, j]
    return w, vals


def _estimate_g2_and_rho(C, low_rank, var_y, *, g2_grid=300,
                         projection_components=1):
    """Tenzer Eqs. (7), (14): recover rho from L and select g2."""
    m = C.shape[0]
    A, pairs = additive_design(m)
    b = np.array([low_rank[i, j] for i, j in pairs], dtype=float)
    rho0 = solve_additive(A, b, loss="l2")

    kp = min(max(1, int(projection_components)), m)
    evecs = eigh(C, subset_by_index=[m - kp, m - 1])[1][:, ::-1]
    best = (float("inf"), 0.0, rho0)
    for q in np.linspace(0.0, float(var_y), max(2, int(g2_grid))):
        rho = rho0 + 0.5 * q
        proj = evecs @ (evecs.T @ rho)
        residual = float(np.linalg.norm(rho - proj)
                         / (np.linalg.norm(rho) + 1e-12))
        if residual < best[0]:
            best = (residual, float(q), rho.copy())
    return best[1], best[2], best[0]


@dataclass
class SparseUPCRResult:
    """Shared SU-PCR/SDSF fit; only the final weight vector differs."""

    w_pcr: np.ndarray
    w_structured: np.ndarray
    rho_hat: np.ndarray
    g2_hat: float
    var_y: float
    covariance: np.ndarray
    decomposition: SparseDecomposition
    pcr_eigenvalues: np.ndarray
    projection_residual: float
    ridge_diagnostics: dict
    meta: dict = field(default_factory=dict)


def sparse_upcr_fit(
    F,
    *,
    var_y=None,
    scale_ratio=0.25,
    rank=2,
    n_components=2,
    g2_projection_components=1,
    g2_grid=300,
    threshold_multiplier=1.0,
    max_iter=100,
    inner_completion_iter=40,
    decomposition_tol=1e-8,
    max_sparse_fraction=None,
    target_condition=100.0,
    fixed_support=None,
):
    """Fit sparse-error U-PCR and dependency-weighted SDSF in one pass.

    ``F`` has shape ``(m_features, n_samples)`` and must already carry the same
    relative sign orientation used by the incumbent U-PCR arm.  The global sign
    remains unidentifiable and is resolved by the caller using the project's
    existing anchor.  No labels are accepted by this function.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError("F must have shape (features, samples)")
    if not np.isfinite(F).all():
        raise ValueError("F contains non-finite values")
    m, n = F.shape
    if m < 3 or n < 3:
        raise ValueError("sparse U-PCR needs at least 3 features and 3 samples")

    C = _symmetrize((F @ F.T) / n)
    diag_mean = float(np.mean(np.diag(C)))
    if var_y is None:
        var_y = float(scale_ratio) * diag_mean

    decomp = projected_sparse_decomposition(
        C,
        rank=rank,
        threshold_multiplier=threshold_multiplier,
        max_iter=max_iter,
        inner_completion_iter=inner_completion_iter,
        tol=decomposition_tol,
        max_sparse_fraction=max_sparse_fraction,
        fixed_support=fixed_support,
    )
    g2, rho, proj_res = _estimate_g2_and_rho(
        C,
        decomp.low_rank,
        float(var_y),
        g2_grid=g2_grid,
        projection_components=g2_projection_components,
    )
    w_pcr, pcr_evals = _pcr_weights(C, rho, n_components=n_components)
    w_structured, ridge_diag = regularized_covariance_weights(
        decomp.structured_cov,
        rho,
        target_condition=target_condition,
    )

    return SparseUPCRResult(
        w_pcr=w_pcr,
        w_structured=w_structured,
        rho_hat=rho,
        g2_hat=float(g2),
        var_y=float(var_y),
        covariance=C,
        decomposition=decomp,
        pcr_eigenvalues=pcr_evals,
        projection_residual=float(proj_res),
        ridge_diagnostics=ridge_diag,
        meta={
            "m": int(m), "n": int(n), "scale_ratio": float(scale_ratio),
            "diag_mean": diag_mean, "rank": int(rank),
            "n_components": int(n_components),
            "g2_projection_components": int(g2_projection_components),
            "g2_at_ceiling": bool(g2 >= float(var_y)
                                  * (1.0 - 1.5 / max(int(g2_grid), 2))),
        },
    )
