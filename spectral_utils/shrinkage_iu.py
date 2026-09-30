"""Structured covariance shrinkage inside the answer-only IU-PCR solve.

Motivation (Omri / Claude / Codex discussion, 2026-09-08/09). Joint L-SML's
model says the feature covariance of one answer is
``v v^T + blockdiag_g(u_g u_g^T) + D``: cross-group co-movement is explained by
the shared signal alone, extra structure lives only inside groups. Its
iterative fit failed on 25-30% of answers and needed a learned partition.
This module keeps the assumption and drops the fit: it shrinks the empirical
window covariance ``C`` toward a closed-form target ``T`` built from ``C``
itself,

    C_alpha = (1 - alpha) C + alpha T,

with a FIXED partition read off the feature names of the bank that was
actually used (moment bank: stream x {level, sd, slope}; context bank:
stream x {level, ema8, ema32}), and feeds ``C_alpha`` into the unchanged
IU-PCR machinery at one of three declared depths:

* ``solve``    keep IU's rho estimate and its two spectral directions U from
               the ORIGINAL C; use C_alpha only in the 2x2 solve
               ``w = U [U^T C_alpha U]^{-1} U^T rho``  (Codex's isolation of
               "does a steadier covariance improve the weights alone?").
* ``subspace`` rho from the original C; U from C_alpha; same solve.
* ``full``     the entire IU-PCR fit (rho, U, solve) on C_alpha via the
               existing ``upcr_fit_covariance`` seam (the original proposal).

Targets (all label-free, all from the answer's own windows):

* ``joint``  cross-block entries replaced by the rank-1 prediction v_i v_j from
             the leading eigenpair of C; within-block entries and the diagonal
             kept.  This is the Joint model without the iterative fit.
* ``block``  cross-block entries set to zero (block-diagonal).
* ``diag``   every off-diagonal entry set to zero (ordinary diagonal shrinkage;
             the mechanism control asked for by Codex: is any gain due to the
             group structure or just to generic stabilisation?).

alpha: a declared constant (0.5, 1.0) or the Ledoit-Wolf / Schafer-Strimmer
analytic estimate restricted to the entries the target changes,
``alpha* = sum Var(c_ij) / sum (c_ij - t_ij)^2``, clipped to [0, 1].  The
variance term assumes independent windows, which adjacent 8-token windows are
not; alpha* is therefore reported as a label-free automatic rule, not as an
optimum.  Nothing here uses labels or other answers.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

TARGETS = ("joint", "block", "diag")
LEVELS = ("solve", "subspace", "full")


def partition(names, by="stream"):
    """Fixed group labels from feature names ``<stream>__<op>``."""
    key = 0 if by == "stream" else 1
    labels = [n.split("__")[key] for n in names]
    uniq = sorted(set(labels))
    return np.asarray([uniq.index(l) for l in labels], int), uniq


def target_matrix(C, groups, kind):
    C = np.asarray(C, float)
    same = groups[:, None] == groups[None, :]
    T = np.array(C, copy=True)
    if kind == "diag":
        return np.diag(np.diag(C))
    if kind == "block":
        T[~same] = 0.0
        return T
    if kind == "joint":
        lam, vec = eigh(C, subset_by_index=[C.shape[0] - 1, C.shape[0] - 1])
        v = vec[:, 0] * np.sqrt(max(float(lam[0]), 0.0))
        rank1 = np.outer(v, v)
        T[~same] = rank1[~same]
        return T
    raise ValueError(kind)


def ledoit_wolf_alpha(Z, C, T):
    """Schafer-Strimmer alpha on the entries where T differs from C."""
    Z = np.asarray(Z, float)
    n = Z.shape[0]
    changed = ~np.isclose(C, T)
    np.fill_diagonal(changed, False)
    if not changed.any() or n < 3:
        return 0.0
    # Var(c_ij) with c_ij = mean_k z_ki z_kj over the fitting windows.
    prod = Z[:, :, None] * Z[:, None, :]                       # n x p x p
    var = prod.var(axis=0, ddof=1) / n
    num = float(var[changed].sum())
    den = float(((C - T)[changed] ** 2).sum())
    if den <= 0:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def shrink(C, T, alpha):
    return (1.0 - alpha) * np.asarray(C, float) + alpha * np.asarray(T, float)


def top_subspace(C, k=2):
    m = C.shape[0]
    ev, evec = eigh(C, subset_by_index=[m - k, m - 1])
    return evec[:, ::-1]


def solve_weights(C_alpha, U, rho):
    """w = U [U^T C_alpha U]^{-1} U^T rho  (same form as laplacian_upcr)."""
    G = U.T @ C_alpha @ U
    return U @ np.linalg.solve(G, U.T @ rho)


def iu_reference_weights(C, U, rho):
    """Sanity: the two-component IU solve with the original C must equal
    sum_c (v_c.rho)/lambda_c v_c when U are exact eigenvectors of C."""
    return solve_weights(C, U, rho)


def shrunk_iu_weights(Z, names, iu_result, level, kind, alpha, by="stream", upcr_fit_covariance=None,
                      iu_kwargs=None):
    """Weights for one (level, target, alpha) variant on one answer.

    Z: standardized, oriented fitting windows (n x p), as passed to IU (fit).
    iu_result: the UPCRResult of the original IU fit on Z (rho_hat, meta).
    Returns (w, info).
    """
    Z = np.asarray(Z, float)
    n = Z.shape[0]
    C = Z.T @ Z / n
    groups, _ = partition(names, by)
    T = target_matrix(C, groups, kind)
    a = ledoit_wolf_alpha(Z, C, T) if alpha == "lw" else float(alpha)
    C_a = shrink(C, T, a)
    info = dict(alpha=a, target=kind, level=level, partition=by, groups=int(groups.max() + 1))
    if level == "full":
        assert upcr_fit_covariance is not None
        res = upcr_fit_covariance(C_a, **(iu_kwargs or {}))
        info["abstained"] = bool(res.abstained)
        return np.asarray(res.w, float), info
    rho = np.asarray(iu_result.rho_hat, float)
    U = top_subspace(C if level == "solve" else C_a, 2)
    return solve_weights(C_a, U, rho), info


def self_test():
    rng = np.random.default_rng(0)
    n, p = 40, 6
    names = [f"s{i}__{op}" for i in range(2) for op in ("level", "sd", "slope")]
    Z = rng.standard_normal((n, p)); Z -= Z.mean(0); Z /= Z.std(0)
    C = Z.T @ Z / n
    g, _ = partition(names); assert g.tolist() == [0, 0, 0, 1, 1, 1]
    Tb = target_matrix(C, g, "block"); assert Tb[0, 3] == 0 and Tb[0, 1] == C[0, 1]
    Td = target_matrix(C, g, "diag"); assert np.allclose(Td, np.diag(np.diag(C)))
    Tj = target_matrix(C, g, "joint"); assert Tj[0, 1] == C[0, 1] and Tj[0, 3] != C[0, 3]
    assert np.allclose(np.diag(Tj), np.diag(C))
    a = ledoit_wolf_alpha(Z, C, Tb); assert 0.0 <= a <= 1.0
    assert np.allclose(shrink(C, Tb, 0.0), C) and np.allclose(shrink(C, Tb, 1.0), Tb)
    U = top_subspace(C); rho = rng.standard_normal(p)
    ev, evec = eigh(C); w_ref = sum((evec[:, -k] @ rho) / ev[-k] * evec[:, -k] for k in (1, 2))
    assert np.allclose(solve_weights(C, U, rho), w_ref, atol=1e-10)
    return True
