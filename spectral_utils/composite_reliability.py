"""
composite_reliability.py — label-free SET-LEVEL objectives for feature selection.

WHY THIS EXISTS
---------------
Steps 220-222 established that the only channel with room in U-PCR is which features
get kept (+2.25pp held out), and that NO per-feature ranking reaches it: the true
correlation with correctness is +0.08pp, and all six label-free rankers in Step 222
land on or below the matched pruning floor. So the property that makes a set good is
a property of the SET.

Step 222's `redundancy` ranker (mean |corr| to the whole pool, keep the lowest) was
the WORST of the eight arms at -3.13pp, yet the good sets are measurably LESS
internally correlated than matched random subsets. Those are compatible only one way:
mean |corr| conflates TWO things that point in opposite directions —

    loading on the shared factor      (signal — you want MORE of it)
    sharing errors with the others    (contamination — you want LESS of it)

and a marginal statistic cannot separate them. The quantity that does is the
correlation NOT explained by the factor, i.e. exactly the `Delta_ij = E[h_i h_j]`
that U-PCR (upcr.py:86, Eq. 14) assumes to be ZERO.

    C_S  =  lambda lambda^T  +  Psi  +  Delta
            \_____________/     \_/     \___/
              shared factor    unique   SHARED ERROR — assumed 0, measured 0.46
                                noise   of the off-diagonal norm on our full pool

THE IDENTIFIABILITY PROBLEM, AND WHY SPARSITY IS THE RIGHT PRIOR
----------------------------------------------------------------
A FULL Delta is not identifiable: m(m-1)/2 pair equations against m+1 unknowns leaves
360 spare equations at m=28.4, but a full Delta adds exactly m(m-1)/2 unknowns — the
system saturates and the residual is zero by construction.

A SPARSE Delta fits comfortably inside the spare degrees of freedom, and on our data
the violation IS sparse rather than low-rank: the top 10% of pairs carry ~44% of the
residual mass (uniform would be 10%), while the leading eigenvalue share of |Delta| is
only ~0.33. There is no single second latent factor; there are a few strongly
dependent pairs. `delta_structure` recomputes both diagnostics so the prior is
re-checked per run rather than assumed.

The sparsity prior is ALSO what makes `omega` usable. Second moments alone cannot
distinguish "both load on the factor" from "both share an error", so plain McDonald's
omega is maximised by m EXACT DUPLICATES (lambda_i = 1, Delta = 0, omega = 1 — verified
exactly). Under a sparse-Delta fit a duplicate pair's correlation is absorbed into one
large Delta entry instead of being credited to lambda, which is what breaks the tie.

    M1's NAMED FAILURE MODE, measured not argued. The fix holds only while the
    duplicate block fits the sparsity budget. A block of b identical views costs
    C(b,2) dependent pairs against a budget of sparsity * C(m,2), so omega is fooled
    once

            b  >~  m * sqrt(sparsity)          (~0.32 m at the 10% default)

    Measured at m=12, sparsity=0.10 (budget 6 pairs): blocks of 2 and 3 are correctly
    penalised BELOW an honest set, blocks of 4+ are not — the crossover lands exactly
    where C(b,2) reaches the budget. At the deployed keep-set size m~21 the budget is
    21 pairs and the crossover is b~7.

    This matters on our pool, which contains by-construction near-duplicate triples
    (`epr`/`epr_spilled`/`epr_energy`, the `sw_var_peak` triple, the `cusum_max`
    triple). exp15 sweeps `sparsity` as a sensitivity for exactly this reason, and M2
    / M3 / M4 sit beside M1 partly to catch it if it fires.

The degeneracy and its boundary are tested in
`scripts/test_composite_reliability.py::test_duplicate_degeneracy` and must be
reported alongside every omega — not hidden.

WHAT IS IN HERE
---------------
    fit_sparse_factor   C -> (lambda, psi, Delta) with Delta sparse
    delta_structure     is the violation sparse? is it low-rank?
    omega_sparse        M1  composite reliability      HIGHER = keep   SELF-SIZING
    resid_dep           M2  sum |Delta_ij|             LOWER  = keep   fixed k
    m_eff               M3  effective independent views HIGHER = keep  SELF-SIZING
    cohesion_set        M4  mean |corr| within S       LOWER  = keep   fixed k
    loading_sum         M5  sum lambda_i               HIGHER = keep   fixed k
    greedy_backward     the search — drop from an incumbent while J improves

None of these touches the deployed path. They produce a column subset; the estimator
is unchanged and is fed that subset through `fit_cols(..., exclusion=False)`.
"""
import numpy as np
from scipy.linalg import eigh

__all__ = [
    "fit_sparse_factor", "delta_structure", "FactorFit",
    "omega_sparse", "resid_dep", "m_eff", "cohesion_set", "loading_sum",
    "OBJECTIVES", "greedy_backward",
]

# Default share of pairs allowed to carry a shared-error term. Set from the measured
# structure of the violation (top decile carries ~44% of the residual mass), NOT tuned
# against performance. `exp15` sweeps it as a sensitivity.
DEFAULT_SPARSITY = 0.10
# Both loops are FIXED-POINT iterations with early exit on `tol`, not optimisers, so
# the caps are safety rails rather than budgets — the greedy refits per candidate drop
# (~21 fits x ~18 steps x 120 splits x 5 arms), which is what makes the cost matter.
N_ALT = 15          # alternating passes: rank-1 fit <-> sparse residual
N_DIAG = 12         # diagonal-imputation passes inside the off-diagonal rank-1 fit
MIN_SET = 3         # U-PCR's own Assumption A4 (upcr.py:240)


class FactorFit:
    """C ~ lambda lambda^T + diag(psi) + Delta, Delta sparse with zero diagonal."""

    __slots__ = ("lam", "psi", "delta", "resid_raw", "C", "n_nonzero", "converged")

    def __init__(self, lam, psi, delta, resid_raw, C, n_nonzero, converged):
        self.lam = lam
        self.psi = psi
        self.delta = delta            # AFTER soft-thresholding — what the objectives use
        self.resid_raw = resid_raw    # BEFORE it — what the structure diagnostic uses
        self.C = C
        self.n_nonzero = n_nonzero
        self.converged = converged


def _top_eigpair(B):
    """Leading eigenpair only. `subset_by_index` is ~4x a full decomposition here and
    this is the innermost call in the whole experiment."""
    m = B.shape[0]
    ev, evec = eigh(B, subset_by_index=[m - 1, m - 1])
    return float(ev[0]), evec[:, 0]


def _rank1_offdiag(M, n_iter=N_DIAG, tol=1e-10):
    """Best rank-1 fit to the OFF-DIAGONAL of M (the diagonal is free).

    The standard principal-factor move: the diagonal carries unique variance we do not
    want to explain, so it is imputed with the current lambda_i^2 rather than fitted.
    Deterministic fixed-point iteration, no random start, early exit on `tol`.
    """
    A = M.copy()
    np.fill_diagonal(A, 0.0)
    top, vec = _top_eigpair(A)
    lam = vec * np.sqrt(max(top, 0.0))
    B = A.copy()
    for _ in range(n_iter):
        np.fill_diagonal(B, lam ** 2)
        top, vec = _top_eigpair(B)
        lam_new = vec * np.sqrt(max(top, 0.0))
        # the eigenvector's sign is arbitrary; compare on the aligned representative
        if float(lam_new @ lam) < 0:
            lam_new = -lam_new
        if float(np.max(np.abs(lam_new - lam))) < tol:
            lam = lam_new
            break
        lam = lam_new
    # Orient so the composite is keyed with the majority of the views. AUROC is
    # scale-invariant and the anchor fixes the global sign downstream, but
    # `sum(lambda)` is NOT sign-invariant and M1/M5 read it directly.
    if np.sum(lam) < 0:
        lam = -lam
    return lam


def _soft_threshold(R, frac):
    """Keep the `frac` largest-magnitude off-diagonal entries, soft-thresholded.

    Soft rather than hard: a hard threshold makes the fit discontinuous in the data,
    so a feature entering or leaving a subset could flip Delta's support and move the
    objective by a step. The greedy search is then descending a staircase.
    """
    m = R.shape[0]
    iu = np.triu_indices(m, 1)
    vals = np.abs(R[iu])
    if vals.size == 0:
        return np.zeros_like(R), 0
    n_keep = int(np.floor(frac * vals.size))
    if n_keep <= 0:
        return np.zeros_like(R), 0
    tau = np.sort(vals)[::-1][n_keep - 1]
    D = np.sign(R) * np.maximum(np.abs(R) - tau, 0.0)
    np.fill_diagonal(D, 0.0)
    return D, int(np.sum(np.abs(D[iu]) > 0))


def fit_sparse_factor(C, sparsity=DEFAULT_SPARSITY, n_alt=N_ALT, tol=1e-9):
    """Decompose a correlation matrix into a rank-1 factor plus a SPARSE shared-error
    term: C ~ lam lam^T + diag(psi) + Delta.

    Alternates (a) rank-1 fit to the off-diagonal of C - Delta, (b) soft-threshold of
    the residual into Delta. `sparsity` is the share of pairs allowed a shared-error
    term; at sparsity=0 this reduces to the plain off-diagonal rank-1 fit that SML
    uses, which is the right degenerate case to have.
    """
    C = np.asarray(C, dtype=float)
    m = C.shape[0]
    delta = np.zeros((m, m))
    lam = np.zeros(m)
    converged = False
    R = np.zeros((m, m))
    for _ in range(n_alt):
        lam_new = _rank1_offdiag(C - delta)
        R = C - np.outer(lam_new, lam_new)
        np.fill_diagonal(R, 0.0)
        delta_new, nnz = _soft_threshold(R, sparsity)
        shift = max(float(np.max(np.abs(lam_new - lam))),
                    float(np.max(np.abs(delta_new - delta))) if m > 1 else 0.0)
        lam, delta = lam_new, delta_new
        if shift < tol:
            converged = True
            break
    else:
        nnz = int(np.sum(np.abs(delta[np.triu_indices(m, 1)]) > 0))
    psi = np.clip(np.diag(C) - lam ** 2, 0.0, None)
    return FactorFit(lam, psi, delta, R, C, nnz, converged)


def delta_structure(fit):
    """Is the shared-error term SPARSE (few big pairs) or LOW-RANK (a second factor)?

    Returns (top-decile share of squared residual mass, leading eigenvalue share).
    Sparse looks like (high, low); a second latent factor looks like (low, high).

    MEASURED ON THE RAW RESIDUAL, NOT ON `fit.delta`. Soft-thresholding at the 10%
    level zeroes everything outside the top decile by construction, so a diagnostic
    read off `fit.delta` returns 1.000 whatever the data looks like — it would be
    testing the threshold, not the violation. The first cut of this function did
    exactly that and the pilot printed 1.000 on every cell. Same class of defect as
    the Step-216 audit that stopped measuring the cell it existed to justify.
    """
    D = fit.resid_raw
    m = D.shape[0]
    if m < 4:
        return np.nan, np.nan
    iu = np.triu_indices(m, 1)
    sq = D[iu] ** 2
    tot = float(sq.sum())
    if tot <= 0:
        return np.nan, np.nan
    k = max(1, int(np.ceil(0.10 * sq.size)))
    top = float(np.sort(sq)[::-1][:k].sum() / tot)
    ev = np.abs(np.linalg.eigvalsh(D))
    r1 = float(ev.max() / (ev.sum() + 1e-30))
    return top, r1


# ---------------------------------------------------------------------------
# the five objectives. Every one takes (C, fit) and returns a scalar to MAXIMISE.
# Directions are folded in here (a LOWER-is-better quantity is returned negated) so
# the search never has to know which way an arm points — Step 222's lesson that a
# free direction doubles the degrees of freedom.
# ---------------------------------------------------------------------------
def omega_sparse(C, fit):
    """M1 — McDonald's omega under the sparse-error factor model. SELF-SIZING.

        omega = (sum lambda)^2 / (1' C 1)

    The denominator is the variance of the equally-weighted composite, so it already
    contains Psi and Delta; adding a redundant view grows it by 1 + 2*sum_j C_new,j
    while the numerator grows only by 2*lambda_new*sum(lambda) + lambda_new^2. A view
    that duplicates the set therefore LOWERS omega, which is what lets this arm pick
    its own size with no K to set.
    """
    tot = float(np.sum(C))
    if tot <= 1e-9:
        return np.nan
    return float(np.sum(fit.lam) ** 2 / tot)


def resid_dep(C, fit):
    """M2 — the shared-error mass itself, sum |Delta_ij|. LOWER = keep.

    Omri's relaxation of E[h_i h_j] = 0 used directly as a selection criterion, in
    ABSOLUTE form. The NORMALISED form (divided by ||C_off||) was measured in the
    Step-223 diagnostic and pointed the wrong way — it falls as the denominator
    grows, so it rewards cohesive sets. Absolute is what the mechanism predicts.
    """
    m = C.shape[0]
    iu = np.triu_indices(m, 1)
    return -float(np.sum(np.abs(fit.delta[iu])))


def m_eff(C, fit):
    """M3 — effective number of independent views. SELF-SIZING.

        m_eff = m / (1 + (m-1) * mean(Delta_offdiag))

    The classical effective-sample-size correction for correlated measurements, with
    the correlation taken as the SHARED-ERROR term rather than the raw one: views that
    agree because they measure the same latent are not redundant, views that agree
    because they share an error are. Negative mean residual correlation legitimately
    pushes m_eff above m.
    """
    m = C.shape[0]
    if m < 2:
        return float(m)
    iu = np.triu_indices(m, 1)
    rbar = float(np.mean(fit.delta[iu]))
    den = 1.0 + (m - 1) * rbar
    if den <= 1e-6:
        return float(m) / 1e-6
    return float(m) / den


def cohesion_set(C, fit):
    """M4 — mean |corr| within S. LOWER = keep. THE CONTROL.

    The SAME quantity as Step 222's `redundancy` arm (-3.13pp, worst of eight), but
    evaluated as a SET objective under a set search instead of a marginal top-k. This
    is the experiment's cleanest test of the thesis claim: if it clears the floor here
    where the marginal version lost by 3pp, "it is a set property, not a per-feature
    one" is demonstrated by a controlled comparison rather than argued.
    """
    m = C.shape[0]
    if m < 2:
        return 0.0
    iu = np.triu_indices(m, 1)
    return -float(np.mean(np.abs(C[iu])))


def loading_sum(C, fit):
    """M5 — sum of factor loadings. HIGHER = keep.

    The SIGNAL half of M1 with the contamination denominator removed. M4 and M5
    bracket M1, so a win for M1 can be attributed to one half or the other rather
    than left unexplained. Monotone increasing in set size, so fixed-k only.
    """
    return float(np.sum(fit.lam))


# name -> (fn, self_sizing, needs_factor_fit). `cohesion_set` reads only C, and
# skipping its fit is ~1/5 of the whole run.
OBJECTIVES = {
    "omega_sparse": (omega_sparse, True, True),
    "resid_dep": (resid_dep, False, True),
    "m_eff": (m_eff, True, True),
    "cohesion_set": (cohesion_set, False, False),
    "loading_sum": (loading_sum, False, True),
}


# ---------------------------------------------------------------------------
def _score(Cfull, cols, fn, sparsity, needs_fit=True):
    cols = sorted(int(c) for c in cols)
    if len(cols) < MIN_SET:
        return np.nan
    Cs = Cfull[np.ix_(cols, cols)]
    try:
        fit = fit_sparse_factor(Cs, sparsity=sparsity) if needs_fit else None
    except np.linalg.LinAlgError:
        return np.nan
    v = fn(Cs, fit)
    return float(v) if np.isfinite(v) else np.nan


def greedy_backward(Cfull, start, fn, sparsity=DEFAULT_SPARSITY,
                    target_k=None, min_set=MIN_SET, tiebreak=None,
                    needs_fit=True):
    """Drop features one at a time from `start`, best-first by `fn` (MAXIMISED).

    target_k=None   free-size: stop as soon as no single drop improves the objective.
                    This is the deployable form — the arm picks its own size.
    target_k=k      fixed-size: keep dropping to exactly k whether or not it improves,
                    so the arm is comparable to the matched pruning floor at the same
                    size.

    Backward from the incumbent, not forward from scratch: it matches how the good
    sets were built, and it means an uninformative objective degrades to the deployed
    keep set rather than to something arbitrary.

    `tiebreak` is a per-feature array used to order equal-scoring drops. Ties are NOT
    broken by column index: the pool is laid out spectral-then-energy-then-logprob, so
    an index tie-break silently means "prefer spectral" — a prior, in a study whose
    point is not carrying one (Step 222, exp14:134).
    """
    cur = sorted(int(c) for c in start)
    if tiebreak is None:
        tiebreak = np.zeros(Cfull.shape[0])
    floor_k = max(min_set, int(target_k) if target_k is not None else min_set)
    best = _score(Cfull, cur, fn, sparsity, needs_fit)
    n_steps = 0
    while len(cur) > floor_k:
        best_key, cand_drop = None, None
        for j in cur:
            s = _score(Cfull, [c for c in cur if c != j], fn, sparsity, needs_fit)
            if not np.isfinite(s):
                continue
            key = (s, float(tiebreak[j]))
            if best_key is None or key > best_key:
                best_key, cand_drop = key, j
        if cand_drop is None:
            break
        cand_score = best_key[0]
        if target_k is None and not (cand_score > best):
            break                          # free-size: stop when no drop improves
        cur = [c for c in cur if c != cand_drop]
        best = cand_score
        n_steps += 1
    return sorted(cur), float(best), n_steps
