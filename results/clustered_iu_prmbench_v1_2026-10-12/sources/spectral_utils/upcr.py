"""
upcr.py — paper-faithful Unsupervised Principal Components Regression.

Dror, Nadler, Bilal & Kluger, "Unsupervised Ensemble Regression",
arXiv:1703.02965 (2017).  (`fusion_utils.upcr_fuse`'s header credits Tenzer et
al. AISTATS 2022 — that is a different paper, "Crowdsourcing regression: a
spectral approach"; the algorithm implemented here is the 2017 one.)

WHY A SECOND MODULE
-------------------
`fusion_utils.upcr_fuse` stays byte-identical so every committed number and the
GOOD_6 = 0.7594 anchor remain reproducible. This module is the faithful
implementation, with each documented deviation exposed as a flag so the A/B can
be measured rather than argued.

WHAT THE METHOD DOES, IN ONE PARAGRAPH
--------------------------------------
We want per-feature fusion weights but have no answer key. Lemma 1 says the
optimal weights solve `rho = C w`, where `C` is the observable feature covariance
and `rho_i = Cov(f_i, Y)` is the unobservable thing we need. Theorem 1 recovers
`rho` without labels: IF features make uncorrelated errors then every off-diagonal
obeys `C_ij = g2 + a_i + a_j` (Eq. 14) — one linear equation per feature PAIR, so
m >= 3 features over-determine the m unknowns. `g2` is the shared signal, and
`rho_i = g2 + a_i` (Eq. 11). `g2` itself is picked by a model-selection criterion
(Eq. 20). Weights then come from the top one or two eigenvectors of `C` (Eq. 21),
which is what makes the method stable when `C` is nearly rank-one.

DEVIATIONS FROM `fusion_utils.upcr_fuse`, ALL VERIFIED AGAINST THE PAPER
------------------------------------------------------------------------
1.  loss='l1'                  Remark 1 / Eq. 15 also uses the ABSOLUTE loss, for
                               robustness when the uncorrelated-error assumption
                               is violated. The old path was squared-loss only.
2.  scale_ratio                `var_y` was hardcoded to 0.25 while the covariance
                               of z-scored features has diagonal EXACTLY 1.0, so
                               the g2 search only ever explored the bottom quarter
                               of its meaningful range. Only the RATIO
                               `var_y / mean(diag C)` is identifiable — see
                               `moment_match` below.
3.  difficulty_gate            Algorithm 1 STOPS when `g2 < eps_L * Var(Y)`. The
                               old path always returned weights.
4.  simple_avg_fallback        Sec. 4.3: with <= 4 experts left after exclusion,
                               the paper uses a plain average. The old path kept
                               the top 3 and still applied Eq. 21.
5.  recompute_after_exclusion  Algorithm 1: "Recalculate v1, rho(q), g2 on
                               remaining experts". The old path reused g2 from the
                               full-C grid.
6.  g2_projection_k=1          Eq. 20's residual is against v1 ALONE. The 2-component
                               rule (Sec. 4.3) replaces Eq. 21 only. The old path
                               projected onto whatever the weight-component count
                               was — and since lambda2 > 0.1*Trace fires on 28/29
                               cells, that was 2 almost everywhere.
7.  exclusion=False            `min_frac=0, exclude_frac=inf` does NOT disable
                               exclusion in the old path — it collapses to
                               `rho >= 0`, which still drops weak features.

WHY THE g2 GRID IS CHEAPER HERE (measured 13.8x, not the 300x first claimed)
----------------------------------------------------------------------------
Eq. 16: `rho(q) = rho(0) + (q/2) * 1`, and the paper notes this holds "regardless
of the loss function L" because `q` enters the residual additively. Since each
design row is `e_i + e_j`, `A @ 1 == 2`, so shifting `rho` by `q/2` shifts every
fitted value by exactly `q` — the residual VECTOR is unchanged. So the system is
solved ONCE at q=0 and shifted analytically. The old path re-ran `lstsq` at all
300 grid points and got the same answer each time.

This removes the 300 `lstsq` calls but NOT the 300 eigen-projections, so the
end-to-end gain is 13.8x (74.4ms -> 5.4ms on ars_gsm8k_r1distill8b), not 300x.
What it does buy outright is the exact L1 solve: one linear program instead of
300. For L1 the identity holds up to LP non-uniqueness -- the residual vector is
invariant, the argmin need not be unique -- which is why `smoke_upcr.py` checks
residual equality rather than coefficient equality.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

from .fusion_utils import zscore

__all__ = [
    "UPCRResult", "upcr_fit", "upcr_fit_covariance", "upcr_pipeline_faithful",
    "additive_design", "solve_additive", "moment_match",
]


# ---------------------------------------------------------------------------
# the additive system  C_ij = rho_i + rho_j - g2   (Eq. 14 + Eq. 11)
# ---------------------------------------------------------------------------
def additive_design(m, pairs=None):
    """Design matrix for the pair equations. Row for pair (i, j) is e_i + e_j.

    ``pairs`` restricts the system to a subset of pairs — this is the seam the
    clustered variant uses to drop same-cluster pairs, whose shared error term
    breaks Eq. 14. Returns (A, pairs).
    """
    if pairs is None:
        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    pairs = list(pairs)
    A = np.zeros((len(pairs), m))
    for k, (i, j) in enumerate(pairs):
        A[k, i] += 1.0
        A[k, j] += 1.0
    return A, pairs


def solve_additive(A, b, loss="l2"):
    """Solve  A @ rho ~ b  under squared or absolute loss (Eq. 15 / Remark 1).

    'l2' is `lstsq`. 'l1' is the exact linear program
        min sum(t)  s.t.  -t <= A rho - b <= t
    which is what "absolute loss" means; it falls back to IRLS if the LP solver
    is unavailable, and to lstsq if that also fails.
    """
    if loss == "l2":
        return np.linalg.lstsq(A, b, rcond=None)[0]
    if loss != "l1":
        raise ValueError(f"loss must be 'l2' or 'l1', got {loss!r}")

    n_eq, m = A.shape
    try:
        from scipy.optimize import linprog
        from scipy.sparse import hstack, eye, csr_matrix
        # variables: [rho (m), t (n_eq)]
        c = np.concatenate([np.zeros(m), np.ones(n_eq)])
        As = csr_matrix(A)
        I = eye(n_eq, format="csr")
        A_ub = hstack([hstack([As, -I]), ]).tocsr()          # A rho - t <= b
        A_ub2 = hstack([hstack([-As, -I]), ]).tocsr()        # -A rho - t <= -b
        res = linprog(
            c,
            A_ub=csr_matrix(np.vstack([A_ub.toarray(), A_ub2.toarray()])),
            b_ub=np.concatenate([b, -b]),
            bounds=[(None, None)] * m + [(0, None)] * n_eq,
            method="highs",
        )
        if res.success:
            return np.asarray(res.x[:m], dtype=float)
    except Exception:
        pass

    # IRLS fallback: reweighted least squares converging to the L1 solution
    rho = np.linalg.lstsq(A, b, rcond=None)[0]
    for _ in range(60):
        r = A @ rho - b
        w = 1.0 / np.maximum(np.abs(r), 1e-6)
        sw = np.sqrt(w)[:, None]
        rho_new = np.linalg.lstsq(A * sw, b * sw[:, 0], rcond=None)[0]
        if np.linalg.norm(rho_new - rho) < 1e-11:
            rho = rho_new
            break
        rho = rho_new
    return rho


def moment_match(F, p):
    """Rescale z-scored rows to the label's first two moments (Assumption A1).

    A1 assumes E[Y] and Var(Y) are known. For a binary correctness label with
    accuracy ``p`` those are ``p`` and ``p*(1-p)``.

    This is exactly equivalent to leaving F z-scored and setting
    ``scale_ratio=1``: every row gets the SAME multiplier, and multiplying C by a
    positive constant leaves its eigenvectors fixed while rho and lambda scale
    together and cancel in Eq. 21. `smoke_upcr.py` asserts it.

    READ THAT CAREFULLY -- it is weaker than "A1 is free". What cancels is the
    accuracy rate p, so we never need to KNOW p. What does NOT follow is that the
    scale ratio is settled: moment-matching sets Var(f_i) = Var(Y) for every
    feature, which is the ratio-1 assumption, i.e. it presumes noiseless
    regressors. A1 is what would PIN the ratio, not what frees it. And the ratio
    is not inert once the thresholds are on: 0.25 -> 1.0 is -0.03pp with the
    difficulty gate off but -2.08pp with it on (abstentions 0.00 -> 17.62 of 25).
    """
    return p + np.sqrt(max(p * (1.0 - p), 1e-12)) * np.asarray(F, dtype=float)


# ---------------------------------------------------------------------------
@dataclass
class UPCRResult:
    w: np.ndarray                  # fusion weights over the FULL feature set
    rho_hat: np.ndarray            # estimated Cov(f_i, Y), full set (0 where dropped)
    rho_hat_full: np.ndarray       # pre-exclusion estimate, every feature
    g2_hat: float
    var_y: float
    keep: np.ndarray               # bool mask of surviving features
    abstained: bool                # difficulty gate fired (Algorithm 1 STOP)
    used_simple_average: bool      # Sec. 4.3 fallback fired
    n_components_used: int
    lambda2_frac: float            # lambda2 / Trace(C)
    proj_residual: float           # Eq. 20 value at the chosen g2
    g2_at_ceiling: bool            # argmin sat on the grid's upper bound
    g2_frac_of_var_y: float        # g2_hat / var_y — the paper's difficulty index
    diag_mean: float               # mean(diag C) — what var_y is measured against
    meta: dict = field(default_factory=dict)


def upcr_fit(
    F,
    *,
    var_y=None,
    scale_ratio=1.0,
    loss="l2",
    n_components=1,
    auto_components=True,
    lambda2_threshold=0.1,
    g2_projection_k=1,
    exclusion=True,
    min_frac=0.05,
    exclude_frac=3.0,
    simple_avg_fallback=True,
    min_experts_for_eq21=5,
    difficulty_gate=True,
    eps_L=0.1,
    on_abstain="flag",
    recompute_after_exclusion=True,
    g2_grid=300,
    pairs=None,
):
    """Faithful U-PCR. Returns a :class:`UPCRResult`.

    Args:
        F: (m, n) — m features x n samples, already sign-oriented. Rows are
            expected to be z-scored (so ``mean(diag C) == 1``); any other common
            scaling is fine as long as ``scale_ratio`` is interpreted against it.
        var_y: Var(Y). If None, set to ``scale_ratio * mean(diag C)`` — the only
            identifiable form. Pass explicitly for the paper's literal spelling.
        scale_ratio: ``var_y / mean(diag C)``. THE free parameter. The legacy path
            is 1/0.25 = 0.25 against a unit diagonal; 1.0 is the paper's
            "an accurate regressor has Var(f_i) ~ Var(Y)" calibration.
        loss: 'l2' (legacy) or 'l1' (Remark 1, robust to dependence).
        g2_projection_k: components in Eq. 20's residual. 1 is the paper.
            'match' reproduces the legacy behaviour of following the weight count.
        exclusion: False genuinely disables the Algorithm 1 drop step.
        on_abstain: 'flag' (score anyway, set ``abstained``), 'mean' (simple
            average), or 'nan'.
        pairs: restrict the additive system to these (i, j) pairs — the seam for
            the clustered variant.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError("F must be a feature-by-sample matrix")
    m, n = F.shape
    if n < 1:
        raise ValueError("F must contain at least one sample")
    return upcr_fit_covariance(
        (F @ F.T) / n,
        var_y=var_y,
        scale_ratio=scale_ratio,
        loss=loss,
        n_components=n_components,
        auto_components=auto_components,
        lambda2_threshold=lambda2_threshold,
        g2_projection_k=g2_projection_k,
        exclusion=exclusion,
        min_frac=min_frac,
        exclude_frac=exclude_frac,
        simple_avg_fallback=simple_avg_fallback,
        min_experts_for_eq21=min_experts_for_eq21,
        difficulty_gate=difficulty_gate,
        eps_L=eps_L,
        on_abstain=on_abstain,
        recompute_after_exclusion=recompute_after_exclusion,
        g2_grid=g2_grid,
        pairs=pairs,
    )


def upcr_fit_covariance(
    covariance,
    *,
    var_y=None,
    scale_ratio=1.0,
    loss="l2",
    n_components=1,
    auto_components=True,
    lambda2_threshold=0.1,
    g2_projection_k=1,
    exclusion=True,
    min_frac=0.05,
    exclude_frac=3.0,
    simple_avg_fallback=True,
    min_experts_for_eq21=5,
    difficulty_gate=True,
    eps_L=0.1,
    on_abstain="flag",
    recompute_after_exclusion=True,
    g2_grid=300,
    pairs=None,
):
    """Fit U-PCR from an already estimated feature covariance matrix.

    This is the moment-level seam used by conditional estimators.  Calling it
    with ``F @ F.T / n`` is numerically identical to :func:`upcr_fit`; no
    sample labels, means, or higher moments are accepted here.
    """
    C = np.asarray(covariance, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("covariance must be a square matrix")
    m = C.shape[0]
    if m < 3:
        raise ValueError(f"U-PCR needs m >= 3 features (Assumption A4), got {m}")
    if not np.isfinite(C).all():
        raise ValueError("covariance contains non-finite values")
    if not np.allclose(C, C.T, atol=1e-10, rtol=1e-10):
        raise ValueError("covariance must be symmetric")
    C = 0.5 * (C + C.T)
    trace_C = float(np.trace(C))
    diag_mean = float(np.mean(np.diag(C)))
    if var_y is None:
        var_y = float(scale_ratio) * diag_mean

    k_probe = min(2, m)
    evals_probe = eigh(C, subset_by_index=[m - k_probe, m - 1])[0][::-1]
    lambda2_frac = (float(evals_probe[1] / (trace_C + 1e-12))
                    if k_probe >= 2 else 0.0)
    if auto_components:
        n_components = 2 if (k_probe >= 2 and lambda2_frac > lambda2_threshold) else 1

    def _fit_block(Cb, idx_pairs, var_y_b):
        """One full estimation pass over a feature block: solve once, shift over
        the g2 grid, pick g2 by Eq. 20."""
        mb = Cb.shape[0]
        A, prs = additive_design(mb, idx_pairs)
        b = np.array([Cb[i, j] for i, j in prs], dtype=float)

        rho0 = solve_additive(A, b, loss=loss)          # Eq. 15 at q = 0

        kproj = (min(n_components, mb) if g2_projection_k == "match"
                 else min(int(g2_projection_k), mb))
        evecs = eigh(Cb, subset_by_index=[mb - kproj, mb - 1])[1][:, ::-1]

        grid = np.linspace(0.0, var_y_b, int(g2_grid))
        best = (np.inf, grid[0], rho0)
        for q in grid:
            rho = rho0 + 0.5 * q                        # Eq. 16, loss-independent
            proj = evecs @ (evecs.T @ rho)
            res = np.linalg.norm(rho - proj) / (np.linalg.norm(rho) + 1e-12)
            if res < best[0]:
                best = (float(res), float(q), rho)
        return Cb, best[0], best[1], best[2]

    C_full, res_full, g2_full, rho_full = _fit_block(C, pairs, var_y)

    # ---- Algorithm 1: difficulty gate -------------------------------------
    g2_frac = g2_full / (var_y + 1e-12)
    abstained = bool(difficulty_gate and g2_frac <= eps_L)

    # ---- Algorithm 1: exclude weak experts --------------------------------
    if exclusion:
        rho_max = float(np.max(rho_full))
        keep = ((rho_full >= min_frac * var_y)
                & (rho_full >= rho_max / exclude_frac))
        if keep.sum() < 3:                      # Eq. 21 needs >= 3 (A4)
            keep = np.zeros(m, dtype=bool)
            keep[np.argsort(rho_full)[-3:]] = True
    else:
        keep = np.ones(m, dtype=bool)

    n_keep = int(keep.sum())
    C_k_input = C[np.ix_(keep, keep)]

    # ---- recompute on survivors (Algorithm 1: "Recalculate ... ") ---------
    # A restricted `pairs` set MUST be carried through the refit. Passing None
    # here silently reverts to an all-pairs fit on the survivors, which makes any
    # pair-restricted variant a no-op the moment exclusion fires (it fires on
    # 23/25 in-scope cells). Found in review, 2026-07-26.
    sub_pairs, pairs_degraded = None, False
    if pairs is not None and n_keep < m:
        remap = {old: new for new, old in enumerate(np.where(keep)[0])}
        sub_pairs = [(remap[i], remap[j]) for (i, j) in pairs
                     if keep[i] and keep[j]]
        A_chk, _ = additive_design(n_keep, sub_pairs) if sub_pairs else (None, None)
        if A_chk is None or np.linalg.matrix_rank(A_chk) < n_keep:
            # the surviving pair graph no longer identifies rho; fall back to all
            # pairs on the survivors and SAY SO rather than returning a wrong fit
            sub_pairs, pairs_degraded = None, True
    elif pairs is not None:
        sub_pairs = pairs

    if recompute_after_exclusion and n_keep < m:
        C_k, res_k, g2_k, rho_k = _fit_block(C_k_input, sub_pairs, var_y)
    else:
        C_k = C_k_input
        g2_k, res_k = g2_full, res_full
        if n_keep < m:
            A_k, prs_k = additive_design(n_keep, sub_pairs)
            b_k = np.array([C_k[i, j] for i, j in prs_k], dtype=float)
            rho_k = solve_additive(A_k, b_k, loss=loss) + 0.5 * g2_k
        else:
            rho_k = rho_full

    # ---- Eq. 21 weights, or the Sec. 4.3 simple-average fallback ----------
    used_mean = bool(simple_avg_fallback and n_keep < min_experts_for_eq21)
    if used_mean:
        w_k = np.ones(n_keep) / n_keep
        evals_k = np.array([np.nan])
    else:
        k2 = min(n_components, n_keep)
        ev, evec = eigh(C_k, subset_by_index=[n_keep - k2, n_keep - 1])
        evals_k, evecs_k = ev[::-1], evec[:, ::-1]
        w_k = np.zeros(n_keep)
        for cix in range(k2):
            vc = evecs_k[:, cix]
            w_k += (vc @ rho_k) / (evals_k[cix] + 1e-12) * vc

    if abstained and on_abstain == "mean":
        w_k = np.ones(n_keep) / n_keep
        used_mean = True
    elif abstained and on_abstain == "nan":
        w_k = np.full(n_keep, np.nan)

    w = np.zeros(m)
    w[keep] = w_k
    rho_out = np.zeros(m)
    rho_out[keep] = rho_k

    return UPCRResult(
        w=w, rho_hat=rho_out, rho_hat_full=rho_full,
        g2_hat=float(g2_k), var_y=float(var_y), keep=keep,
        abstained=abstained, used_simple_average=used_mean,
        n_components_used=int(n_components), lambda2_frac=lambda2_frac,
        proj_residual=float(res_k),
        g2_at_ceiling=bool(g2_k >= var_y * (1.0 - 1.5 / max(int(g2_grid), 2))),
        g2_frac_of_var_y=float(g2_k / (var_y + 1e-12)),
        diag_mean=diag_mean,
        meta={"n_kept": n_keep, "m": m, "loss": loss,
              "scale_ratio": float(var_y / (diag_mean + 1e-12)),
              "trace_C": trace_C, "evals_top2": evals_probe.tolist(),
              "g2_full_pool": float(g2_full),
              "proj_residual_full_pool": float(res_full),
              "pairs_restricted": pairs is not None,
              "pairs_degraded_after_exclusion": pairs_degraded},
    )


def upcr_pipeline_faithful(feats_dict, feat_names, signs=None, *,
                           orient="signs", anchor=None, **kwargs):
    """Orient -> z-score -> :func:`upcr_fit` -> ensemble score.

    orient:
      'signs'  use the supplied hand-derived polarity dict (legacy behaviour).
      'derived' RELATIVE polarity from sign(rho) on unoriented features -- no hand
               signs, no subset. Measured at +1.46pp over the 42 hand polarities
               (20W/5L, p<0.001), and +7.10pp with exclusion off (23W/1L). The
               GLOBAL +-1 still has to come from ``anchor``; see below.
      'anchor' resolve the global sign against ``anchor`` via anchor_orient.

    THERE IS NO ANCHOR-FREE MODE, AND THERE CANNOT BE. rho is a function of C
    alone, and C(-F) == C(F) because a global flip flips both i and j in every
    product. Measured: max|d rho| under a global flip is 0.000e+00 across all 25
    in-scope cells. So the global +-1 is NOT identifiable from the covariance
    structure, and any rule reading rho for it is reading noise -- an earlier
    'rho leans positive' rule was wrong in 25/25 cells (0.2449 vs 0.7551).
    Something outside C must supply that one bit.

    Returns (score, UPCRResult).
    """
    if orient not in ("signs", "derived", "anchor"):
        raise ValueError("orient must be 'signs', 'derived' or 'anchor'; "
                         "'rho' was removed -- the global sign is provably not "
                         "recoverable from rho (see the docstring)")

    F = np.stack([zscore(np.asarray(feats_dict[f], dtype=float))
                  for f in feat_names], axis=0)
    if orient == "signs":
        if signs is None:
            raise ValueError("orient='signs' needs a signs dict")
        F = F * np.array([signs.get(f, 1) for f in feat_names], float)[:, None]
    elif orient == "derived":
        # one throwaway fit to read each feature's polarity off sign(rho)
        probe = upcr_fit(F, **kwargs)
        pol = np.sign(probe.rho_hat_full)
        pol[pol == 0] = 1.0
        F = F * pol[:, None]
        res_pol = pol
    else:
        res_pol = None

    res = upcr_fit(F, **kwargs)
    if orient == "derived":
        res.meta["derived_polarity"] = res_pol.tolist()
    score = res.w @ F

    # The global +-1 is not identifiable from C (see docstring), so it must be
    # resolved against something external whenever one is available.
    if anchor is not None:
        from .streaming_utils import anchor_orient
        score, flipped = anchor_orient(score, np.asarray(anchor, dtype=float))
        res.meta["global_flip"] = bool(flipped)
    elif orient == "anchor":
        raise ValueError("orient='anchor' needs an anchor array")

    return score, res
