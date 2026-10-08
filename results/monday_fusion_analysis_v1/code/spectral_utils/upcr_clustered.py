"""
upcr_clustered.py — U-PCR for DEPENDENT features. Our extension, not in either paper.

THE GAP THIS FILLS
------------------
    unsupervised ensemble        independent experts      dependent experts
    classification               SML (Parisi 2014)        L-SML (Jaffe/Fetaya/Nadler 2016)
    regression                   U-PCR (Dror 2017)        <- empty

U-PCR has no dependence handling at all: no clustering, no cluster-count knob, no
cluster-size knob. L-SML has clustering but only estimates a COUNT (K); it has no
size control either. Our features are heavily dependent — every in-scope cell has
30-100 feature pairs correlated above 0.75, with maxima of 0.996-1.000 — so the
empty cell is exactly the regime we are in.

THE IDEA
--------
U-PCR's Theorem 1 needs `E[h_i h_j] = 0` for ALL pairs, which buys the identity
`C_ij = g2 + a_i + a_j` — one equation per pair. Weaken it to hold only ACROSS
clusters, which is precisely L-SML's latent-variable model (features in different
groups are conditionally independent given Y; features in the same group are only
conditionally independent given their shared latent). Then:

  cross-cluster pair   C_ij = g2 + a_i + a_j            <- clean, keep it
  same-cluster pair    C_ij = g2 + a_i + a_j + E[h_i h_j] <- one extra unknown, drop it

So: FIT THE ADDITIVE SYSTEM ON CROSS-CLUSTER PAIRS ONLY. Delete the corrupted
equations instead of regularising around them.

WHY THIS SHOULD MATTER (the mechanism, stated so it can be refuted)
-------------------------------------------------------------------
For a near-duplicate pair, C_ij is close to 1. The additive model has nothing but
`a_i + a_j` to explain that with, so least squares inflates BOTH features' scores.
The near-duplicate confidence cluster (epr, epr_spilled, epr_energy,
mean_top1_logprob) therefore gets systematically over-weighted — which is the same
group Step 203 flagged as worst-fitting, and a precise mechanism for the
top-5-overlap-of-1/5 disagreement with the supervised weights.

Measured premise (Phase B2, `exp04_violation_map.py`): same-cluster pairs fit the
additive model 2.03x worse than cross-cluster pairs, in 25/25 cells, 24 at p<0.05.

IDENTIFIABILITY — the constraint neither paper supplies
--------------------------------------------------------
Each equation row is `e_i + e_j`. That design has full rank m only if the pair
graph is connected AND non-bipartite. The cross-cluster graph is complete
multipartite, so:

    K >= 3   contains a triangle -> non-bipartite -> rank m -> IDENTIFIABLE
    K == 2   complete bipartite  -> rank m-1     -> a is unidentifiable up to an
                                                    alternating shift along the split
    K == 1   no cross-cluster pairs at all       -> nothing to fit

This is the exact analogue of the paper's own "m >= 3 experts" (Assumption A4),
lifted from features to clusters. `check_identifiability` enforces it rather than
letting a rank-deficient solve return a plausible-looking wrong answer.
"""
import collections

import numpy as np
from scipy.linalg import eigh

from .upcr import upcr_fit, additive_design, UPCRResult
from .fusion_utils import detect_dependent_groups, zscore

__all__ = ["cross_cluster_pairs", "check_identifiability",
           "upcr_clustered_fit", "upcr_hierarchical_fit", "discover_clusters"]

MIN_CLUSTERS = 3          # identifiability floor, derived above
MIN_CLUSTER_SIZE_FOR_UPCR = 5   # the paper's own Sec. 4.3 rule, applied per cluster


def discover_clusters(F, loading_scale="complete", K_range=None):
    """L-SML spectral clustering over the features.

    loading_scale='complete' is not the historical default: Phase 0 showed the
    unit-norm loading inflates the Eq.14 misfit by group size, piling K up at the
    ceiling (15/25 cells at K>=7), while the literal sqrt(lam1) fix over-corrects
    and drops 6/25 cells to K=2 — where this variant is not identifiable.
    'complete' keeps every cell at K>=3.
    """
    m = F.shape[0]
    K, c, residual, _ = detect_dependent_groups(
        [F[i] for i in range(m)], K_range=K_range, method="residual",
        loading_scale=loading_scale)
    return int(K), np.asarray(c, dtype=int), float(residual)


def cross_cluster_pairs(c):
    """Pairs (i, j) whose features live in DIFFERENT clusters — the equations
    the weakened assumption still licenses."""
    c = np.asarray(c)
    m = len(c)
    return [(i, j) for i in range(m) for j in range(i + 1, m) if c[i] != c[j]]


def check_identifiability(c, m=None):
    """Can the cross-cluster-only system be solved uniquely on this partition?"""
    c = np.asarray(c)
    m = len(c) if m is None else m
    sizes = collections.Counter(c)
    K = len(sizes)
    pairs = cross_cluster_pairs(c)
    if not pairs:
        return {"ok": False, "K": K, "reason": "no cross-cluster pairs (K=1)",
                "n_cross_pairs": 0, "rank": 0, "sigma_min": 0.0,
                "cluster_sizes": sorted(sizes.values())}
    A, _ = additive_design(m, pairs)
    sv = np.linalg.svd(A, compute_uv=False)
    rank = int((sv > 1e-9).sum())
    ok = K >= MIN_CLUSTERS and rank == m
    reason = ("ok" if ok else
              f"K={K} < {MIN_CLUSTERS} (bipartite/degenerate pair graph)"
              if K < MIN_CLUSTERS else
              f"cross-cluster design is rank {rank} < m={m}")
    return {"ok": bool(ok), "K": K, "reason": reason,
            "n_cross_pairs": len(pairs), "rank": rank,
            "sigma_min": float(sv[-1]), "cluster_sizes": sorted(sizes.values())}


def upcr_clustered_fit(F, *, groups=None, loading_scale="complete",
                       require_identifiable=True, **kwargs):
    """U-PCR with the additive system restricted to cross-cluster pairs.

    Everything downstream of the pair-fit — the g2 grid, exclusion, Eq. 21 — is the
    faithful implementation, unchanged. The ONLY difference from `upcr_fit` is
    which equations enter the estimate of rho. That is deliberate: it isolates the
    dependence fix from every other moving part, so a delta can be attributed.

    Returns (UPCRResult, info) where info carries the partition and the
    identifiability verdict.
    """
    F = np.asarray(F, dtype=float)
    m = F.shape[0]
    if groups is None:
        K, c, residual = discover_clusters(F, loading_scale=loading_scale)
    else:
        c = np.asarray(groups, dtype=int)
        K, residual = len(np.unique(c)), float("nan")

    ident = check_identifiability(c, m)
    ident["lsml_residual"] = residual
    if require_identifiable and not ident["ok"]:
        raise ValueError(f"cross-cluster system not identifiable: {ident['reason']}")

    res = upcr_fit(F, pairs=cross_cluster_pairs(c), **kwargs)
    res.meta["clusters"] = c.tolist()
    res.meta["identifiability"] = ident
    return res, ident


def upcr_hierarchical_fit(F, *, groups=None, loading_scale="complete",
                          min_cluster_size=MIN_CLUSTER_SIZE_FOR_UPCR,
                          require_identifiable=True, **kwargs):
    """Two-level U-PCR, the direct analogue of L-SML's own algorithm.

    Within a cluster, features are conditionally independent GIVEN that cluster's
    latent variable, so U-PCR is legitimate there — it just estimates weights for
    the latent rather than for Y. Clusters below `min_cluster_size` fall back to a
    plain average, which is the paper's own Sec. 4.3 rule (missing from the legacy
    implementation, and load-bearing here because K=3..8 over ~29 features leaves
    several small clusters in every cell).

    Across clusters the virtual experts ARE independent by construction, so the
    outer U-PCR runs on all pairs — no restriction needed. It still needs K >= 3
    experts, which is the same identifiability floor by a different route.
    """
    F = np.asarray(F, dtype=float)
    m, n = F.shape
    if groups is None:
        K, c, residual = discover_clusters(F, loading_scale=loading_scale)
    else:
        c = np.asarray(groups, dtype=int)
        K, residual = len(np.unique(c)), float("nan")

    ident = check_identifiability(c, m)
    ident["lsml_residual"] = residual
    if require_identifiable and not ident["ok"]:
        raise ValueError(f"hierarchical U-PCR not identifiable: {ident['reason']}")

    virtual, inner_w, order = [], [], []
    for g in np.unique(c):
        idx = np.where(c == g)[0]
        order.append(idx)
        if len(idx) == 1:
            w_in = np.array([1.0])
        elif len(idx) < min_cluster_size:
            w_in = np.ones(len(idx)) / len(idx)          # Sec. 4.3 fallback
        else:
            sub = upcr_fit(F[idx], **kwargs)
            w_in = sub.w
            if not np.isfinite(w_in).all() or np.allclose(w_in, 0):
                w_in = np.ones(len(idx)) / len(idx)
        inner_w.append(w_in)
        virtual.append(zscore(w_in @ F[idx]))            # re-standardise per level

    Xi = np.asarray(virtual)                              # (K, n)
    outer = upcr_fit(Xi, **kwargs)

    # fold the two levels into one per-feature weight vector
    w = np.zeros(m)
    for gi, idx in enumerate(order):
        w[idx] = inner_w[gi] * outer.w[gi]

    res = UPCRResult(
        w=w, rho_hat=np.zeros(m), rho_hat_full=np.zeros(m),
        g2_hat=outer.g2_hat, var_y=outer.var_y, keep=np.ones(m, bool),
        abstained=outer.abstained, used_simple_average=outer.used_simple_average,
        n_components_used=outer.n_components_used,
        lambda2_frac=outer.lambda2_frac, proj_residual=outer.proj_residual,
        g2_at_ceiling=outer.g2_at_ceiling,
        g2_frac_of_var_y=outer.g2_frac_of_var_y, diag_mean=outer.diag_mean,
        meta={"clusters": c.tolist(), "identifiability": ident,
              "outer_weights": outer.w.tolist(),
              "n_clusters_averaged": int(sum(1 for i in order
                                             if 1 < len(i) < min_cluster_size)),
              "level": "hierarchical"},
    )
    return res, ident
