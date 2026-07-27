"""
Statistical fusion utilities for the Spectral Meta-Learner (SML) framework
of Parisi-Nadler-Kluger [PNAS 2014] and Jaffé-Fetaya-Nadler [2016].

Two SML variants are provided:
  * sml_fuse:    direct rank-1 estimator — leading eigenvector of off-diagonal
                 covariance R_off (Paper 2 Lemma 1, theoretically pure).
  * nadler_fuse: M-matrix variant attributed to Parisi et al. 2014 PNAS,
                 which reweights the rank-1 signal by the precision matrix C⁻¹.
                 Empirically gives more peaked weights than sml_fuse.

Both assume binary ±1 classifier inputs for theoretical guarantees;
continuous z-scored arrays are accepted as an empirical adaptation.

Key design decisions
--------------------
* z-score normalization is applied INSIDE best_nadler_on, after sign orientation
  and BEFORE the covariance matrix is computed.  This ensures SML weights
  reflect statistical complementarity, not feature scale.  The Spearman ρ filter
  is rank-invariant and therefore unaffected.

* simple_average_fusion is provided alongside the SML variants so every
  experiment can report "SML Lift" = AUC_SML − AUC_mean over the same subset.
  This directly justifies the use of the more complex SML algorithm.

* best_nadler_on switches between sml_fuse and nadler_fuse based on the
  `binarize` flag: binarize=True selects sml_fuse (Lemma 1 exact, paper-
  aligned); binarize=False keeps nadler_fuse (M-matrix, backward-compatible
  with pre-Step-105 consolidated results).
"""
import itertools

import numpy as np
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# ── Normalization ──────────────────────────────────────────────────────────────

def zscore(arr: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit-variance standardization.
    Returns mean-centered array if std < 1e-8 (constant feature).

    NON-FINITE INPUT IS HANDLED EXPLICITLY. With a NaN anywhere in ``arr``,
    ``arr.std()`` is NaN and ``NaN > 1e-8`` is False, so the old code silently
    took the constant-feature branch and returned an all-NaN array — a feature
    that looks standardized and poisons every covariance it enters. The current
    25-cell pool has no non-finite column (measured, Step 205), so this is
    defensive; the point is that it now fails loudly instead of quietly.
    """
    arr = np.asarray(arr, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(
            f"zscore received {int((~np.isfinite(arr)).sum())} non-finite value(s) "
            f"out of {arr.size}. Fix the feature upstream — silently returning NaNs "
            f"here would propagate into every covariance downstream.")
    std = arr.std()
    return (arr - arr.mean()) / std if std > 1e-8 else arr - arr.mean()


# ── AUC with bootstrap CI ─────────────────────────────────────────────────────

def _fast_auc(y, s):
    """
    Exact AUROC via the Mann–Whitney U rank statistic (ties → average ranks).
    Numerically identical to sklearn.metrics.roc_auc_score for binary y, but
    ~50–100× faster inside a bootstrap loop (one argsort + bincount, no ROC-curve
    construction). Assumes y ∈ {0,1} with BOTH classes present and no NaNs —
    callers pre-filter and guard the single-class bootstrap case.
    """
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y)
    n = len(s)
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    is_start = np.empty(n, dtype=bool)
    is_start[0] = True
    np.not_equal(sorted_s[1:], sorted_s[:-1], out=is_start[1:])
    grp = np.cumsum(is_start) - 1                 # tie-group id per sorted position
    cnt = np.bincount(grp)
    first = np.zeros_like(cnt)
    first[1:] = np.cumsum(cnt)[:-1]               # first 0-indexed sorted pos of each group
    avg_rank_grp = first + (cnt + 1) / 2.0        # 1-indexed average rank per group
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg_rank_grp[grp]
    ypos = y > 0.5
    n_pos = int(ypos.sum())
    n_neg = n - n_pos
    r_pos = ranks[ypos].sum()
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def boot_auc(y, scores, n: int = 1000):
    """
    Bootstrap AUROC with 95% confidence interval.

    Returns (auc, ci_lo, ci_hi).  Returns (nan, nan, nan) if the label array
    has only one class or the scores are constant.
    NaN rows in y or scores are silently dropped before computing AUROC.
    """
    y, s = np.array(y, dtype=float), np.array(scores, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(s))
    y, s = y[mask], s[mask]
    if len(y) < 2 or len(np.unique(y)) < 2 or np.std(s) < 1e-8:
        return float("nan"), float("nan"), float("nan")

    base = roc_auc_score(y, s)
    rng  = np.random.default_rng(42)
    boots = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        yi = y[idx]
        npos = yi.sum()
        if npos == 0 or npos == len(yi):
            continue
        boots.append(_fast_auc(yi, s[idx]))

    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (base, base)
    return base, lo, hi


def paired_boot_delta_auc(y, scores_a, scores_b, n: int = 1000,
                          seed: int = 42, clusters=None):
    """
    Paired bootstrap CI for the AUROC difference of two scores on the SAME
    samples (delta = AUC(a) - AUC(b)).

    Joint index resampling keeps the pairing, so shared sample noise cancels —
    this is the correct test for "method a beats method b on this cell",
    unlike comparing two marginal boot_auc CIs.  With clusters given (e.g.
    question ids for K>1 sampling caches), whole clusters are resampled to
    respect within-question correlation.

    Returns (delta, ci_lo, ci_hi); (nan, nan, nan) if either score is
    degenerate or y is single-class.
    """
    y = np.asarray(y, dtype=float)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(a) | np.isnan(b))
    y, a, b = y[mask], a[mask], b[mask]
    if len(y) < 2 or len(np.unique(y)) < 2 or a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan"), float("nan"), float("nan")

    delta = roc_auc_score(y, a) - roc_auc_score(y, b)
    rng = np.random.default_rng(seed)
    if clusters is not None:
        clusters = np.asarray(clusters)[mask]
        uniq = np.unique(clusters)
        members = {c: np.nonzero(clusters == c)[0] for c in uniq}
    boots = []
    for _ in range(n):
        if clusters is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            picked = uniq[rng.integers(0, len(uniq), len(uniq))]
            idx = np.concatenate([members[c] for c in picked])
        yi = y[idx]
        npos = yi.sum()
        if npos == 0 or npos == len(yi):
            continue
        boots.append(_fast_auc(yi, a[idx]) - _fast_auc(yi, b[idx]))
    if not boots:
        return float(delta), float(delta), float(delta)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(delta), float(lo), float(hi)


# ── Binarization ──────────────────────────────────────────────────────────────

def binarize_classifiers(feats_dict: dict, signs: dict,
                         quantiles: dict = None) -> dict:
    """
    Convert continuous uncertainty features into binary ±1 classifiers.

    Each feature is sign-oriented then thresholded at its empirical quantile
    (default: median, q=0.50), yielding a binary classifier fᵢ ∈ {−1, +1}.
    A prediction of +1 means "likely correct"; −1 means "likely wrong."

    This binarization satisfies the input assumption of the Spectral
    Meta-Learner (SML, Parisi-Nadler-Kluger [PNAS 2014]).  Under Lemma 1
    of that paper, the off-diagonal entries of the covariance matrix of
    binary ±1 classifiers form a rank-1 matrix vvᵀ where vᵢ ∝ (2αᵢ − 1)·√(1−b²),
    with αᵢ the balanced accuracy of classifier i and b the true class
    imbalance (a dataset property, not affected by the median split).

    The median split (q=0.50) produces a balanced prediction distribution
    (50 % +1, 50 % −1), the standard convention for unsupervised classifier
    construction.  An informative feature will yield αᵢ > 0.5 after this
    thresholding; a non-informative feature will yield αᵢ ≈ 0.5 and receive
    near-zero SML weight automatically.

    Offline-calibrated thresholds (quantiles != None): if per-feature quantiles
    are supplied (derived once from historical labeled data and then fixed),
    the method remains unsupervised at test time.  The quantile is applied to
    the oriented feature distribution of the current test set, so the threshold
    adapts to local scale while the split point is externally calibrated.

    Args:
        feats_dict: {feature_name: np.ndarray} of continuous feature values.
        signs:      {feature_name: +1 | −1}.
                    +1 = higher raw value → more likely correct.
                    −1 = higher raw value → more likely wrong (flip sign).
        quantiles:  Optional {feature_name: float in (0,1)}.
                    Per-feature quantile for the binarization threshold.
                    Missing keys and None fall back to 0.50 (median).

    Returns:
        {feature_name: np.ndarray of ±1.0 values, same length as input}
    """
    _q = quantiles or {}
    binary = {}
    for name, arr in feats_dict.items():
        s = signs.get(name, +1)
        oriented = np.array(arr, dtype=float) * s
        q = _q.get(name, 0.5)
        thr = np.quantile(oriented, q)
        binary[name] = np.where(oriented > thr, 1.0, -1.0)
    return binary


# ── Fusion algorithms ──────────────────────────────────────────────────────────

def sml_fuse(*classifiers: np.ndarray) -> tuple:
    """
    Spectral Meta-Learner (SML) — Parisi-Nadler-Kluger [PNAS 2014].

    Estimates the balanced accuracy αᵢ of each binary ±1 classifier from
    the rank-1 structure of the off-diagonal covariance matrix (Lemma 1):

        R_off ≈ v vᵀ,   vᵢ = √(1 − b²) · (2αᵢ − 1)

    The leading eigenvector of R_off is proportional to (2α − 1), so
    weights are proportional to classifier balanced accuracies.  The fused
    score is a weighted vote: score = Σᵢ wᵢ · fᵢ(x).

    This is the theoretically pure SML variant.  For the M-matrix variant
    from the same paper (Parisi et al. 2014), see nadler_fuse().

    Assumes classifiers make conditionally independent errors (Eq. 1 of
    Parisi et al. 2014).  Strongly correlated pairs should be filtered
    before calling — see the Spearman ρ filter in best_nadler_on(), which
    approximates the full dependent-classifier detection of Jaffé-Fetaya-
    Nadler [2016] (Section 3).

    Args:
        *classifiers: Binary ±1 np.ndarrays, all of length n_samples.
                      Must be sign-oriented so that +1 predicts correct.

    Returns:
        (fused_scores: np.ndarray, weights: np.ndarray)
        fused_scores = X @ weights  (continuous weighted vote)
        weights      = L1-normalized |leading eigenvector of R_off|
    """
    X = np.column_stack(classifiers)
    _n, k = X.shape
    R = np.cov(X.T)
    if R.ndim == 0:
        R = np.array([[float(R)]])
    R_off = R - np.diag(np.diag(R))
    try:
        _, vecs = eigh(R_off)
        w = np.abs(vecs[:, -1])
        w /= w.sum() + 1e-12
    except Exception:
        w = np.ones(k) / k
    return X @ w, w


def nadler_fuse(*views) -> tuple:
    """
    Spectral Meta-Learner — M-matrix variant (Parisi-Nadler-Kluger, PNAS 2014).

    Implements the M-matrix weight construction from Parisi et al. [2014],
    which reweights the rank-1 covariance signal by the precision matrix C⁻¹.
    For the theoretically pure SML (direct leading eigenvector of off-diagonal
    covariance R_off), see sml_fuse().

    Input contract: expects sign-oriented arrays.  For full theoretical
    alignment with Lemma 1, inputs should be binary ±1 classifiers produced
    by binarize_classifiers().  Continuous z-scored arrays are accepted as an
    empirical adaptation that preserves the rank-1 covariance intuition but
    lacks the binary classifier guarantee.

    Algorithm:
        1. Stack views into matrix X (n_samples × k).
        2. Compute sample covariance C.
        3. Build M = diag(row_sums_off_diag) @ C⁻¹ @ diag(col_sums_off_diag).
        4. Weights = |leading eigenvector of M|, L1-normalized.
        5. Fused score = X @ weights.

    Returns:
        (fused_scores: np.ndarray, weights: np.ndarray)
    """
    X = np.column_stack(views)
    _n, k = X.shape
    C = np.cov(X.T)
    if C.ndim == 0:
        C = np.array([[float(C)]])
    try:
        off    = C - np.diag(np.diag(C))
        rs, cs = off.sum(1), off.sum(0)
        M      = np.diag(rs) @ np.linalg.pinv(C) @ np.diag(cs)
        _, vecs = eigh(M)
        w = np.abs(vecs[:, -1])
        w /= w.sum() + 1e-12
    except Exception:
        w = np.ones(k) / k
    return X @ w, w


def simple_average_fusion(*views) -> tuple:
    """
    Unweighted mean fusion (equal weights baseline).

    Expects z-scored, sign-oriented feature arrays (same input contract as
    sml_fuse / nadler_fuse).  Used to compute the SML Lift over equal-weight
    ensemble: Lift = AUC_SML − AUC_mean.

    Returns:
        (fused_scores: np.ndarray, weights: np.ndarray)
    """
    X = np.column_stack(views)
    k = X.shape[1]
    w = np.ones(k) / k
    return X @ w, w


# ── L-SML: Latent Spectral Meta-Learner (Jaffé-Fetaya-Nadler 2016) ─────────────

def sml_fuse_signed(*classifiers: np.ndarray) -> tuple:
    """
    SML with signed weights and ±1 sign resolution via Paper 2 assumption (iii).

    Like sml_fuse, but keeps the leading eigenvector's sign rather than taking
    |·|.  Used in the fully unsupervised pipeline where classifiers are NOT
    pre-oriented: the sign of each weight vᵢ encodes the classifier's natural
    orientation (positive = informative as-is, negative = inversely informative).

    Sign resolution (Paper 2 assumption (iii) — "most classifiers beat random"):
    if fewer than half of v's components are positive, flip the global sign of v.

    Args:
        *classifiers: Binary ±1 arrays of length n_samples.

    Returns:
        (fused_scores, signed_weights)
    """
    X = np.column_stack(classifiers)
    _n, k = X.shape
    R = np.cov(X.T)
    if R.ndim == 0:
        R = np.array([[float(R)]])
    R_off = R - np.diag(np.diag(R))
    try:
        _, vecs = eigh(R_off)
        v = vecs[:, -1]
        if np.sum(v > 0) < k / 2:
            v = -v
        elif k % 2 == 0 and int(np.sum(v > 0)) == k // 2:
            # EXACT TIE, and the majority rule cannot break it: at even k with
            # exactly k/2 positive entries the test above is False for BOTH +v
            # and -v, so the returned sign is whatever LAPACK happened to
            # produce — which varies with BLAS build and CPU, not with the data.
            # Fires on real data (1 of 52 group calls on the GOOD_6 path: a k=2
            # group with v = [-0.707, +0.707]). Break it on the first non-zero
            # component so the answer is a function of the data alone.
            #
            # This CANNOT move a published number: L-SML is exactly gauge-
            # invariant to per-group sign, re-verified in Step 205 by flipping
            # EVERY group's v (GOOD_6 moved 0.00pp on 0/25 cells). It is a
            # cross-environment reproducibility fix, not a correctness fix.
            nz = np.flatnonzero(np.abs(v) > 1e-12)
            if nz.size and v[nz[0]] < 0:
                v = -v
    except Exception:
        v = np.ones(k) / k
    return X @ v, v


def _score_matrix_lsml(R: np.ndarray) -> np.ndarray:
    """
    Paper 1 Eq. (15): s_ij = Σ_{k,l ≠ i,j} |r_ij·r_kl − r_il·r_kj|.
    Large values indicate dependent (same-group) classifier pairs.

    Vectorised over (k, l): for a fixed pair (i, j) the summand matrix is
    ``A = |R[i,j]*R - outer(R[:,j], R[i,:])|``, and the excluded index sets are
    removed by subtracting the corresponding rows, columns and diagonal rather
    than by branching. Identical output to the original quadruple loop, ~100x
    faster at m=30 (the loop dominated every subset sweep).

    m < 4 IS SPECIAL, and getting it wrong cost this project a set of published
    numbers. The double sum is EMPTY there: {i, j} already uses two of at most
    three indices, k is forced to the remaining one, and then no l is left that
    differs from i, j and k. The original loop therefore returned exactly 0.0.
    The vectorised form computes the same empty sum as a difference of large
    partial sums, and catastrophic cancellation leaves ~1e-17 of float noise
    instead. Spectral clustering of an all-zero similarity is arbitrary either
    way, but the noise makes it arbitrary DIFFERENTLY: on a real 3-feature subset
    the assignment flipped [0,0,1] -> [1,0,1], the Eq.14 residual moved
    0.00297 -> 0.05284, and the fused AUROC moved 0.6023 -> 0.5891. That silently
    invalidated every size-3 row in results/selector_bench/ written before the
    vectorisation landed (Step 203) — which is most of the a1_residual family,
    since the residual-guided selectors converge on 3 features.

    Returning the exact zero restores the documented equality with the loop. It
    does NOT make L-SML meaningful at m=3: with three features Eq.15 carries no
    information at all, so the group assignment is whatever the clustering's
    tie-break yields. Treat any size-3 fusion as structurally undetermined.
    """
    m = R.shape[0]
    s = np.zeros((m, m))
    if m < 4:
        return s
    for i in range(m):
        Ri = R[i, :]
        for j in range(i + 1, m):
            A = np.abs(R[i, j] * R - np.outer(R[:, j], Ri))
            rows = A.sum(axis=1)
            # k ranges over all indices except i and j
            total = rows.sum() - rows[i] - rows[j]
            # for those k, drop l == i, l == j and l == k
            col_i = A[:, i].sum() - A[i, i] - A[j, i]
            col_j = A[:, j].sum() - A[i, j] - A[j, j]
            dg = np.trace(A) - A[i, i] - A[j, j]
            s[i, j] = s[j, i] = total - col_i - col_j - dg
    return s


def _spectral_cluster_precomputed(similarity: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    """Spectral clustering on a precomputed similarity matrix."""
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(
        n_clusters=K, affinity='precomputed',
        assign_labels='kmeans', random_state=seed,
    )
    return sc.fit_predict(similarity + 1e-12)


# ---------------------------------------------------------------------------
# small-m exact solve — Step 205
# ---------------------------------------------------------------------------
# Spectral clustering is a HEURISTIC for "partition minimising the Eq.14
# residual". At small m the heuristic ties, and the tie is settled by float
# noise rather than by the data:
#
#   Eq.15's double sum has ZERO terms at m=3 (the score matrix is identically
#   zero, so the partition is pure tie-break) and exactly TWO at m=4. At m=4 the
#   score matrix barely separates partitions, and a 5.55e-17 difference in R —
#   the gap between np.cov on a non-contiguous column slice and on a
#   column_stack copy, i.e. BLAS summation order alone — flips the K=3 partition
#   from [0,1,2,0] (residual 0.6018) to [0,1,0,2] (0.3927). The residual grid
#   then prefers a different K, and the fused AUROC moves 9.7pp on
#   lapeigvals_gsm8k_phi35 / consensus_4.
#
# At m <= 4 there is no reason to approximate: Bell(3)=5 and Bell(4)=15, so we
# enumerate every partition and take the exact argmin, with a deterministic
# tie-break. Measured over all 30 all-size-4 bench rows this is +0.03pp
# (15W/10L, p=0.696) — determinacy is free, accuracy was never on offer here.
#
# WHY THE CUTOFF IS 4 AND NOT 5. Exhaustive is affordable at m=5 too (Bell=52)
# and does find lower residuals there, because spectral clustering is only a
# heuristic for this argmin at every m. But m=5 shows no DETERMINACY defect:
# the Step-205 stability audit puts the 5-6 band at 0.000pp median spread under
# a 1e-10 jitter, with 3 of 87 rows above 0.5pp, against 0.439pp median and 18
# of 36 at m=4. Extending the exact solve to m=5 was measured (+0.22pp on that
# band) and DELIBERATELY NOT ADOPTED: it would move a published reference anchor
# (ref.LOCO_5 0.7705 -> 0.7673) to fix something that is not broken. "Is spectral
# clustering a good optimiser for Eq.14 at larger m" is a real question, but it
# is a research question, not this defect — and Steps 203/204 already found the
# residual criterion itself to be a poor guide, so a better optimiser for it is
# not self-evidently an improvement.
SMALL_M_EXACT = 4
DEGENERATE_REL_TOL = 1e-9


def _canonical_partitions(m: int, k_allowed):
    """Every set partition of ``m`` items whose block count is in ``k_allowed``.

    Emitted as restricted-growth strings (``c[0] == 0`` and
    ``c[i] <= max(c[:i]) + 1``), which are canonical — one string per partition,
    no relabelling duplicates — and generated in lexicographic order. That order
    IS the tie-break: the search keeps a candidate only on a strict improvement,
    so among equal residuals the lexicographically first wins, deterministically.
    """
    k_allowed = set(int(k) for k in k_allowed)
    out, cur = [], [0] * m

    def rec(i, mx):
        if i == m:
            if (mx + 1) in k_allowed:
                out.append(tuple(cur))
            return
        for v in range(mx + 2):
            cur[i] = v
            rec(i + 1, max(mx, v))

    rec(1, 0)
    return out


def _exact_small_m_groups(R, K_range, loading_scale):
    """Exact Eq.14 residual argmin over all partitions with K in ``K_range``.

    Returns ``(best_K, assignment, best_residual, curve, degenerate)`` where
    ``curve`` mirrors the spectral path's per-K entries (best partition found at
    each K) and ``degenerate`` says the winner was not separated from the runner
    up by more than float noise — i.e. this answer is a coin flip and any number
    derived from it is one draw, not a measurement.
    """
    m = R.shape[0]
    parts = _canonical_partitions(m, K_range)
    if not parts:
        return None, None, float('inf'), [], True

    scored = [(_residual_lsml(R, np.asarray(p, dtype=int),
                              loading_scale=loading_scale), p) for p in parts]
    best_r, best_p = scored[0]
    for r, p in scored[1:]:
        if r < best_r:                       # strict: lexicographic first wins ties
            best_r, best_p = r, p

    per_K = {}
    for r, p in scored:
        k = len(set(p))
        if k not in per_K or r < per_K[k][0]:
            per_K[k] = (r, p)
    curve = [(k, per_K[k][0], np.asarray(per_K[k][1], dtype=int))
             for k in sorted(per_K)]

    gap = _rel_gap([r for r, p in scored if p != best_p], best_r)
    degenerate = bool(gap is not None and gap <= DEGENERATE_REL_TOL)

    return (len(set(best_p)), np.asarray(best_p, dtype=int), best_r,
            curve, degenerate, gap)


def _rel_gap(others, best):
    """Separation between the winner and the best alternative, relative to scale.

    ``None`` when there is no alternative (nothing to be degenerate about).
    """
    alts = [r for r in others if np.isfinite(r) and r > best]
    runner_up = min(alts) if alts else None
    if runner_up is None:
        # every alternative ties or is non-finite: degenerate iff a finite tie exists
        return 0.0 if any(np.isfinite(r) for r in others) else None
    scale = max(abs(best), abs(runner_up), 1e-300)
    return float((runner_up - best) / scale)


def _diag(m, exact, gap, degenerate):
    return {'m': int(m), 'exact_small_m': bool(exact),
            'residual_gap_rel': gap, 'degenerate': bool(degenerate)}


def _pack(K, c, resid, s, curve, diag, return_curve, return_diag):
    """Assemble the return tuple. Legacy shape is (K, c, residual, s); ``curve``
    and then ``diag`` are appended only when asked for, so existing 4- and
    5-tuple unpacking at every call site keeps working."""
    out = (K, c, resid, s)
    if return_curve:
        out = out + (curve,)
    if return_diag:
        out = out + (diag,)
    return out


LOADING_SCALES = ('unit', 'eigen', 'complete')


def _rank1_masked(M: np.ndarray, unknown: np.ndarray, scale: str,
                  max_iter: int = 100, tol: float = 1e-12,
                  return_info: bool = False):
    """Fit a rank-one factor v to the OBSERVED entries of M (``unknown`` masks
    the entries Lemma 1 says nothing about), under one of three scalings.

    Lemma 1 constrains only ``i != j`` (and, for v^off, only cross-group pairs);
    every masked entry is a free parameter of the fit, not a zero.

    scale='unit'      leading eigenvector of M with masked entries zeroed,
                      NORMALISED TO 1. This is the historical behaviour and the
                      default everywhere — it is what every committed number in
                      the repo was computed with. It does NOT satisfy Lemma 1:
                      the unit vector obeys ``lam1 * v_i*v_j ~ r_ij``, so it is
                      short by a factor of ``sqrt(lam1)``.
    scale='eigen'     ``sqrt(lam1) * v`` — the literal fix proposed in
                      SPEC_residual_scaling_fix.md. Removes the group-size blow-up
                      but is still biased: zeroing the masked entries removes the
                      rank-one matrix's own diagonal, shrinking the estimate by
                      ``sqrt((m-1)/m)`` on a uniform block. Measured misfit/pair on
                      a perfect m-duplicate block: 0.2500 (m=2) -> 0.0083 (m=11).
    scale='complete'  masked-entry completion fixed point: fill the masked entries
                      with the current outer-product estimate, re-decompose, repeat.
                      Exact under Lemma 1 — misfit/pair ~1e-25 on a perfect
                      duplicate block for every m, and recovers unequal loadings to
                      ~1e-14. 3-30x lower misfit than 'eigen' under noise.

    The best iterate by observed-entry misfit is kept, so a non-converging or
    diverging fixed point can never return something worse than its start. Pass
    ``return_info=True`` to get (v, info) with the convergence record — the
    keep-best guard would otherwise MASK non-convergence silently, which on real
    covariance blocks happens often enough to be worth reporting (review finding).
    """
    S = np.array(M, dtype=float, copy=True)
    S[unknown] = 0.0
    try:
        vals, vecs = eigh(S)
    except Exception:
        return np.ones(S.shape[0]) / np.sqrt(S.shape[0])

    u = vecs[:, -1]
    if scale == 'unit':
        return (u, {'converged': True, 'iters': 0, 'guard_fired': False})             if return_info else u

    lam = max(float(vals[-1]), 0.0)
    v = u * np.sqrt(lam)
    if scale == 'eigen':
        return (v, {'converged': True, 'iters': 0, 'guard_fired': False})             if return_info else v

    if scale != 'complete':
        raise ValueError(f"loading_scale must be one of {LOADING_SCALES}, got {scale!r}")

    observed = ~unknown

    def _misfit(vec):
        return float((((np.outer(vec, vec) - M) ** 2)[observed]).sum())

    best_v, best_r = v, _misfit(v)
    converged, used_iters, guard_fired = False, 0, False
    for _it in range(max_iter):
        used_iters = _it + 1
        filled = np.where(unknown, np.outer(v, v), M)
        try:
            vals_i, vecs_i = eigh(filled)
        except Exception:
            guard_fired = True
            break
        lam_i = max(float(vals_i[-1]), 0.0)
        v_new = vecs_i[:, -1] * np.sqrt(lam_i)
        r_new = _misfit(v_new)
        if r_new < best_r:
            best_v, best_r = v_new, r_new
        else:
            guard_fired = True
        if np.linalg.norm(v_new - v) < tol:
            converged = True
            break
        v = v_new
    if return_info:
        return best_v, {'converged': converged, 'iters': used_iters,
                        'guard_fired': guard_fired}
    return best_v


def _estimate_von_voff(R: np.ndarray, c: np.ndarray,
                       loading_scale: str = 'unit') -> tuple:
    """
    Estimate v^on and v^off per Paper 1 Lemma 1.

    v^on_i for i ∈ group g: rank-one factor of the within-group submatrix of R,
    with the diagonal treated as unobserved (Lemma 1 constrains only i != j).
    v^off: rank-one factor of R with all within-group entries unobserved.

    ``loading_scale`` selects how the factor is scaled — see _rank1_masked.
    Default 'unit' reproduces every committed number in the repo.
    """
    m = R.shape[0]
    v_on = np.zeros(m)
    for g in np.unique(c):
        idx = np.where(c == g)[0]
        if len(idx) == 1:
            v_on[idx] = 1.0
            continue
        sub = R[np.ix_(idx, idx)]
        unknown = np.eye(len(idx), dtype=bool)
        try:
            v_on[idx] = _rank1_masked(sub, unknown, loading_scale)
        except ValueError:
            raise
        except Exception:
            v_on[idx] = 1.0 / np.sqrt(len(idx))

    # within-group entries (incl. the diagonal) are unobserved for v^off
    same = np.asarray(c)[:, None] == np.asarray(c)[None, :]
    try:
        v_off = _rank1_masked(R, same, loading_scale)
    except ValueError:
        raise
    except Exception:
        v_off = np.ones(m) / np.sqrt(m)

    return v_on, v_off


def _residual_lsml(R: np.ndarray, c: np.ndarray,
                   loading_scale: str = 'unit') -> float:
    """Paper 1 Eq. (14) residual under assignment c.

    Vectorised (was an O(m^2) Python double loop called once per candidate K,
    i.e. up to 7x per fusion). Bit-identical up to float summation order.

    NOTE: the residual's units change with ``loading_scale``, so absolute
    residual values are comparable only within one scale (SPEC §8).
    """
    v_on, v_off = _estimate_von_voff(R, c, loading_scale=loading_scale)
    c = np.asarray(c)
    same = c[:, None] == c[None, :]
    pred = np.where(same, np.outer(v_on, v_on), np.outer(v_off, v_off))
    d = (pred - R) ** 2
    np.fill_diagonal(d, 0.0)
    return float(d.sum())


def _eigengap_K(s: np.ndarray, max_K: int = 8, return_gaps: bool = False):
    """
    Eigengap heuristic on the normalized Laplacian of similarity matrix s.
    Returns K maximizing the gap between consecutive smallest Laplacian
    eigenvalues (K = index of largest gap + 1, with K ≥ 2). With
    ``return_gaps=True`` also returns the gap vector (selector-bench
    diagnostic, Step 186+).
    """
    m = s.shape[0]
    deg = s.sum(axis=1) + 1e-12
    D_inv_sqrt = 1.0 / np.sqrt(deg)
    S_norm = (D_inv_sqrt[:, None] * s) * D_inv_sqrt[None, :]
    L_norm = np.eye(m) - S_norm
    eigvals = np.sort(np.linalg.eigvalsh(L_norm))[:max_K + 1]
    gaps = np.diff(eigvals)
    K = max(int(np.argmax(gaps)) + 1, 2)
    if return_gaps:
        return K, gaps
    return K


def detect_dependent_groups(binary_classifiers, K_range=None, method: str = 'residual',
                            return_curve: bool = False, loading_scale: str = 'unit',
                            return_diag: bool = False):
    """
    Paper 1 Algorithm 1: detect groups of dependent binary classifiers.

    Algorithm:
        1. Compute m×m covariance matrix R of binary ±1 classifier outputs.
        2. Compute score matrix s_ij = Σ |r_ij·r_kl − r_il·r_kj| (Eq. 15).
           Large s_ij ⇒ classifiers i and j are likely in the same group.
        3. Spectral-cluster on s to obtain assignment c : {1,...,m} → {1,...,K}.
        4. Choose K either by:
             method='residual'  → minimise Eq. (14) residual over K_range
                                 (most paper-faithful, ~K_range × spectral)
             method='eigengap'  → Laplacian eigengap on score matrix
                                 (standard heuristic, single spectral run)

    Args:
        binary_classifiers: iterable of ±1 arrays, length n each.
        K_range:            iterable of K to try (default 2..min(m,8)).
        method:             'residual' or 'eigengap'.
        return_curve:       if True, also return the per-K residual curve
                            [(K, residual, assignment), ...] that the search
                            already computes (selector-bench diagnostic,
                            Step 186+; default False keeps the legacy 4-tuple).
        loading_scale:      'unit' (default, historical) | 'eigen' | 'complete'.
                            Passed to _residual_lsml, so with method='residual'
                            it changes WHICH K is selected. See _rank1_masked.
        return_diag:        append a diagnostics dict with keys ``m``,
                            ``exact_small_m``, ``residual_gap_rel`` and
                            ``degenerate``. ``degenerate=True`` means the chosen
                            grouping beat its nearest rival by less than float
                            noise, so the fused score derived from it is one draw
                            rather than a measurement (Step 205).

    At ``m <= SMALL_M_EXACT`` the K search is replaced by an EXACT enumeration of
    every partition — spectral clustering is only a heuristic for that argmin,
    and at small m the heuristic ties and the tie is broken by float noise. See
    :func:`_exact_small_m_groups`.

    Returns:
        (best_K, assignment_c, residual_at_best_K, score_matrix_s)
        then ``curve`` if return_curve, then ``diag`` if return_diag.
    """
    # np.cov is NOT bit-identical across memory layouts: a non-contiguous column
    # slice and a contiguous copy of the same numbers sum in a different order
    # and differ by ~5e-17. At small m that is enough to change the answer
    # (Step 205), so pin the layout here — the covariance is then a function of
    # the data alone, whatever the caller passed in.
    X = np.ascontiguousarray(np.column_stack(binary_classifiers), dtype=float)
    m = X.shape[1]
    R = np.cov(X.T)
    if R.ndim == 0:
        R = np.array([[float(R)]])
    s = _score_matrix_lsml(R)

    if K_range is None:
        # Cap at m-1 so K never equals m (all-singletons is degenerate: no
        # within-group fusion happens, and cross-group SML reduces to naive SML).
        K_range = list(range(2, min(m, 9)))
    K_range = list(K_range)

    if method == 'eigengap':
        K = _eigengap_K(s, max_K=max(K_range))
        try:
            c = _spectral_cluster_precomputed(s, K)
        except Exception:
            out = (1, np.zeros(m, dtype=int), float('inf'), s)
            return out + ([],) if return_curve else out
        r = _residual_lsml(R, c, loading_scale=loading_scale)
        out = (K, c, r, s)
        return out + ([(K, r, c)],) if return_curve else out

    if method != 'residual':
        raise ValueError(f"Unknown method {method!r}; use 'residual' or 'eigengap'.")

    # -- small m: solve exactly instead of clustering (see _exact_small_m_groups)
    if m <= SMALL_M_EXACT:
        best_K, best_c, best_resid, curve, degen, gap = _exact_small_m_groups(
            R, K_range, loading_scale)
        if best_K is None:
            return _pack(1, np.zeros(m, dtype=int), float('inf'), s, [],
                         _diag(m, True, None, True), return_curve, return_diag)
        return _pack(best_K, best_c, best_resid, s, curve,
                     _diag(m, True, gap, degen), return_curve, return_diag)

    curve = []
    best_K, best_c, best_resid = None, None, float('inf')
    for K in K_range:
        try:
            c = _spectral_cluster_precomputed(s, K)
        except Exception:
            continue
        r = _residual_lsml(R, c, loading_scale=loading_scale)
        curve.append((K, r, c))
        if r < best_resid:
            best_resid, best_K, best_c = r, K, c
    if best_K is None:
        return _pack(1, np.zeros(m, dtype=int), float('inf'), s, curve,
                     _diag(m, False, None, True), return_curve, return_diag)
    # Degeneracy detector, live at EVERY m — the generalisable half of the
    # Step-205 fix. If the winner is not separated from the runner-up by more
    # than float noise, the choice was decided in the last bits, and the number
    # downstream is one draw rather than a measurement. Report it; never absorb it.
    gap = _rel_gap([r for _K, r, _c in curve], best_resid)
    return _pack(best_K, best_c, best_resid, s, curve,
                 _diag(m, False, gap,
                       bool(gap is not None and gap <= DEGENERATE_REL_TOL)),
                 return_curve, return_diag)


def lsml_fuse(*binary_classifiers, K_range=None, method: str = 'residual',
              groups=None, loading_scale: str = 'unit'):
    """
    Latent SML (L-SML) — Paper 1 Algorithm 2 (Jaffé-Fetaya-Nadler 2016).

    Pipeline:
        1. Detect dependent classifier groups via detect_dependent_groups.
        2. Within each group g: run sml_fuse_signed on the group's binary
           classifiers, then binarize the weighted score to a ±1 virtual
           latent classifier ξ_g (Paper 1 Algorithm 2 line 5).
        3. Across groups: run sml_fuse_signed on the K virtual classifiers
           ξ_1, ..., ξ_K (which are conditionally independent by construction
           per the latent-variable model, Fig. 1 right of Paper 1).

    All inputs must be binary ±1.  Real labels never enter this function.

    Args:
        groups: optional precomputed group assignment (int array, length m).
            When given, step 1's spectral clustering is SKIPPED and this
            assignment is used verbatim — the clustering-swap seam for
            external group-discovery methods (selector bench, Step 186+).
            K, residual, and score_matrix are still computed for the meta
            dict so downstream consumers are unaffected.

    Returns:
        (fused_scores, meta_dict)
        meta_dict:
            K, c (group assignment), residual, method, score_matrix,
            group_weights (list of (idx_array, signed_weights)),
            cross_weights (signed weights across virtual classifiers),
            virtual_classifiers (n_samples × K binary array)
    """
    X = np.column_stack(binary_classifiers)
    n, m = X.shape

    if groups is not None:
        c = np.asarray(groups, dtype=int)
        if len(c) != m:
            raise ValueError(f"groups has length {len(c)}, expected m={m}")
        R = np.cov(X.T)
        if R.ndim == 0:
            R = np.array([[float(R)]])
        s_mat = _score_matrix_lsml(R)
        K = len(np.unique(c))
        residual = _residual_lsml(R, c, loading_scale=loading_scale)
    else:
        K, c, residual, s_mat = detect_dependent_groups(
            binary_classifiers, K_range=K_range, method=method,
            loading_scale=loading_scale,
        )

    virtual = []
    group_weights = []
    for g in np.unique(c):
        idx = np.where(c == g)[0]
        if len(idx) == 1:
            xi_g = X[:, idx[0]].astype(float)
            w = np.array([1.0])
        else:
            score, w = sml_fuse_signed(*[X[:, i] for i in idx])
            xi_g = np.sign(score)
            xi_g[xi_g == 0] = 1.0
        virtual.append(xi_g)
        group_weights.append((idx, w))

    virtual_arr = np.column_stack(virtual)

    if len(virtual) == 1:
        if len(group_weights[0][0]) > 1:
            fused = X[:, group_weights[0][0]] @ group_weights[0][1]
        else:
            fused = virtual[0]
        cross_w = np.array([1.0])
    else:
        fused, cross_w = sml_fuse_signed(*virtual)

    return fused, {
        'K': K, 'c': c, 'residual': residual, 'method': method,
        'score_matrix': s_mat,
        'group_weights': group_weights,
        'cross_weights': cross_w,
        'virtual_classifiers': virtual_arr,
    }


def lsml_continuous(*views, K_range=None, method: str = 'residual',
                    groups=None, compute_score_matrix: bool = True,
                    loading_scale: str = 'unit'):
    """
    Continuous L-SML — same group detection as lsml_fuse but skips binarization
    of virtual classifiers.

    Accepts z-scored continuous arrays instead of binary ±1.  The group
    detection step (detect_dependent_groups) runs on the continuous covariance
    matrix and is otherwise identical.  The within-group virtual classifier
    is the continuous weighted sum from sml_fuse_signed rather than np.sign
    of that sum.

    No theoretical guarantee from Parisi-Nadler-Kluger Lemma 1 (which assumes
    binary ±1 inputs), but preserves continuous signal that is lost by median
    binarization (~4 pp on math500 per local experiments).

    Args:
        *views:   z-scored, sign-oriented continuous np.ndarrays (n_samples each).
        K_range:  iterable of K values to try (default: range(2, min(m,9))).
        method:   'residual' or 'eigengap' — K-selection method.
        groups:   optional precomputed group assignment (int array, length m).
                  When given, the spectral-clustering detection is SKIPPED and
                  this assignment is used verbatim — the clustering-swap seam
                  for external group-discovery methods (selector bench,
                  Step 186+). K/residual/score_matrix still fill the meta dict.
        compute_score_matrix: only consulted when ``groups`` is given. The Eq.15
                  score matrix is O(m^4) and costs ~238 ms at m=30, yet nothing
                  downstream reads it on the groups-given path. Pass False to
                  skip it (meta['score_matrix'] becomes None) for a ~500x
                  speedup in subset sweeps. Default True preserves the old
                  meta dict exactly.

    Returns:
        (fused_scores, meta_dict) — same format as lsml_fuse.
    """
    X = np.column_stack(views)
    n, m = X.shape

    if groups is not None:
        c = np.asarray(groups, dtype=int)
        if len(c) != m:
            raise ValueError(f"groups has length {len(c)}, expected m={m}")
        R = np.cov(X.T)
        if R.ndim == 0:
            R = np.array([[float(R)]])
        s_mat = _score_matrix_lsml(R) if compute_score_matrix else None
        K = len(np.unique(c))
        residual = _residual_lsml(R, c, loading_scale=loading_scale)
        # groups were supplied, so nothing was chosen here and nothing can be
        # degenerate — say that explicitly rather than leaving the key absent.
        diag = {'m': int(m), 'exact_small_m': False, 'residual_gap_rel': None,
                'degenerate': False, 'groups_supplied': True}
    else:
        K, c, residual, s_mat, diag = detect_dependent_groups(
            views, K_range=K_range, method=method,
            loading_scale=loading_scale, return_diag=True,
        )
        diag = dict(diag, groups_supplied=False)

    virtual = []
    group_weights = []
    for g in np.unique(c):
        idx = np.where(c == g)[0]
        if len(idx) == 1:
            xi_g = X[:, idx[0]].astype(float)
            w = np.array([1.0])
        else:
            score, w = sml_fuse_signed(*[X[:, i] for i in idx])
            xi_g = score  # keep continuous — unlike lsml_fuse which applies np.sign
        virtual.append(xi_g)
        group_weights.append((idx, w))

    virtual_arr = np.column_stack(virtual)

    if len(virtual) == 1:
        if len(group_weights[0][0]) > 1:
            fused = X[:, group_weights[0][0]] @ group_weights[0][1]
        else:
            fused = virtual[0]
        cross_w = np.array([1.0])
    else:
        fused, cross_w = sml_fuse_signed(*virtual)

    return fused, {
        'K': K, 'c': c, 'residual': residual, 'method': method,
        'score_matrix': s_mat,
        'group_weights': group_weights,
        'cross_weights': cross_w,
        'virtual_classifiers': virtual_arr,
        # Step 205: True means the grouping this score rests on beat its nearest
        # rival by less than float noise. Callers that tabulate AUROC should
        # carry it, so a coin flip is never presented as a measurement.
        'grouping_diag': diag,
        'degenerate': bool(diag.get('degenerate', False)),
    }


def lsml_continuous_pipeline(feats_dict: dict, feat_names: list, signs: dict,
                              K_range=None, method: str = 'residual'):
    """
    Pipeline wrapper for lsml_continuous: orient, z-score, then fuse.

    Applies sign orientation from ``signs`` and z-scores each feature before
    calling lsml_continuous.  No binarization.

    Args:
        feats_dict: {feature_name: np.ndarray} of raw continuous features.
        feat_names: feature names to include.
        signs:      {feature_name: +1|-1} orientation dict.
        K_range, method: forwarded to lsml_continuous.

    Returns:
        (fused_scores, meta_dict) — same format as lsml_fuse.
    """
    views = []
    for f in feat_names:
        arr = np.array(feats_dict[f], dtype=float)
        s = signs.get(f, +1)
        views.append(zscore(arr * s))
    return lsml_continuous(*views, K_range=K_range, method=method)


def multipass_lsml_continuous(runs_feats: list, feat_names: list, signs: dict,
                              anchor_feature: str = 'epr', K_range=None,
                              method: str = 'residual'):
    """
    Hierarchical multi-pass fusion: continuous L-SML within each generation
    pass, then across passes.

    Stage 1 — per pass: lsml_continuous_pipeline over ``feat_names``, then the
    fused score's global ± (a leading-eigenvector coin flip, Step 148) is
    resolved label-free with anchor_orient against the sign-oriented, z-scored
    ``anchor_feature`` of that pass.

    Stage 2 — cross-pass: the oriented per-pass scores are z-scored into K
    score-views and fused with lsml_continuous, anchored against their mean.
    L-SML needs ≥ 3 views: with 2 passes 'fused' falls back to the simple
    average; with 1 pass it is that pass's score (so the same call serves the
    single-pass AUROC-vs-T curve).  'avg_fused' is always the equal-weight
    baseline over the score-views.

    Args:
        runs_feats:     list of feats_dicts, one per pass — each
                        {feature_name: np.ndarray}, index-aligned across passes
                        (same samples, same order).
        feat_names:     features fused within each pass.
        signs:          {feature_name: +1|-1} orientation dict.
        anchor_feature: anchor view for label-free global-sign resolution.
        K_range, method: forwarded to lsml_continuous.

    Returns dict:
        'per_pass_scores': list of oriented per-pass fused scores
        'fused':           cross-pass L-SML score (oriented; see fallbacks)
        'avg_fused':       equal-weight mean of the z-scored per-pass scores
        'rho':             K×K Spearman matrix of the per-pass scores
        'meta':            {'per_pass_meta', 'per_pass_flipped', 'cross_meta',
                            'cross_flipped', 'fallback'}
    """
    # streaming_utils depends only on feature_utils; import locally to keep
    # fusion_utils importable without the streaming module.
    from .streaming_utils import anchor_orient

    per_scores, per_metas, per_flips = [], [], []
    for feats in runs_feats:
        score, meta = lsml_continuous_pipeline(feats, feat_names, signs,
                                               K_range=K_range, method=method)
        anchor = zscore(np.asarray(feats[anchor_feature], dtype=float)
                        * signs.get(anchor_feature, +1))
        score, flipped = anchor_orient(score, anchor)
        per_scores.append(np.asarray(score, dtype=float))
        per_metas.append(meta)
        per_flips.append(flipped)

    k = len(per_scores)
    views = [zscore(s) for s in per_scores]
    avg_fused, _ = simple_average_fusion(*views)

    fallback, cross_meta, cross_flip = None, None, False
    if k >= 3:
        fused, cross_meta = lsml_continuous(*views, K_range=K_range, method=method)
        fused, cross_flip = anchor_orient(fused, avg_fused)
    elif k == 2:
        fused, fallback = avg_fused, 'K=2 passes < 3 views — simple average'
    else:
        fused, fallback = views[0], 'single pass — per-pass score returned'

    rho = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            r = spearmanr(per_scores[i], per_scores[j])[0]
            rho[i, j] = rho[j, i] = r

    return {
        'per_pass_scores': per_scores,
        'fused': np.asarray(fused, dtype=float),
        'avg_fused': avg_fused,
        'rho': rho,
        'meta': {'per_pass_meta': per_metas, 'per_pass_flipped': per_flips,
                 'cross_meta': cross_meta, 'cross_flipped': cross_flip,
                 'fallback': fallback},
    }


def sml_unsupervised(feats_dict: dict, feat_names: list,
                     K_range=None, method: str = 'residual'):
    """
    Pure unsupervised binary L-SML pipeline.

    Fully aligned with Parisi-Nadler-Kluger [PNAS 2014] (SML, Lemma 1) and
    Jaffé-Fetaya-Nadler [2016] (L-SML for dependent classifiers):

      Step 1.  Binarize each continuous feature at its empirical median
               to obtain a binary ±1 classifier.  NO sign orientation
               (orientation is resolved by sml_fuse_signed internally via
               Paper 2 assumption (iii)).
      Step 2.  L-SML: detect groups of dependent classifiers, run SML
               within each group to obtain a binary virtual classifier per
               group, then run SML across groups.
      Step 3.  Return continuous fused scores for downstream AUROC
               evaluation.  Real labels never used inside this function.

    All m features are always used — no subset search, no Spearman ρ
    filter, no label-based selection.

    Args:
        feats_dict: {feature_name: np.ndarray} of continuous features.
        feat_names: list of feature names to include (uses all of them).
        K_range:    iterable of K values to try (default 2..min(m,8)).
        method:     'residual' (Paper 1 Algorithm 1 — paper-faithful) or
                    'eigengap' (Laplacian eigengap heuristic — fast).

    Returns:
        (fused_scores, meta_dict) — see lsml_fuse for meta_dict contents.
    """
    binary = []
    for f in feat_names:
        arr = np.array(feats_dict[f], dtype=float)
        med = np.median(arr)
        binary.append(np.where(arr > med, 1.0, -1.0))
    return lsml_fuse(*binary, K_range=K_range, method=method)


def sml_unsupervised_compare(feats_dict: dict, feat_names: list,
                             K_range=None, labels=None):
    """
    Run sml_unsupervised with both K-selection methods and report agreement.

    Useful for assessing whether the eigengap heuristic is redundant with
    the more expensive Paper 1 Algorithm 1 residual-minimisation approach.

    If `labels` are provided (used for evaluation only, not for fusion),
    also reports each method's AUROC.

    Returns:
        dict with keys:
            'residual_fused', 'residual_meta'   (sml_unsupervised method='residual')
            'eigengap_fused', 'eigengap_meta'   (sml_unsupervised method='eigengap')
            'K_residual', 'K_eigengap',
            'same_K', 'group_ARI' (adjusted Rand index between assignments),
            'residual_auc', 'eigengap_auc' (only if labels given)
    """
    from sklearn.metrics import adjusted_rand_score

    fused_r, meta_r = sml_unsupervised(feats_dict, feat_names,
                                       K_range=K_range, method='residual')
    fused_e, meta_e = sml_unsupervised(feats_dict, feat_names,
                                       K_range=K_range, method='eigengap')

    result = {
        'residual_fused': fused_r, 'residual_meta': meta_r,
        'eigengap_fused': fused_e, 'eigengap_meta': meta_e,
        'K_residual': meta_r['K'], 'K_eigengap': meta_e['K'],
        'same_K': meta_r['K'] == meta_e['K'],
        'group_ARI': float(adjusted_rand_score(meta_r['c'], meta_e['c'])),
    }
    if labels is not None:
        r_auc = boot_auc(labels, fused_r)
        n_auc = boot_auc(labels, -fused_r)
        result['residual_auc'] = max(r_auc[0], n_auc[0])
        r_auc = boot_auc(labels, fused_e)
        n_auc = boot_auc(labels, -fused_e)
        result['eigengap_auc'] = max(r_auc[0], n_auc[0])
    return result


# ── Combined search ────────────────────────────────────────────────────────────

def best_nadler_on(feats_dict: dict, feat_names: list, labels_,
                   max_size: int = 4, label: str = "",
                   compare_mean: bool = True,
                   binarize: bool = False):
    """
    Exhaustive subset search for best SML-SS (Spectral Meta-Learner with
    Supervised Subset Search) fusion score.

    Adapts the SML framework of Parisi-Nadler-Kluger [PNAS 2014] by:
      1. Estimating each feature's discrimination direction from ground-truth
         labels (sign orientation via boot_auc — the supervised step).
      2. Z-scoring after orientation to fix scale bias across features with
         different units (e.g. trace_length ~300 vs epr ~1.5).
      3. Optionally binarizing to ±1 via median threshold (binarize=True),
         which satisfies the binary input assumption of Lemma 1 in Parisi
         et al. [2014] for theoretically grounded weight estimation.
         The weight estimation algorithm depends on the binarize flag:
           - binarize=True  → sml_fuse (Lemma 1 exact: leading eigenvector
             of off-diagonal R_off).  Paper-aligned.
           - binarize=False → nadler_fuse (M-matrix variant from Parisi 2014
             PNAS).  Backward-compatible with pre-Step-105 results.
         Either way the fused score is computed by applying the estimated
         weights to the z-scored continuous arrays, preserving AUROC
         discrimination power.
      4. Exhaustive search over all subsets (size 2 to max_size) for the
         combination with highest fused AUROC against ground-truth labels.

    Conditional independence filter: subsets containing any pair with
    |Spearman ρ| ≥ 0.75 are skipped. This approximates the dependent-
    classifier detection of Jaffé-Fetaya-Nadler [2016] (Section 3), which
    uses 2×2 determinants of the covariance matrix to identify groups of
    strongly correlated classifiers.

    NOTE — in-sample selection bias: the winning subset is chosen by
    maximizing AUROC on the same N samples used for reporting. Bootstrap CI
    does not correct this. Proper evaluation requires a held-out test split.
    This function implements the supervised variant (SML-SS); for the zero-
    label variant see best_nadler_pseudo_label (SML-PL).

    Args:
        feats_dict:   {feature_name: np.ndarray} mapping of raw feature arrays.
        feat_names:   Ordered list of feature names to consider.
        labels_:      Binary correctness labels (1 = correct, 0 = wrong).
        max_size:     Maximum subset size to search.
        label:        String tag for progress print-outs.
        compare_mean: If True, also compute the simple-average AUC for the best
                      subset and print the SML Lift over equal-weight ensemble.
        binarize:     If True, binarize each oriented feature to ±1 via median
                      threshold and use sml_fuse (Lemma 1 exact) for weight
                      estimation — fully paper-aligned.
                      If False (default), use nadler_fuse (M-matrix variant)
                      on continuous z-scored arrays — preserves backward
                      compatibility with pre-Step-105 consolidated results.

    Returns:
        (best_auc, best_lo, best_hi, best_subset, best_weights)
        - best_subset:  tuple of feature name strings, in fusion order
        - best_weights: np.ndarray aligned with best_subset, L1-normalized
                        SML weights (from sml_fuse if binarize=True,
                        else from nadler_fuse). None if no valid subset.
    """
    labels_ = np.array(labels_)

    # ── 1. Orient each feature so higher score → more likely correct ──────────
    auc_m, sign_m = {}, {}
    for n_ in feat_names:
        ap, *_ = boot_auc(labels_,  feats_dict[n_])
        an, *_ = boot_auc(labels_, -feats_dict[n_])
        if ap >= an:
            auc_m[n_], sign_m[n_] = ap, +1
        else:
            auc_m[n_], sign_m[n_] = an, -1

    # ── 2. Z-score after sign orientation (scale normalisation fix) ───────────
    oriented = {
        n_: zscore(feats_dict[n_] * sign_m[n_])
        for n_ in feat_names
    }

    # ── 2b. (Optional) Binarize to ±1 for paper-aligned weight estimation ─────
    # When binarize=True: weights estimated from binary {-1,+1} classifiers
    # (satisfies Lemma 1, Parisi-Nadler-Kluger PNAS 2014); fused score uses
    # the continuous oriented arrays to preserve AUROC discrimination.
    if binarize:
        binary_for_weights = {
            n_: np.where(oriented[n_] > np.median(oriented[n_]), 1.0, -1.0)
            for n_ in feat_names
        }
    else:
        binary_for_weights = oriented  # continuous z-scored (empirical adaptation)

    # ── 3. Precompute Spearman ρ on z-scored, oriented arrays ─────────────────
    rho = {}
    for a, b in itertools.combinations_with_replacement(feat_names, 2):
        r, _ = spearmanr(oriented[a], oriented[b])
        rho[(a, b)] = rho[(b, a)] = r

    # ── 4. Only search features that individually beat chance ─────────────────
    info = [n_ for n_ in feat_names if auc_m[n_] > 0.50]
    total_combos = sum(
        sum(1 for _ in itertools.combinations(info, size))
        for size in range(2, min(len(info) + 1, max_size + 1))
    )
    print(f"  [{label}] {len(feat_names)} features, {len(info)} informative, "
          f"max_size={max_size} → {total_combos} raw combos")

    best_a, best_lo, best_hi, best_s, best_w = 0.0, 0.0, 0.0, None, None
    checked, skipped = 0, 0

    for size in range(2, min(len(info) + 1, max_size + 1)):
        size_combos   = list(itertools.combinations(info, size))
        valid_in_size = 0
        for s in size_combos:
            if any(abs(rho[(a, b)]) >= 0.75
                   for a, b in itertools.combinations(s, 2)):
                skipped += 1
                continue
            # Weight estimation algorithm depends on binarize flag:
            #   binarize=True  → sml_fuse  (Lemma 1 exact, paper-aligned)
            #   binarize=False → nadler_fuse (M-matrix variant, backward-compat)
            # Fused score is always continuous oriented arrays for AUROC discrimination.
            if binarize:
                _, w = sml_fuse(*[binary_for_weights[n_] for n_ in s])
            else:
                _, w = nadler_fuse(*[binary_for_weights[n_] for n_ in s])
            fused = np.column_stack([oriented[n_] for n_ in s]) @ w
            a, lo, hi = boot_auc(labels_, fused)
            if a > best_a:
                best_a, best_lo, best_hi, best_s, best_w = a, lo, hi, s, w
            checked += 1
            valid_in_size += 1
        print(f"    size={size}: {len(size_combos)} combos, "
              f"{valid_in_size} passed ρ-filter, best so far={100*best_a:.1f}%")

    print(f"  [{label}] done — checked={checked}, skipped(ρ)={skipped}, "
          f"best={100*best_a:.1f}%")

    # ── 5. Optionally compare against simple average on the best subset ───────
    if compare_mean and best_s is not None:
        mean_fused, _ = simple_average_fusion(*[oriented[n_] for n_ in best_s])
        mean_auc, mean_lo, mean_hi = boot_auc(labels_, mean_fused)
        lift = (best_a - mean_auc) * 100
        print(f"\n  SML Lift over equal-weight ensemble (subset: {'+'.join(best_s)}):")
        print(f"    SML-SS : {100*best_a:.1f}%  [{100*best_lo:.1f}, {100*best_hi:.1f}]")
        print(f"    Mean   : {100*mean_auc:.1f}%  [{100*mean_lo:.1f}, {100*mean_hi:.1f}]")
        print(f"    Lift   : {lift:+.1f} pp")

    return best_a, best_lo, best_hi, best_s, best_w


def best_nadler_pseudo_label(
    feats_dict: dict,
    feat_names: list,
    seed_features: list,
    seed_signs: dict,
    real_labels=None,
    max_size: int = 4,
    label: str = "",
    compare_mean: bool = False,
):
    """
    Fully unsupervised SML-PL (Spectral Meta-Learner with Pseudo-Label subset
    selection). Zero ground-truth labels are required during feature selection.

    The SML framework (Parisi-Nadler-Kluger [PNAS 2014]) is label-free in its
    weight estimation step: only unlabeled binary classifier outputs are needed
    to estimate balanced accuracies from the rank-1 covariance structure.
    This function extends label-free operation to the subset selection step by
    replacing ground-truth labels with majority-vote pseudo-labels derived from
    seed classifiers whose discrimination direction is known a priori.

    Design:
        1. Orient each seed feature by its known direction (seed_signs).
        2. Build pseudo-labels: sample i is "correct" if the majority of oriented
           seed features are above their median (higher oriented value → more
           confident → less likely hallucinated).
        3. Run the exhaustive SML-SS subset search against pseudo-labels
           (same as best_nadler_on — sign orientation for non-seed features is
           also determined by the pseudo-labels).
        4. If real_labels provided, re-evaluate the best-subset fused score
           against ground truth and return the real AUROC.

    Seed classifiers are continuous features with empirically validated
    discrimination directions for reasoning tasks (math, science MCQ):
    higher entropy / uncertainty → more likely wrong (sign = −1).
    These defaults must NOT be used for factual-recall QA (Phase 9 showed
    entropy signals are anti-predictive on recall tasks).

    Real labels, if provided, are used ONLY for final AUROC reporting, never
    for subset selection or weight estimation.

    Args:
        feats_dict:    {feature_name: np.ndarray} of raw feature arrays.
        feat_names:    Feature names to include in the search.
        seed_features: Reference classifiers whose uncertainty direction is trusted.
        seed_signs:    {feature_name: +1|-1}.  +1 = higher value → more correct;
                       -1 = higher value → more uncertain / wrong.
        real_labels:   Optional 1-D ground-truth array.  If provided, the returned
                       auc/lo/hi are against real labels; pseudo-label AUC is also
                       printed for comparison.  If None, reported AUC is against
                       pseudo-labels.
        max_size, label, compare_mean: forwarded to best_nadler_on.

    Returns:
        (auc, lo, hi, best_subset, best_weights) — same 5-tuple as best_nadler_on.
        auc/lo/hi are against real_labels if provided, else pseudo-labels.
    """
    n = len(next(iter(feats_dict.values())))

    # ── 1. Orient seed features, build pseudo-labels by majority vote ─────────
    votes = np.zeros(n, dtype=float)
    for f in seed_features:
        s = seed_signs.get(f, +1)
        arr = np.array(feats_dict[f], dtype=float) * s
        med = np.median(arr)
        votes += (arr > med).astype(float)

    threshold = len(seed_features) / 2.0
    pseudo = (votes > threshold).astype(int)
    n_pos = int(pseudo.sum())
    print(f"  [{label or 'pseudo'}] pseudo-labels: {n_pos}/{n} positive "
          f"({100 * n_pos / max(n, 1):.1f}%)")

    # Fallback if majority vote produces near-degenerate split
    if min(n_pos, n - n_pos) < 2:
        med_v = np.median(votes)
        pseudo = (votes >= med_v).astype(int)
        print(f"  [{label or 'pseudo'}] fallback median split: {pseudo.sum()} pos")

    # ── 2. Optional: report pseudo-label agreement with real labels ───────────
    if real_labels is not None:
        rl_int = np.array(real_labels, dtype=int)
        agree = float(np.mean(pseudo == rl_int))
        print(f"  [{label or 'pseudo'}] pseudo-label accuracy vs real: {100 * agree:.1f}%")

    # ── 3. Exhaustive subset search against pseudo-labels ─────────────────────
    # The pseudo-labels encode the correct sign for seed features by construction
    # (pseudo=1 for low-entropy = high oriented-seed samples), so best_nadler_on
    # will naturally orient seed features to match seed_signs.
    pl_auc, pl_lo, pl_hi, best_s, best_w = best_nadler_on(
        feats_dict, feat_names, pseudo,
        max_size=max_size,
        label=f"{label}(pseudo)" if label else "pseudo",
        compare_mean=compare_mean,
    )

    if best_s is None:
        return pl_auc, pl_lo, pl_hi, best_s, best_w

    # ── 4. Re-evaluate best subset against real labels ────────────────────────
    if real_labels is not None:
        rl = np.array(real_labels, dtype=float)
        # Reconstruct oriented, z-scored arrays using the same sign logic as
        # best_nadler_on used internally (both use pseudo-labels for orientation).
        oriented = []
        for n_ in best_s:
            ap, *_ = boot_auc(pseudo, feats_dict[n_])
            an, *_ = boot_auc(pseudo, -np.array(feats_dict[n_]))
            s = +1 if ap >= an else -1
            oriented.append(zscore(np.array(feats_dict[n_], dtype=float) * s))
        fused, _ = nadler_fuse(*oriented)
        # AUROC is invariant to score sign — take the better orientation
        auc_p, *_ = boot_auc(rl, fused)
        auc_n, *_ = boot_auc(rl, -fused)
        if auc_p >= auc_n:
            real_auc, real_lo, real_hi = boot_auc(rl, fused)
        else:
            real_auc, real_lo, real_hi = boot_auc(rl, -fused)
        print(f"  [{label or 'pseudo'}] SML-PL pseudo-label AUROC: {100 * pl_auc:.1f}% | "
              f"SML-PL real AUROC: {100 * real_auc:.1f}% "
              f"[{100 * real_lo:.1f}, {100 * real_hi:.1f}]")
        return real_auc, real_lo, real_hi, best_s, best_w

    return pl_auc, pl_lo, pl_hi, best_s, best_w


# ────────────────────────────────────────────────────────────────────────────
# U-PCR — Unsupervised Principal Components Regression
# Tenzer, Dror, Nadler, Bilal, Kluger (AISTATS 2022 / arXiv:1703.02965)
# ────────────────────────────────────────────────────────────────────────────

def _upcr_g2_grid(A, c_vals, evecs, var_y):
    """
    U-PCR g² grid search: for each q on a 300-point grid over [0, var_y],
    solve the linear system rho(q) = lstsq(A, c_vals + q) and score the
    relative eigen-projection residual ||rho − proj_V rho|| / ||rho||.

    Returns (best_res, best_q, best_rho). Extracted verbatim from upcr_fuse
    (Step 186 selector bench) so the projection residual — a label-free
    structural-fit statistic — is also computable standalone via
    upcr_proj_residual without running the full fusion.
    """
    q_grid = np.linspace(0, var_y, 300)
    best_res, best_q, best_rho = np.inf, q_grid[0], None
    for q in q_grid:
        rho, *_ = np.linalg.lstsq(A, c_vals + q, rcond=None)
        proj = evecs @ (evecs.T @ rho)  # generalizes v1*(v1@rho) to k dimensions
        res = np.linalg.norm(rho - proj) / (np.linalg.norm(rho) + 1e-12)
        if res < best_res:
            best_res, best_q, best_rho = res, q, rho.copy()
    return best_res, best_q, best_rho


def upcr_fuse(F, var_y=0.25, n_components=1, auto_components=True,
              lambda2_threshold=0.1, min_frac=0.05, exclude_frac=3.0,
              return_diagnostics=False):
    """
    U-PCR fusion (Tenzer et al., AISTATS 2022 / arXiv:1703.02965).

    Under the uncorrelated-error assumption E[h_i h_j] = 0, the off-diagonal
    of the expert covariance matrix satisfies C_ij = rho_i + rho_j - g^2,
    where rho_i = Cov(f_i, Y) and g^2 = Var(Y) - min_i Var(f_i - Y).
    The leading eigenvector(s) of C are proportional to rho, giving optimal weights.

    Args:
        F: (m, n) float array — m experts x n samples, already z-scored and
           sign-oriented so higher value = more likely correct.
        var_y: Var(Y) for the response. For binary Y with prevalence p: p*(1-p).
               Default 0.25 (balanced). Used as the upper bound for the g^2 grid.
        n_components: 1 or 2 — number of leading eigenvectors for weight projection.
                      Only used when auto_components=False.
        auto_components: if True (default), automatically select 2 components when
                         lambda_2 > lambda2_threshold * Trace(C) (paper criterion).
        lambda2_threshold: fraction of trace for auto 2-component selection (default 0.1).
        min_frac: exclude experts with rho_hat_i < min_frac * var_y (too weak).
        exclude_frac: exclude experts with rho_hat_i < rho_hat_max / exclude_frac.
        return_diagnostics: if True, return a 4th value — dict with eigenvalue diagnostics.

    Returns:
        w: (m,) fusion weight vector (zero for excluded experts)
        rho_hat: (m,) estimated expert-response covariances (full, including excluded)
        g2_hat: float — estimated minimal attainable MSE
        diagnostics: dict (only if return_diagnostics=True) — keys: n_components_used,
                     lambda2_frac, evals_full (top-2 eigenvalues of full C)
    """
    import numpy as np
    from scipy.linalg import eigh

    m, n = F.shape
    C = (F @ F.T) / n  # sample covariance (m x m)
    trace_C = float(np.trace(C))

    # Auto-select n_components based on eigenspectrum (paper criterion: Sec. 3.2)
    k_probe = min(2, m)
    evals_probe, _ = eigh(C, subset_by_index=[m - k_probe, m - 1])
    evals_probe = evals_probe[::-1]  # descending
    lambda2_frac = float(evals_probe[1] / (trace_C + 1e-12)) if k_probe >= 2 else 0.0

    if auto_components:
        n_components = 2 if (k_probe >= 2 and lambda2_frac > lambda2_threshold) else 1

    n_components_used = n_components

    # Build linear system: C_ij = rho_i + rho_j - q  (off-diagonal, i < j)
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    A = np.zeros((len(pairs), m))
    c_vals = np.array([C[i, j] for i, j in pairs])
    for k, (i, j) in enumerate(pairs):
        A[k, i] = 1.0
        A[k, j] = 1.0

    # Leading eigenvector(s) of full C (for g2 grid search)
    k = min(n_components, m)
    evals, evecs = eigh(C, subset_by_index=[m - k, m - 1])
    evals = evals[::-1]
    evecs = evecs[:, ::-1]  # (m, k) — columns are eigenvectors in descending order

    # Grid search: g2_hat = argmin_q || rho(q) - proj_{V_k} rho(q) || / ||rho(q)||
    best_res, best_q, best_rho = _upcr_g2_grid(A, c_vals, evecs, var_y)

    g2_hat = best_q
    rho_hat_full = best_rho

    # Exclusion: drop weak experts (Algorithm 1, step 4)
    rho_max = np.max(rho_hat_full)
    keep = (rho_hat_full >= min_frac * var_y) & (rho_hat_full >= rho_max / exclude_frac)
    if keep.sum() < 3:  # fallback: keep top-3 by rho
        keep = np.zeros(m, dtype=bool)
        keep[np.argsort(rho_hat_full)[-3:]] = True

    # Refit on kept experts only
    F_k = F[keep]
    n_k = keep.sum()
    C_k = (F_k @ F_k.T) / n
    pairs_k = [(i, j) for i in range(n_k) for j in range(i + 1, n_k)]
    A_k = np.zeros((len(pairs_k), n_k))
    c_vals_k = np.array([C_k[i, j] for i, j in pairs_k])
    for idx, (i, j) in enumerate(pairs_k):
        A_k[idx, i] = 1.0
        A_k[idx, j] = 1.0
    rho_k, *_ = np.linalg.lstsq(A_k, c_vals_k + g2_hat, rcond=None)

    k2 = min(n_components, n_k)
    evals_k, evecs_k = eigh(C_k, subset_by_index=[n_k - k2, n_k - 1])
    evals_k = evals_k[::-1]
    evecs_k = evecs_k[:, ::-1]  # (n_k, k2)

    # Weights: w = sum_c (v_c^T rho / lambda_c) * v_c   (Eq. 9, generalized)
    # Each eigenvector contributes an additive correction to the weight vector.
    # With k2=1: reduces to original 1-component formula.
    w_k = np.zeros(n_k)
    for c in range(k2):
        vc = evecs_k[:, c]
        w_k += (vc @ rho_k) / (evals_k[c] + 1e-12) * vc

    # Embed back to full m-length vectors
    w_full = np.zeros(m)
    w_full[keep] = w_k
    rho_hat_out = np.zeros(m)
    rho_hat_out[keep] = rho_k

    if return_diagnostics:
        diag = {
            'n_components_used': n_components_used,
            'lambda2_frac': lambda2_frac,
            'evals_top2': evals_probe.tolist(),
            'trace_C': trace_C,
            'n_kept': int(keep.sum()),
            # Step-186 selector-bench additions (previously computed then
            # discarded): the label-free structural-fit residual, the
            # weak-expert keep mask, and the pre-exclusion rho estimates.
            'proj_residual': float(best_res),
            'keep': keep.copy(),
            'rho_hat_full': rho_hat_full.copy(),
        }
        return w_full, rho_hat_out, g2_hat, diag

    return w_full, rho_hat_out, g2_hat


def upcr_proj_residual(F, var_y=0.25, n_components=1, auto_components=True,
                       lambda2_threshold=0.1):
    """
    Label-free U-PCR structural-fit residual (selector bench, Step 186+).

    Runs only the ESTIMATION stage of upcr_fuse — sample covariance, linear
    system C_ij = rho_i + rho_j − g², eigen-projection, g² grid search — and
    returns the best relative projection residual ||rho − proj rho||/||rho||.
    Small residual ⇒ the additive uncorrelated-error structural model fits
    this expert set well. Mirrors upcr_fuse's numerics exactly (same grid,
    same lstsq); no expert exclusion, no weights, no fusion.

    IMPORTANT (smoke-test finding, 2026-07-17): for STRUCTURAL-MODEL
    DIAGNOSIS (the A1 router / selection objective) call with
    ``auto_components=False, n_components=1``. The auto 2-component mode is
    upcr_fuse's own robustness mechanism for dependent experts — it widens
    the projection subspace exactly when block dependence inflates λ₂, so the
    auto-mode residual ABSORBS the violation instead of measuring it (block
    data scored LOWER than independent data in the known-answer test).

    Args:
        F: (m, n) float array — m experts × n samples, z-scored + oriented.
        var_y / n_components / auto_components / lambda2_threshold: as in
            upcr_fuse.

    Returns:
        (proj_residual, g2_hat, rho_hat_full) — floats + (m,) array.
    """
    from scipy.linalg import eigh

    F = np.asarray(F, dtype=float)
    m, n = F.shape
    if m < 3:
        raise ValueError(f"U-PCR needs m >= 3 experts, got {m}")
    C = (F @ F.T) / n
    trace_C = float(np.trace(C))

    k_probe = min(2, m)
    evals_probe, _ = eigh(C, subset_by_index=[m - k_probe, m - 1])
    evals_probe = evals_probe[::-1]
    lambda2_frac = float(evals_probe[1] / (trace_C + 1e-12)) if k_probe >= 2 else 0.0
    if auto_components:
        n_components = 2 if (k_probe >= 2 and lambda2_frac > lambda2_threshold) else 1

    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    A = np.zeros((len(pairs), m))
    c_vals = np.array([C[i, j] for i, j in pairs])
    for k, (i, j) in enumerate(pairs):
        A[k, i] = 1.0
        A[k, j] = 1.0

    k = min(n_components, m)
    evals, evecs = eigh(C, subset_by_index=[m - k, m - 1])
    evecs = evecs[:, ::-1]

    best_res, best_q, best_rho = _upcr_g2_grid(A, c_vals, evecs, var_y)
    return float(best_res), float(best_q), best_rho


def upcr_pipeline(feats_dict, feat_names, signs, var_y=0.25, return_diagnostics=False,
                  **kwargs):
    """
    Full U-PCR pipeline: orient signs -> z-score -> upcr_fuse -> ensemble score.

    Args:
        feats_dict: {feat_name: np.array(n,)} — raw feature arrays
        feat_names: list of feature names to use (length m >= 3)
        signs: dict {feat_name: +1 or -1} — orient so higher = more likely correct
        var_y: Var(Y) passed to upcr_fuse (default 0.25 for balanced binary labels)
        return_diagnostics: if True, also return eigenvalue diagnostics dict
        **kwargs: forwarded to upcr_fuse (n_components, auto_components,
                  lambda2_threshold, min_frac, exclude_frac)

    Returns:
        score: (n,) ensemble score (higher = more likely correct)
        w: (m,) fusion weights
        rho_hat: (m,) estimated expert-response covariances
        g2_hat: float
        diagnostics: dict (only if return_diagnostics=True)
    """
    import numpy as np
    F = np.stack(
        [signs.get(f, 1) * zscore(np.array(feats_dict[f], dtype=float))
         for f in feat_names],
        axis=0,
    )  # (m, n)
    result = upcr_fuse(F, var_y=var_y, return_diagnostics=return_diagnostics, **kwargs)
    if return_diagnostics:
        w, rho_hat, g2_hat, diag = result
        score = w @ F
        return score, w, rho_hat, g2_hat, diag
    else:
        w, rho_hat, g2_hat = result
        score = w @ F
        return score, w, rho_hat, g2_hat
