"""
Statistical fusion utilities: bootstrap AUC, Nadler spectral fusion, simple average fusion.

Key design decisions
--------------------
* z-score normalization is applied INSIDE best_nadler_on, after sign orientation and
  BEFORE the covariance matrix is computed.  This ensures Nadler weights reflect
  statistical complementarity, not feature scale.  The Spearman ρ filter is
  rank-invariant and therefore unaffected.

* simple_average_fusion is provided alongside nadler_fuse so that every experiment
  can report "Nadler Lift" = AUC_nadler − AUC_mean over the same feature subset.
  This directly justifies the use of the more complex Nadler algorithm.
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
    """
    std = arr.std()
    return (arr - arr.mean()) / std if std > 1e-8 else arr - arr.mean()


# ── AUC with bootstrap CI ─────────────────────────────────────────────────────

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
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], s[idx]))

    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (base, base)
    return base, lo, hi


# ── Fusion algorithms ──────────────────────────────────────────────────────────

def nadler_fuse(*views) -> tuple:
    """
    Nadler combinatorial spectral fusion.

    Expects z-scored, sign-oriented feature arrays (call best_nadler_on, which
    handles normalization, rather than calling this directly on raw features).

    Algorithm:
        1. Stack views into matrix X (n_samples × k_features).
        2. Compute sample covariance C.
        3. Build M = diag(row_sums_off_diag) @ C^{-1} @ diag(col_sums_off_diag).
        4. Weights = absolute value of the leading eigenvector of M, L1-normalized.
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
    nadler_fuse).  Used to compute Nadler Lift = AUC_nadler − AUC_mean.

    Returns:
        (fused_scores: np.ndarray, weights: np.ndarray)
    """
    X = np.column_stack(views)
    k = X.shape[1]
    w = np.ones(k) / k
    return X @ w, w


# ── Combined search ────────────────────────────────────────────────────────────

def best_nadler_on(feats_dict: dict, feat_names: list, labels_,
                   max_size: int = 4, label: str = "",
                   compare_mean: bool = True):
    """
    Exhaustive search over feature subsets for the best Nadler fusion score.

    Normalization: every feature is z-scored after sign orientation and before
    the covariance matrix is computed.  This fixes the scale-bias issue where
    high-variance features (e.g. trace_length ~300) would dominate low-variance
    features (e.g. epr ~1.5) purely due to units.

    Subset filtering: subsets containing any pair with |Spearman ρ| ≥ 0.75 are
    skipped (redundant views hurt Nadler's covariance structure).

    Args:
        feats_dict:   {feature_name: np.ndarray} mapping of raw feature arrays.
        feat_names:   Ordered list of feature names to consider.
        labels_:      Binary correctness labels (1 = correct, 0 = wrong).
        max_size:     Maximum subset size to search.
        label:        String tag for progress print-outs.
        compare_mean: If True, also compute the simple-average AUC for the best
                      Nadler subset and print the Nadler Lift.

    Returns:
        (best_auc, best_lo, best_hi, best_subset, best_weights)
        - best_subset:  tuple of feature name strings, in fusion order
        - best_weights: np.ndarray aligned with best_subset (L1-normalized
                        leading-eigenvector weights from nadler_fuse). None
                        if no valid subset was found.
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
            fused, w = nadler_fuse(*[oriented[n_] for n_ in s])
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
        print(f"\n  Nadler Lift over simple average (subset: {'+'.join(best_s)}):")
        print(f"    Nadler : {100*best_a:.1f}%  [{100*best_lo:.1f}, {100*best_hi:.1f}]")
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
    Fully unsupervised Nadler fusion via feature majority-vote pseudo-labels.

    Enables running the complete pipeline with zero ground-truth labels.
    Feature subset selection uses pseudo-labels derived from the seed features;
    real labels (if provided) are used only for final AUROC reporting.

    Design:
        1. Orient each seed feature by its known direction (seed_signs).
        2. Build pseudo-labels: sample i is "correct" if the majority of oriented
           seed features are above their median (higher oriented value → more
           confident → less likely hallucinated).
        3. Run the exhaustive Nadler subset search against the pseudo-labels
           (same as best_nadler_on — sign orientation for non-seed features is
           also determined by the pseudo-labels).
        4. If real_labels provided, re-evaluate the best-subset fused score
           against ground truth and return the real AUROC.

    Args:
        feats_dict:    {feature_name: np.ndarray} of raw feature arrays.
        feat_names:    Feature names to include in the search.
        seed_features: Known-good features whose uncertainty direction is trusted.
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

    Note:
        The default seed_signs assume *reasoning tasks* (math, science MCQ) where
        higher entropy → more uncertain → wrong.  They are empirically validated
        for those domains but must NOT be used for factual recall QA (Phase 9
        showed entropy signals are anti-predictive on recall tasks).
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
        print(f"  [{label or 'pseudo'}] pseudo AUROC: {100 * pl_auc:.1f}% | "
              f"real AUROC: {100 * real_auc:.1f}% "
              f"[{100 * real_lo:.1f}, {100 * real_hi:.1f}]")
        return real_auc, real_lo, real_hi, best_s, best_w

    return pl_auc, pl_lo, pl_hi, best_s, best_w
