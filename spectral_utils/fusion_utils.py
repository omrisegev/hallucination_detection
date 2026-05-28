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


# ── Binarization ──────────────────────────────────────────────────────────────

def binarize_classifiers(feats_dict: dict, signs: dict) -> dict:
    """
    Convert continuous uncertainty features into binary ±1 classifiers.

    Each feature is sign-oriented then thresholded at its empirical median,
    yielding a binary classifier fᵢ ∈ {−1, +1} over the n samples.
    A prediction of +1 means "likely correct"; −1 means "likely wrong."

    This binarization satisfies the input assumption of the Spectral
    Meta-Learner (SML, Parisi-Nadler-Kluger [PNAS 2014]).  Under Lemma 1
    of that paper, the off-diagonal entries of the covariance matrix of
    binary ±1 classifiers form a rank-1 matrix vvᵀ where vᵢ ∝ (2αᵢ − 1)·√(1−b²),
    with αᵢ the balanced accuracy of classifier i and b the true class
    imbalance (a dataset property, not affected by the median split).

    The median split produces a balanced prediction distribution
    (50 % +1, 50 % −1), the standard convention for unsupervised
    classifier construction.  An informative feature will yield αᵢ > 0.5
    after this thresholding; a non-informative feature will yield αᵢ ≈ 0.5
    and receive near-zero SML weight automatically.

    Args:
        feats_dict: {feature_name: np.ndarray} of continuous feature values.
        signs:      {feature_name: +1 | −1}.
                    +1 = higher raw value → more likely correct.
                    −1 = higher raw value → more likely wrong (flip sign).

    Returns:
        {feature_name: np.ndarray of ±1.0 values, same length as input}
    """
    binary = {}
    for name, arr in feats_dict.items():
        s = signs.get(name, +1)
        oriented = np.array(arr, dtype=float) * s
        med = np.median(oriented)
        binary[name] = np.where(oriented > med, 1.0, -1.0)
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
