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
    except Exception:
        v = np.ones(k) / k
    return X @ v, v


def _score_matrix_lsml(R: np.ndarray) -> np.ndarray:
    """
    Paper 1 Eq. (15): s_ij = Σ_{k,l ≠ i,j} |r_ij·r_kl − r_il·r_kj|.
    Large values indicate dependent (same-group) classifier pairs.
    """
    m = R.shape[0]
    s = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            total = 0.0
            for k in range(m):
                if k == i or k == j:
                    continue
                for l in range(m):
                    if l == i or l == j or l == k:
                        continue
                    total += abs(R[i, j] * R[k, l] - R[i, l] * R[k, j])
            s[i, j] = s[j, i] = total
    return s


def _spectral_cluster_precomputed(similarity: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    """Spectral clustering on a precomputed similarity matrix."""
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(
        n_clusters=K, affinity='precomputed',
        assign_labels='kmeans', random_state=seed,
    )
    return sc.fit_predict(similarity + 1e-12)


def _estimate_von_voff(R: np.ndarray, c: np.ndarray) -> tuple:
    """
    Estimate v^on and v^off per Paper 1 Lemma 1.

    v^on_i for i ∈ group g: leading eigenvector of the within-group submatrix
    of R (diagonal zeroed).
    v^off: leading eigenvector of R with within-group entries zeroed.
    """
    m = R.shape[0]
    v_on = np.zeros(m)
    for g in np.unique(c):
        idx = np.where(c == g)[0]
        if len(idx) == 1:
            v_on[idx] = 1.0
            continue
        sub = R[np.ix_(idx, idx)].copy()
        sub -= np.diag(np.diag(sub))
        try:
            _, vecs = eigh(sub)
            v_on[idx] = vecs[:, -1]
        except Exception:
            v_on[idx] = 1.0 / np.sqrt(len(idx))

    R_off_only = R.copy()
    for i in range(m):
        for j in range(m):
            if c[i] == c[j]:
                R_off_only[i, j] = 0.0
    try:
        _, vecs = eigh(R_off_only)
        v_off = vecs[:, -1]
    except Exception:
        v_off = np.ones(m) / np.sqrt(m)

    return v_on, v_off


def _residual_lsml(R: np.ndarray, c: np.ndarray) -> float:
    """Paper 1 Eq. (14) residual under assignment c."""
    m = R.shape[0]
    v_on, v_off = _estimate_von_voff(R, c)
    resid = 0.0
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            if c[i] == c[j]:
                resid += (v_on[i] * v_on[j] - R[i, j]) ** 2
            else:
                resid += (v_off[i] * v_off[j] - R[i, j]) ** 2
    return float(resid)


def _eigengap_K(s: np.ndarray, max_K: int = 8) -> int:
    """
    Eigengap heuristic on the normalized Laplacian of similarity matrix s.
    Returns K maximizing the gap between consecutive smallest Laplacian
    eigenvalues (K = index of largest gap + 1, with K ≥ 2).
    """
    m = s.shape[0]
    deg = s.sum(axis=1) + 1e-12
    D_inv_sqrt = 1.0 / np.sqrt(deg)
    S_norm = (D_inv_sqrt[:, None] * s) * D_inv_sqrt[None, :]
    L_norm = np.eye(m) - S_norm
    eigvals = np.sort(np.linalg.eigvalsh(L_norm))[:max_K + 1]
    gaps = np.diff(eigvals)
    return max(int(np.argmax(gaps)) + 1, 2)


def detect_dependent_groups(binary_classifiers, K_range=None, method: str = 'residual'):
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

    Returns:
        (best_K, assignment_c, residual_at_best_K, score_matrix_s)
    """
    X = np.column_stack(binary_classifiers)
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
            return 1, np.zeros(m, dtype=int), float('inf'), s
        return K, c, _residual_lsml(R, c), s

    if method != 'residual':
        raise ValueError(f"Unknown method {method!r}; use 'residual' or 'eigengap'.")

    best_K, best_c, best_resid = None, None, float('inf')
    for K in K_range:
        try:
            c = _spectral_cluster_precomputed(s, K)
        except Exception:
            continue
        r = _residual_lsml(R, c)
        if r < best_resid:
            best_resid, best_K, best_c = r, K, c
    if best_K is None:
        return 1, np.zeros(m, dtype=int), float('inf'), s
    return best_K, best_c, best_resid, s


def lsml_fuse(*binary_classifiers, K_range=None, method: str = 'residual'):
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

    K, c, residual, s_mat = detect_dependent_groups(
        binary_classifiers, K_range=K_range, method=method,
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


def lsml_continuous(*views, K_range=None, method: str = 'residual'):
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

    Returns:
        (fused_scores, meta_dict) — same format as lsml_fuse.
    """
    X = np.column_stack(views)
    n, m = X.shape

    K, c, residual, s_mat = detect_dependent_groups(
        views, K_range=K_range, method=method,
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
