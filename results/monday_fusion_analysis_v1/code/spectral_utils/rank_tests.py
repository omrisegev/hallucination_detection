"""
Label-free estimators of the number of "signal" eigenvalues (effective rank)
of a sample covariance matrix.

Used by the A1 selector track (selector bench, Step 186+) as alternative K
rules for L-SML's latent-group count and as per-cell structure diagnostics.

Two estimators:

- ``ahn_horenstein_K`` — the Eigenvalue-Ratio (ER) estimator of Ahn &
  Horenstein (Econometrica 2013): K̂ = argmax_k λ_k / λ_{k+1}. Threshold-free,
  no distributional constants; primary rule.

- ``kritchman_nadler_K`` — the sequential Tracy-Widom hypothesis test of
  Kritchman & Nadler (IEEE Trans. Signal Processing 57(10), 2009; see also
  Chemometrics 94(1), 2008). Tests, for k = 1, 2, ..., whether the k-th
  largest eigenvalue exceeds the Marchenko-Pastur noise edge of the remaining
  bulk at level alpha; K̂ = first k where the test fails to reject, minus 1.

IMPORTANT CAVEATS (recorded per the Step-185 memo, Thread B):

1. Both estimators assume n (samples) and p (features) are large. In this
   project n/p is ~2-30 (n≈100-4500, p≤16/46), well below the asymptotic
   regime — treat K̂ as an anchored prior / diagnostic to compare against the
   Eq-14 residual-grid K, never as a hard cutoff.

2. The mapping "covariance signal rank" → "L-SML latent-group count K" is
   itself heuristic: under the L-SML latent-group model the feature covariance
   has one consensus direction plus group-block structure, so signal rank and
   group count are related but not identical objects.

3. ``kritchman_nadler_K`` here uses Johnstone's TW1 centering/scaling
   constants and a spiked-model bias-corrected noise estimate (fixed-point
   iteration). The TW1 quantile table is hardcoded from the standard RMT
   tables; the noise estimator follows the standard spiked-covariance
   inversion rather than the paper's exact coupled system. Behavior is gated
   by the planted-K smoke test (scripts/smoke_selectors.py) at this project's
   actual n/p scales.
"""

import math

import numpy as np

# Tracy-Widom beta=1 upper-tail quantiles s(alpha): P(TW1 > s) = alpha.
# Standard tabulated values (Johnstone 2001 / RMT tables).
TW1_QUANTILES = {
    0.10: 0.4501,
    0.05: 0.9793,
    0.01: 2.0234,
    0.005: 2.4224,
    0.001: 3.2724,
}


def _sorted_eigvals(eigvals):
    lam = np.sort(np.asarray(eigvals, dtype=float))[::-1]
    return np.clip(lam, 0.0, None)


def ahn_horenstein_K(eigvals, max_K=8, n=None):
    """
    Eigenvalue-Ratio estimator (Ahn & Horenstein 2013).

    K̂ = argmax_k lambda_k / lambda_{k+1} over candidate k. When ``n`` (the
    sample count) is provided, a mock zeroth eigenvalue lambda_0 =
    trace / log(min(n, p)) is prepended per the paper's own device, allowing
    K̂ = 0 ("no signal") — without it, pure-noise input yields an arbitrary
    argmax because all bulk ratios hover near 1.

    Args:
        eigvals: eigenvalues of a (sample) covariance matrix, any order.
        max_K:   largest admissible K̂ (capped at p-1).
        n:       sample count; enables the mock eigenvalue and K̂ = 0.

    Returns:
        int K̂ in [0 if n else 1, max_K].
    """
    lam = _sorted_eigvals(eigvals)
    p = len(lam)
    if p < 2:
        return 1 if n is None else 0
    kmax = min(max_K, p - 1)
    # Floor the denominator so a near-zero trailing eigenvalue cannot force
    # the ratio to explode at the last index by numerical accident.
    eps = 1e-12 * max(lam[0], 1e-30)
    if n is not None:
        mock = lam.sum() / max(math.log(max(min(n, p), 3)), 1.0)
        seq = np.concatenate([[mock], lam])          # seq[k] = lambda_k, k=0..p
        ratios = seq[:kmax + 1] / np.maximum(seq[1:kmax + 2], eps)
        return int(np.argmax(ratios))                # k = 0..kmax
    ratios = lam[:kmax] / np.maximum(lam[1:kmax + 1], eps)
    return int(np.argmax(ratios)) + 1                # k = 1..kmax


def _tw1_quantile(alpha):
    if alpha in TW1_QUANTILES:
        return TW1_QUANTILES[alpha]
    keys = sorted(TW1_QUANTILES)
    if alpha <= keys[0]:
        return TW1_QUANTILES[keys[-1]] if alpha <= 0 else TW1_QUANTILES[keys[0]]
    # log-linear interpolation between tabulated alphas
    for lo, hi in zip(keys, keys[1:]):
        if lo <= alpha <= hi:
            t = (math.log(alpha) - math.log(lo)) / (math.log(hi) - math.log(lo))
            return TW1_QUANTILES[lo] * (1 - t) + TW1_QUANTILES[hi] * t
    return TW1_QUANTILES[keys[-1]]


def _johnstone_constants(n, q):
    """TW1 centering mu and scaling xi for the largest eigenvalue of a
    q-dimensional white Wishart with n samples (Johnstone 2001), in the
    normalization where the sample covariance divides by n."""
    a = math.sqrt(n - 0.5)
    b = math.sqrt(q - 0.5)
    mu = (a + b) ** 2 / n
    xi = (a + b) / n * (1.0 / a + 1.0 / b) ** (1.0 / 3.0)
    return mu, xi


def _noise_var_spiked(lam, k, n, iters=20):
    """
    Noise-variance estimate assuming the top-k eigenvalues are signal.

    Fixed-point bias correction: for each assumed signal eigenvalue, invert
    the spiked-model forward map lambda ≈ rho * (1 + gamma * sigma^2 / (rho -
    sigma^2)) (gamma = q/n) to recover rho, then re-estimate sigma^2 from the
    total variance with the signal contribution removed. Falls back to the
    trimmed mean when the inversion is infeasible.
    """
    p = len(lam)
    q = p - k
    if q <= 0:
        return float(np.mean(lam))
    sigma2 = float(np.mean(lam[k:]))
    if k == 0:
        return sigma2
    gamma = q / n
    for _ in range(iters):
        rho_sum = 0.0
        ok = True
        for j in range(k):
            lj = lam[j]
            # invert lambda = rho + gamma*sigma2*rho/(rho - sigma2):
            # rho^2 - rho*(lj + sigma2 - gamma*sigma2) + lj*sigma2 = 0
            bq = lj + sigma2 - gamma * sigma2
            disc = bq * bq - 4.0 * lj * sigma2
            if disc <= 0:
                ok = False
                break
            rho_sum += 0.5 * (bq + math.sqrt(disc))
        if not ok:
            return float(np.mean(lam[k:]))
        new_sigma2 = max((lam.sum() - rho_sum) / q, 1e-30)
        if abs(new_sigma2 - sigma2) <= 1e-12 * sigma2:
            sigma2 = new_sigma2
            break
        sigma2 = new_sigma2
    return float(sigma2)


def kritchman_nadler_K(eigvals, n, alpha=0.01, max_K=8):
    """
    Sequential Tracy-Widom signal-count test (Kritchman & Nadler 2009).

    For k = 1, 2, ...: H0 = "at most k-1 signals". Reject H0 (i.e. the k-th
    eigenvalue is signal) when

        lambda_k > sigma2_hat(k) * ( mu(n, p-k+1) + s(alpha) * xi(n, p-k+1) )

    where sigma2_hat(k) is the bias-corrected noise variance estimated from
    the p-k smallest eigenvalues, and mu/xi are Johnstone's centering/scaling
    for the largest noise eigenvalue of the remaining bulk. K̂ = first k that
    fails to reject, minus 1.

    Args:
        eigvals: covariance eigenvalues, any order.
        n:       number of samples the covariance was estimated from.
        alpha:   test level (tabulated: 0.10, 0.05, 0.01, 0.005, 0.001).
        max_K:   largest admissible K̂.

    Returns:
        int K̂ in [0, max_K].
    """
    lam = _sorted_eigvals(eigvals)
    p = len(lam)
    if p < 2 or n < 4:
        return 0
    s_alpha = _tw1_quantile(alpha)
    kmax = min(max_K, p - 1)
    for k in range(1, kmax + 1):
        q = p - k + 1          # dimension of the bulk λ_k is largest of, under H0
        sigma2 = _noise_var_spiked(lam, k - 1, n)
        mu, xi = _johnstone_constants(n, q)
        if lam[k - 1] <= sigma2 * (mu + s_alpha * xi):
            return k - 1
    return kmax


def cov_eigvals(V):
    """Descending eigenvalues of the sample covariance of column-views V (n, m)."""
    X = np.asarray(V, dtype=float)
    R = np.cov(X.T)
    if R.ndim == 0:
        return np.array([float(R)])
    return _sorted_eigvals(np.linalg.eigvalsh(R))
