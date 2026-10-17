"""
Unsupervised temporal models over per-token entropy traces.

Track B of the pivot-alternatives pilot (Step 151): do explicit temporal
models — regime-switching (HMM), change-point detection (BOCPD), prediction
residuals (AR / Kalman innovations) — carry hallucination signal that the
static spectral features and DeepConf sliding windows miss?

The 2-state Gaussian HMM is the honest version of an IMM filter: IMM needs
a-priori dynamics for both regimes, which we don't have; EM estimates them.
The AR/Kalman innovation scores are the go/no-go precursor for KalmanNet:
if innovations carry zero label signal, a learned Kalman gain cannot fix it.

Conventions:
    - All scores here are ANOMALY-oriented: higher = more hallucination-
      suspect.  Negate for the package-wide confidence orientation.
    - The HMM is hand-rolled (log-space EM, ~no deps beyond numpy/scipy) so
      spectral_utils stays importable with the standard Colab pip set —
      hmmlearn is deliberately avoided (sklearn-version ABI fragility).
    - HMM state identity is label-free: states are ordered by the mean of
      observation dim 0 (entropy H), so the LAST state is always the
      high-entropy regime.  Labels never touch the fit.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t


# ---------------------------------------------------------------------------
# 2-state (K-state) diagonal Gaussian HMM — pooled log-space EM
# ---------------------------------------------------------------------------

def _as_obs(x):
    a = np.asarray(x, dtype=float)
    return a[:, None] if a.ndim == 1 else a


def _log_gauss_diag(obs, means, variances):
    """(T, K) log density of each row under each diagonal Gaussian state."""
    T, d = obs.shape
    K = means.shape[0]
    out = np.empty((T, K))
    for k in range(K):
        diff2 = (obs - means[k]) ** 2 / variances[k]
        out[:, k] = -0.5 * (d * np.log(2 * np.pi)
                            + np.log(variances[k]).sum()
                            + diff2.sum(axis=1))
    return out


def _forward_backward(log_b, log_trans, log_start, need_xi=True):
    """Scaled forward-backward in log space.

    Returns (gamma (T,K), xi_sum (K,K) or None, loglik).
    """
    T, K = log_b.shape
    la = np.empty((T, K))
    lb = np.empty((T, K))
    la[0] = log_start + log_b[0]
    for t in range(1, T):
        la[t] = log_b[t] + logsumexp(la[t - 1][:, None] + log_trans, axis=0)
    lb[-1] = 0.0
    for t in range(T - 2, -1, -1):
        lb[t] = logsumexp(log_trans + (log_b[t + 1] + lb[t + 1])[None, :], axis=1)
    ll = float(logsumexp(la[-1]))
    gamma = np.exp(la + lb - ll)
    xi_sum = None
    if need_xi and T > 1:
        xi_sum = np.zeros((K, K))
        for t in range(T - 1):
            lx = la[t][:, None] + log_trans + (log_b[t + 1] + lb[t + 1])[None, :] - ll
            xi_sum += np.exp(lx)
    return gamma, xi_sum, ll


def fit_gaussian_hmm(obs_list, n_states: int = 2, n_iter: int = 50,
                     tol: float = 1e-4, seeds=(0, 1, 2),
                     var_floor: float = 1e-3) -> dict:
    """
    Fit a K-state diagonal-covariance Gaussian HMM by pooled EM over a list
    of observation sequences (each (T_i, d), or 1-D arrays).

    Init per seed: states assigned by quantile split of observation dim 0
    (entropy) with seed jitter; sticky transition prior (0.9 self).  Fits with
    any state's pooled occupancy < 2% are rejected in favour of the next seed
    (state-collapse guard); if every seed collapses the best fit is returned
    with degenerate=True.

    States are reordered by mean of dim 0 ascending, so index n_states-1 is
    the high-entropy regime.  Label-free by construction.

    Returns dict: means (K,d), variances (K,d), trans (K,K), start (K,),
    loglik, occupancy (K,), degenerate, seed, n_iter_used.
    """
    seqs = [_as_obs(o) for o in obs_list if len(o) >= 2]
    if not seqs:
        raise ValueError("no usable sequences (need length >= 2)")
    pooled = np.concatenate(seqs, axis=0)
    d = pooled.shape[1]
    K = n_states

    candidates = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        # quantile-split init on dim 0 with jitter
        qs = np.quantile(pooled[:, 0], np.linspace(0, 1, K + 1)[1:-1])
        assign = np.digitize(pooled[:, 0], qs)
        means = np.empty((K, d))
        variances = np.empty((K, d))
        for k in range(K):
            sel = pooled[assign == k]
            if len(sel) < 2:
                sel = pooled
            means[k] = sel.mean(axis=0) + rng.normal(0, 0.05, d)
            variances[k] = np.maximum(sel.var(axis=0), var_floor)
        trans = np.full((K, K), 0.1 / max(K - 1, 1))
        np.fill_diagonal(trans, 0.9)
        start = np.full(K, 1.0 / K)

        prev_ll = -np.inf
        n_used = 0
        for it in range(n_iter):
            n_used = it + 1
            log_trans = np.log(trans)
            log_start = np.log(start)
            g_sum = np.zeros(K)
            g0_sum = np.zeros(K)
            xi_tot = np.zeros((K, K))
            mean_num = np.zeros((K, d))
            var_num = np.zeros((K, d))
            ll_tot = 0.0
            gammas = []
            for obs in seqs:
                log_b = _log_gauss_diag(obs, means, variances)
                gamma, xi_sum, ll = _forward_backward(log_b, log_trans, log_start)
                gammas.append(gamma)
                ll_tot += ll
                g0_sum += gamma[0]
                g_sum += gamma.sum(axis=0)
                if xi_sum is not None:
                    xi_tot += xi_sum
                mean_num += gamma.T @ obs
            means = mean_num / g_sum[:, None]
            for obs, gamma in zip(seqs, gammas):
                for k in range(K):
                    var_num[k] += gamma[:, k] @ (obs - means[k]) ** 2
            variances = np.maximum(var_num / g_sum[:, None], var_floor)
            trans = xi_tot / np.maximum(xi_tot.sum(axis=1, keepdims=True), 1e-12)
            trans = np.maximum(trans, 1e-6)
            trans /= trans.sum(axis=1, keepdims=True)
            start = np.maximum(g0_sum / g0_sum.sum(), 1e-6)
            start /= start.sum()
            if abs(ll_tot - prev_ll) < tol * max(abs(prev_ll), 1.0):
                break
            prev_ll = ll_tot

        occupancy = g_sum / g_sum.sum()
        candidates.append({
            "means": means, "variances": variances, "trans": trans,
            "start": start, "loglik": ll_tot, "occupancy": occupancy,
            "degenerate": bool(occupancy.min() < 0.02),
            "seed": seed, "n_iter_used": n_used,
        })

    ok = [c for c in candidates if not c["degenerate"]]
    best = max(ok or candidates, key=lambda c: c["loglik"])

    order = np.argsort(best["means"][:, 0])
    best["means"] = best["means"][order]
    best["variances"] = best["variances"][order]
    best["trans"] = best["trans"][np.ix_(order, order)]
    best["start"] = best["start"][order]
    best["occupancy"] = best["occupancy"][order]
    return best


def hmm_posteriors(params: dict, obs) -> np.ndarray:
    """(T, K) smoothed state posteriors for one sequence under fitted params."""
    o = _as_obs(obs)
    log_b = _log_gauss_diag(o, params["means"], params["variances"])
    gamma, _, _ = _forward_backward(
        log_b, np.log(params["trans"]), np.log(params["start"]), need_xi=False)
    return gamma


def hmm_trace_scores(params: dict, obs_list, tail: int = 32) -> dict:
    """
    Per-trace anomaly scores under a fitted HMM (higher = more suspect).

    Returns dict of np.ndarray over traces:
        occupancy      — mean posterior of the high-entropy state (PRIMARY)
        tail_occupancy — same over the final `tail` tokens
        switch_rate    — expected state transitions per token
    """
    occ, tail_occ, switch = [], [], []
    for obs in obs_list:
        o = _as_obs(obs)
        log_b = _log_gauss_diag(o, params["means"], params["variances"])
        gamma, xi_sum, _ = _forward_backward(
            log_b, np.log(params["trans"]), np.log(params["start"]))
        occ.append(gamma[:, -1].mean())
        tail_occ.append(gamma[-tail:, -1].mean())
        if xi_sum is not None and len(o) > 1:
            switch.append((xi_sum.sum() - np.trace(xi_sum)) / (len(o) - 1))
        else:
            switch.append(0.0)
    return {
        "occupancy": np.asarray(occ),
        "tail_occupancy": np.asarray(tail_occ),
        "switch_rate": np.asarray(switch),
    }


# ---------------------------------------------------------------------------
# Bayesian online change-point detection (Adams & MacKay 2007)
# ---------------------------------------------------------------------------

def bocpd_gaussian(x, hazard_lambda: float = 100.0, r_max: int = 256,
                   mu0=None, kappa0: float = 1.0, alpha0: float = 1.0,
                   beta0: float = 1.0) -> dict:
    """
    BOCPD on a scalar series with unknown-mean/unknown-variance Gaussian
    segments (Normal-Inverse-Gamma conjugate prior, Student-t predictive).

    Constant hazard 1/hazard_lambda; run-length posterior truncated at r_max.
    mu0=None uses the mean of the first 8 observations (causal init).

    The change-point branch evaluates x_t under the PRIOR predictive (a fresh
    segment is independent of the old run).  Reusing the run predictive there
    — a common implementation slip — makes P(r_t = 0) identically equal to
    the hazard, killing every change-point statistic.

    Returns dict (higher = more change-suspect):
        ecp          — expected number of change points, sum_t P(r_t = 0)
        mean_p0      — mean_t P(r_t = 0)
        map_cp_count — number of steps whose MAP run length resets to 0
        p0           — per-step P(r_t = 0) array (for plots)
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    if T < 2:
        return {"ecp": np.nan, "mean_p0": np.nan, "map_cp_count": np.nan,
                "p0": np.array([])}
    if mu0 is None:
        mu0 = float(x[: min(8, T)].mean())
    h = 1.0 / hazard_lambda
    log_h, log_1mh = np.log(h), np.log(1.0 - h)

    # sufficient statistics per run length (index = run length)
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])
    log_r = np.array([0.0])  # log P(r_0 = 0) = 1

    prior_scale = np.sqrt(beta0 * (kappa0 + 1) / (alpha0 * kappa0))

    p0_hist = np.empty(T - 1)
    map_resets = 0
    for t in range(1, T):
        xt = x[t]
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        pred_ll = student_t.logpdf(xt, df=2 * alpha, loc=mu, scale=scale)
        prior_ll = student_t.logpdf(xt, df=2 * alpha0, loc=mu0,
                                    scale=prior_scale)

        log_growth = log_r + pred_ll + log_1mh              # r -> r+1
        log_cp = logsumexp(log_r + log_h) + prior_ll        # r -> 0, fresh segment
        log_r = np.concatenate([[log_cp], log_growth])
        log_r -= logsumexp(log_r)

        # posterior updates: run length 0 restarts from the prior
        mu_new = (kappa * mu + xt) / (kappa + 1)
        beta_new = beta + kappa * (xt - mu) ** 2 / (2 * (kappa + 1))
        mu = np.concatenate([[mu0], mu_new])
        kappa = np.concatenate([[kappa0], kappa + 1])
        alpha = np.concatenate([[alpha0], alpha + 0.5])
        beta = np.concatenate([[beta0], beta_new])

        if len(log_r) > r_max:
            # fold overflow mass into the last kept run length
            log_r[r_max - 1] = logsumexp(log_r[r_max - 1:])
            log_r = log_r[:r_max]
            mu, kappa, alpha, beta = mu[:r_max], kappa[:r_max], alpha[:r_max], beta[:r_max]

        p0_hist[t - 1] = np.exp(log_r[0])
        if int(np.argmax(log_r)) == 0:
            map_resets += 1

    return {"ecp": float(p0_hist.sum()), "mean_p0": float(p0_hist.mean()),
            "map_cp_count": float(map_resets), "p0": p0_hist}


# ---------------------------------------------------------------------------
# Prediction-residual (innovation) scores — KalmanNet go/no-go precursor
# ---------------------------------------------------------------------------

def ar_innovation_scores(x, order: int = 2) -> dict:
    """
    Least-squares AR(order) fit per trace; innovation = one-step residual.

    Returns dict (higher = more suspect):
        mse_innov   — mean squared innovation (PRIMARY)
        innov_ratio — residual variance / trace variance (1 - R^2), clipped
    """
    x = np.asarray(x, dtype=float)
    if len(x) < order + 5:
        return {"mse_innov": np.nan, "innov_ratio": np.nan}
    Y = x[order:]
    A = np.column_stack([x[order - 1 - j: len(x) - 1 - j] for j in range(order)]
                        + [np.ones(len(Y))])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    res = Y - A @ coef
    var_x = max(float(x.var()), 1e-12)
    return {"mse_innov": float((res ** 2).mean()),
            "innov_ratio": float(np.clip(res.var() / var_x, 0.0, 2.0))}


def kalman_innovation_scores(x, q: float = 0.01, r: float = 1.0) -> dict:
    """
    Constant-velocity scalar Kalman filter; scores from the innovation stream.

    State = [level, slope]; F = [[1,1],[0,1]]; observation = level + noise.
    Fixed q/r (no tuning per trace — the point is whether innovations carry
    label signal at all, not to optimize the filter).

    Returns dict (higher = more suspect):
        mse_innov — mean squared innovation
        nis       — mean normalized innovation squared (v^2 / S)
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 4:
        return {"mse_innov": np.nan, "nis": np.nan}
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])
    Q = q * np.array([[0.25, 0.5], [0.5, 1.0]])
    s = np.array([x[0], 0.0])
    P = np.eye(2)
    v2, nis = [], []
    for t in range(1, len(x)):
        s = F @ s
        P = F @ P @ F.T + Q
        S = float(H @ P @ H + r)
        v = float(x[t] - H @ s)
        K_gain = (P @ H) / S
        s = s + K_gain * v
        P = P - np.outer(K_gain, H @ P)
        v2.append(v * v)
        nis.append(v * v / S)
    return {"mse_innov": float(np.mean(v2)), "nis": float(np.mean(nis))}
