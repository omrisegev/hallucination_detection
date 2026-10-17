"""
Unsupervised multivariate anomaly scorers over trace-level feature vectors.

Track A of the pivot-alternatives pilot (Step 151): can an anomaly/density
model over the same 16 spectral features replace L-SML as the unsupervised
aggregation layer?  Every scorer here is fit per eval cell on ALL unlabeled
samples of that cell and scores those same samples.  That transductive
fit-and-score protocol is the matched comparison, not leakage: L-SML
continuous itself estimates its eigenvector weights from the unlabeled
covariance of the eval cell it scores, so candidates and comparator have
identical information access (features of the cell, zero labels).

Conventions:
    - All scorers return ANOMALY scores: higher = more hallucination-suspect.
      Negate to get the package-wide confidence orientation (higher = correct).
    - Input X is raw (unstandardized); each scorer z-scores columns internally.
    - torch is imported lazily so the module stays importable in sklearn-only
      environments.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Feature-matrix assembly (canonical copy of scripts/logistic_oracle.py)
# ---------------------------------------------------------------------------

def is_saturated(arr, threshold: float = 0.40) -> bool:
    """True if more than `threshold` of the values equal the median."""
    a = np.asarray(arr, dtype=float)
    return float(np.mean(a == np.median(a))) > threshold


def build_feature_matrix(fd: dict, feat_list: list, n_samples: int):
    """
    Stack available features into X of shape (n_samples, n_features).

    Mirrors scripts/logistic_oracle.py build_X exactly: drops missing,
    length-mismatched and saturated features; median-imputes non-finite
    entries; returns (None, available) if fewer than 3 features survive.
    No sign orientation — anomaly models are direction-free in feature space.
    """
    available = [
        f for f in feat_list
        if f in fd and fd[f] is not None
        and len(fd[f]) == n_samples
        and not is_saturated(fd[f])
    ]
    if len(available) < 3:
        return None, available

    X = np.column_stack([np.asarray(fd[f], dtype=float) for f in available])
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        bad = ~np.isfinite(X[:, j])
        if bad.any():
            X[bad, j] = col_medians[j]
    return X, available


def _zscore_cols(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X - mu) / sd


# ---------------------------------------------------------------------------
# Density / distance baselines (sklearn only)
# ---------------------------------------------------------------------------

def mahalanobis_scores(X) -> np.ndarray:
    """Squared Mahalanobis distance under a Ledoit-Wolf shrunk covariance."""
    from sklearn.covariance import LedoitWolf
    Z = _zscore_cols(np.asarray(X, dtype=float))
    lw = LedoitWolf().fit(Z)
    return lw.mahalanobis(Z)


def gmm_nll_scores(X, n_components: int = 2, seed: int = 42,
                   reg_covar: float = 1e-3, n_init: int = 5) -> np.ndarray:
    """Negative log-likelihood under a full-covariance Gaussian mixture."""
    from sklearn.mixture import GaussianMixture
    Z = _zscore_cols(np.asarray(X, dtype=float))
    n_components = min(n_components, max(1, len(Z) // 20))
    gm = GaussianMixture(n_components=n_components, covariance_type='full',
                         reg_covar=reg_covar, n_init=n_init,
                         random_state=seed).fit(Z)
    return -gm.score_samples(Z)


def kde_nll_scores(X, bandwidth="scott") -> np.ndarray:
    """Negative log-likelihood under a Gaussian KDE on z-scored features."""
    from sklearn.neighbors import KernelDensity
    Z = _zscore_cols(np.asarray(X, dtype=float))
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(Z)
    return -kde.score_samples(Z)


def iforest_scores(X, seed: int = 42, n_estimators: int = 300) -> np.ndarray:
    """Isolation Forest anomaly score (negated sklearn score_samples)."""
    from sklearn.ensemble import IsolationForest
    Z = _zscore_cols(np.asarray(X, dtype=float))
    clf = IsolationForest(n_estimators=n_estimators, random_state=seed).fit(Z)
    return -clf.score_samples(Z)


# ---------------------------------------------------------------------------
# Autoencoders (torch, lazy import)
# ---------------------------------------------------------------------------

def _make_ae(d: int, hidden: int, bottleneck: int, seed: int):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(d, hidden), nn.ReLU(),
        nn.Linear(hidden, bottleneck),
        nn.Linear(bottleneck, hidden), nn.ReLU(),
        nn.Linear(hidden, d),
    )


def _ae_dims(d: int):
    hidden = max(3, min(8, d - 1))
    bottleneck = max(2, min(4, d // 2))
    return hidden, bottleneck


def ae_scores(X, bottleneck: int = None, hidden: int = None,
              epochs: int = 500, lr: float = 1e-2, weight_decay: float = 1e-3,
              seeds=(0, 1, 2, 3, 4)):
    """
    Plain autoencoder reconstruction error, averaged over seeds.

    Tiny architecture (d -> hidden -> bottleneck -> hidden -> d, ~450 params
    at d=16), full-batch Adam with a fixed epoch budget (deterministic, no
    early stopping).  Seed averaging of the per-sample reconstruction MSE is
    the main variance reduction at n of a few hundred.

    Returns (anomaly_scores, meta).
    """
    import torch
    Z = _zscore_cols(np.asarray(X, dtype=float))
    n, d = Z.shape
    h, b = _ae_dims(d)
    if hidden is not None:
        h = hidden
    if bottleneck is not None:
        b = bottleneck
    Xt = torch.tensor(Z, dtype=torch.float32)

    per_seed = []
    losses = []
    for seed in seeds:
        net = _make_ae(d, h, b, seed)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            rec = net(Xt)
            loss = ((rec - Xt) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            r = ((net(Xt) - Xt) ** 2).mean(dim=1).numpy()
        per_seed.append(r)
        losses.append(float(loss))
    scores = np.mean(per_seed, axis=0)
    return scores, {"hidden": h, "bottleneck": b, "final_losses": losses,
                    "seed_score_corr": float(np.corrcoef(per_seed[0], per_seed[-1])[0, 1])
                    if len(per_seed) > 1 else 1.0}


def prae_scores(X, inlier_frac: float = 0.8, gate_sigma: float = 0.5,
                lam: float = 10.0, bottleneck: int = None, hidden: int = None,
                epochs: int = 500, lr: float = 1e-2, weight_decay: float = 1e-3,
                seeds=(0, 1, 2, 3, 4)):
    """
    Probabilistic robust autoencoder scores (Lindenbaum, Aizenbud & Kluger,
    ICML 2021 — reimplemented in spirit with STG-style gates).

    Each sample i gets a stochastic gate z_i = clamp(mu_i + sigma*eps, 0, 1),
    eps ~ N(0,1) resampled per epoch.  Training loss:

        mean_i(z_i * r_i)  +  lam * (mean_i P(z_i > 0) - inlier_frac)^2

    where r_i is the per-sample reconstruction MSE and P(z_i > 0) =
    Phi(mu_i / sigma).  The gates let the AE exclude suspected outliers from
    shaping the manifold; `inlier_frac` is the target fraction kept (pre-
    registered, never tuned per cell — per-cell tuning would be label leakage).

    Score = reconstruction error of ALL samples under the robustly trained AE
    (gates affect training only), seed-averaged.  meta['gate_probs'] holds the
    seed-averaged P(z_i > 0) as a secondary anomaly signal (1 - gate).

    Returns (anomaly_scores, meta).
    """
    import torch
    from torch.distributions import Normal
    Z = _zscore_cols(np.asarray(X, dtype=float))
    n, d = Z.shape
    h, b = _ae_dims(d)
    if hidden is not None:
        h = hidden
    if bottleneck is not None:
        b = bottleneck
    Xt = torch.tensor(Z, dtype=torch.float32)
    std_normal = Normal(0.0, 1.0)

    per_seed, gates_seed, losses = [], [], []
    for seed in seeds:
        torch.manual_seed(seed)
        net = _make_ae(d, h, b, seed)
        mu = torch.full((n,), 0.5, requires_grad=True)
        opt = torch.optim.Adam(
            [{"params": net.parameters(), "weight_decay": weight_decay},
             {"params": [mu], "weight_decay": 0.0}], lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            eps = torch.randn(n) * gate_sigma
            z = torch.clamp(mu + eps, 0.0, 1.0)
            r = ((net(Xt) - Xt) ** 2).mean(dim=1)
            open_prob = std_normal.cdf(mu / gate_sigma)
            loss = (z * r).mean() + lam * (open_prob.mean() - inlier_frac) ** 2
            loss.backward()
            opt.step()
        with torch.no_grad():
            r = ((net(Xt) - Xt) ** 2).mean(dim=1).numpy()
            g = std_normal.cdf(mu / gate_sigma).numpy()
        per_seed.append(r)
        gates_seed.append(g)
        losses.append(float(loss))
    scores = np.mean(per_seed, axis=0)
    gate_probs = np.mean(gates_seed, axis=0)
    return scores, {"hidden": h, "bottleneck": b, "final_losses": losses,
                    "gate_probs": gate_probs,
                    "mean_gate": float(gate_probs.mean()),
                    "frac_closed": float((gate_probs < 0.5).mean())}


# ---------------------------------------------------------------------------
# Method registry (Track A runner iterates this)
# ---------------------------------------------------------------------------

TRACKA_METHODS = {
    "maha":    lambda X: (mahalanobis_scores(X), {}),
    "gmm2":    lambda X: (gmm_nll_scores(X, n_components=2), {}),
    "kde":     lambda X: (kde_nll_scores(X), {}),
    "iforest": lambda X: (iforest_scores(X), {}),
    "ae":      ae_scores,
    "prae":    prae_scores,
}

# ae/prae need enough samples to shape a manifold; below this the density
# baselines still run but the autoencoders are skipped (returns None).
AE_MIN_SAMPLES = 80
