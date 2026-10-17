"""ell-0 sparse Canonical Correlation Analysis — the linear arm, for feature selection.

Reimplementation of

    Ofir Lindenbaum (Yale), Moshe Salhov (TAU), Amir Averbuch (TAU), Yuval Kluger (Yale),
    "l0-based Sparse Canonical Correlation Analysis", arXiv:2010.05620v2, 8 Jun 2021.

grounded against `papers/extracted/l0-based-sparse-canonical-correlation-analysis.md`;
digest card at `papers/digests/l0-based-sparse-canonical-correlation-analysis.md`. Every
equation number below is the paper's own.

WHY IT IS HERE. Every closure in the U-PCR feature-selection item so far concerns
`rho_hat`, the correlation with correctness. This method's criterion is a DIFFERENT
quantity — shared structure between two measurement channels — so those floors do not
bound it, and its gates enter the loss JOINTLY with the canonical directions, so a
feature's gate depends on the others. Step 222 closed the marginal per-feature family;
this is the pre-registered set-level follow-on.

WHAT IS IMPLEMENTED

  * `cca_leverage_scores`  — no training, no free parameter. |leading left/right singular
    vectors| of the cross-covariance C_xy. This is the paper's gate initialization (S3.2)
    with the thresholding removed, so it has no percentile knob.
  * `cca_init_mu`          — the paper's S3.2 initialization LITERALLY: threshold |C_xy| at
    the r-th percentile, leading singular vectors of the thresholded matrix, threshold
    their absolute values at the same percentile, mu = ubar + 0.5.
  * `train_l0cca`          — the trained arm. Stochastic gates (Eq. 4) on both views,
    trained by Adam on the l0-regularized total-correlation loss (Eq. 3 / Eq. 5 with
    identity transforms), total correlation as the S3.4 trace.
  * `l0cca_gates`          — seed-averaged `train_l0cca`, matching the stability protocol
    the DUFS selector in `a2_groupfs.py` already uses.
  * `heldout_total_correlation` — the paper's own LABEL-FREE model-selection criterion for
    lambda: fit the canonical directions on train rows, evaluate total correlation on
    held-out rows.

WHICH CORRELATION TERM THIS IS. The paper carries two. Eq. 1/Eq. 3 use rho, the SINGLE
canonical PAIR's correlation, and the paper's own linear l0-CCA estimates one pair
(phi, eta). Eq. 5 uses rho-bar, the TOTAL correlation over d coordinates, which S3.4 then
expresses as the trace of Cy^{-1/2} Cyx Cx^{-1} Cxy Cy^{-1/2}. What is implemented here is
the Eq. 5 / S3.4 form at full rank, d = min(Dx, Dy) — NOT Eq. 3. The trace is the sum of
the SQUARED canonical correlations, which is the paper's trace identity rather than its
prose gloss ("the sum over d correlation values"); the squared/trace form is the one
reproduced, in both the training loss and the held-out criterion.

THE LINEAR SIMPLIFICATION, STATED EXPLICITLY. Eq. 5 puts two neural sub-nets between the
gates and the loss. For the LINEAR model that trace, evaluated on the gated inputs
themselves, already equals the CCA objective at its optimum over all linear maps. So the
linear arm needs no weight matrices at all: the loss is closed-form in the gates, and
every gradient step solves the CCA exactly rather than descending towards it. Two linear
solves replace the matrix square roots, since
    tr(Cy^{-1/2} Cyx Cx^{-1} Cxy Cy^{-1/2}) = tr(Cy^{-1} Cyx Cx^{-1} Cxy).

DEVIATIONS from the paper, all declared:
  1. Rows are SUBSAMPLED to `R_MAX` per fit. Full-batch gradient descent is the paper's
     own choice (S3.4: "we apply full batch gradient decent to the loss in Eq. 5") and is
     kept; the subsample is ours, and is not binding since N >> D here.
  2. Adam rather than plain gradient descent; `GATE_LR = 1e-2`, TWICE the paper's 0.005;
     and 300 epochs against its 10,000. The paper's setting is for D = 800 in N << D; we
     have D <= 30 in N >> D. `EPOCHS` was fixed from the synthetic check in
     `tests/test_l0cca_recovery.py`, which is also the artifact for the claim that 500
     epochs changes no support.
  3. `sigma = 0.25` follows the paper's own Appendix A.1 ("we used sigma = 0.25, which
     worked well in our experiments"), NOT the sigma = 0.5 of STG/DUFS that
     `a2_groupfs.py` uses. The two methods disagree about this and each is followed at
     home.
  4. `cca_init_mu` implements the S3.2 gate initialization faithfully, but the TRAINED arm
     does not use it — `train_l0cca` starts from the fair mu = 0.5 of Algorithm 1 unless a
     caller passes `mu_init_*`, and the harness does not. So the trained arm is the
     paper's method with Algorithm 1's initialization rather than S3.2's; S3.2 is scored
     separately as a standalone ranking instead.
  5. Our regime is N >> D; the paper's motivating case is N << D. The degeneracy argument
     for sparsity does not transfer — only the nuisance-removal one does.
"""
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants. Fixed here so an experiment cannot quietly tune them per cell.
# ---------------------------------------------------------------------------
STG_SIGMA = 0.25         # paper Appendix A.1, explicitly NOT the STG/DUFS 0.5
MU_INIT = 0.5            # paper S3.2 / Algorithm 1: "fair" Bernoulli gates
GATE_LR = 1e-2
EPOCHS = 300             # see tests/test_l0cca_recovery.py; 500 changes no support
RIDGE = 1e-3             # the gamma*I of S3.4, on z-scored columns (diag(C) ~ 1)
R_MAX = 2000             # rows subsampled per fit; N >> D, so this is not binding
N_SEEDS_STABILITY = 5    # matches a2_groupfs.N_SEEDS_STABILITY
_SQRT2 = 1.4142135623730951


def _prep(X, Y, rng=None, r_max=R_MAX):
    """Center both views and subsample rows. Returns float64 numpy arrays."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    if rng is not None and r_max is not None and n > r_max:
        keep = np.sort(rng.choice(n, size=int(r_max), replace=False))
        X, Y = X[keep], Y[keep]
    return X - X.mean(0), Y - Y.mean(0)


def _cov(A, B, ridge=0.0):
    n = A.shape[0]
    C = (A.T @ B) / max(n - 1, 1)
    if ridge:
        C = C + ridge * torch.eye(C.shape[0], dtype=C.dtype)
    return C


def total_correlation(Xg, Yg, ridge=RIDGE):
    """The S3.4 total correlation, tr(Cy^-1 Cyx Cx^-1 Cxy). Differentiable in the inputs.

    Equals the sum of the squared canonical correlations over ALL components — the
    paper's own expression for the linear case, with no `d` to choose.
    """
    Cx = _cov(Xg, Xg, ridge)
    Cy = _cov(Yg, Yg, ridge)
    Cxy = _cov(Xg, Yg)
    A = torch.linalg.solve(Cx, Cxy)          # Cx^-1 Cxy   (Dx x Dy)
    B = torch.linalg.solve(Cy, Cxy.T)        # Cy^-1 Cyx   (Dy x Dx)
    return (B * A.T).sum()                   # tr(B @ A)


def _p_open(mu, sigma=STG_SIGMA):
    """E||z||_0 = sum_i P(z_i >= 0), the Eq.-4 relaxation's l0 surrogate."""
    return 0.5 * (1.0 + torch.erf(mu / (sigma * _SQRT2)))


def train_l0cca(X, Y, lam, seed=0, epochs=EPOCHS, sigma=STG_SIGMA, lr=GATE_LR,
                ridge=RIDGE, mu_init_x=None, mu_init_y=None):
    """Train the gates. Returns (mu_x, mu_y) as float64 numpy arrays.

    Loss (Eq. 3, continuous relaxation Eq. 4, total correlation S3.4):
        L = -tr(Cy^-1 Cyx Cx^-1 Cxy)  +  lam * ( sum_i P(zx_i>=0) + sum_i P(zy_i>=0) )
    with  z = clamp(mu + eps, 0, 1),  eps ~ N(0, sigma^2), and C computed on the gated,
    centered views. One Monte Carlo sample per gradient step, as the paper specifies.

    lam_x = lam_y = lam throughout, which is what the paper does in every experiment
    (Fig. 2, S5.2, A.2 all sweep a single lambda = lambda_x = lambda_y). It also makes the
    two views' mu directly comparable, which the ranking downstream relies on.
    """
    # An explicit generator, not `torch.manual_seed`: this is called thousands of times
    # per run and must not mutate process-global RNG state that other code shares. Matches
    # `a2_groupfs._train_dufs`, which takes the same care.
    gen = torch.Generator().manual_seed(int(seed))
    Xt = torch.tensor(X, dtype=torch.float64)
    Yt = torch.tensor(Y, dtype=torch.float64)
    dx, dy = Xt.shape[1], Yt.shape[1]

    mx = torch.full((dx,), MU_INIT, dtype=torch.float64) if mu_init_x is None else \
        torch.tensor(np.asarray(mu_init_x, dtype=np.float64))
    my = torch.full((dy,), MU_INIT, dtype=torch.float64) if mu_init_y is None else \
        torch.tensor(np.asarray(mu_init_y, dtype=np.float64))
    mx.requires_grad_(True)
    my.requires_grad_(True)
    opt = torch.optim.Adam([mx, my], lr=lr)

    for _ in range(int(epochs)):
        opt.zero_grad()
        ex = torch.randn(dx, dtype=torch.float64, generator=gen)
        ey = torch.randn(dy, dtype=torch.float64, generator=gen)
        zx = torch.clamp(mx + ex * sigma, 0.0, 1.0)
        zy = torch.clamp(my + ey * sigma, 0.0, 1.0)
        Xg, Yg = Xt * zx, Yt * zy
        loss = -total_correlation(Xg, Yg, ridge) + lam * (
            _p_open(mx, sigma).sum() + _p_open(my, sigma).sum())
        loss.backward()
        opt.step()

    return mx.detach().numpy().copy(), my.detach().numpy().copy()


def l0cca_gates(X, Y, lam, seeds=None, **kw):
    """Seed-averaged gates. Mirrors the stability protocol of `a2_groupfs.dufs_pf_gates`:
    the objective is non-convex and one seed is not a measurement."""
    seeds = list(range(N_SEEDS_STABILITY)) if seeds is None else list(seeds)
    outs = [train_l0cca(X, Y, lam, seed=s, **kw) for s in seeds]
    return (np.mean([o[0] for o in outs], axis=0),
            np.mean([o[1] for o in outs], axis=0))


# ---------------------------------------------------------------------------
# closed form — no training
# ---------------------------------------------------------------------------
def cca_leverage_scores(X, Y):
    """|leading left/right singular vectors| of C_xy. Returns (score_x, score_y).

    The paper's S3.2 initialization with the percentile thresholding removed, so it
    carries no knob at all. Higher = more cross-channel leverage.
    """
    Xc, Yc = _prep(X, Y)
    Cxy = (Xc.T @ Yc) / max(Xc.shape[0] - 1, 1)
    U, _, Vt = np.linalg.svd(Cxy, full_matrices=False)
    return np.abs(U[:, 0]), np.abs(Vt[0, :])


def cca_init_mu(X, Y, r=50.0):
    """The paper's S3.2 gate initialization, literally. Returns (mu_x, mu_y).

    Threshold |C_xy| at the r-th percentile, take the leading singular vectors of the
    thresholded matrix, threshold their absolute values at the SAME percentile, and set
    mu = ubar + 0.5. Everything below the percentile ties at exactly 0.5 — a real
    property of the construction, not a bug, and one an experiment using this as a
    ranking has to report.
    """
    Xc, Yc = _prep(X, Y)
    Cxy = (Xc.T @ Yc) / max(Xc.shape[0] - 1, 1)
    delta = np.percentile(np.abs(Cxy), r)
    Cbar = np.where(np.abs(Cxy) > delta, Cxy, 0.0)
    U, _, Vt = np.linalg.svd(Cbar, full_matrices=False)
    u, v = np.abs(U[:, 0]), np.abs(Vt[0, :])
    ubar = np.where(u > np.percentile(u, r), u, 0.0)
    vbar = np.where(v > np.percentile(v, r), v, 0.0)
    return ubar + MU_INIT, vbar + MU_INIT


def heldout_total_correlation(Xtr, Ytr, Xva, Yva, zx, zy, ridge=RIDGE, d=None):
    """The paper's label-free criterion for lambda: canonical directions fitted on TRAIN
    rows, total correlation measured on HELD-OUT rows.

    Returns the sum of squared per-component correlations on the validation rows. Without
    the held-out step the criterion is monotone in the number of open gates and would
    always pick lambda = 0.

    Two details that decide whether the criterion measures anything:

    * The validation rows are centered on the TRAIN mean, not their own. Centering them on
      their own mean forces the projected scores to exactly mean zero, which inflates the
      measured correlation by O(1/n_val) — an optimistic bias, not a label leak, but one
      that grows precisely on the small cells.
    * Components whose TRAIN singular value is numerically zero are DROPPED. A closed gate
      zeroes a column without removing it, so T is rank deficient with
      rank(T) <= min(#open_x, #open_y). Those null directions carry no train correlation
      but a nonzero noise correlation on the validation rows, each contributing ~1/n_val.
      Keeping them would make the criterion's noise floor a function of how many gates are
      CLOSED — i.e. of the very quantity lambda controls. Found in review.
    """
    zx = np.asarray(zx, dtype=np.float64)
    zy = np.asarray(zy, dtype=np.float64)
    Xt, Yt = np.asarray(Xtr, float) * zx, np.asarray(Ytr, float) * zy
    mx, my = Xt.mean(0), Yt.mean(0)
    Xt, Yt = Xt - mx, Yt - my
    n = Xt.shape[0]
    Cx = (Xt.T @ Xt) / max(n - 1, 1) + ridge * np.eye(Xt.shape[1])
    Cy = (Yt.T @ Yt) / max(n - 1, 1) + ridge * np.eye(Yt.shape[1])
    Cxy = (Xt.T @ Yt) / max(n - 1, 1)
    ix = np.linalg.inv(np.linalg.cholesky(Cx))          # Cx^-1/2 up to rotation
    iy = np.linalg.inv(np.linalg.cholesky(Cy))
    T = ix @ Cxy @ iy.T
    U, sv, Vt = np.linalg.svd(T, full_matrices=False)
    rank = int(np.sum(sv > max(sv[0], 1.0) * 1e-8)) if sv.size else 0
    d = rank if d is None else int(min(d, rank))
    if d <= 0:
        return 0.0
    Wx, Wy = ix.T @ U[:, :d], iy.T @ Vt[:d, :].T

    Xv = (np.asarray(Xva, float) * zx) - mx
    Yv = (np.asarray(Yva, float) * zy) - my
    P, Q = Xv @ Wx, Yv @ Wy
    tot = 0.0
    for i in range(d):
        p, q = P[:, i], Q[:, i]
        sp, sq = p.std(), q.std()
        if sp < 1e-12 or sq < 1e-12:
            continue
        c = float(np.mean((p - p.mean()) * (q - q.mean())) / (sp * sq))
        tot += c * c
    return float(tot)
