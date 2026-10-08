r"""
A11 — two remaining published unsupervised conditions (Step 224).

Both are grounded against their extracted text and neither had been implemented here.
They are packaged together because each is a single objective with a single readout.

===========================================================================
RFAE — Sun, Li & Han, "Fractal autoencoder with redundancy regularization for
unsupervised feature selection", SCIENCE CHINA Information Sciences, Feb 2025,
Vol. 68 Iss. 2, 122103.  Extract: papers/extracted/fractal-autoencoder.md
===========================================================================
Motivation, in the paper's own words: "existing unsupervised feature selection methods
tend to prioritize selecting highly correlated features over exploring feature
diversity" — i.e. it is aimed at exactly the trap this channel is investigating.

Architecture (§3.3): a "fractal AE" = a correspondence network (CoNN) and a selection
network (SeNN). CoNN adds an EXTRACTOR layer (an elementwise weight vector w1 over the
features) before the hidden layer; SeNN adds a SELECTOR layer that takes only the top-k
features by |w1|. Both reconstruct the full input.

Objective, Eq. (7):
    min  ||X - Xhat||_F^2 + ||X - Xtil||_F^2  +  alpha*R(W_I) + beta*N(V)  +  phi*Q(W)
         \_________ L1 (CoNN + SeNN recon) __/  \____ L2 (sparsity) ____/    \_ L3 _/
  R(W_I) = ||w1||_1                                              Eq. (4)
  N(V)   = sum_j sqrt(sum_i v_ij^2)      group lasso on hidden   Eq. (5)
  Q(W)   = 1/(m(m-1)) sum_i sum_{j!=i} ||w_i||_2 ||w_j||_2 rho_ij^2   Eq. (6)
Selection: the top k features by w1 ("the feature index w1kmax is utilized").
Initialisation (Fig. 1 caption): extractor weights in [0.999, 0.9999]; extractor-to-
output weights in [0, 1].

DEVIATIONS
  R1. Eq. (6) is printed with the outer sums running to p (hidden nodes) while rho_ij is
      defined as being "between the ith and jth FEATURES". Those are inconsistent. We
      read w_i as the i-th ROW of the extractor-to-hidden matrix (feature i's outgoing
      weights) and sum over FEATURES with the 1/(m(m-1)) normaliser, which is the only
      reading under which rho_ij is defined. Flagged rather than silently chosen.
  R2. rho_ij is called "the mutual information coefficient". No estimator is given in
      the extracted text, and MI between continuous z-scored views at n~250 needs a
      binning choice that would itself be a free parameter. We use the squared
      |Spearman| already carried on the cell (`cell.rho`), which is the same quantity
      the paper's redundancy term is reaching for. Documented, not hidden.
  R3. alpha, beta, phi are "non-negative balancing parameters" with no stated values.
      We set them label-free by magnitude matching at initialisation, the same device
      LS-CAE uses: each regulariser is scaled so its epoch-0 magnitude equals the
      reconstruction term's. No value is tuned against our results.

===========================================================================
SCFS — Parsa, Zare & Ghatee, "Unsupervised Feature Selection based on Adaptive
Similarity Learning and Subspace Clustering", arXiv:1912.05458v1, 10 Dec 2019.
Extract: papers/extracted/unsupervised-feature-selection-based-on-adaptive-similarity.md
===========================================================================
NOTE ON ATTRIBUTION: the Step-224 reading list attributed this slot to "Ma, Wei, Huang &
Wang, Minimum-Redundant Subspace Learning with Self-Weighted Adaptive Graph, 2024". The
PDF in `papers/` is a DIFFERENT paper by DIFFERENT authors. Cite it as Parsa, Zare &
Ghatee 2019.

Objective, Eq. (4):
    min_{W, G>=0, GG^T 1 = 1}
        ||X - G G^T X||_F^2  +  alpha ||X W - G||_F^2  +  beta ||W||_{2,1}
  Term 1 is a SELF-EXPRESSIVE subspace-clustering model: G G^T acts as an implicitly
  learned sample-similarity matrix. Term 2 is a regression from the features onto the
  learned cluster structure. Term 3 is the row-sparsity that performs the selection.
Closed-form W step, Eq. (6)-(7):
    W = (alpha X^T X + beta D)^{-1} alpha X^T G,   D_ii = 1 / (2||w_i||_2 + eps)
Selection: rank features by the row norms ||w_i||_2 — "the more a feature is related to
the clusters, the more it is likely to be selected".

DEVIATIONS
  S1. The G step is not given in closed form in the extracted text. We use projected
      gradient descent on Eq. (4) with the non-negativity projection G <- max(G, 0)
      followed by row normalisation to satisfy GG^T 1 = 1 approximately. Alternating
      W (closed form) and G (projected gradient) to convergence, as the paper's §3.3
      describes ("fixing one element and finding the optimum value for the other").
  S2. c (number of clusters) is not derivable from the data without a rule; we use
      c = 2, the binary correct/incorrect structure this pipeline is built around, and
      report it. This IS a prior about the task, though not about which features matter.
  S3. alpha, beta set by the same label-free magnitude-matching as RFAE.

Both selectors emit fixed sizes plus one data-driven size. Every size rule here is OURS:
neither paper provides one (RFAE takes k as given; SCFS ranks and takes a top-k).

Determinism: all draws from the passed `rng`; torch.set_num_threads(1). Never raises.
"""

import numpy as np
import torch

from . import register

K_GRID = (3, 4, 5, 6, 7, 8)
EPOCHS = 220
BATCH = 64
HIDDEN = 32
LR = 5e-3
REG_FRAC = 0.1               # deviation R3: regulariser magnitude as a fraction of L1
R_MAX = 1200
SCFS_ITERS = 200
SCFS_C = 2
SCFS_LR = 5e-2
MIN_SET = 3
_EPS = 1e-8

_EXPECTED = (tuple(f"rfae.k{k}" for k in K_GRID) + ("rfae",)
             + tuple(f"scfs.k{k}" for k in K_GRID) + ("scfs",))


def _fallback(variant, p, err):
    return {'variant': variant, 'cols': np.arange(p, dtype=np.int64),
            'fallback': True, 'diag': {'error': str(err)}}


def _elbow(ks, vals):
    """Our size rule (both papers take k as given). Smallest k within 10% of the grid's
    total improvement from the best; guarded against a flat/non-monotone curve."""
    v = np.asarray(vals, dtype=float)
    span = float(np.nanmax(v) - np.nanmin(v))
    if not np.isfinite(span) or span < 1e-12:
        return list(ks)[len(ks) // 2]
    th = float(np.nanmin(v)) + 0.10 * span
    for k, x in zip(ks, v):
        if np.isfinite(x) and x <= th:
            return k
    return list(ks)[int(np.nanargmin(v))]


# ---------------------------------------------------------------------------
# RFAE
# ---------------------------------------------------------------------------

def _rfae_fit(X, rho2, k, seed):
    """One RFAE fit. Returns (selected cols, val recon, w1)."""
    torch.manual_seed(int(seed))
    gen = torch.Generator().manual_seed(int(seed))
    n, m = X.shape
    B = int(min(BATCH, n))
    # Fig. 1: extractor weights init in [0.999, 0.9999]
    w1 = (0.999 + 0.0009 * torch.rand(m, generator=gen)).requires_grad_(True)
    W = (torch.randn(m, HIDDEN, generator=gen) * 0.3).requires_grad_(True)  # extr->hidden
    V = (torch.randn(HIDDEN, m, generator=gen) * 0.3).requires_grad_(True)  # hidden->out
    bh = torch.zeros(HIDDEN, requires_grad=True)
    bo = torch.zeros(m, requires_grad=True)
    opt = torch.optim.Adam([w1, W, V, bh, bo], lr=LR)
    R2 = torch.tensor(rho2, dtype=torch.float32)

    def decode(Z):
        # tanh hidden ("the nonlinear activation function f") into a LINEAR output ("g").
        # The output must be linear: the inputs are z-scored and therefore mean-zero, and
        # a non-negative output nonlinearity cannot reconstruct them at all — an earlier
        # ReLU-output draft selected noise for exactly that reason.
        return torch.tanh(Z @ W + bh) @ V + bo

    scales = {}
    for ep in range(EPOCHS):
        idx = torch.randperm(n, generator=gen)[:B]
        Xb = X[idx]
        # CoNN: elementwise extractor gate over ALL features
        Xhat = decode(Xb * w1[None, :])
        # SeNN: selector layer keeps only the current top-k features by w1
        topk = torch.topk(w1.detach(), k).indices
        mask = torch.zeros(m)
        mask[topk] = 1.0
        Xtil = decode(Xb * mask[None, :] * w1[None, :])
        L1 = ((Xb - Xhat) ** 2).mean() + ((Xb - Xtil) ** 2).mean()
        R = w1.abs().sum()                                        # Eq. (4)
        N = torch.sqrt((V ** 2).sum(dim=0) + _EPS).sum()          # Eq. (5)
        wn = torch.sqrt((W ** 2).sum(dim=1) + _EPS)               # ||w_i||_2 per FEATURE
        Q = ((wn[:, None] * wn[None, :]) * R2).sum() / max(m * (m - 1), 1)   # Eq. (6)
        if ep == 0:      # deviation R3: magnitude-match each regulariser once, label-free
            # Each regulariser is set to REG_FRAC of the reconstruction magnitude, not to
            # equal it. At parity the L1 term's constant gradient swamps the
            # reconstruction signal and drives every extractor weight down uniformly,
            # which makes the top-k readout arbitrary (observed directly).
            base = REG_FRAC * (float(L1.detach().abs()) + _EPS)
            scales = {'a': base / (float(R.detach().abs()) + _EPS),
                      'b': base / (float(N.detach().abs()) + _EPS),
                      'p': base / (float(Q.detach().abs()) + _EPS)}
        loss = L1 + scales['a'] * R + scales['b'] * N + scales['p'] * Q   # Eq. (7)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            w1.clamp_(0.0, 1.0)

    with torch.no_grad():
        w = w1.detach().numpy()
        sel = np.array(sorted(np.argsort(w)[::-1][:k].tolist()), dtype=np.int64)
        Xs = X[:, torch.tensor(sel)]
        B_, *_ = np.linalg.lstsq(Xs.numpy(), X.numpy(), rcond=None)
        err = float(np.mean((Xs.numpy() @ B_ - X.numpy()) ** 2))
    return sel, err, w


# ---------------------------------------------------------------------------
# SCFS
# ---------------------------------------------------------------------------

def _scfs_fit(Xn, c, iters, rng):
    """Alternating minimisation of Eq. (4). Returns row norms ||w_i||_2."""
    n, m = Xn.shape
    X = torch.tensor(Xn, dtype=torch.float64)
    G = torch.tensor(rng.random((n, c)), dtype=torch.float64)
    G = G / G.sum(1, keepdim=True).clamp_min(_EPS)
    XtX = X.T @ X
    # deviation S3: alpha, beta by magnitude matching on the initial G, so the
    # subspace-clustering and regression terms start comparable.
    a0 = float(((X - G @ G.T @ X) ** 2).sum())
    alpha = float(np.clip(a0 / (float((G ** 2).sum()) + _EPS), 1e-3, 1e3))
    beta = alpha * 0.1
    W = torch.zeros(m, c, dtype=torch.float64)
    for _ in range(iters):
        # --- W step, closed form, Eq. (6)-(7) ---------------------------------------
        d = 1.0 / (2.0 * torch.sqrt((W ** 2).sum(1) + _EPS) + _EPS)
        A = alpha * XtX + beta * torch.diag(d)
        W = torch.linalg.solve(A + _EPS * torch.eye(m, dtype=torch.float64),
                               alpha * X.T @ G)
        # --- G step, projected gradient on Eq. (4) (deviation S1) -------------------
        # The gradient of ||X - G G^T X||_F^2 with respect to G is taken by AUTOGRAD, not
        # by hand. A hand-derived version of this term selected a pure-noise column in
        # the planted smoke world; differentiating the objective as written removes the
        # derivation as a source of error.
        Gv = G.detach().requires_grad_(True)
        obj = ((X - Gv @ Gv.T @ X) ** 2).sum() + alpha * ((X @ W - Gv) ** 2).sum()
        obj.backward()
        with torch.no_grad():
            g = Gv.grad
            g = g / (g.norm() + _EPS)          # normalised step: the two terms differ in
            G = torch.clamp(Gv - SCFS_LR * g, min=0.0)   # scale by orders of magnitude
            G = G / G.sum(1, keepdim=True).clamp_min(_EPS)   # GG^T 1 = 1, approximately
    return np.sqrt((W.numpy() ** 2).sum(axis=1))


@register('a11_rfae_scfs')
def a11_rfae_scfs(cell, rng, cache=None):
    torch.set_num_threads(1)
    p = cell.p
    out = []
    try:
        V0 = np.asarray(cell.V, dtype=np.float64)
        n = V0.shape[0]
        R = int(min(n, R_MAX))
        rows = np.sort(rng.choice(n, size=R, replace=False)) if R < n else np.arange(n)
        Vr = V0[rows]
        Xt = torch.tensor(Vr, dtype=torch.float32)
        rho2 = np.asarray(cell.rho, dtype=float) ** 2       # deviation R2
        np.fill_diagonal(rho2, 0.0)

        # ---- RFAE ---------------------------------------------------------------
        seed = int(rng.integers(2 ** 31))
        errs, sels = [], {}
        for k in K_GRID:
            if k >= p:
                out.append(_fallback(f'rfae.k{k}', p, f'k={k} >= pool {p}'))
                errs.append(np.nan)
                continue
            s, e, w = _rfae_fit(Xt, rho2, k, seed + k)
            sels[k] = s
            errs.append(e)
            out.append({'variant': f'rfae.k{k}', 'cols': s,
                        'diag': {'k': int(k), 'recon_mse': round(float(e), 6),
                                 'w1': [round(float(x), 4) for x in w],
                                 'note': 'Eq.(7): CoNN+SeNN reconstruction + L1 + group '
                                         'lasso + redundancy Q(W)'}}
                       if len(s) >= MIN_SET
                       else _fallback(f'rfae.k{k}', p, 'selection < 3'))
        if sels:
            kk = _elbow(sorted(sels), [errs[list(K_GRID).index(k)] for k in sorted(sels)])
            out.append({'variant': 'rfae', 'cols': sels[kk],
                        'diag': {'k_chosen': int(kk),
                                 'note': 'size rule is OURS — RFAE takes k as given'}})
        else:
            out.append(_fallback('rfae', p, 'no admissible k'))

        # ---- SCFS ---------------------------------------------------------------
        wnorm = _scfs_fit(Vr, SCFS_C, SCFS_ITERS,
                          np.random.default_rng(int(rng.integers(2 ** 31))))
        order = np.argsort(wnorm)[::-1]
        for k in K_GRID:
            if k >= p:
                out.append(_fallback(f'scfs.k{k}', p, f'k={k} >= pool {p}'))
                continue
            s = np.array(sorted(order[:k].tolist()), dtype=np.int64)
            out.append({'variant': f'scfs.k{k}', 'cols': s,
                        'diag': {'k': int(k), 'c': SCFS_C,
                                 'row_norms': [round(float(x), 5) for x in wnorm],
                                 'note': 'Eq.(4) self-expressive subspace clustering + '
                                         'L2,1 regression; rank by ||w_i||_2'}})
        gaps = np.diff(np.sort(wnorm)[::-1][:max(K_GRID) + 1])
        kk = int(np.argmin(gaps[MIN_SET - 1:]) + MIN_SET) if len(gaps) > MIN_SET else 5
        kk = int(np.clip(kk, MIN_SET, min(8, p - 1)))
        out.append({'variant': 'scfs', 'cols': np.array(sorted(order[:kk].tolist()),
                                                        dtype=np.int64),
                    'diag': {'k_chosen': kk,
                             'note': 'size by the largest gap in the row-norm profile — '
                                     'OURS; the paper ranks and takes a top-k'}})
        return out

    except Exception as e:
        return [_fallback(v, p, e) for v in _EXPECTED]


# ---------------------------------------------------------------------------
# smoke()
# ---------------------------------------------------------------------------

def smoke():
    import time
    from ..selector_bench import UnlabeledCell
    from ..fusion_utils import zscore
    from ..subset_sweep import CANONICAL_POOL

    # 3 latent directions x 3 correlated copies (9 informative, cols 0-8) + 3 pure noise
    # (cols 9-11). Correlated BLOCKS are required for a reconstruction-based method to
    # have any signal at all: if every column were independent, none could be
    # reconstructed from the others and reconstruction error could not rank them. A
    # redundancy-penalising rule should take roughly one representative per block and
    # avoid the noise, so the fair assertion is on WHICH columns are picked at small k,
    # not on saturating the informative set.
    rng = np.random.default_rng(20260805)
    n = 400
    cols, info, block = [], [], []
    for g in range(3):
        lat = rng.standard_normal(n)
        for _ in range(3):
            cols.append(zscore(lat + 0.2 * rng.standard_normal(n)))
            info.append(True)
            block.append(g)
    for _ in range(3):
        cols.append(zscore(rng.standard_normal(n)))
        info.append(False)
        block.append(-1)
    V = np.column_stack(cols)
    p = V.shape[1]
    info = np.array(info)
    block = np.array(block)
    cell = UnlabeledCell(domain='smoke', cell_key='rfae_scfs',
                         pool=list(CANONICAL_POOL[:p]),
                         pool_bits=np.arange(p, dtype=np.uint8), V=V,
                         anchor=zscore(V[:, 0]), anchor_name=CANONICAL_POOL[0],
                         rho=np.abs(np.corrcoef(V.T)))

    t0 = time.time()
    s1 = a11_rfae_scfs(cell, np.random.default_rng([0, 9]))
    el = time.time() - t0
    s2 = a11_rfae_scfs(cell, np.random.default_rng([0, 9]))
    by1 = {s['variant']: s for s in s1}
    by2 = {s['variant']: s for s in s2}
    assert set(by1) == set(_EXPECTED), \
        f"variant set mismatch: {sorted(set(_EXPECTED) ^ set(by1))}"
    for v in _EXPECTED:
        assert list(by1[v]['cols']) == list(by2[v]['cols']), f"{v}: not deterministic"
        assert not by1[v].get('fallback'), f"{v} fell back: {by1[v]['diag']}"

    # At k=3 the informative blocks are the only reconstructable structure, so a working
    # rule must beat chance on signal-vs-noise. Chance at k=3 from 9 informative of 12
    # is 2.25 informative; requiring all 3 is a real bar without demanding a particular
    # block assignment.
    for fam in ('rfae.k3', 'scfs.k3'):
        sel = [int(c) for c in by1[fam]['cols']]
        n_info = int(info[sel].sum())
        assert n_info == 3, f"{fam} picked {n_info}/3 informative (chance 2.25): {sel}"

    for fam in ('rfae.k6', 'scfs.k6'):
        sel = [int(c) for c in by1[fam]['cols']]
        assert int(info[sel].sum()) >= 4, \
            f"{fam} at k=6 picked {int(info[sel].sum())}/6 informative: {sel}"

    r3 = sorted(int(c) for c in by1['rfae.k3']['cols'])
    s3 = sorted(int(c) for c in by1['scfs.k3']['cols'])
    print(f"    [note] a11 smoke: rfae.k3={r3} blocks={block[r3].tolist()} | "
          f"scfs.k3={s3} blocks={block[s3].tolist()} | "
          f"informative=0..8 noise=9..11 | rfae k*={by1['rfae']['diag'].get('k_chosen')} "
          f"scfs k*={by1['scfs']['diag'].get('k_chosen')} {el:.1f}s")
    assert el < 400.0, f"runtime {el:.1f}s — pathological"
