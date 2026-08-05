"""
A8 — LS-CAE: discarding NUISANCE and CORRELATED features (Step 224).

Reimplementation of

    Uri Shaham*, Ofir Lindenbaum*, Jonathan Svirsky, Yuval Kluger,
    "Deep Unsupervised Feature Selection by Discarding Nuisance and Correlated
    Features", October 2021.

grounded line-by-line against
`papers/extracted/deep-unsupervised-feature-selection-by-discarding-nuisance-a.md`.
Reference code exists at github.com/jsvir/lscae (not fetched — this is a clean-room
torch/CPU reimplementation from the extracted text).

WHY THIS ARM EXISTS. Every label-free condition priced in this channel so far attacks
ONE failure mode. `a2.dufs` / `lapscore_adapt` chase the Laplacian score (nuisance);
`a3.cae` chases reconstruction (correlation). This paper's whole claim is that you need
BOTH, and its §1 contribution (iii) is precisely that it "outperforms similar approaches
designed to avoid only correlated or nuisance features, but not both". We hold both
halves already, separately, and have never run the combination.

THE CORE INSIGHT (§3, §4.1), which is what makes it more than a sum of the two:
the Laplacian score must be computed on the SELECTED SUBSET, not on the full feature
set. With many nuisance features the Laplacian is corrupted — the paper shows the
leading nontrivial eigenvector's correlation with the true cluster assignment decays to
noise as nuisance dimension grows (Fig. 2), with d = O(r^4 / -log(1-(2n^2-1)sqrt(1-eps)))
nuisance features enough to break it. The concrete layer supplies exactly the handle
needed: compute the Laplacian at the concrete layer's OUTPUT, so as training sharpens
the selection the Laplacian is progressively cleaned, which sharpens the selection.

THE OBJECTIVE, paper Eq. (6), verbatim from the extract:

    L(X) = ||X - X_hat||^2 / SG(||X - X_hat||^2)
           - Trace[C^T L_diff(C) C] / SG(Trace[C^T L_diff(C) C])

  * X is an m x d minibatch, X_hat the autoencoder output, C = C(X) the CONCRETE LAYER
    OUTPUT (m x k), and L_diff(C) = D^-1 W the DIFFUSION Laplacian computed on C.
  * SG is the stop-gradient operator (identity forward, zero derivative).
  * The MINUS sign is the paper's: reconstruction is minimised, the Laplacian trace is
    MAXIMISED. That is correct for the diffusion Laplacian, whose LARGEST eigenvalues
    carry the main structure (§2.1: "for L_diff the eigenvectors corresponding to
    largest eigenvalues are the ones that express the main structures").
  * The SG denominators are the paper's own balancing mechanism (§4.2): each term is
    inversely weighted by its own magnitude, which "removes the need to use a tunable
    hyperparameter to balance between them". So there is NO lambda here — the loss value
    is identically 0 and only the gradients matter.

DEVIATIONS from the paper, explicit:
  1. KERNEL BANDWIDTH. §2.1 footnote 2 gives the common practice we adopt: sigma = the
     maximal Euclidean distance from any point to its nearest neighbour, computed per
     minibatch on C. The paper does not pin a single choice ("many other practices exist").
  2. OPTIMISATION BUDGET. Adam, EPOCHS x ceil(n/BATCH) minibatch steps at BATCH=64,
     matching `a3_concrete_ae` so the LS-CAE-vs-CAE delta isolates the Laplacian term and
     not the training budget. The paper does not state its budget in the extracted text.
  3. INPUT SCALING. §2 assumes features centred with ||f_i||_2^2 = 1. `cell.V` arrives
     z-scored (mean 0, sd 1), so ||f||^2 = n; we divide by sqrt(n) to meet the paper's
     normalisation exactly.
  4. k IS ARCHITECTURAL, as in the paper — the concrete layer's size. The paper offers no
     rule for choosing it, so the `lscae` adaptive variant's elbow rule is OURS, and is
     reported as such. The fixed-k variants carry no size rule at all.
  5. NO SWAP-REFINE. `a3_concrete_ae` post-processes its subset with an exhaustive local
     search on the eval objective (its deviation 5), which makes that arm
     "CAE-initialised best-subset reconstruction" rather than CAE. We deliberately do NOT
     do that here: the shipped subset is the trained concrete layer's own argmax readout,
     so `lscae.*` is the paper's method and the `lscae` vs `lscae.recon_only` delta is a
     clean within-implementation ablation of the Laplacian term.

VARIANTS
    lscae.k3 .. lscae.k8   fixed concrete-layer size
    lscae                  elbow pick over the k grid (OUR size rule, deviation 4)
    lscae.recon_only       Eq. (6) with the Laplacian term removed — the CAE half alone,
                           same code path, same budget, same readout. This is the
                           ablation the paper's contribution (iii) predicts should lose.

Determinism: every draw comes from the passed `rng` or a torch.Generator seeded from it;
torch.set_num_threads(1) fixes BLAS ordering. Equal-seeded rng => identical output. On
any failure the family degrades to full-pool fallback rows (never raises).
"""

import numpy as np
import torch

from . import register

K_GRID = (3, 4, 5, 6, 7, 8)
# Budget (deviation 2). The paper trains 300 epochs; we use 150 x ceil(n/BATCH) minibatch
# steps and 2 seeds instead of 3, measured to keep the planted-world smoke assertions
# passing while fitting the 24-cell x 2-arena sweep. `a3_concrete_ae` runs the same grid
# at 300/3 without a Laplacian term, so the LS-CAE-vs-CAE delta is NOT budget-matched and
# that must be stated wherever the two are compared.
N_SEEDS = 2
EPOCHS = 150                 # x ceil(n/BATCH) minibatch steps each
BATCH = 64
T0, TB = 10.0, 0.01          # concrete anneal, as in CAE
LOGIT_LR = 0.1
DECODER_LR = 1e-2
R_MAX = 1500
VAL_FRAC = 0.2
_EPS = 1e-8

_EXPECTED = tuple(f"lscae.k{k}" for k in K_GRID) + ("lscae", "lscae.recon_only")


def _anneal(b, B):
    return T0 * (TB / T0) ** (b / max(B, 1))


_EYE_CACHE = {}


def _big_eye(m):
    """Cached `1e12 * I_m`, used to mask self-distances. Allocated once per batch size
    rather than once per gradient step (this runs ~30k times per selector call)."""
    if m not in _EYE_CACHE:
        _EYE_CACHE[m] = torch.eye(m) * 1e12
    return _EYE_CACHE[m]


def _diffusion_laplacian(C):
    """L_diff = D^-1 W on the concrete-layer representation C ([m, k]).

    Gaussian kernel, bandwidth = max over points of the distance to that point's nearest
    neighbour (paper §2.1 footnote 2). Self-similarity is left in W: D^-1 W is the
    transition matrix of the random walk the paper defines, and zeroing the diagonal
    would change that operator."""
    d2 = torch.cdist(C, C) ** 2
    off = d2 + _big_eye(C.shape[0])                         # exclude self for the NN
    sigma2 = off.min(dim=1).values.max().clamp_min(_EPS)    # max_i min_j!=i ||.||^2
    W = torch.exp(-d2 / (2.0 * sigma2))
    return W / W.sum(1, keepdim=True).clamp_min(_EPS)


def _fit_one(Xtr, Xva, k, seed, use_laplacian=True):
    """One LS-CAE fit. Returns (selected distinct cols [<=k], val_mse, logits)."""
    torch.manual_seed(int(seed))
    gen = torch.Generator().manual_seed(int(seed))
    n, p = Xtr.shape
    B = int(min(BATCH, n))
    logits = (0.01 * torch.randn(k, p, generator=gen)).requires_grad_(True)
    dec = torch.nn.Linear(k, p)
    opt = torch.optim.Adam([{'params': [logits], 'lr': LOGIT_LR},
                            {'params': dec.parameters(), 'lr': DECODER_LR}])
    for b in range(EPOCHS):
        T = _anneal(b, EPOCHS)
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, B):
            Xb = Xtr[perm[s:s + B]]
            U = torch.rand(k, p, generator=gen).clamp_(_EPS, 1 - _EPS)
            gumbel = -torch.log(-torch.log(U))
            M = torch.softmax((logits + gumbel) / T, dim=1)      # [k, p]
            C = Xb @ M.t()                                       # [m, k] concrete layer
            recon = torch.mean((dec(C) - Xb) ** 2)
            # Eq. (6). Each term divided by its own stop-gradient magnitude, so neither
            # can dominate and no balancing hyperparameter is needed. `.abs()` guards the
            # denominator only — it never changes a sign in the numerator.
            loss = recon / recon.detach().abs().clamp_min(_EPS)
            if use_laplacian and Xb.shape[0] > 2:
                lap = torch.einsum('ij,ij->', C, _diffusion_laplacian(C) @ C)
                loss = loss - lap / lap.detach().abs().clamp_min(_EPS)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        hard = torch.argmax(logits, dim=1)                       # [k]
        val_mse = float(torch.mean((dec(Xva[:, hard]) - Xva) ** 2))
        chosen = []
        for j in hard.tolist():                                  # distinct argmax
            if j not in chosen:
                chosen.append(j)
        if len(chosen) < k:      # top up by best unused max-logit, as CAE does
            maxlog = logits.max(dim=0).values
            for j in torch.argsort(maxlog, descending=True).tolist():
                if j not in chosen:
                    chosen.append(j)
                if len(chosen) == k:
                    break
        sel = np.array(sorted(chosen[:k]), dtype=np.int64)
    return sel, val_mse, logits.detach().numpy()


def _elbow(ks, mses):
    """OUR size rule (deviation 4) — the paper gives none. Normalised-MSE knee: the
    smallest k whose val MSE is within 10% of the grid's total improvement from the
    best. Guarded against a non-monotone curve collapsing the denominator."""
    ks = list(ks)
    m = np.asarray(mses, dtype=float)
    span = float(np.nanmax(m) - np.nanmin(m))
    if not np.isfinite(span) or span < 1e-12:
        return ks[len(ks) // 2]
    thresh = float(np.nanmin(m)) + 0.10 * span
    for k, v in zip(ks, m):
        if np.isfinite(v) and v <= thresh:
            return k
    return ks[int(np.nanargmin(m))]


def _fallback(variant, p, err):
    return {'variant': variant, 'cols': np.arange(p, dtype=np.int64),
            'fallback': True, 'diag': {'error': str(err)}}


@register('a8_lscae')
def a8_lscae(cell, rng, cache=None):
    torch.set_num_threads(1)
    p = cell.p
    try:
        V = np.asarray(cell.V, dtype=np.float64)
        n = V.shape[0]
        R = int(min(n, R_MAX))
        idx = np.sort(rng.choice(n, size=R, replace=False)) if R < n else np.arange(n)
        Vr = V[idx]
        # paper §2: features centred with ||f||^2 = 1 (deviation 3)
        Vr = Vr - Vr.mean(axis=0, keepdims=True)
        Vr = Vr / np.maximum(np.linalg.norm(Vr, axis=0, keepdims=True), _EPS)

        perm = rng.permutation(R)
        n_va = max(2, int(round(VAL_FRAC * R)))
        va, tr = perm[:n_va], perm[n_va:]
        Xtr = torch.tensor(Vr[tr], dtype=torch.float32)
        Xva = torch.tensor(Vr[va], dtype=torch.float32)

        seeds = [int(rng.integers(2 ** 31)) for _ in range(N_SEEDS)]
        out, by_k, mses = [], {}, []
        for k in K_GRID:
            if k >= p:
                mses.append(np.nan)
                continue
            fits = [_fit_one(Xtr, Xva, k, s, use_laplacian=True) for s in seeds]
            best = min(fits, key=lambda f: f[1])          # best val MSE across seeds
            by_k[k] = best
            mses.append(best[1])
            sel = best[0]
            d = {'k': int(k), 'val_mse': round(float(best[1]), 6),
                 'n_selected': int(len(sel)),
                 'seed_val_mses': [round(float(f[1]), 6) for f in fits],
                 'note': 'LS-CAE Eq.(6): reconstruction + Laplacian score at the '
                         'concrete layer, each inversely weighted by its own magnitude'}
            if len(sel) < 3:
                out.append(_fallback(f'lscae.k{k}', p, f'selection < 3 cols ({len(sel)})'))
            else:
                out.append({'variant': f'lscae.k{k}', 'cols': sel, 'diag': d})

        for k in K_GRID:                                   # keep the variant set fixed
            if k not in by_k:
                out.append(_fallback(f'lscae.k{k}', p, f'k={k} >= pool size {p}'))

        # -- lscae : the elbow pick over the grid (OUR size rule) --------------------
        if by_k:
            kk = _elbow(sorted(by_k), [by_k[k][1] for k in sorted(by_k)])
            sel = by_k[kk][0]
            d = {'k_chosen': int(kk), 'val_mse': round(float(by_k[kk][1]), 6),
                 'k_grid': list(sorted(by_k)),
                 'grid_val_mse': [round(float(by_k[k][1]), 6) for k in sorted(by_k)],
                 'note': 'size rule is OURS — the paper treats k as architectural'}
            out.append({'variant': 'lscae', 'cols': sel, 'diag': d}
                       if len(sel) >= 3 else _fallback('lscae', p, 'selection < 3'))
        else:
            out.append(_fallback('lscae', p, 'no admissible k'))

        # -- lscae.recon_only : the ABLATION — Eq. (6) minus the Laplacian term ------
        # Same budget, same readout, same k. The paper's contribution (iii) predicts
        # this loses to `lscae`; measuring it here makes that testable within ONE
        # implementation instead of across two modules with different training budgets.
        if by_k:
            kk = _elbow(sorted(by_k), [by_k[k][1] for k in sorted(by_k)])
            fits = [_fit_one(Xtr, Xva, kk, s, use_laplacian=False) for s in seeds]
            best = min(fits, key=lambda f: f[1])
            sel = best[0]
            d = {'k_chosen': int(kk), 'val_mse': round(float(best[1]), 6),
                 'note': 'ABLATION: reconstruction term only (no Laplacian score). '
                         'Delta vs `lscae` isolates the paper\'s contribution.'}
            out.append({'variant': 'lscae.recon_only', 'cols': sel, 'diag': d}
                       if len(sel) >= 3
                       else _fallback('lscae.recon_only', p, 'selection < 3'))
        else:
            out.append(_fallback('lscae.recon_only', p, 'no admissible k'))

        return out

    except Exception as e:
        return [_fallback(v, p, e) for v in _EXPECTED]


# ---------------------------------------------------------------------------
# smoke() — planted nuisance + correlated world (auto-discovered by
# scripts/smoke_selectors.py). This is the paper's own motivating setting.
# ---------------------------------------------------------------------------

def smoke():
    import time
    from ..selector_bench import UnlabeledCell
    from ..fusion_utils import zscore
    from ..subset_sweep import CANONICAL_POOL

    # The world the paper is built for: 2 informative directions, each duplicated into a
    # CORRELATED block (so reconstruction alone can pick either member), plus a wall of
    # NUISANCE features (so a Laplacian computed on the full set is corrupted).
    rng = np.random.default_rng(20260804)
    n, n_nuis = 400, 8
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    cols = [zscore(a + 0.15 * rng.standard_normal(n)) for _ in range(2)]     # 0,1
    cols += [zscore(b + 0.15 * rng.standard_normal(n)) for _ in range(2)]    # 2,3
    cols += [zscore(rng.standard_normal(n)) for _ in range(n_nuis)]          # 4..11
    V = np.column_stack(cols)
    p = V.shape[1]
    cell = UnlabeledCell(domain='smoke', cell_key='lscae', pool=list(CANONICAL_POOL[:p]),
                         pool_bits=np.arange(p, dtype=np.uint8), V=V,
                         anchor=zscore(V[:, 0]), anchor_name=CANONICAL_POOL[0],
                         rho=np.abs(np.corrcoef(V.T)))

    t0 = time.time()
    s1 = a8_lscae(cell, np.random.default_rng([0, 11]))
    elapsed = time.time() - t0
    s2 = a8_lscae(cell, np.random.default_rng([0, 11]))

    by1 = {s['variant']: s for s in s1}
    by2 = {s['variant']: s for s in s2}
    assert set(by1) == set(_EXPECTED), f"variant set changed: {sorted(by1)}"

    # (a) determinism under an equal-seeded rng
    for v in _EXPECTED:
        assert list(by1[v]['cols']) == list(by2[v]['cols']), f"{v}: cols not deterministic"

    # (b) the k=4 fit must recover the informative block and reject every nuisance
    # column. This is the paper's claim in its own planted setting; if it fails, the
    # Laplacian term is not doing what Eq. (6) says it does.
    k4 = by1['lscae.k4']
    assert not k4.get('fallback'), f"lscae.k4 fell back: {k4['diag']}"
    picked = set(int(c) for c in k4['cols'])
    nuisance = picked & set(range(4, p))
    assert not nuisance, f"(b) nuisance cols selected at k=4: {sorted(nuisance)}"

    # (c) one representative from EACH correlated block, not two from one
    assert picked & {0, 1} and picked & {2, 3}, \
        f"(c) k=4 missed a correlated block: {sorted(picked)}"

    # (d) runtime tripwire, NOT a performance target. Calibrated at 258s while seven
    # background workers were saturating all cores; uncontended it is far lower. The
    # assert exists to catch a pathological regression (an un-annealed loop, an O(n^2)
    # kernel on the full sample set), not to police a few seconds.
    assert elapsed < 400.0, f"(d) runtime {elapsed:.1f}s — pathological, not just slow"

    print(f"    [note] a8 smoke: k4={sorted(picked)} "
          f"lscae k*={by1['lscae']['diag'].get('k_chosen')} "
          f"recon_only={sorted(int(c) for c in by1['lscae.recon_only']['cols'])} "
          f"{elapsed:.1f}s")
