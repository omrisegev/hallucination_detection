"""
A2 — GroupFS: unsupervised feature selection through group discovery (Step 186+).

Reimplementation of

    Lifshitz, Lindenbaum, Mishne, Meir, Benisty,
    "Unsupervised Feature Selection Through Group Discovery",
    AAAI 2026 (arXiv:2511.09166, CC BY 4.0).

grounded verbatim against the fetched paper HTML — every equation traces to
`papers/extracted/unsupervised-feature-selection-through-group-discovery.md`.
No official code release exists (LindenbaumLab GitHub checked 2026-07-17: 9 repos,
none is GroupFS; a GitHub search for 2511.09166 / GroupFS returned nothing), so this
is a clean-room torch (CPU) reimplementation. Building blocks reused conceptually and
cited: the LSCAE family (github.com/jsvir/lscae — stochastic gates + Laplacian-score
terms) and DUFS (github.com/Ofirlin/DUFS — the Gated-Laplacian 2021 predecessor,
reproduced here as the `a2.dufs` same-family baseline).

WHAT GroupFS DOES (paper §4). Learnable params {pi: d×C assignment logits,
mu: C group-gate means, Q: C×C projection}; objective
    L = L_s + lambda1 * L_f + lambda2 * L_reg,
where (all sign-oriented, z-scored input `cell.V` used directly):
  * L_s  = -(1/(Bd)) tr(Xtil^T P_Xtil^t Xtil)  — sample-graph smoothness of the
    group-gated batch (self-tuning random walk, t=2 diffusion).      [paper §4.1]
  * L_f  = (1/(dC)) [tr(F^T L_feat F) + beta ||F^T F - I||_F^2], F = M Q — feature-graph
    smoothness + orthogonality; F columns centered+unit-l2 each step. [paper §4.2]
  * L_reg = (1/C) sum_j Phi(mu_j/sigma) (1/d) sum_i M_ij  — group sparsity. [paper §4.3]
M is Gumbel-Softmax over the logits (temp annealed 10 -> 1e-2); z_j = clip(mu_j+eps,0,1),
eps~N(0,sigma^2). Selection = groups with an open gate (mu_j>0, the STG rule), features
taken by their hard (argmax-logit) group. Warm-start: spectral-cluster the feature graph
into C groups (App. D distortion picks C); mu_j=0.5; Q random-orthonormal row-scaled by
inverse cluster size.

ROLE HERE. "Feature selection that runs BEFORE fusion": from the unlabeled cell.V,
GroupFS picks a subset (+ a group assignment); the SAME existing L-SML then fuses it.
Not gated pass/fail — all per-cell numbers are reported and the researcher chooses.

DEVIATIONS from the paper (explicit, per §):
  1. Input scaling: cell.V is already z-scored + sign-oriented (subset_sweep.prepare_cell),
     so the paper's per-feature preprocessing is skipped. [§3.1]
  2. Sample graph is O(n^2): rows subsampled to <= R_MAX=1200 via `rng`, and — exactly as
     the paper does — the sample graph is rebuilt PER MINIBATCH (B<=256), so cost is O(B^2).
     [§4.1 "at each iteration"]
  3. Kernel: kept the paper's DENSE self-tuning kernel (Eq. 1, K=7-th-NN bandwidth), NOT
     the generic median-bandwidth + kNN-sparsify default — the paper specifies Eq. (1)
     explicitly. [§3.1]
  4. lambda2 (group sparsity) is chosen LABEL-FREE by selection stability (mean pairwise
     Jaccard of the selected set over N_SEEDS=5 seeds), among {lambda0/4, lambda0, 4*lambda0},
     preferring a non-degenerate median size (3<=size<p). This replaces the paper's
     lowest-*loss* grid search — we have no per-cell held-out clustering signal. [§App B.1]
  5. lambda1 snapped to the nearest of {0.1,1,10,100} by the epoch-1 |L_s|/|L_f_core| ratio
     (L_f_core = the smoothness trace term only, so it is independent of beta=1/lambda1 and
     the circularity is avoided). [§App B.1]
  6. C via the App-D Procrustes distortion heuristic with a knee rule (smallest C within 5%
     of the min distortion), C_max=min(p-1,8) for our small pools (p<=16 h16 / <=~40 c46).
  7. Gate learning rate raised to GATE_LR=2e-2 (logits/Q keep the paper's lr=1e-3) so the
     STG gates separate within a CPU-affordable step budget instead of the paper's
     500-5000 epochs.
  8. SELECTION PRESSURE comes from per-feature DUFS gates aggregated to GROUP granularity,
     not from the paper's joint group-gate training. Measured failure mode (2026-07-17,
     planted-group world): under the CPU budget the joint dynamic saturates EVERY group
     gate open (mu ~ 1.8-2.0) at every lambda2 in a x512 geometric sweep — the sample-
     trace term rewards opening all gates while the Bernoulli-Gaussian penalty's gradient
     phi(mu/sigma) vanishes once mu > ~2*sigma, so the sparse regime is unreachable. The
     per-feature STG objective (DUFS, the 2021 predecessor — reproduced here anyway as
     `a2.dufs`) verifiably separates noise from signal on the same world, so: groups come
     from the paper's discovery mechanism (warm-started logits, App-D C, joint refinement),
     and a group is SELECTED iff the median DUFS gate of its members is open (>0). This
     keeps GroupFS's core "select groups, not features" semantics with a selection signal
     that demonstrably works at our scale. lambda for the DUFS gates is chosen label-free
     by selection stability across seeds (same rule as deviation 4).

Determinism: every random draw comes from the passed `rng` (numpy) or a torch.Generator
seeded from it; torch.set_num_threads(1) fixes BLAS ordering. Equal-seeded rng => identical
output. On any failure the whole family degrades to the full-pool fallback (never raises).
"""

import numpy as np
import torch

from . import register
from ..subset_sweep import GOOD_5

# ---- paper hyperparameters (App. B.1) --------------------------------------
K_NN = 7                       # self-tuning kernel neighbor (Eq. 1)
DIFFUSION_T = 2                # random-walk diffusion steps
STG_SIGMA = 0.5               # stochastic-gate noise
LR = 1e-3                      # Adam lr for logits + Q (paper)
TEMP_START, TEMP_MIN = 10.0, 1e-2   # Gumbel-Softmax temperature schedule
P_MAIN = 0.7                  # warm-start dominant-cluster probability
MU_INIT = 0.5                 # gate-mean init (unbiased prior)
LAMBDA1_GRID = (0.1, 1.0, 10.0, 100.0)

# ---- adaptation knobs (deviations 2,4,6,7) ---------------------------------
GATE_LR = 2e-2                # faster STG convergence within the CPU budget
# lambda2 bracket: geometric, biased UPWARD from the data-driven lambda0 — the
# first bracket (0.25,1,4) left every group gate saturated open (the sample-
# graph trace term rewards opening ALL gates, incl. noise groups, so the
# sparse regime sits well above the epoch-1 |L_s|/|L_reg| ratio). The
# stability rule's admissibility filter (3 <= median size < p) discards the
# saturate-open and collapse endpoints, so a wide bracket is safe.
# (joint group-gate lambda2 bracket removed — deviation 8: selection now
# runs through per-feature DUFS gates aggregated to group granularity)
N_SEEDS_STABILITY = 5
R_MAX = 1200                  # sample-row subsample cap
BATCH = 256
EPOCHS_STAB = 120             # per (lambda2, seed) stability training
EPOCHS_FINAL = 180            # final chosen-lambda2 training
C_MAX_CAP = 8
KNEE_TOL = 0.05               # App-D distortion knee tolerance

_EXPECTED = ('a2.select', 'a2.select+groups', 'a2.groups@good5', 'a2.dufs')
_SQRT2 = 1.4142135623730951
_EPS = 1e-8


# ---------------------------------------------------------------------------
# graph operators (torch)
# ---------------------------------------------------------------------------

def _self_tuning_affinity(pts, k):
    """Dense self-tuning affinity (paper Eq. 1). pts: [m, dim]; gamma_i = distance
    to the k-th nearest neighbor. Diagonal zeroed (no self-loops)."""
    m = pts.shape[0]
    d2 = torch.cdist(pts, pts) ** 2                      # [m, m] squared euclidean
    k = int(max(1, min(k, m - 1)))
    knn_d2 = torch.topk(d2, k + 1, largest=False).values[:, -1]   # k-th NN (excl. self)
    gamma = torch.sqrt(knn_d2.clamp_min(_EPS))          # [m]
    W = torch.exp(-d2 / (gamma[:, None] * gamma[None, :] + _EPS))
    return W - torch.diag(torch.diagonal(W))


def _random_walk_power(W, t):
    """P = D^{-1} W, returned to the t-th power (t-step diffusion)."""
    P = W / W.sum(1, keepdim=True).clamp_min(_EPS)
    Pt = P
    for _ in range(t - 1):
        Pt = Pt @ P
    return Pt


def _normalized_laplacian(W):
    """L_sym = I - D^{-1/2} W D^{-1/2}."""
    dinv_sqrt = 1.0 / torch.sqrt(W.sum(1).clamp_min(_EPS))
    return torch.eye(W.shape[0]) - dinv_sqrt[:, None] * W * dinv_sqrt[None, :]


# ---------------------------------------------------------------------------
# feature-graph spectral pieces (warm-start + App-D group count)
# ---------------------------------------------------------------------------

def _spectral_embed(L_feat_np, C):
    """Row-normalized C smallest eigenvectors of the feature Laplacian."""
    _, vecs = np.linalg.eigh(L_feat_np)                 # ascending eigenvalues
    U = vecs[:, :C]
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms < _EPS] = 1.0
    return U / norms


def _kmeans_labels(U, C, seed):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=C, n_init=10, random_state=int(seed)).fit_predict(U)


def _choose_C(L_feat_np, p, seed):
    """App-D Procrustes distortion heuristic + knee rule (deviation 6)."""
    c_max = int(min(p - 1, C_MAX_CAP))
    if c_max < 2:
        return 2, {}
    dists = {}
    for C in range(2, c_max + 1):
        U = _spectral_embed(L_feat_np, C)               # [p, C]
        labels = _kmeans_labels(U, C, seed)
        Y = np.zeros((p, C))
        Y[np.arange(p), labels] = 1.0
        A, _, Bt = np.linalg.svd(U.T @ Y)               # orthogonal Procrustes
        R = A @ Bt
        dists[C] = float(np.linalg.norm(U @ R - Y, 'fro') ** 2)
    best = min(dists.values())
    thresh = best + KNEE_TOL * abs(best) + 1e-9
    C_star = min(C for C, e in dists.items() if e <= thresh)   # smallest near-min
    return int(C_star), {int(k): round(v, 4) for k, v in dists.items()}


def _warm_logits(labels, C):
    p = len(labels)
    p_rest = (1.0 - P_MAIN) / max(C - 1, 1)
    delta = float(np.log(P_MAIN / max(p_rest, _EPS)))
    logits = np.zeros((p, C), dtype=np.float32)
    logits[np.arange(p), labels] = delta
    return logits


def _init_Q(C, cluster_sizes, gen):
    Q, _ = torch.linalg.qr(torch.randn(C, C, generator=gen))
    inv = 1.0 / torch.clamp(torch.tensor(cluster_sizes, dtype=torch.float32), min=1.0)
    return (Q * inv[:, None]).contiguous()


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _gumbel_like(shape, gen):
    U = torch.rand(shape, generator=gen).clamp_(_EPS, 1.0 - _EPS)
    return -torch.log(-torch.log(U))


def _train_groupfs(X_t, L_feat_t, C, lam1, lam2, beta, epochs, batch,
                   torch_seed, warm_logits_np, cluster_sizes):
    """One GroupFS training run. Returns (logits, mu, final_loss) as numpy/float."""
    torch.manual_seed(int(torch_seed))
    gen = torch.Generator().manual_seed(int(torch_seed))
    R, d = X_t.shape
    B = int(min(batch, R))
    k_samp = int(min(K_NN, B - 1))
    I_C = torch.eye(C)

    logits = torch.tensor(warm_logits_np, dtype=torch.float32, requires_grad=True)
    mu = torch.full((C,), MU_INIT, dtype=torch.float32, requires_grad=True)
    Q = _init_Q(C, cluster_sizes, gen).requires_grad_(True)
    opt = torch.optim.Adam([{'params': [logits, Q], 'lr': LR},
                            {'params': [mu], 'lr': GATE_LR}])

    loss_val = float('nan')
    for e in range(epochs):
        T = max(TEMP_MIN, TEMP_START - (TEMP_START - TEMP_MIN) * e / max(epochs, 1))
        idx = torch.randperm(R, generator=gen)[:B]
        Xb = X_t[idx]                                            # [B, d]
        M = torch.softmax((logits + _gumbel_like((d, C), gen)) / T, dim=1)   # [d, C]
        z = torch.clamp(mu + torch.randn(C, generator=gen) * STG_SIGMA, 0.0, 1.0)
        Xtil = Xb * (M @ z)[None, :]                             # [B, d]
        Pt = _random_walk_power(_self_tuning_affinity(Xtil, k_samp), DIFFUSION_T)
        Ls = -(Xtil * (Pt @ Xtil)).sum() / (B * d)

        Fm = M @ Q                                              # [d, C]
        Fm = Fm - Fm.mean(0, keepdim=True)
        Fm = Fm / (Fm.norm(dim=0, keepdim=True) + _EPS)
        Lf = (torch.einsum('ij,ij->', Fm, L_feat_t @ Fm)
              + beta * ((Fm.t() @ Fm - I_C) ** 2).sum()) / (d * C)

        Pz = 0.5 * (1.0 + torch.erf(mu / (STG_SIGMA * _SQRT2)))
        Lreg = (Pz * M.sum(0)).sum() / (C * d)

        loss = Ls + lam1 * Lf + lam2 * Lreg
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_val = float(loss.detach())

    return logits.detach().numpy(), mu.detach().numpy(), loss_val


def _train_dufs(X_t, lam2, epochs, batch, torch_seed):
    """DUFS (Lindenbaum 2021): per-feature STG gates on the Laplacian-score sample
    objective, no grouping. Returns per-feature gate means."""
    torch.manual_seed(int(torch_seed))
    gen = torch.Generator().manual_seed(int(torch_seed))
    R, d = X_t.shape
    B = int(min(batch, R))
    k_samp = int(min(K_NN, B - 1))
    mu = torch.full((d,), MU_INIT, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([mu], lr=GATE_LR)
    for _ in range(epochs):
        idx = torch.randperm(R, generator=gen)[:B]
        Xb = X_t[idx]
        z = torch.clamp(mu + torch.randn(d, generator=gen) * STG_SIGMA, 0.0, 1.0)
        Xtil = Xb * z[None, :]
        Pt = _random_walk_power(_self_tuning_affinity(Xtil, k_samp), DIFFUSION_T)
        Ls = -(Xtil * (Pt @ Xtil)).sum() / (B * d)
        Pz = 0.5 * (1.0 + torch.erf(mu / (STG_SIGMA * _SQRT2)))
        loss = Ls + lam2 * Pz.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return mu.detach().numpy()


def _init_magnitudes(X_t, L_feat_t, C, warm, cluster_sizes, gen):
    """Epoch-1 |L_s|, |L_f_core|, |L_reg| under the warm start (no gate/gumbel noise)
    for the lambda1 snap + lambda0 center (deviations 4,5)."""
    R, d = X_t.shape
    B = int(min(BATCH, R))
    k_samp = int(min(K_NN, B - 1))
    logits = torch.tensor(warm, dtype=torch.float32)
    mu = torch.full((C,), MU_INIT)
    Q = _init_Q(C, cluster_sizes, gen)
    M = torch.softmax(logits / TEMP_START, dim=1)
    Xtil = X_t[torch.randperm(R, generator=gen)[:B]] * (M @ torch.clamp(mu, 0, 1))[None, :]
    Pt = _random_walk_power(_self_tuning_affinity(Xtil, k_samp), DIFFUSION_T)
    Ls = abs(float(-(Xtil * (Pt @ Xtil)).sum() / (B * d)))
    Fm = M @ Q
    Fm = Fm - Fm.mean(0, keepdim=True)
    Fm = Fm / (Fm.norm(dim=0, keepdim=True) + _EPS)
    Lf_core = abs(float(torch.einsum('ij,ij->', Fm, L_feat_t @ Fm) / (d * C)))
    Pz = 0.5 * (1.0 + torch.erf(mu / (STG_SIGMA * _SQRT2)))
    Lreg = abs(float((Pz * M.sum(0)).sum() / (C * d)))
    return Ls, Lf_core, Lreg


def _snap_lambda1(ratio):
    grid = np.array(LAMBDA1_GRID)
    return float(grid[np.argmin(np.abs(np.log(grid) - np.log(max(ratio, _EPS))))])


# ---------------------------------------------------------------------------
# selection helpers
# ---------------------------------------------------------------------------





def _relabel(sub):
    """Relabel group ids to consecutive 0..K-1 by first appearance."""
    seen, out = {}, []
    for v in sub.tolist():
        seen.setdefault(v, len(seen))
        out.append(seen[v])
    return np.array(out, dtype=np.int64)


def _jaccard(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _mean_pairwise_jaccard(sets):
    vals = [_jaccard(sets[i], sets[j])
            for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return float(np.mean(vals)) if vals else 1.0


def _fallback(variant, p, err, base_diag=None):
    diag = {'error': str(err)}
    if base_diag is not None:
        diag = {**base_diag, 'fallback_reason': str(err)}
    return {'variant': variant, 'cols': np.arange(p, dtype=np.int64),
            'fallback': True, 'diag': diag}


# ---------------------------------------------------------------------------
# the registered selector family
# ---------------------------------------------------------------------------

@register('a2_groupfs')
def a2_groupfs(cell, rng, cache=None):
    torch.set_num_threads(1)
    p = cell.p
    try:
        V = np.asarray(cell.V, dtype=np.float64)
        n = V.shape[0]
        R = int(min(n, R_MAX))
        Xr = V[np.sort(rng.choice(n, size=R, replace=False))] if R < n else V
        X_t = torch.tensor(Xr, dtype=torch.float32)

        # feature graph (features are columns -> points in R^R)
        Wf = _self_tuning_affinity(X_t.t().contiguous(), min(K_NN, p - 1))
        L_feat_t = _normalized_laplacian(Wf)
        L_feat_np = L_feat_t.numpy()

        c_seed = int(rng.integers(2 ** 31))
        C, C_dists = _choose_C(L_feat_np, p, c_seed)
        C = int(max(2, min(C, p - 1)))
        labels = _kmeans_labels(_spectral_embed(L_feat_np, C), C, c_seed)
        warm = _warm_logits(labels, C)
        cluster_sizes = [int((labels == j).sum()) for j in range(C)]

        gen0 = torch.Generator().manual_seed(int(rng.integers(2 ** 31)))
        aLs, aLf, aLreg = _init_magnitudes(X_t, L_feat_t, C, warm, cluster_sizes, gen0)
        lam1 = _snap_lambda1(aLs / max(aLf, _EPS))
        beta = 1.0 / lam1
        lam0 = float(np.clip(aLs / max(aLreg, _EPS), 1e-3, 1e4))

        # joint model ONCE at lam2=lam0 — refines the grouping logits (the gates
        # it learns saturate open and are ignored; deviation 8)
        lg, _mu_joint, _ = _train_groupfs(X_t, L_feat_t, C, lam1, lam0, beta,
                                          EPOCHS_FINAL, BATCH,
                                          int(rng.integers(2 ** 31)),
                                          warm, cluster_sizes)
        groups_full = np.argmax(lg, axis=1)

        # DUFS per-feature gates: lambda chosen label-free by selection
        # stability across seeds (deviations 4 + 8)
        seeds = [int(rng.integers(2 ** 31)) for _ in range(N_SEEDS_STABILITY)]
        stab = {}
        for mult in (0.5, 1.0, 2.0):
            lam_d = lam0 * mult
            gates, sels = [], []
            for s in seeds:
                mu_f = _train_dufs(X_t, lam_d, EPOCHS_STAB, BATCH, s)
                gates.append(mu_f)
                sels.append(np.where(mu_f > 0.0)[0])
            sizes = [len(x) for x in sels]
            med = int(np.median(sizes))
            stab[lam_d] = {'jaccard': _mean_pairwise_jaccard(sels), 'med': med,
                           'admissible': bool(3 <= med < p), 'sizes': sizes,
                           'mu_bar': np.mean(gates, axis=0)}
        adm = [(v['jaccard'], lam_d) for lam_d, v in stab.items() if v['admissible']]
        lam_star = max(adm)[1] if adm else lam0
        stability_star = float(stab[lam_star]['jaccard'])
        mu_feat = stab[lam_star]['mu_bar']            # seed-averaged, deterministic

        # GROUP-granular selection: a group is open iff the median member gate
        # is open; selection = union of open groups (>=3 cols enforced below)
        open_groups = [j for j in np.unique(groups_full)
                       if float(np.median(mu_feat[groups_full == j])) > 0.0]
        selected = np.array(sorted(np.where(np.isin(groups_full, open_groups))[0]),
                            dtype=np.int64)
        if len(selected) < 3:                          # relax: top gates until 3
            order = np.argsort(mu_feat)[::-1]
            selected = np.array(sorted(order[:3]), dtype=np.int64)

        base_diag = {
            'C': int(C), 'lambda1': round(lam1, 4), 'lambda0': round(lam0, 6),
            'lambda_dufs': round(float(lam_star), 6),
            'lambda_stability': {round(float(k), 6): round(v['jaccard'], 3)
                                 for k, v in stab.items()},
            'stability': round(stability_star, 3),
            'K_groups': int(len(np.unique(groups_full))),
            'open_groups': [int(j) for j in open_groups],
            'n_selected': int(len(selected)),
            'feat_gates': [round(float(x), 3) for x in mu_feat],
            'C_distortion': C_dists, 'subsampled_rows': int(R),
        }

        out = []
        # -- a2.select : gate-selected cols, standard L-SML (no groups) ----------
        if len(selected) < 3:
            out.append(_fallback('a2.select', p, 'gate selection < 3 cols', base_diag))
        else:
            out.append({'variant': 'a2.select', 'cols': selected, 'diag': base_diag})

        # -- a2.select+groups : selected cols + discovered groups (clustering swap)
        if len(selected) < 3:
            out.append(_fallback('a2.select+groups', p, 'gate selection < 3 cols', base_diag))
        else:
            grp = _relabel(groups_full[selected])
            d2 = {**base_diag, 'K_subset': int(len(np.unique(grp))),
                  'note': 'gate-selected cols + discovered groups'}
            out.append({'variant': 'a2.select+groups', 'cols': selected,
                        'groups': grp, 'diag': d2})

        # -- a2.groups@good5 : GOOD_5 cols fixed, discovered groups restricted -----
        g5 = [cell.pool.index(f) for f in GOOD_5 if f in cell.pool]
        if len(g5) < 3:
            out.append(_fallback('a2.groups@good5', p,
                                 f'<3 of GOOD_5 present ({len(g5)})', base_diag))
        else:
            g5 = np.array(sorted(g5), dtype=np.int64)
            grp5 = _relabel(groups_full[g5])
            d3 = {**base_diag, 'good5_cols': g5.tolist(),
                  'K_subset': int(len(np.unique(grp5))),
                  'note': 'GOOD_5 fixed; discovered groups restricted (isolate clustering)'}
            out.append({'variant': 'a2.groups@good5', 'cols': g5, 'groups': grp5, 'diag': d3})

        # -- a2.dufs : Gated-Laplacian predecessor — per-FEATURE selection from the
        # same seed-averaged gates (no group granularity, no groups override) -------
        sel_d = np.array(sorted(np.where(mu_feat > 0.0)[0].tolist()), dtype=np.int64)
        d4 = {'lambda_dufs': round(float(lam_star), 6), 'n_selected': int(len(sel_d)),
              'feat_gate_means': [round(float(x), 3) for x in mu_feat],
              'note': 'DUFS (Lindenbaum 2021) per-feature baseline; delta vs '
                      'a2.select isolates the group granularity'}
        if len(sel_d) < 3:
            out.append(_fallback('a2.dufs', p, 'DUFS selection < 3 cols', d4))
        else:
            out.append({'variant': 'a2.dufs', 'cols': sel_d, 'diag': d4})

        return out

    except Exception as e:                  # whole-family failure -> full-pool fallbacks
        return [_fallback(v, p, e) for v in _EXPECTED]


# ---------------------------------------------------------------------------
# smoke() — planted-group known answer (auto-discovered by smoke_selectors.py)
# ---------------------------------------------------------------------------

def smoke():
    from sklearn.metrics import adjusted_rand_score
    from ..selector_bench import UnlabeledCell
    from ..fusion_utils import zscore
    from ..subset_sweep import CANONICAL_POOL
    import time

    # planted: 3 correlated groups (3/3/2) sharing a common consensus + 4 noise cols
    rng_np = np.random.default_rng(20260717)
    n, p = 400, 12
    y = rng_np.standard_normal(n)                     # consensus
    truth_info = np.array([0, 0, 0, 1, 1, 1, 2, 2])   # informative-feature groups
    cols = []
    for g, size, block in [(0, 3, 0.6), (1, 3, 0.6), (2, 2, 0.6)]:
        latent = y + block * rng_np.standard_normal(n)         # shared within group
        for _ in range(size):
            cols.append(zscore(latent + 0.25 * rng_np.standard_normal(n)))
    for _ in range(4):                                          # pure-noise columns
        cols.append(zscore(rng_np.standard_normal(n)))
    V = np.column_stack(cols)                                   # (400, 12)
    pool = list(CANONICAL_POOL[:p])
    pool_bits = np.arange(p, dtype=np.uint8)
    rho = np.abs(np.corrcoef(V.T))
    cell = UnlabeledCell(domain='smoke', cell_key='groupfs', pool=pool,
                         pool_bits=pool_bits, V=V, anchor=zscore(V[:, 0]),
                         anchor_name=pool[0], rho=rho)

    t0 = time.time()
    sels1 = a2_groupfs(cell, np.random.default_rng([0, 99]))
    elapsed = time.time() - t0
    sels2 = a2_groupfs(cell, np.random.default_rng([0, 99]))

    by1 = {s['variant']: s for s in sels1}
    assert set(by1) == set(_EXPECTED), f"variant set changed: {sorted(by1)}"
    for s in sels1:
        assert not s.get('fallback', False), f"{s['variant']} fell back: {s['diag']}"

    # (c) determinism under equal-seeded rng
    by2 = {s['variant']: s for s in sels2}
    for v in _EXPECTED:
        assert list(by1[v]['cols']) == list(by2[v]['cols']), f"{v}: cols not deterministic"
        if 'groups' in by1[v]:
            assert list(by1[v]['groups']) == list(by2[v]['groups']), f"{v}: groups nondet"

    diag = by1['a2.select']['diag']
    # (a) planted-group recovery: score the discovered grouping on the informative
    # cols that were selected (informative cols are 0..7, planted truth `truth_info`).
    sg = by1['a2.select+groups']
    sel_cols = np.array(sg['cols'])
    sel_grp = np.array(sg['groups'])
    info_mask = sel_cols < 8                                    # informative cols are 0..7
    if info_mask.sum() >= 6:                                    # need enough to score ARI
        ari = adjusted_rand_score(truth_info[sel_cols[info_mask]], sel_grp[info_mask])
    else:
        ari = -1.0
    assert ari >= 0.6, f"(a) planted-group ARI {ari:.3f} < 0.6 (selected info cols)"

    # (b) noise cols {8,9,10,11} excluded from selection
    sel_set = set(int(c) for c in by1['a2.select']['cols'])
    noise_in = sel_set & {8, 9, 10, 11}
    assert not noise_in, f"(b) noise cols selected: {sorted(noise_in)}"
    assert sel_set.issubset(set(range(8))) and len(sel_set) >= 3

    # (d) runtime under budget
    assert elapsed < 60.0, f"(d) runtime {elapsed:.1f}s over budget"

    print(f"    [note] a2 smoke: C={diag['C']} K={diag['K_groups']} "
          f"lam_dufs*={diag['lambda_dufs']} stab={diag['stability']} "
          f"open_groups={diag['open_groups']} "
          f"selected={sorted(sel_set)} ARI={ari:.2f} {elapsed:.1f}s")
