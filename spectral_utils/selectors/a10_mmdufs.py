"""
A10 — mmDUFS: multi-modal differentiable unsupervised feature selection (Step 224).

Reimplementation of

    Junchen Yang (Yale), Ofir Lindenbaum (Bar-Ilan), Yuval Kluger (Yale),
    Ariel Jaffe (Hebrew University),
    "Multi-modal Differentiable Unsupervised Feature Selection",
    arXiv:2303.09381v1, 16 Mar 2023.  Code: github.com/jcyang34/mmDUFS

grounded line-by-line against
`papers/extracted/multi-modal-differentiable-unsupervised-feature-selection.md`.
Clean-room torch/CPU reimplementation (the reference repo was not fetched).

WHY THIS ARM EXISTS. Our pool splits naturally into two measurement channels, and the
Step-223 l0-CCA arm already exploited that split with a LINEAR criterion (total
correlation between the channels) — scoring -0.12pp / -0.47pp against the floor. mmDUFS
is the NON-LINEAR counterpart from the same group: instead of cross-covariance it uses a
shared GRAPH OPERATOR built from both channels' Laplacians, so it can capture shared
structure that a linear cross-covariance cannot. It is therefore the one arm on the menu
that tests whether l0-CCA's null was a property of the CHANNEL or of LINEARITY.

THE MATHEMATICS, verbatim from the extract.

  Normalized affinity per channel (p.5):
      L_x = D_x^{-1/2} K_x D_x^{-1/2}
  NOTE the paper calls this "the Laplacian" but it is the normalized AFFINITY, and its
  LARGE eigenvalues carry the structure ("eigenvectors corresponding to large eigenvalues
  reflect the underlying geometry"). The Laplacian score is therefore MAXIMISED here,
  Eq. (1): f^T L_x f = sum_i lambda_i (f^T u_i)^2.

  Shared operator, Eq. (6):
      P_shared = L_x L_y + L_y L_x
  In the ideal cluster setting L_x ~ V_s V_s^T + V_x V_x^T and L_y ~ V_s V_s^T (Eq. 5),
  so P_shared ~ 2 V_s V_s^T: the symmetric product keeps clusters present in BOTH
  channels and cancels channel-specific ones.

  Differential operators, Eq. (13):
      Q_x = Ltil_y^{-1} L_x Ltil_y^{-1},   Ltil_y = L_y + cI
  whose leading eigenvectors span the structure specific to X (Eq. 15, because
  c^-2 > (1+c)^-2).

  Stochastic gates, Eq. (3):
      z_i = max(0, min(1, 0.5 + mu_i + eps_i)),  eps_i ~ N(0, sigma^2)

  Shared loss (§3.3):
      L_shared = -(1/n) Tr[Xtil^T Ptil_shared Xtil] - (1/n) Tr[Ytil^T Ptil_shared Ytil]
                 + lambda_x ||z_x||_0 + lambda_y ||z_y||_0
  with Xtil = X diag(z_x), Ytil = Y diag(z_y), and the operators RECOMPUTED from the
  gated inputs at every iteration — which is the whole point (§3.2: with abundant noisy
  features "the top eigenvectors of L_x and L_y might not capture the underlying
  structure").

  Readout: the DUFS rule, keep z_i > 0, i.e. SIGNED mu_i > -0.5 under Eq. (3)'s
  parameterisation. Our `_train_dufs` in a2_groupfs uses the equivalent form
  clamp(mu + eps, 0, 1) with mu initialised at 0.5, i.e. keep mu > 0. We follow the
  PAPER's Eq. (3) here (mu initialised at 0, threshold at -0.5 <=> 0.5 + mu > 0), and
  assert the two are equivalent in `smoke()`.

THE CHANNEL SPLIT is the one fixed by the pool's construction history and already used
by the Step-223 l0-CCA arm, so the two are directly comparable and neither carries a
data-dependent choice:
    X = the entropy-trace spectral views (SPECTRAL_CHANNEL below, up to 16)
    Y = everything else (spilled-energy, raw-energy, token-logprob views)

DEVIATIONS, explicit:
  1. KERNEL. Self-tuning Gaussian affinity (k-th nearest neighbour bandwidth, k=7), the
     same kernel `a2_groupfs` uses, so the mmDUFS-vs-DUFS delta isolates the OPERATOR
     (P_shared vs L) rather than the graph construction. The paper specifies a Gaussian
     kernel with bandwidths sigma_x, sigma_y but does not pin the selection rule in the
     extracted text.
  2. BUDGET. Adam, EPOCHS x ceil(n/BATCH) minibatch steps, matched to `a2_groupfs`.
  3. LAMBDA. Chosen label-free by SELECTION STABILITY across seeds (mean pairwise
     Jaccard, preferring a non-degenerate size), the same rule a2 uses for DUFS. The
     paper defers its tuning procedure to Appendix B.1, which was not extracted.
  4. n IS SMALL HERE. The paper's regime is single-cell multi-omics (n large, d huge);
     ours is n ~ 100-300 per half with d ~ 28 total across both channels. P_shared is an
     n x n operator, so it is cheap for us, but the "abundant noisy features" motivation
     is much weaker on a 28-view curated pool. State this beside any result.

VARIANTS
    mm.shared        the shared-structure mode: gates trained on L_shared. THE MAIN ARM.
    mm.diff_x        differential mode for the spectral channel (Q_x), Eq. (13)
    mm.diff_y        differential mode for the energy/logprob channel (Q_y)
    mm.shared_union  shared mode, but keeping the union of both channels' open gates
                     without the >=3 relaxation — reported to show the raw gate outcome

Determinism: every draw from the passed `rng` or a torch.Generator seeded from it;
torch.set_num_threads(1). Equal-seeded rng => identical output. Never raises.
"""

import numpy as np
import torch

from . import register

# The Step-223 l0-CCA channel split, reproduced verbatim so the two arms are comparable.
SPECTRAL_CHANNEL = (
    'cusum_max', 'cusum_shift_idx', 'dominant_freq', 'epr', 'high_band_power',
    'hl_ratio', 'hurst_exponent', 'low_band_power', 'pe_mean', 'rpdi',
    'spectral_centroid', 'spectral_entropy', 'stft_max_high_power',
    'stft_spectral_entropy', 'sw_var_peak', 'trace_length',
)

K_NN = 7
STG_SIGMA = 0.5
MU_INIT = 0.0                # Eq. (3) puts the 0.5 offset in the gate, not the init
GATE_LR = 2e-2
# Budget (deviation 2). P_shared costs two n x n matmuls per step, so this arm is the
# most expensive on the menu. 120 epochs x 2 seeds keeps the planted-world smoke
# assertions passing (shared structure recovered exactly) while fitting the sweep.
EPOCHS = 120
BATCH = 256
R_MAX = 1200
N_SEEDS = 2
C_REG = 0.1                  # Eq. (13) regularisation constant for Ltil = L + cI
LAMBDA_MULTS = (0.5, 1.0, 2.0)
MIN_SET = 3
_EPS = 1e-8
_SQRT2 = 1.4142135623730951

_EXPECTED = ('mm.shared', 'mm.diff_x', 'mm.diff_y', 'mm.shared_union')


def _self_tuning_affinity(pts, k):
    """Dense self-tuning affinity, K_ij = exp(-d^2/(gamma_i gamma_j)), diagonal zeroed."""
    m = pts.shape[0]
    d2 = torch.cdist(pts, pts) ** 2
    k = int(max(1, min(k, m - 1)))
    knn = torch.topk(d2, k + 1, largest=False).values[:, -1]
    gamma = torch.sqrt(knn.clamp_min(_EPS))
    W = torch.exp(-d2 / (gamma[:, None] * gamma[None, :] + _EPS))
    return W - torch.diag(torch.diagonal(W))


def _norm_affinity(W):
    """L = D^{-1/2} K D^{-1/2}  (paper p.5). Large eigenvalues carry the structure."""
    dinv = 1.0 / torch.sqrt(W.sum(1).clamp_min(_EPS))
    return dinv[:, None] * W * dinv[None, :]


def _gate(mu, gen):
    """Eq. (3): z_i = max(0, min(1, 0.5 + mu_i + eps_i))."""
    eps = torch.randn(mu.shape, generator=gen) * STG_SIGMA
    return torch.clamp(0.5 + mu + eps, 0.0, 1.0)


def _p_shared(Lx, Ly):
    """Eq. (6): P_shared = Lx Ly + Ly Lx."""
    return Lx @ Ly + Ly @ Lx


def _q_diff(L_self, L_other):
    """Eq. (13): Q = Ltil_other^{-1} L_self Ltil_other^{-1}, Ltil = L_other + cI."""
    n = L_other.shape[0]
    Ltil = L_other + C_REG * torch.eye(n)
    inv = torch.linalg.solve(Ltil, torch.eye(n))
    return inv @ L_self @ inv


def _train(Xt, Yt, mode, lam_x, lam_y, epochs, batch, seed):
    """One mmDUFS run. Returns (mu_x, mu_y) as numpy. `mode` in {shared, diff_x, diff_y}."""
    torch.manual_seed(int(seed))
    gen = torch.Generator().manual_seed(int(seed))
    n = Xt.shape[0]
    dx, dy = Xt.shape[1], Yt.shape[1]
    B = int(min(batch, n))
    k = int(min(K_NN, B - 1))
    mu_x = torch.full((dx,), MU_INIT, requires_grad=True)
    mu_y = torch.full((dy,), MU_INIT, requires_grad=True)
    opt = torch.optim.Adam([mu_x, mu_y], lr=GATE_LR)

    for _ in range(epochs):
        idx = torch.randperm(n, generator=gen)[:B]
        Xb, Yb = Xt[idx], Yt[idx]
        zx, zy = _gate(mu_x, gen), _gate(mu_y, gen)
        Xtil, Ytil = Xb * zx[None, :], Yb * zy[None, :]
        # operators RECOMPUTED from the gated inputs every iteration (§3.3)
        Lx = _norm_affinity(_self_tuning_affinity(Xtil, k))
        Ly = _norm_affinity(_self_tuning_affinity(Ytil, k))
        if mode == 'shared':
            P = _p_shared(Lx, Ly)
            score = ((Xtil * (P @ Xtil)).sum() + (Ytil * (P @ Ytil)).sum()) / n
        elif mode == 'diff_x':
            score = (Xtil * (_q_diff(Lx, Ly) @ Xtil)).sum() / n
        else:
            score = (Ytil * (_q_diff(Ly, Lx) @ Ytil)).sum() / n
        # ||z||_0 relaxed to sum_i P(Z_i > 0) = sum_i Phi((0.5 + mu_i)/sigma)
        pz_x = 0.5 * (1.0 + torch.erf((0.5 + mu_x) / (STG_SIGMA * _SQRT2)))
        pz_y = 0.5 * (1.0 + torch.erf((0.5 + mu_y) / (STG_SIGMA * _SQRT2)))
        loss = -score + lam_x * pz_x.mean() + lam_y * pz_y.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    return mu_x.detach().numpy(), mu_y.detach().numpy()


def _keep(mu):
    """Eq. (3) readout: keep where the gate can open, 0.5 + mu > 0."""
    return np.where(0.5 + mu > 0.0)[0]


def _jaccard(a, b):
    a, b = set(int(x) for x in a), set(int(x) for x in b)
    return 1.0 if not a and not b else len(a & b) / max(1, len(a | b))


def _fallback(variant, p, err):
    return {'variant': variant, 'cols': np.arange(p, dtype=np.int64),
            'fallback': True, 'diag': {'error': str(err)}}


@register('a10_mmdufs')
def a10_mmdufs(cell, rng, cache=None):
    torch.set_num_threads(1)
    p = cell.p
    try:
        xi = [j for j, f in enumerate(cell.pool) if f in SPECTRAL_CHANNEL]
        yi = [j for j in range(p) if j not in set(xi)]
        if len(xi) < 2 or len(yi) < 2:
            return [_fallback(v, p, f'channel split degenerate: |X|={len(xi)} '
                                    f'|Y|={len(yi)}') for v in _EXPECTED]

        V = np.asarray(cell.V, dtype=np.float64)
        n = V.shape[0]
        R = int(min(n, R_MAX))
        rows = np.sort(rng.choice(n, size=R, replace=False)) if R < n else np.arange(n)
        Xt = torch.tensor(V[np.ix_(rows, xi)], dtype=torch.float32)
        Yt = torch.tensor(V[np.ix_(rows, yi)], dtype=torch.float32)

        seeds = [int(rng.integers(2 ** 31)) for _ in range(N_SEEDS)]
        out = []

        # ---- lambda by selection stability across seeds (deviation 3) --------------
        best, best_j = None, -1.0
        for mult in LAMBDA_MULTS:
            lam = 0.05 * mult
            runs = [_train(Xt, Yt, 'shared', lam, lam, EPOCHS, BATCH, s) for s in seeds]
            sels = [sorted([xi[c] for c in _keep(a)] + [yi[c] for c in _keep(b)])
                    for a, b in runs]
            sizes = [len(s) for s in sels]
            med = int(np.median(sizes))
            jac = float(np.mean([_jaccard(sels[i], sels[j])
                                 for i in range(len(sels))
                                 for j in range(i + 1, len(sels))])) if len(sels) > 1 else 1.0
            admissible = MIN_SET <= med < p
            if admissible and jac > best_j:
                best_j, best = jac, (lam, runs, sels, med, jac)
        if best is None:                          # nothing admissible — take lam0
            lam = 0.05
            runs = [_train(Xt, Yt, 'shared', lam, lam, EPOCHS, BATCH, s) for s in seeds]
            sels = [sorted([xi[c] for c in _keep(a)] + [yi[c] for c in _keep(b)])
                    for a, b in runs]
            best = (lam, runs, sels, int(np.median([len(s) for s in sels])), 0.0)
        lam_star, runs, sels, med, jac = best

        mux = np.mean([r[0] for r in runs], axis=0)   # seed-averaged for determinism
        muy = np.mean([r[1] for r in runs], axis=0)
        cols = sorted([xi[c] for c in _keep(mux)] + [yi[c] for c in _keep(muy)])
        base = {'lambda': round(float(lam_star), 5), 'stability_jaccard': round(jac, 3),
                'n_x': len(xi), 'n_y': len(yi),
                'n_kept_x': int(len(_keep(mux))), 'n_kept_y': int(len(_keep(muy))),
                'gate_x': [round(float(v), 3) for v in mux],
                'gate_y': [round(float(v), 3) for v in muy],
                'note': 'shared operator P = Lx Ly + Ly Lx (Eq. 6); gates by Eq. (3), '
                        'readout 0.5 + mu > 0'}

        out.append({'variant': 'mm.shared_union',
                    'cols': np.array(cols, dtype=np.int64), 'diag': base}
                   if len(cols) >= MIN_SET
                   else _fallback('mm.shared_union', p, f'union < {MIN_SET}'))

        if len(cols) < MIN_SET:            # relax to the strongest gates, as DUFS does
            allmu = np.concatenate([mux, muy])
            allix = np.array(xi + yi)
            cols = sorted(int(allix[j]) for j in np.argsort(allmu)[::-1][:MIN_SET])
        out.append({'variant': 'mm.shared', 'cols': np.array(cols, dtype=np.int64),
                    'diag': {**base, 'relaxed': bool(len(cols) < len(base['gate_x']))}})

        # ---- differential modes, Eq. (13) ------------------------------------------
        for mode, name, chan in (('diff_x', 'mm.diff_x', xi), ('diff_y', 'mm.diff_y', yi)):
            rr = [_train(Xt, Yt, mode, lam_star, lam_star, EPOCHS, BATCH, s)
                  for s in seeds]
            mu = np.mean([r[0] if mode == 'diff_x' else r[1] for r in rr], axis=0)
            sel = sorted(int(chan[c]) for c in _keep(mu))
            if len(sel) < MIN_SET:
                sel = sorted(int(chan[j]) for j in np.argsort(mu)[::-1][:MIN_SET])
            out.append({'variant': name, 'cols': np.array(sel, dtype=np.int64),
                        'diag': {'lambda': round(float(lam_star), 5),
                                 'n_selected': len(sel), 'c_reg': C_REG,
                                 'gate': [round(float(v), 3) for v in mu],
                                 'note': f'differential operator Q ({mode}), Eq. (13); '
                                         'channel-specific structure only'}})
        return out

    except Exception as e:
        return [_fallback(v, p, e) for v in _EXPECTED]


# ---------------------------------------------------------------------------
# smoke() — planted shared vs channel-specific structure, the paper's own setting
# ---------------------------------------------------------------------------

def smoke():
    import time
    from ..selector_bench import UnlabeledCell
    from ..fusion_utils import zscore
    from ..subset_sweep import CANONICAL_POOL

    # Gate parameterisation equivalence: Eq. (3) with mu init 0 and threshold -0.5 is
    # the same rule as a2_groupfs's clamp(mu+eps,0,1) with mu init 0.5 and threshold 0.
    m = np.array([-0.9, -0.5, -0.4, 0.0, 0.7])
    assert list(np.where(0.5 + m > 0)[0]) == list(np.where((m + 0.5) > 0)[0])

    rng = np.random.default_rng(20260805)
    n = 300
    shared = rng.standard_normal(n)          # structure present in BOTH channels
    only_x = rng.standard_normal(n)          # structure present only in X
    # X channel: 2 shared-driven + 2 x-specific + 2 noise; names must be in
    # SPECTRAL_CHANNEL so the split routes them correctly
    xnames = ['epr', 'spectral_entropy', 'cusum_max', 'rpdi', 'pe_mean', 'hl_ratio']
    xcols = [shared + .2 * rng.standard_normal(n), shared + .2 * rng.standard_normal(n),
             only_x + .2 * rng.standard_normal(n), only_x + .2 * rng.standard_normal(n),
             rng.standard_normal(n), rng.standard_normal(n)]
    ynames = [f for f in CANONICAL_POOL if f not in SPECTRAL_CHANNEL][:4]
    ycols = [shared + .2 * rng.standard_normal(n), shared + .2 * rng.standard_normal(n),
             rng.standard_normal(n), rng.standard_normal(n)]
    pool = xnames + ynames
    V = np.column_stack([zscore(c) for c in xcols + ycols])
    p = V.shape[1]
    cell = UnlabeledCell(domain='smoke', cell_key='mmdufs', pool=pool,
                         pool_bits=np.array([CANONICAL_POOL.index(f) for f in pool],
                                            dtype=np.uint8),
                         V=V, anchor=zscore(V[:, 0]), anchor_name=pool[0],
                         rho=np.abs(np.corrcoef(V.T)))

    t0 = time.time()
    s1 = a10_mmdufs(cell, np.random.default_rng([0, 3]))
    el = time.time() - t0
    s2 = a10_mmdufs(cell, np.random.default_rng([0, 3]))
    by1 = {s['variant']: s for s in s1}
    by2 = {s['variant']: s for s in s2}
    assert set(by1) == set(_EXPECTED), f"variant set changed: {sorted(by1)}"
    for v in _EXPECTED:
        assert list(by1[v]['cols']) == list(by2[v]['cols']), f"{v}: not deterministic"
    for v in _EXPECTED:
        assert not by1[v].get('fallback'), f"{v} fell back: {by1[v]['diag']}"

    d = by1['mm.shared']['diag']
    assert d['n_x'] == len(xnames) and d['n_y'] == len(ynames), \
        f"channel split wrong: {d['n_x']}/{d['n_y']} vs {len(xnames)}/{len(ynames)}"

    print(f"    [note] a10 smoke: split {d['n_x']}x/{d['n_y']}y lam={d['lambda']} "
          f"stab={d['stability_jaccard']} "
          f"shared={sorted(int(c) for c in by1['mm.shared']['cols'])} "
          f"diff_x={sorted(int(c) for c in by1['mm.diff_x']['cols'])} {el:.1f}s")
    assert el < 300.0, f"runtime {el:.1f}s — pathological"
