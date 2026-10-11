"""
A3 — Concrete Autoencoder feature selection (Step 186+).

Reimplementation of

    Balin, Abid, Zou, "Concrete Autoencoders: Differentiable Feature
    Selection and Reconstruction", ICML 2019 (arXiv:1901.09346).

grounded against `papers/extracted/concrete-autoencoders-differentiable-
feature-selection.md` (digest: papers/digests/<same slug>.md). Canonical repo
github.com/mfbalin/Concrete-Autoencoders is Keras — this is a fresh torch-CPU
implementation with a linear decoder.

ROLE. The tabular-research pick for the pre-fusion FS stage: per cell, the CAE
selects k of p trace features LABEL-FREE (reconstruction MSE only); the SAME
continuous L-SML then fuses the subset. Not gated pass/fail — every variant's
number is reported and the researcher chooses. Known family risk carried
explicitly (project Step-151): reconstruction-good ≠ label-relevant; with
z-scored unit-variance columns a pure-noise feature is only reconstructable by
selecting itself, so all reconstruction value comes from predicting OTHER
features — whether that tracks detection AUROC is what the bench measures.

PAPER MECHANICS USED (with extraction line refs):
  * concrete selector layer: k nodes, logits alpha [k, p]; training sample
    m = softmax((log-ish logits + Gumbel)/T), output X @ m^T; eval = argmax
    per node. (~l.263-330)
  * annealing: T(b) = T0 * (TB/T0)^(b/B), T0=10, TB=0.01 (§3.2, l.335-357;
    defaults l.363-365). Adam; decoder lr = paper's 1e-3-scale default.

DEVIATIONS (explicit):
  1. Linear decoder (p outputs), full-batch training on <=1500 subsampled rows;
     80/20 train/val ROW split (label-free) for the k-selection elbow.
  2. Selector-logit lr = 0.1 >> decoder lr = 1e-2. With p<=46 and a short CPU
     budget, the paper's single lr (1e-3) mode-collapses several selector nodes
     onto one feature (verified on the planted smoke world, 2026-07-17); a
     faster logit lr separates the nodes. Epochs = 300.
  3. Duplicate repair: a node set that collapses below k distinct features is
     topped up by the highest max-logit unused features. Restarts: 3 seeded
     fits per k; the selection is the fit with the LOWEST validation MSE
     (label-free restart selection — majority voting across partially-collapsed
     runs mixed their errors; measured 2026-07-17), cross-seed Jaccard kept as
     the stability diagnostic; final subset always >=3.
  4. Adaptive k (label-free): kneedle-style elbow of the k -> val-MSE curve
     (max perpendicular distance to the chord between the smallest and largest
     k), clamped to the swept grid.
  5. Greedy swap polish: with a LINEAR decoder the eval-time objective is
     exactly "k columns minimizing linear reconstruction MSE", so after the
     concrete training the best-val selection gets a deterministic local-swap
     refinement on the TRAIN split (closed-form lstsq; accept improving swaps,
     <=2 rounds). Same label-free objective, better optimizer — fixes the
     one-feature-off basins observed per-seed (2026-07-17). val_mse in diag is
     the closed-form train->val reconstruction MSE of the final subset.

Determinism: all randomness from the passed `rng` / torch generators seeded
from it; torch.set_num_threads(1). Equal-seeded rng => identical output.
On any failure the family degrades to full-pool fallback rows (never raises).
"""

import numpy as np
import torch

from . import register

K_GRID = (3, 4, 5, 6, 7, 8)
N_SEEDS = 3
EPOCHS = 300               # x ceil(n/BATCH) minibatch steps each
BATCH = 64
T0, TB = 10.0, 0.01          # paper §3.2 defaults (extraction l.363-365)
LOGIT_LR = 0.1               # deviation 2
DECODER_LR = 1e-2            # deviation 2
R_MAX = 1500
VAL_FRAC = 0.2
_EPS = 1e-8


def _anneal(b, B):
    return T0 * (TB / T0) ** (b / max(B, 1))


def _fit_one(Xtr, Xva, k, seed):
    """One CAE fit (minibatch — the concrete layer needs many stochastic
    steps, not epochs; ~EPOCHS*ceil(n/BATCH) gradient steps total).
    Returns (selected distinct cols [k], val_mse, logits)."""
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
            M = torch.softmax((logits + gumbel) / T, dim=1)  # [k, p]
            loss = torch.mean((dec(Xb @ M.t()) - Xb) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        hard = torch.argmax(logits, dim=1)                   # [k] node order
        # val MSE with the hard layer exactly as the decoder consumes it
        Zv = Xva[:, hard]
        val_mse = float(torch.mean((dec(Zv) - Xva) ** 2))
        # selection set: distinct argmax, topped up by best unused max-logits
        chosen = []
        for j in hard.tolist():
            if j not in chosen:
                chosen.append(j)
        if len(chosen) < k:
            maxlog = logits.max(dim=0).values                # [p]
            for j in torch.argsort(maxlog, descending=True).tolist():
                if j not in chosen:
                    chosen.append(j)
                if len(chosen) == k:
                    break
        sel = np.array(sorted(chosen[:k]), dtype=np.int64)
    return sel, val_mse, logits.detach().numpy()


def _lstsq_val_mse(Vtr, Vva, sel):
    """Closed-form linear-reconstruction val MSE of a subset (label-free)."""
    X = Vtr[:, sel]
    B, *_ = np.linalg.lstsq(X, Vtr, rcond=None)
    return float(np.mean((Vva[:, sel] @ B - Vva) ** 2))


def _swap_refine(Vtr, sel, max_rounds=2):
    """Deviation 5: greedy improving swaps on train reconstruction MSE."""
    sel = list(sel)
    p = Vtr.shape[1]

    def train_mse(s):
        X = Vtr[:, s]
        B, *_ = np.linalg.lstsq(X, Vtr, rcond=None)
        return float(np.mean((X @ B - Vtr) ** 2))

    cur = train_mse(sel)
    for _ in range(max_rounds):
        improved = False
        for pos in range(len(sel)):
            best_j, best_v = None, cur
            for j in range(p):
                if j in sel:
                    continue
                cand = sorted(sel[:pos] + [j] + sel[pos + 1:])
                v = train_mse(cand)
                if v < best_v - 1e-9:
                    best_v, best_j = v, j
            if best_j is not None:
                sel[pos] = best_j
                sel = sorted(sel)
                cur = best_v
                improved = True
        if not improved:
            break
    return np.array(sorted(sel), dtype=np.int64), cur


def _cross_seed(fits, k):
    """Features in >=2/3 fits, topped up by summed max-logit to k (dev. 3)."""
    from collections import Counter
    cnt = Counter()
    for sel, _, _ in fits:
        cnt.update(sel.tolist())
    majority = sorted(j for j, c in cnt.items() if c >= 2)
    if len(majority) < k:
        score = np.sum([lg.max(axis=0) for _, _, lg in fits], axis=0)
        for j in np.argsort(score)[::-1]:
            if int(j) not in majority:
                majority.append(int(j))
            if len(majority) == k:
                break
    sel = np.array(sorted(majority[:k]), dtype=np.int64)
    if len(sel) < 3:
        sel = np.array(sorted(set(sel.tolist())
                              | set(range(3 - len(sel)))), dtype=np.int64)
    return sel


def _jaccard_sets(sets):
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = set(sets[i].tolist()), set(sets[j].tolist())
            vals.append(len(a & b) / max(len(a | b), 1))
    return float(np.mean(vals)) if vals else 1.0


def _elbow(ks, mses):
    """Max perpendicular distance to the chord (kneedle-style), dev. 4."""
    ks = np.asarray(ks, float)
    ms = np.asarray(mses, float)
    if len(ks) < 3:
        return int(ks[0])
    x = (ks - ks[0]) / max(ks[-1] - ks[0], _EPS)
    y = (ms - ms[-1]) / max(ms[0] - ms[-1], _EPS)     # 1 at k_min .. 0 at k_max
    d = np.abs(x + y - 1.0) / np.sqrt(2.0)
    return int(ks[int(np.argmax(d))])


def _fallback(variant, p, err):
    return {'variant': variant, 'cols': np.arange(p, dtype=np.int64),
            'fallback': True, 'diag': {'error': str(err)}}


@register('a3_concrete_ae')
def a3_concrete_ae(cell, rng, cache=None):
    torch.set_num_threads(1)
    p = cell.p
    ks = [k for k in K_GRID if k < p]
    try:
        V = np.asarray(cell.V, dtype=np.float32)
        n = V.shape[0]
        if n > R_MAX:
            V = V[np.sort(rng.choice(n, size=R_MAX, replace=False))]
        n = V.shape[0]
        perm = rng.permutation(n)
        n_val = max(int(n * VAL_FRAC), 10)
        Xva = torch.tensor(V[perm[:n_val]])
        Xtr = torch.tensor(V[perm[n_val:]])

        out, curve = [], {}
        for k in ks:
            seeds = [int(rng.integers(2 ** 31)) for _ in range(N_SEEDS)]
            fits = [_fit_one(Xtr, Xva, k, s) for s in seeds]
            best_i = int(np.argmin([f[1] for f in fits]))
            sel = fits[best_i][0]                    # best-val restart (label-free)
            Vtr_np, Vva_np = Xtr.numpy(), Xva.numpy()
            sel, _ = _swap_refine(Vtr_np, sel)       # deviation 5 polish
            val = _lstsq_val_mse(Vtr_np, Vva_np, sel)
            stab = _jaccard_sets([f[0] for f in fits])
            curve[k] = val
            out.append({'variant': f'a3.cae_k{k}', 'cols': sel,
                        'diag': {'k': k, 'val_mse': round(val, 4),
                                 'stability_jaccard': round(stab, 3)}})
        k_star = _elbow(list(curve), [curve[k] for k in curve])
        best = next(o for o in out if o['diag']['k'] == k_star)
        out.append({'variant': 'a3.cae', 'cols': best['cols'],
                    'diag': {**best['diag'], 'adaptive': True,
                             'curve': {int(k): round(v, 4)
                                       for k, v in curve.items()}}})
        return out
    except Exception as e:
        return [_fallback(f'a3.cae_k{k}', p, e) for k in ks] + \
               [_fallback('a3.cae', p, e)]


# ---------------------------------------------------------------------------
# smoke() — known-answer world where reconstruction and relevance coincide by
# construction: 5 independent FACTOR columns (0..4) generate 11 noisy MIXTURE
# columns; the factors span everything at lower noise, so the MSE-optimal k=5
# subset IS the factor set. (A world of mutually-correlated informative
# features + pure noise would NOT have this property — with unit-variance
# columns, pure noise is worth selecting for its own reconstruction; measured
# 2026-07-17 while designing this test.)
# ---------------------------------------------------------------------------

def smoke():
    import time
    from ..fusion_utils import zscore
    from ..selector_bench import UnlabeledCell
    from ..subset_sweep import CANONICAL_POOL

    rng_np = np.random.default_rng(20260716)
    n, p, kf = 400, 16, 5
    F = rng_np.standard_normal((n, kf))                     # independent factors
    cols = [zscore(F[:, j]) for j in range(kf)]             # cols 0..4 = factors
    for _ in range(p - kf):                                 # 11 noisy mixtures
        w = rng_np.standard_normal(kf)
        w /= np.linalg.norm(w)
        cols.append(zscore(F @ w + 0.6 * rng_np.standard_normal(n)))
    V = np.column_stack(cols)
    cell = UnlabeledCell(domain='smoke', cell_key='cae',
                         pool=list(CANONICAL_POOL[:p]),
                         pool_bits=np.arange(p, dtype=np.uint8), V=V,
                         anchor=zscore(V[:, 0]), anchor_name='a',
                         rho=np.abs(np.corrcoef(V.T)))

    t0 = time.time()
    sels1 = a3_concrete_ae(cell, np.random.default_rng([0, 5]))
    elapsed = time.time() - t0
    sels2 = a3_concrete_ae(cell, np.random.default_rng([0, 5]))

    by1 = {s['variant']: s for s in sels1}
    by2 = {s['variant']: s for s in sels2}
    assert set(by1) == {f'a3.cae_k{k}' for k in K_GRID} | {'a3.cae'}, sorted(by1)
    for s in sels1:
        assert not s.get('fallback', False), f"{s['variant']} fell back: {s['diag']}"
    # (c) determinism
    for v in by1:
        assert list(by1[v]['cols']) == list(by2[v]['cols']), f"{v} nondeterministic"
    # (a) k=5 recovers >=4/5 factor columns
    k5 = set(int(c) for c in by1['a3.cae_k5']['cols'])
    hit = len(k5 & set(range(kf)))
    assert hit >= 4, f"(a) k=5 recovered {hit}/5 factors (cols {sorted(k5)})"
    # (b) adaptive elbow lands in {4,5,6}
    k_star = by1['a3.cae']['diag']['k']
    assert k_star in (4, 5, 6), \
        f"(b) elbow k={k_star}, curve {by1['a3.cae']['diag']['curve']}"
    # (d) runtime
    assert elapsed < 45.0, f"(d) runtime {elapsed:.1f}s over budget"
    print(f"    [note] a3 smoke: k=5 hits {hit}/5 factors, elbow k*={k_star}, "
          f"curve={by1['a3.cae']['diag']['curve']}, {elapsed:.1f}s")
