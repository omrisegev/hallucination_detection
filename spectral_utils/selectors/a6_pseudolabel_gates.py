"""
a6_pseudolabel_gates — DUFS gates supervised by an L-SML pseudo-label (Step 194).

Motivation (Step 193d, measured):
    The Gated-Laplacian selector `a2.dufs` optimises a Laplacian-smoothness
    objective over the SAMPLE graph. That objective never sees a label, and
    empirically it is nearly orthogonal to what we actually want:

        rho(gate value, that view's own oriented AUROC) = +0.149
        (mean over the 25 in-scope cells; per-cell range -0.085 ... +0.342)

    So the selector is not picking *badly*, it is picking for a DIFFERENT
    criterion. Pool size (16<->30 within 0.11pp) and anchor choice (`epr`
    resolves the sign on 25/25 cells) have both already been ruled out as
    levers. What has not been tried is giving the gates a separability signal
    that is still label-free.

Idea (Omri, 2026-07-22):
    Fuse K strong views with their own L-SML into a continuous score, then use
    that score as a PSEUDO-LABEL to supervise the gates -- not merely to fix the
    sign (the anchor sweep showed sign has no headroom left). The DUFS loss

        L = L_smooth + lambda2 * E[Pz]

    gains an agreement reward, on the CENTERED agreement
    a_f = |corr(X_f, y_hat)| - mean_f |corr(X_f, y_hat)|:

        L = L_smooth + lambda2 * E[Pz] - lambda3 * E[Pz * a_f]

    Mechanically the new term TILTS the uniform sparsity prior per feature: it
    is constant w.r.t. mu, so the gradient flows only through Pz, making the
    effective per-feature penalty (lambda2 - lambda3 * a_f). Centering is what
    makes this a redistribution rather than a relaxation -- sum_f a_f = 0, so
    the total sparsity budget is unchanged and views that agree with the fused
    consensus more than average get cheaper strictly at the expense of views
    that agree less. An UNcentered term would simply open every gate wider and
    the comparison against the unsupervised control would be meaningless
    (verified: it kept 4/6 planted noise columns vs the control's 0).

    The sparsity level lambda2 itself is chosen label-free by exactly the same
    cross-seed-stability rule the control uses, BEFORE lambda3 is introduced, so
    the supervised and unsupervised arms get identical sparsity freedom.

Circularity guard:
    The seed views that BUILD the pseudo-label are removed from the selectable
    pool -- the gates can never select the views that supervise them. They are
    still allowed in the final scored subset (`a6.pl_dufs`), because that is a
    fixed a-priori prior, not label information; `a6.pl_dufs_noseed` isolates
    how much of any gain is just "the seed views are good views".

Prior art in this repo:
    `fusion_utils.best_nadler_pseudo_label` (Step ~100) already builds a
    pseudo-label from seed features -- but by MAJORITY-VOTE BINARIZATION at the
    median, and then spends it on an exhaustive subset search. This module uses
    the CONTINUOUS L-SML fusion instead and spends it on the gate objective,
    which is a different (and far cheaper) mechanism. The seed set and the
    orientation convention are deliberately borrowed from it.

Contract: sees only an UnlabeledCell -- no labels, no positive rate, ever.
On any failure the whole family degrades to the full-pool fallback (never raises).
"""

import numpy as np
import torch

from . import register
from ..fusion_utils import lsml_continuous, zscore
from ..subset_sweep import ANCHOR_PRIORITY

# ---- inherited from a2_groupfs's DUFS arm (kept identical on purpose, so the
# a6.dufs control is comparable to a2.dufs) ----------------------------------
K_NN = 7                       # self-tuning kernel neighbor
DIFFUSION_T = 2                # random-walk diffusion steps
STG_SIGMA = 0.5                # stochastic-gate noise
MU_INIT = 0.5                  # gate-mean init (unbiased prior)
GATE_LR = 2e-2
N_SEEDS_STABILITY = 5
R_MAX = 1200                   # sample-row subsample cap
BATCH = 256
EPOCHS_STAB = 120              # per (lambda, seed) stability training
# Label-free bracket around the data-driven lambda0, biased UPWARD. a2_groupfs
# learned this the hard way (its own comment: the first bracket "left every group
# gate saturated open" because the sample-graph trace term rewards opening ALL
# gates). a6 needs the extra headroom more than a2 did: removing the seed views
# shrinks the selectable pool, and lambda0 is derived from that pool's own
# smoothness magnitude, so a seed-depleted pool yields a weaker penalty and the
# fixed `mu > 0` cut stops separating. The stability rule's admissibility filter
# (3 <= median size < p) discards the saturate-open and collapse endpoints.
LAM_MULTS = (0.5, 1.0, 2.0, 4.0)

# ---- a6-specific -----------------------------------------------------------
N_SEED_VIEWS = 4               # K in "fuse K strong views into the anchor"
MIN_SEED_VIEWS = 3             # L-SML invariant: fewer than 3 views collapses

_EXPECTED = ('a6.pl_dufs', 'a6.pl_dufs_noseed', 'a6.pl_rank', 'a6.dufs')
_SQRT2 = 1.4142135623730951
_EPS = 1e-8


# ---------------------------------------------------------------------------
# graph operators (verbatim from a2_groupfs -- copied, not imported, because a2
# lives on its own branch lineage and a cross-import would couple the two)
# ---------------------------------------------------------------------------

def _self_tuning_affinity(pts, k):
    """Dense self-tuning affinity. pts: [m, dim]; gamma_i = distance to the
    k-th nearest neighbor. Diagonal zeroed (no self-loops)."""
    m = pts.shape[0]
    d2 = torch.cdist(pts, pts) ** 2
    k = int(max(1, min(k, m - 1)))
    knn_d2 = torch.topk(d2, k + 1, largest=False).values[:, -1]
    gamma = torch.sqrt(knn_d2.clamp_min(_EPS))
    W = torch.exp(-d2 / (gamma[:, None] * gamma[None, :] + _EPS))
    return W - torch.diag(torch.diagonal(W))


def _random_walk_power(W, t):
    """P = D^{-1} W, returned to the t-th power (t-step diffusion)."""
    P = W / W.sum(1, keepdim=True).clamp_min(_EPS)
    Pt = P
    for _ in range(t - 1):
        Pt = Pt @ P
    return Pt


# ---------------------------------------------------------------------------
# pseudo-label
# ---------------------------------------------------------------------------

def _seed_cols(cell):
    """Label-free seed choice: ANCHOR_PRIORITY order, intersected with the
    cell's available pool, capped at N_SEED_VIEWS. The cell's own anchor is
    force-included (it is the orientation reference downstream)."""
    cols, names = [], []
    for f in ANCHOR_PRIORITY:
        if f in cell.pool and len(cols) < N_SEED_VIEWS:
            cols.append(cell.pool.index(f))
            names.append(f)
    if cell.anchor_name in cell.pool:
        a = cell.pool.index(cell.anchor_name)
        if a not in cols:
            cols.insert(0, a)
            names.insert(0, cell.anchor_name)
            cols, names = cols[:N_SEED_VIEWS], names[:N_SEED_VIEWS]
    order = np.argsort(cols)
    return ([cols[i] for i in order], [names[i] for i in order])


def _pseudo_label(cell, seed_cols):
    """Continuous L-SML fusion of the seed views -> z-scored pseudo-label.

    The fused score's global sign is arbitrary (L-SML recovers the consensus up
    to a flip), so orient it against `cell.anchor` -- the same label-free
    anchor-orientation convention used everywhere else in this project.
    Returns (y_hat, meta) or (anchor, {'degraded': ...}) when < 3 seeds exist.
    """
    V = np.asarray(cell.V, dtype=np.float64)
    if len(seed_cols) < MIN_SEED_VIEWS:
        return zscore(np.asarray(cell.anchor, dtype=np.float64)), {
            'degraded': f'only {len(seed_cols)} seed views (<{MIN_SEED_VIEWS}); '
                        'pseudo-label falls back to the raw anchor'}
    fused, meta = lsml_continuous(*[V[:, c] for c in seed_cols])
    y = zscore(np.asarray(fused, dtype=np.float64))
    a = np.asarray(cell.anchor, dtype=np.float64)
    if np.corrcoef(y, a)[0, 1] < 0:
        y = -y
    return y, {'K': int(meta['K']), 'residual': round(float(meta['residual']), 5),
               'cross_weights': [round(float(w), 4) for w in meta['cross_weights']]}


def _corr_with(Xr, y):
    """Per-column Pearson |corr| against y, on the subsampled rows."""
    Xc = Xr - Xr.mean(0, keepdims=True)
    yc = y - y.mean()
    den = np.linalg.norm(Xc, axis=0) * np.linalg.norm(yc)
    return np.abs((Xc * yc[:, None]).sum(0) / np.maximum(den, _EPS))


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _train_gates(X_t, lam2, lam3, agree_t, epochs, batch, torch_seed):
    """DUFS per-feature STG gates on the Laplacian-score sample objective, plus
    the pseudo-label agreement reward. lam3=0 / agree_t=None reduces exactly to
    the a2.dufs objective. Returns per-feature gate means."""
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
        if lam3 > 0.0 and agree_t is not None:
            loss = loss - lam3 * (Pz * agree_t).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return mu.detach().numpy()


def _lambda0(X_t, gen):
    """Epoch-1 |L_s| under the unbiased gate prior, divided by the DUFS-only
    sparsity magnitude. At mu = MU_INIT the STG survival prob is Pz = 0.5
    exactly (erf(0) = 0), so |L_reg| = 0.5 and lambda0 = 2 |L_s|.

    NOTE this is NOT bit-identical to a2_groupfs's lambda0, which divides by the
    GROUP-weighted (Pz * M.sum(0)).sum() / (C*d). a6 has no grouping stage, so
    the DUFS-only ratio is the honest analogue -- the `a6.dufs` control is
    therefore a close but not exact reproduction of `a2.dufs`.
    """
    R, d = X_t.shape
    B = int(min(BATCH, R))
    k_samp = int(min(K_NN, B - 1))
    Xtil = X_t[torch.randperm(R, generator=gen)[:B]] * MU_INIT
    Pt = _random_walk_power(_self_tuning_affinity(Xtil, k_samp), DIFFUSION_T)
    Ls = abs(float(-(Xtil * (Pt @ Xtil)).sum() / (B * d)))
    return float(np.clip(Ls / 0.5, 1e-3, 1e4))


# ---------------------------------------------------------------------------
# selection helpers
# ---------------------------------------------------------------------------

def _jaccard(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _mean_pairwise_jaccard(sets):
    vals = [_jaccard(sets[i], sets[j])
            for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return float(np.mean(vals)) if vals else 1.0


def _stability_pick(X_t, lam2_of, lam3_of, agree_t, seeds, p_sel):
    """Label-free lambda choice: highest cross-seed selection Jaccard among the
    admissible multipliers (3 <= median selected size < p). Returns
    (lam_star_mult, stability, seed-averaged gates, per-mult diagnostics)."""
    stab = {}
    for mult in LAM_MULTS:
        gates, sels = [], []
        for s in seeds:
            g = _train_gates(X_t, lam2_of(mult), lam3_of(mult), agree_t,
                             EPOCHS_STAB, BATCH, s)
            gates.append(g)
            sels.append(np.where(g > 0.0)[0])
        med = int(np.median([len(x) for x in sels]))
        stab[mult] = {'jaccard': _mean_pairwise_jaccard(sels), 'med': med,
                      'admissible': bool(3 <= med < p_sel),
                      'mu_bar': np.mean(gates, axis=0)}
    adm = [(v['jaccard'], m) for m, v in stab.items() if v['admissible']]
    m_star = max(adm)[1] if adm else 1.0
    diag = {float(m): {'jaccard': round(v['jaccard'], 3), 'med_size': v['med'],
                       'admissible': v['admissible']} for m, v in stab.items()}
    return m_star, float(stab[m_star]['jaccard']), stab[m_star]['mu_bar'], diag


def _fallback(variant, p, err, base_diag=None):
    diag = {'error': str(err)}
    if base_diag is not None:
        diag = {**base_diag, 'fallback_reason': str(err)}
    return {'variant': variant, 'cols': np.arange(p, dtype=np.int64),
            'fallback': True, 'diag': diag}


def _full_pool_gates(mu_sel, sel_pool_cols, p, seed_cols, fill=1.0):
    """Scatter selectable-pool gates back to full-pool order. Seed columns are
    not gated (they are always on) and take `fill`."""
    out = np.full(p, float(fill), dtype=float)
    out[np.asarray(sel_pool_cols, dtype=int)] = mu_sel
    _ = seed_cols
    return out


# ---------------------------------------------------------------------------
# the registered selector family
# ---------------------------------------------------------------------------

@register('a6_pseudolabel_gates')
def a6_pseudolabel_gates(cell, rng, cache=None):
    torch.set_num_threads(1)
    p = cell.p
    try:
        V = np.asarray(cell.V, dtype=np.float64)
        n = V.shape[0]
        R = int(min(n, R_MAX))
        row_idx = np.sort(rng.choice(n, size=R, replace=False)) if R < n else np.arange(n)
        Xr = V[row_idx]
        X_full_t = torch.tensor(Xr, dtype=torch.float32)

        # ---- 1. pseudo-label from the seed views ---------------------------
        s_cols, s_names = _seed_cols(cell)
        y_hat, pl_meta = _pseudo_label(cell, s_cols)
        y_r = y_hat[row_idx]

        # ---- 2. selectable pool = everything the pseudo-label did NOT build --
        sel_cols = np.array([c for c in range(p) if c not in set(s_cols)], dtype=np.int64)
        if len(sel_cols) < 3:
            return [_fallback(v, p, f'selectable pool is {len(sel_cols)} cols '
                              f'after removing {len(s_cols)} seeds') for v in _EXPECTED]
        X_sel_t = torch.tensor(Xr[:, sel_cols], dtype=torch.float32)

        agree = _corr_with(Xr[:, sel_cols], y_r)
        # CENTERED: sum_f a_f = 0, so lambda3 redistributes the sparsity budget
        # across features instead of uniformly relaxing it. See module docstring.
        agree_c = agree - agree.mean()
        agree_t = torch.tensor(agree_c, dtype=torch.float32)

        gen0 = torch.Generator().manual_seed(int(rng.integers(2 ** 31)))
        lam0_sel = _lambda0(X_sel_t, gen0)
        lam0_full = _lambda0(X_full_t, torch.Generator().manual_seed(
            int(rng.integers(2 ** 31))))
        seeds = [int(rng.integers(2 ** 31)) for _ in range(N_SEEDS_STABILITY)]

        # ---- 3. supervised gates (two stages) -------------------------------
        # Stage 1: pick the SPARSITY level lambda2 with lambda3 = 0, by the same
        # cross-seed-stability rule the unsupervised control uses. Doing this
        # first is what makes the two arms comparable -- pinning lambda2 while
        # the control got to tune it was the first version's bug.
        m2s, _stab2s, _mu2s, lam2_diag_sel = _stability_pick(
            X_sel_t, lam2_of=lambda m: lam0_sel * m, lam3_of=lambda m: 0.0,
            agree_t=None, seeds=seeds, p_sel=len(sel_cols))
        lam2_sel = lam0_sel * m2s

        # Stage 2: scan lambda3 at that fixed sparsity. lambda3 is scaled by
        # lambda2 so the tilt is commensurate with the penalty it redistributes
        # (the centered agreement is already O(0.1-1)).
        m3, stab3, mu_sel, lam3_diag = _stability_pick(
            X_sel_t, lam2_of=lambda m: lam2_sel, lam3_of=lambda m: lam2_sel * m,
            agree_t=agree_t, seeds=seeds, p_sel=len(sel_cols))

        picked_local = np.where(mu_sel > 0.0)[0]
        picked = sel_cols[picked_local]

        gates_full = _full_pool_gates(mu_sel, sel_cols, p, s_cols, fill=1.0)
        gates_learned = [None] * p
        for j, c in enumerate(sel_cols):
            gates_learned[int(c)] = round(float(mu_sel[j]), 4)

        base_diag = {
            'seed_views': s_names, 'seed_cols': [int(c) for c in s_cols],
            'pseudo_label': pl_meta,
            'lambda0_selectable': round(lam0_sel, 6),
            'lambda2_mult': float(m2s), 'lambda2': round(lam2_sel, 6),
            'lambda2_stability': lam2_diag_sel,
            'lambda3_mult': float(m3), 'lambda3': round(lam2_sel * m3, 6),
            'lambda3_stability': lam3_diag, 'stability': round(stab3, 3),
            'n_selectable': int(len(sel_cols)), 'n_gated_open': int(len(picked)),
            'agreement_with_pl': {cell.pool[int(c)]: round(float(a), 3)
                                  for c, a in zip(sel_cols, agree)},
            'agreement_mean': round(float(agree.mean()), 4),
            # full-pool order; seeds = 1.0 (they ARE always on) -- consumed by
            # scripts/selector_choice_analysis.py's rho(gate, view AUROC)
            'feat_gate_means': [round(float(x), 3) for x in gates_full],
            # same, but seeds are None -- the honest measurement of the gates
            # that were actually LEARNED, without the always-on seeds inflating rho
            'feat_gate_means_learned': gates_learned,
            'subsampled_rows': int(R),
        }

        out = []
        s_arr = np.array(s_cols, dtype=np.int64)

        # -- a6.pl_dufs : headline -- seeds + pseudo-label-supervised picks ----
        cols_hl = np.array(sorted(set(picked.tolist()) | set(s_cols)), dtype=np.int64)
        if len(cols_hl) < 3:
            out.append(_fallback('a6.pl_dufs', p, 'selection < 3 cols', base_diag))
        else:
            out.append({'variant': 'a6.pl_dufs', 'cols': cols_hl,
                        'diag': {**base_diag, 'n_selected': int(len(cols_hl)),
                                 'note': 'seed views + gates supervised by the '
                                         'L-SML pseudo-label'}})

        # -- a6.pl_dufs_noseed : gated picks only ------------------------------
        if len(picked) < 3:
            out.append(_fallback('a6.pl_dufs_noseed', p, 'gate selection < 3 cols',
                                 base_diag))
        else:
            out.append({'variant': 'a6.pl_dufs_noseed', 'cols': np.sort(picked),
                        'diag': {**base_diag, 'n_selected': int(len(picked)),
                                 'note': 'gated picks WITHOUT the seed views -- '
                                         'isolates how much of any gain is just '
                                         '"the seed views are good views"'}})

        # -- a6.pl_rank : size-matched ablation, no gates ----------------------
        # Same budget as pl_dufs, chosen purely by |corr| with the pseudo-label.
        # If this ties pl_dufs, the gate machinery adds nothing over ranking.
        m = int(len(picked))
        if m < 3:
            out.append(_fallback('a6.pl_rank', p, 'size-match budget < 3', base_diag))
        else:
            top = sel_cols[np.argsort(agree)[::-1][:m]]
            cols_rank = np.array(sorted(set(top.tolist()) | set(s_cols)), dtype=np.int64)
            out.append({'variant': 'a6.pl_rank', 'cols': cols_rank,
                        'diag': {**base_diag, 'n_selected': int(len(cols_rank)),
                                 'rank_budget': m,
                                 'note': 'size-matched to pl_dufs; top-m by |corr| '
                                         'with the pseudo-label, no gate training'}})

        # -- a6.dufs : unsupervised control (lambda3 = 0, FULL pool) -----------
        # Must land near a2.dufs 0.7502. See _lambda0's note: the lambda0
        # derivation differs slightly from a2's (no grouping stage), so this is
        # a close reproduction, not a bit-identical one.
        m2, stab2, mu_full, lam2_diag = _stability_pick(
            X_full_t, lam2_of=lambda m: lam0_full * m, lam3_of=lambda m: 0.0,
            agree_t=None, seeds=seeds, p_sel=p)
        sel_ctrl = np.array(sorted(np.where(mu_full > 0.0)[0].tolist()), dtype=np.int64)
        d_ctrl = {'lambda0_full': round(lam0_full, 6), 'lambda2_mult': float(m2),
                  'lambda2_stability': lam2_diag, 'stability': round(stab2, 3),
                  'n_selected': int(len(sel_ctrl)),
                  'feat_gate_means': [round(float(x), 3) for x in mu_full],
                  'note': 'lambda3 = 0 control over the FULL pool -- the '
                          'unsupervised reference a2.dufs is meant to reproduce; '
                          'delta vs a6.pl_dufs is the pseudo-label supervision effect'}
        if len(sel_ctrl) < 3:
            out.append(_fallback('a6.dufs', p, 'control selection < 3 cols', d_ctrl))
        else:
            out.append({'variant': 'a6.dufs', 'cols': sel_ctrl, 'diag': d_ctrl})

        _ = s_arr
        return out

    except Exception as e:              # whole-family failure -> full-pool fallbacks
        return [_fallback(v, p, e) for v in _EXPECTED]


# ---------------------------------------------------------------------------
# smoke() — planted-signal known answer (auto-discovered by smoke_selectors.py)
# ---------------------------------------------------------------------------

def smoke():
    from ..selector_bench import UnlabeledCell
    from ..subset_sweep import CANONICAL_POOL
    import time

    # Planted world: a latent consensus y drives N_INFO informative columns; the
    # rest are pure noise. The FIRST columns are laid out so the ANCHOR_PRIORITY
    # names land on informative columns (they become the seed views), and the
    # gates must then rediscover the REMAINING informative columns from the
    # pseudo-label without ever being allowed to pick the seeds back.
    #
    # N_INFO is deliberately > N_SEED_VIEWS + 3: the seed views are REMOVED from
    # the selectable pool, so a world with only a couple of informative columns
    # left over would be mostly-noise by construction and the >=3-column
    # admissibility floor alone would force noise into the selection. That is a
    # property of the toy, not of the method. Real pools are the other way round
    # (c46 has ~28 of 30 views better than chance), so the toy matches that.
    rng_np = np.random.default_rng(20260722)
    n, p, N_INFO = 400, 14, 8
    y = rng_np.standard_normal(n)
    cols = [zscore(y + 0.55 * rng_np.standard_normal(n)) for _ in range(N_INFO)]
    cols += [zscore(rng_np.standard_normal(n)) for _ in range(p - N_INFO)]
    V = np.column_stack(cols)

    # Pool names must put ANCHOR_PRIORITY members on informative cols 0..5.
    pool = list(ANCHOR_PRIORITY)                       # 4 names -> cols 0..3
    pool += [f for f in CANONICAL_POOL if f not in pool][:p - len(pool)]
    assert len(pool) == p
    pool_bits = np.arange(p, dtype=np.uint8)
    rho = np.abs(np.corrcoef(V.T))
    cell = UnlabeledCell(domain='smoke', cell_key='pseudolabel', pool=pool,
                         pool_bits=pool_bits, V=V, anchor=zscore(V[:, 0]),
                         anchor_name=pool[0], rho=rho)

    t0 = time.time()
    sels1 = a6_pseudolabel_gates(cell, np.random.default_rng([0, 99]))
    elapsed = time.time() - t0
    sels2 = a6_pseudolabel_gates(cell, np.random.default_rng([0, 99]))

    by1 = {s['variant']: s for s in sels1}
    assert set(by1) == set(_EXPECTED), f"variant set changed: {sorted(by1)}"
    for s in sels1:
        assert not s.get('fallback', False), f"{s['variant']} fell back: {s['diag']}"

    # (a) determinism under equal-seeded rng
    by2 = {s['variant']: s for s in sels2}
    for v in _EXPECTED:
        assert list(by1[v]['cols']) == list(by2[v]['cols']), f"{v}: cols not deterministic"

    diag = by1['a6.pl_dufs']['diag']

    # (b) the circularity guard actually holds: seeds are never gate-selected
    seed_cols = set(diag['seed_cols'])
    assert len(seed_cols) >= MIN_SEED_VIEWS, f"(b) only {len(seed_cols)} seeds"
    noseed = set(int(c) for c in by1['a6.pl_dufs_noseed']['cols'])
    assert not (noseed & seed_cols), \
        f"(b) seed cols leaked into the gated selection: {sorted(noseed & seed_cols)}"

    # (c) the pseudo-label is informative: informative cols agree with it more
    # than noise cols do (this is the premise the whole module rests on).
    ag = diag['agreement_with_pl']
    info = [ag[pool[c]] for c in range(N_INFO) if pool[c] in ag]
    noise = [ag[pool[c]] for c in range(N_INFO, p) if pool[c] in ag]
    assert info and noise, "(c) agreement diag missing informative/noise cols"
    assert np.mean(info) > np.mean(noise) + 0.20, \
        f"(c) pseudo-label not informative: info {np.mean(info):.3f} vs noise {np.mean(noise):.3f}"

    # (d) the supervised gates exclude the planted noise. Compared against the
    # unsupervised control on the SAME per-column basis (the control runs on the
    # full pool, so compare noise RATE, not raw counts).
    noise_cols = set(range(N_INFO, p))
    picked_noise = len(noseed & noise_cols)
    ctrl = set(int(c) for c in by1['a6.dufs']['cols'])
    ctrl_noise = len(ctrl & noise_cols)
    assert picked_noise == 0, \
        f"(d) supervised gates kept planted noise: {sorted(noseed & noise_cols)}"

    # (e) the gates ORDER views correctly -- the property the mechanism gate
    # measures on real cells (rho(gate, separability)). Stricter than (d), and it
    # still holds when the >0 cut is mis-calibrated by pool composition.
    gl = diag['feat_gate_means_learned']
    g_info = [gl[c] for c in range(N_INFO) if gl[c] is not None]
    g_noise = [gl[c] for c in range(N_INFO, p) if gl[c] is not None]
    assert g_info and g_noise, "(e) no learned gates on both sides"
    assert min(g_info) > max(g_noise), \
        f"(e) gate ordering violated: min informative {min(g_info):.3f} <= max noise {max(g_noise):.3f}"

    # (f) runtime under budget
    assert elapsed < 180.0, f"(f) runtime {elapsed:.1f}s over budget"

    print(f"    [note] a6 smoke: seeds={diag['seed_views']} "
          f"pl_K={diag['pseudo_label'].get('K')} lam3x{diag['lambda3_mult']} "
          f"stab={diag['stability']} gated_open={diag['n_gated_open']}"
          f"/{diag['n_selectable']} noise_kept={picked_noise} (ctrl {ctrl_noise}) "
          f"agree info={np.mean(info):.2f} noise={np.mean(noise):.2f} {elapsed:.1f}s")
