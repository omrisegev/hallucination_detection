"""
adaptive_k — D1: label-free per-cell subset size K*.

WHY
---
Every selector in this repo commits to a subset size chosen globally: GOOD_5 fixes
5, `a6.pruned_dufs` hard-slices the top 11 gates + seeds = 15, and the pilot script
`eval_unsupervised_dufs_pruned.py:100-106` used a two-value step rule
(`6 if gap >= 3 else 15`). Measured, a fixed budget is the wrong abstraction: the
K=15 cap scored 0.7487 macro, BELOW the un-pruned `a6.pl_dufs` at 0.7524. Math cells
are rank-one dominant and want a compact set; QA cells carry higher-rank structure
and want broader coverage, so one global K necessarily mis-serves one domain.

Step 197 observed that the L-SML covariance residual correlates with downstream
AUROC (Spearman r=+0.648) and the spectral gap weakly so (r=+0.423). That was only
ever a CORRELATION STUDY -- an earlier draft of the advisor letter described a
"continuous residual elbow" cutoff as if it were deployed, but no such cutoff
existed anywhere in the codebase. This module is that cutoff, built properly.

THE HONEST TEST
---------------
Correlating with AUROC is NOT the same as predicting the optimal K. `validate()`
therefore compares each label-free rule against the per-cell oracle K (the prefix
size that actually maximises AUROC, computed WITH labels, for validation only).
If no rule tracks oracle-K, D1 does not work and should be reported as such rather
than shipped on the strength of the AUROC correlation.

CONTRACT
--------
`predict_k` sees only unlabeled data (V + a ranking). Labels enter `oracle_k` and
`validate` exclusively, and those are audit entry points, never selector paths.
"""

import numpy as np

from ..fusion_utils import lsml_continuous, zscore

# L-SML needs >= 3 views; 15 matches the K_max budget the pruning sweeps used.
K_MIN = 3
K_MAX = 15
_EPS = 1e-12


def spectral_gap(V):
    """lambda1 / lambda2 of the feature covariance. High gap => one dominant
    consensus direction (rank-one-ish) => a compact subset should suffice."""
    V = np.asarray(V, dtype=np.float64)
    if V.shape[1] < 2:
        return 1.0
    cov = np.cov(V, rowvar=False)
    ev = np.linalg.eigvalsh(cov)[::-1]
    return float(ev[0] / max(ev[1], _EPS))


def participation_ratio(V):
    """Effective rank of cov(V) via the participation ratio (sum l)^2 / sum l^2.

    H2 (Extension H). Reads the pool's own "signal dimension" off the covariance
    spectrum instead of fixing K: a rank-one-dominant pool gives PR -> 1, a pool
    with d comparable directions gives PR -> d. Label-free.
    """
    V = np.asarray(V, dtype=np.float64)
    if V.shape[1] < 2:
        return 1.0
    ev = np.linalg.eigvalsh(np.cov(V, rowvar=False))
    ev = ev[ev > _EPS]
    if ev.size == 0:
        return 1.0
    return float(ev.sum() ** 2 / np.sum(ev ** 2))


def mp_signal_dim(V):
    """Count of corr(V) eigenvalues above the Marchenko-Pastur upper edge.

    H2 (Extension H). Under the null that the views are pure noise, the sample
    correlation eigenvalues concentrate below lam+ = (1 + sqrt(p/n))^2. Anything
    above that edge is signal, so the count estimates the number of genuinely
    informative directions in the pool. Label-free.
    """
    V = np.asarray(V, dtype=np.float64)
    n, p = V.shape
    if p < 2 or n < 2:
        return 1
    C = np.corrcoef(V, rowvar=False)
    C[~np.isfinite(C)] = 0.0
    ev = np.linalg.eigvalsh(C)
    lam_plus = (1.0 + np.sqrt(p / n)) ** 2
    return int(max(1, np.sum(ev > lam_plus)))


def bootstrap_elbow_k(V, ranking, k_min=K_MIN, k_max=K_MAX, n_boot=25,
                      seed=0):
    """Median of the forward-difference elbow over row bootstraps.

    H2 (Extension H). `elbow_fwd` reads one elbow off one realisation of the
    residual curve; on a finite sample that argmax is noisy. Resampling rows and
    taking the median elbow is the stability-selection analogue. Label-free.
    """
    V = np.asarray(V, dtype=np.float64)
    n = V.shape[0]
    rng = np.random.default_rng(seed)
    picks = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        ks, eps = residual_curve(V[idx], ranking, k_min, k_max)
        if len(ks) < 3:
            continue
        picks.append(ks[int(np.argmax(np.diff(_norm_curve(eps))))])
    if not picks:
        return int(min(max(k_min, 1), max(len(list(ranking)), 1)))
    return int(np.median(picks))


def residual_curve(V, ranking, k_min=K_MIN, k_max=K_MAX):
    """eps(k) for k in [k_min, k_max] over the top-k prefix of `ranking`.

    eps(k) is the canonical project residual: the Paper-1 Eq.(14) L-SML fit error
    returned as `meta['residual']` by `lsml_continuous`. Reused deliberately
    rather than re-deriving a ||R - w w^T||_F^2 by hand, so the curve is the same
    quantity the rest of the pipeline reports.

    Returns (ks, eps) as parallel lists; ks may be shorter than the request when
    the ranking or the pool runs out.
    """
    V = np.asarray(V, dtype=np.float64)
    ranking = list(ranking)
    hi = int(min(k_max, len(ranking)))
    ks, eps = [], []
    for k in range(int(k_min), hi + 1):
        cols = ranking[:k]
        try:
            _, meta = lsml_continuous(*[V[:, c] for c in cols])
        except Exception:
            continue
        r = float(meta.get('residual', np.nan))
        if np.isfinite(r):
            ks.append(k)
            eps.append(r)
    return ks, eps


def _norm_curve(eps):
    """Scale eps to [0,1]; the residual's magnitude varies by ~2 orders across
    cells, so any threshold rule has to work on the shape, not the level."""
    e = np.asarray(eps, dtype=np.float64)
    lo, hi = float(np.min(e)), float(np.max(e))
    return (e - lo) / max(hi - lo, _EPS)


def raw_k(V, ranking, rule):
    """UNCLAMPED estimate for the spectrum rules — the degeneracy diagnostic.

    `predict_k` clamps into [k_min, k_max], which hides the failure mode found in
    Step 201: on rank-one-dominant cells the participation ratio is ~1-2, so
    `eff_rank` clamps to k_min on 25/25 cells and looks like a constant K rather
    than an adaptive rule. Reporting the raw value makes that visible instead of
    presenting a clamped constant as "per-cell adaptive".
    """
    V = np.asarray(V, dtype=np.float64)
    if rule == 'eff_rank':
        return participation_ratio(V)
    if rule == 'mp_floor':
        return float(mp_signal_dim(V))
    if rule == 'stability':
        # bootstrap the participation ratio over resampled rows
        n = V.shape[0]
        rng = np.random.default_rng(42)
        vals = []
        for _ in range(10):
            idx = rng.integers(0, n, size=n)
            vals.append(participation_ratio(V[idx]))
        return float(np.median(vals)) if vals else float(K_MIN)
    if rule == 'boot_elbow':
        return float(bootstrap_elbow_k(V, ranking))
    raise ValueError(f"raw_k: unknown rule {rule!r}")


def predict_k(V, ranking, rule='elbow_fwd', k_min=K_MIN, k_max=K_MAX,
              gap_threshold=3.0, compact_k=6, broad_k=15, return_curve=False):
    """Label-free per-cell K*. Never sees labels.

    Rules
      elbow_fwd  argmax_k [eps(k+1) - eps(k)]  -- the forward-difference elbow
                 the advisor draft described. Picks the k just before the
                 largest jump in fit error.
      knee       max discrete curvature eps(k-1) - 2 eps(k) + eps(k+1): the
                 point where the residual curve bends hardest.
      plateau    smallest k whose normalised residual is within 10% of the
                 curve minimum -- "stop once the fit stops improving".
      gap_step   legacy two-value rule (compact_k if gap >= threshold else
                 broad_k), kept so the benchmark can show what it replaces.
      fixed      always k_max (the a6.pruned_dufs status quo).
      eff_rank   Participation ratio (sum lambda)^2 / sum (lambda^2) of feature covariance.
      mp_floor   Marchenko-Pastur upper edge threshold count: eigenvalues > sigma^2 (1 + sqrt(p/n))^2.
      stability  Bootstrap feature stability cutoff.
    """
    ranking = list(ranking)
    if rule == 'fixed':
        k = int(min(k_max, len(ranking)))
        return (k, ([], [])) if return_curve else k
    if rule == 'gap_step':
        g = spectral_gap(V)
        k = compact_k if g >= gap_threshold else broad_k
        k = int(min(max(k, k_min), len(ranking)))
        return (k, ([], [])) if return_curve else k

    V = np.asarray(V, dtype=np.float64)
    n, p = V.shape

    # Spectrum rules delegate to the module-level estimators above rather than
    # re-implementing them inline (Step 201: the two had drifted into duplicate,
    # slightly different definitions of the same quantity).
    #
    # IMPORTANT (Step 201, defect 3): these return a raw estimate that is then
    # CLAMPED into [k_min, k_max]. On the real in-scope cells the pool is
    # rank-one dominant, so the participation ratio lands ~1-2 and `eff_rank`
    # clamps to k_min on every cell -- i.e. it silently becomes a fixed K, the
    # very prior it was meant to remove. Use `raw_k()` to see the unclamped
    # value, and `validate()` to test any rule against oracle-K before adopting.
    if rule in ('eff_rank', 'mp_floor', 'stability'):
        k = int(round(raw_k(V, ranking, rule)))
        k = int(min(max(k, k_min), min(k_max, len(ranking))))
        return (k, ([], [])) if return_curve else k

    ks, eps = residual_curve(V, ranking, k_min, k_max)
    if len(ks) < 3:
        k = int(min(max(k_min, 1), max(len(ranking), 1)))
        return (k, (ks, eps)) if return_curve else k

    e = _norm_curve(eps)
    if rule == 'elbow_fwd':
        d = np.diff(e)                      # eps(k+1) - eps(k)
        k = ks[int(np.argmax(d))]
    elif rule == 'knee':
        curv = e[:-2] - 2.0 * e[1:-1] + e[2:]
        k = ks[int(np.argmax(np.abs(curv))) + 1]
    elif rule == 'plateau':
        thr = float(np.min(e)) + 0.10
        idx = int(np.argmax(e <= thr))
        k = ks[idx]
    else:
        raise ValueError(f"unknown rule {rule!r}")

    k = int(min(max(k, k_min), len(ranking)))
    return (k, (ks, eps)) if return_curve else k


# ---------------------------------------------------------------------------
# validation only -- these take labels and must never be called from a selector
# ---------------------------------------------------------------------------

def oracle_k(V, ranking, labels, k_min=K_MIN, k_max=K_MAX):
    """The prefix size that actually maximises fused AUROC. Uses labels; this is
    the yardstick D1 is measured against, not something the pipeline may call."""
    from sklearn.metrics import roc_auc_score

    V = np.asarray(V, dtype=np.float64)
    ranking = list(ranking)
    hi = int(min(k_max, len(ranking)))
    best_k, best_auc, curve = None, -1.0, []
    for k in range(int(k_min), hi + 1):
        cols = ranking[:k]
        try:
            fused, _ = lsml_continuous(*[V[:, c] for c in cols])
        except Exception:
            continue
        y = zscore(np.asarray(fused, dtype=np.float64))
        try:
            a = float(roc_auc_score(labels, y))
        except ValueError:
            continue
        a = max(a, 1.0 - a)          # unsigned: orientation is a separate concern
        curve.append((k, a))
        if a > best_auc:
            best_k, best_auc = k, a
    return best_k, best_auc, curve


def validate(cells, rankings, rules=('elbow_fwd', 'knee', 'plateau', 'gap_step', 'fixed', 'eff_rank', 'mp_floor', 'stability')):
    """Per-rule agreement with oracle-K across cells.

    `cells`    : {cell_key: {'V':..., 'labels':...}}
    `rankings` : {cell_key: [col, ...]} label-free feature order
    Returns (per_cell_rows, per_rule_summary).
    """
    from scipy.stats import spearmanr

    rows = []
    for ck, cd in cells.items():
        rank = rankings.get(ck)
        if not rank:
            continue
        V, labels = cd['V'], cd['labels']
        k_star, auc_star, _ = oracle_k(V, rank, labels)
        if k_star is None:
            continue
        row = {'cell': ck, 'group': cd.get('group', '?'),
               'oracle_k': k_star, 'oracle_auc': round(auc_star, 4),
               'spectral_gap': round(spectral_gap(V), 4)}
        for r in rules:
            kh = predict_k(V, rank, rule=r)
            row[f'k_{r}'] = kh
            # AUROC actually obtained at the predicted k
            _, _, curve = oracle_k(V, rank, labels, k_min=kh, k_max=kh)
            row[f'auc_{r}'] = round(curve[0][1], 4) if curve else None
        rows.append(row)

    summary = []
    for r in rules:
        pred = [x[f'k_{r}'] for x in rows]
        orac = [x['oracle_k'] for x in rows]
        aucs = [x[f'auc_{r}'] for x in rows if x.get(f'auc_{r}') is not None]
        oaucs = [x['oracle_auc'] for x in rows if x.get(f'auc_{r}') is not None]
        rs, p = (spearmanr(pred, orac) if len(set(pred)) > 1 else (float('nan'),) * 2)
        summary.append({
            'rule': r,
            'spearman_r': round(float(rs), 4) if np.isfinite(rs) else None,
            'p_value': round(float(p), 4) if np.isfinite(p) else None,
            'mean_abs_dk': round(float(np.mean(np.abs(np.array(pred) - np.array(orac)))), 3),
            'median_k': int(np.median(pred)),
            'macro_auc': round(float(np.mean(aucs)), 4) if aucs else None,
            'oracle_macro_auc': round(float(np.mean(oaucs)), 4) if oaucs else None,
            'auc_gap_to_oracle': (round(float(np.mean(oaucs) - np.mean(aucs)), 4)
                                  if aucs else None),
        })
    summary.sort(key=lambda d: -(d['macro_auc'] or 0))
    return rows, summary
