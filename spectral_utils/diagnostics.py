"""
diagnostics.py — L-SML pipeline decomposition + visualizations.

Used by LSML_Diagnostics.ipynb to isolate which stage of the
binary-unsupervised pipeline costs the most AUROC vs the old supervised
continuous Nadler. The five "stages" are:

  1. continuous + supervised sign + simple average  (upper bound)
  2. continuous + supervised sign + SML weights     (fusion-vs-avg cost)
  3. binary     + supervised sign + SML weights     (binarization cost)
  4. binary     + L-SML sign      + SML (1 group)   (+ sign-resolution cost)
  5. binary     + L-SML sign      + L-SML (K groups)(+ group-detection cost)
     [= the official Step 107 number]

Each stage swaps exactly one variable from the previous, so the AUROC gap
between consecutive rows attributes the loss to one transformation.

All helpers are pure functions that take (feats_dict, feat_names, labels)
and return numpy/dict outputs. Plotting helpers take a matplotlib Axes.
"""
from __future__ import annotations

import numpy as np
from typing import Iterable

from .fusion_utils import (
    boot_auc, zscore,
    sml_fuse_signed, lsml_fuse,
    sml_unsupervised_compare,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sign & binarization primitives
# ─────────────────────────────────────────────────────────────────────────────

def _supervised_signs(feats_dict: dict, feat_names: list, labels) -> dict:
    """For each feature, sign = +1 if higher → likely correct (AUROC ≥ 0.5)."""
    signs = {}
    for f in feat_names:
        a, _, _ = boot_auc(labels, np.asarray(feats_dict[f], dtype=float), n=100)
        signs[f] = +1 if a >= 0.5 else -1
    return signs


def _binarize(feats_dict: dict, feat_names: list, signs: dict | None = None,
              quantile: float = 0.5) -> dict:
    """Binarize each feature to ±1 via quantile threshold. signs=None ⇒ no orientation."""
    out = {}
    for f in feat_names:
        x = np.asarray(feats_dict[f], dtype=float)
        if signs is not None:
            x = x * signs[f]
        t = np.quantile(x, quantile)
        out[f] = np.where(x > t, 1.0, -1.0)
    return out


def _resolve_global_sign(scores: np.ndarray,
                         binary_classifiers: list) -> np.ndarray:
    """
    Paper 2 assumption (iii): the majority of classifiers beat random.

    The simple equal-weight average of binary ±1 classifiers therefore
    points in the correct ensemble direction (with noise that vanishes
    as the majority margin grows). Flip the fused score whenever it is
    anti-correlated with the equal-weight average — this is independent
    of any internal sign-rule a fusion subroutine may have applied, so
    it works for both single-group SML and group-aware L-SML output.
    """
    X = np.column_stack([np.asarray(b, dtype=float) for b in binary_classifiers])
    avg = X.mean(axis=1)
    if np.std(scores) == 0 or np.std(avg) == 0:
        return scores
    if np.corrcoef(scores, avg)[0, 1] < 0:
        return -scores
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Core decomposition
# ─────────────────────────────────────────────────────────────────────────────

def decompose_auroc(feats_dict: dict, feat_names: list, labels,
                    K_range: Iterable[int] = range(2, 7),
                    boot_n: int = 500) -> dict:
    """
    Compute AUROC of fused scores at five pipeline settings and per-feature
    AUROC at each. Reveals which transformation costs the most.

    Returns dict with:
        rows:     [{'name', 'auc', 'lo', 'hi'}] × 5
        per_feat: (n_feats, 5) array of per-feature AUROC
        signs:    {'supervised': {f: ±1}, 'lsml': {f: ±1}, 'agree': {f: bool}}
        groups:   {'K': int, 'assignment': np.ndarray, 'ARI_vs_eigengap': float}
        binary:   {'supervised': {f: arr}, 'unsupervised': {f: arr}}
        scores:   {'final': fused score array of stage 5 (label-oriented sign)}
    """
    labels = np.asarray(labels)
    n_feats = len(feat_names)

    # ── Supervised signs (oracle) ────────────────────────────────────────
    sup_signs = _supervised_signs(feats_dict, feat_names, labels)

    # ── Continuous oriented and z-scored ─────────────────────────────────
    oriented_cont = {f: np.asarray(feats_dict[f], dtype=float) * sup_signs[f]
                     for f in feat_names}
    z_cont = {f: zscore(oriented_cont[f]) for f in feat_names}

    # Stage 1: continuous + sup_sign + simple average
    X1 = np.column_stack([z_cont[f] for f in feat_names])
    s1 = X1.mean(axis=1)
    a1, l1, h1 = boot_auc(labels, s1, n=boot_n)

    # Stage 2: continuous + sup_sign + SML weights
    s2, _ = sml_fuse_signed(*[z_cont[f] for f in feat_names])
    a2, l2, h2 = boot_auc(labels, s2, n=boot_n)

    # Stage 3: binary + sup_sign + SML weights
    bin_sup = _binarize(feats_dict, feat_names, signs=sup_signs, quantile=0.5)
    s3, _ = sml_fuse_signed(*[bin_sup[f] for f in feat_names])
    a3, l3, h3 = boot_auc(labels, s3, n=boot_n)

    # Stage 4: binary + L-SML sign + SML (single group, no group detection)
    bin_raw = _binarize(feats_dict, feat_names, signs=None, quantile=0.5)
    bin_raw_list = [bin_raw[f] for f in feat_names]
    s4_raw, _ = sml_fuse_signed(*bin_raw_list)
    s4 = _resolve_global_sign(s4_raw, bin_raw_list)
    a4, l4, h4 = boot_auc(labels, s4, n=boot_n)

    # Stage 5: binary + L-SML sign + L-SML (K groups, official)
    cmp = sml_unsupervised_compare(feats_dict, feat_names,
                                   K_range=K_range, labels=labels)
    s5 = _resolve_global_sign(np.asarray(cmp['residual_fused']), bin_raw_list)
    a5, l5, h5 = boot_auc(labels, s5, n=boot_n)

    rows = [
        {'name': '1. continuous + sup. sign + simple avg', 'auc': a1, 'lo': l1, 'hi': h1},
        {'name': '2. continuous + sup. sign + SML',         'auc': a2, 'lo': l2, 'hi': h2},
        {'name': '3. binary     + sup. sign + SML',         'auc': a3, 'lo': l3, 'hi': h3},
        {'name': '4. binary     + L-SML sign + SML (1 grp)','auc': a4, 'lo': l4, 'hi': h4},
        {'name': '5. binary     + L-SML sign + L-SML',      'auc': a5, 'lo': l5, 'hi': h5},
    ]

    # ── Per-feature AUROC at each stage ──────────────────────────────────
    per_feat = np.zeros((n_feats, 5))

    # Stages 1 & 2 share the same per-feature input (continuous oriented).
    for i, f in enumerate(feat_names):
        a, _, _ = boot_auc(labels, z_cont[f], n=100)
        per_feat[i, 0] = a
        per_feat[i, 1] = a
    # Stage 3 input: binary supervised-oriented
    for i, f in enumerate(feat_names):
        a, _, _ = boot_auc(labels, bin_sup[f], n=100)
        per_feat[i, 2] = a
    # Stages 4 & 5: binary with L-SML-inferred sign per feature.
    # Recover per-feature L-SML sign by correlating each raw binary feature with
    # the final fused score; sign = sign of that correlation.
    lsml_signs = {}
    for i, f in enumerate(feat_names):
        b = bin_raw[f]
        corr = float(np.corrcoef(b, s5)[0, 1]) if np.std(b) > 0 else 0.0
        lsml_signs[f] = +1 if corr >= 0 else -1
        b_oriented = b * lsml_signs[f]
        a, _, _ = boot_auc(labels, b_oriented, n=100)
        per_feat[i, 3] = a
        per_feat[i, 4] = a

    agree = {f: sup_signs[f] == lsml_signs[f] for f in feat_names}

    return {
        'rows':     rows,
        'per_feat': per_feat,
        'signs':    {'supervised': sup_signs, 'lsml': lsml_signs, 'agree': agree},
        'groups':   {
            'K':                cmp['K_residual'],
            'assignment':       np.asarray(cmp['residual_meta']['c']),
            'ARI_vs_eigengap':  cmp['group_ARI'],
            'K_eigengap':       cmp['K_eigengap'],
        },
        'binary':   {'supervised': bin_sup, 'unsupervised': bin_raw},
        'scores':   {'final': s5, 'continuous_avg': s1, 'sml_continuous': s2},
        'n':        len(labels),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def threshold_sensitivity(feats_dict: dict, feat_names: list, labels,
                          thresholds=(0.25, 0.5, 0.75),
                          K_range: Iterable[int] = range(2, 7),
                          boot_n: int = 500) -> dict:
    """
    Sweep binarization quantile and rerun full L-SML at each. Returns
    {q: {'auc', 'lo', 'hi', 'K'}}. Median (0.5) is the default L-SML setting;
    sweeping flags whether the threshold choice matters.
    """
    labels = np.asarray(labels)
    out = {}
    for q in thresholds:
        bin_dict = _binarize(feats_dict, feat_names, signs=None, quantile=q)
        bin_list = [bin_dict[f] for f in feat_names]
        scores, meta = lsml_fuse(*bin_list, K_range=K_range)
        scores = _resolve_global_sign(np.asarray(scores), bin_list)
        a, lo, hi = boot_auc(labels, scores, n=boot_n)
        out[q] = {'auc': a, 'lo': lo, 'hi': hi, 'K': meta.get('K')}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_decomposition(rows: list, ax, title: str = ''):
    """Horizontal bar chart of the 5 stage AUROCs with 95% CIs."""
    names = [r['name'] for r in rows]
    aucs  = [r['auc'] for r in rows]
    lows  = [r['auc'] - r['lo'] for r in rows]
    highs = [r['hi'] - r['auc']  for r in rows]
    ypos  = np.arange(len(rows))
    colors = ['#2c7fb8', '#41b6c4', '#a1dab4', '#fdae61', '#d7191c']
    ax.barh(ypos, aucs, xerr=[lows, highs], color=colors, ecolor='#333', capsize=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9, family='monospace')
    ax.invert_yaxis()
    ax.set_xlim(0.45, 1.0)
    ax.axvline(0.5, color='#888', lw=0.8, ls='--')
    ax.set_xlabel('AUROC')
    if title:
        ax.set_title(title, fontsize=11)
    for i, (a, lo, hi) in enumerate(zip(aucs,
                                        [r['lo'] for r in rows],
                                        [r['hi'] for r in rows])):
        ax.text(a + 0.005, i, f'{100*a:.1f}', va='center', fontsize=8)


def plot_per_feature_heatmap(per_feat: np.ndarray, feat_names: list, ax,
                             title: str = ''):
    """16 × 5 heatmap of per-feature AUROC at each pipeline stage."""
    im = ax.imshow(per_feat, aspect='auto', cmap='RdYlGn', vmin=0.4, vmax=0.9)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['1.cont/sup/avg', '2.cont/sup/SML', '3.bin/sup/SML',
                        '4.bin/lsml/SML', '5.bin/lsml/LSML'],
                       fontsize=7, rotation=30, ha='right')
    for i in range(per_feat.shape[0]):
        for j in range(per_feat.shape[1]):
            ax.text(j, i, f'{100*per_feat[i, j]:.0f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if per_feat[i, j] < 0.55 else 'black')
    if title:
        ax.set_title(title, fontsize=11)
    return im


def plot_sign_agreement(signs: dict, feat_names: list, ax, title: str = ''):
    """Bar chart of sign agreement per feature (sup vs L-SML)."""
    sup = [signs['supervised'][f] for f in feat_names]
    uns = [signs['lsml'][f]       for f in feat_names]
    match = [s == u for s, u in zip(sup, uns)]
    ypos = np.arange(len(feat_names))
    colors = ['#2ca02c' if m else '#d62728' for m in match]
    ax.barh(ypos, [1]*len(feat_names), color=colors, alpha=0.6)
    for i, (s, u, m) in enumerate(zip(sup, uns, match)):
        glyph = f'sup={s:+d}  lsml={u:+d}'
        ax.text(0.02, i, glyph, va='center', fontsize=8,
                color='black')
    ax.set_yticks(ypos)
    ax.set_yticklabels(feat_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    if title:
        n_agree = sum(match)
        ax.set_title(f'{title}  ({n_agree}/{len(feat_names)} agree)', fontsize=11)


def plot_threshold_sweep(sweep: dict, ax, title: str = ''):
    """Line plot of AUROC vs binarization quantile."""
    qs   = sorted(sweep.keys())
    aucs = [sweep[q]['auc'] for q in qs]
    los  = [sweep[q]['lo']  for q in qs]
    his  = [sweep[q]['hi']  for q in qs]
    ax.errorbar(qs, aucs,
                yerr=[[a - l for a, l in zip(aucs, los)],
                      [h - a for a, h in zip(aucs, his)]],
                marker='o', capsize=4, color='#d7191c')
    for q in qs:
        ax.text(q, sweep[q]['auc'] + 0.005,
                f'K={sweep[q]["K"]}', fontsize=8, ha='center')
    ax.set_xlabel('Binarization quantile')
    ax.set_ylabel('L-SML AUROC')
    ax.set_xticks(qs)
    ax.set_ylim(0.45, max(0.9, max(his) + 0.05))
    ax.axhline(0.5, color='#888', lw=0.8, ls='--')
    if title:
        ax.set_title(title, fontsize=11)


def plot_correlation_with_groups(feats_dict: dict, feat_names: list,
                                 group_assignment: np.ndarray, ax,
                                 title: str = ''):
    """Spearman ρ heatmap of continuous features, reordered by L-SML group."""
    from scipy.stats import spearmanr
    X = np.column_stack([np.asarray(feats_dict[f], dtype=float) for f in feat_names])
    rho, _ = spearmanr(X)
    if np.isscalar(rho):
        rho = np.array([[1.0, rho], [rho, 1.0]])
    order = np.argsort(group_assignment)
    rho_sorted   = rho[np.ix_(order, order)]
    names_sorted = [feat_names[i] for i in order]
    groups_sorted = group_assignment[order]
    im = ax.imshow(rho_sorted, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels(names_sorted, fontsize=7, rotation=90)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=7)
    # Group boundaries
    boundaries = np.where(np.diff(groups_sorted) != 0)[0]
    for b in boundaries:
        ax.axhline(b + 0.5, color='black', lw=1.2)
        ax.axvline(b + 0.5, color='black', lw=1.2)
    if title:
        ax.set_title(title + f'  (K={int(group_assignment.max())+1} groups)', fontsize=11)
    return im


__all__ = [
    'decompose_auroc',
    'threshold_sensitivity',
    'plot_decomposition',
    'plot_per_feature_heatmap',
    'plot_sign_agreement',
    'plot_threshold_sweep',
    'plot_correlation_with_groups',
]
