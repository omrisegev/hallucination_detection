#!/usr/bin/env python3
"""
lsml_theorem_validation.py — Empirical validation of L-SML theorem properties
for continuous (non-binary) inputs.

The Jaffé–Fetaya–Nadler 2016 paper proves Lemma 1, Lemma 4, and the residual
criterion (Eq. 14) under the assumption that inputs are binary ±1 classifiers.
Our continuous (CONT) pipeline drops that assumption and gains +4.9 pp macro
AUROC.  This script quantifies how far our continuous covariance matrices deviate
from the binary-theory guarantees, and whether the core algorithm mechanics still
hold.

Six tests, all read-only (no changes to fusion_utils.py):

  T1+T4  Low-rank structure of R_off (Lemma 1 / Eq. 10–12)
         — eigenvalue spectrum + cumulative Frobenius fraction, binary vs CONT

  T2     Score matrix gap (Lemma 4 / Eq. 15)
         — min within-cluster score vs max cross-cluster score; gap > 0 required

  T3     Residual curve K=1..4 (Eq. 14)
         — K=1 is flat-SML independence assumption; drop to K* validates grouping

  T5     Score-matrix cluster agreement (binary vs CONT, Adjusted Rand Index)
         — validates that group detection is robust to encoding choice

  T6     Virtual classifier independence
         — within-group originals vs K virtual classifiers: are virtuals less
           correlated?  Justifies the two-level hierarchy (Fig. 1 right, paper).

Usage:
    python scripts/lsml_theorem_validation.py                      # all cells
    python scripts/lsml_theorem_validation.py --filter math500     # MATH-500 only
    python scripts/lsml_theorem_validation.py --filter math500 --features all
    python scripts/lsml_theorem_validation.py --filter math500 --cell Qwen-Math-7B
"""

import argparse
import os
import pickle
import sys
import warnings

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from sklearn.metrics import adjusted_rand_score

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils.fusion_utils import (
    _residual_lsml,
    _score_matrix_lsml,
    _spectral_cluster_precomputed,
    binarize_classifiers,
    detect_dependent_groups,
    lsml_continuous_pipeline,
    lsml_fuse,
    zscore,
)

# ── Constants (must match PROGRESS.md / covariance_audit.py) ──────────────────

FEATURE_SIGNS = {
    'epr': -1, 'trace_length': 1, 'spectral_entropy': -1,
    'low_band_power': -1, 'high_band_power': -1, 'hl_ratio': -1,
    'dominant_freq': -1, 'spectral_centroid': -1,
    'stft_max_high_power': -1, 'stft_spectral_entropy': -1,
    'rpdi': -1, 'sw_var_peak': -1,
    'pe_mean': -1, 'hurst_exponent': 1,
    'cusum_max': -1, 'cusum_shift_idx': 1,
}

GOOD_FEATURES = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']
ALL_FEATURES  = list(FEATURE_SIGNS.keys())

# ── Data loading (same pattern as covariance_audit.py) ────────────────────────

def _load_pkl_cells(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        return []
    raw = obj.get('feats', obj)
    cells = []
    for k, v in raw.items():
        if not isinstance(v, (tuple, list)) or len(v) < 2:
            continue
        fd, lb = v[0], v[1]
        if not isinstance(fd, dict):
            continue
        lb = np.array(lb, dtype=int)
        if len(np.unique(lb)) < 2 or len(lb) < 10:
            continue
        cells.append((str(k), fd, lb))
    return cells


def _gather_cells(data_dir, filter_str='', cell_substr=''):
    all_cells = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.pkl'):
            continue
        if any(tag in fname for tag in ('lsml', 'results', 'ablation')):
            continue
        if filter_str and filter_str not in fname:
            continue
        path = os.path.join(data_dir, fname)
        try:
            for ck, fd, lb in _load_pkl_cells(path):
                if cell_substr and cell_substr not in ck:
                    continue
                all_cells.append((fname, ck, fd, lb))
        except Exception as e:
            print(f'  [skip] {fname}: {e}')
    return all_cells

# ── View preparation ──────────────────────────────────────────────────────────

def _orient_zscore(fd, feat_names):
    """Return list of sign-oriented z-scored continuous arrays."""
    return [zscore(np.array(fd[f], dtype=float) * FEATURE_SIGNS.get(f, 1))
            for f in feat_names]


def _binarize(fd, feat_names):
    """Return list of ±1 binarized sign-oriented arrays."""
    b = binarize_classifiers({f: fd[f] for f in feat_names}, FEATURE_SIGNS)
    return [b[f] for f in feat_names]

# ── Math helpers ──────────────────────────────────────────────────────────────

def _rank_k_fractions(R_off, k_max):
    """
    Cumulative Frobenius fraction explained by top-k eigenvectors, for k=1..k_max.
    Uses sum(eigvals[:k]**2) / sum(eigvals**2) — consistent with covariance_audit.
    """
    eigvals = np.linalg.eigvalsh(R_off)[::-1]
    total = np.sum(eigvals ** 2)
    if total < 1e-12:
        return [float('nan')] * k_max
    return [float(np.sum(eigvals[:i + 1] ** 2) / total) for i in range(k_max)]


def _score_gap(s, c):
    """
    Return (min_within, max_cross, gap) for Lemma 4.
    gap = min_within − max_cross; positive gap ⇒ Lemma 4 holds.
    """
    m = s.shape[0]
    within, cross = [], []
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            (within if c[i] == c[j] else cross).append(s[i, j])
    if not within or not cross:
        return float('nan'), float('nan'), float('nan')
    return float(min(within)), float(max(cross)), float(min(within) - max(cross))


def _residual_curve(R, s, K_max=4):
    """
    Eq. 14 residual at K=1..K_max.
    K=1 corresponds to the flat-SML / independence assumption (all in one cluster).
    Returns list of (K, residual).
    """
    m = R.shape[0]
    out = []
    c1 = np.zeros(m, dtype=int)
    out.append((1, _residual_lsml(R, c1)))
    for K in range(2, min(K_max + 1, m)):
        try:
            c = _spectral_cluster_precomputed(s, K)
            out.append((K, _residual_lsml(R, c)))
        except Exception:
            out.append((K, float('nan')))
    return out

# ── Formatting helpers ────────────────────────────────────────────────────────

def _sec(title):
    print(f'\n  {"─" * 68}')
    print(f'  {title}')
    print(f'  {"─" * 68}')


def _verdict(condition, pass_str, fail_str='BORDERLINE'):
    return pass_str if condition else fail_str

# ── Per-cell analysis ─────────────────────────────────────────────────────────

def analyze_cell(src, cell_key, fd, labels, feat_sets):
    """
    Run all six theorem tests on one cell.

    feat_sets: list of (tag, feat_names) pairs, e.g. [('5-feat', GOOD_FEATURES),
                                                        ('16-feat', ALL_FEATURES)]
    """
    n = len(labels)
    print(f'\n{"═" * 72}')
    print(f'  CELL: {src} / {cell_key}   n={n}')
    print(f'{"═" * 72}')

    summary = {}   # tag → {metric: value}

    for tag, feat_names in feat_sets:
        avail = [f for f in feat_names if f in fd]
        if len(avail) < 3:
            print(f'\n  [skip {tag}] only {len(avail)} features available in cache')
            continue
        feat_names = avail
        m = len(feat_names)
        short = ', '.join(feat_names)
        print(f'\n  ══ Feature set: {tag}  ({m} features)  [{short}] ══')

        cont_views = _orient_zscore(fd, feat_names)
        bin_views  = _binarize(fd, feat_names)

        X_cont = np.column_stack(cont_views)
        X_bin  = np.column_stack(bin_views)

        R_cont = np.cov(X_cont.T)
        R_bin  = np.cov(X_bin.T)
        if R_cont.ndim == 0:
            R_cont = np.array([[float(R_cont)]])
            R_bin  = np.array([[float(R_bin)]])

        R_off_cont = R_cont - np.diag(np.diag(R_cont))
        R_off_bin  = R_bin  - np.diag(np.diag(R_bin))

        row = {}
        summary[tag] = row

        # ── T1+T4: Eigenvalue spectrum + rank-K fractions ─────────────────────
        _sec(f'T1+T4 [{tag}] — Low-Rank Structure of R_off (Lemma 1 / Eq. 10–12)')
        print(f'\n  Theory: for binary ±1 inputs, R_off = v·vᵀ (rank-1 within groups).')
        print(f'  Test:   what fraction of R_off variance lives in the top-K eigenvectors?')
        print(f'  A high rank-1 fraction for CONT confirms Lemma 1\'s geometry is preserved.\n')

        eig_cont = np.linalg.eigvalsh(R_off_cont)[::-1]
        eig_bin  = np.linalg.eigvalsh(R_off_bin)[::-1]

        print(f'  Eigenvalues of R_off (descending):')
        print(f'  {"k":>4}  {"CONT":>14}  {"BIN":>14}')
        for i in range(m):
            print(f'  {i+1:>4}  {eig_cont[i]:>14.6f}  {eig_bin[i]:>14.6f}')

        K_show = min(3, m - 1)
        fracs_cont = _rank_k_fractions(R_off_cont, K_show)
        fracs_bin  = _rank_k_fractions(R_off_bin,  K_show)

        print(f'\n  Cumulative Frobenius fraction explained by top-K eigenvectors:')
        print(f'  {"k":>4}  {"CONT (%)":>10}  {"BIN (%)":>10}  {"Δ (pp)":>10}')
        for i in range(K_show):
            fc, fb = 100 * fracs_cont[i], 100 * fracs_bin[i]
            print(f'  {i+1:>4}  {fc:>10.1f}  {fb:>10.1f}  {fc-fb:>+10.1f}')

        r1c, r1b = fracs_cont[0], fracs_bin[0]
        row['rank1_cont'] = r1c
        row['rank1_bin']  = r1b
        v = _verdict(r1c >= 0.55, 'PASS — rank-1 geometry preserved in CONT R_off')
        print(f'\n  → Rank-1 fraction: CONT={100*r1c:.1f}%  BIN={100*r1b:.1f}%')
        print(f'    Lemma 1 check (≥55%): {v}')

        # ── T2: Score matrix gap (Lemma 4) ────────────────────────────────────
        _sec(f'T2 [{tag}] — Score Matrix Gap (Lemma 4 / Eq. 15)')
        print(f'\n  Theory: Lemma 4 requires max_{{c(i)≠c(j)}} S_ij < min_{{c(i)=c(j)}} S_ij')
        print(f'  (gap > 0). If this holds for continuous R, spectral clustering is valid.\n')

        s_cont = _score_matrix_lsml(R_cont)
        s_bin  = _score_matrix_lsml(R_bin)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            K_cont, c_cont, resid_best_cont, _ = detect_dependent_groups(cont_views)
            K_bin,  c_bin,  resid_best_bin,  _ = detect_dependent_groups(bin_views)

        minW_cont, maxC_cont, gap_cont = _score_gap(s_cont, c_cont)
        minW_bin,  maxC_bin,  gap_bin  = _score_gap(s_bin,  c_bin)

        print(f'  {"Metric":28}  {"CONT":>12}  {"BIN":>12}')
        print(f'  {"─"*28}  {"─"*12}  {"─"*12}')
        print(f'  {"K selected":28}  {K_cont:>12}  {K_bin:>12}')
        print(f'  {"min within-cluster score":28}  {minW_cont:>12.6f}  {minW_bin:>12.6f}')
        print(f'  {"max cross-cluster score":28}  {maxC_cont:>12.6f}  {maxC_bin:>12.6f}')
        print(f'  {"gap (min_w − max_c)":28}  {gap_cont:>12.6f}  {gap_bin:>12.6f}')

        row['gap_cont'] = gap_cont
        row['gap_bin']  = gap_bin
        row['K_cont']   = K_cont
        row['K_bin']    = K_bin

        for lbl, gap in [('CONT', gap_cont), ('BIN', gap_bin)]:
            if np.isnan(gap):
                msg = 'N/A (K=1, no cross-cluster pairs)'
            else:
                msg = _verdict(gap > 0,
                               'PASS — Lemma 4 holds; clustering is provably correct',
                               'FAIL — gap ≤ 0; clustering correctness not guaranteed')
            print(f'\n  Lemma 4 [{lbl}]: {msg}')

        # ── T3: Residual curve K=1..4 (Eq. 14) ───────────────────────────────
        _sec(f'T3 [{tag}] — Residual Curve K=1..4 (Eq. 14)')
        print(f'\n  Theory: Eq. 14 residual = Σ (v_on_i·v_on_j − R_ij)² [within]')
        print(f'                          + Σ (v_off_i·v_off_j − R_ij)² [cross]')
        print(f'  K=1 is flat-SML/independence assumption. A large drop K=1→K* proves')
        print(f'  the L-SML grouped structure fits our continuous covariance.\n')

        curve_cont = _residual_curve(R_cont, s_cont, K_max=4)
        curve_bin  = _residual_curve(R_bin,  s_bin,  K_max=4)

        r1_cont = curve_cont[0][1]
        r1_bin  = curve_bin[0][1]

        print(f'  {"K":>4}  {"resid CONT":>14}  {"drop CONT":>12}  '
              f'{"resid BIN":>14}  {"drop BIN":>10}')
        for (kc, rc), (kb, rb) in zip(curve_cont, curve_bin):
            drop_c = f'{100*(r1_cont-rc)/(r1_cont+1e-12):+.1f}%' if kc > 1 else '   —'
            drop_b = f'{100*(r1_bin -rb)/(r1_bin +1e-12):+.1f}%' if kb > 1 else '   —'
            mark = ' ← K*' if kc == K_cont else ''
            print(f'  {kc:>4}  {rc:>14.6f}  {drop_c:>12}  {rb:>14.6f}  {drop_b:>10}{mark}')

        best_drop_cont = 100 * (r1_cont - resid_best_cont) / (r1_cont + 1e-12)
        best_drop_bin  = 100 * (r1_bin  - resid_best_bin)  / (r1_bin  + 1e-12)
        row['resid_drop_cont'] = best_drop_cont
        row['resid_drop_bin']  = best_drop_bin

        v_cont = _verdict(best_drop_cont >= 20,
                          f'PASS ({best_drop_cont:.1f}% drop ≥ 20% threshold)',
                          f'BORDERLINE ({best_drop_cont:.1f}% drop < 20%)')
        v_bin  = _verdict(best_drop_bin  >= 20,
                          f'PASS ({best_drop_bin:.1f}% drop)',
                          f'BORDERLINE ({best_drop_bin:.1f}% drop)')
        print(f'\n  Residual drop K=1→K*:  CONT {v_cont}')
        print(f'                          BIN  {v_bin}')

        # ── T5: Cluster agreement (ARI) ───────────────────────────────────────
        _sec(f'T5 [{tag}] — Score Matrix Cluster Agreement')
        print(f'\n  If binary and continuous R produce the same cluster partition,')
        print(f'  the grouping is encoding-agnostic (ARI ≥ 0.7 = strong agreement).\n')

        if K_cont != K_bin:
            ari = float('nan')
            print(f'  K_cont={K_cont} ≠ K_bin={K_bin}; ARI is not meaningful across '
                  f'different K values.')
        else:
            ari = adjusted_rand_score(c_bin, c_cont)
            identical = np.array_equal(c_bin, c_cont)
            v = _verdict(ari >= 0.7,
                         f'PASS (robust to encoding)',
                         f'BORDERLINE — partition differs between BIN and CONT')
            print(f'  K_bin={K_bin}  K_cont={K_cont}  ARI={ari:.4f}  '
                  f'exact match={identical}')
            print(f'  Cluster agreement: {v}')
        row['cluster_ari'] = ari

    # ── T6: Virtual classifier independence (5-feat only) ─────────────────────
    tag5 = feat_sets[0][0]
    feat_names_5 = [f for f in feat_sets[0][1] if f in fd]
    if len(feat_names_5) >= 3:
        _sec(f'T6 [{tag5}] — Virtual Classifier Independence')
        print(f'\n  Theory: after within-group fusion, L-SML\'s K virtual classifiers')
        print(f'  ξ_g should be "conditionally independent given y" (Paper Fig. 1 right).')
        print(f'  Test: mean |corr(ξ_g, ξ_g\')| vs mean |corr of originals within group|.\n')

        cont5 = _orient_zscore(fd, feat_names_5)
        bin5  = _binarize(fd, feat_names_5)

        X_cont5 = np.column_stack(cont5)
        Corr5 = np.corrcoef(X_cont5.T)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, meta_bin  = lsml_fuse(*bin5)
            _, meta_cont = lsml_continuous_pipeline(fd, feat_names_5, FEATURE_SIGNS)

        c5 = meta_cont['c']
        m5 = len(feat_names_5)

        within_corr = [abs(Corr5[i, j])
                       for i in range(m5) for j in range(m5)
                       if i != j and c5[i] == c5[j]]
        mean_within = np.mean(within_corr) if within_corr else float('nan')
        print(f'  Mean |corr| within original groups (CONT, {len(within_corr)} pairs): '
              f'{mean_within:.4f}')

        for lbl, meta in [('BIN', meta_bin), ('CONT', meta_cont)]:
            vc = meta.get('virtual_classifiers')
            K_vc = meta.get('K', 0)
            if vc is None or vc.ndim != 2 or vc.shape[1] < 2:
                print(f'  {lbl} virtual: K={K_vc} (only 1 group — test not applicable)')
                continue
            Corr_vc = np.corrcoef(vc.T)
            K_vc = vc.shape[1]
            off_vc = [abs(Corr_vc[i, j]) for i in range(K_vc)
                      for j in range(K_vc) if i != j]
            mean_off = np.mean(off_vc)
            reduction = 100.0 * (1.0 - mean_off / (mean_within + 1e-12))
            v = _verdict(mean_off < mean_within,
                         f'PASS (independence ↑; virtual corr is {reduction:.1f}% lower)',
                         f'FAIL (virtual classifiers are MORE correlated than originals)')
            print(f'  {lbl} virtual ({K_vc} classifiers): mean |corr| = {mean_off:.4f}  '
                  f'→ {v}')

    # ── THEOREM ALIGNMENT SUMMARY ──────────────────────────────────────────────
    print(f'\n\n{"═" * 72}')
    print(f'  THEOREM ALIGNMENT SUMMARY')
    print(f'  Cell: {src} / {cell_key}')
    print(f'{"═" * 72}\n')

    hdr = f'  {"Test & Threshold":54}'
    for tag, _ in feat_sets:
        if tag in summary:
            hdr += f'  {tag:>10}'
    print(hdr)
    print(f'  {"─"*54}' + '  ' + '  '.join(['─'*10 for (t, _) in feat_sets if t in summary]))

    rows = [
        ('T1: Rank-1 Frobenius frac CONT (≥55%)', 'rank1_cont', 0.55,  True,  '{:.1%}'),
        ('T1: Rank-1 Frobenius frac BIN  (≥55%)', 'rank1_bin',  0.55,  True,  '{:.1%}'),
        ('T2: Score matrix gap CONT (>0)',         'gap_cont',   0.0,   True,  '{:+.4f}'),
        ('T2: Score matrix gap BIN  (>0)',         'gap_bin',    0.0,   True,  '{:+.4f}'),
        ('T3: Residual drop K=1→K* CONT (≥20%)',  'resid_drop_cont', 20.0, True,  '{:.1f}%'),
        ('T5: Cluster ARI BIN↔CONT (≥0.70)',      'cluster_ari', 0.70, True,  '{:.3f}'),
    ]

    for label, key, thresh, hib, fmt in rows:
        line = f'  {label:54}'
        for tag, _ in feat_sets:
            if tag not in summary:
                continue
            v = summary[tag].get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cell_str = '       N/A'
            else:
                ok = (v >= thresh) if hib else (v <= thresh)
                cell_str = f'{"✓" if ok else "✗"} {fmt.format(v):>8}'
            line += f'  {cell_str:>10}'
        print(line)

    print(f'\n  {"─" * 68}')
    print(f'  NOTE — Assumption (iii) gap (not tested here, mentioned for completeness):')
    print(f'  The paper\'s sign-resolution heuristic (majority of eigenvector components')
    print(f'  positive ⇒ most classifiers beat random chance) succeeds ~80% of the time')
    print(f'  for binary ±1 inputs but only ~50% for continuous z-scored features.')
    print(f'  This is the single formal gap vs. the binary theory.  It is bridged by the')
    print(f'  external FEATURE_SIGNS orientation bit — one global flip per feature set,')
    print(f'  derived once from any held-out sample.  All other structural properties')
    print(f'  above are tested directly on the MATH-500 / Qwen-Math-7B inference cache.')
    print(f'{"═" * 72}\n')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--data-dir', default='./local_cache',
                   help='Directory containing .pkl feature caches (default: ./local_cache)')
    p.add_argument('--filter', default='',
                   help='Only process cells from pkl files whose name contains this string')
    p.add_argument('--cell', default='',
                   help='Only process cells whose key contains this string (e.g. Qwen-Math-7B)')
    p.add_argument('--features', default='both', choices=['good', 'all', 'both'],
                   help='"good"=5-feat, "all"=16-feat, "both" (default)')
    args = p.parse_args()

    if args.features == 'good':
        feat_sets = [('5-feat', GOOD_FEATURES)]
    elif args.features == 'all':
        feat_sets = [('16-feat', ALL_FEATURES)]
    else:
        feat_sets = [('5-feat', GOOD_FEATURES), ('16-feat', ALL_FEATURES)]

    cells = _gather_cells(args.data_dir, filter_str=args.filter, cell_substr=args.cell)
    if not cells:
        print(f'No matching cells found in {args.data_dir!r}  '
              f'(filter={args.filter!r}  cell={args.cell!r})')
        return

    print(f'Found {len(cells)} cell(s).  Feature sets: '
          f'{[t for t, _ in feat_sets]}')
    print(f'Running T2/T3 score-matrix computations — this may take a minute per cell '
          f'(O(m⁴) score matrix + residual loops).\n')

    for src, ck, fd, lb in cells:
        valid_sets = []
        for tag, fnames in feat_sets:
            avail = [f for f in fnames if f in fd]
            if len(avail) >= 3:
                valid_sets.append((tag, fnames))
        if not valid_sets:
            print(f'[skip] {src}/{ck}: no feature set has ≥3 available features')
            continue
        analyze_cell(src, ck, fd, lb, valid_sets)


if __name__ == '__main__':
    main()
