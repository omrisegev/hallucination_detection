"""
method_comparison.py — Compare 10 L-SML pipeline variants on cached feature pkls.

Answers:
  Bracha Q1: What happens without feature selection? (all-16 vs GOOD_5)
  Bracha Q2: Is there a consistent subset across datasets? (Table 3 rho + Table 1)
  Claude/Gemini dispute 1: PROD on MATH-500/Qwen-7B = 88.2% or 90.0%?
  Claude/Gemini dispute 2: vAUROC_g_cont for lsml_16_nosigns best group = 0.65-0.75 or >0.85?
  Mechanism: flat SML vs L-SML grouping; binarized vs continuous; signs vs unsupervised

Usage:
    python scripts/method_comparison.py --smoke-test
    python scripts/method_comparison.py --data-dir ./local_cache/
"""

import argparse
import csv
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.stats import spearmanr

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils import (
    FEAT_NAMES, boot_auc,
    binarize_classifiers, sml_fuse, lsml_fuse,
    lsml_continuous_pipeline, simple_average_fusion,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_SIGNS = {
    'epr': -1, 'trace_length': 1, 'spectral_entropy': -1,
    'low_band_power': -1, 'high_band_power': -1, 'hl_ratio': -1,
    'dominant_freq': -1, 'spectral_centroid': -1,
    'stft_max_high_power': -1, 'stft_spectral_entropy': -1,
    'rpdi': -1, 'sw_var_peak': -1,
    'pe_mean': -1, 'hurst_exponent': 1,
    'cusum_max': -1, 'cusum_shift_idx': 1,
    'epr_spilled': -1, 'sw_var_peak_spilled': -1,
    'cusum_max_spilled': -1, 'min_spilled': -1,
    'verb_conf': +1, 'verb_conf_1p': +1,
}

GOOD_5 = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']

# First 16 entries are H(n) features; indices 16-19 are spilled energy (if present)
ALL_H16 = FEAT_NAMES[:16]

M9_FEATURES = [
    'epr', 'cusum_max', 'sw_var_peak',
    'epr_spilled', 'cusum_max_spilled', 'min_spilled',
    'rpdi', 'dominant_freq', 'stft_max_high_power',
]

SPILLED_REQUIRED = ['epr_spilled', 'cusum_max_spilled', 'min_spilled']

PKL_NAMES = {
    'math500': 'math500_res.pkl',
    'gsm8k':   'gsm8k_res.pkl',
    'gpqa':    'gpqa_res.pkl',
    'rag':     'rag_feats_all.pkl',
    'qa':      'qa_res.pkl',
}

RESULTS_DIR = os.path.join(REPO_DIR, 'results')

# Short column names for tables (order = variant run order)
VARIANT_ORDER = [
    'flat_sml_16_signs',
    'lsml_16_nosigns',
    'lsml_5_nosigns',
    'lsml_5_signs_binary',
    'lsml_5_signs_continuous',
    'flat_sml_5_signs',
    'simple_avg_5_signs',
    'best_individual_5',
    'lsml_9_signs_binary',
    'lsml_20_signs_binary',
]

VARIANT_SHORT = {
    'flat_sml_16_signs':      'flat16',
    'lsml_16_nosigns':        'lsml16',
    'lsml_5_nosigns':         'lsml5n',
    'lsml_5_signs_binary':    'PROD',
    'lsml_5_signs_continuous':'CONT',
    'flat_sml_5_signs':       'flat5',
    'simple_avg_5_signs':     'avg5',
    'best_individual_5':      'best1',
    'lsml_9_signs_binary':    'lsml9',
    'lsml_20_signs_binary':   'lsml20',
}

# Variants that produce per-group statistics
LSML_VARIANTS = {
    'lsml_16_nosigns', 'lsml_5_nosigns',
    'lsml_5_signs_binary', 'lsml_5_signs_continuous',
    'lsml_9_signs_binary', 'lsml_20_signs_binary',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cached_feats(pkl_path):
    """Load feature pkl; handles {'feats': ...} wrapper and bare dict."""
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'feats' in obj:
        return obj['feats']
    return obj


def is_saturated(arr, threshold=0.40):
    """True if >threshold fraction of values equal the median (median binarization gives no signal)."""
    a = np.asarray(arr, dtype=float)
    return float(np.mean(a == np.median(a))) > threshold


def usable_feats(fd, feat_list, threshold=0.40):
    """Return (kept_list, dropped_list) — feat_list minus missing and saturated features."""
    kept, dropped = [], []
    for f in feat_list:
        if f not in fd:
            continue
        if is_saturated(fd[f], threshold):
            dropped.append(f)
        else:
            kept.append(f)
    return kept, dropped


def safe_auc(lbl, scores):
    """Bootstrap AUC with automatic orientation (returns max of both signs)."""
    lbl = np.asarray(lbl, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(set(lbl.tolist())) < 2:
        return float('nan'), float('nan'), float('nan')
    if np.all(scores == scores[0]):
        return 0.5, 0.5, 0.5
    p, pl, ph = boot_auc(lbl, scores)
    n, nl, nh = boot_auc(lbl, -scores)
    return (p, pl, ph) if p >= n else (n, nl, nh)


def _median_binarize(fd, feat_list):
    """Median-binarize each feature in feat_list to ±1 (no sign orientation)."""
    binary = []
    for f in feat_list:
        arr = np.array(fd[f], dtype=float)
        binary.append(np.where(arr > np.median(arr), 1.0, -1.0))
    return binary


def extract_group_stats(lbl, meta, X_binary, feat_names_list=None, is_continuous=False):
    """
    Per-group vAUROC_bin and vAUROC_cont.
    For lsml_fuse (binary): virtual_classifiers[:,g] is ±1; vAUROC_cont from X_binary[:,idx]@w.
    For lsml_continuous (is_continuous=True): virtual_classifiers[:,g] is continuous.
    """
    stats = []
    for g, (idx, w) in enumerate(meta['group_weights']):
        if is_continuous:
            score_cont = meta['virtual_classifiers'][:, g]
            auc_cont, _, _ = safe_auc(lbl, score_cont)
            auc_bin, _, _ = safe_auc(lbl, np.sign(score_cont))
        else:
            xi_bin = meta['virtual_classifiers'][:, g]
            auc_bin, _, _ = safe_auc(lbl, xi_bin)
            if X_binary is not None and len(idx) > 1:
                score_cont = X_binary[:, idx] @ w
            elif X_binary is not None:
                score_cont = X_binary[:, int(idx[0])]
            else:
                score_cont = xi_bin
            auc_cont, _, _ = safe_auc(lbl, score_cont)

        group_feat_names = []
        if feat_names_list is not None:
            group_feat_names = [feat_names_list[i] for i in idx.tolist() if i < len(feat_names_list)]

        stats.append({
            'group': g,
            'size': int(len(idx)),
            'classifier_indices': idx.tolist(),
            'feature_names': group_feat_names,
            'vAUROC_bin': round(float(auc_bin), 4),
            'vAUROC_cont': round(float(auc_cont), 4),
        })
    return stats


# ---------------------------------------------------------------------------
# Core: run all 10 variants for one (fd, lbl) cell
# ---------------------------------------------------------------------------

def run_cell(fd, lbl, cell_key):
    """Run all 10 variants. Returns dict with 'variants', 'rho_matrix', 'saturated_feats'."""
    lbl = np.asarray(lbl, dtype=int)
    n = len(lbl)
    n_pos = int(lbl.sum())

    variants = {}
    all_saturated = []

    # Pre-compute binarized features (sign-oriented, median threshold) — reused by V1/V4/V6/V9/V10
    binary_dict = binarize_classifiers(fd, FEATURE_SIGNS)

    # --- Variant 1: flat_sml_16_signs ---
    feat_list16, dropped16 = usable_feats(fd, ALL_H16)
    all_saturated.extend(dropped16)
    try:
        views1 = [binary_dict[f] for f in feat_list16 if f in binary_dict]
        if len(views1) >= 3:
            fused, _ = sml_fuse(*views1)
            a, lo, hi = safe_auc(lbl, fused)
            variants['flat_sml_16_signs'] = _vres(a, lo, hi)
        else:
            variants['flat_sml_16_signs'] = None
    except Exception as e:
        print(f'  [{cell_key}] flat_sml_16_signs: {e}')
        variants['flat_sml_16_signs'] = None

    # --- Variant 2: lsml_16_nosigns ---
    try:
        avail16 = [f in fd for f in feat_list16] # This looks wrong in the start_session read, let me check
        # Wait, the start session read said:
        # avail16 = [f for f in feat_list16 if f in fd]
        avail16 = [f for f in feat_list16 if f in fd]
        if len(avail16) >= 3:
            bin16 = _median_binarize(fd, avail16)
            X16 = np.column_stack(bin16)
            fused, meta = lsml_fuse(*bin16)
            a, lo, hi = safe_auc(lbl, fused)
            gs = extract_group_stats(lbl, meta, X16, feat_names_list=avail16)
            variants['lsml_16_nosigns'] = _vres(a, lo, hi, meta['K'], gs)
        else:
            variants['lsml_16_nosigns'] = None
    except Exception as e:
        print(f'  [{cell_key}] lsml_16_nosigns: {e}')
        variants['lsml_16_nosigns'] = None

    # --- Variant 3: lsml_5_nosigns ---
    try:
        avail5 = [f for f in GOOD_5 if f in fd]
        if len(avail5) >= 3:
            bin5n = _median_binarize(fd, avail5)
            X5n = np.column_stack(bin5n)
            fused, meta = lsml_fuse(*bin5n)
            a, lo, hi = safe_auc(lbl, fused)
            gs = extract_group_stats(lbl, meta, X5n, feat_names_list=avail5)
            variants['lsml_5_nosigns'] = _vres(a, lo, hi, meta['K'], gs)
        else:
            variants['lsml_5_nosigns'] = None
    except Exception as e:
        print(f'  [{cell_key}] lsml_5_nosigns: {e}')
        variants['lsml_5_nosigns'] = None

    # --- Variant 4: lsml_5_signs_binary [PROD] ---
    try:
        filt4 = {f: binary_dict[f] for f in GOOD_5 if f in binary_dict}
        if len(filt4) >= 3:
            names4 = list(filt4.keys())
            bl4 = list(filt4.values())
            X4 = np.column_stack(bl4)
            fused, meta = lsml_fuse(*bl4)
            a, lo, hi = safe_auc(lbl, fused)
            gs = extract_group_stats(lbl, meta, X4, feat_names_list=names4)
            variants['lsml_5_signs_binary'] = _vres(a, lo, hi, meta['K'], gs)
        else:
            variants['lsml_5_signs_binary'] = None
    except Exception as e:
        print(f'  [{cell_key}] lsml_5_signs_binary: {e}')
        variants['lsml_5_signs_binary'] = None

    # --- Variant 5: lsml_5_signs_continuous [CAND] ---
    try:
        avail5c = [f for f in GOOD_5 if f in fd]
        if len(avail5c) >= 3:
            fused, meta = lsml_continuous_pipeline(fd, avail5c, FEATURE_SIGNS)
            a, lo, hi = safe_auc(lbl, fused)
            gs = extract_group_stats(lbl, meta, None, is_continuous=True)
            variants['lsml_5_signs_continuous'] = _vres(a, lo, hi, meta['K'], gs)
        else:
            variants['lsml_5_signs_continuous'] = None
    except Exception as e:
        print(f'  [{cell_key}] lsml_5_signs_continuous: {e}')
        variants['lsml_5_signs_continuous'] = None

    # --- Variant 6: flat_sml_5_signs ---
    try:
        views6 = [binary_dict[f] for f in GOOD_5 if f in binary_dict]
        if len(views6) >= 3:
            fused, _ = sml_fuse(*views6)
            a, lo, hi = safe_auc(lbl, fused)
            variants['flat_sml_5_signs'] = _vres(a, lo, hi)
        else:
            variants['flat_sml_5_signs'] = None
    except Exception as e:
        print(f'  [{cell_key}] flat_sml_5_signs: {e}')
        variants['flat_sml_5_signs'] = None

    # --- Variant 7: simple_avg_5_signs ---
    try:
        views7 = [np.array(fd[f], dtype=float) * FEATURE_SIGNS.get(f, 1) for f in GOOD_5 if f in fd]
        if len(views7) >= 3:
            fused, _ = simple_average_fusion(*views7)
            a, lo, hi = safe_auc(lbl, fused)
            variants['simple_avg_5_signs'] = _vres(a, lo, hi)
        else:
            variants['simple_avg_5_signs'] = None
    except Exception as e:
        print(f'  [{cell_key}] simple_avg_5_signs: {e}')
        variants['simple_avg_5_signs'] = None

    # --- Variant 8: best_individual_5 ---
    try:
        best_a, best_feat = float('nan'), None
        for f in GOOD_5:
            if f not in fd:
                continue
            arr = np.array(fd[f], dtype=float) * FEATURE_SIGNS.get(f, 1)
            a, _, _ = safe_auc(lbl, arr)
            if a == a and (best_a != best_a or a > best_a):
                best_a, best_feat = a, f
        if best_feat is not None:
            res8 = _vres(best_a, float('nan'), float('nan'))
            res8['best_feat'] = best_feat
            variants['best_individual_5'] = res8
        else:
            variants['best_individual_5'] = None
    except Exception as e:
        print(f'  [{cell_key}] best_individual_5: {e}')
        variants['best_individual_5'] = None

    # --- Variant 9: lsml_9_signs_binary (skip if spilled energy absent) ---
    if not all(f in fd for f in SPILLED_REQUIRED):
        variants['lsml_9_signs_binary'] = None
    else:
        try:
            avail9 = [f for f in M9_FEATURES if f in binary_dict]
            if len(avail9) >= 3:
                bl9 = [binary_dict[f] for f in avail9]
                X9 = np.column_stack(bl9)
                fused, meta = lsml_fuse(*bl9)
                a, lo, hi = safe_auc(lbl, fused)
                gs = extract_group_stats(lbl, meta, X9, feat_names_list=avail9)
                variants['lsml_9_signs_binary'] = _vres(a, lo, hi, meta['K'], gs)
            else:
                variants['lsml_9_signs_binary'] = None
        except Exception as e:
            print(f'  [{cell_key}] lsml_9_signs_binary: {e}')
            variants['lsml_9_signs_binary'] = None

    # --- Variant 10: lsml_20_signs_binary (skip if spilled energy absent) ---
    if not all(f in fd for f in SPILLED_REQUIRED):
        variants['lsml_20_signs_binary'] = None
    else:
        try:
            feat_list20, dropped20 = usable_feats(fd, list(FEAT_NAMES))
            for f in dropped20:
                if f not in all_saturated:
                    all_saturated.append(f)
            avail20 = [f for f in feat_list20 if f in binary_dict]
            if len(avail20) >= 3:
                bl20 = [binary_dict[f] for f in avail20]
                X20 = np.column_stack(bl20)
                fused, meta = lsml_fuse(*bl20)
                a, lo, hi = safe_auc(lbl, fused)
                gs = extract_group_stats(lbl, meta, X20, feat_names_list=avail20)
                variants['lsml_20_signs_binary'] = _vres(a, lo, hi, meta['K'], gs)
            else:
                variants['lsml_20_signs_binary'] = None
        except Exception as e:
            print(f'  [{cell_key}] lsml_20_signs_binary: {e}')
            variants['lsml_20_signs_binary'] = None

    # --- Spearman rho matrix for GOOD_5 pairs ---
    rho_matrix = {}
    for i, fi in enumerate(GOOD_5):
        for j, fj in enumerate(GOOD_5):
            if j <= i or fi not in fd or fj not in fd:
                continue
            a = np.array(fd[fi], dtype=float) * FEATURE_SIGNS.get(fi, 1)
            b = np.array(fd[fj], dtype=float) * FEATURE_SIGNS.get(fj, 1)
            try:
                rho, _ = spearmanr(a, b)
                rho_matrix[(fi, fj)] = abs(float(rho))
            except Exception:
                pass

    if all_saturated:
        print(f'  [{cell_key}] saturated features dropped: {all_saturated}')

    return {
        'n': n, 'n_pos': n_pos, 'n_neg': n - n_pos,
        'variants': variants,
        'rho_matrix': rho_matrix,
        'saturated_feats': all_saturated,
    }


def _vres(auroc, ci_lo, ci_hi, K=None, group_stats=None):
    return {
        'auroc': round(float(auroc), 4) if auroc == auroc else float('nan'),
        'ci_lo': round(float(ci_lo), 4) if ci_lo == ci_lo else float('nan'),
        'ci_hi': round(float(ci_hi), 4) if ci_hi == ci_hi else float('nan'),
        'K': int(K) if K is not None else None,
        'group_stats': group_stats,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _fmt(v):
    """Format a 0–1 AUROC as XX.X% or empty string."""
    if v is None or (isinstance(v, float) and v != v):
        return ''
    return f'{100 * v:.1f}%'


def write_table1(all_results, out_path):
    """Per-cell AUROC for all 10 variants + per-domain and overall macro means."""
    domain_order = ['math500', 'gsm8k', 'gpqa', 'rag', 'qa']
    domain_cells = {d: [] for d in domain_order}

    data_rows = []
    for (domain, cell_key), res in all_results.items():
        row = {'domain': domain, 'cell_key': cell_key,
               'n': res['n'], 'n_pos': res['n_pos']}
        for vname in VARIANT_ORDER:
            vres = res['variants'].get(vname)
            row[VARIANT_SHORT[vname]] = vres['auroc'] if vres else None
        row['saturated_feats'] = ','.join(res.get('saturated_feats', []))
        data_rows.append(row)
        if domain in domain_cells:
            domain_cells[domain].append(row)

    # Per-domain means
    domain_mean_rows = []
    overall_accum = {vname: [] for vname in VARIANT_ORDER}
    for domain in domain_order:
        drows = domain_cells[domain]
        if not drows:
            continue
        mrow = {'domain': domain, 'cell_key': 'DOMAIN_MEAN', 'n': '', 'n_pos': '', 'saturated_feats': ''}
        for vname in VARIANT_ORDER:
            short = VARIANT_SHORT[vname]
            vals = [r[short] for r in drows if r[short] is not None and r[short] == r[short]]
            m = float(np.mean(vals)) if vals else None
            mrow[short] = m
            if m is not None:
                overall_accum[vname].append(m)
        domain_mean_rows.append(mrow)

    # Overall macro mean (mean of domain means — each domain = 1/5 weight)
    macro = {'domain': 'MACRO_AVG', 'cell_key': 'MEAN_OF_DOMAIN_MEANS', 'n': '', 'n_pos': '', 'saturated_feats': ''}
    for vname in VARIANT_ORDER:
        vals = overall_accum[vname]
        macro[VARIANT_SHORT[vname]] = float(np.mean(vals)) if vals else None

    fieldnames = (['domain', 'cell_key', 'n', 'n_pos'] +
                  [VARIANT_SHORT[v] for v in VARIANT_ORDER] + ['saturated_feats'])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in data_rows:
            w.writerow({**row, **{VARIANT_SHORT[v]: _fmt(row[VARIANT_SHORT[v]]) for v in VARIANT_ORDER}})
        w.writerow({k: '' for k in fieldnames})
        for row in domain_mean_rows:
            w.writerow({**row, **{VARIANT_SHORT[v]: _fmt(row[VARIANT_SHORT[v]]) for v in VARIANT_ORDER}})
        w.writerow({k: '' for k in fieldnames})
        w.writerow({**macro, **{VARIANT_SHORT[v]: _fmt(macro[VARIANT_SHORT[v]]) for v in VARIANT_ORDER}})

    print(f'Table 1 -> {out_path}')
    return domain_cells, overall_accum, macro


def write_table2(all_results, out_path):
    """Per-group stats for all L-SML variants."""
    fieldnames = ['domain', 'cell_key', 'variant', 'K', 'group_idx',
                  'feature_names', 'size', 'vAUROC_bin', 'vAUROC_cont']
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (domain, cell_key), res in all_results.items():
            for vname in VARIANT_ORDER:
                if vname not in LSML_VARIANTS:
                    continue
                vres = res['variants'].get(vname)
                if not vres or not vres.get('group_stats'):
                    continue
                for gstat in vres['group_stats']:
                    w.writerow({
                        'domain': domain, 'cell_key': cell_key,
                        'variant': vname, 'K': vres.get('K'),
                        'group_idx': gstat['group'],
                        'feature_names': ','.join(gstat.get('feature_names', [])),
                        'size': gstat['size'],
                        'vAUROC_bin': _fmt(gstat['vAUROC_bin']),
                        'vAUROC_cont': _fmt(gstat['vAUROC_cont']),
                    })
    print(f'Table 2 -> {out_path}')


def write_table3(all_results, out_path):
    """Spearman rho cross-dataset consistency for GOOD_5 pairs."""
    pair_rhos = {}
    for res in all_results.values():
        for pair, rho in res.get('rho_matrix', {}).items():
            pair_rhos.setdefault(pair, []).append(rho)

    rows = []
    for pair, rhos in pair_rhos.items():
        rows.append({
            'feature_pair': f'{pair[0]}|{pair[1]}',
            'mean_rho': round(float(np.mean(rhos)), 4),
            'max_rho': round(float(np.max(rhos)), 4),
            'n_cells_above_0.75': sum(1 for r in rhos if r >= 0.75),
        })
    rows.sort(key=lambda r: -r['mean_rho'])
    n_high = sum(1 for r in rows if r['mean_rho'] >= 0.75)

    fieldnames = ['feature_pair', 'mean_rho', 'max_rho', 'n_cells_above_0.75']
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
        w.writerow({'feature_pair': f'SUMMARY: {n_high} pairs with mean_rho >= 0.75',
                    'mean_rho': '', 'max_rho': '', 'n_cells_above_0.75': ''})
    print(f'Table 3 -> {out_path}')


def write_json(all_results, out_path):
    """Full raw results as JSON for programmatic access."""
    def _serial(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Not serializable: {type(obj)}')

    serializable = {}
    for (domain, cell_key), res in all_results.items():
        cell_out = {
            'n': res['n'], 'n_pos': res['n_pos'], 'n_neg': res['n_neg'],
            'saturated_feats': res.get('saturated_feats', []),
            'variants': {},
            'rho_matrix': {f'{k[0]}|{k[1]}': v for k, v in res.get('rho_matrix', {}).items()},
        }
        for vname, vres in res['variants'].items():
            if vres is None:
                cell_out['variants'][vname] = None
            else:
                cell_out['variants'][vname] = {k: v for k, v in vres.items()}
        serializable[f'{domain}/{cell_key}'] = cell_out

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=_serial)
    print(f'JSON   -> {out_path}')


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(all_results):
    """Stdout summary table + saturation report + 88.2% vs 90.0% check."""
    domain_order = ['math500', 'gsm8k', 'gpqa', 'rag', 'qa']
    domain_label = {'math500': 'MATH-500', 'gsm8k': 'GSM8K', 'gpqa': 'GPQA', 'rag': 'RAG', 'qa': 'QA'}

    domain_aurocs = {d: {v: [] for v in VARIANT_ORDER} for d in domain_order}
    for (domain, cell_key), res in all_results.items():
        if domain not in domain_aurocs:
            continue
        for vname in VARIANT_ORDER:
            vres = res['variants'].get(vname)
            if vres and vres.get('auroc') == vres.get('auroc'):
                domain_aurocs[domain][vname].append(vres['auroc'])

    shorts = [VARIANT_SHORT[v] for v in VARIANT_ORDER]
    col_w = 7
    dom_w = 10
    hdr = f'{"Domain":<{dom_w}}' + ''.join(f'{s:>{col_w}}' for s in shorts)
    sep = '-' * len(hdr)

    print('\n' + '=' * len(hdr))
    print('METHOD COMPARISON SUMMARY (macro-mean per domain, all cells equal weight)')
    print('=' * len(hdr))
    print(hdr)
    print(sep)

    overall = {v: [] for v in VARIANT_ORDER}
    for domain in domain_order:
        if domain not in domain_aurocs:
            continue
        dvals = domain_aurocs[domain]
        row = f'{domain_label[domain]:<{dom_w}}'
        for vname in VARIANT_ORDER:
            vals = dvals[vname]
            if vals:
                m = float(np.mean(vals))
                overall[vname].append(m)
                row += f'{100*m:{col_w}.1f}'
            else:
                row += f'{"--":>{col_w}}'
        print(row)

    print(sep)
    macro_row = f'{"MACRO AVG":<{dom_w}}'
    for vname in VARIANT_ORDER:
        vals = overall[vname]
        if vals:
            macro_row += f'{100*np.mean(vals):{col_w}.1f}'
        else:
            macro_row += f'{"--":>{col_w}}'
    print(macro_row)
    print('=' * len(hdr))

    # Saturation report
    print('\n=== SATURATION REPORT ===')
    sat_by_domain = {}
    for (domain, cell_key), res in all_results.items():
        sf = res.get('saturated_feats', [])
        if sf:
            sat_by_domain.setdefault(domain, set()).update(sf)
    if sat_by_domain:
        for domain in domain_order:
            if domain in sat_by_domain:
                print(f'  {domain}: {sorted(sat_by_domain[domain])}')
    else:
        print('  No saturated features detected.')

    # Dispute check: 88.2% vs 90.0%
    print('\n=== 88.2% vs 90.0% DISPUTE CHECK ===')
    print('PROD (lsml_5_signs_binary) on math500 cells:')
    found = False
    for (domain, cell_key), res in all_results.items():
        if domain != 'math500':
            continue
        vres = res['variants'].get('lsml_5_signs_binary')
        if vres and vres.get('auroc') == vres.get('auroc'):
            found = True
            ci_lo = vres.get('ci_lo', float('nan'))
            ci_hi = vres.get('ci_hi', float('nan'))
            ci_str = f'[{100*ci_lo:.1f},{100*ci_hi:.1f}]' if ci_lo == ci_lo else ''
            print(f'  {cell_key}: {100*vres["auroc"]:.1f}% {ci_str}')
    if not found:
        print('  (no math500 cells found in data)')
    print('  Gemini predicts 88.2%; Claude predicts 90.0%')

    # Dispute check: vAUROC_g_cont for lsml_16_nosigns best group on math500
    print('\n=== vAUROC_g_cont DISPUTE CHECK ===')
    print('lsml_16_nosigns best-group vAUROC_cont on math500 cells:')
    found2 = False
    for (domain, cell_key), res in all_results.items():
        if domain != 'math500':
            continue
        vres = res['variants'].get('lsml_16_nosigns')
        if not vres or not vres.get('group_stats'):
            continue
        found2 = True
        best_g = max(vres['group_stats'], key=lambda g: g['vAUROC_cont'])
        print(f'  {cell_key}: best group {best_g["group"]} vAUROC_cont={100*best_g["vAUROC_cont"]:.1f}%'
              f' (size={best_g["size"]} feats={best_g["feature_names"]})')
    if not found2:
        print('  (no math500/lsml_16_nosigns results)')
    print('  Gemini predicts 65-75%; Claude predicts >85%')


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test():
    rng = np.random.default_rng(42)
    n = 60
    fd_synth = {f: rng.standard_normal(n).tolist() for f in ALL_H16}
    # Force trace_length saturation: need >50% at the same value so median = 2048.0
    fd_synth['trace_length'] = ([2048.0] * 35) + rng.standard_normal(n - 35).tolist()
    lbl_synth = (rng.random(n) > 0.4).astype(int)

    print('=== SMOKE TEST ===')
    print(f'Synthetic: n={n}, n_pos={int(lbl_synth.sum())}, features={len(fd_synth)}')
    result = run_cell(fd_synth, lbl_synth, 'smoke_test')

    print(f'\nSaturated features detected: {result["saturated_feats"]}')
    print()
    for vname in VARIANT_ORDER:
        vres = result['variants'].get(vname)
        if vres is None:
            status = 'skipped (expected: spilled energy absent)'
        else:
            status = f"auroc={vres['auroc']:.3f}"
            if vres.get('K') is not None:
                status += f" K={vres['K']}"
            if vres.get('group_stats'):
                best = max(vres['group_stats'], key=lambda g: g['vAUROC_cont'])
                status += f" groups={len(vres['group_stats'])} best_cont={best['vAUROC_cont']:.3f}"
        print(f'  {vname:<35} -> {status}')

    # Basic correctness checks
    must_run = ['lsml_5_signs_binary', 'lsml_5_signs_continuous', 'flat_sml_5_signs',
                'simple_avg_5_signs', 'best_individual_5', 'lsml_5_nosigns',
                'flat_sml_16_signs', 'lsml_16_nosigns']
    must_skip = ['lsml_9_signs_binary', 'lsml_20_signs_binary']  # no spilled feats in synthetic
    for v in must_run:
        assert result['variants'].get(v) is not None, f'FAIL: {v} returned None'
    for v in must_skip:
        assert result['variants'].get(v) is None, f'FAIL: {v} should be skipped'
    assert 'trace_length' in result['saturated_feats'], 'FAIL: trace_length saturation not detected'

    print('\nAll assertions passed. Smoke test OK.')
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare 10 L-SML pipeline variants on cached feature pkls.')
    parser.add_argument('--data-dir', default='./local_cache',
                        help='Directory with the 5 pkl files (default: ./local_cache)')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run on synthetic data and exit (no Drive files needed)')
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()

    data_dir = os.path.abspath(args.data_dir)
    run_id = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%S')
    print(f'run_id   : {run_id}')
    print(f'data_dir : {data_dir}')
    print()

    all_results = {}
    for domain, pkl_name in PKL_NAMES.items():
        path = os.path.join(data_dir, pkl_name)
        feats = load_cached_feats(path)
        if feats is None:
            print(f'[MISSING] {pkl_name}')
            continue
        print(f'\n--- {domain.upper()} ({len(feats)} cells) ---')
        for cell_key, payload in feats.items():
            fd, lbl = payload
            lbl_arr = np.asarray(lbl, dtype=int)
            if len(set(lbl_arr.tolist())) < 2:
                print(f'  [{cell_key}] single class — skip')
                continue
            print(f'  [{cell_key}] N={len(lbl_arr)} (+{int(lbl_arr.sum())}/-{int(len(lbl_arr)-lbl_arr.sum())})', end=' ')
            result = run_cell(fd, lbl_arr, cell_key)
            prod = result['variants'].get('lsml_5_signs_binary')
            if prod:
                print(f'PROD={100*prod["auroc"]:.1f}%')
            else:
                print()
            all_results[(domain, cell_key)] = result

    if not all_results:
        print('\nNo results — download pkl files to', data_dir)
        print('Required:', list(PKL_NAMES.values()))
        sys.exit(1)

    runs_dir = os.path.join(RESULTS_DIR, 'runs')
    os.makedirs(runs_dir, exist_ok=True)

    write_table1(all_results, os.path.join(RESULTS_DIR, 'method_comparison_table1.csv'))
    write_table2(all_results, os.path.join(RESULTS_DIR, 'method_comparison_table2.csv'))
    write_table3(all_results, os.path.join(RESULTS_DIR, 'method_comparison_table3.csv'))
    write_json(all_results, os.path.join(runs_dir, f'{run_id}_method_comparison.json'))
    print_summary(all_results)


if __name__ == '__main__':
    main()
