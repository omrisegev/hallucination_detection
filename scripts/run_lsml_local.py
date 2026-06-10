"""
run_lsml_local.py — L-SML v2 fusion on locally downloaded feature pkls.

Usage:
    python scripts/run_lsml_local.py --data-dir ./local_cache/
    python scripts/run_lsml_local.py --data-dir ./local_cache/ --label "8feat-test" --features epr low_band_power sw_var_peak cusum_max spectral_entropy hurst_exponent high_band_power hl_ratio

Download these 5 files from Drive before running:
    MyDrive/hallucination_detection/consolidated_results/math500_res.pkl
    MyDrive/hallucination_detection/consolidated_results/gsm8k_res.pkl
    MyDrive/hallucination_detection/consolidated_results/gpqa_res.pkl
    MyDrive/hallucination_detection/consolidated_results/rag_feats_all.pkl
    MyDrive/hallucination_detection/consolidated_results/qa_res.pkl
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.stats import spearmanr

# Allow running from repo root or from scripts/
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils import FEAT_NAMES, boot_auc, binarize_classifiers, lsml_fuse

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
}

DEFAULT_GOOD_FEATURES = [
    'epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy'
]

PKL_NAMES = {
    'math500': 'math500_res.pkl',
    'gsm8k':   'gsm8k_res.pkl',
    'gpqa':    'gpqa_res.pkl',
    'rag':     'rag_feats_all.pkl',
    'qa':      'qa_res.pkl',
}

RESULTS_DIR = os.path.join(REPO_DIR, 'results')
ARCHIVE_PATH = os.path.join(RESULTS_DIR, 'archive.jsonl')
LATEST_CSV   = os.path.join(RESULTS_DIR, 'latest.csv')


# ---------------------------------------------------------------------------
# Helpers (mirror of notebook)
# ---------------------------------------------------------------------------

def load_cached_feats(pkl_path):
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'feats' in obj:
        return obj['feats']
    return obj


def run_lsml_v2(fd, lbl, key_str, good_features):
    lbl = np.asarray(lbl, dtype=int)
    if len(set(lbl.tolist())) < 2:
        print(f'  [{key_str}] only one class — skip')
        return None
    n_pos = int(lbl.sum())
    n_neg = int(len(lbl) - n_pos)

    ind_aucs = {}
    raw_feats = {}  # oriented raw arrays for ρ computation
    for fn in FEAT_NAMES:
        try:
            sign = FEATURE_SIGNS.get(fn, +1)
            arr = np.array(fd[fn], dtype=float) * sign
            a, *_ = boot_auc(lbl, arr)
            ind_aucs[fn] = float(a) if not (a != a) else float('nan')
            raw_feats[fn] = arr
        except Exception:
            ind_aucs[fn] = float('nan')

    try:
        binary = binarize_classifiers(fd, FEATURE_SIGNS)
        binary_filt = {fn: binary[fn] for fn in good_features if fn in binary}
        if len(binary_filt) < 3:
            print(f'  [{key_str}] fewer than 3 features available after filter — skip')
            return None
        fused, meta = lsml_fuse(*binary_filt.values())
    except Exception as e:
        print(f'  [{key_str}] L-SML v2 error: {e}')
        return None

    p_auc, p_lo, p_hi = boot_auc(lbl,  fused)
    n_auc, n_lo, n_hi = boot_auc(lbl, -fused)
    if p_auc >= n_auc:
        v2_auc, v2_lo, v2_hi = p_auc, p_lo, p_hi
    else:
        v2_auc, v2_lo, v2_hi = n_auc, n_lo, n_hi

    print(f'  [{key_str}] N={len(lbl)} (+{n_pos}/-{n_neg}) | '
          f'L-SML={v2_auc:.3f} [{v2_lo:.3f},{v2_hi:.3f}] K={meta["K"]}')

    return {
        'n': len(lbl), 'n_pos': n_pos, 'n_neg': n_neg,
        'ind_aucs': ind_aucs,
        'raw_feats': raw_feats,
        'v2_auc': v2_auc, 'v2_lo': v2_lo, 'v2_hi': v2_hi,
        'K': meta['K'],
    }


def print_diagnostics(domain_results, good_features):
    """Print per-feature AUROC and pairwise Spearman ρ aggregated across all cells."""
    feat_aucs   = {fn: [] for fn in good_features}
    fusion_aucs = []
    best_single = []
    lift_vals   = []

    # Per-cell pairwise ρ — collect as list of matrices
    rho_accum = np.zeros((len(good_features), len(good_features)))
    rho_count = 0

    print('\n' + '=' * 70)
    print('DIAGNOSTIC: per-feature AUROC vs L-SML fusion (good_features only)')
    print('=' * 70)
    hdr = f'{"Cell":<42}' + ''.join(f'{fn[:8]:>9}' for fn in good_features) + f'{"BEST":>9}{"FUSION":>9}{"LIFT":>7}'
    print(hdr)
    print('-' * len(hdr))

    for domain, dres in domain_results.items():
        for key, res in dres.items():
            if res is None:
                continue
            aucs = [res['ind_aucs'].get(fn, float('nan')) for fn in good_features]
            v2   = res['v2_auc']
            best = max((a for a in aucs if a == a), default=float('nan'))
            lift = v2 - best
            label = f'{domain}/{key}'[:41]
            auc_str = ''.join(f'{100*a:>8.1f}%' if a == a else f'{"nan":>9}' for a in aucs)
            lift_str = f'{lift:+.3f}'
            print(f'{label:<42}{auc_str}{100*best:>8.1f}%{100*v2:>8.1f}%{lift_str:>7}')

            for i, fn in enumerate(good_features):
                if aucs[i] == aucs[i]:
                    feat_aucs[fn].append(aucs[i])
            fusion_aucs.append(v2)
            if best == best:
                best_single.append(best)
                lift_vals.append(lift)

            # Pairwise rho for this cell
            arrs = [res['raw_feats'].get(fn) for fn in good_features]
            if all(a is not None for a in arrs):
                mat = np.column_stack(arrs)
                if mat.shape[0] > 5:
                    for i in range(len(good_features)):
                        for j in range(len(good_features)):
                            if i != j:
                                r, _ = spearmanr(mat[:, i], mat[:, j])
                                rho_accum[i, j] += abs(r)
                    rho_count += 1

    print('-' * len(hdr))
    # Aggregate row
    means = [np.nanmean(feat_aucs[fn]) if feat_aucs[fn] else float('nan') for fn in good_features]
    mean_best   = np.mean(best_single) if best_single else float('nan')
    mean_fusion = np.mean(fusion_aucs) if fusion_aucs else float('nan')
    mean_lift   = np.mean(lift_vals)   if lift_vals   else float('nan')
    auc_str = ''.join(f'{100*m:>8.1f}%' if m == m else f'{"nan":>9}' for m in means)
    print(f'{"MEAN":<42}{auc_str}{100*mean_best:>8.1f}%{100*mean_fusion:>8.1f}%{mean_lift:>+7.3f}')

    # Pairwise rho matrix
    print('\n' + '=' * 70)
    print('Pairwise Spearman |rho| (mean across cells, oriented features)')
    print('Values > 0.75 -> L-SML correlation filter would discard this pair')
    print('=' * 70)
    if rho_count > 0:
        rho_mean = rho_accum / rho_count
        col_w = max(len(fn) for fn in good_features) + 2
        header = ' ' * col_w + ''.join(f'{fn:>{col_w}}' for fn in good_features)
        print(header)
        for i, fn_i in enumerate(good_features):
            row = f'{fn_i:<{col_w}}'
            for j, fn_j in enumerate(good_features):
                if i == j:
                    row += f'{"--":>{col_w}}'
                else:
                    val = rho_mean[i, j]
                    tag = '!' if val >= 0.75 else ' '
                    row += f'{val:.2f}{tag}'.rjust(col_w)
            print(row)
    else:
        print('  (not enough data for rho computation)')

    print(f'\nSummary: mean lift = {mean_lift:+.3f}  '
          f'(fusion {100*mean_fusion:.1f}% vs best-single {100*mean_best:.1f}%)')
    n_pos_lift = sum(1 for v in lift_vals if v > 0)
    n_neg_lift = sum(1 for v in lift_vals if v < 0)
    print(f'         positive lift in {n_pos_lift}/{len(lift_vals)} cells, '
          f'negative in {n_neg_lift}/{len(lift_vals)} cells')


def print_feature_summary(domain_results, domain_filter=None):
    """Transposed table: all 16 features as rows, sorted by mean AUROC.
    Shows per-cell AUROCs + flip count. domain_filter: e.g. 'math500,gsm8k'"""
    filters = [d.strip().lower() for d in domain_filter.split(',')] if domain_filter else None

    # Collect cells matching filter
    cells = []  # list of (label, ind_aucs_dict)
    for domain, dres in domain_results.items():
        if filters and domain.lower() not in filters:
            continue
        for key, res in dres.items():
            if res is None:
                continue
            short = key[:20]
            cells.append((f'{domain}/{short}', res['ind_aucs']))

    if not cells:
        print('No cells matched the domain filter.')
        return

    col_w = 7  # width per cell column
    label_w = 20

    # Header
    header = f'{"Feature":<20}' + ''.join(f'{lbl[:col_w]:>{col_w}}' for lbl, _ in cells)
    header += f'{"Flips":>{col_w}}{"Mean":>{col_w}}'
    print('\n' + '=' * len(header))
    domains_shown = domain_filter or 'all'
    print(f'All-feature AUROC table  (domain filter: {domains_shown},  {len(cells)} cells)')
    print(f'  AUROC shown after FEATURE_SIGNS orientation. <50% = wrong direction.')
    print('=' * len(header))
    print(header)
    print('-' * len(header))

    rows = []
    for fn in FEAT_NAMES:
        aucs = [ia.get(fn, float('nan')) for _, ia in cells]
        valid = [a for a in aucs if a == a]
        mean_a = np.mean(valid) if valid else float('nan')
        flips  = sum(1 for a in valid if a < 0.50)
        rows.append((fn, aucs, mean_a, flips))

    rows.sort(key=lambda r: -r[2])  # sort by mean AUROC descending

    for fn, aucs, mean_a, flips in rows:
        flip_tag = f'{flips}!' if flips > 0 else ' 0'
        auc_str = ''.join(
            f'{100*a:>{col_w}.1f}' if a == a else f'{"nan":>{col_w}}'
            for a in aucs
        )
        flip_col = f'{flip_tag:>{col_w}}'
        mean_col = f'{100*mean_a:>{col_w}.1f}' if mean_a == mean_a else f'{"nan":>{col_w}}'
        marker = ' <-- FLIPS' if flips > 0 else ''
        print(f'{fn:<20}{auc_str}{flip_col}{mean_col}{marker}')

    print('-' * len(header))
    print(f'  Sorted by mean AUROC descending. Flips = cells where AUROC < 50%.')


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_run(run_id, label, good_features, domain_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'runs'), exist_ok=True)

    # Per-domain summary for JSON (drop ind_aucs to keep it readable)
    summary = {}
    for domain, dres in domain_results.items():
        summary[domain] = {}
        for key, res in dres.items():
            if res is None:
                continue
            summary[domain][key] = {
                'auroc': round(res['v2_auc'], 4),
                'ci_low':  round(res['v2_lo'], 4),
                'ci_high': round(res['v2_hi'], 4),
                'K': res['K'],
                'n': res['n'],
                'n_pos': res['n_pos'],
                'n_neg': res['n_neg'],
            }

    record = {
        'run_id':   run_id,
        'label':    label,
        'features': good_features,
        'algorithm': 'lsml_v2',
        'results':  summary,
    }

    # 1. Timestamped JSON in runs/
    slug = label.replace(' ', '_') if label else 'run'
    json_path = os.path.join(RESULTS_DIR, 'runs', f'{run_id}_{slug}.json')
    with open(json_path, 'w') as f:
        json.dump(record, f, indent=2)
    print(f'\nSaved run -> {json_path}')

    # 2. Append to archive.jsonl
    with open(ARCHIVE_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(f'Appended -> {ARCHIVE_PATH}')

    # 3. latest.csv
    rows = []
    for domain, dres in domain_results.items():
        for key, res in dres.items():
            if res is None:
                continue
            rows.append({
                'domain': domain.upper(),
                'model_key': key,
                'auroc': round(res['v2_auc'], 4),
                'ci_low':  round(res['v2_lo'], 4),
                'ci_high': round(res['v2_hi'], 4),
                'K': res['K'],
                'n': res['n'],
                'n_pos': res['n_pos'],
                'n_neg': res['n_neg'],
                'run_id': run_id,
                'label': label,
                'features': ','.join(good_features),
            })
    rows.sort(key=lambda r: -r['auroc'])

    import csv
    with open(LATEST_CSV, 'w', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f'Saved   -> {LATEST_CSV}')

    return rows


def print_table(rows):
    print()
    print('=' * 90)
    print(f'{"Domain/Model":<50} {"AUROC":>8} {"95% CI":>18} {"K":>3} {"N":>6}')
    print('=' * 90)
    for r in rows:
        ci = f'[{r["ci_low"]:.3f},{r["ci_high"]:.3f}]'
        label = f'{r["domain"]}/{r["model_key"]}'
        print(f'{label:<50} {100*r["auroc"]:>7.1f}% {ci:>18} {r["K"]:>3} {r["n"]:>6}')
    print('=' * 90)
    print(f'Total cells: {len(rows)}  |  Beating chance (>=0.50): {sum(1 for r in rows if r["auroc"] >= 0.5)}/{len(rows)}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Run L-SML v2 fusion locally on cached feature pkls.')
    parser.add_argument('--data-dir', default='./local_cache',
                        help='Directory containing the 5 feature pkl files (default: ./local_cache)')
    parser.add_argument('--label', default='',
                        help='Human-readable label for this run (e.g. "5feat-lsml-v2")')
    parser.add_argument('--features', nargs='+', default=DEFAULT_GOOD_FEATURES,
                        help='Feature subset to fuse (default: 5-feature GOOD_FEATURES)')
    parser.add_argument('--diagnose', action='store_true',
                        help='Print per-feature AUROCs and pairwise Spearman rho; skip saving to archive')
    parser.add_argument('--feature-summary', action='store_true',
                        help='Print all-16-feature AUROC table sorted by mean; use with --domain')
    parser.add_argument('--domain', default='',
                        help='Comma-separated domain filter for --feature-summary (e.g. math500,gsm8k)')
    args = parser.parse_args()

    data_dir     = os.path.abspath(args.data_dir)
    good_features = args.features
    label        = args.label or f'{len(good_features)}feat-lsml-v2'
    run_id       = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%S')

    print(f'run_id   : {run_id}')
    print(f'label    : {label}')
    print(f'features : {good_features}')
    print(f'data_dir : {data_dir}')
    print()

    # Load feature pkls
    domain_feats = {}
    for domain, fname in PKL_NAMES.items():
        path = os.path.join(data_dir, fname)
        feats = load_cached_feats(path)
        if feats is None:
            print(f'  {domain}: MISSING ({path})')
        else:
            print(f'  {domain}: {len(feats)} keys loaded')
        domain_feats[domain] = feats

    if all(v is None for v in domain_feats.values()):
        print('\nERROR: no pkl files found. Download them from Drive first.')
        print('Expected files in', data_dir)
        for fname in PKL_NAMES.values():
            print(f'  {fname}')
        sys.exit(1)

    # Run L-SML v2 per domain
    domain_results = {}
    for domain, feats in domain_feats.items():
        if feats is None:
            domain_results[domain] = {}
            continue
        print(f'\n--- {domain.upper()} ({len(feats)} cells) ---')
        dres = {}
        for key, (fd, lbl) in feats.items():
            dres[key] = run_lsml_v2(fd, lbl, f'{domain}/{key}', good_features)
        domain_results[domain] = dres

    if args.feature_summary:
        print_feature_summary(domain_results, domain_filter=args.domain or None)
    elif args.diagnose:
        print_diagnostics(domain_results, good_features)
    else:
        rows = save_run(run_id, label, good_features, domain_results)
        print_table(rows)


if __name__ == '__main__':
    main()
