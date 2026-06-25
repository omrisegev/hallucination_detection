"""
Compare CONT (continuous L-SML) vs U-PCR fusion across 5 / 9 / 16 feature sets.

Usage:
    python scripts/run_upcr_comparison.py [--data-dir ./local_cache] [--smoke-test]

Reads:  local_cache/{math500_res,gsm8k_res,gpqa_res,qa_res,rag_feats_all}.pkl
        Schema per pkl: {cell_key: (feat_dict, labels)}
Writes: results/upcr_comparison.pkl
"""

import os
import sys
import pickle
import warnings
import argparse
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils import (
    lsml_continuous_pipeline,
    upcr_pipeline,
    boot_auc,
    FEAT_NAMES,
)


def safe_auc(lbl, scores):
    """AUC taking the better of both sign orientations (matches method_comparison.py)."""
    lbl = np.asarray(lbl, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(set(lbl.tolist())) < 2 or np.all(scores == scores[0]):
        return 0.5, 0.5, 0.5
    p, pl, ph = boot_auc(lbl, scores)
    n, nl, nh = boot_auc(lbl, -scores)
    return (p, pl, ph) if p >= n else (n, nl, nh)

# ── Feature sets (mirrors method_comparison.py definitions) ──────────────────

GOOD_5 = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']

STABLE_H9 = [
    'epr', 'low_band_power', 'high_band_power', 'hl_ratio',
    'spectral_centroid', 'sw_var_peak', 'rpdi', 'pe_mean', 'cusum_max',
]

ALL_H16 = FEAT_NAMES[:16]   # epr … cusum_shift_idx

FEATURE_SETS = {'5': GOOD_5, '9': STABLE_H9, '16': ALL_H16}

FEATURE_SIGNS = {
    'epr': -1, 'trace_length': 1, 'spectral_entropy': -1,
    'low_band_power': -1, 'high_band_power': -1, 'hl_ratio': -1,
    'dominant_freq': -1, 'spectral_centroid': -1,
    'stft_max_high_power': -1, 'stft_spectral_entropy': -1,
    'rpdi': -1, 'sw_var_peak': -1,
    'pe_mean': -1, 'hurst_exponent': 1,
    'cusum_max': -1, 'cusum_shift_idx': 1,
}

PKL_NAMES = {
    'math500': 'math500_res.pkl',
    'gsm8k':   'gsm8k_res.pkl',
    'gpqa':    'gpqa_res.pkl',
    'rag':     'rag_feats_all.pkl',
    'qa':      'qa_res.pkl',
}

OUT_PATH = os.path.join(REPO_DIR, 'results', 'upcr_comparison.pkl')

# ── Data loading (matches method_comparison.py) ───────────────────────────────

def load_cached_feats(pkl_path):
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'feats' in obj:
        return obj['feats']
    return obj


def is_saturated(arr, threshold=0.40):
    a = np.asarray(arr, dtype=float)
    return float(np.mean(a == np.median(a))) > threshold

# ── Per-cell runner ───────────────────────────────────────────────────────────

def run_cell(cell_name, fd, labels):
    """Run CONT and U-PCR for all three feature sets on one cell."""
    labels = np.array(labels, dtype=float)
    p = float(np.mean(labels))
    var_y = max(p * (1.0 - p), 0.05)

    cell_res = {'cell': cell_name, 'n': len(labels), 'prevalence': p, 'var_y': var_y}

    for fs_name, feat_list in FEATURE_SETS.items():
        # Keep only features present in fd and not saturated
        available = [
            f for f in feat_list
            if f in fd and fd[f] is not None
            and len(fd[f]) == len(labels)
            and not is_saturated(fd[f])
        ]
        cell_res[f'avail_{fs_name}'] = available
        n_avail = len(available)

        if n_avail < 3:
            cell_res[f'cont_{fs_name}'] = None
            cell_res[f'upcr_{fs_name}'] = None
            cell_res[f'delta_{fs_name}'] = None
            continue

        feat_signs = {f: FEATURE_SIGNS.get(f, -1) for f in available}

        # CONT
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                cont_score, _ = lsml_continuous_pipeline(fd, available, feat_signs)
                cont_auc, _, _ = safe_auc(labels, cont_score)
            except Exception:
                cont_auc = None

        # U-PCR
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                upcr_score, w, rho_hat, g2_hat = upcr_pipeline(
                    fd, available, feat_signs, var_y=var_y
                )
                upcr_auc, _, _ = safe_auc(labels, upcr_score)
            except Exception:
                upcr_auc = None
                w = rho_hat = g2_hat = None

        delta = (upcr_auc - cont_auc) if (cont_auc is not None and upcr_auc is not None) else None
        cell_res[f'cont_{fs_name}'] = cont_auc
        cell_res[f'upcr_{fs_name}'] = upcr_auc
        cell_res[f'delta_{fs_name}'] = delta
        if w is not None:
            cell_res[f'upcr_weights_{fs_name}'] = w.tolist()
            cell_res[f'upcr_rho_{fs_name}'] = rho_hat.tolist()
            cell_res[f'upcr_g2_{fs_name}'] = float(g2_hat)

    return cell_res

# ── Table printer ─────────────────────────────────────────────────────────────

def print_table(results):
    cols = ['5', '9', '16']
    hdr = (f"{'Cell':<36} | "
           + ' | '.join(f"{'CONT'+k:>8} {'UPCR'+k:>8} {'d'+k:>6}" for k in cols))
    sep = '-' * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    accum = {f'cont_{k}': [] for k in cols}
    accum.update({f'upcr_{k}': [] for k in cols})

    for r in results:
        if r.get('skipped'):
            continue
        parts = []
        for k in cols:
            c = r.get(f'cont_{k}')
            u = r.get(f'upcr_{k}')
            d = r.get(f'delta_{k}')
            cs = f'{100*c:.1f}%' if c is not None else '   N/A'
            us = f'{100*u:.1f}%' if u is not None else '   N/A'
            ds = f'{100*d:+.1f}' if d is not None else '   N/A'
            parts.append(f'{cs:>8} {us:>8} {ds:>6}')
            if c is not None: accum[f'cont_{k}'].append(c)
            if u is not None: accum[f'upcr_{k}'].append(u)
        print(f"  {r['cell']:<34} | " + ' | '.join(parts))

    print(sep)
    macro_parts = []
    for k in cols:
        mc = np.mean(accum[f'cont_{k}']) if accum[f'cont_{k}'] else None
        mu = np.mean(accum[f'upcr_{k}']) if accum[f'upcr_{k}'] else None
        md = (mu - mc) if (mc is not None and mu is not None) else None
        cs = f'{100*mc:.1f}%' if mc is not None else '   N/A'
        us = f'{100*mu:.1f}%' if mu is not None else '   N/A'
        ds = f'{100*md:+.1f}' if md is not None else '   N/A'
        macro_parts.append(f'{cs:>8} {us:>8} {ds:>6}')
    print(f"  {'MACRO':<34} | " + ' | '.join(macro_parts))
    print(sep)

# ── Smoke test ────────────────────────────────────────────────────────────────

def run_smoke_test():
    rng = np.random.default_rng(42)
    n = 80
    fd = {f: rng.standard_normal(n).tolist() for f in ALL_H16}
    lbl = (rng.random(n) > 0.45).astype(int)
    print('=== SMOKE TEST ===')
    r = run_cell('smoke_test', fd, lbl)
    for k in ['5', '9', '16']:
        c = r.get(f'cont_{k}'); u = r.get(f'upcr_{k}')
        avail = r.get(f'avail_{k}', [])
        cs = f'{c:.3f}' if c is not None else 'N/A'
        us = f'{u:.3f}' if u is not None else 'N/A'
        print(f'  feat{k}: avail={len(avail)} cont={cs} upcr={us}')
    print('Smoke test OK.')
    sys.exit(0)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./local_cache')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()

    data_dir = os.path.abspath(args.data_dir)
    print(f'data_dir: {data_dir}\n')

    all_results = []
    for domain, pkl_name in PKL_NAMES.items():
        path = os.path.join(data_dir, pkl_name)
        feats = load_cached_feats(path)
        if feats is None:
            print(f'[MISSING] {pkl_name}')
            continue
        print(f'\n--- {domain.upper()} ({len(feats)} cells) ---')
        for cell_key, payload in feats.items():
            if isinstance(payload, (list, tuple)) and len(payload) == 2:
                fd, lbl = payload
            elif isinstance(payload, dict) and 'feats' in payload:
                fd, lbl = payload['feats'], payload['labels']
            else:
                print(f'  [{cell_key}] unknown schema — skip')
                all_results.append({'cell': f'{domain}/{cell_key}', 'skipped': True})
                continue
            lbl_arr = np.asarray(lbl, dtype=int)
            if len(set(lbl_arr.tolist())) < 2:
                print(f'  [{cell_key}] single class — skip')
                continue
            r = run_cell(f'{domain}/{cell_key}', fd, lbl_arr)
            all_results.append(r)
            c5 = r.get('cont_5'); u5 = r.get('upcr_5')
            status = (f"CONT-5={100*c5:.1f}% U-PCR-5={100*u5:.1f}%"
                      if (c5 is not None and u5 is not None) else 'N/A')
            print(f'  [{cell_key}] {status}')

    if not all_results:
        print('\nNo results. Download pkl files to', data_dir)
        sys.exit(1)

    print()
    print_table(all_results)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'\nSaved to {OUT_PATH}')


if __name__ == '__main__':
    main()
