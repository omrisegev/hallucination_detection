"""
Compare CONT (continuous L-SML) vs U-PCR fusion on all cached inference cells.

Usage:
    python scripts/run_upcr_comparison.py

Reads: consolidated_results/features_all.pkl  (or the per-cell pkl files under
       consolidated_results/cells/)
Writes: results/upcr_comparison.pkl

Output table example:
    Cell                       | CONT AUROC | U-PCR AUROC |    Δ
    MATH500/Qwen-Math-7B       |    90.0%   |     89.4%   | -0.6
    ...
    MACRO                      |    70.1%   |     XX.X%   | +Y.Y
"""

import os
import sys
import pickle
import warnings
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils import (
    lsml_continuous_pipeline,
    upcr_pipeline,
    boot_auc,
    zscore,
)

# ── Config ────────────────────────────────────────────────────────────────────

GOOD_FEATURES = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']
FEATURE_SIGNS = {f: -1 for f in GOOD_FEATURES}  # all features: higher = more wrong

FEATURES_PKL = os.path.join(REPO_DIR, 'consolidated_results', 'features_all.pkl')
OUT_PATH = os.path.join(REPO_DIR, 'results', 'upcr_comparison.pkl')

# ── Load data ─────────────────────────────────────────────────────────────────

def load_features():
    if not os.path.exists(FEATURES_PKL):
        raise FileNotFoundError(
            f"Features pkl not found: {FEATURES_PKL}\n"
            "Run the method_comparison.py script first to generate features_all.pkl."
        )
    with open(FEATURES_PKL, 'rb') as f:
        return pickle.load(f)


def run_cell(cell_name, feats_dict, labels):
    """Run CONT and U-PCR on one cell. Returns dict with both AUROCs."""
    labels = np.array(labels, dtype=float)

    # Check which GOOD_FEATURES are available
    available = [f for f in GOOD_FEATURES if f in feats_dict and
                 feats_dict[f] is not None and len(feats_dict[f]) == len(labels)]
    if len(available) < 3:
        return {'cell': cell_name, 'n_feat': len(available), 'skipped': True,
                'reason': f'only {len(available)} features available (need >=3)'}

    feat_signs = {f: FEATURE_SIGNS[f] for f in available}

    # ── CONT (continuous L-SML) ───────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            cont_score, cont_meta = lsml_continuous_pipeline(
                feats_dict, available, feat_signs, normalize=True
            )
            cont_auc, cont_lo, cont_hi = boot_auc(labels, cont_score)
        except Exception as e:
            cont_auc, cont_lo, cont_hi = None, None, None
            cont_meta = {'error': str(e)}

    # ── U-PCR ─────────────────────────────────────────────────────────────────
    p = np.mean(labels)
    var_y = p * (1.0 - p)  # Var(Y) for binary labels at observed prevalence
    var_y = max(var_y, 0.05)  # clamp — very skewed cells still need a sensible g^2 bound
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            upcr_score, w, rho_hat, g2_hat = upcr_pipeline(
                feats_dict, available, feat_signs, var_y=var_y
            )
            upcr_auc, upcr_lo, upcr_hi = boot_auc(labels, upcr_score)
        except Exception as e:
            upcr_auc, upcr_lo, upcr_hi = None, None, None
            w, rho_hat, g2_hat = None, None, None

    delta = (upcr_auc - cont_auc) if (cont_auc is not None and upcr_auc is not None) else None

    return {
        'cell': cell_name,
        'n_feat': len(available),
        'features': available,
        'n_samples': int(len(labels)),
        'prevalence': float(p),
        'var_y': float(var_y),
        'cont_auc': cont_auc, 'cont_lo': cont_lo, 'cont_hi': cont_hi,
        'upcr_auc': upcr_auc, 'upcr_lo': upcr_lo, 'upcr_hi': upcr_hi,
        'delta': delta,
        'upcr_weights': w.tolist() if w is not None else None,
        'upcr_rho_hat': rho_hat.tolist() if rho_hat is not None else None,
        'upcr_g2_hat': float(g2_hat) if g2_hat is not None else None,
        'skipped': False,
    }


def print_table(results):
    header = f"{'Cell':<35} | {'CONT':>10} | {'U-PCR':>10} | {'Δ':>7}"
    sep = '-' * len(header)
    print(sep)
    print(header)
    print(sep)
    cont_aucs, upcr_aucs = [], []
    for r in results:
        if r.get('skipped'):
            print(f"  {r['cell']:<33} | {'SKIP':>10} | {'SKIP':>10} | {r.get('reason','')}")
            continue
        c = f"{100*r['cont_auc']:.1f}%" if r['cont_auc'] is not None else '  N/A '
        u = f"{100*r['upcr_auc']:.1f}%" if r['upcr_auc'] is not None else '  N/A '
        d = f"{100*r['delta']:+.1f}pp" if r['delta'] is not None else '  N/A '
        print(f"  {r['cell']:<33} | {c:>10} | {u:>10} | {d:>7}")
        if r['cont_auc'] is not None:
            cont_aucs.append(r['cont_auc'])
        if r['upcr_auc'] is not None:
            upcr_aucs.append(r['upcr_auc'])
    print(sep)
    if cont_aucs and upcr_aucs:
        mac_c = np.mean(cont_aucs)
        mac_u = np.mean(upcr_aucs)
        d_mac = mac_u - mac_c
        print(f"  {'MACRO':<33} | {100*mac_c:>9.1f}% | {100*mac_u:>9.1f}% | {100*d_mac:>+6.1f}pp")
    print(sep)


def main():
    print(f"Loading features from {FEATURES_PKL} ...")
    data = load_features()

    # features_all.pkl structure: {cell_name: {'feats': {...}, 'labels': [...]}}
    # Adapt if your pkl has a different schema.
    results = []
    cell_names = sorted(data.keys())
    print(f"Found {len(cell_names)} cells.\n")

    for cell_name in cell_names:
        entry = data[cell_name]
        if isinstance(entry, dict) and 'feats' in entry and 'labels' in entry:
            feats_dict = entry['feats']
            labels = entry['labels']
        elif isinstance(entry, dict) and 'features' in entry and 'labels' in entry:
            feats_dict = entry['features']
            labels = entry['labels']
        else:
            print(f"  [{cell_name}] unrecognised entry schema — skipping")
            results.append({'cell': cell_name, 'skipped': True, 'reason': 'unknown schema'})
            continue

        r = run_cell(cell_name, feats_dict, labels)
        results.append(r)
        status = 'SKIP' if r.get('skipped') else (
            f"CONT {100*r['cont_auc']:.1f}% | U-PCR {100*r['upcr_auc']:.1f}%"
            if r['cont_auc'] is not None and r['upcr_auc'] is not None
            else 'ERROR'
        )
        print(f"  [{cell_name}] {status}")

    print()
    print_table(results)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == '__main__':
    main()
