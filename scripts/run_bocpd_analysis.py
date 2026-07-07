import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr

# Ensure repository root is in sys.path
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPTS_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils import (
    FEAT_NAMES, FEATURE_SIGNS, extract_all_features,
    zscore, boot_auc, anchor_orient,
    lsml_continuous_pipeline, sml_fuse_signed, upcr_pipeline,
)
from spectral_utils.streaming_utils import iter_entropy_traces
from spectral_utils.temporal_models import bocpd_gaussian

# Define feature sets
GOOD_5 = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']
SWEET_7 = ['cusum_max', 'epr', 'spectral_entropy', 'sw_var_peak', 'stft_spectral_entropy', 'low_band_power', 'trace_length']
SWEET_7_BOCPD = SWEET_7 + ['bocpd_ecp']
ALL_16 = list(FEAT_NAMES[:16])
ALL_16_BOCPD = ALL_16 + ['bocpd_ecp']

FEATURE_SETS = {
    'GOOD_5': GOOD_5,
    'SWEET_7': SWEET_7,
    'SWEET_7_BOCPD': SWEET_7_BOCPD,
    'ALL_16': ALL_16,
    'ALL_16_BOCPD': ALL_16_BOCPD,
}

# Signs mapping
SIGNS = {**FEATURE_SIGNS, 'bocpd_ecp': -1}

DATASETS = {
    'gsm8k': {
        'path': 'local_cache/raw_traces/p1_gsm8k_llama8b.pkl',
        'name': 'GSM8K (Llama-3.1-8B)',
    },
    'math500': {
        'path': 'local_cache/raw_traces/math500_T1.0.pkl',
        'name': 'MATH-500 (Qwen2.5-Math-1.5B)',
    },
    'gpqa': {
        'path': 'local_cache/raw_traces/p2c_gpqa_deepseek_r1_7b_inference.pkl',
        'name': 'GPQA Diamond (DeepSeek-R1-Distill-7B)',
    }
}

def flat_sml_continuous_pipeline(feats_dict, feat_names, signs):
    views = []
    for f in feat_names:
        arr = np.array(feats_dict[f], dtype=float)
        s = signs.get(f, 1)
        views.append(zscore(arr * s))
    fused_score, weights = sml_fuse_signed(*views)
    return fused_score, weights

def upcr_continuous_pipeline(feats_dict, feat_names, signs):
    score, w, _, _ = upcr_pipeline(feats_dict, feat_names, signs)
    return score, w

def process_dataset(name, info):
    print(f"\nProcessing {info['name']}...")
    if not os.path.exists(info['path']):
        print(f"  [Error] Cache file {info['path']} not found, skipping.")
        return None

    with open(info['path'], 'rb') as f:
        cache_obj = pickle.load(f)

    traces_and_labels = list(iter_entropy_traces(cache_obj))
    print(f"  Loaded {len(traces_and_labels)} raw traces.")

    processed_feats = []
    labels = []

    for tr, lbl in tqdm(traces_and_labels, desc=f"Extracting features for {name}"):
        if len(tr) < 8:
            continue
        fd = extract_all_features(tr)
        if fd is None:
            continue
        
        # Compute BOCPD change-point count
        b = bocpd_gaussian(tr, hazard_lambda=100.0)
        fd['bocpd_ecp'] = b['ecp']
        
        processed_feats.append(fd)
        labels.append(lbl)

    labels = np.array(labels, dtype=int)
    n_samples = len(labels)
    print(f"  Valid traces (len >= 8): {n_samples}, pos rate: {labels.mean():.2f}")

    # Build feature dict of vectors
    feats_dict = {}
    all_keys = list(processed_feats[0].keys())
    for k in all_keys:
        feats_dict[k] = np.array([f[k] for f in processed_feats])

    # Check correlation between BOCPD and EPR
    rho, _ = spearmanr(feats_dict['bocpd_ecp'], feats_dict['epr'])
    print(f"  Spearman rho(bocpd_ecp, epr) = {rho:.3f}")

    results = {}

    for fs_name, feat_names in FEATURE_SETS.items():
        results[fs_name] = {}
        for algo_name, pipeline_fn in [
            ('L-SML', lambda fd, fn, s: lsml_continuous_pipeline(fd, fn, s)[0]),
            ('Flat SML', lambda fd, fn, s: flat_sml_continuous_pipeline(fd, fn, s)[0]),
            ('U-PCR', lambda fd, fn, s: upcr_continuous_pipeline(fd, fn, s)[0])
        ]:
            try:
                raw_score = pipeline_fn(feats_dict, feat_names, SIGNS)
                # Resolve sign ambiguity using oriented EPR anchor (epr * -1 = higher is correct)
                score, flipped = anchor_orient(raw_score, feats_dict['epr'] * -1)
                auc, lo, hi = boot_auc(labels, score)
                results[fs_name][algo_name] = (auc, lo, hi)
            except Exception as e:
                print(f"    Error in {algo_name} on {fs_name}: {e}")
                results[fs_name][algo_name] = (np.nan, np.nan, np.nan)

    return results

def main():
    all_results = {}
    for name, info in DATASETS.items():
        res = process_dataset(name, info)
        if res is not None:
            all_results[name] = res

    # Output formatting
    print("\n\n==================== RESULTS COMPARISON ====================")
    for name, info in DATASETS.items():
        if name not in all_results:
            continue
        print(f"\nDataset: {info['name']}")
        res = all_results[name]
        
        # Header
        print(f"{'Feature Set':<15} | {'L-SML AUROC':<17} | {'Flat SML AUROC':<17} | {'U-PCR AUROC':<17}")
        print("-" * 75)
        for fs_name in ['GOOD_5', 'SWEET_7', 'SWEET_7_BOCPD', 'ALL_16', 'ALL_16_BOCPD']:
            row_vals = []
            for algo in ['L-SML', 'Flat SML', 'U-PCR']:
                auc, lo, hi = res[fs_name][algo]
                if np.isnan(auc):
                    row_vals.append("      N/A        ")
                else:
                    row_vals.append(f"{auc:.4f} [{lo:.3f},{hi:.3f}]")
            print(f"{fs_name:<15} | {row_vals[0]} | {row_vals[1]} | {row_vals[2]}")
    
    # Save the results to a pkl file
    with open('results/bocpd_analysis_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print("\nSaved results to results/bocpd_analysis_results.pkl")

if __name__ == '__main__':
    main()
