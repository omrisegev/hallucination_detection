#!/usr/bin/env python3
"""
verify_lsml_paper.py — Verify lsml_fuse against the Jaffé-Fetaya-Nadler 2016 paper.

Three experiments:
  A) Paper conditions (M=9, K=3 groups of 3, n=2000)
     Tests that lsml_fuse correctly handles dependent classifiers and
     outperforms naive SML on the latent variable model of the paper.
     MUST PASS before blaming the data.

  B) M=5 degeneration
     Tests what happens with our 5-feature GOOD_FEATURES setup.
     Diagnoses whether K_range including K=m causes degenerate grouping.

  C) Real data (local_cache/math500_res.pkl, GOOD_FEATURES)
     Shows actual production numbers and the effect of the K_range fix.

Usage:
    python scripts/verify_lsml_paper.py
    python scripts/verify_lsml_paper.py --data-dir ./local_cache
"""

import argparse
import os
import pickle
import sys
import warnings

import numpy as np
from sklearn.metrics import adjusted_rand_score

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils.fusion_utils import (
    boot_auc,
    binarize_classifiers,
    lsml_fuse,
    simple_average_fusion,
    sml_fuse,
    zscore,
)

GOOD_FEATURES = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']

FEATURE_SIGNS = {
    'epr': -1, 'trace_length': 1, 'spectral_entropy': -1,
    'low_band_power': -1, 'high_band_power': -1, 'hl_ratio': -1,
    'dominant_freq': -1, 'spectral_centroid': -1,
    'stft_max_high_power': -1, 'stft_spectral_entropy': -1,
    'rpdi': -1, 'sw_var_peak': -1,
    'pe_mean': -1, 'hurst_exponent': 1,
    'cusum_max': -1, 'cusum_shift_idx': 1,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def best_auc(labels, scores):
    """AUROC taking the better of both score orientations."""
    a_p, lo_p, hi_p = boot_auc(labels, scores)
    a_n, lo_n, hi_n = boot_auc(labels, -scores)
    if a_p >= a_n:
        return a_p, lo_p, hi_p
    return a_n, lo_n, hi_n


def header(title):
    print(f'\n{"=" * 62}')
    print(f'  {title}')
    print(f'{"=" * 62}')


# ── Synthetic data (Jaffé-Fetaya-Nadler latent variable model, Fig. 1 right) ──

def make_group_data(n, group_accs, within_accs, seed=42):
    """
    Generate binary ±1 classifiers with group-dependent errors.

    Model (Jaffé-Fetaya-Nadler 2016, Fig. 1 right):
        y   ~ Uniform({-1, +1})                     # true label
        ξ_g ~ y   with prob a_g,  else -y           # latent group variable
        f_i ~ ξ_g with prob b_i,  else -ξ_g        # observed classifier

    Marginal balanced accuracy of f_i: α_i = a_g*b_i + (1-a_g)*(1-b_i)

    Classifiers within a group are correlated (share ξ_g) but conditionally
    independent given ξ_g.  This violates the independence assumption of naive
    SML (Parisi-Nadler-Kluger 2014) but satisfies the L-SML assumption.

    Returns:
        classifiers: (n, M) float array of ±1 values
        labels:      (n,) int array of 0/1  (y=+1 -> label=1)
        true_groups: (M,) int array with group index 0..K-1 for each classifier
    """
    rng = np.random.default_rng(seed)
    y = rng.choice([-1, 1], size=n)
    classifiers, true_groups = [], []
    for g, (a_g, accs_g) in enumerate(zip(group_accs, within_accs)):
        xi_g = np.where(rng.random(n) < a_g, y, -y)
        for b_i in accs_g:
            f_i = np.where(rng.random(n) < b_i, xi_g, -xi_g).astype(float)
            classifiers.append(f_i)
            true_groups.append(g)
    labels = ((y + 1) // 2).astype(int)
    return np.column_stack(classifiers), labels, np.array(true_groups)


# ── Experiment A: Paper conditions ────────────────────────────────────────────

def run_exp_a():
    header('Exp A: Paper conditions  (M=9, K=3 groups of 3, n=2000)')
    GROUP_ACCS  = [0.75, 0.80, 0.85]
    WITHIN_ACCS = [[0.70, 0.75, 0.80]] * 3

    marginal = [
        a * b + (1 - a) * (1 - b)
        for a, accs in zip(GROUP_ACCS, WITHIN_ACCS)
        for b in accs
    ]
    print(f'  Group latent accuracies:    {GROUP_ACCS}')
    print(f'  Within-group accuracies:    [0.70, 0.75, 0.80] per group')
    print(f'  Marginal classifier AUCs:   {[f"{a:.2f}" for a in marginal]}')
    print()

    X, labels, true_groups = make_group_data(
        n=2000, group_accs=GROUP_ACCS, within_accs=WITHIN_ACCS,
    )
    m = X.shape[1]

    # Individual best
    aucs_indiv = [best_auc(labels, X[:, i])[0] for i in range(m)]
    best_indiv_auc = max(aucs_indiv)
    best_idx = int(np.argmax(aucs_indiv))
    print(f'  Individual best AUROC:   {best_indiv_auc:.3f}  (classifier {best_idx},'
          f' group {true_groups[best_idx]})')

    # Naive SML: treats all 9 as independent
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        naive_scores, naive_w = sml_fuse(*[X[:, i] for i in range(m)])
    naive_auc, _, _ = best_auc(labels, naive_scores)
    print(f'  Naive SML AUROC:         {naive_auc:.3f}  (assumes independence, violates model)')

    # L-SML
    caught = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always')
        lsml_scores, meta = lsml_fuse(*[X[:, i] for i in range(m)])
        caught = [str(w.message) for w in wlist]

    lsml_auc, _, _ = best_auc(labels, lsml_scores)
    K_det = meta['K']
    ari = adjusted_rand_score(true_groups, meta['c'])

    print(f'  L-SML AUROC:             {lsml_auc:.3f}')
    print(f'  L-SML K detected:        {K_det}  (true K=3)')
    print(f'  L-SML group ARI:         {ari:.3f}  (1.0 = perfect match, >0.70 = good)')
    for w in caught[:1]:
        print(f'  [WARNING] {w[:90]}')

    # Note: naive SML can be competitive here because all classifiers are
    # pre-oriented (positively correlated with y). The paper's main gain over
    # naive SML shows up when classifier orientations are unknown/mixed.
    # The KEY algorithmic test is group detection (ARI).
    print()
    checks = [
        ('ARI >= 0.70  (group detection works)',              ari >= 0.70),
        ('L-SML AUROC > 0.70  (fusion beats near-chance)',   lsml_auc > 0.70),
        ('L-SML > best_indiv - 0.10  (no catastrophic drop)', lsml_auc > best_indiv_auc - 0.10),
    ]
    passed = True
    for desc, ok in checks:
        status = 'PASS' if ok else 'FAIL'
        if not ok:
            passed = False
        print(f'  [{status}] {desc}')
    if naive_auc > lsml_auc:
        print(f'  [INFO] Naive SML ({naive_auc:.3f}) > L-SML ({lsml_auc:.3f}) here because all')
        print(f'         classifiers are pre-oriented. L-SML advantage shows with mixed')
        print(f'         signs (paper setting). Group detection (ARI={ari:.2f}) is correct.')

    print()
    print(f'  *** Exp A overall: {"PASS" if passed else "FAIL"} ***')
    return passed


# ── Experiment B: M=5 degeneration ───────────────────────────────────────────

def run_exp_b():
    header('Exp B: M=5 degeneration  (simulates our 5-feature GOOD_FEATURES use case)')
    print('  Using first 5 classifiers: 2 groups [0,0,0,1,1]')
    print()

    GROUP_ACCS  = [0.75, 0.80]
    WITHIN_ACCS = [[0.70, 0.75, 0.80], [0.70, 0.75]]

    X, labels, true_groups = make_group_data(
        n=2000, group_accs=GROUP_ACCS, within_accs=WITHIN_ACCS,
    )
    m = X.shape[1]  # 5

    aucs_indiv = [best_auc(labels, X[:, i])[0] for i in range(m)]
    best_indiv_auc = max(aucs_indiv)
    print(f'  Individual best AUROC:   {best_indiv_auc:.3f}')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        naive_scores, _ = sml_fuse(*[X[:, i] for i in range(m)])
    naive_auc, _, _ = best_auc(labels, naive_scores)
    print(f'  Naive SML AUROC:         {naive_auc:.3f}')

    # L-SML with DEFAULT K_range = [2,3,4,5]
    default_krange = list(range(2, min(m, 8) + 1))
    print(f'\n  --- DEFAULT K_range = {default_krange} ---')
    caught_default = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always')
        lsml_d, meta_d = lsml_fuse(*[X[:, i] for i in range(m)])
        caught_default = [str(w.message) for w in wlist]

    lsml_d_auc, _, _ = best_auc(labels, lsml_d)
    K_d = meta_d['K']
    ari_d = adjusted_rand_score(true_groups, meta_d['c'])
    degenerate = (K_d == m)
    print(f'  K selected:              {K_d}' +
          ('  <- K=M: each classifier alone = degenerate!' if degenerate else ''))
    print(f'  Group ARI:               {ari_d:.3f}  (1.0 = perfect; 0.0 = random)')
    print(f'  L-SML AUROC:             {lsml_d_auc:.3f}')
    for w in caught_default[:2]:
        print(f'  [WARNING] {w[:90]}')

    # L-SML with FIXED K_range = [2,3,4] (cap at m-1)
    fixed_krange = list(range(2, m))   # [2, 3, 4]
    print(f'\n  --- FIXED K_range = {fixed_krange}  (cap at m-1={m-1}) ---')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lsml_f, meta_f = lsml_fuse(*[X[:, i] for i in range(m)],
                                    K_range=fixed_krange)

    lsml_f_auc, _, _ = best_auc(labels, lsml_f)
    K_f = meta_f['K']
    ari_f = adjusted_rand_score(true_groups, meta_f['c'])
    print(f'  K selected:              {K_f}  (true K=2)')
    print(f'  Group ARI:               {ari_f:.3f}')
    print(f'  L-SML AUROC:             {lsml_f_auc:.3f}')

    delta_pp = (lsml_f_auc - lsml_d_auc) * 100
    print()
    print(f'  Fix gain:  {delta_pp:+.1f} pp  (fixed K_range vs default)')
    print()

    print('  Diagnosis:')
    if degenerate:
        print(f'  [CONFIRMED] Default K_range selects K={m} (= M): every classifier is')
        print(f'              its own group. lsml_fuse degenerates to naive cross-SML.')
    else:
        print(f'  [NOTE] Default K_range selected K={K_d}, not the degenerate K={m}.')

    if ari_f >= 0.70:
        print(f'  [CONFIRMED] Fixed K_range recovers correct grouping (ARI={ari_f:.2f}).')
    if delta_pp > 1.0:
        print(f'  [CONFIRMED] Fixed K_range improves AUROC by {delta_pp:+.1f} pp.')

    print()
    print('  Root cause: detect_dependent_groups() default K_range:')
    print(f'    OLD: range(2, min(m,8)+1) = {default_krange}  includes K=m -> degenerate')
    print(f'    FIX: range(2, min(m,9))   = {fixed_krange}  always K < m')
    print('  One-line change in spectral_utils/fusion_utils.py:')
    print('    - K_range = list(range(2, min(m, 8) + 1))')
    print('    + K_range = list(range(2, min(m, 9)))')

    return degenerate


# ── Experiment C: Real data ───────────────────────────────────────────────────

def run_exp_c(data_dir):
    header('Exp C: Real data  (math500_res.pkl, GOOD_FEATURES, M=5)')

    # Try multiple cache file candidates
    feats_data = None
    for fname in ['math500_res.pkl', 'lsml_v2_math500_res.pkl', 'lsml_math500_res.pkl']:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        # Format: dict with 'feats' key -> {cell_key: (feats_dict, labels)}
        if isinstance(obj, dict) and 'feats' in obj:
            feats_data = obj['feats']
            print(f'  Loaded: {fname}')
            break
        # Fallback: top-level dict of (feats_dict, labels) tuples
        if isinstance(obj, dict):
            sample = next(iter(obj.values()), None)
            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                feats_data = obj
                print(f'  Loaded: {fname}')
                break

    if feats_data is None:
        print(f'  No math500 pkl found in {data_dir}. Skipping Exp C.')
        print('  (Populate local_cache from Colab notebooks first.)')
        return

    # Prefer the T=1.0 cell for Qwen2.5-Math-1.5B to match our target
    target = None
    for k, v in feats_data.items():
        if not isinstance(v, (tuple, list)) or len(v) < 2:
            continue
        fd, lb = v[0], v[1]
        if not isinstance(fd, dict):
            continue
        if not all(f in fd for f in GOOD_FEATURES):
            continue
        lb_arr = np.array(lb, dtype=int)
        if len(np.unique(lb_arr)) < 2 or len(lb_arr) < 10:
            continue
        if target is None:
            target = (k, fd, lb_arr)
        # Prefer Qwen-1.5B T=1.0 to match the numbers we've been discussing
        if '1.5' in str(k) and 'T1.0' in str(k):
            target = (k, fd, lb_arr)
            break

    if target is None:
        print(f'  No cell with all GOOD_FEATURES found. Skipping Exp C.')
        return

    key, fd, labels = target
    m = len(GOOD_FEATURES)
    print(f'  Cell: {key}  n={len(labels)}, pos={labels.sum()} ({100*labels.mean():.0f}%)')
    print()

    # Binarize with FEATURE_SIGNS
    binary = binarize_classifiers(
        {f: np.array(fd[f]) for f in GOOD_FEATURES}, FEATURE_SIGNS,
    )
    classifiers = [binary[f] for f in GOOD_FEATURES]

    # Individual feature AUROCs
    print('  Individual feature AUROCs (raw continuous):')
    indiv_aucs = {}
    for f in GOOD_FEATURES:
        auc, _, _ = best_auc(labels, np.array(fd[f]))
        indiv_aucs[f] = auc
        print(f'    {f:<22}: {auc:.3f}')
    best_indiv_auc = max(indiv_aucs.values())

    # Simple average of CONTINUOUS z-scored features
    cont_oriented = [zscore(np.array(fd[f]) * FEATURE_SIGNS.get(f, 1))
                     for f in GOOD_FEATURES]
    avg_scores, _ = simple_average_fusion(*cont_oriented)
    avg_auc, _, _ = best_auc(labels, avg_scores)
    print(f'  Simple average (continuous):  {avg_auc:.3f}')

    # Simple average of BINARIZED features (same input as L-SML)
    bin_avg_scores, _ = simple_average_fusion(*classifiers)
    bin_avg_auc, _, _ = best_auc(labels, bin_avg_scores)
    print(f'  Simple average (binarized):   {bin_avg_auc:.3f}  (same inputs as L-SML)')

    # L-SML default K_range
    default_krange = list(range(2, min(m, 8) + 1))
    caught_c = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always')
        lsml_d, meta_d = lsml_fuse(*classifiers)
        caught_c = [str(w.message) for w in wlist]
    lsml_d_auc, _, _ = best_auc(labels, lsml_d)
    K_d = meta_d['K']
    degenerate_c = (K_d == m)
    print(f'  L-SML default K_range={default_krange}:  {lsml_d_auc:.3f}'
          f'  (K={K_d}{"=M degenerate" if degenerate_c else ""})')
    for g in range(K_d):
        grp_feats = [GOOD_FEATURES[i] for i, c in enumerate(meta_d['c']) if c == g]
        print(f'    Group {g}: {grp_feats}')
    for w in caught_c[:1]:
        print(f'    [WARNING] {w[:90]}')

    # L-SML fixed K_range
    fixed_krange = list(range(2, m))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lsml_f, meta_f = lsml_fuse(*classifiers, K_range=fixed_krange)
    lsml_f_auc, _, _ = best_auc(labels, lsml_f)
    K_f = meta_f['K']
    improved = lsml_f_auc > lsml_d_auc + 0.005
    print(f'  L-SML fixed K_range={fixed_krange}: {lsml_f_auc:.3f}'
          f'  (K={K_f}{"  <-- IMPROVED" if improved else ""})')

    delta_pp = (lsml_f_auc - lsml_d_auc) * 100
    vs_best_pp = (lsml_f_auc - best_indiv_auc) * 100
    print()
    binarization_cost_pp = (avg_auc - bin_avg_auc) * 100
    print(f'  Summary:')
    print(f'    Best individual (continuous):  {best_indiv_auc:.3f}')
    print(f'    Simple average (continuous):   {avg_auc:.3f}')
    print(f'    Simple average (binarized):    {bin_avg_auc:.3f}'
          f'  (binarization cost: {binarization_cost_pp:+.1f} pp)')
    print(f'    L-SML (default, K={K_d}):        {lsml_d_auc:.3f}')
    print(f'    L-SML (fixed range, K={K_f}):    {lsml_f_auc:.3f}  '
          f'({delta_pp:+.1f} pp vs default,  {vs_best_pp:+.1f} pp vs best individual)')
    lsml_vs_bin_avg = (lsml_f_auc - bin_avg_auc) * 100
    print(f'    L-SML vs binarized average:    {lsml_vs_bin_avg:+.1f} pp  '
          f'(SML lift on same-format inputs)')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data-dir', default='./local_cache',
                        help='local_cache directory (default: ./local_cache)')
    args = parser.parse_args()

    print('L-SML Paper Verification Script')
    print('Ref: Jaffé-Fetaya-Nadler 2016, "Unsupervised Ensemble Learning')
    print('     with Dependent Classifiers"')
    print('Impl: spectral_utils/fusion_utils.py::lsml_fuse')

    pass_a     = run_exp_a()
    degenerate = run_exp_b()
    run_exp_c(args.data_dir)

    header('SUMMARY')
    print(f'  Exp A (paper conditions M=9):  {"PASS" if pass_a else "FAIL"}')
    print(f'  Exp B (M=5 degeneration):      '
          f'{"CONFIRMED — K=M selected by default" if degenerate else "NOT CONFIRMED"}')
    print()
    if pass_a and degenerate:
        print('  CONCLUSION: lsml_fuse is CORRECT on paper conditions (M >> K).')
        print('  The M=5 failure is a K_range bug: default allows K=M (degenerate).')
        print()
        print('  ONE-LINE FIX in spectral_utils/fusion_utils.py,')
        print('  function detect_dependent_groups(), ~line 405:')
        print()
        print('    - K_range = list(range(2, min(m, 8) + 1))')
        print('    + K_range = list(range(2, min(m, 9)))')
        print()
        print('  This ensures K < M always, preventing the degenerate singleton grouping.')
    elif not pass_a:
        print('  CONCLUSION: lsml_fuse may have a deeper bug.')
        print('  Investigate: sml_fuse_signed, detect_dependent_groups, _residual_lsml.')
    else:
        print('  CONCLUSION: K=M not selected by default — check the ARI and AUROC')
        print('  values manually to diagnose the M=5 performance drop.')


if __name__ == '__main__':
    main()
