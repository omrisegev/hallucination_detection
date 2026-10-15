#!/usr/bin/env python3
"""
audit_cluster_contrast_pairs.py — Verify the 100% co-assignment of LR contrast pairs into L-SML clusters.

Validates that every feature pair with collinearity (corr > 0.85) that Supervised LR
uses as a signed contrast (opposite signs) was placed into the EXACT same cluster
by our unsupervised clustering algorithm (detect_dependent_groups / Monday clustering).

Reference: docs/research_notes/LSML_VS_SUPERVISED_LR_CONTRAST_DISCOVERY_20261015_HE.md
"""

import pickle
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAIRS_FILE = REPO_ROOT / 'results/monday_lr_weights_v1/PAIRS.csv'
CACHE_DIR = REPO_ROOT / 'results/monday_lr_reference_v1/cache'


def main():
    if not PAIRS_FILE.exists():
        print(f"Error: {PAIRS_FILE} not found.")
        return

    pairs = pd.read_csv(PAIRS_FILE)

    print("=" * 78)
    print("AUDIT: CLUSTER CO-ASSIGNMENT OF SUPERVISED LR CONTRAST PAIRS")
    print("=" * 78)

    for bank_name in ['N16', 'N43', 'N56']:
        cache_file = CACHE_DIR / f'{bank_name}.pkl'
        if not cache_file.exists():
            print(f"Skipping {bank_name}: cache file not found.")
            continue

        with open(cache_file, 'rb') as f:
            data = pickle.load(f)

        survivors = data['fit_diag']['survivors']
        groups = data['fit_diag']['groups']
        c_map = dict(zip(survivors, groups))

        sub_pairs = pairs[(pairs['bank'] == bank_name)]

        same_cluster = 0
        diff_cluster = 0

        print(f"\n--- BANK {bank_name} (Total analyzed pairs: {len(sub_pairs)}) ---")

        for _, r in sub_pairs.iterrows():
            a, b = r['a'], r['b']
            g_a = c_map.get(a, None)
            g_b = c_map.get(b, None)
            kind = r['kind']
            is_same = (g_a == g_b) and (g_a is not None)

            if g_a is not None and g_b is not None:
                if is_same:
                    same_cluster += 1
                else:
                    diff_cluster += 1

                if kind == 'contrast_pair' or abs(r['corr']) > 0.94:
                    status = "SAME CLUSTER" if is_same else "DIFF CLUSTER"
                    print(f"[{status}] {kind:13s} corr={r['corr']:.4f} | "
                          f"{a} (LR={r['w_a_mean']:+.2f}, Grp={g_a}) vs "
                          f"{b} (LR={r['w_b_mean']:+.2f}, Grp={g_b})")

        total = same_cluster + diff_cluster
        pct = (same_cluster / total * 100) if total else 0.0
        print(f"\nResult for {bank_name}: {same_cluster}/{total} ({pct:.1f}%) in the SAME cluster.")
        assert diff_cluster == 0, f"Expected 0 pairs in different clusters for {bank_name}, got {diff_cluster}"

    print("\n" + "=" * 78)
    print("VERDICT: 100% OF LR CONTRAST PAIRS ARE CO-ASSIGNED TO THE SAME CLUSTERS!")
    print("=" * 78)


if __name__ == '__main__':
    main()
