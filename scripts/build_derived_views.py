"""
build_derived_views.py — Stage 0 of the subset sweep.

(a) Computes the 6 anomaly-scorer views (mahalanobis, gmm_nll, kde_nll,
    iforest, ae, prae) from each cached cell's feature matrix — label-free,
    sample-aligned with the cell by construction — and saves them to
    local_cache/derived_views.pkl keyed (domain, cell_key).
(b) Builds self-contained trace-derived cells from local_cache/raw_traces/*:
    re-extracts the 16 H-features from each raw trace (so temporal views are
    sample-aligned) plus BOCPD / HMM / AR / Kalman views, spilled-energy
    features and BOCPD-on-ΔE(n) where token_spilled exists. Saved to
    local_cache/trace_cells.pkl. These are DIFFERENT sample sets from the
    cached cells — the sweep treats them as extra cells, never joins them.

Incremental: saves after every cell/file; already-computed keys are skipped
(delete the pkl or pass --force to recompute).

Usage:
    python scripts/build_derived_views.py [--data-dir ./local_cache] [--force]
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils.subset_sweep import (
    ANOMALY_VIEWS, H16, compute_anomaly_views, compute_trace_views, iter_cells,
)

TRACE_FILES = {
    'gsm8k_llama8b_raw':   'raw_traces/p1_gsm8k_llama8b.pkl',
    'math500_qwen1.5b_raw': 'raw_traces/math500_T1.0.pkl',
    'gpqa_r1_7b_raw':      'raw_traces/p2c_gpqa_deepseek_r1_7b_inference.pkl',
}


def atomic_dump(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=os.path.join(REPO_DIR, 'local_cache'))
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    # -- (a) anomaly views over the cached cells --------------------------------
    dv_path = os.path.join(args.data_dir, 'derived_views.pkl')
    derived = {}
    if os.path.exists(dv_path) and not args.force:
        with open(dv_path, 'rb') as f:
            derived = pickle.load(f)
    for domain, cell_key, fd, labels in iter_cells(args.data_dir):
        key = (domain, cell_key)
        if key in derived:
            print(f"[anomaly] {domain}/{cell_key}: cached, skipping")
            continue
        t0 = time.time()
        views = compute_anomaly_views(fd, labels, feat_list=H16)
        derived[key] = views
        atomic_dump(derived, dv_path)
        ok = [v for v in ANOMALY_VIEWS if views.get(v) is not None]
        print(f"[anomaly] {domain}/{cell_key}: n={len(labels)} "
              f"views={ok} ({time.time() - t0:.1f}s)")

    # -- (b) trace-derived cells -------------------------------------------------
    tc_path = os.path.join(args.data_dir, 'trace_cells.pkl')
    trace_cells = {}
    if os.path.exists(tc_path) and not args.force:
        with open(tc_path, 'rb') as f:
            trace_cells = pickle.load(f)
    for cell_key, rel in TRACE_FILES.items():
        if cell_key in trace_cells:
            print(f"[trace] {cell_key}: cached, skipping")
            continue
        path = os.path.join(args.data_dir, rel)
        if not os.path.exists(path):
            print(f"[trace] {cell_key}: {rel} missing — skipped")
            continue
        t0 = time.time()
        with open(path, 'rb') as f:
            cache_obj = pickle.load(f)
        built = compute_trace_views(cache_obj)
        if built is None:
            print(f"[trace] {cell_key}: <10 usable traces — skipped")
            continue
        trace_cells[cell_key] = built
        atomic_dump(trace_cells, tc_path)
        fd, labels = built
        print(f"[trace] {cell_key}: n={len(labels)} pos={labels.mean():.2f} "
              f"views={len(fd)} ({time.time() - t0:.1f}s)")

    # -- coverage table -----------------------------------------------------------
    print("\n=== Derived-view coverage ===")
    n_cells = len(derived)
    for v in ANOMALY_VIEWS:
        n_ok = sum(1 for views in derived.values() if views.get(v) is not None)
        print(f"  {v:<12} {n_ok}/{n_cells} cached cells")
    for cell_key, (fd, labels) in trace_cells.items():
        extras = sorted(k for k in fd if k not in H16)
        print(f"  trace/{cell_key}: n={len(labels)}, extra views: {extras}")
    print(f"\nSaved: {dv_path}\n       {tc_path}")


if __name__ == '__main__':
    main()
