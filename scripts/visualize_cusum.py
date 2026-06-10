"""
visualize_cusum.py — Plot entropy trace with CUSUM onset marker.

Shows a 2×2 grid (correct vs incorrect × one dataset cell each):
  - Token entropy trace (blue line)
  - Mean entropy (dashed gray)
  - CUSUM onset marker (vertical red dashed line at cusum_shift_idx × trace_length)
  - CUSUM cumulative sum (orange, second y-axis)

Data sources (tried in order):
  1. Phase 13 Drive cache: list of problem dicts with 'traces' and 'corrects' keys.
     Pass --phase13-cache /path/to/cache/mathcomp_phase13
  2. Local diagnostics pkl: global_df.pkl or diagnostics_all.pkl in --data-dir.
     Falls back to feature-level data if raw traces are unavailable.

Usage:
    # From Colab (after mounting Drive):
    python scripts/visualize_cusum.py \\
        --phase13-cache /content/drive/MyDrive/hallucination_detection/cache/mathcomp_phase13 \\
        --dataset math500 --temp 1.0

    # Locally (uses local_cache feature data — no raw traces, plots CUSUM stats only):
    python scripts/visualize_cusum.py

    # Save to specific path:
    python scripts/visualize_cusum.py --out-path local_cache/cusum_visualization.png
"""

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils import FEAT_NAMES


# ── CUSUM helpers ──────────────────────────────────────────────────────────────

def cusum_from_trace(ents):
    """Return (cusum array, max_idx, shift_idx_fraction)."""
    e = np.array(ents, dtype=float)
    residuals = e - e.mean()
    cusum = np.cumsum(residuals)
    max_idx = int(np.argmax(np.abs(cusum)))
    shift_frac = max_idx / max(len(e) - 1, 1)
    return cusum, max_idx, shift_frac


def plot_trace(ax_top, ax_bot, ents, title, color='steelblue'):
    """
    ax_top: entropy trace + onset marker
    ax_bot: CUSUM cumulative sum
    """
    e = np.array(ents, dtype=float)
    n = len(e)
    xs = np.arange(n)

    cusum, max_idx, shift_frac = cusum_from_trace(e)

    # Top panel — entropy trace
    ax_top.plot(xs, e, color=color, lw=1.4, alpha=0.9, label='H (token entropy)')
    ax_top.axhline(e.mean(), color='gray', linestyle='--', lw=1, alpha=0.7, label='Mean H')
    ax_top.axvline(max_idx, color='crimson', linestyle='--', lw=1.8,
                   label=f'CUSUM onset (t={max_idx}, {shift_frac:.0%})')
    ax_top.set_ylabel('Token entropy (nats)')
    ax_top.set_title(title, fontsize=9)
    ax_top.legend(fontsize=7, loc='upper right')
    ax_top.set_xlim(0, n - 1)

    # Bottom panel — CUSUM
    ax_bot.plot(xs, cusum, color='darkorange', lw=1.4, alpha=0.9)
    ax_bot.axhline(0, color='gray', linestyle=':', lw=0.8)
    ax_bot.axvline(max_idx, color='crimson', linestyle='--', lw=1.8, alpha=0.7)
    ax_bot.fill_between(xs, 0, cusum, alpha=0.15, color='darkorange')
    ax_bot.set_xlabel('Token index')
    ax_bot.set_ylabel('CUSUM Σ(H − μ)')
    ax_bot.set_xlim(0, n - 1)


# ── Data loading: Phase 13 format (Drive cache) ───────────────────────────────

def load_phase13_traces(cache_dir, slug, dataset, temp):
    """
    Load raw entropy traces from Phase 13 format.
    Returns (correct_traces, incorrect_traces) — lists of 1-D arrays.
    """
    pkl_path = os.path.join(cache_dir, slug, f'{dataset}_T{temp:.1f}.pkl')
    if not os.path.exists(pkl_path):
        # Try without slug subdir
        pkl_path = os.path.join(cache_dir, f'{dataset}_T{temp:.1f}.pkl')
    if not os.path.exists(pkl_path):
        print(f'  [phase13] Not found: {pkl_path}')
        return [], []

    with open(pkl_path, 'rb') as f:
        problems = pickle.load(f)

    correct_traces, incorrect_traces = [], []
    for prob in problems:
        traces = prob.get('traces', [])
        corrects = prob.get('corrects', [])
        for trace, ok in zip(traces, corrects):
            if trace is None or len(trace) < 8:
                continue
            if ok:
                correct_traces.append(np.array(trace, dtype=float))
            else:
                incorrect_traces.append(np.array(trace, dtype=float))

    print(f'  [phase13] {pkl_path}: {len(correct_traces)} correct, '
          f'{len(incorrect_traces)} incorrect traces')
    return correct_traces, incorrect_traces


# ── Data loading: local_cache diagnostics format ──────────────────────────────

def load_local_traces(data_dir):
    """
    Try to load raw entropy traces from local_cache.
    Checks diagnostics_all.pkl and global_df.pkl.
    Returns (correct_traces, incorrect_traces, source_name) or ([], [], None).
    """
    # Try diagnostics_all.pkl — may have per-sample trace data
    for fname in ['diagnostics_all.pkl', 'diagnostics_consensus_all.pkl']:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        # Look for any dict/list with 'token_entropies' or 'ents' or 'traces'
        traces_found = _extract_traces_from_obj(obj)
        if traces_found:
            correct, incorrect = traces_found
            print(f'  [local] Loaded {len(correct)} correct, {len(incorrect)} incorrect '
                  f'traces from {fname}')
            return correct, incorrect, fname

    print('  [local] No raw entropy traces found in local_cache.')
    print('  Run from Colab with --phase13-cache to get raw trace plots.')
    return [], [], None


def _extract_traces_from_obj(obj):
    """Recursively search obj for per-sample entropy traces + labels."""
    correct, incorrect = [], []
    candidates = obj if isinstance(obj, list) else (
        obj.values() if isinstance(obj, dict) else []
    )
    for item in candidates:
        if not isinstance(item, dict):
            continue
        trace = (item.get('token_entropies') or item.get('ents')
                 or item.get('trace') or item.get('traces'))
        label = item.get('label') or item.get('correct') or item.get('is_correct')

        if isinstance(trace, list) and len(trace) >= 8 and label is not None:
            arr = np.array(trace, dtype=float)
            if bool(label):
                correct.append(arr)
            else:
                incorrect.append(arr)
        elif isinstance(trace, list) and isinstance(trace[0], (list, np.ndarray)):
            # Phase 13 nested format
            corrects = item.get('corrects', [True] * len(trace))
            for t, ok in zip(trace, corrects):
                if len(t) >= 8:
                    arr = np.array(t, dtype=float)
                    (correct if ok else incorrect).append(arr)

    return (correct, incorrect) if (correct or incorrect) else None


# ── Select representative traces ──────────────────────────────────────────────

def pick_representative(traces, n=2, prefer_long=True):
    """Pick n traces — prefer longer ones for cleaner visualisation."""
    if not traces:
        return []
    if prefer_long:
        traces = sorted(traces, key=len, reverse=True)
    return traces[:n]


# ── Main plot ─────────────────────────────────────────────────────────────────

def make_figure(correct_traces, incorrect_traces, title_prefix, out_path):
    n_correct = min(2, len(correct_traces))
    n_incorrect = min(2, len(incorrect_traces))
    n_cols = n_correct + n_incorrect

    if n_cols == 0:
        print('No traces to plot.')
        return

    fig = plt.figure(figsize=(6 * n_cols, 6))
    gs = gridspec.GridSpec(2, n_cols, hspace=0.45, wspace=0.35,
                           height_ratios=[2, 1])

    col = 0
    for i, trace in enumerate(pick_representative(correct_traces, n_correct)):
        ax_top = fig.add_subplot(gs[0, col])
        ax_bot = fig.add_subplot(gs[1, col])
        n = len(trace)
        cusum, max_idx, shift_frac = cusum_from_trace(trace)
        title = (f'CORRECT — trace #{i+1}  (n={n} tokens)\n'
                 f'CUSUM onset at {shift_frac:.0%} of trace')
        plot_trace(ax_top, ax_bot, trace, title, color='#2196F3')
        col += 1

    for i, trace in enumerate(pick_representative(incorrect_traces, n_incorrect)):
        ax_top = fig.add_subplot(gs[0, col])
        ax_bot = fig.add_subplot(gs[1, col])
        n = len(trace)
        cusum, max_idx, shift_frac = cusum_from_trace(trace)
        title = (f'INCORRECT — trace #{i+1}  (n={n} tokens)\n'
                 f'CUSUM onset at {shift_frac:.0%} of trace')
        plot_trace(ax_top, ax_bot, trace, title, color='#E53935')
        col += 1

    fig.suptitle(f'CUSUM Onset Visualisation — {title_prefix}',
                 fontsize=11, fontweight='bold', y=1.01)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out_path}')


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase13-cache', metavar='DIR',
                        help='Path to Phase 13 Drive cache dir (contains slug subdirs)')
    parser.add_argument('--slug', default=None,
                        help='Model slug (e.g. qwen25_math_1b). Auto-detected if omitted.')
    parser.add_argument('--dataset', default='math500',
                        choices=['gsm8k', 'math500', 'amc23', 'aime24'],
                        help='Dataset to visualise (default: math500)')
    parser.add_argument('--temp', type=float, default=1.0,
                        help='Temperature to load (default: 1.0)')
    parser.add_argument('--data-dir', default='./local_cache',
                        help='Local cache dir for fallback (default: ./local_cache)')
    parser.add_argument('--out-path', default='./local_cache/cusum_visualization.png',
                        help='Output PNG path')
    args = parser.parse_args()

    correct_traces, incorrect_traces = [], []
    source = 'unknown'

    if args.phase13_cache:
        # Auto-detect slug if not given
        slug = args.slug
        if slug is None:
            try:
                slugs = [d for d in os.listdir(args.phase13_cache)
                         if os.path.isdir(os.path.join(args.phase13_cache, d))]
                slug = slugs[0] if slugs else ''
                print(f'  Auto-detected slug: {slug}')
            except FileNotFoundError:
                print(f'  Cache dir not found: {args.phase13_cache}')
                slug = ''

        correct_traces, incorrect_traces = load_phase13_traces(
            args.phase13_cache, slug, args.dataset, args.temp
        )
        source = f'{args.dataset} T={args.temp}'

    if not correct_traces and not incorrect_traces:
        correct_traces, incorrect_traces, src = load_local_traces(args.data_dir)
        source = src or 'local_cache'

    if not correct_traces and not incorrect_traces:
        print('\nNo raw entropy traces available. Options:')
        print('  1. Run from Colab:')
        print('     python scripts/visualize_cusum.py \\')
        print('       --phase13-cache /content/drive/.../cache/mathcomp_phase13 \\')
        print('       --dataset math500 --temp 1.0')
        print('  2. Make sure local_cache contains diagnostics_all.pkl with trace data.')
        return

    make_figure(correct_traces, incorrect_traces, source, args.out_path)


if __name__ == '__main__':
    main()
