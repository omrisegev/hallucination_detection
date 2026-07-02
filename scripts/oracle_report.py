"""
Oracle report — common-cell validation tables from logistic_oracle.pkl.

Reads results/logistic_oracle.pkl (produced by scripts/logistic_oracle.py) and
re-aggregates it on a strict COMMON-CELL basis: a cell counts toward a feature
set's macro only when BOTH the unsupervised CONT score and the supervised LR
score exist for that set. This removes the CONT-only trivia_qa_traces cell that
inflated the CONT macro in the original email (understating the supervised gap
by ~1pp).

Emits:
  1. Macro table (CONT / LR-CV / gap / balanced in-sample ceiling / std ceiling)
     for feature sets 5 / 9 / 16.
  2. Per-domain breakdown (Reasoning = math500+gsm8k, GPQA, RAG+QA).
  3. One-line-per-cell dump so the advisor-reply tables are reproducible.

Usage:
    python scripts/oracle_report.py [--pkl results/logistic_oracle.pkl]

Read-only: does not recompute any model, only re-reads the saved pkl.
"""

import os
import sys
import pickle
import argparse
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PKL = os.path.join(REPO_DIR, 'results', 'logistic_oracle.pkl')

FEAT_SETS = ['5', '9', '16']
FEAT_LABEL = {'5': '5-feat (GOOD_5)', '9': '9-feat (STABLE_H9)', '16': '16-feat (ALL_H16)'}

# Domain grouping used in the advisor email.
DOMAIN_GROUPS = {
    'Reasoning (math500+gsm8k)': ('math500', 'gsm8k'),
    'GPQA':                      ('gpqa',),
    'RAG + QA':                  ('rag', 'qa'),
}


def domain_of(cell):
    return cell.split('/', 1)[0]


def load_rows(pkl_path):
    with open(pkl_path, 'rb') as f:
        rows = pickle.load(f)
    return [r for r in rows if not r.get('skipped')]


def common_cells(rows, k):
    """Cells where BOTH cont_k and lr_k exist for feature set k."""
    return [r for r in rows
            if r.get(f'cont_{k}') is not None and r.get(f'lr_{k}') is not None]


def macro(vals):
    return float(np.mean(vals)) if vals else None


def fmt_pct(x):
    return f'{100 * x:.1f}%' if x is not None else '  N/A'


def fmt_pp(x):
    return f'{100 * x:+.1f}' if x is not None else '  N/A'


def aggregate(rows, k, domains=None):
    """Return dict of common-cell macros for feature set k over the given domains."""
    cc = common_cells(rows, k)
    if domains is not None:
        cc = [r for r in cc if domain_of(r['cell']) in domains]
    cont = macro([r[f'cont_{k}'] for r in cc])
    lrcv = macro([r[f'lr_{k}'] for r in cc])           # bal_cv (primary LR oracle)
    bal_in = macro([r[f'lr_bal_in_{k}'] for r in cc if r.get(f'lr_bal_in_{k}') is not None])
    std_in = macro([r[f'lr_std_in_{k}'] for r in cc if r.get(f'lr_std_in_{k}') is not None])
    gap = (lrcv - cont) if (cont is not None and lrcv is not None) else None
    return {'n': len(cc), 'cont': cont, 'lrcv': lrcv, 'gap': gap,
            'bal_in': bal_in, 'std_in': std_in}


def print_macro_table(rows, title, domains=None):
    print(title)
    print('-' * 82)
    print(f"{'Feat set':<20} | {'#cells':>6} | {'CONT':>7} | {'LR CV':>7} | "
          f"{'gap':>6} | {'bal ceil':>8} | {'std ceil':>8}")
    print('-' * 82)
    for k in FEAT_SETS:
        a = aggregate(rows, k, domains)
        print(f"{FEAT_LABEL[k]:<20} | {a['n']:>6} | {fmt_pct(a['cont']):>7} | "
              f"{fmt_pct(a['lrcv']):>7} | {fmt_pp(a['gap']):>6} | "
              f"{fmt_pct(a['bal_in']):>8} | {fmt_pct(a['std_in']):>8}")
    print('-' * 82)
    print()


def print_per_cell(rows):
    print('Per-cell dump (CONT / LR-CV / gap  per feature set; common cells marked *)')
    print('-' * 118)
    hdr = f"{'cell':<42} | {'n':>4} | {'prev':>5} | "
    hdr += ' | '.join(f"{'CONT-'+k:>7} {'LR-'+k:>6} {'D-'+k:>5}" for k in FEAT_SETS)
    print(hdr)
    print('-' * 118)
    for r in sorted(rows, key=lambda x: x['cell']):
        parts = []
        for k in FEAT_SETS:
            c = r.get(f'cont_{k}')
            lr = r.get(f'lr_{k}')
            d = (lr - c) if (c is not None and lr is not None) else None
            parts.append(f"{fmt_pct(c):>7} {fmt_pct(lr):>6} {fmt_pp(d):>5}")
        prev = r.get('prevalence')
        prevs = f'{prev:.2f}' if prev is not None else ' N/A'
        print(f"{r['cell']:<42} | {r.get('n', 0):>4} | {prevs:>5} | " + ' | '.join(parts))
    print('-' * 118)
    print()


def make_feature_count_plot(rows, out_path):
    """Grouped bars of CONT vs LR-CV vs in-sample ceiling for the named 5/9/16 sets.

    The discrete companion to lr_convergence.png: shows the 9-feat dip (spectral_
    entropy dropped) appears in BOTH CONT and LR, and that the ceiling rises with
    features while CV stays roughly flat.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[plot] matplotlib not available, skipping')
        return

    aggs = {k: aggregate(rows, k) for k in FEAT_SETS}
    x = np.arange(len(FEAT_SETS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 6))

    cont = [aggs[k]['cont'] * 100 for k in FEAT_SETS]
    lrcv = [aggs[k]['lrcv'] * 100 for k in FEAT_SETS]
    ceil = [aggs[k]['bal_in'] * 100 for k in FEAT_SETS]

    b0 = ax.bar(x - w, cont, w, label='CONT (L-SML, unsup.)', color='#2196F3', alpha=0.9)
    b1 = ax.bar(x,     lrcv, w, label='LR 5-fold CV (supervised)', color='#E91E63', alpha=0.9)
    b2 = ax.bar(x + w, ceil, w, label='LR in-sample ceiling', color='#9E9E9E', alpha=0.9)
    for b in (b0, b1, b2):
        ax.bar_label(b, fmt='%.1f', padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([FEAT_LABEL[k] for k in FEAT_SETS])
    ax.set_ylabel('Macro AUROC (%) — common cells')
    ax.set_ylim(50, max(ceil) + 6)
    ax.axhline(50, color='gray', ls='--', lw=0.7, alpha=0.5)
    ax.set_title('Named feature sets: CONT vs LR CV vs in-sample ceiling\n'
                 'ceiling rises with #features while CV stays ~flat (overfitting); '
                 '9-feat dip (no spectral_entropy) shows in both methods', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis='y')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plot -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default=DEFAULT_PKL)
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.pkl):
        print(f'[ERROR] {args.pkl} not found. Run scripts/logistic_oracle.py first.')
        sys.exit(1)

    rows = load_rows(args.pkl)
    print(f'Loaded {len(rows)} non-skipped cells from {args.pkl}\n')

    print_macro_table(rows, 'MACRO — all common cells (CONT and LR both present)')

    for gname, domains in DOMAIN_GROUPS.items():
        print_macro_table(rows, f'MACRO — {gname}', domains=domains)

    print_per_cell(rows)

    if not args.no_plot:
        make_feature_count_plot(rows, os.path.join(REPO_DIR, 'results', 'oracle_feature_count.png'))


if __name__ == '__main__':
    main()
