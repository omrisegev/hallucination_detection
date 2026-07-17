#!/usr/bin/env python
"""
run_selector_bench — thin driver for spectral_utils.selector_bench.

Examples
    python scripts/run_selector_bench.py --self-check
    python scripts/run_selector_bench.py --selector a1_residual --pool h16
    python scripts/run_selector_bench.py --selector a1_residual --pool c46 \
        --domains repgrid

Worktree note (Step 186): local_cache/ and results/subset_sweep/*.npz are
UNTRACKED — they exist only in the main checkout. A bench run from a git
worktree must point --data-root (or the HD_DATA_ROOT env var) at the main
checkout; --out defaults into THIS checkout's results/selector_bench/, so
worktree results stay on the worktree's branch.
"""

import argparse
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


def main():
    from spectral_utils.selector_bench import bench_selector, self_check

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selector', default='a1_residual')
    ap.add_argument('--pool', choices=['h16', 'c46'], default='h16')
    ap.add_argument('--data-root', default=os.environ.get('HD_DATA_ROOT', REPO_DIR),
                    help='checkout holding local_cache/ + results/subset_sweep/ '
                         '(env HD_DATA_ROOT; MUST be the main checkout when '
                         'running from a worktree)')
    ap.add_argument('--npz-dir', default=None,
                    help='default: <data-root>/results/subset_sweep')
    ap.add_argument('--out', default=None,
                    help='default: <this repo>/results/selector_bench/'
                         '<selector>__<pool>.csv')
    ap.add_argument('--domains', default=None,
                    help='comma list: math500,gsm8k,gpqa,rag,qa,repgrid,trace')
    ap.add_argument('--cells', default=None, help='substring filter on cell keys')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rand-R', type=int, default=32,
                    help='random-floor sample size for live pools (c46 arm)')
    ap.add_argument('--self-check', action='store_true',
                    help='run the bench integrity gate (lookup-vs-live, GOOD_5 '
                         'reproduction, no-label-leak) and exit')
    args = ap.parse_args()

    npz_dir = args.npz_dir or os.path.join(args.data_root, 'results', 'subset_sweep')

    if args.self_check:
        report = self_check(args.data_root, npz_dir,
                            sweep_summary_csv=os.path.join(npz_dir, 'sweep_summary.csv'))
        print(f"self-check PASS: {report['cells']} cells, "
              f"{report['pairs']} lookup-vs-live pairs, "
              f"GOOD_5 reproduced on {report['good5_checked']} cells "
              f"(max |diff| {report['good5_max_abs']:.2e})")
        return

    out = args.out or os.path.join(REPO_DIR, 'results', 'selector_bench',
                                   f"{args.selector}__{args.pool}.csv")
    domains = args.domains.split(',') if args.domains else None
    cells = args.cells.split(',') if args.cells else None
    n = bench_selector(args.selector, args.pool, args.data_root, npz_dir, out,
                       seed=args.seed, domains=domains, cells=cells,
                       rand_R=args.rand_R)
    print(f"wrote {n} new rows -> {out}")


if __name__ == '__main__':
    main()
