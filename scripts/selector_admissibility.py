#!/usr/bin/env python
"""
selector_admissibility — A1.0 pre-registered objective validation (Step 186).

Before ANY residual-guided selector result is interpreted, this script
answers, on real cells: does a label-free objective actually track AUROC?
This is the same discipline that refuted the ρ≥0.75 filter (Step 153 /
memo §1.5) — an objective is trusted only if the data say so.

Objectives tested (within-size Spearman vs stored AUROC, per cell × size):
  eq14_raw   raw Eq-14 residual (stored per subset by the Step-153 sweep)
  eq14_rel   raw / Σ_{i≠j} C_ij² — the structure-seeking relative form
  upcr_k1    U-PCR projection residual, n_components=1 (live, sampled)
  rho_mean / rho_max   the old correlation-filter diagnostics (expected to
                       REPLICATE the §1.5 refutation, i.e. positive Spearman)
  K          detected group count (diagnostic)

PRE-REGISTERED CRITERION (memo D1, fixed before looking):
  an objective is ADMISSIBLE as a selection criterion iff, across cell×size
  units, median Spearman ≤ −0.10 AND Spearman < 0 in ≥ 60% of units
  (evaluated overall and per domain — RAG/GPQA is where the prize lives).
  Anything else → REFUTED/WEAK: a documented negative result, not a tweak
  invitation.

Router validity (memo §2.4): on sampled subsets, does
  (lsml_rel_residual ≤ upcr_k1_residual)  predict  (lsml AUROC ≥ upcr AUROC)?
Compared against the always-lsml base rate; pre-registered usefulness bar:
router accuracy > base rate + 0.03 overall.

Outputs (results/selector_bench/):
  admissibility_within_size.csv   one row per cell × size × objective
  admissibility_summary.csv       per objective × domain verdicts
  admissibility_router.csv        per-subset router records
  admissibility_router_summary.csv per-domain router accuracy
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils.selector_bench import (           # noqa: E402
    canonical_mask_to_local_cols, eval_subset_flex, iter_prepared_cells,
    load_npz_table)
from spectral_utils.selectors.a1_residual import (     # noqa: E402
    _mask_bits_matrix, _pearson, _upcr_k1_residual)

MIN_UNITS = 30          # min finite subsets per (cell, size) unit
ROUTER_SIZES = (4, 5, 6)
PER_SIZE = 34           # sampled subsets per size for upcr/router


def analyze_cell(ctx, table, rng, rows_ws, rows_router, per_size=PER_SIZE):
    auroc = table['auroc'].astype(float)
    resid = table['residual'].astype(float)
    sizes = table['size'].astype(int)
    K = table['K'].astype(float)
    rho_mean = table['rho_mean'].astype(float)
    rho_max = table['rho_max'].astype(float)

    C2 = _pearson(ctx.V) ** 2
    B = _mask_bits_matrix(table['mask'], ctx.pool_bits)
    energy = ((B @ C2) * B).sum(axis=1) - B.sum(axis=1)
    rel = resid / np.maximum(energy, 1e-12)

    objectives = {'eq14_raw': resid, 'eq14_rel': rel,
                  'rho_mean': rho_mean, 'rho_max': rho_max, 'K': K}

    base_ok = np.isfinite(auroc) & np.isfinite(resid)
    for s in np.unique(sizes):
        m = base_ok & (sizes == s)
        if m.sum() < MIN_UNITS:
            continue
        for name, vals in objectives.items():
            v = vals[m]
            if np.std(v) < 1e-12:
                continue
            r = spearmanr(v, auroc[m]).statistic
            rows_ws.append({'domain': ctx.domain, 'cell': ctx.cell_key,
                            'size': int(s), 'n': int(m.sum()),
                            'objective': name,
                            'spearman': round(float(r), 4)})

    # sampled: upcr_k1 objective + router validity
    lsml_rel_full = None
    upcr_res_by_size = {}
    for s in ROUTER_SIZES:
        idx_pool = np.flatnonzero(base_ok & (sizes == s))
        if len(idx_pool) < MIN_UNITS:
            continue
        pick = rng.choice(idx_pool, size=min(per_size, len(idx_pool)),
                          replace=False)
        res_list, auc_list = [], []
        for i in pick:
            cols = canonical_mask_to_local_cols(table['mask'][i], ctx.pool_bits)
            if cols is None:
                continue
            u_res = _upcr_k1_residual(ctx.V, cols)
            u_out = eval_subset_flex(ctx, cols, fusion='upcr')
            res_list.append(u_res)
            auc_list.append(float(auroc[i]))
            rows_router.append({
                'domain': ctx.domain, 'cell': ctx.cell_key, 'size': int(s),
                'lsml_auroc': round(float(auroc[i]), 4),
                'upcr_auroc': (round(u_out['auroc'], 4)
                               if np.isfinite(u_out['auroc']) else ''),
                'lsml_rel_residual': round(float(rel[i]), 5),
                'upcr_k1_residual': round(u_res, 5),
            })
        if len(res_list) >= MIN_UNITS and np.std(res_list) > 1e-12:
            r = spearmanr(res_list, auc_list).statistic
            rows_ws.append({'domain': ctx.domain, 'cell': ctx.cell_key,
                            'size': int(s), 'n': len(res_list),
                            'objective': 'upcr_k1',
                            'spearman': round(float(r), 4)})


def summarize(rows_ws, rows_router):
    """Aggregate to per-objective / per-domain verdicts (pre-registered)."""
    import pandas as pd
    ws = pd.DataFrame(rows_ws)
    summary = []
    for (obj, dom), g in ws.groupby(['objective', 'domain']):
        summary.append(_verdict_row(obj, dom, g['spearman'].to_numpy()))
    for obj, g in ws.groupby('objective'):
        summary.append(_verdict_row(obj, 'ALL', g['spearman'].to_numpy()))

    rt = pd.DataFrame(rows_router)
    rt = rt[rt['upcr_auroc'] != '']
    rt['upcr_auroc'] = rt['upcr_auroc'].astype(float)
    rt['route_lsml'] = rt['lsml_rel_residual'] <= rt['upcr_k1_residual']
    rt['lsml_wins'] = rt['lsml_auroc'] >= rt['upcr_auroc']
    rt['router_correct'] = rt['route_lsml'] == rt['lsml_wins']
    router_summary = []
    groups = [('ALL', rt)] + [(d, g) for d, g in rt.groupby('domain')]
    for dom, g in groups:
        acc = float(g['router_correct'].mean())
        base = max(float(g['lsml_wins'].mean()), 1 - float(g['lsml_wins'].mean()))
        router_summary.append({
            'domain': dom, 'n_subsets': len(g),
            'router_acc': round(acc, 4),
            'best_constant_acc': round(base, 4),
            'delta': round(acc - base, 4),
            'lsml_win_rate': round(float(g['lsml_wins'].mean()), 4),
            'verdict': 'USEFUL' if acc > base + 0.03 else 'NOT-USEFUL',
        })
    return pd.DataFrame(summary), rt, pd.DataFrame(router_summary)


def _verdict_row(obj, dom, sp):
    sp = sp[np.isfinite(sp)]
    med = float(np.median(sp)) if len(sp) else float('nan')
    frac_neg = float(np.mean(sp < 0)) if len(sp) else float('nan')
    admissible = (len(sp) >= 10 and med <= -0.10 and frac_neg >= 0.60)
    return {'objective': obj, 'domain': dom, 'n_units': len(sp),
            'median_spearman': round(med, 4) if np.isfinite(med) else np.nan,
            'frac_negative': round(frac_neg, 3) if np.isfinite(frac_neg) else np.nan,
            'verdict': 'ADMISSIBLE' if admissible else 'NOT-ADMISSIBLE'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', default=os.environ.get('HD_DATA_ROOT', REPO_DIR))
    ap.add_argument('--npz-dir', default=None)
    ap.add_argument('--out-dir',
                    default=os.path.join(REPO_DIR, 'results', 'selector_bench'))
    ap.add_argument('--per-size', type=int, default=PER_SIZE)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--domains', default=None)
    args = ap.parse_args()

    npz_dir = args.npz_dir or os.path.join(args.data_root, 'results', 'subset_sweep')
    os.makedirs(args.out_dir, exist_ok=True)
    domains = args.domains.split(',') if args.domains else None

    rows_ws, rows_router = [], []
    t0 = time.time()
    n_cells = 0
    for ctx in iter_prepared_cells(args.data_root, 'h16', domains=domains):
        table = load_npz_table(npz_dir, ctx.domain, ctx.cell_key)
        if table is None:
            print(f"[admis] {ctx.domain}/{ctx.cell_key}: no npz — skipped")
            continue
        rng = np.random.default_rng([args.seed, n_cells])
        analyze_cell(ctx, table, rng, rows_ws, rows_router,
                     per_size=args.per_size)
        n_cells += 1
        print(f"[admis] {ctx.domain}/{ctx.cell_key} done "
              f"({time.time() - t0:.0f}s elapsed)")

    summary, router_detail, router_summary = summarize(rows_ws, rows_router)

    import pandas as pd
    pd.DataFrame(rows_ws).to_csv(
        os.path.join(args.out_dir, 'admissibility_within_size.csv'), index=False)
    summary.to_csv(
        os.path.join(args.out_dir, 'admissibility_summary.csv'), index=False)
    router_detail.to_csv(
        os.path.join(args.out_dir, 'admissibility_router.csv'), index=False)
    router_summary.to_csv(
        os.path.join(args.out_dir, 'admissibility_router_summary.csv'), index=False)

    print(f"\n=== objective admissibility ({n_cells} cells, "
          f"{time.time() - t0:.0f}s) ===")
    print(summary[summary['domain'] == 'ALL'].to_string(index=False))
    print("\nper-domain (ADMISSIBLE rows only):")
    adm = summary[(summary['domain'] != 'ALL')
                  & (summary['verdict'] == 'ADMISSIBLE')]
    print(adm.to_string(index=False) if len(adm) else "  (none)")
    print("\n=== router validity ===")
    print(router_summary.to_string(index=False))


if __name__ == '__main__':
    main()
