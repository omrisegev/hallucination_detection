#!/usr/bin/env python
"""
exp09_add_test.py — the ADD test: offering the two strongest never-used views.

`topk_tail_mass` and `renyi_entropy_2` rank #1 and #5 of 30 by informativeness
(|AUROC - 0.5|) yet had never appeared in ANY scored fixed subset — the gap
flagged at PROGRESS.md:548 and never closed. Removal has three independent
negatives (WS3 LOCO, the pool-size experiment, feature_inclusion_audit_c46);
addition was simply never tried, so this is the mirror-image question.

The six variants are PRE-REGISTERED together in `subset_sweep.ADD_VARIANTS`
before any was scored, so the best of them reads as a ceiling, not a discovery.

Scoring goes through `selector_bench.eval_subset_flex` — the same function the
bench calls for every reference macro — so these numbers are directly comparable
to the leaderboard and are reproduced independently by
`run_eval_pipeline.py --inscope-only` once it benches the reference_macros family.

The honest bar is `ref.LOCO_5` (0.7705), not `ref.GOOD_6` (0.7594): LOCO_5 already
CONTAINS `topk_tail_mass`, picked independently by the Step-195 exhaustive LOCO
search. Reported against both.

Run:  python scripts/upcr_study/exp09_add_test.py
Out:  results/upcr_study/09_add_test/{per_cell.csv,summary.csv,index.html}
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                  # noqa: E402

from spectral_utils.selector_bench import (                         # noqa: E402
    iter_prepared_cells, eval_subset_flex)
from spectral_utils.selectors.reference_macros import MACROS        # noqa: E402
from spectral_utils.subset_sweep import ADD_VARIANTS                # noqa: E402
from inscope_cells import INSCOPE, GROUP                            # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASELINES = ['ref.GOOD_5', 'ref.GOOD_6', 'ref.LOCO_5']
VARIANTS = BASELINES + [f'ref.{k}' for k in ADD_VARIANTS]

# Anti-regression anchors — a drift here means the data is not the data our
# published numbers came from, so nothing downstream is worth reporting.
ANCHORS = {'ref.GOOD_5': 0.7519, 'ref.GOOD_6': 0.7594, 'ref.LOCO_5': 0.7705}


def collect():
    rows = []
    for ctx in iter_prepared_cells(REPO, 'c46', ['repgrid'], None):
        if ctx.cell_key not in INSCOPE:
            continue
        for v in VARIANTS:
            names = MACROS[v]
            cols = [ctx.pool.index(f) for f in names if f in ctx.pool]
            # every ADD variant is all-or-nothing: a partial mask IS the base
            # subset, which would silently duplicate the baseline row.
            if len(cols) < len(names) or len(cols) < 3:
                continue
            r = eval_subset_flex(ctx, cols)
            rows.append(dict(cell=ctx.cell_key, group=GROUP.get(ctx.cell_key, ''),
                             variant=v, size=len(cols), auroc=r['auroc'],
                             K=r['K'], flipped=r['flipped']))
    return pd.DataFrame(rows)


def paired(p, v, bar):
    """Paired delta of v against bar on their SHARED cells (coverage-matched —
    LOCO_5 scores 24/25, and comparing raw macros across different cell sets is
    the bug run_eval_pipeline's delta_vs_current_best_ref_MATCHED exists to fix)."""
    if v == bar or bar not in p.columns or v not in p.columns:
        return None
    both = p[[v, bar]].dropna()
    if len(both) < 5:
        return None
    d = both[v] - both[bar]
    try:
        pv = float(wilcoxon(both[v], both[bar]).pvalue)
    except Exception:
        pv = float('nan')
    return dict(delta_pp=float(100 * d.mean()), W=int((d > 0).sum()),
                L=int((d < 0).sum()), p=pv, n=int(len(both)))


def main():
    out = S.outdir('09_add_test')
    df = collect()
    df.to_csv(os.path.join(out, 'per_cell.csv'), index=False)

    p = df.pivot(index='cell', columns='variant', values='auroc')

    for v, want in ANCHORS.items():
        got = float(p[v].mean())
        if abs(got - want) > 2e-3:
            raise SystemExit(
                f"ANCHOR FAILED: {v} macro = {got:.4f}, expected {want:.4f}. "
                "Refusing to report an ADD result on data that does not "
                "reproduce the published baselines.")
    print("  anchors OK: " + ", ".join(
        f"{v}={float(p[v].mean()):.4f}" for v in ANCHORS))

    rows = []
    for v in VARIANTS:
        if v not in p.columns:
            continue
        s = p[v]
        qa = [c for c in s.index if GROUP.get(c) == 'QA']
        ma = [c for c in s.index if GROUP.get(c) == 'math']
        r = dict(variant=v, size=int(df[df.variant == v]['size'].max()),
                 cells=int(s.notna().sum()), macro=float(s.mean()),
                 QA=float(s[qa].mean()), math=float(s[ma].mean()))
        for bar in ('ref.GOOD_5', 'ref.GOOD_6', 'ref.LOCO_5'):
            st = paired(p, v, bar)
            r[f'vs_{bar[4:]}'] = (
                f"{st['delta_pp']:+.2f}pp {st['W']}W/{st['L']}L p={st['p']:.3f}"
                if st else '-')
        rows.append(r)

    summ = pd.DataFrame(rows).sort_values('macro', ascending=False)
    summ.to_csv(os.path.join(out, 'summary.csv'), index=False)
    pd.set_option('display.width', 250)
    print()
    print(summ.to_string(index=False))

    best_add = summ[summ.variant.isin([f'ref.{k}' for k in ADD_VARIANTS])].iloc[0]
    S.write_page(
        os.path.join(out, 'index.html'),
        'The ADD test — the two strongest never-used views',
        'U-PCR study, exp09 — does offering topk_tail_mass / renyi_entropy_2 to '
        'the fixed subsets help?',
        [f"Best of six pre-registered ADD variants: <b>{best_add['variant']}</b> at "
         f"<b>{best_add['macro']:.4f}</b> — {best_add['vs_GOOD_6']} vs GOOD_6 and "
         f"{best_add['vs_LOCO_5']} vs LOCO_5.",
         "<code>topk_tail_mass</code> and <code>renyi_entropy_2</code> rank #1 and "
         "#5 of 30 by individual informativeness, yet neither improves a "
         "hand-curated subset. High individual informativeness does not imply "
         "additive value — the information is already covered.",
         "The honest bar is <b>LOCO_5</b> (0.7705), not GOOD_6: LOCO_5 already "
         "contains <code>topk_tail_mass</code>, picked independently by the "
         "Step-195 exhaustive LOCO search. The view is valuable; adding it to a "
         "subset that already covers it is not.",
         "All six were pre-registered together, so the best is a ceiling."],
        S.html_table(
            ['subset', 'size', 'cells', 'macro', 'QA', 'math',
             'vs GOOD_5', 'vs GOOD_6', 'vs LOCO_5'],
            [[r['variant'], r['size'], r['cells'], f"{r['macro']:.4f}",
              f"{r['QA']:.4f}", f"{r['math']:.4f}",
              r['vs_GOOD_5'], r['vs_GOOD_6'], r['vs_LOCO_5']]
             for r in summ.to_dict('records')],
            numeric_cols=(1, 2, 3, 4, 5)))
    print(f"  wrote {os.path.join(out, 'index.html')}")


if __name__ == '__main__':
    main()
