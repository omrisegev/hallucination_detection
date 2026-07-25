#!/usr/bin/env python
"""
d2_loco_ksweep.py — corrected D2 budget sweep with honest held-out K selection.

WHAT WENT WRONG IN THE FIRST SWEEP
----------------------------------
The PL-MRMR ranking is built seeds-first, and the seeds ARE GOOD_6 (Step 2.1).
Verified on 25/25 cells: the ranking's top-5 is exactly GOOD_5 and its top-6 is
exactly GOOD_6. So every "D2 (K<=6)" number in the first sweep was the fixed
baseline wearing the selector's name, and "K_QA=6 -> 0.7274" was just GOOD_6's QA
macro. PL-MRMR contributes nothing until K >= 7.

The first sweep also picked K_QA and K_Math by reading which budget maximised each
domain's macro ON THE 25 CELLS BEING REPORTED -- hyperparameter selection on the
evaluation set, over 16 configurations, for a +0.15pp margin.

WHAT THIS SCRIPT DOES INSTEAD
-----------------------------
1. Two ranking variants, so PL-MRMR's contribution is isolated rather than
   masked by the seed prior:
       D2_seeded : seeds + mRMR fill, K >= 7   (the deployable selector)
       D2_pure   : mRMR only, seeds excluded   (the mechanism on its own)
2. LOCO CV budget selection: for each held-out cell, K is chosen on the OTHER
   cells only (per-domain and, as a control, a single global K). The held-out
   score is what gets reported.
3. Paired Wilcoxon for the specific claim "Math at K=18 beats GOOD_6".

Scored with the canonical path (selector_bench.eval_subset_flex), so every number
is comparable to comparison_inscope.csv.

Outputs: results/advisor_inscope/d2_loco_ksweep.{csv,html} + _summary.csv
"""

import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import wilcoxon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from inscope_cells import GROUP
from spectral_utils.selector_bench import eval_subset_flex
from spectral_utils.subset_sweep import GOOD_5, GOOD_6
from spectral_utils.selectors.a6_pseudolabel_gates import (
    _seed_cols, _pseudo_label, _corr_with, _plmrmr_order, MRMR_ALPHA,
)
from compare_anchor_quality import load_all_inscope_cells

AI_DIR = os.path.join(REPO, "results", "advisor_inscope")
os.makedirs(AI_DIR, exist_ok=True)

K_SEEDED = list(range(7, 21))    # K<=6 is tautologically the seed set: excluded
K_PURE = list(range(3, 21))
VARIANTS = ('D2_seeded', 'D2_pure')


class Ctx:
    def __init__(self, u, labels):
        self.V = np.asarray(u.V, dtype=np.float64)
        self.anchor = u.anchor
        self.labels = labels
        self.pool = u.pool
        self.pool_bits = u.pool_bits


def _auc(ctx, cols):
    cols = sorted(set(int(c) for c in cols))
    if len(cols) < 3:
        return None
    try:
        return round(float(eval_subset_flex(ctx, cols)['auroc']), 4)
    except Exception:
        return None


def run_cell(ck, cdata):
    u = cdata['unlabeled']
    ctx = Ctx(u, cdata['labels'])
    V, p = ctx.V, ctx.V.shape[1]

    s_cols, _ = _seed_cols(u)
    y_hat, _ = _pseudo_label(u, s_cols)
    sel = np.array([c for c in range(p) if c not in set(s_cols)], dtype=np.int64)
    if len(sel) < 3:
        return None
    order = _plmrmr_order(V[:, sel], _corr_with(V[:, sel], y_hat), alpha=MRMR_ALPHA)
    rank_pure = [int(sel[j]) for j in order]
    rank_seeded = [int(c) for c in s_cols] + rank_pure

    row = {'cell': ck, 'group': GROUP.get(ck, '?'), 'p': p,
           'n_seeds': len(s_cols)}
    row['auc_GOOD_5'] = _auc(ctx, [u.pool.index(f) for f in GOOD_5 if f in u.pool])
    row['auc_GOOD_6'] = _auc(ctx, [u.pool.index(f) for f in GOOD_6 if f in u.pool])
    for k in K_SEEDED:
        row[f'D2_seeded_k{k}'] = _auc(ctx, rank_seeded[:k]) if k <= len(rank_seeded) else None
    for k in K_PURE:
        row[f'D2_pure_k{k}'] = _auc(ctx, rank_pure[:k]) if k <= len(rank_pure) else None
    return row


def _ks(variant):
    return K_SEEDED if variant == 'D2_seeded' else K_PURE


def _mean(rows, variant, k, group=None):
    vals = [r[f'{variant}_k{k}'] for r in rows
            if (group is None or r['group'] == group) and r.get(f'{variant}_k{k}') is not None]
    return float(np.mean(vals)) if vals else None


def loco_eval(rows, variant, per_domain=True):
    """Leave-one-cell-out: K chosen on the other 24 cells only."""
    out = []
    for r in rows:
        others = [o for o in rows if o['cell'] != r['cell']]
        pool = [o for o in others if o['group'] == r['group']] if per_domain else others
        best_k, best_v = None, -1.0
        for k in _ks(variant):
            vals = [o[f'{variant}_k{k}'] for o in pool if o.get(f'{variant}_k{k}') is not None]
            if not vals:
                continue
            m = float(np.mean(vals))
            if m > best_v:
                best_k, best_v = k, m
        held = r.get(f'{variant}_k{best_k}') if best_k else None
        out.append({'cell': r['cell'], 'group': r['group'],
                    'k_chosen': best_k, 'auc': held})
    return out


def summarize(rows):
    summary = []

    # fixed references
    for name in ('GOOD_5', 'GOOD_6'):
        vals = [r[f'auc_{name}'] for r in rows if r.get(f'auc_{name}') is not None]
        qa = [r[f'auc_{name}'] for r in rows if r['group'] == 'QA' and r.get(f'auc_{name}')]
        ma = [r[f'auc_{name}'] for r in rows if r['group'] == 'math' and r.get(f'auc_{name}')]
        summary.append({'arm': f'ref.{name}', 'kind': 'fixed', 'k': len(GOOD_5) if name == 'GOOD_5' else len(GOOD_6),
                        'macro_all': round(float(np.mean(vals)), 4),
                        'macro_qa': round(float(np.mean(qa)), 4),
                        'macro_math': round(float(np.mean(ma)), 4)})

    # in-sample best K (reported ONLY to show the optimism gap)
    for v in VARIANTS:
        for scope, grp in (('all', None), ('qa', 'QA'), ('math', 'math')):
            best = max(((k, _mean(rows, v, k, grp)) for k in _ks(v)),
                       key=lambda t: (t[1] if t[1] is not None else -1))
            summary.append({'arm': f'{v} in-sample best K ({scope})', 'kind': 'in-sample',
                            'k': best[0],
                            'macro_all': round(best[1], 4) if scope == 'all' else None,
                            'macro_qa': round(best[1], 4) if scope == 'qa' else None,
                            'macro_math': round(best[1], 4) if scope == 'math' else None})

    # honest LOCO CV
    for v in VARIANTS:
        for per_domain in (True, False):
            res = loco_eval(rows, v, per_domain=per_domain)
            vals = [x['auc'] for x in res if x['auc'] is not None]
            qa = [x['auc'] for x in res if x['group'] == 'QA' and x['auc'] is not None]
            ma = [x['auc'] for x in res if x['group'] == 'math' and x['auc'] is not None]
            ks = [x['k_chosen'] for x in res if x['k_chosen']]
            summary.append({
                'arm': f"{v} LOCO-CV ({'per-domain K' if per_domain else 'global K'})",
                'kind': 'loco', 'k': f"{int(np.median(ks))} (med)",
                'macro_all': round(float(np.mean(vals)), 4),
                'macro_qa': round(float(np.mean(qa)), 4) if qa else None,
                'macro_math': round(float(np.mean(ma)), 4) if ma else None,
                '_res': res})
    return summary


def wilcoxon_math_k18(rows):
    """The specific claim: Math at K=18 beats GOOD_6."""
    pairs = [(r['D2_seeded_k18'], r['auc_GOOD_6']) for r in rows
             if r['group'] == 'math' and r.get('D2_seeded_k18') is not None
             and r.get('auc_GOOD_6') is not None]
    if len(pairs) < 5:
        return None
    a = np.array([x for x, _ in pairs]); b = np.array([y for _, y in pairs])
    try:
        pv = float(wilcoxon(a, b).pvalue)
    except ValueError:
        pv = float('nan')
    return {'n': len(pairs), 'mean_d2': round(float(a.mean()), 4),
            'mean_good6': round(float(b.mean()), 4),
            'delta': round(float((a - b).mean()), 4), 'p': round(pv, 4),
            'wins': int((a > b).sum()), 'losses': int((a < b).sum())}


def build_html(rows, summary, wil):
    def f(v, nd=4):
        return '-' if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    kind_col = {'fixed': '#475569', 'in-sample': '#b45309', 'loco': '#15803d'}
    srows = []
    for d in summary:
        c = kind_col.get(d['kind'], '#333')
        srows.append(
            f"<tr><td style='color:{c}'><b>{d['arm']}</b></td><td>{d['k']}</td>"
            f"<td><b>{f(d['macro_all'])}</b></td><td>{f(d['macro_qa'])}</td>"
            f"<td>{f(d['macro_math'])}</td></tr>")

    # curve table
    crows = []
    for v in VARIANTS:
        for k in _ks(v):
            crows.append(f"<tr><td>{v}</td><td>{k}</td>"
                         f"<td>{f(_mean(rows, v, k))}</td>"
                         f"<td>{f(_mean(rows, v, k, 'QA'))}</td>"
                         f"<td>{f(_mean(rows, v, k, 'math'))}</td></tr>")

    wtxt = ('-' if wil is None else
            f"Math, D2_seeded@K=18 ({wil['mean_d2']}) vs GOOD_6 ({wil['mean_good6']}): "
            f"delta {wil['delta']:+.4f}, Wilcoxon p = <b>{wil['p']}</b>, "
            f"{wil['wins']} wins / {wil['losses']} losses of {wil['n']} cells "
            f"&rarr; <b>{'SIGNIFICANT' if wil['p'] < 0.05 else 'NOT significant'}</b>")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>D2 corrected K-sweep + LOCO CV</title>
<style>
 body{{font-family:'Segoe UI',Tahoma,sans-serif;background:#f4f7f9;color:#333;margin:0;padding:20px}}
 h1,h2{{color:#1e293b}}
 .card{{background:#fff;padding:20px;border-radius:10px;box-shadow:0 4px 6px rgba(0,0,0,.05);margin-bottom:22px}}
 table{{width:100%;border-collapse:collapse;margin-top:12px;font-size:13px}}
 th,td{{padding:8px 10px;border:1px solid #e2e8f0;text-align:left}}
 th{{background:#f8fafc;color:#475569}}
 code{{background:#f1f5f9;padding:1px 5px;border-radius:3px}}
 .note{{font-size:13px;color:#64748b}}
 .warn{{background:#fef3c7;border-left:6px solid #b45309;padding:14px;border-radius:8px}}
</style></head><body>

<div class="card">
  <h1>D2 corrected budget sweep + honest LOCO CV</h1>
  <div class="warn"><b>Why this supersedes the first sweep.</b> The PL-MRMR ranking is
   seeds-first and the seeds are GOOD_6, so on <b>25/25 cells</b> the ranking's top-5 is
   exactly GOOD_5 and its top-6 is exactly GOOD_6. Every <code>D2 (K&le;6)</code> figure in
   the earlier sweep was a fixed baseline relabelled, and "K_QA=6 &rarr; 0.7274" was simply
   GOOD_6's QA macro. PL-MRMR does nothing until K&ge;7, so K&le;6 is excluded here.
   The earlier 0.7609 also picked K per domain on the very cells it reported.</div>
  <p class="note" style="margin-top:14px"><b>Rows are colour-coded:</b>
   <span style="color:#475569">fixed reference</span> ·
   <span style="color:#b45309">in-sample best K (optimistic, shown only to expose the gap)</span> ·
   <span style="color:#15803d">LOCO CV (honest, held-out)</span>.</p>
</div>

<div class="card">
  <h2>Headline comparison</h2>
  <table><thead><tr><th>Arm</th><th>K</th><th>Macro (25)</th><th>QA (10)</th>
   <th>Math (15)</th></tr></thead><tbody>{''.join(srows)}</tbody></table>
</div>

<div class="card">
  <h2>Paired significance test</h2>
  <p>{wtxt}</p>
</div>

<div class="card">
  <h2>Full budget curves</h2>
  <table><thead><tr><th>Variant</th><th>K</th><th>Macro</th><th>QA</th><th>Math</th></tr></thead>
   <tbody>{''.join(crows)}</tbody></table>
</div>
</body></html>
"""


def main():
    cells = load_all_inscope_cells()
    print(f"\n--- Corrected D2 K-sweep + LOCO CV over {len(cells)} cells ---", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_cell, ck, cells[ck]): ck for ck in sorted(cells)}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                rows.append(r)
                print(f"  [{i}/{len(futs)}] {r['cell']:38s} "
                      f"g6={r['auc_GOOD_6']} k7={r.get('D2_seeded_k7')} "
                      f"k15={r.get('D2_seeded_k15')} k18={r.get('D2_seeded_k18')}",
                      flush=True)
    rows.sort(key=lambda r: (r['group'], r['cell']))

    summary = summarize(rows)
    wil = wilcoxon_math_k18(rows)

    with open(os.path.join(AI_DIR, "d2_loco_ksweep.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(AI_DIR, "d2_loco_ksweep_summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        flat = [{k: v for k, v in d.items() if k != '_res'} for d in summary]
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader(); w.writerows(flat)
    html_path = os.path.join(AI_DIR, "d2_loco_ksweep.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html(rows, summary, wil))

    print("\n=== budget curves (mean AUROC) ===", flush=True)
    for v in VARIANTS:
        print(f"  -- {v} --", flush=True)
        for k in _ks(v):
            print(f"     K={k:2d}  macro={_mean(rows,v,k):.4f}  "
                  f"QA={_mean(rows,v,k,'QA'):.4f}  math={_mean(rows,v,k,'math'):.4f}",
                  flush=True)

    print("\n=== summary ===", flush=True)
    print(f"  {'arm':46s}{'K':>10s}{'macro':>9s}{'QA':>9s}{'math':>9s}", flush=True)
    for d in summary:
        g = lambda k: ('   -   ' if d.get(k) is None else f"{d[k]:.4f}")
        print(f"  {d['arm']:46s}{str(d['k']):>10s}{g('macro_all'):>9s}"
              f"{g('macro_qa'):>9s}{g('macro_math'):>9s}", flush=True)

    if wil:
        print(f"\n=== Wilcoxon: Math D2_seeded@K=18 vs GOOD_6 ===", flush=True)
        print(f"  {wil['mean_d2']} vs {wil['mean_good6']}  delta={wil['delta']:+.4f}  "
              f"p={wil['p']}  wins={wil['wins']} losses={wil['losses']} n={wil['n']}",
              flush=True)
        print(f"  -> {'SIGNIFICANT' if wil['p'] < 0.05 else 'NOT significant'}", flush=True)

    print(f"\n  HTML : {html_path}", flush=True)
    print("\nCorrected sweep complete. STOPPING for review.", flush=True)


if __name__ == '__main__':
    main()
