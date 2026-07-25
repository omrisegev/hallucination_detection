#!/usr/bin/env python
"""
bench_seven_arms.py — Step 3: the D1/D2 ablation across the 25 in-scope cells.

Scored with the CANONICAL path (`selector_bench.eval_subset_flex`: lsml_continuous
fusion + label-free anchor_orient + raw AUROC), so every number here is directly
comparable to results/selector_bench/comparison_inscope.csv.

ARMS
  1 ref.GOOD_5        fixed reference subset
  2 ref.GOOD_6        fixed reference subset
  3 a6.pl_dufs        un-pruned gate selector (previous selector of record)
  4 a6.pruned_dufs    fixed K=15 gate selector (post-mu_sel-fix baseline)
  5 D1_alone          DUFS gate ranking + knee-adaptive K   (negative control)
  6 D2_alone          PL-MRMR ranking + fixed K=15
  7 D1_D2             PL-MRMR ranking + knee-adaptive K

Arms 3-7 are RE-RUN rather than lifted from existing CSVs: Step 2.1 changed the
pseudo-label seed rule to GOOD_6, so the older a6 numbers are not comparable.

D1 already failed its own validation (Step 2.2: the residual predicts AUROC but
NOT the optimal K; the letter's elbow rule scored r_s=+0.007, p=0.975). Arms 5
and 7 are kept as documented controls, not candidates.

Output:
  results/advisor_inscope/seven_arm_comparison.{csv,html}
  results/advisor_inscope/seven_arm_summary.csv
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

os.environ.setdefault('A6_K_RULE', 'knee')      # D1 uses its best-measured rule

from inscope_cells import GROUP
from spectral_utils.selector_bench import eval_subset_flex
from spectral_utils.subset_sweep import GOOD_5, GOOD_6
from spectral_utils.selectors.adaptive_k import predict_k
from spectral_utils.selectors.a6_pseudolabel_gates import (
    a6_pseudolabel_gates, _seed_cols, _pseudo_label, _corr_with,
    _plmrmr_order, MRMR_ALPHA,
)
from compare_anchor_quality import load_all_inscope_cells

AI_DIR = os.path.join(REPO, "results", "advisor_inscope")
os.makedirs(AI_DIR, exist_ok=True)

FIXED_K = 15
K_RULE = 'knee'
ARMS = ['ref.GOOD_5', 'ref.GOOD_6', 'a6.pl_dufs', 'a6.pruned_dufs',
        'D1_alone', 'D2_alone', 'D1_D2']


class Ctx:
    """Minimal labeled context for eval_subset_flex."""
    def __init__(self, u, labels):
        self.V = np.asarray(u.V, dtype=np.float64)
        self.anchor = u.anchor
        self.labels = labels
        self.pool = u.pool
        self.pool_bits = u.pool_bits


def _named_cols(pool, names):
    return [pool.index(f) for f in names if f in pool]


def _gate_ranking(diag, s_cols, p):
    """Seeds first, then selectable columns by learned gate value (desc)."""
    gates = diag.get('feat_gate_means_learned') or []
    scored = [(c, gates[c]) for c in range(min(p, len(gates)))
              if gates[c] is not None]
    scored.sort(key=lambda t: -t[1])
    return [int(c) for c in s_cols] + [int(c) for c, _ in scored]


def run_cell(ck, cdata):
    u = cdata['unlabeled']
    labels = cdata['labels']
    ctx = Ctx(u, labels)
    V = ctx.V
    p = V.shape[1]
    rng = np.random.default_rng(abs(hash(ck)) % (2 ** 31))

    subsets, notes = {}, {}

    subsets['ref.GOOD_5'] = _named_cols(u.pool, GOOD_5)
    subsets['ref.GOOD_6'] = _named_cols(u.pool, GOOD_6)

    # --- one a6 run supplies arms 3, 4 and the gate ranking for arm 5 -------
    try:
        res = a6_pseudolabel_gates(u, rng, cache=None)
        by_var = {r['variant']: r for r in res}
        for v in ('a6.pl_dufs', 'a6.pruned_dufs'):
            r = by_var.get(v)
            if r is not None and not r.get('fallback'):
                subsets[v] = [int(c) for c in r['cols']]
                notes[v] = 'ok'
            else:
                notes[v] = 'fallback'
        diag = (by_var.get('a6.pl_dufs') or {}).get('diag', {})
        s_cols, _ = _seed_cols(u)
        grank = _gate_ranking(diag, s_cols, p)
        if len(grank) >= 3:
            k1 = predict_k(V, grank, rule=K_RULE)
            subsets['D1_alone'] = grank[:k1]
            notes['D1_alone'] = f'k={k1}'
    except Exception as e:
        notes['a6'] = f'a6 failed: {e}'

    # --- PL-MRMR ranking for arms 6 and 7 ----------------------------------
    try:
        s_cols, _ = _seed_cols(u)
        y_hat, _ = _pseudo_label(u, s_cols)
        sel = np.array([c for c in range(p) if c not in set(s_cols)], dtype=np.int64)
        if len(sel) >= 3:
            agree = _corr_with(V[:, sel], y_hat)
            order = _plmrmr_order(V[:, sel], agree, alpha=MRMR_ALPHA)
            mrank = [int(c) for c in s_cols] + [int(sel[j]) for j in order]
            subsets['D2_alone'] = mrank[:min(FIXED_K, len(mrank))]
            k2 = predict_k(V, mrank, rule=K_RULE)
            subsets['D1_D2'] = mrank[:k2]
            notes['D1_D2'] = f'k={k2}'
    except Exception as e:
        notes['mrmr'] = f'mrmr failed: {e}'

    row = {'cell': ck, 'group': GROUP.get(ck, '?'), 'p': p}
    for arm in ARMS:
        cols = subsets.get(arm)
        if not cols or len(set(cols)) < 3:
            row[f'auc_{arm}'] = None
            row[f'n_{arm}'] = 0
            continue
        try:
            r = eval_subset_flex(ctx, sorted(set(cols)))
            row[f'auc_{arm}'] = round(float(r['auroc']), 4)
            row[f'n_{arm}'] = int(r['size'])
        except Exception as e:
            row[f'auc_{arm}'] = None
            row[f'n_{arm}'] = 0
            notes[arm] = f'score failed: {e}'
    row['notes'] = json.dumps(notes) if notes else ''
    return row


def summarize(rows):
    out = []
    for arm in ARMS:
        col = f'auc_{arm}'
        vals = [r[col] for r in rows if r.get(col) is not None]
        qa = [r[col] for r in rows if r['group'] == 'QA' and r.get(col) is not None]
        ma = [r[col] for r in rows if r['group'] == 'math' and r.get(col) is not None]
        sizes = [r[f'n_{arm}'] for r in rows if r.get(col) is not None]
        if not vals:
            continue
        d = {'arm': arm, 'n_cells': len(vals),
             'macro_all': round(float(np.mean(vals)), 4),
             'macro_qa': round(float(np.mean(qa)), 4) if qa else None,
             'macro_math': round(float(np.mean(ma)), 4) if ma else None,
             'mean_size': round(float(np.mean(sizes)), 2)}
        for ref in ('ref.GOOD_5', 'a6.pl_dufs'):
            pair = [(r[col], r[f'auc_{ref}']) for r in rows
                    if r.get(col) is not None and r.get(f'auc_{ref}') is not None]
            if arm == ref or len(pair) < 5:
                d[f'delta_vs_{ref}'] = 0.0 if arm == ref else None
                d[f'p_vs_{ref}'] = None
                continue
            a = np.array([x for x, _ in pair]); b = np.array([y for _, y in pair])
            d[f'delta_vs_{ref}'] = round(float(np.mean(a - b)), 4)
            try:
                d[f'p_vs_{ref}'] = round(float(wilcoxon(a, b).pvalue), 5)
            except ValueError:
                d[f'p_vs_{ref}'] = None
        out.append(d)
    out.sort(key=lambda d: -d['macro_all'])
    return out


def build_html(rows, summary):
    def f(v, nd=4):
        return '-' if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    srows = []
    for d in summary:
        star = ' &#11088;' if d['arm'] == summary[0]['arm'] else ''
        def sig(pv, dv):
            if pv is None or dv is None:
                return ''
            if pv < 0.05:
                return ' <b style="color:%s">%s</b>' % (
                    '#15803d' if dv > 0 else '#b91c1c', '&#10003;' if dv > 0 else '&#10007;')
            return ' <span style="color:#94a3b8">n.s.</span>'
        srows.append(
            f"<tr><td><b>{d['arm']}</b>{star}</td><td><b>{f(d['macro_all'])}</b></td>"
            f"<td>{f(d['macro_qa'])}</td><td>{f(d['macro_math'])}</td>"
            f"<td>{f(d['mean_size'],2)}</td>"
            f"<td>{f(d.get('delta_vs_ref.GOOD_5'))}{sig(d.get('p_vs_ref.GOOD_5'), d.get('delta_vs_ref.GOOD_5'))}</td>"
            f"<td>{f(d.get('p_vs_ref.GOOD_5'),5)}</td>"
            f"<td>{f(d.get('delta_vs_a6.pl_dufs'))}{sig(d.get('p_vs_a6.pl_dufs'), d.get('delta_vs_a6.pl_dufs'))}</td>"
            f"<td>{f(d.get('p_vs_a6.pl_dufs'),5)}</td></tr>")

    hdr = ''.join(f"<th>{a}</th>" for a in ARMS)
    crows = []
    for r in sorted(rows, key=lambda x: (x['group'], x['cell'])):
        best = max([r[f'auc_{a}'] for a in ARMS if r.get(f'auc_{a}') is not None],
                   default=None)
        tds = ''
        for a in ARMS:
            v = r.get(f'auc_{a}')
            bold = ' style="font-weight:bold;background:#dcfce7"' if (
                v is not None and best is not None and abs(v - best) < 1e-9) else ''
            tds += f"<td{bold}>{f(v)}<br><span style='font-size:10px;color:#64748b'>n={r.get('n_'+a)}</span></td>"
        crows.append(f"<tr><td><b>{r['cell']}</b></td><td>{r['group']}</td>{tds}</tr>")

    top = summary[0]
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Step 3 - 7-Arm D1/D2 Ablation</title>
<style>
 body{{font-family:'Segoe UI',Tahoma,sans-serif;background:#f4f7f9;color:#333;margin:0;padding:20px}}
 h1,h2{{color:#1e293b}}
 .card{{background:#fff;padding:20px;border-radius:10px;box-shadow:0 4px 6px rgba(0,0,0,.05);margin-bottom:22px}}
 table{{width:100%;border-collapse:collapse;margin-top:12px;font-size:13px}}
 th,td{{padding:8px 10px;border:1px solid #e2e8f0;text-align:left}}
 th{{background:#f8fafc;color:#475569}}
 code{{background:#f1f5f9;padding:1px 5px;border-radius:3px}}
 .note{{font-size:13px;color:#64748b}}
</style></head><body>

<div class="card">
  <h1>Step 3 — 7-arm D1/D2 ablation (25 in-scope cells)</h1>
  <p class="note">Scored with the canonical path
    (<code>selector_bench.eval_subset_flex</code>: L-SML continuous fusion +
    label-free <code>anchor_orient</code> + raw AUROC), so these are directly
    comparable to <code>comparison_inscope.csv</code>. Pseudo-label seeds =
    GOOD_6 (Step 2.1). D1 rule = <code>{K_RULE}</code>, fixed K = {FIXED_K},
    mRMR alpha = {MRMR_ALPHA}.</p>
  <p class="note"><b>D1 context:</b> Step 2.2 showed no label-free rule predicts
    oracle-K (the advisor draft's elbow rule scored r<sub>s</sub>=+0.007, p=0.975).
    Arms <code>D1_alone</code> and <code>D1_D2</code> are documented controls,
    not candidates.</p>
  <p>Top arm by macro AUROC: <b>{top['arm']}</b> at <b>{top['macro_all']}</b>
     (QA {top['macro_qa']}, Math {top['macro_math']}).</p>
</div>

<div class="card">
  <h2>Arm summary</h2>
  <p class="note">Deltas are paired per-cell means; p-values are Wilcoxon
     signed-rank. &#10003; = significant gain, &#10007; = significant loss.</p>
  <table><thead><tr><th>Arm</th><th>Macro (25)</th><th>QA (10)</th><th>Math (15)</th>
   <th>Mean size</th><th>&Delta; vs GOOD_5</th><th>p</th>
   <th>&Delta; vs pl_dufs</th><th>p</th></tr></thead>
   <tbody>{''.join(srows)}</tbody></table>
</div>

<div class="card">
  <h2>Per-cell AUROC</h2>
  <table><thead><tr><th>Cell</th><th>Group</th>{hdr}</tr></thead>
   <tbody>{''.join(crows)}</tbody></table>
</div>
</body></html>
"""


def main():
    cells = load_all_inscope_cells()
    print(f"\n--- Step 3: 7-arm ablation over {len(cells)} cells ---", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_cell, ck, cells[ck]): ck for ck in sorted(cells)}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            print(f"  [{i}/{len(futs)}] {r['cell']:38s} " +
                  ' '.join(f"{a.split('.')[-1][:9]}={str(r.get('auc_'+a)):7s}"
                           for a in ARMS), flush=True)

    rows.sort(key=lambda r: (r['group'], r['cell']))
    summary = summarize(rows)

    csv_path = os.path.join(AI_DIR, "seven_arm_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    sum_path = os.path.join(AI_DIR, "seven_arm_summary.csv")
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    html_path = os.path.join(AI_DIR, "seven_arm_comparison.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html(rows, summary))

    print("\n=== 7-arm summary ===", flush=True)
    print(f"  {'arm':16s} {'macro':>7s} {'QA':>7s} {'math':>7s} {'size':>6s} "
          f"{'dG5':>8s} {'pG5':>8s} {'dPL':>8s} {'pPL':>8s}", flush=True)
    for d in summary:
        g = lambda k: (float('nan') if d.get(k) is None else d[k])
        print(f"  {d['arm']:16s} {d['macro_all']:7.4f} "
              f"{g('macro_qa'):7.4f} {g('macro_math'):7.4f} {d['mean_size']:6.2f} "
              f"{g('delta_vs_ref.GOOD_5'):8.4f} {g('p_vs_ref.GOOD_5'):8.4f} "
              f"{g('delta_vs_a6.pl_dufs'):8.4f} {g('p_vs_a6.pl_dufs'):8.4f}",
              flush=True)

    print(f"\n  CSV  : {csv_path}\n  SUM  : {sum_path}\n  HTML : {html_path}",
          flush=True)
    print("\nStep 3 complete. STOPPING for final review.", flush=True)


if __name__ == '__main__':
    main()
