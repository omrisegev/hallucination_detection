#!/usr/bin/env python
"""
validate_adaptive_k.py — Step 2.2 make-or-break check for D1.

THE QUESTION
------------
Step 197 found that the L-SML covariance residual correlates with downstream
AUROC (Spearman r=+0.648). An earlier advisor draft turned that into a claimed
"continuous residual elbow" cutoff, K*_cell = argmax_k (eps(k+1) - eps(k)) --
except no such cutoff existed in the code, and correlating with AUROC is NOT the
same thing as predicting the best subset SIZE.

This script builds the cutoff for real and tests it honestly: for every in-scope
cell, compare each label-free K rule against the ORACLE K (the prefix size that
actually maximises AUROC, computed with labels for validation only).

    If no rule tracks oracle-K, D1 does not work, and we say so.

The ranking the prefixes are taken over is the D2 PL-MRMR order (seeds first,
then pseudo-label-relevance minus redundancy) -- i.e. exactly the order the
combined D1+D2 selector consumes, so this validates the pair as deployed.

Outputs:
  results/advisor_inscope/adaptive_k_validation.csv
  results/advisor_inscope/adaptive_k_validation_rules.csv
  results/advisor_inscope/adaptive_k_validation.html
"""

import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from inscope_cells import GROUP
from spectral_utils.selectors.adaptive_k import validate, spectral_gap, K_MIN, K_MAX
from spectral_utils.selectors.a6_pseudolabel_gates import (
    _seed_cols, _pseudo_label, _corr_with, _plmrmr_order, MRMR_ALPHA,
)
from compare_anchor_quality import load_all_inscope_cells

AI_DIR = os.path.join(REPO, "results", "advisor_inscope")
os.makedirs(AI_DIR, exist_ok=True)

RULES = ('elbow_fwd', 'knee', 'plateau', 'gap_step', 'fixed')


def build_ranking(cell):
    """The D2 PL-MRMR ranking, identical to what the selector computes."""
    V = np.asarray(cell.V, dtype=np.float64)
    p = V.shape[1]
    s_cols, _ = _seed_cols(cell)
    y_hat, _ = _pseudo_label(cell, s_cols)
    sel_cols = np.array([c for c in range(p) if c not in set(s_cols)], dtype=np.int64)
    if len(sel_cols) < 3:
        return None
    agree = _corr_with(V[:, sel_cols], y_hat)
    order = _plmrmr_order(V[:, sel_cols], agree, alpha=MRMR_ALPHA)
    return [int(c) for c in s_cols] + [int(sel_cols[j]) for j in order]


def build_html(rows, summary):
    best = summary[0]
    scatter = {r: [{'x': x['oracle_k'], 'y': x[f'k_{r}'],
                    'label': x['cell'], 'g': x['group']} for x in rows]
               for r in RULES}
    colours = {'elbow_fwd': '#3b82f6', 'knee': '#8b5cf6', 'plateau': '#10b981',
               'gap_step': '#f59e0b', 'fixed': '#ef4444'}

    srows = []
    for d in summary:
        rs = '-' if d['spearman_r'] is None else f"{d['spearman_r']:+.4f}"
        pv = '-' if d['p_value'] is None else f"{d['p_value']:.4f}"
        sig = ' ✓' if (d['p_value'] is not None and d['p_value'] < 0.05) else ''
        srows.append(
            f"<tr><td><b>{d['rule']}</b></td><td>{rs}{sig}</td><td>{pv}</td>"
            f"<td>{d['mean_abs_dk']:.2f}</td><td>{d['median_k']}</td>"
            f"<td><b>{d['macro_auc']}</b></td><td>{d['oracle_macro_auc']}</td>"
            f"<td>{d['auc_gap_to_oracle']}</td></tr>")

    crows = []
    for x in sorted(rows, key=lambda r: (r['group'], r['cell'])):
        ks = ''.join(f"<td>{x[f'k_{r}']}</td>" for r in RULES)
        crows.append(
            f"<tr><td><b>{x['cell']}</b></td><td>{x['group']}</td>"
            f"<td><b>{x['oracle_k']}</b></td>{ks}"
            f"<td>{x['oracle_auc']}</td><td>{x['spectral_gap']}</td></tr>")

    rule_hdr = ''.join(f"<th>{r}</th>" for r in RULES)
    verdict_ok = (best['spearman_r'] is not None and best['spearman_r'] > 0
                  and best['p_value'] is not None and best['p_value'] < 0.05)
    vcol = '#15803d' if verdict_ok else '#b91c1c'
    vbg = '#dcfce7' if verdict_ok else '#fee2e2'
    vtxt = ("D1 VIABLE — a label-free rule tracks oracle-K significantly."
            if verdict_ok else
            "D1 NOT SUPPORTED — no label-free rule tracks oracle-K significantly. "
            "The residual correlates with AUROC but does not predict the optimal size.")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Step 2.2 - Adaptive K* vs Oracle K*</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
 body{{font-family:'Segoe UI',Tahoma,sans-serif;background:#f4f7f9;color:#333;margin:0;padding:20px}}
 h1,h2{{color:#1e293b}}
 .card{{background:#fff;padding:20px;border-radius:10px;box-shadow:0 4px 6px rgba(0,0,0,.05);margin-bottom:22px}}
 table{{width:100%;border-collapse:collapse;margin-top:12px;font-size:13px}}
 th,td{{padding:8px 10px;border:1px solid #e2e8f0;text-align:left}}
 th{{background:#f8fafc;color:#475569}}
 .vbox{{background:{vbg};padding:16px;border-radius:8px;border-left:6px solid {vcol}}}
 .verdict{{font-size:20px;font-weight:bold;color:{vcol}}}
 .chart{{position:relative;height:420px}}
 code{{background:#f1f5f9;padding:1px 5px;border-radius:3px}}
</style></head><body>

<div class="card">
  <h1>Step 2.2 — D1 adaptive K* vs oracle K*</h1>
  <p>Does a <b>label-free</b> rule predict the subset size that actually maximises AUROC?
     Prefixes are taken over the D2 PL-MRMR ranking (seeds first, then pseudo-label
     relevance minus redundancy), k in [{K_MIN}, {K_MAX}].</p>
  <div class="vbox"><div class="verdict">{vtxt}</div>
  <p style="margin:8px 0 0">Best rule: <b>{best['rule']}</b> —
     Spearman r<sub>s</sub> = <b>{best['spearman_r']}</b> (p = {best['p_value']}),
     mean |&Delta;K| = <b>{best['mean_abs_dk']:.2f}</b>,
     macro AUROC <b>{best['macro_auc']}</b> vs oracle {best['oracle_macro_auc']}
     (gap {best['auc_gap_to_oracle']}).</p></div>
</div>

<div class="card">
  <h2>Rule comparison</h2>
  <p style="font-size:13px;color:#64748b"><code>fixed</code> = always K_MAX (the
    a6.pruned_dufs status quo). <code>gap_step</code> = the legacy two-value spectral-gap
    rule. The rest read the residual curve. ✓ marks p &lt; 0.05.</p>
  <table><thead><tr><th>Rule</th><th>Spearman r<sub>s</sub></th><th>p</th>
   <th>mean |&Delta;K|</th><th>median K</th><th>macro AUROC</th>
   <th>oracle macro</th><th>gap</th></tr></thead>
   <tbody>{''.join(srows)}</tbody></table>
</div>

<div class="card">
  <h2>Predicted K* vs oracle K* (perfect prediction = diagonal)</h2>
  <div class="chart"><canvas id="sc"></canvas></div>
</div>

<div class="card">
  <h2>Per-cell detail</h2>
  <table><thead><tr><th>Cell</th><th>Group</th><th>oracle K</th>{rule_hdr}
   <th>oracle AUROC</th><th>spectral gap</th></tr></thead>
   <tbody>{''.join(crows)}</tbody></table>
</div>

<script>
const S = {json.dumps(scatter)}, C = {json.dumps(colours)};
new Chart(document.getElementById('sc'), {{
  type:'scatter',
  data:{{datasets: Object.keys(S).map(r => ({{
      label:r, data:S[r], backgroundColor:C[r], pointRadius:6 }}))
      .concat([{{label:'perfect', type:'line',
        data:[{{x:{K_MIN},y:{K_MIN}}},{{x:{K_MAX},y:{K_MAX}}}],
        borderColor:'#94a3b8', borderDash:[6,6], pointRadius:0, fill:false}}])}},
  options:{{responsive:true,maintainAspectRatio:false,
    scales:{{x:{{title:{{display:true,text:'oracle K*'}},min:{K_MIN-1},max:{K_MAX+1}}},
             y:{{title:{{display:true,text:'predicted K*'}},min:{K_MIN-1},max:{K_MAX+1}}}}},
    plugins:{{tooltip:{{callbacks:{{label:c=>c.raw.label+' ('+c.raw.g+') oracle='+c.raw.x+' pred='+c.raw.y}}}}}}}}
}});
</script>
</body></html>
"""


def main():
    cells = load_all_inscope_cells()
    print(f"\n--- Step 2.2: adaptive-K validation over {len(cells)} cells ---",
          flush=True)

    cdict, rankings = {}, {}
    for ck in sorted(cells):
        u = cells[ck]['unlabeled']
        rank = build_ranking(u)
        if rank is None:
            print(f"  skip {ck}: selectable pool too small", flush=True)
            continue
        rankings[ck] = rank
        cdict[ck] = {'V': np.asarray(cells[ck]['V'], dtype=np.float64),
                     'labels': cells[ck]['labels'],
                     'group': GROUP.get(ck, '?')}
        print(f"  ranked {ck:38s} p={len(rank)}", flush=True)

    rows, summary = validate(cdict, rankings, rules=RULES)

    csv_path = os.path.join(AI_DIR, "adaptive_k_validation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rules_path = os.path.join(AI_DIR, "adaptive_k_validation_rules.csv")
    with open(rules_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    html_path = os.path.join(AI_DIR, "adaptive_k_validation.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html(rows, summary))

    print("\n=== D1 rule vs oracle-K ===", flush=True)
    print(f"  {'rule':11s} {'r_s':>8s} {'p':>8s} {'|dK|':>6s} {'medK':>5s} "
          f"{'macroAUC':>9s} {'oracle':>7s} {'gap':>7s}", flush=True)
    for d in summary:
        rs = float('nan') if d['spearman_r'] is None else d['spearman_r']
        pv = float('nan') if d['p_value'] is None else d['p_value']
        print(f"  {d['rule']:11s} {rs:8.4f} {pv:8.4f} {d['mean_abs_dk']:6.2f} "
              f"{d['median_k']:5d} {d['macro_auc']:9.4f} "
              f"{d['oracle_macro_auc']:7.4f} {d['auc_gap_to_oracle']:7.4f}", flush=True)

    oks = [r['oracle_k'] for r in rows]
    qa = [r['oracle_k'] for r in rows if r['group'] == 'QA']
    ma = [r['oracle_k'] for r in rows if r['group'] == 'math']
    print(f"\n  oracle K: median {int(np.median(oks))} | QA median "
          f"{int(np.median(qa))} | math median {int(np.median(ma))}", flush=True)
    print(f"\n  CSV  : {csv_path}\n  RULES: {rules_path}\n  HTML : {html_path}",
          flush=True)
    print("\nStep 2.2 complete. STOPPING for review before Step 3.", flush=True)


if __name__ == '__main__':
    main()
