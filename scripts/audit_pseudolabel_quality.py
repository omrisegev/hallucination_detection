#!/usr/bin/env python
"""
audit_pseudolabel_quality.py — Step 1 prerequisite GATE for the D1+D2 work.

WHY THIS RUNS FIRST
-------------------
D2 (`a6.adaptive_pl_mrmr`) makes pseudo-label agreement the PRIMARY feature-relevance
driver instead of a lambda3 correction on a Laplacian-smoothness objective. That only
works if the pseudo-label y_hat is actually informative on the cells we are trying to
fix. The QA gap is real (verified: losnet_hotpotqa is closed-book QA and inside_coqa is
open-book conversational QA -- neither is RAG, so neither can be scoped away), and the
weakest QA cells sit near chance (inside_coqa 0.523, seiclr_triviaqa/OPT-30B 0.568).

If y_hat is near-chance on those cells, D2 would select y_hat-correlated-but-wrong
features and make QA worse. y_hat is also built FROM seed views, so "select features
agreeing with y_hat" carries a circularity risk. This audit bounds both before any D2
code is written.

WHAT IT MEASURES (per cell, all 25 in-scope)
-------------------------------------------
  auc_pl            AUROC of y_hat as-is (anchor-oriented -- the form D2 would consume)
  auc_pl_abs        max(auc, 1-auc): is there signal even if the SIGN is wrong?
  auc_pl_lpm        y_hat re-oriented against `logprob_margin` instead of the cell anchor
  auc_best_seed     best single seed view (unsigned) -- the fallback D2 competes with
  auc_logprob_margin  the candidate alternative anchor on its own
  auc_good5         L-SML continuous fusion over GOOD_5 -- the subset to beat

PRE-REGISTERED GATE (declared before running; see the plan file)
----------------------------------------------------------------
  GREEN  : median QA auc_pl >= 0.70            -> D2 proceeds using y_hat as-is
  AMBER  : neither GREEN nor RED               -> D2 proceeds, low-consensus cells flagged
  RED    : auc_pl < 0.60 on >= 2 QA cells      -> D2 must anchor y_hat to logprob_margin
                                                  (or fall back to it on those cells)

Outputs:
  results/advisor_inscope/pseudolabel_quality_audit.csv
  results/advisor_inscope/pseudolabel_quality_audit.html
"""

import csv
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from inscope_cells import GROUP
from spectral_utils.fusion_utils import zscore, lsml_continuous
from spectral_utils.selectors.a6_pseudolabel_gates import (
    _seed_cols, _pseudo_label, MIN_SEED_VIEWS, N_SEED_VIEWS,
)
# Import the reference subsets from the ONE canonical definition. Do NOT hardcode
# them: `compare_anchor_quality.py:141` hardcodes a "good5_indices" list
# (['epr','cusum_max','logprob_margin','spectral_entropy','topk_tail_mass']) that is
# neither GOOD_5 nor LOCO_5 but an undocumented hybrid -- every number derived from
# it (including the "GOOD_5 + logprob_margin anchor = 0.7596" figure) is mislabeled.
from spectral_utils.subset_sweep import GOOD_5, GOOD_6, LOCO_5, ANCHOR_PRIORITY
# Reuse the 25-cell loader rather than duplicating it (Step-193 lesson: a roster
# copy-pasted into N scripts drifts). This is the same loader Stage 2 used.
from compare_anchor_quality import load_all_inscope_cells

AI_DIR = os.path.join(REPO, "results", "advisor_inscope")
os.makedirs(AI_DIR, exist_ok=True)

# The hybrid list compare_anchor_quality.py used, kept ONLY so the audit can show
# what that script was actually measuring.
HYBRID_CAQ = ['epr', 'cusum_max', 'logprob_margin', 'spectral_entropy', 'topk_tail_mass']

# Candidate seed rules for the pseudo-label (Option A decides between these on data).
SEED_RULES = {
    'anchor4': ANCHOR_PRIORITY,   # status quo: what _seed_cols uses today
    'good5': GOOD_5,
    'good6': GOOD_6,
    'loco5': LOCO_5,
    'hybrid_caq': HYBRID_CAQ,
}

# Pre-registered gate thresholds
GATE_GREEN_MEDIAN_QA = 0.70
GATE_RED_CELL_AUC = 0.60
GATE_RED_MIN_CELLS = 2


def _safe_auc(labels, scores):
    """AUROC, or None when the vector is degenerate/unusable."""
    s = np.asarray(scores, dtype=float)
    if s.size == 0 or not np.all(np.isfinite(s)) or np.allclose(s, s[0]):
        return None
    try:
        return float(roc_auc_score(labels, s))
    except ValueError:
        return None


def _fuse_named(cell, names, orient_to=None):
    """L-SML fuse the named views present in this cell's pool.

    Returns dict with the oriented score, the columns used, the L-SML residual,
    and `consensus` = mean |corr(y_fused, seed_view)|. `consensus` is the
    LABEL-FREE degeneracy detector: a healthy rank-one consensus should track its
    own inputs, so a low value means the fusion no longer represents the views it
    was built from (which is exactly what happens on the cells where the fused
    pseudo-label inverts).
    """
    V = np.asarray(cell.V, dtype=np.float64)
    cols = [cell.pool.index(f) for f in names if f in cell.pool]
    if len(cols) < MIN_SEED_VIEWS:
        return None
    fused, meta = lsml_continuous(*[V[:, c] for c in cols])
    y = zscore(np.asarray(fused, dtype=np.float64))
    a = np.asarray(cell.anchor if orient_to is None else orient_to, dtype=np.float64)
    if np.corrcoef(y, a)[0, 1] < 0:
        y = -y
    cons = float(np.mean([abs(np.corrcoef(y, V[:, c])[0, 1]) for c in cols]))
    return {'y': y, 'cols': cols, 'n': len(cols),
            'residual': float(meta['residual']), 'consensus': cons}


def audit_cell(cell_key, cdata):
    """Pseudo-label quality under every candidate seed rule, for one cell."""
    u = cdata['unlabeled']
    V = np.asarray(cdata['V'], dtype=np.float64)
    labels = cdata['labels']

    seed_cols, seed_names = _seed_cols(u)
    y_hat, pl_meta = _pseudo_label(u, seed_cols)

    row = {
        'cell': cell_key,
        'group': GROUP.get(cell_key, '?'),
        'n': int(len(labels)),
        'pos_rate': round(float(np.mean(labels)), 4),
        'n_seeds': len(seed_cols),
        'seeds': '|'.join(seed_names),
        'degraded': bool(pl_meta.get('degraded')),
        'lsml_residual': pl_meta.get('residual'),
    }

    # --- status quo pseudo-label (what a6 ships today) ---------------------
    auc_pl = _safe_auc(labels, y_hat)
    row['auc_pl'] = round(auc_pl, 4) if auc_pl is not None else None
    row['auc_pl_abs'] = round(max(auc_pl, 1.0 - auc_pl), 4) if auc_pl is not None else None
    row['sign_wrong'] = bool(auc_pl is not None and auc_pl < 0.5)

    # --- candidate seed rules ----------------------------------------------
    for rule, names in SEED_RULES.items():
        r = _fuse_named(u, names)
        if r is None:
            row[f'auc_{rule}'] = None
            row[f'cons_{rule}'] = None
            row[f'nviews_{rule}'] = 0
            continue
        a = _safe_auc(labels, r['y'])
        row[f'auc_{rule}'] = round(a, 4) if a is not None else None
        row[f'cons_{rule}'] = round(r['consensus'], 4)
        row[f'nviews_{rule}'] = r['n']
        if rule == 'anchor4':
            row['residual_anchor4'] = round(r['residual'], 5)

    # --- per-seed individual AUCs + the label-free fallback target ----------
    seed_aucs = {}
    for c, nm in zip(seed_cols, seed_names):
        a = _safe_auc(labels, V[:, c])
        if a is not None:
            seed_aucs[nm] = max(a, 1.0 - a)
    row['auc_best_seed'] = round(max(seed_aucs.values()), 4) if seed_aucs else None
    row['best_seed_name'] = (max(seed_aucs, key=seed_aucs.get) if seed_aucs else None)

    # Option-B fallback chain, fully label-free: logprob_margin if the cell has it,
    # else the cell's own anchor. (logprob_margin is ABSENT on inside_coqa, which is
    # why "fall back to logprob_margin" cannot be the whole rule.)
    if 'logprob_margin' in u.pool:
        fb_name, fb_vec = 'logprob_margin', V[:, u.pool.index('logprob_margin')]
    else:
        fb_name, fb_vec = f'anchor:{u.anchor_name}', np.asarray(u.anchor, dtype=np.float64)
    a = _safe_auc(labels, fb_vec)
    row['fallback_name'] = fb_name
    row['auc_fallback'] = round(a, 4) if a is not None else None
    row['auc_fallback_abs'] = round(max(a, 1.0 - a), 4) if a is not None else None

    return row


def gate_for(rows, col):
    """Apply the pre-registered gate to one pseudo-label column."""
    qa = [r for r in rows if r['group'] == 'QA' and r.get(col) is not None]
    aucs = [r[col] for r in qa]
    median_qa = float(np.median(aucs)) if aucs else float('nan')
    weak = [r['cell'] for r in qa if r[col] < GATE_RED_CELL_AUC]
    if len(weak) >= GATE_RED_MIN_CELLS:
        verdict = 'RED'
    elif median_qa >= GATE_GREEN_MEDIAN_QA:
        verdict = 'GREEN'
    else:
        verdict = 'AMBER'
    return verdict, median_qa, weak


def rule_summary(rows):
    """Per-seed-rule macro summary + gate, so Option A is decided on data."""
    out = []
    for rule in ['anchor4'] + [r for r in SEED_RULES if r != 'anchor4']:
        col = f'auc_{rule}'
        vals = [r[col] for r in rows if r.get(col) is not None]
        qa = [r[col] for r in rows if r['group'] == 'QA' and r.get(col) is not None]
        ma = [r[col] for r in rows if r['group'] == 'math' and r.get(col) is not None]
        if not vals:
            continue
        verdict, median_qa, weak = gate_for(rows, col)
        inverted = [r['cell'] for r in rows
                    if r.get(col) is not None and r[col] < 0.5]
        out.append({
            'rule': rule,
            'views': '|'.join(SEED_RULES[rule]),
            'n_cells': len(vals),
            'macro_all': round(float(np.mean(vals)), 4),
            'macro_qa': round(float(np.mean(qa)), 4) if qa else None,
            'macro_math': round(float(np.mean(ma)), 4) if ma else None,
            'median_qa': round(median_qa, 4),
            'n_weak_qa': len(weak),
            'weak_qa': ','.join(weak),
            'n_inverted': len(inverted),
            'inverted': ','.join(inverted),
            'gate': verdict,
        })
    out.sort(key=lambda d: -d['macro_all'])
    return out


def evaluate_gate(rows):
    """Gate on the SHIPPING pseudo-label (status quo), plus supporting detail."""
    verdict, median_qa, weak = gate_for(rows, 'auc_pl')
    if verdict == 'RED':
        msg = (f"{len(weak)} QA cells below {GATE_RED_CELL_AUC} "
               f"({', '.join(weak)}) with the shipping seed rule.")
    elif verdict == 'GREEN':
        msg = (f"Median QA pseudo-label AUROC {median_qa:.4f} >= "
               f"{GATE_GREEN_MEDIAN_QA}. D2 may consume y_hat as-is.")
    else:
        msg = (f"Median QA pseudo-label AUROC {median_qa:.4f} is below "
               f"{GATE_GREEN_MEDIAN_QA} but fewer than {GATE_RED_MIN_CELLS} cells "
               f"are under {GATE_RED_CELL_AUC}.")
    flips = [r['cell'] for r in rows if r.get('sign_wrong')]

    # Does the label-free consensus statistic actually flag the broken cells?
    cons, inv = [], []
    for r in rows:
        if r.get('cons_anchor4') is not None and r.get('auc_pl') is not None:
            cons.append(r['cons_anchor4'])
            inv.append(1.0 if r['auc_pl'] < 0.5 else 0.0)
    cons_inverted = [c for c, i in zip(cons, inv) if i]
    cons_ok = [c for c, i in zip(cons, inv) if not i]

    return verdict, {
        'median_qa': median_qa,
        'weak_qa_cells': weak,
        'message': msg,
        'sign_wrong_cells': flips,
        'cons_inverted_mean': float(np.mean(cons_inverted)) if cons_inverted else float('nan'),
        'cons_ok_mean': float(np.mean(cons_ok)) if cons_ok else float('nan'),
        'cons_inverted_max': float(np.max(cons_inverted)) if cons_inverted else float('nan'),
        'cons_ok_min': float(np.min(cons_ok)) if cons_ok else float('nan'),
    }


def build_dashboard(rows, verdict, detail, rules):
    """Self-contained HTML: gate verdict, seed-rule comparison, per-cell table."""
    colour = {'GREEN': '#15803d', 'AMBER': '#b45309', 'RED': '#b91c1c'}[verdict]
    bg = {'GREEN': '#dcfce7', 'AMBER': '#fef3c7', 'RED': '#fee2e2'}[verdict]

    srt = sorted(rows, key=lambda r: (r['group'], -(r['auc_pl'] or 0)))

    def fmt(v, nd=4):
        return '-' if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    rule_order = [d['rule'] for d in rules]
    rule_rows = []
    for d in rules:
        gc = {'GREEN': '#15803d', 'AMBER': '#b45309', 'RED': '#b91c1c'}[d['gate']]
        star = ' &#11088;' if d['rule'] == rules[0]['rule'] else ''
        rule_rows.append(
            f"<tr><td><b>{d['rule']}</b>{star}<br><span style='font-size:11px;color:#64748b'>"
            f"{d['views']}</span></td><td><b>{fmt(d['macro_all'])}</b></td>"
            f"<td>{fmt(d['macro_qa'])}</td><td>{fmt(d['macro_math'])}</td>"
            f"<td>{fmt(d['median_qa'])}</td><td>{d['n_weak_qa']}</td>"
            f"<td>{d['n_inverted']}<br><span style='font-size:11px;color:#b91c1c'>"
            f"{d['inverted']}</span></td>"
            f"<td style='color:{gc};font-weight:bold'>{d['gate']}</td></tr>")

    body_rows = []
    for r in srt:
        pl = r['auc_pl']
        cls = ''
        if pl is not None and pl < GATE_RED_CELL_AUC:
            cls = ' style="background:#fee2e2"'
        elif pl is not None and pl < GATE_GREEN_MEDIAN_QA:
            cls = ' style="background:#fef3c7"'
        cells = ''.join(f"<td>{fmt(r.get('auc_' + ru))}</td>" for ru in rule_order)
        body_rows.append(
            f"<tr{cls}><td><b>{r['cell']}</b></td><td>{r['group']}</td>"
            f"<td>{r['n']}</td>{cells}"
            f"<td>{fmt(r.get('cons_anchor4'))}</td>"
            f"<td>{fmt(r['auc_best_seed'])} <span style='font-size:11px;color:#64748b'>"
            f"{r.get('best_seed_name') or ''}</span></td>"
            f"<td>{fmt(r.get('auc_fallback_abs'))} <span style='font-size:11px;color:#64748b'>"
            f"{r.get('fallback_name') or ''}</span></td></tr>")

    qa = [r for r in srt if r['group'] == 'QA']
    math_ = [r for r in srt if r['group'] == 'math']
    chart = {
        'qa_labels': [r['cell'] for r in qa],
        'qa_vals': [r['auc_pl'] for r in qa],
        'math_labels': [r['cell'] for r in math_],
        'math_vals': [r['auc_pl'] for r in math_],
    }
    rule_hdr = ''.join(f"<th>{ru}</th>" for ru in rule_order)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Step 1 - Pseudo-Label Quality Audit (D1+D2 Gate)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background:#f4f7f9; color:#333;
         margin:0; padding:20px; }}
  h1,h2 {{ color:#1e293b; }}
  .card {{ background:#fff; padding:20px; border-radius:10px;
           box-shadow:0 4px 6px rgba(0,0,0,.05); margin-bottom:22px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:12px; font-size:13px; }}
  th,td {{ padding:8px 10px; border:1px solid #e2e8f0; text-align:left; }}
  th {{ background:#f8fafc; color:#475569; }}
  .verdict {{ font-size:22px; font-weight:bold; color:{colour}; }}
  .vbox {{ background:{bg}; padding:16px; border-radius:8px; border-left:6px solid {colour}; }}
  .chart {{ position:relative; height:340px; }}
  code {{ background:#f1f5f9; padding:1px 5px; border-radius:3px; }}
</style>
</head>
<body>

<div class="card">
  <h1>Step 1 - Pseudo-Label Quality Audit</h1>
  <p>Prerequisite gate for <b>D2</b> (<code>a6.adaptive_pl_mrmr</code>), which would make
     pseudo-label agreement the primary feature-relevance driver. This measures whether
     <code>y_hat</code> is trustworthy enough to play that role, especially on the weak QA cells.</p>
  <div class="vbox">
    <div class="verdict">GATE: {verdict}</div>
    <p style="margin:8px 0 0">{detail['message']}</p>
  </div>
  <p style="margin-top:14px">
    Median QA <code>auc_pl</code>: <b>{detail['median_qa']:.4f}</b> &nbsp;|&nbsp;
    Sign-inverted cells: <b>{len(detail['sign_wrong_cells'])}</b>
    {(' (' + ', '.join(detail['sign_wrong_cells']) + ')') if detail['sign_wrong_cells'] else ''}
  </p>
  <p style="font-size:13px;color:#64748b">
    Label-free degeneracy detector (<code>consensus</code> = mean |corr(y_hat, its own seed
    views)|): mean on inverted cells <b>{detail['cons_inverted_mean']:.4f}</b>
    (max {detail['cons_inverted_max']:.4f}) vs mean on healthy cells
    <b>{detail['cons_ok_mean']:.4f}</b> (min {detail['cons_ok_min']:.4f}). A clean separation
    means Option B can catch the broken fusions without labels.
  </p>
</div>

<div class="card">
  <h2>Option A: seed-rule comparison (decided on data)</h2>
  <p style="font-size:13px;color:#64748b">
    Each row re-builds the pseudo-label by L-SML fusing a different reference subset.
    <code>anchor4</code> is what <code>_seed_cols</code> ships today.
    <code>hybrid_caq</code> is the undocumented list hardcoded at
    <code>compare_anchor_quality.py:141</code> and mislabeled there as GOOD_5.</p>
  <table>
    <thead><tr><th>Seed rule</th><th>Macro (25)</th><th>QA</th><th>Math</th>
    <th>Median QA</th><th>#QA&lt;0.60</th><th>#inverted</th><th>Gate</th></tr></thead>
    <tbody>{''.join(rule_rows)}</tbody>
  </table>
</div>

<div class="card">
  <h2>Pseudo-Label AUROC by cell (shipping rule)</h2>
  <div class="chart"><canvas id="c1"></canvas></div>
</div>

<div class="card">
  <h2>Per-cell detail</h2>
  <p style="font-size:13px;color:#64748b">
    Columns are the fused pseudo-label AUROC under each seed rule (as-is, anchor-oriented --
    values &lt; 0.5 are sign-inverted). <code>consensus</code> is the label-free degeneracy
    statistic for the shipping rule. Red rows &lt; {GATE_RED_CELL_AUC},
    amber &lt; {GATE_GREEN_MEDIAN_QA}.</p>
  <table>
    <thead><tr><th>Cell</th><th>Group</th><th>N</th>{rule_hdr}
    <th>consensus</th><th>best single seed</th><th>fallback view</th></tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</div>

<script>
const D = {json.dumps(chart)};
new Chart(document.getElementById('c1'), {{
  type:'bar',
  data:{{ labels: D.qa_labels.concat(D.math_labels),
    datasets:[{{ label:'QA cells', backgroundColor:'#ef4444',
                 data: D.qa_vals.concat(D.math_labels.map(()=>null)) }},
              {{ label:'Math cells', backgroundColor:'#3b82f6',
                 data: D.qa_labels.map(()=>null).concat(D.math_vals) }}] }},
  options:{{ responsive:true, maintainAspectRatio:false,
    scales:{{ y:{{ min:0.4, max:1.0, title:{{display:true,text:'Pseudo-label AUROC'}} }},
              x:{{ ticks:{{ font:{{size:9}}, maxRotation:90, minRotation:60 }} }} }},
    plugins:{{ annotation:false }} }}
}});
</script>

</body>
</html>
"""
    path = os.path.join(AI_DIR, "pseudolabel_quality_audit.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def main():
    cells = load_all_inscope_cells()
    print(f"\n--- Step 1: pseudo-label quality audit over {len(cells)} cells ---",
          flush=True)

    rows = []
    for ck in sorted(cells):
        rows.append(audit_cell(ck, cells[ck]))
        r = rows[-1]
        print(f"  {r['group']:5s} {r['cell']:38s} auc_pl={str(r['auc_pl']):8s} "
              f"abs={str(r['auc_pl_abs']):8s} good5={str(r.get('auc_good5')):8s} "
              f"loco5={str(r.get('auc_loco5')):8s} cons={str(r.get('cons_anchor4')):8s}",
              flush=True)

    verdict, detail = evaluate_gate(rows)
    rules = rule_summary(rows)

    rule_cols = []
    for ru in SEED_RULES:
        rule_cols += [f'auc_{ru}', f'cons_{ru}', f'nviews_{ru}']
    fields = (['cell', 'group', 'n', 'pos_rate', 'auc_pl', 'auc_pl_abs', 'sign_wrong']
              + rule_cols
              + ['auc_best_seed', 'best_seed_name', 'fallback_name', 'auc_fallback',
                 'auc_fallback_abs', 'n_seeds', 'seeds', 'degraded', 'lsml_residual'])
    csv_path = os.path.join(AI_DIR, "pseudolabel_quality_audit.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    rules_path = os.path.join(AI_DIR, "pseudolabel_seed_rules.csv")
    with open(rules_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rules[0].keys()))
        w.writeheader()
        w.writerows(rules)

    html_path = build_dashboard(rows, verdict, detail, rules)

    qa = [r['auc_pl'] for r in rows if r['group'] == 'QA' and r['auc_pl'] is not None]
    ma = [r['auc_pl'] for r in rows if r['group'] == 'math' and r['auc_pl'] is not None]
    print("\n=== Pseudo-label quality summary (shipping seed rule) ===", flush=True)
    print(f"  QA   mean {np.mean(qa):.4f} | median {np.median(qa):.4f} | n={len(qa)}",
          flush=True)
    print(f"  Math mean {np.mean(ma):.4f} | median {np.median(ma):.4f} | n={len(ma)}",
          flush=True)

    print("\n=== Option A: seed-rule comparison ===", flush=True)
    print(f"  {'rule':12s} {'macro':>7s} {'QA':>7s} {'math':>7s} {'medQA':>7s} "
          f"{'<0.60':>5s} {'inv':>4s}  gate", flush=True)
    for d in rules:
        print(f"  {d['rule']:12s} {d['macro_all']:7.4f} "
              f"{(d['macro_qa'] if d['macro_qa'] is not None else float('nan')):7.4f} "
              f"{(d['macro_math'] if d['macro_math'] is not None else float('nan')):7.4f} "
              f"{d['median_qa']:7.4f} {d['n_weak_qa']:5d} {d['n_inverted']:4d}  {d['gate']}",
              flush=True)

    print("\n=== Option B: label-free degeneracy detector (consensus) ===", flush=True)
    print(f"  inverted cells: mean {detail['cons_inverted_mean']:.4f} "
          f"(max {detail['cons_inverted_max']:.4f})", flush=True)
    print(f"  healthy  cells: mean {detail['cons_ok_mean']:.4f} "
          f"(min {detail['cons_ok_min']:.4f})", flush=True)
    separable = detail['cons_inverted_max'] < detail['cons_ok_min']
    print(f"  separable by a threshold? {'YES' if separable else 'NO'}", flush=True)

    print(f"\n  GATE VERDICT (shipping rule): {verdict}\n  {detail['message']}", flush=True)
    print(f"\n  CSV  : {csv_path}\n  RULES: {rules_path}\n  HTML : {html_path}", flush=True)
    print("\nStep 1 re-run complete. STOPPING for review before Step 2.", flush=True)


if __name__ == '__main__':
    main()
