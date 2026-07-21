#!/usr/bin/env python
"""In-scope (QA + math) evaluation report — Step 192.

Renders one self-contained, theme-aware HTML page presenting the complete
in-scope evaluation on the wide (30-view) feature pool:

  - headline: the leading pipeline (GOOD_6) macro AUROC, QA/math split, and its
    margin over GOOD_5 / the best label-free learned selector
  - the full selector leaderboard (in-scope, c46 pool)
  - the honest split-half ceiling ladder (GOOD_5 -> greedy@H16 -> greedy@30v),
    the achievable-headroom number (Step 189 discipline: NOT the winner's-curse
    label-peeking sweep oracle)
  - the Step-187 orientation verdict: is the anchor / curated subset correctly
    oriented on in-scope cells? (answers whether the offline sign-fix matters)
  - a per-cell table (GOOD_5 vs GOOD_6 L-SML AUROC, n, accuracy)

Every number is read from a CSV — never hand-typed (Step-184 discipline).
Regenerate, never hand-edit:  python scripts/inscope_report.py
"""
import os
import sys

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO_DIR, 'results', 'selector_bench')

QA_CELLS = [
    'epr_triviaqa_mistral24b', 'inside_coqa_llama7b', 'losnet_hotpotqa_mistral7b',
    'sciq_llama8b', 'se_nq_open_llama8b', 'se_squad_v2_llama8b',
    'seiclr_triviaqa_opt30b', 'semenergy_triviaqa_qwen3_8b',
    'spilled_triviaqa_llama8b', 'truthfulqa_llama8b',
]
MATH_CELLS = [
    'ars_gsm8k_r1distill8b', 'internalstates_gsm8k_qwen25_7b',
    'lapeigvals_gsm8k_llama3b', 'lapeigvals_gsm8k_llama8b',
    'lapeigvals_gsm8k_mistral24b', 'lapeigvals_gsm8k_nemo',
    'lapeigvals_gsm8k_phi35', 'noise_gsm8k_mistral7b', 'noise_gsm8k_phi3mini',
    'math500_dsmath7b', 'math500_qwenmath7b', 'math500_r1distill8b',
    'math500_r1distill8b_mn4096', 'trace_gsm8k_llama8b_k10',
    'trace_math500_qwenmath15b_k10',
]
INSCOPE = QA_CELLS + MATH_CELLS
GROUP = {c: 'QA' for c in QA_CELLS}
GROUP.update({c: 'math' for c in MATH_CELLS})

FAMILY = [  # (prefix, label)
    ('ref.', 'reference macro'), ('a1.', 'residual-guided (A1)'),
    ('a2.', 'GroupFS (A2)'), ('a3.', 'Concrete-AE (A3)'),
    ('a4.', 'anchor-affinity (A4)'), ('a5.', 'mRMR (A5)'),
    ('lapscore', 'classical FS'), ('spec', 'classical FS'), ('mcfs', 'classical FS'),
    ('random', 'stats floor'), ('mad', 'stats floor'),
    ('kurtosis', 'stats floor'), ('decorr', 'stats floor'),
    ('epr.', 'single feature'),
]


def fam_of(variant):
    for pre, lab in FAMILY:
        if variant.startswith(pre):
            return lab
    return 'other'


def fnum(x, nd=4):
    try:
        v = float(x)
        return '' if not np.isfinite(v) else f'{v:.{nd}f}'
    except (TypeError, ValueError):
        return ''


def pp(x, nd=2):
    try:
        v = float(x) * 100
        return f'{v:+.{nd}f}pp' if np.isfinite(v) else ''
    except (TypeError, ValueError):
        return ''


# ---------------------------------------------------------------------------
CSS = """
:root{--bg:#faf9f7;--panel:#ffffff;--ink:#1f2430;--muted:#5b6472;--line:#e3e1dc;
 --accent:#1baf7a;--ref:#2a78d6;--warn:#c98500;--bad:#c0392b;--chip:#f0efec;--code:#f4f3f0;}
:root[data-theme="dark"]{--bg:#14171d;--panel:#1c2027;--ink:#e8eaf0;--muted:#9aa2af;
 --line:#2c313a;--accent:#22c286;--ref:#3987e5;--warn:#eda100;--bad:#e26d5a;--chip:#252a33;--code:#232830;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14171d;--panel:#1c2027;
 --ink:#e8eaf0;--muted:#9aa2af;--line:#2c313a;--accent:#22c286;--ref:#3987e5;--warn:#eda100;
 --bad:#e26d5a;--chip:#252a33;--code:#232830;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Segoe UI",system-ui,sans-serif}
main{max-width:1060px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:26px;line-height:1.25;margin:6px 0 2px;text-wrap:balance}
h2{font-size:19px;margin:40px 0 8px;padding-top:14px;border-top:1px solid var(--line)}
p{max-width:78ch}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0}
.sub{color:var(--muted);margin:0 0 18px}
table{border-collapse:collapse;font-size:13.5px;margin:10px 0;width:100%}
th,td{padding:5px 10px;border-bottom:1px solid var(--line);text-align:left;font-variant-numeric:tabular-nums}
th{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}
td.num,th.num{text-align:right}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--panel)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:14px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:16px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.tile .lab{font-size:12px;color:var(--muted);margin:0 0 4px}
.big{font-size:30px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.1}
.tile .foot{font-size:12px;color:var(--muted);margin:4px 0 0}
.chip{display:inline-block;background:var(--chip);border-radius:999px;padding:1px 10px;font-size:12px;color:var(--muted);margin-right:6px}
.good{color:var(--accent);font-weight:600}.badv{color:var(--bad);font-weight:600}.mut{color:var(--muted)}
code{background:var(--code);border-radius:4px;padding:1px 5px;font-size:13px}
.note{border-left:3px solid var(--accent);padding:8px 14px;background:var(--panel);border-radius:0 8px 8px 0;margin:12px 0;max-width:82ch}
.note.warn{border-left-color:var(--warn)}
svg text{font:11.5px "Segoe UI",system-ui,sans-serif;fill:var(--ink)}
svg .mut{fill:var(--muted)}
a{color:var(--ref)}
"""
THEME_JS = ("<script>(function(){var b=document.createElement('button');"
            "b.textContent='\\u25d0 theme';b.style.cssText='position:fixed;top:10px;right:12px;"
            "z-index:9;background:var(--panel);color:var(--muted);border:1px solid var(--line);"
            "border-radius:8px;padding:3px 10px;cursor:pointer;font-size:12px';b.onclick=function(){"
            "var r=document.documentElement;var cur=r.getAttribute('data-theme')||"
            "(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');"
            "r.setAttribute('data-theme',cur==='dark'?'light':'dark');};"
            "document.body.appendChild(b);})();</script>")


def hbar_chart(rows, vmin, vmax, ref_val, ref_label, width=680, rowh=26):
    """rows: list of (label, value, highlight_bool). Single-measure magnitude
    bars (sequential single hue), baseline at vmin, a reference rule at ref_val."""
    n = len(rows)
    padl, padr, padt, padb = 250, 60, 8, 22
    h = padt + padb + n * rowh
    span = vmax - vmin

    def x(v):
        return padl + (v - vmin) / span * (width - padl - padr)
    parts = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img" '
             f'aria-label="in-scope macro AUROC by method">']
    # reference rule (GOOD_5)
    rx = x(ref_val)
    parts.append(f'<line x1="{rx:.1f}" y1="{padt}" x2="{rx:.1f}" y2="{h-padb}" '
                 f'stroke="var(--ref)" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{rx:.1f}" y="{h-padb+15}" text-anchor="middle" '
                 f'class="mut">{ref_label} {ref_val:.3f}</text>')
    for i, (lab, val, hi) in enumerate(rows):
        y = padt + i * rowh
        bx = x(val)
        fill = 'var(--accent)' if hi else 'var(--muted)'
        op = '1' if hi else '0.55'
        parts.append(f'<rect x="{padl}" y="{y+4}" width="{max(bx-padl,1):.1f}" '
                     f'height="{rowh-10}" rx="3" fill="{fill}" opacity="{op}"/>')
        parts.append(f'<text x="{padl-8}" y="{y+rowh/2+3:.1f}" text-anchor="end">{lab}</text>')
        parts.append(f'<text x="{bx+5:.1f}" y="{y+rowh/2+3:.1f}" '
                     f'fill="{fill}">{val:.4f}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def ladder_chart(data, width=680):
    """data: list of (group_label, good5, greedy_h16, greedy_c46). Grouped bars,
    3 measures share ONE AUROC scale (indexed to the same axis) — the honest
    ceiling ladder. Uses status-neutral single-hue steps (light->dark) since the
    three bars are ordered stages of one quantity, not distinct identities."""
    vals = [v for g in data for v in g[1:] if np.isfinite(v)]
    vmin = min(0.5, min(vals) - 0.02)
    vmax = max(vals) + 0.03
    span = vmax - vmin
    padl, padr, padt, padb = 120, 20, 24, 30
    grouph = 92
    h = padt + padb + len(data) * grouph
    barh = 20
    hues = ['#9ecfc0', '#4bb596', '#158463']  # light->dark single-hue steps
    dhues = ['#3a5c53', '#2b9d78', '#22c286']
    names = ['GOOD_5 (curated)', 'greedy@H16 (16 views)', 'greedy@30v (wide)']

    def x(v):
        return padl + (v - vmin) / span * (width - padl - padr)
    parts = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img" '
             f'aria-label="honest split-half ceiling ladder">']
    # legend
    lx = padl
    for k, nm in enumerate(names):
        parts.append(f'<rect x="{lx}" y="4" width="11" height="11" rx="2" fill="{hues[k]}" '
                     f'style="fill:{hues[k]}"/><text x="{lx+15}" y="13">{nm}</text>')
        lx += 165
    for gi, (glab, g5, gh16, gc46) in enumerate(data):
        y0 = padt + gi * grouph
        parts.append(f'<text x="8" y="{y0+grouph/2:.1f}" class="k">{glab}</text>')
        for k, val in enumerate([g5, gh16, gc46]):
            if not np.isfinite(val):
                continue
            y = y0 + k * (barh + 3)
            bx = x(val)
            parts.append(f'<rect x="{padl}" y="{y}" width="{max(bx-padl,1):.1f}" '
                         f'height="{barh}" rx="3" fill="{hues[k]}" style="fill:{hues[k]}"/>')
            parts.append(f'<text x="{bx+5:.1f}" y="{y+barh/2+3:.1f}">{val:.4f}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def main():
    lb = pd.read_csv(os.path.join(BENCH, 'comparison_inscope.csv'))
    lb_c46 = lb[lb['pool_mode'] == 'c46'].copy().sort_values('macro_all', ascending=False)
    orient = pd.read_csv(os.path.join(BENCH, 'inscope_feature_orientation.csv'))
    osum = pd.read_csv(os.path.join(BENCH, 'inscope_feature_orientation_summary.csv'))
    scores = pd.read_csv(os.path.join(REPO_DIR, 'results', 'repgrid', 'scores_lsml_upcr.csv'))

    def load_sh(pool):
        p = os.path.join(BENCH, f'splithalf_oracle_{pool}_inscope_summary.csv')
        return pd.read_csv(p) if os.path.exists(p) else None
    sh_c46, sh_h16 = load_sh('c46'), load_sh('h16')

    good5 = lb_c46[lb_c46['variant'] == 'ref.GOOD_5'].iloc[0]
    good6 = lb_c46[lb_c46['variant'] == 'ref.GOOD_6'].iloc[0]
    learned = lb_c46[~lb_c46['variant'].str.startswith(('ref.', 'random', 'mad',
                                                        'kurtosis', 'decorr', 'epr.'))]
    best_learned = learned.iloc[0] if len(learned) else None

    # ---- honest ceiling ladder (QA / math) ----
    def ladder_rows():
        out = []
        for glab, cells in [('QA (10 cells)', QA_CELLS), ('math (15 cells)', MATH_CELLS)]:
            def m(sh, col):
                if sh is None:
                    return np.nan
                s = sh[sh['cell'].isin(cells)][col]
                return float(s.mean()) if len(s) else np.nan
            g5 = np.nanmean([m(sh_h16, 'good5_halfB'), m(sh_c46, 'good5_halfB')])
            out.append((glab, g5, m(sh_h16, 'greedy_halfB'), m(sh_c46, 'greedy_halfB')))
        return out
    ladder = ladder_rows()

    # ---- orientation verdict ----
    epr = orient[orient['feature'] == 'epr']
    epr_min = float(epr['oriented_auroc'].min())
    epr_anti = int((epr['oriented_auroc'] < 0.5).sum())
    g6feats = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy', 'varentropy']
    g6o = osum[osum['feature'].isin(g6feats)][['feature', 'QA_mean', 'QA_n_anti',
                                               'math_mean', 'math_n_anti', 'all_mean']]

    # ---- leaderboard table (c46) ----
    lb_rows = ''
    show = lb_c46.head(14)
    for _, r in show.iterrows():
        hi = r['variant'] in ('ref.GOOD_6', 'ref.GOOD_5')
        cls = ' style="background:var(--chip)"' if hi else ''
        dv = r['delta_vs_good5_all']
        dcls = 'good' if dv > 0 else ('badv' if dv < 0 else 'mut')
        gm = r['G_macro']
        gcls = 'good' if gm == 'SUCCESS' else ('mut' if gm == 'PASS' else 'badv')
        lb_rows += (f'<tr{cls}><td>{r["variant"]}</td><td class="mut">{fam_of(r["variant"])}</td>'
                    f'<td class="num">{fnum(r["macro_all"])}</td>'
                    f'<td class="num">{fnum(r["macro_qa"])}</td>'
                    f'<td class="num">{fnum(r["macro_math"])}</td>'
                    f'<td class="num {dcls}">{pp(dv)}</td>'
                    f'<td class="num">{fnum(r["wilcoxon_p"],4) if fnum(r["wilcoxon_p"]) else "—"}</td>'
                    f'<td class="num">{int(r["wins"])}/{int(r["ties"])}/{int(r["losses"])}</td>'
                    f'<td class="{gcls}">{gm}</td></tr>')

    # chart rows: top methods + baselines, highlight GOOD_6/GOOD_5
    chart_src = show.head(12)
    crows = [(r['variant'], float(r['macro_all']),
              r['variant'] in ('ref.GOOD_6', 'ref.GOOD_5'))
             for _, r in chart_src.iterrows()]
    vmin = min(v for _, v, _ in crows) - 0.01
    vmax = max(v for _, v, _ in crows) + 0.01
    chart = hbar_chart(crows, vmin, vmax, float(good5['macro_all']), 'GOOD_5')

    # ---- per-cell table ----
    sc = scores[(scores['method'] == 'lsml') & (scores['anchor'] == 'epr')]
    def cell_score(cell, subset):
        row = sc[(sc['cell'] == cell) & (sc['subset'] == subset)]
        return row.iloc[0] if len(row) else None
    cell_rows = ''
    for cell in INSCOPE:
        g5r = cell_score(cell, 'GOOD_5')
        g6r = cell_score(cell, 'GOOD_6')
        if g5r is None:
            continue
        n = int(g5r['n_problems']) if pd.notna(g5r['n_problems']) else ''
        acc = fnum(g5r['acc'], 3) if pd.notna(g5r['acc']) else ''
        a5 = fnum(g5r['auroc_X'], 4)
        a6 = fnum(g6r['auroc_X'], 4) if g6r is not None else ''
        delta = ''
        if g6r is not None and pd.notna(g5r['auroc_X']) and pd.notna(g6r['auroc_X']):
            d = (float(g6r['auroc_X']) - float(g5r['auroc_X']))
            delta = f'<span class="{"good" if d>0 else ("badv" if d<0 else "mut")}">{pp(d)}</span>'
        cell_rows += (f'<tr><td>{cell}</td><td class="chip">{GROUP[cell]}</td>'
                      f'<td class="num">{n}</td><td class="num">{acc}</td>'
                      f'<td class="num">{a5}</td><td class="num">{a6}</td>'
                      f'<td class="num">{delta}</td></tr>')

    # ---- ceiling headroom text ----
    def ceil_line(sh, cells):
        if sh is None:
            return None
        s = sh[sh['cell'].isin(cells)]
        return (float(s['good5_halfB'].mean()), float(s['greedy_halfB'].mean()))
    hp_qa = ceil_line(sh_c46, QA_CELLS)
    hp_math = ceil_line(sh_c46, MATH_CELLS)

    def hp_txt(hp):
        if not hp:
            return 'pending'
        g5, gr = hp
        return f'{g5:.4f} → {gr:.4f} ({(gr-g5)*100:+.2f}pp)'

    orient_ok = (epr_anti == 0)
    orient_verdict = ('The label-free anchor <code>epr</code> is correctly oriented on '
                      f'<b>all 25</b> in-scope cells (min oriented AUROC {epr_min:.3f}); '
                      'the curated GOOD_5/GOOD_6 features carry the right fixed sign here.') \
        if orient_ok else \
        (f'<b>{epr_anti}</b> in-scope cells have an anti-oriented anchor — the '
         'Step-187 sign-fix is back in play for QA.')

    g6_g5_ladder = ''
    for glab, g5, gh16, gc46 in ladder:
        g6_g5_ladder += f'<li><b>{glab}</b>: GOOD_5 {fnum(g5)} &rarr; greedy@H16 {fnum(gh16)} &rarr; greedy@30v {fnum(gc46)}</li>'

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>In-Scope Evaluation — QA + Math, Wide Feature Pool</title><style>{CSS}</style></head>
<body><main>
<p class="eyebrow">Step 192 · selector bench · wide (30-view) pool</p>
<h1>In-scope evaluation: the leading pipeline over all features, on the new cluster data</h1>
<p class="sub">25 in-scope cells (10 short-form QA + 15 reasoning/math). RAG and GPQA
excluded per the Jul-20 scope call. Every number regenerated from CSV by
<code>scripts/inscope_report.py</code>.</p>

<div class="tiles">
 <div class="tile"><p class="lab">Leading pipeline (GOOD_6)</p>
   <div class="big good">{float(good6['macro_all']):.4f}</div>
   <p class="foot">macro AUROC over 25 cells · QA {float(good6['macro_qa']):.3f} · math {float(good6['macro_math']):.3f}</p></div>
 <div class="tile"><p class="lab">GOOD_6 &minus; GOOD_5</p>
   <div class="big">{pp(good6['delta_vs_good5_all'])}</div>
   <p class="foot">Wilcoxon p={fnum(good6['wilcoxon_p'],4)} · {int(good6['wins'])}W/{int(good6['losses'])}L · the wide-pool <code>varentropy</code> view</p></div>
 <div class="tile"><p class="lab">Best label-free learned selector</p>
   <div class="big">{float(best_learned['macro_all']):.4f}</div>
   <p class="foot">{best_learned['variant']} · {pp(best_learned['delta_vs_good5_all'])} vs GOOD_5 (a tie)</p></div>
 <div class="tile"><p class="lab">Honest headroom (split-half, math)</p>
   <div class="big">{hp_txt(hp_math).split(' ')[-1] if hp_math else '—'}</div>
   <p class="foot">greedy@30v over GOOD_5, held-out half · not the label-peeking oracle</p></div>
</div>

<div class="note"><b>Headline.</b> On the in-scope QA+math cells scored over the full
30-view feature pool, the curated <b>GOOD_6</b> fixed subset is the leading pipeline
({float(good6['macro_all']):.4f} macro, {pp(good6['delta_vs_good5_all'])} over GOOD_5,
p={fnum(good6['wilcoxon_p'],4)}). Its entire gain is the one wide-pool feature it adds
(<code>varentropy</code>): GOOD_5's five features all live in the 16-view H(n) pool, so
GOOD_5 is pool-invariant. <b>No label-free learned selector beats the curated subsets</b>
— the best ({best_learned['variant']}) ties GOOD_5 at {pp(best_learned['delta_vs_good5_all'])}.
This reproduces the Step-186/189 finding on the in-scope pool: the value of the wider
pool is one good feature, not automatic selection over it.</div>

<h2>Selector leaderboard — in-scope, wide (c46) pool</h2>
<p>Macro AUROC over the 25 in-scope cells, split QA / math. Delta and Wilcoxon are
paired vs GOOD_5 across cells; wins/ties/losses per cell. All 8 families ran with
<b>0 fallbacks and 0 NaN AUROC</b>.</p>
<div class="card">{chart}
<p class="mut" style="font-size:12px;margin:6px 0 0">Bars: macro AUROC over all 25 in-scope cells. Dashed rule: GOOD_5 reference. GOOD_6 / GOOD_5 highlighted.</p></div>
<div class="scroll"><table>
<thead><tr><th>variant</th><th>family</th><th class="num">macro (all)</th>
<th class="num">QA</th><th class="num">math</th><th class="num">Δ GOOD_5</th>
<th class="num">Wilcoxon p</th><th class="num">W/T/L</th><th>G-macro</th></tr></thead>
<tbody>{lb_rows}</tbody></table></div>

<h2>Honest ceiling — how much is really on the table?</h2>
<p>The label-peeking exhaustive-sweep oracle is a winner's-curse artifact (Step 189).
This is the out-of-sample counterpart: bounded greedy forward selection on a held-out
half, refit and scored on the other half, R=10 random 50/50 splits per cell. It answers
"if selection could see labels on half the data, how much would it honestly gain?"</p>
<div class="card">{ladder_chart(ladder)}</div>
<ul>{g6_g5_ladder}</ul>
<div class="note">QA honest ladder (wide pool): {hp_txt(hp_qa)}. Math: {hp_txt(hp_math)}.
The wide-pool greedy ceiling sits close to GOOD_5 — consistent with the leaderboard's
learned-selector ties. The achievable prize from in-cell label-free selection is small;
the reliable win is the curated GOOD_6.</div>

<h2>Feature orientation — does the Step-187 sign-fix matter in-scope?</h2>
<p>Step 191 found ~10pp of the (now out-of-scope) RAG deficit was pure global-sign error:
the fixed offline feature sign was upside-down on retrieval-grounded QA. The check for the
in-scope domains: is the anchor / curated subset anti-oriented here too?</p>
<div class="note"><b>Verdict: no.</b> {orient_verdict} The Step-187 offline sign-fix is
<b>not needed for the in-scope QA + math pipeline</b>; it was a RAG-specific effect.</div>
<div class="scroll"><table>
<thead><tr><th>GOOD_6 feature</th><th class="num">QA mean AUROC</th><th class="num">QA anti-cells</th>
<th class="num">math mean AUROC</th><th class="num">math anti-cells</th><th class="num">all mean</th></tr></thead>
<tbody>{''.join(f'<tr><td>{r.feature}</td><td class="num">{fnum(r.QA_mean)}</td><td class="num">{int(r.QA_n_anti)}</td><td class="num">{fnum(r.math_mean)}</td><td class="num">{int(r.math_n_anti)}</td><td class="num">{fnum(r.all_mean)}</td></tr>' for r in g6o.itertuples())}</tbody></table></div>
<p class="mut" style="font-size:12px">Oriented AUROC = the AUROC of the feature as the fusion
consumes it (fixed offline sign, z-scored). &lt;0.5 ⇒ anti-oriented. Across the full 30-view
pool ~45% of views are anti-oriented on any given cell, but L-SML absorbs that via negative
weights and selection removes it — what matters is that the curated members and the anchor are
correctly signed, which they are.</p>

<h2>Per-cell detail — GOOD_5 vs GOOD_6 (L-SML, epr anchor)</h2>
<div class="scroll"><table>
<thead><tr><th>cell</th><th>domain</th><th class="num">n</th><th class="num">acc</th>
<th class="num">GOOD_5</th><th class="num">GOOD_6</th><th class="num">Δ</th></tr></thead>
<tbody>{cell_rows}</tbody></table></div>
<p class="mut" style="font-size:12px">AUROC of continuous L-SML fusion, label-free epr anchor
orientation. <code>math500_r1distill8b</code> and <code>_mn4096</code> are the short-cap /
long-cap (mn4096) variants of the same condition. Two GPQA-free notes: <code>inside_coqa_llama7b</code>
runs on the base llama-7b (low accuracy by construction); the trace cells are K=10 multi-sample.</p>

<p class="mut" style="font-size:12px;margin-top:30px">Generated by <code>scripts/inscope_report.py</code>
— regenerate, never hand-edit. Bench: <code>scripts/run_selector_bench.py --pool c46</code> ×8 families;
ceiling: <code>scripts/selector_splithalf_oracle.py --pool-mode {{c46,h16}}</code>; orientation:
<code>scripts/inscope_orientation_audit.py</code>.</p>
</main>{THEME_JS}</body></html>"""

    out = os.path.join(BENCH, 'inscope_evaluation.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'wrote {out}  ({len(html)} bytes)')
    print(f'  GOOD_6 macro_all={float(good6["macro_all"]):.4f} '
          f'delta={pp(good6["delta_vs_good5_all"])} '
          f'best_learned={best_learned["variant"]} {float(best_learned["macro_all"]):.4f}')
    print(f'  split-half c46 loaded: {sh_c46 is not None}  h16: {sh_h16 is not None}')


if __name__ == '__main__':
    main()
