#!/usr/bin/env python
"""
selector_chosen_sets_report — one HTML page answering "what did the selector
actually pick, per cell?": the old grid-search-best subset (label-peeking
oracle, sweep_summary.csv) next to the new label-free a2.select choice
(GroupFS, Step 186), with both scores, plus a chosen-frequency table for
every pool feature (availability-aware — energy views exist on 7/19 cells).

Sources: results/selector_bench/{a2_groupfs__*,baselines,reference_macros__c46}.csv,
results/subset_sweep/sweep_summary.csv + *.manifest.json (per-cell H16 pool),
local_cache/repgrid_cells.pkl (c46 per-cell feature availability).

    python scripts/selector_chosen_sets_report.py   # -> results/selector_bench/chosen_sets.html
"""

import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from selector_deep_report import CSS, THEME_JS, esc  # noqa: E402
from spectral_utils.subset_sweep import GOOD_5  # noqa: E402

BENCH = os.path.join(REPO_DIR, 'results', 'selector_bench')
SWEEP = os.path.join(REPO_DIR, 'results', 'subset_sweep')
OUT = os.path.join(BENCH, 'chosen_sets.html')

GOOD5 = set(GOOD_5)
# Views not derived from token probabilities: trace_length (stopping property)
# + the four Z_n log-partition views (softmax destroys absolute logit scale).
NON_PROB = {'trace_length', 'epr_energy', 'min_energy',
            'sw_var_peak_energy', 'cusum_max_energy'}

EXTRA_CSS = """
.chip.g5{border:1px solid var(--accent);color:var(--accent)}
.chip.np{border:1px dashed var(--warn);color:var(--warn)}
.chipset{line-height:2.0;max-width:520px}
td.set{min-width:280px}
.bar{height:8px;border-radius:4px;background:var(--accent);display:inline-block;
  vertical-align:middle}
.bar.ref{background:var(--ref)}
"""


def pct(v):
    return '' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{100*v:.1f}'


def chips(feats, pool_n=None):
    feats = [f for f in feats if f]
    if pool_n and len(feats) == pool_n:
        return (f'<span class="chip">saturated — all {pool_n} pool features</span>')
    out = []
    for f in feats:
        cls = 'chip g5' if f in GOOD5 else ('chip np' if f in NON_PROB else 'chip')
        out.append(f'<span class="{cls}">{esc(f)}</span>')
    return f'<span class="chipset">{"".join(out)}</span>'


# ── dumbbell: GOOD_5 -> a2.select, oracle grid-best as a tick ────────────────

def dumbbell(rows, d0, d1, W=980):
    """rows: (label, good5, a2, oracle) in AUROC%. oracle may be None (c46 arm)."""
    ROW, TOP, X0 = 26, 10, 300
    X1 = W - 46
    n = len(rows)
    H = TOP + n * ROW + 34
    x = lambda v: X0 + (v - d0) / (d1 - d0) * (X1 - X0)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" preserveAspectRatio="xMidYMid meet" '
         f'style="width:100%;height:auto">']
    for gv in range(int(np.ceil(d0 / 10) * 10), int(d1) + 1, 10):
        s.append(f'<line x1="{x(gv):.1f}" y1="{TOP}" x2="{x(gv):.1f}" y2="{TOP+n*ROW}" '
                 f'stroke="var(--line)" stroke-width="1"/>')
        s.append(f'<text x="{x(gv):.1f}" y="{TOP+n*ROW+16}" text-anchor="middle" '
                 f'class="mut">{gv}</text>')
    s.append(f'<line x1="{x(50):.1f}" y1="{TOP}" x2="{x(50):.1f}" y2="{TOP+n*ROW}" '
             f'stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 3"/>')
    for i, (lbl, g5, a2, orc) in enumerate(rows):
        cy = TOP + i * ROW + ROW / 2
        s.append(f'<text x="{X0-8}" y="{cy+4:.1f}" text-anchor="end">{esc(lbl)}</text>')
        if orc is not None:
            s.append(f'<line x1="{x(orc):.1f}" y1="{cy-7:.1f}" x2="{x(orc):.1f}" '
                     f'y2="{cy+7:.1f}" stroke="var(--muted)" stroke-width="2.5">'
                     f'<title>grid-search best (oracle): {orc:.1f}</title></line>')
        s.append(f'<line x1="{x(g5):.1f}" y1="{cy:.1f}" x2="{x(a2):.1f}" y2="{cy:.1f}" '
                 f'stroke="var(--line)" stroke-width="1.5"/>')
        s.append(f'<circle cx="{x(g5):.1f}" cy="{cy:.1f}" r="5" fill="var(--ref)">'
                 f'<title>GOOD_5: {g5:.1f}</title></circle>')
        s.append(f'<circle cx="{x(a2):.1f}" cy="{cy:.1f}" r="5" fill="var(--accent)">'
                 f'<title>a2.select: {a2:.1f}</title></circle>')
    s.append(f'<text x="{(X0+X1)//2}" y="{H-2}" text-anchor="middle" class="mut">'
             f'AUROC (%) — dashed line = chance</text>')
    s.append('</svg>')
    return ''.join(s)


LEGEND = ('<p class="legend"><span class="dot" style="background:var(--ref)"></span>'
          'GOOD_5 (fixed) &nbsp; <span class="dot" style="background:var(--accent)">'
          '</span>a2.select (label-free, per cell) &nbsp; '
          '<span style="color:var(--muted)">▍</span> grid-search best subset '
          '(label-peeking oracle ceiling)</p>')


# ── data assembly ────────────────────────────────────────────────────────────

def load_h16_pools():
    """Per-cell sweep pool from the subset_sweep manifests: (domain, cell) -> [feat]."""
    pools = {}
    for m in glob.glob(os.path.join(SWEEP, '*.manifest.json')):
        d = json.load(open(m))
        pools[(d['domain'], d['cell_key'])] = list(d['pool'])
    return pools


def load_c46_avail():
    """Per-cell available features from the repgrid featcache (19 c46 cells)."""
    path = os.path.join(REPO_DIR, 'local_cache', 'repgrid_cells.pkl')
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        cells = pickle.load(f)
    out = {}
    for name, c in cells.items():
        feats = c['feats']
        out[name] = [k for k in feats
                     if np.isfinite(np.asarray(feats[k], dtype=float)).any()]
    return out


def main():
    sweep = pd.read_csv(os.path.join(SWEEP, 'sweep_summary.csv'))
    base = pd.read_csv(os.path.join(BENCH, 'baselines.csv'))
    h16 = pd.read_csv(os.path.join(BENCH, 'a2_groupfs__h16.csv'))
    c46 = pd.read_csv(os.path.join(BENCH, 'a2_groupfs__c46.csv'))
    refc = pd.read_csv(os.path.join(BENCH, 'reference_macros__c46.csv'))

    sel_h = h16[h16.variant == 'a2.select'].set_index(['domain', 'cell'])
    sel_c = c46[c46.variant == 'a2.select'].set_index('cell')
    g5_c = refc[refc.variant == 'ref.GOOD_5'].set_index('cell')
    g6_c = refc[refc.variant == 'ref.GOOD_6'].set_index('cell')
    sweep_ix = sweep.set_index(['domain', 'cell_key'])
    base_ix = base.set_index(['domain', 'cell'])

    # Step-182 augmentation arm: GOOD_5 + one extra view at a time (the
    # "GOOD_5 + logprobs" comparison from the Subset_Sweep_Report). Keep the
    # best-delta view per repgrid cell.
    aug = pd.read_csv(os.path.join(SWEEP, 'augmentation.csv'))
    aug = aug[(aug.domain == 'repgrid') & (aug.base == 'GOOD_5')]
    best_aug = (aug.loc[aug.groupby('cell_key').aug_auroc.idxmax()]
                .set_index('cell_key')[['view', 'aug_auroc']])

    h16_pools = load_h16_pools()
    c46_avail = load_c46_avail()

    # ---- h16 arm rows -------------------------------------------------------
    h_rows = []
    for (dom, cell), r in sel_h.sort_index().iterrows():
        sw = sweep_ix.loc[(dom, cell)] if (dom, cell) in sweep_ix.index else None
        bs = base_ix.loc[(dom, cell)] if (dom, cell) in base_ix.index else None
        h_rows.append({
            'domain': dom, 'cell': cell,
            'pool_n': int(r['p_pool']),
            'oracle': float(sw['best_auroc']) if sw is not None else np.nan,
            'best_feats': str(sw['best_feats']).split('|') if sw is not None else [],
            'good5': float(bs['good5_auroc']) if bs is not None else np.nan,
            'a2': float(r['auroc']),
            'a2_feats': str(r['chosen']).split('|'),
            'a2_size': int(r['size']),
        })

    # ---- c46 arm rows (19 repgrid cells) ------------------------------------
    c_rows = []
    for cell, r in sel_c.sort_index().iterrows():
        g5 = g5_c.loc[cell] if cell in g5_c.index else None
        sw = (sweep_ix.loc[('repgrid', cell)]
              if ('repgrid', cell) in sweep_ix.index else None)
        c_rows.append({
            'cell': cell, 'pool_n': int(r['p_pool']),
            'oracle_h16': float(sw['best_auroc']) if sw is not None else np.nan,
            'good5': float(g5['auroc']) if g5 is not None else np.nan,
            'good6': (float(g6_c.loc[cell, 'auroc'])
                      if cell in g6_c.index else np.nan),
            'aug_view': (str(best_aug.loc[cell, 'view'])
                         if cell in best_aug.index else ''),
            'aug': (float(best_aug.loc[cell, 'aug_auroc'])
                    if cell in best_aug.index else np.nan),
            'a2': float(r['auroc']),
            'a2_feats': str(r['chosen']).split('|'),
            'a2_size': int(r['size']),
        })

    # ---- chosen-frequency, availability-aware -------------------------------
    freq = {}
    for row in h_rows:
        pool = h16_pools.get((row['domain'], row['cell']))
        pool = pool if pool else [f for f in row['a2_feats']]
        for f in pool:
            freq.setdefault(f, [0, 0, 0, 0])[1] += 1
            if f in row['a2_feats']:
                freq[f][0] += 1
    for row in c_rows:
        avail = c46_avail.get(row['cell'])
        avail = avail if avail else row['a2_feats']
        for f in avail:
            freq.setdefault(f, [0, 0, 0, 0])[3] += 1
            if f in row['a2_feats']:
                freq[f][2] += 1

    # ---- render -------------------------------------------------------------
    B = []
    B.append('<h2>What this page answers</h2>')
    B.append(
        '<p>Per cell: the subset the <b>old exhaustive grid search</b> picked '
        '(<code>sweep_summary.csv</code> — an oracle: it scores every subset '
        'against the labels and reports the argmax, so it is a ceiling, not a '
        'method), the fixed <b>GOOD_5</b> default, and the subset the '
        '<b>label-free A2 GroupFS selector</b> (<code>a2.select</code>, Step 186) '
        'chose on its own — with all three AUROCs. Chips: '
        '<span class="chip g5">GOOD_5 member</span> '
        '<span class="chip np">non-probability view (trace_length / Z_n energy)</span> '
        '<span class="chip">other pool feature</span>.</p>')

    # h16 dumbbell + table
    B.append('<h2>Curated 16-feature pool — 51 cells</h2>')
    B.append(LEGEND)
    rows = [(f'{r["domain"]} · {r["cell"]}'[:46], 100 * r['good5'], 100 * r['a2'],
             100 * r['oracle'] if np.isfinite(r['oracle']) else None)
            for r in sorted(h_rows, key=lambda r: (r['domain'], -r['good5']))]
    lo = min(min(v for _, g, a, o in rows for v in (g, a)) - 3, 48)
    B.append(f'<div class="card">{dumbbell(rows, lo, 100)}</div>')

    B.append('<div class="scroll"><table><tr><th>domain</th><th>cell</th>'
             '<th class="num">grid best<br>(oracle)</th><th class="num">GOOD_5</th>'
             '<th class="num">a2.select</th><th class="num">Δ vs GOOD_5</th>'
             '<th class="num">size</th><th class="set">grid-search best subset</th>'
             '<th class="set">a2.select chosen subset</th></tr>')
    for r in sorted(h_rows, key=lambda r: (r['domain'], -r['good5'])):
        d = 100 * (r['a2'] - r['good5'])
        dc = 'good' if d > 0.5 else ('badv' if d < -0.5 else 'mut')
        B.append(
            f'<tr><td>{esc(r["domain"])}</td><td>{esc(r["cell"])}</td>'
            f'<td class="num mut">{pct(r["oracle"])}</td>'
            f'<td class="num">{pct(r["good5"])}</td>'
            f'<td class="num k">{pct(r["a2"])}</td>'
            f'<td class="num {dc}">{d:+.1f}</td>'
            f'<td class="num">{r["a2_size"]}/{r["pool_n"]}</td>'
            f'<td class="set">{chips(r["best_feats"])}</td>'
            f'<td class="set">{chips(r["a2_feats"], r["pool_n"])}</td></tr>')
    B.append('</table></div>')

    # c46 dumbbell + table
    B.append('<h2>Full 46-view pool — 19 replication-grid cells</h2>')
    B.append('<p>No exhaustive grid search exists on this pool (2<sup>46</sup> '
             'subsets) — the oracle tick is the <i>H16-pool</i> grid best, shown '
             'as context only; a2.select here draws on the full pool including '
             'logprob and (on 7 cells) Z<sub>n</sub> energy views.</p>')
    B.append(LEGEND)
    rows = [(r['cell'][:46], 100 * r['good5'], 100 * r['a2'],
             100 * r['oracle_h16'] if np.isfinite(r['oracle_h16']) else None)
            for r in sorted(c_rows, key=lambda r: -r['good5'])]
    lo = min(min(v for _, g, a, o in rows for v in (g, a)) - 3, 48)
    B.append(f'<div class="card">{dumbbell(rows, lo, 100)}</div>')

    B.append(
        '<p>The <b>GOOD_5 + one view</b> columns are the Step-182 augmentation arm '
        '(<code>augmentation.csv</code>): add a single extra view (a logprob, energy, '
        'spilled, or H16 view) to GOOD_5 and re-fuse — shown here as the best such '
        'view per cell — note "best per cell" is picked by AUROC, so that column is '
        'mildly label-peeking (a per-cell ceiling over 17 candidate views). '
        '<b>GOOD_6</b> is the promoted case (GOOD_5 + varentropy) — the honest '
        'fixed recipe; a2.select is the free label-free selection.</p>')
    B.append('<div class="scroll"><table><tr><th>cell</th>'
             '<th class="num">grid best<br>(H16 oracle)</th><th class="num">GOOD_5</th>'
             '<th class="num">GOOD_6</th>'
             '<th>best GOOD_5+view</th><th class="num">its AUROC</th>'
             '<th class="num">a2.select</th><th class="num">Δ vs GOOD_5</th>'
             '<th class="num">size</th><th class="set">a2.select chosen subset</th></tr>')
    for r in sorted(c_rows, key=lambda r: -r['good5']):
        d = 100 * (r['a2'] - r['good5'])
        dc = 'good' if d > 0.5 else ('badv' if d < -0.5 else 'mut')
        B.append(
            f'<tr><td>{esc(r["cell"])}</td>'
            f'<td class="num mut">{pct(r["oracle_h16"])}</td>'
            f'<td class="num">{pct(r["good5"])}</td>'
            f'<td class="num">{pct(r["good6"])}</td>'
            f'<td>{chips([r["aug_view"]]) if r["aug_view"] else "—"}</td>'
            f'<td class="num">{pct(r["aug"])}</td>'
            f'<td class="num k">{pct(r["a2"])}</td>'
            f'<td class="num {dc}">{d:+.1f}</td>'
            f'<td class="num">{r["a2_size"]}/{r["pool_n"]}</td>'
            f'<td class="set">{chips(r["a2_feats"], r["pool_n"])}</td></tr>')
    B.append('</table></div>')

    # frequency table
    B.append('<h2>How often does a2.select pick each feature?</h2>')
    B.append('<p>Chosen / available. A feature counts as available only when the '
             'cell actually has it (the 4 Z<sub>n</sub> energy views exist on 7 of '
             'the 19 c46 cells; logprob views on the AIRCC-captured cells only). '
             'Dashed rows are the non-probability views.</p>')
    B.append('<div class="scroll"><table><tr><th>feature</th>'
             '<th class="num">h16 arm (51 cells)</th><th></th>'
             '<th class="num">c46 arm (19 cells)</th><th></th>'
             '<th>probability-derived?</th></tr>')

    def rate_cells(ch, av):
        if av == 0:
            return '<td class="num mut">—</td><td></td>'
        w = int(120 * ch / av)
        return (f'<td class="num">{ch}/{av}</td>'
                f'<td><span class="bar" style="width:{w}px"></span></td>')

    order = sorted(freq, key=lambda f: -(freq[f][2] / max(freq[f][3], 1)
                                         + freq[f][0] / max(freq[f][1], 1)))
    for f in order:
        ch_h, av_h, ch_c, av_c = freq[f]
        mark = ('<span class="warnv">no — ' +
                ('generation length' if f == 'trace_length' else 'logit scale (Z<sub>n</sub>)') +
                '</span>') if f in NON_PROB else 'yes'
        name = (f'<span class="chip g5">{esc(f)}</span>' if f in GOOD5
                else f'<span class="chip np">{esc(f)}</span>' if f in NON_PROB
                else esc(f))
        B.append(f'<tr><td>{name}</td>{rate_cells(ch_h, av_h)}'
                 f'{rate_cells(ch_c, av_c)}<td>{mark}</td></tr>')
    B.append('</table></div>')

    B.append('<div class="note">Reading note: a2.select saturates (keeps the whole '
             'pool) on 8 of 19 c46 cells and 13 of 51 h16 cells — on those rows the '
             '"chosen subset" carries no selection information, which inflates every '
             'feature\'s frequency equally. The Δ column is the honest summary: '
             'selection roughly ties GOOD_5 on the 46-view pool and loses to it on '
             'the curated 16-pool (Step 186 verdict).</div>')

    html = (f'<title>Chosen feature sets — grid search vs label-free selection</title>\n'
            f'<style>{CSS}{EXTRA_CSS}</style>\n<main>\n'
            f'<p class="eyebrow">Step 186-187 · label-free feature-subset selection</p>\n'
            f'<h1>Which features were chosen, cell by cell</h1>'
            f'<p class="sub">Old exhaustive grid search (oracle) vs GOOD_5 vs the '
            f'label-free A2 GroupFS selection — sets and scores side by side.</p>\n'
            f'{"".join(B)}\n</main>\n{THEME_JS}')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'wrote {OUT} ({len(html)/1024:.0f} KB, {len(h_rows)} h16 + {len(c_rows)} c46 cells)')


if __name__ == '__main__':
    main()
