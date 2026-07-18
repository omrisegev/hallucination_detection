#!/usr/bin/env python
"""
selector_viz — self-contained HTML dashboard of the feature-selection bench
(Step 186). Reads every results/selector_bench/*__{h16,c46}.csv plus the sweep
comparators, and renders one theme-aware page: a best-per-family bar chart with
GOOD_5 / random / oracle reference lines, and the full per-variant leaderboard
with per-domain columns. No pass/fail gating — all methods shown, sorted by
macro AUROC, for the researcher to read and choose.

Design: dataviz skill. Categorical hue per method family (validated slots 1-6,
adjacent pairlist); direct value labels + full table satisfy the contrast-relief
rule; light/dark both selected.

    python scripts/selector_viz.py            # -> results/selector_bench/dashboard.html
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# variant-prefix -> (family key, display label, light hue, dark hue)
FAMILIES = [
    ('ref.',      'reference', 'Reference macros (fixed subsets)', '#2a78d6', '#3987e5'),
    ('a1.',       'residual',  'Residual-guided (Nadler/Kluger)',  '#008300', '#008300'),
    ('lapscore',  'classical', 'Classical spectral FS',            '#e87ba4', '#d55181'),
    ('spec',      'classical', 'Classical spectral FS',            '#e87ba4', '#d55181'),
    ('mcfs',      'classical', 'Classical spectral FS',            '#e87ba4', '#d55181'),
    ('random',    'stats',     'Simple-stats floor',               '#eda100', '#c98500'),
    ('mad',       'stats',     'Simple-stats floor',               '#eda100', '#c98500'),
    ('kurtosis',  'stats',     'Simple-stats floor',               '#eda100', '#c98500'),
    ('decorr',    'stats',     'Simple-stats floor',               '#eda100', '#c98500'),
    ('a2.',       'groupfs',   'GroupFS (Lindenbaum, AAAI 2026)',  '#1baf7a', '#199e70'),
    ('a3.',       'cae',       'Concrete-AE (ICML 2019)',          '#eb6834', '#d95926'),
    ('a4.',       'antigrav',  'Antigravity (anchor/intrinsic-dim/CSSP)', '#8a4fd6', '#7a3fc0'),
    ('a5.',       'mrmr',      'mRMR hybrid (relevance-redundancy)', '#0891b2', '#0e7490'),
    ('epr.',      'reference', 'Reference macros (fixed subsets)', '#2a78d6', '#3987e5'),
]
FAM_ORDER = ['reference', 'residual', 'classical', 'stats', 'groupfs', 'cae',
            'antigrav', 'mrmr']
FAM_LABEL = {k: lbl for _, k, lbl, _, _ in FAMILIES}
FAM_LIGHT = {k: lo for _, k, _, lo, _ in FAMILIES}
FAM_DARK = {k: dk for _, k, _, _, dk in FAMILIES}
DOMAINS = ['math500', 'gsm8k', 'gpqa', 'rag', 'qa', 'repgrid', 'trace']


def family_of(variant):
    for pref, key, *_ in FAMILIES:
        if variant.startswith(pref):
            return key
    return 'other'


def load(bench_dir, sweep_dir):
    files = sorted(glob.glob(os.path.join(bench_dir, '*__h16.csv')) +
                   glob.glob(os.path.join(bench_dir, '*__c46.csv')))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['auroc'] = pd.to_numeric(df['auroc'], errors='coerce')
    df['family'] = df['variant'].map(family_of)
    ss = pd.read_csv(os.path.join(sweep_dir, 'sweep_summary.csv')).rename(
        columns={'cell_key': 'cell', 'best_auroc': 'oracle_auroc'})
    return df, ss


def summarize(df, pool):
    """Per-variant macro + per-domain macro (NaN AUROC = loss)."""
    d = df[df['pool_mode'] == pool]
    rows = []
    for variant, g in d.groupby('variant'):
        auc = g.set_index('cell')['auroc']
        auc_loss = auc.fillna(0.0)
        rec = {'variant': variant, 'family': family_of(variant),
               'n_cells': len(g), 'macro': float(auc_loss.mean()),
               'mean_size': float(g['size'].mean()),
               'mean_pctile': float(pd.to_numeric(g['pctile_within_size'],
                                                  errors='coerce').mean()),
               'n_fallback': int(g['fallback'].astype(str)
                                 .isin(['True', 'true', '1']).sum())}
        for dom in DOMAINS:
            gd = g[g['domain'] == dom]
            rec[dom] = float(gd['auroc'].fillna(0.0).mean()) if len(gd) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def refs(ss, pool):
    """GOOD_5 / random-median / oracle macro reference lines for a pool."""
    if pool == 'c46':
        ss = ss[ss['domain'] == 'repgrid']
    return {'GOOD_5': float(ss['good5_auroc'].mean()),
            'oracle': float(ss['oracle_auroc'].mean()),
            'n': int(len(ss))}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _bar_chart(summ, ref, pool):
    """Best-variant-per-family horizontal bars + reference lines (inline SVG)."""
    best = (summ.sort_values('macro', ascending=False)
            .groupby('family', as_index=False).first())
    best = best[best['family'].isin(FAM_ORDER)]
    best['ord'] = best['family'].map(lambda f: FAM_ORDER.index(f))
    best = best.sort_values('macro', ascending=False)
    rows = best.to_dict('records')

    lo, hi = 0.50, 0.78
    W, rowh, padL, padR, padT = 720, 42, 210, 60, 44
    H = padT + rowh * len(rows) + 40

    def x(v):
        return padL + (max(lo, min(hi, v)) - lo) / (hi - lo) * (W - padL - padR)

    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
           f'aria-label="best variant per family, {pool}">']
    # reference lines
    for name, val, col in [('random', ref.get('random', np.nan), 'var(--muted)'),
                           ('GOOD_5', ref['GOOD_5'], 'var(--ink2)'),
                           ('oracle', ref['oracle'], 'var(--ink2)')]:
        if not np.isfinite(val):
            continue
        xv = x(val)
        dash = '' if name == 'GOOD_5' else ' stroke-dasharray="4 3"'
        svg.append(f'<line x1="{xv:.1f}" y1="{padT-8}" x2="{xv:.1f}" y2="{H-40}" '
                   f'stroke="{col}"{dash} stroke-width="1.5"/>')
        svg.append(f'<text x="{xv:.1f}" y="{padT-14}" text-anchor="middle" '
                   f'class="reflab">{name} {val:.3f}</text>')
    # bars
    for i, r in enumerate(rows):
        y = padT + i * rowh + 6
        xv = x(r['macro'])
        col = f'var(--fam-{r["family"]})'
        svg.append(f'<text x="{padL-10}" y="{y+16}" text-anchor="end" '
                   f'class="barlab">{FAM_LABEL[r["family"]].split(" (")[0]}</text>')
        svg.append(f'<rect x="{padL}" y="{y}" width="{max(xv-padL,1):.1f}" '
                   f'height="22" rx="4" fill="{col}"/>')
        svg.append(f'<text x="{xv+6:.1f}" y="{y+16}" class="barval">'
                   f'{r["macro"]:.3f}</text>')
        svg.append(f'<text x="{padL+6}" y="{y+16}" class="barin">'
                   f'{r["variant"]}</text>')
    # axis ticks
    for t in np.arange(0.50, 0.79, 0.05):
        xv = x(t)
        svg.append(f'<line x1="{xv:.1f}" y1="{H-40}" x2="{xv:.1f}" y2="{H-36}" '
                   f'stroke="var(--muted)"/>')
        svg.append(f'<text x="{xv:.1f}" y="{H-24}" text-anchor="middle" '
                   f'class="axlab">{t:.2f}</text>')
    svg.append('</svg>')
    return '\n'.join(svg)


def _leaderboard(summ, ref):
    dom_cols = ''.join(f'<th>{d}</th>' for d in DOMAINS)
    head = ('<tr><th>variant</th><th>family</th><th>size</th><th>macro</th>'
            f'<th>Δ GOOD_5</th><th>pctile</th>{dom_cols}<th>n</th></tr>')
    g5 = ref['GOOD_5']
    body = []
    order = summ.assign(ord=summ['family'].map(
        lambda f: FAM_ORDER.index(f) if f in FAM_ORDER else 9))
    order = order.sort_values(['ord', 'macro'], ascending=[True, False])
    for r in order.to_dict('records'):
        delta = r['macro'] - g5
        dcol = 'pos' if delta > 0.001 else ('neg' if delta < -0.001 else '')
        chip = (f'<span class="chip" style="background:var(--fam-{r["family"]})">'
                f'</span>{r["family"]}')
        doms = ''.join(
            f'<td class="num">{r[d]:.3f}</td>' if np.isfinite(r[d]) else '<td class="num">·</td>'
            for d in DOMAINS)
        pct = f'{r["mean_pctile"]:.0f}' if np.isfinite(r['mean_pctile']) else '·'
        fb = f' <span class="fb">fb{r["n_fallback"]}</span>' if r['n_fallback'] else ''
        body.append(
            f'<tr><td class="var">{r["variant"]}{fb}</td><td>{chip}</td>'
            f'<td class="num">{r["mean_size"]:.1f}</td>'
            f'<td class="num strong">{r["macro"]:.3f}</td>'
            f'<td class="num {dcol}">{delta:+.3f}</td>'
            f'<td class="num">{pct}</td>{doms}'
            f'<td class="num">{r["n_cells"]}</td></tr>')
    return f'<table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>'


def render(df, ss, out_html):
    parts = []
    for pool, title in [('h16', 'H16 pool — 51 cells (all domains)'),
                        ('c46', 'Full 46-view pool — 19 repgrid cells')]:
        summ = summarize(df, pool)
        if summ.empty:
            continue
        ref = refs(ss, pool)
        # random-median reference from the stats family's random variants
        rnd = summ[summ['variant'].str.startswith('random')]['macro']
        ref['random'] = float(rnd.mean()) if len(rnd) else np.nan
        parts.append(f'<section><h2>{title}</h2>'
                     f'<p class="sub">GOOD_5 {ref["GOOD_5"]:.3f} · random '
                     f'{ref.get("random", float("nan")):.3f} · oracle ceiling '
                     f'{ref["oracle"]:.3f} (label-peeking) · {ref["n"]} cells</p>'
                     f'<div class="chart">{_bar_chart(summ, ref, pool)}</div>'
                     f'<details open><summary>Full leaderboard '
                     f'({len(summ)} variants, sorted within family)</summary>'
                     f'<div class="tablewrap">{_leaderboard(summ, ref)}</div>'
                     f'</details></section>')

    fam_vars = '\n'.join(
        f'    --fam-{k}: {FAM_LIGHT[k]};' for k in FAM_ORDER)
    fam_vars_d = '\n'.join(
        f'    --fam-{k}: {FAM_DARK[k]};' for k in FAM_ORDER)
    legend = ' '.join(
        f'<span class="lg"><span class="chip" style="background:var(--fam-{k})">'
        f'</span>{FAM_LABEL[k]}</span>' for k in FAM_ORDER)

    html = f"""<h1>Feature-selection bench — all methods</h1>
<p class="lede">Each method selects a feature subset <b>label-free</b> from the
unlabeled trace features; the <b>same L-SML</b> then fuses that subset. AUROC is
raw (anchor-oriented). Nothing is gated pass/fail — every variant is shown,
sorted by macro AUROC, for you to read and choose. GOOD_5 and the other curated
macros are baselines; the oracle line is a label-peeking ceiling, not achievable.</p>
<div class="legend">{legend}</div>
{''.join(parts)}
<style>
.viz-root{{color-scheme:light;--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;
  --ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--pos:#006300;--neg:#c0392b;
{fam_vars}}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme=light])) .viz-root{{
  color-scheme:dark;--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;
  --muted:#898781;--grid:#2c2c2a;--pos:#0ca30c;--neg:#e66767;
{fam_vars_d}}}}}
:root[data-theme=dark] .viz-root{{color-scheme:dark;--surface:#1a1a19;--plane:#0d0d0d;
  --ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--pos:#0ca30c;--neg:#e66767;
{fam_vars_d}}}
.viz-root{{background:var(--plane);color:var(--ink);font:14px/1.5 system-ui,
  -apple-system,"Segoe UI",sans-serif;padding:24px;max-width:1080px;margin:0 auto}}
h1{{font-size:22px;margin:0 0 6px}} h2{{font-size:16px;margin:26px 0 2px}}
.lede{{color:var(--ink2);max-width:70ch}} .sub{{color:var(--ink2);font-size:13px;margin:2px 0 10px}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;margin:14px 0 4px;font-size:13px}}
.lg{{display:inline-flex;align-items:center;gap:6px;color:var(--ink2)}}
.chip{{width:11px;height:11px;border-radius:3px;display:inline-block;flex:none}}
.chart{{background:var(--surface);border:1px solid var(--grid);border-radius:10px;
  padding:10px 8px;margin:8px 0}}
.reflab{{fill:var(--ink2);font-size:11px}} .axlab{{fill:var(--muted);font-size:10px;
  font-variant-numeric:tabular-nums}}
.barlab{{fill:var(--ink2);font-size:12px}} .barval{{fill:var(--ink);font-size:12px;
  font-weight:600;font-variant-numeric:tabular-nums}}
.barin{{fill:var(--surface);font-size:11px;opacity:.9}}
.tablewrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:12.5px;
  margin-top:8px}} th,td{{text-align:left;padding:4px 8px;border-bottom:1px solid var(--grid)}}
th{{color:var(--ink2);font-weight:600;position:sticky;top:0;background:var(--plane)}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}} td.strong{{font-weight:700}}
td.var{{font-family:ui-monospace,monospace;font-size:11.5px}}
.pos{{color:var(--pos)}} .neg{{color:var(--neg)}}
.fb{{color:var(--neg);font-size:10px}} details summary{{cursor:pointer;color:var(--ink2);
  font-size:13px;margin:6px 0}}
</style>
<div class="viz-root" hidden></div>"""
    # wrap the whole body so the CSS vars apply
    html = ('<div class="viz-root">' + html.replace(
        '<div class="viz-root" hidden></div>', '') + '</div>')
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html)
    return out_html


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bench-dir',
                    default=os.path.join(REPO_DIR, 'results', 'selector_bench'))
    ap.add_argument('--sweep-dir',
                    default=os.path.join(os.environ.get('HD_DATA_ROOT', REPO_DIR),
                                         'results', 'subset_sweep'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.bench_dir, 'dashboard.html')
    df, ss = load(args.bench_dir, args.sweep_dir)
    render(df, ss, out)
    fams = sorted(df['family'].unique())
    print(f"families present: {fams}")
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
