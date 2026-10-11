"""Add two advisor views to the Codex presentation (advisor_review_claude.html):
  1. bank x method tables (within-answer AUC and Q80 PRMScore) with a mean per method and a mean per bank;
  2. a PRMBench-leaderboard view: the official radar over the 9 sub-categories and the paper's Table 3 layout.
Usage: python3 add_advisor_views.py <repo_root> <official_prmscore_dir> <in_html> <out_html>
  repo_root: checkout of omrisegev/hallucination_detection at claude/thinkfix-rescore-v1 (080877e or later)."""
import sys, math
import numpy as np, pandas as pd
from pathlib import Path

REPO, OFF, IN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
D = 'results/thinkfix_rescore_v1'
F_PB = f'{D}/PRMBENCH_COMPARISON.csv'
BANKS = ['N11', 'N13', 'N22', 'N40', 'N53', 'N14', 'N16', 'N25', 'N43', 'N56']
METH = ['ALL_EQ', 'DS_FILTER_EQ', 'LSML_CONT', 'IU_PCR', 'HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']
NAME = dict(zip(METH, ['Plain average', 'DS-filter average', 'Continuous L-SML', 'IU-PCR', 'HEM / cc', 'MEAN / cc', 'MEAN / ga', 'TCI / ga']))
NEW = 'thinkfix_mask1'

pb = pd.read_csv(REPO / F_PB)
pnl, cb = pb[pb.table == 'prmbench_panel_b'], pb[pb.table == 'cont_baselines']
lrr = pb[(pb.table == 'lr_reference') & (pb.scorer == 'LR')]
off = pd.concat([pd.read_csv(OFF / f'OFFICIAL_PRMSCORE_{b}.csv') for b in BANKS])

def auc(b, m):
    r = cb[(cb.bank == b) & (cb.scorer == m) & (cb.metric == 'prmb_within_auc')] if m in ('LSML_CONT', 'IU_PCR') \
        else pnl[(pnl.bank == b) & (pnl.scorer == m) & (pnl.metric == 'within_auc')]
    return float(r[NEW].iloc[0])
def q80(b, m):
    return float(off[(off.bank == b) & (off.method == m) & (off.rule == 'R0')].overall.iloc[0])
def lr_auc(b):
    return float(lrr[(lrr.bank == b) & (lrr.metric == 'prmb_within_auc')][NEW].iloc[0])

def matrix(fn, caption, lr=None, note=''):
    X = np.array([[fn(b, m) for m in METH] for b in BANKS])
    h = '<tr><th>Bank</th>' + ''.join(f'<th>{NAME[m]}</th>' for m in METH) + '<th class="mm">Bank mean</th>' + ('<th class="lr">LR (supervised)</th>' if lr else '') + '</tr>'
    rows = []
    for i, b in enumerate(BANKS):
        best = X[i].max()
        rows.append(f'<tr{" class=hl" if b == "N16" else ""}><th>{b}</th>' + ''.join(f'<td{" class=b" if v == best else ""}>{v:.3f}</td>' for v in X[i])
                    + f'<td class="mm">{X[i].mean():.3f}</td>' + (f'<td class="lr">{lr(b):.3f}</td>' if lr else '') + '</tr>')
    M = X.mean(axis=0)
    rows.append('<tr class="mr"><th>Method mean</th>' + ''.join(f'<td{" class=b" if v == M.max() else ""}>{v:.3f}</td>' for v in M)
                + f'<td class="mm">{X.mean():.3f}</td>' + (f'<td class="lr">{np.mean([lr(b) for b in BANKS]):.3f}</td>' if lr else '') + '</tr>')
    return f'<div class="mt"><table><caption>{caption}</caption>{h}{"".join(rows)}</table></div><p class="small">{note}</p>'

T1 = matrix(auc, 'Within-answer AUROC · all banks × all label-free methods (corrected + mask1)', lr_auc,
            'Bold = highest label-free value in the row or in the mean row. Means are equal-weight and descriptive; LR is not in the bank mean. '
            f'Source: <code>{F_PB}</code>, column <code>{NEW}</code> (tables <code>prmbench_panel_b</code>, <code>cont_baselines</code>, <code>lr_reference</code>).')
T2 = matrix(q80, 'PRMScore with the global answer-z Q80 rule · all banks × all label-free methods', None,
            'Pooled over the 6,211 non-control answers by the repository port of the PRMBench evaluator. Qwen2.5-Math-PRM-7B with its own 0.5 threshold: 0.655. '
            'Source: <code>official_prmscore_v1.py</code> outputs, branch <code>claude/advisor-page-v6-official-prmscore</code>, '
            '<code>results/advisor_page_v6_2026-10-11/official_prmscore_thinkfix_mask1/</code>.')

# ---- leaderboard view (PRMBench paper layout and radar)
SUB = [('NR', 'redundency'), ('NCL', 'circular'), ('MS', 'multi_solutions'), ('DR', 'deception'), ('PS', 'missing_condition'),
       ('CI', 'confidence'), ('DC', 'domain_inconsistency'), ('SC', 'step_contradiction'), ('ES', 'counterfactual')]
# arXiv 2501.03124v1, Table 3 (percent). Columns: Overall, NR, NCL, ES, SC, DC, CI, PS, DR, MS
PAPER = {
    'o1-mini': (68.8, 65.6, 63.7, 74.5, 67.7, 73.8, 72.3, 61.8, 64.8, 100.0),
    'GPT-4o': (66.8, 57.0, 62.4, 72.0, 69.7, 70.7, 71.1, 62.5, 65.7, 99.2),
    'Gemini-2.0-thinking-exp-1219': (68.8, 68.5, 63.8, 72.9, 71.3, 71.0, 71.8, 60.3, 65.7, 99.8),
    'ReasonEval-34B': (60.5, 54.8, 48.1, 66.4, 60.3, 57.8, 67.5, 57.7, 64.3, 97.2),
    'ReasonEval-7B': (60.0, 61.0, 50.1, 62.1, 65.9, 61.5, 66.0, 55.6, 58.0, 99.5),
    'RLHFlow-PRM-Deepseek-8B': (54.2, 46.4, 48.9, 55.7, 55.0, 53.2, 66.2, 49.0, 55.4, 99.8),
    'MathShepherd-Mistral-7B': (47.0, 44.0, 50.3, 49.4, 44.5, 41.3, 47.7, 47.2, 48.6, 86.1),
    'Skywork-PRM-7B': (36.2, 35.7, 41.2, 36.7, 29.1, 30.6, 34.4, 36.8, 37.4, 88.8),
}
KEYS = ['overall', 'NR', 'NCL', 'ES', 'SC', 'DC', 'CI', 'PS', 'DR', 'MS']
def paper_row(v): return dict(zip(KEYS, [x / 100 for x in v]))
def ours_row(q):
    m = {'overall': q.overall}
    for k, c in SUB: m[k] = float(q[c])
    return m
o16 = off[off.bank == 'N16']
OURS = {'Continuous L-SML · N16 · Q80 (ours, label-free)': ours_row(o16[(o16.method == 'LSML_CONT') & (o16.rule == 'R0')].iloc[0]),
        'IU-PCR · N16 · Q80 (ours, label-free)': ours_row(o16[(o16.method == 'IU_PCR') & (o16.rule == 'R0')].iloc[0]),
        'Qwen2.5-Math-PRM-7B · 0.5 (our evaluator run)': ours_row(off[off.bank == '-'].iloc[0])}
def cats(r):
    return dict(simp=np.mean([r['NR'], r['NCL']]), sound=np.mean([r['ES'], r['SC'], r['DC'], r['CI']]), sens=np.mean([r['PS'], r['DR'], r['MS']]))

H = ('<tr><th rowspan="2">Model</th><th rowspan="2">Overall</th><th colspan="3">Simplicity</th><th colspan="5">Soundness</th><th colspan="4">Sensitivity</th></tr>'
     '<tr><th>NR</th><th>NCL</th><th class="mm">Avg</th><th>ES</th><th>SC</th><th>DC</th><th>CI</th><th class="mm">Avg</th><th>PS</th><th>DR</th><th>MS</th><th class="mm">Avg</th></tr>')
def lrow(name, r, cls=''):
    c = cats(r); f = lambda x: f'{100 * x:.1f}'
    return (f'<tr{cls}><th>{name}</th><td><b>{f(r["overall"])}</b></td><td>{f(r["NR"])}</td><td>{f(r["NCL"])}</td><td class="mm">{f(c["simp"])}</td>'
            f'<td>{f(r["ES"])}</td><td>{f(r["SC"])}</td><td>{f(r["DC"])}</td><td>{f(r["CI"])}</td><td class="mm">{f(c["sound"])}</td>'
            f'<td>{f(r["PS"])}</td><td>{f(r["DR"])}</td><td>{f(r["MS"])}</td><td class="mm">{f(c["sens"])}</td></tr>')
rows = [lrow(n, r, ' class=hl' if 'ours' in n else '') for n, r in OURS.items()]
rows.append('<tr class="sep"><td colspan="14">Published in the PRMBench paper (arXiv 2501.03124v1, Table 3)</td></tr>')
rows += [lrow(n, paper_row(v), ' class=pub') for n, v in PAPER.items()]
T3 = (f'<div class="mt"><table class="lb"><caption>PRMBench leaderboard layout · PRMScore × 100</caption>{H}{"".join(rows)}</table></div>'
      '<p class="small">Same layout and category averages as the PRMBench paper: NR = non-redundancy (redundency), NCL = non-circular logic, '
      'ES = empirical soundness (counterfactual), SC = step consistency (step_contradiction), DC = domain consistency, CI = confidence invariance, '
      'PS = prerequisite sensitivity (missing_condition), DR = deception resistance, MS = multi-solution consistency (valid-step F1; no error steps). '
      'Overall is pooled over all steps, so it is not the mean of the three averages. Here the Sensitivity average includes MS, as in the paper. '
      'Our rows use the label-free Q80 rule and our port of the evaluator; the paper rows are copied from its Table 3 and were not re-run. '
      'The live leaderboard on prmbench.github.io may differ from paper v1.</p>')

# radar (same axis order as the PRMBench teaser figure)
SER = [('Continuous L-SML · Q80', OURS['Continuous L-SML · N16 · Q80 (ours, label-free)'], '#246c9c', '', 2.6),
       ('IU-PCR · Q80', OURS['IU-PCR · N16 · Q80 (ours, label-free)'], '#2a9d8f', '6 3', 2.2),
       ('Qwen2.5-Math-PRM-7B', OURS['Qwen2.5-Math-PRM-7B · 0.5 (our evaluator run)'], '#bf6235', '', 2.2),
       ('o1-mini (paper)', paper_row(PAPER['o1-mini']), '#7a6aa8', '2 3', 1.6),
       ('GPT-4o (paper)', paper_row(PAPER['GPT-4o']), '#8a8a8a', '5 3', 1.6),
       ('ReasonEval-34B (paper)', paper_row(PAPER['ReasonEval-34B']), '#b9a33a', '1 3', 1.6)]
CX, CY, R0, R1, LO, HI = 300, 265, 0, 200, 0.2, 1.0
def pt(i, v):
    a = -math.pi / 2 + 2 * math.pi * i / len(SUB); r = R0 + (R1 - R0) * (max(v, LO) - LO) / (HI - LO)
    return CX + r * math.cos(a), CY + r * math.sin(a)
g = []
for lv in [0.4, 0.6, 0.8, 1.0]:
    g.append('<polygon points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in (pt(i, lv) for i in range(len(SUB)))) + '" fill="none" stroke="#d6dfe4"/>')
    x, y = pt(0, lv); g.append(f'<text x="{x + 4:.0f}" y="{y + 4:.0f}" font-size="10" fill="#5f707a">{int(lv * 100)}</text>')
for i, (k, _) in enumerate(SUB):
    x, y = pt(i, HI); g.append(f'<line x1="{CX}" y1="{CY}" x2="{x:.1f}" y2="{y:.1f}" stroke="#d6dfe4"/>')
    lx, ly = pt(i, HI + 0.09); g.append(f'<text x="{lx:.0f}" y="{ly + 4:.0f}" font-size="13" font-weight="600" text-anchor="middle" fill="#21323d">{k}</text>')
for n, r, col, dash, w in SER[::-1]:
    P = ' '.join(f'{x:.1f},{y:.1f}' for x, y in (pt(i, r[k]) for i, (k, _) in enumerate(SUB)))
    fill = 'none'
    g.append(f'<polygon points="{P}" fill="{fill}" fill-opacity="0.08" stroke="{col}" stroke-width="{w}" stroke-dasharray="{dash}"/>')
    for i, (k, _) in enumerate(SUB):
        x, y = pt(i, r[k]); g.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2.6 if w > 2 else 1.8}" fill="{col}"/>')
leg = ''.join(f'<g transform="translate(610,{70 + 30 * j})"><line x1="0" y1="0" x2="34" y2="0" stroke="{c}" stroke-width="{w}" stroke-dasharray="{d}"/>'
              f'<text x="44" y="4" font-size="13" fill="#21323d">{n}</text></g>' for j, (n, _, c, d, w) in enumerate(SER))
RADAR = (f'<figure><svg viewBox="0 0 900 530" role="img" aria-label="Radar of PRMScore per PRMBench sub-category">{"".join(g)}{leg}'
         '<text x="610" y="275" font-size="12" fill="#5f707a">Radius: sub-category score × 100,</text>'
         '<text x="610" y="292" font-size="12" fill="#5f707a">from 20 (centre) to 100 (outer ring).</text></svg>'
         '<figcaption>PRMBench sub-category scores in the axis order of the official PRMBench figure. Ours (N16, label-free, Q80) against '
         'Qwen2.5-Math-PRM-7B (our evaluator run) and three models from the PRMBench paper (Table 3). Different decision rules; development data.</figcaption></figure>')

ls, isc = OURS['Continuous L-SML · N16 · Q80 (ours, label-free)'], OURS['IU-PCR · N16 · Q80 (ours, label-free)']
prm = OURS['Qwen2.5-Math-PRM-7B · 0.5 (our evaluator run)']
best_open = max(v[0] for k, v in PAPER.items() if k.startswith(('ReasonEval', 'RLHF', 'Math', 'Sky')))
SEC = f'''<section class="section" id="leaderboard"><header><span class="num">05b</span><h2>Where we sit on the PRMBench leaderboard</h2></header>
<p>The radar uses the same nine axes as the official PRMBench figure. Continuous L-SML and IU-PCR (label-free, Q80) reach an overall PRMScore of {100 * ls["overall"]:.1f} and {100 * isc["overall"]:.1f}.
Qwen2.5-Math-PRM-7B gives {100 * prm["overall"]:.1f} in our run. The best open PRM in the paper's table gives {best_open:.1f}; the proprietary LLMs give 66.0 to 68.8.</p>
{RADAR}
<div class="two"><div><h3>Where the shape differs</h3><p>On NR and NCL (simplicity) we are higher than every model in the table, and simplicity is the weakest category for most published models. IU-PCR lies almost on top of L-SML. We are low on MS: Q80 flags about 20% of the steps in an answer with no error, so valid multi-solution answers lose points. PRM-7B is higher on ES, CI and DR.</p></div>
<div><h3>How to read the comparison</h3><p>Our rows and the PRM-7B row use one evaluator port and one export. The paper rows are copied, not re-run, and each published model uses its own decision rule. This is development data on PRMBench, not a leaderboard submission.</p></div></div>
{T3}
</section>
'''

CSS = ('<style>.mt{overflow-x:auto;margin:16px 0 6px}.mt table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;min-width:100%;background:white}'
       '.mt caption{text-align:left;font-weight:600;padding:0 0 6px}.mt th,.mt td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:right;white-space:nowrap}'
       '.mt tr th:first-child{text-align:left}.mt td.b{font-weight:700;color:var(--blue)}.mt .mm{border-left:2px solid var(--line);font-weight:600}'
       '.mt .lr{color:var(--muted);font-style:italic;border-left:2px solid var(--line)}.mt tr.hl th,.mt tr.hl td{background:var(--tint)}'
       '.mt tr.mr th,.mt tr.mr td{border-top:2px solid var(--ink);font-weight:600}.mt tr.pub th,.mt tr.pub td{color:var(--muted)}'
       '.mt tr.sep td{text-align:left;font-size:12px;color:var(--muted);padding-top:12px}.mt table.lb th[colspan]{text-align:center}</style>')

s = IN.read_text(encoding='utf8')
assert 'id="leaderboard"' not in s, 'already patched'
A = '<details><summary>Full bank × scorer matrix'
assert s.count(A) == 1
s = s.replace(A, '<h3 style="margin-top:28px">All banks × all methods</h3>' + T1 + T2 + A)
i = s.index('<section class="section" id="types"'); j = s.index('</section>', i) + len('</section>')
s = s[:j] + '\n' + SEC + s[j:]
s = s.replace('</head>', CSS + '</head>', 1)
s = s.replace('<a href="#types">Error types</a>', '<a href="#types">Error types</a><a href="#leaderboard">Leaderboard</a>', 1)
OUT.write_text(s, encoding='utf8')
print('ok', len(s), 'nav' if 'href="#leaderboard"' in s else 'NO NAV LINK')
