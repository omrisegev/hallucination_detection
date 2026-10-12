"""Build advisor page v13 from published v12, following FLOW_OUTLINE_v2_2026-10-15.md (approved by Omri 2026-10-12).

Usage: python3 build_v14.py <published_v12.html> <v11_package_dir> <project_files_dir> <out.html> [external_lr_prm.csv]
v14 (2026-10-12, Omri's notes): section 3 names Qwen3-8B and adds Qwen3-4B rows; section 5 uses the same methods as sections 0-4
(plain average, DS-filter, L-SML, IU-PCR at N16, plus LR and PRM-7B when the optional CSV exists).

Freeze rule: no new experiments. The six new figures are new plots of saved numbers:
  0 heatmap      controls/ALL_BANK_TOP1_AUC.csv + report/legacy/PRMBENCH_COMPARISON.csv (LR)
  1 dumbbell     report/legacy/PRMBENCH_COMPARISON.csv (frozen = 8 Oct input, thinkfix_mask1 = corrected)
  3 leaderboard  report/paper_table3.csv + report/legacy/OFFICIAL_PRMSCORE_N16.csv (R0) + EXACT_DECISION_RULES.csv (PRM)
  5 external     <project_files>/qwen3_4b_vs_8b__external_lsml_locked.csv + __external_panel.csv (STEP_INDEX)
  8a ladder      controls/DIAGNOSTIC_CEILINGS.json, EXACT_DECISION_RULES.csv, BUDGET_CURVES.csv,
                 results/count_structure_check_v1/README.md @ 17aa1e4 (count from error type, 68.9)
  8b rules       EXACT_DECISION_RULES.csv, statistics/MOM_QUOTA.csv,
                 results/label_free_error_count_v1_2026-10-12/FINDINGS_HE.md @ 6a0b870 (Codex arms, OOF)
  8c counts      controls/PER_ANSWER_COUNTS.csv (mean per length bin; aggregation only)
All other figures are moved unchanged from v12. The v12 sections go to the collapsed appendix.
"""
import html, json, re, sys
from pathlib import Path
import pandas as pd

V12, PKG, PF, OUT = sys.argv[1:5]
PKG = Path(PKG); PF = Path(PF)
s = open(V12, encoding='utf-8').read()

GH = 'https://github.com/omrisegev/hallucination_detection/blob/'
PK = GH + '07cb7a45fb01ab62ade51fa5e59717c1f27fd8ff/results/advisor_completion_2026-10-11/'
COUNT = GH + '17aa1e473cf189ad9b2d76eb277e64d3ba3bd447/results/count_structure_check_v1/README.md'
CODEX = GH + '6a0b870/results/label_free_error_count_v1_2026-10-12/FINDINGS_HE.md'
SELFGEN = GH + '080877e/results/thinkfix_rescore_v1/selfgen/B16_SELFGEN_VS_PB_TF.csv'
PAPER = 'https://arxiv.org/abs/2501.03124'
OCT8 = GH + '8996379/docs/meetings/advisor_meeting_2026-10-08/findings_page/findings.html'

FONT = 'system-ui,-apple-system,Segoe UI,sans-serif'
INK, MUTED, GRID = '#223548', '#60778d', '#e3eaf1'
COL = {'LSML_CONT': '#1968b4', 'IU_PCR': '#168f80', 'PRM': '#d37438', 'ALL_EQ': '#8a9aab', 'LR': '#6a4c9c',
       'other': '#5f7d99', 'judge': '#7d6aa8', 'open': '#a9b7c6', 'idx': '#9a6b4f', 'equal': '#7a8b9c', 'ct7': '#b07d2b'}
NAME = {'ALL_EQ': 'Plain average', 'DS_FILTER_EQ': 'DS-filter average', 'LSML_CONT': 'Continuous L-SML', 'IU_PCR': 'IU-PCR',
        'ALL_EQ_ORIENTED': 'ALL_EQ_ORIENTED', 'HEM_EQ': 'HEM_EQ', 'HEM_MOM_cc': 'HEM_MOM_cc', 'HEM_MOM_ga': 'HEM_MOM_ga',
        'MEAN_EQ': 'MEAN_EQ', 'MEAN_MOM_cc': 'MEAN_MOM_cc', 'MEAN_MOM_ga': 'MEAN_MOM_ga', 'TCI_ONLY_ga': 'TCI_ONLY_ga'}
METHODS = list(NAME)
BANKS = ['N11', 'N13', 'N14', 'N16', 'N22', 'N25', 'N40', 'N43', 'N53', 'N56']


def esc(x):
    return html.escape(str(x), quote=True)


def a(href, text):
    return f'<a href="{href}" target="_blank" rel="noopener">{esc(text)}</a>'


def T(x, y, txt, size=12, anchor='start', w=400, fill=INK, extra=''):
    txt = re.sub(r'(?<![A-Za-z0-9])-(?=\d)', '−', str(txt))
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{w}" '
            f'fill="{fill}" font-family="{FONT}"{(" " + extra) if extra else ""}>{esc(txt)}</text>')


def svg(w, h, body, label, maxw=None):
    st = f' style="max-width:{maxw}px;margin:auto"' if maxw else ''
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}"{st}>'
            f'<title>{esc(label)}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}</svg>')


def mix(c1, c2, t):
    a1 = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]; a2 = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(x + (y - x) * t):02x}' for x, y in zip(a1, a2))


# ---------------------------------------------------------------- data
AB = pd.read_csv(PKG / 'controls/ALL_BANK_TOP1_AUC.csv')
AB = AB[(AB.classification == 'ALL_NONCONTROL') & (AB.control == 'unrestricted') & (AB.variant == 'thinkfix_mask1')]
AUC = AB.pivot(index='bank', columns='method', values='within_auc')
assert (AUC['IU_PCR'] - AUC['IU_PCR_LEVELSIGN']).abs().max() == 0 and (AUC['LSML_CONT'] - AUC['LSML_CONT_LEVELSIGN']).abs().max() == 0
CMP = pd.read_csv(PKG / 'report/legacy/PRMBENCH_COMPARISON.csv')
LRG = CMP[(CMP.table == 'lr_gaps') & (CMP.metric == 'lr_within_auc')].set_index('bank')
LR_NEW, LR_OLD = LRG.thinkfix_mask1, LRG.frozen
OLDNEW = CMP[CMP.table.isin(['panel_a', 'cont_baselines']) & (CMP.metric == 'prmb_within_auc')].drop_duplicates(['bank', 'scorer'])
OLDNEW = OLDNEW.set_index(['bank', 'scorer'])
for m in METHODS:  # corrected values in the comparison file equal the package matrix
    assert abs(OLDNEW.loc[('N16', m), 'thinkfix_mask1'] - AUC.loc['N16', m]) < 1e-9, m
RULES = pd.read_csv(PKG / 'controls/EXACT_DECISION_RULES.csv')
RA = RULES[RULES.classification == 'ALL_NONCONTROL'].set_index(['method', 'rule'])
MS = RULES[RULES.classification == 'multi_solutions'].set_index(['method', 'rule']).category_score
CEIL = json.load(open(PKG / 'controls/DIAGNOSTIC_CEILINGS.json'))
BUD = pd.read_csv(PKG / 'controls/BUDGET_CURVES.csv')
PRM_BEST = BUD[BUD.method == 'PRM_NATIVE'].sort_values('prm_score').iloc[-1]
MOM = pd.read_csv(PKG / 'statistics/MOM_QUOTA.csv')
MOM_IU = MOM[(MOM.method == 'IU_PCR') & (MOM.type == 'ALL_NONCONTROL')].prm_score.iloc[0]
OFF = pd.read_csv(PKG / 'report/legacy/OFFICIAL_PRMSCORE_N16.csv')
PAPER3 = pd.read_csv(PKG / 'report/paper_table3.csv')
EXT = pd.read_csv(PF / 'qwen3_4b_vs_8b__external_lsml_locked.csv').set_index(['cell', 'arm'])
EXP = pd.read_csv(PF / 'qwen3_4b_vs_8b__external_panel.csv')
IDX = EXP[(EXP.scorer == 'STEP_INDEX') & (EXP['rule'] == 'ANSWER_Z_Q80')].set_index('cell')
PAC = pd.read_csv(PKG / 'controls/PER_ANSWER_COUNTS.csv')
PAC = PAC[PAC.classification != 'correct']

q80 = {m: RA.loc[(m, 'ANSWER_Z_Q80'), 'prm_score'] * 100 for m in ('LSML_CONT', 'IU_PCR')}
assert abs(q80['LSML_CONT'] - 65.381) < 0.001 and abs(q80['IU_PCR'] - 65.359) < 0.001
PRM_NAT = RA.loc[('PRM_NATIVE', 'NATIVE_0_5'), 'prm_score'] * 100


# ---------------------------------------------------------------- figure 0: heatmap
def fig_heatmap():
    cols = METHODS + ['MEAN', 'LR']
    vals = {b: {m: AUC.loc[b, m] for m in METHODS} for b in BANKS}
    for b in BANKS:
        vals[b]['MEAN'] = sum(vals[b][m] for m in METHODS) / len(METHODS)
        vals[b]['LR'] = LR_NEW[b]
    mrow = {c: sum(vals[b][c] for b in BANKS) / len(BANKS) for c in cols}
    lo, hi = 0.725, 0.825
    x0, cw, ch, top = 92, 62, 30, 128
    def cx(j):
        return x0 + j * cw + (12 if j >= len(METHODS) else 0) + (12 if j == len(METHODS) + 1 else 0)
    W = cx(len(cols) - 1) + cw + 80
    H = top + (len(BANKS) + 1) * ch + 10 + 64
    out = []
    labels = [NAME[m] for m in METHODS] + ['Mean of the 12', 'LR · supervised']
    for j, lab in enumerate(labels):
        x = cx(j) + cw / 2 + 4
        out.append(T(x, top - 8, lab, 11.5, 'start', 650 if j in (2, 3, 13) else 400,
                     extra=f'transform="rotate(-42 {x:.1f} {top - 8:.1f})"'))
    rows = BANKS + ['Method mean']
    for i, b in enumerate(rows):
        y = top + i * ch + (10 if i == len(BANKS) else 0)
        out.append(T(x0 - 10, y + ch / 2 + 4, b, 12, 'end', 700 if b in ('N16', 'Method mean') else 400))
        for j, c in enumerate(cols):
            v = mrow[c] if i == len(BANKS) else vals[b][c]
            t = min(1, max(0, (v - lo) / (hi - lo)))
            fill = mix('#f3f7fb', '#164f86', t)
            out.append(f'<rect x="{cx(j)}" y="{y}" width="{cw - 2}" height="{ch - 2}" rx="3" fill="{fill}"/>')
            out.append(T(cx(j) + (cw - 2) / 2, y + ch / 2 + 3, f'{v:.3f}', 11, 'middle', 500,
                         '#ffffff' if t > 0.52 else INK))
    yn = top + BANKS.index('N16') * ch
    out.append(f'<rect x="{x0 - 3}" y="{yn - 2}" width="{cx(len(cols) - 1) + cw - x0 + 3}" height="{ch + 2}" '
               f'rx="4" fill="none" stroke="{INK}" stroke-width="1.6"/>')
    ly = H - 34
    for k in range(11):
        t = k / 10
        out.append(f'<rect x="{x0 + k * 24}" y="{ly}" width="24" height="10" fill="{mix("#f3f7fb", "#164f86", t)}"/>')
    out.append(T(x0, ly + 26, f'{lo:.3f}', 11, 'start', fill=MUTED))
    out.append(T(x0 + 264, ly + 26, f'{hi:.3f}', 11, 'end', fill=MUTED))
    out.append(T(x0 + 290, ly + 9, 'Within-answer AUC, one colour scale for all cells. Frame: N16, the reference bank.', 11.5, fill=MUTED))
    out.append(T(x0 + 290, ly + 26, 'Banks are nested feature sets, not independent replications.', 11.5, fill=MUTED))
    return svg(W, H, ''.join(out), 'Within-answer AUC for 12 label-free methods and LR on ten banks')


# ---------------------------------------------------------------- figure 1: dumbbell
def fig_dumbbell():
    rows = [(NAME[m], OLDNEW.loc[('N16', m), 'frozen'], OLDNEW.loc[('N16', m), 'thinkfix_mask1'], m) for m in METHODS]
    rows.sort(key=lambda r: -r[2])
    rows = [('LR · supervised', LR_OLD['N16'], LR_NEW['N16'], 'LR')] + rows
    lo, hi, X0, X1, top, rh = 0.765, 0.830, 250, 820, 58, 27
    sx = lambda v: X0 + (v - lo) / (hi - lo) * (X1 - X0)
    H = top + len(rows) * rh + 58
    out = []
    for k in range(14):
        v = lo + k * 0.005
        if round(v * 1000) % 10:
            continue
        out.append(f'<line x1="{sx(v):.1f}" y1="{top - 10}" x2="{sx(v):.1f}" y2="{top + len(rows) * rh}" stroke="{GRID}"/>')
        out.append(T(sx(v), top + len(rows) * rh + 18, f'{v:.2f}', 11, 'middle', fill=MUTED))
    out.append(f'<line x1="{sx(0.80118):.1f}" y1="{top - 22}" x2="{sx(0.80118):.1f}" y2="{top + len(rows) * rh}" '
               f'stroke="{COL["PRM"]}" stroke-width="1.6" stroke-dasharray="5 4"/>')
    out.append(T(sx(0.80118) - 6, top - 16, 'Qwen2.5-Math-PRM-7B 0.801 (does not use our input)', 11.5, 'end', fill=COL['PRM']))
    for i, (lab, o, n, m) in enumerate(rows):
        y = top + i * rh + rh / 2
        c = COL.get(m, COL['other'])
        bold = 700 if m in ('LR', 'LSML_CONT', 'IU_PCR', 'ALL_EQ') else 400
        out.append(T(X0 - 12, y + 4, lab, 12, 'end', bold))
        out.append(f'<line x1="{sx(n):.1f}" y1="{y}" x2="{sx(o):.1f}" y2="{y}" stroke="{c}" stroke-width="2" stroke-opacity=".45"/>')
        out.append(f'<circle cx="{sx(o):.1f}" cy="{y}" r="5" fill="#ffffff" stroke="{c}" stroke-width="1.8"/>')
        out.append(f'<circle cx="{sx(n):.1f}" cy="{y}" r="5.5" fill="{c}"/>')
        out.append(T(X1 + 14, y + 4, f'{o:.3f} → {n:.3f}  ({(n - o) * 100:+.1f})', 11.5, fill=INK))
        if i == 0:
            out.append(f'<line x1="{X0 - 200}" y1="{y + rh / 2}" x2="{X1 + 160}" y2="{y + rh / 2}" stroke="{GRID}"/>')
    ly = H - 16
    out.append(f'<circle cx="{X0}" cy="{ly - 4}" r="5" fill="#ffffff" stroke="{INK}" stroke-width="1.6"/>')
    out.append(T(X0 + 12, ly, 'Before the fix (8 Oct input)', 11.5))
    out.append(f'<circle cx="{X0 + 210}" cy="{ly - 4}" r="5.5" fill="{INK}"/>')
    out.append(T(X0 + 222, ly, 'Corrected input (think block + mask1)', 11.5))
    out.append(T(X1 + 14, top - 16, 'old → new  (Δ AUC points)', 11, fill=MUTED))
    return svg(1000, H, ''.join(out), 'N16 within-answer AUC before and after the think-block fix, per method')


# ---------------------------------------------------------------- figure 3: leaderboard bars
JUDGES = {'GPT-4o', 'o1-mini', 'Gemini-2.0-flash-exp', 'Gemini-2.0-thinking-exp-1219'}


R4 = pd.read_csv(PF / 'qwen3_4b_vs_8b__N16_R0_LSML_IUPCR_by_subtype.csv').query("model == '4b'").set_index('method').overall * 100


def lb_rows():
    r0 = OFF[OFF['rule'] == 'R0'].set_index('method').overall * 100
    rows = [('Continuous L-SML · Qwen3-8B · ours', r0['LSML_CONT'], 'LSML_CONT'),
            ('IU-PCR · Qwen3-8B · ours', r0['IU_PCR'], 'IU_PCR'),
            ('Continuous L-SML · Qwen3-4B · ours', R4['LSML_CONT'], 'LSML_4B'),
            ('IU-PCR · Qwen3-4B · ours', R4['IU_PCR'], 'IU_4B'),
            ('Qwen2.5-Math-PRM-7B · our run, 0.5 threshold', PRM_NAT, 'PRM')]
    for _, p in PAPER3.iterrows():
        lab = p.model + (' · paper, N=394' if p.model == 'o1-mini' else ' · paper')
        rows.append((lab, p.overall, 'judge' if p.model in JUDGES else 'open'))
    return sorted(rows, key=lambda r: -r[1])


def fig_leaderboard():
    rows = lb_rows()
    X0, X1, top, rh = 330, 900, 46, 25
    sx = lambda v: X0 + v / 75 * (X1 - X0)
    H = top + len(rows) * rh + 34
    out = []
    for v in range(0, 76, 10):
        out.append(f'<line x1="{sx(v):.1f}" y1="{top - 6}" x2="{sx(v):.1f}" y2="{top + len(rows) * rh}" stroke="{GRID}"/>')
        out.append(T(sx(v), top + len(rows) * rh + 16, str(v), 11, 'middle', fill=MUTED))
    for i, (lab, v, k) in enumerate(rows):
        y = top + i * rh
        c = {'LSML_4B': mix(COL['LSML_CONT'], '#ffffff', 0.45), 'IU_4B': mix(COL['IU_PCR'], '#ffffff', 0.45)}.get(k, COL.get(k, COL['other']))
        ours = k in ('LSML_CONT', 'IU_PCR', 'PRM', 'LSML_4B', 'IU_4B')
        out.append(T(X0 - 10, y + 16, lab, 12, 'end', 700 if ours else 400))
        out.append(f'<rect x="{X0}" y="{y + 5}" width="{sx(v) - X0:.1f}" height="15" rx="2" fill="{c}"/>')
        out.append(T(sx(v) + 6, y + 17, f'{v:.1f}', 11.5, 'start', 700 if ours else 400))
    lx = 40
    for lab, c in (('Ours, Qwen3-8B / light: Qwen3-4B', COL['LSML_CONT']), ('Qwen2.5-Math-PRM-7B, our run', COL['PRM']),
                   ('Open PRMs, paper Table 3', COL['open']), ('LLM judges, paper Table 3', COL['judge'])):
        out.append(f'<rect x="{lx}" y="12" width="12" height="12" rx="2" fill="{c}"/>')
        out.append(T(lx + 18, 22, lab, 11.5))
        lx += 18 + len(lab) * 6.4 + 22
    return svg(1000, H, ''.join(out), 'Overall PRMScore: our methods, Qwen2.5-Math-PRM-7B and paper Table 3 rows')


# ---------------------------------------------------------------- figure 5: external
EXT_CONT = pd.read_csv(PF / 'qwen3_4b_vs_8b__external_cont.csv')
EXT_NEST = pd.read_csv(PF / 'qwen3_4b_vs_8b__external_nested.csv')
EXTN16 = pd.concat([EXT_CONT, EXT_NEST])
EXTN16 = EXTN16[(EXTN16.bank == 'N16') & (EXTN16['rule'] == 'ANSWER_Z_Q80')].drop_duplicates(['cell', 'scorer']).set_index(['cell', 'scorer'])
PUB = json.load(open(PF / 'advisor_page' / 'PUBLISHED_CONTEXT_external.json'))
PUB_PRM = {'socratic_qwen3_8b': PUB['socratic']['rows']['Qwen2.5-Math-PRM-7B'] / 100,
           'hard2verify_qwen3_8b': PUB['hard2verify']['rows']['Qwen2.5-Math-PRM-7B'] / 100}
# Optional same-protocol LR and PRM-7B rows (run requested 2026-10-12); columns benchmark, model, method, rule, official_metric
EXTRA = None
if len(sys.argv) > 5 and Path(sys.argv[5]).exists():
    EXTRA = pd.read_csv(sys.argv[5])
EXT_METHODS = [('ALL_EQ', 'Plain avg.'), ('DS_FILTER_EQ', 'DS-filter'), ('LSML_CONT', 'L-SML'), ('IU_PCR', 'IU-PCR'),
               ('LR', 'LR · sup.'), ('PRM', 'PRM-7B')]


def ext_val(cell, m, model, col='official_metric'):
    # for PRM, model is the rule: 'nat' (native 0.5) or 'q80' (ANSWER_Z_Q80)
    if m in ('LR', 'PRM'):
        if EXTRA is None:
            return None
        bench = 'socratic' if cell.startswith('socratic') else 'hard2verify'
        rule = ('NATIVE_0_5' if model == 'nat' else 'ANSWER_Z_Q80') if m == 'PRM' else 'ANSWER_Z_Q80'
        r = EXTRA[(EXTRA.benchmark == bench) & (EXTRA.method == m) & (EXTRA['rule'] == rule)
                  & (EXTRA.model == ('-' if m == 'PRM' else model))]
        return None if r.empty else float(r[col].iloc[0])
    return EXTN16.loc[(cell, m), f'{col}_{model}']


def fig_external():
    out = []
    panels = [('socratic_qwen3_8b', 'Socratic-PRMBench · PRMScore · 2,995 answers'),
              ('hard2verify_qwen3_8b', 'Hard2Verify · Balanced F1 · 200 answers')]
    cols = {'ALL_EQ': COL['ALL_EQ'], 'DS_FILTER_EQ': COL['other'], 'LSML_CONT': COL['LSML_CONT'], 'IU_PCR': COL['IU_PCR'],
            'LR': COL['LR'], 'PRM': COL['PRM']}
    top, base, ymax = 56, 330, 0.7
    sy = lambda v: base - v / ymax * (base - top)
    PH = 420
    for p, (cell, title) in enumerate(panels):
        oy = p * PH
        sy2 = lambda v: oy + sy(v)
        out.append(T(40, oy + 26, title, 13.5, 'start', 700))
        for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            out.append(f'<line x1="70" y1="{sy2(v):.1f}" x2="760" y2="{sy2(v):.1f}" stroke="{GRID}"/>')
            out.append(T(62, sy2(v) + 4, f'{v:.1f}', 10.5, 'end', fill=MUTED))
        for j, (m, lab) in enumerate(EXT_METHODS):
            gx = 92 + j * 112
            models = ['nat', 'q80'] if m == 'PRM' else ['8b', '4b']
            for k, model in enumerate(models):
                v = ext_val(cell, m, model)
                bx = gx + k * 40
                c = mix(cols[m], '#ffffff', 0.45 if model in ('4b', 'q80') else 0)
                if v is None:
                    out.append(f'<rect x="{bx}" y="{sy2(0.7):.1f}" width="36" height="{base - sy(0.7):.1f}" fill="#f7f9fb" '
                               f'stroke="#b8c4d0" stroke-dasharray="4 3"/>')
                    out.append(T(bx + 18, sy2(0.35), 'running', 10.5, 'middle', fill=MUTED, extra=f'transform="rotate(-90 {bx + 18} {sy2(0.35):.1f})"'))
                else:
                    out.append(f'<rect x="{bx}" y="{sy2(v):.1f}" width="36" height="{oy + base - sy2(v):.1f}" fill="{c}"/>')
                    out.append(T(bx + 18, sy2(v) - 6, f'{v:.3f}', 10.5, 'middle', 600 if model in ('8b', 'nat') else 400))
            out.append(T(gx + 38, oy + base + 18, lab, 11.5, 'middle', 700 if m in ('LSML_CONT', 'IU_PCR') else 400))
            out.append(T(gx + 38, oy + base + 32, '0.5 | Q80' if m == 'PRM' else '8B | 4B', 10, 'middle', fill=MUTED))
        for v, lab, c, dash in ((IDX.loc[cell, 'official_metric_8b'], 'Step index alone', COL['idx'], '6 4'),
                                (PUB_PRM[cell], 'PRM-7B, published (other protocol)', COL['PRM'], '2 3')):
            out.append(f'<line x1="70" y1="{sy2(v):.1f}" x2="760" y2="{sy2(v):.1f}" stroke="{c}" stroke-width="1.6" stroke-dasharray="{dash}"/>')
        refs = sorted(((IDX.loc[cell, 'official_metric_8b'], 'Step index alone', COL['idx']),
                       (PUB_PRM[cell], 'PRM-7B, published*', COL['PRM'])), key=lambda r: -r[0])
        ys = [sy2(refs[0][0]) + 4, max(sy2(refs[1][0]) + 4, sy2(refs[0][0]) + 20)]
        for (v, lab, c), yy in zip(refs, ys):
            out.append(T(768, yy, f'{lab} {v:.3f}', 11.5, 'start', 600, c))
    H = 2 * PH + 10
    out.append(T(40, H - 6, 'Ours and LR: dark Qwen3-8B, light Qwen3-4B, N16, Q80 · PRM-7B: dark native 0.5, light Q80 · *published: authors\' thresholds', 11.5, fill=MUTED))
    return svg(1000, H + 4, ''.join(out), 'External benchmarks: our methods, LR and PRM-7B on Socratic and Hard2Verify')


# ---------------------------------------------------------------- figure 8a: ladder
def ladder_rows():
    L = CEIL['LSML_CONT']
    return [('Q80 · our frozen rule', q80['LSML_CONT'], 'free'),
            ('Best global cutoff', L['global_threshold']['prm_score'] * 100, 'oracle'),
            ('Constant 2 flags per answer (TOP_2)', RA.loc[('LSML_CONT', 'TOP_2_DIAGNOSTIC'), 'prm_score'] * 100, 'posthoc'),
            ('Count from error type only', 68.9, 'oracle'),
            ('True count per answer', L['true_count']['prm_score'] * 100, 'oracle'),
            ('Perfect ranking at our own counts', L['perfect_ranking_fixed_counts']['prm_score'] * 100, 'oracle')]


def fig_ladder():
    rows = ladder_rows()
    lo, hi, X0, X1, top, rh = 60, 85, 300, 850, 64, 34
    sx = lambda v: X0 + (v - lo) / (hi - lo) * (X1 - X0)
    H = top + len(rows) * rh + 60
    cmap = {'free': COL['LSML_CONT'], 'posthoc': COL['ct7'], 'oracle': '#8e9bab'}
    out = []
    for v in range(60, 86, 5):
        out.append(f'<line x1="{sx(v):.1f}" y1="{top - 8}" x2="{sx(v):.1f}" y2="{top + len(rows) * rh}" stroke="{GRID}"/>')
        out.append(T(sx(v), top + len(rows) * rh + 17, str(v), 11, 'middle', fill=MUTED))
    for v, lab, yy in ((PRM_NAT, f'PRM-7B, native 0.5: {PRM_NAT:.1f}', 22), (PRM_BEST.prm_score * 100, f'PRM-7B, best global cut: {PRM_BEST.prm_score * 100:.1f}', 42)):
        out.append(f'<line x1="{sx(v):.1f}" y1="{yy + 4}" x2="{sx(v):.1f}" y2="{top + len(rows) * rh}" stroke="{COL["PRM"]}" stroke-width="1.5" stroke-dasharray="5 4"/>')
        out.append(T(sx(v) + 5, yy, lab, 11.5, fill=COL['PRM']))
    b = rows[0][1]
    for i, (lab, v, k) in enumerate(rows):
        y = top + i * rh + rh / 2
        c = cmap[k]
        out.append(T(X0 - 12, y + 4, lab, 12.5, 'end', 700 if i == 0 else 400))
        if i:
            out.append(f'<line x1="{sx(b):.1f}" y1="{y}" x2="{sx(v):.1f}" y2="{y}" stroke="{c}" stroke-width="7" stroke-linecap="round" stroke-opacity=".35"/>')
        out.append(f'<circle cx="{sx(v):.1f}" cy="{y}" r="7" fill="{c}"/>')
        out.append(T(sx(v) + 12, y + 4, f'{v:.1f}' + (f'  ({v - b:+.1f})' if i else ''), 12, 'start', 600))
    ly = H - 18
    lx = X0
    for lab, c in (('Label-free rule', cmap['free']), ('Post hoc choice (fits how PRMBench inserts errors)', cmap['posthoc']),
                   ('Uses labels: a ceiling, not a method', cmap['oracle'])):
        out.append(f'<circle cx="{lx + 6}" cy="{ly - 4}" r="6" fill="{c}"/>')
        out.append(T(lx + 16, ly, lab, 11.5))
        lx += 16 + len(lab) * 6.2 + 24
    return svg(1000, H, ''.join(out), 'PRMScore when only the number of flags per answer changes, with the same ranking')


# ---------------------------------------------------------------- figure 8b: what we tried
def tried_rows():
    b = q80['IU_PCR']
    codex_q80 = 65.386
    return [('R2_COUNT · registered rule', RA.loc[('IU_PCR', 'R2_COUNT'), 'prm_score'] * 100 - b, 'free'),
            ('Entropy gate (ENTROPY_GATE_Q80) · registered', RA.loc[('IU_PCR', 'ENTROPY_GATE_Q80'), 'prm_score'] * 100 - b, 'free'),
            ('MoM group posterior as the count', MOM_IU * 100 - b, 'free'),
            ('Codex LOCAL_MOM_MEAN (best arm) · OOF', -0.4815, 'free'),
            ('Codex B_TRIPLE_MEAN · OOF', 64.450 - codex_q80, 'free'),
            ('Codex D_ZERO_MEAN (allows clean answers) · OOF', 63.760 - codex_q80, 'free'),
            ('Constant 2 flags (TOP_2) · post hoc reference', RA.loc[('IU_PCR', 'TOP_2_DIAGNOSTIC'), 'prm_score'] * 100 - b, 'posthoc')]


def fig_tried():
    rows = tried_rows()
    lo, hi, X0, X1, top, rh = -2, 3, 380, 900, 30, 32
    sx = lambda v: X0 + (v - lo) / (hi - lo) * (X1 - X0)
    H = top + len(rows) * rh + 44
    out = []
    for v in range(-2, 4):
        out.append(f'<line x1="{sx(v):.1f}" y1="{top - 6}" x2="{sx(v):.1f}" y2="{top + len(rows) * rh}" stroke="{INK if v == 0 else GRID}" stroke-width="{1.4 if v == 0 else 1}"/>')
        out.append(T(sx(v), top + len(rows) * rh + 17, f'{v:+d}' if v else '0 = Q80', 11, 'middle', fill=MUTED))
    for i, (lab, d, k) in enumerate(rows):
        y = top + i * rh
        c = COL['ct7'] if k == 'posthoc' else ('#b9583c' if d < 0 else COL['LSML_CONT'])
        if k == 'posthoc':
            out.append(f'<line x1="{X0 - 340}" y1="{y - 3}" x2="{X1}" y2="{y - 3}" stroke="{GRID}" stroke-dasharray="3 3"/>')
        out.append(T(X0 - 12, y + rh / 2 + 4, lab, 12, 'end'))
        x1, x2 = sorted((sx(0), sx(d)))
        out.append(f'<rect x="{x1:.1f}" y="{y + 7}" width="{max(x2 - x1, 1.5):.1f}" height="{rh - 14}" rx="2" fill="{c}"/>')
        tx = sx(d) + (6 if d >= 0 else -6)
        out.append(T(tx, y + rh / 2 + 4, f'{d:+.2f}', 11.5, 'start' if d >= 0 else 'end', 600))
    out.append(T(X0 - 12, H - 6, 'Δ PRMScore points vs Q80 (IU-PCR ranking, N16)', 11.5, 'end', fill=MUTED))
    return svg(1000, H, ''.join(out), 'PRMScore change against Q80 for each count rule that we tested')


# ---------------------------------------------------------------- figure 8c: counts vs length
BINS = [(1, 7, '≤7'), (8, 9, '8–9'), (10, 11, '10–11'), (12, 13, '12–13'), (14, 15, '14–15'), (16, 18, '16–18'), (19, 10 ** 6, '≥19')]


def count_table():
    rows = {}
    def agg(g, col):
        return [g[(g.n_steps >= lo) & (g.n_steps <= hi)][col].mean() for lo, hi, _ in BINS]
    ns = None
    for key, (m, r) in {'truth': ('IU_PCR', 'ANSWER_Z_Q80'), 'Q80': ('IU_PCR', 'ANSWER_Z_Q80'), 'R2_COUNT': ('IU_PCR', 'R2_COUNT'),
                        'ENTROPY': ('IU_PCR', 'ENTROPY_GATE_Q80'), 'PRM': ('PRM_NATIVE', 'NATIVE_0_5')}.items():
        g = PAC[(PAC.method == m) & (PAC['rule'] == r)]
        assert len(g) == 6211
        rows[key] = agg(g, 'true_count' if key == 'truth' else 'flagged_count')
        rho = g.n_steps.rank().corr((g.true_count if key == 'truth' else g.flagged_count).rank())
        rows[key + '_rho'] = rho
        ns = [int(((g.n_steps >= lo) & (g.n_steps <= hi)).sum()) for lo, hi, _ in BINS]
        rows['truth_mean'] = g.true_count.mean()
    rows['n'] = ns
    return rows


def fig_counts(ct):
    X0, X1, top, base, ymax = 80, 790, 30, 330, 6.5
    sx = lambda i: X0 + i * (X1 - X0) / (len(BINS) - 1)
    sy = lambda v: base - v / ymax * (base - top)
    out = []
    for v in range(0, 7):
        out.append(f'<line x1="{X0 - 10}" y1="{sy(v):.1f}" x2="{X1 + 10}" y2="{sy(v):.1f}" stroke="{GRID}"/>')
        out.append(T(X0 - 16, sy(v) + 4, str(v), 11, 'end', fill=MUTED))
    for i, (_, _, lab) in enumerate(BINS):
        out.append(T(sx(i), base + 18, lab, 11.5, 'middle'))
        out.append(T(sx(i), base + 33, f'n={ct["n"][i]:,}', 10, 'middle', fill=MUTED))
    out.append(T(X0 - 50, top - 10, 'Errors or flags per answer (mean)', 11.5, fill=MUTED))
    out.append(T((X0 + X1) / 2, base + 52, 'Steps in the answer', 11.5, 'middle', fill=MUTED))
    series = [('truth', 'True errors', '#111d29', 3.2, ''), ('Q80', 'Q80 flags', COL['LSML_CONT'], 2.4, ''),
              ('R2_COUNT', 'R2_COUNT flags', COL['LR'], 2, '6 4'), ('ENTROPY', 'Entropy-gate flags', COL['IU_PCR'], 2, '3 3'),
              ('PRM', 'PRM-7B flags (0.5)', COL['PRM'], 2, '')]
    for key, lab, c, w, dash in series:
        pts = ' '.join(f'{sx(i):.1f},{sy(v):.1f}' for i, v in enumerate(ct[key]))
        da = f' stroke-dasharray="{dash}"' if dash else ''
        out.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="{w}"{da}/>')
        for i, v in enumerate(ct[key]):
            out.append(f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="3.2" fill="{c}"/>')
        v = ct[key][-1]
        out.append(T(X1 + 16, sy(v) + 4, f'{lab} {v:.1f} · ρ={ct[key + "_rho"]:.2f}', 11.5, 'start', 700 if key == 'truth' else 400, c))
    return svg(1000, 400, ''.join(out), 'Mean true errors and mean flags per answer, by answer length')


# ---------------------------------------------------------------- section 2 schematic (cards reused from the v12 SVG)
def fig_schematic():
    i0 = s.find('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1260 635"')
    i1 = s.find('</svg>', i0)
    v = s[i0:i1]
    pane = v[v.find('<rect x="4" y="7"'):v.find('<g transform=')]
    cards = {}
    for m in re.finditer(r'<g transform="translate\((\d+),(\d+)\)">(.*?)</g>', v, flags=re.S):
        name = re.search(r'font-weight="700">([A-Z]+) ·', m.group(3)).group(1)
        cards[name] = m.group(3)
    assert len(cards) == 9, cards.keys()
    groups = [('SIMPLICITY', ['NR', 'NCL'], '#2f7d58'), ('SOUNDNESS', ['ES', 'SC', 'DC', 'CI'], '#9a6b12'),
              ('SENSITIVITY', ['PS', 'DR', 'MS'], '#5a5a9a')]
    out = ['<defs><marker id="arrowhead2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" '
           'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#526675"/></marker></defs>',
           pane.replace('url(#arrowhead)', 'url(#arrowhead2)')]
    for k, (g, names, c) in enumerate(groups):
        x = 250 + k * 248
        out.append(T(x + 4, 30, g, 12.5, 'start', 750, c, 'letter-spacing="1.2"'))
        for j, n in enumerate(names):
            out.append(f'<g transform="translate({x},{44 + j * 114})">{cards[n].replace("url(#arrowhead)", "url(#arrowhead2)")}</g>')
    out.append(T(250, 520, 'Each card is one error type. Green node: valid step; red node: error step; dashed node: missing premise.', 11.5, fill=MUTED))
    out.append(T(250, 540, 'MS answers have no error: several valid paths must stay accepted. It tests false flags.', 11.5, fill=MUTED))
    out.append(T(250, 560, 'Simplicity errors can leave every number and the final answer correct.', 11.5, fill=MUTED))
    return svg(990, 585, ''.join(out), 'How PRMBench is built and its nine error types in three groups')


def radar_only():
    i0 = s.find('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1260 635"')
    i1 = s.find('</svg>', i0) + 6
    v = s[i0:i1].replace('viewBox="0 0 1260 635"', 'viewBox="482 14 520 600" style="max-width:620px;margin:auto"', 1)
    v = re.sub(r'<text x="237" y="616"[^>]*>Concept diagrams are illustrative[^<]*</text>', '', v)
    v = re.sub(r'<rect x="4" y="7".*?(?=<polygon)', '', v, count=1, flags=re.S)  # data pane and cards are outside the crop
    return v


# ---------------------------------------------------------------- v12 pieces
def take_fig(key):
    """Return (figure_html, start, end) for the v12 figure whose caption starts with key; end includes its takeaway box."""
    i = s.find('<figcaption>' + key)
    assert i > 0, key
    f = s.rfind('<figure', 0, i)
    e = s.find('</figure>', i) + 9
    e2 = e
    if s.startswith('<ul class="takeaways">', e):
        e2 = s.find('</ul>', e) + 5
    fig = s[f:e]
    fig = re.sub(r'<!DOCTYPE svg[^>]*>', '', fig)
    fig = re.sub(r'<\?xml[^>]*\?>', '', fig)
    return fig, f, e2


MOVED = {'Matched AUC controls': '4b', 'All 56 channels plus step index': '4c', 'New OOF gap across all ten banks': '6a',
         'Four-way top1 outcomes by error type': '6b', 'Change in within-answer AUC from Qwen3-8B to Qwen3-4B': '6c',
         'Budget curves and oracle diagnostics': '7', 'Actual category scores for three evaluated methods': '4a'}
FIG = {k: take_fig(k)[0] for k in MOVED}


def section(sid):
    i = s.find(f'<section id="{sid}"')
    assert i > 0, sid
    j = s.find('</section>', i) + 10
    return s[i:j]


def appx_part(sid, drop=()):
    h = section(sid)
    lo = s.find(f'<section id="{sid}"'); hi = lo + len(h)
    for k, sec in MOVED.items():
        fig, f, e = take_fig(k)
        if lo <= f and e <= hi:
            seg = s[f:e]
            h = h.replace(seg, f'<p class="small moved">This figure is in the main text, section {sec}.</p>')
    for d in drop:
        h = h.replace(d, '')
    h = h.replace(f'<section id="{sid}"', f'<div class="appx-part" id="{sid}"', 1)
    assert h.endswith('</section>')
    return h[:-10] + '</div>'


def details_by_summary(h, start):
    i = h.find('<details><summary>' + start)
    assert i >= 0, start
    depth, k = 0, i
    while True:
        o = h.find('<details', k + 1); c = h.find('</details>', k + 1)
        if o != -1 and o < c:
            depth += 1; k = o
        else:
            if depth == 0:
                return h[i:c + 10]
            depth -= 1; k = c


# ---------------------------------------------------------------- page text
def exp(txt):
    return f'<div class="exp"><span class="kt">Experiment · what we tested and why</span>{txt}</div>'


def concl(items):
    assert len(items) == 3, items
    return '<ul class="takeaways concl"><span class="kt">Conclusions</span>' + ''.join(f'<li>{t}</li>' for t in items) + '</ul>'


def src(*parts):
    return '<p class="source">Source: ' + ' · '.join(parts) + '</p>'


def fig(svgtxt, cap):
    return f'<figure>{svgtxt}<figcaption>{cap}</figcaption></figure>'


def sec(sid, eyebrow, title, body):
    return f'<section id="{sid}"><div class="eyebrow">{eyebrow}</div><h2>{title}</h2>{body}</section>'


ct = count_table()
lad = ladder_rows()
tried = tried_rows()
lb = lb_rows()
n16 = {m: AUC.loc['N16', m] for m in METHODS}
spread = {b: max(AUC.loc[b, m] for m in METHODS) - min(AUC.loc[b, m] for m in METHODS) for b in BANKS}
assert all(max(AUC.loc[b, m] for m in METHODS if m != 'ALL_EQ') > AUC.loc[b, 'ALL_EQ'] for b in BANKS)
assert all(LR_NEW[b] > max(AUC.loc[b, m] for m in METHODS) for b in BANKS)
assert max(spread, key=spread.get) == 'N53' and round(spread['N53'], 3) == 0.036 and round(spread['N16'], 3) == 0.010
old_rank = sorted(METHODS, key=lambda m: -OLDNEW.loc[('N16', m), 'frozen'])
new_rank = sorted(METHODS, key=lambda m: -OLDNEW.loc[('N16', m), 'thinkfix_mask1'])
assert old_rank[-1] == new_rank[-1] == 'ALL_EQ' and old_rank != new_rank
drops = {m: (OLDNEW.loc[('N16', m), 'thinkfix_mask1'] - OLDNEW.loc[('N16', m), 'frozen']) * 100 for m in METHODS}
assert round(min(drops.values()), 1) == -1.9 and round(max(drops.values()), 1) == -0.9
lr_drop = (LR_NEW['N16'] - LR_OLD['N16']) * 100
assert round(lr_drop, 1) == -2.1
assert min(p.overall for p in PAPER3.itertuples() if p.model in JUDGES) == 66.0
assert max(p.overall for p in PAPER3.itertuples() if p.model not in JUDGES) == 60.5
assert PAPER3.NR.max() < 69.95
assert abs(ct['truth_mean'] - 2.117) < 0.001 and round(ct['truth_rho'], 2) == 0.09 and round(ct['Q80_rho'], 2) == 0.78
assert (round(ct["R2_COUNT_rho"], 2), round(ct["ENTROPY_rho"], 2), round(ct["PRM_rho"], 2)) == (0.44, 0.48, -0.03)
assert all(x > 0 for x in (ct['Q80'][-1] - ct['Q80'][0], ct['R2_COUNT'][-1] - ct['R2_COUNT'][0], ct['ENTROPY'][-1] - ct['ENTROPY'][0]))
assert all(d < 0 for _, d, k in tried if k == 'free')
ms_iu = {r: MS[('IU_PCR', r)] * 100 for r in ('ANSWER_Z_Q80', 'R2_COUNT', 'ENTROPY_GATE_Q80')}
cff = {r: RA.loc[('IU_PCR', r), 'controls_false_flag_rate'] * 100 for r in ('ANSWER_Z_Q80', 'R2_COUNT', 'ENTROPY_GATE_Q80')}
ext = {(c, a_, m): EXT.loc[(c, a_), m] for c in ('socratic_qwen3_8b', 'hard2verify_qwen3_8b')
       for a_ in ('frozen_lsml', 'frozen_equal', 'ct7') for m in ('metric_8b', 'metric_4b')}
idx = {c: IDX.loc[c, 'official_metric_8b'] for c in ('socratic_qwen3_8b', 'hard2verify_qwen3_8b')}
f3 = lambda v: f'{v:.3f}'
f1 = lambda v: f'{v:.1f}'

S0 = sec('s0', '0 · WHERE WE STOPPED ON 8 OCTOBER', 'The fusion methods are close to each other; LR is above all of them',
         exp('We fuse the step signals of one teacher-forced pass of Qwen3-8B with label-free methods. On 8 October we compared '
             'all methods on all feature banks with within-answer AUC. Here we show the same comparison on the corrected input. '
             'The purpose is to give the same starting view as last time.')
         + fig(fig_heatmap(), 'Within-answer AUC on 6,030 answers, corrected input (think block + mask1). 12 distinct label-free methods; '
               'the LEVELSIGN variants are identical to L-SML and IU-PCR in all banks and are not shown. LR is the supervised '
               'cross-fitted reference on the same features. Old 8 October values are in appendix A.')
         + concl([f'In each bank, the label-free methods are close to each other. At N16 all 12 are between {f3(min(n16.values()))} and '
                  f'{f3(max(n16.values()))}. The spread grows in the larger banks, up to {spread["N53"]:.3f} at N53. The choice of fusion method has a small effect.',
                  f'The best fusion method is above the plain average in every bank. At N16: IU-PCR {f3(n16["IU_PCR"])} and L-SML '
                  f'{f3(n16["LSML_CONT"])}, against {f3(n16["ALL_EQ"])} for the plain average.',
                  f'Supervised LR on the same features is above all label-free methods in every bank (N16: {f3(LR_NEW["N16"])}). This gap is the headroom of the features.'])
         + src(a(PK + 'controls/ALL_BANK_TOP1_AUC.csv', 'controls/ALL_BANK_TOP1_AUC.csv'),
               a(PK + 'report/legacy/PRMBENCH_COMPARISON.csv', 'report/legacy/PRMBENCH_COMPARISON.csv') + ' (LR)',
               'Top1 and PRMScore matrices: appendix D'))

S1 = sec('s1', '1 · DID THE CORRECTIONS CHANGE THE PICTURE?', 'All methods went down by a similar amount; the picture stays',
         exp('After 8 October we found that the teacher-forced prompt did not have the empty think block that Qwen3 expects. '
             'We fixed the prompt, ran the forward pass again and scored all methods again. We also masked the first answer token. '
             'The purpose: to make sure that the numbers are valid, and to see if the conclusions change.')
         + fig(fig_dumbbell(), 'N16 within-answer AUC per method, before the fix and on the corrected input (6,030 answers). '
               'Qwen2.5-Math-PRM-7B does not use our teacher-forced input, so it does not change.')
         + concl([f'The fix lowers every method by {abs(max(drops.values())):.1f} to {abs(min(drops.values())):.1f} AUC points (LR: {abs(lr_drop):.1f}). '
                  'LR stays first and the plain average stays last. The fusion methods stay inside a band of about 0.01; their order inside the band changes.',
                  'The decline is not a first-token effect. Without step 0 the decline is larger (−2.2 vs −1.6 points for L-SML). '
                  'The first-token mask changes AUC by less than 0.1 point.',
                  'The same conclusions hold on Qwen3-4B (N16 L-SML 0.775 vs 0.779; section 6c) and on a held-out half of the '
                  'source groups (IU-PCR 0.786; appendix A).'])
         + src(a(PK + 'report/legacy/PRMBENCH_COMPARISON.csv', 'report/legacy/PRMBENCH_COMPARISON.csv'),
               a(PK + 'controls/PREFIX_MASK_ATTRIBUTION.csv', 'controls/PREFIX_MASK_ATTRIBUTION.csv'),
               a(OCT8, '8 October page'))
         + '<div class="box"><span class="kt">Without teacher forcing</span><p>On answers that Qwen3-4B and Qwen3-8B wrote themselves '
           '(GSM8K and MATH, not PRMBench; FINAL890 labels), the N16 signal gives within-answer AUC 0.757. With teacher forcing on human-written answers '
           '(ProcessBench GSM8K/MATH) it gives 0.741. Difference +0.015, 95% CI [−0.021, +0.052]. We found no difference, but we do '
           'not claim equivalence.</p>' + src(a(SELFGEN, 'selfgen/B16_SELFGEN_VS_PB_TF.csv') + ' (FINAL890, thinkfix_mask1, pooled)') + '</div>'
         + '<div class="box warn"><span class="kt">Claims we withdraw</span><ul>'
           '<li>"realized_drv leads 6 of 8 error types". On the corrected input it leads 0 of 8 (section 4c).</li>'
           '<li>"All ~40 literature rows are weaker on Simplicity". In paper Table 3, 10 of 15 rows have Simplicity below both other '
           'category means (appendix E).</li>'
           '<li>"Self-generated and teacher-forced answers are equivalent". See the box above: no difference found, no equivalence shown.</li></ul></div>')

S2 = sec('s2', '2 · BACKGROUND', 'What is a PRM, and what does PRMBench test?',
         '<div class="prose"><p>A <b>process reward model (PRM)</b> gives a score to each step of a solution. It is trained on step labels. '
         'Qwen2.5-Math-PRM-7B gives each step a probability of being correct; a step below 0.5 is an error.</p>'
         '<p><b>Our method does not train.</b> It reads token-level signals from one teacher-forced pass of Qwen3 over the given answer, '
         'and fuses them without labels (L-SML, IU-PCR).</p>'
         '<p><b>PRMBench</b> (Song et al., 2025, ' + a(PAPER, 'arXiv 2501.03124') + '): GPT-4o inserted errors into correct PRM800K solutions, '
         'and the results were filtered. QwQ supplied correct alternative solutions for MS. Each step has a label. Our population is '
         '6,211 answers with errors plus 758 clean controls (the paper has 6,216 answers).</p>'
         '<p>The nine error types are in three groups: <b>Simplicity</b> (NR, NCL), <b>Soundness</b> (ES, SC, DC, CI) and '
         '<b>Sensitivity</b> (PS, DR, MS).</p>'
         '<p>The official metric, <b>PRMScore</b>, is the mean of the F1 on valid steps and the F1 on error steps. It needs a flag rule. '
         'Within-answer AUC and top1-on-error need no rule.</p></div>'
         + exp('No experiment. This figure shows how the benchmark is built and what each error type looks like, so that the '
               'category results in sections 3-8 can be read.')
         + fig(fig_schematic(), 'PRMBench construction and the nine error types. Concept diagrams are illustrative; they reuse the v12 schematics, '
               'inspired by the PRMBench overview.')
         + concl(['NR and NCL steps can be wrong without a wrong number: a removable step, or a step that assumes the answer. '
                  'The final answer can stay correct.',
                  'CI, ES and DR errors are written to look plausible. A model can read them with low surprise.',
                  'PRMScore rewards both finding errors and not flagging valid steps. So it tests the flag rule, not only the ranking.'])
         + '<div class="box"><span class="kt">What the paper says about Simplicity</span><p>The paper reports Simplicity as the hardest '
           'category: "Simplicity is more challenging for PRMs" (§4.3, main results), and on its STEM extension "the Simplicity category '
           'remains the most challenging" (appendix B.5). It does not give a cause and does not test one. The idea that correctness-focused '
           'training is the cause is our hypothesis only. The claim is also not universal: 10 of 15 rows in Table 3 (v1) have Simplicity below '
           'both other category means.</p>' + src(a(PAPER, 'PRMBench paper, v1 and v5 checked 12 Oct'),
                                                  a(PK + 'report/LITERATURE.json', 'report/LITERATURE.json')) + '</div>'
         + '<details open><summary>Two actual examples: NR and NCL</summary>'
         + details_by_summary(s, 'Two actual examples').split('</summary>', 1)[1][:-10] + '</details>'
         + '<details><summary>Nine-category legend</summary>'
         + re.search(r'<div class="legendgrid">.*?</div></div>', s, flags=re.S).group(0) + '</details>')

r0 = OFF[OFF['rule'] == 'R0'].set_index('method').overall * 100
res_table = re.search(r'<section id="results">.*?(<div class="scroll"><table>.*?</table></div>)', s, flags=re.S).group(1)
S3 = sec('s3', '3 · THE OFFICIAL METRIC', 'Where are we on the PRMBench leaderboard?',
         exp('Our features come from one teacher-forced pass of Qwen3-8B over each answer (no fine-tuning, no labels). We repeated '
             'the full pipeline with Qwen3-4B. We computed the official PRMScore with the PRMBench evaluator on all answers. Our flag rule is Q80 '
             '(fixed before the corrected results). We ran Qwen2.5-Math-PRM-7B on the same answers with its native 0.5 threshold. '
             'The purpose: to place a label-free method on the published leaderboard.')
         + fig(fig_leaderboard(), 'Overall PRMScore × 100, N16, Q80. Our rows use features from Qwen3-8B (dark) or Qwen3-4B (light). Our rows and the PRM-7B row use the same 6,211 answers and the same evaluator port. '
               'Paper rows are published values with their own decision rules, not reruns.')
         + concl([f'With Qwen3-8B features our Overall score is {f1(r0["LSML_CONT"])}; Qwen2.5-Math-PRM-7B gets {f1(PRM_NAT)}. The paired difference is '
                  f'−0.08 [−0.77, +0.64]: a tie. With Qwen3-4B features both our methods get {f1(R4["LSML_CONT"])}.',
                  'We are above all open PRMs in paper Table 3 (best 60.5) and below the four LLM judges (66.0 to 68.8). '
                  'We use one pass, no labels and no training.',
                  'On ranking only, PRM has the higher AUC (0.801 vs 0.779), but our top-ranked step is an error more often (59.9% vs 57.8%).'])
         + res_table
         + src(a(PK + 'report/legacy/OFFICIAL_PRMSCORE_N16.csv', 'report/legacy/OFFICIAL_PRMSCORE_N16.csv'),
               a(PK + 'report/paper_table3.csv', 'report/paper_table3.csv'),
               a(PK + 'report/legacy/PAIRED_DECISION_ANALYSIS.json', 'PAIRED_DECISION_ANALYSIS.json') + ' (5,000 group resamples)',
               'Full Table 3: appendix D'))

S4 = sec('s4', '4 · ERROR TYPES', 'Which errors do we find, and which signals find them?',
         '<h3>4a · Category profile</h3>'
         + exp('We split the official PRMScore by the nine error types, for us and for PRM-7B. The purpose: to see what is behind the tie.')
         + fig(radar_only(), 'Official PRMScore per error type, N16, Q80 for our methods and the native 0.5 threshold for PRM-7B. '
               'Radius 0-100. MS uses valid-step F1 alone.')
         + concl(['We lead in Simplicity: NR 70.0 vs 49.0 and NCL 68.0 vs 55.0. No row in paper Table 3 is higher on NR.',
                  'PRM-7B leads on the plausible-looking errors: Soundness 71.0 vs 66.9 and Sensitivity 75.5 vs 69.3 (Sensitivity includes MS).',
                  'Our MS score is low (88.4 vs 99.9), because Q80 flags a step in almost every clean answer. Section 8 comes back to this.'])
         + '<h3>4b · Is the Simplicity lead real?</h3>'
         + exp('Step position, step length and text repetition can predict some error types. We matched error and valid steps on relative '
               'position and length inside each answer, and compared AUC on the matched pairs. The purpose: to test if the lead comes '
               'from the signal or from these simple cues.')
         + FIG['Matched AUC controls']
         + concl(['The NR lead stays under every control: +19.7 AUC points over PRM-7B [15.9, 23.6] after position and length matching. '
                  'This is our strongest result.',
                  'The NCL lead becomes +4.3 points, and the interval crosses zero. We do not claim it.',
                  'On all answers, position matching removes PRM\'s AUC lead and our lead over the plain average '
                  '(4,691 of 6,030 answers can be matched; a caution, not a final result).'])
         + '<h3>4c · Which signals find which errors?</h3>'
         + exp('We computed the within-answer AUC of each of the 56 channels alone, per error type, with a label-free sign rule. '
               'Step index is a baseline row. The purpose: to see which signal families catch which error types.')
         + FIG['All 56 channels plus step index']
         + concl(['No channel wins on all types. Uncertainty channels lead on redundancy (0.84), domain (0.81) and circular (0.77); '
                  'digit_spread leads on deception (0.73).',
                  'Step index alone reaches 0.87 on confidence, above every channel, but it is near chance on domain (0.54) and circular (0.57). '
                  'So our strong types are not position effects.',
                  'One channel (rolling spectral entropy) is below 0.5 on all types: the label-free sign rule turns it the wrong way.'])
         + src(a(PK + 'controls/NUISANCE_CONTROLS.csv', 'controls/NUISANCE_CONTROLS.csv'),
               a(PK + 'statistics/CHANNEL_BY_TYPE.csv', 'statistics/CHANNEL_BY_TYPE.csv'),
               a(PK + 'report/legacy/OFFICIAL_PRMSCORE_N16.csv', 'OFFICIAL_PRMSCORE_N16.csv')))

LF = ('ALL_EQ', 'DS_FILTER_EQ', 'LSML_CONT', 'IU_PCR')
SO, HV = 'socratic_qwen3_8b', 'hard2verify_qwen3_8b'
assert EXTRA is not None, 'v15 needs the LR/PRM CSV as 5th argument'
off = lambda c, m, md: ext_val(c, m, md)
auc = lambda c, m, md: ext_val(c, m, md, 'within_auc')
s8b = [off(SO, m, '8b') for m in LF]; s4b = [off(SO, m, '4b') for m in LF]
h_lf = [off(HV, m, md) for m in LF for md in ('8b', '4b')]
lf_auc_s = [auc(SO, m, md) for m in LF for md in ('8b', '4b')]
lf_auc_h = [auc(HV, m, md) for m in LF for md in ('8b', '4b')]
lr_s, lr_h = [off(SO, 'LR', md) for md in ('8b', '4b')], [off(HV, 'LR', md) for md in ('8b', '4b')]
lr_auc_s, lr_auc_h = [auc(SO, 'LR', md) for md in ('8b', '4b')], [auc(HV, 'LR', md) for md in ('8b', '4b')]
prm_s, prm_h = off(SO, 'PRM', 'nat'), off(HV, 'PRM', 'nat')
prm_sq, prm_hq = off(SO, 'PRM', 'q80'), off(HV, 'PRM', 'q80')
prm_auc_s, prm_auc_h = auc(SO, 'PRM', 'nat'), auc(HV, 'PRM', 'nat')
assert max(abs(a_ - b_) for a_, b_ in zip(s4b, s8b)) < 0.003 and min(s8b) > idx[SO]
assert prm_s > max(lr_s) > max(s8b + s4b) and abs(prm_s - PUB_PRM[SO]) < 0.001
assert min(lr_h) > max(h_lf + [prm_h, prm_hq]) and idx[HV] > max(lr_h)
assert min(lr_auc_s) > max(lf_auc_s) and min(lr_auc_h) > max(lf_auc_h)
assert prm_auc_s > max(lr_auc_s) and prm_auc_h < min(lr_auc_h)
r2 = lambda xs: f'{f3(min(xs))} to {f3(max(xs))}'
S5 = sec('s5', '5 · TRANSFER', 'Does it transfer outside PRMBench?',
         exp('We applied the same N16 pipeline, with the same methods as in sections 0-4, to two other benchmarks: Socratic-PRMBench '
             '(same error taxonomy, 2,995 answers) and Hard2Verify (200 answers). The label-free methods fit on each benchmark without labels. '
             'LR is trained only on PRMBench labels and applied without target labels. We ran Qwen2.5-Math-PRM-7B on the same answers and steps, '
             'with its native 0.5 threshold and with our Q80 rule. Each benchmark uses its own official metric. We repeated our methods '
             'and LR with Qwen3-4B features. The purpose: to test the methods on data that we did not use for development.')
         + fig(fig_external(), 'Official metric per benchmark (PRMScore for Socratic, Balanced F1 for Hard2Verify). Our methods and LR: N16, '
               'ANSWER_Z_Q80. PRM-7B: our run on the same answers, native 0.5 threshold (dark) and Q80 (light). Dotted line: the published '
               'PRM-7B value, with thresholds tuned by the benchmark authors. Step index uses no model features. LR and PRM-7B were run on '
               '12 October. The frozen-weight transfer test (weights fitted on the old input) is in appendix D.')
         + concl([f'On Socratic, PRM-7B is the strongest: {f3(prm_s)} at its 0.5 threshold, the same as the published value. '
                  f'LR, trained on PRMBench only, gets {f3(lr_s[0])} (8B) and {f3(lr_s[1])} (4B). The four label-free methods get '
                  f'{r2(s8b + s4b)}, above step index ({f3(idx[SO])}).',
                  f'On Hard2Verify, LR is the best method on the same answers: {f3(lr_h[0])} (8B) and {f3(lr_h[1])} (4B), against '
                  f'{f3(prm_hq)} for PRM-7B with Q80 and {f3(prm_h)} with its 0.5 threshold (it flags very few steps). But step index alone '
                  f'({f3(idx[HV])}) is above all methods, and there are only 200 answers with no CIs.',
                  f'On ranking (within-answer AUC), supervision from PRMBench transfers. LR is above all label-free methods on both benchmarks '
                  f'(Socratic {r2(lr_auc_s)} vs {r2(lf_auc_s)}; Hard2Verify {r2(lr_auc_h)} vs {r2(lf_auc_h)}). PRM-7B ranks best on Socratic '
                  f'({f3(prm_auc_s)}) but below LR on Hard2Verify ({f3(prm_auc_h)}).'])
         + src('qwen3_4b_vs_8b__external_cont.csv and __external_nested.csv (project folder; results/thinkfix_rescore_4b_v1, not in Git)',
               'LR and PRM-7B: results/external_lr_prm_v1/RESULTS.csv (local commit adc32b5fc; AIRCC jobs 282806, 282807)',
               'published values: lsml_external_generalization_v1/evaluation/PUBLISHED_CONTEXT.json (' + a('https://arxiv.org/html/2505.23474v1', 'Socratic-PRMBench Table 3')
               + ', ' + a('https://arxiv.org/html/2510.13744v1', 'Hard2Verify Table 2') + ')'))

S6 = sec('s6', '6 · SUPERVISION', 'What does supervision add?',
         '<h3>6a · The size of the gap</h3>'
         + exp('We trained logistic regression (LR) on the same features with step labels, with source-group cross-fitting, in all '
               'ten banks. The purpose: to measure how much a supervised fusion of the same signals can add.')
         + FIG['New OOF gap across all ten banks']
         + concl(['The AUC gap is smallest at N16 (0.021) and largest at N40 (0.063).',
                  'On top1 we lead LR up to N25. At N16, L-SML is 2.0 points above LR [1.0, 3.0].',
                  'LR does not show a weighting that we can copy. Its weights agree with ours at cosine about 0.03. Its two largest weights '
                  '(+0.44 and −0.44) sit on two almost identical digit channels (r = 0.9998) and cancel; all other LR weights are 0.025 or less.'])
         + '<h3>6b · Where each one is right</h3>'
         + exp('For each answer with an error, we checked if the top-ranked step is an error, for L-SML and for LR. We split the answers '
               'into four groups (both right, only ours, only LR, both wrong) and by error type. The purpose: to see if the two methods find different errors.')
         + FIG['Four-way top1 outcomes by error type']
         + concl(['L-SML alone finds the error in 485 answers that LR misses; LR alone finds 366.',
                  'We win on redundancy (90 vs 27), domain (79 vs 32) and circular (77 vs 42). This matches the channel heatmap in 4c.',
                  'LR wins on confidence (60 vs 28) and deception (75 vs 58), the plausible-looking errors.'])
         + '<h3>6c · Model size</h3>'
         + exp('We repeated the full pipeline with Qwen3-4B as the teacher-forcing model. Answers, labels, banks and methods are the same. '
               'The purpose: to see if the results depend on the model that produces the features.')
         + FIG['Change in within-answer AUC from Qwen3-8B to Qwen3-4B']
         + concl(['The label-free methods change by less than 0.01 AUC in every bank, in both directions.',
                  'LR is lower on 4B in all ten banks (−0.003 to −0.010). It depends on the model more than our methods do.',
                  'The category profile and the PRMScore almost do not change (65.4 → 65.0; NR stays at about 70).'])
         + src(a(PK + 'lr/METRICS.csv', 'lr/METRICS.csv'), a(PK + 'lr/FOURWAY_SUMMARY.csv', 'lr/FOURWAY_SUMMARY.csv'),
               a(PK + 'lr/WEIGHTS.csv', 'lr/WEIGHTS.csv'), 'qwen3_4b_vs_8b__prmbench_*.csv (project folder)',
               'LR weights, family mass, omission and crossover: appendix B'))

S7 = sec('s7', '7 · THE DECISION', 'Why is the score only a tie?',
         exp('We moved one global threshold over all steps and computed PRMScore at each flag rate, for us and for PRM-7B. '
             'The purpose: to see if a better global threshold closes the gap.')
         + FIG['Budget curves and oracle diagnostics']
         + concl([f'At the same flag rate (about 18% of steps), PRM-7B gets {PRM_BEST.prm_score * 100:.1f} and we get '
                  f'{CEIL["LSML_CONT"]["global_threshold"]["prm_score"] * 100:.1f}. PRM-7B ranks better.',
                  'PRM-7B\'s 0.5 threshold flags only 8.7% of steps, far from its best point. That is why the default scores are equal.',
                  'Our Q80 point is almost at the top of our curve: the best global threshold adds only 0.3 points. The problem is inside each answer.'])
         + src(a(PK + 'controls/BUDGET_CURVES.csv', 'controls/BUDGET_CURVES.csv'),
               a(PK + 'controls/DIAGNOSTIC_CEILINGS.json', 'controls/DIAGNOSTIC_CEILINGS.json')))

L = {r[0]: r[1] for r in lad}
S8 = sec('s8', '8 · WHAT IS LEFT', 'What is left to gain, and where did we stop?',
         '<h3>8a · The value of the per-answer count</h3>'
         + exp('We kept our ranking and changed only the number of flags in each answer, with label-based oracles as ceilings. '
               'The purpose: to find which part of the decision has the largest headroom.')
         + fig(fig_ladder(), 'N16, L-SML ranking, 6,211 noncontrol answers. Each row changes only the number of flags per answer '
               '(except the last row, which changes the ranking). The gains do not add.')
         + concl([f'With the same ranking, the true number of errors per answer adds +{L["True count per answer"] - L["Q80 · our frozen rule"]:.1f} points. '
                  f'This is above PRM-7B at its best threshold ({PRM_BEST.prm_score * 100:.1f}).',
                  f'A constant two flags per answer gives {L["Constant 2 flags per answer (TOP_2)"]:.1f}. This is a post hoc choice, not a method, '
                  'but it shows that the right count is small.',
                  f'Perfect ranking adds more (+{L["Perfect ranking at our own counts"] - L["Q80 · our frozen rule"]:.1f}), but the LR gap shows that our '
                  'features do not have this information now. The count is the part that we can work on without new signals.'])
         + src(a(PK + 'controls/DIAGNOSTIC_CEILINGS.json', 'controls/DIAGNOSTIC_CEILINGS.json'),
               a(PK + 'controls/EXACT_DECISION_RULES.csv', 'controls/EXACT_DECISION_RULES.csv'),
               a(COUNT, 'count_structure_check_v1') + ' (commit 17aa1e4; count from error type)')
         + '<h3>8b · What we tried</h3>'
         + exp('We tested label-free rules that choose the count per answer: two registered rules (R2_COUNT, entropy gate), MoM on the '
               'L-SML group votes, and the Codex estimator families A-D. The purpose: to find a label-free count that beats Q80.')
         + fig(fig_tried(), 'Change in PRMScore against Q80 with the same IU-PCR ranking. Registered rules and MoM: full population '
               f'(Q80 = {q80["IU_PCR"]:.2f}). Codex arms: out-of-fold, against their own OOF Q80 (65.39). Arm A is not shown: '
               'it failed in 4 of 5 N16 folds (1,269 of 6,211 answers covered).')
         + concl(['No label-free rule beats Q80 on PRMScore.',
                  f'R2_COUNT and the entropy gate make many clean answers clean again: false flags on clean controls fall from {cff["ANSWER_Z_Q80"]:.1f}% '
                  f'to {cff["R2_COUNT"]:.1f}% and {cff["ENTROPY_GATE_Q80"]:.1f}%, and MS rises from {ms_iu["ANSWER_Z_Q80"]:.1f} to '
                  f'{ms_iu["ENTROPY_GATE_Q80"]:.1f} with the entropy gate. They help a different goal: the clean-answer decision.',
                  'The MoM error rate follows the vote quota, not the data: quotas of 10%, 20% and 30% give estimates of 13.6%, 22.9% and 32.6%.'])
         + src(a(PK + 'controls/EXACT_DECISION_RULES.csv', 'controls/EXACT_DECISION_RULES.csv'),
               a(PK + 'statistics/MOM_QUOTA.csv', 'statistics/MOM_QUOTA.csv'),
               a(CODEX, 'codex/label-free-error-count-v1 @ 6a0b870'))
         + '<h3>8c · Why they failed</h3>'
         + exp('We compared the true number of errors per answer with answer length, and with the counts that our rules give. '
               'The purpose: to explain the failure in 8b.')
         + fig(fig_counts(ct), 'Mean per length bin, 6,211 noncontrol answers; IU-PCR ranking for our rules. ρ: Spearman correlation '
               'with the number of steps over all answers. Aggregated from the saved per-answer counts; no new scoring.')
         + concl([f'In PRMBench, an answer has about 2 errors (mean {ct["truth_mean"]:.2f}). This number almost does not depend on length '
                  f'(from {ct["truth"][0]:.1f} in short answers to {ct["truth"][-1]:.1f} in long ones; ρ = {ct["truth_rho"]:.2f}).',
                  f'Q80 gives more flags in longer answers: from {ct["Q80"][0]:.1f} to {ct["Q80"][-1]:.1f} (ρ = {ct["Q80_rho"]:.2f}). '
                  f'R2_COUNT and the entropy gate also grow with length (ρ = {ct["R2_COUNT_rho"]:.2f} and {ct["ENTROPY_rho"]:.2f}).',
                  f'PRM-7B gives a flat count, but too low (about {sum(ct["PRM"]) / len(ct["PRM"]):.1f}). A rule with counts that grow with length '
                  'cannot match this benchmark; a good count rule needs a different signal.'])
         + src(a(PK + 'controls/PER_ANSWER_COUNTS.csv', 'controls/PER_ANSWER_COUNTS.csv'), a(COUNT, 'count_structure_check_v1'))
         + '<div class="box open"><span class="kt">Open direction · for discussion</span><ol>'
           '<li>First check if the fused scores inside an answer (for example the gap between the top steps) predict the count beyond length. '
           'Use a supervised count model as the ceiling. If this fails, move the effort to ranking.</li>'
           '<li>If the information exists: estimate the count from the continuous scores (for example a mixture model per answer), '
           'not from thresholded votes.</li>'
           '<li>Keep two problems apart: how many errors a wrong answer has (PRMScore), and when an answer is clean (MS and the clean controls).</li>'
           '</ol></div>')

# ---------------------------------------------------------------- appendix
old_rows = ''
for b in BANKS:
    cells = ''
    for m in ('ALL_EQ', 'DS_FILTER_EQ', 'LSML_CONT', 'IU_PCR'):
        o, n = OLDNEW.loc[(b, m), 'frozen'], OLDNEW.loc[(b, m), 'thinkfix_mask1']
        cells += f'<td>{o:.3f} → {n:.3f}</td>'
    cells += f'<td>{LR_OLD[b]:.3f} → {LR_NEW[b]:.3f}</td>'
    old_rows += f'<tr><th>{b}</th>{cells}</tr>'
OLD_TABLE = ('<div class="appx-part"><div class="eyebrow">8 OCTOBER VALUES</div><h2>Within-answer AUC: 8 October input → corrected input</h2>'
             '<div class="scroll"><table><thead><tr><th>Bank</th><th>Plain average</th><th>DS-filter average</th><th>Continuous L-SML</th>'
             '<th>IU-PCR</th><th>LR · supervised</th></tr></thead><tbody>' + old_rows + '</tbody></table></div>'
             '<p class="small">"8 October input" is the frozen value before the think-block fix. Corrected = think block + mask1. '
             'The 8 October control was DS-signed; this table uses DS-filter on both sides.</p>'
             + src(a(PK + 'report/legacy/PRMBENCH_COMPARISON.csv', 'report/legacy/PRMBENCH_COMPARISON.csv') + ' (frozen, thinkfix_mask1)',
                   a(OCT8, '8 October presentation')) + '</div>')

app_html = section('appendix')
D_MATRICES = ''.join(details_by_summary(app_html, k) for k in ('All banks × methods: AUC', 'Top1-on-error', 'PRMScore · all defined'))
E_PROV = ''.join(details_by_summary(app_html, k) for k in ('Dataset construction', 'External context', 'Raw tables'))
results_details = details_by_summary(section('results'), 'Paired comparisons')
lead = appx_part('leaderboard', drop=(details_by_summary(section('leaderboard'), 'Two actual examples'),))
opendir = appx_part('open-direction').replace(' (what matters for ProcessBench and the clean controls; PRMScore rewards it very little)',
                                              ' (what matters for the clean controls; PRMScore rewards it very little)')
assert 'ProcessBench and the clean' not in opendir


def appx(letter, title, body):
    return f'<details class="appx"><summary>{letter} · {title}</summary>{body}</details>'


APPX = ('<section id="appendix-v13"><div class="eyebrow">APPENDIX · COLLAPSED</div><h2>Evidence and earlier material</h2>'
        '<p>The v12 sections are here without changes, except that the figures used in the main text are not repeated.</p>'
        + appx('A', 'Corrections and robustness', OLD_TABLE + appx_part('changed') + appx_part('simplicity-controls') + appx_part('internal-validation') + appx_part('q80-provenance'))
        + appx('B', 'LR analysis', appx_part('lr-diagnostics') + appx_part('features'))
        + appx('C', 'Decision rules and estimators', appx_part('decision') + appx_part('registered-rules') + opendir
               + appx_part('mom-quota') + appx_part('dependence'))
        + appx('D', 'Full matrices and tables', '<div class="appx-part"><h2>All banks × methods</h2>' + D_MATRICES + results_details + '</div>'
               + lead + appx_part('model-size'))
        + appx('E', 'Provenance and task status', '<div class="appx-part"><h2>Sources and audits</h2>' + E_PROV + '</div>'
               + appx_part('tasks') + appx_part('remaining-work'))
        + '</section>')

# ---------------------------------------------------------------- assemble
css = re.search(r'<style>(.*?)</style>', s, flags=re.S).group(1)
css += ('.exp{background:#f6f8fb;border:1px solid #dfe7f0;border-radius:10px;margin:16px 0 6px;padding:12px 16px;font-size:15px;'
        'line-height:1.55;max-width:1000px}.exp .kt,.box .kt{display:block;font-size:11px;letter-spacing:1.2px;font-weight:750;'
        'color:#657a91;text-transform:uppercase;margin-bottom:4px}.concl .kt{color:#1768ac}'
        '.box{border:1px solid #dfe7f0;border-radius:10px;padding:12px 16px;margin:18px 0;max-width:1000px;font-size:14.5px;background:#fbfcfd}'
        '.box p{margin:4px 0}.box ul,.box ol{font-size:14.5px;margin:4px 0;padding-left:22px}.box.warn{background:#fff8ef;border-color:#efd7b4}'
        '.box.open{background:#f3f7fb;border-color:#c9d9ea}.prose p{font-size:15.5px;margin:8px 0}'
        'h3{font-size:20px;margin:30px 0 4px;letter-spacing:-.2px}section h3:first-of-type{margin-top:6px}'
        '.appx{border-top:1px solid #dfe7f0}.appx>summary{font-size:18px;font-weight:700;color:#223548;padding:6px 0}'
        '.appx-part{border:1px solid #e6ecf2;border-radius:12px;padding:18px;margin:16px 0;background:#fff}'
        '.appx-part h2{font-size:22px}.moved{font-style:italic}:root{color-scheme:light}@media(max-width:720px){main{padding:20px 16px}}')
NAV = ''.join(f'<a href="#{i}">{t}</a>' for i, t in (('s0', '0 Start point'), ('s1', '1 Corrections'), ('s2', '2 PRM & PRMBench'),
                                                       ('s3', '3 Leaderboard'), ('s4', '4 Error types'), ('s5', '5 Transfer'),
                                                       ('s6', '6 Supervision'), ('s7', '7 Why a tie'), ('s8', '8 What is left'),
                                                       ('appendix-v13', 'Appendix')))
page = ('<title>PRMBench Advisor Review</title><style>' + css + '</style><main>'
        '<header id="top"><div class="eyebrow">ADVISOR REVIEW · MEETING 15 OCTOBER 2026 · V15</div>'
        '<h1>What our label-free signals<br>capture in PRMBench</h1>'
        '<p class="lead">One teacher-forced pass, no labels, no training: where we are after the corrections, which errors we find, '
        'and what is left to gain.</p><div class="tag">PRMBench · N16 · Qwen3-8B · corrected prefix + mask1 · development evidence</div></header>'
        '<aside class="notice">Version 15, 12 October 2026: section 5 adds LR and Qwen2.5-Math-PRM-7B on the same external answers (new runs on 12 October, approved by Omri). Version 14: section 3 names the feature model and adds Qwen3-4B; section 5 now compares the same methods as the other sections. Version 13: new order, approved by Omri (outline v2): PRMBench only; each figure has the '
        'experiment before it and three conclusions after it. All numbers come from saved results; v13 ran no new experiments. Six figures are '
        'new plots of saved numbers (sections 0, 1, 3, 5 and 8). The v12 material is in the collapsed appendix.</aside>'
        '<nav>' + NAV + '</nav>' + S0 + S1 + S2 + S3 + S4 + S5 + S6 + S7 + S8 + APPX
        + '<footer><b>Evidence state:</b> v11 package (commit 07cb7a4), count check (17aa1e4), Codex count study (6a0b870), '
          'think-fix re-score (080877e) and the Qwen3-4B re-score (project folder, not in Git). No new GPU or CPU experiments for this page.</footer>'
        '</main>')
for sid in [x for x in re.findall(r'<section id="([^"]+)"', s) if x not in ('results', 'appendix')]:
    assert f'id="{sid}"' in page, ('v12 section missing', sid)
for bad in ('Mind the Gap',):
    assert bad not in page, bad
main_text = page[:page.find('<section id="appendix-v13"')]
assert 'ProcessBench' not in main_text.replace('(ProcessBench GSM8K/MATH)', ''), 'ProcessBench in main text'
assert main_text.count('<span class="kt">Conclusions</span>') == 15
open(OUT, 'w', encoding='utf-8').write(page)
print('written', OUT, len(page))
