"""Build the PRMBench-first advisor page from the think-block re-score (thinkfix_mask1).
Usage: python3 gen_prmb_v2.py <repo_root> <official_prmscore_dir> <out_html>
  repo_root: checkout of omrisegev/hallucination_detection at claude/thinkfix-rescore-v1 (8cc6498)
  official_prmscore_dir: OFFICIAL_PRMSCORE_<bank>.csv from official_prmscore_v1.py on prmscore_inputs/"""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path

REPO, OFF, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
D = 'results/thinkfix_rescore_v1'
F_PB, F_PBB = f'{D}/PRMBENCH_COMPARISON.csv', f'{D}/PROCESSBENCH_COMPARISON.csv'
BANKS = ['N11', 'N13', 'N22', 'N40', 'N53', 'N14', 'N16', 'N25', 'N43', 'N56']
PIPE = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']
METH = ['ALL_EQ', 'DS_FILTER_EQ', 'LSML_CONT', 'IU_PCR'] + PIPE
NAME = dict(zip(METH, ['Plain average', 'DS-filter average', 'Continuous L-SML', 'IU-PCR'] + PIPE))
PRM = 'Qwen2.5-Math-PRM-7B (supervised)'
NEW, OLD = 'thinkfix_mask1', 'old_replay'

pb = pd.read_csv(REPO / F_PB); pbb = pd.read_csv(REPO / F_PBB)
pnl = pb[pb.table == 'prmbench_panel_b']; cb = pb[pb.table == 'cont_baselines']; cat = pb[pb.table == 'prmbench_by_category']
gaps = pb[pb.table == 'lr_gaps']
off = pd.concat([pd.read_csv(OFF / f'OFFICIAL_PRMSCORE_{b}.csv') for b in BANKS])

def v(bank, m, metric, col=NEW):  # metric: within_auc | prmscore_top1 | top1_on_error
    if m in ('LSML_CONT', 'IU_PCR'):
        r = cb[(cb.bank == bank) & (cb.scorer == m) & (cb.metric == 'prmb_' + metric)]
    else:
        r = pnl[(pnl.bank == bank) & (pnl.scorer == m) & (pnl.metric == metric)]
    return float(r[col].iloc[0]) if len(r) else np.nan

def ref(s, metric, col=NEW):
    r = pnl[(pnl.scorer == s) & (pnl.metric == metric)]; return float(r[col].iloc[0])

f3 = lambda x: '–' if np.isnan(x) else f'{x:.3f}'
sg = lambda x: '–' if np.isnan(x) else f'{x:+.3f}'
def wrap(caption, head, rows, note):
    return f'<div class="tw"><table><caption>{caption}</caption>{head}{"".join(rows)}</table></div><p class="note">{note}</p>'

SRC = (f'Source: <code>{F_PB}</code>, column <code>{NEW}</code>; table <code>prmbench_panel_b</code> (plain average, DS-filter average, pipeline) '
       'and table <code>cont_baselines</code> (L-SML, IU-PCR).')

def bank_table(metric, caption):
    head = '<tr><th>Bank</th>' + ''.join(f'<th>{NAME[m]}</th>' for m in METH) + '<th class="mcol">Mean, label-free</th><th class="lrc">LR (supervised)</th></tr>'
    rows, allv = [], []
    for b in BANKS:
        x = np.array([v(b, m, metric) for m in METH]); allv.append(x); bx = np.nanmax(x)
        lead = b == 'N16'
        rows.append(f'<tr{" class=lead" if lead else ""}><th>{"N16 (leading bank)" if lead else b}</th>'
                    + ''.join(f'<td{" class=best" if y == bx else ""}>{f3(y)}</td>' for y in x)
                    + f'<td class="mcol">{f3(np.nanmean(x))}</td><td class="lr">pending</td></tr>')
    M = np.nanmean(allv, axis=0); bm = M.max()
    rows.append('<tr class="mrow"><th>Mean, 10 banks</th>' + ''.join(f'<td{" class=best" if y == bm else ""}>{y:.3f}</td>' for y in M)
                + f'<td class="mcol">{M.mean():.3f}</td><td class="lr">pending</td></tr>')
    note = ('Bold = best label-free method in the row. "Mean, label-free" does not include LR. '
            'LR is pending: the supervised re-score on the fixed input is still running. ' + SRC)
    return wrap(caption, head, rows, note), np.array(allv), M

T_AUC, AUC, M_AUC = bank_table('within_auc', 'PRMBench · within-answer AUC (higher is better)')
T_PS, PS, M_PS = bank_table('prmscore_top1', 'PRMBench · PRMScore, one flag per answer (higher is better)')
i16 = BANKS.index('N16')
bank_means = AUC.mean(axis=1)

# ---- before / after the think-block fix
old_auc = np.array([[v(b, m, 'within_auc', OLD) for m in METH] for b in BANKS])
dlt = AUC - old_auc
BA = ['<tr><th></th>' + ''.join(f'<th>{NAME[m]}</th>' for m in METH) + '<th class="mcol">Mean</th></tr>']
for lab, x in [('N16, before (missing think block)', old_auc[i16]), ('N16, after (fixed + first token masked)', AUC[i16]), ('N16, change', dlt[i16])]:
    fm = sg if 'change' in lab else f3
    BA.append(f'<tr{" class=lead" if "after" in lab else ""}><th>{lab}</th>' + ''.join(f'<td>{fm(y)}</td>' for y in x) + f'<td class="mcol">{fm(x.mean())}</td></tr>')
BA.append('<tr class="mrow"><th>Change, mean over banks</th>' + ''.join(f'<td>{sg(y)}</td>' for y in np.nanmean(dlt, axis=0)) + f'<td class="mcol">{sg(np.nanmean(dlt))}</td></tr>')
old_best_bank = BANKS[int(np.nanargmax(np.nanmean(old_auc, axis=1)))]
T_BA = wrap('Within-answer AUC before and after the think-block fix', '', BA,
            f'"Before" is the replay of the frozen pipeline on the old input. It is equal to the numbers on the previous version of this page '
            f'(the DS-filter average replaces the DS-signed column). The pipeline did not run on N11 before the fix, so the last row uses 9 banks for the pipeline columns. Source: <code>{F_PB}</code>, columns <code>{OLD}</code> and <code>{NEW}</code>.')

# ---- official PRMScore
RULES = ['R0', 'R2', 'TOP1', 'TOP2']
o16 = off[off.bank == 'N16']
def o(bank, m, rule, col='overall'):
    r = off[(off.bank == bank) & (off.method == m) & (off.rule == rule)]; return float(r[col].iloc[0])
prm_off = off[off.bank == '-'].iloc[0]
OH = '<tr><th>Method (N16)</th>' + ''.join(f'<th>{r}</th>' for r in RULES) + '</tr>'
OR = []
for m in METH:
    x = [o('N16', m, r) for r in RULES]
    OR.append(f'<tr><th>{NAME[m]}</th>' + ''.join(f'<td>{y:.3f}</td>' for y in x) + '</tr>')
xm = [np.mean([o('N16', m, r) for m in METH]) for r in RULES]
OR.append('<tr class="mrow"><th>Mean, 8 methods</th>' + ''.join(f'<td>{y:.3f}</td>' for y in xm) + '</tr>')
r1 = o('N16', 'MEAN_MOM_ga', 'R1')
OR.append(f'<tr class="lead"><th>R1 (one global score, same for all methods)</th><td colspan="4" style="text-align:left">{r1:.3f}</td></tr>')
OR.append(f'<tr class="refrow"><th>Qwen2.5-Math-PRM-7B, its own 0.5 threshold</th><td colspan="4" style="text-align:left">{prm_off.overall:.3f}</td></tr>')
T_OFF = wrap('PRMBench · official PRMScore, N16, by flag rule', OH, OR,
             'Official PRMScore = ½ · (F1 on correct steps + F1 on error steps), pooled over all non-control answers, scored by the repository port of the '
             'PRMBench evaluator (<code>spectral_utils/prmbench.py::prmbench_evaluate</code>). No rule uses labels. '
             'Source: <code>/mnt/project-files/advisor_page/official_prmscore_thinkfix_mask1/OFFICIAL_PRMSCORE_N16.csv</code> '
             f'(script <code>official_prmscore_v1.py</code>, input <code>{D}/prmscore_inputs/bank_N16.npz</code>).')

# rule behaviour (N16, MEAN_MOM_ga)
FH = '<tr><th>Rule (N16, MEAN_MOM_ga)</th><th>Steps flagged</th><th>Answers with ≥1 flag</th><th>Error-free controls with ≥1 flag</th><th>PRMScore</th></tr>'
FR = []
for r in ['R0', 'R1', 'R2', 'TOP1', 'TOP2']:
    q = o16[(o16.method == 'MEAN_MOM_ga') & (o16.rule == r)].iloc[0]
    FR.append(f'<tr><th>{r}</th><td>{q.step_flag_rate:.1%}</td><td>{q.answers_with_flag:.1%}</td><td>{q.controls_with_flag:.1%}</td><td>{q.overall:.3f}</td></tr>')
T_FLAG = wrap('What each rule flags', FH, FR, 'Controls = the PRMBench answers with no error (the <code>correct_*</code> rows). They are not in the pooled PRMScore. Same source as above.')

# categories
CL = ['redundency', 'circular', 'counterfactual', 'step_contradiction', 'domain_inconsistency', 'confidence', 'missing_condition', 'deception']
CH = ('<tr><th></th><th>Overall</th><th>Simplicity</th><th>Soundness</th><th>Sensitivity</th>' + ''.join(f'<th>{c}</th>' for c in CL) + '</tr>')
CR = []
def crow(lab, q, cls=''):
    return (f'<tr{cls}><th>{lab}</th><td>{q["overall"]:.3f}</td><td>{q["simplicity"]:.3f}</td><td>{q["soundness"]:.3f}</td><td>{q["sensitivity"]:.3f}</td>'
            + ''.join(f'<td>{q[c]:.3f}</td>' for c in CL) + '</tr>')
for r, cls in [('R0', ' class=lead'), ('R1', ''), ('TOP2', '')]:
    CR.append(crow(f'Ours, MEAN_MOM_ga, {r}' if r != 'R1' else 'Ours, R1', o16[(o16.method == 'MEAN_MOM_ga') & (o16.rule == r)].iloc[0], cls))
CR.append(crow('Qwen2.5-Math-PRM-7B, 0.5 threshold', prm_off, ' class=refrow'))
q0 = o16[(o16.method == 'MEAN_MOM_ga') & (o16.rule == 'R0')].iloc[0]
T_CAT = wrap('Official PRMScore by PRMBench category, N16', CH, CR,
             'Simplicity = redundency + circular. Soundness = counterfactual + step_contradiction + domain_inconsistency + confidence. '
             'Sensitivity = missing_condition + deception. multi_solutions has no error steps, so it is not in the Sensitivity mean. '
             '"redundency" is the PRMBench spelling. We have no redundancy head, so redundency and circular use the evaluator\'s validity fallback. '
             'PRM-7B row: from <code>dataset_cache/four_localization/prmbench_qwen25math7b_full/manifest.json</code>. Same source as above.')

# all banks, R0 / R2 / TOP2
BH = '<tr><th>Bank</th><th>R0, mean of 8 methods</th><th>R0, range</th><th>R1</th><th>R2, mean</th><th>TOP2, mean</th></tr>'
BR = []
for b in BANKS:
    x0 = [o(b, m, 'R0') for m in METH]
    lead = b == 'N16'
    BR.append(f'<tr{" class=lead" if lead else ""}><th>{b}</th><td>{np.mean(x0):.3f}</td><td>{min(x0):.3f} to {max(x0):.3f}</td><td>{o(b, "MEAN_MOM_ga", "R1"):.3f}</td>'
              f'<td>{np.mean([o(b, m, "R2") for m in METH]):.3f}</td><td>{np.mean([o(b, m, "TOP2") for m in METH]):.3f}</td></tr>')
T_OFFB = wrap('Official PRMScore on all banks', BH, BR, 'Source: <code>/mnt/project-files/advisor_page/official_prmscore_thinkfix_mask1/OFFICIAL_PRMSCORE_&lt;bank&gt;.csv</code>.')

# ---- references (same protocol)
RH = '<tr><th></th><th>Within AUC</th><th>PRMScore, one flag</th><th>Top step is an error</th></tr>'
RR = [f'<tr class="lead"><th>Ours, N16, mean of 8 label-free methods</th><td>{AUC[i16].mean():.3f}</td><td>{PS[i16].mean():.3f}</td>'
      f'<td>{np.mean([v("N16", m, "top1_on_error") for m in METH]):.3f}</td></tr>',
      '<tr><th>LR on N16 (supervised headroom)</th><td colspan="3" style="text-align:left">pending</td></tr>']
for s in [PRM, 'Step index (later = riskier)', 'Step token length']:
    RR.append(f'<tr class="refrow"><th>{s}</th><td>{ref(s, "within_auc"):.3f}</td><td>{ref(s, "prmscore_top1"):.3f}</td><td>{ref(s, "top1_on_error"):.3f}</td></tr>')
T_REF = wrap('PRMBench references, same protocol', RH, RR,
             f'The Mind the Gap row is not here: its input file is gone, so the re-score could not recompute it. Source: <code>{F_PB}</code>, table <code>prmbench_panel_b</code>.')

# ---- error types, N16 within-AUC
TS = ['ALL_EQ'] + PIPE
TH = ('<tr><th>Error type</th><th>Category</th>' + ''.join(f'<th>{NAME[m]}</th>' for m in TS)
      + '<th class="mcol">Ours, best minus PRM</th><th class="lrc">PRM-7B</th><th class="lrc">Step index</th><th>MEAN_MOM_ga before fix</th></tr>')
CATOF = {'redundency': 'Simplicity', 'circular': 'Simplicity', 'missing_condition': 'Sensitivity', 'deception': 'Sensitivity'}
def cv(bank, s, g, col=NEW):
    r = cat[(cat.bank == bank) & (cat.scorer == s) & (cat.group == g) & (cat.metric == 'within_auc')]; return float(r[col].iloc[0])
rows_t = []
for g in CL:
    x = [cv('N16', m, g) for m in TS]; p = cv('-', PRM, g); d = max(x) - p
    rows_t.append((d, f'<tr{" class=good" if d > 0.02 else (" class=bad" if d < -0.02 else "")}><th>{g}</th><td style="text-align:left">{CATOF.get(g, "Soundness")}</td>'
                   + ''.join(f'<td>{y:.3f}</td>' for y in x) + f'<td class="mcol">{d:+.3f}</td><td class="lr">{p:.3f}</td>'
                   f'<td class="lr">{cv("-", "Step index (later = riskier)", g):.3f}</td><td>{cv("N16", "MEAN_MOM_ga", g, OLD):.3f}</td></tr>'))
T_TY = wrap('PRMBench error types · within-answer AUC, N16', TH, [r for _, r in sorted(rows_t, key=lambda t: -t[0])],
            'Green = our best method is more than 0.02 above PRM-7B. Red = more than 0.02 below. "Best" is picked after we saw the scores. '
            f'Source: <code>{F_PB}</code>, table <code>prmbench_by_category</code>.')
red = {g: (max(cv('N16', m, g) for m in TS), cv('-', PRM, g)) for g in CL}

# ---- ProcessBench, N16
def pv(m, col=NEW):
    t = 'cont_baselines' if m in ('LSML_CONT', 'IU_PCR') else 'panel_a'
    r = pbb[(pbb.table == t) & (pbb.bank == 'N16') & (pbb.scorer == m)]; return float(r[col].iloc[0])
def pref(s):
    r = pbb[(pbb.table == 'processbench_strata') & (pbb.stratum == 'ALL') & (pbb.scorer == s)]; return float(r[NEW].iloc[0])
pn, po = [pv(m) for m in METH], [pv(m, OLD) for m in METH]
PH = '<tr><th></th>' + ''.join(f'<th>{NAME[m]}</th>' for m in METH) + '<th class="lrc">LR</th><th class="lrc">PRM-7B, its top step</th></tr>'
PR = [f'<tr class="lead"><th>After fix</th>' + ''.join(f'<td>{y:.3f}</td>' for y in pn) + f'<td class="lr">pending</td><td class="lr">{pref("Qwen2.5-Math-PRM-7B argmax"):.3f}</td></tr>',
      '<tr class="refrow"><th>Before fix</th>' + ''.join(f'<td>{y:.3f}</td>' for y in po) + '<td class="lr"></td><td class="lr"></td></tr>']
T_PB = wrap('ProcessBench, N16 only · top step = first error (wrong answers, 8-cell mean)', PH, PR,
            f'Source: <code>{F_PBB}</code>, tables <code>panel_a</code>, <code>cont_baselines</code>, <code>processbench_strata</code> (PRM-7B, stratum ALL).')

# ---- LR before the fix (for the pending note)
gf = gaps[gaps.metric == 'gap_vs_best'].frozen

S = dict(
    n16_auc=AUC[i16].mean(), n16_auc_lo=AUC[i16].min(), n16_auc_hi=AUC[i16].max(),
    n16_ps_lo=PS[i16].min(), n16_ps_hi=PS[i16].max(), n16_ps=PS[i16].mean(),
    prm_auc=ref(PRM, 'within_auc'), prm_ps=ref(PRM, 'prmscore_top1'), prm_own=ref(PRM, 'prmscore_own'),
    d_all=np.nanmean(dlt), d_lo=np.nanmin(dlt), d_hi=np.nanmax(dlt), d16=dlt[i16].mean(), old16=old_auc[i16].mean(),
    best_bank=BANKS[int(np.argmax(bank_means))], old_best_bank=old_best_bank,
    spread=M_AUC.max() - M_AUC.min(), mean_best=NAME[METH[int(np.argmax(M_AUC))]], mean_best_v=M_AUC.max(),
    mean_worst=NAME[METH[int(np.argmin(M_AUC))]], mean_worst_v=M_AUC.min(), ps_spread=M_PS.max() - M_PS.min(),
    n16_best=NAME[METH[int(np.argmax(AUC[i16]))]], n16_best_v=AUC[i16].max(),
    r0_lo=min(o('N16', m, 'R0') for m in METH), r0_hi=max(o('N16', m, 'R0') for m in METH), r0_ga=o('N16', 'MEAN_MOM_ga', 'R0'),
    r1=r1, r2_ga=o('N16', 'MEAN_MOM_ga', 'R2'), top2_lo=min(o('N16', m, 'TOP2') for m in METH), top2_hi=max(o('N16', m, 'TOP2') for m in METH),
    prm_off=float(prm_off.overall), prm_simp=float(prm_off.simplicity), prm_sound=float(prm_off.soundness), prm_sens=float(prm_off.sensitivity),
    ga_simp=float(q0.simplicity), ga_sound=float(q0.soundness), ga_sens=float(q0.sensitivity),
    r1_ctrl=float(o16[(o16.method == 'MEAN_MOM_ga') & (o16.rule == 'R1')].controls_with_flag.iloc[0]),
    r0_ctrl=float(q0.controls_with_flag),
    red_ours=red['redundency'][0], red_prm=red['redundency'][1], circ_ours=red['circular'][0], circ_prm=red['circular'][1],
    lr_old_lo=float(gf.min()), lr_old_hi=float(gf.max()),
    pb_plain=pn[0], pb_pipe_lo=min(pn[4:]), pb_pipe_hi=max(pn[4:]), pb_plain_old=po[0],
    pb_ga=pn[METH.index('MEAN_MOM_ga')], pb_ga_old=po[METH.index('MEAN_MOM_ga')],
)
assert S['best_bank'] == 'N16' == S['old_best_bank'], (S['best_bank'], S['old_best_bank'])
json.dump(S, open(OUT.with_suffix('.numbers.json'), 'w'), indent=1, default=float)

tpl = Path(__file__).with_name('template_prmb_v2.html').read_text(encoding='utf8')
for k, val in dict(T_AUC=T_AUC, T_PS=T_PS, T_BA=T_BA, T_OFF=T_OFF, T_FLAG=T_FLAG, T_CAT=T_CAT, T_OFFB=T_OFFB, T_REF=T_REF, T_TY=T_TY, T_PB=T_PB).items():
    tpl = tpl.replace('{{' + k + '}}', val)
for k, val in S.items():
    if isinstance(val, float): tpl = tpl.replace('{{' + k + '}}', f'{val:.3f}').replace('{{+' + k + '}}', f'{val:+.3f}').replace('{{%' + k + '}}', f'{val:.0%}')
    else: tpl = tpl.replace('{{' + k + '}}', str(val))
assert '{{' not in tpl, [l for l in tpl.split('\n') if '{{' in l][:5]
OUT.write_text(tpl, encoding='utf8'); print('ok', len(tpl)); print(json.dumps(S, indent=1, default=lambda x: round(float(x), 4)))
