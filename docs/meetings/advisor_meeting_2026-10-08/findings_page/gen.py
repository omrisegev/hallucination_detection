import pandas as pd, numpy as np, json
from pathlib import Path
R = Path(r'c:\Users\omris\TAU\hallucination_detection\results')
OUT = Path(__file__).parent / 'findings.html'
BANKS = ['N11', 'N13', 'N22', 'N40', 'N53', 'N14', 'N16', 'N25', 'N43', 'N56']
PIPE = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']

pbp = pd.read_csv(R / 'monday_nested_banks_v2/PANEL_B_PRMBENCH.csv')
pbb = pd.read_csv(R / 'monday_nested_banks_v2/PANEL_B_PROCESSBENCH.csv'); pbb = pbb[pbb.stratum == 'ALL']
ds = pd.read_csv(R / 'monday_ds_signed_v1/ABSOLUTE.csv'); ds = ds[ds.scorer == 'DS_SIGNED']
cb = pd.read_csv(R / 'monday_cont_baselines_v1/ABSOLUTE.csv')
lr = pd.read_csv(R / 'monday_lr_reference_v1/LR_SCORES.csv'); lr = lr[lr.scorer == 'LR']
f1 = pd.read_csv(R / 'monday_nested_banks_v2/PANEL_B_PROCESSBENCH_F1.csv')
ty = pd.read_csv(R / 'monday_ds_signed_v1/TYPES_SUMMARY_DS.csv')


def get(df, bank, scorer, col, sc='scorer'):
    v = df[(df.bank == bank) & (df[sc] == scorer)][col]
    return float(v.iloc[0]) if len(v) else np.nan


def table(metric_rows, title, note, better_high=True):
    cols = ['Plain average', 'DS-signed average', 'Continuous L-SML', 'IU-PCR'] + PIPE + ['LR (supervised)']
    h = [f'<div class="tw"><table><caption>{title}</caption><tr><th>Bank</th>' + ''.join(f'<th>{c}</th>' for c in cols) + '<th class="mcol">Mean of label-free methods</th></tr>']
    fmt = lambda v: "–" if np.isnan(v) else f"{v:.3f}"
    rows = {b: metric_rows(b) for b in BANKS}
    for b, vals in rows.items():
        if all(np.isnan(v) for v in vals): continue
        lab = vals[:-1]; best = np.nanmax(lab) if better_high else np.nanmin(lab)
        cells = []
        for i, v in enumerate(vals):
            cls = ' class="lr"' if i == len(vals) - 1 else (' class="best"' if v == best else '')
            cells.append(f'<td{cls}>{fmt(v)}</td>')
        n_lab = int(np.sum(~np.isnan(lab)))
        star = '' if n_lab == len(lab) else f'<sup>{n_lab} methods</sup>'
        rcls = ' class="lead"' if b == 'N16' else ''
        name = 'N16 (leading bank)' if b == 'N16' else b
        h.append(f'<tr{rcls}><th>{name}</th>' + ''.join(cells) + f'<td class="mcol">{fmt(np.nanmean(lab))}{star}</td></tr>')
    # Method means over the banks where every method ran (the pipeline did not run on N11).
    full = [b for b, v in rows.items() if not any(np.isnan(v))]
    means = np.mean([rows[b] for b in full], axis=0)
    best = np.max(means[:-1]) if better_high else np.min(means[:-1])
    cells = ''.join(f'<td class="{"lr" if i == len(means) - 1 else ("best" if v == best else "")}">{v:.3f}</td>' for i, v in enumerate(means))
    h.append(f'<tr class="mrow"><th>Mean over {len(full)} banks</th>{cells}<td class="mcol">{np.mean(means[:-1]):.3f}</td></tr>')
    note += f' Last row: mean of each method over the {len(full)} banks where all methods ran (no N11: the pipeline did not run on N11). Last column: mean of the label-free methods on that bank, without LR; on N11 it is the mean of the 4 methods that ran.'
    h.append(f'</table></div><p class="note">{note}</p>')
    return '\n'.join(h)


def prmb_rows(col_pb, col_cb, col_ds, col_lr):
    return lambda b: [get(pbp, b, 'ALL_EQ', col_pb), get(ds, b, 'DS_SIGNED', col_ds), get(cb, b, 'LSML_CONT', col_cb), get(cb, b, 'IU_PCR', col_cb)] + \
        [get(pbp, b, s, col_pb) for s in PIPE] + [get(lr, b, 'LR', col_lr)]


def pb_rows(b):
    return [get(pbb, b, 'ALL_EQ', 'argmax_hit'), get(ds, b, 'DS_SIGNED', 'pb_sla'), get(cb, b, 'LSML_CONT', 'pb_argmax_hit'), get(cb, b, 'IU_PCR', 'pb_argmax_hit')] + \
        [get(pbb, b, s, 'argmax_hit') for s in PIPE] + [get(lr, b, 'LR', 'pb_argmax_hit')]


T_PRMB_AUC = table(prmb_rows('within_auc', 'prmb_within_auc', 'prmb_within_auc', 'prmb_within_auc'), 'PRMBench · within-answer AUC (higher is better)',
                   'Bold = best label-free method in the row. LR = supervised logistic regression on the same channels, source-group cross-fit (headroom, not a competitor). Plain average = mean of the raw channels, no filter, no sign flip. DS-signed average = flip a channel when Dawid–Skene says it points the wrong way, then average.')
T_PRMB_TOP = table(prmb_rows('top1_on_error', 'prmb_top1_on_error', 'prmb_top1_on_error', 'prmb_top1_on_error'), 'PRMBench · top step is an error (wrong answers only)', 'Same methods and banks.')
T_PB = table(pb_rows, 'ProcessBench · SLA (top step = the first error, wrong answers only, 8-cell mean)',
             'SLA is Mind the Gap\'s metric. LR here is trained on PRMBench and applied to ProcessBench (transfer, no ProcessBench labels). DS-signed numbers from monday_ds_signed_v1.')

# reference rows
def ref(df, scorer, col):
    v = df[df.scorer == scorer][col]; return float(v.iloc[0]) if len(v) else np.nan
refs_pb = [('Qwen2.5-Math-PRM-7B, its top step', ref(pbb, 'Qwen2.5-Math-PRM-7B argmax', 'argmax_hit')),
           ('Qwen2.5-Math-PRM-7B, its own 0.5 rule', ref(pbb, 'Qwen2.5-Math-PRM-7B native (0.5)', 'argmax_hit')),
           ('Qwen2.5-72B critic', ref(pbb, 'Qwen2.5-72B critic native', 'argmax_hit')),
           ('Mind the Gap (our run)', ref(pbb, 'Mind the Gap (local repro.)', 'argmax_hit')),
           ('Mind the Gap (paper, Table 3)', 0.3927),
           ('Always the longest step', ref(pbb, 'Longest step', 'argmax_hit')), ('Always step 0', ref(pbb, 'Always step 0', 'argmax_hit'))]
refs_prmb = [(s, ref(pbp, s, 'within_auc'), ref(pbp, s, 'top1_on_error')) for s in ['Qwen2.5-Math-PRM-7B (supervised)', 'Mind the Gap (local repro.)', 'Step index (later = riskier)', 'Step token length']]

# PB F1 by rule, N16
f = f1[f1.bank == 'N16']
F1T = ['<div class="tw"><table><caption>ProcessBench official F1 on N16, by decision rule</caption><tr><th>Method</th><th>Rule</th><th>F1</th><th>First error found (wrong answers)</th><th>Clean answers left clean</th></tr>']
for _, r in f.sort_values(['rule', 'scorer']).iterrows():
    F1T.append(f'<tr><th>{r.scorer}</th><td>{r.rule}</td><td>{r.official_f1:.3f}</td><td>{r.error_hit:.3f}</td><td>{r.clean_correct:.3f}</td></tr>')
o = f1[f1.bank.isna() | ~f1.bank.isin(BANKS)]
for _, r in o.iterrows():
    F1T.append(f'<tr class="refrow"><th>{r.scorer}</th><td>{r.rule}</td><td>{r.official_f1:.3f}</td><td>{r.error_hit:.3f}</td><td>{r.clean_correct:.3f}</td></tr>')
F1T.append('</table></div><p class="note">R2_COUNT = flag as many steps as the answer has in the cell\'s top 20% of the raw channel mean; none means clean. ANSWER_Z_Q80 = top 20% of the cell. ENTROPY_GATE_Q80 = entropy gate. R2_COUNT is the best of the three registered rules, so it is mildly optimistic.</p>')
F1T = '\n'.join(F1T)

TY = ['<div class="tw"><table><caption>PRMBench error types · within-answer AUC vs Qwen2.5-Math-PRM-7B, over 80 label-free configurations (DS-signed)</caption><tr><th>Error type</th><th>Answers</th><th>Configs clearly above PRM</th><th>Clearly below</th><th>Median minus PRM</th><th>PRM AUC</th><th>Step index alone</th></tr>']
for _, r in ty.sort_values('median_minus_PRM', ascending=False).iterrows():
    cls = ' class="good"' if r.clearly_above_PRM >= 60 else (' class="bad"' if r.clearly_below_PRM >= 60 else '')
    TY.append(f'<tr{cls}><th>{r.type}</th><td>{r.n_answers}</td><td>{r.clearly_above_PRM}/{r.n_configs}</td><td>{r.clearly_below_PRM}/{r.n_configs}</td><td>{r.median_minus_PRM:+.3f}</td><td>{r.PRM_auc:.3f}</td><td>{r.step_index_auc:.3f}</td></tr>')
TY.append('</table></div>'); TY = '\n'.join(TY)

json.dump({'refs_pb': refs_pb, 'refs_prmb': refs_prmb}, open(Path(__file__).parent / 'refs.json', 'w'))
tpl = (Path(__file__).parent / 'template.html').read_text(encoding='utf8')
def reftab_pb():
    return '<div class="tw"><table><caption>ProcessBench references (SLA)</caption>' + ''.join(f'<tr><th>{n}</th><td>{v:.3f}</td></tr>' for n, v in refs_pb) + '</table></div>'
def reftab_prmb():
    return '<div class="tw"><table><caption>PRMBench references</caption><tr><th></th><th>Within AUC</th><th>Top step is an error</th></tr>' + ''.join(f'<tr><th>{n}</th><td>{a:.3f}</td><td>{b:.3f}</td></tr>' for n, a, b in refs_prmb) + '</table></div>'
for k, v in {'T_PB': T_PB, 'T_PRMB_AUC': T_PRMB_AUC, 'T_PRMB_TOP': T_PRMB_TOP, 'F1T': F1T, 'TY': TY, 'REF_PB': reftab_pb(), 'REF_PRMB': reftab_prmb()}.items():
    tpl = tpl.replace('{{' + k + '}}', v)
OUT.write_text(tpl, encoding='utf8'); print('ok', len(tpl))
