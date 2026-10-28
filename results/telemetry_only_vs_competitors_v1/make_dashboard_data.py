"""RESULTS.csv + the published numbers (tables.py) -> dashboard/competitor_protocols.csv, the scorecard's
"Telemetry only vs PRISM, SLIDER and CROP" dataset. One row per (table, sub-table, metric, pair); published values
sit on every row of their table with their paper and table. A few 'note' rows carry the numbers the captions quote.
'auto' is AUTO (the PRM as one of 17 channels with the PRM-backbone telemetry, free L-SML grouping): it uses the PRM,
unlike tele/avg, which are the standalone base model's telemetry only."""
import csv
from pathlib import Path
import tables as T

HERE = Path(__file__).resolve().parent
OUT = HERE / 'dashboard'; OUT.mkdir(exist_ok=True)
get, PRMS = T.get, T.PRMS
PRISM_T2 = 'PRISM, arXiv 2606.09078, Table 2'
rows = []


def row(section, sub, metric, order, pair, dec, **v):
    rows.append(dict(key='|'.join([section, sub, metric, str(order)]), section=section, sub=sub, metric=metric,
                     pair_order=order, pair=pair, decimals=dec, **v))


def pairs():
    for i, (prm, pl, base) in enumerate(PRMS):
        yield i, prm, f'{base} → {pl}'


# PRISM Table 2 and the matched-TPR operating point
T2PUB = {'PRISM': dict(FPR=47.13, FNR=12.69, Precision=91.93, TPR=87.30, TNR=52.86, PRMScore=68.00),
         'base': dict(FPR=69.34, FNR=4.27, Precision=89.40, TPR=95.36, TNR=30.66, PRMScore=65.50)}
for m in ('PRMScore', 'FPR', 'FNR', 'Precision', 'TPR', 'TNR'):
    for i, prm, pair in pairs():
        g = lambda arm, rule: get(prm, 'prmbench', 'PRISM T2', arm, m, 'non-control', rule)
        row('prism_t2', '', m, i, pair, 2, tele=g('TELE_N16', 'GAP'), avg=g('MEAN_N16', 'GAP'), auto=g('AUTO_N16', 'GAP'), prm=g('PRM native', 'native'),
            prm2=g('PRM (same rule)', 'GAP'), pub_value=T2PUB['PRISM'][m], pub_label='PRISM, re-trained Qwen-PRM-7B (' + PRISM_T2 + ')',
            pub2_value=T2PUB['base'][m], pub2_label='Qwen-PRM-7B as printed by PRISM (' + PRISM_T2 + ')')
        h = lambda arm: get(prm, 'prmbench', 'PRISM matched TPR 87.3', arm, m, rule='global threshold, labels')
        row('prism_mtpr', '', m, i, pair, 2, tele=h('TELE_N16'), avg=h('MEAN_N16'), auto=h('AUTO_N16'), prm=h('PRM native'),
            pub_value=T2PUB['PRISM'][m], pub_label='PRISM at its own operating point (' + PRISM_T2 + ')')
# PRISM Table 7
C7 = ['Overall', 'NR', 'NCL', 'Simplicity', 'ES', 'SC', 'DC', 'CI', 'Soundness', 'PS', 'DR', 'MS', 'Sensitivity']
t7 = list(T.T7.values())
for j, m in enumerate(C7):
    for i, prm, pair in pairs():
        g = lambda arm, rule: get(prm, 'prmbench', 'PRISM T7', arm, m, rule=rule)
        row('prism_t7', '', m, i, pair, 1, tele=g('TELE_N16', 'GAP'), avg=g('MEAN_N16', 'GAP'), auto=g('AUTO_N16', 'GAP'), prm=g('PRM native', 'native'),
            prm2=g('PRM (same rule)', 'GAP'), pub_value=t7[1][j], pub_label='PRISM 132k curriculum-4 (PRISM, arXiv 2606.09078, Table 7)',
            pub2_value=t7[2][j], pub2_label='PRISM 132k curriculum-3 (PRISM, arXiv 2606.09078, Table 7)')
# PRISM ProcessBench Table 6
pbm = [f'{s} {k}' for s, _ in T.subs for k in ('TPR', 'TNR', 'F1')] + ['average F1']
pubpb = {k: v + [sum(v[2::3]) / 4] for k, v in T.pub.items()}
for j, m in enumerate(pbm):
    for i, prm, pair in pairs():
        g = lambda arm, rule: get(prm, 'processbench', 'PRISM PB T6', arm, m, rule=rule)
        row('prism_pb', '', m, i, pair, 1, tele=g('TELE_N16', 'Q80'), avg=g('MEAN_N16', 'Q80'), x1=g('TELE_N16', 'NC'),
            x2=g('MEAN_N16', 'NC'), auto=g('AUTO_N16', 'Q80'), auto2=g('AUTO_N16', 'NC'), prm=g('PRM native', 'native'),
            pub_value=pubpb['Qwen-PRM-7B baseline'][j], pub_label='Qwen-PRM-7B as printed by PRISM (PRISM, arXiv 2606.09078, Table 6)',
            pub2_value=pubpb['PRISM CL1'][j], pub2_label='PRISM CL1, its best average (PRISM, arXiv 2606.09078, Table 6)')
# SLIDER
SL = [('acc_lr', 'acc', 'LR threshold balanced', 61.91), ('balacc_lr', 'bal_acc', 'LR threshold balanced', None),
      ('acc_rule', 'acc', 'label-free {r}', None), ('balacc_rule', 'bal_acc', 'label-free {r}', None)]
for name, met, rule, pubv in SL:
    for i, prm, pair in pairs():
        g = lambda arm, r: get(prm, 'prmbench', 'SLIDER', arm, met, rule=rule.format(r=r), scope='all_steps')
        row('slider', '', name, i, pair, 1, tele=g('TELE_N16', 'GAP'), avg=g('MEAN_N16', 'GAP'), auto=g('AUTO_N16', 'GAP'), prm=g('PRM native', 'native'),
            prm2=g('PRM (same rule)', 'GAP'), pub_value=pubv,
            pub_label='Step-RRI (SLIDER, arXiv 2610.00571, Fig. 7)' if pubv else 'not reported by SLIDER')
# CROP Table 6 and Table 17
C6P = {b: {n.split(' (')[0].replace(' LR', ''): (a, k, r) for n, a, k, r in v} for b, v in T.C6.items()}
for b in T.C6:
    for m, idx in (('kept', 1), ('auroc', 0), ('risk', 2)):
        for i, prm, pair in pairs():
            g = lambda arm, grid: get(prm, b, 'CROP T6', arm, m, grid=grid)
            P = C6P[b]
            row('crop_t6', b, m, i, pair, 3 if m == 'auroc' else 1, tele=g('TELE_N16', 'q201'), avg=g('MEAN_N16', 'q201'), auto=g('AUTO_N16', 'q201'),
                prm=g('PRM (1-reward)', 'fixed101'), prm2=g('PRM (1-reward)', 'q201'),
                pub_value=P['Trace+PRM'][idx], pub_label='Trace+PRM LR, supervised (CROP, arXiv 2605.30085, Table 6)',
                pub2_value=P['Direct PRM'][idx], pub2_label='Direct PRM, Qwen-PRM-7B (CROP, arXiv 2605.30085, Table 6)',
                pub3_value=P['Trace features'][idx], pub3_label='Trace features LR, Llama-3.1-8B (CROP, arXiv 2605.30085, Table 6)')
T17 = {'crv_arith->crv_gsm8k': dict(risk=14.8, kept=69.9, FA=13.3, accept=62.5), 'crv_gsm8k->crv_arith': dict(risk=5.1, kept=69.0, FA=4.4, accept=54.7)}
for d, pv in T17.items():
    for m in ('risk', 'kept', 'FA', 'accept'):
        for i, prm, pair in pairs():
            g = lambda arm, grid: get(prm, d, 'CROP T17', arm, m, grid=grid)
            row('crop_t17', d, m, i, pair, 1, tele=g('TELE_N16', 'q201'), avg=g('MEAN_N16', 'q201'), auto=g('AUTO_N16', 'q201'), prm=g('PRM (1-reward)', 'fixed101'),
                prm2=g('PRM (1-reward)', 'q201'), pub_value=pv[m], pub_label='CROP, proxy not named (CROP, arXiv 2605.30085, Table 17)')
# numbers quoted in the captions
cos = next(csv.DictReader(open(HERE.parent / 'slider_cosine_bridge_v1/COSINE_BRIDGE.csv', encoding='utf-8')))
row('note', '', 'slider_cosine', 0, '', 1, tele=float(cos['all_steps__LR_balanced__acc']), pub_value=51.60,
    pub_label='Cosine, MiniLM (SLIDER, arXiv 2610.00571, implied by Fig. 7 and its text)')
row('note', '', 'slider_majority', 0, '', 1, tele=get('qwen_prm_7b', 'prmbench', 'SLIDER', 'majority class', 'acc',
                                                     rule='always not redundant', scope='all_steps'))
cols = ['key', 'section', 'sub', 'metric', 'pair_order', 'pair', 'decimals', 'tele', 'avg', 'x1', 'x2', 'auto', 'auto2', 'prm', 'prm2',
        'pub_value', 'pub_label', 'pub2_value', 'pub2_label', 'pub3_value', 'pub3_label']
with open(OUT / 'competitor_protocols.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, cols); w.writeheader()
    for r in rows:
        w.writerow({k: ('' if r.get(k) is None else round(r[k], 6) if isinstance(r.get(k), float) else r.get(k, '')) for k in cols})
missing = [r['key'] for r in rows if r['section'] != 'note' and (r.get('tele') is None or r.get('prm') is None or r.get('auto') is None)]
print(len(rows), 'rows; missing', missing[:5])
