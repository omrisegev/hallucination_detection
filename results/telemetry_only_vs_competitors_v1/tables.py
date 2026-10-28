"""RESULTS.csv + the published numbers -> TABLES_BODY.md (one table per competitor item; README/TABLES.md add the text)."""
import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = list(csv.DictReader(open(HERE / 'RESULTS.csv', encoding='utf-8')))
PRMS = [('qwen_prm_7b', 'Qwen-PRM-7B', 'Qwen2.5-Math-7B-Instruct'), ('qwen_prm_72b', 'Qwen-PRM-72B', 'Qwen2.5-Math-72B-Instruct'),
        ('pure_prm', 'PURE-PRM-7B', 'Qwen2.5-Math-7B'), ('rlhflow_prm', 'RLHFlow-PRM-8B', 'Llama-3.1-8B-Instruct')]
ARM_LABEL = {'PRM native': 'PRM, native threshold', 'PRM (same rule)': 'PRM, same rule as telemetry',
             'TELE_N16': 'TELE16 (L-SML)', 'MEAN_N16': 'TELE16 plain average', 'PRM (1-reward)': 'PRM (1 − reward)'}
V = defaultdict(dict)
for r in R:
    k = (r['prm'], r['benchmark'], r['protocol'], r['arm'], r['pool'], r['rule'], r['scope'], r['grid'])
    V[k][r['metric']] = float(r['value'])


def get(prm, bench, proto, arm, metric, pool='', rule='', scope='', grid=''):
    return V[(prm, bench, proto, arm, pool, rule, scope, grid)].get(metric)


def f(x, d=1):
    return '—' if x is None else f'{x:.{d}f}'


def table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    out += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    return '\n'.join(out)


def who(prm_label, base, arm):
    return f'{base} → {prm_label}', ARM_LABEL.get(arm, arm)


md = []
# ------------------------------------------------------------------ PRISM Table 2
cols = ['FPR', 'FNR', 'Precision', 'TPR', 'TNR', 'PRMScore']
rows = [['published', 'Qwen-PRM-7B baseline (PRISM T2)', 'trained PRM', '69.34', '4.27', '89.40', '95.36', '30.66', '65.50'],
        ['published', 'PRISM, re-trained Qwen-PRM-7B (PRISM T2)', 'trained on PRM800K pairs', '47.13', '12.69', '91.93', '87.30', '52.86', '68.00']]
for prm, pl, base in PRMS:
    for arm, rule, acc in (('PRM native', 'native', 'trained PRM'), ('PRM (same rule)', 'GAP', 'trained PRM, label-free rule'),
                           ('TELE_N16', 'GAP', 'label-free'), ('MEAN_N16', 'GAP', 'label-free')):
        p, a = who(pl, base, arm)
        rows.append([p, a, acc] + [f(get(prm, 'prmbench', 'PRISM T2', arm, c, 'non-control', rule), 2) for c in cols])
md += ['## PRISM, Table 2: PRMBench error-detection rates', '',
       'Positive class = a correct step. FPR = error steps passed as correct. Pool: 6,211 non-control answers, 83,371 steps.', '',
       table(['pair (standalone → PRM)', 'arm', 'access'] + cols, rows), '']
# ------------------------------------------------------------------ PRISM matched TPR
rows = [['published', 'PRISM (its own operating point)', '47.13', '12.69', '91.93', '87.30', '52.86', '68.00']]
for prm, pl, base in PRMS:
    for arm in ('PRM native', 'TELE_N16', 'MEAN_N16'):
        p, a = who(pl, base, arm)
        rows.append([p, a] + [f(get(prm, 'prmbench', 'PRISM matched TPR 87.3', arm, c, rule='global threshold, labels'), 2) for c in cols])
md += ['## PRISM, same operating point: global threshold set (with labels) so TPR ≈ 87.30', '',
       'Same threshold access for every row; the PRM row moves the untouched PRM\'s threshold.', '',
       table(['pair', 'arm'] + cols, rows), '']
# ------------------------------------------------------------------ PRISM Table 7
c7 = ['Overall', 'NR', 'NCL', 'Simplicity', 'ES', 'SC', 'DC', 'CI', 'Soundness', 'PS', 'DR', 'MS', 'Sensitivity']
T7 = {'Qwen-PRM-7B baseline (PRISM T7)': [65.5, 49.1, 55.0, 52.1, 71.7, 67.4, 66.3, 78.5, 71.0, 57.7, 69.1, 99.7, 75.5],
      'PRISM 132k curriculum-4 (PRISM T7)': [68.0, 53.5, 66.0, 59.7, 73.8, 67.5, 67.1, 75.6, 71.0, 62.4, 70.1, 98.7, 77.1],
      'PRISM 132k curriculum-3 (PRISM T7)': [68.0, 52.3, 63.7, 58.0, 73.9, 68.5, 67.4, 77.4, 71.8, 61.1, 70.6, 99.3, 77.0]}
rows = [['published', k] + [f(x) for x in v] for k, v in T7.items()]
for prm, pl, base in PRMS:
    for arm, rule in (('PRM native', 'native'), ('PRM (same rule)', 'GAP'), ('TELE_N16', 'GAP'), ('MEAN_N16', 'GAP')):
        p, a = who(pl, base, arm)
        rows.append([p, a] + [f(get(prm, 'prmbench', 'PRISM T7', arm, c, rule=rule)) for c in c7])
md += ['## PRISM, Table 7: PRMBench PRMScore by error category', '',
       'NR redundancy, NCL circular logic, ES empirical soundness, SC step consistency, DC domain consistency, CI confidence '
       'invariance, PS prerequisite sensitivity, DR deception resistance, MS multi-solution (no error steps: valid-step F1). '
       'NR and NCL go through the validity path (no redundancy head), as in every earlier table.', '',
       table(['pair', 'arm'] + c7, rows), '']
# ------------------------------------------------------------------ PRISM ProcessBench Table 6
subs = [('gsm8k', 'GSM8K'), ('math', 'MATH'), ('omnimath', 'Omni-MATH'), ('olympiadbench', 'OlympiadBench')]
pub = {'Qwen-PRM-7B baseline': [71.0, 96.4, 81.8, 67.0, 91.1, 77.2, 54.8, 66.1, 83.4, 54.6, 85.5, 66.7],
       'PRISM CL1': [72.0, 94.8, 81.8, 69.4, 86.9, 77.2, 55.6, 66.2, 81.7, 58.2, 79.6, 67.3],
       'PRISM CL2': [73.4, 94.3, 82.6, 70.5, 85.0, 77.1, 58.2, 65.8, 75.5, 58.9, 75.5, 66.1],
       'PRISM CL3': [73.4, 93.8, 82.4, 72.6, 82.0, 77.0, 57.8, 64.8, 72.6, 58.5, 72.6, 64.8]}
head = ['pair', 'arm', 'rule'] + [f'{n} {m}' for _, n in subs for m in ('TPR', 'TNR', 'F1')] + ['avg F1']
rows = []
for k, v in pub.items():
    rows.append(['published', f'{k} (PRISM T6)', 'native'] + [f(x) for x in v] + [f(sum(v[2::3]) / 4)])
for prm, pl, base in PRMS:
    for arm, rule, rl in (('PRM native', 'native', 'native'), ('PRM (same rule)', 'Q80', 'Q80'),
                          ('TELE_N16', 'Q80', 'Q80 (telemetry only)'), ('MEAN_N16', 'Q80', 'Q80 (telemetry only)'),
                          ('TELE_N16', 'NC', 'NC (PRM gate + count)'), ('MEAN_N16', 'NC', 'NC (PRM gate + count)')):
        p, a = who(pl, base, arm)
        rows.append([p, a, rl] + [f(get(prm, 'processbench', 'PRISM PB T6', arm, f'{s} {m}', rule=rule)) for s, _ in subs for m in ('TPR', 'TNR', 'F1')]
                    + [f(get(prm, 'processbench', 'PRISM PB T6', arm, 'average F1', rule=rule))])
md += ['## PRISM, Table 6 (appendix 10.2): ProcessBench', '',
       'TPR = accuracy on samples with an error (exact first-error step), TNR = accuracy on error-free samples, F1 = their '
       'harmonic mean. PRISM prints Omni-MATH and OlympiadBench cells that do not fit F1 = harmonic mean (Omni-MATH '
       '54.8/66.1 → 83.4); they are copied as printed. Q80 is pure telemetry (first flagged step, or "no error" when '
       'nothing is flagged). NC uses the PRM\'s native gate and error count with the arm\'s order, so it is NOT telemetry only.', '',
       table(head, rows), '']
# ------------------------------------------------------------------ SLIDER
rows = [['published', 'Step-RRI (SLIDER Fig. 7)', '61.91', '—', '—', '—', '—'],
        ['published', 'InfoGain, QwQ-32B (implied: 61.91 − 13.53)', '48.38', '—', '—', '—', '—'],
        ['published', 'Cosine, MiniLM (implied: 61.91 − 10.31)', '51.60', '—', '—', '—', '—']]
maj = get('qwen_prm_7b', 'prmbench', 'SLIDER', 'majority class', 'acc', rule='always not redundant', scope='all_steps')
rows.append(['ours', 'always "not redundant" (majority)', f(maj, 2), '—', '—', '—', '50.00'])
for prm, pl, base in PRMS:
    for arm, rule in (('PRM native', 'native'), ('PRM (same rule)', 'GAP'), ('TELE_N16', 'GAP'), ('MEAN_N16', 'GAP')):
        p, a = who(pl, base, arm); g = lambda m, r: get(prm, 'prmbench', 'SLIDER', arm, m, rule=r, scope='all_steps')
        rows.append([p, a, f(g('acc', 'LR threshold balanced'), 2), f(g('bal_acc', 'LR threshold balanced'), 2),
                     f(g('acc', 'LR threshold unweighted'), 2), f(g('acc', f'label-free {rule}'), 2), f(g('bal_acc', f'label-free {rule}'), 2)])
md += ['## SLIDER / Step-RRI: PRMBench redundancy class (758 answers, positive = redundant step)', '',
       'SLIDER calibrates a threshold with a linear classifier on a held-out labelled set and reports "accuracy"; split, '
       'class weighting and step-1 handling are not stated. Our reading: 10 trace-level 50/50 splits, 1-D logistic regression '
       'on the score. About 87% of steps are not redundant, so plain accuracy rewards "never flag" (majority row). '
       'steps ≥ 2 variants are in RESULTS.csv.', '',
       table(['pair', 'arm', 'acc, LR threshold (balanced)', 'balanced acc, LR (balanced)', 'acc, LR threshold (unweighted)',
              'acc, label-free rule', 'balanced acc, label-free rule'], rows), '']
# ------------------------------------------------------------------ CROP Table 6
C6 = {'prmbench': [('Random', .4994, 8.7, 4.7), ('Token/format LR', .7751, 16.2, 5.0), ('Trace features LR (Llama-3.1-8B)', .7812, 15.9, 4.6),
                   ('Direct PRM (Qwen-PRM-7B)', .8010, 4.8, 0.6), ('Trace+PRM LR', .8336, 21.8, 5.0)],
      'processbench': [('Random', .4887, 9.8, 4.9), ('Token/format LR', .7057, 10.6, 4.8), ('Trace features LR (Llama-3.1-8B)', .7058, 11.3, 4.5),
                       ('Direct PRM (Qwen-PRM-7B)', .8801, 46.7, 4.5), ('Trace+PRM LR', .8935, 42.4, 4.5)],
      'crv_arith': [('Random', .5049, 62.0, 4.6), ('Token/format LR', .8168, 77.5, 4.7), ('Trace features LR (Llama-3.1-8B)', .8564, 90.3, 4.5),
                    ('Direct PRM (Qwen-PRM-7B)', .9421, 97.5, 4.2), ('Trace+PRM LR', .9701, 97.1, 4.1)],
      'crv_gsm8k': [('Random', .5046, 27.2, 4.3), ('Token/format LR', .7081, 39.8, 4.9), ('Trace features LR (Llama-3.1-8B)', .7495, 43.2, 4.3),
                    ('Direct PRM (Qwen-PRM-7B)', .8659, 81.7, 5.1), ('Trace+PRM LR', .8812, 63.1, 4.5)],
      'prm800k': [('Random', .5002, 20.5, 5.4), ('Token/format LR', .9919, 89.0, 4.8), ('Trace features LR (Llama-3.1-8B)', .9925, 89.4, 4.8),
                  ('Direct PRM (Qwen-PRM-7B)', .9327, 67.2, 4.5), ('Trace+PRM LR', .9934, 89.5, 4.6)]}
BN = {'prmbench': 'PRMBench', 'processbench': 'ProcessBench', 'crv_arith': 'CRV-Arithmetic', 'crv_gsm8k': 'CRV-GSM8K', 'prm800k': 'PRM800K'}
md += ['## CROP, Table 6: conformal clean prefix at α = 0.05', '',
       'Step AUROC pooled over test steps; kept = mean retained-prefix fraction; risk = share of test traces whose kept prefix '
       'contains an error (target ≤ 5%). 10 splits 60/20/20, seeds 2806–2815. CROP calibrates every row\'s threshold with '
       'labels on the 20% calibration split; their LR rows are also TRAINED on the 60% split. Our rows use no training split.', '']
for b, pubs in C6.items():
    rows = [['published', f'{n} (CROP T6)', f(a, 4), f(k), f(r)] for n, a, k, r in pubs]
    for prm, pl, base in PRMS:
        for arm, grid in (('PRM (1-reward)', 'fixed101'), ('PRM (1-reward)', 'q201'), ('TELE_N16', 'q201'), ('MEAN_N16', 'q201')):
            p, a = who(pl, base, arm); g = lambda m: get(prm, b, 'CROP T6', arm, m, grid=grid)
            gl = 'CROP fixed 101 grid' if grid == 'fixed101' else '201-quantile grid'
            rows.append([p, f'{a}, {gl}', f(g('auroc'), 4), f(g('kept')), f(g('risk'))])
    note = {'prmbench': 'Our PRMBench uses all 6,211 non-control answers (CROP sampled 5,000), split assignment re-implemented.',
            'processbench': 'Same 3,400 traces / 25,697 steps as CROP Table 4; labels mark the first-error step only '
                            '(8.6% of steps here; CROP lists 6.6%, not explained).',
            'crv_arith': 'Inputs rebuilt with CROP\'s selection rules (counts match Table 4). CROP used 20 seeds on CRV; we use 10.',
            'crv_gsm8k': 'As CRV-Arithmetic.', 'prm800k': 'Inputs rebuilt with CROP\'s selection rules (counts match Table 4).'}[b]
    md += [f'### {BN[b]}', '', note, '', table(['pair', 'arm', 'step AUROC', 'kept %', 'risk %'], rows), '']
# ------------------------------------------------------------------ CROP Table 17
rows = [['published', 'CROP (CROP T17)', 'CRV-Arith → CRV-GSM8K', '14.8', '69.9', '13.3', '62.5'],
        ['published', 'CROP (CROP T17)', 'CRV-GSM8K → CRV-Arith', '5.1', '69.0', '4.4', '54.7']]
for d, dl in (('crv_arith->crv_gsm8k', 'CRV-Arith → CRV-GSM8K'), ('crv_gsm8k->crv_arith', 'CRV-GSM8K → CRV-Arith')):
    for prm, pl, base in PRMS:
        for arm, grid in (('PRM (1-reward)', 'fixed101'), ('PRM (1-reward)', 'q201'), ('TELE_N16', 'q201'), ('MEAN_N16', 'q201')):
            p, a = who(pl, base, arm); g = lambda m: get(prm, d, 'CROP T17', arm, m, grid=grid)
            rows.append([p, f'{a}, {"fixed grid" if grid == "fixed101" else "q201"}', dl, f(g('risk')), f(g('kept')), f(g('FA')), f(g('accept'))])
md += ['## CROP, Table 17: cross-domain transfer (calibrate on one CRV set, test on the other)', '',
       'Exchangeability is broken on purpose; CROP presents this as a failure diagnostic. The paper does not name the proxy '
       'in this table. FA = whole traces accepted that contain an error; accept = whole traces accepted (our reading of '
       'the columns; % of target test traces).', '',
       table(['pair', 'arm', 'calibrate → test', 'risk %', 'kept %', 'FA %', 'accept %'], rows), '']
(HERE / 'TABLES_BODY.md').write_text('\n'.join(md), encoding='utf-8')
print(len(md))
