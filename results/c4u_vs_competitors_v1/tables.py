"""RESULTS.csv (C4U) + RESULTS_BACKBONE.csv (PRM, AUTO; same protocol code) + published numbers -> TABLES.md, SUMMARY.csv."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
BB = HERE.parent / 'telemetry_only_vs_competitors_v1/RESULTS_BACKBONE.csv'
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
NAME = dict(qwen_prm_7b='Qwen-PRM-7B', qwen_prm_72b='Qwen-PRM-72B', pure_prm='PURE', rlhflow_prm='RLHFlow')
R = list(csv.DictReader(open(HERE / 'RESULTS.csv'))) + [r for r in csv.DictReader(open(BB)) if r['arm'] == 'AUTO_N16']
# backbone TELE16, GAP first flag (Step 483, results/processbench_gap_sla_v1 via standalone-gap HANDOFF)
TELE_SLA = dict(qwen_prm_7b=47.23, qwen_prm_72b=50.64, pure_prm=33.21, rlhflow_prm=32.52)


def v(prm, bench, proto, arm, met, **kw):
    x = [float(r['value']) for r in R if r['prm'] == prm and r['benchmark'] == bench and r['protocol'] == proto
         and r['arm'] == arm and r['metric'] == met and all(r.get(k, '') == w for k, w in kw.items())]
    assert len(x) == 1, (prm, bench, proto, arm, met, kw, x)
    return x[0]


def f(x, d=1):
    return f'{x:.{d}f}'


out, summ = [], []
W = out.append


def table(title, head, rows, notes=()):
    W(f'\n## {title}\n'); W('| ' + ' | '.join(head) + ' |'); W('|' + '---|' * len(head))
    for r in rows: W('| ' + ' | '.join(r) + ' |')
    for n in notes: W(f'- {n}')


W('# C4U (frozen, C4U-parisi-within-jaffe-cross-v1) vs PRISM, CROP, SLIDER, Mind the Gap\n')
W('Input R1 in every cell (raw banks from AIRCC, see README). PRM and AUTO rows: RESULTS_BACKBONE.csv of '
  'telemetry_only_vs_competitors_v1 (same code; PRM rows re-computed here and parity-gated). '
  'C4U flags = its own DS-posterior count; threshold protocols use the C4U rank.')

# PRISM
rows = []
for p in PRMS:
    a = dict(pool='non-control', rule='GAP')
    rows.append([NAME[p], f(v(p, 'prmbench', 'PRISM T2', 'PRM native', 'PRMScore', pool='non-control', rule='native'), 2),
                 f(v(p, 'prmbench', 'PRISM T2', 'AUTO_N16', 'PRMScore', **a), 2), f(v(p, 'prmbench', 'PRISM T2', 'C4U', 'PRMScore', **a), 2),
                 f(v(p, 'prmbench', 'PRISM T2', 'C4U', 'TPR', **a)), f(v(p, 'prmbench', 'PRISM T2', 'C4U', 'FPR', **a)),
                 f(v(p, 'prmbench', 'PRISM matched TPR 87.3', 'C4U', 'PRMScore'), 2),
                 f(v(p, 'prmbench', 'PRISM matched TPR 87.3', 'AUTO_N16', 'PRMScore'), 2),
                 f(v(p, 'prmbench', 'PRISM matched TPR 87.3', 'PRM native', 'PRMScore'), 2)])
    summ.append(dict(protocol='PRISM PRMBench PRMScore', prm=p, PRM=rows[-1][1], AUTO=rows[-1][2], C4U=rows[-1][3], paper='68.00 (PRISM, Qwen-PRM-7B)'))
table('PRISM: PRMBench PRMScore (Table 2)', ['PRM', 'PRM own thr', 'AUTO', 'C4U', 'C4U TPR', 'C4U FPR',
                                               'C4U @TPR 87.3', 'AUTO @TPR 87.3', 'PRM @TPR 87.3'], rows,
      ['PRISM paper (retrained Qwen-PRM-7B): PRMScore 68.00 at TPR 87.30 / FPR 47.13. Baseline Qwen-PRM-7B 65.50.',
       '"@TPR 87.3" = global threshold chosen with labels to match PRISM\'s TPR (fairness check, not a method).'])

# SLIDER
rows = []
for p in PRMS:
    k = dict(scope='all_steps', rule='LR threshold balanced')
    rows.append([NAME[p], f(v(p, 'prmbench', 'SLIDER', 'PRM native', 'acc', **k), 2), f(v(p, 'prmbench', 'SLIDER', 'AUTO_N16', 'acc', **k), 2),
                 f(v(p, 'prmbench', 'SLIDER', 'C4U', 'acc', **k), 2),
                 f(v(p, 'prmbench', 'SLIDER', 'C4U', 'bal_acc', scope='all_steps', rule='label-free GAP'), 2),
                 f(v(p, 'prmbench', 'SLIDER', 'AUTO_N16', 'bal_acc', scope='all_steps', rule='label-free GAP'), 2)])
    summ.append(dict(protocol='SLIDER redundancy acc (LR balanced)', prm=p, PRM=rows[-1][1], AUTO=rows[-1][2], C4U=rows[-1][3], paper='61.91 (Step-RRI)'))
table('SLIDER: PRMBench redundancy class', ['PRM', 'PRM acc', 'AUTO acc', 'C4U acc', 'C4U bal-acc (own flags)', 'AUTO bal-acc (GAP)'], rows,
      ['acc = plain accuracy, class-balanced 1-D LR threshold, 10 trace-level 50/50 splits (SLIDER protocol). Paper Step-RRI 61.91.',
       'Majority class (always "not redundant") = 86.8 plain accuracy; bal-acc is the informative number.'])

# CROP T6
for bench, title, paper in (('prmbench', 'PRMBench', 'Trace+PRM LR 21.8, Direct PRM 4.8 (fixed grid), random 8.7'),
                            ('processbench', 'ProcessBench', 'Trace+PRM LR 42.4, Direct PRM 46.7, random 9.8'),
                            ('crv_arith', 'CRV-Arith', 'Trace+PRM LR 97.1, Direct PRM 97.5, random 62.0'),
                            ('crv_gsm8k', 'CRV-GSM8K', 'Trace+PRM LR 63.1, Direct PRM 81.7, random 27.2'),
                            ('prm800k', 'PRM800K', 'Trace+PRM LR 89.5, Direct PRM 67.2, random 20.5')):
    rows = []
    for p in PRMS:
        g = dict(grid='q201')
        rows.append([NAME[p], f(v(p, bench, 'CROP T6', 'PRM (1-reward)', 'kept', **g)), f(v(p, bench, 'CROP T6', 'AUTO_N16', 'kept', **g)),
                     f(v(p, bench, 'CROP T6', 'C4U', 'kept', **g)), f(v(p, bench, 'CROP T6', 'C4U', 'risk', **g)),
                     f(v(p, bench, 'CROP T6', 'PRM (1-reward)', 'auroc', **g), 3), f(v(p, bench, 'CROP T6', 'AUTO_N16', 'auroc', **g), 3),
                     f(v(p, bench, 'CROP T6', 'C4U', 'auroc', **g), 3)])
        summ.append(dict(protocol=f'CROP T6 kept% {title}', prm=p, PRM=rows[-1][1], AUTO=rows[-1][2], C4U=rows[-1][3], paper=paper))
    table(f'CROP Table 6, alpha 0.05: {title}', ['PRM', 'PRM kept', 'AUTO kept', 'C4U kept', 'C4U risk', 'PRM AUROC', 'AUTO AUROC', 'C4U AUROC'], rows,
          [f'CROP paper (Qwen-PRM-7B): {paper}. 201-quantile grid for every row; threshold calibrated with labels for every row.'])

# CROP T17
rows = []
for p in PRMS:
    for d in ('crv_arith->crv_gsm8k', 'crv_gsm8k->crv_arith'):
        g = dict(grid='q201')
        rows.append([NAME[p], d, f(v(p, d, 'CROP T17', 'PRM (1-reward)', 'risk', **g)) + ' / ' + f(v(p, d, 'CROP T17', 'PRM (1-reward)', 'kept', **g)),
                     f(v(p, d, 'CROP T17', 'AUTO_N16', 'risk', **g)) + ' / ' + f(v(p, d, 'CROP T17', 'AUTO_N16', 'kept', **g)),
                     f(v(p, d, 'CROP T17', 'C4U', 'risk', **g)) + ' / ' + f(v(p, d, 'CROP T17', 'C4U', 'kept', **g))])
table('CROP Table 17: cross-domain (risk % / kept %)', ['PRM', 'direction', 'PRM', 'AUTO', 'C4U'], rows,
      ['CROP paper: Arith->GSM8K risk 14.8 / kept 69.9; GSM8K->Arith 5.1 / 69.0. Target risk 5%.'])

# ProcessBench
rows = []
for p in PRMS:
    lo, hi = v(p, 'processbench', 'MtG T3 SLA', 'C4U', 'macro SLA lo'), v(p, 'processbench', 'MtG T3 SLA', 'C4U', 'macro SLA hi')
    rows.append([NAME[p], f(v(p, 'processbench', 'MtG T3 SLA', 'PRM native', 'macro SLA'), 2), f(TELE_SLA[p], 2),
                 f(v(p, 'processbench', 'MtG T3 SLA', 'C4U', 'macro SLA'), 2) + f' [{lo:.1f}, {hi:.1f}]',
                 f(v(p, 'processbench', 'MtG T3 SLA', 'C4U', 'overall SLA'), 2), 'yes' if lo > 39.39 else ('no, below' if hi < 39.39 else 'tie')])
    summ.append(dict(protocol='Mind the Gap ProcessBench macro SLA', prm=p, PRM=rows[-1][1], AUTO='', C4U=f(v(p, 'processbench', 'MtG T3 SLA', 'C4U', 'macro SLA'), 2),
                     paper='39.39 (MtG Qwen3-8B, no PRM)'))
table('Mind the Gap Table 3: ProcessBench SLA (2,221 error answers)', ['PRM', 'PRM own thr', 'backbone TELE16 GAP', 'C4U macro [95% CI]', 'C4U overall', 'C4U > MtG 39.39'], rows,
      ['SLA = exact match of the first error step; prediction = first flagged step. MtG (Shannon Drop, Qwen3-8B, no PRM): macro 39.39 / overall 38.11.',
       'CI: 2,000 answer resamples stratified by subset, no refit.'])
rows = []
for p in PRMS:
    rows.append([NAME[p], f(v(p, 'processbench', 'PRISM PB T6', 'PRM native', 'average F1', rule='native', input='R1'), 2),
                 f(v(p, 'processbench', 'PRISM PB T6', 'AUTO_N16', 'average F1', rule='NC'), 2),
                 f(v(p, 'processbench', 'PRISM PB T6', 'C4U', 'average F1', rule='C4U own count'), 2),
                 f(v(p, 'processbench', 'PRISM PB T6', 'C4U', 'gsm8k TNR', rule='C4U own count'), 1)])
    summ.append(dict(protocol='PRISM ProcessBench avg F1', prm=p, PRM=rows[-1][1], AUTO=rows[-1][2] + ' (NC gate)', C4U=rows[-1][3], paper=''))
table('PRISM Table 6: ProcessBench average F1', ['PRM', 'PRM own thr', 'AUTO (PRM gate, NC)', 'C4U (own count)', 'C4U correct-answer acc, GSM8K'], rows,
      ['C4U flags at least one step in almost every answer, so its accuracy on correct answers is near 0 and F1 collapses. AUTO borrows the PRM gate (NC rule) for this metric.'])
W('\n## ARES\n- Not run. ARES was dropped from deck v2 (Omri, 2026-10-29); no ARES protocol cell was asked here.')
(HERE / 'TABLES.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
with open(HERE / 'SUMMARY.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
print('\n'.join(out))
