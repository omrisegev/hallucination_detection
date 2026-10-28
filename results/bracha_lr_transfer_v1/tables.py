"""RESULTS.csv, BOOTSTRAP.csv, WEIGHTS.csv -> TABLES_BODY.md (one table per benchmark; SUMMARY.md and TABLES.md add the text)."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = list(csv.DictReader(open(HERE / 'RESULTS.csv', encoding='utf-8')))
BOOT = {(r['benchmark'], r['prm']): r for r in csv.DictReader(open(HERE / 'BOOTSTRAP.csv', encoding='utf-8'))}
PRMS = [('qwen_prm_7b', 'Qwen-PRM-7B'), ('qwen_prm_72b', 'Qwen-PRM-72B'), ('pure_prm', 'PURE-PRM-7B'), ('rlhflow_prm', 'RLHFlow-PRM-8B')]
BENCH = [('prmbench', 'PRMBench', 'PRMScore'), ('socratic', 'Socratic-PRMBench', 'macro_f1'), ('hard2verify', 'Hard2Verify (177 answers)', 'balanced_f1'),
         ('processbench', 'ProcessBench', 'avg_f1'), ('crv_arith', 'CRV-Arithmetic', 'macro_f1'), ('crv_gsm8k', 'CRV-GSM8K', 'macro_f1'),
         ('deltabench', 'DeltaBench', 'macro_f1')]
V = {(r['benchmark'], r['telemetry'], r['prm'], r['arm'], r['metric']): float(r['value']) for r in R if r['value'] not in ('', 'nan')}
ARMS = [('PRM native', 'backbone', 'PRM, native threshold'), ('PRM', 'backbone', 'PRM, GAP rule'),
        ('LR_c_clean', 'backbone', 'LR on the PRM only (supervised, PRM800K)'),
        ('AUTO_N16', 'backbone', 'AUTO: PRM + backbone telemetry, L-SML (label-free)'),
        ('CHAIN_N16', 'backbone', 'CHAIN: PRM + backbone telemetry, L-SML (label-free)'),
        ('LR_a_clean', 'backbone', 'LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed)'),
        ('LR_a_all', 'backbone', 'LR: PRM + backbone telemetry (supervised, all PRM800K)'),
        ('TELE_N16', 'standalone', 'TELE16 standalone, L-SML (label-free)'),
        ('MEAN_N16', 'standalone', 'TELE16 standalone, plain average'),
        ('LR_b_clean', 'standalone', 'LR: standalone telemetry only (supervised, PRM800K)')]
f = lambda x, d=1: '—' if x is None else f'{x:.{d}f}'


def table(head, rows):
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


md = []
for b, name, m in BENCH:
    main = {'processbench': 'avg_f1_NC'}.get(b, f'{m}_GAP'); thr = f'{m}_THR05'; nat = {'processbench': 'avg_f1_native'}.get(b, f'{m}_native')
    rows = []
    for arm, tel, label in ARMS:
        if arm == 'PRM native':
            vals = [V.get((b, tel, p, arm, nat)) for p, _ in PRMS]; rule = 'native'
        else:
            vals = [V.get((b, tel, p, arm, main)) for p, _ in PRMS]; rule = 'NC' if b == 'processbench' else 'GAP'
        rows.append([label, rule] + [f(v) for v in vals] +
                    [f(V.get((b, tel, PRMS[0][0], arm, 'within_auc')), 3) + ' / ' + f(V.get((b, tel, PRMS[1][0], arm, 'within_auc')), 3) + ' / '
                     + f(V.get((b, tel, PRMS[2][0], arm, 'within_auc')), 3) + ' / ' + f(V.get((b, tel, PRMS[3][0], arm, 'within_auc')), 3)])
        if arm.startswith('LR') and b != 'processbench':
            rows.append([label + ', probability > 0.5', 'THR05 (label-trained)'] + [f(V.get((b, tel, p, arm, thr))) for p, _ in PRMS] + [''])
    md += [f'## {name}', '', table(['arm', 'rule'] + [pl for _, pl in PRMS] + ['within-AUC (4 PRMs)'], rows), '']
    extra = []
    for arm, tel, label in ARMS[1:]:
        if any(V.get((b, tel, p, arm, 'pooled_auroc')) is not None for p, _ in PRMS):
            extra.append([label, 'pooled step AUROC'] + [f(V.get((b, tel, p, arm, 'pooled_auroc')), 3) for p, _ in PRMS])
        ca = 'PRM (1-reward)' if (b == 'prmbench' and arm == 'PRM') else arm
        if any(V.get((b, tel, p, ca, 'crop_kept_a05')) is not None for p, _ in PRMS):
            extra.append([label, 'CROP kept % (α 0.05)'] + [f(V.get((b, tel, p, ca, 'crop_kept_a05'))) for p, _ in PRMS])
    md += [table(['arm', 'metric'] + [pl for _, pl in PRMS], extra), '']
    if b != 'processbench':
        md += ['LR (PRM + telemetry, overlap removed) minus AUTO, main metric, paired bootstrap 95% CI: ' +
               '; '.join(f"{pl} {float(BOOT[(b, p)]['delta']):+.2f} [{float(BOOT[(b, p)]['ci_lo']):+.2f}, {float(BOOT[(b, p)]['ci_hi']):+.2f}]"
                         for p, pl in PRMS) + f" (resampling {BOOT[(b, PRMS[0][0])]['grouping']}s).", '']
if __name__ == '__main__':
    (HERE / 'TABLES_BODY.md').write_text('\n'.join(md), encoding='utf-8')
    print(len(md))
