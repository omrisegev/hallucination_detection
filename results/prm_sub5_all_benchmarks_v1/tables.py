"""SUB5_ALL.csv -> TABLES.md: one table per benchmark, mode and metric (rows = arms, columns = the 4 PRMs)."""
import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
ARMS = ['PRM native', 'PRM', 'PRM (1-reward)', 'TELE_N16', 'MEAN_N16', 'CHAIN_N16', 'SUB5 alone', 'SUB5 +PRM']
METRICS = ['PRMScore_GAP', 'PRMScore_sameTPR87.3', 'PRMScore_native', 'avg_f1_NC', 'avg_f1_native', 'balanced_f1_GAP',
           'balanced_f1_native', 'macro_f1_GAP', 'macro_f1_native', 'within_auc', 'crop_kept_a05', 'crop_auroc']
ORDER = ['prmbench', 'socratic', 'hard2verify', 'processbench', 'crv_arith', 'crv_gsm8k', 'prm800k', 'deltabench']


def fmt(m, v):
    return f'{v:.3f}' if m in ('within_auc', 'crop_auroc', 'pooled_auroc') else f'{v:.2f}'


def main():
    T = defaultdict(dict)
    for r in csv.DictReader(open(HERE / 'SUB5_ALL.csv', encoding='utf-8')):
        if r['value'] in ('', 'nan', 'None'): continue
        T[(r['benchmark'], r['mode'], r['metric'])][(r['arm'], r['prm'])] = float(r['value'])
    out = ['# SUB5 on every benchmark (development data; see README for what is in-sample)', '']
    for b in ORDER:
        for mode in ('backbone', 'standalone'):
            for m in METRICS:
                d = T.get((b, mode, m))
                if not d: continue
                out += [f'### {b} / {mode} / {m}', '', '| arm | ' + ' | '.join(PRMS) + ' |', '|---|' + '---|' * len(PRMS)]
                for a in ARMS:
                    if any((a, p) in d for p in PRMS):
                        out.append(f'| {a} | ' + ' | '.join(fmt(m, d[(a, p)]) if (a, p) in d else '-' for p in PRMS) + ' |')
                out.append('')
    (HERE / 'TABLES.md').write_text('\n'.join(out), encoding='utf-8')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
