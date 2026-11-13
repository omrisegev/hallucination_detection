import csv, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
OUT = Path(__file__).resolve().parent
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
MET = {'prmbench': 'PRMScore', 'socratic': 'PRMScore', 'hard2verify': 'balanced F1', 'processbench': 'macro SLA'}
lines = []
P = lambda *a: lines.append(' '.join(str(x) for x in a))
best = {}
for b in MET:
    R = list(csv.DictReader(open(OUT / f'RESULTS_{b}.csv'))); B = list(csv.DictReader(open(OUT / f'BOOT_{b}.csv')))
    v = defaultdict(dict)
    for r in R: v[(r['arm'], r['metric'])][r['prm']] = float(r['value'])
    d = {(r['arm'], r['prm']): r for r in B if r['kind'] == 'diff_vs_F2R0'}
    arms = list(dict.fromkeys(r['arm'] for r in R))
    sec = 'FE-AUC position-matched' if b == 'processbench' else 'within-AUC'
    P(f'\n### {b}: {MET[b]} (GAP flags); diff vs F2 L-SML | R0 with 95% paired bootstrap CI; {sec}; K; effN (mean over 4 cells)')
    P('| arm | mean | ' + ' | '.join(PRMS) + ' | mean diff | CI>0 | CI<0 | ' + f'{sec} mean | K (per cell) | effN mean |')
    P('|' + '---|' * 12)
    sc = {}
    for a in arms:
        x = [v[(a, MET[b])][p] for p in PRMS]; dd = [float(d[(a, p)]['value']) for p in PRMS]
        up = sum(float(d[(a, p)]['lo']) > 0 for p in PRMS); dn = sum(float(d[(a, p)]['hi']) < 0 for p in PRMS)
        s2 = np.mean([v[(a, sec)][p] for p in PRMS]); K = [int(v[(a, 'K groups')][p]) for p in PRMS]
        en = np.mean([v[(a, 'eff N features')][p] for p in PRMS])
        sc[a] = np.mean(x)
        P(f'| {a.replace(" | ", " @ ")} | {np.mean(x):.2f} | ' + ' | '.join(f'{y:.2f}' for y in x) + f' | {np.mean(dd):+.2f} | {up} | {dn} | {s2:.4f} | {K} | {en:.1f} |')
    best[b] = sc
P('\n### rank of arms by mean main metric (per benchmark)')
arms = list(best['prmbench'])
for a in arms:
    P(f'| {a.replace(" | ", " @ ")} | ' + ' | '.join(f'{best[b][a]:.2f} (#{sorted(best[b].values(), reverse=True).index(best[b][a]) + 1})' for b in MET) + ' |')
open(OUT / 'TABLES.md', 'w').write('\n'.join(lines)); print('\n'.join(lines))
# overall: mean rank across benchmarks and gates
P('\n### overall (mean rank over 4 benchmarks; mean diff vs F2@R0 averaged over 16 cells; CI>0 / CI<0 counts over 16 cells)')
P('| arm | mean rank | PRMB rank | Soc rank | H2V rank | PB rank | mean diff (16 cells) | CI>0 | CI<0 |'); P('|' + '---|' * 9)
allB = {b: list(csv.DictReader(open(OUT / f'BOOT_{b}.csv'))) for b in MET}
ov = []
for a in arms:
    rk = [sorted(best[b].values(), reverse=True).index(best[b][a]) + 1 for b in MET]
    dd = [r for b in MET for r in allB[b] if r['kind'] == 'diff_vs_F2R0' and r['arm'] == a]
    assert len(dd) == 16
    ov.append((np.mean(rk), a, rk, np.mean([float(r['value']) for r in dd]), sum(float(r['lo']) > 0 for r in dd), sum(float(r['hi']) < 0 for r in dd)))
for m, a, rk, md, up, dn in sorted(ov):
    P(f'| {a.replace(" | ", " @ ")} | {m:.2f} | ' + ' | '.join(map(str, rk)) + f' | {md:+.2f} | {up} | {dn} |')
open(OUT / 'TABLES.md', 'w').write('\n'.join(lines)); print('\n'.join(lines[-(len(arms) + 3):]))
