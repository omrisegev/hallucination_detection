import csv
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parent
MET = {'prmbench': 'PRMScore', 'socratic': 'PRMScore', 'hard2verify': 'balanced F1', 'processbench': 'macro SLA'}
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
v = {}
for b, m in MET.items():
    for r in csv.DictReader(open(OUT / f'RESULTS_{b}.csv')):
        if r['metric'] == m: v[(b, r['prm'], r['arm'])] = float(r['value'])
C = []
for rep in ('R0', 'R1'):
    C += [(f'binary vs equal across (F3 - F4a) {rep}', f'F3 GSP-T | {rep}', f'F4a GSP-within + equal across | {rep}'),
          (f'binary vs rank-one across (F3 - F4b) {rep}', f'F3 GSP-T | {rep}', f'F4b GSP-within + rank-one across | {rep}'),
          (f'binary vs signed-SML across, L-SML within (F5 - F2) {rep}', f'F5 L-SML-within + binary across | {rep}', f'F2 L-SML | {rep}'),
          (f'GSP vs L-SML within, binary across (F3 - F5) {rep}', f'F3 GSP-T | {rep}', f'F5 L-SML-within + binary across | {rep}'),
          (f'GSP-within+rank-one vs L-SML (F4b - F2) {rep}', f'F4b GSP-within + rank-one across | {rep}', f'F2 L-SML | {rep}'),
          (f'A2 readout on L-SML (F6F2 - F2) {rep}', f'F6 A2 -> F2 L-SML | {rep}', f'F2 L-SML | {rep}'),
          (f'A2 readout on GSP-T (F6F3 - F3) {rep}', f'F6 A2 -> F3 GSP-T | {rep}', f'F3 GSP-T | {rep}')]
for arm in ('F1 mean', 'F2 L-SML', 'F3 GSP-T', 'F4a GSP-within + equal across', 'F4b GSP-within + rank-one across', 'F5 L-SML-within + binary across', 'F6 A2 -> F2 L-SML', 'F6 A2 -> F3 GSP-T'):
    C.append((f'R1 - R0 ({arm})', f'{arm} | R1', f'{arm} | R0'))
lines = ['| contrast (point estimates) | ' + ' | '.join(f'{b} mean diff (wins/4)' for b in MET) + ' | all 16: wins |', '|' + '---|' * 6]
for name, a, bb in C:
    cells = []; tot = 0
    for b in MET:
        d = [v[(b, p, a)] - v[(b, p, bb)] for p in PRMS]; w = sum(x > 0 for x in d); tot += w
        cells.append(f'{np.mean(d):+.2f} ({w}/4)')
    lines.append(f'| {name} | ' + ' | '.join(cells) + f' | {tot}/16 |')
open(OUT / 'CONTRASTS.md', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
