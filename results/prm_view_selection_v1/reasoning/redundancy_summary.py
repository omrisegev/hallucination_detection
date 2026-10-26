"""Print the pre-declared redundancy/SLIDER comparison from REDUNDANCY_SLIDER.csv."""
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'REDUNDANCY_SLIDER.csv')))
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
ARMS = ['PRM', 'POSITION', 'CHAIN_N16', 'TELE_N16', 'LSML_CONT', 'CHAINRES_N16', 'TELERES_N16',
        'ALL16|RAW|+PRM', 'ALL16|POS|+PRM', 'ALL16|RAW|alone', 'ALL16|POS|alone',
        'SUB5|RAW|+PRM', 'SUB5|POS|+PRM', 'SUB5|RAW|alone', 'SUB5|POS|alone',
        'S_R|RAW|+PRM', 'S_R|POS|+PRM', 'S_R|RAW|alone', 'S_R|POS|alone',
        'C3|RAW|+PRM', 'C3|POS|+PRM', 'C3|RAW|alone', 'C3|POS|alone']
for bench, subs in (('prmbench', ['redundency']),
                    ('socratic', ['all_redundancy', 'premise_redundancy', 'decomposition_redundancy',
                                  'regather_redundancy', 'integration_redundancy'])):
    for sub in subs:
        print(f'===== {bench} / {sub}: SLIDER-protocol accuracy | balanced accuracy | GAP balanced accuracy | AUC  '
              '(values per PRM: Qwen-7B, Qwen-72B, PURE, RLHFlow)')
        for a in ARMS:
            R = [next((r for r in rows if r['cell'] == f'{bench}__{p}' and r['subset'] == sub and r['arm'] == a), None) for p in PRMS]
            if any(r is None for r in R): continue
            g = lambda k: [float(r[k]) for r in R]
            print(f"{a:16s} acc {' '.join(f'{x:5.1f}' for x in g('slider_acc'))} | bal {' '.join(f'{x:5.1f}' for x in g('slider_bal_acc'))}"
                  f" | gapbal {' '.join(f'{x:5.1f}' for x in g('gap_bal_acc'))} | auc {' '.join(f'{x:.3f}' for x in g('within_auc'))}")
