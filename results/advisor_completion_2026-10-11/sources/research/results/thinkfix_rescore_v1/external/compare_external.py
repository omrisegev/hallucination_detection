"""Old vs think-fix locked LSML external evaluation (Hard2Verify, Socratic; Qwen3-8B).

    python results/thinkfix_rescore_v1/external/compare_external.py
Reads the frozen results/lsml_external_generalization_v1/evaluation/METRICS.json (old records, 2026-09-24) and the
thinkfix / thinkfix_mask1 re-evaluations here; writes EXTERNAL_LSML_COMPARISON.csv.
'metric' is each benchmark's official metric as registered (Hard2Verify / PRMScore-style), computed by the
unchanged locked evaluator; within_auc is the mean per-answer within-AUC.
"""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent; MAIN = HERE.parents[2]
SRC = {'old': MAIN / 'results/lsml_external_generalization_v1/evaluation/METRICS.json',
       'thinkfix': HERE / 'thinkfix/lsml_locked/METRICS.json', 'thinkfix_mask1': HERE / 'thinkfix_mask1/lsml_locked/METRICS.json'}
rows = []
for v, p in SRC.items():
    m = json.loads(p.read_text(encoding='utf8'))
    for cell, r in m.items():
        if cell == 'socratic_qwq32b': continue
        for arm, a in r['arms'].items():
            for k in ('metric', 'within_auc', 'correct_recall', 'error_recall', 'disjoint_metric'):
                rows.append({'cell': cell, 'arm': arm, 'group': 'ALL', 'measure': k, 'variant': v, 'value': a.get(k)})
        for cat, c in r.get('categories', {}).items():
            for arm, val in c['arms'].items():
                rows.append({'cell': cell, 'arm': arm, 'group': cat, 'measure': 'metric', 'variant': v, 'value': val})
df = pd.DataFrame(rows).pivot_table(index=['cell', 'arm', 'group', 'measure'], columns='variant', values='value', aggfunc='first').reset_index()
df['thinkfix_minus_old'] = df.thinkfix - df.old; df['mask1_minus_old'] = df.thinkfix_mask1 - df.old
df.to_csv(HERE / 'EXTERNAL_LSML_COMPARISON.csv', index=False)
print(df[(df.group == 'ALL') & df.measure.isin(['metric', 'within_auc'])].round(4).to_string(index=False))
