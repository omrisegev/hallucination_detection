"""Old vs think-fix external results of the Monday panel methods (step 3 part 2).

    python results/thinkfix_rescore_v1/external/compare_panel_external.py
Tables compared (old = frozen results in the main checkout; new = external/<variant>/panel_chain/...):
  nested  results/monday_nested_banks_v2/METRICS.csv        (N banks x Monday methods x rules: official_metric, within_auc)
  panel   results/monday_external_panel_v1/METRICS.csv      (B banks)
  cont    results/monday_cont_baselines_v1/EXTERNAL_METRICS.csv (continuous L-SML / IU-PCR)
  orient  results/monday_nested_banks_v2/EXTERNAL_ORIENTED_CONTROL.csv
QwQ rows exist only in the frozen tables and are dropped. Writes EXTERNAL_PANEL_COMPARISON.csv.
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent; MAIN = HERE.parents[2]
TABLES = {'nested': 'results/monday_nested_banks_v2/METRICS.csv', 'panel': 'results/monday_external_panel_v1/METRICS.csv',
          'cont': 'results/monday_cont_baselines_v1/EXTERNAL_METRICS.csv', 'orient': 'results/monday_nested_banks_v2/EXTERNAL_ORIENTED_CONTROL.csv'}
METRICS = ['official_metric', 'within_auc', 'f1_correct', 'f1_error']
out = []
for name, rel in TABLES.items():
    frames = {'old': pd.read_csv(MAIN / rel)}
    for v in ('thinkfix', 'thinkfix_mask1'):
        p = HERE / v / 'panel_chain' / rel
        if p.exists(): frames[v] = pd.read_csv(p)
    keys = [c for c in ('cell', 'benchmark', 'bank', 'scorer', 'rule') if c in frames['old'].columns]
    mets = [c for c in frames['old'].columns if c in METRICS or (c not in keys and pd.api.types.is_numeric_dtype(frames['old'][c]) and c not in ('ci95_lo', 'ci95_hi'))]
    long = []
    for v, df in frames.items():
        df = df[df.cell != 'socratic_qwq32b'] if 'cell' in df else df
        m = df[keys + [c for c in mets if c in df.columns]].melt(id_vars=keys, var_name='measure', value_name=v)
        long.append(m.set_index(keys + ['measure']))
    t = pd.concat(long, axis=1).reset_index(); t.insert(0, 'table', name); out.append(t)
df = pd.concat(out, ignore_index=True)
for v in ('thinkfix', 'thinkfix_mask1'):
    df[f'{v}_minus_old'] = pd.to_numeric(df[v], errors='coerce') - pd.to_numeric(df['old'], errors='coerce')
df.to_csv(HERE / 'EXTERNAL_PANEL_COMPARISON.csv', index=False)
sel = df[(df.table == 'nested') & (df.bank == 'N16') & (df.rule == 'ANSWER_Z_Q80') & df.measure.isin(['official_metric', 'within_auc'])]
print(sel[['cell', 'scorer', 'measure', 'old', 'thinkfix', 'thinkfix_mask1', 'thinkfix_mask1_minus_old']].round(4).to_string(index=False))
d = df[(df.table == 'nested') & (df.measure == 'within_auc')]
print('nested within_auc mask1-old: mean', round(d.thinkfix_mask1_minus_old.mean(), 4), 'by cell', d.groupby('cell').thinkfix_mask1_minus_old.mean().round(4).to_dict())
