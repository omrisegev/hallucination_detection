"""Which single channel catches which PRMBench error type. PRE-FIX input (old telemetry), CPU, labels for evaluation only.
Mirrors results/monday_types_singles_v1/run.py part B (label-free sign = sign of corr with the cell LEVEL mean), restricted to the
PRMBench cell, and averages the per-answer within-answer AUC by error type. Adds step index and step length as references."""
import importlib.util, json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); sys.path.insert(0, str(MAIN))
import spectral_utils.fusion_utils  # noqa
spec = importlib.util.spec_from_file_location('dr', MAIN / 'results/monday_nested_banks_v2/dev_run.py'); dr = importlib.util.module_from_spec(spec); spec.loader.exec_module(dr)
ext = dr.ext
OUT = Path(__file__).parent


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


A = pd.read_csv(dr.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); cells = A.cell.to_numpy(); ids = A.id.astype(str).to_numpy()
with np.load(dr.SE / 'OOF_STEP_SCORES.npz') as z: OFF = z['offsets'].astype(np.int64); LAB = z['labels'].astype(bool)
fr = json.loads((dr.SE / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
meta = {m['idx']: m for m in pickle.load(open(fr['prm_metadata']['path'], 'rb')).values()}
F = np.load(ext.FE / 'source/FEATURES.npz'); pop = ext.Population(F, ['N56'])
ans = np.flatnonzero(cells == dr.PRM); bank = pop.cell_bank('N56', ans); V = bank['V']; off = bank['off']; lev = pop.level[bank['rows']]; names = bank['names']
cls = np.array([meta[ids[i]]['classification'] for i in ans])
sg = np.array([(np.sign(np.corrcoef(V[:, j], lev)[0, 1]) or 1.) if V[:, j].std() > 0 else 1. for j in range(V.shape[1])])
n16 = set(dr.NB['N16'])
R = np.full((len(ans), len(names) + 1), np.nan)
for q, i in enumerate(ans):
    y = LAB[OFF[i]:OFF[i + 1]]; seg = slice(off[q], off[q + 1])
    if len(y) != off[q + 1] - off[q]: raise SystemExit(f'length mismatch {i}')
    for j in range(len(names)): R[q, j] = auc(y, V[seg, j] * sg[j])
    R[q, -1] = auc(y, np.arange(len(y)).astype(float))
cols = list(names) + ['STEP_INDEX']
D = pd.DataFrame(R, columns=cols); D['type'] = cls
types = [t for t in sorted(set(cls)) if t not in ('correct', 'multi_solutions')]
M = D[D.type.isin(types)].groupby('type')[cols].mean().T
M['ALL'] = D[D.type.isin(types)][cols].mean()
M['n16'] = [c in n16 for c in M.index]; M['sign'] = list(sg) + [1.]
M.to_csv(OUT / 'CHANNEL_BY_TYPE_PREFIX_OLD.csv')
counts = D[D.type.isin(types)].groupby('type')[cols[0]].apply(lambda x: int(x.notna().sum()))
print('answers with AUC per type:', counts.to_dict())
for t in types + ['ALL']:
    s = M[t].drop('STEP_INDEX').sort_values(ascending=False)
    print(f'\n{t}: step_index {M.loc["STEP_INDEX", t]:.3f} | top5', ', '.join(f'{c} {v:.3f}' for c, v in s.head(5).items()), '| bottom2', ', '.join(f'{c} {v:.3f}' for c, v in s.tail(2).items()))
