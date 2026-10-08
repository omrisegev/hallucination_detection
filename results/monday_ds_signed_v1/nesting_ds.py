"""Nesting contrasts for the DS-signed average (same aggregation/bootstrap as run.py and NESTING_CONTRASTS.csv)."""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent; MAIN = OUT.parents[1]; V2 = MAIN / 'results/monday_nested_banks_v2'
spec = importlib.util.spec_from_file_location('dev_run', V2 / 'dev_run.py'); dr = importlib.util.module_from_spec(spec); spec.loader.exec_module(dr)
A = pd.read_csv(dr.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); cells = A.cell.to_numpy(); target = A.target.to_numpy().astype(int); groups = A.source_group.to_numpy()
pb = np.array([c.startswith('pb_') for c in cells]); prm_idx = np.flatnonzero(~pb); pb_idx = np.flatnonzero(pb)
pg, pinv = np.unique(groups[prm_idx], return_inverse=True); bg, binv = np.unique(groups[pb_idx], return_inverse=True)
rng = np.random.default_rng(dr.SEED)
Wp = np.vstack([np.ones(len(pg)), rng.multinomial(len(pg), np.full(len(pg), 1 / len(pg)), size=dr.DRAWS)]).astype(float)
Wb = np.vstack([np.ones(len(bg)), rng.multinomial(len(bg), np.full(len(bg), 1 / len(bg)), size=dr.DRAWS)]).astype(float)
pb_cells = sorted(set(cells[pb_idx])); err = target[pb_idx] >= 0
def gs(X, inv, G): o = np.zeros((G, X.shape[1])); np.add.at(o, inv, X); return o
def prm_auc(a):
    a = a[prm_idx]; S = Wp @ gs(np.column_stack([np.nan_to_num(a), np.isfinite(a)]).astype(float), pinv, len(pg)); return S[:, 0] / S[:, 1]
def pb_hit(am):
    hit = am[pb_idx] == target[pb_idx]; cols = []
    for c in pb_cells: ck = cells[pb_idx] == c; cols += [hit & err & ck, err & ck]
    S = (Wb @ gs(np.column_stack(cols).astype(float), binv, len(bg))).reshape(len(Wb), len(pb_cells), 2); return np.nanmean(S[:, :, 0] / S[:, :, 1], 1)
ci = lambda x: (float(x[0]), float(np.quantile(x[1:], .025)), float(np.quantile(x[1:], .975)))
P = np.load(OUT / 'PER_ANSWER.npz'); PA = np.load(V2 / 'PER_ANSWER.npz'); rows = []
# parity of this aggregation with v2 NESTING_CONTRASTS for ALL_EQ N22->N40
for sc, src in (('DS_SIGNED', P), ('ALL_EQ', PA)):
    for a, b in (('N11', 'N13'), ('N13', 'N22'), ('N22', 'N40'), ('N40', 'N53')):
        for metric, f, k in (('prmb_within_auc', prm_auc, 'auc'), ('pb_sla', pb_hit, 'argmax')):
            pt, lo, hi = ci(f(src[f'{b}__{sc}__{k}']) - f(src[f'{a}__{sc}__{k}'])); rows.append({'scorer': sc, 'from': a, 'to': b, 'metric': metric, 'delta': pt, 'lo95': lo, 'hi95': hi})
pd.DataFrame(rows).to_csv(OUT / 'NESTING_DS.csv', index=False); print(pd.DataFrame(rows).round(4).to_string())
