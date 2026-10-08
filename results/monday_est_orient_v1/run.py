"""monday_est_orient_v1: orient each channel with our own label-free estimators (DS, MoM, SML) instead of the level anchor.
See PROTOCOL.json."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent; MAIN = OUT.parents[1]; V2 = MAIN / 'results/monday_nested_banks_v2'; CB = MAIN / 'results/monday_cont_baselines_v1'
sys.path.insert(0, str(MAIN))
import spectral_utils.fusion_utils  # noqa: F401
spec = importlib.util.spec_from_file_location('dev_run', V2 / 'dev_run.py'); dr = importlib.util.module_from_spec(spec); spec.loader.exec_module(dr)
ext = dr.ext; BANKS = dr.BANKS; M4 = dr.METHODS4
import er_stage_a as SA  # noqa: E402  (path set by the adapter)
SC = ['DS_ORIENTED', 'MOM_ORIENTED', 'SML_ORIENTED']


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    t0 = time.time(); assert json.loads((OUT / 'PROTOCOL.json').read_text(encoding='utf8'))['status'] == 'FROZEN'
    A = pd.read_csv(dr.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); cells = A.cell.to_numpy(); N = len(A)
    target = A.target.to_numpy().astype(int); groups = A.source_group.to_numpy()
    with np.load(dr.SE / 'OOF_STEP_SCORES.npz') as z: OFF = z['offsets'].astype(np.int64); LAB = z['labels'].astype(bool)
    pb = np.array([c.startswith('pb_') for c in cells]); prm_idx = np.flatnonzero(~pb); pb_idx = np.flatnonzero(pb)
    F = np.load(ext.FE / 'source/FEATURES.npz'); pop = ext.Population(F, BANKS)
    R = {(bk, s): {'auc': np.full(N, np.nan), 'argmax': np.full(N, -9, int)} for bk in BANKS for s in SC}
    flips = []
    for bk in BANKS:
        for ce in sorted(set(cells)):
            ans = np.flatnonzero(cells == ce); bank = pop.cell_bank(bk, ans); V = bank['V']; off = bank['off']; marks = bank['marks']; names = bank['names']
            lev = pop.level[bank['rows']]
            lev_sg = np.array([(np.sign(np.corrcoef(V[:, j], lev)[0, 1]) or 1.) if V[:, j].std() > 0 else 1. for j in range(V.shape[1])])
            sg = {'DS_ORIENTED': np.where(SA.em_estimate(marks, 'ds')['pi'] > .5, 1., -1.),
                  'MOM_ORIENTED': np.where(SA.tensor_mom_estimate(marks, 0)['pi'] > .5, 1., -1.),
                  'SML_ORIENTED': np.where(SA.sml_estimate(marks, 0)['t'] > 0, 1., -1.)}
            for nm, s in sg.items():
                flips.append({'bank': bk, 'cell': ce, 'scorer': nm, 'n_flipped': int((s < 0).sum()), 'n_flipped_level_anchor': int((lev_sg < 0).sum()),
                              'same_flip_set_as_level_anchor': bool(np.array_equal(s, lev_sg)), 'flipped': [names[j] for j in np.flatnonzero(s < 0)],
                              'flipped_level_anchor': [names[j] for j in np.flatnonzero(lev_sg < 0)]})
                score = (V * s).mean(1); st = R[(bk, nm)]
                for q, i in enumerate(ans):
                    ss = score[off[q]:off[q + 1]]; st['argmax'][i] = int(np.flatnonzero(ss >= ss.max() - 8 * np.finfo(float).eps)[0])
                    if not pb[i]: st['auc'][i] = auc(LAB[OFF[i]:OFF[i + 1]], ss)
        print(bk, round(time.time() - t0), flush=True)
    pd.DataFrame(flips).to_csv(OUT / 'FLIPS.csv', index=False)
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
    PA = np.load(V2 / 'PER_ANSWER.npz'); PC = np.load(CB / 'PER_ANSWER.npz'); rows, cons = [], []
    for bk in BANKS:
        cur = {nm: (prm_auc(R[(bk, nm)]['auc']), pb_hit(R[(bk, nm)]['argmax'])) for nm in SC}
        refs = {'ALL_EQ': (prm_auc(PA[f'{bk}__ALL_EQ__auc']), pb_hit(PA[f'{bk}__ALL_EQ__argmax']))}
        for m in ('ALL_EQ_ORIENTED', 'IU_PCR', 'LSML_CONT'): refs[m] = (prm_auc(PC[f'{bk}__{m}__auc']), pb_hit(PC[f'{bk}__{m}__argmax']))
        for m in M4 + ['DS_FILTER_EQ']:
            fk = f'{bk}__{m}__failed'
            refs[m] = (prm_auc(PA[f'{bk}__{m}__auc']) if not PA[fk][prm_idx].any() else None, pb_hit(PA[f'{bk}__{m}__argmax']) if not PA[fk][pb_idx].any() else None)
        for nm, (a, h) in cur.items():
            rows.append({'bank': bk, 'scorer': nm, 'prmb_within_auc': a[0], 'pb_argmax_hit': h[0]})
            for ref, (ra, rh) in refs.items():
                for k, metric, x, y in ((0, 'prmb_within_auc', a, ra), (1, 'pb_argmax_hit', h, rh)):
                    if y is None: continue
                    pt, lo, hi = ci(x - y); cons.append({'bank': bk, 'left': nm, 'right': ref, 'metric': metric, 'delta': pt, 'lo95': lo, 'hi95': hi})
    pd.DataFrame(rows).to_csv(OUT / 'ABSOLUTE.csv', index=False); pd.DataFrame(cons).to_csv(OUT / 'CONTRASTS.csv', index=False)
    print('done', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
