"""monday_types_singles_v1 part B rerun with Dawid-Skene signs (Omri, 2026-10-08): every N56 channel alone, its sign per cell
from er_stage_a.em_estimate(marks, 'ds') on the cell's tail votes (pi > 0.5 -> +1, else -1) instead of the level-mean correlation.
Same metrics, seed and bootstrap as part B. Fusion contrasts on N16 use the DS-signed average in place of the oriented average.
Outputs in singles/; the original part-B files are untouched."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent / 'singles'; OUT.mkdir(exist_ok=True); MAIN = OUT.parents[2]
V2 = MAIN / 'results/monday_nested_banks_v2'; CB = MAIN / 'results/monday_cont_baselines_v1'; TS = MAIN / 'results/monday_types_singles_v1'; DSD = OUT.parent
sys.path.insert(0, str(MAIN))
import spectral_utils.fusion_utils  # noqa: F401
spec = importlib.util.spec_from_file_location('dev_run', V2 / 'dev_run.py'); dr = importlib.util.module_from_spec(spec); spec.loader.exec_module(dr)
ext = dr.ext; M4 = dr.METHODS4
import er_stage_a as SA  # noqa: E402


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    t0 = time.time()
    A = pd.read_csv(dr.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); cells = A.cell.to_numpy(); N = len(A)
    target = A.target.to_numpy().astype(int); groups = A.source_group.to_numpy()
    with np.load(dr.SE / 'OOF_STEP_SCORES.npz') as z: OFF = z['offsets'].astype(np.int64); LAB = z['labels'].astype(bool)
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
    ci = lambda x: (float(x[0]), float(np.nanquantile(x[1:], .025)), float(np.nanquantile(x[1:], .975)))

    F = np.load(ext.FE / 'source/FEATURES.npz'); pop = ext.Population(F, ['N56'])
    names = None; S_auc = {}; S_am = {}; signs = []
    for ce in sorted(set(cells)):
        ans = np.flatnonzero(cells == ce); bank = pop.cell_bank('N56', ans); V = bank['V']; off = bank['off']; lev = pop.level[bank['rows']]
        if names is None:
            names = bank['names']; S_auc.update({c: np.full(N, np.nan) for c in names}); S_am.update({c: np.full(N, -9, int) for c in names})
        pi = SA.em_estimate(bank['marks'], 'ds')['pi']; sg = np.where(pi > .5, 1., -1.)
        lev_sg = np.array([(np.sign(np.corrcoef(V[:, j], lev)[0, 1]) or 1.) if V[:, j].std() > 0 else 1. for j in range(V.shape[1])])
        for j, c in enumerate(names):
            signs.append({'cell': ce, 'channel': c, 'ds_pi': float(pi[j]), 'ds_sign': int(sg[j]), 'level_sign': int(lev_sg[j])})
            s = V[:, j] * sg[j]
            for q, i in enumerate(ans):
                ss = s[off[q]:off[q + 1]]; S_am[c][i] = int(np.flatnonzero(ss >= ss.max() - 8 * np.finfo(float).eps)[0])
                if not pb[i]: S_auc[c][i] = auc(LAB[OFF[i]:OFF[i + 1]], ss)
        print('singles', ce, round(time.time() - t0), flush=True)
    SG = pd.DataFrame(signs); SG.to_csv(OUT / 'SIGNS.csv', index=False)
    old = pd.read_csv(TS / 'SINGLE_CHANNELS.csv').set_index('channel'); rows = []
    for c in names:
        a = prm_auc(S_auc[c]); h = pb_hit(S_am[c]); g = SG[SG.channel == c]
        rows.append({'channel': c, 'prmb_auc': a[0], 'prmb_lo': float(np.quantile(a[1:], .025)), 'prmb_hi': float(np.quantile(a[1:], .975)),
                     'pb_sla': h[0], 'pb_lo': float(np.quantile(h[1:], .025)), 'pb_hi': float(np.quantile(h[1:], .975)),
                     'n_cells_ds_flipped': int((g.ds_sign < 0).sum()), 'n_cells_level_flipped': int((g.level_sign < 0).sum()),
                     'n_cells_sign_differs': int((g.ds_sign != g.level_sign).sum()), 'flipped_on_prmbench_ds': bool(g[g.cell == dr.PRM].ds_sign.iloc[0] < 0),
                     'prmb_auc_level_sign': float(old.loc[c, 'prmb_auc']), 'pb_sla_level_sign': float(old.loc[c, 'pb_hit'])})
    SC = pd.DataFrame(rows).sort_values('prmb_auc', ascending=False); SC.to_csv(OUT / 'SINGLE_CHANNELS_DS.csv', index=False)
    best_p = SC.iloc[0].channel; best_b = SC.sort_values('pb_sla', ascending=False).iloc[0].channel
    PA = np.load(V2 / 'PER_ANSWER.npz'); PC = np.load(CB / 'PER_ANSWER.npz'); PD = np.load(DSD / 'PER_ANSWER.npz')
    fus = {f'N16/{m}': (PA[f'N16__{m}__auc'], PA[f'N16__{m}__argmax']) for m in M4 + ['ALL_EQ']}
    fus.update({f'N16/{m}': (PC[f'N16__{m}__auc'], PC[f'N16__{m}__argmax']) for m in ('LSML_CONT', 'IU_PCR')})
    fus['N16/DS_SIGNED'] = (PD['N16__DS_SIGNED__auc'], PD['N16__DS_SIGNED__argmax'])
    crow = []
    for fn, (fa, fm) in fus.items():
        for ch in sorted({best_p, best_b, 'realized_drv'}):
            pt, lo, hi = ci(prm_auc(fa) - prm_auc(S_auc[ch])); crow.append({'fusion': fn, 'single': ch, 'metric': 'prmb_within_auc', 'delta': pt, 'lo95': lo, 'hi95': hi})
            pt, lo, hi = ci(pb_hit(fm) - pb_hit(S_am[ch])); crow.append({'fusion': fn, 'single': ch, 'metric': 'pb_sla', 'delta': pt, 'lo95': lo, 'hi95': hi})
    pd.DataFrame(crow).to_csv(OUT / 'FUSION_VS_SINGLE_DS.csv', index=False)
    json.dump({'best_single_prmbench': best_p, 'best_single_processbench': best_b, 'median_prmb_auc': float(SC.prmb_auc.median()), 'median_pb_sla': float(SC.pb_sla.median()),
               'n_channels_with_any_sign_difference': int((SC.n_cells_sign_differs > 0).sum()), 'runtime_s': round(time.time() - t0)}, open(OUT / 'STATUS.json', 'w'), indent=1)
    print('done', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
