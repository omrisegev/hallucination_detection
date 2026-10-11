"""monday_types_singles_v1: (A) PRMBench error-type profile with intervals across all label-free configurations;
(B) every single channel vs fusion on both benchmarks. See PROTOCOL.json."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import importlib.util, json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent; MAIN = OUT.parents[1]; V2 = MAIN / 'results/monday_nested_banks_v2'; CB = MAIN / 'results/monday_cont_baselines_v1'
sys.path.insert(0, str(MAIN))
import spectral_utils.fusion_utils  # noqa: F401  main package first (same order as monday_cont_baselines_v1)
spec = importlib.util.spec_from_file_location('dev_run', V2 / 'dev_run.py'); dr = importlib.util.module_from_spec(spec); spec.loader.exec_module(dr)
ext = dr.ext; BANKS = dr.BANKS; M4 = dr.METHODS4


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    t0 = time.time()
    assert json.loads((OUT / 'PROTOCOL.json').read_text(encoding='utf8'))['status'] == 'FROZEN'
    A = pd.read_csv(dr.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); cells = A.cell.to_numpy(); N = len(A)
    ids = A.id.astype(str).to_numpy(); target = A.target.to_numpy().astype(int); groups = A.source_group.to_numpy()
    with np.load(dr.SE / 'OOF_STEP_SCORES.npz') as z: OFF = z['offsets'].astype(np.int64); LAB = z['labels'].astype(bool)
    pb = np.array([c.startswith('pb_') for c in cells]); prm_idx = np.flatnonzero(~pb); pb_idx = np.flatnonzero(pb)
    fr = json.loads((dr.SE / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
    meta = {m['idx']: m for m in pickle.load(open(fr['prm_metadata']['path'], 'rb')).values()}
    cls = np.array([meta[ids[i]]['classification'] for i in prm_idx])
    PA = np.load(V2 / 'PER_ANSWER.npz'); PC = np.load(CB / 'PER_ANSWER.npz')
    pg, pinv = np.unique(groups[prm_idx], return_inverse=True); bg, binv = np.unique(groups[pb_idx], return_inverse=True)
    rng = np.random.default_rng(dr.SEED)
    Wp = np.vstack([np.ones(len(pg)), rng.multinomial(len(pg), np.full(len(pg), 1 / len(pg)), size=dr.DRAWS)]).astype(float)
    Wb = np.vstack([np.ones(len(bg)), rng.multinomial(len(bg), np.full(len(bg), 1 / len(bg)), size=dr.DRAWS)]).astype(float)
    pb_cells = sorted(set(cells[pb_idx])); err = target[pb_idx] >= 0
    def gs(X, inv, G): o = np.zeros((G, X.shape[1])); np.add.at(o, inv, X); return o
    def prm_auc(a, mask=None):
        a = a[prm_idx].copy()
        if mask is not None: a[~mask] = np.nan
        S = Wp @ gs(np.column_stack([np.nan_to_num(a), np.isfinite(a)]).astype(float), pinv, len(pg)); return S[:, 0] / S[:, 1]
    def pb_hit(am):
        hit = am[pb_idx] == target[pb_idx]; cols = []
        for c in pb_cells: ck = cells[pb_idx] == c; cols += [hit & err & ck, err & ck]
        S = (Wb @ gs(np.column_stack(cols).astype(float), binv, len(bg))).reshape(len(Wb), len(pb_cells), 2); return np.nanmean(S[:, :, 0] / S[:, :, 1], 1)
    ci = lambda x: (float(x[0]), float(np.nanquantile(x[1:], .025)), float(np.nanquantile(x[1:], .975)))
    # parity: v2 N16 HEM_MOM_cc PRMBench AUC
    v2b = pd.read_csv(V2 / 'PANEL_B_PRMBENCH.csv'); ref = v2b[(v2b.scorer == 'HEM_MOM_cc') & (v2b.bank == 'N16')].within_auc.iloc[0]
    got = prm_auc(PA['N16__HEM_MOM_cc__auc'])[0]
    if abs(got - ref) > 1e-12: raise SystemExit(f'parity {got} {ref}')

    # ---------------- A: error types
    conf = {}
    for bk in BANKS:
        for m in M4 + ['ALL_EQ']:
            k = f'{bk}__{m}__auc'
            if k in PA.files and not PA[f'{bk}__{m}__failed'][prm_idx].any(): conf[(bk, m)] = PA[k]
        for m in ('ALL_EQ_ORIENTED', 'LSML_CONT', 'IU_PCR'): conf[(bk, m)] = PC[f'{bk}__{m}__auc']
    refs = {'PRM': PA['PRM__auc'], 'STEP_INDEX': PA['STEP_INDEX__auc'], 'MTG': PA['MTG__auc']}
    types = sorted(set(cls) - {'correct', 'multi_solutions'}); rows, summ = [], []
    for t in types:
        mask = cls == t; n_t = int((mask & np.isfinite(PA['PRM__auc'][prm_idx])).sum()); rp = prm_auc(refs['PRM'], mask)
        for (bk, m), a in conf.items():
            v = prm_auc(a, mask); pt, lo, hi = ci(v - rp)
            rows.append({'type': t, 'n_answers': n_t, 'bank': bk, 'scorer': m, 'family': 'method' if m in M4 else m, 'auc': v[0], 'auc_lo': float(np.nanquantile(v[1:], .025)),
                         'auc_hi': float(np.nanquantile(v[1:], .975)), 'minus_PRM': pt, 'minus_PRM_lo': lo, 'minus_PRM_hi': hi})
        for r, a in refs.items():
            v = prm_auc(a, mask); rows.append({'type': t, 'n_answers': n_t, 'bank': '-', 'scorer': r, 'family': 'reference', 'auc': v[0],
                                               'auc_lo': float(np.nanquantile(v[1:], .025)), 'auc_hi': float(np.nanquantile(v[1:], .975))})
    T = pd.DataFrame(rows); T.to_csv(OUT / 'TYPES_ALL_CONFIGS.csv', index=False)
    L = T[T.family != 'reference']
    for t in types:
        x = L[L.type == t]; prm = T[(T.type == t) & (T.scorer == 'PRM')].iloc[0]; si = T[(T.type == t) & (T.scorer == 'STEP_INDEX')].iloc[0]
        n16 = x[(x.bank == 'N16') & x.scorer.isin(M4)]
        summ.append({'type': t, 'n_answers': int(x.n_answers.iloc[0]), 'n_configs': len(x), 'clearly_above_PRM': int((x.minus_PRM_lo > 0).sum()),
                     'clearly_below_PRM': int((x.minus_PRM_hi < 0).sum()), 'median_minus_PRM': float(x.minus_PRM.median()),
                     'min_minus_PRM': float(x.minus_PRM.min()), 'max_minus_PRM': float(x.minus_PRM.max()), 'PRM_auc': prm.auc, 'PRM_lo': prm.auc_lo, 'PRM_hi': prm.auc_hi,
                     'step_index_auc': si.auc, 'N16_methods_auc_min': n16.auc.min(), 'N16_methods_auc_max': n16.auc.max(),
                     'N16_HEM_MOM_cc_minus_PRM': float(n16[n16.scorer == 'HEM_MOM_cc'].minus_PRM.iloc[0]),
                     'N16_HEM_MOM_cc_lo': float(n16[n16.scorer == 'HEM_MOM_cc'].minus_PRM_lo.iloc[0]), 'N16_HEM_MOM_cc_hi': float(n16[n16.scorer == 'HEM_MOM_cc'].minus_PRM_hi.iloc[0])})
    pd.DataFrame(summ).to_csv(OUT / 'TYPES_SUMMARY.csv', index=False)
    print('types done', round(time.time() - t0), flush=True)

    # ---------------- B: single channels (N56 = all channels)
    F = np.load(ext.FE / 'source/FEATURES.npz'); pop = ext.Population(F, ['N56'])
    names = None; S_auc = {}; S_am = {}; S_auc_raw = {}; S_am_raw = {}; flips = {}
    for ce in sorted(set(cells)):
        ans = np.flatnonzero(cells == ce); bank = pop.cell_bank('N56', ans); V = bank['V']; off = bank['off']; lev = pop.level[bank['rows']]
        if names is None:
            names = bank['names']
            for d in (S_auc, S_auc_raw): d.update({c: np.full(N, np.nan) for c in names})
            for d in (S_am, S_am_raw): d.update({c: np.full(N, -9, int) for c in names})
        sg = np.array([(np.sign(np.corrcoef(V[:, j], lev)[0, 1]) or 1.) if V[:, j].std() > 0 else 1. for j in range(V.shape[1])])
        flips[ce] = [names[j] for j in range(len(sg)) if sg[j] < 0]
        for j, c in enumerate(names):
            for sgn, da, dm in ((sg[j], S_auc, S_am), (1., S_auc_raw, S_am_raw)):
                s = V[:, j] * sgn
                for q, i in enumerate(ans):
                    ss = s[off[q]:off[q + 1]]; dm[c][i] = int(np.flatnonzero(ss >= ss.max() - 8 * np.finfo(float).eps)[0])
                    if not pb[i]: da[c][i] = auc(LAB[OFF[i]:OFF[i + 1]], ss)
        print('singles', ce, round(time.time() - t0), flush=True)
    srows = []
    for c in names:
        a = prm_auc(S_auc[c]); h = pb_hit(S_am[c]); a2 = prm_auc(S_auc_raw[c]); h2 = pb_hit(S_am_raw[c])
        srows.append({'channel': c, 'prmb_auc': a[0], 'prmb_lo': float(np.quantile(a[1:], .025)), 'prmb_hi': float(np.quantile(a[1:], .975)),
                      'pb_hit': h[0], 'pb_lo': float(np.quantile(h[1:], .025)), 'pb_hi': float(np.quantile(h[1:], .975)),
                      'prmb_auc_raw_sign': a2[0], 'pb_hit_raw_sign': h2[0], 'flipped_on_prmbench': c in flips[dr.PRM]})
    SC = pd.DataFrame(srows).sort_values('prmb_auc', ascending=False); SC.to_csv(OUT / 'SINGLE_CHANNELS.csv', index=False)
    # contrasts: fusion configs on N16 vs the best single channel per benchmark (label-chosen)
    best_p = SC.iloc[0].channel; best_b = SC.sort_values('pb_hit', ascending=False).iloc[0].channel
    fus = {f'N16/{m}': (PA[f'N16__{m}__auc'], PA[f'N16__{m}__argmax']) for m in M4 + ['ALL_EQ']}
    fus.update({f'N16/{m}': (PC[f'N16__{m}__auc'], PC[f'N16__{m}__argmax']) for m in ('ALL_EQ_ORIENTED', 'LSML_CONT', 'IU_PCR')})
    crow = []
    for fn, (fa, fm) in fus.items():
        for ch in sorted({best_p, best_b, 'realized_drv'}):
            pt, lo, hi = ci(prm_auc(fa) - prm_auc(S_auc[ch])); crow.append({'fusion': fn, 'single': ch, 'metric': 'prmb_within_auc', 'delta': pt, 'lo95': lo, 'hi95': hi})
            pt, lo, hi = ci(pb_hit(fm) - pb_hit(S_am[ch])); crow.append({'fusion': fn, 'single': ch, 'metric': 'pb_argmax_hit', 'delta': pt, 'lo95': lo, 'hi95': hi})
    pd.DataFrame(crow).to_csv(OUT / 'FUSION_VS_SINGLE.csv', index=False)
    json.dump({'best_single_prmbench': best_p, 'best_single_processbench': best_b, 'flipped_channels_by_cell': flips, 'runtime_s': round(time.time() - t0),
               'parity_N16_HEM_MOM_cc': got}, open(OUT / 'STATUS.json', 'w'), indent=1)
    print('done', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
