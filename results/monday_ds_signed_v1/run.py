"""monday_ds_signed_v1: the DS-signed average (Dawid-Skene sign per channel, then plain average) as the deck's
sign-corrected control, with every development metric the decks show. See PROTOCOL.json."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import importlib.util, json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent; MAIN = OUT.parents[1]
V2 = MAIN / 'results/monday_nested_banks_v2'; CB = MAIN / 'results/monday_cont_baselines_v1'
EO = MAIN / 'results/monday_est_orient_v1'; TS = MAIN / 'results/monday_types_singles_v1'
sys.path.insert(0, str(MAIN))
import spectral_utils.fusion_utils  # noqa: F401  (same import order as the earlier runs)
spec = importlib.util.spec_from_file_location('dev_run', V2 / 'dev_run.py'); dr = importlib.util.module_from_spec(spec); spec.loader.exec_module(dr)
ext = dr.ext; BANKS = dr.BANKS; M4 = dr.METHODS4
import er_stage_a as SA  # noqa: E402
SC = ['DS_SIGNED', 'MOM_SIGNED']


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    t0 = time.time(); assert json.loads((OUT / 'PROTOCOL.json').read_text(encoding='utf8'))['status'] == 'FROZEN'
    A = pd.read_csv(dr.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); cells = A.cell.to_numpy(); N = len(A)
    ids = A.id.astype(str).to_numpy(); target = A.target.to_numpy().astype(int); groups = A.source_group.to_numpy()
    with np.load(dr.SE / 'OOF_STEP_SCORES.npz') as z: OFF = z['offsets'].astype(np.int64); LAB = z['labels'].astype(bool)
    pb = np.array([c.startswith('pb_') for c in cells]); prm_idx = np.flatnonzero(~pb); pb_idx = np.flatnonzero(pb)
    fr = json.loads((dr.SE / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
    meta = {m['idx']: m for m in pickle.load(open(fr['prm_metadata']['path'], 'rb')).values()}
    noncontrol = np.array([(not pb[i]) and meta[ids[i]]['classification'] != 'correct' for i in range(N)])
    cls = np.array([meta[ids[i]]['classification'] for i in prm_idx])
    F = np.load(ext.FE / 'source/FEATURES.npz'); pop = ext.Population(F, BANKS)
    R = {(bk, s): {'auc': np.full(N, np.nan), 'top1err': np.full(N, np.nan), 'argmax': np.full(N, -9, int), 'c1': np.zeros((N, 4), int)} for bk in BANKS for s in SC}
    flips = []
    for bk in BANKS:
        for ce in sorted(set(cells)):
            ans = np.flatnonzero(cells == ce); bank = pop.cell_bank(bk, ans); V = bank['V']; off = bank['off']; marks = bank['marks']; names = bank['names']
            lev = pop.level[bank['rows']]
            lev_sg = np.array([(np.sign(np.corrcoef(V[:, j], lev)[0, 1]) or 1.) if V[:, j].std() > 0 else 1. for j in range(V.shape[1])])
            sg = {'DS_SIGNED': np.where(SA.em_estimate(marks, 'ds')['pi'] > .5, 1., -1.),
                  'MOM_SIGNED': np.where(SA.tensor_mom_estimate(marks, 0)['pi'] > .5, 1., -1.)}
            for nm, s in sg.items():
                flips.append({'bank': bk, 'cell': ce, 'scorer': nm, 'n_channels': len(names), 'n_flipped': int((s < 0).sum()), 'n_flipped_level_anchor': int((lev_sg < 0).sum()),
                              'n_level_anchor_flips_also_flipped': int(((s < 0) & (lev_sg < 0)).sum()),
                              'same_flip_set_as_level_anchor': bool(np.array_equal(s, lev_sg)), 'flipped': ';'.join(names[j] for j in np.flatnonzero(s < 0)),
                              'flipped_level_anchor': ';'.join(names[j] for j in np.flatnonzero(lev_sg < 0))})
                score = (V * s).mean(1); st = R[(bk, nm)]
                for q, i in enumerate(ans):
                    ss = score[off[q]:off[q + 1]]; y = LAB[OFF[i]:OFF[i + 1]]
                    am = int(np.flatnonzero(ss >= ss.max() - 8 * np.finfo(float).eps)[0]); st['argmax'][i] = am
                    if not pb[i]:
                        st['auc'][i] = auc(y, ss)
                        if y.any(): st['top1err'][i] = float(y[am])
                        if noncontrol[i]:
                            f = np.zeros(len(ss), bool); f[np.lexsort((np.arange(len(ss)), -ss))[:1]] = True
                            st['c1'][i] = dr.conf(f, y)
        print(bk, round(time.time() - t0), flush=True)
    pd.DataFrame(flips).to_csv(OUT / 'FLIPS.csv', index=False)

    # ---- aggregation: identical weights to dev_run (same seed and draw order)
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
    def prmscore(c1):
        tp, fp, tn, fn = c1[prm_idx].sum(0).astype(float)
        return .5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn))
    ci = lambda x: (float(x[0]), float(np.nanquantile(x[1:], .025)), float(np.nanquantile(x[1:], .975)))

    # parity gate vs monday_est_orient_v1 (DS_ORIENTED / MOM_ORIENTED)
    EA = pd.read_csv(EO / 'ABSOLUTE.csv'); worst = 0.
    for bk in BANKS:
        for nm, old in (('DS_SIGNED', 'DS_ORIENTED'), ('MOM_SIGNED', 'MOM_ORIENTED')):
            r = EA[(EA.bank == bk) & (EA.scorer == old)].iloc[0]
            worst = max(worst, abs(prm_auc(R[(bk, nm)]['auc'])[0] - r.prmb_within_auc), abs(pb_hit(R[(bk, nm)]['argmax'])[0] - r.pb_argmax_hit))
    if worst > 1e-12: raise SystemExit(f'parity with monday_est_orient_v1 failed: {worst}')

    PA = np.load(V2 / 'PER_ANSWER.npz'); PC = np.load(CB / 'PER_ANSWER.npz'); rows, cons = [], []
    for bk in BANKS:
        for nm in SC:
            st = R[(bk, nm)]
            rows.append({'bank': bk, 'scorer': nm, 'prmb_within_auc': prm_auc(st['auc'])[0], 'prmb_top1_on_error': float(np.nanmean(st['top1err'][prm_idx])),
                         'prmb_prmscore_top1': prmscore(st['c1']), 'pb_sla': pb_hit(st['argmax'])[0]})
        ds = (prm_auc(R[(bk, 'DS_SIGNED')]['auc']), pb_hit(R[(bk, 'DS_SIGNED')]['argmax']))
        refs = {'ALL_EQ': (prm_auc(PA[f'{bk}__ALL_EQ__auc']), pb_hit(PA[f'{bk}__ALL_EQ__argmax'])),
                'MOM_SIGNED': (prm_auc(R[(bk, 'MOM_SIGNED')]['auc']), pb_hit(R[(bk, 'MOM_SIGNED')]['argmax']))}
        for m in ('ALL_EQ_ORIENTED', 'IU_PCR', 'LSML_CONT'): refs[m] = (prm_auc(PC[f'{bk}__{m}__auc']), pb_hit(PC[f'{bk}__{m}__argmax']))
        for m in M4 + ['DS_FILTER_EQ']:
            fk = f'{bk}__{m}__failed'
            refs[m] = (prm_auc(PA[f'{bk}__{m}__auc']) if not PA[fk][prm_idx].any() else None, pb_hit(PA[f'{bk}__{m}__argmax']) if not PA[fk][pb_idx].any() else None)
        for m, (ra, rh) in refs.items():
            for metric, x, y in (('prmb_within_auc', ra, ds[0]), ('pb_sla', rh, ds[1])):
                if x is None: continue
                pt, lo, hi = ci(x - y); cons.append({'bank': bk, 'method': m, 'reference': 'DS_SIGNED', 'metric': metric, 'delta': pt, 'lo95': lo, 'hi95': hi})
    pd.DataFrame(rows).to_csv(OUT / 'ABSOLUTE.csv', index=False); pd.DataFrame(cons).to_csv(OUT / 'CONTRASTS.csv', index=False)
    np.savez_compressed(OUT / 'PER_ANSWER.npz', **{f'{bk}__{nm}__{f}': R[(bk, nm)][f] for bk in BANKS for nm in SC for f in ('auc', 'argmax', 'top1err')})

    # ---- error types: DS_SIGNED replaces ALL_EQ_ORIENTED in the 80 configurations
    TA = pd.read_csv(TS / 'TYPES_ALL_CONFIGS.csv'); types = sorted(set(cls) - {'correct', 'multi_solutions'}); trows = []
    for t in types:
        mask = cls == t; rp = prm_auc(PA['PRM__auc'], mask)
        chk = TA[(TA.type == t) & (TA.scorer == 'PRM')].auc.iloc[0]
        if abs(rp[0] - chk) > 1e-12: raise SystemExit(f'types parity failed {t}')
        n_t = int(TA[TA.type == t].n_answers.iloc[0])
        for bk in BANKS:
            v = prm_auc(R[(bk, 'DS_SIGNED')]['auc'], mask); pt, lo, hi = ci(v - rp)
            trows.append({'type': t, 'n_answers': n_t, 'bank': bk, 'scorer': 'DS_SIGNED', 'family': 'DS_SIGNED', 'auc': v[0], 'auc_lo': float(np.nanquantile(v[1:], .025)),
                          'auc_hi': float(np.nanquantile(v[1:], .975)), 'minus_PRM': pt, 'minus_PRM_lo': lo, 'minus_PRM_hi': hi})
    TN = pd.concat([TA[TA.scorer != 'ALL_EQ_ORIENTED'], pd.DataFrame(trows)], ignore_index=True); TN.to_csv(OUT / 'TYPES_ALL_CONFIGS_DS.csv', index=False)
    L = TN[TN.family != 'reference']; summ = []
    for t in types:
        x = L[L.type == t]; prm = TN[(TN.type == t) & (TN.scorer == 'PRM')].iloc[0]; si = TN[(TN.type == t) & (TN.scorer == 'STEP_INDEX')].iloc[0]
        summ.append({'type': t, 'n_answers': int(x.n_answers.iloc[0]), 'n_configs': len(x), 'clearly_above_PRM': int((x.minus_PRM_lo > 0).sum()),
                     'clearly_below_PRM': int((x.minus_PRM_hi < 0).sum()), 'median_minus_PRM': float(x.minus_PRM.median()),
                     'min_minus_PRM': float(x.minus_PRM.min()), 'max_minus_PRM': float(x.minus_PRM.max()), 'PRM_auc': prm.auc, 'PRM_lo': prm.auc_lo, 'PRM_hi': prm.auc_hi,
                     'step_index_auc': si.auc})
    pd.DataFrame(summ).to_csv(OUT / 'TYPES_SUMMARY_DS.csv', index=False)

    # ---- N16: DS_SIGNED vs level-anchor oriented, per answer (singles figure reuses the N16 oriented value)
    n16_same = bool(np.array_equal(np.nan_to_num(R[('N16', 'DS_SIGNED')]['auc'], nan=-1), np.nan_to_num(PC['N16__ALL_EQ_ORIENTED__auc'], nan=-1))
                    and np.array_equal(R[('N16', 'DS_SIGNED')]['argmax'], PC['N16__ALL_EQ_ORIENTED__argmax']))
    (OUT / 'STATUS.json').write_text(json.dumps({'status': 'BUILT', 'runtime_s': round(time.time() - t0), 'parity_est_orient_max_abs': worst,
                                                 'N16_DS_SIGNED_identical_to_level_oriented_per_answer': n16_same}, indent=1), encoding='utf8')
    print('done', round(time.time() - t0), 'N16 identical', n16_same, flush=True)


if __name__ == '__main__':
    main()
