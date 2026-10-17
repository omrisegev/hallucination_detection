# PATCHED COPY of results/monday_cont_baselines_v1/run.py for thinkfix_rescore_v1 (variant=thinkfix).
# Changes: V2 = this variant's nested_banks run; ext.FE set by the wrapper; export hook.
"""monday_cont_baselines_v1: continuous L-SML and IU-PCR on the v2 nested banks (development).

Same per-cell inputs (bank['V'] = channels as they are), same metrics and bootstrap as
monday_nested_banks_v2/dev_run.py. See PROTOCOL.json.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import hashlib, importlib.util, json, pickle, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent; MAIN = Path(r'/shared/cycle2_tau_averbuch_prj/omrisegev1/thinkfix_downstream/C:\Users\omris\TAU\hallucination_detection'); V2 = Path(r'/shared/cycle2_tau_averbuch_prj/omrisegev1/results/thinkfix_rescore_v1/thinkfix/nested_banks')
sys.path.insert(0, str(MAIN))
from spectral_utils.fusion_utils import lsml_continuous
from spectral_utils.upcr import upcr_fit
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS
spec = importlib.util.spec_from_file_location('dev_run', V2 / 'dev_run_patched.py'); dr = importlib.util.module_from_spec(spec); spec.loader.exec_module(dr)
ext = dr.ext; NB = dr.NB; BANKS = dr.BANKS; M4 = dr.METHODS4
BASE = ['LSML_CONT', 'IU_PCR']; SCORERS = BASE + [b + '_LEVELSIGN' for b in BASE] + ['ALL_EQ_ORIENTED']


def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit_lsml(V):
    s, meta = lsml_continuous(*[V[:, j] for j in range(V.shape[1])])
    K = int(meta['K'])
    if K < 3: raise RuntimeError(f'L-SML selected K={K} groups: fewer than 3 groups')
    return np.asarray(s, float), {'K': K, 'degenerate': bool(meta['degenerate']), 'group_sizes': np.bincount(np.unique(meta['c'], return_inverse=True)[1]).tolist()}


def fit_iu(V):
    sd = V.std(0); keep = sd > 0
    if keep.sum() < 3: raise RuntimeError('fewer than 3 non-constant channels')
    F = ((V[:, keep] - V[:, keep].mean(0)) / sd[keep]).T
    res = upcr_fit(F, **IU_FIT_DEFAULTS)
    if not np.all(np.isfinite(res.w)): raise RuntimeError('non-finite IU-PCR weights')
    return np.asarray(res.w @ F, float), {'n_neg_weights': int((res.w < 0).sum()), 'm': int(keep.sum()), 'abstained': bool(getattr(res, 'abstained', False))}


def main():
    t0 = time.time()
    P = json.loads((OUT / 'PROTOCOL.json').read_text(encoding='utf8')); assert P['status'] == 'FROZEN'
    if (OUT / 'PER_ANSWER.npz').exists(): raise SystemExit('outputs exist')
    A = pd.read_csv(dr.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); cells = A.cell.to_numpy(); N = len(A)
    ids = A.id.astype(str).to_numpy(); target = A.target.to_numpy().astype(int); groups = A.source_group.to_numpy()
    with np.load(dr.SE / 'OOF_STEP_SCORES.npz') as z: OFF = z['offsets'].astype(np.int64); LAB = z['labels'].astype(bool)
    pb = np.array([c.startswith('pb_') for c in cells])
    fr = json.loads((dr.SE / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
    meta = {m['idx']: m for m in pickle.load(open(fr['prm_metadata']['path'], 'rb')).values()}
    noncontrol = np.array([(not pb[i]) and meta[ids[i]]['classification'] != 'correct' for i in range(N)])
    F = np.load(ext.FE / 'source/FEATURES.npz'); pop = ext.Population(F, BANKS)
    PA = np.load(V2 / 'PER_ANSWER.npz')
    R = {(bk, s): {'auc': np.full(N, np.nan), 'top1err': np.full(N, np.nan), 'argmax': np.full(N, -9, int), 'c1': np.zeros((N, 4), int), 'failed': np.zeros(N, bool)}
         for bk in BANKS for s in SCORERS}
    diag, failures, parity = {}, [], []
    for bk in BANKS:
        for ce in sorted(set(cells)):
            ans = np.flatnonzero(cells == ce); bank = pop.cell_bank(bk, ans); V = bank['V']; off = bank['off']
            lev = pop.level[bank['rows']]
            # parity: the plain average of V must reproduce v2's ALL_EQ argmax on this cell
            eq = V.mean(1); am_eq = np.array([int(np.flatnonzero(eq[a:b] >= eq[a:b].max() - 8 * np.finfo(float).eps)[0]) for a, b in zip(off[:-1], off[1:])])
            parity.append({'bank': bk, 'cell': ce, 'ALL_EQ_argmax_mismatches': int((am_eq != PA[f'{bk}__ALL_EQ__argmax'][ans]).sum())})
            sg = np.array([np.sign(np.corrcoef(V[:, j], lev)[0, 1]) or 1. if V[:, j].std() > 0 else 1. for j in range(V.shape[1])])
            scores = {'ALL_EQ_ORIENTED': (V * sg).mean(1)}
            for name, fn in (('LSML_CONT', fit_lsml), ('IU_PCR', fit_iu)):
                t1 = time.time()
                try:
                    s, d = fn(V)
                    if not np.all(np.isfinite(s)): raise RuntimeError('non-finite score')
                    r = np.corrcoef(s, lev)[0, 1]; d['corr_with_level'] = float(r); d['seconds'] = round(time.time() - t1, 1)
                    scores[name] = s; scores[name + '_LEVELSIGN'] = s * (np.sign(r) or 1.)
                    diag[f'{bk}/{ce}/{name}'] = d
                except Exception as exc:
                    failures.append({'bank': bk, 'cell': ce, 'scorer': name, 'reason': f'{type(exc).__name__}: {exc}'[:300]})
                    diag[f'{bk}/{ce}/{name}'] = {'FAILED': str(exc)[:300]}
                    for nm in (name, name + '_LEVELSIGN'): R[(bk, nm)]['failed'][ans] = True
            _HOOK(bk, ce, bank, scores)
            for nm, s in scores.items():
                st = R[(bk, nm)]
                for j, i in enumerate(ans):
                    ss = s[off[j]:off[j + 1]]; y = LAB[OFF[i]:OFF[i + 1]]
                    am = int(np.flatnonzero(ss >= ss.max() - 8 * np.finfo(float).eps)[0]); st['argmax'][i] = am
                    if not pb[i]:
                        st['auc'][i] = auc(y, ss)
                        if y.any(): st['top1err'][i] = float(y[am])
                        if noncontrol[i]:
                            f = np.zeros(len(ss), bool); f[np.lexsort((np.arange(len(ss)), -ss))[:1]] = True
                            st['c1'][i] = dr.conf(f, y)
            print(bk, ce, {k: v for k, v in diag.items() if k.startswith(f'{bk}/{ce}/')}, round(time.time() - t0), flush=True)
    pd.DataFrame(parity).to_csv(OUT / 'PARITY_ALL_EQ.csv', index=False)
    (OUT / 'FIT_DIAGNOSTICS.json').write_text(json.dumps(diag, indent=1), encoding='utf8')
    (OUT / 'COVERAGE.json').write_text(json.dumps({'failures': failures}, indent=1), encoding='utf8')

    # ---- aggregation: identical weights to dev_run (same seed and draw order)
    prm_idx = np.flatnonzero(~pb); pb_idx = np.flatnonzero(pb)
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
    def prmscore(c1):
        tp, fp, tn, fn = c1[prm_idx].sum(0).astype(float)
        return .5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn))
    ci = lambda x: (float(x[0]), float(np.quantile(x[1:], .025)), float(np.quantile(x[1:], .975)))
    # the v2 ALL_EQ point estimate must be reproduced by the shared aggregation
    v2abs = pd.read_csv(V2 / 'PANEL_A_ABSOLUTE.csv')
    chk = max(abs(prm_auc(PA[f'{bk}__ALL_EQ__auc'])[0] - v2abs[(v2abs.bank == bk) & (v2abs.scorer == 'ALL_EQ')].prmb_within_auc.iloc[0]) for bk in BANKS)
    if chk > 1e-12: raise SystemExit(f'aggregation parity failed {chk}')
    rows, cons = [], []
    for bk in BANKS:
        cur = {}
        for nm in SCORERS:
            st = R[(bk, nm)]; f_prm = bool(st['failed'][prm_idx].any()); f_pb = bool(st['failed'][pb_idx].any())
            a = prm_auc(st['auc']) if not f_prm else None; h = pb_hit(st['argmax']) if not f_pb else None; cur[nm] = (a, h)
            rows.append({'bank': bk, 'n_channels': len(NB[bk]), 'scorer': nm, 'prmb_within_auc': a[0] if a is not None else np.nan,
                         'prmb_top1_on_error': float(np.nanmean(st['top1err'][prm_idx])) if not f_prm else np.nan,
                         'prmb_prmscore_top1': prmscore(st['c1']) if not f_prm else np.nan, 'pb_argmax_hit': h[0] if h is not None else np.nan,
                         'prmbench_complete': not f_prm, 'processbench_complete': not f_pb})
        refs = {'ALL_EQ': (prm_auc(PA[f'{bk}__ALL_EQ__auc']), pb_hit(PA[f'{bk}__ALL_EQ__argmax'])), 'ALL_EQ_ORIENTED': cur['ALL_EQ_ORIENTED']}
        for m in M4:
            fk = f'{bk}__{m}__failed'
            refs[m] = (prm_auc(PA[f'{bk}__{m}__auc']) if not PA[fk][prm_idx].any() else None, pb_hit(PA[f'{bk}__{m}__argmax']) if not PA[fk][pb_idx].any() else None)
        for b in SCORERS[:4]:
            for ref in ('ALL_EQ', 'ALL_EQ_ORIENTED'):
                for k, metric in ((0, 'prmb_within_auc'), (1, 'pb_argmax_hit')):
                    if cur[b][k] is None or refs[ref][k] is None: continue
                    pt, lo, hi = ci(cur[b][k] - refs[ref][k]); cons.append({'bank': bk, 'left': b, 'right': ref, 'metric': metric, 'delta': pt, 'lo95': lo, 'hi95': hi})
            for m in M4:
                for k, metric in ((0, 'prmb_within_auc'), (1, 'pb_argmax_hit')):
                    if cur[b][k] is None or refs[m][k] is None: continue
                    pt, lo, hi = ci(refs[m][k] - cur[b][k]); cons.append({'bank': bk, 'left': m, 'right': b, 'metric': metric, 'delta': pt, 'lo95': lo, 'hi95': hi})
    pd.DataFrame(rows).to_csv(OUT / 'ABSOLUTE.csv', index=False); pd.DataFrame(cons).to_csv(OUT / 'CONTRASTS.csv', index=False)
    np.savez_compressed(OUT / 'PER_ANSWER.npz', **{f'{bk}__{nm}__{f}': R[(bk, nm)][f] for bk in BANKS for nm in SCORERS for f in ('auc', 'argmax', 'top1err', 'failed')})
    (OUT / 'STATUS.json').write_text(json.dumps({'status': 'BUILT', 'runtime_s': round(time.time() - t0), 'n_failures': len(failures),
        'parity_ALL_EQ_argmax_mismatches': int(sum(p['ALL_EQ_argmax_mismatches'] for p in parity)), 'aggregation_parity_max_abs': chk,
        'run_sha256': sha(__file__), 'protocol_sha256': sha(OUT / 'PROTOCOL.json'), 'adapter_sha256': sha(dr.EXT / 'score.py')}, indent=1), encoding='utf8')
    print('done', round(time.time() - t0), 'failures', len(failures), flush=True)


if __name__ == '__main__':
    main()
