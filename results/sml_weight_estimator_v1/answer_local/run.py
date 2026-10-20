"""Answer-local SML vs pooled SML vs equal, with shrinkage; answer-local L-SML feasibility.

Phase 1 (label-free): all scores/weights. Phase 2 (read_labels + evaluate): metrics only.
Loading mirrors results/sml_weight_estimator_v1/processbench_banks/score_banks.py.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '4'
import importlib.util, json, pickle, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path('/Users/osegev/Desktop/hallucination_detection')
SCR = Path('/private/tmp/claude-501/-Users-osegev-Desktop-hallucination-detection/98d49308-5ca3-48f7-a452-63da0bb7d2a1/scratchpad')
FEAT = SCR / 'pb_rank1/bank_old/FEATURES.npz'
PRM_RAW = REPO / 'dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl'
OUT = SCR / 'answer_local'
sys.path.insert(0, str(REPO))
import spectral_utils.fusion_utils as FU  # noqa: E402

fs = importlib.util.spec_from_file_location('prm_fuse', REPO / 'results/prm_boosting_v1/fuse.py')
FUSE = importlib.util.module_from_spec(fs); fs.loader.exec_module(FUSE); ext = FUSE.ext
BANKS = ['N13', 'N16', 'N25', 'N56']
LAMS = [.25, .5, .75, .9]
EPS = 8 * np.finfo(float).eps
BINS = [(1, 6, '<=6'), (7, 10, '7-10'), (11, 15, '11-15'), (16, 24, '16-24'), (25, 10 ** 9, '>=25')]

A = pd.read_csv(REPO / 'results/step_evidence_v1/OOF_ANSWERS.csv', encoding='utf-8-sig')
F = dict(np.load(FEAT))
assert list(map(str, F['uids'])) == A.uid.astype(str).tolist(), 'answer order differs from OOF_ANSWERS'
cells = A.cell.to_numpy(); N = len(A)
pop = ext.Population(F, BANKS)
NS = np.diff(F['offsets'].astype(np.int64))


def first_argmax(s):
    return int(np.flatnonzero(s >= s.max() - EPS)[0])


def lead_abs(M):
    """|leading eigvec| of offdiag(M) for a stack of symmetric matrices (..., m, m)."""
    M = M.copy(); idx = np.arange(M.shape[-1]); M[..., idx, idx] = 0.
    return np.abs(np.linalg.eigh(M)[1][..., :, -1])


def local_corr(X):
    """Within-answer correlation of X (n x m); constant channels get zero rows/cols (diag 1). Also returns Z."""
    n = len(X); mu = X.mean(0); sd = X.std(0)
    ok = sd > 1e-10; Z = np.zeros_like(X); Z[:, ok] = (X[:, ok] - mu[ok]) / sd[ok]
    S = Z.T @ Z / n; np.fill_diagonal(S, 1.)
    return S, Z, ok


def phase1():
    rec = {}  # (bank, method) -> per-answer argmax (int) ; PRMB step scores separately
    steps = {}  # (bank, method) -> step scores for prmbench cell (concatenated in cell order)
    info = []   # per answer per bank: n, cos(local,pooled), lam_ss, lam_an, rank, cell
    lsml_rows = []
    t0 = time.time()
    for bk in BANKS:
        for ce in sorted(set(cells)):
            ans = np.flatnonzero(cells == ce); bank = pop.cell_bank(bk, ans); V = np.asarray(bank['V'], float); off = bank['off']
            lev = pop.level[bank['rows']]
            sg = np.array([(np.sign(np.corrcoef(V[:, j], lev)[0, 1]) or 1.) if V[:, j].std() > 0 else 1. for j in range(V.shape[1])])
            Vo = V * sg; m = Vo.shape[1]
            Rp = np.corrcoef(Vo.T); Rp = np.nan_to_num(Rp); np.fill_diagonal(Rp, 1.)
            wp = lead_abs(Rp[None])[0]
            meths = {}
            def put(name, j, s):
                meths.setdefault(name, {})[j] = s
            for j, i in enumerate(ans):
                X = Vo[off[j]:off[j + 1]]; n = len(X)
                S, Z, ok = local_corr(X)
                offm = ~np.eye(m, dtype=bool)
                # Schafer-Strimmer empirical variance of each correlation entry, target = pooled R
                if n >= 3:
                    W = Z[:, :, None] * Z[:, None, :]
                    varr = n / (n - 1.) ** 3 * ((W - W.mean(0)) ** 2).sum(0)
                    Sr = S * n / (n - 1.)
                    den = ((Sr - Rp)[offm] ** 2).sum()
                    lam_ss = float(np.clip(varr[offm].sum() / den, 0, 1)) if den > 0 else 1.
                    van = (1 - np.clip(S, -1, 1) ** 2) ** 2 / (n - 1.)
                    lam_an = float(np.clip(van[offm].sum() / den, 0, 1)) if den > 0 else 1.
                else:
                    lam_ss = lam_an = 1.
                Ms = [S] + [l * Rp + (1 - l) * S for l in LAMS] + [lam_ss * Rp + (1 - lam_ss) * S, lam_an * Rp + (1 - lam_an) * S]
                Wt = lead_abs(np.stack(Ms))
                degenerate = not np.any(np.abs(S[offm]) > 1e-12)
                wl = Wt[0] if not degenerate else np.ones(m) / np.sqrt(m)
                names = ['LOCAL'] + [f'SHR_POOL_{l}' for l in LAMS] + ['SHR_POOL_SS', 'SHR_POOL_AN']
                for nm, w in zip(names, Wt):
                    if degenerate: w = np.ones(m)
                    put(nm, j, X @ w)
                put('EQ', j, X.mean(1)); put('POOLED', j, X @ wp)
                wl1 = wl / wl.sum()
                for l in LAMS:
                    put(f'SHR_EQ_{l}', j, X @ ((1 - l) * wl1 + l / m))
                cos = float(wl @ wp / (np.linalg.norm(wl) * np.linalg.norm(wp)))
                cos_ss = float(Wt[5] @ wp / (np.linalg.norm(Wt[5]) * np.linalg.norm(wp)))
                info.append({'bank': bk, 'cell': ce, 'answer': int(i), 'n': n, 'm': m, 'rank': min(n - 1, int(ok.sum())),
                             'cos_local_pooled': cos, 'cos_ss_pooled': cos_ss, 'lam_ss': lam_ss, 'lam_an': lam_an, 'degenerate': degenerate})
            for nm, d in meths.items():
                am = rec.setdefault((bk, nm), np.full(N, -9, int))
                for j, i in enumerate(ans): am[i] = first_argmax(d[j])
                if ce == 'prmbench_qwen3_8b':
                    steps[(bk, nm)] = [d[j] for j in range(len(ans))]
            # median |offdiag pooled R| (nuisance redundancy level)
            info.append({'bank': bk, 'cell': ce, 'answer': -1, 'pool_abs_offdiag_median': float(np.median(np.abs(Rp[~np.eye(m, dtype=bool)]))),
                         'pool_eig_top_frac': float(np.linalg.eigvalsh(Rp)[-1] / m),
                         'pool_eff_rank': float(np.linalg.eigvalsh(Rp).clip(0).sum() ** 2 / (np.linalg.eigvalsh(Rp).clip(0) ** 2).sum())})
            # store Vo for prmbench (signal-vs-noise) only
            if ce == 'prmbench_qwen3_8b':
                steps[(bk, '_Vo')] = [Vo[off[j]:off[j + 1]] for j in range(len(ans))]
            print(bk, ce, round(time.time() - t0), flush=True)
    return rec, steps, pd.DataFrame(info)


def phase1_lsml(n_per=500, seed=0):
    """Answer-local grouped L-SML (lsml_continuous defaults) on single answers, label-free."""
    rng = np.random.default_rng(seed)
    pr = np.flatnonzero(cells == 'prmbench_qwen3_8b'); pbs = np.flatnonzero(np.char.startswith(cells.astype(str), 'pb_'))
    samp = np.r_[rng.choice(pr, n_per, replace=False), rng.choice(pbs, n_per, replace=False)]
    out = []
    for bk in ['N13', 'N16']:
        for ce in sorted(set(cells[samp])):
            ans = samp[cells[samp] == ce]; ans.sort(); bank = pop.cell_bank(bk, np.flatnonzero(cells == ce))
            V = np.asarray(bank['V'], float); off = bank['off']; lev = pop.level[bank['rows']]
            sg = np.array([(np.sign(np.corrcoef(V[:, j], lev)[0, 1]) or 1.) if V[:, j].std() > 0 else 1. for j in range(V.shape[1])])
            Vo = V * sg; pos = {int(a): k for k, a in enumerate(np.flatnonzero(cells == ce))}
            for i in ans:
                j = pos[int(i)]; X = Vo[off[j]:off[j + 1]]; r = {'bank': bk, 'cell': ce, 'answer': int(i), 'n': len(X)}
                t = time.time()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        s, meta = FU.lsml_continuous(*[X[:, c] for c in range(X.shape[1])])
                    s = np.asarray(s, float); r['K'] = int(meta['K'])
                    if not np.all(np.isfinite(s)): r['status'] = 'nonfinite'
                    elif r['K'] < 3: r['status'] = 'K<3'
                    else: r['status'] = 'ok'; r['argmax'] = first_argmax(s); r['scores'] = s
                except Exception as exc:
                    r['status'] = 'exception:' + type(exc).__name__ + ':' + str(exc)[:80]
                r['sec'] = time.time() - t
                r['eq_scores'] = X.mean(1); r['eq_argmax'] = first_argmax(X.mean(1))
                out.append(r)
        print('lsml', bk, flush=True)
    return out


# ======================= labels: read only after all scores exist =======================
def read_labels():
    from spectral_utils.prm_label_contract import prm_error_flags
    target = A.target.to_numpy().astype(int)
    raw = pickle.load(open(PRM_RAW, 'rb')); rows = list(raw.values()) if isinstance(raw, dict) else raw
    err = {str(r['idx']): r.get('error_steps') for r in rows if isinstance(r, dict) and 'idx' in r}
    ids = A.id.astype(str).to_numpy(); flags = {}
    for i in np.flatnonzero(cells == 'prmbench_qwen3_8b'):
        es = err.get(ids[i])
        if es is None: raise SystemExit('missing error_steps for ' + ids[i])
        flags[int(i)] = np.asarray(prm_error_flags(es, int(NS[i])), bool)
    return target, flags


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def binof(n):
    for lo, hi, nm in BINS:
        if lo <= n <= hi: return nm


def evaluate(rec, steps, info, lsml):
    target, flags = read_labels()
    pr = np.flatnonzero(cells == 'prmbench_qwen3_8b'); pb_cells = sorted(c for c in set(cells) if c.startswith('pb_'))
    rows = []
    for (bk, nm), am in rec.items():
        aucs = np.array([auc(flags[int(i)], s) for i, s in zip(pr, steps[(bk, nm)])])
        hits = [float(np.mean(am[(cells == c) & (target >= 0)] == target[(cells == c) & (target >= 0)])) for c in pb_cells]
        r = {'bank': bk, 'method': nm, 'bin': 'ALL', 'prmb_within_auc': np.nanmean(aucs), 'prmb_n': int(np.isfinite(aucs).sum()),
             'pb_hit_cell_equal': float(np.mean(hits))}
        rows.append(r)
        pe = np.flatnonzero(np.char.startswith(cells.astype(str), 'pb_') & (target >= 0))
        for lo, hi, b in BINS:
            mpr = (NS[pr] >= lo) & (NS[pr] <= hi); mpb = pe[(NS[pe] >= lo) & (NS[pe] <= hi)]
            rows.append({'bank': bk, 'method': nm, 'bin': b, 'prmb_within_auc': np.nanmean(aucs[mpr]), 'prmb_n': int(np.isfinite(aucs[mpr]).sum()),
                         'pb_hit_pooled': float(np.mean(am[mpb] == target[mpb])), 'pb_n': len(mpb)})
    R = pd.DataFrame(rows); R.to_csv(OUT / 'RESULTS.csv', index=False)
    # weight stability
    I = info[info.answer >= 0].copy(); I['bin'] = I.n.map(binof)
    stab = I.groupby(['bank', 'bin']).agg(cos_local_med=('cos_local_pooled', 'median'), cos_ss_med=('cos_ss_pooled', 'median'),
                                          lam_ss_med=('lam_ss', 'median'), lam_an_med=('lam_an', 'median'), rank_med=('rank', 'median'),
                                          n_ans=('n', 'size')).reset_index()
    stab.to_csv(OUT / 'STABILITY.csv', index=False)
    info[info.answer < 0].dropna(axis=1, how='all').to_csv(OUT / 'POOLED_STRUCTURE.csv', index=False)
    # signal vs noise (PRMBench)
    sn = []
    for bk in BANKS:
        Vos = steps[(bk, '_Vo')]; per = []
        for i, X in zip(pr, Vos):
            y = flags[int(i)]; n = len(y)
            if y.all() or not y.any(): continue
            p = y.mean(); dX = X[y].mean(0) - X[~y].mean(0); per.append((n, p, dX))
        for lo, hi, b in BINS:
            sel = [t for t in per if lo <= t[0] <= hi]
            if not sel: continue
            dmean = np.mean([t[2] for t in sel], 0); m = len(dmean); offm = ~np.eye(m, dtype=bool)
            ins = [np.median(np.abs(t[1] * (1 - t[1]) * np.outer(t[2], t[2]))[offm]) for t in sel]
            sysm = [np.median(np.abs(t[1] * (1 - t[1]) * np.outer(dmean, dmean))[offm]) for t in sel]
            sysmax = [np.max(np.abs(t[1] * (1 - t[1]) * np.outer(dmean, dmean))[offm]) for t in sel]
            ns = np.array([t[0] for t in sel])
            sn.append({'bank': bk, 'bin': b, 'n_answers': len(sel), 'median_n': float(np.median(ns)), 'median_p': float(np.median([t[1] for t in sel])),
                       'mean_abs_dX_systematic': float(np.mean(np.abs(dmean))), 'max_abs_dX_systematic': float(np.max(np.abs(dmean))),
                       'mean_abs_dX_inanswer': float(np.mean([np.mean(np.abs(t[2])) for t in sel])),
                       'signal_inanswer_med': float(np.median(ins)), 'signal_systematic_med': float(np.median(sysm)),
                       'signal_systematic_maxpair_med': float(np.median(sysmax)),
                       'null_inanswer_expect_1_over_n': float(np.median(1. / ns)),
                       'SE_corr_1_over_sqrt_n_minus_3': float(np.median(1 / np.sqrt(np.maximum(ns - 3, 1))))})
    pd.DataFrame(sn).to_csv(OUT / 'SIGNAL_NOISE.csv', index=False)
    # answer-local L-SML
    L = []
    for r in lsml:
        i = r['answer']; d = {k: r.get(k) for k in ('bank', 'cell', 'answer', 'n', 'K', 'status', 'sec')}
        if r['cell'] == 'prmbench_qwen3_8b':
            y = flags[i]; d['eq_auc'] = auc(y, r['eq_scores']); d['lsml_auc'] = auc(y, r['scores']) if r['status'] == 'ok' else np.nan
        elif target[i] >= 0:
            d['eq_hit'] = float(r['eq_argmax'] == target[i]); d['lsml_hit'] = float(r['argmax'] == target[i]) if r['status'] == 'ok' else np.nan
        L.append(d)
    L = pd.DataFrame(L); L['bin'] = L.n.map(binof); L.to_csv(OUT / 'LSML_LOCAL.csv', index=False)
    return R, stab, pd.DataFrame(sn), L


if __name__ == '__main__':
    rec, steps, info = phase1()
    lsml = phase1_lsml()
    R, stab, sn, L = evaluate(rec, steps, info, lsml)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
    print(R[R.bin == 'ALL'].pivot(index='method', columns='bank', values=['prmb_within_auc', 'pb_hit_cell_equal']).round(4))
    print('DONE')
