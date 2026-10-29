"""Does the per-answer z-score create dependence between the telemetry channels? (Omri, 2026-10-29)

Diagnostic, CPU only, stored per-step features, no inference. Labels are used only to split steps by class and to score.

Three representations of the same 16 N16 channels (+ the PRM risk on ProcessBench):
  AZ      per-answer z-score: (x - answer mean) / answer sd. This is what every fusion uses today
          (reconstructed from the raw features and checked against the stored channel files).
  CENTER  per-answer centring only, then one pooled sd per channel (no per-answer division).
  RAW     pooled z-score: one global mean and sd per channel; nothing per answer.
Digit channels use their active steps only, as in build_bank (inactive steps = 0).

Dependence = mean |Pearson r| between channels inside one class (error steps / correct steps), pooled over steps,
i.e. the class-conditional dependence that SML assumes is zero. Also by answer length.
Quality = within-answer AUC (mean over answers with both classes) and pooled step AUC, for MEAN16 (plain average),
TELE16 (L-SML, free grouping, no PRM) and, on ProcessBench, AUTO (PRM + 16 channels, free grouping).

Cells: PRMBench with each PRM's base LM as standalone telemetry (raw banks restored from Drive
cluster_results/prm_standalone_telemetry_v1/bank_<prm>; the PRM-backbone PRMBench raw banks are only on AIRCC, which was
unreachable), ProcessBench with each of the 4 PRM backbones (hd_collection_data/sub5_all_v1/pb_prm).

    python zscore_dependence.py
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'bracha_lr_transfer_v1'))
import lr_transfer as L  # noqa: E402
E, S8 = L.E, L.S8
from sklearn.metrics import roc_auc_score  # noqa: E402

TAU = Path(r'C:\Users\omris\TAU')
PB = {'qwen_prm_7b': 'bank_qwen_prm', 'qwen_prm_72b': 'bank_qwen_prm_72b', 'pure_prm': 'bank_pure_prm', 'rlhflow_prm': 'bank_rlhflow_prm'}
N16 = json.loads((HERE.parents[1] / 'results/monday_nested_banks_v2/BANKS.json').read_text())['N16']
DIG = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
LEN_BINS = [(2, 3), (4, 6), (7, 10), (11, 10 ** 6)]


def raw_matrix(F, rows):
    nm = [str(x) for x in F['names']]
    X = F['values'][rows][:, [nm.index(c) for c in N16]].astype(float)
    act = np.ones_like(X, bool)
    act[:, [N16.index(c) for c in DIG]] = F['digit_active'][rows].astype(bool)
    return X, act


def represent(X, act, off, how):
    """X raw (steps x channels), act = which entries are defined. Returns the representation; undefined -> 0."""
    out = np.zeros_like(X)
    if how == 'RAW':
        for j in range(X.shape[1]):
            v = X[act[:, j], j]; sd = v.std()
            if sd > 1e-12: out[act[:, j], j] = (v - v.mean()) / sd
        return out
    for a, b in zip(off[:-1], off[1:]):
        for j in range(X.shape[1]):
            m = act[a:b, j]; y = X[a:b, j][m]
            if not len(y): continue
            if how == 'AZ':
                if y.std() > 1e-12: out[a:b, j][m] = (y - y.mean()) / y.std()
            else:                                                   # CENTER
                out[a:b, j][m] = y - y.mean()
    if how == 'CENTER':
        sd = np.array([out[act[:, j], j].std() for j in range(X.shape[1])]); out = out / np.where(sd > 1e-12, sd, 1.)
    return out


def score_1d(s, off, how):
    return represent(s[:, None], np.ones((len(s), 1), bool), off, how)[:, 0]


def mean_abs_r(M):
    M = M[:, M.std(0) > 1e-12]
    R = np.corrcoef(M.T); iu = np.triu_indices_from(R, 1)
    return float(np.mean(np.abs(R[iu]))), float(np.mean(R[iu]))


def pooled_auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float('nan')


def run(tag, bench, c, X, act, prm=None):
    off = c.off; n = np.diff(off); lens = np.repeat(n, n)
    keep = c.keep_gold.astype(bool) if bench == 'processbench' else np.ones(off[-1], bool)   # PB: prefix domain
    if bench == 'prmbench':
        keep = keep & np.repeat(c.cls != 'correct', n)               # non-control answers
    y = c.err.astype(bool)
    rows_dep, rows_auc = [], []
    stored = E.channels(c, 'N16')
    for how in ('AZ', 'CENTER', 'RAW'):
        V = represent(X, act, off, how)
        if how == 'AZ' and stored is not None:
            assert np.max(np.abs(V - stored)) < 1e-4, f'{tag}: reconstructed answer-z differs from the stored channels'
        for cls_name, m in (('error_steps', keep & y), ('correct_steps', keep & ~y)):
            r_abs, r_sgn = mean_abs_r(V[m])
            d = dict(cell=tag, representation=how, class_=cls_name, n_steps=int(m.sum()), mean_abs_r=r_abs, mean_r=r_sgn)
            for lo, hi in LEN_BINS:
                mm = m & (lens >= lo) & (lens <= hi)
                d[f'mean_abs_r_len{lo}-{hi if hi < 999 else "+"}'] = mean_abs_r(V[mm])[0] if mm.sum() > 50 else float('nan')
            rows_dep.append(d)
        arms = {'MEAN16': V.mean(1)}
        arms['TELE16'], dt = S8.fit(V, anchor=V.mean(1)); Ktele = int(dt['K'])
        Kauto = None
        if prm is not None:
            zp = score_1d(prm, off, how)
            arms['PRM'] = zp
            arms['AUTO'], da = S8.auto(V, zp); Kauto = da['K']
        for a, s in arms.items():
            wa = E.per_answer_auc(c.err, s, off, keep)
            rows_auc.append(dict(cell=tag, representation=how, arm=a, within_auc=float(np.nanmean(wa)),
                                 pooled_auc=pooled_auc(y[keep], s[keep]),
                                 K=Ktele if a == 'TELE16' else Kauto if a == 'AUTO' else None))
        print(tag, how, ' '.join(f"{r['arm']} w{r['within_auc']:.4f} p{r['pooled_auc']:.4f}" for r in rows_auc if r['representation'] == how),
              '| |r| err/ok', ' '.join(f"{d['mean_abs_r']:.3f}" for d in rows_dep if d['representation'] == how), flush=True)
    return rows_dep, rows_auc


def main():
    dep, auc = [], []
    for prm in PB:                                                  # PRMBench, the PRM's own base LM as standalone telemetry
        F = np.load(TAU / 'hd_collection_data/prm_standalone_banks' / f'bank_{prm}' / 'FEATURES.npz')
        c = L.load('prmbench', prm, 'sa')
        off_all = F['offsets'].astype(np.int64); rid = F['row_ids'].astype(str); pos = {r: i for i, r in enumerate(rid)}
        rows = np.concatenate([np.arange(off_all[pos[u]], off_all[pos[u] + 1]) for u in c.ids]); assert len(rows) == c.off[-1]
        X, act = raw_matrix(F, rows)
        d, a = run(f'prmbench/{prm}_standalone', 'prmbench', c, X, act); dep += d; auc += a
    for prm, bank in PB.items():
        c = L.load('processbench', prm, 'prm')
        G = np.load(TAU / 'hd_collection_data/sub5_all_v1/pb_prm' / bank / 'FEATURES.npz')
        u = G['uids'].astype(str); o = G['offsets'].astype(np.int64); p = {x: i for i, x in enumerate(u)}
        rows = np.concatenate([np.arange(o[p[x]], o[p[x] + 1]) for x in c.ids]); assert len(rows) == c.off[-1]
        X, act = raw_matrix(G, rows)
        d, a = run(f'processbench/{prm}', 'processbench', c, X, act, prm=np.asarray(c.arms['PRM'], float)); dep += d; auc += a
    E.write_csv(HERE / 'DEPENDENCE.csv', dep); E.write_csv(HERE / 'AUC.csv', auc)


if __name__ == '__main__':
    main()
