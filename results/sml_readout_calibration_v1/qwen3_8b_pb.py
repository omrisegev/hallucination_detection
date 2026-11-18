"""ProcessBench, Mind the Gap's scorer Qwen3-8B (thinkfix mask1 bank). This bank stores one value per step and channel,
not the token readout grid, so only A0 (as stored) and A4 (residualize on the nuisance covariates) can run. The bank has
no token count: A4 here residualizes on relative position and its square only. Same cell construction as
gsp_ablation_baseline_v1/qwen3_8b_pb.py. No label is read before scoring."""
import sys, types
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibrate as CB  # noqa: E402
AB, T, E, Rn = CB.AB, CB.T, CB.E, CB.Rn
F = np.load(Path(E.DATA) / 'thinkfix_source_bank_mask1' / 'FEATURES.npz')
cells = F['cells'].astype(str); keep = np.char.endswith(cells, '_q8') & np.char.startswith(cells, 'pb_')
rid = F['row_ids'].astype(str); off = F['offsets'].astype(np.int64); ne = F['nonempty'].astype(bool)
L = F['nonempty_lengths'].astype(np.int64); fst = np.concatenate([[0], np.cumsum(L)])
names = list(F['names'].astype(str))
ids, rows, pos, offs = [], [], [], [0]
for j in np.flatnonzero(keep):
    u = rid[j].split('::', 1)[1]
    full = np.flatnonzero(ne[fst[j]:fst[j + 1]]); assert len(full) == off[j + 1] - off[j]
    ids.append(u); rows.append(np.arange(off[j], off[j + 1])); pos.append(full); offs.append(offs[-1] + len(full))
rows = np.concatenate(rows)
c = types.SimpleNamespace(ids=ids, off=np.array(offs))
V = F['values'][rows]; act = F['digit_active'][rows].astype(bool)
rv = np.column_stack([V[:, names.index(n)] for n in T.N16]).astype(float)
G = np.column_stack([T.rank_gauss(rv[:, j], act[:, T.DIG.index(n)] if n in T.DIG else None) for j, n in enumerate(T.N16)])
X = np.column_stack([E.answer_z(G[:, j], c.off) for j in range(16)])
n = np.diff(c.off); k = np.arange(c.off[-1]) - np.repeat(c.off[:-1], n)
rel = k / np.maximum(np.repeat(n, n) - 1, 1)
Z = np.column_stack([np.ones(len(rel)), rel, rel ** 2])
Xr = np.column_stack([CB.residualize(X[:, j], Z, c.off) for j in range(16)])
lab = np.array([T.A.PB_GOLD[u]['label'] for u in ids])
rows_out = []
for arm, Xt in (('A0 top10', X), ('A4 residualize (position only)', Xr)):
    CB.RAW.clear(); tel = Rn.c4u_t(c, Xt)
    for cnt, f in (('GAP', AB.gap(c, tel['rank'])), ('own count', tel['flags'])):
        pr = T.pb_predict(c, f); pr = np.array([pos[j][p] if p >= 0 else -1 for j, p in enumerate(pr)])
        m = T.pb_metrics(c, pr)
        rows_out.append(dict(bench='processbench', prm='qwen3_8b (MtG scorer)', arm=f'{arm} | {cnt}', metric='macro SLA', value=m['macro SLA']))
    # position-matched FE-AUC on full step positions (nonempty steps only)
    num = den = 0.
    fullpos = np.concatenate(pos)
    for kk in range(int(fullpos.max()) + 1):
        P = [tel['rank'][c.off[j] + np.flatnonzero(pos[j] == kk)[0]] for j in range(len(ids)) if lab[j] == kk and kk in pos[j]]
        Q = [tel['rank'][c.off[j] + np.flatnonzero(pos[j] == kk)[0]] for j in range(len(ids)) if (lab[j] < 0 or lab[j] > kk) and kk in pos[j]]
        if P and Q:
            Q = np.sort(Q); P = np.array(P)
            num += ((np.searchsorted(Q, P, 'left') + np.searchsorted(Q, P, 'right')) / 2 / len(Q)).sum(); den += len(P)
    extra = {'FE-AUC position-matched': num / den, 'misfit (label-free)': CB.misfit(Xt),
             'mean |corr(col, rel_pos)| (label-free)': float(np.mean([abs(np.corrcoef(Xt[:, j], rel)[0, 1]) for j in range(16)])),
             'n psi/eta out of [0,1]': int(np.sum((CB.RAW['psi'] < 0) | (CB.RAW['psi'] > 1)) + np.sum((CB.RAW['eta'] < 0) | (CB.RAW['eta'] > 1))),
             'max raw eta': float(np.max(CB.RAW['eta'])), 'K groups': len(tel['groups'])}
    for mk, mv in extra.items():
        rows_out.append(dict(bench='processbench', prm='qwen3_8b (MtG scorer)', arm=arm, metric=mk, value=float(mv)))
    print(arm, [(r['arm'], r['metric'], round(r['value'], 3)) for r in rows_out if r['arm'].startswith(arm)], flush=True)
E.write_csv(HERE / 'RESULTS_qwen3_8b_pb.csv', rows_out)
