"""GSP on Mind the Gap's own scorer model: ProcessBench teacher-forced telemetry of Qwen3-8B (thinkfix, mask1 bank,
cells pb_*_q8; 3,400 answers). Same R1 construction as bracha_panel.sa_X (global rank -> normal over the 3,400
answers with the digit-active mask, then answer-z) and the same arms as ablation.py / pb_cp.py, without the PRM.
Bank rows are the nonempty steps; a prediction maps back to the full step position before scoring.
Metric: macro SLA (Mind the Gap) and ProcessBench average F1 via c4u_competitors.pb_metrics."""
import sys, types
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation as AB  # noqa: E402
import pb_cp as CP  # noqa: E402
T, E, Rn = AB.T, AB.E, AB.Rn
F = np.load(Path(E.DATA) / 'thinkfix_source_bank_mask1' / 'FEATURES.npz')
cells = F['cells'].astype(str); keep = np.char.endswith(cells, '_q8') & np.char.startswith(cells, 'pb_')
rid = F['row_ids'].astype(str); off = F['offsets'].astype(np.int64); ne = F['nonempty'].astype(bool)
L = F['nonempty_lengths'].astype(np.int64); fst = np.concatenate([[0], np.cumsum(L)])
names = list(F['names'].astype(str))
ids, rows, pos, offs = [], [], [], [0]
for j in np.flatnonzero(keep):
    u = rid[j].split('::', 1)[1]; g = T.A.PB_GOLD[u]
    assert L[j] == g['n_steps'], (u, L[j], g['n_steps'])
    full = np.flatnonzero(ne[fst[j]:fst[j + 1]]); assert len(full) == off[j + 1] - off[j]
    ids.append(u); rows.append(np.arange(off[j], off[j + 1])); pos.append(full); offs.append(offs[-1] + len(full))
rows = np.concatenate(rows); assert len(ids) == 3400 and len(set(ids)) == 3400
c = types.SimpleNamespace(ids=ids, off=np.array(offs))
V = F['values'][rows]; act = F['digit_active'][rows].astype(bool)
rv = np.column_stack([V[:, names.index(n)] for n in T.N16]).astype(float)
G = np.column_stack([T.rank_gauss(rv[:, j], act[:, T.DIG.index(n)] if n in T.DIG else None) for j, n in enumerate(T.N16)])
X = np.column_stack([E.answer_z(G[:, j], c.off) for j in range(16)])
tel = Rn.c4u_t(c, X)
arms = {'GSP-T own count': tel['flags'], 'GSP-T + GAP': AB.gap(c, tel['rank'])}
fw1, fw2, _ = CP.free_init(tel['rank'], c.off)
for k, kw in (('CP GSP-T rank (free EM)', {}), ('CP GSP-T rank (free, pi = 1/2)', dict(pi=.5))):
    arms[k] = CP.to_flags(CP.R.map_pred(CP.em(tel['rank'], c.off, fw1, fw2, B=10, iters=30, h=CP.h_from(fw1, c.off), **kw)), c.off)
rows_out = []
for k, f in arms.items():
    pr = T.pb_predict(c, f)
    pr = np.array([pos[j][p] if p >= 0 else -1 for j, p in enumerate(pr)])        # nonempty row -> full step position
    m = T.pb_metrics(c, pr)
    rows_out.append(dict(bench='processbench', scorer='Qwen3-8B (thinkfix)', arm=k, macro_SLA=m['macro SLA'],
                         lo=m['macro SLA lo'], hi=m['macro SLA hi'], avg_F1=m['average F1'],
                         no_error_pct=100 * float(np.mean(pr < 0)), steps_dropped_empty=int(L[keep].sum() - len(rows))))
    print(k, {x: round(v, 2) for x, v in rows_out[-1].items() if isinstance(v, float)}, flush=True)
E.write_csv(HERE / 'PB_QWEN3_8B.csv', rows_out)
print('prevalence', tel['prevalence'], 'K', len(tel['groups']))
