"""Diagnosis: why GSP ranks worse than L-SML / equal mean on Qwen3-8B ProcessBench. First-error AUC = within each
erroneous answer with >= 1 clean step before the first error, P(score(first error) > score(earlier clean step)),
averaged over answers (ties 1/2). Valid on ProcessBench (uses only steps up to the first error).
Also: SLA of argmax, GSP groups, signs and group weights."""
import runpy, json
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
g = runpy.run_path(str(HERE / 'qwen3_8b_pb.py'))
c, pos, T, X, tel, N = g['c'], g['pos'], g['T'], g['X'], g['tel'], g['AB'].N
lab = np.array([T.A.PB_GOLD[u]['label'] for u in c.ids])


def fe_auc(s):
    v = []
    for j, (a, b) in enumerate(zip(c.off[:-1], c.off[1:])):
        if lab[j] < 0: continue
        r = np.flatnonzero(pos[j] == lab[j])
        if not len(r): continue
        k = r[0]
        if k == 0: continue
        x, y = s[a + k], s[a:a + k]
        v.append(np.mean((x > y) + .5 * (x == y)))
    return float(np.mean(v)), len(v)


def sla(s):
    pr = np.array([pos[j][int(np.argmax(s[a:b]))] for j, (a, b) in enumerate(zip(c.off[:-1], c.off[1:]))])
    return T.pb_metrics(c, pr)['macro SLA']


s_lsml, grp, _ = N.lsml(X, X.mean(1))
cand = {'GSP rank': tel['rank'], 'L-SML score (same groups, before GSP)': s_lsml, 'equal mean of 16': X.mean(1)}
out = {}
for k, s in cand.items():
    a, n = fe_auc(s); out[k] = dict(first_error_AUC=a, n_answers=n, SLA_argmax=sla(s))
for j, nm in enumerate(T.N16):
    a, n = fe_auc(X[:, j]); out[nm] = dict(first_error_AUC=a, n_answers=n, SLA_argmax=sla(X[:, j]))
sign = [1. if np.corrcoef(X[:, j], s_lsml)[0, 1] >= 0 else -1. for j in range(16)]
info = dict(groups={int(k): [T.N16[i] for i in np.flatnonzero(grp == k)] for k in np.unique(grp[grp >= 0])},
            ungrouped=[T.N16[i] for i in np.flatnonzero(grp < 0)], signs=dict(zip(T.N16, sign)),
            anchor_group=tel['anchor_group'], keys=[k for k in tel if k not in ('rank', 'flags', 'k')])
for k in ('psi', 'eta', 'group_weights', 'w', 'lw'):
    if k in tel: info[k] = np.asarray(tel[k]).tolist()
json.dump(dict(scores=out, gsp=info), open(HERE / 'QWEN3_8B_DIAG.json', 'w'), indent=1, default=str)
for k, v in out.items(): print(f"{k:40s} FE-AUC {v['first_error_AUC']:.3f}  SLA {v['SLA_argmax']:.1f}")
print(json.dumps(info, default=str)[:3000])
