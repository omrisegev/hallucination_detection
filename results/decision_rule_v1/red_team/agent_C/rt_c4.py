"""Red-team C, part 4: paired source-group bootstrap for the decomposition contrasts."""
import json
import numpy as np, pandas as pd
exec(open(r'C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/41041b9e-b636-4238-bb1a-111fd1f803fe/scratchpad/rt_dr_C/rt_c_setup.py').read())
c0 = cnt['R0_frozen']; c2 = cnt['R2_allocate']; d = c2 - c0
zS = np.empty(S_)
for i in range(n):
    s = S[off[i]:off[i + 1]]; zS[off[i]:off[i + 1]] = (s - s.mean()) / max(s.std(), 1e-8)
def r0_at(q):
    f = np.zeros(S_, bool)
    for k in range(5):
        c = (k + 1) % 5; calm = prms & (fold[aid] == c); evm = fold[aid] == k; f[evm] = zS[evm] >= float(np.quantile(zS[calm], q))
    return f
inc = np.zeros(n)
for i in np.flatnonzero(prm):
    m = prm & (fold != fold[i]) & (ns == ns[i]); inc[i] = d[m].mean() if m.any() else d[prm & (fold != fold[i])].mean()
rules = {'R0': F['R0_frozen'], 'R1': F['R1_global'], 'R2': F['R2_allocate'], 'R0_rate_matched_q0.7862': r0_at(0.7862431251167554), 'R0_best_q0.845_ORACLE': r0_at(0.845),
         'a5_R0_plus_len_increment': by_count(np.clip(c0 + np.floor(inc + 0.5).astype(int), 0, ns))}
grp = ans.source_group.to_numpy(); ii = np.flatnonzero(nc); Gu, gi = np.unique(grp[ii], return_inverse=True); gmap = np.full(n, -1); gmap[ii] = gi
good = ~lab
def agg(f):
    v = ~f; a = np.zeros((len(Gu), 4)); m = ncs
    for j, x in enumerate([v & good, v & ~good, ~v & ~good, ~v & good]):
        a[:, j] = np.bincount(gmap[aid[m]], weights=x[m], minlength=len(Gu))
    return a
AG = {r: agg(f) for r, f in rules.items()}
def prm_of(c):
    tp, fp, tn, fn = np.moveaxis(c, -1, 0); return 0.5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fn + fp))
rng = np.random.default_rng(777005); B = 4000; W = rng.multinomial(len(Gu), np.full(len(Gu), 1 / len(Gu)), size=B).astype(float)
bs = {r: prm_of(W @ AG[r]) for r in rules}; pt = {r: float(prm_of(AG[r].sum(0))) for r in rules}
out = {'point': pt, 'groups': int(len(Gu))}
for a, b in [('R2', 'R0'), ('R1', 'R0'), ('R0_rate_matched_q0.7862', 'R0'), ('R2', 'R0_rate_matched_q0.7862'), ('R1', 'R0_rate_matched_q0.7862'),
             ('R2', 'R0_best_q0.845_ORACLE'), ('R2', 'a5_R0_plus_len_increment'), ('a5_R0_plus_len_increment', 'R0')]:
    dd = bs[a] - bs[b]; out[f'{a} - {b}'] = dict(delta=pt[a] - pt[b], ci95=[float(np.quantile(dd, .025)), float(np.quantile(dd, .975))])
print(json.dumps(out, indent=1))
json.dump(out, open(OUT / 'rt_c4_results.json', 'w'), indent=1)
