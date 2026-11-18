"""Simulated step tables for a dry run of run_analysis.py (2026-11-18). NOT data: token values are generated
around the stored per-step mean readouts; V2/V1 follow the real (PB) or simulated (PRMBench) sidecars.

    python make_sim.py <stored pb7.npz> <pbcell7.npz> <prmb7.npz> <real make_variants out (PB sidecars + SPLIT)> <OUT>
"""
import json, sys
from pathlib import Path
import numpy as np

pb7, pbc, prm7, var, OUT = map(Path, sys.argv[1:6])
OUT = Path(OUT); (OUT / 'steps').mkdir(parents=True, exist_ok=True); (OUT / 'inputs').mkdir(exist_ok=True)
rng = np.random.default_rng(7); K = 48


def gen_tokens(mu, d, n):
    """(11,) step means, (11,) spreads, n tokens -> sorted top-K (11, K), stats (11, 4)."""
    v = mu[None] + d[None] * (rng.exponential(1., (n, 11)) - 1.)
    s = -np.sort(-v, 0); top = np.full((11, K), np.nan, np.float32); top[:, :min(n, K)] = s[:K].T
    st = np.stack([s[:min(10, n)].mean(0), v.mean(0), np.median(v, 0), np.percentile(v, 90, 0)], 1)
    return top, st


def build(uids, steps_of, bench, V, step5_of, act_of):
    """steps_of(u) -> list of (mu (11,), d (11,), ntok, bump); writes STEPS npz."""
    T, S, S5, A, NT, OFF = [], [], [], [], [], [0]
    for u in uids:
        for (mu, d, n, bump), s5, a in zip(steps_of(u), step5_of(u), act_of(u)):
            t, st = gen_tokens(mu + bump * d, d, n); T.append(t); S.append(st); S5.append(s5); A.append(a); NT.append(n)
        OFF.append(len(T))
    ne = np.ones(len(T), bool)
    np.savez_compressed(OUT / 'steps' / f'STEPS_{bench}_{V}.npz', uids=np.array(uids), offsets=np.array(OFF), nonempty=ne,
                        ntok=np.array(NT), nchar=np.array(NT) * 3, top=np.stack(T), stats=np.stack(S).astype(np.float32),
                        step5=np.array(S5, np.float32), digit_active=np.array(A, bool),
                        info=np.array(json.dumps(dict(sim=True, gate=dict(status='PASS')))))
    print(bench, V, len(uids), len(T), flush=True)


def side(bench, V):
    return {s['uid']: s for s in map(json.loads, open(var / f'SIDECAR_{bench}_{V}.jsonl')) if 'skip' not in s}


# ---------------------------------------------------------------- ProcessBench: real uids, real split and sidecars
D = np.load(pb7); C = np.load(pbc)
ids = D['uids'].astype(str); off = D['offsets']; mean = D['readouts'][:, :, 1]; d = (D['readouts'][:, :, 0] - mean).mean(0) / 2.5
ntok = D['ntok']; s5 = C['sa_N16'][:, 11:].astype(float); act = C['digit_active']
assert (C['sa_ids'].astype(str) == ids).all()
pos = {u: j for j, u in enumerate(ids)}
rows = lambda u: np.arange(off[pos[u]], off[pos[u] + 1])
pb_steps = lambda u: [(mean[r], d, int(ntok[r]), 0.) for r in rows(u)]
build(list(ids), pb_steps, 'processbench', 'C', lambda u: s5[rows(u)], lambda u: act[rows(u)])
S2 = side('processbench', 'V2'); S1 = side('processbench', 'V1')


def v2_parts(sc, parent_rows, donor_rows):
    k = sc['k'] - 1; r = list(parent_rows); dr = donor_rows[sc['donor_step'] - 1]
    return r, k, dr


def pb_v2(u):
    sc = S2[u]; r, k, dr = v2_parts(sc, rows(sc['parent_uid']), rows(sc['donor_uid']))
    return [(mean[dr], d, int(ntok[dr]), .6) if i == k else (mean[x], d, int(ntok[x]), 0.) for i, x in enumerate(r)]


def pb_v2_s5(u):
    sc = S2[u]; r, k, dr = v2_parts(sc, rows(sc['parent_uid']), rows(sc['donor_uid']))
    return [s5[dr] + .4 if i == k else s5[x] for i, x in enumerate(r)]


build(sorted(S2), pb_v2, 'processbench', 'V2', pb_v2_s5, lambda u: act[rows(S2[u]['parent_uid'])])
pb_v1 = lambda u: [pb_steps(S1[u]['parent_uid'])[j] for j in S1[u]['perm']]
build(sorted(S1), pb_v1, 'processbench', 'V1', lambda u: s5[rows(S1[u]['parent_uid'])][S1[u]['perm']],
      lambda u: act[rows(S1[u]['parent_uid'])][S1[u]['perm']])

# ---------------------------------------------------------------- PRMBench: 2,000 real answers, simulated labels / split
P = np.load(prm7); pids = P['uids'].astype(str); poff = P['offsets']; pmean = P['readouts'][:, :, 1]; pnt = P['ntok']
pd_ = (P['readouts'][:, :, 0] - pmean).mean(0) / 2.5
sel = np.sort(rng.choice(len(pids), 2000, replace=False)); sel = [j for j in sel if poff[j + 1] - poff[j] >= 2]
pu = [pids[j] for j in sel]; ppos = {pids[j]: j for j in sel}
prow = lambda u: np.arange(poff[ppos[u]], poff[ppos[u] + 1])
z = (pmean - pmean.mean(0)) / pmean.std(0); risk = z[:, [0, 2, 5]].mean(1)
labels, err = [], {}
for u in pu:
    r = prow(u); n = len(r)
    if rng.random() < .134: es = []; cls = 'correct' if rng.random() < .6 else 'redundency'
    else:
        p = np.exp(1.5 * risk[r]); p /= p.sum(); es = sorted(set((rng.choice(n, rng.integers(1, 3), p=p) + 1).tolist())); cls = 'confidence'
    err[u] = np.isin(np.arange(1, n + 1), es)
    labels.append(dict(idx=u, classification=cls, category='sim', error_steps=es, n_steps=n, source_idx=u))
ps5 = {u: rng.normal(0, 1, (len(prow(u)), 5)) + .5 * err[u][:, None] for u in pu}
pact = {u: rng.random((len(prow(u)), 3)) < .7 for u in pu}
ngr = 400; grp = {u: f'prmb_source_v2:sim{rng.integers(ngr):04d}' for u in pu}
gside = {g: ('dev' if rng.random() < .5 else 'test') for g in sorted(set(grp.values()))}
prm_steps = lambda u: [(pmean[x], pd_, int(pnt[x]), 0.) for x in prow(u)]
build(pu, prm_steps, 'prmbench', 'C', lambda u: ps5[u], lambda u: pact[u])
side_rows = {'V1': [], 'V2': []}
byside = {s: [u for u in pu if gside[grp[u]] == s] for s in ('dev', 'test')}
for u in pu:
    n = len(prow(u)); s = gside[grp[u]]
    k = int(rng.integers(2, n + 1)); donor = byside[s][rng.integers(len(byside[s]))]
    while grp[donor] == grp[u]: donor = byside[s][rng.integers(len(byside[s]))]
    side_rows['V2'].append(dict(uid=u + '__v2', parent_uid=u, variant='V2', group=grp[u], side=s, k=k, donor_uid=donor,
                                donor_step=int(rng.integers(1, len(prow(donor)) + 1)), relax_level=0))
    if n >= 3:
        while True:
            perm = [0] + list(rng.permutation(np.arange(1, n)))
            if perm[-1] != n - 1: break
        side_rows['V1'].append(dict(uid=u + '__v1', parent_uid=u, variant='V1', group=grp[u], side=s, perm=[int(x) for x in perm],
                                    single_adjacent_swap=n == 3))
for V in ('V1', 'V2'):
    with open(OUT / 'inputs' / f'SIDECAR_prmbench_{V}.jsonl', 'w') as f:
        for r in side_rows[V]: f.write(json.dumps(r) + '\n')
S2p = {r['uid']: r for r in side_rows['V2']}; S1p = {r['uid']: r for r in side_rows['V1']}


def prm_v2(u):
    sc = S2p[u]; r = prow(sc['parent_uid']); dr = prow(sc['donor_uid'])[sc['donor_step'] - 1]; k = sc['k'] - 1
    return [(pmean[dr], pd_, int(pnt[dr]), .6) if i == k else (pmean[x], pd_, int(pnt[x]), 0.) for i, x in enumerate(r)]


def prm_v2_s5(u):
    sc = S2p[u]; k = sc['k'] - 1; v = ps5[sc['parent_uid']].copy(); v[k] = ps5[sc['donor_uid']][sc['donor_step'] - 1] + .4
    return v


build(sorted(S2p), prm_v2, 'prmbench', 'V2', prm_v2_s5, lambda u: pact[S2p[u]['parent_uid']])
build(sorted(S1p), lambda u: [prm_steps(S1p[u]['parent_uid'])[j] for j in S1p[u]['perm']], 'prmbench', 'V1',
      lambda u: ps5[S1p[u]['parent_uid']][S1p[u]['perm']], lambda u: pact[S1p[u]['parent_uid']][S1p[u]['perm']])
json.dump(labels, open(OUT / 'inputs' / 'prmbench_C_labels.json', 'w'))
sp = json.load(open(var / 'SPLIT.json'))
sp['prmbench'] = {u: [grp[u], gside[grp[u]]] for u in pu}
sp['group_side'].update(gside)
json.dump(sp, open(OUT / 'inputs' / 'SPLIT.json', 'w'))
for V in ('V1', 'V2'):
    (OUT / 'inputs' / f'SIDECAR_processbench_{V}.jsonl').write_text((var / f'SIDECAR_processbench_{V}.jsonl').read_text())
print('done')
