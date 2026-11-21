"""V3 / V4 children for Hard2Verify and DeltaBench (rule C transfer, 2026-11-21). The V3 / V4 rules are the UNCHANGED
functions make_v3 / make_v4 of results/ssl_one_experiment_v1/make_variants_v34.py (imported, not copied). No gold label
is read. No dev/test split: every answer with C telemetry is a parent (side 'all'). Refuses to overwrite.

    python make_variants_ext.py --bench hard2verify --answers <answers.json | run_identity.json> \
        --steps STEPS_<bench>_C.npz --groups GROUPS_<bench>.json --variants V3,V4 --out DIR
Note: make_v3 / make_v4 seed their RNG with [SEED, 3|4, 0 if bench == 'processbench' else 1]; external benchmarks get
the PRMBench seed index 1 (no rule change).
Outputs: <bench>_<V>_answers.json (driver --answers format: uid, question, steps, benchmark, source_id),
         <bench>_preC_<V>_answers.json (C rows of the parents of the first 12 children), SIDECAR_<bench>_<V>.jsonl,
         VARIANTS_EXT_<bench>_REPORT.json
"""
import argparse, json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import make_variants_v34 as V34

a = argparse.ArgumentParser()
for k in ('--bench', '--answers', '--steps', '--groups', '--variants', '--out'): a.add_argument(k, required=True)
a = a.parse_args(); t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
VS = a.variants.split(',')
R = json.load(open(a.answers, encoding='utf8'))
if isinstance(R, dict) and 'row_ids' in R:        # H2V run_identity.json of the stored C run
    src = [dict(uid=u, question=qs[0], steps=list(qs[1]), benchmark=a.bench, source_id=u) for u, qs in zip(R['row_ids'], R['questions_steps'])]
else:
    src = [dict(uid=r['uid'], question=r['question'], steps=list(r['steps']), benchmark=r.get('benchmark', a.bench),
                source_id=r.get('source_id', r['uid']), generator=r.get('generator')) for r in R]
have = set(np.load(a.steps)['uids'].astype(str)); G = json.load(open(a.groups))
src = [r for r in src if r['uid'] in have]
assert len(src) == len(have), (len(src), len(have))
gen = lambda u, r: r.get('generator') or (u.rsplit('-', 1)[0] and next((g for g in ('gemini-2.5-pro', 'claude-sonnet-4', 'GPT-5') if u.endswith(g)), None))
L = [dict(uid=r['uid'], steps=r['steps'], group=G[r['uid']], side='all', subset=None, generator=gen(r['uid'], r)) for r in src]
dev = set(range(len(L)))
rep = dict(seed=V34.SEED, bench=a.bench, parents=len(L), groups=len(set(G[r['uid']] for r in src)), rules='make_variants_v34.make_v3/make_v4 unchanged')
for V in VS:
    D = V34.make_v3(L, a.bench, dev) if V == 'V3' else V34.make_v4(L, a.bench, dev)[0]
    rows, lines, pre = [], [], []
    for i in sorted(dev):
        d = D[i]; p = L[i]
        line = dict(uid=f"{p['uid']}__{V.lower()}", parent_uid=p['uid'], variant=V, group=p['group'], side='all')
        if 'skip' in d: line['skip'] = d['skip']; lines.append(line); continue
        line.update({k: v for k, v in d.items() if k != 'steps'}); lines.append(line)
        if len(pre) < V34.N_PRE: pre.append(i)
        rows.append(dict(uid=line['uid'], question=src[i]['question'], steps=d['steps'], benchmark=a.bench, source_id=p['uid']))
    V34.write_new(out / f'{a.bench}_{V}_answers.json', lambda f: json.dump(rows, open(f, 'w')))
    V34.write_new(out / f'{a.bench}_preC_{V}_answers.json', lambda f: json.dump([src[i] for i in pre], open(f, 'w')))
    V34.write_new(out / f'SIDECAR_{a.bench}_{V}.jsonl', lambda f: open(f, 'w').write(''.join(json.dumps(s) + '\n' for s in lines)))
    built = [s for s in lines if 'skip' not in s]
    info = dict(rows=len(built), share_built=round(len(built) / len(L), 4), skipped=dict(Counter(s['skip'] for s in lines if 'skip' in s)),
                k_mean=float(np.mean([s['k'] for s in built])) if built else None,
                rel_parent_mean=float(np.mean([s['rel_parent'] for s in built])) if built else None)
    if V == 'V3':
        info['old_new_examples'] = [(s['old'], s['new']) for s in built[:8]]
    else:
        info['relax_level'] = dict(Counter(s['relax_level'] for s in built))
        info['same_generator_share'] = float(np.mean([s['donor_same_generator'] for s in built])) if built else None
    rep[V] = info; print(a.bench, V, json.dumps(info), flush=True)
rep['seconds'] = round(time.time() - t0, 1)
V34.write_new(out / f'VARIANTS_EXT_{a.bench}_REPORT.json', lambda f: json.dump(rep, open(f, 'w'), indent=1))
