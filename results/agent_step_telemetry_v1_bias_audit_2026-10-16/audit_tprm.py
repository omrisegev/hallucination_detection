"""ToolPRMBench length / provenance / format bias audit (label-free NLL/entropy, Qwen3-8B)."""
import json, collections, sys, numpy as np
S = sys.argv[1]   # dir with tprm_feats.json
D = '/Users/osegev/Desktop/hallucination_detection/local_cache/agent_step_telemetry_v1/inputs_v2_20261013'
F = {r['uid']: r for r in json.load(open(f'{S}/tprm_feats.json'))}
L = {d['uid']: d for d in map(json.loads, open(f'{D}/evaluator_only/tprm.jsonl'))}
A = {}
for l in open(f'{D}/tprm_answers.jsonl'):
    r = json.loads(l); m = r['messages'][r['step_message_indices'][-1]]
    A[r['uid']] = m
def struct(m):
    c = m.get('content') or ''
    tc = bool(m.get('tool_calls')) or '"function"' in c or '"name"' in c or '{"' in c
    prose = len(c.strip()) > 0 and not c.strip().startswith('{')
    return dict(tool=tc, prose=prose, chars=len(c) + len(json.dumps(m.get('tool_calls') or [])))
pairs = collections.defaultdict(dict)
for u, r in F.items():
    st = [s for s in r['steps'] if s]
    if not st: continue
    pairs[L[u]['pair_id']][L[u]['action_chosen']] = dict(f=st[-1], sub=r['subset'], st=struct(A[u]))
P = [v for v in pairs.values() if True in v and False in v]
win = lambda c, r: 1.0 if c < r else 0.5 if c == r else 0.0
# label-free length residualisation: per subset OLS of feature on log(len) over all actions (no labels)
res = {}
for f in ['mean_nll', 'mean_ent', 'max_nll']:
    for s in {p[True]['sub'] for p in P}:
        xs = [(p[k]['f'], k) for p in P if p[True]['sub'] == s for k in (True, False)]
        x = np.log([a['len'] for a, _ in xs]); y = np.array([a[f] for a, _ in xs])
        b = np.polyfit(x, y, 1); res[(f, s)] = b
def resid(a, f, s): return a[f] - np.polyval(res[(f, s)], np.log(a['len']))
out = {}
lines = ['| subset | n | P(chosen shorter) | med len ch/rej | tool-call ch/rej | prose ch/rej | mean_nll acc | mean_ent acc | len-matched n | nll acc (matched) | ent acc (matched) | nll acc (len-resid) | ent acc (len-resid) | sum_nll acc | same-format n | ent acc (same fmt) |',
         '|' + '---|' * 16]
for s in sorted({p[True]['sub'] for p in P}) + ['ALL']:
    Q = [p for p in P if s == 'ALL' or p[True]['sub'] == s]
    c = lambda p: p[True]; r = lambda p: p[False]
    sh = np.mean([win(c(p)['f']['len'], r(p)['f']['len']) for p in Q])
    acc = {f: np.mean([win(c(p)['f'][f], r(p)['f'][f]) for p in Q]) for f in ['mean_nll', 'mean_ent']}
    M = [p for p in Q if 0.8 <= c(p)['f']['len'] / r(p)['f']['len'] <= 1.25]
    accm = {f: np.mean([win(c(p)['f'][f], r(p)['f'][f]) for p in M]) if M else np.nan for f in ['mean_nll', 'mean_ent']}
    accr = {f: np.mean([win(resid(c(p)['f'], f, c(p)['sub']), resid(r(p)['f'], f, c(p)['sub'])) for p in Q]) for f in ['mean_nll', 'mean_ent']}
    sumnll = np.mean([win(c(p)['f']['mean_nll'] * c(p)['f']['len'], r(p)['f']['mean_nll'] * r(p)['f']['len']) for p in Q])
    SF = [p for p in Q if c(p)['st']['tool'] == r(p)['st']['tool'] and c(p)['st']['prose'] == r(p)['st']['prose']]
    accsf = np.mean([win(c(p)['f']['mean_ent'], r(p)['f']['mean_ent']) for p in SF]) if SF else np.nan
    row = dict(n=len(Q), p_chosen_shorter=sh,
               med_len=(np.median([c(p)['f']['len'] for p in Q]), np.median([r(p)['f']['len'] for p in Q])),
               tool=(np.mean([c(p)['st']['tool'] for p in Q]), np.mean([r(p)['st']['tool'] for p in Q])),
               prose=(np.mean([c(p)['st']['prose'] for p in Q]), np.mean([r(p)['st']['prose'] for p in Q])),
               acc=acc, n_matched=len(M), acc_matched=accm, acc_resid=accr, acc_sum_nll=sumnll, n_same_fmt=len(SF), ent_acc_same_fmt=accsf)
    out[s] = row
    lines.append(f"| {s} | {len(Q)} | {sh:.3f} | {row['med_len'][0]:.0f}/{row['med_len'][1]:.0f} | {row['tool'][0]:.2f}/{row['tool'][1]:.2f} | {row['prose'][0]:.2f}/{row['prose'][1]:.2f} | "
                 f"{acc['mean_nll']:.3f} | {acc['mean_ent']:.3f} | {len(M)} | {accm['mean_nll']:.3f} | {accm['mean_ent']:.3f} | {accr['mean_nll']:.3f} | {accr['mean_ent']:.3f} | {sumnll:.3f} | {len(SF)} | {accsf:.3f} |")
# GTA format breakdown
gta = [p for p in P if p[True]['sub'] == 'GTA']
cnt = collections.Counter((('tool' if p[True]['st']['tool'] else '') + ('+prose' if p[True]['st']['prose'] else ''),
                           ('tool' if p[False]['st']['tool'] else '') + ('+prose' if p[False]['st']['prose'] else '')) for p in gta)
lines.append('\nGTA (chosen fmt, rejected fmt) counts: ' + str(dict(cnt)))
json.dump(out, open(sys.argv[2] + '/tprm_audit.json', 'w'), indent=1, default=float)
print('\n'.join(lines))
