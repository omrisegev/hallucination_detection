"""CPU pilot: label-free fusion of competitor judgments with Qwen3-8B telemetry scores.

TPRM : competitor (Skywork-o1-PRM score, Nemotron-49B-GenRM ranking_score, Qwen3-Coder-Next vote) + our
       per-variant score (mean_ent / IU-PCR cellz / L-SML cellz). Cross-backbone: our telemetry is Qwen3-8B.
APB  : Qwen3-8B LLM-judge step label + our Qwen3-8B telemetry score (same backbone).

Fusion is fixed a priori, no labels: equal-weight sum of each view scaled to unit size.
  TPRM: pair-level preference d = (evidence for variant chosen over rejected); each view divided by its
        per-subset RMS over all pairs (RMS is sign-invariant, so the scale needs no label).
  APB : per-subset z-score over all labelled steps of judge-error-score and of our error score.
Protocols and helpers are imported from fair_protocols/fair.py (bias-audit definitions, unchanged).

    python -I fusion_pilot.py <baseline_feats_dir> <n16_dir> <inputs_v2_dir> <apb_ref_dir> <apb_judge_dir> <tprm_ref_dir> <out_dir>
"""
import collections, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TPRM_REF, OUT = sys.argv[6], sys.argv[7]
sys.argv = sys.argv[:6] + [OUT]
sys.path.insert(0, os.path.join(HERE, '..', 'agent_step_telemetry_v1_n16_2026-10-16', 'fair_protocols'))
import fair  # noqa: E402  (reads argv[1:7] at import)

rng = np.random.default_rng(1)
B = 2000
SUBS = ['bfcl', 'tooltalk', 'ToolSandbox', 'GTA']
OURS = ['mean_ent', 'IU-PCR cellz', 'L-SML cellz']
COMP = {'Skywork-o1-PRM-7B': 'Skywork_Skywork-o1-Open-PRM-Qwen-2.5-7B',
        'Nemotron-49B-GenRM': 'nvidia_Llama-3_3-Nemotron-Super-49B-GenRM',
        'Qwen3-Coder-Next': 'Qwen_Qwen3-Coder-Next'}


def comp_pref(name, x):
    """Competitor evidence that the gold-chosen action is better (>0) — orientation by its own output semantics."""
    if name == 'Skywork-o1-PRM-7B':
        if x.get('score_chosen') is None or x.get('score_rejected') is None: return 0.0   # missing output = abstain
        return float(x['score_chosen']) - float(x['score_rejected'])
    if name == 'Nemotron-49B-GenRM':           # ranking_score 1..6, <=3 prefers response_1
        rs = x.get('ranking_score')
        if rs is None: return 0.0
        return (3.5 - rs) if x['ground_truth'] == 'response_1' else (rs - 3.5)
    jr = x.get('judgment_result'); ch = jr.get('chosen action') if isinstance(jr, dict) else None
    if ch not in ('Action1', 'Action2'): return 0.0
    return 1.0 if ch == x['ground_truth'] else -1.0


def acc(d): return np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0))


def paired_ci(a, b):
    n = len(a); idx = rng.integers(0, n, (B, n))
    g = a[idx].mean(1) - b[idx].mean(1)
    return [float(x) for x in np.percentile(g, [2.5, 97.5])]


def tprm():
    T = fair.load('tprm'); L = fair.lab('tprm'); A = {}
    for l in open(f'{fair.INP}/tprm_answers.jsonl'):
        r = json.loads(l); A[r['uid']] = r['messages'][r['step_message_indices'][-1]]

    def struct(m):   # verbatim audit_tprm.py (via fair.py)
        c = m.get('content') or ''
        return dict(tool=bool(m.get('tool_calls')) or '"function"' in c or '"name"' in c or '{"' in c,
                    prose=len(c.strip()) > 0 and not c.strip().startswith('{'))
    pairs = collections.defaultdict(dict)
    for t in T:
        if t['st']: pairs[L[t['uid']]['pair_id']][L[t['uid']]['action_chosen']] = dict(x=t['st'][-1], sub=t['sub'], st=struct(A[t['uid']]))
    P = {k: v for k, v in pairs.items() if True in v and False in v}
    groups = collections.defaultdict(list)
    for p in P.values():
        for k in (True, False): groups[p[True]['sub']].append(p[k]['x'])
    res_fn = fair.ols_resid(groups, OURS, lambda x: [1., np.log(x['f']['len'])])
    # competitor preferences aligned by release row: pair id ends with case<row-in-subset-file>
    C = {}
    for s in SUBS:
        for name, f in COMP.items():
            for i, x in enumerate(json.load(open(f'{TPRM_REF}/prmbench_{s}_{f}.json'))):
                pid = [k for k in P if k.startswith(f'tprm:{s}:') and k.endswith(f':case{i}')]
                assert len(pid) == 1, (s, i, pid)
                C[(name, pid[0])] = comp_pref(name, x)
    out = {}
    for s in SUBS:
        allp = [k for k in P if P[k][True]['sub'] == s]
        same = [k for k in allp if P[k][True]['st'] == P[k][False]['st']]
        use = same if s != 'GTA' else allp
        R = {'n_pairs': len(use), 'n_same_format': len(same)}
        for mode in ('raw', 'resid'):
            ours = {}
            for f in OURS:
                g = (lambda x: x['f'][f]) if mode == 'raw' else (lambda x, f=f: res_fn(s, x, f))
                d = np.array([g(P[k][False]['x']) - g(P[k][True]['x']) for k in allp])   # higher score = error
                rms = np.sqrt(np.mean(d ** 2)); ours[f] = dict(zip(allp, d / rms))
            for name in COMP:
                dc = np.array([C[(name, k)] for k in allp]); rms = np.sqrt(np.mean(dc ** 2)) or 1.0
                comp = dict(zip(allp, dc / rms))
                a_c = acc(np.array([comp[k] for k in use]))
                row = {'competitor_alone': float(a_c.mean()),
                       'competitor_alone_all_pairs': float(acc(dc).mean()), 'n_abstain_all_pairs': int((dc == 0).sum())}
                for f in OURS:
                    a_o = acc(np.array([ours[f][k] for k in use]))
                    a_f = acc(np.array([comp[k] + ours[f][k] for k in use]))
                    row[f] = {'ours_alone': float(a_o.mean()), 'fused': float(a_f.mean()),
                              'gain': float(a_f.mean() - a_c.mean()), 'gain_ci': paired_ci(a_f, a_c),
                              'flips_to_right': int(((a_f > a_c)).sum()), 'flips_to_wrong': int(((a_f < a_c)).sum())}
                R[f'{mode}/{name}'] = row
        out[s] = R
    # pooled over the three same-format subsets
    return out


def apb():
    T = fair.load('apb'); L = fair.lab('apb'); J = fair.judge_labels()
    for t in T:
        sl = L[t['uid']]['step_labels']
        t['st'] = [dict(x, y=sl[str(x['k'])]) for x in t['st'] if str(x['k']) in sl]
        for x in t['st']: x['f']['judge'] = {-1: 1., 0: .5, 1: 0.}.get(J.get(t['uid'], {}).get(str(x['k'])), .5)
    T = [t for t in T if t['st']]
    ours = ['mean_ent', 'IU-PCR ansz', 'L-SML ansz']
    groups = collections.defaultdict(list)
    for t in T: groups[t['sub']] += t['st']
    rlen = fair.ols_resid(groups, ours, lambda x: [1., np.log(x['f']['len'])])
    rlp = fair.ols_resid(groups, ours, lambda x: [1., np.log(x['f']['len']), x['i']])
    # per-subset z stats (label-free, all labelled steps)
    def zstats(fn):
        return {s: (np.mean([fn(s, x) for x in st]), np.std([fn(s, x) for x in st]) or 1.0) for s, st in groups.items()}
    judge_z = zstats(lambda s, x: x['f']['judge'])
    out = {}
    units = []
    for t in T:
        ys = [x['y'] for x in t['st']]
        if -1 not in ys: continue
        j = ys.index(-1); pre = [x for x in t['st'][:j] if x['y'] == 1]
        if pre: units.append((t['sub'], t['st'][j], pre))
    allsteps = [(t['sub'], x) for t in T for x in t['st'] if x['y'] != 0]
    for mode in ('raw', 'len', 'lenpos'):
        for f in ours:
            raw = (lambda s, x, f=f: x['f'][f]) if mode == 'raw' else ((lambda s, x, f=f: rlen(s, x, f)) if mode == 'len' else (lambda s, x, f=f: rlp(s, x, f)))
            oz = zstats(raw)
            views = {'judge': lambda s, x: (x['f']['judge'] - judge_z[s][0]) / judge_z[s][1],
                     'ours': lambda s, x: (raw(s, x) - oz[s][0]) / oz[s][1]}
            views['fused'] = lambda s, x: views['judge'](s, x) + views['ours'](s, x)
            R = {}
            for v, fn in views.items():
                a = np.array([np.mean([fair.win(fn(s, e), fn(s, p)) for p in pre]) for s, e, pre in units])
                y = np.array([x['y'] == -1 for _, x in allsteps]); sc = np.array([fn(s, x) for s, x in allsteps])
                R[v] = {'first_err_pairwise': float(a.mean()), 'step_auroc_all': fair.auroc(y, sc), '_a': a,
                        'per_subset': {s: float(a[np.array([u[0] for u in units]) == s].mean()) for s in sorted(groups)}}
            R['fused']['gain_vs_judge'] = float(R['fused']['_a'].mean() - R['judge']['_a'].mean())
            R['fused']['gain_ci'] = paired_ci(R['fused']['_a'], R['judge']['_a'])
            for v in R: R[v].pop('_a')
            R['n_units'] = len(units)
            out[f'{mode}/{f}'] = R
    return out


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    res = {'tprm': tprm(), 'apb': apb()}
    json.dump(res, open(os.path.join(OUT, 'FUSION_PILOT.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1)[:200])
