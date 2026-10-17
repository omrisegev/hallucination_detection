"""AFTraj-2K injected-vs-diagnosed audit: paired safe/unsafe twins, prefix identity, signal location."""
import json, sys, collections, re, numpy as np
from scipy.stats import rankdata
S, OUT = sys.argv[1], sys.argv[2]
D = '/Users/osegev/Desktop/hallucination_detection/local_cache/agent_step_telemetry_v1/inputs_v2_20261013'
F = {r['uid']: r for r in json.load(open(f'{S}/aft_feats.json'))}
L = {d['uid']: d for d in map(json.loads, open(f'{D}/evaluator_only/aft.jsonl'))}
TX = {}
for l in open(f'{D}/aft_answers.jsonl'):
    r = json.loads(l); c = r['messages'][-1]['content']
    TX[r['uid']] = dict(group=r['group'], steps=[c[s:e] for s, e in r['entry_spans']], sel=r['step_entry_indices'], e2h=r['entry_to_history'])
def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def steps(u):  # list of (history_key, text, feats)
    r = F[u]; t = TX[u]; out = []
    for j, (k, f) in enumerate(zip(r['keys'], r['steps'])):
        out.append((k, t['steps'][t['sel'][j]], f))
    return out
W = lambda a, b: 1.0 if a > b else 0.5 if a == b else 0.0
G = collections.defaultdict(list)
for u in L: G[TX[u]['group']].append(u)
pairs = []
for us in G.values():
    s = [u for u in us if L[u]['label'] == 'safe']; t = [u for u in us if L[u]['unsafe_source'] == 'injected']
    if len(s) == 1 and len(t) == 1: pairs.append((s[0], t[0]))
res = {}; lines = []
kind = lambda u: L[u]['domain'] + '/' + L[u]['conv_id'].split('_')[-1]
for dom in ['math', 'coding', 'agentic', 'ALL']:
    P = [(s, t) for s, t in pairs if dom == 'ALL' or L[t]['domain'] == dom]
    R = dict(n_pairs=len(P), kinds=dict(collections.Counter(kind(t) for _, t in P)))
    first_diff_eq_m, prefix_feat_maxdiff, nsafe, nuns, post_s, post_u, ends_at_m = [], [], [], [], [], [], []
    w_traj, w_m, w_post, w_pre, w_suffix_mean, hit_m, rnd, z_m, len_m = [], [], [], [], [], [], [], [], []
    for s, t in P:
        A, B = steps(s), steps(t); m = L[t]['mistake_step']
        ka = [k for k, _, _ in A]; kb = [k for k, _, _ in B]
        fd = next((i for i in range(min(len(A), len(B))) if A[i][1] != B[i][1]), min(len(A), len(B)))
        first_diff_eq_m.append(kb[fd] == m if fd < len(B) else False)
        pre = [i for i in range(fd) if A[i][2] and B[i][2]]
        if pre: prefix_feat_maxdiff.append(max(abs(A[i][2]['mean_ent'] - B[i][2]['mean_ent']) for i in pre))
        nsafe.append(len(A)); nuns.append(len(B))
        if m in kb:
            im = kb.index(m); post_u.append(len(B) - im - 1); ends_at_m.append(im >= len(B) - 2)
            ia = ka.index(m) if m in ka else None
            if ia is not None: post_s.append(len(A) - ia - 1)
            ent = lambda X, lo, hi: [x[2]['mean_ent'] for x in X[lo:hi] if x[2]]
            ma, mb = np.mean(ent(A, 0, None)), np.mean(ent(B, 0, None)); w_traj.append(W(mb, ma))
            if ia is not None and A[ia][2] and B[im][2]:
                w_m.append(W(B[im][2]['mean_ent'], A[ia][2]['mean_ent']))
                sa, sb = ent(A, ia, None), ent(B, im, None)
                if sa and sb: w_suffix_mean.append(W(np.mean(sb), np.mean(sa)))
                pa, pb = ent(A, ia + 1, None), ent(B, im + 1, None)
                if pa and pb: w_post.append(W(np.mean(pb), np.mean(pa)))
                qa, qb = ent(A, 0, ia), ent(B, 0, im)
                if qa and qb: w_pre.append(W(np.mean(qb), np.mean(qa)))
            if B[im][2]:
                e = [x[2]['mean_ent'] for x in B if x[2]]
                hit_m.append(max(range(len(B)), key=lambda i: B[i][2]['mean_ent'] if B[i][2] else -1) == im); rnd.append(1 / len(B))
                z_m.append((B[im][2]['mean_ent'] - np.mean(e)) / (np.std(e) + 1e-9))
                len_m.append((len(B[im][1]), len(A[ia][1]) if ia is not None else np.nan))
    R.update(first_diff_is_mistake_step=np.mean(first_diff_eq_m), prefix_mean_ent_maxabsdiff_median=float(np.median(prefix_feat_maxdiff)) if prefix_feat_maxdiff else np.nan,
             med_steps_safe=float(np.median(nsafe)), med_steps_unsafe=float(np.median(nuns)), P_unsafe_fewer_steps=float(np.mean([b < a for a, b in zip(nsafe, nuns)])),
             med_steps_after_m_safe=float(np.median(post_s)), med_steps_after_m_unsafe=float(np.median(post_u)), P_unsafe_ends_within1_of_m=float(np.mean(ends_at_m)),
             paired_win_traj_mean_ent=float(np.mean(w_traj)), paired_win_pre_m=float(np.mean(w_pre)) if w_pre else np.nan,
             paired_win_at_m=float(np.mean(w_m)), paired_win_suffix_from_m=float(np.mean(w_suffix_mean)), paired_win_post_m=float(np.mean(w_post)) if w_post else np.nan,
             argmax_ent_is_m=float(np.mean(hit_m)), argmax_random=float(np.mean(rnd)), med_z_ent_at_m=float(np.median(z_m)),
             med_chars_m_unsafe_vs_safe=(float(np.median([a for a, _ in len_m])), float(np.nanmedian([b for _, b in len_m]))))
    res[dom] = R
# unpaired controls: diagnosed runs vs all safe runs of the same domain; and injected unsafe vs safe NOT in its own pair
for dom in ['math', 'agentic']:
    for src in ['diagnosed', 'injected']:
        us = [u for u in L if L[u]['unsafe_source'] == src and L[u]['domain'] == dom]
        sf = [u for u in L if L[u]['label'] == 'safe' and L[u]['domain'] == dom]
        mt = lambda u, f: np.mean([s[f] for s in F[u]['steps'] if s])
        y = [1] * len(us) + [0] * len(sf)
        # signal restricted to steps up to and including mistake (online prefix) vs safe full
        res[f'{dom}/{src}'] = dict(n=len(us), auc_mean_ent=auroc(y, [mt(u, 'mean_ent') for u in us + sf]), auc_nsteps=auroc(y, [len(F[u]['steps']) for u in us + sf]),
                                   med_steps=float(np.median([len(F[u]['steps']) for u in us])), med_steps_safe=float(np.median([len(F[u]['steps']) for u in sf])),
                                   med_rel_mistake_pos=float(np.median([F[u]['keys'].index(L[u]['mistake_step']) / max(len(F[u]['keys']) - 1, 1) for u in us if L[u]['mistake_step'] in F[u]['keys']])))
# within-unsafe localization for diagnosed
for dom in ['math', 'agentic']:
    us = [u for u in L if L[u]['unsafe_source'] == 'diagnosed' and L[u]['domain'] == dom and L[u]['mistake_step'] in F[u]['keys']]
    h, rr, z = [], [], []
    for u in us:
        e = [s['mean_ent'] if s else -1 for s in F[u]['steps']]; im = F[u]['keys'].index(L[u]['mistake_step'])
        h.append(int(np.argmax(e)) == im); rr.append(1 / len(e)); ee = [v for v in e if v >= 0]
        if e[im] >= 0: z.append((e[im] - np.mean(ee)) / (np.std(ee) + 1e-9))
    res[f'{dom}/diagnosed'].update(argmax_ent_is_m=float(np.mean(h)), argmax_random=float(np.mean(rr)), med_z_ent_at_m=float(np.median(z)))
json.dump(res, open(f'{OUT}/aft_audit.json', 'w'), indent=1, default=float)
for k, v in res.items():
    print(f'### {k}'); print(json.dumps(v, default=lambda x: round(float(x), 3)))
