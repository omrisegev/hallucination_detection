"""Re-score the existing (label-free, already fitted) N16 L-SML / IU-PCR step scores under the fair protocols of
results/agent_step_telemetry_v1_bias_audit_2026-10-16 (same-format pairs, 0.8-1.25 length matching, per-subset
label-free OLS residualisation on log length, first-error-vs-prior-clean, paired twins). No refitting.

    python -I fair.py <baseline_feats_dir> <n16_dir> <inputs_v2_dir> <apb_ref_dir> <apb_judge_dir> <out_dir>
"""
import collections, json, sys
import numpy as np
from scipy.stats import rankdata

SF, N16, INP, APB_REF, APB_JUDGE, OUT = sys.argv[1:7]
LAB = f'{INP}/evaluator_only'
B = 1000
rng = np.random.default_rng(0)
METHODS = {'L-SML cellz': 'cellz__bench__LSML_CONT', 'IU-PCR cellz': 'cellz__bench__IU_PCR',
           'IU-PCR cellz per-subset (post-hoc)': 'cellz__subset__IU_PCR',
           'L-SML ansz': 'ansz__bench__LSML_CONT', 'IU-PCR ansz': 'ansz__bench__IU_PCR'}


def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def win(a, b): return 1.0 if a > b else 0.5 if a == b else 0.0      # a = error candidate, higher = error


def ci_mean(w):
    w = np.asarray(w, float)
    if not len(w): return [np.nan, np.nan]
    return [float(x) for x in np.percentile([w[rng.integers(0, len(w), len(w))].mean() for _ in range(B)], [2.5, 97.5])]


def ci_units(fn, n):
    v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]


def lab(b): return {d['uid']: d for d in map(json.loads, open(f'{LAB}/{b}.jsonl'))}


def load(b):
    """Per step: baseline mean_ent / mean_nll / len (agent_baselines.py) + the N16 scores, keyed (uid, key)."""
    rows = json.load(open(f'{SF}/{b}_feats.json'))
    z = np.load(f'{N16}/SCORES_{b}.npz'); off = z['offsets']; keys = z['step_keys']
    sc = {}
    for i, u in enumerate(z['uids'].astype(str)):
        for j in range(off[i], off[i + 1]):
            sc[(u, int(keys[j]))] = {m: float(z[k][j]) for m, k in METHODS.items()}
    T = []
    for r in rows:
        st = []
        for i, (k, s) in enumerate(zip(r['keys'], r['steps'])):
            if s is None: continue
            if b == 'tprm': k = i
            f = dict(mean_ent=s['mean_ent'], mean_nll=s['mean_nll'], len=s['len'], pos=float(i), **sc.get((r['uid'], int(k)), {}))
            st.append(dict(k=int(k), i=i, f=f))
        T.append(dict(uid=r['uid'], sub=r['subset'], keys=r['keys'], st=st))
    return T


def ols_resid(groups, feats, design):
    """Per-subset label-free OLS of each feature on `design(step)`; returns resid(sub, step, feat)."""
    coef = {}
    for s, steps in groups.items():
        X = np.array([design(x) for x in steps])
        for f in feats:
            y = np.array([x['f'][f] for x in steps]); coef[(s, f)] = np.linalg.lstsq(X, y, rcond=None)[0]
    return lambda s, x, f: x['f'][f] - float(np.dot(coef[(s, f)], design(x)))


# ------------------------------------------------------------------ TPRM
def tprm():
    T = load('tprm'); L = lab('tprm'); A = {}
    for l in open(f'{INP}/tprm_answers.jsonl'):
        r = json.loads(l); A[r['uid']] = r['messages'][r['step_message_indices'][-1]]

    def struct(m):   # verbatim audit_tprm.py
        c = m.get('content') or ''
        tc = bool(m.get('tool_calls')) or '"function"' in c or '"name"' in c or '{"' in c
        prose = len(c.strip()) > 0 and not c.strip().startswith('{')
        return dict(tool=tc, prose=prose)
    pairs = collections.defaultdict(dict)
    for t in T:
        if not t['st']: continue
        pairs[L[t['uid']]['pair_id']][L[t['uid']]['action_chosen']] = dict(x=t['st'][-1], sub=t['sub'], st=struct(A[t['uid']]))
    P = [v for v in pairs.values() if True in v and False in v]
    feats = ['mean_ent', 'mean_nll', 'len'] + [m for m in METHODS if 'cellz' in m]
    groups = collections.defaultdict(list)
    for p in P:
        for k in (True, False): groups[p[True]['sub']].append(p[k]['x'])
    res_fn = ols_resid(groups, [f for f in feats if f != 'len'], lambda x: [1., np.log(x['f']['len'])])   # audit: polyfit(log len) per subset
    out = {}
    for s in ['bfcl', 'tooltalk', 'ToolSandbox', 'GTA']:
        Q = [p for p in P if p[True]['sub'] == s]
        same = [p for p in Q if p[True]['st'] == p[False]['st']]
        use = same if s != 'GTA' else Q          # GTA: format probe, all pairs (only 4 same-format)
        M = [p for p in use if 0.8 <= p[True]['x']['f']['len'] / p[False]['x']['f']['len'] <= 1.25]
        R = {'n_pairs': len(use), 'n_same_format': len(same), 'n_len_matched': len(M)}
        for f in feats:
            w = [win(p[False]['x']['f'][f], p[True]['x']['f'][f]) for p in use]       # rejected scored higher = correct
            wm = [win(p[False]['x']['f'][f], p[True]['x']['f'][f]) for p in M]
            R[f] = {'raw': float(np.mean(w)), 'raw_ci': ci_mean(w), 'matched': float(np.mean(wm)) if wm else np.nan, 'matched_ci': ci_mean(wm)}
            if f != 'len':
                wr = [win(res_fn(s, p[False]['x'], f), res_fn(s, p[True]['x'], f)) for p in use]
                R[f].update(resid=float(np.mean(wr)), resid_ci=ci_mean(wr))
            if s == 'GTA':
                ws = [win(p[False]['x']['f'][f], p[True]['x']['f'][f]) for p in same]; R[f]['same_format_only'] = float(np.mean(ws)) if ws else np.nan
        out[s] = R
    return out


# ------------------------------------------------------------------ APB
OFFS = {'hotpotqa': 0, 'gaia_dev': 250, 'bfcl': 500, 'tau2': 750}


def judge_labels():
    pred = {}
    for d, o in OFFS.items():
        for r in map(json.loads, open(f'{APB_JUDGE}/{d}__blind_Qwen3-8B-Non-Thinking.jsonl')):
            pred[f'apb:{d}:{o + r["query_index"] * 5 + r["sample_index"]}'] = {
                str(k): int(v) for k, v in (r.get('step_labels') or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return pred


def apb():
    T = load('apb'); L = lab('apb'); J = judge_labels()
    for t in T:
        sl = L[t['uid']]['step_labels']
        t['st'] = [dict(x, y=sl[str(x['k'])]) for x in t['st'] if str(x['k']) in sl]
        for x in t['st']: x['f']['judge'] = {-1: 1., 0: .5, 1: 0.}.get(J.get(t['uid'], {}).get(str(x['k'])), .5)
    T = [t for t in T if t['st']]
    feats = ['mean_ent', 'mean_nll', 'len', 'pos'] + list(METHODS)
    groups = collections.defaultdict(list)
    for t in T: groups[t['sub']] += t['st']                       # audit: all labelled steps of the subset
    rlen = ols_resid(groups, [f for f in feats if f not in ('len', 'pos')], lambda x: [1., np.log(x['f']['len'])])
    rlp = ols_resid(groups, [f for f in feats if f not in ('len', 'pos')], lambda x: [1., np.log(x['f']['len']), x['i']])
    units = []   # one per error trajectory with >=1 prior +1 step
    pool = collections.defaultdict(list)   # audit first_err_pooled: first -1 steps vs all +1 steps before any error (all trajs)
    for t in T:
        ys = [x['y'] for x in t['st']]
        j = ys.index(-1) if -1 in ys else None
        pre = [x for x in (t['st'][:j] if j is not None else t['st']) if x['y'] == 1]
        e = t['st'][j] if j is not None else None
        pool[t['sub']].append((e, pre))
        if e is not None and pre: units.append(dict(sub=t['sub'], e=e, pre=pre))
    def val(x, f, mode, sub):
        if mode == 'raw' or f in ('len', 'pos', 'judge'): return x['f'][f]
        return (rlen if mode == 'len' else rlp)(sub, x, f)
    out = {}
    for mode in ('raw', 'len', 'lenpos'):
        for f in feats + ['judge']:
            if mode != 'raw' and f in ('len', 'pos', 'judge'): continue
            acc, hit, rnd, sub = [], [], [], []
            for u in units:
                ev = val(u['e'], f, mode, u['sub']); pv = [val(p, f, mode, u['sub']) for p in u['pre']]
                acc.append(np.mean([win(ev, p) for p in pv]))
                allv = pv + [ev]; mx = max(allv); nt = sum(v == mx for v in allv)
                hit.append((1 / nt) if ev == mx else 0.); rnd.append(1 / len(allv)); sub.append(u['sub'])
            acc = np.array(acc); hit = np.array(hit); sub = np.array(sub)
            R = {'pairwise_acc': float(acc.mean()), 'pairwise_acc_ci': ci_mean(acc), 'loc_hit': float(hit.mean()), 'loc_hit_ci': ci_mean(hit),
                 'loc_random': float(np.mean(rnd)), 'n': int(len(acc))}
            yy, ss, ssub = [], [], []
            for s_, lst in pool.items():
                for e, pre in lst:
                    if e is not None: yy.append(True); ss.append(val(e, f, mode, s_)); ssub.append(s_)
                    for p in pre: yy.append(False); ss.append(val(p, f, mode, s_)); ssub.append(s_)
            yy = np.array(yy); ss = np.array(ss); ssub = np.array(ssub)
            R['pooled_auc_ALL'] = auroc(yy, ss)
            for s_ in sorted(set(sub)):
                m = sub == s_; mm = ssub == s_
                R[s_] = {'n': int(m.sum()), 'pairwise_acc': float(acc[m].mean()), 'loc_hit': float(hit[m].mean()),
                         'loc_random': float(np.mean(np.array(rnd)[m])), 'pooled_auc': auroc(yy[mm], ss[mm])}
            out[f'{mode}/{f}'] = R
    return out


# ------------------------------------------------------------------ AFT
def aft():
    T = {t['uid']: t for t in load('aft')}; L = lab('aft'); grp = {}
    for l in open(f'{INP}/aft_answers.jsonl'):
        r = json.loads(l); grp[r['uid']] = r['group']
    feats = ['mean_ent', 'mean_nll', 'len', 'L-SML cellz', 'IU-PCR cellz']
    out = {}
    # (a) diagnosed, online prefix up to and including the mistake step; safe runs truncated to the step count of a
    # randomly paired diagnosed run of the same domain (fixed seed 20261016; the audit defines no safe-prefix rule).
    prng = np.random.default_rng(20261016)
    for dom in ('agentic', 'math'):
        dg = []
        for u in sorted(L):
            d = L[u]
            if d['unsafe_source'] == 'diagnosed' and d['domain'] == dom:
                ks = [x['k'] for x in T[u]['st']]
                if d['mistake_step'] in ks: dg.append((u, ks.index(d['mistake_step']) + 1))
        safe = sorted(u for u in L if L[u]['label'] == 'safe' and L[u]['domain'] == dom and T[u]['st'])
        cut = {u: dg[prng.integers(len(dg))][1] for u in safe}
        R = {'n_diagnosed': len(dg), 'n_safe': len(safe), 'safe_prefix_rule': 'steps = mistake-step count of a random same-domain diagnosed run (seed 20261016)',
             'safe_runs_shorter_than_cut': int(sum(len(T[u]['st']) < cut[u] for u in safe))}
        y = np.array([True] * len(dg) + [False] * len(safe))
        for f in feats + ['n_steps_prefix']:
            v = []
            for u, n in dg: v.append(n if f == 'n_steps_prefix' else np.mean([x['f'][f] for x in T[u]['st'][:n]]))
            for u in safe:
                st = T[u]['st'][:cut[u]]; v.append(len(st) if f == 'n_steps_prefix' else np.mean([x['f'][f] for x in st]))
            v = np.array(v, float); R[f] = {'auc_prefix_mean': auroc(y, v), 'ci': ci_units(lambda i: auroc(y[i], v[i]), len(y))}
            if f == 'n_steps_prefix': continue
            hit, hit_on, rnd = [], [], []
            for u, n in dg:    # localization over the full run (audit) and over the online prefix (argmax == last step)
                vals = [x['f'][f] for x in T[u]['st']]; m = n - 1
                hit.append(int(np.argmax(vals)) == m); hit_on.append(int(np.argmax(vals[:n])) == m); rnd.append((1 / len(vals), 1 / n))
            R[f].update(loc_full=float(np.mean(hit)), loc_full_ci=ci_mean(hit), loc_full_random=float(np.mean([a for a, _ in rnd])),
                        loc_prefix=float(np.mean(hit_on)), loc_prefix_random=float(np.mean([b for _, b in rnd])))
        out[f'diagnosed/{dom}'] = R
    # (b) injected paired twins (audit pairing: group with exactly one safe and one injected run), mistake step only
    G = collections.defaultdict(list)
    for u in L: G[grp[u]].append(u)
    pairs = []
    for us in G.values():
        s = [u for u in us if L[u]['label'] == 'safe']; t = [u for u in us if L[u]['unsafe_source'] == 'injected']
        if len(s) == 1 and len(t) == 1: pairs.append((s[0], t[0]))
    # extra control: per-domain label-free OLS on log step length over all steps of the domain (audit residualisation rule)
    groups = collections.defaultdict(list)
    for u, t in T.items(): groups[L[u]['domain']] += t['st']
    rlen = ols_resid(groups, [f for f in feats if f != 'len'], lambda x: [1., np.log(x['f']['len'])])
    for dom in ('coding', 'math', 'agentic', 'ALL'):
        P = [(s, t) for s, t in pairs if dom == 'ALL' or L[t]['domain'] == dom]
        R = {'n_pairs_total': len(P)}
        for f in feats:
            w, wr = [], []
            for s, t in P:
                m = L[t]['mistake_step']; xs = {x['k']: x for x in T[s]['st']}; xt = {x['k']: x for x in T[t]['st']}
                if m in xs and m in xt:
                    w.append(win(xt[m]['f'][f], xs[m]['f'][f]))
                    if f != 'len': d = L[t]['domain']; wr.append(win(rlen(d, xt[m], f), rlen(d, xs[m], f)))
            R[f] = {'win_at_m': float(np.mean(w)), 'ci': ci_mean(w), 'n': len(w)}
            if wr: R[f].update(win_at_m_lenresid=float(np.mean(wr)), lenresid_ci=ci_mean(wr))
        out[f'injected_paired/{dom}'] = R
    return out


if __name__ == '__main__':
    res = {'tprm': tprm(), 'apb': apb(), 'aft': aft()}
    json.dump(res, open(f'{OUT}/FAIR_METRICS.json', 'w'), indent=1, default=float)
    print('DONE')
