"""AgentProcessBench confound audit: length, position, label propagation, protocol variants."""
import json, sys, collections, numpy as np
from scipy.stats import rankdata, spearmanr
S, OUT = sys.argv[1], sys.argv[2]
D = '/Users/osegev/Desktop/hallucination_detection/local_cache/agent_step_telemetry_v1/inputs_v2_20261013'
rows = json.load(open(f'{S}/apb_feats.json'))
L = {d['uid']: d for d in map(json.loads, open(f'{D}/evaluator_only/apb.jsonl'))}
def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
T = []  # trajectories: list of dict(sub, steps=[(pos, relpos, feats, y)])
for r in rows:
    lab = L[r['uid']]['step_labels']; st = []
    n = len(r['keys'])
    for i, (k, s) in enumerate(zip(r['keys'], r['steps'])):
        y = lab.get(str(k))
        if s is None or y is None: continue
        st.append(dict(i=i, rel=i / max(n - 1, 1), f=s, y=y))
    if st: T.append(dict(sub=r['subset'], st=st, final=L[r['uid']]['final_label']))
subs = sorted({t['sub'] for t in T}) + ['ALL']
FE = ['mean_ent', 'mean_nll', 'len']
# label-free per-subset residualisation on log len AND position
coef = {}
for s in subs[:-1]:
    X = [(np.log(x['f']['len']), x['i']) for t in T if t['sub'] == s for x in t['st']]
    X = np.c_[np.ones(len(X)), np.array(X)]
    for f in ['mean_ent', 'mean_nll']:
        y = np.array([x['f'][f] for t in T if t['sub'] == s for x in t['st']])
        coef[(s, f, 'len')] = np.linalg.lstsq(X[:, :2], y, rcond=None)[0]
        coef[(s, f, 'lenpos')] = np.linalg.lstsq(X, y, rcond=None)[0]
def resid(t, x, f, mode):
    b = coef[(t['sub'], f, mode)]; v = [1, np.log(x['f']['len'])] + ([x['i']] if mode == 'lenpos' else [])
    return x['f'][f] - np.dot(b, v)
for t in T:
    for x in t['st']:
        x['f']['mean_ent_lr'] = resid(t, x, 'mean_ent', 'len'); x['f']['mean_nll_lr'] = resid(t, x, 'mean_nll', 'len')
FE = FE + ['mean_ent_lr', 'mean_nll_lr']
res = {}; lines = []
for s in subs:
    TT = [t for t in T if s == 'ALL' or t['sub'] == s]
    steps = [(t, x) for t in TT for x in t['st']]
    lab = collections.Counter(x['y'] for _, x in steps)
    nz = [(t, x) for t, x in steps if x['y'] != 0]
    y = np.array([x['y'] == -1 for _, x in nz])
    R = dict(n_traj=len(TT), labels=dict(lab))
    for f in FE: R[f'auc_{f}'] = auroc(y, [x['f'][f] for _, x in nz])
    R['auc_abs_pos'] = auroc(y, [x['i'] for _, x in nz]); R['auc_rel_pos'] = auroc(y, [x['rel'] for _, x in nz])
    # propagation: after the first -1, are all later labelled steps -1?
    absorb, post_frac, n_err_traj, recover = [], [], 0, []
    tot_err = sum(y); post = 0
    for t in TT:
        ys = [x['y'] for x in t['st']]
        if -1 in ys:
            j = ys.index(-1); after = ys[j + 1:]
            n_err_traj += 1
            if after:
                absorb.append(all(v == -1 for v in after)); recover.append(1 in after)
            post += sum(v == -1 for v in after)
    R['traj_with_err'] = n_err_traj; R['P_all_after_first_err_are_-1'] = float(np.mean(absorb)) if absorb else np.nan
    R['P_+1_after_first_err'] = float(np.mean(recover)) if recover else np.nan
    R['frac_err_steps_not_first'] = post / max(tot_err, 1)
    # length strata (quintiles within subset)
    for f in ['mean_ent', 'mean_nll']:
        lens = np.array([x['f']['len'] for _, x in nz]); vals = np.array([x['f'][f] for _, x in nz])
        q = np.quantile(lens, [0.2, 0.4, 0.6, 0.8]); b = np.digitize(lens, q)
        a = [(auroc(y[b == k], vals[b == k]), (b == k).sum()) for k in range(5)]
        a = [(v, n) for v, n in a if not np.isnan(v)]
        R[f'auc_{f}_lenstrat'] = sum(v * n for v, n in a) / sum(n for _, n in a)
        if s != 'ALL':
            R[f'auc_{f}_lenresid'] = auroc(y, [resid(t, x, f, 'len') for t, x in nz])
            R[f'auc_{f}_lenposresid'] = auroc(y, [resid(t, x, f, 'lenpos') for t, x in nz])
        R[f'spearman_{f}_vs_pos'] = spearmanr([x['i'] for _, x in steps], [x['f'][f] for _, x in steps]).correlation
        R[f'spearman_{f}_vs_loglen'] = spearmanr([x['f']['len'] for _, x in steps], [x['f'][f] for _, x in steps]).correlation
    # protocol A: within-trajectory AUROC (traj with both -1 and +1), mean over trajectories
    for f in FE:
        w = [auroc([x['y'] == -1 for x in t['st'] if x['y'] != 0], [x['f'][f] for x in t['st'] if x['y'] != 0]) for t in TT]
        R[f'within_traj_auc_{f}'] = float(np.nanmean(w)); R['n_within'] = int(np.sum(~np.isnan(w)))
    # protocol B: first-error-only: first -1 step vs +1 steps BEFORE it in the same trajectory (pairwise acc)
    for f in FE:
        acc = []
        pool_y, pool_s = [], []
        for t in TT:
            ys = [x['y'] for x in t['st']]
            if -1 not in ys: 
                pool_y += [False] * sum(v == 1 for v in ys); pool_s += [x['f'][f] for x in t['st'] if x['y'] == 1]; continue
            j = ys.index(-1); e = t['st'][j]['f'][f]; pre = [x['f'][f] for x in t['st'][:j] if x['y'] == 1]
            pool_y.append(True); pool_s.append(e)
            pool_y += [False] * len(pre); pool_s += pre
            if pre: acc.append(np.mean([1.0 if e > p else 0.5 if e == p else 0.0 for p in pre]))
        R[f'first_err_vs_prior_ok_{f}'] = float(np.mean(acc)) if acc else np.nan
        R['n_first_err_with_prior'] = len(acc)
        R[f'first_err_pooled_auc_{f}'] = auroc(pool_y, pool_s)  # first-error steps vs all clean (+1) steps before any error
    # label-0 as error / as ok
    y0e = np.array([x['y'] != 1 for _, x in steps]); R['auc_mean_ent_0_as_err'] = auroc(y0e, [x['f']['mean_ent'] for _, x in steps])
    res[s] = R
json.dump(res, open(f'{OUT}/apb_audit.json', 'w'), indent=1, default=float)
keys = ['n_traj', 'labels', 'traj_with_err', 'P_all_after_first_err_are_-1', 'P_+1_after_first_err', 'frac_err_steps_not_first',
        'auc_len', 'auc_abs_pos', 'auc_rel_pos', 'auc_mean_ent', 'auc_mean_ent_lenstrat', 'auc_mean_ent_lenresid', 'auc_mean_ent_lenposresid',
        'auc_mean_nll', 'auc_mean_nll_lenstrat', 'auc_mean_nll_lenposresid', 'spearman_mean_ent_vs_pos', 'spearman_mean_ent_vs_loglen',
        'n_within', 'within_traj_auc_mean_ent', 'within_traj_auc_mean_nll', 'within_traj_auc_len',
        'n_first_err_with_prior', 'first_err_vs_prior_ok_mean_ent', 'first_err_vs_prior_ok_mean_nll', 'first_err_vs_prior_ok_len',
        'first_err_pooled_auc_mean_ent', 'first_err_pooled_auc_mean_nll', 'first_err_pooled_auc_len', 'auc_mean_ent_0_as_err', 'within_traj_auc_mean_ent_lr', 'within_traj_auc_mean_nll_lr', 'first_err_vs_prior_ok_mean_ent_lr', 'first_err_vs_prior_ok_mean_nll_lr', 'first_err_pooled_auc_mean_ent_lr']
print('| metric | ' + ' | '.join(subs) + ' |'); print('|' + '---|' * (len(subs) + 1))
fm = lambda v: f'{v:.3f}' if isinstance(v, float) else str(v)
for k in keys: print(f'| {k} | ' + ' | '.join(fm(res[s].get(k, '')) for s in subs) + ' |')
