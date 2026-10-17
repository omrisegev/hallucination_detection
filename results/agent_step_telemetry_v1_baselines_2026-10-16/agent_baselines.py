"""Label-free per-step NLL/entropy baselines on agent_step_telemetry_v1.

Step 1: extract compact per-step features from the raw records (label-blind).
Step 2: join evaluator labels and score with each benchmark's standard metric.
Orientation is fixed a priori: higher NLL / entropy => more likely an error.
"""
import glob, json, os, sys
from multiprocessing import Pool
import numpy as np
from scipy.stats import rankdata

ROOT = sys.argv[1]          # dir with tprm/ apb/ aft/ record dirs
LAB = sys.argv[2]           # evaluator_only dir
OUT = sys.argv[3]
FEATS = ['mean_nll', 'max_nll', 'mean_ent', 'len']
rng = np.random.default_rng(0)


def step_feats(path):
    r = json.load(open(path))
    t = r['payload']['telemetry']
    nll = np.asarray(t['token_spilled_energies'], float)
    ent = np.asarray(t['token_entropy_full'], float)
    steps = []
    for i, (s, e) in enumerate(t['step_token_spans']):
        if e <= s:
            steps.append(None); continue
        steps.append(dict(mean_nll=float(nll[s:e].mean()), max_nll=float(nll[s:e].max()),
                          mean_ent=float(ent[s:e].mean()), len=float(e - s)))
    key = (t['step_message_indices'] if t['benchmark'] == 'apb'
           else [t['entry_to_history'][j] for j in t['step_entry_indices']] if t['benchmark'] == 'aft'
           else list(range(len(steps))))
    return dict(uid=r['uid'], subset=t['subset'], group=t['group'], steps=steps, keys=key)


def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot(fn, n, B=1000):
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return np.nanpercentile(vals, [2.5, 97.5])


def fmt(v, ci): return f'{v:.3f} [{ci[0]:.3f},{ci[1]:.3f}]'


def load(bench):
    cache = os.path.join(OUT, f'{bench}_feats.json')
    if os.path.exists(cache): return json.load(open(cache))
    files = [f for f in glob.glob(os.path.join(ROOT, bench, '*.record.json'))]
    with Pool(8) as p: rows = p.map(step_feats, files, chunksize=8)
    json.dump(rows, open(cache, 'w'))
    return rows


def labels(bench):
    return {d['uid']: d for d in map(json.loads, open(os.path.join(LAB, f'{bench}.jsonl')))}


def tprm(lines):
    rows = load('tprm'); L = labels('tprm')
    pairs = {}
    for r in rows:
        st = [s for s in r['steps'] if s]
        if not st: continue
        lab = L[r['uid']]
        pairs.setdefault(lab['pair_id'], {})[lab['action_chosen']] = (st[-1], r['subset'])
    pairs = [v for v in pairs.values() if True in v and False in v]
    lines.append(f'\n## ToolPRMBench — pairwise accuracy (chosen action less "bad"), {len(pairs)} pairs; chance = 0.500')
    subsets = sorted({v[True][1] for v in pairs})
    lines.append('| feature | all | ' + ' | '.join(subsets) + ' |')
    lines.append('|' + '---|' * (2 + len(subsets)))
    for f in FEATS:
        w = np.array([1.0 if v[True][0][f] < v[False][0][f] else 0.5 if v[True][0][f] == v[False][0][f] else 0.0 for v in pairs])
        sub = np.array([v[True][1] for v in pairs])
        cells = [fmt(w.mean(), boot(lambda i: w[i].mean(), len(w)))]
        cells += [f'{w[sub == s].mean():.3f} (n={int((sub == s).sum())})' for s in subsets]
        lines.append(f'| {f} | ' + ' | '.join(cells) + ' |')


def apb(lines):
    rows = load('apb'); L = labels('apb')
    S, Y, SUB, TR = [], [], [], []   # step-level, -1 vs +1
    loc_hit, loc_rand, traj_y, traj_s = [], [], [], {f: [] for f in FEATS}
    for r in rows:
        lab = L[r['uid']]['step_labels']
        trs = []
        for k, s in zip(r['keys'], r['steps']):
            y = lab.get(str(k))
            if s is None or y is None: continue
            trs.append((k, s, y))
            if y != 0:
                S.append(s); Y.append(y == -1); SUB.append(r['subset'])
        if not trs: continue
        errs = [k for k, s, y in trs if y == -1]
        if errs:
            loc_hit.append({f: max(trs, key=lambda x: x[1][f])[0] == errs[0] for f in FEATS})
            loc_rand.append(1 / len(trs))
        fl = L[r['uid']]['final_label']
        if fl in (1, -1):
            traj_y.append(fl == -1)
            for f in FEATS: traj_s[f].append(max(x[1][f] for x in trs))
    Y = np.array(Y); SUB = np.array(SUB); subsets = sorted(set(SUB))
    lines.append(f'\n## AgentProcessBench — step AUROC, error(-1) vs correct(+1); {Y.sum()} err / {(~Y).sum()} ok steps (label-0 excluded)')
    lines.append('| feature | all | ' + ' | '.join(subsets) + ' |')
    lines.append('|' + '---|' * (2 + len(subsets)))
    for f in FEATS:
        x = np.array([s[f] for s in S])
        cells = [fmt(auroc(Y, x), boot(lambda i: auroc(Y[i], x[i]), len(Y)))]
        cells += [f'{auroc(Y[SUB == s], x[SUB == s]):.3f}' for s in subsets]
        lines.append(f'| {f} | ' + ' | '.join(cells) + ' |')
    lines.append(f'\nAPB first-error localization (argmax step == first -1 step), {len(loc_hit)} trajectories; random = {np.mean(loc_rand):.3f}')
    for f in FEATS: lines.append(f'- {f}: {np.mean([h[f] for h in loc_hit]):.3f}')
    ty = np.array(traj_y)
    lines.append(f'\nAPB trajectory AUROC (final_label -1 vs +1, max over steps), {ty.sum()} fail / {(~ty).sum()} ok')
    for f in FEATS: lines.append(f'- {f}: {auroc(ty, traj_s[f]):.3f}')


def aft(lines):
    rows = load('aft'); L = labels('aft')
    y, sub, agg = [], [], {(f, a): [] for f in FEATS for a in ('mean', 'max')}
    hit, rnd = {f: [] for f in FEATS}, []
    for r in rows:
        trs = [(k, s) for k, s in zip(r['keys'], r['steps']) if s]
        if not trs: continue
        lab = L[r['uid']]
        y.append(lab['label'] != 'safe'); sub.append(r['subset'])
        for f in FEATS:
            v = [s[f] for _, s in trs]
            agg[(f, 'mean')].append(np.mean(v)); agg[(f, 'max')].append(np.max(v))
        if lab['label'] != 'safe' and lab['mistake_step'] in [k for k, _ in trs]:
            rnd.append(1 / len(trs))
            for f in FEATS: hit[f].append(max(trs, key=lambda x: x[1][f])[0] == lab['mistake_step'])
    y = np.array(y); sub = np.array(sub); subsets = sorted(set(sub))
    lines.append(f'\n## AFTraj-2K — trajectory AUROC unsafe vs safe; {y.sum()} unsafe / {(~y).sum()} safe')
    lines.append('| feature | agg | all | ' + ' | '.join(subsets) + ' |')
    lines.append('|' + '---|' * (3 + len(subsets)))
    for (f, a), v in agg.items():
        v = np.array(v)
        cells = [fmt(auroc(y, v), boot(lambda i: auroc(y[i], v[i]), len(y)))]
        cells += [f'{auroc(y[sub == s], v[sub == s]):.3f}' for s in subsets]
        lines.append(f'| {f} | {a} | ' + ' | '.join(cells) + ' |')
    lines.append(f'\nAFT mistake-step localization (argmax step == mistake_step), {len(rnd)} unsafe runs; random = {np.mean(rnd):.3f}')
    for f in FEATS: lines.append(f'- {f}: {np.mean(hit[f]):.3f}')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    lines = ['# Agent step telemetry v1 — label-free NLL/entropy baselines (Qwen3-8B)']
    for b in sys.argv[4].split(','):
        {'tprm': tprm, 'apb': apb, 'aft': aft}[b](lines)
    txt = '\n'.join(lines); print(txt)
    open(os.path.join(OUT, f'REPORT_{sys.argv[4].replace(",", "_")}.md'), 'w').write(txt)
