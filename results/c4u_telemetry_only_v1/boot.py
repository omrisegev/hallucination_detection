"""Paired answer-bootstrap CIs (2,000 draws) for PRMScore (PRMBench) and average F1 / macro SLA (ProcessBench).
PRMBench: resample source-question groups (E.boot_weights groups), PRMScore from summed counts (official_from_counts).
ProcessBench: answers resampled within each subset (as c4u_competitors.pb_metrics). -> CI.csv"""
import csv, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'results/prm_boost_collection_v1'))
import evaluate_collection as E  # noqa: E402
D = np.load(HERE / 'PER_ANSWER.npz')
P = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
ARMS = ['PRM native', 'C4U (PRM + telemetry)', 'C4U-T (telemetry only)']
SUB = ('gsm8k', 'math', 'olympiadbench', 'omnimath'); B = 2000
rows = []
q = lambda x: (float(np.quantile(x, .025)), float(np.quantile(x, .975)))
for mode in ('sa', 'prm'):
    for prm in P:
        inv = D[f'{mode}|prmbench|{prm}|groups']; ng = inv.max() + 1
        rng = np.random.default_rng(20261111)
        W = np.vstack([np.ones(ng), rng.multinomial(ng, np.full(ng, 1 / ng), size=B)])
        st = {}
        for a in ARMS:
            G = np.zeros((ng, 4)); np.add.at(G, inv, D[f'{mode}|prmbench|{prm}|{a}|counts'])
            st[a] = E.official_from_counts(W @ G, 'prmbench')
        for a in ARMS:
            lo, hi = q(st[a][1:]); rows.append(dict(mode=mode, prm=prm, bench='prmbench', metric='PRMScore', arm=a, point=st[a][0], lo=lo, hi=hi))
        for a, b in (('C4U-T (telemetry only)', 'C4U (PRM + telemetry)'), ('C4U-T (telemetry only)', 'PRM native')):
            d = st[a] - st[b]; lo, hi = q(d[1:])
            rows.append(dict(mode=mode, prm=prm, bench='prmbench', metric='PRMScore', arm=f'{a} minus {b}', point=d[0], lo=lo, hi=hi))
        lab = D[f'{mode}|processbench|{prm}|label']; sub = D[f'{mode}|processbench|{prm}|subset']
        rng = np.random.default_rng(20261111)
        idx = {s: np.flatnonzero(sub == s) for s in SUB}
        draws = [np.concatenate([idx[s] for s in SUB])] + [np.concatenate([rng.choice(idx[s], len(idx[s])) for s in SUB]) for _ in range(B)]
        def met(pred, ii):
            l, p, s_ = lab[ii], pred[ii], sub[ii]; f1, sla = [], []
            for s in SUB:
                e = (s_ == s) & (l != -1); k = (s_ == s) & (l == -1)
                ea = 100 * (p[e] == l[e]).mean(); ca = 100 * (p[k] == -1).mean()
                f1.append(2 * ea * ca / (ea + ca) if ea + ca else 0.); sla.append(ea)
            return np.mean(f1), np.mean(sla)
        st = {a: np.array([met(D[f'{mode}|processbench|{prm}|{a}|pred'], ii) for ii in draws]) for a in ARMS}
        for j, m in enumerate(('average F1', 'macro SLA')):
            for a in ARMS:
                lo, hi = q(st[a][1:, j]); rows.append(dict(mode=mode, prm=prm, bench='processbench', metric=m, arm=a, point=st[a][0, j], lo=lo, hi=hi))
            for a, b in (('C4U-T (telemetry only)', 'C4U (PRM + telemetry)'), ('C4U-T (telemetry only)', 'PRM native')):
                d = st[a][:, j] - st[b][:, j]; lo, hi = q(d[1:])
                rows.append(dict(mode=mode, prm=prm, bench='processbench', metric=m, arm=f'{a} minus {b}', point=d[0], lo=lo, hi=hi))
        print(mode, prm, flush=True)
with open(HERE / 'CI.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
