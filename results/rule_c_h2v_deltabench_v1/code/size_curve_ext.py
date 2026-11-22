"""Label-free stability of rule C on the external pseudo-benchmark: subsample n pseudo pairs (200 draws), rerun the
frozen beta / tau choice, report quantiles. No label is read.   python size_curve_ext.py RULEC_npz BENCH OUT_JSON"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); WS = 'C:/Users/omris/TAU/rule_c_ws/run'; sys.path.insert(0, f'{WS}/agentX1'); sys.path.insert(0, f'{WS}/agentS2')
import pseudo as PS  # noqa
import deltabench_official as O  # noqa
Z = np.load(sys.argv[1]); bench = sys.argv[2]
kerr, plen, pf, psg = Z['pseudo_kerr'], Z['pseudo_len'], Z['pseudo_flags'].astype(bool), Z['pseudo_sg']
npair = int((kerr < 0).sum()); off = np.r_[0, np.cumsum(plen)].astype(np.int64)
rng = np.random.default_rng(20261122); out = {}
def choose(idx):
    ii = np.r_[idx, idx + npair]; rows = np.concatenate([np.arange(off[i], off[i + 1]) for i in ii]); o = np.r_[0, np.cumsum(plen[ii])].astype(np.int64)
    f, sg, ke = pf[rows], psg[rows], kerr[ii]; nc = o[len(idx)]; refv = sg[:nc][f[:nc]]
    if not len(refv): return None
    cur = []
    for be in PS.ALPHAS:
        tau = -np.inf if be >= 1 else (np.max(refv) if be <= 0 else np.quantile(refv, 1 - be)); fl = f & (sg > tau)
        if bench == 'deltabench':
            cur.append(100 * O.per_sample(O.sets_from_flags(fl, o), [set() if k < 0 else {int(k) + 1} for k in ke])['F1_empty1'].mean())
        else:
            cur.append(PS.pseudo_metric(bench, fl, o, ke, None))
    cur = np.array(cur); b = PS.ALPHAS[np.flatnonzero(cur >= cur.max() - 1e-12)].max()
    return float(b), float(-np.inf if b >= 1 else (np.max(refv) if b <= 0 else np.quantile(refv, 1 - b)))
for n in [x for x in (25, 50, 100, 200, 400, 800) if x < npair] + [npair]:
    reps = 1 if n == npair else 200
    R = [choose(np.sort(rng.choice(npair, n, replace=False)) if n < npair else np.arange(npair)) for _ in range(reps)]
    R = [r for r in R if r]; b = np.array([r[0] for r in R]); t = np.array([r[1] for r in R])
    out[n] = dict(reps=len(R), beta_q=np.quantile(b, [.05, .5, .95]).tolist(), tau_q=np.quantile(t, [.05, .5, .95]).tolist())
    print(bench, n, out[n])
json.dump(out, open(sys.argv[3], 'w'), indent=1)
