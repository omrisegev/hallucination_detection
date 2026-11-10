import numpy as np, sys
sys.path.insert(0, '.')
import cp_rule as R, pbeval as V
def load(prm):
    z = np.load(f'cells/PB_{prm}.npz'); return {k: z[k] for k in z.files}
def first_pass(flag, off):
    w1 = np.zeros(len(flag)); w2 = np.zeros(len(flag)); pred = []
    for a, b in zip(off[:-1], off[1:]):
        f = np.flatnonzero(flag[a:b])
        if len(f): w1[a + f[0]] = 1; w2[a + f[0] + 1:b] = 1; pred.append(f[0])
        else: pred.append(-1)
    return w1, w2, np.array(pred)
def em(x, off, w1, w2, B=10, iters=1, model='first', temper=1., pi=None, h=None, uids=None, trace=False):
    bins = R.qbins(x if x.ndim == 2 else x[:, None], B); hist = []
    for it in range(iters):
        l1, l2 = R.llrs(bins, w1, w2, B, 'first')
        l1, l2 = l1 * temper, l2 * temper
        p_, h_ = (np.mean([w1[a:b].sum() == 0 for a, b in zip(off[:-1], off[1:])]) if pi is None else pi), (.15 if h is None else h)
        P, w1, w2 = R.posterior(l1, off, p_, h_, 'first', l2)
        if trace: m = V.metrics(uids, R.map_pred(P)); hist.append((it, round(m['avg F1'], 2), round(m['macro SLA'], 2)))
    return P, hist
