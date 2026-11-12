"""Arm E: ProcessBench change-point first-error rule (3-state chain clean -> first error -> after, MAP over
{1..n, no error}) on the ablation arms of ablation.py (standalone base-model telemetry).

Same rule and EM code as results/pb_first_error_rule_v1 (cp_rule.py; the anchored and FREE variants of its run.py):
  CP PRM (anchored)                 evidence = PRM risk; states named by the PRM native partition (p > .5), 1 EM step
  CP PRM + GSP full rank (anchored) evidence = [PRM risk, GSP full rank], same anchor, 10 bins
  CP GSP-T rank (free EM)           evidence = GSP-T rank; no PRM anywhere; start = first step above the global
                                    quantile that flags half of the answers; EM 30 steps, pi estimated
  CP GSP-T rank (free, pi = 1/2)    same, prior P(no error) fixed at 1/2 (no-information prior)
The prediction (first error step or none) goes through the same official metric code as ablation.py
(c4u_competitors.pb_predict / pb_metrics) as a one-hot flag vector. Needs cells/processbench__<prm>.npz.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation as AB  # noqa: E402
import cp_rule as R  # noqa: E402

T, E = AB.T, AB.E
VARIANT = sys.argv[1] if len(sys.argv) > 1 else ''          # '' = top10 readout; 'hyb10' / 'hyb30' later


def first_pass(flag, off):
    w1 = np.zeros(len(flag)); w2 = np.zeros(len(flag)); pred = []
    for a, b in zip(off[:-1], off[1:]):
        f = np.flatnonzero(flag[a:b])
        if len(f): w1[a + f[0]] = 1; w2[a + f[0] + 1:b] = 1; pred.append(f[0])
        else: pred.append(-1)
    return w1, w2, np.array(pred)


def em(x, off, w1, w2, B=10, iters=1, pi=None, h=None):
    bins = R.qbins(x if x.ndim == 2 else x[:, None], B)
    for _ in range(iters):
        l1, l2 = R.llrs(bins, w1, w2, B, 'first')
        p_ = np.mean([w1[a:b].sum() == 0 for a, b in zip(off[:-1], off[1:])]) if pi is None else pi
        P, w1, w2 = R.posterior(l1, off, p_, .15 if h is None else h, 'first', l2)
    return P


def h_from(w1, off):
    tau = [np.flatnonzero(w1[a:b])[0] for a, b in zip(off[:-1], off[1:]) if w1[a:b].any()]
    lens = [b - a for a, b in zip(off[:-1], off[1:]) if w1[a:b].any()]
    best, bh = -np.inf, .15
    for h in R.H_GRID:
        v = sum(np.log(h) + t * np.log1p(-h) - np.log1p(-(1 - h) ** n) for t, n in zip(tau, lens))
        if v > best: best, bh = v, h
    return float(bh)


def free_init(m, off, frac=.5):
    mx = np.array([m[a:b].max() for a, b in zip(off[:-1], off[1:])]); q = np.quantile(mx, 1 - frac)
    return first_pass(m >= q, off)


def to_flags(pred, off):
    f = np.zeros(off[-1], bool)
    for p, a in zip(pred, off[:-1]):
        if p >= 0: f[a + p] = True
    return f


def main():
    rows = []
    tag = f'_{VARIANT}' if VARIANT else ''
    for prm in AB.PRMS:
        c, X, ab = AB.load_cell('processbench', prm)
        S = np.load(HERE / 'cells' / f'processbench__{prm}{tag}.npz')
        off = np.asarray(c.off); assert (S['off'] == off).all()
        risk = c.arms['PRM']
        arms = {}
        w1, w2, nat = first_pass(risk > 0, off); kw = dict(pi=float(np.mean(nat < 0)), h=h_from(w1, off))
        if not VARIANT:
            arms['CP PRM (anchored)'] = R.map_pred(em(risk, off, w1, w2, B=20, **kw))
            fr = S['rank|GSP full (PRM + tele) own count']
            arms['CP PRM + GSP full rank (anchored)'] = R.map_pred(em(np.column_stack([risk, fr]), off, w1, w2, B=10, **kw))
        tr = S['rank|GSP-T own count']
        fw1, fw2, _ = free_init(tr, off)
        arms['CP GSP-T rank (free EM)'] = R.map_pred(em(tr, off, fw1, fw2, B=10, iters=30, h=h_from(fw1, off)))
        arms['CP GSP-T rank (free, pi = 1/2)'] = R.map_pred(em(tr, off, fw1, fw2, B=10, iters=30, pi=.5, h=h_from(fw1, off)))
        for k, pred in arms.items():
            f = to_flags(pred, off)
            m = T.pb_metrics(c, T.pb_predict(c, f))
            name = k if not VARIANT else k.replace('GSP-T', f'GSP-T {VARIANT}')
            rows += [dict(bench='processbench', prm=prm, arm=name, metric='avg F1', value=m['average F1']),
                     dict(bench='processbench', prm=prm, arm=name, metric='macro SLA', value=m['macro SLA']),
                     dict(bench='processbench', prm=prm, arm=name, metric='answers with no error %', value=100 * float(np.mean(pred < 0)))]
            np.save(HERE / 'cells' / f'pbcp__{prm}__{name.replace(" ", "_").replace("/", "")}.npy', pred)
            print(prm, name, round(m['average F1'], 2), round(m['macro SLA'], 2), round(100 * float(np.mean(pred < 0)), 1), flush=True)
        del c
    E.write_csv(HERE / f'RESULTS_pb_cp{tag}.csv', rows)


if __name__ == '__main__':
    main()
