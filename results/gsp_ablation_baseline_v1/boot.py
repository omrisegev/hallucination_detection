"""Bootstrap 95% CIs for every arm (absolute) and for the ablation steps (paired differences).

PRMBench / Socratic / Hard2Verify: E.boot_weights (resampling source-question groups, 2,000 draws, fixed seeds of
the collection), official metric from resampled per-answer counts. ProcessBench: 2,000 answer resamples within each
subset, official average F1 of the stored predictions (pb_predict of the stored flags; CP arms from pbcp__*.npy).
Reads cells/*.npz written by ablation.py (+ hybrid suffixes when present). Output: BOOT.csv.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation as AB  # noqa: E402
E, T = AB.E, AB.T

PAIRS = [('GSP-T + GAP', 'GSP-T own count'),
         ('GSP-T own count', 'PRM native'), ('GSP-T + GAP', 'PRM + GAP'),
         ('GSP full (PRM + tele) own count', 'GSP-T own count'), ('GSP full (PRM + tele) + GAP', 'GSP-T + GAP'),
         ('GSP-T hyb10 own count', 'GSP-T own count'), ('GSP-T hyb10 + GAP', 'GSP-T + GAP'),
         ('GSP-T hyb30 own count', 'GSP-T own count'), ('GSP-T hyb30 + GAP', 'GSP-T + GAP'),
         ('CP GSP-T rank (free, pi = 1/2)', 'GSP-T own count'), ('CP PRM + GSP full rank (anchored)', 'PRM native'),
         ('CP PRM + GSP full rank (anchored)', 'GSP full (PRM + tele) own count')]


def arms_of(bench, prm):
    F = {}
    for p in sorted((HERE / 'cells').glob(f'{bench}__{prm}*.npz')):
        S = np.load(p)
        for k in S.files:
            if k.startswith('flags|'):
                F[k[6:]] = S[k]
    return F


def pb_boot(c, preds, draws=2000, seed=7):
    lab = np.array([T.A.PB_GOLD[u]['label'] for u in c.ids]); sub = np.array([T.A.PB_GOLD[u]['subset'] for u in c.ids])
    rng = np.random.default_rng(seed)
    idx = {s: np.flatnonzero(sub == s) for s in T.SUBSETS}
    samples = [np.concatenate([rng.choice(v, len(v)) for v in idx.values()]) for _ in range(draws)]
    out = {}
    for k, pred in preds.items():
        hit = np.where(lab >= 0, pred == lab, pred == -1).astype(float)
        vals = []
        for ix in [np.concatenate(list(idx.values()))] + samples:
            f1 = []
            for s in T.SUBSETS:
                m = ix[sub[ix] == s]; e = m[lab[m] >= 0]; cc = m[lab[m] < 0]
                ea, ca = 100 * hit[e].mean(), 100 * hit[cc].mean()
                f1.append(2 * ea * ca / (ea + ca) if ea + ca else 0.)
            vals.append(np.mean(f1))
        out[k] = np.array(vals)
    return out


def main():
    rows = []
    for bench in AB.BENCHES:
        for prm in AB.PRMS:
            F = arms_of(bench, prm)
            if not F:
                continue
            c, X, ab = AB.load_cell(bench, prm)
            if bench == 'processbench':
                preds = {k: T.pb_predict(c, f) for k, f in F.items()}
                for p in (HERE / 'cells').glob(f'pbcp__{prm}__*.npy'):
                    name = p.stem.split('__', 2)[2].replace('_', ' ').replace('pi = 12', 'pi = 1/2')
                    f = np.zeros(c.off[-1], bool); pr = np.load(p)
                    for q, a in zip(pr, c.off[:-1]):
                        if q >= 0: f[a + q] = True
                    preds[name] = T.pb_predict(c, f)
                D = pb_boot(c, preds)
            else:
                W, inv, ng = E.boot_weights(c)
                wa = np.zeros(len(c.ids))
                D = {k: E.boot_stats(c, E.counts_by_answer(c, f), wa, W, inv, ng)[0] * (1 if bench == 'prmbench' else 100)
                     for k, f in F.items()}
            for k, v in D.items():
                rows.append(dict(bench=bench, prm=prm, arm=k, kind='abs', value=v[0], lo=np.percentile(v[1:], 2.5),
                                 hi=np.percentile(v[1:], 97.5)))
            for a, b in PAIRS:
                if a in D and b in D:
                    d = D[a] - D[b]
                    rows.append(dict(bench=bench, prm=prm, arm=f'{a} MINUS {b}', kind='diff', value=d[0],
                                     lo=np.percentile(d[1:], 2.5), hi=np.percentile(d[1:], 97.5)))
            print(bench, prm, 'boot done', flush=True)
            del c
    E.write_csv(HERE / 'BOOT.csv', rows)


if __name__ == '__main__':
    main()
