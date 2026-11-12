"""Bootstrap 95% CIs: each arm minus A0, paired. PRMBench / Socratic / Hard2Verify: official metric via
E.boot_weights (source-question groups, 2,000 draws). ProcessBench: macro SLA, 2,000 answer resamples within subset."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibrate as CB  # noqa: E402
AB, E, T = CB.AB, CB.E, CB.T
BASE = 'A0 top10'


def pb_sla_boot(c, preds, draws=2000, seed=7):
    lab = np.array([T.A.PB_GOLD[u]['label'] for u in c.ids]); sub = np.array([T.A.PB_GOLD[u]['subset'] for u in c.ids])
    rng = np.random.default_rng(seed); idx = {s: np.flatnonzero((sub == s) & (lab >= 0)) for s in T.SUBSETS}
    samples = [{s: rng.choice(v, len(v)) for s, v in idx.items()} for _ in range(draws)]
    out = {}
    for k, pr in preds.items():
        hit = (pr == lab).astype(float)
        out[k] = np.array([100 * np.mean([hit[d[s]].mean() for s in T.SUBSETS]) for d in [idx] + samples])
    return out


def main(benches):
    rows = []
    for bench in benches:
        for prm in AB.PRMS:
            S = np.load(HERE / 'cells' / f'{bench}__{prm}.npz')
            F = {k[6:]: S[k] for k in S.files if k.startswith('flags|')}
            c, X, ab = AB.load_cell(bench, prm)
            if bench == 'processbench':
                D = pb_sla_boot(c, {k: T.pb_predict(c, f) for k, f in F.items()})
            else:
                W, inv, ng = E.boot_weights(c); wa = np.zeros(len(c.ids))
                D = {k: E.boot_stats(c, E.counts_by_answer(c, f), wa, W, inv, ng)[0] * (1 if bench == 'prmbench' else 100)
                     for k, f in F.items()}
            for k, v in D.items():
                rows.append(dict(bench=bench, prm=prm, arm=k, kind='abs', value=v[0], lo=np.percentile(v[1:], 2.5), hi=np.percentile(v[1:], 97.5)))
                if k != BASE:
                    d = v - D[BASE]
                    rows.append(dict(bench=bench, prm=prm, arm=f'{k} MINUS {BASE}', kind='diff', value=d[0],
                                     lo=np.percentile(d[1:], 2.5), hi=np.percentile(d[1:], 97.5)))
            print(bench, prm, 'boot done', flush=True); del c
    E.write_csv(HERE / f'BOOT_{"_".join(benches)}.csv', rows)


if __name__ == '__main__':
    main(tuple(sys.argv[1:]) or AB.BENCHES)
