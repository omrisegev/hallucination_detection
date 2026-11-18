"""Hybrid and length-bin step readouts (2026-11-12, follow-up to evaluate.py; Omri's questions).

Note: top10 = mean of the min(10, n) largest tokens, so a step with n < 10 already gets the mean of all its tokens.
The short-step penalty is an extreme-value effect of LONG steps (top-10 of hundreds of tokens is high), not a short-step
readout choice. Two families tested, label-free:
  hyb_<r>_<X>   per step: answer-z of readout r (median / mean) if ntok < X, else answer-z of top10 (X = 10, 20, 30)
  top10_lenbin  raw top10 z-scored inside its token-length bin (pooled over the cell, bins = 10 ntok quantiles), then
                answer-z: each step is compared with steps of similar length (non-linear length normalization)
  median_lenbin same for the median readout
Outputs T_hybrid_cells.csv / T_hybrid_bench.csv (avg11 and avg3, mean over 4 PRMs). Reference rows top10 / median /
top10_lenres / PRM are recomputed identically for comparison.
"""
import numpy as np
import pandas as pd
import evaluate as E

A = E.A


def lenbin_z(x, ntok, nb=10):
    edges = np.unique(np.quantile(ntok, np.linspace(0, 1, nb + 1)))
    b = np.clip(np.searchsorted(edges, ntok, side='right') - 1, 0, len(edges) - 2)
    out = np.empty_like(x, dtype=float)
    for k in np.unique(b):
        m = b == k; v = x[m]; s = v.std(0)
        out[m] = (v - v.mean(0)) / np.where(s > 1e-8, s, 1.)
    return out


def main():
    rows = []
    for b in E.BENCHES:
        for p in A.PRMS:
            d = A.load(b, p); S = np.load(E.STEPS / f'{b}__{p}.npz')
            ix = E.align(d, S); R = S['readouts'][ix]; names = list(S['channel_names'].astype(str))
            rn = list(S['readout_names'].astype(str)); ntok = S['ntok'][ix].astype(float); off = d['off']
            az = lambda M: np.column_stack([E.answer_z(M[:, j], off) for j in range(M.shape[1])])
            Z = {r: az(R[:, :, rn.index(r)]) for r in ('top10', 'median', 'mean')}
            reads = dict(Z)
            lz = E.answer_z(np.log(ntok), off); L = np.column_stack([np.ones(len(lz)), lz])
            reads['top10_lenres'] = Z['top10'] - L @ np.linalg.lstsq(L, Z['top10'], rcond=None)[0]
            for r in ('median', 'mean'):
                for X in (10, 20, 30):
                    reads[f'hyb_{r}_{X}'] = np.where((ntok < X)[:, None], Z[r], Z['top10'])
            reads['top10_lenbin'] = az(lenbin_z(R[:, :, rn.index('top10')], ntok))
            reads['median_lenbin'] = az(lenbin_z(R[:, :, rn.index('median')], ntok))
            for r, M in reads.items():
                for sname, s in (('avg11', M.mean(1)), ('avg3', M[:, [names.index(n) for n in E.MAIN]].mean(1))):
                    rows.append(dict(bench=b, prm=p, readout=r, signal=sname, short_lt10=float((ntok < 10).mean()), **E.metrics(d, s)))
            rows.append(dict(bench=b, prm=p, readout='-', signal='PRM', **E.metrics(d, d['prm'])))
            print(b, p, 'done', flush=True)
    T = pd.DataFrame(rows); T.to_csv(E.HERE / 'T_hybrid_cells.csv', index=False)
    G = T.groupby(['bench', 'readout', 'signal'], sort=False).mean(numeric_only=True).reset_index()
    G.to_csv(E.HERE / 'T_hybrid_bench.csv', index=False)


if __name__ == '__main__':
    main()
