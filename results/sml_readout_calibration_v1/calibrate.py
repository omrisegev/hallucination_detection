"""Label-free per-channel readout calibration toward the SML assumptions (2026-11-12, Omri's idea).

Question: can a label-free criterion choose the step readout of each of the 11 TELE16 token channels (today top-10
mean for all) so that the 16 columns better meet the SML assumptions (one latent factor, conditional independence
given the step label)? Pipeline is fixed: GSP-T (c4u_telemetry_only_v1.run.c4u_t, no PRM) + GAP count, on the
standalone base-model telemetry.

Readout grid (from readout_steps_sa_v1, 7 per token channel): top10, mean, top20pct, p90, median, frac_above,
top10_drop1. Every candidate goes through the frozen R1 column construction (global rank -> normal, then answer-z).
The 5 step-statistic channels (realized_z, realized_drv, 3 digit channels) have no readout grid and stay as stored.

Label-free nuisance covariates per step: relative position k/(n-1) in the answer, its square, log(1 + ntok).

Arms:
  A0  top10 for all 11 token channels (baseline; must equal GSP-T + GAP of gsp_ablation_baseline_v1)
  A1  per channel, the readout with the smallest max(|corr(col, rel_pos)|, |corr(col, log ntok)|)       (criterion C2)
  A2  coordinate descent (2 sweeps) that minimizes the rank-1 misfit of the 16 x 16 correlation matrix     (C1)
  A3  residualize all 16 columns on the nuisance covariates, then A2's search on the residualized columns  (C3, main)
  A4  residualize all 16 columns, top10 readout                                                           (control)
Rank-1 misfit = sum_{i != j} (R_ij - u_i u_j)^2 / sum_{i != j} R_ij^2, u = rank_one_offdiag(R).
No label is read in A1-A4. Labels are read only for scoring and for the conditional-correlation diagnostic.

    python calibrate.py [bench ...]
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'results/gsp_ablation_baseline_v1'))
import ablation as AB  # noqa: E402
import hybrid as HY  # noqa: E402
from spectral_utils.binary_view_estimation import rank_one_offdiag  # noqa: E402
E, T, Rn = AB.E, AB.T, AB.Rn
CELLS = HERE / 'cells'; CELLS.mkdir(exist_ok=True)
TOK11 = HY.TOK11
READOUTS = ('top10', 'mean', 'top20pct', 'p90', 'median', 'frac_above', 'top10_drop1')

# keep the unclipped psi / eta of the cross-group fit (spectral_binary clips them to [0, 1])
_sb = Rn.spectral_binary
RAW = {}


def _spectral_binary(V, anchor=0):
    sp = _sb(V, anchor)
    RAW['psi'] = (1 + sp['delta'] + sp['mu'] - sp['delta'] * sp['b']) / 2
    RAW['eta'] = (1 + sp['delta'] - sp['mu'] + sp['delta'] * sp['b']) / 2
    return sp


Rn.spectral_binary = _spectral_binary


def nuisance(off, ntok):
    n = np.diff(off); k = np.arange(off[-1]) - np.repeat(off[:-1], n)
    rel = k / np.maximum(np.repeat(n, n) - 1, 1)
    return rel, np.column_stack([np.ones(len(rel)), rel, rel ** 2, np.log1p(ntok)])


def residualize(x, Z, off):
    beta, *_ = np.linalg.lstsq(Z, x, rcond=None)
    return E.answer_z(x - Z @ beta, off)


def misfit(X):
    if not np.all(np.isfinite(X)) or np.any(X.std(0) < 1e-8):
        return np.inf                                          # constant / invalid candidate column: not eligible
    R = np.corrcoef(X.T); u = rank_one_offdiag(R); o = ~np.eye(len(R), dtype=bool)
    return float(((R - np.outer(u, u))[o] ** 2).sum() / (R[o] ** 2).sum())


def nuis_score(x, rel, logn):
    if not np.all(np.isfinite(x)) or x.std() < 1e-8:
        return np.inf
    return max(abs(np.corrcoef(x, rel)[0, 1]), abs(np.corrcoef(x, logn)[0, 1]))


def descend(cand, fixed, sweeps=2):
    """Coordinate descent over the readout of each token channel; start = top10. cand[j][r]: column."""
    ch = [0] * len(cand)
    X = np.column_stack([cand[j][0] for j in range(len(cand))] + [fixed])
    best = misfit(X); trace = [best]
    for _ in range(sweeps):
        for j in range(len(cand)):
            for r in range(len(READOUTS)):
                if r == ch[j]:
                    continue
                X2 = X.copy(); X2[:, j] = cand[j][r]; m = misfit(X2)
                if m < best - 1e-9:
                    best, ch[j], X = m, r, X2
            trace.append(best)
    return X, ch, trace


def labels(c, bench):
    """Step labels for scoring and diagnostics only. ProcessBench: steps before the first error = 0, first error = 1,
    later steps unlabeled; correct answers all 0."""
    if bench == 'processbench':
        lab = np.array([T.A.PB_GOLD[u]['label'] for u in c.ids])
        y = np.full(c.off[-1], -1)
        for j, (a, b) in enumerate(zip(c.off[:-1], c.off[1:])):
            if lab[j] < 0:
                y[a:b] = 0
            elif lab[j] < b - a:
                y[a:a + lab[j]] = 0; y[a + lab[j]] = 1
        return y
    y = np.asarray(c.err).astype(int)
    if getattr(c, 'keep', None) is not None:
        y = np.where(c.keep, y, -1)
    return y


def cond_abs_corr(X, y):
    out = {}
    for v in (0, 1):
        R = np.corrcoef(X[y == v].T); o = ~np.eye(len(R), dtype=bool)
        out[v] = float(np.abs(R[o]).mean())
    return out


def pos_matched_fe_auc(c, rank):
    """ProcessBench: first-error step at position k against step k of answers that are still correct at k
    (correct answers, or first error later than k). AUC per k, weighted by the number of first errors at k."""
    lab = np.array([T.A.PB_GOLD[u]['label'] for u in c.ids]); n = np.diff(c.off)
    num = den = 0.
    for k in range(int(n.max())):
        P = [rank[c.off[j] + k] for j in np.flatnonzero((lab == k) & (n > k))]
        Q = [rank[c.off[j] + k] for j in np.flatnonzero(((lab < 0) | (lab > k)) & (n > k))]
        if not P or not Q:
            continue
        P, Q = np.array(P), np.sort(np.array(Q))
        a = (np.searchsorted(Q, P, 'left') + np.searchsorted(Q, P, 'right')) / 2 / len(Q)
        num += a.sum(); den += len(P)
    return num / den


def fe_auc(c, rank):
    lab = np.array([T.A.PB_GOLD[u]['label'] for u in c.ids]); r = []
    for j, (a, b) in enumerate(zip(c.off[:-1], c.off[1:])):
        k = lab[j]
        if 0 < k < b - a:
            r.append(np.mean((rank[a + k] > rank[a:a + k]) + .5 * (rank[a + k] == rank[a:a + k])))
    return float(np.mean(r))


def main(benches):
    rows, fits = [], {}
    for bench in benches:
        for prm in AB.PRMS:
            c, X, ab = AB.load_cell(bench, prm)
            S = np.load(HY.STEPS / f'{bench}__{prm}.npz'); ix = HY.align(c, S, bench, prm)
            names = list(S['channel_names'].astype(str)); rn = list(S['readout_names'].astype(str))
            R = S['readouts'][ix]; ntok = S['ntok'][ix].astype(float)
            cand = [[HY.column(R[:, names.index(n), rn.index(r)], c.off) for r in READOUTS] for n in TOK11]
            gate = float(np.abs(np.column_stack([cand[j][0] for j in range(11)]) - X[:, 2:13]).max())
            assert gate < 1e-4, (bench, prm, gate)
            fixed = X[:, 13:18]
            rel, Z = nuisance(c.off, ntok); logn = np.log1p(ntok)
            cand_r = [[residualize(x, Z, c.off) for x in cj] for cj in cand]
            fixed_r = np.column_stack([residualize(fixed[:, j], Z, c.off) for j in range(5)])

            A = {}
            A['A0 top10'] = (X[:, 2:].copy(), [0] * 11)
            ch1 = [int(np.argmin([nuis_score(x, rel, logn) for x in cj])) for cj in cand]
            A['A1 nuisance-decorrelated readout'] = (np.column_stack([cand[j][ch1[j]] for j in range(11)] + [fixed]), ch1)
            X2, ch2, tr2 = descend(cand, fixed)
            A['A2 min rank-1 misfit'] = (X2, ch2)
            X3, ch3, tr3 = descend(cand_r, fixed_r)
            A['A3 residualize + min misfit'] = (X3, ch3)
            A['A4 residualize, top10'] = (np.column_stack([cand_r[j][0] for j in range(11)] + [fixed_r]), [0] * 11)

            y = labels(c, bench)
            save = dict(ids=np.asarray(c.ids).astype(str), off=np.asarray(c.off))
            for arm, (Xt, ch) in A.items():
                RAW.clear()
                tel = Rn.c4u_t(c, Xt)
                gf = AB.gap(c, tel['rank'])
                sc = AB.score(c, bench, tel['rank'], gf)
                if bench == 'processbench':
                    sc['FE-AUC (vs earlier steps)'] = fe_auc(c, tel['rank'])
                    sc['FE-AUC position-matched'] = pos_matched_fe_auc(c, tel['rank'])
                cc = cond_abs_corr(Xt, y)
                diag = {'misfit (label-free)': misfit(Xt),
                        'mean |corr(col, rel_pos)| (label-free)': float(np.mean([abs(np.corrcoef(Xt[:, j], rel)[0, 1]) for j in range(16)])),
                        'mean |corr(col, rel_pos)| of rank': abs(float(np.corrcoef(tel['rank'], rel)[0, 1])),
                        'cond |corr| y=0': cc[0], 'cond |corr| y=1': cc[1],
                        'K groups': len(tel['groups']), 'prevalence': tel['prevalence'],
                        'max raw psi': float(np.max(RAW['psi'])), 'max raw eta': float(np.max(RAW['eta'])),
                        'n psi/eta out of [0,1]': int(np.sum((RAW['psi'] < 0) | (RAW['psi'] > 1)) + np.sum((RAW['eta'] < 0) | (RAW['eta'] > 1)))}
                for mk, mv in {**sc, **diag}.items():
                    rows.append(dict(bench=bench, prm=prm, arm=arm, metric=mk, value=float(mv)))
                fits[f'{bench}/{prm}/{arm}'] = dict(readouts={n: READOUTS[r] for n, r in zip(TOK11, ch)},
                                                     groups=[[T.N16[j] for j in g] for g in tel['groups']],
                                                     psi_raw=[float(v) for v in RAW['psi']], eta_raw=[float(v) for v in RAW['eta']])
                save[f'rank|{arm}'] = tel['rank']; save[f'flags|{arm}'] = gf
                print(bench, prm, arm, {k: round(v, 3) for k, v in {**sc, **diag}.items() if k in (
                    'PRMScore', 'balanced F1', 'macro SLA', 'within-AUC', 'FE-AUC position-matched', 'misfit (label-free)',
                    'cond |corr| y=0', 'n psi/eta out of [0,1]')}, flush=True)
            fits[f'{bench}/{prm}/trace'] = dict(A2=tr2, A3=tr3)
            np.savez_compressed(CELLS / f'{bench}__{prm}.npz', **save)
            del c
    tag = '_'.join(benches)
    E.write_csv(HERE / f'RESULTS_{tag}.csv', rows)
    json.dump(fits, open(HERE / f'FIT_{tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main(tuple(sys.argv[1:]) or AB.BENCHES)
