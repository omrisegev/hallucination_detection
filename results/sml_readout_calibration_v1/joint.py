"""Joint label-free choice of readout, grouping and vote threshold (2026-11-13, Omri follow-up).

The GSP model is hierarchical: columns in one L-SML group share a group factor; across groups the columns are
conditionally independent given the step label (one error factor). Its second-moment form is a bifactor
correlation model:   R_ij = l_i l_j            (i, j in different groups)
                     R_ij = l_i l_j + m_i m_j    (i, j in the same group)
Hierarchical misfit = sum over off-diagonal (i, j) of (R_ij - model_ij)^2 / sum of R_ij^2.

B1  alternate until the grouping does not change (max 4 rounds):
      (a) fix the L-SML groups, coordinate descent (2 sweeps) over the 11 token readouts on the hierarchical misfit
          of the continuous columns;
      (b) recompute the L-SML groups on the new columns (same call as GSP-T: N.lsml(X, X.mean(1))).
    Then GSP-T + GAP as usual.
B2  as B1, and each of the 16 columns also chooses its vote rule; the misfit is computed on the +-1 votes (phi
    correlations), because spectral_binary consumes the votes. Vote rules after orientation (sign of corr with the
    channel-mean anchor, as GSP-T): GAP (current), top-1 and top-2 per answer, global quantile .8 and .9 of the column.
    A vote vector with error share < 2% or > 98% is not eligible. GSP-T then uses these votes instead of GAP votes;
    rank and count are unchanged (rank = group scores x log-odds weights, count = GAP on the rank).
No label is read before scoring. Labels are read only for scoring and the conditional-correlation diagnostic.

    python joint.py [bench ...]
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibrate as CB  # noqa: E402
AB, E, T, Rn, HY = CB.AB, CB.E, CB.T, CB.Rn, CB.HY
N, S8 = Rn.N, Rn.S8
from spectral_utils.binary_view_estimation import rank_one_offdiag  # noqa: E402
READOUTS, TOK11 = CB.READOUTS, CB.TOK11
RULES = ('GAP', 'top1', 'top2', 'q80', 'q90')


def rank1_masked(R, mask, iters=300):
    """u with R_ij ~= u_i u_j on the entries in mask (symmetric, no diagonal); other entries imputed."""
    M = np.where(mask, R, 0.); w, V = np.linalg.eigh(M); u = V[:, -1] * np.sqrt(max(w[-1], 0.))
    for _ in range(iters):
        M = np.where(mask, R, np.outer(u, u)); w, V = np.linalg.eigh(M)
        un = V[:, -1] * np.sqrt(max(w[-1], 0.))
        if un @ u < 0: un = -un
        if np.max(np.abs(un - u)) < 1e-10: u = un; break
        u = un
    return u


def hier_misfit(R, g):
    """Bifactor misfit of correlation matrix R for group labels g (-1 = unassigned, own group)."""
    p = len(R); g = np.where(np.asarray(g) < 0, 1000 + np.arange(p), g)
    off = ~np.eye(p, dtype=bool); same = (g[:, None] == g[None, :]) & off; cross = off & ~same
    if cross.sum() == 0:
        return np.inf
    l = rank1_masked(R, cross); Rr = R - np.outer(l, l); model = np.outer(l, l)
    for k in np.unique(g):
        idx = np.flatnonzero(g == k)
        if len(idx) >= 3:
            m = rank_one_offdiag(Rr[np.ix_(idx, idx)])
            model[np.ix_(idx, idx)] += np.outer(m, m)
        elif len(idx) == 2:                                       # a pair is fit exactly by its group factor
            model[idx[0], idx[1]] = model[idx[1], idx[0]] = R[idx[0], idx[1]]
    return float(((R - model)[off] ** 2).sum() / (R[off] ** 2).sum())


def corr_ok(X):
    if not np.all(np.isfinite(X)) or np.any(X.std(0) < 1e-8):
        return None
    return np.corrcoef(X.T)


def within_rank(x, off):
    """0 = largest value of its answer (stable, ties by position)."""
    aid = np.repeat(np.arange(len(off) - 1), np.diff(off))
    o = np.lexsort((np.arange(len(x)), -x, aid))
    r = np.empty(len(x), int); r[o] = np.arange(len(x)) - np.repeat(off[:-1], np.diff(off))
    return r


def vote_rules(c, x):
    """x already oriented (+ = error side). Returns {rule: +-1 vote vector or None if not eligible}."""
    r = within_rank(x, c.off); out = {}
    for rule in RULES:
        if rule == 'GAP': f = AB.gap(c, x)
        elif rule == 'top1': f = r < 1
        elif rule == 'top2': f = r < 2
        else: f = x > np.quantile(x, float(rule[1:]) / 100)
        out[rule] = np.where(f, 1., -1.) if .02 < f.mean() < .98 else None
    return out


def descend(cols, g, fixed=None, sweeps=2, init=None):
    """cols[j] = list of candidate vectors (None = not eligible); start at init (default all 0). fixed = extra columns."""
    ch = list(init) if init is not None else [0] * len(cols)
    X = np.column_stack([cols[j][ch[j]] for j in range(len(cols))] + ([] if fixed is None else [fixed]))
    best = hier_misfit(np.corrcoef(X.T), g)
    for _ in range(sweeps):
        for j in range(len(cols)):
            for r in range(len(cols[j])):
                if r == ch[j] or cols[j][r] is None:
                    continue
                X2 = X.copy(); X2[:, j] = cols[j][r]; R = corr_ok(X2)
                if R is None: continue
                m = hier_misfit(R, g)
                if m < best - 1e-9: best, ch[j], X = m, r, X2
    return ch, best


def c4u_t_votes(c, X, V):
    """c4u_telemetry_only_v1.run.c4u_t with the vote matrix V given (same body otherwise)."""
    off = c.off
    s, g, _ = N.lsml(X, X.mean(1))
    sign = np.array([1. if np.corrcoef(X[:, j], s)[0, 1] >= 0 else -1. for j in range(X.shape[1])])
    Xo = X * sign; Xo = (Xo - Xo.mean(0)) / Xo.std(0)
    VG, XG, gid = [], [], []
    for k in np.unique(g[g >= 0]):
        idx = np.flatnonzero(g == k); wt = np.ones(len(idx))
        if len(idx) >= 3: wt = np.abs(rank_one_offdiag(np.cov(Xo[:, idx].T)))
        wt = wt / wt.sum()
        VG.append(np.where(V[:, idx] @ wt > 0, 1., -1.)); xs = Xo[:, idx] @ wt; XG.append((xs - xs.mean()) / xs.std()); gid.append(idx)
    VG, XG = np.column_stack(VG), np.column_stack(XG)
    ag = int(np.argmax([np.corrcoef(XG[:, k], s)[0, 1] for k in range(XG.shape[1])]))
    sp = Rn.spectral_binary(VG, anchor=ag)
    rank = XG @ N.lw(sp['psi'], sp['eta'])
    return dict(rank=rank, groups=gid, prevalence=float(sp['prevalence']), psi=sp['psi'], eta=sp['eta'])


def main(benches):
    rows, fits = [], {}
    for bench in benches:
        for prm in AB.PRMS:
            c, X, ab = AB.load_cell(bench, prm)
            S = np.load(HY.STEPS / f'{bench}__{prm}.npz'); ix = HY.align(c, S, bench, prm)
            names = list(S['channel_names'].astype(str)); rn = list(S['readout_names'].astype(str))
            Rd = S['readouts'][ix]
            cand = [[HY.column(Rd[:, names.index(n), rn.index(r)], c.off) for r in READOUTS] for n in TOK11]
            assert float(np.abs(np.column_stack([cand[j][0] for j in range(11)]) - X[:, 2:13]).max()) < 1e-4
            fixed = X[:, 13:18]; rel, _ = CB.nuisance(c.off, S['ntok'][ix].astype(float)); y = CB.labels(c, bench)
            cand = [[x if np.isfinite(x).all() and x.std() > 1e-8 else None for x in cj] for cj in cand]

            # ---------------- B1: readout <-> grouping
            ch = [0] * 11; hist = []
            for it in range(4):
                Xc = np.column_stack([cand[j][ch[j]] for j in range(11)] + [fixed])
                _, g, _ = N.lsml(Xc, Xc.mean(1)); g = np.asarray(g, int)
                if hist and np.array_equal(g, hist[-1]['g']): break
                ch, m = descend(cand, g, fixed, init=ch)
                hist.append(dict(g=g, ch=ch, misfit=m))
            XB1 = np.column_stack([cand[j][ch[j]] for j in range(11)] + [fixed]); chB1 = ch

            # ---------------- B2: readout + vote rule <-> grouping (misfit on votes)
            s0 = XB1.mean(1)
            allc = [[cand[j][r] for r in range(len(READOUTS))] for j in range(11)] + [[fixed[:, j]] for j in range(5)]
            vc, meta = [], []
            for j, cj in enumerate(allc):
                vj, mj = [], []
                for r, x in enumerate(cj):
                    if x is None: continue
                    xo = x if np.corrcoef(x, s0)[0, 1] >= 0 else -x
                    for rule, v in vote_rules(c, xo).items():
                        vj.append(v); mj.append((r, rule))
                # default first: (current readout of B1 for token channels / 0, GAP)
                d = mj.index(((chB1[j] if j < 11 else 0), 'GAP'))
                vj.insert(0, vj.pop(d)); mj.insert(0, mj.pop(d)); vc.append(vj); meta.append(mj)
            chv = [0] * 16; histv = []
            for it in range(4):
                Xc = np.column_stack([allc[j][meta[j][chv[j]][0]] for j in range(16)])
                _, g, _ = N.lsml(Xc, Xc.mean(1)); g = np.asarray(g, int)
                if histv and np.array_equal(g, histv[-1]['g']): break
                chv, m = descend(vc, g, init=chv)
                histv.append(dict(g=g, misfit=m))
            XB2 = np.column_stack([allc[j][meta[j][chv[j]][0]] for j in range(16)])
            VB2 = np.column_stack([vc[j][chv[j]] for j in range(16)])

            # ---------------- score
            A = {'A0 top10': (X[:, 2:], None), 'B1 joint readout+grouping': (XB1, None), 'B2 joint readout+vote rule+grouping': (XB2, VB2)}
            save = dict(ids=np.asarray(c.ids).astype(str), off=np.asarray(c.off))
            for arm, (Xt, Vt) in A.items():
                CB.RAW.clear()
                tel = Rn.c4u_t(c, Xt) if Vt is None else c4u_t_votes(c, Xt, Vt)
                gf = AB.gap(c, tel['rank']); sc = AB.score(c, bench, tel['rank'], gf)
                if bench == 'processbench':
                    sc['FE-AUC position-matched'] = CB.pos_matched_fe_auc(c, tel['rank'])
                Vd = Vt if Vt is not None else np.column_stack([np.where(AB.gap(c, Xt[:, j] * (1 if np.corrcoef(Xt[:, j], Xt.mean(1))[0, 1] >= 0 else -1)), 1., -1.) for j in range(16)])
                _, gA, _ = N.lsml(Xt, Xt.mean(1))
                cc, cv = CB.cond_abs_corr(Xt, y), CB.cond_abs_corr(Vd, y)
                diag = {'hier misfit continuous (label-free)': hier_misfit(np.corrcoef(Xt.T), gA),
                        'hier misfit votes (label-free)': hier_misfit(np.corrcoef(Vd.T), gA),
                        'cond |corr| y=0': cc[0], 'cond |corr| y=1': cc[1],
                        'cond |corr| votes y=0': cv[0], 'cond |corr| votes y=1': cv[1],
                        'K groups': len(tel['groups']), 'prevalence': tel['prevalence'],
                        'max raw eta': float(np.max(CB.RAW['eta'])),
                        'n psi/eta out of [0,1]': int(np.sum((CB.RAW['psi'] < 0) | (CB.RAW['psi'] > 1)) + np.sum((CB.RAW['eta'] < 0) | (CB.RAW['eta'] > 1)))}
                for mk, mv in {**sc, **diag}.items():
                    rows.append(dict(bench=bench, prm=prm, arm=arm, metric=mk, value=float(mv)))
                save[f'rank|{arm}'] = tel['rank']; save[f'flags|{arm}'] = gf
                print(bench, prm, arm, {k: round(float(v), 3) for k, v in {**sc, **diag}.items() if k in (
                    'PRMScore', 'balanced F1', 'macro SLA', 'within-AUC', 'hier misfit votes (label-free)',
                    'cond |corr| votes y=0', 'K groups', 'n psi/eta out of [0,1]')}, flush=True)
            fits[f'{bench}/{prm}'] = dict(
                B1_readouts={n: READOUTS[r] for n, r in zip(TOK11, chB1)}, B1_rounds=len(hist), B1_misfit=[h['misfit'] for h in hist],
                B2_choice={T.N16[j]: [READOUTS[meta[j][chv[j]][0]] if j < 11 else 'stored', meta[j][chv[j]][1]] for j in range(16)},
                B2_rounds=len(histv), B2_misfit=[h['misfit'] for h in histv])
            np.savez_compressed(CB.CELLS / f'{bench}__{prm}__joint.npz', **save)
            del c
    tag = '_'.join(benches)
    E.write_csv(HERE / f'RESULTS_joint_{tag}.csv', rows)
    json.dump(fits, open(HERE / f'FIT_joint_{tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main(tuple(sys.argv[1:]) or AB.BENCHES)
