"""GSP-T on R0 telemetry columns (2026-11-12, Omri: 'try R0 as you know it').
R0 = answer-z of the raw channel value within each answer (digit channels: over digit-active steps only, 0 elsewhere)
= c4u_competitors.answer_z_masked, the stored N16 that build_X gates R1 against. R1 (used everywhere else in this
ablation) = answer-z of the global rank -> normal. Only the column construction changes; GSP-T code is unchanged.
Arms: GSP-T own count, GSP-T + GAP; PB also change-point free EM, macro SLA and first-error AUC (also for R1).
Output: RESULTS_r0.csv."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation as AB  # noqa: E402
import pb_cp as CP  # noqa: E402
T, E, Rn, BP, N = AB.T, AB.E, AB.Rn, AB.BP, AB.N


def raw_rows(c, prm, bench):
    if bench in ('prmbench', 'processbench'):
        return T.raw16(c, prm, bench)
    p = Rn.RA.ALL / f'full_sa_{T.PFILE.get(prm, prm)}_{bench}' / 'bank_v3' / 'FEATURES.npz'
    F = np.load(p); names = list(F['names'].astype(str)); V = F['values']; act = F['digit_active'].astype(bool)
    S = np.load(Path(E.DATA) / {'socratic': 'socratic', 'hard2verify': 'h2v'}[bench] / f'SCORES_{T.PFILE.get(prm, prm)}.npz')
    lay = []
    for L_ in (F, S):
        ne = L_['nonempty'].astype(bool); L = L_['nonempty_lengths'].astype(np.int64)
        lay.append((ne, np.concatenate([[0], np.cumsum(L)]), np.cumsum(ne) - 1, {u: j for j, u in enumerate(L_['uids'].astype(str))}))
    (bne, bst, brid, bpos), (pne, pst, _, ppos) = lay
    rows = []
    for u in c.ids:
        a, b = bpos[u], ppos[u]
        for k in range(pst[b + 1] - pst[b]):
            if pne[pst[b] + k]:
                assert bne[bst[a] + k]; rows.append(brid[bst[a] + k])
    rows = np.array(rows); assert len(rows) == c.off[-1]
    return np.column_stack([V[rows, names.index(n)] for n in T.N16]).astype(float), act[rows]


def r0_cols(rv, act, off):
    return np.column_stack([T.answer_z_masked(rv[:, j], off, act[:, T.DIG.index(n)] if n in T.DIG else np.ones(len(rv), bool))
                            for j, n in enumerate(T.N16)])


def fe_auc(c, s, pos=None):
    lab = np.array([T.A.PB_GOLD[u]['label'] for u in c.ids]); v = []
    for j, (a, b) in enumerate(zip(c.off[:-1], c.off[1:])):
        if lab[j] < 0: continue
        k = lab[j] if pos is None else (np.flatnonzero(pos[j] == lab[j])[0] if (pos[j] == lab[j]).any() else -1)
        if k <= 0 or k >= b - a: continue
        v.append(np.mean((s[a + k] > s[a:a + k]) + .5 * (s[a + k] == s[a:a + k])))
    return float(np.mean(v))


def pb_extra(c, tel, name, pos=None):
    out = []
    fw1, fw2, _ = CP.free_init(tel['rank'], c.off)
    arms = {f'{name} + GAP': AB.gap(c, tel['rank']), f'{name} own count': tel['flags'],
            f'CP {name} rank (free EM)': CP.to_flags(CP.R.map_pred(CP.em(tel['rank'], c.off, fw1, fw2, B=10, iters=30, h=CP.h_from(fw1, c.off))), c.off)}
    for k, f in arms.items():
        pr = T.pb_predict(c, f)
        if pos is not None: pr = np.array([pos[j][p] if p >= 0 else -1 for j, p in enumerate(pr)])
        m = T.pb_metrics(c, pr)
        out += [(k, 'macro SLA', m['macro SLA']), (k, 'macro SLA lo', m['macro SLA lo']), (k, 'macro SLA hi', m['macro SLA hi']), (k, 'avg F1', m['average F1'])]
    out.append((f'{name} rank', 'first-error AUC', fe_auc(c, tel['rank'], pos)))
    return out


def main():
    rows = []
    for bench in AB.BENCHES:
        for prm in AB.PRMS:
            c, X, ab = AB.load_cell(bench, prm)
            rv, act = raw_rows(c, prm, bench); X0 = r0_cols(rv, act, c.off)
            if bench in ('prmbench', 'processbench'):        # R0 must equal the stored N16 channels
                d = float(np.abs(X0 - E.channels(c, 'N16')).max()); assert d < 1e-4, d
            for name, Xt in (('GSP-T R0', X0), ('GSP-T R1', X[:, 2:])):
                tel = Rn.c4u_t(c, Xt)
                if bench == 'processbench':
                    for k, mk, v in pb_extra(c, tel, name):
                        rows.append(dict(bench=bench, prm=prm, arm=k, metric=mk, value=v))
                    continue
                for k, f in ((f'{name} own count', tel['flags']), (f'{name} + GAP', AB.gap(c, tel['rank']))):
                    for mk, mv in AB.score(c, bench, tel['rank'], f).items():
                        rows.append(dict(bench=bench, prm=prm, arm=k, metric=mk, value=mv))
            print(bench, prm, 'done', flush=True)
            del c
    # Qwen3-8B (Mind the Gap scorer), ProcessBench
    import runpy
    g = runpy.run_path(str(HERE / 'qwen3_8b_pb.py'))
    c, pos, rows_q = g['c'], g['pos'], g['rows']
    F = g['F']; names = g['names']
    rv = np.column_stack([F['values'][rows_q, names.index(n)] for n in T.N16]).astype(float)
    act = F['digit_active'][rows_q].astype(bool)
    for name, Xt in (('GSP-T R0', r0_cols(rv, act, c.off)), ('GSP-T R1', g['X'])):
        tel = Rn.c4u_t(c, Xt)
        for k, mk, v in pb_extra(c, tel, name, pos):
            rows.append(dict(bench='processbench', prm='qwen3_8b_mtg', arm=k, metric=mk, value=v))
    E.write_csv(HERE / 'RESULTS_r0.csv', rows)


if __name__ == '__main__':
    main()
