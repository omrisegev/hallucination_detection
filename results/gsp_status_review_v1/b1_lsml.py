"""B1 (joint readout + grouping on the bifactor misfit) with L-SML as the final fusion, on R0 and R1 (2026-11-13).
Omri asked: readout optimization in parallel with grouping, on TELE16 L-SML. Same alternation as
results/sml_readout_calibration_v1/joint.py B1 (fix L-SML groups -> coordinate descent of the 11 token readouts on the
hierarchical misfit -> recompute L-SML groups, max 4 rounds); final score = L-SML s on the chosen columns, flags = GAP.
References in the same run: F2 L-SML (R0 = TELE16, R1) and A2 -> L-SML. Paired bootstrap vs F2 L-SML R0.
Labels used only for scoring.  python -B b1_lsml.py <bench>"""
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import ablate as AL  # noqa  (loaders, CB, BT, R0)
import importlib.util as _iu
_sp = _iu.spec_from_file_location('joint', AL.ROOT / 'results/sml_readout_calibration_v1/joint.py'); J = _iu.module_from_spec(_sp); _sp.loader.exec_module(J)
CB, BT, R0, AB, E, T, HY, N = AL.CB, AL.BT, AL.R0, AL.AB, AL.E, AL.T, AL.HY, AL.N
BASE = 'F2 L-SML | R0'


def lsml_s(X):
    s, g, _ = N.lsml(X, X.mean(1)); return s, np.asarray(g, int)


def b1(cand, fixed):
    cand = [[x if np.isfinite(x).all() and x.std() > 1e-8 else None for x in cj] for cj in cand]
    ch, hist = [0] * 11, []
    for it in range(4):
        Xc = np.column_stack([cand[j][ch[j]] for j in range(11)] + [fixed])
        _, g = lsml_s(Xc)
        if hist and np.array_equal(g, hist[-1]['g']):
            break
        ch, m = J.descend(cand, g, fixed, init=ch)
        hist.append(dict(g=g, misfit=m))
    Xb = np.column_stack([cand[j][ch[j]] for j in range(11)] + [fixed])
    return Xb, ch, hist


def main(bench):
    t0 = time.time(); rows, boots, fits = [], [], {}
    for prm in AB.PRMS:
        c, X, ab = AB.load_cell(bench, prm); X1 = X[:, 2:]
        rv, act = R0.raw_rows(c, prm, bench); X0 = R0.r0_cols(rv, act, c.off)
        S = np.load(HY.STEPS / f'{bench}__{prm}.npz'); ix = HY.align(c, S, bench, prm)
        names = list(S['channel_names'].astype(str)); rn = list(S['readout_names'].astype(str))
        Rd = S['readouts'][ix]; ones = np.ones(len(ix), bool)
        cand1 = [[HY.column(Rd[:, names.index(n), rn.index(r)], c.off) for r in CB.READOUTS] for n in CB.TOK11]
        cand0 = [[T.answer_z_masked(Rd[:, names.index(n), rn.index(r)].astype(float), c.off, ones) for r in CB.READOUTS] for n in CB.TOK11]
        assert np.abs(np.column_stack([x[0] for x in cand0]) - X0[:, :11]).max() < 1e-4
        assert np.abs(np.column_stack([x[0] for x in cand1]) - X1[:, :11]).max() < 1e-4
        A, fit = {}, {}
        for tag, Xa, cand in (('R0', X0, cand0), ('R1', X1, cand1)):
            A[f'F2 L-SML | {tag}'] = lsml_s(Xa)[0]
            X2, ch2, _ = CB.descend(cand, Xa[:, 11:]); A[f'A2 -> L-SML | {tag}'] = lsml_s(X2)[0]
            Xb, chb, hist = b1(cand, Xa[:, 11:]); sb, gb = lsml_s(Xb); A[f'B1 -> L-SML | {tag}'] = sb
            fit[tag] = dict(B1_readouts={n: CB.READOUTS[r] for n, r in zip(CB.TOK11, chb)}, B1_rounds=len(hist),
                            B1_misfit=[h['misfit'] for h in hist], B1_K_final=int(len(np.unique(gb[gb >= 0]))),
                            B1_final_groups_equal_last=bool(np.array_equal(gb, hist[-1]['g'])),
                            A2_readouts={n: CB.READOUTS[r] for n, r in zip(CB.TOK11, ch2)})
        F = {}
        for arm, rank in A.items():
            f = AB.gap(c, rank); F[arm] = f; sc = AB.score(c, bench, rank, f)
            if bench == 'processbench': sc['FE-AUC position-matched'] = CB.pos_matched_fe_auc(c, rank)
            for mk, mv in sc.items(): rows.append(dict(bench=bench, prm=prm, arm=arm, metric=mk, value=float(mv)))
        if bench == 'processbench':
            D = BT.pb_sla_boot(c, {k: T.pb_predict(c, f) for k, f in F.items()})
        else:
            W, inv, ng = E.boot_weights(c); wa = np.zeros(len(c.ids))
            D = {k: E.boot_stats(c, E.counts_by_answer(c, f), wa, W, inv, ng)[0] * (1 if bench == 'prmbench' else 100) for k, f in F.items()}
        for k, v in D.items():
            d = v - D[BASE]
            boots.append(dict(bench=bench, prm=prm, arm=k, abs=v[0], diff=d[0], lo=np.percentile(d[1:], 2.5), hi=np.percentile(d[1:], 97.5)))
        fits[prm] = fit
        print(bench, prm, {k: round(float(D[k][0]), 2) for k in D}, fit['R0']['B1_rounds'], fit['R1']['B1_rounds'], flush=True)
        del c
    E.write_csv(OUT / f'B1LSML_RESULTS_{bench}.csv', rows); E.write_csv(OUT / f'B1LSML_BOOT_{bench}.csv', boots)
    json.dump(dict(fits=fits, runtime_s=time.time() - t0), open(OUT / f'B1LSML_FIT_{bench}.json', 'w'), indent=1, default=float)
    print('done', bench, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1])
