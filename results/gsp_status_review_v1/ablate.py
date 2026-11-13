"""Matched label-free fusion ablation: is GSP's cross-group binary step worth keeping? (2026-11-13)
Arms on R0 and R1 inputs (all + GAP on final score): F1 mean, F2 L-SML s, F3 GSP-T, F4a GSP-within + equal across,
F4b GSP-within + |rank-one| across, F5 L-SML-within + GSP binary across, F6 A2 readouts -> F2/F1/F3/F4a/F4b.
Labels used only for scoring.  python -B ablate.py <bench>"""
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
ROOT = Path('/home/user/hallucination_detection')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'results/sml_readout_calibration_v1'))
sys.path.insert(0, str(ROOT / 'results/gsp_ablation_baseline_v1'))
import calibrate as CB  # noqa
import importlib.util as _iu
_sp = _iu.spec_from_file_location('boot_cal', ROOT / 'results/sml_readout_calibration_v1/boot.py'); BT = _iu.module_from_spec(_sp); _sp.loader.exec_module(BT)
import r0 as R0  # noqa
AB, E, T, Rn, HY = CB.AB, CB.E, CB.T, CB.Rn, CB.HY
N, S8 = Rn.N, Rn.S8
from spectral_utils.binary_view_estimation import rank_one_offdiag  # noqa
from spectral_utils.fusion_utils import sml_fuse_signed  # noqa
BASE = 'F2 L-SML | R0'


def std(x):
    return (x - x.mean(0)) / x.std(0)


def eff(w):
    w = np.abs(np.asarray(w, float)); p = w / w.sum(); return float(1 / (p ** 2).sum())


def fusions(c, X, tag, pref=''):
    """All fusion arms on one 16-column matrix X. Returns {arm: (rank, K, eff_n)} and internal-gate info."""
    out, info = {}, {}
    m = X.shape[1]
    out[f'{pref}F1 mean'] = (X.mean(1), 1, float(m))
    tel = Rn.c4u_t(c, X)
    s = tel['s']; groups = tel['groups']; K = len(groups)
    # L-SML internals (recomputed from the same groups): signed eigvec within, signed SML across
    virt, wv = [], []
    for idx in groups:
        if len(idx) == 1:
            virt.append(X[:, idx[0]]); wv.append(np.array([1.]))
        else:
            sc, w = sml_fuse_signed(*[X[:, i] for i in idx]); virt.append(sc); wv.append(w)
    fz, cw = sml_fuse_signed(*virt) if K > 1 else (virt[0], np.array([1.]))
    info['lsml_recon_abs_corr'] = float(abs(np.corrcoef(fz, s)[0, 1]))
    wcol = np.zeros(m)
    for idx, w, cwk in zip(groups, wv, cw):
        wcol[idx] = w * cwk
    out[f'{pref}F2 L-SML'] = (s, K, eff(wcol))
    # GSP-T (F3) and its within-group pieces, rebuilt from tel
    sign = np.array([1. if np.corrcoef(X[:, j], s)[0, 1] >= 0 else -1. for j in range(m)])
    Xo = std(X * sign)
    XG, sdk = [], []
    for idx, wt in zip(groups, tel['weights']):
        xs = Xo[:, idx] @ wt; XG.append((xs - xs.mean()) / xs.std()); sdk.append(xs.std())
    XG = np.column_stack(XG)
    lw = N.lw(tel['psi'], tel['eta'])
    info['gspt_recon_maxdiff'] = float(np.abs(XG @ lw - tel['rank']).max())
    def colw(gw):
        v = np.zeros(m)
        for idx, wt, g, sd in zip(groups, tel['weights'], gw, sdk):
            v[idx] = g * wt / sd
        return v
    out[f'{pref}F3 GSP-T'] = (tel['rank'], K, eff(colw(lw)))
    out[f'{pref}F4a GSP-within + equal across'] = (XG.mean(1), K, eff(colw(np.ones(K))))
    w4 = np.abs(rank_one_offdiag(np.cov(XG.T)))
    r4 = XG @ w4
    if np.corrcoef(r4, s)[0, 1] < 0:
        r4 = -r4
    out[f'{pref}F4b GSP-within + rank-one across'] = (r4, K, eff(colw(w4)))
    if not pref:   # F5 only on the stored (top10) columns
        U = np.column_stack(virt)
        U = U * np.array([1. if np.corrcoef(U[:, k], s)[0, 1] >= 0 else -1. for k in range(K)])
        sdu = U.std(0); UG = std(U)
        VG = np.column_stack([np.where(S8.gap_flags(c, UG[:, k]), 1., -1.) for k in range(K)])
        ag = int(np.argmax([np.corrcoef(UG[:, k], s)[0, 1] for k in range(K)]))
        sp = Rn.spectral_binary(VG, anchor=ag)
        lw5 = N.lw(sp['psi'], sp['eta'])
        v = np.zeros(m)
        for idx, w, g, sd in zip(groups, wv, lw5, sdu):
            v[idx] = g * w / sd
        out[f'{pref}F5 L-SML-within + binary across'] = (UG @ lw5, K, eff(v))
    return {f'{k} | {tag}': v for k, v in out.items()}, info


def main(bench):
    t0 = time.time()
    rows, boots, gates, fits = [], [], {}, {}
    for prm in AB.PRMS:
        tc = time.time()
        c, X, ab = AB.load_cell(bench, prm)
        X1 = X[:, 2:]
        rv, act = R0.raw_rows(c, prm, bench); X0 = R0.r0_cols(rv, act, c.off)
        g = {}
        if bench in ('prmbench', 'processbench'):
            g['R0_vs_stored_N16'] = float(np.abs(X0 - E.channels(c, 'N16')).max())
        # readout candidates
        S = np.load(HY.STEPS / f'{bench}__{prm}.npz'); ix = HY.align(c, S, bench, prm)
        names = list(S['channel_names'].astype(str)); rn = list(S['readout_names'].astype(str))
        Rd = S['readouts'][ix]; ones = np.ones(len(ix), bool)
        cand1 = [[HY.column(Rd[:, names.index(n), rn.index(r)], c.off) for r in CB.READOUTS] for n in CB.TOK11]
        cand0 = [[T.answer_z_masked(Rd[:, names.index(n), rn.index(r)].astype(float), c.off, ones) for r in CB.READOUTS] for n in CB.TOK11]
        g['R1_top10_vs_X'] = float(np.abs(np.column_stack([cand1[j][0] for j in range(11)]) - X1[:, :11]).max())
        g['R0_top10_vs_r0cols'] = float(np.abs(np.column_stack([cand0[j][0] for j in range(11)]) - X0[:, :11]).max())
        g['R0_nonfinite_candidates'] = int(sum((~np.isfinite(x)).sum() for cj in cand0 for x in cj))
        assert g['R1_top10_vs_X'] < 1e-4 and g['R0_top10_vs_r0cols'] < 1e-4, (bench, prm, g)
        if 'R0_vs_stored_N16' in g:
            assert g['R0_vs_stored_N16'] < 1e-4, g
        X2_1, ch1, _ = CB.descend(cand1, X1[:, 11:])
        X2_0, ch0, _ = CB.descend(cand0, X0[:, 11:])
        A = {}
        for tag, Xa, X2 in (('R0', X0, X2_0), ('R1', X1, X2_1)):
            a, inf = fusions(c, Xa, tag); A.update(a); g[f'{tag} internal'] = inf
            a, inf = fusions(c, X2, tag, pref='F6 A2 -> '); A.update(a); g[f'{tag} A2 internal'] = inf
        for k in ('R0 internal', 'R0 A2 internal', 'R1 internal', 'R1 A2 internal'):
            assert g[k]['gspt_recon_maxdiff'] < 1e-8 and g[k]['lsml_recon_abs_corr'] > 1 - 1e-9, (bench, prm, k, g[k])
        fits[prm] = dict(A2_readouts_R1={n: CB.READOUTS[r] for n, r in zip(CB.TOK11, ch1)},
                         A2_readouts_R0={n: CB.READOUTS[r] for n, r in zip(CB.TOK11, ch0)},
                         K={k: v[1] for k, v in A.items()}, eff_n={k: v[2] for k, v in A.items()})
        gates[prm] = g
        F = {}
        for arm, (rank, K, en) in A.items():
            f = AB.gap(c, rank); F[arm] = f
            sc = AB.score(c, bench, rank, f)
            if bench == 'processbench':
                sc['FE-AUC position-matched'] = CB.pos_matched_fe_auc(c, rank)
            sc['K groups'] = K; sc['eff N features'] = en
            for mk, mv in sc.items():
                rows.append(dict(bench=bench, prm=prm, arm=arm, metric=mk, value=float(mv)))
        # paired bootstrap vs BASE
        if bench == 'processbench':
            D = BT.pb_sla_boot(c, {k: T.pb_predict(c, f) for k, f in F.items()})
        else:
            W, inv, ng = E.boot_weights(c); wa = np.zeros(len(c.ids))
            D = {k: E.boot_stats(c, E.counts_by_answer(c, f), wa, W, inv, ng)[0] * (1 if bench == 'prmbench' else 100)
                 for k, f in F.items()}
        for k, v in D.items():
            boots.append(dict(bench=bench, prm=prm, arm=k, kind='abs', value=v[0], lo=np.percentile(v[1:], 2.5), hi=np.percentile(v[1:], 97.5)))
            d = v - D[BASE]
            boots.append(dict(bench=bench, prm=prm, arm=k, kind='diff_vs_F2R0', value=d[0], lo=np.percentile(d[1:], 2.5), hi=np.percentile(d[1:], 97.5)))
        np.savez_compressed(OUT / 'cells' / f'{bench}__{prm}.npz', ids=np.asarray(c.ids).astype(str), off=np.asarray(c.off),
                            **{f'rank|{k}': v[0] for k, v in A.items()}, **{f'flags|{k}': f for k, f in F.items()})
        print(bench, prm, f'{time.time() - tc:.0f}s', json.dumps({k: v for k, v in g.items() if not isinstance(v, dict)}), flush=True)
        del c
    E.write_csv(OUT / f'RESULTS_{bench}.csv', rows)
    E.write_csv(OUT / f'BOOT_{bench}.csv', boots)
    json.dump(dict(gates=gates, fits=fits, runtime_s=time.time() - t0), open(OUT / f'FIT_{bench}.json', 'w'), indent=1, default=float)
    print('done', bench, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1])
