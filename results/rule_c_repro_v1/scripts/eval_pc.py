"""Frozen evaluation of agentS2 rules A, B, C on dev (LABEL-USING), then marked post-evaluation diagnostics.
Rules come from out/RULES_<bench>.npz written by pseudo.py before any label of them was read (FROZEN.md)."""
import sys, json, numpy as np
WS = 'C:/Users/omris/TAU/rule_c_ws/run'
sys.path.insert(0, f'{WS}/agentX1'); sys.path.insert(0, f'{WS}/agentS2')
from common import load_dev, RA, L
import pseudo as PS
OUT = f'{WS}/agentS2/out'
PIHAT = {'processbench': .580, 'prmbench': .704}


def zero_share(fl, off):
    return float(np.mean(np.add.reduceat(fl.astype(int), off[:-1]) == 0))


def main():
    R = {}
    for bench in ('processbench', 'prmbench'):
        B = load_dev(bench, draws=2000)
        Cd, f0, sC, bd, gid = B['Cd'], B['f0'], B['sC'], B['bd'], B['gid']
        key = 'F1' if bench == 'processbench' else 'PRMScore'
        Z = np.load(f'{OUT}/RULES_{bench}.npz'); assert (Z['uids'] == Cd.uids).all()
        LN = np.load(f'{WS}/agentX1/out/LN_SCORES_{bench}.npz'); assert (LN['uids'] == Cd.uids).all(); print(bench, 'PC-vs-Mac sC max abs diff', float(np.abs(LN['sC'] - sC).max()), flush=True)  # PC repro: bitwise check relaxed
        ex = lambda op: f0 & L.expand(op, Cd.off)
        arms = {'no_gate': f0, 'ref_pihat_sC': ex(RA.gate_open(sC, PIHAT[bench])), 'ref_pihat_d2': ex(RA.gate_open(LN['d2'], PIHAT[bench])),
                'A': ex(Z['openA']), 'B': ex(Z['openB']), 'C': Z['flagsC']}
        M = {k: RA.score_flags(Cd, v, bd.W, gid) for k, v in arms.items()}
        res = {}
        for k, v in arms.items():
            m = M[k]; d = dict(zero_flag_share=zero_share(v, Cd.off), flag_share=float(v.sum() / f0.sum()))
            for kk in ([key, 'SLA'] if bench == 'processbench' else [key]):
                d[kk] = float(m[kk]); d[kk + '_ci'] = L.ci(m['_boot'][kk])
            if bench == 'processbench': d.update(err_acc=m['err_acc'], clean_acc=m['clean_acc'])
            if k in ('A', 'B', 'C'):
                d['vs'] = {ref: {kk: RA.diff_ci(m, M[ref], kk) for kk in ([key, 'SLA'] if bench == 'processbench' else [key])}
                           for ref in ('no_gate', 'ref_pihat_sC', 'ref_pihat_d2')}
            res[k] = d
        # ---------------- post-evaluation diagnostics (LABEL-USING; rules unchanged)
        diag = {}
        nC = np.diff(Cd.off); edges = np.quantile(nC, [.2, .4, .6, .8]); stC = PS.strata(nC, edges)
        sA, kerr, plen, keepB = Z['pseudo_sA'], Z['pseudo_kerr'], Z['pseudo_len'], Z['pseudo_keepB']
        isw = kerr >= 0; stP = PS.strata(plen, edges); al = Z['alphas']
        for name, msk in (('A', np.ones(len(sA), bool)), ('B', keepB)):
            cv = []; sh = []
            for a in al:
                t = PS.strat_thresholds(sA[msk & ~isw], stP[msk & ~isw], a, 5); op = sC > t[stC]
                cv.append(RA.score_flags(Cd, ex(op))[key]); sh.append(float(op.mean()))
            cv = np.array(cv); ib = int(np.argmax(cv))
            diag[f'real_curve_{name}'] = dict(real_argmax_alpha=float(al[ib]), real_max=float(cv[ib]), real_open_share_at_argmax=sh[ib],
                                              synthetic_alpha=float(al[np.argmin(np.abs(sh - Z[f'open{name}'].mean()))]) if False else None,
                                              curve={f'{a:.2f}': [round(float(c), 3), round(s, 3)] for a, c, s in zip(al, cv, sh) if round(a * 100) % 5 == 0})
        npair = int((~isw).sum()); off = np.r_[0, np.cumsum(plen)].astype(np.int64)
        ref = Z['pseudo_sg'][:off[npair]][Z['pseudo_flags'][:off[npair]]]; sgC = B['sgC']
        cv = []
        for b in al:
            tau = -np.inf if b >= 1 else (np.max(ref) if b <= 0 else np.quantile(ref, 1 - b))
            fl = f0 & (sgC > tau); cv.append((RA.score_flags(Cd, fl)[key], zero_share(fl, Cd.off)))
        cv = np.array(cv); ib = int(np.argmax(cv[:, 0]))
        diag['real_curve_C'] = dict(real_argmax_beta=float(al[ib]), real_max=float(cv[ib, 0]), zero_share_at_argmax=float(cv[ib, 1]),
                                    curve={f'{b:.2f}': [round(float(c), 3), round(float(z), 3)] for b, (c, z) in zip(al, cv) if round(b * 100) % 5 == 0})
        # oracle share of the sC rank gate (best share on real labels)
        shares = np.round(np.arange(.3, 1.0001, .01), 2)
        sc = [RA.score_flags(Cd, ex(RA.gate_open(sC, s)))[key] for s in shares]
        diag['sC_rank_gate_real_best'] = dict(share=float(shares[int(np.argmax(sc))]), value=float(max(sc)))
        # pseudo-clean contamination and real vs pseudo hit rates
        pos = {u: j for j, u in enumerate(Cd.uids)}; pp = np.array([pos[u] for u in Z['pseudo_parent'][:npair]]); kk = kerr[npair:]
        fam = Z['pseudo_family'][:npair]
        if bench == 'processbench':
            lab = Cd.lab[pp]; cont = (lab >= 0) & (lab <= kk); err_at_k = lab == kk
            pred = L.pb_predict(f0, Cd.off); y = Cd.y_ans
            diag['real_gap_hit_rate_wrong'] = float(np.mean(pred[y] == Cd.lab[y]))
        else:
            cont = np.array([Cd.err[Cd.off[p]:Cd.off[p] + k + 1].any() for p, k in zip(pp, kk)])
            err_at_k = np.array([Cd.err[Cd.off[p] + k] for p, k in zip(pp, kk)]); y = Cd.y_ans
            hit = np.add.reduceat((f0 & Cd.err).astype(int), Cd.off[:-1]) > 0
            diag['real_any_flag_hits_error_wrong'] = float(hit[y].mean())
            pred = L.pb_predict(f0, Cd.off); fe = np.array([np.flatnonzero(Cd.err[a:b])[0] if Cd.err[a:b].any() else -1 for a, b in zip(Cd.off[:-1], Cd.off[1:])])
            diag['real_first_flag_is_first_error_wrong'] = float(np.mean(pred[y] == fe[y]))
        diag['pseudo_clean_contaminated'] = dict(all=float(cont.mean()), V3=float(cont[fam == 'V3'].mean()), V4=float(cont[fam == 'V4'].mean()),
                                                 B_filtered=float(cont[keepB[:npair]].mean()), parent_error_at_k_V4=float(err_at_k[fam == 'V4'].mean()))
        diag['real_sC_auroc_wrong_vs_clean'] = L.auroc(Cd.y_ans, sC); diag['true_pi'] = float(Cd.y_ans.mean())
        # pseudo AUROC on uncontaminated pairs only
        okp = np.r_[~cont, ~cont]
        diag['pseudo_sA_auroc_uncontaminated_pairs'] = L.auroc(isw[okp], sA[okp])
        R[bench] = dict(metric=key, arms=res, diagnostics_LABEL_USING_post_eval=diag)
        print(bench, json.dumps(RA.rnd({k: {a: b for a, b in v.items() if a != 'vs'} for k, v in res.items()}, 3)))
        for k in ('A', 'B', 'C'):
            print(' ', k, {ref: {kk: (round(v['delta'], 2), np.round(v['ci'], 2).tolist()) for kk, v in d.items()} for ref, d in res[k]['vs'].items()})
        print(' diag', json.dumps(RA.rnd({k: v for k, v in diag.items()}, 3)))
    json.dump(RA.rnd(R, 4), open(f'{OUT}/EVAL_pc.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
