"""The one SSL experiment: analysis of the 6 step tables (2026-11-18). CPU, numpy/scipy/sklearn (no torch).

Design: ANSWERS_TO_MAC_2026-11-18.md (answers 1-56) and ONE_EXPERIMENT.md (rules). Order per benchmark:
  P0   parity of run C on the full C pool (method of record: hard top10 readout, R0, L-SML, GAP) vs stored numbers
  R    realism: C2ST + correlation-pattern agreement, V2 step k vs real wrong dev steps (step level), V2 / V1 answers
       vs real wrong dev answers (answer level). Stop rules of answer 55.
  S    token readout (count-based soft top-N per channel) fitted on dev V2 half A; lambda chosen on dev (answer 28);
       channel revert guard (31); kept only if the dev conditional |corr| drops with clustered-bootstrap CI > 0.
       ILP over hard candidates (30) and hard top-round(N) are reported only.
  S2   clean proxy (bottom 30% of s_g on dev C) added to the objective (32); kept only if it improves S (CI > 0).
  A    answer features (ECDF + soft top-M per channel + Simes of s_g), answer L-SML; M fitted on the gate positive
       (V2, else V1); lambda and keep as stage S, on answer-level dev checks.
  G    pi-hat by BBE on the answer score, 2-fold cross-fit by group; PU posterior P(wrong) by isotonic ratio.
  Mark PRMBench: close the (1 - pi-hat) share of answers with the lowest answer score, GAP in the open ones.
       ProcessBench: the same ("GAP + gate"), and the change point (free EM, per-answer prior 1 - P_i fixed, B = 20).
  TEST once: label-free parts fitted on test C; N, M, lambda and the keep decisions frozen from dev.
Gold labels: dev choices named above, P0, the reports, and the test scoring. Never inside a label-free fit.

    python run_analysis.py --W <folder with steps/ and inputs/> --gold <processbench_gold.json> --out <OUT>
           [--stored-readouts <dir with processbench__qwen_prm_7b.npz, prmbench__qwen_prm_7b.npz>]
           [--stored-pb-cell <PB_qwen_prm_7b.npz>] [--steps 500] [--draws 2000] [--bench processbench,prmbench]
"""
import argparse, hashlib, json, time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

import ssl_data as D
import ssl_lib as L
import ssl_stages as S

LAMBDAS = (.1, .3, 1., 3.)
STORED = {'processbench': ('SLA', 38.47), 'prmbench': ('PRMScore', 68.26)}     # stored method-of-record values (7B, sa)
SEED = 20261116
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a); LOG.append(s); print(s, flush=True)


def ghash(g, salt):
    return int(hashlib.sha256((salt + g).encode()).hexdigest()[:8], 16)


def halves(groups, salt):
    """Deterministic 50/50 split of the groups (hash order): True = half A / fold 1."""
    u = sorted(set(groups), key=lambda g: ghash(g, salt)); a = set(u[:(len(u) + 1) // 2])
    return np.array([g in a for g in groups])


def rnd(x, k=4):
    if isinstance(x, dict): return {a: rnd(b, k) for a, b in x.items()}
    if isinstance(x, (list, tuple)): return [rnd(b, k) for b in x]
    if isinstance(x, (float, np.floating)): return round(float(x), k)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.ndarray): return rnd(x.tolist(), k)
    return x


# ------------------------------------------------------------------ metrics with clustered bootstrap
def pb_boot(C, pred, Wd, gi):
    """Avg F1 and macro SLA over draws (row 0 = point). Resampling unit = corrected group."""
    G = Wd.shape[1]; hit = (pred == C.lab).astype(float); err = C.lab != -1
    F1, SLA = np.zeros(len(Wd)), np.zeros(len(Wd))
    for s in L.SUBSETS:
        e = (C.sub == s) & err; k = (C.sub == s) & ~err
        he = np.bincount(gi[e], hit[e], G); ne = np.bincount(gi[e], None, G)
        hk = np.bincount(gi[k], (pred[k] == -1).astype(float), G); nk = np.bincount(gi[k], None, G)
        ea = 100 * (Wd @ he) / np.maximum(Wd @ ne, 1); ca = 100 * (Wd @ hk) / np.maximum(Wd @ nk, 1)
        F1 += np.where(ea + ca > 0, 2 * ea * ca / np.maximum(ea + ca, 1e-12), 0) / 4; SLA += ea / 4
    return dict(F1=F1, SLA=SLA)


def prm_boot(C, flags, Wd, gi):
    Cn = L.counts_by_answer(C.err, flags, C.keep, C.off); G = Wd.shape[1]
    Cg = np.zeros((G, 4)); np.add.at(Cg, gi, Cn)
    return dict(PRMScore=L.official_from_counts(Wd @ Cg))


def score_flags(C, flags, Wd=None, gi=None, pred=None):
    if C.bench == 'processbench':
        pred = L.pb_predict(flags, C.off) if pred is None else pred
        m = L.pb_metrics(pred, C.lab, C.sub)
        out = dict(SLA=m['macro SLA'], F1=m['average F1'], err_acc=m['error acc (macro)'], clean_acc=m['clean acc (macro)'],
                   open_share=100 * float(np.mean(pred >= 0)))
        if Wd is not None: out['_boot'] = pb_boot(C, pred, Wd, gi)
    else:
        out = dict(PRMScore=L.prmscore(C.err, flags, C.keep, C.off), flag_rate=100 * float(flags.mean()),
                   open_share=100 * float(np.mean(np.add.reduceat(flags.astype(int), C.off[:-1]) > 0)))
        k = C.keep; tp = np.sum(~C.err & ~flags & k); fn = np.sum(~C.err & flags & k)
        tn = np.sum(C.err & flags & k); fp = np.sum(C.err & ~flags & k)
        out.update(TPR_err=100 * tn / max(tn + fp, 1), FPR_err=100 * fn / max(fn + tp, 1),
                   pi_step=float(np.sum(C.err & k) / k.sum()))
        if Wd is not None: out['_boot'] = prm_boot(C, flags, Wd, gi)
    return out


def strip(d):
    return {k: v for k, v in d.items() if not k.startswith('_')}


def diff_ci(a, b, key):
    d = a['_boot'][key] - b['_boot'][key]
    return dict(delta=float(d[0]), ci=L.ci(d))


# ------------------------------------------------------------------ realism (answer 55)
def c2st_steps(Cd, V2d, rawC, rawV, seed=SEED):
    X = L.r0_cols(rawC, Cd.act, Cd.off); Xv = L.r0_cols(rawV, V2d.act, V2d.off)
    kidx = V2d.off[:-1] + V2d.k; w = Cd.wrong_step
    gC = np.repeat(Cd.group, np.diff(Cd.off))[w]; Xw = X[w]; ntC = Cd.ntok[w]
    Xk = Xv[kidx]; ntV = V2d.ntok[kidx]; gV = V2d.group
    edges = np.quantile(np.r_[ntC, ntV], [.2, .4, .6, .8])
    ia, ib = L.balance_by_strata(np.searchsorted(edges, ntV), np.searchsorted(edges, ntC), np.random.default_rng(seed))
    auc = L.c2st(Xk[ia], Xw[ib], gV[ia], gC[ib])
    agree = float(spearmanr(L.offdiag(L.spearman_matrix(Xk)), L.offdiag(L.spearman_matrix(Xw)))[0])
    return dict(auroc=auc, agreement=agree, n_each=int(len(ia)), pass_=bool(auc <= .65 and agree >= .5))


def c2st_answers(Fd, yd, gd, nd, Fv, gv, nv, edges, seed=SEED):
    w = yd.astype(bool)
    ia, ib = L.balance_by_strata(np.searchsorted(edges, nv), np.searchsorted(edges, nd[w]), np.random.default_rng(seed))
    auc = L.c2st(Fv[ia], Fd[w][ib], gv[ia], gd[w][ib])
    agree = float(spearmanr(L.offdiag(L.spearman_matrix(Fv)), L.offdiag(L.spearman_matrix(Fd[w])))[0])
    return dict(auroc=auc, agreement=agree, n_each=int(len(ia)), pass_=bool(auc <= .65 and agree >= .5))


# ------------------------------------------------------------------ answer-level helpers
def answer_cond(F, y, gi, Wd):
    w, c = y.astype(bool), ~y.astype(bool)
    return L.cond_abs_corr(L.ranks(F[w]), gi[w], L.ranks(F[c]), gi[c], Wd)


def feats(AM, P, raw, M):
    Ss = AM.sorted_ecdf(P, raw); sim, sg = AM.simes(P, raw)
    return np.column_stack([L.soft_topn(Ss, np.broadcast_to(M, (16,))), sim]), Ss, sim, sg


def gate_open(score, share):
    """Open the `share` of answers with the highest answer score (rank rule of answers 40, 44)."""
    n = len(score); k = int(round(share * n)); o = np.zeros(n, bool)
    if k: o[np.argsort(-score, kind='stable')[:k]] = True
    return o


def aligned(V, C):
    """Keep the condition answers whose parent is in C with the same number of (kept) steps."""
    pos = {u: j for j, u in enumerate(C.uids)}; n = np.diff(C.off)
    m = np.array([p in pos and n[pos[p]] == nv for p, nv in zip(V.parent, np.diff(V.off))], bool)
    return (V if m.all() else V.subset(m)), int((~m).sum())


def parent_index(Pv, C):
    pos = {u: j for j, u in enumerate(C.uids)}
    return np.array([pos[p] for p in Pv.parent])


# ------------------------------------------------------------------ one benchmark
def run_bench(bench, pools, a, R, SC):
    t0 = time.time(); out = R.setdefault(bench, {}); fit = out.setdefault('fit', {})
    C = pools[bench, 'C']; V1 = pools.get((bench, 'V1')); V2 = pools.get((bench, 'V2'))
    main = STORED[bench][0]
    # ---------------- P0 parity (full C pool, hard top10)
    raw0 = D.readout(C); ev, s0, f0, X0 = S.step_eval(C, raw0)
    p0 = dict(metric=main, value=ev['main'], stored=STORED[bench][1], pass_=bool(abs(ev['main'] - STORED[bench][1]) <= 1.),
              lsml=ev['lsml'], n_answers=C.n, n_steps=int(C.off[-1]))
    if bench == 'processbench': p0['F1'] = ev['F1']
    if a.stored_readouts:
        p = Path(a.stored_readouts) / f'{bench}__qwen_prm_7b.npz'
        if p.exists():
            Ds = np.load(p); pos = {u: j for j, u in enumerate(Ds['uids'].astype(str))}; so = Ds['offsets']
            mine, ref = [], []
            for j, u in enumerate(C.uids):
                if u in pos and so[pos[u] + 1] - so[pos[u]] == C.off[j + 1] - C.off[j]:
                    mine.append(raw0[C.off[j]:C.off[j + 1], :11]); ref.append(Ds['readouts'][so[pos[u]]:so[pos[u] + 1], :, 0])
            m_, r_ = np.concatenate(mine), np.concatenate(ref)
            p0['token_top10_vs_stored'] = dict(answers=len(mine), spearman=[float(spearmanr(m_[:, j], r_[:, j])[0]) for j in range(11)],
                                               max_abs_diff=float(np.abs(m_ - r_).max()))
    if a.stored_pb_cell and bench == 'processbench':
        Cs = np.load(a.stored_pb_cell); pos = {u: j for j, u in enumerate(Cs['sa_ids'].astype(str))}; so = Cs['sa_off']
        mine, ref = [], []
        for j, u in enumerate(C.uids):
            if u in pos and so[pos[u] + 1] - so[pos[u]] == C.off[j + 1] - C.off[j]:
                mine.append(X0[C.off[j]:C.off[j + 1]]); ref.append(Cs['sa_N16'][so[pos[u]]:so[pos[u] + 1]])
        m_, r_ = np.concatenate(mine), np.concatenate(ref)
        p0['R0_16_vs_stored_sa_N16'] = dict(answers=len(mine), spearman=[float(spearmanr(m_[:, j], r_[:, j])[0]) for j in range(16)])
    out['P0'] = p0; log(bench, 'P0', main, round(ev['main'], 2), 'stored', STORED[bench][1])
    if V2 is None:                          # PB V2 pending (2026-11-18): P0 only, no dev / test stages
        out['stopped'] = 'V2 pool missing: P0 only'; out['seconds'] = round(time.time() - t0, 1)
        log(bench, 'V2 pool missing: P0 only'); return

    # ---------------- split
    Cd, Ct = C.subset(C.side == 'dev'), C.subset(C.side == 'test')
    V2d = V2.subset(V2.side == 'dev') if V2 is not None else None; V2t = V2.subset(V2.side == 'test') if V2 is not None else None
    V1d = V1.subset(V1.side == 'dev') if V1 is not None else None; V1t = V1.subset(V1.side == 'test') if V1 is not None else None
    drops = {}
    for nm in ('V2d', 'V2t', 'V1d', 'V1t'):
        P_ = locals()[nm]
        if P_ is not None:
            P2, nd = aligned(P_, Cd if nm.endswith('d') else Ct); drops[nm] = nd
            if nm == 'V2d': V2d = P2
            elif nm == 'V2t': V2t = P2
            elif nm == 'V1d': V1d = P2
            else: V1t = P2
    out['condition_answers_dropped_parent_mismatch'] = drops
    out['split'] = {k: dict(answers=P.n, steps=int(P.off[-1])) for k, P in
                    (('C dev', Cd), ('C test', Ct), ('V2 dev', V2d), ('V2 test', V2t), ('V1 dev', V1d), ('V1 test', V1t)) if P is not None}
    out['split']['true_pi'] = dict(dev=float(Cd.y_ans.mean()), test=float(Ct.y_ans.mean()))
    bd = S.Boot(Cd.group, a.draws); gid = bd.gi
    hA = halves(Cd.group, 'halfA'); hA_v2 = np.isin(V2d.group, Cd.group[hA])

    # ---------------- dev baseline (method of record)
    raw_top10 = D.readout(Cd); ev0, s_d0, f_d0, X_d0 = S.step_eval(Cd, raw_top10, bd, gid)
    out['dev_top10'] = dict(main=ev0['main'], cond=float(ev0['cond']['primary'][0]), cond_ci=L.ci(ev0['cond']['primary']),
                            cond_wrong=float(ev0['cond']['wrong'][0]), cond_clean=float(ev0['cond']['clean'][0]),
                            auc_channel=ev0['auc_channel'], lsml=ev0['lsml'])

    # ---------------- realism, step level (top10 readout)
    rs = c2st_steps(Cd, V2d, raw_top10, D.readout(V2d)); out['realism_step_V2'] = rs
    if a.test_ignore_stops: rs = dict(rs, pass_=True); out['TEST_ONLY_stop_rules_ignored'] = True
    log(bench, 'C2ST step V2', rnd(rs))

    # ---------------- stage S
    N_final, revert_final, stageS = None, None, dict(run=False)
    if rs['pass_']:
        dA = S.StageSData(V2d.subset(hA_v2), Cd.subset(hA)); cand = []
        for lam in LAMBDAS:
            ts = time.time(); r = S.fit_stage_s(dA, lam, steps=a.steps); N = r['vals']
            evS, *_ = S.step_eval(Cd, D.readout(Cd, N), bd, gid)
            rev = np.array([(evS['auc_channel'][c] < .52) and (evS['auc_channel'][c] < ev0['auc_channel'][c]) for c in range(11)])
            if rev.any():
                evS, *_ = S.step_eval(Cd, D.readout(Cd, N, rev), bd, gid)
            cand.append(dict(lam=lam, N=N.tolist(), revert=rev.tolist(), L=r['L'], J=r['J'], auc_soft=r['auc'], inits=r['inits'],
                             main=evS['main'], cond=float(evS['cond']['primary'][0]), _cond=evS['cond']['primary'],
                             guard=bool(evS['main'] >= ev0['main'] - 1.), sec=round(time.time() - ts, 1)))
            log(bench, 'S lam', lam, 'N', np.round(N, 1).tolist(), 'rev', int(rev.sum()), main, round(evS['main'], 2),
                'cond', round(cand[-1]['cond'], 4), 'top10', round(float(ev0['cond']['primary'][0]), 4), f"{cand[-1]['sec']}s")
        ok = [c for c in cand if c['guard']]
        stageS = dict(run=True, candidates=[{k: v for k, v in c.items() if not k.startswith('_')} for c in cand])
        if ok:
            best = min(ok, key=lambda c: c['cond'])
            near = [c for c in ok if c['cond'] <= best['cond'] + .005]
            best = max(near, key=lambda c: c['main'])
            d = ev0['cond']['primary'] - best['_cond']
            stageS.update(chosen_lam=best['lam'], N=best['N'], revert=best['revert'], cond_drop=float(d[0]), cond_drop_ci=L.ci(d),
                          keep=bool(L.ci(d)[0] > 0))
            # stability: half B
            dB = S.StageSData(V2d.subset(~hA_v2), Cd.subset(~hA))
            rB = S.fit_stage_s(dB, best['lam'], steps=a.steps)
            stageS['halfB'] = dict(L_at_N_A=S.objective_at(dB, best['lam'], best['N'])['L'], L_at_10=S.objective_at(dB, best['lam'], np.full(11, 10.))['L'],
                                   N_B=rB['vals'].tolist(), max_abs_N_A_minus_N_B=float(np.max(np.abs(np.array(best['N']) - rB['vals']))))
            # hard top-round(N) check (answer 23)
            hard = [f"top{int(min(max(round(n), 1), 48))}" for n in best['N']]
            rawH = np.column_stack([np.column_stack([D.candidate(Cd, c, hard[c]) for c in range(11)]), Cd.step5])
            evH, *_ = S.step_eval(Cd, rawH, bd, gid)
            stageS['hard_round_check'] = dict(readout=hard, main=evH['main'], cond=float(evH['cond']['primary'][0]))
            # ILP over hard candidates (answer 30): report only
            il = S.ilp_readout(V2d.subset(hA_v2), Cd.subset(hA), best['lam'])
            evI, *_ = S.step_eval(Cd, S.ilp_raw(Cd, il['choice']), bd, gid)
            stageS['ilp'] = dict(il, main=evI['main'], cond=float(evI['cond']['primary'][0]))
            if stageS['keep']:
                N_final, revert_final = np.array(best['N']), np.array(best['revert'])
            log(bench, 'S chosen', best['lam'], 'keep', stageS['keep'], 'cond drop', rnd(stageS['cond_drop']), rnd(stageS['cond_drop_ci']))
        else:
            stageS.update(keep=False, reason='no lambda passed the dev guard')
    else:
        stageS['reason'] = 'V2 failed the step-level realism check'
    out['stage_S'] = stageS

    # ---------------- stage S2 (clean proxy)
    stageS2 = dict(run=False)
    if N_final is not None:
        rawS = D.readout(Cd, N_final, revert_final); evS, *_ = S.step_eval(Cd, rawS, bd, gid)
        G = L.g16_cols(rawS, Cd.act); sg, _, _ = L.lsml_linear(G)
        proxy = sg <= np.quantile(sg, .3)
        CdA = Cd.subset(hA); proxyA = np.flatnonzero(proxy[np.repeat(hA, np.diff(Cd.off))])
        d2 = S.StageSData(V2d.subset(hA_v2), CdA, proxy_rows=proxyA)
        lam = stageS['chosen_lam']; r2 = S.fit_stage_s(d2, lam, inits=(N_final, 3., 10., 30.), steps=a.steps)
        N2 = r2['vals']; ev2, *_ = S.step_eval(Cd, D.readout(Cd, N2), bd, gid)
        rev2 = np.array([(ev2['auc_channel'][c] < .52) and (ev2['auc_channel'][c] < ev0['auc_channel'][c]) for c in range(11)])
        if rev2.any(): ev2, *_ = S.step_eval(Cd, D.readout(Cd, N2, rev2), bd, gid)
        d = evS['cond']['primary'] - ev2['cond']['primary']
        stageS2 = dict(run=True, proxy_share=float(proxy.mean()), proxy_true_clean_share=float(Cd.clean_step[proxy].mean()),
                       N=N2.tolist(), revert=rev2.tolist(), main=ev2['main'], guard=bool(ev2['main'] >= ev0['main'] - 1.),
                       cond=float(ev2['cond']['primary'][0]), cond_drop_vs_S=float(d[0]), ci=L.ci(d), J_sets=r2['J_sets'])
        stageS2['keep'] = bool(stageS2['guard'] and stageS2['ci'][0] > 0)
        if stageS2['keep']: N_final, revert_final = N2, rev2
        log(bench, 'S2 keep', stageS2['keep'], 'drop vs S', rnd(stageS2['cond_drop_vs_S']), rnd(stageS2['ci']))
    out['stage_S2'] = stageS2
    fit.update(N=None if N_final is None else N_final.tolist(), revert=None if revert_final is None else revert_final.tolist())

    # ---------------- final step readout on dev
    rawC = D.readout(Cd, N_final, revert_final); evF, sF, fF, XF = S.step_eval(Cd, rawC, bd, gid)
    out['dev_final_readout'] = dict(main=evF['main'], cond=float(evF['cond']['primary'][0]), cond_ci=L.ci(evF['cond']['primary']),
                                    cond_wrong=float(evF['cond']['wrong'][0]), cond_clean=float(evF['cond']['clean'][0]),
                                    lsml=evF['lsml'], auc_channel=evF['auc_channel'])
    # position / length reports (answer 33): within-class Spearman of each R0 channel with position and log length
    pl = {}
    for name, Xr in (('top10', X_d0), ('final', XF)):
        for cls, m in (('wrong', Cd.wrong_step), ('clean', Cd.clean_step)):
            pl[f'{name} {cls}'] = dict(position=[float(spearmanr(Xr[m, j], Cd.pos[m])[0]) for j in range(16)],
                                       length=[float(spearmanr(Xr[m, j], Cd.ntok[m])[0]) for j in range(16)])
    out['position_length_corr'] = pl

    # ---------------- stage A (answer features), gate positive
    AM = S.AnswerModel(Cd, rawC, sF)
    rawV2 = D.readout(V2d, N_final, revert_final); rawV1 = D.readout(V1d, N_final, revert_final)
    one = np.ones(16)
    FC, SsC, simC, sgC = feats(AM, Cd, rawC, one); FV2, SsV2, simV2, _ = feats(AM, V2d, rawV2, one)
    FV1, SsV1, simV1, _ = feats(AM, V1d, rawV1, one)
    nC = np.diff(Cd.off); edges = np.quantile(nC, [.2, .4, .6, .8]); out['length_strata_edges'] = edges.tolist()
    rA2 = c2st_answers(FC, Cd.y_ans, Cd.group, nC, FV2, V2d.group, np.diff(V2d.off), edges)
    rA1 = c2st_answers(FC, Cd.y_ans, Cd.group, nC, FV1, V1d.group, np.diff(V1d.off), edges)
    out['realism_answer_V2'] = rA2; out['realism_answer_V1'] = rA1
    if a.test_ignore_stops: rA2 = dict(rA2, pass_=True)
    pos_name = 'V2' if rA2['pass_'] else ('V1' if rA1['pass_'] else None)
    if not rs['pass_'] and pos_name == 'V2':
        pos_name = 'V1' if rA1['pass_'] else None              # answer 55: V2 failed -> no stage A/G on V2
    out['gate_positive'] = pos_name; log(bench, 'C2ST answer V2', rnd(rA2), 'V1', rnd(rA1), 'positive', pos_name)
    AM.fit_fusion(FC); sA_ref = AM.score(FC)
    bA = S.Boot(Cd.group, a.draws)
    cond_ref = answer_cond(FC, Cd.y_ans, bA.gi, bA.W); auc_ref = L.auroc(Cd.y_ans, sA_ref)
    out['dev_answer_ref'] = dict(cond=float(cond_ref['primary'][0]), auroc=auc_ref, lsml=AM.ameta)
    M_final, stageA = one.copy(), dict(run=False)
    Ppos, SsP, simP = {'V2': (V2d, SsV2, simV2), 'V1': (V1d, SsV1, simV1)}.get(pos_name, (None, None, None))
    if pos_name:
        dAa = S.StageAData(SsP, SsC, simP, parent_index(Ppos, Cd)); cand = []
        for lam in LAMBDAS:
            r = S.fit_stage_a(dAa, lam, steps=a.steps); M = r['vals']
            F = np.column_stack([L.soft_topn(SsC, M), simC]); AM.fit_fusion(F); sA = AM.score(F)
            cc = answer_cond(F, Cd.y_ans, bA.gi, bA.W); au = L.auroc(Cd.y_ans, sA)
            cand.append(dict(lam=lam, M=M.tolist(), L=r['L'], J=r['J'], inits=r['inits'], auroc=au, cond=float(cc['primary'][0]),
                             _cond=cc['primary'], guard=bool(au >= auc_ref - .01)))
            log(bench, 'A lam', lam, 'M', np.round(M, 2).tolist(), 'auroc', round(au, 4), 'cond', round(cand[-1]['cond'], 4),
                'ref', round(float(cond_ref['primary'][0]), 4), round(auc_ref, 4))
        ok = [c for c in cand if c['guard']]
        stageA = dict(run=True, positive=pos_name, candidates=[{k: v for k, v in c.items() if not k.startswith('_')} for c in cand])
        if ok:
            best = min(ok, key=lambda c: c['cond']); near = [c for c in ok if c['cond'] <= best['cond'] + .005]
            best = max(near, key=lambda c: c['auroc']); d = cond_ref['primary'] - best['_cond']
            stageA.update(chosen_lam=best['lam'], M=best['M'], cond_drop=float(d[0]), cond_drop_ci=L.ci(d), keep=bool(L.ci(d)[0] > 0))
            if stageA['keep']: M_final = np.array(best['M'])
        else:
            stageA.update(keep=False, reason='no lambda passed the dev guard')
        log(bench, 'A keep', stageA.get('keep'), rnd(stageA.get('cond_drop')), rnd(stageA.get('cond_drop_ci')))
    out['stage_A'] = stageA; fit['M'] = M_final.tolist()
    FC = np.column_stack([L.soft_topn(SsC, M_final), simC]); AM.fit_fusion(FC); sA_C = AM.score(FC)
    cond_fin = answer_cond(FC, Cd.y_ans, bA.gi, bA.W)
    out['dev_answer_final'] = dict(cond=float(cond_fin['primary'][0]), auroc=L.auroc(Cd.y_ans, sA_C), lsml=AM.ameta,
                                   auroc_by_length=[L.auroc(Cd.y_ans[m], sA_C[m]) for m in
                                                    [np.searchsorted(edges, nC) == q for q in range(5)]])

    # ---------------- stage G (dev): BBE, 2-fold cross-fit by group
    G = dict(run=False)
    if pos_name:
        f1 = halves(Cd.group, 'gatefold'); fpos = np.isin(Ppos.group, Cd.group[f1]); pis, P_dev = [], np.zeros(Cd.n)
        lamA = stageA.get('chosen_lam') if stageA.get('keep') else None
        for fold in (True, False):
            if lamA is not None:
                dF = S.StageAData(SsP[fpos == fold], SsC, simP[fpos == fold], parent_index(Ppos.subset(fpos == fold), Cd))
                Mf = S.fit_stage_a(dF, lamA, steps=a.steps)['vals']
            else:
                Mf = one
            FCf = np.column_stack([L.soft_topn(SsC, Mf), simC]); AM.fit_fusion(FCf); sc_C = AM.score(FCf)
            sc_P = AM.score(np.column_stack([L.soft_topn(SsP, Mf), simP]))
            o_pos = fpos != fold; o_C = f1 != fold
            pi_f, info = L.bbe(sc_P[o_pos], sc_C[o_C]); pis.append(dict(pi_hat=pi_f, **info))
            P_dev[o_C] = L.pu_posterior(sc_P[o_pos], sc_C[o_C], pi_f, sc_C[o_C])
        pi_hat = float(np.mean([p['pi_hat'] for p in pis]))
        AM.fit_fusion(FC)
        alt = 'V1' if pos_name == 'V2' else 'V2'; Palt, SsA_, simA_ = {'V2': (V2d, SsV2, simV2), 'V1': (V1d, SsV1, simV1)}[alt]
        pi_alt, _ = L.bbe(AM.score(np.column_stack([L.soft_topn(SsA_, M_final), simA_])), sA_C)
        G = dict(run=True, pi_hat=pi_hat, folds=pis, pi_true=float(Cd.y_ans.mean()), pi_hat_other_positive=dict(positive=alt, pi_hat=pi_alt),
                 P_mean=float(P_dev.mean()), P_auroc=L.auroc(Cd.y_ans, P_dev))
        log(bench, 'G dev pi-hat', round(pi_hat, 3), 'true', round(G['pi_true'], 3), alt, round(pi_alt, 3))
    out['stage_G_dev'] = G

    # ---------------- marking on dev
    out['dev_marking'] = marking(Cd, sF, sA_C, G.get('pi_hat'), P_dev if pos_name else None, sgC, bd.W, gid, s_top10=s_d0)
    SC[f'{bench}_dev'] = dict(s=sF, sA=sA_C, y_step=Cd.wrong_step.astype(np.int8) - Cd.clean_step.astype(np.int8),
                              y_ans=Cd.y_ans, s_top10=s_d0)
    fit['dev_seconds'] = round(time.time() - t0, 1)

    # ================= TEST (once): label-free fits on test C; N, M, lambda, keep frozen
    bt = S.Boot(Ct.group, a.draws); git = bt.gi
    rawT = D.readout(Ct, N_final, revert_final); evT, sT, fT, XT = S.step_eval(Ct, rawT, bt, git)
    rawT0 = D.readout(Ct); evT0, sT0, fT0, XT0 = S.step_eval(Ct, rawT0, bt, git)
    AMt = S.AnswerModel(Ct, rawT, sT)
    FT, SsT, simT, sgT = feats(AMt, Ct, rawT, M_final); AMt.fit_fusion(FT); sA_T = AMt.score(FT)
    T = dict(answer_auroc=L.auroc(Ct.y_ans, sA_T), pi_true=float(Ct.y_ans.mean()), lsml_step=evT['lsml'], lsml_answer=AMt.ameta,
             cond_step=dict(top10=dict(point=float(evT0['cond']['primary'][0]), ci=L.ci(evT0['cond']['primary'])),
                            final=dict(point=float(evT['cond']['primary'][0]), ci=L.ci(evT['cond']['primary']))))
    btA = S.Boot(Ct.group, a.draws)
    FT1 = np.column_stack([L.soft_topn(SsT, one), simT])
    AMt.fit_fusion(FT1); T['cond_answer'] = dict(ref=float(answer_cond(FT1, Ct.y_ans, btA.gi, btA.W)['primary'][0]))
    AMt.fit_fusion(FT); T['cond_answer']['final'] = float(answer_cond(FT, Ct.y_ans, btA.gi, btA.W)['primary'][0])
    T['answer_auroc_by_length'] = [L.auroc(Ct.y_ans[m], sA_T[m]) for m in [np.searchsorted(edges, np.diff(Ct.off)) == q for q in range(5)]]
    pi_t, P_t = None, None
    if pos_name:
        Pt = {'V2': V2t, 'V1': V1t}[pos_name]; rawP = D.readout(Pt, N_final, revert_final)
        FP, *_ = feats(AMt, Pt, rawP, M_final); sP = AMt.score(FP)
        pi_t, info = L.bbe(sP, sA_T); P_t = L.pu_posterior(sP, sA_T, pi_t, sA_T)
        T.update(pi_hat=pi_t, bbe=info, abs_err=abs(pi_t - T['pi_true']), pi_check_pass=bool(abs(pi_t - T['pi_true']) <= .10),
                 P_mean=float(P_t.mean()), P_auroc=L.auroc(Ct.y_ans, P_t))
        alt = 'V1' if pos_name == 'V2' else 'V2'; Pa = {'V2': V2t, 'V1': V1t}[alt]
        FA, *_ = feats(AMt, Pa, D.readout(Pa, N_final, revert_final), M_final)
        T['pi_hat_other_positive'] = dict(positive=alt, pi_hat=L.bbe(AMt.score(FA), sA_T)[0])
    T['marking'] = marking(Ct, sT, sA_T, pi_t, P_t, sgT, bt.W, git, s_top10=sT0, baselines=(Ct, rawT0, XT0))
    out['test'] = T
    SC[f'{bench}_test'] = dict(s=sT, sA=sA_T, y_step=Ct.wrong_step.astype(np.int8) - Ct.clean_step.astype(np.int8),
                               y_ans=Ct.y_ans, s_top10=sT0, P=np.zeros(0) if P_t is None else P_t)
    if pos_name:
        SC[f'{bench}_test']['sA_pos'] = sP
        kk = Pt.off[:-1] + Pt.k if pos_name == 'V2' else np.zeros(0, int)
        if pos_name == 'V2':
            XP = L.r0_cols(rawP, Pt.act, Pt.off); SC[f'{bench}_test']['s_pos_k'] = (XP @ np.array(evT['W']))[kk]
    fit['seconds'] = round(time.time() - t0, 1)
    log(bench, 'TEST', rnd({k: strip(v) for k, v in T['marking'].items() if isinstance(v, dict) and 'open_share' in v}, 2))
    return out


def marking(C, s, sA, pi_hat, P, sg, Wd, gi, s_top10, baselines=None):
    """All marking arms on one C pool. s: final fused step score; sA: answer score; sg: Simes step score."""
    arms = {}
    f_mor = L.gap_flags(s_top10, C.off); arms['method of record (top10, no gate)'] = score_flags(C, f_mor, Wd, gi)
    f_gap = L.gap_flags(s, C.off); arms['final readout, no gate'] = score_flags(C, f_gap, Wd, gi)
    ex = lambda o: L.expand(o, C.off)
    if pi_hat is not None:
        op = gate_open(sA, pi_hat); arms['final + gate (pi-hat)'] = score_flags(C, f_gap & ex(op), Wd, gi)
        arms['final + gate (pi-hat)']['pi_hat'] = pi_hat
        arms['final + gate (pi-hat)']['opened_wrong_share'] = float(C.y_ans[op].mean()) if op.any() else float('nan')
        if C.bench == 'processbench':
            fitcp = L.cp_fit(s, C.off, 1 - P, B=20, iters=30); pred = L.cp_predict(fitcp, s, C.off, 1 - P)
            arms['final + change point (prior 1 - P)'] = score_flags(C, None, Wd, gi, pred=pred)
            arms['final + change point (prior 1 - P)']['h'] = fitcp['h']
    arms['final + label-perfect gate (oracle)'] = score_flags(C, f_gap & ex(C.y_ans), Wd, gi)
    simes = L.simes(sg, C.off)
    arms['final + Simes gate alpha .05'] = score_flags(C, f_gap & ex(simes <= .05), Wd, gi)
    best = None
    for al in (.001, .002, .005, .01, .02, .05, .1, .2, .3, .5, .7, 1.):
        r = score_flags(C, f_gap & ex(simes <= al)); v = r['F1'] if C.bench == 'processbench' else r['PRMScore']
        if best is None or v > best[1]: best = (al, v)
    arms[f'final + Simes gate oracle alpha {best[0]}'] = score_flags(C, f_gap & ex(simes <= best[0]), Wd, gi)
    if baselines is not None:
        Cb, raw0, X0 = baselines
        arms['baseline: equal mean of 16 R0 + GAP'] = score_flags(C, L.gap_flags(X0.mean(1), C.off), Wd, gi)
        arms['baseline: position only + GAP'] = score_flags(C, L.gap_flags(C.pos.astype(float), C.off), Wd, gi)
        arms['baseline: length only + GAP'] = score_flags(C, L.gap_flags(np.log(np.maximum(C.ntok, 1)).astype(float), C.off), Wd, gi)
        arms['baseline: mean q15_H1 (entropy) + GAP'] = score_flags(C, L.gap_flags(C.hard[:, 0, 1], C.off), Wd, gi)
        arms['baseline: mean chosen_surprisal (perplexity) + GAP'] = score_flags(C, L.gap_flags(C.hard[:, 2, 1], C.off), Wd, gi)
    key = 'F1' if C.bench == 'processbench' else 'PRMScore'
    ref = arms['method of record (top10, no gate)']
    for k, v in arms.items():
        if k != 'method of record (top10, no gate)':
            v['vs_method_of_record'] = {key: diff_ci(v, ref, key)}
            if C.bench == 'processbench': v['vs_method_of_record']['SLA'] = diff_ci(v, ref, 'SLA')
    return {k: strip(v) | ({'vs_method_of_record': v['vs_method_of_record']} if 'vs_method_of_record' in v else {}) for k, v in arms.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', required=True); ap.add_argument('--gold', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--stored-readouts'); ap.add_argument('--stored-pb-cell')
    ap.add_argument('--steps', type=int, default=500); ap.add_argument('--draws', type=int, default=2000)
    ap.add_argument('--bench', default='processbench,prmbench')
    ap.add_argument('--test-ignore-stops', action='store_true', help='code test on simulated data only: run every stage')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    pools, split = D.load_all(a.W, a.gold)
    log('pools', {f'{b}/{v}': (P.n, int(P.off[-1])) for (b, v), P in pools.items()})
    R, SC = dict(created=time.strftime('%Y-%m-%d %H:%M'), W=str(a.W), steps=a.steps, draws=a.draws, lambdas=LAMBDAS), {}
    for bench in a.bench.split(','):
        if (bench, 'C') in pools:
            run_bench(bench, pools, a, R, SC)
            json.dump(rnd(R), open(out / 'RESULTS.json', 'w'), indent=1, default=str)
            (out / 'analysis.log').write_text('\n'.join(LOG))
    R['seconds'] = round(time.time() - t0, 1)
    json.dump(rnd(R), open(out / 'RESULTS.json', 'w'), indent=1, default=str)
    np.savez_compressed(out / 'SCORES.npz', **{f'{k}__{f}': v for k, d in SC.items() for f, v in d.items()})
    (out / 'analysis.log').write_text('\n'.join(LOG))
    log('done', R['seconds'], 's')


if __name__ == '__main__':
    main()
