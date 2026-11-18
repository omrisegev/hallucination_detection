"""Dev-only check of a MIXTURE of V3 and V4 as the known-wrong set (Omri, thread, 2026-11-19 00:21 Israel:
"check the V3 + V4 mixture, and whether it also passes the clean proxy"). CPU. No test data is read into any statistic.

Mixture = all dev V3 answers + all dev V4 answers (union; a parent can give one V3 and one V4 child).
PRMBench is about 50/50. ProcessBench is about 80/20 V3/V4, because V4 exists for only 25% of the PB dev answers.

1. Realism (the pre-registered stop rule, the same functions and seed as run_analysis.py / realism_v34.py):
   step level: RA.c2st_steps of the mixture's step k vs the real wrong dev steps; answer level: RA.c2st_answers on the
   17 answer features (M = 1). Pass: AUROC <= .65 and pattern agreement >= .5. Also (report only): loudness and the
   answer-level classifier vs the answers' own parents (realism_v34.py).
   Code check: V3 alone and V4 alone must give the REALISM_V34.json numbers.
2. pi-hat (the pihat_dev.py computation, stage G with M = 1, 2-fold cross-fit by group): mixture as the known-wrong
   set, both benchmarks; V3 alone and V4 alone as reference rows. Check: |pi-hat - true pi| <= .10.
3. Only on a benchmark where the mixture passes the STEP-level rule: stage S (token readout) with the mixture in place
   of V2, then stage S2 (clean proxy: bottom 30% of the new fused score s_g on dev C), exactly as run_analysis.py:
   same lambdas, dev guard (main metric >= top10 - 1), choice rule, keep rule (clustered-bootstrap CI of the drop of
   the dev conditional |corr| > 0), S2 only after a kept S. Report only: half-B stability, hard top-round(N) check.
   The ILP report of stage S is not run. Stages A and G after S are not run (not asked).
Gold labels: realism sets, true pi, AUROC values, the dev checks of S / S2 (as pre-registered). Never inside a fit.

    python mix_v34.py --W <dir with steps/ inputs/> --gold <processbench_gold.json> --out <OUT>
"""
import argparse, json, time
from pathlib import Path

import numpy as np

import realism_v34 as RV
import run_analysis as RA
import ssl_data as D
import ssl_lib as L
import ssl_stages as S

EXPECT = {('prmbench', 'step', 'V3'): .6987, ('prmbench', 'step', 'V4'): .6539, ('prmbench', 'answer', 'V3'): .5583,
          ('prmbench', 'answer', 'V4'): .6172, ('processbench', 'step', 'V3'): .6761, ('processbench', 'step', 'V4'): .7865,
          ('processbench', 'answer', 'V3'): .6366, ('processbench', 'answer', 'V4'): .6753}
MIX = 'V3+V4'
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a); LOG.append(s); print(s, flush=True)


def concat(Ps, name):
    """One pool from several condition pools (answers stacked; step arrays stacked; offsets rebuilt)."""
    P = D.Pool(); P.bench, P.V = Ps[0].bench, name
    for k in ('uids', 'group', 'parent', 'k', 'side'):
        setattr(P, k, np.concatenate([getattr(p, k) for p in Ps]))
    for k in ('top', 'hard', 'step5', 'act', 'ntok', 'pos', 'nonempty'):
        setattr(P, k, np.concatenate([getattr(p, k) for p in Ps]))
    P.off = np.r_[0, np.cumsum(np.concatenate([np.diff(p.off) for p in Ps]))].astype(np.int64)
    assert len(set(P.uids)) == P.n and P.off[-1] == len(P.top)
    return P


def pihat(Cd, AM, sC, Pd, f1):
    """pihat_dev.py: BBE(scores of the positive answers of the other fold, scores of the C answers of the other fold)."""
    sP = AM.score(RA.feats(AM, Pd, D.readout(Pd), np.ones(16))[0])
    fpos = np.isin(Pd.group, Cd.group[f1]); folds, P_dev = [], np.zeros(Cd.n)
    for fold in (True, False):
        o_pos, o_C = fpos != fold, f1 != fold
        pi_f, info = L.bbe(sP[o_pos], sC[o_C]); folds.append(dict(pi_hat=pi_f, n_pos=int(o_pos.sum()), n_C=int(o_C.sum()), **info))
        P_dev[o_C] = L.pu_posterior(sP[o_pos], sC[o_C], pi_f, sC[o_C])
    ph = float(np.mean([f['pi_hat'] for f in folds])); pt = float(Cd.y_ans.mean()); op = RA.gate_open(sC, ph)
    return dict(n_positive=Pd.n, pi_hat=ph, pi_true=pt, abs_err=abs(ph - pt), pi_check_pass_10pts=bool(abs(ph - pt) <= .10),
                folds=folds, positive_vs_C_score_auroc=L.auroc(np.r_[np.ones(Pd.n), np.zeros(Cd.n)].astype(bool), np.r_[sP, sC]),
                P_mean=float(P_dev.mean()), P_auroc=L.auroc(Cd.y_ans, P_dev), gate_open_share=float(op.mean()),
                gate_open_true_wrong_share=float(Cd.y_ans[op].mean()) if op.any() else None)


def stage_s_s2(Cd, Pd, steps, draws):
    """run_analysis.py stages S and S2 on dev, with the positive pool Pd in place of V2 (same rules)."""
    bd = S.Boot(Cd.group, draws); gid = bd.gi
    hA = RA.halves(Cd.group, 'halfA'); hA_p = np.isin(Pd.group, Cd.group[hA])
    raw_top10 = D.readout(Cd); ev0, *_ = S.step_eval(Cd, raw_top10, bd, gid)
    main = 'SLA' if Cd.bench == 'processbench' else 'PRMScore'
    out = dict(metric=main, dev_top10=dict(main=ev0['main'], cond=float(ev0['cond']['primary'][0]), cond_ci=L.ci(ev0['cond']['primary']),
                                          auc_channel=ev0['auc_channel']))
    N_final, revert_final = None, None
    dA = S.StageSData(Pd.subset(hA_p), Cd.subset(hA)); cand = []
    for lam in RA.LAMBDAS:
        ts = time.time(); r = S.fit_stage_s(dA, lam, steps=steps); N = r['vals']
        evS, *_ = S.step_eval(Cd, D.readout(Cd, N), bd, gid)
        rev = np.array([(evS['auc_channel'][c] < .52) and (evS['auc_channel'][c] < ev0['auc_channel'][c]) for c in range(11)])
        if rev.any():
            evS, *_ = S.step_eval(Cd, D.readout(Cd, N, rev), bd, gid)
        cand.append(dict(lam=lam, N=N.tolist(), revert=rev.tolist(), L=r['L'], J=r['J'], auc_soft=r['auc'], inits=r['inits'],
                         main=evS['main'], cond=float(evS['cond']['primary'][0]), _cond=evS['cond']['primary'],
                         guard=bool(evS['main'] >= ev0['main'] - 1.), sec=round(time.time() - ts, 1)))
        log(Cd.bench, 'S lam', lam, 'N', np.round(N, 1).tolist(), 'rev', int(rev.sum()), main, round(evS['main'], 2),
            'cond', round(cand[-1]['cond'], 4), 'top10', round(float(ev0['cond']['primary'][0]), 4), f"{cand[-1]['sec']}s")
    ok = [c for c in cand if c['guard']]
    stageS = dict(run=True, candidates=[{k: v for k, v in c.items() if not k.startswith('_')} for c in cand])
    if ok:
        best = min(ok, key=lambda c: c['cond'])
        near = [c for c in ok if c['cond'] <= best['cond'] + .005]
        best = max(near, key=lambda c: c['main'])
        d = ev0['cond']['primary'] - best['_cond']
        stageS.update(chosen_lam=best['lam'], N=best['N'], revert=best['revert'], main=best['main'], cond=best['cond'],
                      cond_drop=float(d[0]), cond_drop_ci=L.ci(d), keep=bool(L.ci(d)[0] > 0))
        dB = S.StageSData(Pd.subset(~hA_p), Cd.subset(~hA))
        rB = S.fit_stage_s(dB, best['lam'], steps=steps)
        stageS['halfB'] = dict(L_at_N_A=S.objective_at(dB, best['lam'], best['N'])['L'],
                               L_at_10=S.objective_at(dB, best['lam'], np.full(11, 10.))['L'],
                               N_B=rB['vals'].tolist(), max_abs_N_A_minus_N_B=float(np.max(np.abs(np.array(best['N']) - rB['vals']))))
        hard = [f"top{int(min(max(round(n), 1), 48))}" for n in best['N']]
        rawH = np.column_stack([np.column_stack([D.candidate(Cd, c, hard[c]) for c in range(11)]), Cd.step5])
        evH, *_ = S.step_eval(Cd, rawH, bd, gid)
        stageS['hard_round_check'] = dict(readout=hard, main=evH['main'], cond=float(evH['cond']['primary'][0]))
        if stageS['keep']:
            N_final, revert_final = np.array(best['N']), np.array(best['revert'])
        log(Cd.bench, 'S chosen', best['lam'], 'keep', stageS['keep'], 'cond drop', RA.rnd(stageS['cond_drop']), RA.rnd(stageS['cond_drop_ci']))
    else:
        stageS.update(keep=False, reason='no lambda passed the dev guard')
    out['stage_S'] = stageS

    stageS2 = dict(run=False, reason='stage S not kept: S2 runs only after a kept S (pre-registered)')
    if N_final is not None:
        rawS = D.readout(Cd, N_final, revert_final); evS, *_ = S.step_eval(Cd, rawS, bd, gid)
        G = L.g16_cols(rawS, Cd.act); sg, _, _ = L.lsml_linear(G)
        proxy = sg <= np.quantile(sg, .3)
        CdA = Cd.subset(hA); proxyA = np.flatnonzero(proxy[np.repeat(hA, np.diff(Cd.off))])
        d2 = S.StageSData(Pd.subset(hA_p), CdA, proxy_rows=proxyA)
        lam = stageS['chosen_lam']; r2 = S.fit_stage_s(d2, lam, inits=(N_final, 3., 10., 30.), steps=steps)
        N2 = r2['vals']; ev2, *_ = S.step_eval(Cd, D.readout(Cd, N2), bd, gid)
        rev2 = np.array([(ev2['auc_channel'][c] < .52) and (ev2['auc_channel'][c] < ev0['auc_channel'][c]) for c in range(11)])
        if rev2.any(): ev2, *_ = S.step_eval(Cd, D.readout(Cd, N2, rev2), bd, gid)
        d = evS['cond']['primary'] - ev2['cond']['primary']
        stageS2 = dict(run=True, proxy_share=float(proxy.mean()), proxy_true_clean_share=float(Cd.clean_step[proxy].mean()),
                       N=N2.tolist(), revert=rev2.tolist(), main=ev2['main'], guard=bool(ev2['main'] >= ev0['main'] - 1.),
                       cond=float(ev2['cond']['primary'][0]), cond_drop_vs_S=float(d[0]), ci=L.ci(d), J_sets=r2['J_sets'])
        stageS2['keep'] = bool(stageS2['guard'] and stageS2['ci'][0] > 0)
        if stageS2['keep']: N_final, revert_final = N2, rev2
        log(Cd.bench, 'S2 keep', stageS2['keep'], 'drop vs S', RA.rnd(stageS2['cond_drop_vs_S']), RA.rnd(stageS2['ci']))
    out['stage_S2'] = stageS2
    rawF = D.readout(Cd, N_final, revert_final); evF, *_ = S.step_eval(Cd, rawF, bd, gid)
    out['dev_final_readout'] = dict(N=None if N_final is None else N_final.tolist(), main=evF['main'],
                                    cond=float(evF['cond']['primary'][0]), cond_ci=L.ci(evF['cond']['primary']))
    return out


def figure(R, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID = '#fcfcfb', '#1f1e1c', '#52514e', '#e4e3df'
    COL = {'V3': '#1baf7a', 'V4': '#d6a12a', MIX: '#2a78d6'}
    plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': GRID, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
                         'legend.frameon': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6})
    benches = [b for b in ('prmbench', 'processbench') if b in R]
    fig, ax = plt.subplots(1, len(benches) + 1, figsize=(5.2 * (len(benches) + 1), 4.6), squeeze=False); ax = ax[0]
    for i, b in enumerate(benches):
        r = R[b]; a0 = ax[i]; vs = [v for v in ('V3', 'V4', MIX) if v in r['answer']]; x = np.arange(len(vs))
        st = [r['step'][v]['auroc'] for v in vs]; an = [r['answer'][v]['auroc'] for v in vs]
        a0.bar(x - .19, st, .36, color=[COL[v] for v in vs], edgecolor=INK2, linewidth=.4)
        a0.bar(x + .19, an, .36, color=[COL[v] for v in vs], alpha=.45, edgecolor=INK2, linewidth=.4)
        for xi, s_, a_ in zip(x, st, an):
            a0.text(xi - .19, s_ + .01, f'{s_:.3f}', ha='center', fontsize=8, color=INK)
            a0.text(xi + .19, a_ + .01, f'{a_:.3f}', ha='center', fontsize=8, color=INK)
        a0.axhline(.65, color=INK, lw=1.2, ls='--'); a0.axhline(.5, color=INK2, lw=.6)
        a0.set_xticks(x); a0.set_xticklabels(vs); a0.set_ylim(.4, 1.07); a0.set_ylabel('classifier AUROC, variant vs real wrong')
        a0.set_title(f'{b}: realism (dark = step, light = answer;\ndashed line = pass limit .65)', loc='left')
    a1 = ax[-1]; rows = [(b, v) for b in benches for v in ('V3', 'V4', MIX) if v in R[b]['pihat']]
    y = np.arange(len(rows))[::-1]
    for yi, (b, v) in zip(y, rows):
        p = R[b]['pihat'][v]; a1.scatter(p['pi_hat'], yi, s=40, color=COL[v], zorder=3)
        a1.scatter(p['pi_true'], yi, s=40, marker='|', color=INK, zorder=4)
        a1.plot([p['pi_true'] - .1, p['pi_true'] + .1], [yi, yi], color=GRID, lw=5, zorder=1)
        a1.text(p['pi_hat'] + .012, yi + .18, f"{p['pi_hat']:.3f}", fontsize=8, color=INK)
    a1.set_yticks(y); a1.set_yticklabels([f'{b} {v}' for b, v in rows]); a1.set_xlim(.4, 1.02)
    a1.set_xlabel('dot = pi-hat; black tick = true pi; grey band = true pi +- .10 (pass)')
    a1.set_title('pi-hat with each set as the known-wrong set', loc='left')
    fig.suptitle('Dev only: the V3 + V4 mixture as the known-wrong set', x=.01, ha='left', fontsize=11.5)
    fig.tight_layout(); fig.savefig(out / 'R1_mix_v34.png', dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', required=True); ap.add_argument('--gold', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=500); ap.add_argument('--draws', type=int, default=2000)
    ap.add_argument('--bench', default='prmbench,processbench')
    ap.add_argument('--test-force-stage-s', action='store_true', help='code test on simulated data only: run S / S2 without the pass')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    W = Path(a.W); split = json.load(open(W / 'inputs' / 'SPLIT.json'))
    gold = {g['uid']: g for g in json.load(open(a.gold))}
    lp = W / 'inputs' / 'prmbench_C_labels.json'; labels = json.load(open(lp)) if lp.exists() else None
    R = dict(created=time.strftime('%Y-%m-%d %H:%M'), W=str(W), side='dev only', mixture='union of all dev V3 and all dev V4 answers',
             rule='AUROC <= .65 and agreement >= .5', steps=a.steps, draws=a.draws)
    if a.test_force_stage_s: R['TEST_ONLY_stage_s_forced'] = True
    for bench in a.bench.split(','):
        C = D.load_pool(W, bench, 'C', split, gold, labels); Cd = C.subset(C.side == 'dev')
        rawC = D.readout(Cd); ev, sF, *_ = S.step_eval(Cd, rawC)
        AM = S.AnswerModel(Cd, rawC, sF); one = np.ones(16)
        FC = RA.feats(AM, Cd, rawC, one)[0]; nC = np.diff(Cd.off); edges = np.quantile(nC, [.2, .4, .6, .8])
        AM.fit_fusion(FC); sC = AM.score(FC); f1 = RA.halves(Cd.group, 'gatefold')
        pools = {}
        for V in ('V3', 'V4'):
            P = D.load_pool(W, bench, V, split); pools[V], nd = RA.aligned(P.subset(P.side == 'dev'), Cd); assert nd == 0, (bench, V, nd)
        pools[MIX] = concat([pools['V3'], pools['V4']], MIX)
        r = R[bench] = dict(dev=dict(answers=Cd.n, real_wrong_steps=int(Cd.wrong_step.sum()), real_wrong_answers=int(Cd.y_ans.sum())),
                            counts={V: P.n for V, P in pools.items()}, step={}, answer={}, answer_vs_own_parent={}, pihat={})
        r['mixture_share_V4'] = pools['V4'].n / pools[MIX].n
        for V, Pd in pools.items():
            r['step'][V] = RV.step_level(Cd, Pd, None)
            FV = RA.feats(AM, Pd, D.readout(Pd), one)[0]
            r['answer'][V] = RA.c2st_answers(FC, Cd.y_ans, Cd.group, nC, FV, Pd.group, np.diff(Pd.off), edges)
            pj = RA.parent_index(Pd, Cd)
            r['answer_vs_own_parent'][V] = dict(auroc=L.c2st(FV, FC[pj], Pd.group, Cd.group[pj]), parents_wrong_share=float(Cd.y_ans[pj].mean()))
            r['pihat'][V] = pihat(Cd, AM, sC, Pd, f1)
            log(bench, V, 'step', RA.rnd({k: r['step'][V][k] for k in ('auroc', 'agreement', 'n_each', 'pass_', 'loudness_median')}),
                '| answer', RA.rnd(r['answer'][V]), '| vs parent', RA.rnd(r['answer_vs_own_parent'][V]['auroc']),
                '| pi-hat', RA.rnd(r['pihat'][V]['pi_hat']), 'true', RA.rnd(r['pihat'][V]['pi_true']))
        r['code_check'] = {f'{lv} {V}': dict(now=r[lv][V]['auroc'], expected=e, pass_=bool(abs(r[lv][V]['auroc'] - e) < .0015))
                           for (b, lv, V), e in EXPECT.items() if b == bench}
        if bench == 'processbench':                  # PIHAT_processbench_V3.json (pihat_dev.py, commit aeff3ac)
            r['code_check']['pi-hat V3'] = dict(now=r['pihat']['V3']['pi_hat'], expected=.889,
                                                pass_=bool(abs(r['pihat']['V3']['pi_hat'] - .889) < .0015))
        log(bench, 'code check', RA.rnd(r['code_check']))
        r['verdict_mixture'] = dict(step=r['step'][MIX]['pass_'], answer=r['answer'][MIX]['pass_'], pi_hat=r['pihat'][MIX]['pi_check_pass_10pts'])
        if r['step'][MIX]['pass_'] or a.test_force_stage_s:
            r['stages_S_S2'] = stage_s_s2(Cd, pools[MIX], a.steps, a.draws)
        else:
            r['stages_S_S2'] = dict(run=False, reason='the mixture fails the step-level realism rule: stage S and S2 do not run (pre-registered)')
        log(bench, 'verdict', r['verdict_mixture'], 'S/S2 run', r['stages_S_S2'].get('run', True))
        json.dump(RA.rnd(R), open(out / 'MIX_V34.json', 'w'), indent=1, default=str)
    R['seconds'] = round(time.time() - t0, 1)
    json.dump(RA.rnd(R), open(out / 'MIX_V34.json', 'w'), indent=1, default=str)
    figure(R, out)
    (out / 'mix_v34.log').write_text('\n'.join(LOG)); log('done', R['seconds'], 's')


if __name__ == '__main__':
    main()
