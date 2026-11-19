"""Answer-level gate check on DEV (2026-11-19, Omri approved in the thread: "יש אישור"). CPU only, no new inference.

The gate is the run_analysis.py stage-G gate, unchanged, with M = 1 (stage A did not run):
  answer features   RA.feats: 16 channels (per-answer orientation, pool ECDF, max over the answer's steps) + Simes of s_g
  answer score      S.AnswerModel.fit_fusion / score: z per feature, answer L-SML (label-free, fitted on the dev C pool)
  threshold (SSL)   pi-hat = BBE(score of the synthetic wrong answers, score of the unlabelled pool), 2-fold cross-fit by
                    question group (RA.halves 'gatefold'); open the pi-hat share of answers with the highest score
  marking           RA.marking: GAP flags in the open answers (+ ProcessBench change point with prior 1 - P)
Positives: V3, V4, V3+V4 (dev), each with ALL parents and with CLEAN parents only (gold y_ans of the parent = 0;
gold used only to choose which synthetic rows are positives and for scoring, as a diagnostic; this is NOT label-free).
Extra report arm: the same gate with the TRUE share pi (separates the threshold error from the ranking quality).
    python gate_dev.py <code dir (commit 4d57e57)> <W> <processbench_gold.json> <out dir> [draws]
"""
import json, sys, time
from pathlib import Path
import numpy as np

CODE, W, GOLD, OUT = sys.argv[1:5]
DRAWS = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
W = Path(W); OUT = Path(OUT); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, CODE)
import ssl_data as D, ssl_lib as L, ssl_stages as S, run_analysis as RA, mix_v34 as MX

t0 = time.time()
LOG = open(OUT / 'gate_dev.log', 'w', encoding='utf-8')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def r4(x):
    return RA.rnd(x)


split = json.load(open(W / 'inputs' / 'SPLIT.json'))
labels = json.load(open(W / 'inputs' / 'prmbench_C_labels.json'))
gold = {g['uid']: g for g in json.load(open(GOLD))}
R = dict(created=time.strftime('%Y-%m-%d %H:%M'), side='dev only', code_commit='4d57e57', draws=DRAWS, M=1)
one = np.ones(16)

for bench in ('processbench', 'prmbench'):
    C = D.load_pool(W, bench, 'C', split, gold, labels); Cd = C.subset(C.side == 'dev')
    rawC = D.readout(Cd); bd = S.Boot(Cd.group, DRAWS); gid = bd.gi
    ev0, s0, f0, X0 = S.step_eval(Cd, rawC, bd, gid)
    AM = S.AnswerModel(Cd, rawC, s0)
    FC, SsC, simC, sgC = RA.feats(AM, Cd, rawC, one); AM.fit_fusion(FC); sC = AM.score(FC)
    f1 = RA.halves(Cd.group, 'gatefold')
    key = 'F1' if bench == 'processbench' else 'PRMScore'
    pv = {}
    for V in ('V3', 'V4'):
        P = D.load_pool(W, bench, V, split); pv[V], nd = RA.aligned(P.subset(P.side == 'dev'), Cd); assert nd == 0, (bench, V, nd)
    pv['V3+V4'] = MX.concat([pv['V3'], pv['V4']], 'V3+V4')
    for V in ('V3', 'V4', 'V3+V4'):
        cm = ~Cd.y_ans[RA.parent_index(pv[V], Cd)]
        pv[V + ' clean parents'] = pv[V].subset(cm) if V != 'V3+V4' else MX.concat([pv['V3'].subset(~Cd.y_ans[RA.parent_index(pv['V3'], Cd)]),
                                                                                   pv['V4'].subset(~Cd.y_ans[RA.parent_index(pv['V4'], Cd)])], 'V3+V4c')
    pi_true = float(Cd.y_ans.mean())
    B = R[bench] = dict(answers=int(Cd.n), wrong_answers=int(Cd.y_ans.sum()), pi_true=pi_true,
                        answer_score_auroc=L.auroc(Cd.y_ans, sC), positives={})
    base = None
    for name, Pd in pv.items():
        # pi-hat and P_dev exactly as mix_v34.pihat (re-implemented here to keep P_dev for the change point)
        sP = AM.score(RA.feats(AM, Pd, D.readout(Pd), one)[0])
        fpos = np.isin(Pd.group, Cd.group[f1]); folds, P_dev = [], np.zeros(Cd.n)
        for fold in (True, False):
            o_pos, o_C = fpos != fold, f1 != fold
            pi_f, info = L.bbe(sP[o_pos], sC[o_C]); folds.append(dict(pi_hat=pi_f, n_pos=int(o_pos.sum()), n_C=int(o_C.sum())))
            P_dev[o_C] = L.pu_posterior(sP[o_pos], sC[o_C], pi_f, sC[o_C])
        ph = float(np.mean([f['pi_hat'] for f in folds]))
        if name in ('V3', 'V4', 'V3+V4'):
            ref = MX.pihat(Cd, AM, sC, Pd, f1)['pi_hat']; assert abs(ref - ph) < 1e-9, (name, ref, ph)
        op = RA.gate_open(sC, ph)
        mk = RA.marking(Cd, s0, sC, ph, P_dev, sgC, bd.W, gid, s_top10=s0)
        if base is None:
            base = mk
            optrue = RA.gate_open(sC, pi_true)
            tg = RA.score_flags(Cd, f0 & L.expand(optrue, Cd.off), bd.W, gid)
            tg['vs_method_of_record'] = {key: RA.diff_ci(tg, RA.score_flags(Cd, f0, bd.W, gid), key)}
            B['no_gate'] = RA.strip(RA.score_flags(Cd, f0))
            B['oracle_gate'] = mk['final + label-perfect gate (oracle)']
            B['gate_true_pi_ranked_by_score'] = RA.strip(tg) | {'vs_method_of_record': tg['vs_method_of_record'],
                                                                 'open_true_wrong_share': float(Cd.y_ans[optrue].mean()),
                                                                 'wrong_opened': float(optrue[Cd.y_ans].mean()),
                                                                 'clean_closed': float((~optrue)[~Cd.y_ans].mean())}
        g = mk['final + gate (pi-hat)']
        row = dict(n_positive=int(Pd.n), pi_hat=ph, abs_err=abs(ph - pi_true), folds=folds,
                   positive_vs_pool_auroc=L.auroc(np.r_[np.ones(Pd.n), np.zeros(Cd.n)].astype(bool), np.r_[sP, sC]),
                   open_share=float(op.mean()), wrong_opened=float(op[Cd.y_ans].mean()), clean_closed=float((~op)[~Cd.y_ans].mean()),
                   P_auroc=L.auroc(Cd.y_ans, P_dev), gate=g)
        if bench == 'processbench':
            row['change_point'] = mk['final + change point (prior 1 - P)']
        B['positives'][name] = row
        msg = (f"{bench} {name} n={Pd.n} pi-hat {ph:.3f} (true {pi_true:.3f}) open {row['open_share']:.3f} "
               f"wrong-opened {row['wrong_opened']:.3f} clean-closed {row['clean_closed']:.3f} | gate {key} {g[key]:.2f} "
               f"delta {g['vs_method_of_record'][key]['delta']:+.2f} {[round(x, 2) for x in g['vs_method_of_record'][key]['ci']]}")
        if bench == 'processbench':
            cp = row['change_point']
            msg += f" SLA {g['SLA']:.2f} | change point F1 {cp['F1']:.2f} SLA {cp['SLA']:.2f}"
        log(msg)
    nb = B['no_gate']; ob = B['oracle_gate']; tb = B['gate_true_pi_ranked_by_score']
    log(bench, 'answers', Cd.n, 'pi true', round(pi_true, 3), 'answer-score AUROC', round(B['answer_score_auroc'], 4),
        '| no gate', key, round(nb[key], 2), '| oracle gate', round(ob[key], 2),
        '| true-pi gate', round(tb[key], 2), 'wrong-opened', round(tb['wrong_opened'], 3), 'clean-closed', round(tb['clean_closed'], 3),
        *(['| SLA no gate', round(nb['SLA'], 2), 'oracle', round(ob['SLA'], 2), 'true-pi', round(tb['SLA'], 2)] if bench == 'processbench' else []))

R['seconds'] = round(time.time() - t0, 1)
(OUT / 'GATE_DEV.json').write_text(json.dumps(r4(R), indent=1, default=str))
log('done', R['seconds'], 's')
