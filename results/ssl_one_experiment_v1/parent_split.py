"""Dev-only diagnostic: split the synthetic answers by the TRUE label of their parent answer (Omri approved in the thread,
2026-11-19 10:26 Israel, "יש אישור", for the check proposed at 10:0x: "split the synthetic answers by the true label of the
parent, as a dev-only diagnostic, and compare their score to real wrong answers"). CPU, seconds. No test data is read.

Why: the synthetic answers were built from ALL dev answers as parents, without labels. On ProcessBench 65% of the parents
are already wrong, on PRMBench 86%. BBE (stage G) needs the known-wrong set to have the same answer-score distribution as
the real wrong answers. This script shows which part of the synthetic set breaks that: the injection into a CLEAN parent
(too quiet?) or the second error in an already WRONG parent (too loud?).

For each benchmark and each set (V3, V4, V3+V4), the synthetic answers are split by the parent label:
  all / clean_parent / wrong_parent.
For each part (answer score sA = the answer L-SML score with M = 1, the same model as pihat_dev.py and mix_v34.py):
  - n, share of the set;
  - sA median; AUROC of sA, part vs real wrong answers (.5 = same distribution; > .5 = louder) and vs real clean answers;
  - shift vs its own parent: median of sA(variant) - sA(parent), share with a shift > 0;
  - answer-level realism C2ST on the 17 features vs real wrong answers (RA.c2st_answers, report only);
  - pi-hat by BBE with this part as the known-wrong set (the pihat_dev.py 2-fold computation), and the gate it gives:
    share opened, TPR (real wrong answers opened), TNR (real clean answers closed).
Reference row (labels, diagnostic only): pi-hat when the known-wrong set is a PERFECT synthetic set, i.e. the real wrong
answers of the OTHER gate fold (an independent sample of the real wrong distribution). It shows the pi-hat error of BBE
itself on this score, without any synthetic mismatch.
Code check: the 'all' rows must give the pi-hat values of PIHAT_processbench_V3.json and MIX_V34.json.
Gold labels: the split, the reference row and all reports. Never inside a fit (the answer model is fitted label-free).

    python parent_split.py --W <dir with steps/ inputs/> --gold <processbench_gold.json> --out <OUT>
"""
import argparse, json, time
from pathlib import Path

import numpy as np

import mix_v34 as MX
import run_analysis as RA
import ssl_data as D
import ssl_lib as L
import ssl_stages as S

EXPECT = {('processbench', 'V3'): .889, ('processbench', 'V4'): .5258, ('processbench', 'V3+V4'): .8129,
          ('prmbench', 'V3'): .9325, ('prmbench', 'V4'): .7927, ('prmbench', 'V3+V4'): .8634}
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a); LOG.append(s); print(s, flush=True)


def gate_rates(sC, W, share):
    op = RA.gate_open(sC, share)
    return dict(open_share=float(op.mean()), tpr_wrong_opened=float(op[W].mean()), tnr_clean_closed=float((~op[~W]).mean()))


def pihat_part(Cd, sC, sP, gP, f1):
    """pihat_dev.py: per gate fold, BBE(positives of that fold, C answers of that fold); pi-hat = mean of the 2 folds."""
    fpos = np.isin(gP, Cd.group[f1]); folds = []
    for fold in (True, False):
        o_pos, o_C = fpos != fold, f1 != fold
        if o_pos.sum() < 20:
            folds.append(dict(pi_hat=None, n_pos=int(o_pos.sum()), reason='fewer than 20 positives')); continue
        pi_f, info = L.bbe(sP[o_pos], sC[o_C]); folds.append(dict(pi_hat=pi_f, n_pos=int(o_pos.sum()), n_C=int(o_C.sum()), **info))
    ok = [f['pi_hat'] for f in folds if f['pi_hat'] is not None]
    ph = float(np.mean(ok)) if ok else None
    return ph, folds


def pihat_perfect(Cd, sC, f1):
    """Reference: known-wrong set = the real wrong answers of the OTHER gate fold; unlabeled = the C answers of this fold."""
    W = Cd.y_ans.astype(bool); folds = []
    for fold in (True, False):
        o_C = f1 != fold; o_pos = (f1 == fold) & W
        pi_f, info = L.bbe(sC[o_pos], sC[o_C]); folds.append(dict(pi_hat=pi_f, n_pos=int(o_pos.sum()), n_C=int(o_C.sum()), **info))
    return float(np.mean([f['pi_hat'] for f in folds])), folds


def part_row(Cd, FC, sC, nC, edges, f1, Pd, FV, sP, pj, m):
    W = Cd.y_ans.astype(bool); pt = float(W.mean())
    row = dict(n=int(m.sum()), share_of_set=float(m.mean()))
    if m.sum() < 20:
        row['reason'] = 'fewer than 20 answers'; return row
    s = sP[m]; d = s - sC[pj[m]]
    lab = lambda a, b: (np.r_[np.ones(len(a)), np.zeros(len(b))].astype(bool), np.r_[a, b])
    row.update(score_median=float(np.median(s)),
               auroc_vs_real_wrong=L.auroc(*lab(s, sC[W])), auroc_vs_real_clean=L.auroc(*lab(s, sC[~W])),
               shift_vs_parent_median=float(np.median(d)), shift_vs_parent_q25_q75=[float(np.quantile(d, .25)), float(np.quantile(d, .75))],
               shift_gt0_share=float((d > 1e-9).mean()))
    row['answer_c2st_vs_real_wrong'] = RA.c2st_answers(FC, Cd.y_ans, Cd.group, nC, FV[m], Pd.group[m], np.diff(Pd.off)[m], edges)
    ph, folds = pihat_part(Cd, sC, s, Pd.group[m], f1)
    row.update(pi_hat=ph, pi_true=pt, abs_err=None if ph is None else abs(ph - pt),
               pi_check_pass_10pts=None if ph is None else bool(abs(ph - pt) <= .10), folds=folds)
    if ph is not None:
        row['gate'] = gate_rates(sC, W, ph)
    return row


def figure(R, curves, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID = '#fcfcfb', '#1f1e1c', '#52514e', '#e4e3df'
    COL = dict(real_wrong='#2a78d6', real_clean='#eb6834', clean_parent='#1baf7a', wrong_parent='#eda100')
    LAB = dict(real_wrong='real wrong answers', real_clean='real clean answers',
               clean_parent='synthetic, clean parent', wrong_parent='synthetic, wrong parent')
    LS = dict(real_wrong='-', real_clean='-', clean_parent='--', wrong_parent='--')
    plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': GRID, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
                         'legend.frameon': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6})
    benches = [b for b in ('processbench', 'prmbench') if b in R]
    fig, ax = plt.subplots(len(benches), 3, figsize=(16, 4.6 * len(benches)), squeeze=False)
    for i, b in enumerate(benches):
        for j, V in enumerate(('V3', 'V4')):
            a0 = ax[i, j]
            for key in ('real_clean', 'real_wrong', 'clean_parent', 'wrong_parent'):
                x = np.sort(curves[b][V][key]); y = np.arange(1, len(x) + 1) / len(x)
                a0.plot(x, y, color=COL[key], lw=2, ls=LS[key], label=f'{LAB[key]} (n={len(x)})')
            a0.set_xlabel('answer score sA (higher = more likely wrong)'); a0.set_ylabel('share of answers with score <= x')
            a0.set_title(f'{b} dev, {V}: answer-score distributions', loc='left'); a0.legend(fontsize=8, loc='upper left')
        a1 = ax[i, 2]; rows = []
        for V in ('V3', 'V4', 'V3+V4'):
            for part in ('all', 'clean_parent', 'wrong_parent'):
                r = R[b]['sets'][V][part]
                if r.get('pi_hat') is not None: rows.append((f'{V} {part.replace("_", " ")}', r['pi_hat'], part))
        rows.append(('perfect synthetic (real wrong)', R[b]['reference_perfect_known_wrong']['pi_hat'], 'real_wrong'))
        pt = R[b]['dev']['pi_true']; y = np.arange(len(rows))[::-1]
        a1.axvspan(pt - .1, pt + .1, color=GRID, zorder=0); a1.axvline(pt, color=INK, lw=1.2)
        PC = dict(all=INK2, clean_parent=COL['clean_parent'], wrong_parent=COL['wrong_parent'], real_wrong=COL['real_wrong'])
        for yi, (name, v, part) in zip(y, rows):
            a1.scatter(v, yi, s=44, color=PC[part], zorder=3, edgecolor=SURF, linewidth=1.5)
            a1.text(v + .012, yi, f'{v:.3f}', va='center', fontsize=8, color=INK)
        a1.set_yticks(y); a1.set_yticklabels([r[0] for r in rows], fontsize=8); a1.set_xlim(.3, 1.08)
        a1.set_xlabel(f'pi-hat (black line = true pi {pt:.3f};\ngrey band = pass, true pi +- .10)')
        a1.set_title(f'{b} dev:\npi-hat by known-wrong set', loc='left')
    fig.suptitle('Dev only: synthetic answers split by the true label of their parent', x=.01, ha='left', fontsize=11.5)
    fig.tight_layout(); fig.savefig(out / 'R1_parent_split.png', dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', required=True); ap.add_argument('--gold', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--bench', default='processbench,prmbench')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    W_ = Path(a.W); split = json.load(open(W_ / 'inputs' / 'SPLIT.json'))
    gold = {g['uid']: g for g in json.load(open(a.gold))}
    lp = W_ / 'inputs' / 'prmbench_C_labels.json'; labels = json.load(open(lp)) if lp.exists() else None
    R = dict(created=time.strftime('%Y-%m-%d %H:%M'), W=str(W_), side='dev only', score='answer L-SML score sA, M = 1',
             labels_use='parent split, perfect-synthetic reference, reports; never inside a fit')
    curves = {}
    for bench in a.bench.split(','):
        C = D.load_pool(W_, bench, 'C', split, gold, labels); Cd = C.subset(C.side == 'dev')
        rawC = D.readout(Cd); ev, sF, *_ = S.step_eval(Cd, rawC)
        AM = S.AnswerModel(Cd, rawC, sF); one = np.ones(16)
        FC = RA.feats(AM, Cd, rawC, one)[0]; nC = np.diff(Cd.off); edges = np.quantile(nC, [.2, .4, .6, .8])
        AM.fit_fusion(FC); sC = AM.score(FC); f1 = RA.halves(Cd.group, 'gatefold'); Wm = Cd.y_ans.astype(bool)
        pools = {}
        for V in ('V3', 'V4'):
            P = D.load_pool(W_, bench, V, split); pools[V], nd = RA.aligned(P.subset(P.side == 'dev'), Cd); assert nd == 0, (bench, V, nd)
        pools['V3+V4'] = MX.concat([pools['V3'], pools['V4']], 'V3+V4')
        ph_ref, folds_ref = pihat_perfect(Cd, sC, f1)
        r = R[bench] = dict(dev=dict(answers=Cd.n, wrong=int(Wm.sum()), clean=int((~Wm).sum()), pi_true=float(Wm.mean()),
                                     answer_auroc_wrong_vs_clean=L.auroc(Wm, sC), score_median_real_wrong=float(np.median(sC[Wm])),
                                     score_median_real_clean=float(np.median(sC[~Wm]))),
                            reference_perfect_known_wrong=dict(pi_hat=ph_ref, abs_err=abs(ph_ref - float(Wm.mean())), folds=folds_ref,
                                                               gate=gate_rates(sC, Wm, ph_ref)),
                            reference_true_pi_gate=gate_rates(sC, Wm, float(Wm.mean())), sets={})
        curves[bench] = {}
        for V, Pd in pools.items():
            pj = RA.parent_index(Pd, Cd); pw = Wm[pj]
            FV = RA.feats(AM, Pd, D.readout(Pd), one)[0]; sP = AM.score(FV)
            r['sets'][V] = {part: part_row(Cd, FC, sC, nC, edges, f1, Pd, FV, sP, pj, m)
                            for part, m in (('all', np.ones(Pd.n, bool)), ('clean_parent', ~pw), ('wrong_parent', pw))}
            curves[bench][V] = dict(real_wrong=sC[Wm], real_clean=sC[~Wm], clean_parent=sP[~pw], wrong_parent=sP[pw])
            for part, x in r['sets'][V].items():
                log(bench, V, part, 'n', x['n'], 'share', RA.rnd(x['share_of_set']), '| vs wrong', RA.rnd(x.get('auroc_vs_real_wrong')),
                    'vs clean', RA.rnd(x.get('auroc_vs_real_clean')), '| shift med', RA.rnd(x.get('shift_vs_parent_median')),
                    '| c2st', RA.rnd(x.get('answer_c2st_vs_real_wrong', {}).get('auroc')), '| pi-hat', RA.rnd(x.get('pi_hat')),
                    'true', RA.rnd(x.get('pi_true')), '| gate', RA.rnd(x.get('gate')))
        log(bench, 'perfect synthetic pi-hat', RA.rnd(ph_ref), 'true', RA.rnd(r['dev']['pi_true']), 'gate', RA.rnd(r['reference_perfect_known_wrong']['gate']))
        r['code_check'] = {V: dict(now=r['sets'][V]['all']['pi_hat'], expected=e, pass_=bool(abs(r['sets'][V]['all']['pi_hat'] - e) < .0015))
                           for (b, V), e in EXPECT.items() if b == bench}
        log(bench, 'code check', RA.rnd(r['code_check']))
        json.dump(RA.rnd(R), open(out / 'PARENT_SPLIT.json', 'w'), indent=1, default=str)
    R['seconds'] = round(time.time() - t0, 1)
    json.dump(RA.rnd(R), open(out / 'PARENT_SPLIT.json', 'w'), indent=1, default=str)
    figure(R, curves, out)
    (out / 'parent_split.log').write_text('\n'.join(LOG)); log('done', R['seconds'], 's')


if __name__ == '__main__':
    main()
