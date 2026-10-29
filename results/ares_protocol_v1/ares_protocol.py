"""ARES (arXiv 2507.12948) READOUT protocol applied to our stored step scores, as frozen in PROTOCOL.json.

Not a rerun of ARES's entailment method. One global threshold per (cell, arm), chosen by 5-fold CV on labels to
maximize Macro-F1 (folds by source group, seed 20261021, 99-quantile grid), out-of-fold decisions pooled.
PRMBench Macro-F1 comes from competitors.ares_rows unchanged; an own loop with the same folds/grid gives Macro-P/R and
the paired bootstrap (and must reproduce ares_rows exactly). DeltaBench (447-answer common set) uses the same loop.

    python ares_protocol.py      # -> ARES_RESULTS.csv, BOOTSTRAP.csv, PARITY.json   (exit 3 on a failed parity gate)
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import json, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'prm_sub5_all_benchmarks_v1'))
import sub5_all as A  # noqa: E402
E, CMP, CAB = A.E, A.CMP, A.CAB
CC = CMP.CC
sys.path.insert(0, str(A.ROOT / 'results/prm_boost_math_rigor_v1'))
import stage8_auto_forced_k as S8  # noqa: E402

PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
FOLD_SEED, BOOT_SEED, B = 20261021, 20261029, 1000
S8_REF = {(r['bench'], r['prm']): r['A']['auto'] for r in
          json.loads((A.ROOT / 'results/prm_boost_math_rigor_v1/stage8/AUTO_FORCEDK_RESULTS.json').read_text())}
AUTO_NEW = {(r['benchmark'], r['prm'], r['arm'], r['metric']): float(r['value']) for r in E.read_csv(A.HERE / 'AUTO_NEW.csv')}
SUB5 = {(r['benchmark'], r['mode'], r['prm'], r['arm'], r['metric']): float(r['value']) for r in E.read_csv(A.HERE / 'SUB5_ALL.csv')}
COMP = {(r['prm'], r['arm']): float(r['macro_f1_oof']) for r in E.read_csv(A.ROOT / 'results/prm_boost_collection_v1/tables/COMP_ARES.csv')
        if r['prm'] != 'paper'}
ROWS, PAR, BOOT = [], [], []
norm = lambda s: re.sub(r'\s+', ' ', str(s or '')).strip().lower()


def gate(name, got, ref, tol=1e-6):
    d = abs(float(got) - float(ref)); PAR.append(dict(check=name, got=float(got), ref=float(ref), abs_diff=d, ok=bool(d <= tol)))
    print(('PASS ' if d <= tol else 'FAIL ') + name, round(float(got), 6), round(float(ref), 6), flush=True)


def macro(C, scale):
    """Two-class Macro-F1 / P / R from pooled [tp, fp, tn, fn] (positive = valid step). F1 matches official_from_counts."""
    tp, fp, tn, fn = np.asarray(C, float)
    q = lambda a, b: a / b if b > 0 else 0.
    f1 = .5 * (q(2 * tp, 2 * tp + fp + fn) + q(2 * tn, 2 * tn + fp + fn))
    p = .5 * (q(tp, tp + fp) + q(tn, tn + fn)); r = .5 * (q(tp, tp + fn) + q(tn, tn + fp))
    return scale * f1, scale * p, scale * r


def fold_of_answers(c, groups):
    """Exactly ares_rows' assignment: permute the unique groups of non-control answers, round-robin into 5 folds."""
    ans_ok = c.cls != 'correct' if c.bench == 'prmbench' else np.ones(len(c.ids), bool)
    ug = np.unique(groups[ans_ok]); rng = np.random.default_rng(FOLD_SEED)
    fold_of = dict(zip(rng.permutation(ug), np.arange(len(ug)) % 5))
    return np.array([fold_of.get(g, -1) for g in groups])


def ares_oof(c, arm, groups):
    """Same loop as ares_rows, on per-answer counts (E.counts_by_answer; includes empty steps on external sets).
    Returns per-answer out-of-fold counts and the 5 per-fold Macro-F1 values."""
    v = CC.thr_score(c, arm); af = fold_of_answers(c, groups); sf = np.repeat(af, np.diff(c.off)); bench = c.bench
    Coof = np.zeros((len(c.ids), 4)); fs = []
    for k in range(5):
        trm = c.keep & (sf != k)
        ts = np.quantile(v[trm], np.linspace(0.01, 0.99, 99))
        Ctr = lambda t: E.counts_by_answer(c, v >= t)[af != k].sum(0)
        best = max(ts, key=lambda t: E.official_from_counts(Ctr(t), bench))
        Cb = E.counts_by_answer(c, v >= best); Coof[af == k] = Cb[af == k]
        fs.append(float(E.official_from_counts(Cb[af == k].sum(0), bench)))
    return Coof, fs


def boot(Ca, Cb, groups, scale):
    ug, gi = np.unique(groups, return_inverse=True)
    Ga, Gb = np.zeros((len(ug), 4)), np.zeros((len(ug), 4)); np.add.at(Ga, gi, Ca); np.add.at(Gb, gi, Cb)
    pt = macro(Ga.sum(0), scale)[0] - macro(Gb.sum(0), scale)[0]
    rng = np.random.default_rng(BOOT_SEED); d = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        d.append(macro(w @ Ga, scale)[0] - macro(w @ Gb, scale)[0])
    d = np.asarray(d)
    return dict(delta=float(pt), ci_lo=float(np.quantile(d, .025)), ci_hi=float(np.quantile(d, .975)),
                p_le_0=float(np.mean(d <= 0)), n_groups=int(len(ug)))


def add_auto(c, zarm):
    s, info = S8.auto(E.channels(c, 'N16'), E.answer_z(c.arms[zarm], c.off)); c.arms['AUTO_N16'] = s
    return info


def emit(c, bench, prm, tele, label, arm, groups, gap_arm, ares_f1=None, fold_sd=None):
    scale = 100.
    Coof, fs = ares_oof(c, arm, groups)
    f1, p, r = macro(Coof.sum(0), scale)
    if ares_f1 is not None:
        gate(f'{bench}/{prm}/{tele}/{label}: own loop == ares_rows', f1, ares_f1, 1e-9)
    gf = c.flags(gap_arm, 'GAP'); Cg = E.counts_by_answer(c, gf)
    g1, gp, gr = macro(Cg.sum(0), scale)
    gap_main = E.official(c, gf)[0] if bench == 'prmbench' else g1
    ROWS.append(dict(prm=prm, telemetry=tele, arm=label, score=arm, benchmark=bench, macro_f1_oof=f1,
                     fold_sd=float(np.std(fs, ddof=1)) * (1. if bench == 'prmbench' else 100.) if fold_sd is None else fold_sd, macro_precision_oof=p, macro_recall_oof=r,
                     gap_arm=gap_arm, gap_main=gap_main, gap_macro_f1_counts=g1, gap_macro_precision=gp, gap_macro_recall=gr,
                     n_answers=len(c.ids), n_steps=int(c.keep.sum()), n_groups=int(len(np.unique(groups)))))
    return Coof, gap_main


def prmbench(prm):
    scale_arm = 'PRM_Q80_PROB' if A.META[f'prmbench/{prm}']['prm_reference_scale'] == '1-reward' else 'PRM_Q80'
    c = E.load('prmbench', prm); info = add_auto(c, 'PRM_Q80')
    arms = [('PRM', 'PRM_native', scale_arm), ('AUTO', 'AUTO_N16', 'AUTO_N16'), ('CHAIN', 'CHAIN_N16', 'CHAIN_N16'),
            ('TELE16 backbone', 'TELE_N16', 'TELE_N16')]
    ref = {r['arm']: r for r in CMP.ares_rows(c, [a for _, a, _ in arms])}
    C = {}
    for label, arm, garm in arms:
        if (prm, arm) in COMP:
            gate(f'prmbench/{prm}/{arm}: ares_rows vs COMP_ARES.csv', ref[arm]['macro_f1_oof'], COMP[(prm, arm)], 1e-9)
        C[label], gm = emit(c, 'prmbench', prm, 'backbone', label, arm, c.groups, garm, ref[arm]['macro_f1_oof'], ref[arm]['macro_f1_fold_sd'])
        sref = {'PRM': ('PRM', 'PRMScore_GAP'), 'CHAIN': ('CHAIN_N16', 'PRMScore_GAP'), 'TELE16 backbone': ('TELE_N16', 'PRMScore_GAP')}
        if label == 'AUTO':
            gate(f'prmbench/{prm}: AUTO GAP PRMScore vs stage8', gm, S8_REF[('prmbench', prm)]['gap_prmscore'])
        else:
            gate(f'prmbench/{prm}/backbone/{label}: GAP vs SUB5_ALL', gm, SUB5[('prmbench', 'backbone', prm) + sref[label]])
    BOOT.append(dict(benchmark='prmbench', prm=prm, contrast='AUTO - PRM (ARES readout, OOF Macro-F1)', auto_K=info['K'],
                     **boot(C['AUTO'], C['PRM'], c.groups, 100.)))
    del c
    s = E.load('prmbench', prm + '__standalone')
    ref = {r['arm']: r for r in CMP.ares_rows(s, ['TELE_N16', 'MEAN_N16'])}
    for label, arm in (('TELE16 standalone (L-SML)', 'TELE_N16'), ('TELE16 standalone mean', 'MEAN_N16')):
        _, gm = emit(s, 'prmbench', prm, 'standalone', label, arm, s.groups, arm, ref[arm]['macro_f1_oof'], ref[arm]['macro_f1_fold_sd'])
        gate(f'prmbench/{prm}/standalone/{arm}: GAP vs SUB5_ALL', gm, SUB5[('prmbench', 'standalone', prm, arm, 'PRMScore_GAP')])
    del s


def deltabench(prm, qtext):
    key = 'cab|prm|deltabench'; A.REG[key] = dict(mode='prm', bench='deltabench', common=A.common_uids('deltabench'))
    c = E.load(key, prm); info = add_auto(c, 'PRM')
    groups = np.array([qtext.get(u, u) for u in c.ids])
    arms = [('PRM', 'PRM', A.prm_scale_arm('deltabench', prm)), ('AUTO', 'AUTO_N16', 'AUTO_N16'),
            ('CHAIN', 'CHAIN_N16', 'CHAIN_N16'), ('TELE16 backbone', 'TELE_N16', 'TELE_N16')]
    C = {}
    for label, arm, garm in arms:
        C[label], gm = emit(c, 'deltabench', prm, 'backbone', label, arm, groups, garm)
        cab = CAB.row(c, 'prm', 'deltabench', 'common', garm, 'GAP')['macro_f1']
        gate(f'deltabench/{prm}/{label}: own GAP == evaluate_cab.row', gm, cab, 1e-9)
        if label == 'AUTO':
            gate(f'deltabench/{prm}: AUTO GAP vs AUTO_NEW', gm, AUTO_NEW[('deltabench', prm, 'AUTO_N16', 'macro_f1_GAP')])
        else:
            gate(f'deltabench/{prm}/{label}: GAP vs SUB5_ALL', gm,
                 SUB5[('deltabench', 'backbone', prm, {'PRM': 'PRM'}.get(label, arm), 'macro_f1_GAP')])
    BOOT.append(dict(benchmark='deltabench', prm=prm, contrast='AUTO - PRM (ARES readout, OOF Macro-F1)', auto_K=info['K'],
                     **boot(C['AUTO'], C['PRM'], groups, 100.)))


def main():
    for prm in PRMS:
        prmbench(prm); print(prm, 'prmbench done', flush=True)
    ans = json.load(open(A.DATA / 'crop_ares_v1/inputs_text/deltabench/answers.json', encoding='utf8'))
    qtext = {a['uid']: 'q:' + norm(a['question']) for a in ans}
    for prm in PRMS:
        deltabench(prm, qtext); print(prm, 'deltabench done', flush=True)
    fails = [p['check'] for p in PAR if not p['ok']]
    json.dump(dict(n_checks=len(PAR), failed=fails, max_abs_diff=max(p['abs_diff'] for p in PAR), checks=PAR),
              open(HERE / 'PARITY.json', 'w'), indent=1)
    print('parity', len(PAR), 'failed', fails)
    if fails: raise SystemExit(3)
    E.write_csv(HERE / 'ARES_RESULTS.csv', ROWS); E.write_csv(HERE / 'BOOTSTRAP.csv', BOOT)
    print('rows', len(ROWS))


if __name__ == '__main__':
    main()
