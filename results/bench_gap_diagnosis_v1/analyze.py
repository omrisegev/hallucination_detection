"""Why do C4U and TELE16 behave differently across benchmarks? Diagnostics on the extracted cells (2026-11-11).

Reads cells/<bench>__<prm>.npz (extract.py) and writes:
  T1_label_structure.csv   label protocol / length / clean share / cascade / position per benchmark
  T2_arm_quality.csv       PRM, TELE16, C4U, position: pooled AUROC, within-AUC, position-stratified first-error AUC,
                           answer-level AUC, per cell; plus gaps vs PRM
  T3_channels.csv          per channel x cell: pooled AUROC, within-AUC, stratified first-error AUC, class-conditional
                           Spearman with the PRM (within correct steps / within error steps), LR increment over PRM
  T4_lr_headroom.csv       answer-grouped 5-fold logistic regression: PRM, 16 channels, PRM+16, PRM+16+position
  T5_attribution.csv       Spearman across the 32 cells: gap vs each candidate factor
All metrics on the evaluation steps (keep mask). Labels are used only for evaluation and the LR headroom reference.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
CELLS = HERE / 'cells'
BENCHES = ['prmbench', 'processbench', 'socratic', 'hard2verify', 'crv_arith', 'crv_gsm8k', 'prm800k', 'deltabench_ext']
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
N16 = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level',
       'energy_innovation', 'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0',
       'realized_z', 'realized_drv', 'digit_alternative', 'digit_spread', 'digit_alternative_innovation']
POSCAP = 12


def auc(y, s):
    y = np.asarray(y, bool)
    if y.all() or not y.any(): return np.nan
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within(err, s, off, keep):
    out = []
    for a, b in zip(off[:-1], off[1:]):
        m = keep[a:b]
        out.append(auc(err[a:b][m], s[a:b][m]))
    return np.array(out)


def step_index(off):
    return np.concatenate([np.arange(b - a) for a, b in zip(off[:-1], off[1:])])


def rel_pos(off):
    return np.concatenate([np.arange(b - a) / max(b - a - 1, 1) for a, b in zip(off[:-1], off[1:])])


def first_error_view(err, off, keep):
    """Steps up to and including the first labelled error of each answer (all steps of clean answers), label =
    'is the first error'. This is the one target every protocol labels the same way (ProcessBench stops there,
    Socratic/H2V/PRM800K-style labels cascade after it)."""
    use = np.zeros(len(err), bool); y = np.zeros(len(err), bool)
    for a, b in zip(off[:-1], off[1:]):
        k = np.flatnonzero(keep[a:b]) + a
        e = k[err[k]]
        if len(e):
            f = e[0]; use[k[k <= f]] = True; y[f] = True
        else:
            use[k] = True
    return use, y


def strat_auc(y, s, strata):
    """Pooled AUROC inside each position stratum, weighted by the number of (error, correct) pairs."""
    num = den = 0.
    for v in np.unique(strata):
        m = strata == v; yy = y[m]
        if yy.all() or not yy.any(): continue
        w = yy.sum() * (~yy).sum(); num += w * auc(yy, s[m]); den += w
    return num / den if den else np.nan


def answer_auc(err, s, off, keep):
    has = np.array([err[a:b][keep[a:b]].any() for a, b in zip(off[:-1], off[1:])])
    mx = np.array([s[a:b][keep[a:b]].max() if keep[a:b].any() else -np.inf for a, b in zip(off[:-1], off[1:])])
    ok = np.isfinite(mx)
    return auc(has[ok], mx[ok])


def load(b, p):
    f = CELLS / f'{b}__{p}.npz'
    return dict(np.load(f, allow_pickle=False)) if f.exists() else None


def label_structure(d):
    err, off, keep = d['err'], d['off'], d['keep']
    n = np.diff(off); rows = []
    clean = cascade = last = first_is_last = 0; nerr_ans = []; fpos = []
    for a, b in zip(off[:-1], off[1:]):
        k = np.flatnonzero(keep[a:b]); e = err[a:b][k]
        if not len(k): continue
        if not e.any():
            clean += 1; continue
        nerr_ans.append(e.sum()); f = np.argmax(e)
        fpos.append(f / max(len(k) - 1, 1))
        cascade += bool(e[f:].all() and len(e) - f > 1)
        last += bool(e[-1]); first_is_last += bool(f == len(k) - 1)
    ne = len(nerr_ans); na = clean + ne
    si = step_index(off); use, y = first_error_view(err, off, keep)
    w = within(err, si.astype(float), off, keep)
    yk = err[keep]; sizes = np.array([keep[a:b].sum() for a, b in zip(off[:-1], off[1:])])
    a_mean = np.repeat([err[a:b][keep[a:b]].mean() if keep[a:b].any() else 0 for a, b in zip(off[:-1], off[1:])], n)[keep]
    icc = float(np.var(a_mean) / np.var(yk)) if np.var(yk) > 0 else np.nan     # share of step-label variance between answers
    return dict(answers=na, steps=int(keep.sum()), steps_per_answer=float(sizes[sizes > 0].mean()),
                step_error_rate=float(yk.mean()), clean_share=clean / na, errors_per_error_answer=float(np.mean(nerr_ans)),
                single_error_share=float(np.mean(np.array(nerr_ans) == 1)),
                cascade_share=cascade / ne, last_step_error_share=last / ne, first_error_is_last_kept=first_is_last / ne,
                first_error_relpos=float(np.mean(fpos)), between_answer_label_share=icc,
                position_within_auc=float(np.nanmean(w)), within_auc_answers=int(np.isfinite(w).sum()),
                position_pooled_auc=auc(yk, rel_pos(off)[keep]), position_first_error_auc=auc(y[use], si[use]))


def arm_quality(d, s):
    err, off, keep = d['err'], d['off'], d['keep']
    use, y = first_error_view(err, off, keep)
    si = np.minimum(step_index(off), POSCAP)
    return dict(pooled=auc(err[keep], s[keep]), within=float(np.nanmean(within(err, s, off, keep))),
                fe_strat=strat_auc(y[use], s[use], si[use]), answer=answer_auc(err, s, off, keep))


def cv_auc(Xf, y, groups, seed=0):
    p = np.zeros(len(y)); gkf = GroupKFold(5)
    for tr, te in gkf.split(Xf, y, groups):
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xf[tr], y[tr]); p[te] = m.decision_function(Xf[te])
    return auc(y, p)


def main():
    T1, T2, T3, T4 = [], [], [], []
    for b in BENCHES:
        for p in PRMS:
            d = load(b, p)
            if d is None:
                print('missing', b, p); continue
            ls = label_structure(d); ls.update(bench=b, prm=p, K=int(d['K']), c4u_prevalence=float(d['prevalence']))
            ls['c4u_prev_ratio'] = ls['c4u_prevalence'] / ls['step_error_rate']
            ls['c4u_flag_rate'] = float(d['c4u_flags'][d['keep']].mean()); ls['tele_flag_rate'] = float(d['tele_flags'][d['keep']].mean())
            T1.append(ls)
            err, off, keep, X = d['err'], d['off'], d['keep'], d['X'].astype(float)
            arms = dict(PRM=d['prm'], TELE16=d['tele'], C4U=d['c4u'], position=rel_pos(off), PRM_answer_z=X[:, 0])
            q = {a: arm_quality(d, s) for a, s in arms.items()}
            for a, v in q.items():
                T2.append(dict(bench=b, prm=p, arm=a, **v))
            # channels
            use, y1 = first_error_view(err, off, keep); si = np.minimum(step_index(off), POSCAP)
            prm = d['prm']; kk = keep
            for j, ch in enumerate(N16):
                x = X[:, 2 + j]
                if np.nanstd(x[kk]) == 0:
                    continue
                r0 = spearmanr(x[kk & ~err], prm[kk & ~err])[0]; r1 = spearmanr(x[kk & err], prm[kk & err])[0]
                T3.append(dict(bench=b, prm=p, channel=ch, pooled=auc(err[kk], x[kk]),
                               within=float(np.nanmean(within(err, x, off, keep))),
                               fe_strat=strat_auc(y1[use], x[use], si[use]),
                               rho_prm_correct=r0, rho_prm_error=r1, rho_prm_all=spearmanr(x[kk], prm[kk])[0]))
            # LR headroom (answer-grouped CV; supervised reference only)
            groups = np.repeat(np.arange(len(off) - 1), np.diff(off))[kk]
            yk = err[kk]; Xk = X[kk]; pos = rel_pos(off)[kk][:, None]
            Xs = (Xk - Xk.mean(0)) / np.where(Xk.std(0) > 0, Xk.std(0), 1)
            lr = dict(bench=b, prm=p, LR_PRM=cv_auc(Xs[:, :2], yk, groups), LR_16=cv_auc(Xs[:, 2:], yk, groups),
                      LR_PRM16=cv_auc(Xs, yk, groups), LR_PRM16_pos=cv_auc(np.column_stack([Xs, pos]), yk, groups),
                      LR_pos=cv_auc(pos, yk, groups))
            T4.append(lr)
            print(b, p, 'PRM %.3f TELE %.3f C4U %.3f | LR PRM %.3f PRM+16 %.3f' % (
                q['PRM']['pooled'], q['TELE16']['pooled'], q['C4U']['pooled'], lr['LR_PRM'], lr['LR_PRM16']), flush=True)
    E = __import__('csv')
    for name, rows in (('T1_label_structure', T1), ('T2_arm_quality', T2), ('T3_channels', T3), ('T4_lr_headroom', T4)):
        with open(HERE / f'{name}.csv', 'w', newline='') as f:
            w = E.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
