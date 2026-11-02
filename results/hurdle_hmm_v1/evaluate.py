"""Label phase. Reads saved fits + bundle labels; writes RESULTS.csv and VERDICT.json.
Usage: python evaluate.py <bundle_dir> <fit_dir> <out_dir>"""
import json, pathlib, sys, numpy as np
from sklearn.metrics import roc_auc_score
B_PER = 1000


def macro_f1(y, f):
    def f1(t, p):
        tp = (t & p).sum(); return 2 * tp / max(t.sum() + p.sum(), 1)
    return 0.5 * (f1(y, f) + f1(~y, ~f))


def t6_f1(off, y, keep, flags, has_err):
    acc_e, acc_c = [], []
    for a, (lo, hi) in enumerate(zip(off[:-1], off[1:])):
        fl = np.where(flags[lo:hi])[0]; pred = fl[0] if len(fl) else -1
        if has_err[a]:
            fe = np.where(y[lo:hi] & keep[lo:hi])[0]; acc_e.append(len(fe) > 0 and pred == fe[0])
        else:
            acc_c.append(pred == -1)
    te, tc = np.mean(acc_e), np.mean(acc_c)
    return te, tc, (2 * te * tc / (te + tc) if te + tc > 0 else 0.0)


def main(bundle, fitdir, out):
    rows = []
    for fp in sorted(pathlib.Path(fitdir).glob('*.npz')):
        c = fp.stem; F = np.load(fp, allow_pickle=False); fit = json.loads(str(F['fit_json']))
        B = np.load(pathlib.Path(bundle) / f'{c}.npz', allow_pickle=False)
        off = B['offsets'].astype(np.int64); n = np.diff(off)
        y = B['step_err_LABEL'].astype(bool); m = B['step_mask'].astype(bool); keep = B['step_keep_eval'].astype(bool)
        ev = B['ans_evaluated'].astype(bool); ha = B['ans_has_err_LABEL'].astype(bool)
        bench = [b for b in ('processbench', 'crv_arith', 'crv_gsm8k', 'prmbench') if b in c][0]
        fb, fk, fg = F['flag_bayes'], F['flag_hhk'], F['flag_gap']
        mf_b, mf_k, mf_g = macro_f1(y[m], fb[m]), macro_f1(y[m], fk[m]), macro_f1(y[m], fg[m])
        # paired answer bootstrap for HH-BAYES - GAP
        rng = np.random.default_rng(0); A = len(n); ans_of = np.repeat(np.arange(A), n)
        diffs = []
        for _ in range(B_PER):
            pick = rng.integers(0, A, A); w = np.bincount(pick, minlength=A)[ans_of] * m
            def wf1(t, p):
                tp = (w * (t & p)).sum(); return 2 * tp / max((w * t).sum() + (w * p).sum(), 1e-12)
            mb = 0.5 * (wf1(y, fb) + wf1(~y, ~fb)); mg = 0.5 * (wf1(y, fg) + wf1(~y, ~fg)); diffs.append(mb - mg)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        auc_pA = roc_auc_score(ha[ev], F['pA'][ev]); auc_raw = roc_auc_score(ha[ev], B['ans_raw_fused_answer_max'][ev])
        r = dict(cell=c, bench=bench, prm=c.split(f'{bench}_')[1], true_pi=ha.mean(), pi_hat=fit['pi'],
                 P1_abs_err=abs(fit['pi'] - ha.mean()), mf1_bayes=mf_b, mf1_hhk=mf_k, mf1_gap=mf_g,
                 P2_diff=mf_b - mf_g, P2_ci_lo=lo, P2_ci_hi=hi, auc_pA=auc_pA, auc_raw_max=auc_raw,
                 P3_diff=auc_pA - auc_raw, share_answers_called_err=(F['pA'] > .5).mean(),
                 share_steps_flagged_bayes=fb[m].mean(), share_steps_err_true=y[m].mean(),
                 delta_over_s0=fit['delta'] / fit['s0'], f=fit['f'], rho=fit['rho'], h=fit['h'], e1=fit['e1'],
                 converged=fit['converged'])
        if bench == 'processbench':
            for name, fl in (('bayes', fb), ('hhk', fk), ('gap', fg)):
                te, tc, f1 = t6_f1(off, y, keep, fl, ha); r[f't6_{name}_tpr'], r[f't6_{name}_tnr'], r[f't6_{name}_f1'] = te, tc, f1
        else:
            E = ha & ev
            cnt = np.add.reduceat((y & m).astype(int), off[:-1]); kb = np.add.reduceat(fb.astype(int), off[:-1])
            kg = np.add.reduceat(fg.astype(int), off[:-1])
            r['count_mae_bayes'] = np.abs(kb - cnt)[E].mean(); r['count_mae_gap'] = np.abs(kg - cnt)[E].mean()
        rows.append(r)
    import pandas as pd
    d = pd.DataFrame(rows); out = pathlib.Path(out); d.to_csv(out / 'RESULTS.csv', index=False)
    verdict = {}
    for b, g in d.groupby('bench'):
        p1 = int((g.P1_abs_err <= .05).sum()); p2 = int((g.P2_diff > 0).sum()); p3 = int((g.P3_diff >= -.02).sum())
        verdict[b] = dict(n_cells=len(g), P1=p1, P2=p2, P3=p3, P1_pass=p1 >= 3, P2_pass=p2 >= 3, P3_pass=p3 >= 3)
    nb1 = sum(v['P1_pass'] for v in verdict.values()); nb2 = sum(v['P2_pass'] for v in verdict.values())
    verdict['line'] = 'works' if (nb1 >= 2 and nb2 >= 2) else ('partial' if (nb1 >= 2 or nb2 >= 2) else 'fails')
    json.dump(verdict, open(out / 'VERDICT.json', 'w'), indent=1, default=bool)
    print(json.dumps(verdict, indent=1, default=bool))


if __name__ == '__main__':
    main(*sys.argv[1:4])
