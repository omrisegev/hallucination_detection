"""v2 evaluation (reads test-half labels). Usage: python evaluate_v2.py <bundle_dir> <run_dir> <out_dir>"""
import json, pathlib, sys, numpy as np, pandas as pd
from common import macro_f1, t6
from run_v2 import PRIMARY, R
ARMS = ['HH_BAYES', 'HH_K', 'GAP', 'SHARE', 'GATE_GAP', 'THR']
CTL = ['GAP', 'SHARE', 'GATE_GAP', 'THR']


def main(bundle, rundir, out):
    rows = []
    for c in PRIMARY:
        B = np.load(pathlib.Path(bundle) / f'{c}.npz', allow_pickle=False)
        off = B['offsets'].astype(np.int64); n = np.diff(off)
        y = B['step_err_LABEL'].astype(bool); m = B['step_mask'].astype(bool); keep = B['step_keep_eval'].astype(bool)
        ev = B['ans_evaluated'].astype(bool); ha = B['ans_has_err_LABEL'].astype(bool)
        bench = [b for b in ('processbench', 'crv_arith', 'crv_gsm8k', 'prmbench') if b in c][0]
        for r in range(R):
            F = np.load(pathlib.Path(rundir) / f'{c}__r{r}.npz', allow_pickle=False); fit = json.loads(str(F['fit_json']))
            test = F['test']; tmask = np.zeros(len(n), bool); tmask[test] = True
            sm = m & np.repeat(tmask, n)
            row = dict(cell=c, bench=bench, prm=c.split(f'{bench}_')[1], rep=r, at_bound=fit['at_bound'],
                       pi_cal=fit['pi_cal'], delta_over_s0=fit['delta'] / fit['s0'], rho=fit['rho'], f=fit['f'])
            for a in ARMS:
                row[f'mf1_{a}'] = macro_f1(y[sm], F[f'flag_{a}'][sm])
                if bench == 'processbench':
                    row[f't6_{a}'] = t6(off, y, keep, F[f'flag_{a}'], ha, test)
            te = test[ev[test]]
            g_hh = F['pA'][te] > .5
            amax = np.array([B['step_raw_fused'][a:b].max() for a, b in zip(off[:-1], off[1:])])
            g_ctl = amax[te] > np.quantile(amax, 1 - fit['pi_cal'])
            def bacc(g):
                t = ha[te]; return 0.5 * ((g & t).sum() / max(t.sum(), 1) + (~g & ~t).sum() / max((~t).sum(), 1))
            row['gate_bacc_HH'], row['gate_bacc_GATE_GAP'] = bacc(g_hh), bacc(g_ctl)
            rows.append(row)
    d = pd.DataFrame(rows); out = pathlib.Path(out); d.to_csv(out / 'RESULTS.csv', index=False)
    cells, verdict = [], {}
    for c, g in d.groupby('cell', sort=False):
        best = max(CTL, key=lambda k: g[f'mf1_{k}'].mean()); D = g['mf1_HH_BAYES'] - g[f'mf1_{best}']
        cr = dict(cell=c, bench=g.bench.iloc[0], best_ctl=best, P2_mean_D=D.mean(), P2_reps_pos=int((D > 0).sum()),
                  P2_pass=bool(D.mean() > 0 and (D > 0).sum() >= 4))
        if g.bench.iloc[0] == 'processbench':
            b3 = max(['GATE_GAP', 'THR'], key=lambda k: g[f't6_{k}'].mean()); D3 = g['t6_HH_BAYES'] - g[f't6_{b3}']
            cr.update(P3_best=b3, P3_mean_D=D3.mean(), P3_reps_pos=int((D3 > 0).sum()), P3_pass=bool(D3.mean() > 0 and (D3 > 0).sum() >= 4))
        cells.append(cr)
    C = pd.DataFrame(cells); C.to_csv(out / 'CELLS.csv', index=False)
    for b, g in C.groupby('bench'):
        verdict[b] = dict(P2_cells=int(g.P2_pass.sum()), P2_pass=bool(g.P2_pass.sum() >= 3))
        if b == 'processbench':
            verdict[b].update(P3_cells=int(g.P3_pass.sum()), P3_pass=bool(g.P3_pass.sum() >= 3))
    nb = sum(v['P2_pass'] for v in verdict.values()); p3 = verdict.get('processbench', {}).get('P3_pass', False)
    verdict['line'] = 'works' if nb >= 2 else ('partial' if (nb == 1 or p3) else 'fails')
    verdict['fits_at_bound'] = int(d.at_bound.sum())
    json.dump(verdict, open(out / 'VERDICT.json', 'w'), indent=1)
    print(json.dumps(verdict, indent=1)); print(C.to_string())


if __name__ == '__main__':
    main(*sys.argv[1:4])
