"""E1 of ds_gate_v1 (PROTOCOL.json): fit the hierarchical Dawid-Skene gate on every cell and compute the five
diagnostics D1-D5. Labels are read only after the fit, by the diagnostics.

    python e1_check.py      # -> E1_RESULTS.csv, E1_CALIBRATION.csv, E1_FITS.json, E1_VERDICT.json
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'bracha_lr_transfer_v1'))
import ds_gate as D  # noqa: E402
import lr_transfer as L  # noqa: E402  (cell loaders; imports sub5_all, stage8)
A_sub5, E, S8 = L.A, L.E, L.S8
from spectral_utils import prm_anchored_lsml as A  # noqa: E402

BENCHES = ['prmbench', 'processbench', 'crv_arith', 'crv_gsm8k']
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']


def run_cell(bench, prm):
    c = L.load(bench, prm, 'prm')
    off = c.off; n = np.diff(off)
    X = E.channels(c, 'N16')
    risk = c.arms['PRM_Q80' if bench == 'prmbench' else 'PRM']
    lens = np.repeat(n, n)
    pool = lens > 1                                             # all answers with >= 2 steps, no label or class field (review S4)
    V, Z, sign = D.clean_and_vote(E, X, risk, off, pool)
    G, K, cg = D.group_votes(A, V, pool)
    info = dict(benchmark=bench, prm=prm, K=int(K), groups=cg.tolist(), n_flipped=int((sign < 0).sum()))
    if K < 3:
        info['valid'] = False; return info, None
    pos = D.relative_position(off)
    fit = D.fit_hds(G, off, pos)
    info.update(valid=True, loglik=fit['loglik'], n_iter=fit['n_iter'], start=fit['start'], params=fit['params'],
                label_switch_warning=bool(fit.get('label_switch_warning', False)), ll_drops=fit['ll_drops'],
                worst_ll_drop=fit['worst_drop'], degenerate=fit['degenerate'])
    # ---------------- labels from here on (diagnostics only)
    keep_eval = c.keep_gold if bench == 'processbench' else c.keep
    has_err = np.array([c.err[a:b][keep_eval[a:b]].any() for a, b in zip(off[:-1], off[1:])])
    true_cnt = np.array([c.err[a:b][keep_eval[a:b]].sum() for a, b in zip(off[:-1], off[1:])])
    ev = (c.cls != 'correct') if bench == 'prmbench' else np.ones(len(n), bool)  # evaluated answers
    w = fit['p_answer']; q = fit['q']; rho = fit['params']['rho']
    sumq = np.add.reduceat(q, off[:-1])
    auto, _ = S8.auto(X, E.answer_z(risk, off))
    za = E.answer_z(auto, off); gapcnt = E.gap_counts(za, off)
    stepmask = keep_eval & np.repeat(ev, n)
    he, wv = has_err[ev], w[ev]
    idx = np.empty(len(wv), int); idx[np.argsort(wv, kind='stable')] = np.arange(len(wv)) * 10 // len(wv)   # equal-count bins by rank (review N4)
    ece = float(sum(abs(wv[idx == k].mean() - he[idx == k].mean()) * (idx == k).mean() for k in range(10) if (idx == k).any()))
    cal = [dict(benchmark=bench, prm=prm, bin=k, mean_p=float(wv[idx == k].mean()), share_with_error=float(he[idx == k].mean()),
                n=int((idx == k).sum())) for k in range(10) if (idx == k).any()]
    row = dict(benchmark=bench, prm=prm, K=int(K),
               D1_rho_hat=rho, D1_truth_fit_population=float(has_err.mean()), D1_mean_p_evaluated=float(wv.mean()),
               D1_truth_evaluated=float(he.mean()),
               D2_mean_q=float(q[stepmask].mean()), D2_true_step_rate=float(c.err[stepmask].mean()),
               D3_ece=ece)
    for tname, t in (('rho', rho), ('0.5', .5)):
        closed = wv < t
        row[f'D4_kept_has_error_at_{tname}'] = float((~closed[he]).mean()) if he.any() else float('nan')
        row[f'D4_closed_clean_at_{tname}'] = float(closed[~he].mean()) if (~he).any() else float('nan')
        row[f'D4_closed_all_at_{tname}'] = float(closed.mean())
    row.update(D5_spearman_sumq_true=float(spearmanr(sumq[ev], true_cnt[ev])[0]),
               D5_spearman_gapcount_true=float(spearmanr(gapcnt[ev], true_cnt[ev])[0]),
               D5_spearman_sumq_length=float(spearmanr(sumq[ev], n[ev])[0]),
               D5_spearman_gapcount_length=float(spearmanr(gapcnt[ev], n[ev])[0]),
               n_answers_evaluated=int(ev.sum()), n_clean_evaluated=int((~he).sum()))
    return info, (row, cal)


def verdict(rows, n_invalid):
    get = lambda b: [r for r in rows if r['benchmark'] == b]
    ext = [r for r in rows if r['benchmark'] in ('processbench', 'crv_arith', 'crv_gsm8k')]
    by = {b: get(b) for b in ('processbench', 'crv_arith', 'crv_gsm8k')}
    d1 = all(sum(abs(r['D1_rho_hat'] - r['D1_truth_fit_population']) <= .05 for r in rs) >= 3 for rs in by.values())
    d3 = all(sum(r['D3_ece'] <= .05 for r in rs) >= 3 for rs in by.values())
    pb = by['processbench']
    d4 = sum(r['D4_closed_clean_at_rho'] >= .30 and r['D4_kept_has_error_at_rho'] >= .85 for r in pb) >= 3 and \
        all(1 - r['D4_kept_has_error_at_rho'] <= .05 for r in get('prmbench'))
    d5 = all(abs(r['D5_spearman_sumq_length']) < abs(r['D5_spearman_gapcount_length']) and r['D5_spearman_sumq_true'] > r['D5_spearman_gapcount_true']
             for r in rows)                                     # reference = the GAP count of the same cell (amendment A3)
    ok = bool(d1 and d3 and d4 and d5 and n_invalid == 0)       # an invalid (K < 3) cell is a failure (review S2)
    return dict(D1=d1, D3=d3, D4=d4, D5=d5, n_invalid_cells=n_invalid, pass_all=ok, n_cells=len(rows), n_external=len(ext))


def main():
    rows, cals, fits = [], [], []
    for bench in BENCHES:
        for prm in PRMS:
            info, out = run_cell(bench, prm); fits.append(info)
            if out:
                rows.append(out[0]); cals += out[1]
                r = out[0]
                print(bench, prm, 'K', r['K'], 'rho', round(r['D1_rho_hat'], 3), 'truth', round(r['D1_truth_fit_population'], 3),
                      'ECE', round(r['D3_ece'], 3), 'closed clean@rho', round(r['D4_closed_clean_at_rho'], 3) if r['D4_closed_clean_at_rho'] == r['D4_closed_clean_at_rho'] else 'nan',
                      'kept err@rho', round(r['D4_kept_has_error_at_rho'], 3), flush=True)
            else:
                print(bench, prm, 'INVALID K', info['K'], flush=True)
    E.write_csv(HERE / 'E1_RESULTS.csv', rows); E.write_csv(HERE / 'E1_CALIBRATION.csv', cals)
    json.dump(fits, open(HERE / 'E1_FITS.json', 'w'), indent=1, default=float)
    v = verdict(rows, sum(not f.get('valid', False) for f in fits)); json.dump(v, open(HERE / 'E1_VERDICT.json', 'w'), indent=1)
    print('VERDICT', v)


if __name__ == '__main__':
    main()
