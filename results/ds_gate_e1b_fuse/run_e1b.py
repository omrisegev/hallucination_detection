"""E1b (PROTOCOL.json): the corrected-E1 hierarchical DS gate (results/ds_gate_v1b_fixed, reused read-only) on FUSE
global-threshold votes (votes_e1b.fuse_votes) instead of within-answer GAP votes. Same diagnostics D1-D5 and decision
rules as the corrected E1, plus within-answer AUC of the step posterior q and S-hat at the FUSE votes vs at the
corrected-E1 GAP votes of the same voters. Labels are read only after the fit.

    python run_e1b.py [arm ...]     # default: AUTO AUTO_NATIVE TELE16 SUB5
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import json, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'ds_gate_v1b_fixed'))
import votes_e1b as VB  # noqa: E402
import run_e1 as R  # noqa: E402  (corrected E1: arm inputs, metrics, verdict; read-only)
D, E, A, gap_flags = R.D, R.E, R.A, R.gap_flags

ARMS = ['AUTO', 'AUTO_NATIVE', 'TELE16', 'SUB5']


def run_cell(arm, bench, prm):
    base_arm = 'AUTO' if arm.startswith('AUTO') else arm
    c, X, prm_risk, score = R.arm_inputs(base_arm, bench, prm)           # stored channels + the arm's fused score (GAP)
    fv = VB.fuse_votes(bench, prm, arm)
    assert np.array_equal(fv['c'].off, c.off) and np.array_equal(fv['c'].ids, c.ids)
    V = fv['V']; off = c.off; n = np.diff(off); pool = np.repeat(n, n) > 1
    Vgap = D.clean_and_vote(E, X, off, pool, prm_risk)                  # corrected-E1 GAP votes, same voters
    if base_arm == 'SUB5': assert Vgap.shape[1] == V.shape[1]
    sh_gap = VB.shat(*VB.moments(np.where(Vgap[pool] > 0, 1., -1.))[1:])
    sh_fuse_pool = VB.shat(*VB.moments(np.where(V[pool] > 0, 1., -1.))[1:])
    G, K, cg = D.group_votes(A, V, pool)
    info = dict(arm=arm, benchmark=bench, prm=prm, K=int(K), groups=cg.tolist(), names=fv['names'],
                tau_quantile=fv['tau_quantile'], vote_rate=fv['vote_rate'], shat_init=fv['shat_init'], shat_final=fv['shat_final'])
    if K < 3:
        info['valid'] = False; return info, None
    fit = D.fit_hds(G, off, D.bin_design(D.relative_position(off)))
    info.update(valid=True, loglik=fit['loglik'], params=fit['params'], label_switch_warning=fit['label_switch_warning'],
                ll_drops=fit['ll_drops'], worst_ll_drop=fit['worst_drop'], degenerate=fit['degenerate'])
    w = fit['p_answer']; q = fit['q']; rho = fit['params']['rho']
    f_gap = gap_flags(c, score)
    nflag = np.add.reduceat(f_gap.astype(int), off[:-1]); sq_f = np.add.reduceat(np.where(f_gap, q, 0.), off[:-1])
    gates = {'rho': w < rho, '0.5': w < .5, 'cost': sq_f < (1 - w) * nflag}
    # ---------------- labels from here on
    keep_gold = c.keep_gold.astype(bool) if bench == 'processbench' else None
    keep_eval = keep_gold if bench == 'processbench' else c.keep
    has_err = np.array([c.err[a:b][keep_eval[a:b]].any() for a, b in zip(off[:-1], off[1:])])
    true_cnt = np.array([c.err[a:b][keep_eval[a:b]].sum() for a, b in zip(off[:-1], off[1:])])
    ev = (c.cls != 'correct') if bench == 'prmbench' else np.ones(len(n), bool)
    he, wv = has_err[ev], w[ev]
    sumq = np.add.reduceat(q, off[:-1]); gapcnt = E.gap_counts(E.answer_z(score, off), off)
    idx = np.empty(len(wv), int); idx[np.argsort(wv, kind='stable')] = np.arange(len(wv)) * 10 // len(wv)
    ece = float(sum(abs(wv[idx == k].mean() - he[idx == k].mean()) * (idx == k).mean() for k in range(10) if (idx == k).any()))
    stepmask = keep_eval & np.repeat(ev, n)
    wa = E.per_answer_auc(c.err, q, off, keep_eval); wa[~ev] = np.nan
    wa_s = E.per_answer_auc(c.err, score, off, keep_eval); wa_s[~ev] = np.nan
    row = dict(arm=arm, benchmark=bench, prm=prm, K=int(K), D1_rho_hat=rho, D1_truth_fit_population=float(has_err.mean()),
               D1_mean_p_evaluated=float(wv.mean()), D1_truth_evaluated=float(he.mean()),
               D2_mean_q=float(q[stepmask].mean()), D2_true_step_rate=float(c.err[stepmask].mean()), D3_ece=ece,
               within_auc_q=float(np.nanmean(wa)), within_auc_fused_score=float(np.nanmean(wa_s)),
               shat_gap_votes=sh_gap, shat_fuse_votes=sh_fuse_pool,
               pi_bins=' '.join(f'{x:.3f}' for x in fit['params']['pi_bins']))
    for tname, g in gates.items():
        gv = g[ev]
        row[f'D4_kept_has_error_at_{tname}'] = float((~gv[he]).mean()) if he.any() else float('nan')
        row[f'D4_closed_clean_at_{tname}'] = float(gv[~he].mean()) if (~he).any() else float('nan')
        row[f'D4_closed_all_at_{tname}'] = float(gv.mean())
    row['metric_GAP'] = R.metric(c, bench, f_gap, keep_gold)
    for tname, g in gates.items():
        row[f'metric_gated_{tname}'] = R.metric(c, bench, f_gap & ~np.repeat(g, n), keep_gold)
    if bench == 'processbench':
        _, ae, ac = R.pb_f1(c, f_gap & ~np.repeat(gates['rho'], n), keep_gold)
        row.update(pb_acc_error_gated_rho=ae, pb_acc_clean_gated_rho=ac)
    row.update(D5_spearman_sumq_true=float(spearmanr(sumq[ev], true_cnt[ev])[0]),
               D5_spearman_gapcount_true=float(spearmanr(gapcnt[ev], true_cnt[ev])[0]),
               D5_spearman_sumq_length=float(spearmanr(sumq[ev], n[ev])[0]),
               D5_spearman_gapcount_length=float(spearmanr(gapcnt[ev], n[ev])[0]),
               n_answers_evaluated=int(ev.sum()), n_clean_evaluated=int((~he).sum()))
    return info, row


def main():
    for arm in (sys.argv[1:] or ARMS):
        rows, fits = [], []
        for bench in R.BENCHES:
            for prm in R.PRMS:
                t0 = time.time()
                try:
                    info, row = run_cell(arm, bench, prm)
                except FileNotFoundError as ex:                          # missing raw bank: reported, counts as invalid
                    info, row = dict(arm=arm, benchmark=bench, prm=prm, valid=False, missing_raw=str(ex)), None
                fits.append(info)
                if row:
                    rows.append(row)
                    print(arm, bench, prm, 'K', row['K'], 'rho', round(row['D1_rho_hat'], 3), 'truth', round(row['D1_truth_fit_population'], 3),
                          'ECE', round(row['D3_ece'], 3), 'closed clean@rho', round(row['D4_closed_clean_at_rho'], 3),
                          'kept err@rho', round(row['D4_kept_has_error_at_rho'], 3), 'wAUCq', round(row['within_auc_q'], 4),
                          'GAP', round(row['metric_GAP'], 2), 'gated rho/cost/.5', round(row['metric_gated_rho'], 2),
                          round(row['metric_gated_cost'], 2), round(row['metric_gated_0.5'], 2),
                          'Shat gap/fuse', round(row['shat_gap_votes'], 3), round(row['shat_fuse_votes'], 3), f'{time.time() - t0:.0f}s', flush=True)
                else:
                    print(arm, bench, prm, 'INVALID', info.get('K'), info.get('missing_raw', ''), flush=True)
        E.write_csv(HERE / f'E1B_RESULTS_{arm}.csv', rows)
        json.dump(fits, open(HERE / f'E1B_FITS_{arm}.json', 'w'), indent=1, default=float)
        n_missing = sum('missing_raw' in f for f in fits)
        v = R.verdict(rows, sum(not f.get('valid', False) for f in fits)); v['n_missing_raw'] = n_missing
        json.dump(v, open(HERE / f'E1B_VERDICT_{arm}.json', 'w'), indent=1); print('VERDICT', arm, v, flush=True)


if __name__ == '__main__':
    main()
