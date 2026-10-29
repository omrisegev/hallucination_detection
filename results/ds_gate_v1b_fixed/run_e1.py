"""E1 rerun with the fixes of PROTOCOL.json (ds_gate_v1b_fixed): stored orientation, every voter (PRM too) cleaned of
position, binned position prior; three arms per cell (AUTO = PRM + N16 backbone telemetry; TELE16 and SUB5 = standalone
telemetry, no PRM). Diagnostics D1-D5 as ds_gate_v1, plus the gated decision rule against plain GAP.
Labels are read only after the fit.

    python run_e1.py [arm ...]     # default: AUTO TELE16 SUB5 -> E1_RESULTS_<arm>.csv, E1_VERDICT_<arm>.json, ...
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
sys.path.insert(0, str(HERE.parent / 'prm_boost_math_rigor_v1'))
import ds_gate_b as D  # noqa: E402
import lr_transfer as L  # noqa: E402
E, S8 = L.E, L.S8
from spectral_utils import prm_anchored_lsml as A  # noqa: E402
from stage7_hygiene import gap_flags  # noqa: E402

BENCHES = ['prmbench', 'processbench', 'crv_arith', 'crv_gsm8k']
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
N16N = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level', 'energy_innovation',
        'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0', 'realized_z', 'realized_drv', 'digit_alternative',
        'digit_spread', 'digit_alternative_innovation']
SUB5 = ['q15_H1', 'chosen_surprisal', 'top15_turnover', 'dominant_freq16', 'bocpd_p0', 'digit_spread']


def check_names(c):
    if c.bench == 'prmbench':
        p = E.DATA / ('prmbench' if '__' not in c.prm else 'prmbench_standalone') / f'ABL5_CHANNELS_{c.prm}.npz'
        if p.exists():
            assert [str(x) for x in np.load(p)['channel_names__N16']] == N16N, 'channel order'


def arm_inputs(arm, bench, prm):
    c = L.load(bench, prm, 'prm' if arm == 'AUTO' else 'sa')
    X = E.channels(c, 'N16'); check_names(c)
    prm_risk, score = None, None
    if arm == 'AUTO':
        prm_risk = np.asarray(c.arms['PRM_Q80' if bench == 'prmbench' else 'PRM'], float)
        score, _ = S8.auto(X, E.answer_z(prm_risk, c.off))
    else:
        if arm == 'SUB5': X = X[:, [N16N.index(nm) for nm in SUB5]]
        score, _ = S8.fit(X, anchor=X.mean(1))
    return c, X, prm_risk, np.asarray(score, float)


def pb_f1(c, f, keep_gold):
    """ProcessBench official: predicted first error = first flagged step (-1 if none) vs gold first error (-1 if clean);
    harmonic mean of accuracy on erroneous and on clean answers, x100. Empty steps are not modelled here."""
    acc_e, acc_c = [], []
    for a, b in zip(c.off[:-1], c.off[1:]):
        g = np.flatnonzero(c.err[a:b] & keep_gold[a:b]); g = int(g[0]) if len(g) else -1
        p = np.flatnonzero(f[a:b]); p = int(p[0]) if len(p) else -1
        (acc_e if g >= 0 else acc_c).append(p == g)
    ae, ac = float(np.mean(acc_e)), float(np.mean(acc_c))
    return 100 * 2 * ae * ac / (ae + ac) if ae + ac > 0 else 0., ae, ac


def metric(c, bench, f, keep_gold=None):
    if bench == 'processbench':
        return pb_f1(c, f, keep_gold)[0]
    v = float(E.official_from_counts(E.counts_by_answer(c, f).sum(0), c.bench))
    return v if bench == 'prmbench' else 100 * v


def run_cell(arm, bench, prm):
    c, X, prm_risk, score = arm_inputs(arm, bench, prm)
    off = c.off; n = np.diff(off); lens = np.repeat(n, n)
    pool = lens > 1
    V = D.clean_and_vote(E, X, off, pool, prm_risk)
    G, K, cg = D.group_votes(A, V, pool)
    info = dict(arm=arm, benchmark=bench, prm=prm, K=int(K), groups=cg.tolist())
    if K < 3:
        info['valid'] = False; return info, None
    Dm = D.bin_design(D.relative_position(off))
    fit = D.fit_hds(G, off, Dm)
    info.update(valid=True, loglik=fit['loglik'], n_iter=fit['n_iter'], start=fit['start'], params=fit['params'],
                label_switch_warning=fit['label_switch_warning'], ll_drops=fit['ll_drops'], worst_ll_drop=fit['worst_drop'],
                degenerate=fit['degenerate'])
    w = fit['p_answer']; q = fit['q']; rho = fit['params']['rho']
    # decision rules (label-free): GAP flags of the arm's fused score; gated answers get k = 0
    f_gap = gap_flags(c, score)
    nflag = np.add.reduceat(f_gap.astype(int), off[:-1]); sq_f = np.add.reduceat(np.where(f_gap, q, 0.), off[:-1])
    gates = {'rho': w < rho, '0.5': w < .5, 'cost': sq_f < (1 - w) * nflag}   # cost: P(err)*E[correct flags] < P(clean)*|F|
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
    row = dict(arm=arm, benchmark=bench, prm=prm, K=int(K), D1_rho_hat=rho, D1_truth_fit_population=float(has_err.mean()),
               D1_mean_p_evaluated=float(wv.mean()), D1_truth_evaluated=float(he.mean()),
               D2_mean_q=float(q[stepmask].mean()), D2_true_step_rate=float(c.err[stepmask].mean()), D3_ece=ece,
               pi_bins=' '.join(f'{x:.3f}' for x in fit['params']['pi_bins']))
    for tname, g in gates.items():
        gv = g[ev]
        row[f'D4_kept_has_error_at_{tname}'] = float((~gv[he]).mean()) if he.any() else float('nan')
        row[f'D4_closed_clean_at_{tname}'] = float(gv[~he].mean()) if (~he).any() else float('nan')
        row[f'D4_closed_all_at_{tname}'] = float(gv.mean())
    row['metric_GAP'] = metric(c, bench, f_gap, keep_gold)
    for tname, g in gates.items():
        row[f'metric_gated_{tname}'] = metric(c, bench, f_gap & ~np.repeat(g, n), keep_gold)
    if bench == 'processbench':
        _, ae, ac = pb_f1(c, f_gap & ~np.repeat(gates['rho'], n), keep_gold)
        row.update(pb_acc_error_gated_rho=ae, pb_acc_clean_gated_rho=ac)
    row.update(D5_spearman_sumq_true=float(spearmanr(sumq[ev], true_cnt[ev])[0]),
               D5_spearman_gapcount_true=float(spearmanr(gapcnt[ev], true_cnt[ev])[0]),
               D5_spearman_sumq_length=float(spearmanr(sumq[ev], n[ev])[0]),
               D5_spearman_gapcount_length=float(spearmanr(gapcnt[ev], n[ev])[0]),
               n_answers_evaluated=int(ev.sum()), n_clean_evaluated=int((~he).sum()))
    return info, row


def verdict(rows, n_invalid):
    get = lambda b: [r for r in rows if r['benchmark'] == b]
    by = {b: get(b) for b in ('processbench', 'crv_arith', 'crv_gsm8k')}
    n3 = lambda rs, ok: sum(ok(r) for r in rs) >= 3
    d1 = all(n3(rs, lambda r: abs(r['D1_rho_hat'] - r['D1_truth_fit_population']) <= .05) for rs in by.values())
    d3 = all(n3(rs, lambda r: r['D3_ece'] <= .05) for rs in by.values())
    d4 = n3(by['processbench'], lambda r: r['D4_closed_clean_at_rho'] >= .30 and r['D4_kept_has_error_at_rho'] >= .85) and \
        all(1 - r['D4_kept_has_error_at_rho'] <= .05 for r in get('prmbench')) and len(get('prmbench')) > 0
    d5 = all(abs(r['D5_spearman_sumq_length']) < abs(r['D5_spearman_gapcount_length']) and
             r['D5_spearman_sumq_true'] > r['D5_spearman_gapcount_true'] for r in rows)
    return dict(D1=bool(d1), D3=bool(d3), D4=bool(d4), D5=bool(d5), n_invalid_cells=n_invalid,
                pass_all=bool(d1 and d3 and d4 and d5 and n_invalid == 0), n_cells=len(rows))


def main():
    for arm in (sys.argv[1:] or ['AUTO', 'TELE16', 'SUB5']):
        rows, fits = [], []
        for bench in BENCHES:
            for prm in PRMS:
                info, row = run_cell(arm, bench, prm); fits.append(info)
                if row:
                    rows.append(row)
                    print(arm, bench, prm, 'K', row['K'], 'rho', round(row['D1_rho_hat'], 3), 'truth', round(row['D1_truth_fit_population'], 3),
                          'ECE', round(row['D3_ece'], 3), 'closed clean@rho', round(row['D4_closed_clean_at_rho'], 3),
                          'kept err@rho', round(row['D4_kept_has_error_at_rho'], 3), 'GAP', round(row['metric_GAP'], 2),
                          'gated rho/cost/.5', round(row['metric_gated_rho'], 2), round(row['metric_gated_cost'], 2),
                          round(row['metric_gated_0.5'], 2), flush=True)
                else:
                    print(arm, bench, prm, 'INVALID K', info['K'], flush=True)
        E.write_csv(HERE / f'E1_RESULTS_{arm}.csv', rows)
        json.dump(fits, open(HERE / f'E1_FITS_{arm}.json', 'w'), indent=1, default=float)
        v = verdict(rows, sum(not f.get('valid', False) for f in fits))
        json.dump(v, open(HERE / f'E1_VERDICT_{arm}.json', 'w'), indent=1); print('VERDICT', arm, v, flush=True)


if __name__ == '__main__':
    main()
