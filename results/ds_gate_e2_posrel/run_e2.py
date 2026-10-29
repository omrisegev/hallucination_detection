"""E2 (PROTOCOL.json): position-dependent reliability. For every arm x cell, the SAME group votes are fitted by
  BASE = the corrected-E1 model (ds_gate_v1b_fixed/ds_gate_b.fit_hds: binned prior, emissions constant in position)
  E2   = ds_gate_e2.fit_e2 (binned prior + emissions per position third),
and both are scored with the corrected-E1 diagnostics and decision rules (run_e1.py, imported read-only), plus
within-answer AUC of the step posterior q and a first-error breakdown by position third. Labels are read only after
both fits.

    python run_e2.py <votes: gap|fuse> [arm ...]     # -> E2_RESULTS_<votes>_<arm>.csv, E2_FITS_..., E2_VERDICT_...
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '3')
import hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
V1B = HERE.parent / 'ds_gate_v1b_fixed'
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(V1B))
import ds_gate_e2 as M  # noqa: E402
import run_e1 as R  # noqa: E402  (read-only reuse: inputs, metrics, verdict)
B, E, A = M.B, R.E, R.A
gap_flags = R.gap_flags

THIRDS = ['early', 'middle', 'late']


def votes_for(kind, arm, bench, prm):
    c, X, prm_risk, score = R.arm_inputs(arm, bench, prm)
    pool = np.repeat(np.diff(c.off), np.diff(c.off)) > 1
    if kind == 'gap':
        V = B.clean_and_vote(E, X, c.off, pool, prm_risk)
    else:
        sys.path.insert(0, str(HERE.parent / 'ds_gate_e1b_fuse'))
        import votes_e1b  # noqa: E402
        V = np.asarray(votes_e1b.fuse_votes(bench, prm, arm))
        assert V.shape[0] == c.off[-1], 'FUSE votes do not align with the cell'
    return c, score, V, pool


def first_flag_hit(f, a, b, g):
    p = np.flatnonzero(f[a:b]); return (int(p[0]) if len(p) else -1) == g


def diagnostics(model, kind, arm, bench, prm, c, score, fit, K):
    off = c.off; n = np.diff(off)
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
    row = dict(model=model, votes=kind, arm=arm, benchmark=bench, prm=prm, K=int(K), D1_rho_hat=rho,
               D1_truth_fit_population=float(has_err.mean()), D1_mean_p_evaluated=float(wv.mean()), D1_truth_evaluated=float(he.mean()),
               D2_mean_q=float(q[stepmask].mean()), D2_true_step_rate=float(c.err[stepmask].mean()), D3_ece=ece,
               within_auc_q=float(np.nanmean(wa)), ll_drops=fit['ll_drops'], label_switch=fit['label_switch_warning'])
    for tname, g in gates.items():
        gv = g[ev]
        row[f'D4_kept_has_error_at_{tname}'] = float((~gv[he]).mean()) if he.any() else float('nan')
        row[f'D4_closed_clean_at_{tname}'] = float(gv[~he].mean()) if (~he).any() else float('nan')
        row[f'D4_closed_error_at_{tname}'] = float(gv[he].mean()) if he.any() else float('nan')
    row['metric_GAP'] = R.metric(c, bench, f_gap, keep_gold)
    for tname, g in gates.items():
        row[f'metric_gated_{tname}'] = R.metric(c, bench, f_gap & ~np.repeat(g, n), keep_gold)
    # first error by position third (evaluated answers with an error): top-1 of q, GAP, GAP gated at rho
    f_rho = f_gap & ~np.repeat(gates['rho'], n)
    hits = {t: [[], [], []] for t in range(3)}
    for j in np.flatnonzero(ev & has_err):
        a, b = off[j], off[j + 1]
        g = int(np.flatnonzero(c.err[a:b] & keep_eval[a:b])[0]); t = int(M.third(np.array([(g + .5) / (b - a)]))[0])
        qq = np.where(keep_eval[a:b], q[a:b], -np.inf)
        hits[t][0].append(int(np.argmax(qq)) == g); hits[t][1].append(first_flag_hit(f_gap, a, b, g)); hits[t][2].append(first_flag_hit(f_rho, a, b, g))
    for t in range(3):
        nm = THIRDS[t]; row[f'n_first_error_{nm}'] = len(hits[t][0])
        for k, lab in enumerate(('top1_q', 'GAP', 'gated_rho')):
            row[f'hit_{lab}_{nm}'] = float(np.mean(hits[t][k])) if hits[t][k] else float('nan')
    row.update(D5_spearman_sumq_true=float(spearmanr(sumq[ev], true_cnt[ev])[0]),
               D5_spearman_gapcount_true=float(spearmanr(gapcnt[ev], true_cnt[ev])[0]),
               D5_spearman_sumq_length=float(spearmanr(sumq[ev], n[ev])[0]),
               D5_spearman_gapcount_length=float(spearmanr(gapcnt[ev], n[ev])[0]),
               n_answers_evaluated=int(ev.sum()), n_clean_evaluated=int((~he).sum()))
    if model == 'E2':
        for k in ('psi', 'phi', 'phi0'):
            arr = np.array(fit['params'][k])
            for t in range(3): row[f'{k}_mean_{THIRDS[t]}'] = float(arr[t].mean())
    return row


def run_cell(kind, arm, bench, prm):
    c, score, V, pool = votes_for(kind, arm, bench, prm)
    G, K, cg = B.group_votes(A, V, pool)
    info = dict(votes=kind, arm=arm, benchmark=bench, prm=prm, K=int(K), groups=cg.tolist())
    if K < 3:
        info['valid'] = False; return info, []
    pos = B.relative_position(c.off); Dm = B.bin_design(pos); t = M.third(pos)
    base = B.fit_hds(G, c.off, Dm); e2 = M.fit_e2(G, c.off, Dm, t)
    info.update(valid=True, base=dict(params=base['params'], loglik=base['loglik'], ll_drops=base['ll_drops']),
                e2=dict(params=e2['params'], loglik=e2['loglik'], ll_drops=e2['ll_drops'],
                        label_switch_by_third=e2['label_switch_by_third'], degenerate=e2['degenerate']),
                n_extra_params=int(6 * K), loglik_gain=float(e2['loglik'] - base['loglik']))
    rows = [diagnostics('BASE', kind, arm, bench, prm, c, score, base, K), diagnostics('E2', kind, arm, bench, prm, c, score, e2, K)]
    return info, rows


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
    kind = sys.argv[1]; arms = sys.argv[2:] or ['AUTO', 'TELE16', 'SUB5']
    prov = {f: sha(V1B / f) for f in ('ds_gate_b.py', 'run_e1.py')}
    for arm in arms:
        rows, fits = [], []
        for bench in R.BENCHES:
            for prm in R.PRMS:
                info, rr = run_cell(kind, arm, bench, prm); fits.append(info); rows += rr
                if rr:
                    b, e = rr
                    print(kind, arm, bench, prm, 'K', b['K'], 'truth', round(b['D1_truth_fit_population'], 3),
                          '| BASE rho', round(b['D1_rho_hat'], 3), 'ECE', round(b['D3_ece'], 3), 'wAUCq', round(b['within_auc_q'], 4),
                          '| E2 rho', round(e['D1_rho_hat'], 3), 'ECE', round(e['D3_ece'], 3), 'wAUCq', round(e['within_auc_q'], 4),
                          'closed clean/err@rho', round(e['D4_closed_clean_at_rho'], 3), round(e['D4_closed_error_at_rho'], 3),
                          'GAP', round(e['metric_GAP'], 2), 'gated rho/cost', round(e['metric_gated_rho'], 2), round(e['metric_gated_cost'], 2),
                          'llgain', round(info['loglik_gain'], 1), flush=True)
                else:
                    print(kind, arm, bench, prm, 'INVALID K', info['K'], flush=True)
        E.write_csv(HERE / f'E2_RESULTS_{kind}_{arm}.csv', rows)
        json.dump(dict(provenance=prov, fits=fits), open(HERE / f'E2_FITS_{kind}_{arm}.json', 'w'), indent=1, default=float)
        n_inv = sum(not f.get('valid', False) for f in fits)
        v = {m: R.verdict([r for r in rows if r['model'] == m], n_inv) for m in ('BASE', 'E2')}
        json.dump(v, open(HERE / f'E2_VERDICT_{kind}_{arm}.json', 'w'), indent=1); print('VERDICT', kind, arm, v, flush=True)


if __name__ == '__main__':
    main()
