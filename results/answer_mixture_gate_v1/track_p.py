"""Track P: the PRM's raw step risk logit(1 - reward), no telemetry, 12 cells (PRMBench, Hard2Verify common-177,
ProcessBench x qwen_prm_7b, qwen_prm_72b, pure_prm, rlhflow_prm). Pre-registration:
docs/experiments/ANSWER_MIXTURE_GATE_PREREG_2026-10-29.md.

Phase 1 (no labels): step risk -> answer score (max; secondary top-2 mean) -> pooled OLS residual on
[log n steps, log answer tokens where available] -> 2-component GMM (primary raw residual; secondary Yeo-Johnson).
Phase 2: labels are read only after every fit of the cell is finished, and only to score.

Covariates per benchmark (the pre-registered "log answer length in tokens where available"):
  PRMBench: log n steps + log answer tokens (sum of expm1(log_step_tokens), the stored log1p step token counts).
  Hard2Verify: log n steps only (no token counts in the score files).
  ProcessBench: log n steps; plus log answer tokens for qwen_prm_7b only, from the raw telemetry spans
    (trackT/PB_QWEN_ANSWER_TOKENS.npz). The other three ProcessBench PRMs have no token counts locally.
    python results/answer_mixture_gate_v1/track_p.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'results/prm_boost_collection_v1'))
import mixture as MX  # noqa: E402
import evaluate_collection as E  # noqa: E402
from collection_parity import h2v_common_uids  # noqa: E402

PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
PB_NAME = {'qwen_prm_7b': 'qwen_prm'}
PB_SCORES = E.DATA / 'processbench_scores'
PB_GOLD = ROOT / 'results/prm_boost_processbench_v1/local/processbench_gold.json'
PB_TOK = HERE / 'trackT/PB_QWEN_ANSWER_TOKENS.npz'


def logit_risk(reward):
    r = np.clip(np.asarray(reward, float), 1e-6, 1 - 1e-6)
    return np.log((1 - r) / r)


def check_raw(step, off, what):
    """A raw score must keep each answer's level: per-answer means must vary across answers."""
    m = np.add.reduceat(step, off[:-1]) / np.diff(off)
    sd = float(m.std())
    if not sd > 1e-3:
        raise SystemExit(f'{what}: per-answer mean sd {sd}; this score is answer-standardized, not raw')
    return sd


# ------------------------------------------------------------------------------------------- phase 1 (label-free)
def inputs(bench, prm):
    """(ids, off, step risk, covariates dict, label loader). The label loader is called only in phase 2."""
    if bench == 'prmbench':
        c = E.load('prmbench', prm)
        step = logit_risk(c.extra['reward'])
        tok = np.add.reduceat(np.expm1(c.extra['log_step_tokens']), c.off[:-1])
        assert (tok > 0).all()
        cov = {'log_n_steps': np.log(np.diff(c.off)), 'log_answer_tokens': np.log(tok)}
        lab = lambda: np.add.reduceat(c.err.astype(int), c.off[:-1]) > 0
        return c.ids, c.off, step, cov, lab
    if bench == 'hard2verify':
        c = E.load('hard2verify', prm, uids_keep=h2v_common_uids())
        assert len(c.ids) == 177 and len(c.empty['correct']) == 0
        step = c.arms['PRM'].copy()          # raw logit(1 - reward), checked against PRMPROB
        cov = {'log_n_steps': np.log(np.diff(c.off))}
        lab = lambda: np.add.reduceat((c.err & c.keep).astype(int), c.off[:-1]) > 0
        return c.ids, c.off, step, cov, lab
    if bench == 'processbench':
        S = np.load(PB_SCORES / f'SCORES_{PB_NAME.get(prm, prm)}.npz')
        uids = S['uids'].astype(str); ne = S['nonempty'].astype(bool); lens = S['nonempty_lengths'].astype(int)
        nne = np.add.reduceat(ne.astype(int), np.r_[0, np.cumsum(lens)][:-1])
        assert len(S['score__PRM']) == ne.sum() and (nne > 0).all()
        off = np.r_[0, np.cumsum(nne)].astype(np.int64)
        step = S['score__PRM'].astype(float)                       # risk on the non-empty steps
        cov = {'log_n_steps': np.log(nne)}
        if prm == 'qwen_prm_7b':
            T = np.load(PB_TOK); pos = {u: j for j, u in enumerate(T['uids'].astype(str))}
            j = np.array([pos[u] for u in uids]); assert np.array_equal(T['n_nonempty'][j], nne)
            cov['log_answer_tokens'] = np.log(T['answer_tokens'][j].astype(float))

        def lab():
            G = {g['uid']: g for g in json.loads(PB_GOLD.read_text())}
            return np.array([G[u]['label'] != -1 for u in uids])   # ProcessBench: label = 0-based first error, -1 clean
        return uids, off, step, cov, lab
    raise ValueError(bench)


def fit_cell(bench, prm):
    ids, off, step, cov, lab = inputs(bench, prm)
    sd = check_raw(step, off, f'{bench}/{prm}')
    out = {'n_answers': len(ids), 'per_answer_mean_sd': sd, 'covariates': list(cov), 'variants': {}}
    for sname, sfun in (('max', MX.answer_max), ('top2', MX.answer_top2)):
        s_a = sfun(step, off)
        res, beta = MX.residualize(s_a, list(cov.values()))
        for model in ('raw', 'yeo-johnson'):
            f = MX.fit_mixture(res, None if model == 'raw' else 'yeo-johnson')
            out['variants'][(sname, model)] = {'s_a': s_a, 'res': res, 'beta': beta.tolist(), 'fit': f}
    return out, lab


def main():
    rows, fits_json = [], {}
    for bench in ('prmbench', 'hard2verify', 'processbench'):
        for prm in PRMS:
            out, lab = fit_cell(bench, prm)                       # all fits of the cell finished here
            y = lab()                                             # phase 2: labels only score
            for (sname, model), v in out['variants'].items():
                s = MX.score(v['fit'], y, v['s_a'])
                row = dict(track='P', bench=bench, prm=prm, score='PRM', answer_score=sname, model=model,
                           primary=(sname == 'max' and model == 'raw'), covariates='+'.join(out['covariates']),
                           per_answer_mean_sd=out['per_answer_mean_sd'],
                           **{k: s[k] for k in ('n', 'pi_true', 'pi_hat', 'abs_err', 'bic1', 'bic2', 'bic2_minus_bic1',
                                                'bic_prefers_2', 'converged', 'clean_closed', 'err_kept',
                                                'share_called_clean', 'gate_useful', 'auroc_s_a')},
                           auroc_residual=MX.auc(y, v['res']))
                rows.append(row)
                fits_json[f'{bench}|{prm}|{sname}|{model}'] = {
                    'beta': v['beta'], 'means': s['means'], 'sds': s['sds'], 'weights': s['weights'],
                    'yj_lambda': s['yj_lambda'], 'pi_hat': s['pi_hat'], 'pi_true': s['pi_true']}
                if row['primary']:
                    print(f"{bench:12s} {prm:12s} pi={s['pi_true']:.3f} pi_hat={s['pi_hat']:.3f} |d|={s['abs_err']:.3f} "
                          f"dBIC={s['bic2_minus_bic1']:.1f} clean_closed={s['clean_closed']:.3f} err_kept={s['err_kept']:.3f} "
                          f"AUROC={s['auroc_s_a']:.3f}", flush=True)
    with open(HERE / 'TRACK_P.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (HERE / 'TRACK_P_FITS.json').write_text(json.dumps(fits_json, indent=1))
    prim = [r for r in rows if r['primary']]
    ok = {b: sum(r['abs_err'] <= .05 for r in prim if r['bench'] == b) for b in ('processbench', 'hard2verify', 'prmbench')}
    verdict = {'rule': '|pi_hat - pi| <= .05 in >= 3 of 4 PRMs on ProcessBench AND on Hard2Verify (primary)',
               'cells_within_05': ok, 'pass': bool(ok['processbench'] >= 3 and ok['hard2verify'] >= 3),
               'gate_useful_cells_primary': [f"{r['bench']}|{r['prm']}" for r in prim if r['gate_useful']]}
    (HERE / 'TRACK_P_VERDICT.json').write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict))


if __name__ == '__main__':
    main()
