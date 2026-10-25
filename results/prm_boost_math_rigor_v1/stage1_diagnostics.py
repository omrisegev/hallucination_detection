"""Stage 1 (plan, 2026-10-24): check the framing's predictions on real cells. LABEL-USING DIAGNOSTICS ONLY.

Nothing here is fitted to labels or feeds a method. Run only after GATE_PREREG.md is committed, so that these
diagnostics cannot shape the gate.

Per cell (PRMBench, Socratic, Hard2Verify x 5 PRMs), on CHAIN_N16's cross-stage views VC = [answer_z(PRM risk),
answer_z(group scores)] and CHAIN's effective weights w (results/prm_boost_math_rigor_v1/common.py):

  P2 check     Gaussian plug-in AUC(w) = Phi(w'D / sqrt(2 w'S w)), with D the class mean difference and S the pooled
               within-class covariance of the within-answer-centered views. Predicted vs actual within-answer AUC
               for the PRM alone, CHAIN, and the label-using Fisher weights S^-1 D (headroom, not a method).
  anchor check within-class correlation of the PRM with each group score after removing the covariate basis
               (P5 assumes ~0 beyond the shared error signal)
  nuisance     label-free: share of the off-diagonal covariance of VC explained by the covariates; second/first
               eigenvalue of the zero-diagonal covariance. Label-using: share of the off-diagonal covariance that is
               within-class (the review's W2 number).
  position     within-class correlation of the PRM and of CHAIN's telemetry part with relative position

    COLLECTION_DATA=<data root> python stage1_diagnostics.py <out_dir>
"""
import csv, json, sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
from spectral_utils.fusion_theory import covariate_basis, within_answer_center, within_answer_residualize  # noqa: E402


def within_class_cov(Xc, y):
    S0 = np.cov(Xc[~y].T); S1 = np.cov(Xc[y].T)
    return ((~y).sum() * S0 + y.sum() * S1) / len(y)


def offdiag_norm(M):
    return float(np.linalg.norm(M - np.diag(np.diag(M))))


def corr_within(a, b, y):
    r = [np.corrcoef(a[m], b[m])[0, 1] for m in (~y, y) if m.sum() > 2]
    n = [m.sum() for m in (~y, y) if m.sum() > 2]
    return float(np.average(r, weights=n))


def cell(bench, prm):
    c, VC, w, fused, info = K.chain_inputs(bench, prm)
    step_mask, ans_mask = K.eval_rows(c, bench)
    use = step_mask & (c.keep if bench != 'prmbench' else np.ones(c.off[-1], bool))
    y = c.err.astype(bool)
    pos, loglen = K.covariates(c)
    Z = covariate_basis(pos, loglen)
    Xc = within_answer_center(VC, c.off)[use]; yu = y[use]
    D = Xc[yu].mean(0) - Xc[~yu].mean(0); S = within_class_cov(Xc, yu)
    pred = lambda v: float(norm.cdf(v @ D / np.sqrt(2 * v @ S @ v)))
    e0 = np.eye(len(w))[0]; wf = np.linalg.solve(S, D)
    a_prm, a_chain, a_fisher = (K.within_auc(c, VC @ v, ans_mask) for v in (e0, w, wf))
    Xr = within_answer_residualize(VC, Z, c.off)[use]
    r_anchor = [corr_within(Xr[:, 0], Xr[:, k], yu) for k in range(1, VC.shape[1])]
    C = np.cov(Xc.T); Cr = np.cov(Xr.T)
    ev = np.sort(np.linalg.eigvalsh(C - np.diag(np.diag(C))))[::-1]
    tele = VC[:, 1:] @ w[1:]
    share = abs(w[0]) * VC[:, 0].std() / (abs(w[0]) * VC[:, 0].std() + tele.std())
    return {'benchmark': bench, 'prm': prm, 'K': info['K'], 'n_steps_eval': int(use.sum()),
            'auc_prm': a_prm, 'auc_chain': a_chain, 'auc_fisher_oracle': a_fisher,
            'pred_prm': pred(e0), 'pred_chain': pred(w), 'pred_fisher_oracle': pred(wf),
            'chain_minus_prm': a_chain - a_prm, 'pred_chain_minus_prm': pred(w) - pred(e0),
            'prm_share_std_units': float(share),
            'anchor_within_corr_mean_abs': float(np.mean(np.abs(r_anchor))),
            'anchor_within_corr_max_abs': float(np.max(np.abs(r_anchor))),
            'nuisance_share_covariates_labelfree': 1 - offdiag_norm(Cr) / offdiag_norm(C),
            'eig2_over_eig1_labelfree': float(ev[1] / ev[0]),
            'within_class_share_of_offdiag': offdiag_norm(S) / offdiag_norm(C),
            'prm_pos_within_corr': corr_within(Xc[:, 0], pos[use] - pos[use].mean(), yu),
            'tele_pos_within_corr': corr_within(within_answer_center(tele[:, None], c.off)[use, 0],
                                                pos[use] - pos[use].mean(), yu)}


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for bench in K.BENCHES:
        for prm in K.E.PRMS:
            rows.append(cell(bench, prm)); r = rows[-1]
            print(bench, prm, 'AUC prm %.3f chain %.3f | pred %.3f %.3f | anchor |r| %.3f' % (
                r['auc_prm'], r['auc_chain'], r['pred_prm'], r['pred_chain'], r['anchor_within_corr_mean_abs']), flush=True)
    with open(out / 'STAGE1_DIAGNOSTICS.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, list(rows[0])); wr.writeheader(); wr.writerows(rows)
    act = np.array([r['chain_minus_prm'] for r in rows]); pr = np.array([r['pred_chain_minus_prm'] for r in rows])
    summ = {'cells': len(rows), 'sign_agreement_pred_vs_actual': int(((act > 0) == (pr > 0)).sum()),
            'corr_pred_actual_delta': float(np.corrcoef(act, pr)[0, 1]),
            'mean_abs_err_auc_chain': float(np.mean([abs(r['pred_chain'] - r['auc_chain']) for r in rows])),
            'mean_abs_err_auc_prm': float(np.mean([abs(r['pred_prm'] - r['auc_prm']) for r in rows]))}
    (out / 'STAGE1_SUMMARY.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
