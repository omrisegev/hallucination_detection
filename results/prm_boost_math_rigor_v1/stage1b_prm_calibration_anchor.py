"""Stage 1b (2026-10-24, after the Step 469 negative result): can the PRM's own probability serve as the anchor?
LABEL-USING DIAGNOSTIC ONLY. No method is fitted or selected here.

Idea: if the PRM is calibrated, P(error | r) ~= 1 - r, so regressing each telemetry group score on p = 1 - r
(both centered within answers) estimates that group's class separation Delta_k without labels: E[x | p] = mu + Delta p.
Shared PRM-telemetry noise biases the slope by Cov(noise_k, p) / Var(p).

Per cell (PRMBench, Socratic, Hard2Verify x 5 PRMs):
  calibration   mean(1 - r) vs the actual error rate; ECE over 10 equal-width bins of 1 - r; within-answer AUC of p
  slopes        slope_k of group score k on p (within-answer centered), vs the true separation Delta_k
                (class mean difference, within-answer centered): Spearman rank agreement across groups and the
                median ratio slope/Delta (1 = unbiased under the calibration assumption)
  PRM itself    the slope of the PRM's own answer-z score on p vs its true separation (same ratio)

    COLLECTION_DATA=<data root> python stage1b_prm_calibration_anchor.py <out_dir>
"""
import csv, json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
from spectral_utils.fusion_theory import within_answer_center  # noqa: E402


def prob_arm(c):
    return 'PRM_Q80_PROB' if c.bench == 'prmbench' else 'PRMPROB'


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1); e = 0.
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def cell(bench, prm):
    c, VC, w, fused, info = K.chain_inputs(bench, prm)
    step_mask, ans_mask = K.eval_rows(c, bench)
    use = step_mask & (c.keep if bench != 'prmbench' else np.ones(c.off[-1], bool))
    y = c.err.astype(float); p = np.clip(c.arms[prob_arm(c)], 0, 1)
    Xc = within_answer_center(np.column_stack([p, VC]), c.off)[use]; yu = y[use]
    pc, V = Xc[:, 0], Xc[:, 1:]
    slope = (V * pc[:, None]).mean(0) / (pc ** 2).mean()
    delta = V[yu == 1].mean(0) - V[yu == 0].mean(0)
    ratio = slope / delta
    tg = slice(1, None)                                  # telemetry groups (column 0 is the PRM's own z-score)
    rho = spearmanr(slope[tg], delta[tg]).correlation if V.shape[1] > 3 else np.nan
    return {'benchmark': bench, 'prm': prm, 'K': info['K'],
            'mean_p': float(p[use].mean()), 'error_rate': float(yu.mean()), 'ece': ece(p[use], yu[...]),
            'auc_p_within': K.within_auc(c, p, ans_mask),
            'prm_slope_over_delta': float(ratio[0]),
            'tele_spearman_slope_vs_delta': float(rho),
            'tele_median_slope_over_delta': float(np.median(ratio[tg])),
            'tele_sign_agree': int((np.sign(slope[tg]) == np.sign(delta[tg])).sum()), 'n_groups': int(V.shape[1] - 1),
            'slopes': np.round(slope, 4).tolist(), 'deltas': np.round(delta, 4).tolist()}


def main(out):
    out.mkdir(parents=True, exist_ok=True); rows = []
    for bench in K.BENCHES:
        for prm in K.E.PRMS:
            r = cell(bench, prm); rows.append(r)
            print('%-11s %-13s mean p %.3f vs err %.3f  ECE %.3f | PRM slope/Delta %.2f | tele rank rho %+.2f, median slope/Delta %.2f, signs %d/%d' % (
                bench, prm, r['mean_p'], r['error_rate'], r['ece'], r['prm_slope_over_delta'],
                r['tele_spearman_slope_vs_delta'], r['tele_median_slope_over_delta'], r['tele_sign_agree'], r['n_groups']), flush=True)
    with open(out / 'PRM_ANCHOR_DIAGNOSTICS.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, list(rows[0])); wr.writeheader(); wr.writerows(rows)
    summ = {'cells': len(rows),
            'median_tele_spearman': float(np.nanmedian([r['tele_spearman_slope_vs_delta'] for r in rows])),
            'tele_sign_agree_total': f"{sum(r['tele_sign_agree'] for r in rows)}/{sum(r['n_groups'] for r in rows)}",
            'median_ece': float(np.median([r['ece'] for r in rows]))}
    (out / 'PRM_ANCHOR_SUMMARY.json').write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
