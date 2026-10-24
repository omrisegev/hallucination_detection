"""Stage 1c (2026-10-24): does the PRM's probability carry the per-answer error COUNT? Label-using diagnostic only.

Per answer: S = sum over scored steps of (1 - reward), the expected error count if the PRM were calibrated. Compared
with the true count (error steps inside the evaluated pool):
  - Spearman(S, true count); Spearman(S, number of steps) and Spearman(true count, number of steps) for the length
    confound;
  - MAE of round(S), of the PRM's native count (reward < .5), of GAP on the PRM probability, and of constant k=2;
  - exact-count rate of round(S);
  - clean answers (true count 0): share with round(S) = 0 and with native count 0, and the answer-level AUC of S for
    clean vs erroneous.
PRMBench includes its 758 control (correct) answers here, so clean-answer detection can be measured; the
official PRMScore pool excludes them. Hard2Verify uses the 177-answer common set.

    COLLECTION_DATA=<data root> python stage1c_prm_count.py <out_dir>
"""
import csv, json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

E = K.E


def auc(y, s):
    y = np.asarray(y, bool)
    return E.auc(y, s) if 0 < y.sum() < len(y) else float('nan')


def cell(bench, prm):
    c = E.load(bench, prm, uids_keep=K.h2v_common_uids()) if bench == 'hard2verify' else E.load(bench, prm)
    p = np.clip(c.arms['PRM_Q80_PROB' if bench == 'prmbench' else 'PRMPROB'], 0, 1)
    keep = np.ones(c.off[-1], bool) if bench == 'prmbench' else c.keep
    st, en = c.off[:-1], c.off[1:]
    S = np.array([p[a:b][keep[a:b]].sum() for a, b in zip(st, en)])
    T = np.array([(c.err[a:b] & keep[a:b]).sum() for a, b in zip(st, en)])
    N = np.array([keep[a:b].sum() for a, b in zip(st, en)])
    nat = np.array([(p[a:b][keep[a:b]] > .5).sum() for a, b in zip(st, en)])
    gap = E.gap_counts(E.answer_z(p, c.off), c.off)
    ok = N > 0; S, T, N, nat, gap = S[ok], T[ok], N[ok], nat[ok], gap[ok]
    clean = T == 0; rS = np.rint(S)
    return {'benchmark': bench, 'prm': prm, 'answers': int(ok.sum()), 'clean_answers': int(clean.sum()),
            'mean_true_count': float(T.mean()), 'mean_S': float(S.mean()),
            'spearman_S_count': float(spearmanr(S, T).correlation),
            'spearman_S_steps': float(spearmanr(S, N).correlation),
            'spearman_count_steps': float(spearmanr(T, N).correlation),
            'mae_roundS': float(np.abs(rS - T).mean()), 'mae_native': float(np.abs(nat - T).mean()),
            'mae_gap': float(np.abs(gap - T).mean()), 'mae_k2': float(np.abs(2 - T).mean()),
            'exact_roundS': float((rS == T).mean()), 'exact_gap': float((gap == T).mean()),
            'clean_zero_roundS': float((rS[clean] == 0).mean()) if clean.any() else float('nan'),
            'clean_zero_native': float((nat[clean] == 0).mean()) if clean.any() else float('nan'),
            'erroneous_zero_roundS': float((rS[~clean] == 0).mean()),
            'auc_S_clean_vs_erroneous': auc(~clean, S)}


def main(out):
    out.mkdir(parents=True, exist_ok=True); rows = []
    for bench in K.BENCHES:
        for prm in E.PRMS:
            r = cell(bench, prm); rows.append(r)
            print('%-11s %-13s ans %4d clean %4d | count %.2f vs S %.2f | rho(S,count) %+.2f rho(S,len) %+.2f rho(count,len) %+.2f | MAE S %.2f nat %.2f gap %.2f k2 %.2f | exact S %.2f gap %.2f | clean->0 S %.2f nat %.2f | err->0 %.2f | AUC clean %.2f' % (
                bench, prm, r['answers'], r['clean_answers'], r['mean_true_count'], r['mean_S'], r['spearman_S_count'],
                r['spearman_S_steps'], r['spearman_count_steps'], r['mae_roundS'], r['mae_native'], r['mae_gap'], r['mae_k2'],
                r['exact_roundS'], r['exact_gap'], r['clean_zero_roundS'], r['clean_zero_native'], r['erroneous_zero_roundS'],
                r['auc_S_clean_vs_erroneous']), flush=True)
    with open(out / 'PRM_COUNT.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, list(rows[0])); wr.writeheader(); wr.writerows(rows)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
