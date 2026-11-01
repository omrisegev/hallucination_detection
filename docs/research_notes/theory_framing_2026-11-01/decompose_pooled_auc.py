"""Check T2 (pooled-AUC decomposition) and T1 (answer-level information after answer-z) on the D1-D3 bundle.

Usage: python decompose_pooled_auc.py <bundle_dir> <out_csv>
The bundle is results/raw_null_gate_diag_v1/bundle/ on branch claude/raw-null-gate-2026-10-30 (commit f446884e).
Labels are used only for these diagnostics. No fitted quantity is changed.

Identity (exact): over all (error, correct) step pairs on the labelled set,
    pooled_auc = pi_w * within_auc_pairs + (1 - pi_w) * between_auc,
    pi_w = sum_j e_j c_j / (E * C)  (share of pairs inside one answer).
"""
import csv, glob, os, sys
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def pair_auc(se, sc):
    if len(se) == 0 or len(sc) == 0:
        return np.nan, 0
    r = rankdata(np.concatenate([se, sc]))
    u = r[:len(se)].sum() - len(se) * (len(se) + 1) / 2
    return u / (len(se) * len(sc)), len(se) * len(sc)


def main(bundle_dir, out_csv):
    rows = []
    for f in sorted(glob.glob(os.path.join(bundle_dir, '*.npz'))):
        B = np.load(f, allow_pickle=False)
        off, m, y = B['offsets'], B['step_mask'], B['step_err_LABEL']
        A = len(off) - 1
        ev, ya = B['ans_evaluated'], B['ans_has_err_LABEL']
        for key in ['step_raw_fused', 'step_z_fused']:
            s = B[key]
            ok = m & np.isfinite(s)
            pooled, P = pair_auc(s[ok & y], s[ok & ~y])
            wsum = wn = 0.0
            per_answer = []
            for a in range(A):
                sl = slice(off[a], off[a + 1])
                o = ok[sl]; yy = y[sl][o]; ss = s[sl][o]
                if yy.any() and (~yy).any():
                    au, p = pair_auc(ss[yy], ss[~yy])
                    wsum += au * p; wn += p; per_answer.append(au)
            amax = np.array([np.nanmax(s[off[a]:off[a + 1]]) for a in range(A)])
            rows.append(dict(
                cell=os.path.basename(f)[:-4], score=key, n_answers=A,
                n_err=int((ok & y).sum()), n_cor=int((ok & ~y).sum()),
                pi_w=wn / P, pooled_auc=pooled,
                within_auc_pairs=wsum / wn if wn else np.nan,
                within_auc_ans_mean=np.mean(per_answer) if per_answer else np.nan,
                between_auc=(pooled * P - wsum) / (P - wn),
                ans_prev=ya[ev].mean(),
                ans_auc_answer_max=roc_auc_score(ya[ev], amax[ev]) if 0 < ya[ev].mean() < 1 else np.nan))
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
