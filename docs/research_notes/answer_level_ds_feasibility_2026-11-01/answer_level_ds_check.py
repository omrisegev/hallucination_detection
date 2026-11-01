"""Feasibility check (development data, post hoc, not pre-registered): answer-level Dawid-Skene on
raw (not answer-standardized) group scores, learned across answers.

Per cell (D1-D3 bundle): voters = each L-SML group's answer max of its pooled-z score (u_g), residualized
on log(n_steps) across answers (label-free). Two label-free fits:
  DSB: binary Dawid-Skene EM, each voter cut at its pooled median (one vote per group).
  GLC: 2-class Gaussian latent class with diagonal covariance (= continuous DS, conditional independence).
Labels are used only for: true prevalence, true per-voter sens/spec, class-conditional |corr| (CI check).
Usage: python answer_level_ds_check.py <bundle_dir> <out_csv>
"""
import csv, glob, json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture


def ds_binary(V, iters=500):
    # V: answers x K in {0,1}; 1 = vote "error". Init from majority vote.
    q = V.mean(1)
    for _ in range(iters):
        pi = q.mean()
        se = (q[:, None] * V).sum(0) / q.sum()
        sp = ((1 - q)[:, None] * (1 - V)).sum(0) / (1 - q).sum()
        se, sp = np.clip(se, 1e-4, 1 - 1e-4), np.clip(sp, 1e-4, 1 - 1e-4)
        l1 = np.log(pi) + (V * np.log(se) + (1 - V) * np.log(1 - se)).sum(1)
        l0 = np.log(1 - pi) + ((1 - V) * np.log(sp) + V * np.log(1 - sp)).sum(1)
        qn = 1 / (1 + np.exp(l0 - l1))
        if np.max(np.abs(qn - q)) < 1e-9: q = qn; break
        q = qn
    return q, pi, se, sp


def main(bundle_dir, out_csv):
    rows = []
    for f in sorted(glob.glob(os.path.join(bundle_dir, '*.npz'))):
        B = np.load(f, allow_pickle=False)
        keys = sorted(k for k in B.files if k.startswith('ans_group') and k.endswith('_answer_max'))
        if len(keys) < 3: continue
        off = B['offsets']; n = np.diff(off).astype(float)
        X = np.column_stack([B[k] for k in keys]).astype(float)
        ok = np.isfinite(X).all(1)
        ev, y = B['ans_evaluated'] & ok, B['ans_has_err_LABEL']
        # label-free length adjustment across answers
        D = np.column_stack([np.ones(ok.sum()), np.log(n[ok])])
        R = np.full_like(X, np.nan)
        for g in range(X.shape[1]):
            b = np.linalg.lstsq(D, X[ok, g], rcond=None)[0]; R[ok, g] = X[ok, g] - D @ b
        Z = (R[ok] - R[ok].mean(0)) / R[ok].std(0)
        evk = ev[ok]; yk = y[ok][evk].astype(bool)
        true_pi = yk.mean()
        if not (0 < true_pi < 1): continue
        # DIAG: CI check and per-voter AUC
        cc = []
        for cls in (0, 1):
            Zc = Z[evk][yk == cls]
            C = np.corrcoef(Zc.T); cc.append(np.abs(C[np.triu_indices(len(C), 1)]).mean())
        aucs = [roc_auc_score(yk, Z[evk][:, g]) for g in range(Z.shape[1])]
        # DSB
        V = (Z > np.median(Z, 0)).astype(float)
        q, pi_b, se_b, sp_b = ds_binary(V)
        Ve = V[evk]
        se_t = Ve[yk].mean(0); sp_t = 1 - Ve[~yk].mean(0)
        # GLC
        gm = GaussianMixture(2, covariance_type='diag', n_init=10, random_state=0).fit(Z)
        err = int(np.argmax(gm.means_.mean(1)))
        post = gm.predict_proba(Z)[:, err]
        rows.append(dict(
            cell=os.path.basename(f)[:-4], K=Z.shape[1], n_answers=int(ok.sum()), true_prev=true_pi,
            mean_voter_auc_DIAG=np.mean(aucs), ci_corr_clean_DIAG=cc[0], ci_corr_error_DIAG=cc[1],
            dsb_prev=pi_b, dsb_post_auc_DIAG=roc_auc_score(yk, q[evk]),
            dsb_sens_mae_DIAG=np.abs(se_b - se_t).mean(), dsb_spec_mae_DIAG=np.abs(sp_b - sp_t).mean(),
            glc_prev=gm.weights_[err], glc_post_auc_DIAG=roc_auc_score(yk, post[evk]),
            raw_fused_max_auc_DIAG=roc_auc_score(yk, B['ans_raw_fused_answer_max'][ok][evk])))
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
