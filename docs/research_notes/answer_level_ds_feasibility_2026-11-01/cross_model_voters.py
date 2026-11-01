"""Post hoc: answer-level DS with voters from two different models' telemetry (TELE16 on the PRM backbone +
TELE16 standalone base model) plus the AUTO groups (contain the PRM). Same answers required (checked by
offsets and labels). Same length adjustment and fits as answer_level_ds_check.py."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_level_ds_check import ds_binary
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
bd = sys.argv[1]
print('bench,prm,K,true_prev,dsb_prev,glc_prev,cross_model_ci_corr_DIAG,within_model_ci_corr_DIAG,dsb_post_auc_DIAG')
for bench in ('processbench', 'crv_arith', 'crv_gsm8k'):
    for prm in ('qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm'):
        P = [np.load(f'{bd}/{a}_{bench}_{prm}.npz') for a in ('TELE16_prm', 'TELE16_sa', 'AUTO_prm')]
        if not all(np.array_equal(P[0]['offsets'], p['offsets']) and np.array_equal(P[0]['ans_has_err_LABEL'], p['ans_has_err_LABEL']) for p in P):
            print(bench, prm, 'answers differ, skipped'); continue
        cols, src = [], []
        for s, p in enumerate(P):
            for k in sorted(x for x in p.files if x.startswith('ans_group') and x.endswith('_answer_max')):
                cols.append(p[k]); src.append(s)
        X = np.column_stack(cols).astype(float); src = np.array(src)
        n = np.diff(P[0]['offsets']).astype(float); ok = np.isfinite(X).all(1)
        Dm = np.column_stack([np.ones(ok.sum()), np.log(n[ok])])
        R = X[ok] - Dm @ np.linalg.lstsq(Dm, X[ok], rcond=None)[0]
        Z = (R - R.mean(0)) / R.std(0)
        ev = P[0]['ans_evaluated'][ok]; y = P[0]['ans_has_err_LABEL'][ok][ev].astype(bool)
        cm, wm = [], []
        for c in (0, 1):
            C = np.abs(np.corrcoef(Z[ev][y == c].T))
            for i in range(len(src)):
                for j in range(i + 1, len(src)):
                    if {src[i], src[j]} == {0, 1}: cm.append(C[i, j])
                    elif src[i] == src[j]: wm.append(C[i, j])
        V = (Z > np.median(Z, 0)).astype(float); q, pb, _, _ = ds_binary(V)
        gm = GaussianMixture(2, covariance_type='diag', n_init=10, random_state=0).fit(Z)
        e = int(np.argmax(gm.means_.mean(1)))
        print(f'{bench},{prm},{Z.shape[1]},{y.mean():.3f},{pb:.3f},{gm.weights_[e]:.3f},{np.mean(cm):.3f},{np.mean(wm):.3f},{roc_auc_score(y, q[ev]):.3f}')
