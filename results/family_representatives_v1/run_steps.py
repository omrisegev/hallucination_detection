"""Spec section 12 step 2: cumulative T variants, label-free structure metrics only."""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
from metrics_a import structure_metrics, ROOT  # noqa: E402
import transforms as T  # noqa: E402

D = ROOT / 'results/advisor_completion_2026-10-11/inputs'
Zf = np.load(D / 'bank_V_N56.npz'); P = np.load(D / 'population.npz')
off = Zf['offsets']; folds = np.repeat(P['fold'], np.diff(off)); X0 = Zf['V']; n0 = [str(c) for c in Zf['columns']]

X1, n1, log1 = T.drop_copies(X0, n0, off)
variants = {'N56': (X0, n0, []), 'T1_copies': (X1, n1, log1)}
for mode in ('mean', 'pc1', 'pcK', 'pc1res'):
    for thr in (.8, .5):
        Z, nm, lg = T.compress_blocks(X1, n1, off, mode=mode, thr=thr)
        variants[f'T2_{mode}_r{thr}'] = (Z, nm, lg)
Xg, ng, _ = T.rank_gauss(X1, n1, off)
variants['T4_rankgauss_on_T1'] = (Xg, ng, [])
Z, nm, lg = T.compress_blocks(Xg, ng, off, mode='pc1', thr=.8); variants['T4+T2_pc1_r0.8'] = (Z, nm, lg)
COV = T.step_covariates(off, P['token_lengths'])   # observed step position and length; no labels
for base in ('T2_pc1_r0.8', 'T4+T2_pc1_r0.8', 'T2_mean_r0.5'):
    Xb, nb, _ = variants[base]
    Z, nm, lg = T.residualize_nuisance(Xb, nb, off, COV); variants[f'T3_res_on_{base}'] = (Z, nm, lg)

rows = {}
for k, (X, nm, lg) in variants.items():
    t = time.time()
    try:
        res, _ = structure_metrics(X, folds, nm)
    except Exception as e:
        res = dict(m=X.shape[1], FAILED=f'{type(e).__name__}: {e}'[:200])
    res['transform_log'] = lg; res['seconds'] = round(time.time() - t, 1); rows[k] = res
    keys = ['m', 'S1_between_rank1_residual', 'S1g_group_rank1_residual', 'S2_within_rank1_residual', 'S3_effective_groups', 'S4_K', 'S4_groups_lt3',
            'S5_pairs_ge_095', 'S5_eff_rank_ratio', 'S6_channel_sign_violations', 'S6_negative_cross_weights', 'S7_fold_ari', 'S7_fold_weight_spearman']
    print(f'{k:22s}', ' '.join(f'{res.get(q, float("nan")):.3f}' if isinstance(res.get(q), float) else str(res.get(q)) for q in keys), res.get('FAILED', ''), flush=True)
(HERE / 'METRICS_A_STEPS.json').write_text(json.dumps(dict(labels_read=False, columns=keys, variants=rows), indent=1))
