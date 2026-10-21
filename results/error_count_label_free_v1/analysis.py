"""Robustness of the best estimator (post hoc checks, all clearly labelled):
paired source-group bootstrap, length-matched count shuffles, per-category table, nnr1 diagnostic, GAP cap sensitivity.
"""
import csv, json
import numpy as np
from scipy.stats import spearmanr
from common_ec import *  # noqa
import common as pc

bk = load_tele_bank('N16'); off = bk['offsets'].astype(np.int64); ids = bk['idx']; n = np.diff(off); A = len(n)
meta = load_meta(); y, cnt, nonc, cls, cat = labels(ids, off, meta)
P = pc.load_population(); groups = P['group']
F = dict(np.load(OUT / 'FLAGS.npz')); C = dict(np.load(OUT / 'COUNTS.npz'))
Sb = dict(np.load(OUT / 'BOOST_SCORES.npz'))
SCORE = {'BOOST_CHAIN_N16': answer_z(Sb['CHAIN_N16'], off), 'TELE_LSML_CONT': bk['S__LSML_CONT'],
         'TELE_ALL_EQ_ORIENTED': bk['S__ALL_EQ_ORIENTED']}
sm = np.repeat(nonc, n); aid = np.repeat(np.arange(A), n)
gu, ginv = np.unique(groups, return_inverse=True)
rng = np.random.default_rng(20261021); B = 2000
W = np.stack([np.bincount(rng.integers(0, len(gu), len(gu)), minlength=len(gu)) for _ in range(B)])  # (B, G)


def conf_by_group(fl):
    m = sm
    parts = [(~y & ~fl & m), (y & ~fl & m), (y & fl & m), (~y & fl & m)]
    return np.stack([np.bincount(ginv[aid], weights=p.astype(float), minlength=len(gu)) for p in parts], 1)  # (G,4)


def ps_from(c):
    tp, fp, tn, fn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    return 100 * .5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn))


out = {}
rows = []
for arm, s in SCORE.items():
    g = {r: conf_by_group(F[f'{arm}__{r}']) for r in ['E1_GAP', 'Q80', 'K2', 'E3_LEVEL', 'ORACLE']}
    for a_, b_ in [('E1_GAP', 'Q80'), ('E1_GAP', 'K2'), ('E3_LEVEL', 'Q80')]:
        d = ps_from(g[a_].sum(0)) - ps_from(g[b_].sum(0))
        db = ps_from(W @ g[a_]) - ps_from(W @ g[b_])
        rows.append(dict(arm=arm, contrast=f'{a_}-{b_}', diff=d, lo=np.quantile(db, .025), hi=np.quantile(db, .975)))
    # count shuffles: GAP counts permuted among answers with the same n (20 seeds) and across all answers
    k = C[f'{arm}__E1_GAP']; sh_same, sh_all = [], []
    for seed in range(20):
        r = np.random.default_rng(seed); ks = k.copy()
        for nn in np.unique(n):
            ii = np.where(n == nn)[0]; ks[ii] = k[r.permutation(ii)]
        sh_same.append(prmscore(y, topk_flags(s, off, ks), sm))
        sh_all.append(prmscore(y, topk_flags(s, off, k[r.permutation(A)]), sm))
    gap_ps = prmscore(y, F[f'{arm}__E1_GAP'], sm)
    # post hoc cap sensitivity (NOT a new estimator)
    caps = {}
    for nm, fn_ in [('cap_n-1', lambda m: m - 1), ('cap_ceil_n/3', lambda m: max(1, int(np.ceil(m / 3)))),
                    ('cap_4', lambda m: min(4, m - 1))]:
        kk = []
        for a, b in zip(off[:-1], off[1:]):
            z = np.sort(s[a:b])[::-1]; m = b - a
            if m == 1: kk.append(1); continue
            km = max(1, min(fn_(m), m - 1)); kk.append(int(np.argmax(z[:km] - z[1:km + 1]) + 1))
        caps[nm] = prmscore(y, topk_flags(s, off, np.array(kk)), sm)
    kn = k[nonc]
    out[arm] = dict(gap_prmscore=gap_ps, shuffle_same_n_mean=float(np.mean(sh_same)), shuffle_same_n_sd=float(np.std(sh_same)),
                    shuffle_all_mean=float(np.mean(sh_all)), gap_count_dist={int(v): int((kn == v).sum()) for v in np.unique(kn)},
                    true_count_dist={int(v): int((cnt[nonc] == v).sum()) for v in np.unique(cnt[nonc])},
                    spearman_gap_n=float(spearmanr(kn, n[nonc]).correlation),
                    spearman_true_n=float(spearmanr(cnt[nonc], n[nonc]).correlation),
                    spearman_gap_true_within_n=float(np.nanmean([spearmanr(kn[n[nonc] == m], cnt[nonc][n[nonc] == m]).correlation
                                                     for m in np.unique(n[nonc]) if (n[nonc] == m).sum() > 30
                                                     and np.std(kn[n[nonc] == m]) > 0 and np.std(cnt[nonc][n[nonc] == m]) > 0])),
                    exact_count_acc={r: float(np.mean(counts_of(F[f'{arm}__{r}'], off)[nonc] == cnt[nonc]))
                                     for r in ['E1_GAP', 'Q80', 'K2', 'E3_LEVEL']},
                    cap_sensitivity_post_hoc=caps)
    # per category (9 PRMBench classes) for Q80 / GAP / K2 / ORACLE
    pcs = {r: per_class(y, cnt, nonc, off, F[f'{arm}__{r}'], cls) for r in ['Q80', 'E1_GAP', 'K2', 'ORACLE', 'E3_LEVEL']}
    out[arm]['per_class'] = pcs

# nnr1 diagnostic: does the answer-local covariance scale c_a track the true count? (scale-free; labels for evaluation)
X, _, _ = load_tele_channels()
info = json.loads((OUT / 'ESTIMATOR_INFO.json').read_text())
v = np.array(info['TELE_LSML_CONT']['v_pool'])
Xo = X * np.sign([np.corrcoef(X[:, j], bk['S__LSML_CONT'])[0, 1] for j in range(X.shape[1])])
U = np.outer(v, v); om = ~np.eye(len(v), dtype=bool); den = (U[om] ** 2).sum()
ca, tc, tp_, nn_ = [], [], [], []
for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
    if b - a < 4 or not nonc[j]: continue
    Z = Xo[a:b] - Xo[a:b].mean(0); sd = Z.std(0); sd[sd < 1e-12] = np.inf
    Cm = (Z / sd).T @ (Z / sd) / (b - a)
    ca.append((Cm[om] * U[om]).sum() / den); tc.append(cnt[j]); pp = cnt[j] / (b - a); tp_.append(pp * (1 - pp)); nn_.append(b - a)
ca, tc, tp_, nn_ = map(np.array, (ca, tc, tp_, nn_))
out['nnr1_diagnostic_TELE'] = dict(n_answers=len(ca), spearman_c_vs_true_count=float(spearmanr(ca, tc).correlation),
                                   spearman_c_vs_true_p1mp=float(spearmanr(ca, tp_).correlation),
                                   spearman_c_vs_n=float(spearmanr(ca, nn_).correlation))
(OUT / 'ROBUSTNESS.json').write_text(json.dumps(out, indent=1, default=float))
with open(OUT / 'CONTRASTS.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
for arm in SCORE:
    o = out[arm]; print(arm, {k: v for k, v in o.items() if k != 'per_class'})
    for c in o['per_class']['Q80']:
        print('   ', f"{c:22s}", ' '.join(f"{r}={o['per_class'][r][c]['prmscore']:.1f}" for r in o['per_class']),
              f"true={o['per_class']['Q80'][c]['mean_true']:.2f} gapk={o['per_class']['E1_GAP'][c]['mean_pred']:.2f} "
              f"q80k={o['per_class']['Q80'][c]['mean_pred']:.2f} n={o['per_class']['Q80'][c]['n']}")
print(out['nnr1_diagnostic_TELE'])
