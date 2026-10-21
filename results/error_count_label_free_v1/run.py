"""Label-free per-answer error-count estimators (DECLARATION.md) + headroom references.
Stage 1 computes every count WITHOUT labels and saves COUNTS.npz; stage 2 reads labels and evaluates.
"""
import csv, json
import numpy as np
from scipy.stats import t as student_t, poisson
from common_ec import *  # noqa
import common as pc
from spectral_utils.fusion_utils import rank1_offdiag_fit

# ---------------- stage 1: label-free inputs ----------------
meta_free = None  # labels are not touched in stage 1
bk = load_tele_bank('N16'); off = bk['offsets'].astype(np.int64); ids = bk['idx']
n = np.diff(off); A = len(n); nbar = float(n.mean())
Xt, off2, ids2 = load_tele_channels()
assert np.array_equal(off2, off) and np.array_equal(ids2, ids)
P = pc.load_population()          # loads y too (label rebuild assert inside); y is NOT used in stage 1
assert np.array_equal(P['off'], off) and np.array_equal(P['ids'], ids)
Sb = dict(np.load(OUT / 'BOOST_SCORES.npz'))
Xb = np.column_stack([P['X'][:, [P['names'].index(c) for c in N16]], answer_z(P['prm'], off)])

ARMS = {
    'BOOST_CHAIN_N16': dict(score=answer_z(Sb['CHAIN_N16'], off), X=Xb, level=P['prm']),
    'TELE_LSML_CONT': dict(score=bk['S__LSML_CONT'], X=Xt, level=bk['G']),
    'TELE_ALL_EQ_ORIENTED': dict(score=bk['S__ALL_EQ_ORIENTED'], X=Xt, level=bk['G']),
}


def orient(X, s):
    sg = np.sign([np.corrcoef(X[:, j], s)[0, 1] for j in range(X.shape[1])]); sg[sg == 0] = 1
    return X * sg


def tensor_prevalence(X):
    N, m = X.shape
    Xc = X - X.mean(0)
    R = Xc.T @ Xc / N
    v = rank1_offdiag_fit(R)
    num = den = 0.
    for i in range(m):
        for j in range(i + 1, m):
            xij = Xc[:, i] * Xc[:, j]
            for k in range(j + 1, m):
                T = float(np.mean(xij * Xc[:, k])); w = v[i] * v[j] * v[k]
                num += T * w; den += w * w
    g = num / den
    return float((1 - g / np.sqrt(g * g + 4)) / 2), float(g), v, R


def gap_count(z):
    m = len(z)
    if m == 1:
        return 1
    s = np.sort(z)[::-1]; kmax = min(int(np.ceil(m / 2)), m - 1)
    return int(np.argmax(s[:kmax] - s[1:kmax + 1]) + 1)


def esd_count(x, alpha=.05):
    w = np.asarray(x, float).copy(); nn = len(w); last = 0
    for i in range(1, nn // 2 + 1):
        mm = len(w)                     # = n - i + 1
        if mm - 1 < 3:
            break
        sd = w.std(ddof=1)
        if sd <= 1e-12:
            break
        j = int(np.argmax(w)); Ri = (w[j] - w.mean()) / sd
        tt = student_t.ppf(1 - alpha / mm, mm - 2)
        lam = (mm - 1) * tt / np.sqrt((mm - 2 + tt * tt) * mm)
        if Ri > lam:
            last = i
        w = np.delete(w, j)
    return last


COUNTS, FLAGS, INFO = {}, {}, {}
for arm, d in ARMS.items():
    s, L = d['score'], d['level']
    Xo = orient(d['X'], s)
    p_hat, gamma, v, R = tensor_prevalence(Xo)
    INFO[arm] = dict(p_hat=p_hat, gamma=gamma, v_pool=v.tolist())
    c = {}
    c['E1_GAP'] = np.array([gap_count(s[a:b]) for a, b in zip(off[:-1], off[1:])])
    c['E2_ESD'] = np.array([esd_count(s[a:b]) for a, b in zip(off[:-1], off[1:])])
    c['E3_LEVEL'] = counts_of(L > np.quantile(L, .8), off)
    c['E5_TCONST'] = np.full(A, max(1, int(round(p_hat * nbar))))
    U = np.outer(v, v); offm = ~np.eye(len(v), dtype=bool); den = float((U[offm] ** 2).sum())
    k6 = np.ones(A, int); ca = np.full(A, np.nan)
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        if b - a < 4:
            continue
        Z = Xo[a:b] - Xo[a:b].mean(0); sd = Z.std(0); sd[sd < 1e-12] = np.inf
        C = (Z / sd).T @ (Z / sd) / (b - a)
        ca[j] = float((C[offm] * U[offm]).sum() / den)
        q = float(np.clip(ca[j] * p_hat * (1 - p_hat), 0, .25))
        k6[j] = int(round((b - a) * (1 - np.sqrt(1 - 4 * q)) / 2))
    c['E6_NNR1'] = k6; INFO[arm]['nnr1_c_quantiles'] = np.nanquantile(ca, [.1, .25, .5, .75, .9]).tolist()
    if arm.startswith('BOOST'):
        c['E7_PRMNAT'] = counts_of(P['prm'] >= 0, off)
    e = np.add.reduceat(L, off[:-1]) / n
    u = (np.argsort(np.argsort(e, kind='stable'), kind='stable') + .5) / A
    c['E8_ANSPOIS'] = np.maximum(1, poisson.ppf(u, p_hat * nbar).astype(int))
    COUNTS[arm] = c
    f = {}
    for name, k in c.items():
        f[name] = topk_flags(s, off, k)
    f['E4_TFRAC'] = s > np.quantile(s, 1 - p_hat)
    # label-free reference rules
    f['Q80'] = s > np.quantile(s, .8)
    for q in (.70, .75, .85, .90):
        f[f'Q{int(q * 100)}'] = s > np.quantile(s, q)
    for kk in (1, 2, 3):
        f[f'K{kk}'] = topk_flags(s, off, kk)
    if arm.startswith('BOOST'):
        f['PRM_NATIVE_own_ranking'] = P['prm'] >= 0
    FLAGS[arm] = f
np.savez_compressed(OUT / 'COUNTS.npz', **{f'{a}__{k}': v for a, c in COUNTS.items() for k, v in c.items()})
(OUT / 'ESTIMATOR_INFO.json').write_text(json.dumps(INFO, indent=1))
print({a: (round(i['p_hat'], 4), round(i['gamma'], 3), [round(x, 3) for x in i['nnr1_c_quantiles']]) for a, i in INFO.items()})

# ---------------- stage 2: evaluation (labels read here for the first time) ----------------
meta = load_meta()
y, cnt, nonc, cls, cat = labels(ids, off, meta)
groups = P['group']
rows = []
for arm, f in FLAGS.items():
    s = ARMS[arm]['score']
    f['ORACLE'] = topk_flags(s if not arm.startswith('BOOST') else ARMS[arm]['score'], off, cnt)
    # label-tuned references: best global fraction, best constant k
    best = max(((q, prmscore(y, s > np.quantile(s, q), np.repeat(nonc, n))) for q in np.arange(.5, .96, .01)),
               key=lambda t: t[1])
    f[f'LABEL_BEST_FRAC_q{best[0]:.2f}'] = s > np.quantile(s, best[0])
    bk_ = max(((kk, prmscore(y, topk_flags(s, off, kk), np.repeat(nonc, n))) for kk in range(1, 7)), key=lambda t: t[1])
    f[f'LABEL_BEST_CONST_k{bk_[0]}'] = topk_flags(s, off, bk_[0])
    for rule, fl in f.items():
        r = dict(arm=arm, rule=rule, **evaluate(y, cnt, nonc, off, fl))
        r['access'] = ('LABEL (ceiling/reference)' if rule == 'ORACLE' or rule.startswith('LABEL_') else
                       'label-free estimator' if rule.startswith('E') else
                       'benchmark prior (k=2 known)' if rule.startswith('K') else
                       'PRM native (supervised PRM)' if rule.startswith('PRM_') else 'label-free reference')
        rows.append(r)
with open(OUT / 'RESULTS.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
np.savez_compressed(OUT / 'FLAGS.npz', **{f'{a}__{k}': v for a, ff in FLAGS.items() for k, v in ff.items()})
for r in rows:
    print(f"{r['arm']:22s} {r['rule']:26s} PS={r['prmscore']:.2f} MAE={r['count_mae']:.3f} mean_k={r['mean_pred']:.2f} "
          f"share={r['flag_share']:.3f} zacc={r['zero_acc']:.3f} rho={r['spearman_count']:.3f}")
