"""Class-conditional correlation diagnostic for the PRMBench telemetry channels (label-using DIAGNOSTIC, not a method).

Question (motivated by Balasubramanian et al., arXiv 2601.22336, Ising-model label aggregation): is the correlation
structure among the 16 backbone telemetry channels (+ the PRM reward) different on erroneous vs correct steps, and if
so does a quadratic rule (QDA / LR with pairwise products) find detection signal beyond a linear rule (LDA / LR)?
Nothing here feeds any fusion weights.

Part A  answer-z features (each channel z-scored within its answer, as L-SML does), non-control answers with >=3
        steps. R_err, R_ok = within-class correlation matrices of the answer-z values.
          frob      = ||offdiag(R_err - R_ok)||_F (both triangles)
          mean_off  = mean off-diagonal r per class, raw channel orientation (as requested)
          mean_abs  = mean off-diagonal |r| per class
          mean_ori  = mean off-diagonal r after orienting every view toward "error" with a LABEL-FREE anchor
                      (sign of its pooled answer-z correlation with -reward; the reward itself is flipped)
        Null: labels permuted WITHIN answer (error count preserved), 1000 draws. CI: cluster bootstrap over source
        groups, 1000 draws. Robustness: the same inside relative-position terciles.
        EXTRA (labelled, added after a design check): the answer-z standardization plus a pure mean shift between
        classes already produces R_err != R_ok, because the class means of the answer-z values vary across answers
        with the answer's error fraction. Two controls:
          (i)  class-centred statistics: subtract each answer x class mean before the correlation (removes every
               per-answer mean shift; classes with a single step in an answer are dropped from that class),
               with its own within-answer permutation null;
          (ii) an equal-covariance Gaussian simulation with the observed answer structure, labels and per-view
               mean shift, pushed through the same answer-z (50 replicates): what frob a pure mean shift produces.
Part B  supervised headroom on answer-z N17: LDA vs QDA (reg_param from {0,.01,.1} by inner 3-fold GroupKFold on
        the training folds, by inner within-answer AUC), LR linear vs LR linear+squares+pairwise products; 5-fold
        GroupKFold by source group. Reference: frozen label-free L-SML score CHAIN_N16 (higher = error, the
        evaluate_collection.auc(err, score) convention) and -reward, on the same test answers.
        EXTRA: LR on linear + squares only (no cross products), to separate marginal nonlinearity from dependence.

    .venv/bin/python results/telemetry_class_correlation_v1/class_correlation.py
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import csv, json, sys, time, warnings
from itertools import combinations
from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'prm_boost_collection_v1'))
import evaluate_collection as E  # noqa: E402

warnings.filterwarnings('ignore')
B_PERM, B_BOOT, N_SIM = 1000, 1000, 50
SEED = 20261025
PRMS = E.PRMS
CHDIR = E.DATA / 'prmbench'
T0 = time.time()


# ------------------------------------------------------------------------------------------------ data
def load_cell(prm):
    c = E.load_prmbench(prm)
    Z = np.load(CHDIR / f'ABL5_CHANNELS_{prm}.npz')
    assert np.array_equal(Z['row_ids'].astype(str), c.ids), prm
    assert np.array_equal(Z['offsets'].astype(np.int64), c.off), prm
    X16 = Z['channels__N16'].astype(float); names = [str(n) for n in Z['channel_names__N16']]
    assert X16.shape == (c.off[-1], 16) and np.isfinite(X16).all()
    n = np.diff(c.off)
    keep_ans = (c.cls != 'correct') & (n >= 3)
    ai = np.flatnonzero(keep_ans)
    rows = np.concatenate([np.arange(c.off[j], c.off[j + 1]) for j in ai])
    off = np.r_[0, np.cumsum(n[ai])]
    raw = np.column_stack([X16, c.extra['reward'].astype(float)])[rows]
    z = np.zeros_like(raw); nzero = np.zeros(raw.shape[1], int)
    for a, b in zip(off[:-1], off[1:]):
        v = raw[a:b]; sd = v.std(0); ok = sd > 1e-12
        z[a:b, ok] = (v[:, ok] - v[:, ok].mean(0)) / sd[ok]; nzero += ~ok
    d = dict(prm=prm, z=z, raw=raw, y=c.err[rows], off=off, ans=np.repeat(np.arange(len(ai)), n[ai]),
             pos=c.extra['position'][rows].astype(float), grp=c.groups[ai], lsml=c.arms['CHAIN_N16'][rows],
             reward=c.extra['reward'][rows].astype(float), names=names + ['prm_reward'], nzero=nzero,
             n_ans_dropped_short=int(((c.cls != 'correct') & (n < 3)).sum()), n_control=int((c.cls == 'correct').sum()))
    assert d['y'].any() and (~d['y']).any()
    return d


# ------------------------------------------------------------------------------------------------ statistics
def corr_from_moments(n, s, ss):
    m = s / n; C = ss / n - np.outer(m, m); sd = np.sqrt(np.clip(np.diag(C), 1e-300, None))
    return C / np.outer(sd, sd)


def stats(Re, Ro, sgn):
    p = Re.shape[0]; off = ~np.eye(p, dtype=bool); D = Re - Ro
    O = np.outer(sgn, sgn)
    return dict(frob=float(np.sqrt((D[off] ** 2).sum())),
                mean_off_err=float(Re[off].mean()), mean_off_ok=float(Ro[off].mean()),
                mean_off_diff=float(Re[off].mean() - Ro[off].mean()),
                mean_abs_err=float(np.abs(Re[off]).mean()), mean_abs_ok=float(np.abs(Ro[off]).mean()),
                mean_ori_err=float((O * Re)[off].mean()), mean_ori_ok=float((O * Ro)[off].mean()),
                mean_ori_diff=float((O * Re)[off].mean() - (O * Ro)[off].mean()))


def class_corrs(X, y, w=None):
    """Within-class correlation; w = optional per-row weights (bootstrap)."""
    out = []
    for cl in (True, False):
        m = y == cl; Xm = X[m]
        if w is None:
            out.append(np.corrcoef(Xm, rowvar=False))
        else:
            ww = w[m]; out.append(corr_from_moments(ww.sum(), ww @ Xm, (Xm * ww[:, None]).T @ Xm))
    return out


def permute_within(y, ans, rng):
    """Shuffle labels within each answer (error count preserved). Rows are grouped contiguously by answer."""
    order = np.lexsort((rng.random(len(y)), ans)); yp = np.empty_like(y); yp[order] = y
    return yp


def class_center(X, y, ans):
    """Subtract the answer x class mean; drop rows whose answer x class cell has a single row."""
    key = ans * 2 + y.astype(int); K = key.max() + 1
    cnt = np.bincount(key, minlength=K)
    mu = np.column_stack([np.bincount(key, X[:, j], minlength=K) for j in range(X.shape[1])]) / np.maximum(cnt, 1)[:, None]
    return X - mu[key], cnt[key] >= 2


def full_stats(X, y, sgn):
    Re, Ro = class_corrs(X, y); return stats(Re, Ro, sgn), Re, Ro


def perm_null(X, y, ans, sgn, rng, centered=False, B=B_PERM):
    """Fast within-class moments: err moments via matmul on err rows, ok = total - err."""
    vals = {'frob': [], 'mean_off_diff': [], 'mean_ori_diff': []}
    if not centered:
        n, s, ss = len(X), X.sum(0), X.T @ X
    for _ in range(B):
        yp = permute_within(y, ans, rng)
        if centered:
            Xc, ok = class_center(X, yp, ans); Re, Ro = class_corrs(Xc[ok], yp[ok])
        else:
            Xe = X[yp]; ne, se, sse = len(Xe), Xe.sum(0), Xe.T @ Xe
            Re = corr_from_moments(ne, se, sse); Ro = corr_from_moments(n - ne, s - se, ss - sse)
        st = stats(Re, Ro, sgn)
        for k in vals: vals[k].append(st[k])
    return {k: np.array(v) for k, v in vals.items()}


def boot_ci(X, y, ans, grp, sgn, rng, B=B_BOOT):
    """Cluster bootstrap over source groups: per-group sufficient statistics, multinomial group weights."""
    gu, gi = np.unique(grp, return_inverse=True); G = len(gu)
    row_g = gi[ans]; p = X.shape[1]
    suff = {}
    for cl in (True, False):
        m = y == cl; Xm = X[m]; g = row_g[m]
        cnt = np.bincount(g, minlength=G).astype(float)
        s = np.zeros((G, p)); np.add.at(s, g, Xm)
        ss = np.zeros((G, p, p)); np.add.at(ss, g, Xm[:, :, None] * Xm[:, None, :])
        suff[cl] = (cnt, s, ss.reshape(G, p * p))
    W = rng.multinomial(G, np.full(G, 1 / G), size=B).astype(float)
    out = {'frob': [], 'mean_off_diff': [], 'mean_ori_diff': []}
    for b in range(B):
        R = []
        for cl in (True, False):
            cnt, s, ss = suff[cl]; w = W[b]
            R.append(corr_from_moments(w @ cnt, w @ s, (w @ ss).reshape(p, p)))
        st = stats(R[0], R[1], sgn)
        for k in out: out[k].append(st[k])
    return {k: (float(np.quantile(v, .025)), float(np.quantile(v, .975))) for k, v in out.items()}


def pval(null, obs, two_sided=False):
    if two_sided: return float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null)))
    return float((1 + np.sum(null >= obs)) / (1 + len(null)))


def answer_z(raw, off):
    z = np.zeros_like(raw)
    for a, b in zip(off[:-1], off[1:]):
        v = raw[a:b]; sd = v.std(0); ok = sd > 1e-12
        z[a:b, ok] = (v[:, ok] - v[:, ok].mean(0)) / sd[ok]
    return z


def sim_mean_shift(d, cols, Ro, rng):
    """Equal-covariance Gaussian null: x = L e + mu * err (L L^T = observed R_ok), then the same answer-z.
    mu = observed per-view standardized class-mean difference of the answer-z values (Cohen's d)."""
    z = d['z'][:, cols]; y = d['y']
    mu = (z[y].mean(0) - z[~y].mean(0)) / np.sqrt(0.5 * (z[y].var(0) + z[~y].var(0)))
    L = np.linalg.cholesky(Ro + 1e-9 * np.eye(len(cols)))
    fr = []
    for _ in range(N_SIM):
        X = rng.standard_normal((len(y), len(cols))) @ L.T + np.outer(y, mu)
        Zs = answer_z(X, d['off']); Re, Rs = class_corrs(Zs, y)
        fr.append(stats(Re, Rs, np.ones(len(cols)))['frob'])
    return np.array(fr), mu


# ------------------------------------------------------------------------------------------------ Part B
def within_auc(y, s, ans):
    v = []
    for a in np.unique(ans):
        m = ans == a
        r = E.auc(y[m], s[m])
        if np.isfinite(r): v.append(r)
    return float(np.mean(v)), len(v)


def within_auc_fast(y, s, off_local):
    v = [E.auc(y[a:b], s[a:b]) for a, b in zip(off_local[:-1], off_local[1:])]
    v = np.array(v); return float(np.nanmean(v)), int(np.isfinite(v).sum())


def subset_offsets(ans_ids, off):
    rows = np.concatenate([np.arange(off[j], off[j + 1]) for j in ans_ids])
    return rows, np.r_[0, np.cumsum(np.diff(off)[ans_ids])]


def part_b(d):
    z = d['z']; y = d['y']; off = d['off']; nA = len(off) - 1
    gkf = GroupKFold(n_splits=5)
    models = ['LDA', 'QDA', 'LR_linear', 'LR_pairwise', 'EXTRA_LR_squares', 'LSML_CHAIN_N16', 'PRM_reward']
    res = {m: {'within': [], 'pooled': [], 'n_auc': []} for m in models}; qda_pick = []
    for tr_a, te_a in gkf.split(np.arange(nA), groups=d['grp']):
        tr, _ = subset_offsets(tr_a, off); te, off_te = subset_offsets(te_a, off)
        Xtr, ytr, Xte, yte = z[tr], y[tr], z[te], y[te]
        # inner selection of the QDA reg_param on training answers only
        best, best_r = -1, None
        inner = GroupKFold(n_splits=3)
        for r in (0.0, 0.01, 0.1):
            sc = []
            for itr, iva in inner.split(tr_a, groups=d['grp'][tr_a]):
                r1, _ = subset_offsets(tr_a[itr], off); r2, o2 = subset_offsets(tr_a[iva], off)
                q = QuadraticDiscriminantAnalysis(reg_param=r).fit(z[r1], y[r1])
                s2 = q.predict_log_proba(z[r2]); s2 = s2[:, 1] - s2[:, 0]
                sc.append(within_auc_fast(y[r2], np.nan_to_num(s2, nan=0.0, posinf=1e300, neginf=-1e300), o2)[0])
            if np.mean(sc) > best: best, best_r = np.mean(sc), r
        qda_pick.append(best_r)
        fitted = {
            'LDA': LinearDiscriminantAnalysis().fit(Xtr, ytr),
            'QDA': QuadraticDiscriminantAnalysis(reg_param=best_r).fit(Xtr, ytr),
            'LR_linear': make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Xtr, ytr),
            'LR_pairwise': make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                                         LogisticRegression(C=1.0, max_iter=5000)).fit(Xtr, ytr),
            # EXTRA: linear + squares, no cross products -> isolates what the cross products (dependence) add
            'EXTRA_LR_squares': make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(
                np.column_stack([Xtr, Xtr ** 2]), ytr)}
        for m in models:
            if m in fitted:
                if m in ('LDA', 'QDA'):
                    lp = fitted[m].predict_log_proba(Xte); s = lp[:, 1] - lp[:, 0]
                elif m == 'EXTRA_LR_squares':
                    s = fitted[m].decision_function(np.column_stack([Xte, Xte ** 2]))
                else:
                    s = fitted[m].decision_function(Xte)
                s = np.nan_to_num(s, nan=0.0, posinf=1e300, neginf=-1e300)
            elif m == 'LSML_CHAIN_N16':
                s = d['lsml'][te]
            else:
                s = -d['reward'][te]
            w, na = within_auc_fast(yte, s, off_te)
            res[m]['within'].append(w); res[m]['n_auc'].append(na); res[m]['pooled'].append(E.auc(yte, s))
    rows = []
    for m in models:
        r = res[m]
        rows.append(dict(prm=d['prm'], model=m, features='answer-z N17' if m not in ('LSML_CHAIN_N16', 'PRM_reward') else
                         ('frozen label-free score' if m == 'LSML_CHAIN_N16' else '-reward (raw PRM)'),
                         access='supervised (5-fold GroupKFold)' if m not in ('LSML_CHAIN_N16', 'PRM_reward') else 'label-free',
                         within_auc_mean=np.mean(r['within']), within_auc_sd=np.std(r['within'], ddof=1),
                         pooled_auroc_mean=np.mean(r['pooled']), pooled_auroc_sd=np.std(r['pooled'], ddof=1),
                         n_answers_with_both_classes=int(np.sum(r['n_auc'])),
                         fold_within=';'.join(f'{v:.4f}' for v in r['within']),
                         qda_reg_param_per_fold=';'.join(str(v) for v in qda_pick) if m == 'QDA' else ''))
    return rows


# ------------------------------------------------------------------------------------------------ main
def main():
    rngP = np.random.default_rng(SEED); rngB = np.random.default_rng(SEED + 1); rngS = np.random.default_rng(SEED + 2)
    A, Bt, TOP, META = [], [], [], {}
    for prm in PRMS:
        t = time.time(); d = load_cell(prm)
        y, ans = d['y'], d['ans']
        # label-free orientation toward "error": sign of pooled answer-z corr with -reward (reward flipped itself)
        zr = d['z'][:, 16]
        sgn_all = np.array([np.sign(np.corrcoef(d['z'][:, j], -zr)[0, 1]) or 1.0 for j in range(16)] + [-1.0])
        META[prm] = dict(n_answers=len(d['off']) - 1, n_steps=len(y), n_err_steps=int(y.sum()), n_ok_steps=int((~y).sum()),
                         n_answers_with_err=int(np.bincount(ans, y).astype(bool).sum()),
                         n_control_excluded=d['n_control'], n_short_excluded=d['n_ans_dropped_short'],
                         zero_std_answer_channel=dict(zip(d['names'], d['nzero'].tolist())),
                         orientation_label_free=dict(zip(d['names'], sgn_all.tolist())))
        for view, cols in (('N16', list(range(16))), ('N17', list(range(17)))):
            X = d['z'][:, cols]; sgn = sgn_all[cols]; names = [d['names'][j] for j in cols]
            st, Re, Ro = full_stats(X, y, sgn)
            nl = perm_null(X, y, ans, sgn, rngP)
            ci = boot_ci(X, y, ans, d['grp'], sgn, rngB)
            row = dict(prm=prm, view=view, scope='all', n_err=int(y.sum()), n_ok=int((~y).sum()), **st,
                       frob_ci_lo=ci['frob'][0], frob_ci_hi=ci['frob'][1],
                       mean_off_diff_ci_lo=ci['mean_off_diff'][0], mean_off_diff_ci_hi=ci['mean_off_diff'][1],
                       mean_ori_diff_ci_lo=ci['mean_ori_diff'][0], mean_ori_diff_ci_hi=ci['mean_ori_diff'][1],
                       frob_perm_p=pval(nl['frob'], st['frob']), frob_perm_null_mean=float(nl['frob'].mean()),
                       frob_perm_null_q95=float(np.quantile(nl['frob'], .95)),
                       mean_off_diff_perm_p=pval(nl['mean_off_diff'], st['mean_off_diff'], True),
                       mean_ori_diff_perm_p=pval(nl['mean_ori_diff'], st['mean_ori_diff'], True))
            D = Re - Ro; pairs = sorted(combinations(range(len(cols)), 2), key=lambda ij: -abs(D[ij]))[:5]
            row['top5_pairs'] = ' | '.join(f'{names[i]}~{names[j]} {Re[i, j]:+.3f} vs {Ro[i, j]:+.3f}' for i, j in pairs)
            for i, j in pairs:
                TOP.append(dict(prm=prm, view=view, pair=f'{names[i]}~{names[j]}', r_err=Re[i, j], r_ok=Ro[i, j], diff=D[i, j]))
            # EXTRA (ii): pure mean-shift equal-covariance Gaussian simulation
            fr, mu = sim_mean_shift(d, cols, Ro, rngS)
            row.update(sim_meanshift_frob_mean=float(fr.mean()), sim_meanshift_frob_q95=float(np.quantile(fr, .95)),
                       sim_meanshift_frob_max=float(fr.max()))
            A.append(row)
            # EXTRA (i): class-centred within answer
            Xc, okc = class_center(X, y, ans)
            stc, Rec, Roc = full_stats(Xc[okc], y[okc], sgn)
            nlc = perm_null(X, y, ans, sgn, rngP, centered=True, B=200)
            Dc = Rec - Roc; pc = sorted(combinations(range(len(cols)), 2), key=lambda ij: -abs(Dc[ij]))[:5]
            A.append(dict(prm=prm, view=view, scope='EXTRA_class_centred', n_err=int((y & okc).sum()), n_ok=int((~y & okc).sum()),
                          **stc, frob_perm_p=pval(nlc['frob'], stc['frob']), frob_perm_null_mean=float(nlc['frob'].mean()),
                          frob_perm_null_q95=float(np.quantile(nlc['frob'], .95)),
                          mean_off_diff_perm_p=pval(nlc['mean_off_diff'], stc['mean_off_diff'], True),
                          mean_ori_diff_perm_p=pval(nlc['mean_ori_diff'], stc['mean_ori_diff'], True),
                          perm_draws=200,
                          top5_pairs=' | '.join(f'{names[i]}~{names[j]} {Rec[i, j]:+.3f} vs {Roc[i, j]:+.3f}' for i, j in pc)))
            # position terciles (z kept from the full answer)
            terc = np.digitize(d['pos'], [1 / 3, 2 / 3])
            for k in range(3):
                m = terc == k; ym = y[m]; am = ans[m]
                if ym.sum() < 50 or (~ym).sum() < 50: continue
                stk, Rek, Rok = full_stats(X[m], ym, sgn)
                nlk = perm_null(X[m], ym, am, sgn, rngP, B=B_PERM)
                cik = boot_ci(X[m], ym, am, d['grp'], sgn, rngB, B=B_BOOT)
                A.append(dict(prm=prm, view=view, scope=f'pos_tercile_{k + 1}', n_err=int(ym.sum()), n_ok=int((~ym).sum()), **stk,
                              frob_ci_lo=cik['frob'][0], frob_ci_hi=cik['frob'][1],
                              mean_off_diff_ci_lo=cik['mean_off_diff'][0], mean_off_diff_ci_hi=cik['mean_off_diff'][1],
                              mean_ori_diff_ci_lo=cik['mean_ori_diff'][0], mean_ori_diff_ci_hi=cik['mean_ori_diff'][1],
                              frob_perm_p=pval(nlk['frob'], stk['frob']), frob_perm_null_mean=float(nlk['frob'].mean()),
                              frob_perm_null_q95=float(np.quantile(nlk['frob'], .95)),
                              mean_off_diff_perm_p=pval(nlk['mean_off_diff'], stk['mean_off_diff'], True),
                              mean_ori_diff_perm_p=pval(nlk['mean_ori_diff'], stk['mean_ori_diff'], True)))
        Bt += part_b(d)
        META[prm]['seconds'] = round(time.time() - t, 1)
        print(f'{prm} done in {time.time() - t:.0f}s', flush=True)
    E.write_csv(HERE / 'CLASS_CORRELATION.csv', A)
    E.write_csv(HERE / 'QUADRATIC_HEADROOM.csv', Bt)
    E.write_csv(HERE / 'TOP_PAIRS.csv', TOP)
    META['_run'] = dict(seconds=round(time.time() - T0, 1), seed=SEED, perm_draws=B_PERM, boot_draws=B_BOOT, sim_reps=N_SIM)
    (HERE / 'RUN_META.json').write_text(json.dumps(META, indent=1))
    print('total', round(time.time() - T0, 1), 's')


if __name__ == '__main__':
    main()
