"""Stage 9: AUTO-F, pre-registered in AUTO_F_PREREG.json / docs/research_notes/math_rigor_2026-10-24/AUTO_F_PREREG.md.

AUTO-F keeps AUTO's label-free partition of [z(PRM), N16 channels] and replaces L-SML's eigenvector weights:
  1. every column is answer-z'd (P1); constant columns (AUTO's c = -1) are dropped;
  2. each group gets one composite by block-vs-rest CCA (ridge 1e-3 x mean diagonal); under the L-SML block model
     this direction is Psi_b^-1 u_b (Fisher within the group). Singletons are their own composite. Each composite is
     oriented to correlate positively with z(PRM) and standardized;
  3. a one-factor ML factor analysis on the composites gives loadings lam and uniquenesses psi (sklearn); loadings
     are oriented so the PRM's composite loads positively, negative loadings are set to 0 (W9), weights = lam / psi;
  4. fused = sum_b w_b z_b, oriented with z(PRM). Fewer than 3 composites: AUTO's score (declared fallback).
Also: AUTO and AUTO-F with forced K = 2..8 (AUTO's partition with K_range=[K]), CHAIN forced K for the matched share
comparison, and a label-using LDA vs QDA diagnostic (5-fold source-group CV, diagnostic only).

    COLLECTION_DATA=<data root> python stage9_auto_f.py <out_dir>
"""
import json, sys, time
from pathlib import Path

import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
from stage8_auto_forced_k import boot, chain, fit, point, stats  # noqa: E402

E = K.E
PRE = json.loads((Path(__file__).resolve().parent / 'AUTO_F_PREREG.json').read_text())


def auto_fit(V, zp, K_range=None):
    VP = np.column_stack([zp, V])
    s, d = fit(VP, K_range=K_range)
    return s, np.asarray(d['c'], int), int(d['K'])


def _inv_sqrt(C, ridge):
    C = C + ridge * np.trace(C) / len(C) * np.eye(len(C))
    w, U = np.linalg.eigh(C)
    return U @ np.diag(w ** -.5) @ U.T


def block_cca(Xb, Xr, ridge):
    m = Xb.shape[1]
    C = np.cov(np.column_stack([Xb, Xr]).T)
    Wb, Wr = _inv_sqrt(C[:m, :m], ridge), _inv_sqrt(C[m:, m:], ridge)
    U, S, _ = np.linalg.svd(Wb @ C[:m, m:] @ Wr)
    return Wb @ U[:, 0], float(S[0])


def auto_f(c, V, zp, part, s_auto):
    """AUTO-F score from AUTO's partition `part` (over [zp, V], -1 = dropped)."""
    VP = np.column_stack([zp, V]); keep = np.flatnonzero(part >= 0)
    X = np.column_stack([E.answer_z(VP[:, j], c.off) for j in keep]); grp = part[keep]
    labels = np.unique(grp)
    if len(labels) < 3:
        return s_auto, {'fallback': True, 'G': int(len(labels))}
    Z, prm_comp = [], None
    for g in labels:
        idx = np.flatnonzero(grp == g)
        if len(idx) == 1:
            s = X[:, idx[0]]
        else:
            rest = np.flatnonzero(grp != g)
            a, _ = block_cca(X[:, idx], X[:, rest], PRE['ridge'])
            s = X[:, idx] @ a
        if np.corrcoef(s, zp)[0, 1] < 0: s = -s
        Z.append((s - s.mean()) / s.std())
        if 0 in keep[idx]: prm_comp = len(Z) - 1
    Z = np.column_stack(Z)
    fa = FactorAnalysis(n_components=1, random_state=0).fit(Z)
    lam = fa.components_[0].copy(); psi = np.maximum(fa.noise_variance_, PRE['psi_floor'])
    if lam[prm_comp] < 0: lam = -lam
    n_neg = int((lam < 0).sum()); lam = np.maximum(lam, 0.)
    w = lam / psi
    s = Z @ w
    if np.corrcoef(s, zp)[0, 1] < 0: s = -s
    return s, {'fallback': False, 'G': int(len(labels)), 'lam': lam.tolist(), 'psi': psi.tolist(),
               'w': (w / np.abs(w).sum()).tolist(), 'prm_composite': int(prm_comp), 'n_negative_loadings': n_neg,
               'prm_group_size': int((grp == grp[keep == 0][0]).sum()) if 0 in keep else 0}


def lda_qda(c, V, zp, ans_eval):
    """Label-using diagnostic: 5-fold source-group CV, LDA vs QDA log-odds, within-AUC on eval answers."""
    X = np.column_stack([zp] + [E.answer_z(V[:, j], c.off) for j in range(V.shape[1]) if V[:, j].std() > 1e-9])
    y = c.err.astype(int); step_ans = np.repeat(np.arange(len(c.ids)), np.diff(c.off))
    groups = c.groups.astype(str)[step_ans]
    out = {'LDA': np.zeros(len(y)), 'QDA': np.zeros(len(y))}
    for tr, te in GroupKFold(n_splits=PRE['cv_folds']).split(X, y, groups):
        out['LDA'][te] = LinearDiscriminantAnalysis().fit(X[tr], y[tr]).decision_function(X[te])
        q = QuadraticDiscriminantAnalysis(reg_param=PRE['qda_reg']).fit(X[tr], y[tr])
        lp = q.predict_log_proba(X[te]); out['QDA'][te] = lp[:, 1] - lp[:, 0]
    keep = c.keep if c.bench != 'prmbench' else None
    res = {}
    for k, s in out.items():
        wa = E.per_answer_auc(c.err, s, c.off, keep); wa[~ans_eval] = np.nan
        res[k] = float(np.nanmean(wa))
    res['QDA_minus_LDA'] = res['QDA'] - res['LDA']
    return res


def run_cell(bench, prm):
    c = E.load(bench, prm)
    V = E.channels(c, 'N16'); zp = E.answer_z(c.arms[K.prm_arm(c)], c.off)
    _, ans_eval = K.eval_rows(c, bench)
    W, inv, _ = E.boot_weights(c); Wa = W[:, inv]
    share = lambda s: E.prm_weight_share(c, 'CHAIN', 'N16', score=s)['prm_weight_share']
    s_ch, i_ch = chain(c, V, zp)
    assert np.max(np.abs(s_ch - c.arms['CHAIN_N16'])) < 1e-5, 'parity'
    s_au, part, k_au = auto_fit(V, zp)
    s_f, i_f = auto_f(c, V, zp, part, s_au)
    st = {n: stats(c, s, ans_eval) for n, s in (('CHAIN', s_ch), ('AUTO', s_au), ('AUTO_F', s_f))}
    A = {n: {**point(c, st[n]), 'lstsq_prm_share': share(s)} for n, s in (('CHAIN', s_ch), ('AUTO', s_au), ('AUTO_F', s_f))}
    A['AUTO']['K'] = k_au; A['AUTO_F'].update(i_f); A['CHAIN'].update(i_ch)
    A['AUTO_F_minus_AUTO'] = boot(c, st['AUTO_F'], st['AUTO'], Wa)
    A['AUTO_F_minus_CHAIN'] = boot(c, st['AUTO_F'], st['CHAIN'], Wa)
    B = []
    for k in PRE['forced_K']:
        sa, pk, kk = auto_fit(V, zp, K_range=[k]); sf, inf = auto_f(c, V, zp, pk, sa); sc, ic = chain(c, V, zp, K_range=[k])
        B.append({'K_forced': k, 'K_auto': kk, 'K_chain': ic['K'], 'fallback': inf['fallback'],
                  'AUTO': {**point(c, stats(c, sa, ans_eval)), 'lstsq_prm_share': share(sa)},
                  'AUTO_F': {**point(c, stats(c, sf, ans_eval)), 'lstsq_prm_share': share(sf)},
                  'CHAIN': {**point(c, stats(c, sc, ans_eval)), 'lstsq_prm_share': share(sc)}})
    rng = lambda arm: float(np.ptp([b[arm]['lstsq_prm_share'] for b in B]))
    Bs = {'share_range_AUTO_F': rng('AUTO_F'), 'share_range_CHAIN': rng('CHAIN'), 'share_range_AUTO': rng('AUTO'),
          'prmscore_range_AUTO_F': float(np.ptp([b['AUTO_F']['gap_prmscore'] for b in B])),
          'prmscore_range_CHAIN': float(np.ptp([b['CHAIN']['gap_prmscore'] for b in B]))}
    D = lda_qda(c, V, zp, ans_eval)
    return {'bench': bench, 'prm': prm, 'A': A, 'B': B, 'B_summary': Bs, 'diag_lda_qda': D}


def verdicts(res):
    pb = [r for r in res if r['bench'] == 'prmbench']
    e1 = sum(r['A']['AUTO_F_minus_AUTO']['auc']['ci'][0] > 0 for r in pb)
    e2 = sum(r['A']['AUTO_F_minus_CHAIN']['auc']['delta'] >= PRE['e2_auc_margin'] and
             r['A']['AUTO_F_minus_CHAIN']['official']['delta'] >= PRE['e2_official_margin'] for r in pb)
    e3 = sum(r['B_summary']['share_range_AUTO_F'] < r['B_summary']['share_range_CHAIN'] for r in pb)
    d = sum(r['diag_lda_qda']['QDA_minus_LDA'] >= PRE['diag_qda_gain'] for r in pb)
    met = {'e1': e1 >= PRE['e_min_prms'], 'e2': e2 >= PRE['e_min_prms'], 'e3': e3 >= PRE['e_min_prms']}
    return {'counts': {'e1': e1, 'e2': e2, 'e3': e3, 'diag_qda_prms': d}, 'met': met,
            'verdict_switch_to_auto_f': 'pass' if met['e1'] and met['e2'] else 'fail',
            'diag_class_dependent_direction': 'worth pursuing' if d >= PRE['diag_min_prms'] else 'not supported'}


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); res = []
    for bench in K.BENCHES:
        for prm in E.PRMS:
            t = time.time(); r = run_cell(bench, prm); r['seconds'] = time.time() - t; res.append(r)
            print(bench, prm, f"{r['seconds']:.0f}s", flush=True)
            (out / 'AUTO_F_RESULTS.json').write_text(json.dumps(res, indent=1))
    v = verdicts(res)
    (out / 'AUTO_F_VERDICT.json').write_text(json.dumps(v, indent=1)); print(json.dumps(v))


if __name__ == '__main__':
    main(sys.argv[1])
