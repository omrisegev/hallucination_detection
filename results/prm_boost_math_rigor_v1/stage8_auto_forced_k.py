"""Stage 8 (W7 part 2, W8), pre-registered in AUTO_FORCEDK_PREREG.json /
docs/research_notes/math_rigor_2026-10-24/AUTO_FORCEDK_PREREG.md. Label-free fitting; labels only score.

  A  AUTO_N16 (L-SML with free grouping over [z(PRM), N16 channels], as in prm_boosting_bank_sweep_v1) against
     CHAIN_N16 (PRM forced into its own cross-stage view), both read out with GAP.
  B  CHAIN_N16 with the telemetry partition forced to K = 2..8: PRM share and GAP PRMScore against K.

    COLLECTION_DATA=<data root> python stage8_auto_forced_k.py <out_dir>
"""
import json, sys, time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
from stage7_hygiene import gap_flags  # noqa: E402

E = K.E
PRE = json.loads((Path(__file__).resolve().parent / 'AUTO_FORCEDK_PREREG.json').read_text())


def fit(V, groups=None, anchor=None, K_range=None):
    kw = {} if K_range is None else {'K_range': K_range}
    s, d = K.lsml_continuous_guarded(*[V[:, j] for j in range(V.shape[1])], groups=groups, **kw)
    s = np.asarray(s, float); ref = V[:, 0] if anchor is None else anchor
    flip = np.corrcoef(s, ref)[0, 1] < 0
    return (-s if flip else s), d


def chain(c, V, zp, K_range=None):
    _, dt = fit(V, anchor=V.mean(1), K_range=K_range)
    vg = np.asarray(dt['virtual_classifiers'], float)
    VC = np.column_stack([zp] + [E.answer_z(vg[:, g], c.off) for g in range(vg.shape[1])])
    s, dc = fit(VC, groups=np.arange(VC.shape[1]))
    cw = np.abs(np.asarray(dc['cross_weights'], float))
    return s, {'K': int(dt['K']), 'cross_prm_share': float(cw[0] / cw.sum())}


def auto(V, zp):
    VP = np.column_stack([zp, V])
    s, d = fit(VP)
    ca = np.asarray(d['c'], int)
    return s, {'K': int(d['K']), 'prm_alone': bool((ca == ca[0]).sum() == 1)}


def stats(c, s, ans_eval):
    keep = c.keep if c.bench != 'prmbench' else None
    wa = E.per_answer_auc(c.err, s, c.off, keep); wa[~ans_eval] = np.nan
    C = E.counts_by_answer(c, gap_flags(c, s)) * ans_eval[:, None]
    return wa, C


def boot(c, a, b, Wa):
    """Paired bootstrap (rows of Wa = answer weights, row 0 = point) of a - b in within-AUC and GAP PRMScore."""
    scale = 1. if c.bench == 'prmbench' else 100.
    out = {}
    for name, (wa1, C1), (wa2, C2) in (('pair', a, b),):
        f1, f2 = np.isfinite(wa1), np.isfinite(wa2)
        a1 = (Wa @ np.where(f1, wa1, 0)) / (Wa @ f1); a2 = (Wa @ np.where(f2, wa2, 0)) / (Wa @ f2)
        o1 = scale * E.official_from_counts(Wa @ C1, c.bench); o2 = scale * E.official_from_counts(Wa @ C2, c.bench)
        for m, d in (('auc', a1 - a2), ('official', o1 - o2)):
            out[m] = {'delta': float(d[0]), 'ci': [float(np.quantile(d[1:], .025)), float(np.quantile(d[1:], .975))]}
    return out


def point(c, st):
    wa, C = st
    return {'within_auc': float(np.nanmean(wa)),
            'gap_prmscore': float(E.official_from_counts(C.sum(0), c.bench)) * (1. if c.bench == 'prmbench' else 100.)}


def run_cell(bench, prm):
    c = E.load(bench, prm)
    V = E.channels(c, 'N16'); zp = E.answer_z(c.arms[K.prm_arm(c)], c.off)
    _, ans_eval = K.eval_rows(c, bench)
    W, inv, _ = E.boot_weights(c); Wa = W[:, inv]
    s_ch, i_ch = chain(c, V, zp)
    assert np.max(np.abs(s_ch - c.arms['CHAIN_N16'])) < 1e-5, 'parity'
    s_au, i_au = auto(V, zp)
    st_ch, st_au = stats(c, s_ch, ans_eval), stats(c, s_au, ans_eval)
    share = lambda s: E.prm_weight_share(c, 'CHAIN', 'N16', score=s)['prm_weight_share']
    A = {'chain': {**point(c, st_ch), **i_ch, 'lstsq_prm_share': share(s_ch)},
         'auto': {**point(c, st_au), **i_au, 'lstsq_prm_share': share(s_au)},
         'auto_minus_chain': boot(c, st_au, st_ch, Wa)}
    B = []
    for k in PRE['forced_K']:
        s_k, i_k = chain(c, V, zp, K_range=[k])
        B.append({'K_forced': k, **i_k, **point(c, stats(c, s_k, ans_eval)), 'lstsq_prm_share': share(s_k)})
    Ks = [r['K'] for r in B]
    rho = spearmanr(Ks, [r['cross_prm_share'] for r in B]).statistic
    best = max(r['gap_prmscore'] for r in B)
    summ = {'spearman_K_cross_share': float(rho),
            'share_times_K1': [r['cross_prm_share'] * (r['K'] + 1) for r in B],
            'autoK_minus_best_forced_prmscore': A['chain']['gap_prmscore'] - best,
            'prmscore_range_forced': [min(r['gap_prmscore'] for r in B), best]}
    return {'bench': bench, 'prm': prm, 'A': A, 'B': B, 'B_summary': summ}


def verdicts(res):
    pb = [r for r in res if r['bench'] == 'prmbench']
    lo = lambda r, m: r['A']['auto_minus_chain'][m]['ci'][0]
    hi = lambda r, m: r['A']['auto_minus_chain'][m]['ci'][1]
    auto_better = sum(lo(r, 'official') > 0 for r in pb)
    auto_better_both = sum(lo(r, 'official') > 0 and lo(r, 'auc') > 0 for r in pb)
    chain_better_auc = sum(hi(r, 'auc') < 0 for r in pb)
    a = ('AUTO preferred' if auto_better >= PRE['a_min_prms'] else
         'CHAIN preferred' if chain_better_auc >= PRE['a_min_prms'] and auto_better_both == 0 else 'inconclusive')
    b1 = sum(r['B_summary']['spearman_K_cross_share'] < 0 for r in pb)
    b2 = sum(r['B_summary']['autoK_minus_best_forced_prmscore'] >= PRE['b2_margin'] for r in pb)
    return {'A': {'verdict': a, 'auto_better_official_prms': auto_better, 'auto_better_both_prms': auto_better_both,
                  'chain_better_auc_prms': chain_better_auc},
            'B1': {'met': b1 >= PRE['b_min_prms'], 'prms_negative_spearman': b1},
            'B2': {'met': b2 >= PRE['b_min_prms'], 'prms_autoK_within_margin': b2}}


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); res = []
    for bench in K.BENCHES:
        for prm in E.PRMS:
            t = time.time(); r = run_cell(bench, prm); r['seconds'] = time.time() - t; res.append(r)
            print(bench, prm, f"{r['seconds']:.0f}s", flush=True)
            (out / 'AUTO_FORCEDK_RESULTS.json').write_text(json.dumps(res, indent=1))
    v = verdicts(res)
    (out / 'AUTO_FORCEDK_VERDICT.json').write_text(json.dumps(v, indent=1)); print(json.dumps(v))


if __name__ == '__main__':
    main(sys.argv[1])
