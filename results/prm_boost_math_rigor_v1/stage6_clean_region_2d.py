"""Calibrated track, 2-D clean region in (S, number of steps); pre-registered in CLEAN_REGION_2D_PREREG.json.

Variants (both certified with the same Learn-then-Test as stage4; labels on the calibration split only):
  mondrian_len3  (primary)   length terciles of the calibration answers; one tau per tercile, each certified at
                             alpha on that tercile's erroneous answers. The overall false-clean rate is a mixture
                             of per-stratum rates, each <= alpha w.p. >= 1 - delta (union over 3 strata: >= 1 - 3*delta
                             for all three at once; the marginal rate is still controlled by the mixture).
  mean_p         (secondary) 1-D score S / n_steps (length-normalized expected error rate)
Everything else (splits, seeds, endpoints, flags, Hard2Verify transfer) as stage4_calibrated_clean.py.

    COLLECTION_DATA=<data root> python stage6_clean_region_2d.py <out_dir>
"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage4_calibrated_clean as S4  # noqa: E402

K, E = S4.K, S4.E
PRE = json.loads((Path(__file__).resolve().parent / 'CLEAN_REGION_2D_PREREG.json').read_text())
P4 = S4.PRE


def fit_rule(variant, S, n, err, cal, alpha):
    if variant == 'mean_p':
        return {'tau': S4.choose_tau(S / n, err, cal, alpha, P4['delta'], P4['grid'])}
    cuts = np.quantile(n[cal], [1 / 3, 2 / 3])
    st = np.digitize(n, cuts)
    taus = [S4.choose_tau(S, err, cal & (st == k), alpha, P4['delta'], P4['grid']) for k in range(3)]
    return {'cuts': cuts.tolist(), 'taus': taus}


def apply_rule(variant, rule, S, n):
    if variant == 'mean_p':
        return (S / n < rule['tau']) if rule['tau'] is not None else np.zeros(len(S), bool)
    st = np.digitize(n, rule['cuts']); out = np.zeros(len(S), bool)
    for k, t in enumerate(rule['taus']):
        if t is not None:
            out |= (st == k) & (S < t)
    return out


def run(variant, prm, alpha):
    c = E.load('prmbench', prm)
    p = np.clip(c.arms['PRM_Q80_PROB'], 0, 1); n = np.diff(c.off).astype(float)
    S = S4.answer_sums(c, p, np.ones(c.off[-1], bool))
    err = np.add.reduceat(c.err.astype(int), c.off[:-1]) > 0
    f_gap = c.flags('CHAIN_N16', 'GAP'); seeds = []
    for seed in range(P4['seeds']):
        cal = S4.split(c, P4['split_seed0'] + seed); test = ~cal
        rule = fit_rule(variant, S, n, err, cal, alpha); clean = apply_rule(variant, rule, S, n)
        f2 = S4.two_step_flags(c, f_gap, clean)
        r = {'seed': seed, 'rule': rule, 'test_false_clean': float(clean[test & err].mean()),
             'test_clean_recall': float(clean[test & ~err].mean()),
             'd_all': S4.step_metric(c, f2, test, True) - S4.step_metric(c, f_gap, test, True),
             'd_official': S4.step_metric(c, f2, test, False) - S4.step_metric(c, f_gap, test, False)}
        if seed == 0:
            W = S4.test_group_weights(c, test, P4['draws'], P4['boot_seed'])
            for name, allst in (('d_all', True), ('d_official', False)):
                d = S4.step_metric(c, f2, test, allst, W) - S4.step_metric(c, f_gap, test, allst, W)
                r[name + '_ci'] = [float(np.quantile(d[1:], .025)), float(np.quantile(d[1:], .975))]
            rule0 = rule
        seeds.append(r)
    # Hard2Verify transfer of the seed-0 rule
    h = E.load('hard2verify', prm, uids_keep=K.h2v_common_uids())
    ph = np.clip(h.arms['PRMPROB'], 0, 1)
    nh = np.array([h.keep[a:b].sum() for a, b in zip(h.off[:-1], h.off[1:])], float); nh = np.maximum(nh, 1)
    Sh = S4.answer_sums(h, ph, h.keep); errh = np.add.reduceat((h.err & h.keep).astype(int), h.off[:-1]) > 0
    ch = apply_rule(variant, rule0, Sh, nh); fg = h.flags('CHAIN_N16', 'GAP'); f2h = S4.two_step_flags(h, fg, ch)
    ones = np.ones(len(h.ids), bool); Wh = E.boot_weights(h)
    dh = S4.step_metric(h, f2h, ones, False, Wh) - S4.step_metric(h, fg, ones, False, Wh)
    hv = {'false_clean': float(ch[errh].mean()), 'clean_recall': float(ch[~errh].mean()), 'd_official': float(dh[0]),
          'd_official_ci': [float(np.quantile(dh[1:], .025)), float(np.quantile(dh[1:], .975))]}
    return seeds, hv


def main(out):
    out.mkdir(parents=True, exist_ok=True); res = {}
    for variant in PRE['variants']:
        for prm in E.PRMS:
            seeds, hv = run(variant, prm, P4['primary_alpha'])
            ok = sum(s['test_false_clean'] <= P4['primary_alpha'] for s in seeds); s0 = seeds[0]
            res[f'{variant}/{prm}'] = {'seeds': seeds, 'h2v': hv, 'guarantee_seeds_ok': ok,
                                      'mean_clean_recall': float(np.mean([s['test_clean_recall'] for s in seeds]))}
            print('%-13s %-13s false-clean %.3f recall %.3f guarantee %d/10 | d_all %+.2f %s | d_off %+.2f %s | H2V fc %.2f rec %.2f d %+.2f %s' % (
                variant, prm, s0['test_false_clean'], s0['test_clean_recall'], ok, s0['d_all'], np.round(s0['d_all_ci'], 2),
                s0['d_official'], np.round(s0['d_official_ci'], 2), hv['false_clean'], hv['clean_recall'], hv['d_official'],
                np.round(hv['d_official_ci'], 1)), flush=True)
    v = PRE['primary_variant']; prim = {prm: res[f'{v}/{prm}'] for prm in E.PRMS}
    e1 = sum(x['guarantee_seeds_ok'] >= P4['e1_min_seeds'] for x in prim.values())
    e2 = sum(x['seeds'][0]['d_all_ci'][0] > 0 for x in prim.values())
    e3 = sum(x['seeds'][0]['d_official'] >= P4['e3_margin'] for x in prim.values())
    e4 = sum(x['h2v']['d_official'] > 0 for x in prim.values())
    verdict = ('pass' if e1 >= 4 and e2 >= 3 and e3 >= 4 else 'partial' if e1 >= 4 and (e2 >= 3 or e3 >= 4) else 'fail')
    summ = {'primary_variant': v, 'e1_guarantee_prms': e1, 'e2_all_answers_gain_prms': e2,
            'e3_official_noninferior_prms': e3, 'h2v_transfer_positive_prms': e4, 'verdict': verdict}
    (out / 'REGION2D_RESULTS.json').write_text(json.dumps(res, indent=1, default=float))
    (out / 'REGION2D_VERDICT.json').write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
