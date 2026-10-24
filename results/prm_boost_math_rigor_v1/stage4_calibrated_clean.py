"""Calibrated track (CALIBRATED_TRACK = labels on a declared calibration split): risk-controlled "clean answer"
threshold, pre-registered in CALIBRATED_CLEAN_PREREG.json / docs/.../CALIBRATED_CLEAN_PREREG.md.

Answer score S = sum over steps of (1 - reward). Called clean iff S < tau. tau is chosen by fixed-sequence
Learn-then-Test (spectral_utils.risk_control) on PRMBench calibration source groups so that the false-clean rate
(erroneous answers called clean) is <= alpha with probability >= 1 - delta. Answers called clean get no flags; the
rest get GAP on CHAIN_N16. Compared with plain GAP on the test source groups; Hard2Verify gets the seed-0 PRMBench
tau (transfer, no guarantee).

    COLLECTION_DATA=<data root> python stage4_calibrated_clean.py <out_dir>
"""
import csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
from spectral_utils.risk_control import fixed_sequence_threshold  # noqa: E402

E = K.E
PRE = json.loads((Path(__file__).resolve().parent / 'CALIBRATED_CLEAN_PREREG.json').read_text())


def answer_sums(c, p, keep):
    return np.array([p[a:b][keep[a:b]].sum() for a, b in zip(c.off[:-1], c.off[1:])])


def split(c, seed):
    """Half of the source groups to calibration, stratified by mean answer length (pairs of adjacent groups)."""
    g = c.groups.astype(str); u, inv = np.unique(g, return_inverse=True)
    nlen = np.bincount(inv, weights=np.diff(c.off)) / np.bincount(inv)
    order = np.argsort(nlen, kind='stable'); rng = np.random.default_rng(seed)
    cal_g = np.zeros(len(u), bool)
    for i in range(0, len(order), 2):
        pair = order[i:i + 2]
        cal_g[pair[rng.integers(len(pair))]] = True
    return cal_g[inv]                                     # per answer


def choose_tau(S, err, cal, alpha, delta, grid_n):
    grid = np.quantile(S[cal], np.linspace(0, 1, grid_n))
    s_err = S[cal & err]
    risks = np.array([(s_err < t).mean() for t in grid])
    j, _ = fixed_sequence_threshold(grid, risks, len(s_err), alpha, delta)
    return None if j is None else float(grid[j])


def step_metric(c, f, ans_mask, all_steps, W=None):
    keep0 = c.keep
    if all_steps:
        c.keep = np.ones(c.off[-1], bool)
    C = E.counts_by_answer(c, f); c.keep = keep0
    C = C * ans_mask[:, None]
    scale = 1. if c.bench == 'prmbench' else 100.
    if W is None:
        return scale * float(E.official_from_counts(C.sum(0), c.bench))
    Wm, inv, ng = W
    G = np.zeros((ng, 4)); np.add.at(G, inv, C)
    return scale * E.official_from_counts(Wm @ G, c.bench)


def test_group_weights(c, test, draws, seed):
    """Paired bootstrap weights over the TEST source groups only (calibration answers get group -1 -> weight 0)."""
    g = np.where(test, c.groups.astype(str), '__cal__')
    u, inv = np.unique(g, return_inverse=True)
    is_test = u != '__cal__'; nt = int(is_test.sum()); rng = np.random.default_rng(seed)
    W = np.zeros((draws + 1, len(u))); W[0, is_test] = 1.
    W[1:, is_test] = rng.multinomial(nt, np.full(nt, 1 / nt), size=draws)
    return W, inv, len(u)


def two_step_flags(c, f_gap, clean):
    return f_gap & ~np.repeat(clean, np.diff(c.off))


def run_prmbench(prm, alpha):
    c = E.load('prmbench', prm)
    p = np.clip(c.arms['PRM_Q80_PROB'], 0, 1); allk = np.ones(c.off[-1], bool)
    S = answer_sums(c, p, allk)
    err = np.add.reduceat(c.err.astype(int), c.off[:-1]) > 0
    f_gap = c.flags('CHAIN_N16', 'GAP')
    seeds = []
    for seed in range(PRE['seeds']):
        cal = split(c, PRE['split_seed0'] + seed); test = ~cal
        tau = choose_tau(S, err, cal, alpha, PRE['delta'], PRE['grid'])
        clean = (S < tau) if tau is not None else np.zeros(len(S), bool)
        f2 = two_step_flags(c, f_gap, clean)
        r = {'seed': seed, 'tau': tau, 'test_false_clean': float(clean[test & err].mean()),
             'test_clean_recall': float(clean[test & ~err].mean()) if (test & ~err).any() else float('nan'),
             'd_all': step_metric(c, f2, test, True) - step_metric(c, f_gap, test, True),
             'd_official': step_metric(c, f2, test, False) - step_metric(c, f_gap, test, False)}   # keep = non-control
        if seed == 0:
            W = test_group_weights(c, test, PRE['draws'], PRE['boot_seed'])
            for name, allst in (('d_all', True), ('d_official', False)):
                d = step_metric(c, f2, test, allst, W) - step_metric(c, f_gap, test, allst, W)
                r[name + '_ci'] = [float(np.quantile(d[1:], .025)), float(np.quantile(d[1:], .975))]
            tau0 = tau
        seeds.append(r)
    return seeds, tau0


def run_h2v(prm, tau):
    c = E.load('hard2verify', prm, uids_keep=K.h2v_common_uids())
    p = np.clip(c.arms['PRMPROB'], 0, 1)
    S = answer_sums(c, p, c.keep)
    err = np.add.reduceat((c.err & c.keep).astype(int), c.off[:-1]) > 0
    clean = (S < tau) if tau is not None else np.zeros(len(S), bool)
    f_gap = c.flags('CHAIN_N16', 'GAP'); f2 = two_step_flags(c, f_gap, clean)
    ones = np.ones(len(c.ids), bool); W = E.boot_weights(c)
    d = step_metric(c, f2, ones, False, W) - step_metric(c, f_gap, ones, False, W)
    return {'tau': tau, 'false_clean': float(clean[err].mean()), 'clean_recall': float(clean[~err].mean()),
            'd_official': float(d[0]), 'd_official_ci': [float(np.quantile(d[1:], .025)), float(np.quantile(d[1:], .975))]}


def main(out):
    out.mkdir(parents=True, exist_ok=True); res = {}
    for alpha in PRE['alphas']:
        for prm in E.PRMS:
            seeds, tau0 = run_prmbench(prm, alpha)
            h = run_h2v(prm, tau0)
            ok = sum(s['test_false_clean'] <= alpha for s in seeds)
            res[f'{alpha}/{prm}'] = {'prmbench_seeds': seeds, 'h2v_transfer': h,
                                    'guarantee_seeds_ok': ok,
                                    'mean_d_all': float(np.mean([s['d_all'] for s in seeds])),
                                    'mean_d_official': float(np.mean([s['d_official'] for s in seeds])),
                                    'mean_clean_recall': float(np.nanmean([s['test_clean_recall'] for s in seeds])),
                                    'n_seeds_never_clean': int(sum(s['tau'] is None for s in seeds))}
            s0 = seeds[0]
            print('alpha %.2f %-13s tau0 %s | test false-clean %.3f recall %.3f | guarantee %d/%d | d_all %+.2f %s | d_off %+.2f %s | H2V false-clean %.2f d %+.2f %s' % (
                alpha, prm, None if s0['tau'] is None else round(s0['tau'], 3), s0['test_false_clean'], s0['test_clean_recall'],
                ok, PRE['seeds'], s0['d_all'], np.round(s0['d_all_ci'], 2), s0['d_official'], np.round(s0['d_official_ci'], 2),
                h['false_clean'], h['d_official'], np.round(h['d_official_ci'], 1)), flush=True)
    prim = {prm: res[f"{PRE['primary_alpha']}/{prm}"] for prm in E.PRMS}
    e1 = sum(v['guarantee_seeds_ok'] >= PRE['e1_min_seeds'] for v in prim.values())
    e2 = sum(v['prmbench_seeds'][0]['d_all_ci'][0] > 0 for v in prim.values())
    e3 = sum(v['prmbench_seeds'][0]['d_official'] >= PRE['e3_margin'] for v in prim.values())
    verdict = ('pass' if e1 >= PRE['e1_min_prms'] and e2 >= PRE['e2_min_prms'] and e3 >= PRE['e3_min_prms'] else
               'partial' if e1 >= PRE['e1_min_prms'] and (e2 >= PRE['e2_min_prms'] or e3 >= PRE['e3_min_prms']) else 'fail')
    summ = {'e1_guarantee_prms': e1, 'e2_all_answers_gain_prms': e2, 'e3_official_noninferior_prms': e3, 'verdict': verdict}
    (out / 'CALIBRATED_RESULTS.json').write_text(json.dumps(res, indent=1, default=float))
    (out / 'CALIBRATED_VERDICT.json').write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
