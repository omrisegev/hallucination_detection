"""Two-step flag count, pre-registered in TWO_STEP_PREREG.json / docs/.../TWO_STEP_COUNT_PREREG.md.

Step 1: an answer is called clean iff the PRM's native rule flags none of its scored steps.
Step 2: other answers get GAP flags on the ranking (CHAIN_N16 primary; the PRM's own probability as a secondary row).
Compared with plain GAP on the same ranking. Labels enter only the evaluation.

    COLLECTION_DATA=<data root> python stage3_two_step_count.py <out_dir>
"""
import csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

E = K.E
PRE = json.loads((Path(__file__).resolve().parent / 'TWO_STEP_PREREG.json').read_text())


def clean_answers(c):
    nf = c.native_flag & (np.ones(c.off[-1], bool) if c.bench == 'prmbench' else c.keep)
    return np.add.reduceat(nf.astype(int), c.off[:-1]) == 0


def metric_draws(c, f, W, all_steps=False):
    keep0 = c.keep
    if all_steps:
        c.keep = np.ones(c.off[-1], bool)
    C = E.counts_by_answer(c, f)
    c.keep = keep0
    Wm, inv, ng = W
    G = np.zeros((ng, 4)); np.add.at(G, inv, C)
    scale = 1. if c.bench == 'prmbench' else 100.
    return scale * E.official_from_counts(Wm @ G, c.bench)       # row 0 = point estimate (all weights 1)


def run_cell(bench, prm):
    c = E.load(bench, prm, uids_keep=K.h2v_common_uids()) if bench == 'hard2verify' else E.load(bench, prm)
    W = E.boot_weights(c)
    clean_pred = clean_answers(c)
    true_clean = np.add.reduceat((c.err & (np.ones(c.off[-1], bool) if bench == 'prmbench' else c.keep)).astype(int), c.off[:-1]) == 0
    rows = []
    for rank_arm in ('CHAIN_N16', 'PRM_Q80_PROB' if bench == 'prmbench' else 'PRMPROB'):
        f_gap = c.flags(rank_arm, 'GAP')
        f_two = f_gap & ~np.repeat(clean_pred, np.diff(c.off))
        r = {'benchmark': bench, 'prm': prm, 'ranking': rank_arm, 'answers': len(c.ids),
             'true_clean': int(true_clean.sum()), 'called_clean': int(clean_pred.sum()),
             'clean_called_clean': float(clean_pred[true_clean].mean()) if true_clean.any() else float('nan'),
             'erroneous_called_clean': float(clean_pred[~true_clean].mean())}
        metrics = [('official', False)] + ([('all_answers', True)] if bench == 'prmbench' else [])
        for name, all_steps in metrics:
            a = metric_draws(c, f_two, W, all_steps); b = metric_draws(c, f_gap, W, all_steps); d = a - b
            r.update({f'{name}_gap': float(b[0]), f'{name}_two_step': float(a[0]), f'{name}_delta': float(d[0]),
                      f'{name}_ci_low': float(np.quantile(d[1:], .025)), f'{name}_ci_high': float(np.quantile(d[1:], .975))})
        rows.append(r)
    return rows


def main(out):
    out.mkdir(parents=True, exist_ok=True); rows = []
    for bench in ('prmbench', 'hard2verify'):
        for prm in E.PRMS:
            for r in run_cell(bench, prm):
                rows.append(r)
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    with open(out / 'TWO_STEP_RESULTS.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r))); wr.writeheader(); wr.writerows(rows)
    prim = [r for r in rows if r['ranking'] == 'CHAIN_N16']
    pb = [r for r in prim if r['benchmark'] == 'prmbench']; hv = [r for r in prim if r['benchmark'] == 'hard2verify']
    cr = PRE['criteria']
    a = sum(r['all_answers_ci_low'] > 0 for r in pb) >= cr['a_prmbench_all_answers_ci_above_0_min_prms']
    b = sum(r['official_delta'] >= cr['b_prmbench_official_noninferiority_margin'] for r in pb) >= cr['b_min_prms']
    cc = sum(r['official_delta'] > 0 for r in hv) >= cr['c_h2v_point_positive_min_prms']
    verdict = 'pass' if (a and b and cc) else ('partial' if b and (a or cc) else 'fail')
    summ = {'a_prmbench_all_answers': a, 'b_prmbench_official_noninferior': b, 'c_h2v_positive': cc, 'verdict': verdict}
    (out / 'TWO_STEP_VERDICT.json').write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
