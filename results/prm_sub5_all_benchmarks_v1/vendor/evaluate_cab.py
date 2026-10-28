"""Label evaluation of the CROP / ARES benchmark collection (local; gold never goes to AIRCC).

Reads the 65 label-free SCORES files (AIRCC $O/scores, copied from Drive) and the gold files written by make_inputs.py
(plus ProcessBench, Socratic and Hard2Verify gold converted to the same {uid, correct, include} layout). Metric code is
imported from results/prm_boost_collection_v1/evaluate_collection.py, so every number uses the same functions as the
PRMBench / Socratic / Hard2Verify collection tables.

Per cell (mode in {prm, sa}, PRM, benchmark) and arm:
  within_auc    mean per-answer AUC of step risk vs error (answers with both classes)
  pooled_auroc  AUROC over all scored steps (CROP's step AUROC)
  macro_f1      0.5*(F1 valid + F1 error) x100 on pooled steps (the PRMScore / ARES Macro-F1 form);
                Hard2Verify uses its official step Balanced F1 x100 instead. Not computed for ProcessBench.
Rules: native (PRM reward <= 0.5, PRM arm only), Q80, GAP (label-free); ORACLE (true count, reference).
Population: 'own' = every answer the cell scored; 'common' = answers scored by all 10 cells of the benchmark
(5 PRM-mode + 5 standalone; 5 standalone for ProcessBench / Socratic / Hard2Verify). Headline tables use 'common'.

    COLLECTION_DATA=<dir with crop_ares_v1/> python evaluate_cab.py
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'results/prm_boost_collection_v1'))
import evaluate_collection as EC  # noqa: E402

D = Path(os.environ.get('CAB_DATA', ROOT.parent / 'hd_collection_data' / 'crop_ares_v1'))
OUT = HERE / 'eval'
PRMS = ['qwen_prm', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm', 'math_shepherd']
NEW = ['crv_arith', 'crv_gsm8k', 'math_shepherd', 'prm800k', 'deltabench']
SA_ONLY = ['processbench', 'socratic', 'hard2verify']
ARMS = {'prm': ['PRM', 'PRMPROB', 'TELE_N16', 'MEAN_N16', 'CHAIN_N16', 'CHAIN_N13', 'PRMG_N16', 'HALF_N16', 'EQ_N16',
                'POS_ONLY'],
        'sa': ['TELE_N16', 'MEAN_N16', 'TELE_N13', 'MEAN_N13', 'CHAIN_N16', 'HALF_N16']}
RULES = ['Q80', 'GAP', 'ORACLE']
METRIC_BENCH = {'hard2verify': 'hard2verify'}            # every other benchmark: PRMScore form (socratic code path)


def gold(bench):
    return {g['uid']: g for g in json.loads((D / 'gold' / f'{bench}.json').read_text(encoding='utf8'))}


def load(mode, prm, bench, keep_uids=None):
    """Same construction as EC.load_external, with this collection's paths and gold."""
    S = dict(np.load(D / 'scores' / f'SCORES_{mode}_{prm}_{bench}.npz'))
    G = gold(bench); c = EC.Cell(); c.bench = METRIC_BENCH.get(bench, 'socratic'); c.prm = prm
    uids = S['uids'].astype(str); lens = S['nonempty_lengths'].astype(int); ne = S['nonempty'].astype(bool)
    starts = np.r_[0, np.cumsum(lens)]
    ans = np.arange(len(uids)) if keep_uids is None else np.flatnonzero(np.isin(uids, list(keep_uids)))
    ne_idx = np.cumsum(ne) - 1
    full_rows = np.concatenate([np.arange(starts[j], starts[j + 1]) for j in ans])
    ne_rows = full_rows[ne[full_rows]]
    c.ids = uids[ans]
    c.off = np.r_[0, np.cumsum([ne[starts[j]:starts[j + 1]].sum() for j in ans])].astype(np.int64)
    corr = np.concatenate([np.asarray(G[u]['correct'], bool) for u in uids])
    inc = np.concatenate([np.asarray(G[u]['include'], bool) for u in uids])
    assert len(corr) == len(ne), (mode, prm, bench, len(corr), len(ne))
    c.err = ~corr[ne_rows]; c.keep = inc[ne_rows]
    c.arms = {k[len('score__'):]: v[ne_idx[ne_rows]].astype(float) for k, v in S.items() if k.startswith('score__')}
    c.native_flag = (S['native_valid'][ne_rows] == 0)
    c.groups = c.ids.copy()
    nv = S['native_valid'].astype(int)
    c.empty = {'correct': corr[full_rows][~ne[full_rows]], 'include': inc[full_rows][~ne[full_rows]],
               'native_valid': nv[full_rows][~ne[full_rows]].astype(bool),
               'answer': np.repeat(np.arange(len(ans)), [(~ne[starts[j]:starts[j + 1]]).sum() for j in ans])}
    c.cat = np.array([G[u].get('category', '') for u in c.ids]); c.extra = {}
    c.n_gold = len(G)
    return c


def cells(bench):
    modes = ['sa'] if bench in SA_ONLY else ['prm', 'sa']
    return [(m, p) for m in modes for p in PRMS if (D / 'scores' / f'SCORES_{m}_{p}_{bench}.npz').exists()]


def row(c, mode, bench, pop, arm, rule):
    f = c.native_flag if rule == 'native' else c.flags(arm, rule)
    C = EC.counts_by_answer(c, f, rule == 'native')
    wa = EC.within(c, arm)
    met = np.nan if bench == 'processbench' else 100 * float(EC.official_from_counts(C.sum(0), c.bench))
    r = EC.rates(C)
    return dict(benchmark=bench, population=pop, mode=mode, prm=c.prm, arm=arm, rule=rule,
                access=EC.access_of(arm) if arm != 'POS_ONLY' else 'label-free control (position only)',
                n_answers=len(c.ids), n_gold=c.n_gold, n_steps=int(c.keep.sum()), error_rate=float(c.err[c.keep].mean()),
                within_auc=float(np.nanmean(wa)), n_auc=int(np.isfinite(wa).sum()),
                pooled_auroc=float(EC.pooled_auroc(c, arm)), macro_f1=met,
                TPR=r['TPR'], TNR=r['TNR'], flag_rate=float(f[c.keep].mean()))


def main():
    OUT.mkdir(exist_ok=True); rows = []
    for bench in NEW + SA_ONLY:
        cl = cells(bench)
        common = set.intersection(*[set(np.load(D / 'scores' / f'SCORES_{m}_{p}_{bench}.npz')['uids'].astype(str))
                                    for m, p in cl])
        for pop in ('own', 'common'):
            for m, p in cl:
                c = load(m, p, bench, None if pop == 'own' else common)
                for arm in ARMS[m]:
                    if arm not in c.arms: continue
                    for rule in (['native'] if arm == 'PRM' else []) + RULES:
                        rows.append(row(c, m, bench, pop, arm, rule))
                print(bench, pop, m, p, len(c.ids), flush=True)
    with open(OUT / 'CAB_EVAL_ALL.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('rows', len(rows), '->', OUT / 'CAB_EVAL_ALL.csv')


if __name__ == '__main__':
    main()
