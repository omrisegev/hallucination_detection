"""POST HOC checks (after stage 2; not declared, not used to choose any rule):
(a) G1 vs R1 vs R2 AUROC for SUCCESS_SET stratified by exact (n, k*) - large G1 coincides with k* = 1;
(b) GAP success / pooled precision on answers that pass vs fail test T1."""
import csv
import numpy as np
from gt_common import *  # noqa
from common_ec import load_meta, labels  # noqa
from stage2_helpers import auroc  # noqa

Z = dict(np.load(HERE / 'STAGE1.npz')); off = Z['PB__off']; ids = Z['PB__ids']; n = np.diff(off)
y, cnt, nonc, cls, cat = labels(ids, off, load_meta())
rows = []


def strat2(pos, s, keys, min_n=20):
    vals, ws = [], []
    for kk in np.unique(keys, axis=0):
        i = np.all(keys == kk, 1)
        if i.sum() >= min_n and 0 < pos[i].sum() < i.sum():
            vals.append(auroc(pos[i], s[i])); ws.append(i.sum())
    return float(np.average(vals, weights=ws)), int(np.sum(ws))


for arm in ('B_CHAIN_N16', 'T_LSML_CONT', 'T_ALL_EQ'):
    key = f'PB__{arm}'; z = Z[f'{key}__z']; k = Z[f'{key}__Q__kstar']
    fl = topk_flags(z, off, k); tp = np.add.reduceat((fl & y).astype(int), off[:-1])
    succ = (tp == k) & (k == cnt); pop = nonc & (n >= 3); keys = np.column_stack([n, k])[pop]
    r = dict(arm=arm)
    for q in ('G1', 'R1', 'R2'):
        r[f'auroc_strat_n_kstar_{q}'], r['coverage'] = strat2(succ[pop], Z[f'{key}__Q__{q}'][pop], keys)
    t1 = Z[f'{key}__pass__T1']
    for nm, m in (('T1_pass', pop & t1), ('T1_fail', pop & ~t1)):
        r[f'{nm}_n'] = int(m.sum()); r[f'{nm}_success_set'] = float(succ[m].mean())
        r[f'{nm}_gap_precision'] = float(tp[m].sum() / k[m].sum()); r[f'{nm}_mean_kstar'] = float(k[m].mean())
    r['share_of_all_GAP_successes_in_T1_fail'] = float(succ[pop & ~t1].sum() / succ[pop].sum())
    rows.append(r); print(r)
with open(HERE / 'POSTHOC.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
