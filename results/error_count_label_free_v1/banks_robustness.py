"""Robustness of the declared E1 GAP rule (no change, no tuning) across all 10 mask1 banks and every exported scorer.
Also the PRM boosting ranking for the other locally refitted arms (LSML_CONT, CHAIN_N13, PRM_native ranking)."""
import csv
import numpy as np
from common_ec import *  # noqa
import common as pc


def gap_counts(s, off):
    k = []
    for a, b in zip(off[:-1], off[1:]):
        m = b - a
        if m == 1: k.append(1); continue
        z = np.sort(s[a:b])[::-1]; km = min(int(np.ceil(m / 2)), m - 1)
        k.append(int(np.argmax(z[:km] - z[1:km + 1]) + 1))
    return np.array(k)


meta = load_meta(); rows = []
for bank in ['N11', 'N13', 'N14', 'N16', 'N22', 'N25', 'N40', 'N43', 'N53', 'N56']:
    bk = load_tele_bank(bank); off = bk['offsets'].astype(np.int64)
    y, cnt, nonc, cls, cat = labels(bk['idx'], off, meta); sm = np.repeat(nonc, np.diff(off))
    for key in [k for k in bk if k.startswith('S__')]:
        s = bk[key]
        q80 = prmscore(y, s > np.quantile(s, .8), sm); gap = prmscore(y, topk_flags(s, off, gap_counts(s, off)), sm)
        k2 = prmscore(y, topk_flags(s, off, 2), sm); orc = prmscore(y, topk_flags(s, off, cnt), sm)
        rows.append(dict(line='tele_mask1', bank=bank, scorer=key[3:], Q80=q80, GAP=gap, K2=k2, ORACLE=orc,
                         GAP_minus_Q80=gap - q80, GAP_minus_K2=gap - k2))
P = pc.load_population(); off = P['off']; S = dict(np.load(OUT / 'BOOST_SCORES.npz'))
y, cnt, nonc, cls, cat = labels(P['ids'], off, meta); sm = np.repeat(nonc, np.diff(off))
S2 = pc.get_scores(P)
for arm in ['PRM_native', 'LSML_CONT', 'CHAIN_N16', 'CHAIN_N13']:
    s = answer_z(S2[arm], off)
    q80 = prmscore(y, s > np.quantile(s, .8), sm); gap = prmscore(y, topk_flags(s, off, gap_counts(s, off)), sm)
    k2 = prmscore(y, topk_flags(s, off, 2), sm); orc = prmscore(y, topk_flags(s, off, cnt), sm)
    rows.append(dict(line='boost_qwen7b', bank='-', scorer=arm, Q80=q80, GAP=gap, K2=k2, ORACLE=orc,
                     GAP_minus_Q80=gap - q80, GAP_minus_K2=gap - k2))
with open(OUT / 'GAP_ALL_BANKS.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
d = np.array([r['GAP_minus_Q80'] for r in rows if r['line'] == 'tele_mask1'])
e = np.array([r['GAP_minus_K2'] for r in rows if r['line'] == 'tele_mask1'])
print(f'tele cells={len(d)}  GAP-Q80 min/median/max = {d.min():.2f}/{np.median(d):.2f}/{d.max():.2f}  positive {np.sum(d > 0)}')
print(f'           GAP-K2  min/median/max = {e.min():.2f}/{np.median(e):.2f}/{e.max():.2f}  positive {np.sum(e > 0)}')
for r in rows:
    if r['line'] == 'boost_qwen7b' or r['scorer'] in ('LSML_CONT', 'ALL_EQ_ORIENTED'):
        print(r['line'], r['bank'], f"{r['scorer']:18s}", ' '.join(f"{k}={r[k]:.2f}" for k in ['Q80', 'GAP', 'K2', 'ORACLE']))
