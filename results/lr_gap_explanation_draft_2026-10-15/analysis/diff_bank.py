"""N56 difference bank: feed LR's correlated contrast pairs as precomputed differences.

Per outer fold (pairs and orientation from that fold's TRAINING LR, lr/CORRELATED_CONTRAST_PAIRS.csv):
  d_ab = answer_z( sign(w_a) * (V_a - V_b) )    for every pair with opposite LR signs
Banks:
  ORIGINAL       the 56 channels
  REPLACE        unpaired channels + differences (paired originals dropped)
  REPLACE_LEVEL  REPLACE + one answer-z mean per connected component of the pair graph
  AUGMENT        56 channels + differences
Methods (all fit on the training fold, scored OOF): plain average, LR (C=10), IU-PCR, L-SML.
NOTE: pair choice and difference orientation use LR, i.e. labels. This is a supervised-prior diagnostic.
"""
import json, sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/Users/osegev/Desktop/hallucination_detection')
sys.path.insert(0, '/Users/osegev/Desktop/hallucination_detection/results/lr_gap_explanation_draft_2026-10-15/analysis')
from spectral_utils.advisor_lr import fit_lr, fit_labelfree
from decomp import within, ROOT

BANK = 'N56'
p = np.load(f'{ROOT}/inputs/population.npz'); off = p['offsets'].astype(int); y = p['y'].astype(int)
sf = np.repeat(p['fold'].astype(int), np.diff(off)); n = np.diff(off); aid = np.repeat(np.arange(len(n)), n)
b = np.load(f'{ROOT}/inputs/bank_V_{BANK}.npz'); V = b['V'].astype(float); names = b['columns'].astype(str).tolist()
PAIRS = pd.read_csv(f'{ROOT}/lr/CORRELATED_CONTRAST_PAIRS.csv'); PAIRS = PAIRS[PAIRS.bank == BANK]


def answer_z(x):
    cnt = np.bincount(aid); m = np.bincount(aid, x) / cnt; d = x - m[aid]
    s = np.sqrt(np.bincount(aid, d * d) / cnt); s = np.where(s > 1e-12, s, 1.)
    return d / s[aid]


def components(edges, nodes):
    parent = {v: v for v in nodes}
    def f(v):
        while parent[v] != v: parent[v] = parent[parent[v]]; v = parent[v]
        return v
    for a, c in edges: parent[f(a)] = f(c)
    comp = {}
    for v in nodes: comp.setdefault(f(v), []).append(v)
    return list(comp.values())


def banks_for_fold(o):
    q = PAIRS[PAIRS.outer == o]
    q = q[np.sign(q.LR_standardized_weight_left) != np.sign(q.LR_standardized_weight_right)]
    idx = {c: j for j, c in enumerate(names)}
    D = np.column_stack([answer_z(np.sign(r.LR_standardized_weight_left) * (V[:, idx[r.left]] - V[:, idx[r.right]]))
                         for r in q.itertuples()])
    paired = sorted(set(q.left) | set(q.right))
    keep = [j for j, c in enumerate(names) if c not in paired]
    comps = components(list(zip(q.left, q.right)), paired)
    LV = np.column_stack([answer_z(V[:, [idx[c] for c in comp]].mean(1)) for comp in comps])
    info = {'n_pairs': len(q), 'n_paired_channels': len(paired), 'n_unpaired': len(keep),
            'components': [len(c) for c in comps]}
    return {'ORIGINAL': V, 'REPLACE': np.column_stack([V[:, keep], D]),
            'REPLACE_LEVEL': np.column_stack([V[:, keep], D, LV]), 'AUGMENT': np.column_stack([V, D])}, info


def main():
    methods = ['PLAIN_AVG', 'LR', 'IU_PCR', 'LSML']
    sc = {}; infos = []; fails = []
    for o in range(5):
        B, info = banks_for_fold(o); infos.append(info); tr, te = sf != o, sf == o
        for bank, X in B.items():
            out = {'PLAIN_AVG': X[te].mean(1), 'LR': fit_lr(X[tr], y[tr], 10.).predict(X[te])}
            for meth in ('IU_PCR', 'LSML_CONT'):
                try:
                    out['LSML' if meth == 'LSML_CONT' else meth] = fit_labelfree(X[tr], meth).predict(X[te])
                except Exception as e:
                    fails.append(f'{bank} fold{o} {meth}: {type(e).__name__}: {e}')
            for k, v in out.items():
                sc.setdefault((bank, k), np.full(len(y), np.nan))[te] = v
            print(o, bank, X.shape[1], 'done', flush=True)
    print('fold info:', infos)
    if fails: print('FAILURES:', fails)
    rows = []
    for bank in ['ORIGINAL', 'REPLACE', 'REPLACE_LEVEL', 'AUGMENT']:
        r = {'bank': bank}
        for k in methods:
            v = sc.get((bank, k))
            r[k] = round(within(v, off, y), 4) if v is not None and np.isfinite(v).all() else None
        rows.append(r)
    df = pd.DataFrame(rows); print(df.to_string(index=False))
    df.to_csv('/Users/osegev/Desktop/hallucination_detection/results/lr_gap_explanation_draft_2026-10-15/analysis/diff_bank_N56.csv', index=False)


if __name__ == '__main__':
    main()
