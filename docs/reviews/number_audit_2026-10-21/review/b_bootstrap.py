"""Reviewer: paired source-group bootstrap of GAP-Q80 and GAP-K2 (TELE LSML_CONT mask1 N16) and BOOST Q80-count/length rho.
Own implementation; groups looked up by answer id from population.npz."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path('/Users/osegev/Desktop/hallucination_detection'); OUT = Path(__file__).resolve().parent
meta = {r['idx']: r for r in json.loads((ROOT / 'results/thinkfix_rescore_v1/prmscore_inputs/prmbench_meta.json').read_text())['rows']}
z = np.load(ROOT / 'results/thinkfix_rescore_v1/prmscore_inputs/bank_N16.npz')
off = z['offsets'].astype(np.int64); ids = z['idx'].astype(str); n = np.diff(off); A = len(n)
with np.load(ROOT / 'results/advisor_completion_2026-10-11/inputs/population.npz', allow_pickle=True) as p:
    gmap = dict(zip(p['idx'].astype(str), p['group'].astype(str)))
grp = np.array([gmap[i] for i in ids]); gu, ginv = np.unique(grp, return_inverse=True)
y = np.zeros(off[-1], bool)
for i, a, b in zip(ids, off[:-1], off[1:]):
    for e in meta[i]['error_steps']:
        if 1 <= e <= b - a: y[a + e - 1] = True
nonc = np.array([meta[i]['classification'] != 'correct' for i in ids]); sm = np.repeat(nonc, n)
s = z['S__LSML_CONT'].astype(float)


def gapk(x):
    if len(x) == 1: return 1
    q = -np.sort(-x); K = int(np.ceil(len(x) / 2)); return 1 + int(np.argmax(q[:K] - q[1:K + 1]))


def topk(s, ks):
    f = np.zeros(len(s), bool)
    for (a, b), k in zip(zip(off[:-1], off[1:]), ks):
        o = np.lexsort((np.arange(b - a), -s[a:b])); f[a + o[:min(k, b - a)]] = True
    return f


fl = {'GAP': topk(s, [gapk(s[a:b]) for a, b in zip(off[:-1], off[1:])]),
      'K2': topk(s, [2] * A), 'Q80': s > np.quantile(s, .8)}
aid = np.repeat(np.arange(A), n)


def conf(f):  # per-group TP, FP, TN, FN with POSITIVE = valid step
    parts = [~y & ~f & sm, y & ~f & sm, y & f & sm, ~y & f & sm]
    return np.stack([np.bincount(ginv[aid], weights=q.astype(float), minlength=len(gu)) for q in parts], 1)


def ps(c):
    tp, fp, tn, fn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    return 50 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn))


C = {k: conf(v) for k, v in fl.items()}
rng = np.random.default_rng(7); W = rng.multinomial(len(gu), np.full(len(gu), 1 / len(gu)), size=4000).astype(float)
out = {'n_groups': int(len(gu))}
for a_, b_ in [('GAP', 'Q80'), ('GAP', 'K2')]:
    d0 = float(ps(C[a_].sum(0)) - ps(C[b_].sum(0))); db = ps(W @ C[a_]) - ps(W @ C[b_])
    out[f'{a_}-{b_}'] = [d0, float(np.quantile(db, .025)), float(np.quantile(db, .975))]
# BOOST arm Q80 count vs length
A_ = dict(np.load(ROOT / 'results/prmbench_paper_metric_parity_v1/ARM_SCORES.npz'))
with np.load(ROOT / 'results/remote_claims_replay_2026-10-15/inputs/unpacked/results/prm_boosting_v1/CHANNEL_CONDCORR_prm_backbone.npz', allow_pickle=True) as q:
    offb = q['offsets'].astype(np.int64); idsb = q['idx'].astype(str)
assert np.array_equal(offb, off) and np.array_equal(idsb, ids)
sb = A_['CHAIN_N16'].astype(float)
zb = np.empty(len(sb))
for a, b in zip(off[:-1], off[1:]):
    x = sb[a:b]; zb[a:b] = (x - x.mean()) / max(x.std(), 1e-8)
kq = np.add.reduceat((zb > np.quantile(zb, .8)).astype(int), off[:-1])
out['BOOST_CHAIN_N16_spearman_Q80count_len_nonc'] = float(spearmanr(kq[nonc], n[nonc]).correlation)
# token-length version for TELE (answer token lengths from population.npz)
with np.load(ROOT / 'results/advisor_completion_2026-10-11/inputs/population.npz', allow_pickle=True) as p:
    atok = dict(zip(p['idx'].astype(str), p['answer_token_lengths'].astype(float)))
at = np.array([atok[i] for i in ids])
kqt = np.add.reduceat(fl['Q80'].astype(int), off[:-1]); cnt = np.add.reduceat(y.astype(int), off[:-1])
out['TELE_LSML_spearman_Q80count_answer_tokens_nonc'] = float(spearmanr(kqt[nonc], at[nonc]).correlation)
out['spearman_truecount_answer_tokens_nonc'] = float(spearmanr(cnt[nonc], at[nonc]).correlation)
(OUT / 'B_BOOTSTRAP.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
