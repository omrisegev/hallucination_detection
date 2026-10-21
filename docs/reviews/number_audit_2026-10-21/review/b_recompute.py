"""Independent reviewer recompute (B1, B2, B3). Does NOT import the error_count agent's code.
PRMScore via the official-evaluator port spectral_utils.prmbench.prmbench_evaluate
(PRMScore = (F1_valid + F1_error)/2 over noncontrol rows; synthetic 'correct' rows excluded by the evaluator).
GAP re-implemented from DECLARATION.md text only.
"""
import json, pickle, sys, csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path('/Users/osegev/Desktop/hallucination_detection')
sys.path.insert(0, str(ROOT))
from spectral_utils.prmbench import prmbench_evaluate  # official port

OUT = Path(__file__).resolve().parent
meta_rows = json.loads((ROOT / 'results/thinkfix_rescore_v1/prmscore_inputs/prmbench_meta.json').read_text())['rows']
META = {r['idx']: r for r in meta_rows}


def prmscore_official(ids, off, flags):
    """flags[t] True = step predicted erroneous. labels: 1 = valid (official POSITIVE)."""
    preds = [{'idx': i, 'labels': [0 if f else 1 for f in flags[a:b]]} for i, a, b in zip(ids, off[:-1], off[1:])]
    md = [dict(idx=i, classification=META[i]['classification'], error_steps=META[i]['error_steps']) for i in ids]
    r = prmbench_evaluate(preds, md)['total']
    return 100 * .5 * (r['f1'] + r['negative_f1'])


def az(s, off):
    o = np.empty(len(s))
    for a, b in zip(off[:-1], off[1:]):
        x = s[a:b]; sd = x.std()
        o[a:b] = (x - x.mean()) / (sd if sd > 1e-8 else 1e-8)
    return o


def q80(s, off, ge=True):
    z = az(s, off); t = np.quantile(z, .8)
    return (z >= t) if ge else (z > t)


def gap_k(x):
    """DECLARATION E1: k = argmax_{k=1..ceil(n/2)} (z_(k) - z_(k+1)), z_(1) the largest; n=1 -> 1."""
    n = len(x)
    if n == 1:
        return 1
    z = -np.sort(-np.asarray(x, float))
    K = int(np.ceil(n / 2))
    gaps = [z[k - 1] - z[k] for k in range(1, K + 1)]   # k is 1-based
    return 1 + int(np.argmax(gaps))


def topk(s, off, ks):
    f = np.zeros(len(s), bool)
    for (a, b), k in zip(zip(off[:-1], off[1:]), ks):
        k = min(int(k), b - a)
        order = np.lexsort((np.arange(b - a), -s[a:b]))   # desc score, ties by position
        f[a + order[:k]] = True
    return f


def labels(ids, off):
    y = np.zeros(off[-1], bool)
    for i, a, b in zip(ids, off[:-1], off[1:]):
        es = META[i]['error_steps']
        for e in es:
            if 1 <= e <= b - a:
                y[a + e - 1] = True
    return y


def within_auc(y, s, off):
    from scipy.stats import rankdata
    v = []
    for a, b in zip(off[:-1], off[1:]):
        yy = y[a:b]; n1 = yy.sum(); n0 = len(yy) - n1
        if n1 and n0:
            v.append((rankdata(s[a:b])[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(v)), len(v)


res = {}
# ---------------- TELE mask1 N16 ----------------
z = np.load(ROOT / 'results/thinkfix_rescore_v1/prmscore_inputs/bank_N16.npz', allow_pickle=False)
off = z['offsets'].astype(np.int64); ids = z['idx'].astype(str); n = np.diff(off)
y = labels(ids, off)
nonc = np.array([META[i]['classification'] != 'correct' for i in ids])
for arm in ['LSML_CONT', 'ALL_EQ_ORIENTED']:
    s = z['S__' + arm].astype(float)
    r = {}
    r['answer_z_idempotent_maxdiff'] = float(np.max(np.abs(az(s, off) - s)))
    r['within_auc'], r['n_auc'] = within_auc(y, s, off)
    r['Q80_ge'] = prmscore_official(ids, off, q80(s, off, True))
    r['Q80_gt'] = prmscore_official(ids, off, q80(s, off, False))
    kg = np.array([gap_k(s[a:b]) for a, b in zip(off[:-1], off[1:])])
    r['GAP'] = prmscore_official(ids, off, topk(s, off, kg))
    r['K2'] = prmscore_official(ids, off, topk(s, off, np.full(len(n), 2)))
    r['K3'] = prmscore_official(ids, off, topk(s, off, np.full(len(n), 3)))
    r['GAP_minus_Q80'] = r['GAP'] - r['Q80_gt']
    r['GAP_minus_K2'] = r['GAP'] - r['K2']
    # B3: count vs answer length (noncontrols)
    kq = np.add.reduceat(q80(s, off, False).astype(int), off[:-1])
    cnt = np.add.reduceat(y.astype(int), off[:-1])
    r['spearman_Q80count_len_nonc'] = float(spearmanr(kq[nonc], n[nonc]).correlation)
    r['spearman_truecount_len_nonc'] = float(spearmanr(cnt[nonc], n[nonc]).correlation)
    r['spearman_GAPcount_len_nonc'] = float(spearmanr(kg[nonc], n[nonc]).correlation)
    r['spearman_Q80count_len_all'] = float(spearmanr(kq, n).correlation)
    r['mean_true_count_nonc'] = float(cnt[nonc].mean()); r['mean_GAP_count_nonc'] = float(kg[nonc].mean())
    r['share_true_count_eq2_nonc'] = float(np.mean(cnt[nonc] == 2))
    res['TELE_' + arm] = r
    np.save(OUT / f'gapk_{arm}.npy', kg)

# compare with agent's GAP counts (if present) -- only as a consistency check
try:
    C = np.load(OUT.parent / 'error_count/COUNTS.npz')
    for arm in ['LSML_CONT', 'ALL_EQ_ORIENTED']:
        res['TELE_' + arm]['gapk_equal_to_agent'] = bool(np.array_equal(np.load(OUT / f'gapk_{arm}.npy'), C[f'TELE_{arm}__E1_GAP']))
except Exception as e:
    res['agent_counts_cmp_error'] = repr(e)

# ---------------- BOOST Qwen-7B (cached cluster/parity arm scores; native from the PRM pkl's own labels) ----------------
with np.load(ROOT / 'results/remote_claims_replay_2026-10-15/inputs/unpacked/results/prm_boosting_v1/CHANNEL_CONDCORR_prm_backbone.npz', allow_pickle=True) as q:
    offb = q['offsets'].astype(np.int64); idsb = q['idx'].astype(str)
A = dict(np.load(ROOT / 'results/prmbench_paper_metric_parity_v1/ARM_SCORES.npz'))
rb = {}
for arm in ['CHAIN_N16', 'CHAIN_N13', 'LSML_CONT']:
    s = A[arm].astype(float)
    rb[arm + '_Q80_gt'] = prmscore_official(idsb, offb, q80(s, offb, False))
    rb[arm + '_Q80_ge'] = prmscore_official(idsb, offb, q80(s, offb, True))
    kg = np.array([gap_k(s[a:b]) for a, b in zip(offb[:-1], offb[1:])])
    rb[arm + '_GAP'] = prmscore_official(idsb, offb, topk(s, offb, kg))
    rb[arm + '_K2'] = prmscore_official(idsb, offb, topk(s, offb, np.full(len(offb) - 1, 2)))
d = pickle.load(open(ROOT / 'dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl', 'rb'))
byid = {v['idx']: v for v in d.values()}
_L = lambda v: json.loads(v) if isinstance(v, str) else list(v)
nat = np.concatenate([np.array(_L(byid[i]['labels'])) == 0 for i in idsb])
rew = np.concatenate([np.array(_L(byid[i]['rewards']), float) for i in idsb])
rb['PRM_native_pkl_labels'] = prmscore_official(idsb, offb, nat)
rb['PRM_native_reward_le_0.5'] = prmscore_official(idsb, offb, rew <= .5)
rb['PRM_native_reward_lt_0.5'] = prmscore_official(idsb, offb, rew < .5)
try:
    Bs = dict(np.load(OUT.parent / 'error_count/BOOST_SCORES.npz'))
    rb['agent_refit_vs_cached_CHAIN_N16_maxabs'] = float(np.max(np.abs(Bs['CHAIN_N16'] - A['CHAIN_N16'])))
except Exception as e:
    rb['agent_cmp_error'] = repr(e)
res['BOOST_qwen7b'] = rb
(OUT / 'B_RESULTS.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
