"""Shared loaders + PRMScore evaluator for the label-free error-count study (CPU, existing data only).

Telemetry-only line: thinkfix_mask1 bank (results/thinkfix_rescore_v1/prmscore_inputs, mask1 per its README/HANDOFF).
Boosting line: Qwen2.5-Math-PRM-7B CHAIN_N16 via results/prmbench_paper_metric_parity_v1/common.py (exact parity).
Labels (error_steps, 1-based) are read ONLY by `labels()` and the metric functions.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path('/Users/osegev/Desktop/hallucination_detection')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'results/prmbench_paper_metric_parity_v1'))
BANK_DIR = ROOT / 'results/thinkfix_rescore_v1/prmscore_inputs'
RAW = ROOT / 'results/remote_claims_replay_2026-10-15/inputs/unpacked/results/prm_boosting_v1'
OUT = Path(__file__).resolve().parent
N16 = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level',
       'energy_innovation', 'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0',
       'realized_z', 'realized_drv', 'digit_alternative', 'digit_spread', 'digit_alternative_innovation']


def load_meta():
    m = json.loads((BANK_DIR / 'prmbench_meta.json').read_text())['rows']
    return {r['idx']: r for r in m}


def load_tele_bank(bank='N16'):
    z = np.load(BANK_DIR / f'bank_{bank}.npz', allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d['idx'] = d['idx'].astype(str)
    return d


def load_tele_channels(chans=N16):
    """mask1 Qwen3-8B per-step channels (answer-z). The exported matrix is label-oriented by the atlas sign;
    undo it (as prmbench_paper_metric_parity_v1/common.py does) so nothing label-derived enters a fit."""
    import csv
    with np.load(RAW / 'CHANNEL_CONDCORR_qwen3_8b_mask1.npz', allow_pickle=True) as z:
        names = z['names'].astype(str).tolist(); X = z['S'].astype(float)
        off = z['offsets'].astype(np.int64); ids = z['idx'].astype(str)
    signs = {r['channel']: float(r['sign']) for r in csv.DictReader((RAW / 'CHANNEL_ATLAS_qwen3_8b_mask1.csv').open())}
    X = X * np.array([signs[c] for c in names])
    return X[:, [names.index(c) for c in chans]], off, ids


def labels(ids, off, meta):
    """Per-step error flags (in-range 1-based error_steps), per-answer true count, control mask, class, category."""
    y = np.zeros(off[-1], bool)
    for j, i in enumerate(ids):
        a, b = off[j], off[j + 1]
        y[a:b] = np.isin(np.arange(1, b - a + 1), meta[i]['error_steps'])
        assert b - a == meta[i]['n_steps']
    cnt = np.add.reduceat(y.astype(int), off[:-1])
    cls = np.array([meta[i]['classification'] for i in ids])
    cat = np.array([meta[i]['category'] for i in ids])
    return y, cnt, cls != 'correct', cls, cat


def answer_z(score, off):
    out = np.zeros(len(score))
    for a, b in zip(off[:-1], off[1:]):
        s = score[a:b]
        out[a:b] = (s - s.mean()) / max(s.std(), 1e-8)
    return out


def q_flags(score, off, q=.8, already_z=False):
    z = score if already_z else answer_z(score, off)
    return z > np.quantile(z, q)


def topk_flags(score, off, k):
    """Flag the k_a highest-scoring steps of answer a (stable order, ties by position)."""
    k = np.broadcast_to(np.asarray(k, int), (len(off) - 1,))
    f = np.zeros(off[-1], bool)
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        kk = int(min(max(k[j], 0), b - a))
        if kk:
            o = np.argsort(-score[a:b], kind='stable')[:kk]
            f[a + o] = True
    return f


def counts_of(flags, off):
    return np.add.reduceat(flags.astype(int), off[:-1])


def prmscore(y, flag, step_mask):
    """Official PRMBench pooled macro-F1, POSITIVE = valid step; synthetic controls excluded via step_mask."""
    m = step_mask
    tp = np.count_nonzero(~y & ~flag & m); fp = np.count_nonzero(y & ~flag & m)
    tn = np.count_nonzero(y & flag & m); fn = np.count_nonzero(~y & flag & m)
    f1p = 2 * tp / (2 * tp + fp + fn) if tp else 0.
    f1n = 2 * tn / (2 * tn + fp + fn) if tn else 0.
    return 100 * .5 * (f1p + f1n)


def evaluate(y, cnt, nonc, off, flags, cls=None):
    """PRMScore + count error on noncontrol answers. Count = number of flagged steps per answer."""
    sm = np.repeat(nonc, np.diff(off))
    k = counts_of(flags, off)
    ps = prmscore(y, flags, sm)
    mae = float(np.mean(np.abs(k[nonc] - cnt[nonc])))
    zt = cnt[nonc] == 0; zp = k[nonc] == 0
    out = dict(prmscore=ps, count_mae=mae, mean_pred=float(k[nonc].mean()), mean_true=float(cnt[nonc].mean()),
               flag_share=float(flags[sm].mean()), zero_acc=float(np.mean(zt == zp)),
               zero_recall=float(zp[zt].mean()) if zt.any() else np.nan,
               nonzero_recall=float((~zp)[~zt].mean()),
               spearman_count=float(_spearman(k[nonc], cnt[nonc])))
    return out


def _spearman(a, b):
    from scipy.stats import spearmanr
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return spearmanr(a, b).correlation


def per_class(y, cnt, nonc, off, flags, cls):
    sm_all = np.repeat(nonc, np.diff(off)); step_cls = np.repeat(cls, np.diff(off))
    k = counts_of(flags, off); res = {}
    for c in sorted(set(cls[nonc])):
        am = (cls == c) & nonc; sm = sm_all & (step_cls == c)
        res[c] = dict(prmscore=prmscore(y, flags, sm), mean_true=float(cnt[am].mean()),
                      mean_pred=float(k[am].mean()), mae=float(np.mean(np.abs(k[am] - cnt[am]))), n=int(am.sum()))
    return res
