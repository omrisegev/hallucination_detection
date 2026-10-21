"""Shared label-free pieces for the gap-significance study (DECLARATION.md). No labels are read here."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import sys
from pathlib import Path
import numpy as np

ROOT = Path('/Users/osegev/Desktop/hallucination_detection')
EC_DIR = ROOT / 'results/error_count_label_free_v1'
sys.path.insert(0, str(EC_DIR))
HERE = Path(__file__).resolve().parent
OLD = Path('/private/tmp/claude-501/-Users-osegev-Desktop-hallucination-detection/98d49308-5ca3-48f7-a452-63da0bb7d2a1/scratchpad')
BOOST_NPZ = OLD / 'error_count/BOOST_SCORES.npz'
OLD_FLAGS = OLD / 'error_count/FLAGS.npz'
SOC_DIR = OLD / 'nonneg/signed'
SOC_GOLD = ROOT / 'results/remote_claims_replay_2026-10-15/inputs/socratic_gold.json'
SOC_REF = ROOT / 'results/sml_weight_estimator_v1/socratic_sign/RESULTS_socratic_signed.csv'
M_NULL = 50_000
NULL_NPZ = HERE / 'NULL.npz'


def window_K(n):
    return min(int(np.ceil(n / 2)), n - 1)


def gap_stats_matrix(Z):
    """Z: (M, n) answer-z rows. Returns G1, k*, G2, Gmed, R1, R2 per row (n >= 3)."""
    M, n = Z.shape
    s = -np.sort(-Z, axis=1)
    g = s[:, :-1] - s[:, 1:]
    K = window_K(n)
    d = g[:, :K]
    kstar = np.argmax(d, axis=1) + 1
    G1 = d[np.arange(M), kstar - 1]
    G2 = np.sort(d, axis=1)[:, -2] if K >= 2 else np.full(M, np.nan)
    Gmed = np.median(g, axis=1)
    R1 = G1 / np.maximum(Gmed, 1e-12)
    R2 = G1 / np.maximum(G2, 1e-12)
    return G1, kstar, G2, Gmed, R1, R2


def build_null(ns, M=M_NULL, seed=20261021):
    """Per-n null samples (sorted) of G1, R1, R2 for n iid N(0,1) draws standardized like answer_z."""
    out = {}
    for n in sorted(set(int(v) for v in ns if v >= 3)):
        rng = np.random.default_rng(seed + n)
        X = rng.standard_normal((M, n))
        Z = (X - X.mean(1, keepdims=True)) / np.maximum(X.std(1, keepdims=True), 1e-8)
        G1, _, _, _, R1, R2 = gap_stats_matrix(Z)
        out[f'G1_{n}'] = np.sort(G1).astype(np.float64)
        out[f'R1_{n}'] = np.sort(R1).astype(np.float64)
        out[f'R2_{n}'] = np.sort(R2).astype(np.float64)
    return out


def load_null(ns):
    need = sorted(set(int(v) for v in ns if v >= 3))
    have = {}
    if NULL_NPZ.exists():
        with np.load(NULL_NPZ) as z:
            have = {k: z[k] for k in z.files}
    missing = [n for n in need if f'G1_{n}' not in have]
    if missing:
        have.update(build_null(missing))
        np.savez_compressed(NULL_NPZ, **have)
    return have


def pval(null_sorted, x):
    """(1 + #{null >= x}) / (M + 1)."""
    M = len(null_sorted)
    ge = M - np.searchsorted(null_sorted, x, side='left')
    return (1 + ge) / (M + 1)


def answer_quantities(z, off, null):
    """Per answer: n, G1, k*, G2, Gmed, R1, R2, p_null, p_R1, p_R2, q95s. n<=2 -> nan stats (k* = 1)."""
    A = len(off) - 1
    cols = {k: np.full(A, np.nan) for k in ('G1', 'G2', 'Gmed', 'R1', 'R2', 'p_null', 'p_R1', 'p_R2',
                                            'q95_G1', 'q95_R1', 'q95_R2')}
    n = np.diff(off).astype(int); kstar = np.ones(A, int)
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        m = b - a
        if m <= 2:
            continue
        G1, ks, G2, Gmed, R1, R2 = gap_stats_matrix(z[a:b][None, :])
        kstar[j] = ks[0]
        cols['G1'][j], cols['G2'][j], cols['Gmed'][j], cols['R1'][j], cols['R2'][j] = G1[0], G2[0], Gmed[0], R1[0], R2[0]
        for q in ('G1', 'R1', 'R2'):
            ns_ = null[f'{q}_{m}']
            cols[f'p_{"null" if q == "G1" else q}'][j] = pval(ns_, cols[q][j])
            cols[f'q95_{q}'][j] = np.quantile(ns_, .95)
    cols['n'] = n; cols['kstar'] = kstar
    return cols


def pooled_null_quantile(ns, null, q=.95):
    """q-quantile of the mixture of per-n null G1 distributions, weighted by the number of answers with each n."""
    ns = np.asarray([v for v in ns if v >= 3], int)
    vals, cnts = np.unique(ns, return_counts=True); w = cnts / cnts.sum()
    lo, hi = 0.0, max(null[f'G1_{v}'][-1] for v in vals)
    F = lambda t: sum(wi * np.searchsorted(null[f'G1_{v}'], t, side='right') / len(null[f'G1_{v}']) for v, wi in zip(vals, w))
    for _ in range(100):
        mid = (lo + hi) / 2
        if F(mid) < q: lo = mid
        else: hi = mid
    return hi


def topk_flags(score, off, k):
    k = np.broadcast_to(np.asarray(k, int), (len(off) - 1,))
    f = np.zeros(off[-1], bool)
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        kk = int(min(max(k[j], 0), b - a))
        if kk:
            o = np.argsort(-score[a:b], kind='stable')[:kk]
            f[a + o] = True
    return f


def policy_counts(Q, pass_mask, fallback, q80_count):
    k = Q['kstar'].copy()
    fb = {'F0': np.zeros_like(k), 'FQ': q80_count, 'F1': np.ones_like(k)}[fallback]
    return np.where(pass_mask, k, fb)


def tests(Q):
    t = Q['n'] >= 3
    with np.errstate(invalid='ignore'):
        return {'T1': t & (Q['p_null'] < .05), 'T2': t & (Q['R1'] > Q['q95_R1']), 'T3': t & (Q['R2'] > Q['q95_R2'])}
