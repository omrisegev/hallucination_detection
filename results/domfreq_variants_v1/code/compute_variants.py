"""domfreq_variants_v1 stage 1: per-step variant values for one (benchmark, PRM) cell. Label-free.

    python3 compute_variants.py --bench prmbench|socratic --tag <prm> --tele <pkl> --feat <FEATURES.npz> --out <dir>

Steps
  1. Load the raw telemetry, apply the frozen mask1 rule (code_prm_boost_v1/mask_fixed first_token_mask.py,
     the version that keeps already-empty source steps empty).
  2. Per answer (in FEATURES.npz order), drop empty steps (as the bank builders do) and rebuild the frozen
     11-channel token matrix with the bank's own code (code_thinkfix_bank_553de79, _bank11
     build_token_feature_matrix, float32, risk-oriented). STEP 0: dominant_freq16 step value = top10 of the
     oriented token column (bank rule) and, for comparison, the plain mean; both are compared later to the
     stored FEATURES column.
  3. Variants from six token series (surprisal, margin, logsumexp energy, top15 turnover, top50 JS, and the
     saved token entropy as REFERENCE), see VARIANTS below. No labels are read here.
Output: <out>/CACHE_<bench>_<tag>.npz with V [steps, n_variants] float32 (rows = nonempty steps in FEATURES
order), names, fallback mask, dom_top10, dom_mean, step_len, step_index, n_steps, answer offsets, ids.
"""
import os
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_n] = '1'
import argparse, importlib.util, json, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

S = Path('/shared/cycle2_tau_averbuch_prj/omrisegev1')
FROZEN = S / 'code_thinkfix_bank_553de79'
MASK = S / 'code_prm_boost_v1/mask_fixed/spectral_utils/first_token_mask.py'
sys.path.insert(0, str(FROZEN))
from spectral_utils.external_generalization._bank11.claude_feature_bank_v1 import (  # noqa: E402
    build_token_feature_matrix, FEATURE_NAMES, FEATURE_RISK_SIGNS)
from spectral_utils.external_generalization.fusion import top10  # noqa: E402

SERIES = ('surprisal', 'margin', 'energy', 'turnover', 'js', 'entropyREF')
WINDOWS = (8, 16, 32)
DETR = ('mc', 'lin')
SPEC_STATS = ('slope', 'low10', 'low20', 'sent', 'flat', 'cent')
TIME_STATS = ('lag1', 'lag2')
PAIRS = (('surprisal', 'energy'), ('margin', 'turnover'))
COH_STATS = ('msc_low', 'msc_high', 'phase_low', 'phase_high')
MIN_STEP = 4          # step-aligned spectral / autocorr need >= 4 tokens
MIN_STEP_DFA = 16     # step-aligned DFA needs two scales (4, 8)
MIN_STEP_COH = 16     # step-aligned coherence: >= 3 Welch segments of 8 tokens, hop 4
EPS = 1e-12


def variant_names():
    out = []
    for s in SERIES:
        for w in WINDOWS:
            for d in DETR:
                for st in SPEC_STATS + TIME_STATS:
                    if st == 'low10' and w == 8: continue        # no nonzero rfft bin <= 0.1 at W=8
                    for ag in ('mean', 'last'): out.append(f'{s}|{st}|w{w}|{d}|{ag}')
        for w in (16, 32):
            for ag in ('mean', 'last'): out.append(f'{s}|dfa|w{w}|-|{ag}')
        for d in DETR:
            for st in SPEC_STATS + TIME_STATS: out.append(f'{s}|{st}|step|{d}|-')
        out.append(f'{s}|dfa|step|-|-')
    for a, b in PAIRS:
        for st in COH_STATS:
            for w in (16, 32):
                for ag in ('mean', 'last'): out.append(f'{a}~{b}|{st}|w{w}|-|{ag}')
            out.append(f'{a}~{b}|{st}|step|-|-')
    return out


NAMES = variant_names()
INDEX = {n: i for i, n in enumerate(NAMES)}


def detrend(Y, kind):
    """Y [m, W]: mean-center, or remove the least-squares line, along axis 1."""
    Y = Y - Y.mean(1, keepdims=True)
    if kind == 'lin':
        t = np.arange(Y.shape[1], dtype=float); t -= t.mean()
        Y = Y - np.outer(Y @ t / (t @ t), t)
    return Y


def spec_stats(Y, nfft=None):
    """Continuous spectral statistics of each row of Y (already detrended). AC bins only (f > 0)."""
    W = Y.shape[1]; nfft = nfft or W
    P = np.abs(np.fft.rfft(Y, n=nfft, axis=1)[:, 1:]) ** 2 + EPS
    f = np.fft.rfftfreq(nfft)[1:]
    tot = P.sum(1)
    pn = P / tot[:, None]
    lf = np.log(f); lfc = lf - lf.mean(); lP = np.log(P)
    out = {'slope': (lP - lP.mean(1, keepdims=True)) @ lfc / (lfc @ lfc)}
    out['low10'] = P[:, f <= 0.1].sum(1) / tot if (f <= 0.1).any() else np.full(len(Y), np.nan)
    out['low20'] = P[:, f <= 0.2].sum(1) / tot
    out['sent'] = -(pn * np.log(pn)).sum(1) / np.log(len(f))
    out['flat'] = np.exp(lP.mean(1)) / P.mean(1)
    out['cent'] = pn @ f            # = power-weighted mean frequency (identical definition, kept once)
    return out


def acf(Y):
    d = (Y * Y).sum(1)
    ok = d > EPS
    l1 = np.where(ok, (Y[:, 1:] * Y[:, :-1]).sum(1) / np.where(ok, d, 1), 0.)
    l2 = np.where(ok, (Y[:, 2:] * Y[:, :-2]).sum(1) / np.where(ok, d, 1), 0.)
    return {'lag1': l1, 'lag2': l2}


def dfa(X, scales):
    """DFA-1 exponent of each row of X: profile, non-overlapping segments, linear detrend, slope of log F(s)."""
    prof = np.cumsum(X - X.mean(1, keepdims=True), 1)
    logF = []
    for s in scales:
        k = prof.shape[1] // s
        seg = prof[:, :k * s].reshape(len(prof), k, s)
        t = np.arange(s, dtype=float); t -= t.mean()
        seg = seg - seg.mean(2, keepdims=True)
        seg = seg - (seg @ t / (t @ t))[:, :, None] * t
        logF.append(0.5 * np.log((seg ** 2).mean((1, 2)) + EPS))
    logF = np.array(logF).T; ls = np.log(scales); ls = ls - ls.mean()
    return (logF - logF.mean(1, keepdims=True)) @ ls / (ls @ ls)


def welch_coh(X, Y, seg=8, hop=4):
    """Welch cross-spectral statistics for each row pair (rows = windows). Hann segments, mean-centred."""
    m, W = X.shape
    starts = np.arange(0, W - seg + 1, hop)
    h = np.hanning(seg + 2)[1:-1]
    Xs = np.stack([X[:, a:a + seg] for a in starts], 1); Ys = np.stack([Y[:, a:a + seg] for a in starts], 1)
    Xs = (Xs - Xs.mean(2, keepdims=True)) * h; Ys = (Ys - Ys.mean(2, keepdims=True)) * h
    FX = np.fft.rfft(Xs, axis=2)[:, :, 1:]; FY = np.fft.rfft(Ys, axis=2)[:, :, 1:]   # f = .125 .25 .375 .5
    Sxx = (np.abs(FX) ** 2).mean(1); Syy = (np.abs(FY) ** 2).mean(1); Sxy = (FX * np.conj(FY)).mean(1)
    den = Sxx * Syy
    msc = np.where(den > EPS, np.abs(Sxy) ** 2 / np.where(den > EPS, den, 1), np.nan)
    lo, hi = slice(0, 2), slice(2, 4)
    out = {'msc_low': msc[:, lo].mean(1), 'msc_high': msc[:, hi].mean(1)}
    for k, b in (('phase_low', lo), ('phase_high', hi)):
        z = Sxy[:, b].sum(1)
        out[k] = np.where(np.abs(z) > EPS, np.angle(z), np.nan)
    return out


def rolling_windows(x, W):
    """Causal windows ending at every token. Tokens before the first full window reuse the first window
    (backfill, as the bank's _causal does); an answer shorter than W uses the whole answer as one window."""
    n = len(x)
    if n >= W:
        Wm = swv(x, W)                                    # window i ends at token i+W-1
        idx = np.maximum(np.arange(n) - (W - 1), 0)
        return Wm, idx
    return x[None, :], np.zeros(n, int)


def agg(tok, spans):
    a = spans[:, 0]; b = spans[:, 1]
    cs = np.r_[0., np.cumsum(np.where(np.isfinite(tok), tok, 0.))]
    cn = np.r_[0, np.cumsum(np.isfinite(tok))]
    cnt = cn[b] - cn[a]
    mean = np.where(cnt > 0, (cs[b] - cs[a]) / np.maximum(cnt, 1), np.nan)
    return mean, tok[b - 1]


def answer_variants(series, spans):
    n_steps = len(spans); V = np.full((n_steps, len(NAMES)), np.nan); n = len(series['surprisal'])
    # rolling-window statistics, token level -> step mean / last
    for s in SERIES:
        x = series[s]
        for w in WINDOWS:
            Wm, idx = rolling_windows(x, w)
            short = Wm.shape[1] < 4
            for d in DETR:
                Y = detrend(Wm, d)
                st = {} if short else {**spec_stats(Y), **acf(Y)}
                for k in SPEC_STATS + TIME_STATS:
                    if k == 'low10' and w == 8: continue
                    tok = st[k][idx] if k in st else np.full(n, np.nan)
                    m, l = agg(tok, spans)
                    V[:, INDEX[f'{s}|{k}|w{w}|{d}|mean']] = m; V[:, INDEX[f'{s}|{k}|w{w}|{d}|last']] = l
            if w in (16, 32):
                tok = dfa(Wm, (4, 8) if w == 16 else (4, 8, 16))[idx] if Wm.shape[1] >= 16 else np.full(n, np.nan)
                m, l = agg(tok, spans)
                V[:, INDEX[f'{s}|dfa|w{w}|-|mean']] = m; V[:, INDEX[f'{s}|dfa|w{w}|-|last']] = l
        # step-aligned: the step's own tokens; zero-padded periodogram, nfft = max(16, next pow2)
        for j, (a, b) in enumerate(spans):
            L = b - a
            if L >= MIN_STEP:
                seg = x[a:b][None, :]; nfft = max(16, 1 << int(np.ceil(np.log2(L))))
                for d in DETR:
                    Y = detrend(seg, d); st = {**spec_stats(Y, nfft), **acf(Y)}
                    for k in SPEC_STATS + TIME_STATS: V[j, INDEX[f'{s}|{k}|step|{d}|-']] = st[k][0]
            if L >= MIN_STEP_DFA:
                sc = tuple(q for q in (4, 8, 16) if q <= L // 2)
                V[j, INDEX[f'{s}|dfa|step|-|-']] = dfa(x[a:b][None, :], sc)[0]
    for pa, pb in PAIRS:
        x, y = series[pa], series[pb]
        for w in (16, 32):
            Wx, idx = rolling_windows(x, w); Wy, _ = rolling_windows(y, w)
            st = welch_coh(Wx, Wy) if Wx.shape[1] >= 16 else None
            for k in COH_STATS:
                tok = st[k][idx] if st is not None else np.full(n, np.nan)
                m, l = agg(tok, spans)
                V[:, INDEX[f'{pa}~{pb}|{k}|w{w}|-|mean']] = m; V[:, INDEX[f'{pa}~{pb}|{k}|w{w}|-|last']] = l
        for j, (a, b) in enumerate(spans):
            if b - a >= MIN_STEP_COH:
                st = welch_coh(x[a:b][None, :], y[a:b][None, :])
                for k in COH_STATS: V[j, INDEX[f'{pa}~{pb}|{k}|step|-|-']] = st[k][0]
    fb = ~np.isfinite(V)
    for c in np.flatnonzero(fb.any(0)):                    # bank fill rule: answer column mean, else 0
        g = ~fb[:, c]; V[fb[:, c], c] = V[g, c].mean() if g.any() else 0.
    return V, fb


def one(item):
    i, row = item
    sp = [tuple(map(int, s)) for s in row['step_token_spans']]
    keep = [k for k, s in enumerate(sp) if s[1] > s[0]]
    spans = np.array([sp[k] for k in keep], int)
    M = build_token_feature_matrix(row)                       # [T, 11] float32, risk-oriented
    sign = np.array([FEATURE_RISK_SIGNS[c] for c in FEATURE_NAMES], float)
    raw = M.astype(float) * sign
    c = {nm: raw[:, FEATURE_NAMES.index(nm)] for nm in FEATURE_NAMES}
    series = {'surprisal': c['chosen_surprisal'], 'margin': c['logprob_margin'], 'energy': c['energy_level'],
              'turnover': c['top15_turnover'], 'js': c['top50_js'],
              'entropyREF': np.asarray(row['token_entropies'], float)}
    dcol = M[:, FEATURE_NAMES.index('dominant_freq16')].astype(float)
    dom_top10 = top10(dcol, spans); dom_mean = np.array([dcol[a:b].mean() for a, b in spans])
    V, fb = answer_variants(series, spans)
    return i, keep, len(sp), V.astype(np.float32), fb, dom_top10, dom_mean, spans[:, 1] - spans[:, 0], len(dcol), spans[:, 0]


def safe_row(row):
    r = {k: row[k] for k in ('gen_token_ids', 'token_entropies', 'token_spilled_energies', 'token_logsumexp',
                             'step_token_spans')}
    r['top_k_logprobs'] = row.get('top_k_logprobs') or row.get('top_k_logprobs_raw')
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', choices=('prmbench', 'socratic'), required=True)
    ap.add_argument('--tag', required=True); ap.add_argument('--tele', type=Path, required=True)
    ap.add_argument('--feat', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); t0 = time.perf_counter()
    spec = importlib.util.spec_from_file_location('ftm', MASK); FM = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(FM)
    raw = pickle.load(open(a.tele, 'rb'))
    masked, failures = FM.mask_cache(raw)
    if failures: raise SystemExit(f'mask failures: {failures[:3]}')
    lookup = {str(e['idx']): e for e in masked.values()}; del raw
    F = np.load(a.feat)
    if a.bench == 'prmbench':
        cells = F['cells'].astype(str); ans = np.flatnonzero(cells == 'prmbench_qwen3_8b')
        ids = F['row_ids'][ans].astype(str)
    else:
        ids = F['uids'].astype(str); ans = np.arange(len(ids))
    if a.limit: ids = ids[:a.limit]
    print(a.bench, a.tag, 'answers', len(ids), 'load+mask', round(time.perf_counter() - t0), 's', flush=True)
    jobs = [(j, safe_row(lookup[i])) for j, i in enumerate(ids)]; del lookup, masked
    res = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, r in enumerate(ex.map(one, jobs, chunksize=8)):
            res[r[0]] = r
            if k % 1000 == 0: print('done', k, round(time.perf_counter() - t0), 's', flush=True)
    keep_counts = np.array([len(r[1]) for r in res]); all_counts = np.array([r[2] for r in res])
    np.savez_compressed(
        a.out / f'CACHE_{a.bench}_{a.tag}.npz', ids=ids, names=np.array(NAMES),
        V=np.concatenate([r[3] for r in res]), fallback=np.concatenate([r[4] for r in res]),
        dom_top10=np.concatenate([r[5] for r in res]), dom_mean=np.concatenate([r[6] for r in res]),
        step_len=np.concatenate([r[7] for r in res]), span_lo=np.concatenate([r[9] for r in res]),
        keep_index=np.concatenate([np.asarray(r[1], int) for r in res]),
        n_steps_all=all_counts, n_steps_kept=keep_counts, n_tokens=np.array([r[8] for r in res]),
        offsets=np.r_[0, np.cumsum(keep_counts)])
    json.dump({'bench': a.bench, 'tag': a.tag, 'tele': str(a.tele), 'feat': str(a.feat), 'answers': len(ids),
               'steps_kept': int(keep_counts.sum()), 'steps_all': int(all_counts.sum()), 'variants': len(NAMES),
               'seconds': round(time.perf_counter() - t0, 1), 'job': os.environ.get('SLURM_JOB_ID', '')},
              open(a.out / f'CACHE_{a.bench}_{a.tag}.json', 'w'), indent=1)
    print('DONE', a.bench, a.tag, round(time.perf_counter() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
