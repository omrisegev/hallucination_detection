"""Re-read the 11 bank11 telemetry channels from raw token-level telemetry with alternative step readouts (2026-11-11).

Input: one raw PRM-backbone telemetry pickle (external_telemetry.pkl / prmbench_telemetry.pkl, restored from the Drive
backup of the AIRCC results with local rclone). Processing mirrors the bank builders exactly:
  * results/prm_boost_external_v1/build_bank_external.py (Socratic, Hard2Verify, ProcessBench): pre-empty steps put
    aside, mask1 (forced token 0 dropped, spectral_utils/first_token_mask.py), empty steps dropped;
  * results/prm_boosting_v1/build_bank.py (PRMBench): mask1, empty steps -> answer column mean.
Token values = spectral_utils.external_generalization.fusion.validate_telemetry(row) (risk-oriented [tokens, 11]),
the same call the frozen bank uses before its top-10 readout.

Readouts per step (all on the risk-oriented token values v of the step, n = len(v)):
  top10        mean of the min(10, n) largest (the current readout; gate against the stored bank)
  mean         mean of all tokens
  top20pct     mean of the max(1, ceil(0.2 n)) largest
  p90          90th percentile (linear interpolation)
  median       median
  frac_above   share of tokens above the cell's global 80th percentile of that channel (all tokens of the cell; label-free)
  top10_drop1  top10 after dropping the step's first token (n > 1; a 1-token step keeps its token)
Empty steps (PRMBench only) are filled with the answer's mean of that readout. No labels are read.

    python extract_readouts.py <raw.pkl> <out.npz> [--bank FEATURES.npz] [--workers 8]
"""
import argparse, json, math, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from spectral_utils.first_token_mask import mask_cache  # noqa: E402
from spectral_utils.external_generalization.fusion import validate_telemetry  # noqa: E402
from spectral_utils.external_generalization._bank11.claude_feature_bank_v1 import FEATURE_NAMES  # noqa: E402

READOUTS = ('top10', 'mean', 'top20pct', 'p90', 'median', 'frac_above', 'top10_drop1')


def safe_row(row):
    s = {k: row[k] for k in ('gen_token_ids', 'token_entropies', 'token_spilled_energies', 'token_logsumexp',
                             'step_token_spans')}
    s['top_k_logprobs'] = row.get('top_k_logprobs') or row.get('top_k_logprobs_raw')
    return s


def tokens(item):
    uid, row = item
    return uid, validate_telemetry(row).astype(np.float64)


def mask_like_builders(raw):
    """build_bank_external.py's pre-empty handling + mask1 (identical to build_bank.py's when no step is pre-empty).
    Entry by entry (raw entries are popped) to keep memory low; only the fields the token bank needs are kept."""
    masked, pre_empty, n_failed = {}, {}, 0
    for k in list(raw):
        e = raw.pop(k)
        if 'failure' in e:
            n_failed += 1; continue
        nchar = [len(t) for t in e.get('steps') or []]
        e = {f: e[f] for f in ('idx', 'gen_token_ids', 'token_entropies', 'token_spilled_energies', 'token_logsumexp',
                               'step_token_spans', 'top_k_logprobs', 'top_k_logprobs_raw') if f in e}
        sp = e['step_token_spans']; emp = [s for s, (lo, hi) in enumerate(sp) if hi <= lo]
        if emp:
            pre_empty[str(e['idx'])] = emp; e['step_token_spans'] = [x for x in sp if x[1] > x[0]]
        m, failures = mask_cache({k: e})
        if failures:
            raise SystemExit(f'mask failure: {failures}')
        m = m[k]
        if emp:
            kept = iter(m['step_token_spans']); full = []
            for s in range(len(sp)):
                if s in emp:
                    at = full[-1][1] if full else 0; full.append((at, at))
                else:
                    full.append(tuple(next(kept)))
            m['step_token_spans'] = full
        m['nchar'] = nchar
        masked[k] = m
    return masked, n_failed, pre_empty


def readouts(V, spans, g80):
    """V: [tokens, 11]; spans: non-empty (a, b). Returns [steps, 11, len(READOUTS)]."""
    out = np.empty((len(spans), V.shape[1], len(READOUTS)))
    for i, (a, b) in enumerate(spans):
        v = V[a:b]; n = b - a
        s = np.sort(v, axis=0)
        out[i, :, 0] = s[-min(10, n):].mean(0)
        out[i, :, 1] = v.mean(0)
        out[i, :, 2] = s[-max(1, math.ceil(.2 * n)):].mean(0)
        out[i, :, 3] = np.percentile(v, 90, axis=0)
        out[i, :, 4] = np.median(v, axis=0)
        out[i, :, 5] = (v > g80).mean(0)
        w = v[1:] if n > 1 else v
        out[i, :, 6] = np.sort(w, axis=0)[-min(10, len(w)):].mean(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--bank', type=Path); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(); t0 = time.perf_counter()
    raw = pickle.load(open(a.raw, 'rb'))
    masked, n_failed, pre_empty = mask_like_builders(raw); del raw
    recs = sorted(masked.values(), key=lambda e: str(e['idx']))
    uids = [str(e['idx']) for e in recs]
    jobs = [(u, safe_row(e)) for u, e in zip(uids, recs)]
    spans_all = [list(e['step_token_spans']) for e in recs]
    texts = [e['nchar'] for e in recs]
    del masked, recs
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        TOK = dict(ex.map(tokens, jobs, chunksize=16))
    del jobs
    allv = np.vstack([TOK[u] for u in uids])
    g80 = np.percentile(allv, 80, axis=0); del allv
    R, NT, NE, OFF, CH, FV = [], [], [], [0], [], []
    for u, sp, tl in zip(uids, spans_all, texts):
        V = TOK.pop(u)
        ne = np.array([b > a_ for a_, b in sp]); nesp = [s for s, k in zip(sp, ne) if k]
        r = np.full((len(sp), V.shape[1], len(READOUTS)), np.nan)
        r[ne] = readouts(V, nesp, g80)
        if (~ne).any():                              # empty step -> answer mean of that readout (bank fill rule)
            r[~ne] = np.nanmean(r[ne], axis=0)
        fv = np.full((len(sp), V.shape[1]), np.nan)  # first token of each step (diagnostic)
        fv[ne] = np.array([V[a_] for a_, _ in nesp])
        R.append(r); FV.append(fv); NT.append(np.array([b - a_ for a_, b in sp])); NE.append(ne); OFF.append(OFF[-1] + len(sp))
        CH.append(np.array(tl if len(tl) == len(sp) else [-1] * len(sp)))
    R = np.concatenate(R).astype(np.float64)
    res = dict(uids=np.array(uids), offsets=np.array(OFF), readouts=R, readout_names=np.array(READOUTS),
               channel_names=np.array(FEATURE_NAMES), ntok=np.concatenate(NT), nonempty=np.concatenate(NE),
               nchar=np.concatenate(CH), first_token=np.concatenate(FV), g80=g80)
    info = dict(raw=str(a.raw), records=len(uids), failed_rows=n_failed, pre_empty_answers=len(pre_empty),
                steps=int(OFF[-1]), seconds=round(time.perf_counter() - t0, 1))
    if a.bank:   # gate: current top-10 readout vs the stored bank values (first 11 columns = bank11)
        F = np.load(a.bank); names = list(F['names'].astype(str)); V = F['values']
        assert names[:11] == list(FEATURE_NAMES)
        bu = F['uids'].astype(str) if 'uids' in F.files and 'cells' not in F.files else None
        if 'cells' in F.files:
            sel = np.flatnonzero(F['cells'].astype(str) == 'prmbench_qwen3_8b'); bu = F['row_ids'].astype(str)[sel]
        else:
            sel = np.arange(len(bu))
        boff = F['offsets']; pos = {u: j for j, u in zip(sel, bu)}
        has_ne = 'nonempty' in F.files and not F['nonempty'].all()
        diffs = []; n_ans = 0
        myoff = np.array(OFF)
        for i, u in enumerate(uids):
            if u not in pos: continue
            j = pos[u]; mine = R[myoff[i]:myoff[i + 1], :, 0]; ne = res['nonempty'][myoff[i]:myoff[i + 1]]
            ref = V[boff[j]:boff[j + 1], :11]
            if has_ne or len(ref) != len(mine):
                mine = mine[ne]
            if len(ref) != len(mine):
                raise SystemExit(f'step count mismatch {u}: bank {len(ref)} mine {len(mine)}')
            diffs.append(np.abs(mine - ref).max(0)); n_ans += 1
        d = np.max(diffs, axis=0)
        info['gate'] = dict(bank=str(a.bank), answers=n_ans, max_abs_diff=dict(zip(FEATURE_NAMES, map(float, d))),
                            max_abs_diff_all=float(d.max()), status='PASS' if d.max() <= 1e-6 else 'FAIL')
        print('GATE', info['gate']['status'], d.max(), n_ans, flush=True)
    res['info'] = np.array(json.dumps(info))
    np.savez_compressed(a.out, **res)
    print('DONE', json.dumps(info)[:400], flush=True)


if __name__ == '__main__':
    main()
