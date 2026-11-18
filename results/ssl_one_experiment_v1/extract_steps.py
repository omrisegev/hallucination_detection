"""Step table of one telemetry run for the one SSL experiment (2026-11-18). CPU. Runs on AIRCC after each GPU run.

Token side = results/bench_gap_diagnosis_v1/readout_alternatives/extract_readouts.py (gate PASS on 16 stored cells):
mask1 (forced token 0 dropped), pre-empty steps put aside as the bank builders do, token values =
spectral_utils.external_generalization.fusion.validate_telemetry(row) (risk-oriented, [tokens, 11]).
Per step and token channel this file keeps the TOPK largest token values, sorted high -> low (NaN padded), plus the
exact top10 / mean / median / p90 over all tokens of the step. The soft top-N readout of stage S needs only these.
Step side = the frozen bank FEATURES.npz of the same run (the 5 step channels realized_z, realized_drv,
digit_alternative, digit_spread, digit_alternative_innovation, and digit_active).
Gate: exact top10 here == the bank's first 11 columns (max abs diff <= 1e-6), as in extract_readouts.py.
No labels are read.

    python extract_steps.py <raw.pkl> <bank FEATURES.npz> <out.npz> [--workers 16] [--topk 48]
Output npz: uids (A,), offsets (A+1,), nonempty (S,), ntok (S,), nchar (S,), top (S, 11, TOPK) float32,
            stats (S, 11, 4) float32 [top10, mean, median, p90], stat_names, step5 (S, 5) float32, digit_active (S, 3),
            tok_names (11,), step_names (5,), info (json: gate, counts).
Empty steps (PRMBench only): nonempty False, top/stats NaN, step5 from the bank if the bank has the row.
"""
import argparse, json, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('HD_ROOT', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'results/bench_gap_diagnosis_v1/readout_alternatives'))
from extract_readouts import mask_like_builders, safe_row  # noqa: E402  (unchanged, gate-passed code)
from spectral_utils.external_generalization.fusion import validate_telemetry  # noqa: E402
from spectral_utils.external_generalization._bank11.claude_feature_bank_v1 import FEATURE_NAMES  # noqa: E402

STEP5 = ['realized_z', 'realized_drv', 'digit_alternative', 'digit_spread', 'digit_alternative_innovation']
STATS = ['top10', 'mean', 'median', 'p90']


def tokens(item):
    uid, row = item
    return uid, validate_telemetry(row).astype(np.float64)


def step_block(V, spans, topk):
    S = len(spans); top = np.full((S, V.shape[1], topk), np.nan, np.float32); st = np.full((S, V.shape[1], 4), np.nan, np.float32)
    for i, (a, b) in enumerate(spans):
        if b <= a: continue
        v = V[a:b]; s = -np.sort(-v, axis=0); n = b - a
        top[i, :, :min(n, topk)] = s[:topk].T
        st[i, :, 0] = s[:min(10, n)].mean(0); st[i, :, 1] = v.mean(0); st[i, :, 2] = np.median(v, axis=0)
        st[i, :, 3] = np.percentile(v, 90, axis=0)
    return top, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw', type=Path); ap.add_argument('bank', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--topk', type=int, default=48)
    a = ap.parse_args(); t0 = time.perf_counter()
    raw = pickle.load(open(a.raw, 'rb'))
    masked, n_failed, pre_empty = mask_like_builders(raw); del raw
    recs = sorted(masked.values(), key=lambda e: str(e['idx']))
    uids = [str(e['idx']) for e in recs]
    jobs = [(u, safe_row(e)) for u, e in zip(uids, recs)]
    spans_all = [list(e['step_token_spans']) for e in recs]; nchar = [e['nchar'] for e in recs]
    del masked, recs
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        TOK = dict(ex.map(tokens, jobs, chunksize=16))
    del jobs
    TOP, ST, NT, NE, NC, OFF = [], [], [], [], [], [0]
    for u, sp, nc in zip(uids, spans_all, nchar):
        V = TOK.pop(u); top, st = step_block(V, sp, a.topk)
        TOP.append(top); ST.append(st); NT.append([b - x for x, b in sp]); NE.append([b > x for x, b in sp])
        NC.append(nc if len(nc) == len(sp) else [-1] * len(sp)); OFF.append(OFF[-1] + len(sp))
    top = np.concatenate(TOP); st = np.concatenate(ST); ne = np.concatenate(NE).astype(bool); off = np.array(OFF)
    # bank: 5 step channels + digit_active, aligned like extract_readouts.py's gate
    F = np.load(a.bank); names = list(F['names'].astype(str)); BV = F['values']; act = F['digit_active'].astype(bool)
    assert names[:11] == list(FEATURE_NAMES), names[:11]
    bu = (F['uids'] if 'uids' in F.files else F['row_ids']).astype(str); boff = F['offsets'].astype(np.int64)  # adapt key names only
    pos = {x: j for j, x in enumerate(bu)}
    step5 = np.full((len(ne), 5), np.nan, np.float32); dact = np.zeros((len(ne), 3), bool)
    diffs, n_ans, miss = [], 0, []
    for i, u in enumerate(uids):
        if u not in pos: miss.append(u); continue
        j = pos[u]; lo, hi = off[i], off[i + 1]; rows = np.arange(boff[j], boff[j + 1])
        sel = np.arange(lo, hi)
        if len(rows) != hi - lo:
            sel = sel[ne[lo:hi]]
        if len(rows) != len(sel):
            raise SystemExit(f'step count mismatch {u}: bank {len(rows)} here {len(sel)}')
        step5[sel] = BV[rows][:, [names.index(n) for n in STEP5]]; dact[sel] = act[rows]
        ok = ne[sel]
        diffs.append(np.abs(st[sel[ok], :, 0] - BV[rows[ok], :11]).max(0) if ok.any() else np.zeros(11)); n_ans += 1
    d = np.max(diffs, axis=0) if diffs else np.full(11, np.nan)
    info = dict(raw=str(a.raw), bank=str(a.bank), records=len(uids), failed_rows=n_failed, pre_empty_answers=len(pre_empty),
                steps=int(off[-1]), empty_steps=int((~ne).sum()), topk=a.topk, answers_not_in_bank=len(miss),
                gate=dict(answers=n_ans, max_abs_diff=dict(zip(FEATURE_NAMES, map(float, d))), max_abs_diff_all=float(np.nanmax(d)),
                          status='PASS' if n_ans == len(uids) and np.nanmax(d) <= 1e-6 else 'FAIL'),
                seconds=round(time.perf_counter() - t0, 1))
    np.savez_compressed(a.out, uids=np.array(uids), offsets=off, nonempty=ne, ntok=np.concatenate(NT), nchar=np.concatenate(NC),
                        top=top, stats=st, stat_names=np.array(STATS), step5=step5, digit_active=dact,
                        tok_names=np.array(FEATURE_NAMES), step_names=np.array(STEP5), info=np.array(json.dumps(info)))
    print('GATE', info['gate']['status'], info['gate']['max_abs_diff_all'], n_ans, len(uids), flush=True)
    print('DONE', json.dumps({k: v for k, v in info.items() if k != 'gate'}), flush=True)
    if info['gate']['status'] != 'PASS':
        sys.exit(3)


if __name__ == '__main__':
    main()
