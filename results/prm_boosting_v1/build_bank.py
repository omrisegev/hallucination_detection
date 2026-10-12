"""Build a PRMBench FEATURES.npz from a Qwen2.5-Math telemetry cell (run_prmbench_prm_telemetry.py),
in the layout fuse.py expects: the think-fix mask1 source bank (13,769 answers, JOINED order) with
the 6,969 PRMBench rows replaced by the new model's features. ProcessBench rows are untouched and
never read by fuse.py (it fits and scores the PRMBench cell only).

Per-answer feature code = the frozen bank builder's own `one()` (scripts/rebuild_source_bank_v4.py at
code_thinkfix_bank_553de79), after the same mask1 rule as the Qwen3 cells (forced token 0 dropped).
Step counts must match the roster; token counts differ by tokenizer path and are not asserted.
No labels are read. Run on AIRCC CPU with cwd = code_thinkfix_bank_553de79:

    python3 build_bank.py --raw <cell>/prmbench_telemetry.pkl --base <mask1 source FEATURES.npz> \
        --mask-code <code_thinkfix_rescore_v1> --out <dir> [--workers 16]
"""
import os
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_n] = '1'
import argparse, hashlib, importlib.util, json, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
import rebuild_source_bank_v4 as RB  # noqa: E402  (importable by name, so worker processes can unpickle RB.one)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def one_safe(item):
    """RB.one on the non-empty steps; empty steps (zero tokens in this tokenization) get NaN rows, filled later
    with the answer's column means (the bank's own fill_missing rule)."""
    i, row = item
    sp = row['step_token_spans']; keep = [k for k, s in enumerate(sp) if s[1] > s[0]]
    if len(keep) == len(sp):
        return RB.one(item)
    r = dict(row); r['step_token_spans'] = [sp[k] for k in keep]
    _, b48, b5, d3, ac, sec = RB.one((i, r))
    def expand(x, fill):
        out = np.full((len(sp),) + x.shape[1:], fill, dtype=x.dtype); out[keep] = x; return out
    return i, expand(b48, np.nan), expand(b5, np.nan), expand(d3, np.nan), expand(ac, False), sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=Path, required=True); ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--mask-code', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--expect-equal', action='store_true')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False); t0 = time.perf_counter()
    mspec = importlib.util.spec_from_file_location('ftm', a.mask_code / 'spectral_utils/first_token_mask.py')
    M = importlib.util.module_from_spec(mspec); mspec.loader.exec_module(M)
    raw = pickle.load(open(a.raw, 'rb'))
    masked, failures = M.mask_cache(raw)
    if failures: json.dump({'status': 'FAIL', 'mask_failures': failures}, open(a.out / 'FAILURE.json', 'w'), indent=1, default=str); raise SystemExit('mask failures')
    lookup = {str(e['idx']): e for e in masked.values()}
    B = dict(np.load(a.base)); off = B['offsets']; cells = B['cells'].astype(str); rows = B['row_ids'].astype(str)
    ans = np.flatnonzero(cells == 'prmbench_qwen3_8b')
    missing = [rows[i] for i in ans if rows[i] not in lookup]
    if missing: raise SystemExit(f'{len(missing)} PRMBench answers missing from raw cell')
    jobs = []
    for i in ans:
        r = RB.safe_row(lookup[rows[i]])
        if len(r['step_token_spans']) != off[i + 1] - off[i]: raise SystemExit('step count ' + rows[i])
        jobs.append((int(i), r))
    filled = []
    vals = B['values'].copy(); act = B['digit_active'].copy(); names = list(B['names'].astype(str))
    assert len(names) == 56
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, b48, b5, d3, ac, _ in ex.map(one_safe, jobs, chunksize=16):
            lo, hi = off[i:i + 2]
            b5, _ = RB.fill_missing(b5)
            v = np.column_stack((b48, b5, d3))
            if not np.isfinite(v).all():   # only allowed on a step that is empty in this tokenization (PRM path: empty step text)
                empty = [k for k, sp in enumerate(lookup[rows[i]]['step_token_spans']) if sp[1] <= sp[0]]
                bad = sorted(set(np.flatnonzero(~np.isfinite(v).all(1)).tolist()))
                if not empty or not set(bad) <= set(empty): raise SystemExit('nonfinite ' + rows[i])
                v, f = RB.fill_missing(v); filled.append({'row': rows[i], 'empty_steps': empty, 'filled_values': int(f.sum())})
            vals[lo:hi] = v; act[lo:hi] = ac
    if a.expect_equal:   # parity mode: rebuilding the base's own raw cell must reproduce the base
        err = float(np.max(np.abs(vals - B['values']))); dmis = int((act != B['digit_active']).sum())
        json.dump({'max_abs_error': err, 'digit_active_mismatch': dmis}, open(a.out / 'PARITY.json', 'w'))
        print('PARITY', err, dmis, flush=True)
        if err > 1e-6 or dmis: raise SystemExit('PARITY FAIL')
    B['values'] = vals; B['digit_active'] = act
    np.savez_compressed(a.out / 'FEATURES.npz', **B)
    json.dump({'raw': str(a.raw), 'raw_sha256': sha(a.raw), 'base': str(a.base), 'base_sha256': sha(a.base),
               'mask_rule': M.RULE, 'n_prmbench_answers': int(len(ans)), 'mask_failures': 0, 'empty_step_fills': filled,
               'features_sha256': sha(a.out / 'FEATURES.npz'), 'seconds': round(time.perf_counter() - t0, 1),
               'job_id': os.environ.get('SLURM_JOB_ID', '')}, open(a.out / 'MANIFEST.json', 'w'), indent=1)
    print('DONE', a.out, round(time.perf_counter() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
