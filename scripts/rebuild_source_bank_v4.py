"""Rebuild external_banks_v4 source/FEATURES.npz (48 + 5 + 3 step channels, 13769 answers in
localization_full_benchmark_v3 JOINED order) from a chosen set of raw teacher-forced cells.

Same per-answer code path as the frozen gates: family_external_features.step_features (48),
external_banks_v4.new_step_features (5) and digit_step_features (3), then the
fill_missing rule of extract_external_banks_v4.source. No labels are read.

    python scripts/rebuild_source_bank_v4.py --variant old|thinkfix|thinkfix_mask1 \
        --results /shared/.../results --out <new dir> [--parity <frozen FEATURES.npz>] [--workers 8]

variant=old is the reproduction check: with --parity it must match the frozen bank (max abs
error per channel <= 1e-6, digit mask identical) or it exits non-zero. For thinkfix_mask1 the
roster token count is one less than JOINED's (forced token 0 removed, spectral_utils/first_token_mask).
An answer whose raw row is missing (e.g. a mask1 failure) is an explicit failure: the run stops.
"""
import os
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_n] = '1'
import argparse, gc, hashlib, json, pickle, sys, time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils import external_banks_v4 as E  # noqa: E402
from spectral_utils.family_external_features import step_features  # noqa: E402
from spectral_utils.family_tail_transfer import load_lock  # noqa: E402

ROSTER = ROOT/'results/localization_full_benchmark_v3/evaluation/JOINED.json'
SUBSETS = ('gsm8k', 'math', 'olympiadbench', 'omnimath')
SUFFIX = {'old': ('', ''), 'thinkfix': ('_thinkfix', '_thinkfix'),
          'thinkfix_mask1': ('_thinkfix_mask1', '_thinkfix_mask1')}
OLD_PRM = 'prmbench_qwen3_8b_telemetry_full'
NEW_PRM = 'prmbench_qwen3_8b_telemetry'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def cells(results, variant):
    s = SUFFIX[variant][0]
    for m in ('4b', '8b'):
        for d in SUBSETS:
            yield f'pb_{d}_q{m[0]}', results/f'pb_qwen3_{m}{s}'/f'processbench_{d}.pkl', d
    prm = OLD_PRM if variant == 'old' else NEW_PRM+s
    yield 'prmbench_qwen3_8b', results/prm/'prmbench_telemetry.pkl', None


def safe_row(row):
    safe = {k: row[k] for k in ('gen_token_ids', 'token_entropies', 'token_spilled_energies',
                                'token_logsumexp', 'step_token_spans')}
    safe['top_k_logprobs'] = row.get('top_k_logprobs') or row.get('top_k_logprobs_raw')
    return safe


def fill_missing(matrix):
    x = np.array(matrix, float); filled = (~np.isfinite(x)).sum(axis=0)
    if np.isinf(x).any():
        raise ValueError('infinite feature values')
    for j in range(x.shape[1]):
        good = np.isfinite(x[:, j])
        x[~good, j] = x[good, j].mean() if good.any() else 0.
    return x, filled


def one(item):
    i, row = item
    t0 = time.process_time()
    base, _ = step_features(row)
    new, _ = E.new_step_features(row)
    digits, active = E.digit_step_features(row)
    return i, base, new, digits, active, time.process_time()-t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=sorted(SUFFIX), required=True)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--parity', type=Path)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    names48 = load_lock()['recipe']['channels_48']
    names = list(names48)+list(E.NEW_NAMES)+list(E.DIGIT_NAMES)
    roster = json.loads(ROSTER.read_text())['records']
    off = np.r_[0, np.cumsum([r['steps'] for r in roster])]
    assert len(roster) == 13769 and off[-1] == 145597
    shift = 1 if a.variant == 'thinkfix_mask1' else 0
    base = np.full((off[-1], 48), np.nan); new = np.full((off[-1], 5), np.nan)
    digits = np.full((off[-1], 3), np.nan); active = np.zeros((off[-1], 3), bool)
    seen = np.zeros(len(roster), bool); sources = {}; cpu = 0.
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for cell, path, dataset in cells(a.results, a.variant):
            sources[cell] = {'path': str(path), 'sha256': sha(path)}
            payload = pickle.load(open(path, 'rb'))
            rows = list(payload.values()) if isinstance(payload, dict) else payload
            lookup = {}
            for row in rows:
                if isinstance(row, dict):
                    key = f'{dataset}::{row.get("id")}' if dataset else str(row.get('idx'))
                    if key in lookup:
                        raise ValueError('duplicate raw ID '+key)
                    lookup[key] = row
            idx = [i for i, r in enumerate(roster) if r['cell'] == cell]
            missing = [roster[i]['uid'] for i in idx if str(roster[i]['row_id']) not in lookup]
            if missing:
                json.dump({'status': 'FAIL', 'cell': cell, 'missing_rows': missing},
                          open(a.out/'FAILURE.json', 'w'), indent=1)
                raise SystemExit(f'{cell}: {len(missing)} roster answers missing from {path}')

            def jobs():
                for i in idx:
                    row = safe_row(lookup[str(roster[i]['row_id'])])
                    assert len(row['step_token_spans']) == roster[i]['steps'], roster[i]['uid']
                    assert len(row['gen_token_ids']) == roster[i]['tokens']-shift, roster[i]['uid']
                    yield i, row
            it = iter(jobs()); pending = set(); done_all = False; n = 0
            while pending or not done_all:
                while not done_all and len(pending) < 3*a.workers:
                    try:
                        pending.add(ex.submit(one, next(it)))
                    except StopIteration:
                        done_all = True
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    i, b48, b5, d3, act, sec = fut.result()
                    lo, hi = off[i:i+2]
                    base[lo:hi] = b48; new[lo:hi] = b5; digits[lo:hi] = d3; active[lo:hi] = act
                    seen[i] = True; cpu += sec; n += 1
            print('CELL', cell, n, 'elapsed', round(time.perf_counter()-started), flush=True)
            del payload, rows, lookup
            gc.collect()
    assert seen.all() and np.isfinite(base).all() and np.isfinite(digits).all()
    filled = np.zeros(5, int)
    for lo, hi in zip(off[:-1], off[1:]):
        new[lo:hi], f = fill_missing(new[lo:hi]); filled += f
    values = np.column_stack((base, new, digits))
    dest = a.out/'FEATURES.npz'
    np.savez_compressed(dest, values=values, names=np.asarray(names), digit_active=active, offsets=off,
                        uids=np.asarray([r['uid'] for r in roster]), cells=np.asarray([r['cell'] for r in roster]),
                        row_ids=np.asarray([str(r['row_id']) for r in roster]),
                        nonempty=np.ones(int(off[-1]), bool), nonempty_lengths=np.diff(off))
    report = {'variant': a.variant, 'answers': len(roster), 'steps': int(off[-1]), 'channels': names,
              'features_sha256': sha(dest), 'sources': sources, 'roster_sha256': sha(ROSTER),
              'new_channel_filled_values': dict(zip(E.NEW_NAMES, filled.tolist())),
              'digit_inactive_steps': dict(zip(E.DIGIT_NAMES, (~active).sum(axis=0).tolist())),
              'code_sha256': {rel: sha(ROOT/rel) for rel in (
                  'scripts/rebuild_source_bank_v4.py', 'spectral_utils/external_banks_v4.py',
                  'spectral_utils/family_external_features.py', 'spectral_utils/family_hist_features.py',
                  'spectral_utils/digit_feature_family.py', 'spectral_utils/digit_alternative_probability.py',
                  'spectral_utils/family_tail_transfer.py')},
              'labels_read': False, 'elapsed_seconds': time.perf_counter()-started, 'cpu_seconds': cpu,
              'job_id': os.environ.get('SLURM_JOB_ID', '')}
    if a.parity:
        with np.load(a.parity) as z:
            ref, ref_active = z['values'], z['digit_active']
            assert list(z['names']) == names and np.array_equal(z['offsets'], off)
        err = np.abs(values-ref).max(axis=0)
        report['parity'] = {'reference': str(a.parity), 'reference_sha256': sha(a.parity),
                            'max_abs_error': dict(zip(names, map(float, err))),
                            'digit_active_mismatches': int((active != ref_active).sum()),
                            'status': 'PASS' if err.max() <= 1e-6 and (active == ref_active).all() else 'FAIL'}
        print('PARITY', report['parity']['status'], 'max', float(err.max()),
              'worst', names[int(err.argmax())], flush=True)
    (a.out/'MANIFEST.json').write_text(json.dumps(report, indent=2)+'\n')
    print('DONE', values.shape, 'filled', filled.tolist(), flush=True)
    if a.parity and report['parity']['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
