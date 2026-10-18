"""Build an external FEATURES.npz (Socratic / Hard2Verify) from a PRM's backbone telemetry
(cluster/run_prmbench_prm_telemetry.py --answers -> external_telemetry.pkl).

Per-step channels are computed exactly as on PRMBench for PRM boosting
(results/prm_boosting_v1/build_bank.py): the frozen bank builder's own `one()` from
scripts/rebuild_source_bank_v4.py (code_thinkfix_bank_553de79), after the mask1 rule (forced token 0
dropped), empty steps filled with the answer's column means. The output follows the external_banks_v4
layout read by results/external_lr_prm_v1/lr_transfer.py:bank_V/evaluate: records sorted by uid,
`nonempty` = flattened per-record step mask, `nonempty_lengths` = total steps per record, `offsets`
index the non-empty rows. A step is empty when its token span is empty in this tokenization.
Rewards and native flags are saved next to it (REWARDS.npz, aligned with all steps). No labels are read.

Run on AIRCC CPU with cwd = code_thinkfix_bank_553de79:
    python3 build_bank_external.py --raw <dir>/external_telemetry.pkl --mask-code <code_thinkfix_rescore_v1> \
        --names-from <PRMBench FEATURES.npz> --out <dir>/bank [--workers 16]
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
import rebuild_source_bank_v4 as RB  # noqa: E402

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def one_nonempty(item):
    """RB.one on the non-empty steps only (the external layout drops empty steps)."""
    i, row = item
    sp = row['step_token_spans']; keep = [k for k, s in enumerate(sp) if s[1] > s[0]]
    r = dict(row); r['step_token_spans'] = [sp[k] for k in keep]
    _, b48, b5, d3, ac, sec = RB.one((i, r))
    return i, keep, b48, b5, d3, ac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=Path, required=True); ap.add_argument('--mask-code', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--names-from', type=Path, required=True, help='any 56-channel source FEATURES.npz (channel order)')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False); t0 = time.perf_counter()
    mspec = importlib.util.spec_from_file_location('ftm', a.mask_code / 'spectral_utils/first_token_mask.py')
    M = importlib.util.module_from_spec(mspec); mspec.loader.exec_module(M)
    raw = pickle.load(open(a.raw, 'rb'))
    failed = {str(e['idx']): e['failure'] for e in raw.values() if 'failure' in e}
    good = {k: e for k, e in raw.items() if 'failure' not in e}
    masked, failures = M.mask_cache(good)
    if failures:
        json.dump({'status': 'FAIL', 'mask_failures': failures}, open(a.out / 'FAILURE.json', 'w'), indent=1, default=str)
        raise SystemExit('mask failures')
    recs = sorted(masked.values(), key=lambda e: str(e['idx']))
    jobs = [(j, RB.safe_row(e)) for j, e in enumerate(recs)]
    out = [None] * len(recs)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for j, keep, b48, b5, d3, ac in ex.map(one_nonempty, jobs, chunksize=16):
            b5, _ = RB.fill_missing(b5)
            v = np.column_stack((b48, b5, d3))
            if not np.isfinite(v).all():
                raise SystemExit('nonfinite on a non-empty step: ' + str(recs[j]['idx']))
            out[j] = (keep, v, ac)
    lens = np.array([len(e['step_token_spans']) for e in recs])
    nonempty = np.zeros(int(lens.sum()), bool); starts = np.r_[0, np.cumsum(lens)]
    for j, (keep, _, _) in enumerate(out):
        nonempty[starts[j] + np.asarray(keep, int)] = True
    values = np.vstack([v for _, v, _ in out]); active = np.vstack([ac for _, _, ac in out])
    offsets = np.r_[0, np.cumsum([len(k) for k, _, _ in out])]
    names = np.load(a.names_from)['names']
    assert values.shape[1] == len(names) == 56, values.shape
    np.savez_compressed(a.out / 'FEATURES.npz', names=names, values=values, digit_active=active, offsets=offsets,
                        uids=np.array([str(e['idx']) for e in recs]), nonempty=nonempty, nonempty_lengths=lens)
    rew = np.concatenate([np.asarray(e['rewards'], float) for e in recs])
    nat = np.concatenate([np.asarray(e['native_labels'], int) for e in recs])
    assert len(rew) == len(nat) == lens.sum()
    np.savez_compressed(a.out / 'REWARDS.npz', uids=np.array([str(e['idx']) for e in recs]), rewards=rew,
                        native_labels=nat, nonempty_lengths=lens)
    json.dump({'raw': str(a.raw), 'raw_sha256': sha(a.raw), 'mask_rule': M.RULE, 'n_records': len(recs),
               'n_failed_rows': len(failed), 'failed_rows': failed, 'steps_total': int(lens.sum()),
               'steps_nonempty': int(nonempty.sum()), 'labels_read': False,
               'features_sha256': sha(a.out / 'FEATURES.npz'), 'seconds': round(time.perf_counter() - t0, 1),
               'job_id': os.environ.get('SLURM_JOB_ID', '')}, open(a.out / 'MANIFEST.json', 'w'), indent=1)
    print('DONE', a.out, len(recs), 'records', round(time.perf_counter() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
