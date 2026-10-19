"""Exact comparison of two bank dirs (FEATURES.npz, REWARDS.npz, MANIFEST.json): old builder vs v2 builder.

Every array must be equal element for element (np.array_equal; no tolerance). From MANIFEST.json the content fields
must agree (raw_sha256, n_records, failed and pre-empty rows, step counts), and the v2 bank must record digit ids that
match Qwen's 15..24. Paths, timings, job ids and the npz sha256 (zip timestamps) are not compared.

    python3 bank_parity.py <summary.json> <name> <ref bank dir> <new bank dir> [<name> <ref> <new> ...]

Prints `PARITY <name> OK|FAIL ...` per pair, writes the summary and exits 1 unless every pair is exact. No labels.
"""
import json, os, sys
from pathlib import Path
import numpy as np

FEATURES = ('names', 'values', 'digit_active', 'offsets', 'uids', 'nonempty', 'nonempty_lengths')
REWARDS = ('uids', 'rewards', 'native_labels', 'nonempty_lengths')
MANIFEST = ('raw_sha256', 'n_records', 'n_failed_rows', 'failed_rows', 'pre_empty_steps', 'steps_total', 'steps_nonempty')


def compare(ref, new):
    out = {'ref': str(ref), 'new': str(new), 'diff': [], 'max_abs': 0.0}
    for f, keys in (('FEATURES.npz', FEATURES), ('REWARDS.npz', REWARDS)):
        with np.load(ref / f) as A, np.load(new / f) as B:
            if set(A.files) != set(B.files):
                out['diff'].append(f'{f}: keys {sorted(A.files)} != {sorted(B.files)}')
            for k in keys:
                a, b = A[k], B[k]
                if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
                    out['diff'].append(f'{f}:{k} {a.dtype}{a.shape} vs {b.dtype}{b.shape}')
                    if a.shape == b.shape and a.dtype.kind in 'fiub' and b.dtype.kind in 'fiub':
                        out['max_abs'] = max(out['max_abs'], float(np.max(np.abs(a.astype(float) - b.astype(float)))))
    mr, mn = (json.loads((d / 'MANIFEST.json').read_text()) for d in (ref, new))
    out['diff'] += [f'MANIFEST:{k}' for k in MANIFEST if mr.get(k) != mn.get(k)]
    dig = mn.get('digit_ids') or {}
    out['new_digit_ids'] = dig.get('digit_ids')
    if not dig.get('matches_qwen_15_24'):
        out['diff'].append(f'MANIFEST:digit_ids of the v2 bank are not 15..24: {dig.get("digit_ids")}')
    out['ok'] = not out['diff']
    return out


def main(argv):
    summary, rest = Path(argv[0]), argv[1:]
    if not rest or len(rest) % 3:
        raise SystemExit('usage: bank_parity.py <summary.json> <name> <ref> <new> [...]')
    pairs = {}
    for name, ref, new in zip(rest[0::3], rest[1::3], rest[2::3]):
        r = pairs[name] = compare(Path(ref), Path(new))
        print('PARITY', name, 'OK' if r['ok'] else 'FAIL', 'max_abs', r['max_abs'], 'diffs', len(r['diff']), *r['diff'][:5], flush=True)
    ok = all(r['ok'] for r in pairs.values())
    summary.write_text(json.dumps({'ok': ok, 'pairs': pairs, 'job_id': os.environ.get('SLURM_JOB_ID', ''),
                                   'labels_read': False}, indent=1))
    print('PARITY SUMMARY', 'OK' if ok else 'FAIL', summary)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
