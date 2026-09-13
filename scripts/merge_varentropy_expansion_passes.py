"""Merge the fast pass (19 non-Joint arms) with the completed full checkpoint (23 arms).

1. Assert the two manifests bind identical code/input hashes.
2. Assert every fast-pass arm replays BIT-IDENTICALLY in the full checkpoint (step scores, standardized
   and effective weights, intercepts, failures, diagnostics; NaN-aware equality).
3. Produce the final 23-arm METRICS.json / SCORES.npz / CSVs with the unchanged driver evaluation.
``--smoke`` compares the two SMOKE.sqlite files and skips the evaluation.
"""
import argparse
import io
import json
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_varentropy_expansion_fusion_v1 as run

base = run.base
FULL_OUT = run.OUT; FAST_OUT = run.OUT / 'fast_pass'


def rows(path):
    con = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    assert con.execute('pragma quick_check').fetchone()[0] == 'ok', path
    manifest = json.loads(con.execute('SELECT payload FROM manifest WHERE id=1').fetchone()[0])
    out = {}
    for i, blob, info in con.execute('SELECT idx,payload,info FROM answers'):
        with np.load(io.BytesIO(blob), allow_pickle=False) as z: arrays = {k: z[k] for k in z.files}
        out[i] = (arrays, json.loads(info))
    con.close(); return manifest, out


def equal_nan(a, b):
    return np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True)


def compare(fast, full, fast_methods, full_methods):
    mismatches = []; checks = 0
    for i, (fa, fi) in fast.items():
        if i not in full: mismatches.append(dict(idx=i, what='missing in full checkpoint')); continue
        ua, ui = full[i]
        if fi['uid'] != ui['uid']: mismatches.append(dict(idx=i, what='uid')); continue
        for m in fast_methods:
            jf, ju = fast_methods.index(m), full_methods.index(m)
            for what, ok in (('steps', equal_nan(fa['steps'][:, jf], ua['steps'][:, ju])),
                             ('weights', equal_nan(fa['weights'][jf], ua['weights'][ju])),
                             ('effective', equal_nan(fa['effective'][jf], ua['effective'][ju])),
                             ('intercept', equal_nan(fa['intercepts'][jf], ua['intercepts'][ju])),
                             ('failure', fi['failures'].get(m) == ui['failures'].get(m)),
                             ('diagnostics', fi['diagnostics'].get(m) == ui['diagnostics'].get(m))):
                checks += 1
                if not ok: mismatches.append(dict(idx=i, uid=fi['uid'], method=m, what=what))
        for key in ('identity_max_discrepancy', 'identity_fusion_vs_k15raw_max_error', 'raw50_max_error', 'hist_replay_max_error'):
            checks += 1
            if fi[key] != ui[key]: mismatches.append(dict(idx=i, uid=fi['uid'], what=key))
    return checks, mismatches


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--skip-evaluation', action='store_true'); args = p.parse_args()
    source = args.source_root.resolve(); base.old.configure_source_root(source); roots = run.default_roots(source)
    name = 'SMOKE.sqlite' if args.smoke else 'CHECKPOINT.sqlite'; started = time.perf_counter()
    fast_manifest, fast = rows(FAST_OUT / name); full_manifest, full = rows(FULL_OUT / name)
    fast_methods = tuple(fast_manifest['methods']); full_methods = tuple(full_manifest['methods'])
    assert full_methods == tuple(run.METHODS) and len(full_methods) == 23 and set(fast_methods) < set(full_methods) and len(fast_methods) == 19
    assert set(fast_manifest['pending_arms']) == set(full_methods) - set(fast_methods)
    drift = [k for k, v in full_manifest['hashes'].items() if fast_manifest['hashes'].get(k) != v]
    assert not drift, 'code/input hash drift between passes: ' + '; '.join(drift[:5])
    for k, v in full_manifest['hashes'].items(): assert base.old.sha256_file(Path(k)) == v, 'on-disk drift: ' + k
    checks, mismatches = compare(fast, full, fast_methods, full_methods)
    status = 'PASS' if not mismatches and len(fast) == len(full) else 'FAIL'
    if not args.smoke and status == 'PASS' and len(full) != 13769: status = 'INCOMPLETE'
    result = dict(status=status, scope='smoke' if args.smoke else 'full', fast_answers=len(fast), full_answers=len(full), bit_identity_checks=checks,
                  mismatches=mismatches[:50], n_mismatches=len(mismatches), fast_methods=list(fast_methods), pending_arms=fast_manifest['pending_arms'],
                  hashes_verified=len(full_manifest['hashes']), seconds=time.perf_counter() - started)
    run.atomic_json_retry(FULL_OUT / ('MERGE_SMOKE_REVIEW.json' if args.smoke else 'MERGE_REVIEW.json'), result)
    print('[merge]', status, len(fast), 'fast /', len(full), 'full answers;', checks, 'bit-identity checks;', len(mismatches), 'mismatches', flush=True)
    if args.smoke or args.skip_evaluation or status != 'PASS': return
    records = json.loads((base.old.BENCH / 'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined = np.load(base.old.BENCH / 'evaluation/JOINED.npz', allow_pickle=False)
    con = base.connect(FULL_OUT / 'CHECKPOINT.sqlite', full_manifest)
    with threadpool_limits(limits=1): run.evaluate(con, records, joined, roots)     # unchanged driver: all 23 arms
    con.close()
    metrics = json.loads((FULL_OUT / 'METRICS.json').read_text(encoding='utf8'))
    metrics.update(**{'pass': 'merged'}, fast_pass_bit_identity='PASS', fast_pass_methods=list(fast_methods))
    run.atomic_json_retry(FULL_OUT / 'METRICS.json', metrics)
    print('[merge] final 23-arm METRICS.json written', flush=True)


if __name__ == '__main__':
    main()
