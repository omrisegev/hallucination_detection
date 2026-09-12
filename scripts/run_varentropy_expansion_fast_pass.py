"""Fast pass of the varentropy expansion protocol: the SAME driver and module, restricted at runtime to the 19 non-Joint arms.

The driver (`run_varentropy_expansion_fusion_v1`) and the fusion module are imported UNCHANGED (their
hashes bind the paused full checkpoint).  This wrapper only rebinds, in-process, the solver roster
(no 'joint'), the method list, the output directory, the worker (an assertion shim) and the
comparison filter.  Because Windows spawn re-imports this module as ``__mp_main__``, the same
rebinding happens in every worker process.  Outputs go to ``results/varentropy_expansion_fusion_v1/fast_pass/``.
The four Joint arms are recorded as ``pending_arms``; ``scripts/merge_varentropy_expansion_passes.py``
produces the final 23-arm evaluation once the full checkpoint completes.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_varentropy_expansion_fusion_v1 as run
from spectral_utils import varentropy_expansion_fusion as vef

base = run.base
FULL_METHODS = vef.METHODS
PENDING_ARMS = tuple(m for m in FULL_METHODS if m.endswith('__joint'))
FAST_METHODS = tuple(m for m in FULL_METHODS if m not in PENDING_ARMS)
FAST_SOLVERS = tuple(s for s in vef.SOLVERS if s != 'joint')
FULL_OUT = run.OUT
OUT = run.OUT / 'fast_pass'
assert len(FAST_METHODS) == 19 and len(PENDING_ARMS) == 4
_original_worker = run.worker
_original_comparisons = run.comparisons

# ---- process-local restriction (executed on import, hence in every spawned worker as well) ----
vef.SOLVERS = FAST_SOLVERS
run.METHODS = FAST_METHODS
run.OUT = OUT


def fast_worker(task):
    if vef.SOLVERS != FAST_SOLVERS or run.METHODS != FAST_METHODS:
        raise RuntimeError('fast-pass restriction not active in this worker process')
    return _original_worker(task)


def fast_comparisons():
    present = set(FAST_METHODS) | {'entropy'}
    return [(t, a, b) for t, a, b in _original_comparisons() if a in present and b in present]


run.worker = fast_worker
run.comparisons = fast_comparisons


def manifest_for(source, roots):
    manifest = run.manifest_for(source, roots)
    manifest['hashes'][str(Path(__file__))] = base.old.sha256_file(Path(__file__))
    manifest.update(schema='varentropy-expansion-fast-pass-v1', methods=list(FAST_METHODS), pending_arms=list(PENDING_ARMS),
                    **{'pass': 'fast'})
    full = FULL_OUT / 'MANIFEST.json'
    if full.exists():
        reference = json.loads(full.read_text(encoding='utf8'))['hashes']
        drift = [k for k, v in reference.items() if manifest['hashes'].get(k) != v]
        if drift: raise ValueError('fast pass code/input hashes differ from the paused full checkpoint: ' + '; '.join(drift[:5]))
        manifest['full_pass_manifest_match'] = True
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    for a in ('v2-root', 'temporal-root', 'varentropy-root'): p.add_argument('--' + a, type=Path, default=None)
    p.add_argument('--workers', type=int, default=1); p.add_argument('--smoke', action='store_true')
    p.add_argument('--max-answers', type=int, default=0); p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--memory-gate-pb-kb', type=int, default=run.MEMORY_GATE_KB['pb']); p.add_argument('--memory-gate-prmb-kb', type=int, default=run.MEMORY_GATE_KB['prm'])
    args = p.parse_args(); source = args.source_root.resolve()
    roots = run.default_roots(source)
    for k in roots:
        override = getattr(args, k + '_root')
        if override is not None: roots[k] = override.resolve()
    base.old.configure_source_root(source); OUT.mkdir(parents=True, exist_ok=True)
    manifest = manifest_for(source, roots)
    con = base.connect(OUT / ('SMOKE.sqlite' if args.smoke else 'CHECKPOINT.sqlite'), manifest)
    base.atomic_json(OUT / ('SMOKE_MANIFEST.json' if args.smoke else 'MANIFEST.json'), manifest)
    records = json.loads((base.old.BENCH / 'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined = np.load(base.old.BENCH / 'evaluation/JOINED.npz', allow_pickle=False)
    if len(records) != 13769 or len({r['uid'] for r in records}) != 13769: raise ValueError('benchmark roster mismatch')
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        if not args.evaluate_only:
            run.score(con, records, joined, roots, workers=args.workers, smoke=args.smoke, max_answers=args.max_answers,
                      gates={'pb': args.memory_gate_pb_kb, 'prm': args.memory_gate_prmb_kb})
        n = con.execute('SELECT COUNT(*) FROM answers').fetchone()[0]
        if args.smoke:
            infos = [json.loads(r[0]) for r in con.execute('SELECT info FROM answers ORDER BY idx')]
            if any(m in r['seconds'] or m in r['diagnostics'] for r in infos for m in PENDING_ARMS): raise RuntimeError('a Joint arm was fitted in the fast pass')
            failures = [dict(uid=r['uid'], method=m, reason=reason, declared=run.is_declared_failure(reason)) for r in infos for m, reason in r['failures'].items()]
            status = 'PASS' if all(f['declared'] for f in failures) else 'FAIL'
            base.atomic_json(OUT / 'SMOKE.json', dict(status=status, scope='FEASIBILITY_ONLY', **{'pass': 'fast'}, pending_arms=list(PENDING_ARMS), n_answers=n,
                                                      failures=failures, undeclared_failures=sum(not f['declared'] for f in failures), rows=infos))
            base.atomic_json(OUT / 'FEASIBILITY.json', run.clean(dict(run.feasibility(infos, args.workers, time.perf_counter() - started), **{'pass': 'fast'}, pending_arms=list(PENDING_ARMS))))
            base.atomic_json(OUT / 'SMOKE_STATE.json', dict(status='SMOKE_' + status, completed=n, expected=len(records)))
            print('[smoke] fast pass', status, n, 'rows; failures:', len(failures), flush=True)
        elif n == len(records):
            run.evaluate(con, records, joined, roots)
            metrics = json.loads((OUT / 'METRICS.json').read_text(encoding='utf8'))
            metrics.update(schema='varentropy-expansion-fast-pass-v1', pending_arms=list(PENDING_ARMS), **{'pass': 'fast'},
                           note='19 non-Joint arms only; the four Joint arms are pending in the full checkpoint and are merged by merge_varentropy_expansion_passes.py.')
            base.atomic_json(OUT / 'METRICS.json', metrics)
            base.atomic_json(OUT / 'RUN_STATE.json', dict(status='FAST_PASS_COMPLETE', completed=n, expected=len(records), pending_arms=list(PENDING_ARMS)))
        else:
            print('[scoring] fast pass', n, '/', len(records), 'answers scored; evaluation waits for completion', flush=True)
    con.close()


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException as error:
        name = 'SMOKE_STATE.json' if '--smoke' in sys.argv else 'RUN_STATE.json'
        path = OUT / name; state = json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED', error=f'{type(error).__name__}: {error}')
        base.atomic_json(path, state)
        raise
