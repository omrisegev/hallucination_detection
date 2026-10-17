"""Joint pass of the varentropy expansion protocol: the SAME driver and module, restricted at runtime to the two
Joint arms on the primary banks (B2_sel__joint, B2d_sel__joint).

The driver and fusion module are imported UNCHANGED (their hashes bind the other checkpoints).  This wrapper
rebinds, in-process only: the bank roster (primary banks), the solver roster ('joint'), the method list
(the three historical arms, which cost ~10 ms and keep the driver's in-worker Step 339 replay assertion, plus
the two Joint arms), the output directory, the worker (assertion shim), the checkpoint reader (appends the
remaining 16 fast-pass arms from ``fast_pass/SCORES.npz`` at evaluation time) and the comparison filter
(every registered pair involving a Joint arm, plus the unchanged primary pair).  Windows spawn re-imports this
module as ``__mp_main__``, so every worker process carries the same restriction.

``B2__joint`` and ``B2d__joint`` (secondary banks) are DEFERRED FOR COMPUTE, not dropped from the protocol.
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
JOINT_ARMS = ('B2d_sel__joint', 'B2_sel__joint')
PENDING_ARMS = ('B2__joint', 'B2d__joint')
PASS_METHODS = vef.HISTORICAL + JOINT_ARMS                     # 5 checkpoint columns
FAST_METHODS = tuple(m for m in FULL_METHODS if not m.endswith('__joint'))   # 19 arms scored by the fast pass
FAST_ONLY = tuple(m for m in FAST_METHODS if m not in vef.HISTORICAL)         # 16 arms appended from fast_pass/SCORES.npz
FULL_OUT = run.OUT
FAST_OUT = run.OUT / 'fast_pass'
OUT = run.OUT / 'joint_pass'
MEMORY_GATE_KB = {'pb': 1_200_000, 'prm': 1_500_000}          # main-agent instruction for this pass
DEFERRED_NOTE = 'B2__joint and B2d__joint are deferred for compute (Joint dominates runtime), not dropped from the protocol.'
assert set(PASS_METHODS) | set(FAST_METHODS) | set(PENDING_ARMS) == set(FULL_METHODS)
_original_worker = run.worker
_original_comparisons = run.comparisons
_original_load_scored = base.load_scored

# ---- process-local restriction (executed on import, hence in every spawned worker as well) ----
vef.BANKS = ('B2d_sel', 'B2_sel')
vef.SOLVERS = ('joint',)
run.METHODS = PASS_METHODS
run.OUT = OUT


def joint_worker(task):
    if vef.SOLVERS != ('joint',) or vef.BANKS != ('B2d_sel', 'B2_sel') or run.METHODS != PASS_METHODS:
        raise RuntimeError('joint-pass restriction not active in this worker process')
    return _original_worker(task)


def joint_comparisons():
    present = set(FAST_METHODS) | set(JOINT_ARMS) | {'entropy'}
    return [(t, a, b) for t, a, b in _original_comparisons()
            if a in present and b in present and ((a in JOINT_ARMS or b in JOINT_ARMS) or (a, b) in run.PRIMARY)]


def joint_load_scored(con, records, offsets):
    """Checkpoint arms (historical + Joint) plus the 16 fast-pass arms; historical arms must agree across passes."""
    scores, telemetry = _original_load_scored(con, records, offsets)
    with np.load(FAST_OUT / 'SCORES.npz', allow_pickle=False) as fast:
        for m in vef.HISTORICAL:
            if not np.array_equal(scores[m], fast['steps__' + m], equal_nan=True): raise ValueError(f'{m} differs between joint pass and fast pass')
        for m in FAST_ONLY: scores[m] = fast['steps__' + m]
    return scores, telemetry


run.worker = joint_worker
run.comparisons = joint_comparisons
base.load_scored = joint_load_scored


def manifest_for(source, roots):
    manifest = run.manifest_for(source, roots)
    manifest['hashes'][str(Path(__file__))] = base.old.sha256_file(Path(__file__))
    manifest.update(schema='varentropy-expansion-joint-pass-v1', methods=list(PASS_METHODS), joint_arms=list(JOINT_ARMS),
                    pending_arms=list(PENDING_ARMS), pending_note=DEFERRED_NOTE, fast_pass_arms=list(FAST_METHODS), **{'pass': 'joint_sel'})
    for other in (FULL_OUT / 'MANIFEST.json', FAST_OUT / 'MANIFEST.json', FAST_OUT / 'SMOKE_MANIFEST.json'):
        if other.exists():
            reference = json.loads(other.read_text(encoding='utf8'))['hashes']
            drift = [k for k, v in reference.items() if k in manifest['hashes'] and manifest['hashes'][k] != v]
            if drift: raise ValueError(f'joint pass code/input hashes differ from {other}: ' + '; '.join(drift[:5]))
    manifest['other_pass_manifests_match'] = True
    return manifest


def evaluate(con, records, joined, roots):
    fast_state = json.loads((FAST_OUT / 'RUN_STATE.json').read_text(encoding='utf8'))
    if fast_state.get('status') != 'FAST_PASS_COMPLETE': raise RuntimeError('fast pass must be complete before the joint-pass evaluation')
    run.evaluate(con, records, joined, roots)                    # unchanged driver evaluation over 21 present arms
    metrics = json.loads((OUT / 'METRICS.json').read_text(encoding='utf8'))
    fast = json.loads((FAST_OUT / 'METRICS.json').read_text(encoding='utf8'))
    for key in ('failures', 'coverage', 'weights', 'telemetry'):
        for m in FAST_ONLY: metrics[key][m] = fast[key][m]
    metrics.update(schema='varentropy-expansion-joint-pass-v1', **{'pass': 'joint_sel'}, joint_arms=list(JOINT_ARMS), pending_arms=list(PENDING_ARMS),
                   pending_note=DEFERRED_NOTE, fast_pass_scores_sha256=base.old.sha256_file(FAST_OUT / 'SCORES.npz'),
                   fast_pass_metrics_sha256=base.old.sha256_file(FAST_OUT / 'METRICS.json'),
                   note='B2_sel__joint and B2d_sel__joint fitted in this pass; the 19 non-Joint arms are the fast-pass scores; references frozen as in the driver.')
    base.atomic_json(OUT / 'METRICS.json', metrics)
    base.atomic_json(OUT / 'RUN_STATE.json', dict(status='JOINT_PASS_COMPLETE', completed=len(records), expected=len(records),
                                                  pending_arms=list(PENDING_ARMS), pending_note=DEFERRED_NOTE))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    for a in ('v2-root', 'temporal-root', 'varentropy-root'): p.add_argument('--' + a, type=Path, default=None)
    p.add_argument('--workers', type=int, default=1); p.add_argument('--smoke', action='store_true')
    p.add_argument('--max-answers', type=int, default=0); p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--memory-gate-pb-kb', type=int, default=MEMORY_GATE_KB['pb']); p.add_argument('--memory-gate-prmb-kb', type=int, default=MEMORY_GATE_KB['prm'])
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
            extra = {m for r in infos for m in list(r['seconds']) + list(r['diagnostics']) + list(r['failures'])} - set(PASS_METHODS) - {'bank_build'}
            if extra: raise RuntimeError(f'arms outside the joint pass were fitted: {sorted(extra)}')
            failures = [dict(uid=r['uid'], method=m, reason=reason, declared=run.is_declared_failure(reason)) for r in infos for m, reason in r['failures'].items()]
            status = 'PASS' if all(f['declared'] for f in failures) else 'FAIL'
            base.atomic_json(OUT / 'SMOKE.json', dict(status=status, scope='FEASIBILITY_ONLY', **{'pass': 'joint_sel'}, joint_arms=list(JOINT_ARMS),
                                                      pending_arms=list(PENDING_ARMS), pending_note=DEFERRED_NOTE, n_answers=n, failures=failures,
                                                      undeclared_failures=sum(not f['declared'] for f in failures), rows=infos))
            base.atomic_json(OUT / 'FEASIBILITY.json', run.clean(dict(run.feasibility(infos, args.workers, time.perf_counter() - started),
                                                                      **{'pass': 'joint_sel'}, joint_arms=list(JOINT_ARMS), pending_arms=list(PENDING_ARMS))))
            base.atomic_json(OUT / 'SMOKE_STATE.json', dict(status='SMOKE_' + status, completed=n, expected=len(records)))
            print('[smoke] joint pass', status, n, 'rows; failures:', len(failures), flush=True)
        elif n == len(records):
            evaluate(con, records, joined, roots)
        else:
            print('[scoring] joint pass', n, '/', len(records), 'answers scored; evaluation waits for completion', flush=True)
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
