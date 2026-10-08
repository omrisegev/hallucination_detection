"""Write existing LR arms ahead of the runner without mutating workflow state.

Assumption: the runner consumes LR in its frozen serial order. Success: identical
LR fields in two fresh processes and sealed content argmax replay before use.
Any changed source, replay mismatch or checkpoint corruption stops this helper.
The 15 minute scheduling margin is a conservative scheduling estimate, not a
lock shared with the older runner; it is rechecked after compute and publication
preparation. Existing artifacts are never overwritten.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from spectral_utils.monday_fusion_protocol import ROOT, WORK, CODE, RUNS, BRANCH, sha, dump
from spectral_utils.monday_fusion_execution import completed_chunk, restore_content, native, storage_guard
from spectral_utils.monday_fusion_layers import lr_rows

VARIANTS = ('original', 'remove_2', 'remove_3', 'random_2', 'random_3')
SUFFIXES = ('.metrics.csv', '.fits.json', '.fold_metrics.json')
SAFE_SECONDS = 900


def now():
    return datetime.now(timezone.utc).isoformat()


def process_identity(pid):
    """Windows PID plus creation time prevents mistaking a reused PID for owner."""
    if os.name != 'nt':
        return {'pid': pid, 'creation_time': None, 'alive': None}
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_ulonglong)] * 4
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, 0, pid)
    if not handle:
        return {'pid': pid, 'creation_time': None, 'alive': False}
    try:
        times = [ctypes.c_ulonglong() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            raise RuntimeError('Cannot verify helper process creation time')
        return {'pid': pid, 'creation_time': times[0].value, 'alive': times[1].value == 0}
    finally:
        kernel.CloseHandle(handle)


@contextmanager
def ownership(path):
    owner = {**process_identity(os.getpid()), 'acquired': now()}
    try:
        with path.open('x', encoding='utf8') as stream:
            json.dump(owner, stream); stream.flush(); os.fsync(stream.fileno())
    except FileExistsError:
        previous = json.loads(path.read_text(encoding='utf8'))
        actual = process_identity(previous['pid'])
        raise RuntimeError('LR helper ownership file exists; never remove automatically. '
                           + json.dumps({'recorded': previous, 'actual': actual}))
    try:
        yield
    finally:
        if json.loads(path.read_text(encoding='utf8')) != owner:
            raise RuntimeError('LR helper ownership changed unexpectedly')
        path.unlink()


def memory_guard(minimum_mb=650):
    if os.name == 'nt':
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ('total_phys', 'available_phys', 'total_page',
                                                     'available_page', 'total_virtual', 'available_virtual', 'extended')]
        status = MemoryStatus(); status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise RuntimeError('Cannot verify free physical memory')
        if status.available_phys < minimum_mb * 1_000_000:
            raise RuntimeError(f'Only {status.available_phys / 1e6:.0f} MB RAM available; helper requires {minimum_mb} MB before next bank')


def code_check(expected_head=None):
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', BRANCH], text=True).strip()
    if expected_head and head != expected_head:
        raise RuntimeError('Branch head changed before helper startup')
    paths = [p.relative_to(CODE).as_posix() for p in (CODE / 'spectral_utils').glob('monday_fusion*.py')]
    paths += ['scripts/experiments/monday_fusion_run.py', 'scripts/experiments/monday_fusion_lr_helper_run.py', 'spectral_utils/monday_lr_helper.py']
    hashes = {}
    for rel in paths:
        content = subprocess.check_output(['git', '-C', str(ROOT), 'show', head + ':' + rel])
        if content != (CODE / rel).read_bytes():
            raise RuntimeError('Helper requires exact committed code: ' + rel)
        hashes[rel] = sha(CODE / rel)
    return head, hashes


def load_population():
    import calfix_common as CC
    path = RUNS['level_reduction_v1'] / 'INPUT_MANIFEST.json'
    manifest = json.loads(path.read_text(encoding='utf8'))
    # Read-only verification, deliberately not gates.audit().
    for name, item in manifest['inputs'].items():
        if sha(item['path']) != item['sha256']:
            raise RuntimeError('Input hash mismatch: ' + name)
    pop = CC.Population()
    folds = json.loads(Path(manifest['inputs']['folds']['path']).read_text(encoding='utf8'))['outer']
    if not all(folds[g] == int(f) for g, f in zip(pop.groups, pop.fold)):
        raise RuntimeError('FOLDS_V2 mismatch')
    return manifest, pop


def runner_order(cells, banks):
    return [(ce, bk, variant, roster, pos) for ce in sorted(cells) for bk in banks
            for variant in VARIANTS for roster in ('full', 'survivors') for pos in (False, True)]


def stem_of(job):
    ce, bk, variant, roster, pos = job
    return f'{ce}__{bk}__{variant}__{roster}__pos{int(pos)}'


def helper_order(jobs, banks):
    # Original PRM across all banks first; bank batching bounds resident memory.
    result = []
    for prm in (True, False):
        for bank in banks:
            result.extend(j for j in jobs if j[1] == bank and j[2] == 'original' and j[0].startswith('prm') == prm)
    for bank in reversed(list(banks)):
        result.extend(j for j in reversed(jobs) if j[1] == bank and j[2] != 'original')
    return result


def read_log(path):
    raw = path.read_bytes() if path.exists() else b''
    return raw.decode('utf-16' if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf8', errors='replace')


def runner_cursor(jobs, directory, status_path, log_path):
    status = json.loads(status_path.read_text(encoding='utf8'))
    if status.get('stage') != 'lr':
        # Later phases must not cause the cursor to rewind to zero.
        if (directory.parent / 'LR_COMPLETE.json').exists():
            return len(jobs)
        if status.get('stage') not in ('position_decision', 'level_contrasts', 'input_contrasts', 'estimator_a', 'estimator_b'):
            raise RuntimeError('Unrecognized runner stage; refusing concurrent writes: ' + str(status))
        return 0
    mapping = {stem_of(j): i for i, j in enumerate(jobs)}
    cursor = 0
    for line in read_log(log_path).splitlines():
        if line.startswith('LR checkpoint '):
            name = line.split()[2]
            if name not in mapping:
                raise RuntimeError('Unrecognized runner LR checkpoint: ' + name)
            cursor = mapping[name] + 1
    while cursor < len(jobs) and (directory / (stem_of(jobs[cursor]) + '.complete.json')).exists():
        cursor += 1
    return cursor


def reach_seconds(jobs, candidate, directory, status_path, log_path):
    cursor = runner_cursor(jobs, directory, status_path, log_path)
    if candidate <= cursor:
        return 0.0
    return float(sum(240 if j[0].startswith('prm') else 12 for j in jobs[cursor:candidate]
                     if not (directory / (stem_of(j) + '.complete.json')).exists()))


def verify_restored(state, cell, bank, variant):
    stem = f'{cell}__{bank}__{variant}'
    path = RUNS['level_reduction_v1'] / 'chunks' / (stem + '.stats.npz')
    with np.load(path) as sealed:
        if not np.array_equal(state['bank']['answers'], sealed['answer_indices']):
            raise RuntimeError('Restored population differs: ' + stem)
        for recipe, score in state['scores'].items():
            actual = np.array([np.flatnonzero(score[a:b] >= score[a:b].max() - 8 * np.finfo(float).eps)[0]
                               for a, b in zip(state['bank']['off'][:-1], state['bank']['off'][1:])])
            if not np.array_equal(actual, sealed[recipe + '__ANSWER_Z_Q80__argmax']):
                raise RuntimeError('Restored argmax differs: ' + stem + '/' + recipe)
    return {'recipes_checked': len(state['scores']), 'answers_checked': len(state['bank']['answers']), 'sealed_sha256': sha(path)}


def compute(data, state, job):
    ce, bk, variant, roster, position = job
    started = time.perf_counter()
    try:
        rows, folds, diag = lr_rows(data, state, bk, ce, variant, roster, position)
        status = 'COMPLETE'
    except (ValueError, ArithmeticError, ConvergenceWarning, np.linalg.LinAlgError) as exc:
        status = 'ARM_FAILURE'; diag = {'failure': str(exc)}; folds = []
        rows = [{'bank': bk, 'variant': variant, 'recipe': 'LR_' + roster,
                 'position_variant': 'POS' if position else 'CONTENT', 'decision_rule': rule,
                 'cell': ce, 'score_scope': scope, 'coverage': 0., 'failures': len(state['bank']['answers']),
                 'n_answers': 0, 'n_total': len(state['bank']['answers']), 'failure_reason': str(exc), 'review_status': 'PENDING'}
                for rule in ['ANSWER_Z_Q80', 'TOP_1', 'TOP_2', 'TOP_3'] for scope in ['OOF', 'IN_SAMPLE_DIAGNOSTIC']]
    for row in rows:
        row['runtime_s'] = time.perf_counter() - started
    return rows, folds, diag, status, time.perf_counter() - started


def deterministic_payload(result):
    rows, folds, diag, status, elapsed = result
    return native({'rows': [{k: v for k, v in row.items() if k != 'runtime_s'} for row in rows],
                   'folds': folds, 'diagnostics': diag, 'status': status})


def exclusive_publish(staging, directory, stem, can_write):
    """Publish complete files without replacing any existing directory entry."""
    if not can_write():
        return False
    paths = [directory / (stem + s) for s in (*SUFFIXES, '.complete.json')]
    if any(p.exists() for p in paths):
        return False
    # All serialization is finished before the final scheduling and seal check.
    if not can_write():
        return False
    for path in paths:  # seal last; os.link fails rather than overwriting a race.
        os.link(staging / path.name, path)
    return True


def determinism_check(provenance_dir, head):
    script = CODE / 'scripts/experiments/monday_fusion_lr_helper_run.py'
    outputs = [provenance_dir / f'determinism_{i}.json' for i in (1, 2)]
    for output in outputs:
        subprocess.run([sys.executable, '-B', '-u', str(script), '--worker-check', str(output), '--expected-head', head], check=True)
    a, b = [json.loads(p.read_text(encoding='utf8')) for p in outputs]
    if a != b or a['status'] != 'COMPLETE':
        raise RuntimeError('Two-process LR determinism check failed')
    result = {'status': 'PASS', 'stem': 'pb_gsm8k_q4__B13__original__full__pos0',
              'rows': len(a['rows']), 'fold_rows': len(a['folds']), 'fields': 'all except runtime_s',
              'artifacts': {p.name: sha(p) for p in outputs}}
    dump(provenance_dir / 'DETERMINISM.json', result)
    return result


def run(args):
    head, hashes = code_check(args.expected_head)
    if args.worker_check:
        from spectral_utils.monday_fusion_replay import HistoricalData
        memory_guard(900)
        manifest, pop = load_population()
        memory_guard()
        data = HistoricalData(manifest, pop, banks=['B13'])
        state = restore_content(data, 'B13', 'pb_gsm8k_q4', 'original')
        verify_restored(state, 'pb_gsm8k_q4', 'B13', 'original')
        dump(Path(args.worker_check), deterministic_payload(compute(data, state, ('pb_gsm8k_q4', 'B13', 'original', 'full', False))))
        return
    provenance_dir = WORK / ('lr_helper_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + str(os.getpid()))
    provenance_dir.mkdir()
    stop = Path(args.stop_file) if args.stop_file else WORK / 'STOP_LR_HELPER'
    provenance = {'status': 'STARTING', 'pid': os.getpid(), 'head': head, 'code_hashes': hashes, 'threads': 2,
                  'started': now(), 'stop_file': str(stop), 'command': sys.argv, 'written': [], 'discarded': [], 'arm_failures': 0}
    record = provenance_dir / 'PROVENANCE.json'
    dump(record, provenance)
    print('LR helper provenance: ' + str(record), flush=True)
    try:
        if stop.exists():
            provenance['status'] = 'STOPPED'; return
        provenance['determinism'] = determinism_check(provenance_dir, head)
        print('LR helper two-process determinism PASS', flush=True)
        memory_guard(900)
        manifest, pop = load_population()
        provenance['input_manifest_sha256'] = sha(RUNS['level_reduction_v1'] / 'INPUT_MANIFEST.json')
        jobs = runner_order(set(pop.cells), manifest['banks'])
        indices = {stem_of(j): i for i, j in enumerate(jobs)}
        directory = RUNS['level_reduction_v1'] / 'lr'; directory.mkdir(exist_ok=True)
        status_path = WORK / 'WORKFLOW_STATUS.json'; log_path = WORK / 'RESUME_20261004.log'
        active_bank = None; data = None; state = None; state_key = None
        from spectral_utils.monday_fusion_replay import HistoricalData
        provenance['status'] = 'RUNNING'; dump(record, provenance)
        for job in helper_order(jobs, manifest['banks']):
            if stop.exists():
                provenance['status'] = 'STOPPED'; break
            stem = stem_of(job); index = indices[stem]
            def can_write():
                if 'Completed checkpoint changed' in read_log(log_path):
                    raise RuntimeError('Runner reported Completed checkpoint changed; helper stopped')
                if stop.exists() or completed_chunk(directory, stem):
                    return False
                return reach_seconds(jobs, index, directory, status_path, log_path) > SAFE_SECONDS
            if not can_write():
                continue
            if any((directory / (stem + s)).exists() for s in SUFFIXES):
                provenance['discarded'].append({'stem': stem, 'reason': 'unsealed artifacts exist'}); continue
            storage_guard()
            ce, bk, variant, roster, pos = job
            if active_bank != bk:
                state = None; data = None; gc.collect()
                memory_guard()
                data = HistoricalData(manifest, pop, banks=[bk]); active_bank = bk; state_key = None
            if state_key != job[:3]:
                state = restore_content(data, bk, ce, variant)
                verification = verify_restored(state, ce, bk, variant); state_key = job[:3]
            # Loading and replay can consume time too: recheck before fitting.
            if not can_write():
                continue
            result = compute(data, state, job)
            if not can_write():
                provenance['discarded'].append({'stem': stem, 'reason': 'runner approached during fit'}); dump(record, provenance); continue
            rows, folds, diag, status, elapsed = result
            staging = provenance_dir / 'pending'; staging.mkdir(exist_ok=True)
            metric_temp = staging / (stem + '.metrics.csv.tmp')
            pd.DataFrame(rows).to_csv(metric_temp, index=False)
            metric_temp.replace(staging / (stem + '.metrics.csv'))
            dump(staging / (stem + '.fits.json'), native(diag))
            dump(staging / (stem + '.fold_metrics.json'), native(folds))
            dump(staging / (stem + '.complete.json'), {'status': status, 'runtime_s': elapsed,
                 'artifacts': {s: sha(staging / (stem + s)) for s in SUFFIXES}})
            storage_guard()
            written = exclusive_publish(staging, directory, stem, can_write)
            if written:
                completed_chunk(directory, stem)
                provenance['written'].append({'stem': stem, 'status': status, 'time': now(), 'runtime_s': elapsed,
                     'complete_sha256': sha(directory / (stem + '.complete.json')), 'restored_verification': verification})
                provenance['arm_failures'] += status == 'ARM_FAILURE'
                print('LR helper checkpoint ' + stem + ' ' + status, flush=True)
            else:
                provenance['discarded'].append({'stem': stem, 'reason': 'publication collision guard'})
            # Remove only this helper's staging links. Published files remain sealed.
            for suffix in (*SUFFIXES, '.complete.json'):
                (staging / (stem + suffix)).unlink()
            provenance['runner_checkpoint_changed_seen'] = 'Completed checkpoint changed' in read_log(log_path)
            dump(record, provenance)
            if provenance['runner_checkpoint_changed_seen']:
                raise RuntimeError('Runner reported Completed checkpoint changed; helper stopped')
            if args.max_stems and len(provenance['written']) >= args.max_stems:
                provenance['status'] = 'LIMIT_REACHED'; break
        else:
            provenance['status'] = 'NO_MORE_SAFE_CANDIDATES'
    except BaseException as exc:
        provenance['status'] = 'FAILED'; provenance['error'] = repr(exc)
        raise
    finally:
        provenance['ended'] = now(); provenance['written_count'] = len(provenance['written'])
        dump(record, provenance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-head')
    parser.add_argument('--stop-file')
    parser.add_argument('--max-stems', type=int)
    parser.add_argument('--worker-check')
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    lock = nullcontext() if args.worker_check else ownership(WORK / 'LR_HELPER_OWNER.json')
    with lock, threadpool_limits(limits=2):
        run(args)
