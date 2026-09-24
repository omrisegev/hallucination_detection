"""TRANSFER_LOCK_V3 external rescoring: V2's eleven arms plus three maximal-step (top-1) tail rows
(F15_tail1s_lsml, B11_tail1s_lsml, B11o_tail1s_lsml), from the step features ALREADY extracted and
sealed by the V1 run.  No label access, no new extraction, no target fitting.  All eleven V2 arms
must replay the sealed V2 records exactly.  Writes results/family_tail_external_v3/.

    python scripts/run_family_external_v3.py
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from spectral_utils.family_tail_transfer import answer_standardize, build_representations, load_lock  # noqa: E402
from spectral_utils.external_generalization.artifacts import RecordStore, atomic_json, file_hash  # noqa: E402

V1OUT = ROOT/'results/family_tail_external_v1'; V2OUT = ROOT/'results/family_tail_external_v2'; OUT = ROOT/'results/family_tail_external_v3'
LOCK2 = ROOT/'results/family_tail_transfer_v2/TRANSFER_LOCK_V2.json'
LOCK2_SHA256 = '0c5c55996503394edf2faeb1bc066e3fdec3c6eb315c79c0e7d59ca7d61c986c'
LOCK3 = ROOT/'results/family_tail_transfer_v2/TRANSFER_LOCK_V3.json'
LOCK3_SHA256 = '0fc561b7e382814cacdaa3a6425916cdba5119d03b48f2839729459a7e58c8ee'
CELLS = ('hard2verify_qwen3_8b', 'socratic_qwen3_8b', 'socratic_qwq32b')
NEW = ('F15_tail1s_lsml', 'B11_tail1s_lsml', 'B11o_tail1s_lsml')


def load_lock3():
    raw, raw2 = LOCK3.read_bytes(), LOCK2.read_bytes()
    if hashlib.sha256(raw).hexdigest() != LOCK3_SHA256 or hashlib.sha256(raw2).hexdigest() != LOCK2_SHA256:
        raise ValueError('frozen transfer lock V3/V2 changed')
    lock, v2 = json.loads(raw), json.loads(raw2)
    if (any(lock['recipe'][k] != v for k, v in v2['recipe'].items()) or any(lock['deployment'][m] != v2['deployment'][m] for m in v2['deployment'])
            or list(lock['rows'])[:len(v2['rows'])] != list(v2['rows'])):
        raise ValueError('V3 must carry every V2 recipe item, row and deployment unchanged')
    return lock


def score(lock, features, names, offsets, methods):
    """score_locked with the lock passed in (identical algebra)."""
    reps = build_representations(features, names, offsets, include_all=True)
    a48, cols48 = reps['A48o']
    reps['B11o'] = (a48[:, [cols48.index(c) for c in lock['recipe']['bank11']]], lock['recipe']['bank11'])
    out = {}
    for method in methods:
        matrix, columns = reps[method.split('_')[0]]; record = lock['deployment'][method]
        w = np.array([record['weights'][c] for c in columns])
        values = answer_standardize(matrix@w, offsets, final=True); tau = record['q80_threshold_fold4']
        out[method] = {'scores': values, 'pred_valid': (values < tau).astype(np.int8)}
    return out


def main():
    lock = load_lock3(); arms = list(lock['rows']); methods = [m for m in arms if m != 'ct7']
    freeze = json.loads((OUT/'IMPLEMENTATION_FREEZE.json').read_text())
    for rel, expected in freeze['files'].items():
        if file_hash(ROOT/rel) != expected:
            raise ValueError('changed frozen code/input: '+rel)
    identity = {'lock_v3': LOCK3_SHA256, 'implementation_freeze': file_hash(OUT/'IMPLEMENTATION_FREEZE.json'),
                'v1_sealed': file_hash(V1OUT/'ALL_CELLS_SEALED.json'), 'v2_sealed': file_hash(V2OUT/'ALL_CELLS_SEALED.json'), 'arms': arms}
    started = time.perf_counter(); replay = {}; n = 0
    for cell in CELLS:
        v1seal = json.loads((V1OUT/cell/'SEAL.json').read_text())
        paths = sorted((V1OUT/cell/'shard_000').glob('*.record.json'))
        if len(paths) != v1seal['answers']:
            raise ValueError('incomplete V1 population: '+cell)
        worst = {m: 0.0 for m in methods if m not in NEW}
        with RecordStore(OUT/cell/'shard_000', identity) as store:
            for path in paths:
                t0 = time.perf_counter(); rec = json.loads(path.read_text()); old = rec['payload']
                ref2 = json.loads((V2OUT/cell/'shard_000'/path.name).read_text())['payload']
                if ref2['v1_record_sha256'] != file_hash(path):
                    raise ValueError('V2 record does not point to this V1 record')
                feats = np.asarray(old['features'], float); valid = np.asarray(old['nonempty'], bool)
                arms_out = score(lock, feats, old['feature_names'], np.array([0, len(feats)]), methods)
                scores, predictions = {}, {}
                for m, item in arms_out.items():
                    values = np.full(len(valid), np.nan); values[valid] = item['scores']
                    pred = np.zeros(len(valid), int); pred[valid] = item['pred_valid']
                    scores[m] = [float(v) if np.isfinite(v) else None for v in values]; predictions[m] = pred.tolist()
                    if m not in NEW:
                        ref = np.asarray([np.nan if v is None else v for v in ref2['scores'][m]])
                        worst[m] = max(worst[m], float(np.max(np.abs(values[valid]-ref[valid]))))
                        if worst[m] > 1e-12 or predictions[m] != ref2['predictions'][m]:
                            raise ValueError(f'V2 arm replay failed: {cell} {m}')
                scores['ct7'], predictions['ct7'] = old['scores']['ct7'], old['predictions']['ct7']
                store.put(rec['uid'], {'scores': scores, 'predictions': predictions, 'nonempty': old['nonempty'],
                                       'telemetry_sha256': old['telemetry_sha256'], 'v1_record_sha256': file_hash(path),
                                       'v2_record_sha256': file_hash(V2OUT/cell/'shard_000'/path.name),
                                       'features_sha256': hashlib.sha256(feats.tobytes()).hexdigest(), 'tokens': old['tokens'],
                                       'cpu_seconds': old['cpu_seconds'], 'process_cpu_seconds': old['process_cpu_seconds'],
                                       'v2_rescoring_seconds': time.perf_counter()-t0,
                                       'ct7_provenance': 'reused sealed V1 prediction on identical telemetry'})
                n += 1
        replay[cell] = {'answers': len(paths), 'v2_arm_max_abs_score_diff': worst, 'v2_arm_predictions_identical': True}
        print(cell, 'rescored', len(paths), 'answers; V2 arm max diff', max(worst.values()), flush=True)
    atomic_json(OUT/'V2_REPLAY.json', replay)
    atomic_json(OUT/'CPU_EXECUTION.json', {'identity': identity, 'records_this_run': n, 'elapsed_seconds': time.perf_counter()-started,
                                           'new_gpu_hours': 0, 'feature_extraction': 'reused from the sealed V1 records (no new extraction)'})


if __name__ == '__main__':
    main()
