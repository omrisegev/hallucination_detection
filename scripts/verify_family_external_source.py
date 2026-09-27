"""Full raw-source -> 48 locked step channels parity gate, without quality scoring."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.family_external_features import step_features
from spectral_utils.family_tail_transfer import answer_standardize, load_lock
from spectral_utils.external_generalization.artifacts import file_hash, atomic_json


def source_specs():
    for model in ('4b', '8b'):
        for dataset in ('gsm8k', 'math', 'olympiadbench', 'omnimath'):
            yield f'pb_{dataset}_q{model[0]}', ROOT/f'dataset_cache/repgrid/pb_qwen3_{model}/processbench_{dataset}.pkl', dataset
    yield 'prmbench_qwen3_8b', ROOT/'dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl', None


def safe_row(row):
    safe = {key: row[key] for key in ('gen_token_ids', 'token_entropies', 'token_spilled_energies', 'token_logsumexp', 'step_token_spans')}
    safe['top_k_logprobs'] = row.get('top_k_logprobs') or row.get('top_k_logprobs_raw')
    return safe


def one(item):
    i, row = item
    started = time.process_time()
    matrix, names = step_features(row)
    return i, matrix, time.process_time()-started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit-per-cell', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a new immutable gate directory: '+str(args.output))
    args.output.mkdir(parents=True)
    lock = load_lock(); names = lock['recipe']['channels_48']
    poolpath = args.pool_dir/'pool_z.npy'
    assert file_hash(poolpath) == 'd9abccfb561f459d2f3a6c5b4346a1f6766df7fdbaddb230338aa40349077a16'
    pool = np.load(poolpath, mmap_mode='r')
    pn = json.loads((args.pool_dir/'pool_names.json').read_text())
    cols = [pn.index(name) for name in names]
    roster = json.loads((ROOT/'results/localization_full_benchmark_v3/evaluation/JOINED.json').read_text())['records']
    off = np.r_[0, np.cumsum([r['steps'] for r in roster])]
    assert len(roster) == 13769 and off[-1] == 145597
    expected_hashes = json.loads((ROOT/'results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.MANIFEST.json').read_text())['sources_sha256']
    maxima = np.zeros(len(names)); checked = 0; checked_steps = 0; cpu = 0.; started = time.perf_counter()
    bycell = {}; sources = {}; feature_paths = []
    report = {'status': 'RUNNING', 'n_total': len(roster), 'channels': names, 'tolerance': 1e-6,
              'external_quality_computed': False, 'limit_per_cell': args.limit_per_cell,
              'command': ' '.join(sys.argv), 'pool_sha256': file_hash(poolpath)}
    atomic_json(args.output/'GATE.json', report)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for cell, path, dataset in source_specs():
            digest = file_hash(path)
            assert digest == expected_hashes[path.relative_to(ROOT).as_posix()], path
            sources[cell] = {'path': str(path), 'sha256': digest}
            with path.open('rb') as stream:
                payload = pickle.load(stream)
            rows = list(payload.values()) if isinstance(payload, dict) else payload
            lookup = {}
            for row in rows:
                if not isinstance(row, dict): continue
                key = f'{dataset}::{row.get("id")}' if dataset else str(row.get('idx'))
                if key in lookup: raise ValueError('duplicate raw ID '+key)
                lookup[key] = row
            indexes = [i for i,r in enumerate(roster) if r['cell']==cell]
            if args.limit_per_cell:
                ordered = sorted(indexes, key=lambda i:(roster[i]['tokens'], roster[i]['uid']))
                indexes = sorted({ordered[int(j)] for j in np.linspace(0, len(ordered)-1, args.limit_per_cell)})
            cell_features = []; cell_indexes = []; cell_max = np.zeros(len(names))
            # Executor buffersize bounds pending raw-row copies and memory use.
            def jobs():
                for i in indexes:
                    row = safe_row(lookup[str(roster[i]['row_id'])])
                    assert len(row['step_token_spans']) == roster[i]['steps']
                    assert len(row['gen_token_ids']) == roster[i]['tokens']
                    yield i, row
            # Python 3.13 has no map buffersize; one small batch per worker keeps memory bounded.
            iterator = iter(jobs())
            while True:
                batch = []
                for _ in range(args.workers*2):
                    try: batch.append(next(iterator))
                    except StopIteration: break
                if not batch: break
                for i, matrix, seconds in executor.map(one, batch):
                    a,b = off[i:i+2]
                    z = answer_standardize(matrix, np.array([0, len(matrix)]))
                    delta = np.max(np.abs(z-pool[a:b][:, cols]), axis=0)
                    maxima = np.maximum(maxima, delta); cell_max = np.maximum(cell_max, delta)
                    checked += 1; checked_steps += len(matrix); cpu += seconds
                    if np.any(delta>1e-6):
                        failure = dict(report, status='FAIL', n_checked=checked, cell=cell, uid=roster[i]['uid'],
                                       errors={name:float(x) for name,x in zip(names,delta) if x>1e-6}, sources=sources)
                        atomic_json(args.output/'GATE.json', failure)
                        np.savez_compressed(args.output/'FAILED_FEATURES.npz', got=matrix, expected=pool[a:b][:,cols], names=names)
                        raise AssertionError(failure['errors'])
                    cell_features.append(matrix); cell_indexes.append(i)
                    if checked%100==0: print('source parity',checked,'/13769',cell,'elapsed',round(time.perf_counter()-started),flush=True)
            dest=args.output/(cell+'.npz')
            np.savez_compressed(dest, features=np.concatenate(cell_features), indexes=cell_indexes, names=names)
            feature_paths.append({'path':str(dest), 'sha256':file_hash(dest)})
            bycell[cell]={'answers':len(cell_indexes), 'maximum_error':float(cell_max.max())}
            print('CELL PASS',cell,len(cell_indexes),float(cell_max.max()),flush=True)
            del payload, rows, lookup, cell_features
            gc.collect()
    report.update(status='PASS', n_checked=checked, steps=checked_steps,
                  scope='FULL' if checked==13769 else 'FEASIBILITY',
                  per_channel_max_error=dict(zip(names,map(float,maxima))), cells=bycell, sources=sources,
                  features=feature_paths, elapsed_seconds=time.perf_counter()-started, process_cpu_seconds=cpu)
    atomic_json(args.output/'GATE.json',report)
    print('PASS',checked,'/13769',float(maxima.max()),flush=True)


if __name__=='__main__': main()
