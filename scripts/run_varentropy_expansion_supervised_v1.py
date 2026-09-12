"""Supervised step-level linear diagnostic on the expansion banks (other-answer access; not a ceiling).

Stage 1 caches each answer's standardized bank (float32) in a sqlite checkpoint;
stage 2 fits one linear model per (cell, bank, outer fold) on the training folds
and scores the held fold; stage 3 evaluates with the frozen gate/readout/metrics.
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_direct_probability_temporal as base
from scripts.run_varentropy_expansion_fusion_v1 import (MEMORY_GATE_KB, OUT as PARENT, atomic_json_retry, clean, smoke_indices, wait_for_memory)
from spectral_utils.varentropy_expansion_supervised import (BANKS, bank_columns, fit, score_steps, standardized_bank, step_labels)
from spectral_utils.varentropy_expansion_fusion import WIDTH

OUT = PARENT / 'supervised'
METHODS = tuple(f'sup__{bank}' for bank in BANKS)
base.atomic_json = atomic_json_retry


def manifest_for(source):
    files = [Path(__file__), ROOT / 'spectral_utils/varentropy_expansion_supervised.py', ROOT / 'spectral_utils/varentropy_expansion_fusion.py',
             ROOT / 'spectral_utils/varentropy_contribution_fusion.py', ROOT / 'scripts/run_varentropy_expansion_fusion_v1.py',
             ROOT / 'docs/experiments/VARENTROPY_EXPANSION_FUSION_V1.md', base.old.BENCH / 'evaluation/JOINED.json',
             base.old.BENCH / 'evaluation/JOINED.npz', base.old.FOLDS, base.old.FIXED_GATE / 'DETECTORS.npz', base.old.FIXED_GATE / 'METRICS.json']
    return dict(schema='varentropy-expansion-supervised-v1', methods=list(METHODS), access='supervised; other-answer step labels; source-group-disjoint outer folds',
                hashes={str(p): base.old.sha256_file(p) for p in files}, source_root=str(source))


def extract(con, records, *, smoke, gates):
    done = {r[0] for r in con.execute('SELECT idx FROM answers')}; started = time.perf_counter()
    for cell, path, kind, dataset in base.source_specs():
        all_indices = [i for i, r in enumerate(records) if r['cell'] == cell]
        indices = [i for i in all_indices if i not in done]
        if not indices: continue
        wait_for_memory(gates[kind]); print('[extract]', cell, len(indices), flush=True)
        rows = base.old._source_row_map(base.old.load_pickle(path), kind=kind, dataset=dataset)
        if smoke: indices = [i for i in smoke_indices(all_indices, rows, records) if i not in done]
        for i in indices:
            r = records[i]; row = rows[r['row_id']]
            lp = np.asarray(base.old._topk_payload(row)['logprobs'], float); spans = np.asarray(row['step_token_spans'], int)
            if spans.shape != (r['steps'], 2): raise ValueError(f'{r["uid"]}: step count mismatch')
            z, keep = standardized_bank(lp, np.asarray(row['token_spilled_energies'], float))
            info = dict(uid=r['uid'], n_tokens=int(len(z)), n_steps=int(len(spans)), active_columns=int(keep.sum()))
            con.execute('INSERT INTO answers VALUES (?,?,?)', (i, base.packed(z=z.astype(np.float32), keep=keep, spans=spans), base.dumps(info)))
            done.add(i)
        con.commit(); del rows
        base.atomic_json(OUT / ('SMOKE_STATE.json' if smoke else 'RUN_STATE.json'),
                         dict(status='EXTRACTING', completed=len(done), expected=len(records), elapsed_seconds=time.perf_counter() - started))


def load_answers(con, indices):
    out = {}
    for i in indices:
        row = con.execute('SELECT payload FROM answers WHERE idx=?', (i,)).fetchone()
        if row is None: continue
        with np.load(io.BytesIO(row[0]), allow_pickle=False) as z: out[i] = (z['z'], z['spans'])
    return out


def fit_folds(con, records, joined, *, smoke):
    con.execute('CREATE TABLE IF NOT EXISTS fits (key TEXT PRIMARY KEY, theta BLOB, info TEXT NOT NULL)')
    folds = json.loads(base.old.FOLDS.read_text(encoding='utf8'))['outer']
    outer = np.array([int(folds[r['group_id']]) for r in records]); offsets = joined['offsets']; labels = joined['labels']; target = joined['target']
    present = {r[0] for r in con.execute('SELECT idx FROM answers')}
    scores = {m: np.full(int(offsets[-1]), np.nan) for m in METHODS}
    fitted = {r[0] for r in con.execute('SELECT key FROM fits')}
    for cell, _, kind, _ in base.source_specs():
        cell_indices = [i for i, r in enumerate(records) if r['cell'] == cell and i in present]
        if not cell_indices: continue
        answers = load_answers(con, cell_indices)
        for bank in BANKS:
            cols = bank_columns(bank)
            for f in range(5):
                key = f'{cell}|{bank}|{f}'
                train = [i for i in cell_indices if outer[i] != f]; test = [i for i in cell_indices if outer[i] == f]
                if key in fitted:
                    theta_blob, info_json = con.execute('SELECT theta,info FROM fits WHERE key=?', (key,)).fetchone()
                    info = json.loads(info_json); theta = np.load(io.BytesIO(theta_blob))['theta'] if theta_blob is not None else None
                else:
                    started = time.perf_counter(); theta = None
                    try:
                        if not train or not test: raise ValueError('empty training or held fold (declared fold failure)')
                        assert not {records[i]['group_id'] for i in train} & {records[i]['group_id'] for i in test}
                        x = np.concatenate([answers[i][0][:, cols] for i in train]); spans = []; y = []; base_offset = 0
                        for i in train:
                            z, sp = answers[i]; spans.append(sp + base_offset); base_offset += len(z)
                            y.append(step_labels(kind, len(sp), target=target[i], labels=labels[offsets[i]:offsets[i + 1]]))
                        theta, info = fit(x, np.concatenate(spans), np.concatenate(y))
                        info.update(n_train_answers=len(train), n_test_answers=len(test), n_train_tokens=int(len(x)))
                        if info['status'] == 'STALLED': theta = None     # declared: a stalled model scores nothing
                    except (ValueError, RuntimeError, AssertionError, np.linalg.LinAlgError) as error:
                        info = dict(status='FAILED', reason=f'{type(error).__name__}: {error}', n_train_answers=len(train), n_test_answers=len(test))
                    info.update(seconds=time.perf_counter() - started, cell=cell, bank=bank, fold=f)
                    con.execute('INSERT INTO fits VALUES (?,?,?)', (key, base.packed(theta=theta) if theta is not None else None, base.dumps(clean(info))))
                    con.commit(); fitted.add(key)
                    print('[fit]', key, info['status'], round(info['seconds'], 1), 's', flush=True)
                if theta is not None:
                    for i in test:
                        z, sp = answers[i]; scores[f'sup__{bank}'][offsets[i]:offsets[i + 1]] = score_steps(z[:, cols], sp, theta)
        del answers
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True); p.add_argument('--smoke', action='store_true')
    p.add_argument('--workers', type=int, default=1); p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--memory-gate-pb-kb', type=int, default=MEMORY_GATE_KB['pb']); p.add_argument('--memory-gate-prmb-kb', type=int, default=MEMORY_GATE_KB['prm'])
    args = p.parse_args(); source = args.source_root.resolve(); base.old.configure_source_root(source); OUT.mkdir(parents=True, exist_ok=True)
    manifest = manifest_for(source)
    con = base.connect(OUT / ('SMOKE.sqlite' if args.smoke else 'CHECKPOINT.sqlite'), manifest)
    base.atomic_json(OUT / ('SMOKE_MANIFEST.json' if args.smoke else 'MANIFEST.json'), manifest)
    records = json.loads((base.old.BENCH / 'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined = np.load(base.old.BENCH / 'evaluation/JOINED.npz', allow_pickle=False)
    if len(records) != 13769: raise ValueError('benchmark roster mismatch')
    with threadpool_limits(limits=1):
        if not args.evaluate_only: extract(con, records, smoke=args.smoke, gates={'pb': args.memory_gate_pb_kb, 'prm': args.memory_gate_prmb_kb})
        n = con.execute('SELECT COUNT(*) FROM answers').fetchone()[0]
        scores = fit_folds(con, records, joined, smoke=args.smoke)
        fits = [json.loads(r[0]) for r in con.execute('SELECT info FROM fits')]
        if args.smoke:
            base.atomic_json(OUT / 'SMOKE.json', dict(status='PASS' if all(f['status'] in ('FIT', 'FAILED', 'STALLED') for f in fits) else 'FAIL', scope='MECHANICS_ONLY',
                                                      n_answers=n, n_fits=sum(f['status'] == 'FIT' for f in fits), n_declared_fold_failures=sum(f['status'] == 'FAILED' for f in fits),
                                                      n_stalled_fits=sum(f['status'] == 'STALLED' for f in fits),
                                                      scored_answers={m: int(sum(np.isfinite(scores[m][joined['offsets'][i]:joined['offsets'][i + 1]]).all() for i in range(len(records)))) for m in METHODS},
                                                      fits=fits, access='supervised; other-answer access; matched diagnostic, not a ceiling'))
            base.atomic_json(OUT / 'SMOKE_STATE.json', dict(status='SMOKE_COMPLETE', completed=n, expected=len(records)))
            print('[smoke] supervised mechanics complete;', len(fits), 'fold fits attempted', flush=True)
        elif n == len(records):
            metrics, per = base.evaluate_arrays(records, joined, scores)
            base.atomic_json(OUT / 'METRICS.json', clean(dict(schema='varentropy-expansion-supervised-v1', access='supervised; other-answer access; matched diagnostic, not a ceiling',
                                                               metrics=metrics, fits=fits)))
            np.savez_compressed(OUT / 'SCORES.npz', **{'steps__' + m: s for m, s in scores.items()},
                                **{'prediction__' + m: p['prediction'] for m, p in per.items()}, **{'valid__' + m: p['valid'] for m, p in per.items()})
            base.atomic_json(OUT / 'RUN_STATE.json', dict(status='COMPLETE', completed=n, expected=len(records)))
            for m, x in metrics.items(): print(m, x['pb_all8'], x['prm_within'], x['prm_pooled'], x['prmscore_q08'], x['valid_answers'], flush=True)
        else:
            print('[extract]', n, '/', len(records), 'cached; full evaluation waits for completion', flush=True)
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
