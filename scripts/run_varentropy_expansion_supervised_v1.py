"""Supervised step-level linear diagnostic on the expansion banks (other-answer access; not a ceiling).

Stage 1 caches each answer's standardized bank (float32) in a sqlite checkpoint;
stage 2 fits one linear model per (cell, bank, outer fold) on the training folds
and scores the held fold; stage 3 evaluates with the frozen gate/readout/metrics.

Correction 2026-09-13 (audit of the completed run; the frozen run itself is not
rewritten, its checkpoint manifest binds the earlier code hashes):
* the manifest hashes the raw source artifacts (8 PB pickles, PRMB telemetry
  and PRMB labels) and the test file, and a checkpoint whose stored manifest
  differs is refused with the differing keys named;
* extraction asserts the uid/row_id mapping, spans equal (values) to the frozen
  BENCH/scores/<uid>.npz boundaries, token count equal to len(token_entropies)
  and to the top-K rows, top-K/selected-surprisal alignment and PB gate
  detector equality, and stores per-answer provenance in the cached info;
* smoke status follows the fit-outcome rule in
  ``spectral_utils.varentropy_expansion_supervised.smoke_status`` (an all-failed
  run cannot PASS), with FIT_HEALTH written beside SMOKE.json.
The PRMScore threshold of this driver's evaluation still follows the shared
``evaluate_arrays`` convention; the held-fold-blind calibration is the separate
``scripts/run_varentropy_expansion_supervised_calibration_correction.py``.
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
from spectral_utils.varentropy_expansion_supervised import (BANKS, bank_columns, fit, fit_health, manifest_differences, score_steps,
                                                            smoke_status, standardized_bank, step_labels, verify_answer)
from spectral_utils.varentropy_expansion_fusion import WIDTH

OUT = PARENT / 'supervised'
METHODS = tuple(f'sup__{bank}' for bank in BANKS)
base.atomic_json = atomic_json_retry


def raw_source_paths():
    """The raw artifacts the extraction reads: 8 PB pickles, PRMB telemetry, PRMB labels."""
    return [path for _, path, _, _ in base.source_specs()] + [base.old.PRMB_LABELS]


def manifest_for(source):
    files = [Path(__file__), ROOT / 'spectral_utils/varentropy_expansion_supervised.py', ROOT / 'spectral_utils/varentropy_expansion_fusion.py',
             ROOT / 'spectral_utils/varentropy_contribution_fusion.py', ROOT / 'scripts/run_varentropy_expansion_fusion_v1.py',
             ROOT / 'scripts/test_varentropy_expansion_supervised.py',
             ROOT / 'docs/experiments/VARENTROPY_EXPANSION_FUSION_V1.md', base.old.BENCH / 'evaluation/JOINED.json',
             base.old.BENCH / 'evaluation/JOINED.npz', base.old.FOLDS, base.old.FIXED_GATE / 'DETECTORS.npz', base.old.FIXED_GATE / 'METRICS.json']
    hashes = {str(p): base.old.sha256_file(p) for p in files}
    raw = {str(p): base.old.sha256_file(p) for p in raw_source_paths()}
    hashes.update(raw)
    return dict(schema='varentropy-expansion-supervised-v1.1', methods=list(METHODS), access='supervised; other-answer step labels; source-group-disjoint outer folds',
                hashes=hashes, raw_source_hashes=raw, source_root=str(source))


def connect(path, manifest):
    """base.connect with the differing manifest keys named when a frozen checkpoint is refused."""
    if path.exists():
        import sqlite3
        con = sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True)
        try:
            existing = con.execute('SELECT payload FROM manifest WHERE id=1').fetchone()
        except sqlite3.OperationalError:
            existing = None
        con.close()
        if existing and json.loads(existing[0]) != manifest:
            diff = manifest_differences(json.loads(existing[0]), manifest)
            raise ValueError(f'checkpoint {path.name} manifest mismatch; do not overwrite frozen checkpoint. Differing keys: {diff}')
    return base.connect(path, manifest)


def extract(con, records, *, smoke, gates, out, manifest):
    done = {r[0] for r in con.execute('SELECT idx FROM answers')}; started = time.perf_counter()
    detector, _ = base.old._gate_contract(records)
    for cell, path, kind, dataset in base.source_specs():
        all_indices = [i for i, r in enumerate(records) if r['cell'] == cell]
        indices = [i for i in all_indices if i not in done]
        if not indices: continue
        wait_for_memory(gates[kind]); print('[extract]', cell, len(indices), flush=True)
        rows = base.old._source_row_map(base.old.load_pickle(path), kind=kind, dataset=dataset)
        source_sha = manifest['raw_source_hashes'][str(path)]
        if smoke: indices = [i for i in smoke_indices(all_indices, rows, records) if i not in done]
        for i in indices:
            r = records[i]; row = rows[r['row_id']]
            with np.load(base.old.BENCH / 'scores' / f'{r["uid"]}.npz', allow_pickle=False) as z:
                provenance = verify_answer(r, row, kind=kind, dataset=dataset, bench_starts=z['step_starts'], bench_ends=z['step_ends'],
                                           detector=detector[i] if kind == 'pb' else None)
            lp = np.asarray(base.old._topk_payload(row)['logprobs'], float); spans = np.asarray(row['step_token_spans'], int)
            if spans.shape != (r['steps'], 2): raise ValueError(f'{r["uid"]}: step count mismatch')
            z, keep = standardized_bank(lp, np.asarray(row['token_spilled_energies'], float))
            if len(z) != provenance['n_tokens']: raise ValueError(f'{r["uid"]}: bank rows != token count')
            info = dict(uid=r['uid'], n_tokens=int(len(z)), n_steps=int(len(spans)), active_columns=int(keep.sum()),
                        provenance=dict(source_path=str(path), source_sha256=source_sha, **provenance))
            con.execute('INSERT INTO answers VALUES (?,?,?)', (i, base.packed(z=z.astype(np.float32), keep=keep, spans=spans), base.dumps(clean(info))))
            done.add(i)
        con.commit(); del rows
        base.atomic_json(out / ('SMOKE_STATE.json' if smoke else 'RUN_STATE.json'),
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
    p.add_argument('--out', type=Path, default=OUT, help='output directory (default: the frozen supervised directory)')
    p.add_argument('--memory-gate-pb-kb', type=int, default=MEMORY_GATE_KB['pb']); p.add_argument('--memory-gate-prmb-kb', type=int, default=MEMORY_GATE_KB['prm'])
    args = p.parse_args(); source = args.source_root.resolve(); base.old.configure_source_root(source); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest = manifest_for(source)
    con = connect(out / ('SMOKE.sqlite' if args.smoke else 'CHECKPOINT.sqlite'), manifest)
    base.atomic_json(out / ('SMOKE_MANIFEST.json' if args.smoke else 'MANIFEST.json'), manifest)
    records = json.loads((base.old.BENCH / 'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined = np.load(base.old.BENCH / 'evaluation/JOINED.npz', allow_pickle=False)
    if len(records) != 13769: raise ValueError('benchmark roster mismatch')
    with threadpool_limits(limits=1):
        if not args.evaluate_only:
            extract(con, records, smoke=args.smoke, gates={'pb': args.memory_gate_pb_kb, 'prm': args.memory_gate_prmb_kb}, out=out, manifest=manifest)
        n = con.execute('SELECT COUNT(*) FROM answers').fetchone()[0]
        scores = fit_folds(con, records, joined, smoke=args.smoke)
        fits = [json.loads(r[0]) for r in con.execute('SELECT info FROM fits')]
        health = fit_health(fits)
        if args.smoke:
            status, _ = smoke_status(fits, BANKS)
            base.atomic_json(out / 'FIT_HEALTH.json', clean(health))
            base.atomic_json(out / 'SMOKE.json', dict(status=status, scope='MECHANICS_ONLY',
                                                      rule='PASS: >=1 FIT/FIT_ITERATION_LIMIT per bank, no UNEXPECTED_FAILURE, no STALLED; '
                                                           'INCONCLUSIVE: a bank without a successful fit but only expected limitations; FAIL otherwise',
                                                      n_answers=n, n_fits=health['n_converged'], n_iteration_limit_fits=health['n_iteration_limit'],
                                                      n_declared_fold_failures=sum(f['status'] == 'FAILED' for f in fits),
                                                      n_expected_limitations=health['n_expected_limitations'], n_unexpected_failures=health['n_unexpected_failures'],
                                                      n_stalled_fits=health['n_stalled'], outcomes=health['by_cell_bank_fold'],
                                                      scored_answers={m: int(sum(np.isfinite(scores[m][joined['offsets'][i]:joined['offsets'][i + 1]]).all() for i in range(len(records)))) for m in METHODS},
                                                      fits=fits, access='supervised; other-answer access; matched diagnostic, not a ceiling'))
            base.atomic_json(out / 'SMOKE_STATE.json', dict(status='SMOKE_COMPLETE', completed=n, expected=len(records)))
            print('[smoke] supervised mechanics complete;', len(fits), 'fold fits attempted; status', status, flush=True)
        elif n == len(records):
            metrics, per = base.evaluate_arrays(records, joined, scores)
            base.atomic_json(out / 'FIT_HEALTH.json', clean(health))
            base.atomic_json(out / 'METRICS.json', clean(dict(schema='varentropy-expansion-supervised-v1', access='supervised; other-answer access; matched diagnostic, not a ceiling',
                                                               metrics=metrics, fits=fits, fit_health={k: v for k, v in health.items() if k != 'by_cell_bank_fold'})))
            np.savez_compressed(out / 'SCORES.npz', **{'steps__' + m: s for m, s in scores.items()},
                                **{'prediction__' + m: p['prediction'] for m, p in per.items()}, **{'valid__' + m: p['valid'] for m, p in per.items()})
            base.atomic_json(out / 'RUN_STATE.json', dict(status='COMPLETE', completed=n, expected=len(records)))
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
        out = OUT
        if '--out' in sys.argv: out = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
        path = out / name; state = json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED', error=f'{type(error).__name__}: {error}')
        base.atomic_json(path, state)
        raise
