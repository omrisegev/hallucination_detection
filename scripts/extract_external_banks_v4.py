"""external_banks_v4 extraction: 48 stored + 5 new + 3 digit raw step channels; no labels.

Stages (default: all, in order):
  tokenizers  digit-ID check: local tokenizer.json files (Qwen3-8B at the exact external
              revision; Qwen2.5 family as corroboration for QwQ-32B) AND an empirical decode
              check on every external record: public step text (inputs/<benchmark>/answers.json,
              no labels) + saved tokenizer character offsets must map ids 15..24 <-> '0'..'9'.
  source      combined source bank in JOINED/OOF order from the two passed parity gates.
  external    per external cell, per record (non-empty steps only, as run_family_external.py):
              new channels + digits joined with the stored 48 features.

Refuses to run unless the v4 source parity gate is PASS/FULL and this module's code hash
equals the hash recorded by that gate. Never reads evaluator_only inputs or any external
metric/contrast/report file; stored family records are read only for features/identity.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils import external_banks_v4 as E  # noqa: E402
from spectral_utils.family_tail_transfer import load_lock  # noqa: E402
from spectral_utils.external_generalization.artifacts import atomic_json, file_hash  # noqa: E402

OUT = ROOT/'results/external_banks_v4'
GATE = OUT/'source_parity/GATE.json'
FAMILY = MAIN/'results/family_tail_external_v1'
PRIVATE = MAIN/'scratch/external_generalization_private'
RAW = PRIVATE/'evaluation_archives'
ROSTER = MAIN/'results/localization_full_benchmark_v3/evaluation/JOINED.json'
CELLS = {'hard2verify_qwen3_8b': ('hard2verify', 200), 'socratic_qwen3_8b': ('socratic', 2995),
         'socratic_qwq32b': ('socratic', 2995)}
DIGIT_CHARS = '0123456789'


def sha_text(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def fill_missing(matrix):
    """family_external_features.step_features rule: NaN -> answer column mean, all-NaN -> 0."""
    x = np.array(matrix, float); filled = (~np.isfinite(x)).sum(axis=0)
    if np.isinf(x).any():
        raise ValueError('infinite feature values')
    for j in range(x.shape[1]):
        good = np.isfinite(x[:, j])
        x[~good, j] = x[good, j].mean() if good.any() else 0.
    return x, filled


def check_gate():
    gate = json.loads(GATE.read_text())
    if gate['status'] != 'PASS' or gate['n_checked'] != 13769 or gate['scope'] != 'FULL':
        raise ValueError('full v4 source parity PASS required before extraction')
    rel = 'spectral_utils/external_banks_v4.py'
    if file_hash(ROOT/rel) != gate['code_sha256'][rel]:
        raise ValueError('external_banks_v4.py changed after the parity gate')
    for rel, digest in gate['code_sha256'].items():
        if rel != 'scripts/verify_external_banks_v4_source.py' and file_hash(ROOT/rel) != digest:
            raise ValueError('dependency changed after the parity gate: '+rel)
    return gate


def code_hashes():
    gate = json.loads(GATE.read_text())
    out = dict(gate['code_sha256'])
    out['scripts/extract_external_banks_v4.py'] = file_hash(Path(__file__))
    return out


# ---------------------------------------------------------------- tokenizers
_STEPS = {}


def public_steps(bench):
    """Public step text per uid (inputs/<bench>/answers.json has no label fields)."""
    if bench not in _STEPS:
        answers = json.loads((PRIVATE/'inputs'/bench/'answers.json').read_text(encoding='utf-8'))
        _STEPS[bench] = {a['uid']: a['steps'] for a in answers}
    return _STEPS[bench]


def decode_check(job):
    """Empirical digit-ID check on one record from saved character offsets; no labels."""
    path, bench = job
    record = json.loads(Path(path).read_text()); t = record['payload']['telemetry']
    text, chars = '', []
    for step in public_steps(bench)[record['uid']]:  # contracts.Answer.chain
        if text:
            text += '\n\n'
        start = len(text); text += step; chars.append([start, len(text)])
    if [list(x) for x in t['step_char_spans']] != chars:
        raise ValueError('public step text does not reproduce saved character spans: '+record['uid'])
    ids = np.asarray(t['gen_token_ids']); offsets = t['token_offsets']
    counts = np.zeros(10, int); bad = 0; digit_char_tokens = 0
    for tid, (a, b) in zip(ids.tolist(), offsets):
        piece = text[a:b]
        if 15 <= tid <= 24:
            counts[tid-15] += 1
            bad += piece != DIGIT_CHARS[tid-15]
        if len(piece) == 1 and piece in DIGIT_CHARS:
            digit_char_tokens += 1
            bad += tid != 15+DIGIT_CHARS.index(piece)
    return counts, int(bad), digit_char_tokens, len(ids)


def tokenizers(workers):
    report = {'digit_ids': list(E.DIGIT_IDS), 'local_tokenizer_files': [], 'empirical': {}}
    files = sorted(glob.glob(str(MAIN/'results/automatic_group_free_phase_a6_s0a_v1/inputs/qwen3-8b*/tokenizer.json')))
    files += sorted(glob.glob(str(Path.home()/'.cache/huggingface/hub/models--Qwen--Q*/snapshots/*/tokenizer.json')))
    for path in files:
        vocab = json.loads(Path(path).read_text(encoding='utf-8'))['model']['vocab']
        ids = [vocab[c] for c in DIGIT_CHARS]
        report['local_tokenizer_files'].append({'path': path, 'sha256': file_hash(path),
                                                'digit_ids': ids, 'ok': ids == list(E.DIGIT_IDS)})
    qwen3_revision = 'b968826d9c46dd6066d109eabc6255188de91218'
    report['qwen3_8b_exact_revision_local'] = any(
        qwen3_revision in r['path'] and r['ok'] for r in report['local_tokenizer_files'])
    report['qwq32b_tokenizer_file_local'] = any('qwq' in r['path'].lower() for r in report['local_tokenizer_files'])
    report['qwen25_family_local_ok'] = [r['path'] for r in report['local_tokenizer_files']
                                        if 'Qwen2.5' in r['path'] and r['ok']]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for cell, (bench, expected) in CELLS.items():
            jobs = [(str(path), bench) for path in sorted((RAW/cell/'records').glob('*.record.json'))]
            assert len(jobs) == expected
            counts = np.zeros(10, int); bad = 0; digit_tokens = 0; tokens = 0
            for c, b, d, n in executor.map(decode_check, jobs, chunksize=16):
                counts += c; bad += b; digit_tokens += d; tokens += n
            identity = json.loads((RAW/cell/'TOKENIZATION.json').read_text())['identity']
            report['empirical'][cell] = {
                'model': identity['model'], 'revision': identity['revision'], 'records': len(jobs),
                'tokens': tokens, 'digit_id_counts': counts.tolist(), 'single_digit_char_tokens': digit_tokens,
                'mismatches': bad, 'all_ten_digits_seen': bool((counts > 0).all()),
                'ok': bad == 0 and bool((counts > 0).all()) and int(counts.sum()) == digit_tokens}
    report['status'] = 'PASS' if all(v['ok'] for v in report['empirical'].values()) and \
        report['qwen3_8b_exact_revision_local'] else 'FAIL'
    report['note'] = ('QwQ-32B tokenizer file is not available locally; its digit IDs are verified '
                      'empirically on every external record (saved tokenizer offsets vs public step text) '
                      'and corroborated by the local Qwen2.5-family vocabularies.'
                      if not report['qwq32b_tokenizer_file_local'] else '')
    atomic_json(OUT/'TOKENIZER_CHECK.json', report)
    print('tokenizers', report['status'], {k: (v['mismatches'], v['digit_id_counts']) for k, v in report['empirical'].items()},
          flush=True)
    if report['status'] != 'PASS':
        raise SystemExit('digit tokenizer check failed')
    return report


# ---------------------------------------------------------------- source
def source(gate):
    names48 = load_lock()['recipe']['channels_48']
    roster = json.loads(ROSTER.read_text())['records']
    off = np.r_[0, np.cumsum([r['steps'] for r in roster])]
    family_gate = json.loads((FAMILY/'source_full_v1/GATE.json').read_text())
    base = np.full((int(off[-1]), 48), np.nan); new = np.full((int(off[-1]), 5), np.nan)
    digits = np.full((int(off[-1]), 3), np.nan); active = np.zeros((int(off[-1]), 3), bool)
    seen48 = np.zeros(len(roster), bool); seen_new = np.zeros(len(roster), bool)
    for item in family_gate['features']:
        assert file_hash(item['path']) == item['sha256'], item['path']
        with np.load(item['path']) as z:  # NpzFile re-decompresses on every key access
            assert list(z['names']) == list(names48)
            indexes, features = z['indexes'], z['features']
        cursor = 0
        for i in indexes:
            a, b = off[i:i+2]; base[a:b] = features[cursor:cursor+b-a]; cursor += b-a
            assert not seen48[i]; seen48[i] = True
        assert cursor == len(features)
    for item in gate['recomputed']:
        assert file_hash(item['path']) == item['sha256'], item['path']
        with np.load(item['path']) as z:
            assert list(z['names']) == list(E.NEW_NAMES) and list(z['digit_names']) == list(E.DIGIT_NAMES)
            arrays = {key: z[key] for key in ('indexes', 'new', 'digits', 'digit_active')}
        cursor = 0
        for i in arrays['indexes']:
            a, b = off[i:i+2]; s = slice(cursor, cursor+b-a)
            new[a:b] = arrays['new'][s]; digits[a:b] = arrays['digits'][s]; active[a:b] = arrays['digit_active'][s]
            cursor += b-a; assert not seen_new[i]; seen_new[i] = True
        assert cursor == len(arrays['new'])
    assert seen48.all() and seen_new.all() and np.isfinite(base).all() and np.isfinite(digits).all()
    filled = np.zeros(5, int)
    for a, b in zip(off[:-1], off[1:]):
        new[a:b], f = fill_missing(new[a:b]); filled += f
    values = np.column_stack((base, new, digits))
    names = list(names48)+list(E.NEW_NAMES)+list(E.DIGIT_NAMES)
    dest = OUT/'source'/'FEATURES.npz'; dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, values=values, names=np.asarray(names), digit_active=active, offsets=off,
                        uids=np.asarray([r['uid'] for r in roster]), cells=np.asarray([r['cell'] for r in roster]),
                        row_ids=np.asarray([str(r['row_id']) for r in roster]),
                        nonempty=np.ones(int(off[-1]), bool), nonempty_lengths=np.diff(off))
    atomic_json(OUT/'source'/'MANIFEST.json', {
        'version': E.VERSION, 'order': 'localization_full_benchmark_v3 JOINED.json (= OOF_STEP_SCORES offsets)',
        'answers': len(roster), 'steps': int(off[-1]), 'channels': names, 'features_sha256': file_hash(dest),
        'provenance': {'channels_48': 'family_tail_external_v1/source_full_v1 recomputed raw features (PASS gate)',
                       'new_and_digits': 'external_banks_v4/source_parity recomputed arrays (PASS gate)'},
        'family_gate_sha256': file_hash(FAMILY/'source_full_v1/GATE.json'), 'v4_gate_sha256': file_hash(GATE),
        'new_channel_filled_values': dict(zip(E.NEW_NAMES, filled.tolist())),
        'digit_inactive_steps': dict(zip(E.DIGIT_NAMES, (~active).sum(axis=0).tolist())),
        'all_steps_nonempty': True, 'code_sha256': code_hashes(), 'labels_read': False})
    print('source bank', values.shape, 'filled', filled.tolist(), flush=True)


# ---------------------------------------------------------------- external
def one(job):
    cell, rawpath = job
    started = time.process_time()
    raw_sha = file_hash(rawpath)
    record = json.loads(Path(rawpath).read_text()); row = record['payload']['telemetry']
    storedpath = FAMILY/cell/'shard_000'/Path(rawpath).name
    stored = json.loads(storedpath.read_text())
    payload = stored['payload']
    if stored['uid'] != record['uid'] or payload['telemetry_sha256'] != raw_sha:
        raise ValueError('stored family record does not match raw telemetry: '+record['uid'])
    spans = np.asarray(row['step_token_spans'], int); valid = spans[:, 1] > spans[:, 0]
    if valid.tolist() != payload['nonempty']:
        raise ValueError('empty-step mask changed: '+record['uid'])
    base = np.asarray(payload['features'], float)
    if base.shape != (int(valid.sum()), 48) or not np.isfinite(base).all():
        raise ValueError('stored 48-channel features malformed: '+record['uid'])
    new, _ = E.new_step_features(row, spans[valid])
    new, filled = fill_missing(new)
    digits, active = E.digit_step_features(row, spans[valid])
    return (record['uid'], valid, base, new, filled, digits, active, raw_sha, file_hash(storedpath),
            payload['feature_names'], len(row['gen_token_ids']), time.process_time()-started)


def external(gate, workers):
    names48 = load_lock()['recipe']['channels_48']
    names = list(names48)+list(E.NEW_NAMES)+list(E.DIGIT_NAMES)
    tok = json.loads((OUT/'TOKENIZER_CHECK.json').read_text())
    assert tok['status'] == 'PASS'
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for cell, (_, expected) in CELLS.items():
            t0 = time.perf_counter()
            paths = sorted((RAW/cell/'records').glob('*.record.json'))
            if len(paths) != expected:
                raise ValueError('incomplete raw population: '+cell)
            results = []
            for n, item in enumerate(executor.map(one, [(cell, str(p)) for p in paths], chunksize=1)):
                if item[9] != list(names48):
                    raise ValueError('stored feature names differ from the lock')
                results.append(item)
                if n % 250 == 0:
                    print('external', cell, n+1, '/', len(paths), 'seconds', round(time.perf_counter()-started), flush=True)
            results.sort(key=lambda r: r[0])
            uids = [r[0] for r in results]
            assert len(set(uids)) == len(uids) == expected
            lengths = np.array([int(r[2].shape[0]) for r in results])
            off = np.r_[0, np.cumsum(lengths)]
            values = np.column_stack((np.concatenate([r[2] for r in results]), np.concatenate([r[3] for r in results]),
                                      np.concatenate([r[5] for r in results])))
            active = np.concatenate([r[6] for r in results])
            nonempty = np.concatenate([r[1] for r in results])
            nonempty_lengths = np.array([len(r[1]) for r in results])
            assert values.shape == (off[-1], len(names)) and np.isfinite(values).all()
            assert nonempty.sum() == off[-1]
            filled = np.sum([r[4] for r in results], axis=0)
            dest = OUT/cell/'FEATURES.npz'; dest.parent.mkdir(parents=True, exist_ok=True)
            telemetry = np.asarray([r[7] for r in results]); stored = np.asarray([r[8] for r in results])
            np.savez_compressed(dest, values=values, names=np.asarray(names), digit_active=active, offsets=off,
                                uids=np.asarray(uids), nonempty=nonempty, nonempty_lengths=nonempty_lengths,
                                telemetry_sha256=telemetry, feature_record_sha256=stored)
            identity = json.loads((RAW/cell/'TOKENIZATION.json').read_text())['identity']
            atomic_json(OUT/cell/'MANIFEST.json', {
                'version': E.VERSION, 'cell': cell, 'model': identity['model'], 'revision': identity['revision'],
                'records': len(uids), 'steps_total': int(nonempty_lengths.sum()), 'steps_nonempty': int(off[-1]),
                'steps_empty_excluded': int((~nonempty).sum()), 'tokens': int(sum(r[10] for r in results)),
                'channels': names, 'order': 'records sorted by uid; rows = non-empty steps in step order',
                'nonempty_layout': 'nonempty is the flattened per-record step mask (all steps); '
                                   'nonempty_lengths[i] = total steps of record i; offsets index non-empty rows',
                'features_sha256': file_hash(dest),
                'records_digest': sha_text('\n'.join(f'{u} {t} {s}' for u, t, s in zip(uids, telemetry, stored))),
                'record_hash_note': 'per-record telemetry_sha256 / feature_record_sha256 arrays are inside FEATURES.npz',
                'new_channel_filled_values': dict(zip(E.NEW_NAMES, filled.tolist())),
                'digit_inactive_steps': dict(zip(E.DIGIT_NAMES, (~active).sum(axis=0).tolist())),
                'v4_gate_sha256': file_hash(GATE), 'tokenizer_check_sha256': file_hash(OUT/'TOKENIZER_CHECK.json'),
                'code_sha256': code_hashes(), 'cpu_seconds': float(sum(r[11] for r in results)),
                'elapsed_seconds': time.perf_counter()-t0, 'labels_read': False,
                'external_quality_computed': False})
            print('CELL', cell, values.shape, 'filled', filled.tolist(), 'seconds', round(time.perf_counter()-t0), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', nargs='?', default='all', choices=['tokenizers', 'source', 'external', 'all'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.stage in ('tokenizers',):
        OUT.mkdir(parents=True, exist_ok=True); tokenizers(args.workers); return
    gate = check_gate()
    if args.stage in ('source', 'all'):
        source(gate)
    if args.stage == 'all':
        tokenizers(args.workers)
    if args.stage in ('external', 'all'):
        external(gate, args.workers)


if __name__ == '__main__':
    main()
