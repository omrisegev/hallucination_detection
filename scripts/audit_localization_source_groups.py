"""Inspect source-question identity without loading the 1.1-GB NumPy payloads.

Trusted project pickle only. This metadata reader discards NumPy/binary values,
never scores an answer, and never uses correctness fields. Such fields are
present in the container and may be deserialized; this is not a label-free file.
"""
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import pickle
import re
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
LIVE = Path('C:/Users/omris/TAU/hd_jlsml_v2_wt')
OUT = ROOT / 'results/localization_source_group_audit_v1'
RAW = ROOT / 'dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()
def text_sha(text): return hashlib.sha256(text.encode()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


class Discarded:
    def __setstate__(self, state): pass


def discard(*args, **kwargs): return Discarded()


class MetadataUnpickler(pickle._Unpickler):
    """Data-only metadata extraction; do not use for any scientific array."""
    dispatch = pickle._Unpickler.dispatch.copy()
    def find_class(self, module, name):
        if module.startswith('numpy'): return discard
        raise pickle.UnpicklingError('Unexpected class in project metadata: ' + module + '.' + name)
    def binary(self, fmt, nbytes):
        size = struct.unpack(fmt, self.read(nbytes))[0]
        if size < 0: raise pickle.UnpicklingError('Negative payload length')
        while size:
            block = self.read(min(size, 1 << 16))
            if not block: raise EOFError('Truncated binary payload')
            size -= len(block)
        self.append(b'')
    def short_binary(self): self.binary('<B', 1)
    def binary32(self): self.binary('<I', 4)
    def binary64(self): self.binary('<Q', 8)
    dispatch[pickle.SHORT_BINBYTES[0]] = short_binary
    dispatch[pickle.BINBYTES[0]] = binary32
    dispatch[pickle.BINBYTES8[0]] = binary64
    dispatch[pickle.BYTEARRAY8[0]] = binary64


def source_seed(group):
    match = re.search(r'(prm_(?:train|test)_p\d+_\d+)$', group)
    if not match: raise ValueError('UNRECOGNIZED_SOURCE_ID: ' + group)
    return match.group(1)


def extract():
    OUT.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    print('Reading question metadata; NumPy payloads are discarded.', flush=True)
    with RAW.open('rb') as stream: data = MetadataUnpickler(stream).load()
    records = []
    for row in data.values():
        row_id, group, question = row['idx'], row['source_idx'], row['question']
        assert all(isinstance(v, str) for v in (row_id, group, question))
        records.append({'row_id': row_id, 'old_group_id': group, 'source_seed': source_seed(group),
                        'question_sha256': text_sha(question),
                        'question_whitespace_sha256': text_sha(' '.join(question.split()))})
    assert len(records) == len({r['row_id'] for r in records})
    del data; gc.collect()
    save(OUT / 'QUESTION_METADATA.json', {'raw_path': str(RAW), 'raw_sha256': sha(RAW), 'rows': records,
        'correctness_fields_used': False, 'container_contains_labels': True, 'numpy_payloads_discarded': True,
        'extractor_sha256': sha(__file__), 'seconds': time.monotonic() - started})
    print('Saved question identities for', len(records), 'rows.', flush=True)
    pb_rows = []; pb_files = {}
    for subset in ('gsm8k', 'math', 'olympiadbench', 'omnimath'):
        path = ROOT / 'dataset_cache/repgrid/pb_qwen3_8b' / ('processbench_' + subset + '.pkl')
        with path.open('rb') as stream: data = MetadataUnpickler(stream).load()
        for row in data.values():
            assert isinstance(row['problem'], str)
            pb_rows.append({'subset': subset, 'row_id': subset + '::' + str(row['id']),
                            'question_sha256': text_sha(row['problem']),
                            'question_whitespace_sha256': text_sha(' '.join(row['problem'].split()))})
        del data; gc.collect(); pb_files[str(path)] = sha(path)
        print('Read PB question metadata:', subset, flush=True)
    pb_index = {r['row_id']: r for r in pb_rows}; model_replays = 0
    for subset in ('gsm8k', 'math', 'olympiadbench', 'omnimath'):
        path = ROOT / 'dataset_cache/repgrid/pb_qwen3_4b' / ('processbench_' + subset + '.pkl')
        with path.open('rb') as stream: data = MetadataUnpickler(stream).load()
        seen = set()
        for row in data.values():
            row_id = subset + '::' + str(row['id']); seen.add(row_id)
            assert text_sha(row['problem']) == pb_index[row_id]['question_sha256']
            model_replays += 1
        assert seen == {r['row_id'] for r in pb_rows if r['subset'] == subset}
        del data; gc.collect(); pb_files[str(path)] = sha(path)
        print('Verified 4b/8b question identity:', subset, flush=True)
    save(OUT / 'PB_QUESTION_METADATA.json', {'rows': pb_rows, 'source_files': pb_files,
        'q4_q8_exact_question_replays': model_replays,
        'correctness_fields_used': False, 'container_contains_labels': True, 'numpy_payloads_discarded': True,
        'extractor_sha256': sha(__file__)})


def audit():
    metadata = load(OUT / 'QUESTION_METADATA.json'); assert metadata['extractor_sha256'] == sha(__file__)
    release_path = ROOT / 'results/answer_localization_representation_pilot_v1/RELEASE.json'
    release = load(release_path); records = metadata['rows']; by_id = {r['row_id']: r for r in records}
    previous = release['cells']['prmbench_qwen3_8b']['rows']
    assert len(previous) == len(records) and {r['row_id'] for r in previous} == set(by_id)
    for r in previous: assert r['group_id'] == by_id[r['row_id']]['old_group_id']
    by_seed = defaultdict(list); by_question = defaultdict(list)
    for r in records:
        by_seed[r['source_seed']].append(r)
        by_question[r['question_whitespace_sha256']].append(r)
    parent = {s: s for s in by_seed}
    def root(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]; s = parent[s]
        return s
    def union(a, b):
        a, b = root(a), root(b)
        if a != b: parent[max(a,b)] = min(a,b)
    question_collisions = []
    for question, rows in by_question.items():
        seeds = sorted({r['source_seed'] for r in rows})
        if len(seeds) > 1:
            question_collisions.append({'question_hash': question, 'seeds': seeds})
            for s in seeds[1:]: union(seeds[0], s)
    members = defaultdict(list)
    for s in parent: members[root(s)].append(s)
    canonical = {s: 'prmb_source_v2:' + text_sha('|'.join(sorted(members[root(s)])))[:24] for s in parent}
    for r in records: r['canonical_group_id'] = canonical[r['source_seed']]
    pb_metadata = load(OUT / 'PB_QUESTION_METADATA.json'); pb_rows = pb_metadata['rows']
    assert pb_metadata['extractor_sha256'] == sha(__file__)
    prm_question_group = {r['question_whitespace_sha256']: r['canonical_group_id'] for r in records}
    pb_question_rows = defaultdict(list)
    for row in pb_rows:
        question = row['question_whitespace_sha256']
        row['canonical_group_id'] = prm_question_group.get(question, 'pb_source_v2:' + question[:24])
        pb_question_rows[question].append(row)
    pb_by_id = {r['row_id']: r for r in pb_rows}; assert len(pb_by_id) == len(pb_rows)
    for cell, info in release['cells'].items():
        if cell.startswith('pb_'):
            for row in info['rows']: assert row['group_id'] == row['row_id'] and row['row_id'] in pb_by_id
    same_question_across_old = []
    for q, rows in by_question.items():
        old = sorted({r['old_group_id'] for r in rows})
        if len(old) > 1: same_question_across_old.append({'question_hash': q, 'old_groups': old, 'rows': len(rows)})
    folds_path = LIVE / 'results/joint_lsml_optimization_v2/folds/folds.json'
    full_folds = load(folds_path); folds = full_folds['prmbench']; fold_overlap = []
    for component, seeds in members.items():
        rows = [r for s in seeds for r in by_seed[s]]
        assignments = sorted({folds['outer'][r['old_group_id']] for r in rows})
        fold_overlap.append({'canonical_group_id': canonical[component], 'source_seeds': sorted(seeds),
                             'outer_folds': assignments, 'rows': len(rows)})
    manifests = [ROOT / 'results/localization_short_cycle01/COHORT.json'] + [ROOT / 'results' / d / 'CONFIG.json'
        for d in ('localization_short_cycle01','localization_short_cycle02','localization_short_cycle03_graph')]
    exposures = {}; excluded = set()
    for p in manifests:
        d = load(p); ids = [r['row_id'] for r in d] if isinstance(d, list) else d['selected_row_ids']
        groups = {by_id[row_id]['canonical_group_id'] for row_id in ids}; excluded.update(groups)
        exposures[str(p)] = {'row_ids': ids, 'canonical_groups': sorted(groups), 'sha256': sha(p)}
    pilot_path = ROOT / 'results/answer_localization_representation_pilot_v1/PREPARED.json'
    selected = load(pilot_path)['selected']; pilot_prm = [r for r in selected if r['cell'].startswith('prm')]
    pilot_groups = [by_id[r['row_id']]['canonical_group_id'] for r in pilot_prm]
    overlaps = sorted(excluded & set(pilot_groups)); excluded.update(pilot_groups)
    exposures[str(pilot_path)] = {'row_ids': [r['row_id'] for r in pilot_prm], 'canonical_groups': sorted(set(pilot_groups)), 'sha256': sha(pilot_path)}
    pb_selected = [r for r in selected if r['cell'].startswith('pb_')]
    pb_pilot_groups = [pb_by_id[r['row_id']]['canonical_group_id'] for r in pb_selected]
    excluded.update(pb_pilot_groups)
    exposures[str(pilot_path)]['pb_row_ids'] = [r['row_id'] for r in pb_selected]
    exposures[str(pilot_path)]['pb_canonical_groups'] = sorted(set(pb_pilot_groups))
    pb_duplicates = [{'question_hash': q, 'rows': [r['row_id'] for r in rr],
                      'subsets': sorted({r['subset'] for r in rr}),
                      'outer_folds': sorted({full_folds['processbench']['outer'][r['row_id']] for r in rr})}
                     for q, rr in pb_question_rows.items() if len(rr) > 1]
    save(OUT / 'CANONICAL_GROUPS.json', {'grouping_version': 'question_source_components_v2',
        'rule': 'Connect the official source-seed suffix and exact whitespace-normalized observed question text. Modified questions retain their source seed.',
        'rows': records, 'pb_rows': pb_rows, 'metadata_sha256': sha(OUT / 'QUESTION_METADATA.json'),
        'pb_metadata_sha256': sha(OUT / 'PB_QUESTION_METADATA.json'),
        'cross_seed_question_collisions': question_collisions, 'correctness_fields_used': False})
    result = {'status': 'SOURCE_GROUP_LEAKAGE_VERIFIED', 'rows': len(records),
        'old_groups': len({r['old_group_id'] for r in records}), 'source_seeds': len(by_seed), 'canonical_components': len(members),
        'identical_question_hashes_spanning_old_groups': len(same_question_across_old),
        'rows_in_identical_question_cross_group_sets': sum(x['rows'] for x in same_question_across_old),
        'same_question_examples': same_question_across_old[:8],
        'canonical_groups_spanning_outer_folds': sum(len(x['outer_folds']) > 1 for x in fold_overlap),
        'pb_rows': len(pb_rows), 'pb_exact_question_groups': len(pb_question_rows),
        'pb_repeated_question_groups': len(pb_duplicates), 'pb_repeated_question_rows': sum(len(x['rows']) for x in pb_duplicates),
        'pb_repeated_groups_spanning_outer_folds': sum(len(x['outer_folds']) > 1 for x in pb_duplicates),
        'pb_cross_subset_question_groups': sum(len(x['subsets']) > 1 for x in pb_duplicates),
        'pb_repeated_question_details': pb_duplicates,
        'pb_prmb_identical_question_hashes': len(set(pb_question_rows) & set(prm_question_group)),
        'pilot_pb_answers': len(pb_selected), 'pilot_pb_canonical_groups': len(set(pb_pilot_groups)),
        'outer_fold_overlap': fold_overlap, 'pilot_prm_answers': len(pilot_prm), 'pilot_prm_canonical_groups': len(set(pilot_groups)),
        'pilot_overlap_with_earlier_shortcycles': overlaps, 'excluded_canonical_groups': sorted(excluded), 'exposure_sources': exposures,
        'parent_release_sha256': sha(release_path), 'claude_folds_sha256': sha(folds_path),
        'canonical_map_sha256': sha(OUT / 'CANONICAL_GROUPS.json'), 'audit_script_sha256': sha(__file__),
        'implication': 'Old within-answer fits are unchanged; old PRMB source-group independence and grouped cross-validation require correction. '
                       'Do not use the old group IDs to claim a disjoint source-question replication.'}
    save(OUT / 'AUDIT.json', result)
    print(json.dumps({k:result[k] for k in ('status','rows','old_groups','source_seeds','canonical_components',
        'identical_question_hashes_spanning_old_groups','canonical_groups_spanning_outer_folds',
        'pilot_prm_answers','pilot_prm_canonical_groups','pilot_overlap_with_earlier_shortcycles',
        'pb_repeated_question_groups','pb_repeated_groups_spanning_outer_folds','pb_cross_subset_question_groups',
        'pb_prmb_identical_question_hashes','pilot_pb_canonical_groups')}, indent=2),flush=True)


if __name__ == '__main__':
    extract(); audit()
