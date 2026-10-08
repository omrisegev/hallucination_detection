"""Round-2 review input freeze: hash every sealed/aggregate input before and after analysis.
Read-only. Live files (logs, in-progress layer chunks without a seal) are listed but not hashed."""
import hashlib, json, sys, datetime
from pathlib import Path
ROOT = Path(r'C:\Users\omris\TAU\hallucination_detection')
OUT = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return p.relative_to(ROOT).as_posix()


tag = sys.argv[1] if len(sys.argv) > 1 else 'before'
rec = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'tag': tag, 'files': {},
       'sealed_chunks': {}, 'layer_checkpoints': {}, 'seal_failures': [], 'skipped_live': []}
for exp in ['level_reduction_v1', 'estimator_input_v1']:
    for run in ['run_20261003', 'run_20261004_resume']:
        d = ROOT / 'results' / exp / run
        for p in sorted(d.iterdir()):
            if p.is_file():
                rec['files'][rel(p)] = sha(p)
        for stamp in sorted((d / 'chunks').glob('*.complete.json')):
            seal = json.loads(stamp.read_text(encoding='utf8')); stem = stamp.name.removesuffix('.complete.json'); ok = True
            for suffix, digest in seal['artifacts'].items():
                if sha(stamp.parent / (stem + suffix)) != digest:
                    ok = False; rec['seal_failures'].append(rel(stamp.parent / (stem + suffix)))
            rec['sealed_chunks'][f'{exp}/{run}/{stem}'] = {'seal_sha256': sha(stamp), 'verified': ok}
        if (d / 'layers').exists():
            sealed = set()
            for stamp in sorted((d / 'layers').glob('*.complete.json')):
                seal = json.loads(stamp.read_text(encoding='utf8')); stem = stamp.name.removesuffix('.complete.json'); ok = True
                sealed.add(stem)
                for suffix, digest in seal['artifacts'].items():
                    if sha(stamp.parent / (stem + suffix)) != digest:
                        ok = False; rec['seal_failures'].append(rel(stamp.parent / (stem + suffix)))
                rec['layer_checkpoints'][f'{exp}/{run}/{stem}'] = {'seal_sha256': sha(stamp), 'verified': ok}
            for p in (d / 'layers').iterdir():
                if p.name.split('.')[0] not in sealed:
                    rec['skipped_live'].append(rel(p))
for sub in ['results/monday_fusion_review_response_20261004', 'docs/meetings/advisor_prep_2026-10-05']:
    for p in sorted((ROOT / sub).iterdir()):
        if p.is_file():
            rec['files'][rel(p)] = sha(p)
for p in sorted((ROOT / 'results/monday_fusion_analysis_v1/code/spectral_utils').glob('monday_fusion*.py')):
    rec['files'][rel(p)] = sha(p)
for p in sorted((ROOT / 'results/monday_fusion_analysis_v1/code/scripts/experiments').glob('*.py')):
    rec['files'][rel(p)] = sha(p)
for name in ['WORKFLOW_STATUS.json', 'RESUME_SESSION_20261004.json', 'SELECTION_REVISION_CHECKS_20261004.json', 'INPUT_AUDIT.json',
             'REPLAY_GATE.json', 'POSITION_REPLAY_GATE.json', 'RAW_BANK_REPLAY.json', 'RAW_INPUT_MANIFEST_V3.json', 'GATE_STATUS.json',
             'RESUME_AUDIT_20261004.json', 'RESUME_COMMIT_MAPPING.json', 'WORKFLOW_STATUS_before_resume_20261004.json',
             'CHECKPOINT_RESTORE_AUDIT.json', 'TEST_RESULTS.json']:
    p = ROOT / 'results/monday_fusion_analysis_v1' / name
    if p.exists():
        rec['files']['results/monday_fusion_analysis_v1/' + name] = sha(p)
rec['skipped_live'] += ['results/monday_fusion_analysis_v1/RESUME_20261004.log',
                        'results/monday_fusion_analysis_v1/WORKFLOW_STATUS.json is a live aggregate; hashed as a snapshot only']
man = json.loads((ROOT / 'results/level_reduction_v1/run_20261003/INPUT_MANIFEST.json').read_text(encoding='utf8'))
for k, spec in man['inputs'].items():
    p = Path(spec['path'])
    if p.exists():
        h = sha(p); rec['files'][str(p)] = h
        if spec.get('sha256') and spec['sha256'] != h:
            rec['seal_failures'].append('INPUT_MANIFEST ' + k)
    else:
        rec['seal_failures'].append('INPUT_MANIFEST missing ' + k)
rec['counts'] = {'files': len(rec['files']), 'sealed_chunks': len(rec['sealed_chunks']), 'layer_checkpoints': len(rec['layer_checkpoints']),
                 'seal_failures': len(rec['seal_failures']), 'skipped_live': len(rec['skipped_live'])}
(OUT / f'INPUT_FREEZE_{tag}.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding='utf8')
print(json.dumps(rec['counts']))
print('layer checkpoints by run:', {k: sum(1 for s in rec['layer_checkpoints'] if s.startswith(k)) for k in
                                    ['level_reduction_v1/run_20261003', 'level_reduction_v1/run_20261004_resume', 'estimator_input_v1/run_20261004_resume']})
print('seal failures:', rec['seal_failures'][:10])
