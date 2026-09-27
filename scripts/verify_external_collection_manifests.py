"""Verify full-population accounting from compact, archived telemetry manifests.

Raw-record validation runs on AIRCC first. This checks its counts, record hashes,
input identity and local manifest integrity; it does not independently read raw
arrays or replace the separate rclone archive checksum verification.
"""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    inputs = json.loads((root / 'INPUT_MANIFEST.json').read_text(encoding='utf-8-sig'))
    protocol = json.loads((root / 'COLLECTION_PROTOCOL.json').read_text(encoding='utf-8-sig'))
    protocol_hash = hashlib.sha256((root / 'COLLECTION_PROTOCOL.json').read_bytes()).hexdigest()
    results = []
    for cell, dataset, model in [('hard2verify_qwen3_8b', 'hard2verify', 'Qwen/Qwen3-8B'),
                                 ('socratic_qwen3_8b', 'socratic', 'Qwen/Qwen3-8B'),
                                 ('socratic_qwq32b', 'socratic', 'Qwen/QwQ-32B')]:
        folder = root / 'full' / cell
        read = lambda name: json.loads((folder / name).read_text(encoding='utf8'))
        manifest, audit = read('MANIFEST.json'), read('AUDIT.json')
        timing, tokenization, gate = read('TIMING.json'), read('TOKENIZATION.json'), read('ALIGNMENT_GATE.json')
        for name, expected in manifest['files'].items():
            if name.endswith('.json'):
                data = (folder / name).read_bytes()
                if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                    raise ValueError(f'{cell}: compact file hash/size mismatch: {name}')
        identity = timing['identity']
        if identity != tokenization['identity'] or identity != manifest['identity']:
            raise ValueError(f'{cell}: mismatched identities')
        expected_identity = dict(model=model, revision=protocol['models'][model], mode='full',
                                 dtype='bfloat16', answers_sha256=inputs[dataset]['answers_sha256'],
                                 protocol_sha256=protocol_hash, alignment_gate_version=2)
        if any(identity.get(k) != v for k, v in expected_identity.items()):
            raise ValueError(f'{cell}: changed input/model/protocol')
        uids = tokenization['uids']
        measured = [r['uid'] for r in timing['measurements']]
        selected = identity['selected_uids']
        count = inputs[dataset]['answers']
        if (any(len(x) != count or len(set(x)) != count for x in [uids, measured, selected])
                or set(uids) != set(measured) or set(uids) != set(selected)
                or tokenization['selected_indices'] != list(range(count))):
            raise ValueError(f'{cell}: incomplete, duplicate or extra answers')
        if (not timing['complete'] or tokenization['truncated'] != 0 or gate['status'] != 'PASS'
                or audit['status'] != 'PASS' or audit['answers'] != count or manifest['answers'] != count
                or len(audit['record_sha256']) != count):
            raise ValueError(f'{cell}: incomplete validation')
        if any(x['quality_evaluated'] for x in [manifest, audit, timing]):
            raise ValueError(f'{cell}: quality was evaluated before method freeze')
        if (audit['totals'] != manifest['totals'] or audit['totals']['steps'] != inputs[dataset]['steps']
                or audit['totals']['tokens'] != tokenization['total_answer_tokens']
                or sum(r['answer_tokens'] for r in timing['measurements']) != audit['totals']['tokens']
                or audit['totals']['empty_steps'] != inputs[dataset]['empty_steps_retained']):
            raise ValueError(f'{cell}: unmatched step/token accounting')
        results.append(dict(cell=cell, status='PASS', answers=count, totals=audit['totals'],
                            archive=manifest['files']['telemetry.tar.gz'], drive=manifest['drive'],
                            peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in timing['measurements']),
                            quality_evaluated=False))
    return dict(status='PASS', scope='compact full-population accounting and hashes; raw audit executed on AIRCC',
                cells=results, quality_evaluated=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'status': result['status'], 'cells': len(result['cells'])}))
