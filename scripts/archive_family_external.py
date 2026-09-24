"""Create a local immutable reproducibility archive; never uploads or deletes."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/family_tail_external_v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for name in ('RED_TEAM.md', 'ALL_CELLS_SEALED.json', 'METRICS.json', 'REPORT.html'):
        if not (OUT/name).is_file():
            raise ValueError('Incomplete final artifacts: '+name)
    if list(OUT.rglob('WRITER.lock')):
        raise ValueError('Active writer')
    excluded = {'__pycache__', '.pytest_cache'}
    entries = [(p, p.relative_to(ROOT).as_posix()) for p in OUT.rglob('*')
               if p.is_file() and not (set(p.parts)&excluded)
               and p.suffix not in ('.pyc', '.tmp')
               and p.name not in ('ARCHIVE.json', 'ARCHIVE_CONTENTS.json')]
    for name in ('pool_z.npy', 'pool_names.json'):
        entries.append((args.pool_dir/name, 'reproduction/source_pool/'+name))
    if sha(args.pool_dir/'pool_z.npy') != 'd9abccfb561f459d2f3a6c5b4346a1f6766df7fdbaddb230338aa40349077a16':
        raise ValueError('Source pool identity mismatch')
    entries.append((ROOT/'results/localization_full_benchmark_v3/evaluation/JOINED.json',
                    'reproduction/source_pool/JOINED.json'))
    entries.append((ROOT/'results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.MANIFEST.json',
                    'reproduction/source_pool/CT7_TOKEN_MATRICES.MANIFEST.json'))
    # No answer/question texts, raw telemetry PKLs or credentials are included.
    ledger = [{'member': member, 'bytes': p.stat().st_size, 'sha256': sha(p)}
              for p, member in sorted(entries, key=lambda entry: entry[1])]
    manifest = OUT/'ARCHIVE_CONTENTS.json'
    manifest.write_text(json.dumps({'files': ledger, 'n_files': len(ledger),
        'uncompressed_bytes': sum(x['bytes'] for x in ledger)}, indent=2)+'\n', encoding='utf8')
    with tarfile.open(args.output, 'w:gz', compresslevel=6) as archive:
        for p, member in sorted(entries, key=lambda entry: entry[1]):
            archive.add(p, arcname=member, recursive=False)
        archive.add(manifest, arcname=manifest.relative_to(ROOT).as_posix(), recursive=False)
    with tarfile.open(args.output, 'r:gz') as archive:
        byname = {x['member']: x for x in ledger}
        checked = 0
        for member in archive:
            if member.name not in byname:
                if member.name != manifest.relative_to(ROOT).as_posix():
                    raise ValueError('Unexpected archive member')
                continue
            h = hashlib.sha256()
            stream = archive.extractfile(member)
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                h.update(block)
            if h.hexdigest() != byname[member.name]['sha256']:
                raise ValueError('Archive member mismatch')
            checked += 1
    assert checked == len(ledger)
    info = {'local_path': str(args.output.resolve()), 'bytes': args.output.stat().st_size,
            'sha256': sha(args.output), 'n_verified_members': checked,
            'contents_manifest_sha256': sha(manifest), 'upload_status': 'NOT_UPLOADED',
            'payload': 'Features, predictions, bootstrap draws, source pool, compact evaluation evidence and plots; no benchmark question/answer text or secrets.'}
    (OUT/'ARCHIVE.json').write_text(json.dumps(info, indent=2)+'\n', encoding='utf8')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
