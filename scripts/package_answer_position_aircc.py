"""Package exact frozen inputs and the isolated portability checkout.

No cache loads, no source mutations and no remote operations. The archive uses
code/ and source/ namespaces, preserving the driver's relative input layout.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--original-worktree', type=Path, required=True)
    parser.add_argument('--frozen-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--overlay-from', type=Path,
                        help='Existing bundle manifest; package changed code only, never data')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('refuse to overwrite an existing bundle')
    files = {}
    frozen = json.loads(args.frozen_manifest.read_text(encoding='utf8'))
    verified = []
    for filename, expected in frozen['hashes'].items():
        path = Path(filename)
        actual = digest(path)
        if actual != expected:
            raise ValueError('frozen source changed: '+filename)
        if path.is_relative_to(args.original_worktree):
            archive = 'code/'+path.relative_to(args.original_worktree).as_posix()
            copied = ROOT/path.relative_to(args.original_worktree)
        else:
            archive = 'source/'+path.relative_to(args.source_root).as_posix()
            copied = path
        files[archive] = copied
        verified.append(dict(source=filename, path=archive, frozen_sha256=actual))
    # Include all Python import dependencies from the isolated checkout, no
    # worktrees, results, binaries, notebooks or source-data directory copies.
    for folder in ('scripts', 'spectral_utils'):
        for path in (ROOT/folder).rglob('*.py'):
            files['code/'+path.relative_to(ROOT).as_posix()] = path
    for relative in ('cluster/answer_position_aircc.sbatch',
                     'docs/experiments/CONDITIONAL_IU_FOLLOWUPS_V1.md'):
        files['code/'+relative] = ROOT/relative
    # Fail rather than silently move a checkpoint with a live WAL journal.
    for path in files.values():
        if path.suffix == '.sqlite':
            wal = Path(str(path)+'-wal')
            if wal.exists() and wal.stat().st_size:
                raise RuntimeError('source database has a nonempty WAL: '+str(path))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(path=name, bytes=path.stat().st_size, sha256=digest(path))
            for name, path in sorted(files.items())]
    permitted = {'code/scripts/run_answer_position_fusion.py',
                 'code/scripts/run_rbm_hierarchical_time.py'}
    hashes = {r['path']: r['sha256'] for r in rows}
    changed = [r['path'] for r in verified if hashes[r['path']] != r['frozen_sha256']]
    if set(changed) != permitted:
        raise ValueError('unexpected differences from frozen source: '+str(changed))
    manifest = dict(schema='answer-position-aircc-bundle-v1', base='481381408',
                    files=rows, original_sources=verified, operational_changes=changed,
                    note='RAM probe and manifest hash list only; new output identity')
    included = set(files)
    if args.overlay_from:
        before = json.loads(args.overlay_from.read_text(encoding='utf8'))
        previous = {row['path']:row['sha256'] for row in before['files']}
        included = {row['path'] for row in rows if previous.get(row['path']) != row['sha256']}
        if not all(name.startswith('code/') for name in included):
            raise ValueError('an operational overlay cannot replace input data')
        if set(previous)-set(files):
            raise ValueError('an overlay cannot silently remove files')
        manifest['requires_previous_manifest_sha256'] = digest(args.overlay_from)
        manifest['overlay_members'] = sorted(included)
        manifest['note'] = 'Portable RAM probe, hashes and scheduler checkpoint handling; no model changes'
    payload = json.dumps(manifest, indent=2).encode()
    with tarfile.open(args.out, 'w:gz', compresslevel=1) as tar:
        for name, path in sorted(files.items()):
            if name in included:tar.add(path, arcname=name, recursive=False)
        item = tarfile.TarInfo('BUNDLE_MANIFEST.json');item.size=len(payload)
        tar.addfile(item, io.BytesIO(payload))
    args.out.with_suffix('.manifest.json').write_bytes(payload)
    review = dict(status='PACKAGED', files=len(rows), bytes=args.out.stat().st_size,
                  archive_sha256=digest(args.out), frozen_sources_verified=len(verified),
                  operational_changes=changed)
    args.out.with_suffix('.review.json').write_text(json.dumps(review, indent=2), encoding='utf8')
    print(json.dumps(review), flush=True)


if __name__ == '__main__':
    main()
