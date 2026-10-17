"""Verify a frozen reading delivery inside a later, integrated checkout.

Package files and numerical runtime files must still match the sealed delivery.
Only the four evolving project documents are checked against the publication
commit, with their current drift explicitly reported. No manifest is rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = '6a0b870023acd4c2af9d05040451b33c2e63894d'
REFERENCE_DOCUMENTS = frozenset({
    'Experiments_Report.md', 'HISTORY.md', 'PROGRESS.md', 'Research_Directions.md',
})


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_path(base, relative):
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError('Missing or unsafe delivery path: ' + relative)
    return path


def verify(package, *, root=ROOT):
    package, root = package.resolve(), root.resolve()
    manifest = json.loads((package / 'GITHUB_DELIVERY_MANIFEST.json').read_text())
    checked = 0
    references = []
    for record in manifest['files']:
        path = checked_path(package, record['path'])
        if sha(path) != record['sha256']:
            raise ValueError('Reading artifact changed: ' + record['path'])
        checked += 1
    for record in manifest['runtime_files']:
        relative = record['path']
        path = checked_path(root, relative)
        if relative in REFERENCE_DOCUMENTS:
            original = subprocess.check_output(
                ['git', 'show', PUBLICATION + ':' + relative], cwd=root)
            if hashlib.sha256(original).hexdigest() != record['sha256']:
                raise ValueError('Publication reference changed: ' + relative)
            references.append({'path': relative, 'publication': PUBLICATION,
                               'current_differs': sha(path) != record['sha256']})
        elif sha(path) != record['sha256']:
            raise ValueError('Numerical/runtime source changed: ' + relative)
        checked += 1
    return {'status': 'PASS', 'checked': checked,
            'scope': 'Current delivery artifacts and numerical sources; project documents at publication revision',
            'reference_documents': references,
            'raw_arrays_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path,
                        default=ROOT/'results/label_free_error_count_v1_2026-10-12')
    args = parser.parse_args()
    print(json.dumps(verify(args.package), indent=2))


if __name__ == '__main__':
    main()
