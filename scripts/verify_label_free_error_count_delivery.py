"""Read-only verification of the error-count delivery and its runtime sources."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, default=ROOT / 'results/label_free_error_count_v1_2026-10-12')
    args = parser.parse_args()
    package = args.package.resolve()
    manifest = json.loads((package / 'PACKAGE_MANIFEST.json').read_text())
    failures = []
    checked = 0
    for base, entries in [(package, manifest['files']), (ROOT, manifest['runtime_files'])]:
        for entry in entries:
            relative = entry.get('path', entry.get('repo_relative_path'))
            path = (base / relative).resolve()
            if not path.is_relative_to(base) or not path.is_file():
                failures.append(relative + ': missing or outside root')
            elif path.stat().st_size != entry['size'] or sha(path) != entry['sha256']:
                failures.append(relative + ': changed')
            checked += 1
    expected = manifest['frozen_advisor_manifest_sha256']
    if sha(ROOT / 'results/advisor_completion_2026-10-11/PACKAGE_MANIFEST.json') != expected:
        failures.append('Frozen advisor manifest changed')
    if failures:
        raise SystemExit(json.dumps({'status': 'FAILED', 'failures': failures}, indent=2))
    print(json.dumps({'status': 'PASS', 'checked_delivery_and_runtime_files': checked,
                      'scope': 'Read-only delivery hashes; use driver --stage verify for full historical input inventory'}))


if __name__ == '__main__':
    main()
