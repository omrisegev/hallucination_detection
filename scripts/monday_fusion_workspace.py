"""Small, auditable code export and isolated commits; never changes the checkout index."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = '54a2c2a8f463c1c48e8f943405cb1bb7f657c85b'
BRANCH = 'codex/monday-fusion-analysis-20261003'
WORK = ROOT / 'results/monday_fusion_analysis_v1'
CODE = WORK / 'code'

def git(*args, data=None, env=None):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], input=data, env=env)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    tmp.replace(path)

def prepare():
    if CODE.exists():
        raise RuntimeError('Export already exists; inspect it before resuming')
    if shutil.disk_usage(ROOT).free < 150_000_000:
        raise RuntimeError('Less than 150 MB free: stop before export')
    paths = ['scripts/experiments', 'spectral_utils']
    payload = git('archive', '--format=tar', BASE, *paths)
    if len(payload) > 30_000_000:
        raise RuntimeError('Unexpected export size')
    CODE.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive.getmembers():
            dst = (CODE / member.name).resolve()
            if not dst.is_relative_to(CODE.resolve()) or not (member.isfile() or member.isdir()):
                raise RuntimeError('Unsafe export member')
        archive.extractall(CODE, filter='data')
    git('branch', BRANCH, BASE)
    write_json(WORK / 'WORKSPACE.json', {'base': BASE, 'branch': BRANCH,
        'export': str(CODE), 'export_bytes': len(payload),
        'checkout_unchanged': git('rev-parse', 'HEAD').decode().strip(),
        'policy': 'Only explicit paths enter isolated index; existing worktrees are read-only.'})
    print(json.dumps({'export_bytes': len(payload), 'branch': BRANCH}))

def commit(paths, message):
    parent = git('rev-parse', BRANCH).decode().strip()
    index = WORK / 'commit.index'
    env = dict(os.environ, GIT_INDEX_FILE=str(index))
    git('read-tree', parent, env=env)
    for dest, src in paths.items():
        src = Path(src).resolve()
        if not src.is_relative_to(ROOT.resolve()) or Path(dest).is_absolute() or '..' in Path(dest).parts:
            raise ValueError('Path outside workspace')
        blob = git('hash-object', '-w', '--no-filters', str(src)).decode().strip()
        git('update-index', '--add', '--cacheinfo', '100644', blob, dest, env=env)
    tree = git('write-tree', env=env).decode().strip()
    head = git('commit-tree', tree, '-p', parent, data=(message + '\n').encode()).decode().strip()
    git('update-ref', 'refs/heads/' + BRANCH, head, parent)
    print(head)
    return head

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'commit'])
    p.add_argument('--mapping')
    p.add_argument('--message')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare()
    else:
        commit(json.loads(Path(a.mapping).read_text(encoding='utf8')), a.message)
