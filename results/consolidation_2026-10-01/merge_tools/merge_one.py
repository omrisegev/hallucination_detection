"""Object-level merge of one ref into a base commit, without touching any worktree or index.

usage: python merge_one.py BASE OTHER MSGFILE OUTDIR [--union PATH ...] [--file PATH=LOCALFILE ...]

- runs `git merge-tree --write-tree BASE OTHER`
- conflicted paths listed with --union are resolved with `git merge-file --union` (both sides kept)
- conflicted paths given with --file take the content of LOCALFILE (manual resolution)
- any other conflicted path: stage blobs are dumped to OUTDIR/<path>.{base,ours,theirs} and the
  script stops with exit 2 (nothing committed)
- on success prints the new merge commit id (not attached to any ref)
- heading-block check: for every resolved markdown file, every heading line of ours and of theirs
  must appear in the result at least as many times as on that side
"""
import collections, os, subprocess, sys, tempfile

REPO = r'C:\Users\omris\TAU\hallucination_detection'


def git(*args, inp=None, env=None, check=True):
    r = subprocess.run(['git', '-C', REPO, *args], input=inp, capture_output=True, env=env)
    if check and r.returncode not in (0,):
        raise SystemExit(f'git {args[:3]} failed rc={r.returncode}: {r.stderr.decode(errors="replace")}')
    return r


def blob(oid):
    return git('cat-file', 'blob', oid).stdout if oid else b''


def headings(b):
    return collections.Counter(l for l in b.decode('utf-8', errors='replace').splitlines() if l.startswith('#'))


def main():
    a = sys.argv[1:]
    base, other, msgfile, outdir = a[:4]; msgfile = os.path.abspath(msgfile)
    union, files = set(), {}
    i = 4
    while i < len(a):
        if a[i] == '--union':
            union.add(a[i + 1]); i += 2
        elif a[i] == '--file':
            p, f = a[i + 1].split('=', 1); files[p] = f; i += 2
        else:
            raise SystemExit(f'bad arg {a[i]}')
    base_c = git('rev-parse', base).stdout.decode().strip()
    other_c = git('rev-parse', other).stdout.decode().strip()
    r = git('merge-tree', '--write-tree', base_c, other_c, check=False)
    if r.returncode not in (0, 1):
        raise SystemExit(r.stderr.decode())
    lines = r.stdout.decode('utf-8', errors='replace').split('\n')
    tree = lines[0].strip()
    stages = collections.defaultdict(dict)
    for l in lines[1:]:
        if not l.strip():
            break
        meta, path = l.split('\t', 1)
        mode, oid, st = meta.split()
        stages[path][int(st)] = (mode, oid)
    print(f'merge-tree rc={r.returncode} tree={tree} conflicted={len(stages)}')
    unresolved = [p for p in stages if p not in union and p not in files]
    if unresolved:
        for p in unresolved:
            d = os.path.join(outdir, p.replace('/', '__'))
            for st, name in ((1, 'base'), (2, 'ours'), (3, 'theirs')):
                if st in stages[p]:
                    open(f'{d}.{name}', 'wb').write(blob(stages[p][st][1]))
            print('UNRESOLVED', p, sorted(stages[p]))
        raise SystemExit(2)
    idx = os.path.abspath(os.path.join(outdir, 'tmp_index'))
    if os.path.exists(idx):
        os.remove(idx)
    env = dict(os.environ, GIT_INDEX_FILE=idx)
    git('read-tree', tree, env=env)
    for p, st in stages.items():
        ours, theirs, b = blob(st.get(2, (0, None))[1]), blob(st.get(3, (0, None))[1]), blob(st.get(1, (0, None))[1])
        mode = (st.get(2) or st.get(3))[0]
        if p in files:
            res = open(files[p], 'rb').read()
        else:
            with tempfile.TemporaryDirectory(dir=outdir) as td:
                fo, fb, ft = (os.path.join(td, n) for n in ('o', 'b', 't'))
                CRLF, LF = bytes([13, 10]), bytes([10])
                crlf = ours.count(CRLF) > ours.count(LF) // 2
                lf = lambda x: x.replace(CRLF, LF)
                open(fo, 'wb').write(lf(ours)); open(fb, 'wb').write(lf(b)); open(ft, 'wb').write(lf(theirs))
                m = subprocess.run(['git', 'merge-file', '--union', '-p', fo, fb, ft], capture_output=True)
                res = m.stdout.replace(LF, CRLF) if crlf else m.stdout
                print(f'  {p}: line endings ours={"CRLF" if crlf else "LF"}; merged on LF-normalized text')
        if b'<<<<<<<' in res or b'>>>>>>>' in res:
            raise SystemExit(f'conflict markers remain in {p}')
        if p.endswith('.md'):
            ho, ht, hr = headings(ours), headings(theirs), headings(res)
            hb = headings(b)
            miss = [(h, n) for side in (ho, ht) for h, n in side.items() if n > hb[h] and hr[h] < n]
            exp = sum(max(ho[h], ht[h]) for h in set(ho) | set(ht))
            extra = [(h, hr[h]) for h in hr if hr[h] > max(ho[h], ht[h])]
            print(f'  {p}: headings ours={sum(ho.values())} theirs={sum(ht.values())} result={sum(hr.values())} expected_union={exp} missing={len(miss)} duplicated={len(extra)}')
            for h, n in extra[:10]:
                print('    DUPLICATED', n, h[:120])
            if len(extra) > 3 and p not in files:
                raise SystemExit(4)
            if miss:
                for h, n in miss[:10]:
                    print('    MISSING', n, h[:120])
                if p not in files:
                    raise SystemExit(3)
                print('    (manual resolution: side chosen deliberately, reported only)')
        oid = git('hash-object', '-w', '--stdin', inp=res).stdout.decode().strip()
        git('update-index', '--cacheinfo', f'{mode},{oid},{p}', env=env)
        print(f'  resolved {p} ({"file" if p in files else "union"}) -> {oid[:10]}')
    for p in files:
        if p in stages:
            continue
        ls = git('ls-tree', tree, '--', p).stdout.decode().split()
        mode = ls[0] if ls else '100644'
        oid = git('hash-object', '-w', '--stdin', inp=open(files[p], 'rb').read()).stdout.decode().strip()
        git('update-index', '--add', '--cacheinfo', f'{mode},{oid},{p}', env=env)
        print(f'  forced {p} (no conflict; manual content applied) -> {oid[:10]}')
    new_tree = git('write-tree', env=env).stdout.decode().strip()
    os.remove(idx)
    c = git('commit-tree', new_tree, '-p', base_c, '-p', other_c, '-F', msgfile).stdout.decode().strip()
    print('COMMIT', c)


main()
