"""Block-level union of two markdown logs.

usage: python blockunion.py OURS_FILE THEIRS_FILE OUT_FILE "<appendix heading>" "<appendix note>"

OURS is kept verbatim. Blocks of THEIRS (a block starts at a line beginning with '#') whose
normalized text (CR stripped, lines right-stripped, trailing blank lines removed) does not occur
in OURS are appended under the appendix heading, in THEIRS order, each distinct block once.
Checks: OURS is a prefix of OUT; every normalized THEIRS block occurs in OUT.
Output keeps OURS line endings.
"""
import sys


def blocks(text):
    out, cur, fence = [], [], False
    for line in text.split('\n'):
        if line.lstrip().startswith('```'):
            fence = not fence
        if line.startswith('#') and cur and not fence:
            out.append(cur); cur = []
        cur.append(line)
    if cur:
        out.append(cur)
    return out


def norm(b):
    lines = [l.rstrip('\r').rstrip() for l in b]
    while lines and not lines[-1]:
        lines.pop()
    return '\n'.join(lines)


ours_b = open(sys.argv[1], 'rb').read()
theirs = open(sys.argv[2], 'rb').read().decode('utf-8').replace('\r\n', '\n')
crlf = ours_b.count(b'\r\n') > ours_b.count(b'\n') // 2
ours = ours_b.decode('utf-8').replace('\r\n', '\n')
have = {norm(b) for b in blocks(ours)}
add, seen = [], set()
tb = blocks(theirs)
for b in tb:
    n = norm(b)
    if not n.strip() or n in have or n in seen:
        continue
    seen.add(n)
    add.append(n)
res = ours.rstrip('\n') + '\n'
if add:
    res += '\n---\n\n' + sys.argv[4] + '\n\n' + sys.argv[5] + '\n\n' + '\n\n'.join(add) + '\n'
have_out = {norm(b) for b in blocks(res)}
missing = [norm(b) for b in tb if norm(b).strip() and norm(b) not in have_out]
assert res.startswith(ours.rstrip('\n')), 'ours not preserved'
print(f'ours_blocks={len(blocks(ours))} theirs_blocks={len(tb)} theirs_distinct={len({norm(b) for b in tb})} appended={len(add)} missing_after={len(missing)}')
for m in missing[:5]:
    print('MISSING:', m[:150].replace('\n', ' | '))
if missing:
    sys.exit(3)
out = res.replace('\n', '\r\n') if crlf else res
open(sys.argv[3], 'wb').write(out.encode('utf-8'))
