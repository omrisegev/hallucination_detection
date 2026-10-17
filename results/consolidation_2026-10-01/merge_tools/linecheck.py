"""Every line of each side (that differs from base) must appear in the merged blob. usage: linecheck.py MERGE_COMMIT PATH..."""
import subprocess, sys, collections
R=r'C:\Users\omris\TAU\hallucination_detection'
g=lambda *a: subprocess.run(['git','-C',R,*a],capture_output=True).stdout
c=sys.argv[1]; p1,p2=g('rev-parse',c+'^1').decode().strip(),g('rev-parse',c+'^2').decode().strip(); mb=g('merge-base',p1,p2).decode().strip()
for p in sys.argv[2:]:
    L=lambda ref: collections.Counter(x.rstrip('\r') for x in g('show',f'{ref}:{p}').decode('utf-8','replace').split('\n'))
    r,a,b,o=L(c),L(p1),L(p2),L(mb)
    miss=sum(max(0,n-r[l]) for side in (a,b) for l,n in side.items() if l.strip() and n>o[l])
    print(f'{p}: lines ours={sum(a.values())} theirs={sum(b.values())} merged={sum(r.values())} nonblank_lines_missing={miss}')
