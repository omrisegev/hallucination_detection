"""Review check: is 'first error = jump, later errors = level' an artefact of the per-answer z (R0)?
Redo the PRMBench dev first-vs-later comparison with the GLOBAL rank-Gauss readout (L.g16_cols, no per-answer z),
and with answers that have exactly 2 or 3 labelled errors only. Dev only, CPU, gold labels as a check only."""
import json, sys
from pathlib import Path
import numpy as np

CODE, W, OUT = sys.argv[1:4]; W = Path(W)
sys.path.insert(0, CODE)
import ssl_data as D, ssl_lib as L

split = json.load(open(W / 'inputs' / 'SPLIT.json')); labels = json.load(open(W / 'inputs' / 'prmbench_C_labels.json'))
C = D.load_pool(W, 'prmbench', 'C', split, None, labels); Cd = C.subset(C.side == 'dev'); raw = D.readout(Cd)
names = list(L.N16); g = np.repeat(Cd.group, np.diff(Cd.off))
first = np.zeros(Cd.off[-1], bool); later = np.zeros(Cd.off[-1], bool); before = np.zeros(Cd.off[-1], bool)
nerr = np.zeros(Cd.off[-1], int)
for a, b in zip(Cd.off[:-1], Cd.off[1:]):
    e = np.where(Cd.wrong_step[a:b])[0]; nerr[a:b] = len(e)
    if len(e): first[a + e[0]] = True; later[a + e[1:]] = True; before[a:a + e[0]] = True
R = {}
for rep, X in (('R0 per-answer z', L.r0_cols(raw, Cd.act, Cd.off)), ('G16 global rank-gauss', L.g16_cols(raw, Cd.act))):
    def c2(ma, mb):
        rng = np.random.default_rng(0); ia, ib = np.where(ma)[0], np.where(mb)[0]; n = min(len(ia), len(ib))
        ia, ib = rng.choice(ia, n, replace=False), rng.choice(ib, n, replace=False)
        return round(float(L.c2st(X[ia], X[ib], g[ia], g[ib])), 4), int(n)
    sub = (nerr >= 2) & (nerr <= 3)
    sel = ['chosen_surprisal', 'q15_H1', 'energy_level', 'bocpd_p0', 'realized_z', 'realized_drv']
    R[rep] = dict(first_vs_later=c2(first, later), first_vs_later_answers_with_2_3_errors=c2(first & sub, later & sub),
                  first_vs_clean_before=c2(first, before),
                  means={k: {n: round(float(X[m, names.index(n)].mean()), 3) for n in sel}
                         for k, m in (('clean before first', before), ('first', first), ('later', later))})
    print(rep, json.dumps(R[rep]), flush=True)
Path(OUT).write_text(json.dumps(R, indent=1))
