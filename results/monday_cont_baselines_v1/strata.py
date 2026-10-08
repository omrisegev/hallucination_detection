"""ProcessBench strata (same definitions as monday_nested_banks_v2/dev_run.py) for IU-PCR, continuous L-SML and the
oriented average on every bank; point estimates (cell-equal mean over the 8 cells). Writes PB_STRATA.csv."""
import json
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent; MAIN = OUT.parents[1]
SE = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
A = pd.read_csv(SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); cells = A.cell.to_numpy(); target = A.target.to_numpy().astype(int)
with np.load(SE / 'OOF_STEP_SCORES.npz') as z: OFF = z['offsets'].astype(np.int64)
fr = json.loads((SE / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
with np.load(fr['tokens']['path']) as z: spans = z['step_spans']
TLEN = (spans[:, 1] - spans[:, 0]).astype(float)
pb_idx = np.flatnonzero([c.startswith('pb_') for c in cells]); err = target[pb_idx] >= 0
longest = np.array([int(np.argmax(TLEN[OFF[i]:OFF[i + 1]])) for i in pb_idx]); pcells = cells[pb_idx]
strata = [('ALL', np.ones(len(pb_idx), bool)), ('content-only (not step 0, not longest)', err & (target[pb_idx] != 0) & (longest != target[pb_idx])),
          ('error at step 0', err & (target[pb_idx] == 0)), ('error in longest step', err & (longest == target[pb_idx]) & (target[pb_idx] != 0))]
strata += [('subset:' + s, np.array([c.split('_')[1] == s for c in pcells])) for s in ['gsm8k', 'math', 'olympiadbench', 'omnimath']]
PC = np.load(OUT / 'PER_ANSWER.npz'); rows = []
banks = sorted({k.split('__')[0] for k in PC.files})
for bk in banks:
    for nm in ('IU_PCR', 'LSML_CONT', 'ALL_EQ_ORIENTED'):
        hit = PC[f'{bk}__{nm}__argmax'][pb_idx] == target[pb_idx]
        for sname, m in strata:
            vals = [(hit & err & m & (pcells == c)).sum() / max((err & m & (pcells == c)).sum(), 1) for c in sorted(set(pcells)) if (err & m & (pcells == c)).sum()]
            rows.append({'stratum': sname, 'scorer': nm, 'bank': bk, 'argmax_hit': float(np.mean(vals))})
D = pd.DataFrame(rows); D.to_csv(OUT / 'PB_STRATA.csv', index=False)
chk = D[(D.stratum == 'ALL')].set_index(['bank', 'scorer']).argmax_hit
ab = pd.read_csv(OUT / 'ABSOLUTE.csv').set_index(['bank', 'scorer']).pb_argmax_hit
print('max diff vs ABSOLUTE ALL:', float(max(abs(chk[k] - ab[k]) for k in chk.index)))
