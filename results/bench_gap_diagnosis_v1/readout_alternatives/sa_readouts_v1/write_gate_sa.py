"""Collect per-cell gate info from the standalone (sa) readout npz files into GATE.json (2026-11-12).

    python write_gate_sa.py <readout_steps_sa_v1 dir>
Reads each <bench>__<prm>.npz 'info' field (written by extract_readouts.py) and sha.txt (raw pickle sha256 check vs
the bank MANIFEST.json raw_sha256, written by the download driver). No labels are read.
"""
import json, os, sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
sha = {}
if (d / 'sha.txt').exists():
    for line in (d / 'sha.txt').read_text().split('\n'):
        if line.strip():
            b, p, ok = line.split()
            sha[f'{b}__{p}'] = ok.endswith('1')
cells = {}
for f in sorted(d.glob('*__*.npz')):
    info = json.loads(str(np.load(f)['info']))
    g = info.get('gate', {})
    cells[f.stem] = dict(gate=g.get('status'), max_abs_diff=g.get('max_abs_diff_all'), answers=info['records'],
                         gate_answers=g.get('answers'), steps=info['steps'], failed_rows=info['failed_rows'],
                         pre_empty_answers=info['pre_empty_answers'], seconds=info['seconds'],
                         raw_sha256_ok=sha.get(f.stem), bank=os.path.basename(os.path.dirname(g.get('bank', ''))),
                         npz_bytes=f.stat().st_size)
out = dict(script='results/bench_gap_diagnosis_v1/readout_alternatives/extract_readouts.py (unchanged)',
           gate_rule='top10 readout equals stored sa bank bank11 columns, max abs diff <= 1e-6',
           n_cells=len(cells), all_pass=all(c['gate'] == 'PASS' for c in cells.values()), cells=cells)
(d / 'GATE.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: (v['gate'], v['max_abs_diff'], v['answers'], v['npz_bytes']) for k, v in cells.items()}, indent=0))
