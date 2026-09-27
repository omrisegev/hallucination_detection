import json, pickle, numpy as np, pandas as pd
from pathlib import Path
R = Path(r'C:/Users/omris/TAU/hallucination_detection/.worktrees/readout-quickest-detection-v1/results/step_evidence_v1')
ans = pd.read_csv(R/'OOF_ANSWERS.csv', encoding='utf-8-sig', usecols=['uid','id','source_group','fold','cell','target'])
Z = np.load(R/'OOF_STEP_SCORES.npz'); off = Z['offsets']; lab = Z['labels'].astype(bool); ns = np.diff(off); n=len(ans)
freeze = json.loads((R/'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
meta = {m['idx']: m for m in pickle.load(open(freeze['prm_metadata']['path'],'rb')).values()}
prm = ~ans.cell.str.startswith('pb_').to_numpy(); ids = ans.id.to_numpy()
cls = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
has_err = np.array([prm[i] and lab[off[i]:off[i+1]].any() for i in range(n)])
nc = prm & (cls!='correct')
P = np.flatnonzero(prm)
lab_ok = all(np.array_equal(lab[off[i]:off[i+1]], np.isin(np.arange(ns[i])+1, meta[ids[i]]['error_steps'])) for i in P)
print('lab_ok', lab_ok)
s = set(cls[nc & ~has_err]) - {'multi_solutions'}
print('gate set (nonempty => STOP):', s, bool(s))
for i in np.flatnonzero(nc & ~has_err & (cls!='multi_solutions')):
    print(ids[i], ns[i], meta[ids[i]]['error_steps'])
