import json, pickle, numpy as np, pandas as pd
from pathlib import Path
R = Path(r'C:/Users/omris/TAU/hallucination_detection/.worktrees/readout-quickest-detection-v1/results/step_evidence_v1')
ans = pd.read_csv(R/'OOF_ANSWERS.csv', encoding='utf-8-sig', usecols=['uid','id','source_group','fold','cell','target'])
Z = np.load(R/'OOF_STEP_SCORES.npz'); off = Z['offsets']; lab = Z['labels'].astype(bool); ns = np.diff(off); n=len(ans)
freeze = json.loads((R/'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
raw = pickle.load(open(freeze['prm_metadata']['path'],'rb'))
print(type(raw), len(raw)); k0 = next(iter(raw)); print('key example', k0, 'fields', list(raw[k0].keys()))
meta = {m['idx']: m for m in raw.values()}
prm = ~ans.cell.str.startswith('pb_').to_numpy(); ids = ans.id.to_numpy()
cls = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
has_err = np.array([prm[i] and lab[off[i]:off[i+1]].any() for i in range(n)])
has_ok = np.array([prm[i] and (~lab[off[i]:off[i+1]]).any() for i in range(n)])
ctrl = prm & (cls=='correct'); nc = prm & ~ctrl
print('prm', prm.sum(), 'ctrl', ctrl.sum(), 'nc', nc.sum(), 'has_err', has_err.sum(), 'eligible', (has_err&has_ok).sum())
print('classes', pd.Series(cls[prm]).value_counts().to_dict())
ms = cls=='multi_solutions'
print('MS total', ms.sum(), 'MS with in-range err', (ms&has_err).sum(), 'MS without err', (ms&~has_err).sum())
print('MS with nonempty error_steps (meta)', sum(1 for i in np.flatnonzero(ms) if meta[ids[i]]['error_steps']))
print('nc answers with only error steps', (nc & has_err & ~has_ok).sum(), 'lengths', ns[nc & has_err & ~has_ok][:20])
print('nc non-MS without in-range err', (nc & ~ms & ~has_err).sum())
print('eight-class answers', (nc & ~ms).sum(), 'eight-class with err', (nc&~ms&has_err).sum())
print('ctrl with err', (ctrl&has_err).sum())
# rewards key
i0 = np.flatnonzero(prm)[0]; print('rewards key present', 'rewards' in meta[ids[i0]], len(meta[ids[i0]].get('rewards',[])), ns[i0])
bad=0
for i in np.flatnonzero(prm):
    r = meta[ids[i]].get('rewards'); 
    if r is None or len(r)!=ns[i] or not np.all(np.isfinite(r)): bad+=1
print('reward length/NaN mismatches', bad)
rw = np.concatenate([np.asarray(meta[ids[i]]['rewards'],float) for i in np.flatnonzero(prm)])
print('rewards exactly 0.5:', (rw==0.5).sum(), 'n', rw.size)
# groups
g = ans.source_group.to_numpy()[prm]; print('PRM source groups', len(np.unique(g)))
print('ns==1 PRM answers', (prm & (ns==1)).sum())
print('folds of prm', np.bincount(ans.fold.to_numpy()[prm]))
# first error position distribution
nce = np.flatnonzero(nc & has_err)
f = np.array([np.flatnonzero(lab[off[i]:off[i+1]])[0] for i in nce])
print('first err at 0:', (f==0).sum(), 'of', len(nce))
q = np.quantile(ns[nc],[.25,.5,.75]); print('len quartile edges', q)
bins = np.searchsorted(q, ns[nc], side='left'); print('len bin counts', np.bincount(bins))
