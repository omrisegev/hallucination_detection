import json, pickle
from pathlib import Path
import numpy as np, pandas as pd
R = Path(r'C:/Users/omris/TAU/hallucination_detection/.worktrees/readout-quickest-detection-v1/results/step_evidence_v1')
ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Z = np.load(R / 'OOF_STEP_SCORES.npz')
off = Z['offsets']; ns = np.diff(off); n = len(ans)
print('steps', int(off[-1]), 'answers', n, 'min/max len', ns.min(), ns.max(), 'keys', Z.files)
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); ids = ans.id.to_numpy()
meta_path = Path(json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))['prm_metadata']['path'])
meta = {m['idx']: m for m in pickle.load(open(meta_path, 'rb')).values()}
cls = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
nc = prm & (cls != 'correct')
k = pd.DataFrame({'len': ns[nc], 'fold': fold[nc]}); sz = k.groupby(['len', 'fold']).size()
single = sz[sz == 1]
print('noncontrol answers', nc.sum(), 'cells', len(sz), 'singleton cells', len(single), 'answers in singletons', int(single.sum()),
      'steps in singletons', int(sum(l for (l, f) in single.index)), 'of', int(ns[nc].sum()))
print('cells of size 2:', int((sz == 2).sum()), 'answers in cells <=3:', int(sz[sz <= 3].sum()))
print('PB cells', sorted(set(ans.cell[pb])))
print('PRM groups', ans.source_group[prm].nunique(), 'PB groups', ans.source_group[pb].nunique())
print('prm lengths 1-step answers', int((ns[prm] == 1).sum()), 'pb 1-step', int((ns[pb] == 1).sum()))
# folds constant within an answer by construction; check groups do not straddle folds
g = ans.groupby('source_group').fold.nunique(); print('source groups spanning >1 fold:', int((g > 1).sum()))
print('PB target values: min', ans.target[pb].min(), 'share -1', float((ans.target[pb] == -1).mean()))
