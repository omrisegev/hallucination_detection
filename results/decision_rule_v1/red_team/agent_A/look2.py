import numpy as np, json, pickle, pandas as pd
R='C:/Users/omris/TAU/hallucination_detection/.worktrees/'
a=pd.read_csv(R+'readout-quickest-detection-v1/results/step_evidence_v1/OOF_ANSWERS.csv', encoding='utf-8-sig', usecols=['uid','id','source_group','fold','cell','target'])
print(a.shape); print(a.cell.value_counts())
print(a.head()); print(a[~a.cell.str.startswith('pb_')].head())
fr=json.load(open(R+'readout-quickest-detection-v1/results/step_evidence_v1/INPUT_FREEZE.json', encoding='utf-8-sig'))
m=pickle.load(open(fr['prm_metadata']['path'],'rb'))
print(type(m), len(m))
it=list(m.items())[:2] if isinstance(m,dict) else m[:2]
for x in it: print(x)
v=list(m.values())
import collections
print(collections.Counter(x.get('classification') for x in v))
oof=np.load(R+'readout-quickest-detection-v1/results/step_evidence_v1/OOF_STEP_SCORES.npz')
print(oof['offsets'][:5], oof['labels'][:20])
