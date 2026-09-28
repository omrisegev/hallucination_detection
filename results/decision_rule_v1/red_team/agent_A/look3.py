import numpy as np, json, pickle, pandas as pd, collections
R='C:/Users/omris/TAU/hallucination_detection/.worktrees/'
a=pd.read_csv(R+'readout-quickest-detection-v1/results/step_evidence_v1/OOF_ANSWERS.csv', encoding='utf-8-sig', usecols=['uid','id','source_group','fold','cell','target'])
pr=a[~a.cell.str.startswith('pb_')]
print(pr[['id','source_group','fold','target']].head(10).to_string())
print(pr.target.value_counts().head())
print(a[a.cell.str.startswith('pb_')].target.value_counts().head())
oof=np.load(R+'readout-quickest-detection-v1/results/step_evidence_v1/OOF_STEP_SCORES.npz')
L=oof['labels']; off=oof['offsets']
print(collections.Counter(L.tolist()))
idx=a.index[~a.cell.str.startswith('pb_')]
print('prm labels', collections.Counter(np.concatenate([L[off[i]:off[i+1]] for i in idx]).tolist()))
print('pb labels', collections.Counter(np.concatenate([L[off[i]:off[i+1]] for i in a.index[a.cell.str.startswith('pb_')]]).tolist()))
fr=json.load(open(R+'readout-quickest-detection-v1/results/step_evidence_v1/INPUT_FREEZE.json', encoding='utf-8-sig'))
m=pickle.load(open(fr['prm_metadata']['path'],'rb'))
ids=set(pr.id); idxs=set(str(x['idx']) for x in m.values())
print(len(ids), len(idxs), len(ids & idxs))
print(list(ids)[:5])
print([k for k in fr['protocol'].keys()])
