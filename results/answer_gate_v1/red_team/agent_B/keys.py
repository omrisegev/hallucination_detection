import numpy as np, json
W=r'C:\Users\omris\TAU\hallucination_detection\.worktrees\decision-rule-v1\results\answer_gate_v1'
D=np.load(W+r'\run_20260930\DECISIONS.npz')
for k in D.files: print(k, D[k].shape, D[k].dtype)
F=np.load(W+r'\ANSWER_FEATURES.npz'); print(F.files, F['X'].shape)
L=[json.loads(l) for l in open(W+r'\run_20260930\FIT_LOG.jsonl')]
print(len(L)); print(L[0]); print(L[1]); print(L[2])
import collections
print(collections.Counter((r['detector'],r['flipped']) for r in L))
