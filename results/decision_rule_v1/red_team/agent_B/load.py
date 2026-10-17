import json, pickle, numpy as np, pandas as pd, sys
from pathlib import Path
MAIN = Path(r'C:/Users/omris/TAU/hallucination_detection')
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
W = MAIN / '.worktrees/decision-rule-v1/results/decision_rule_v1/run_20260929'
SSL = MAIN / '.worktrees/ssl-pseudolabel-residual-v1'
ans = pd.read_csv(R/'OOF_ANSWERS.csv', encoding='utf-8-sig'); Z = np.load(R/'OOF_STEP_SCORES.npz')
off = Z['offsets']; labels = Z['labels'].astype(bool); n=len(ans); ns=np.diff(off); S_=int(off[-1])
meta_path = Path(json.loads((R/'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))['prm_metadata']['path'])
metaraw = pickle.load(open(meta_path,'rb'))
D = np.load(W/'DECISIONS.npz')
