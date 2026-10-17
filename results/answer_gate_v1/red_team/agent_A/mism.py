import json, numpy as np
from common import load_all, ROOT, PRMCELL
ER = ROOT + "/.worktrees/ssl-pseudolabel-residual-v1/results/expectation_realization_v1/"
df, off, lab, d, bymeta = load_all()
n_steps=np.diff(off); folds=df.fold.values
S=np.load(ER+"run_20260927_stage_b/STEP_SCORES.npz")["S_equal"]
thr=json.load(open(ER+"run_20260927_stage_b_thr/THRESHOLDS.json"))["S_equal"]
ans=np.repeat(np.arange(len(df)),n_steps)
az=np.empty_like(S)
for i in range(len(df)):
    x=S[off[i]:off[i+1]]; sd=x.std()
    az[off[i]:off[i+1]]=(x-x.mean())/sd if sd>0 else 0.0
r0=az>=np.array([thr[str(f)] for f in folds[ans]])
for j in np.where(r0!=d['R0_frozen'])[0]:
    i=ans[j]; print('step',j,'answer',i,df.cell.iloc[i],'nsteps',n_steps[i],'az',repr(az[j]),'thr',thr[str(folds[i])],'saved',d['R0_frozen'][j], 'S',S[off[i]:off[i+1]])
