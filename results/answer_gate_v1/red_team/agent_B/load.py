import json, pickle, re
import numpy as np, pandas as pd
W=r'C:\Users\omris\TAU\hallucination_detection\.worktrees\decision-rule-v1'
AG=W+r'\results\answer_gate_v1'
R=r'C:\Users\omris\TAU\hallucination_detection\.worktrees\readout-quickest-detection-v1\results\step_evidence_v1'
RAW=r'C:\Users\DELL\.cache\huggingface\hub\datasets--hitsmy--PRMBench_Preview\snapshots\5cc7683d0ae5797f84d7aeac0607966f277c39e1\prmbench_preview.jsonl'
ans=pd.read_csv(R+r'\OOF_ANSWERS.csv',encoding='utf-8-sig',usecols=['uid','id','source_group','fold','cell','target'])
Zs=np.load(R+r'\OOF_STEP_SCORES.npz'); off=Zs['offsets']; labels=Zs['labels'].astype(bool); n=len(ans); ns=np.diff(off)
pb=ans.cell.str.startswith('pb_').to_numpy(); prm=~pb; ids=ans.id.to_numpy(); fold=ans.fold.to_numpy(); cells=ans.cell.to_numpy(); target=ans.target.to_numpy(); groups=ans.source_group.to_numpy()
mp=json.loads(open(R+r'\INPUT_FREEZE.json',encoding='utf-8-sig').read())['prm_metadata']['path']
meta={m['idx']:m for m in pickle.load(open(mp,'rb')).values()}
cls=np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
control=prm&(cls=='correct'); noncontrol=prm&~control
has_err=np.array([labels[off[i]:off[i+1]].any() for i in range(n)])&prm
ms=noncontrol&(cls=='multi_solutions'); err_nc=has_err&noncontrol
aid=np.repeat(np.arange(n),ns)
assert (int(prm.sum()),int(control.sum()),int(noncontrol.sum()),int(err_nc.sum()),int(ms.sum()))==(6969,758,6211,6035,160)
F=np.load(AG+r'\ANSWER_FEATURES.npz'); X=F['X'].astype(float); names=[str(x) for x in F['names']]
assert np.array_equal(F['ids'],ids.astype(str))
D=np.load(AG+r'\run_20260930\DECISIONS.npz'); assert np.array_equal(D['offsets'],off)
FL=[json.loads(l) for l in open(AG+r'\run_20260930\FIT_LOG.jsonl')]
# raw text
raw={}
for l in open(RAW,encoding='utf8'):
    r=json.loads(l); c=r['classification']; k=f"{c}_{r['idx']}"
    if k not in raw: raw[k]=dict(steps=list(r['modified_process']),q=r['modified_question'],cls=c,rawidx=r['idx'])
    if c=='redundency':
        k2=f"correct_{r['idx']}"
        if k2 not in raw: raw[k2]=dict(steps=list(r['original_process']),q=r['original_question'],cls='correct',rawidx=r['idx'])
def srckey(rawidx,c):
    return rawidx[len(c)+1:] if rawidx.startswith(c+'_') else rawidx
