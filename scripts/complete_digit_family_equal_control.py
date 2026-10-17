"""Complete predeclared each-bank equal control omitted from the primary runner.

Run after primary outcomes; no primary artifact changes or parameter selection.
"""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_digit_family_extension_v1 import final_z,write
from scripts.run_digit_alternative_probability_v1 import rows,sha,auc
from spectral_utils.digit_feature_family import answer_standardize
from spectral_utils.prmbench import prmbench_evaluate

out=ROOT/'results/digit_family_extension_v1';supp=out/'duplicate_equal_completion';supp.mkdir(exist_ok=True)
if (supp/'PREDICTIONS.npz').exists():raise RuntimeError('Refuse to overwrite')
old=ROOT/'results/digit_alternative_probability_v1';rec=json.loads((old/'RECORDS.json').read_text())
inp=json.loads((out/'FIT_INPUTS.json').read_text());assert sha(Path(inp['level']['path']))==inp['level']['sha256']
f=np.load(out/'PREDICTIONS.npz');off=f['offsets'];folds=f['folds'];sf=np.repeat(folds,np.diff(off))
base=answer_standardize(np.load(inp['level']['path'])['level'],off)
feat=np.load(out/'FEATURES.npz');num=answer_standardize(feat['values'],off,feat['active'])[:,0]
bank=np.column_stack((base,num,num,num))
np.testing.assert_allclose(bank.mean(axis=1),(base.sum(axis=1)+3*num)/14,atol=1e-14,rtol=0)
score=final_z(bank.mean(axis=1),off);pred=np.full(len(score),-1,np.int8);taus=[]
for test in range(5):
    tau=float(np.quantile(score[sf==(test+1)%5],.8));pred[sf==test]=score[sf==test]<tau;taus.append(tau)
np.savez_compressed(supp/'PREDICTIONS.npz',score=score,pred=pred,offsets=off)
write(supp/'SEAL.json',{'sha256':sha(supp/'PREDICTIONS.npz'),'thresholds':taus,'code_sha256':sha(Path(__file__)),
    'reason':'Predeclared equal fusion on each bank; duplicate bank equal row was omitted by runner. Completed after primary outcomes, no changes to primary inference.'})
raw=np.load(old/'SCORES.npz');labels=raw['labels'];target=raw['target'];source=ROOT.parents[1]
meta={r['idx']:r for r in rows(source/'dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl')}
prm=[i for i,r in enumerate(rec) if r['cell'].startswith('prm')];hits={};within=[]
for i,r in enumerate(rec):
    a,b=off[i:i+2];s=score[a:b];y=labels[a:b]
    if r['cell'].startswith('pb') and target[i]>=0:
        peak=int(np.flatnonzero(s>=s.max()-8*np.finfo(float).eps)[0]);hits.setdefault(r['cell'],[]).append(peak==target[i])
    elif r['cell'].startswith('prm') and 0 in y and 1 in y:within.append(auc(y==1,s))
official=prmbench_evaluate([{'idx':rec[i]['row_id'],'labels':pred[off[i]:off[i+1]].astype(int).tolist()} for i in prm],
                         [meta[rec[i]['row_id']] for i in prm])['total']
m={'method':'copies3_equal','n_checked':len(rec),'prmscore':.5*(official['f1']+official['negative_f1']),
   'prmscore_n':6211,'within_auc':float(np.mean(within)),'within_n':len(within),
   'pb_macro_exact':float(np.mean([np.mean(v) for v in hits.values()])), 'pb_n':sum(map(len,hits.values())),
   'timing':'predeclared control completed after primary results; descriptive only'}
write(supp/'METRICS.json',m);print(json.dumps(m,indent=2))
