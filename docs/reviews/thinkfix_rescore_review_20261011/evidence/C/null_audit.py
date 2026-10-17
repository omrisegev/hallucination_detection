import numpy as np,json,hashlib,time
from scipy.stats import rankdata
from pathlib import Path
P=Path('/private/tmp/thinkfix-review-080877eb/results/thinkfix_rescore_v1/prmscore_inputs'); OUT=Path('/private/tmp/thinkfix-audit-C')
z=np.load(P/'bank_N16.npz',allow_pickle=False); meta={str(m['idx']):m for m in json.load(open(P/'prmbench_meta.json'))['rows']}; off=z['offsets']; ids=z['idx']; rng=np.random.default_rng(20261011); B=1000
names=[k for k in z.files if k.startswith('S__')]+['G']; z={k:z[k] for k in z.files}; out={}; A=[]; D=[]; Y=[]; idx=[]; labels=[]; counts={}; bad=[]
for j,uid in enumerate(ids):
 a,b=off[j:j+2]; m=meta[str(uid)]; n=b-a
 if n!=m['n_steps']:bad.append([str(uid),int(n),m['n_steps']])
 y=np.isin(np.arange(1,n+1),m['error_steps']);labels.extend(y); counts[m['classification']]=counts.get(m['classification'],0)+1
 if y.any() and not y.all():
  ranks=rankdata(np.column_stack([z[k][a:b] for k in names]),method='average',axis=0); n1=int(y.sum()); n0=n-n1
  assert np.allclose(int(y.sum())/len(y)*((ranks-(n1+1)/2)/(n1*n0)).sum(0),.5)
  A.append((ranks[y].sum(0)-n1*(n1+1)/2)/(n1*n0));Y.append(y);D.append((ranks-(n1+1)/2)/(n1*n0));idx.append(j)
if bad:raise RuntimeError(bad[:10])
obs=np.mean(A,axis=0);null=np.zeros((B,len(names)));t0=time.time()
# Each null uniformly selects exactly n_error positions separately within each answer;
# shared selection across methods preserves pairing. No fit recomputation.
for y,weights in zip(Y,D):
 r=rng.random((B,len(y)));selected=np.argsort(r,axis=1)[:,:int(y.sum())];null+=weights[selected].sum(1)
null/=len(Y)
# Permute one complete score vector across all steps (different null, does not refit).
score=z['S__MEAN_MOM_ga'].copy();permuted=rng.permutation(score);perm_auc=[]
for j in idx:
 a,b=off[j:j+2]; y=np.asarray(labels[a:b]);r=rankdata(permuted[a:b]);n1=int(y.sum());n0=len(y)-n1;perm_auc.append((r[y].sum()-n1*(n1+1)/2)/(n1*n0))
for k,v,draw in zip(names,obs,null.T):out[k]={'observed_within_auc':float(v),'null_mean':float(draw.mean()),'null_sd':float(draw.std(ddof=1)),'null_ci95':np.quantile(draw,[.025,.975]).tolist(),'p_upper_plus_one':float((1+np.sum(draw>=v))/(B+1)),'max_observed_or_flipped':float(max(v,1-v))}
report={'scope':'saved-score within-answer ranking null only; no full feature extraction/fitting null','bank_sha256':hashlib.sha256((P/'bank_N16.npz').read_bytes()).hexdigest(),'meta_sha256':hashlib.sha256((P/'prmbench_meta.json').read_bytes()).hexdigest(),'n_answers':len(ids),'n_steps':int(off[-1]),'n_unique_answer_ids':len(set(ids)),'n_mixed_answers_auc':len(Y),'n_positive_steps':int(sum(labels)),'answer_classification_counts':counts,'B':B,'seed':20261011,'results':out,'permuted_one_score_global_MEAN_MOM_ga_within_auc':float(np.mean(perm_auc)),'elapsed_s':time.time()-t0}
np.savez_compressed(OUT/'null_draws.npz',methods=np.asarray(names),within_auc=null,observed=obs)
(OUT/'NULL_REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
