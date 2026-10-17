"""Independent full canonical-group bootstrap on two primary contrasts."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import csv,json,pickle
import numpy as np
P=Path(__file__).resolve().parents[1];I=P/'inputs';p=np.load(I/'population.npz')
cls=p['classification'];groups=p['group'];y=p['y'];off=p['offsets'];ids=p['idx']
glist=np.unique(groups);gi={g:i for i,g in enumerate(glist)}
W=np.random.default_rng(20261011).multinomial(707,np.full(707,1/707),size=5000)
z=np.load(P/'controls/CONTROL_PER_ANSWER.npz')
delta=z['position_length__LSML_CONT__auc']-z['position_length__PRM_NATIVE__auc']
mask=(cls=='redundency')&np.isfinite(delta);sums=np.zeros(707);counts=np.zeros(707)
for v,g in zip(delta[mask],groups[mask]):sums[gi[g]]+=v;counts[gi[g]]+=1
sample=(W@sums)/(W@counts);interval=np.quantile(sample,[.025,.975])
rows=list(csv.DictReader((P/'controls/PAIRED_AUC_CONTRASTS.csv').open()))
row=next(r for r in rows if r['control']=='position_length' and r['classification']=='redundency'
         and r['left']=='LSML_CONT' and r['right']=='PRM_NATIVE')
assert np.max(abs(interval-[float(row['ci_low']),float(row['ci_high'])]))<1e-12
flags=np.load(P/'controls/DECISION_FLAGS.npz');ai=np.repeat(np.arange(len(ids)),np.diff(off));take=cls[ai]=='circular'
intervals=[];method_values=[]
for method,rule in [('LSML_CONT','ANSWER_Z_Q80'),('PRM_NATIVE','NATIVE_0_5')]:
    f=flags[method+'__'+rule];C=np.zeros((707,4))
    for i,g in enumerate(groups):
        if cls[i]!='circular':continue
        a,b=off[i:i+2];yy=y[a:b];ff=f[a:b]
        C[gi[g]]+=np.array([np.sum(yy&ff),np.sum(~yy&ff),np.sum(yy&~ff),np.sum(~yy&~ff)])
    conf=W@C;tp,fp,fn,tn=conf.T
    result=.5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn));method_values.append(result)
delta_score=method_values[0]-method_values[1];prmint=np.quantile(delta_score,[.025,.975])
rows=list(csv.DictReader((P/'controls/PAIRED_PRMSCORE_CONTRASTS.csv').open()))
row=next(r for r in rows if r['rule']=='ANSWER_Z_Q80' and r['classification']=='circular'
         and r['left']=='LSML_CONT' and r['right']=='PRM_NATIVE')
assert np.max(abs(prmint-[float(row['ci_low']),float(row['ci_high'])]))<1e-12
receipt=dict(status='PASS',draws=5000,unit='707 canonical source groups',seed=20261011,
             matched_NR_AUC_LSML_minus_PRM_interval=interval.tolist(),
             NCL_Q80_PRMScore_LSML_minus_native_PRM_interval=prmint.tolist(),
             inference='Pointwise development intervals; not selection/multiplicity-adjusted')
(P/'review/PAIRED_INTERVAL_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
