import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from load import pb, prmb
from auc import az
N16=['q15_H1','q15_VE1','chosen_surprisal','logprob_margin','true_tail50','energy_level','energy_innovation','top15_turnover','top50_js','dominant_freq16','bocpd_p0','realized_z','realized_drv','digit_alternative','digit_spread','digit_alternative_innovation']
def within(y,x,off,keep):
    r=[]
    for a,b in zip(off[:-1],off[1:]):
        k=keep[a:b]; yy=y[a:b][k]; xx=x[a:b][k]
        if yy.any() and (~yy).any():
            r.append(roc_auc_score(yy,xx))
    return np.mean(r),len(r)
def fe_auc(y,x,off,keep):
    # first-error AUC: first-error step of each error answer vs all steps of clean answers (MtG-like) 
    pos=[];neg=[]
    for a,b in zip(off[:-1],off[1:]):
        yy=y[a:b]
        if yy.any(): pos.append(x[a+np.argmax(yy)])
        else: neg.extend(x[a:b])
    return roc_auc_score(np.r_[np.ones(len(pos)),np.zeros(len(neg))],np.r_[pos,neg])
D={'PB':pb(),'PRMB':prmb()}
print(f'{"feature":30s} | PB within  PRMB within | PB pooled PRMB pooled | PB pooled-z PRMB pooled-z')
res={}
for f in N16:
    row=[]
    for k in ('PB','PRMB'):
        d=D[k]; x=d['V'][:,d['names'].index(f)]
        w,nw=within(d['err'],x,d['off'],d['keep']); p=roc_auc_score(d['err'][d['keep']],x[d['keep']]); pz=roc_auc_score(d['err'][d['keep']],az(x,d['off'])[d['keep']])
        row.append((w,p,pz,nw))
    print(f'{f:30s} | {row[0][0]:.3f}  {row[1][0]:.3f} | {row[0][1]:.3f}  {row[1][1]:.3f} | {row[0][2]:.3f}  {row[1][2]:.3f}')
print('eligible answers',row[0][3],row[1][3])
