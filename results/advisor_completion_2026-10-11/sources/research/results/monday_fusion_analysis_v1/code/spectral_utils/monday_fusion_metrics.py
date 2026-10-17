"""Separate localization, PRMScore, clean-answer and first-error endpoints."""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from spectral_utils.prmbench import prmbench_evaluate

def answer_z(score,off):
    result=np.zeros_like(score,dtype=float)
    for a,b in zip(off[:-1],off[1:]):
        v=score[a:b]; result[a:b]=(v-v.mean())/max(v.std(),1e-8)
    return result

def top_flags(score,off,counts):
    out=np.zeros(len(score),bool)
    for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
        out[a+np.argsort(-score[a:b],kind='stable')[:min(int(counts[i]),b-a)]]=True
    return out

def within_auc(y,s):
    n1=int(y.sum()); n0=len(y)-n1
    if not n1 or not n0: return np.nan
    return float((rankdata(s)[y].sum()-n1*(n1+1)/2)/(n1*n0))

def prmscore_counts(c):
    tp,fp,tn,fn=np.moveaxis(np.asarray(c,float),-1,0)
    def ratio(a,b):return np.divide(a,b,out=np.full(np.shape(a),-1.,float),where=b!=0)
    return .5*(ratio(2*tp,2*tp+fp+fn)+ratio(2*tn,2*tn+fp+fn))

def evaluate(pop,bank,score,flags=None,detector=None):
    ans=bank['answers']; rows=bank['rows']; off=bank['off']
    if not np.isfinite(score).all() or len(score)!=int(off[-1]): raise ValueError('Incomplete/nonfinite scores')
    if flags is None:
        az=answer_z(score,off); flags=az>=np.quantile(az,.8)
    auc=np.full(len(ans),np.nan); counts=np.zeros((len(ans),4),int)
    first=np.full(len(ans),-1,int); argmax=np.zeros(len(ans),int); flagcount=np.zeros(len(ans),int)
    err=np.zeros(len(ans),bool); errorcount=np.full(len(ans),-1,int)
    for j,(i,a,b) in enumerate(zip(ans,off[:-1],off[1:])):
        ff=flags[a:b]; flagcount[j]=ff.sum()
        first[j]=int(np.flatnonzero(ff)[0]) if ff.any() else -1
        argmax[j]=int(np.flatnonzero(score[a:b]>=score[a:b].max()-8*np.finfo(float).eps)[0])
        if pop.prm[i]:
            yy=pop.labels[pop.off[i]:pop.off[i+1]]; errorcount[j]=yy.sum(); err[j]=yy.any()
            auc[j]=within_auc(yy,score[a:b])
            if pop.noncontrol[i]:
                valid=~ff; gold=~yy
                counts[j]=[(valid&gold).sum(),(valid&~gold).sum(),(~valid&~gold).sum(),(~valid&gold).sum()]
        else: err[j]=pop.target[i]>=0
    metric={'within_answer_auc':float(np.nanmean(auc)) if np.isfinite(auc).any() else None,
            'prmscore':None,'pb_first_error_accuracy':None,'pb_official_f1':None,
            'pb_decision_error_accuracy':None,'clean_flag_rate':float(np.mean(flagcount[~err]>0)) if (~err).any() else None,
            'wrong_answer_auroc':None,'coverage':1.,'failures':0,'n_answers':len(ans),'n_steps':len(score),
            'n_auc_answers':int(np.isfinite(auc).sum()),'n_erroneous':int(err.sum()),'n_clean':int((~err).sum()),
            'error_count_distribution':None}
    if pop.prm[ans[0]]:
        value=float(prmscore_counts(counts.sum(0)))
        official=prmbench_evaluate([{'idx':pop.ids[i],'labels':(~flags[a:b]).astype(int).tolist()} for i,a,b in zip(ans,off[:-1],off[1:])],[pop.meta[pop.ids[i]] for i in ans])['total']
        expected=.5*(official['f1']+official['negative_f1'])
        if abs(value-expected)>1e-12: raise RuntimeError(f'PRMScore counts differ from official evaluator: {value} vs {expected}')
        metric['prmscore']=value if counts.sum()>0 else None
        u,c=np.unique(errorcount,return_counts=True); metric['error_count_distribution']=dict(zip(map(str,u),map(int,c)))
    else:
        hit=first==pop.target[ans]
        ae=float(hit[err].mean()) if err.any() else np.nan
        ac=float(hit[~err].mean()) if (~err).any() else np.nan
        metric.update(pb_first_error_accuracy=float((argmax[err]==pop.target[ans][err]).mean()) if err.any() else None,
                      pb_decision_error_accuracy=ae if err.any() else None,
                      pb_official_f1=(2*ae*ac/(ae+ac) if ae+ac>0 else 0.) if err.any() and (~err).any() else None)
    if detector is None: detector=np.array([score[a:b].max() for a,b in zip(off[:-1],off[1:])])
    if min(int(err.sum()),int((~err).sum()))>=10:
        metric['wrong_answer_auroc']=float(roc_auc_score(err,detector))
    return metric,{'auc':auc,'confusion':counts,'first':first,'argmax':argmax,'flag_count':flagcount,'erroneous':err}
