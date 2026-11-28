import numpy as np
from scipy.stats import rankdata, spearmanr
from load import pb, prmb
U='/mnt/user-data/uploads/hd_collection_data/readout_steps_sa_v1/'
def wauc(y,x,off,keep):
    r=[]
    for a,b in zip(off[:-1],off[1:]):
        k=keep[a:b]; yy=y[a:b][k]; xx=x[a:b][k]; n1=yy.sum(); n0=len(yy)-n1
        if n1 and n0:
            rk=rankdata(xx); r.append((rk[yy].sum()-n1*(n1+1)/2)/(n1*n0))
    return float(np.mean(r))
def get():
    D={'PB':pb(),'PRMB':prmb()}
    for k,f in (('PB','processbench'),('PRMB','prmbench')):
        R=np.load(U+f+'__qwen_prm_7b.npz'); d=D[k]
        assert (R['uids'].astype(str)==d['ids'].astype(str)).all() and (R['offsets']==d['off']).all()
        d['ntok']=R['ntok'].astype(float); d['R']=R['readouts']; d['rn']=list(R['readout_names']); d['cn']=list(R['channel_names'])
        off=d['off']; p=np.zeros(len(d['err'])); idx=np.zeros(len(d['err']))
        for a,b in zip(off[:-1],off[1:]): p[a:b]=np.arange(b-a)/max(b-a-1,1); idx[a:b]=np.arange(b-a)
        d['pos']=p; d['idx']=idx; d['col']=lambda f,d=d: d['V'][:,d['names'].index(f)]
    return D
