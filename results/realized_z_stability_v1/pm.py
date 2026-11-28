import numpy as np
from scipy.stats import rankdata
from fast import wauc, get
def az(x,off):
    o=x.copy()
    for a,b in zip(off[:-1],off[1:]):
        s=x[a:b].std(); o[a:b]=(x[a:b]-x[a:b].mean())/s if s>1e-12 else 0
    return o
def pm_auc(d,x,maxj=12):
    """position-matched first-error AUC: at each step index j, first-error steps at j vs steps at j that are still correct (prefix still clean). Weighted mean over j by #positives."""
    off=d['off']; y=d['err']; idx=d['idx'].astype(int)
    fe=np.full(len(off)-1,10**6)
    for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
        if y[a:b].any(): fe[i]=np.argmax(y[a:b])
    ans=np.repeat(np.arange(len(off)-1),np.diff(off)); f=fe[ans]
    pos=idx==f; neg=idx<f
    tot=0;w=0
    for j in range(maxj):
        p=pos&(idx==j); n=neg&(idx==j)
        if p.sum()<5 or n.sum()<5: continue
        r=rankdata(np.r_[x[p],x[n]]); a=(r[:p.sum()].sum()-p.sum()*(p.sum()+1)/2)/(p.sum()*n.sum())
        tot+=a*p.sum(); w+=p.sum()
    return tot/w
if __name__=='__main__':
    D=get()
    N16=['q15_H1','q15_VE1','chosen_surprisal','logprob_margin','true_tail50','energy_level','energy_innovation','top15_turnover','top50_js','dominant_freq16','bocpd_p0','realized_z','realized_drv','digit_alternative','digit_spread','digit_alternative_innovation']
    print(f'{"feature":28s} within PB|PRMB   PM raw PB|PRMB   PM answer-z PB|PRMB')
    for f in N16+['POSITION','LOG_NTOK']:
        o=[]
        for d in D.values():
            x=d['pos'] if f=='POSITION' else np.log(d['ntok']) if f=='LOG_NTOK' else d['col'](f)
            o.append((wauc(d['err'],x,d['off'],d['keep']),pm_auc(d,x),pm_auc(d,az(x,d['off']))))
        print(f'{f:28s} {o[0][0]:.3f}|{o[1][0]:.3f}      {o[0][1]:.3f}|{o[1][1]:.3f}      {o[0][2]:.3f}|{o[1][2]:.3f}')
    print('chosen_surprisal readouts')
    for r in D['PB']['rn']:
        o=[]
        for d in D.values():
            x=d['R'][:,d['cn'].index('chosen_surprisal'),d['rn'].index(r)]
            o.append((wauc(d['err'],x,d['off'],d['keep']),pm_auc(d,x),pm_auc(d,az(x,d['off']))))
        print(f'  cs_{r:24s} {o[0][0]:.3f}|{o[1][0]:.3f}      {o[0][1]:.3f}|{o[1][1]:.3f}      {o[0][2]:.3f}|{o[1][2]:.3f}')
