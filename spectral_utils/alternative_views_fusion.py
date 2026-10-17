"""Raw alternative views and block-variability shrinkage, without labels."""
import numpy as np
from .digit_fusion import digit_streams,add_correction
from .direct_probability_fusion_v2 import residual_tail_mass
from .step_evidence_fusion import standardize
from .predictor_subset_fusion import fast_iu,top10

VIEWS=('surprisal','rank','mass_above','gap','logtail15','logtail50','digit',
       'tail15','tail50')
HEADS=('equal','family_equal','iu','diag','block')
BANKS=('new7','augmented12')
METHODS=tuple('raw__'+v for v in VIEWS)+tuple('add__'+v for v in VIEWS)+tuple(b+'__'+h for b in BANKS for h in HEADS)


def extract(provided,top_ids,lp,spill):
    g=np.asarray(provided);ids=np.asarray(top_ids);lp=np.asarray(lp,float);spill=np.asarray(spill,float)
    if ids.shape!=lp.shape or ids.shape!=(len(g),50) or spill.shape!=g.shape:raise ValueError('Raw shape mismatch')
    if not np.isfinite(lp).all() or not np.isfinite(spill).all():raise ValueError('Nonfinite raw stream')
    if np.any(np.diff(lp,axis=1)>2e-6):raise ValueError('Top-k ordering mismatch')
    hit=ids==g[:,None];present=hit.any(1);rank=np.where(present,hit.argmax(1),50)
    if np.any(hit.sum(1)>1):raise ValueError('Duplicate provided token in top-k')
    if present.any():
        delta=float(np.max(np.abs(spill[present]+lp[np.flatnonzero(present),rank[present]])))
        if delta>2e-5:raise ValueError('Provided surprisal disagrees with saved log probability')
    else:delta=0.
    prob=np.exp(lp);cum=np.cumsum(prob,axis=1)
    above=np.where(present,np.where(rank>0,cum[np.arange(len(g)),np.maximum(rank-1,0)],0.),cum[:,-1])
    gap=lp[:,0]+spill
    if gap.min() < -2e-5:raise ValueError('Negative top1 gap')
    t15=residual_tail_mass(lp[:,:15]);t50=residual_tail_mass(lp)
    d=digit_streams(g,ids[:,0],range(15,25),'unused')[0]
    x=np.column_stack((spill,rank,above,np.maximum(gap,0.),np.log(np.maximum(t15,1e-12)),
                       np.log(np.maximum(t50,1e-12)),d,t15,t50))
    return x,dict(provided_probability_delta=delta,absent_top50=int((~present).sum()))


def shrink_covariance(z,C,groups,kind,alpha_override=None):
    target=np.diag(np.diag(C)) if kind=='diag' else np.where(groups[:,None]==groups[None,:],C,0.)
    changed=np.abs(C-target)>1e-12;np.fill_diagonal(changed,False)
    moments=[];sizes=[]
    for a in range(0,len(z),16):
        block=z[a:a+16];moments.append(block.T@block/len(block));sizes.append(len(block))
    weights=np.asarray(sizes,float)/len(z);blocks=np.array(moments)
    np.testing.assert_allclose(np.einsum('i,ijk->jk',weights,blocks),C,atol=2e-12,rtol=2e-12)
    den=np.square(C-target)[changed].sum()
    if den<=1e-24:alpha=0.
    elif len(weights)<2:alpha=1.
    else:
        variance=np.einsum('i,ijk->jk',weights**2,np.square(blocks-C))/(1-np.square(weights).sum())
        alpha=float(np.clip(variance[changed].sum()/den,0,1))
    if alpha_override is not None:alpha=float(alpha_override)
    return (1-alpha)*C+alpha*target,alpha,len(weights)


def score(bank,new,spans,base):
    singles=top10(new,spans)
    output=[singles[:,j] for j in range(9)]+[add_correction(base,singles[:,j]) for j in range(9)]
    diagnostics={}
    for name,x,g in [('new7',new[:,:7],np.array([0]*4+[1]*2+[2])),
                     ('augmented12',np.column_stack((bank,new[:,:7])),np.array([0]*5+[1]*4+[2]*2+[3]))]:
        z,sd=standardize(x);live=np.flatnonzero(sd>1e-12);z=z[:,live];group=g[live];m=len(live)
        C=z.T@z/len(z);summaries=top10(z,spans)
        equal=np.ones(m)/max(m,1);family=np.zeros(m)
        for k in np.unique(group):family[group==k]=1/(len(np.unique(group))*np.sum(group==k))
        info={};ws=[]
        for head in HEADS:
            reason=None;alpha=None;fitted=C
            if head=='equal':w=equal
            elif head=='family_equal':w=family
            else:
                if head in ('diag','block'):fitted,alpha,_=shrink_covariance(z,C,group,head)
                if m<3:reason='fewer_than_three_live_views';w=equal
                elif np.linalg.eigvalsh(fitted)[-2]<=1e-10:reason='rank_below_two';w=equal
                else:
                    w,_=fast_iu(fitted)
                    if w@C@np.ones(m)<0:w=-w
            aux=summaries@w;output.append(add_correction(base,aux));ws.append(w)
            full=np.zeros(len(g));full[live]=w
            info[head]=dict(weights=full.tolist(),alpha=alpha,fallback=reason)
        diagnostics[name]=dict(live=live.tolist(),covariance=C.tolist(),heads=info)
    return np.column_stack(output),diagnostics
