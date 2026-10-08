"""Context-varying weights on fixed original-feature token evidence; no labels."""
import numpy as np

HEADS=('native','simplex','group')
POLICIES=('static','energy','position','random','amplitude','direction')
METHODS=tuple(h+'__'+p for h in HEADS for p in POLICIES)+('equal',)

def raw_head(head, values, sd, static_values):
    values=np.asarray(values,float);sd=np.asarray(sd,float)
    if head=='native':
        norm=np.abs(np.asarray(static_values)/sd).sum()
        if norm<1e-12:raise ValueError('Unidentifiable static native head')
        return values/sd/norm
    return values.copy()

def decompose(weights,static,sd):
    static=np.asarray(static,float);weights=np.asarray(weights,float);sd=np.asarray(sd,float)
    denom=np.linalg.norm(static*sd)
    if denom<1e-12:raise ValueError('Zero static amplitude')
    A=np.linalg.norm(weights*sd,axis=1)/denom
    if np.any(A<1e-12):raise ValueError('Zero context amplitude')
    return A[:,None]*static,weights/A[:,None],A

def held_weights(n,positions,weights,static):
    positions=np.asarray(positions,int);weights=np.asarray(weights,float);static=np.asarray(static,float)
    if positions.ndim!=1 or len(positions)!=len(weights) or np.any(np.diff(positions)<=0) or np.any(positions<16) or np.any(positions>=n):
        raise ValueError('Invalid chronological weight anchors')
    idx=np.searchsorted(positions,np.arange(n),side='right')-1
    extended=np.vstack([static,weights]);return extended[idx+1]

def selected_tokens(x,spans):
    x=np.asarray(x,float);out=[]
    for start,stop in np.asarray(spans,int):
        if not 0<=start<stop<=len(x):raise ValueError('Invalid span')
        ids=np.argsort(x[start:stop],axis=0,kind='stable')[-min(10,stop-start):]+start
        out.append(ids)
    return out

def readout(x,selected,weights):
    x=np.asarray(x,float);weights=np.asarray(weights,float)
    if weights.shape==x.shape[1:]:weights=np.broadcast_to(weights,x.shape)
    if x.shape!=weights.shape or not np.isfinite(weights).all():raise ValueError('Invalid weight curve')
    j=np.arange(x.shape[1])
    return np.array([(x[ids,j]*weights[ids,j]).mean(axis=0).sum() for ids in selected])

def score_answer(x,spans,positions,landmark_heads,static_heads,sd):
    selected=selected_tokens(x,spans);out={'equal':readout(x,selected,np.ones(x.shape[1])/x.shape[1])}
    for head in HEADS:
        static=raw_head(head,static_heads[head],sd,static_heads[head])
        out[head+'__static']=readout(x,selected,static)
        for arm in ('energy','position','random'):
            raw=raw_head(head,landmark_heads[head][arm],sd,static_heads[head])
            out[head+'__'+arm]=readout(x,selected,held_weights(len(x),positions,raw,static))
            if arm=='energy':
                amplitude,direction,_=decompose(raw,static,sd)
                for policy,v in [('amplitude',amplitude),('direction',direction)]:
                    out[head+'__'+policy]=readout(x,selected,held_weights(len(x),positions,v,static))
    return out
