"""Label-free deterministic perturbations of a step-feature bank."""
import hashlib
import numpy as np


def answer_noise(offsets, identities, width=15, seed=401170):
    offsets=np.asarray(offsets,int)
    if len(offsets)!=len(identities)+1 or np.any(np.diff(offsets)<1):
        raise ValueError('answer identity/offset mismatch')
    output=np.zeros((int(offsets[-1]),width))
    for uid,a,b in zip(identities,offsets[:-1],offsets[1:]):
        key=hashlib.sha256(f'{seed}:{uid}'.encode('utf8')).digest()
        rng=np.random.default_rng(int.from_bytes(key[:8],'little'))
        raw=rng.normal(size=(b-a,width));sd=raw.std(axis=0)
        output[a:b]=np.divide(raw-raw.mean(axis=0),sd,out=np.zeros_like(raw),where=sd>1e-12)
    return output


def augment(base, offsets, identities, kind):
    x=np.asarray(base,float)
    if kind=='duplicates':extra=x[:,:15].copy()
    elif kind=='noise':extra=answer_noise(offsets,identities)
    else:raise ValueError(kind)
    return np.column_stack((x,extra))
