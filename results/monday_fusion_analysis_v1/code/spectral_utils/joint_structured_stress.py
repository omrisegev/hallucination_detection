"""Frozen label-free approximate-copy and temporally structured additions."""
import hashlib
import numpy as np


def standardize_block(x):
    x=np.asarray(x,float);sd=x.std(axis=0)
    return np.divide(x-x.mean(axis=0),sd,out=np.zeros_like(x),where=sd>1e-12)


def structured_augmentation(base,offsets,identities,kind):
    base=np.asarray(base,float);extra=np.zeros((len(base),15))
    if kind not in ('near_copies','structured_noise'):raise ValueError(kind)
    if len(offsets)!=len(identities)+1 or offsets[0]!=0 or offsets[-1]!=len(base):raise ValueError('ANSWER_ROSTER_MISMATCH')
    for uid,a,b in zip(identities,offsets[:-1],offsets[1:]):
        key=hashlib.sha256(f'{kind}:414170:{uid}'.encode('utf8')).digest()
        rng=np.random.default_rng(int.from_bytes(key[:8],'little'))
        if kind=='near_copies':
            noise=standardize_block(rng.normal(size=(b-a,15)))
            block=base[a:b,:15]+.05*noise
        else:
            random=rng.normal(size=(b-a,16));shared=random[:,0].copy()
            for i in range(1,len(shared)):shared[i]=.8*shared[i-1]+.6*random[i,0]
            shared=standardize_block(shared[:,None])
            private=standardize_block(random[:,1:]);block=.8*shared+.6*private
        extra[a:b]=standardize_block(block)
    return np.column_stack((base,extra))
