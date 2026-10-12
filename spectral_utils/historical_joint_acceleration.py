"""Exact ARI arithmetic for small integer feature partitions.

The historical grouping loop repeatedly constructs sparse contingency matrices
for partitions of only 23 features. This helper uses dense integer counts and
the same pair-confusion formula as the installed sklearn implementation.
The private historical module binding is restored on leaving the context;
sklearn and the historical checkout are never modified.
"""
from contextlib import contextmanager
import itertools
import time
import numpy as np
from sklearn.metrics import adjusted_rand_score


def small_partition_ari(left,right):
    a=np.asarray(left);b=np.asarray(right)
    if (a.ndim!=1 or b.shape!=a.shape or len(a)>256 or a.dtype.kind not in 'iu'
        or b.dtype.kind not in 'iu'):
        return adjusted_rand_score(left,right)
    if len(a)==0:return 1.0
    amin,bmin=int(a.min()),int(b.min());amax,bmax=int(a.max()),int(b.max())
    if min(amin,bmin)<0 or max(amax,bmax)>=256:return adjusted_rand_score(left,right)
    a=a.astype(np.int64,copy=False);b=b.astype(np.int64,copy=False)
    c=np.bincount(a*(bmax+1)+b,minlength=(amax+1)*(bmax+1))
    rows=np.bincount(a);columns=np.bincount(b)
    squares=int(c@c);n=len(a)
    tp=squares-n;fp=int(columns@columns)-squares;fn=int(rows@rows)-squares
    tn=n*n-fp-fn-squares
    if fn==0 and fp==0:return 1.0
    return 2.0*(tp*tn-fn*fp)/((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn))


@contextmanager
def compatible_ari(reference):
    module=reference['joint_lsml'];previous=module.adjusted_rand_score
    assert previous is adjusted_rand_score,'Unexpected preexisting ARI override'
    audit=dict(calls=0,mode='dense_integer_pair_confusion; exact sklearn formula')
    def counted(a,b):
        audit['calls']+=1
        return small_partition_ari(a,b)
    module.adjusted_rand_score=counted
    try:yield audit
    finally:module.adjusted_rand_score=previous


def preflight():
    cases=[]
    for n in range(6):
        labels=[np.array(x,dtype=np.int64) for x in itertools.product(range(2),repeat=n)]
        cases.extend((a,b) for a in labels for b in labels)
    rng=np.random.default_rng(20260907329)
    for _ in range(400):
        cases.append((rng.integers(0,int(rng.integers(1,24)),23),rng.integers(0,int(rng.integers(1,24)),23)))
    cases.extend([(np.array([-1,0,1]),np.array([1,0,-1])),(np.array(['a','b']),np.array(['c','d'])),
                  (np.arange(300),np.arange(300)),(np.array([0,300]),np.array([300,0]))])
    started=time.perf_counter();expected=[adjusted_rand_score(a,b) for a,b in cases]
    reference_seconds=time.perf_counter()-started
    started=time.perf_counter();actual=[small_partition_ari(a,b) for a,b in cases]
    fast_seconds=time.perf_counter()-started
    assert expected==actual
    return dict(status='PASS',cases=len(cases),exact_float_equality=True,reference_seconds=reference_seconds,
                fast_seconds=fast_seconds,scope='small-partition ARI arithmetic only; full fit replay separate')
