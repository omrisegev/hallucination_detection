"""Exact source-group bootstrap with cached pair counts, not repeated AUC fits.

The resampling order matches run_answer_localization_v2.paired_intervals:
each draw samples PRMB groups, then PB groups, in original cell insertion order.
Metrics retain repeated source-group multiplicities and undefined draws.
"""
from __future__ import annotations

import numpy as np


def group_strata(rows):
    strata={}
    for i,row in enumerate(rows):strata.setdefault(row['cell'],{}).setdefault(row['group_id'],[]).append(i)
    return [list(group[k] for k in sorted(group)) for group in strata.values()]


def sample_multiplicity(strata,n,rng):
    weights=np.zeros(n,dtype=float)
    for groups in strata:
        sampled=rng.integers(len(groups),size=len(groups))
        counts=np.bincount(sampled,minlength=len(groups))
        for indices,count in zip(groups,counts):weights[indices]=count
    return weights


def auc_counts(rows,arm):
    n=len(rows);wins=np.zeros((n,n));positive=np.zeros(n);negative=np.zeros(n);within=np.zeros(n);mixed=np.zeros(n)
    for i,row in enumerate(rows):
        y=np.asarray(row['target']);x=np.asarray(row['scores'][arm],float)
        pos=x[y==1];positive[i]=len(pos);negative[i]=np.sum(y==0)
        for j,other in enumerate(rows):
            neg=np.asarray(other['scores'][arm],float)[np.asarray(other['target'])==0]
            wins[i,j]=np.sum(pos[:,None]>neg)+.5*np.sum(pos[:,None]==neg)
        if positive[i]*negative[i]:within[i]=wins[i,i]/(positive[i]*negative[i]);mixed[i]=1
    return wins,positive,negative,within,mixed


def auc_from_counts(counts,multiplicity):
    wins,positive,negative,within,mixed=counts;m=multiplicity
    denominator=(m@positive)*(m@negative);within_den=m@mixed
    return ((float(m@wins@m)/denominator if denominator else None),
            (float(m@within)/within_den if within_den else None))


def pb_counts(rows,arm,fixed=False):
    result=[]
    valid_key='fixed_iu_valid' if fixed else 'decision_valid'
    prediction_key='fixed_iu_predictions' if fixed else 'predictions'
    for cell in dict.fromkeys(r['cell'] for r in rows):
        clean=np.asarray([int(r['cell']==cell and r['target']==-1) for r in rows])
        error=np.asarray([int(r['cell']==cell and r['target']!=-1) for r in rows])
        hits=np.asarray([bool(r[valid_key].get(arm,False) and r[prediction_key].get(arm)==r['target']) for r in rows])
        result.append((clean,error,clean*hits,error*hits))
    return result


def pb_from_counts(counts,m):
    f1=[]
    for clean,error,clean_hit,error_hit in counts:
        nc,ne=m@clean,m@error
        if not nc or not ne:return None
        ca,ea=(m@clean_hit)/nc,(m@error_hit)/ne
        f1.append(2*ca*ea/(ca+ea) if ca+ea else 0.)
    return float(np.mean(f1)) if f1 else None


def paired_source_group_intervals(rows,left,right,draws=1000,seed=2026090706):
    prm=[r for r in rows if r['cell'].startswith('prm') and r['valid'].get(left) and r['valid'].get(right)]
    pb=[r for r in rows if r['cell'].startswith('pb_')]
    ps,bs=group_strata(prm),group_strata(pb)
    pc=[auc_counts(prm,a) for a in (left,right)]
    bc=[pb_counts(pb,a) for a in (left,right)]
    fc=[pb_counts(pb,a,True) for a in (left,right)]
    values={k:[] for k in ('prm','within','pb','fixed_iu')};rng=np.random.default_rng(seed)
    for _ in range(draws):
        if prm:
            m=sample_multiplicity(ps,len(prm),rng);a,b=[auc_from_counts(c,m) for c in pc]
            for j,key in enumerate(('prm','within')):
                if a[j] is not None and b[j] is not None:values[key].append(a[j]-b[j])
        m=sample_multiplicity(bs,len(pb),rng)
        for collection,key in ((bc,'pb'),(fc,'fixed_iu')):
            a,b=[pb_from_counts(c,m) for c in collection]
            if a is not None and b is not None:values[key].append(a-b)
    output={'unit':'source group, stratified by cell','seed':seed,'draws':draws,
        'implementation':'cached_pair_counts_exact_parent_draw_order'}
    names={'prm':'prm_common_valid','within':'prm_within_answer_common_valid','pb':'pb_all_population','fixed_iu':'pb_common_iu_gate_all_population'}
    for key,series in values.items():
        output[names[key]+'_ci95']=np.quantile(series,[.025,.975]).tolist() if series else None
        output[names[key]+'_valid_draws']=len(series)
    return output
