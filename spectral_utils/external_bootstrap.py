"""Shared paired source-question bootstrap; same draws for every locked arm."""
import numpy as np
from .external_generalization.evaluation import metric


def bootstrap_all(counts, groups, benchmark, draws=100000, seed=20260924):
    names=list(counts);x=np.stack([counts[n] for n in names],axis=1)
    if x.shape!=(len(groups),len(names),4):raise ValueError('unmatched counts')
    unique,index=np.unique(groups,return_inverse=True)
    if len(unique)<2:raise ValueError('need two source groups')
    grouped=np.zeros((len(unique),len(names),4));np.add.at(grouped,index,x)
    rng=np.random.default_rng(seed);samples=np.empty((draws,len(names)))
    for start in range(0,draws,64):
        stop=min(start+64,draws)
        ix=rng.integers(len(unique),size=(stop-start,len(unique)))
        samples[start:stop]=metric(grouped[ix].sum(axis=1),benchmark)
    return names,samples


def contrast_from_draws(counts,names,samples,left,right,groups,benchmark,family_size=18,seed=20260924):
    d=samples[:,names.index(left)]-samples[:,names.index(right)];d=d[np.isfinite(d)]
    alpha=.05/family_size
    return {'delta':float(metric(np.asarray(counts[left]).sum(0),benchmark)-metric(np.asarray(counts[right]).sum(0),benchmark)),
      'ci_bonferroni':np.quantile(d,[alpha/2,1-alpha/2]).tolist() if len(d) else None,
      'family_size':family_size,'paired_N':len(groups),'paired_groups':len(set(groups)),
      'draws':len(samples),'valid_draws':len(d),'seed':seed}
