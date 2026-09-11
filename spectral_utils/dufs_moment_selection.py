"""Answer-local selection of six of twelve moment columns before fixed-D RBM.

Adapted DUFS uses the repository's frozen 120-epoch, three-seed recipe.
The alternative greedily minimizes squared Pearson redundancy, without labels.
Neither selector supplies a hallucination label or changes the risk anchor.
"""
import time
import numpy as np
from scipy.special import expit
from .adapted_dufs import adapted_dufs_soft_gates
from .higher_moment_fusion import representation_order, feature_names
from .direct_probability_fusion import zscore_columns, _orient
from .moment_rbm_fusion import fit_rbm

BANKS = ('all12', 'original6', 'dufs6', 'correlation6')
METHODS = tuple(b+'__'+s for b in BANKS for s in ('rbm_initial', 'rbm'))


def top_six(probabilities):
    probabilities = np.asarray(probabilities, float)
    if probabilities.ndim != 1 or len(probabilities) < 6 or not np.isfinite(probabilities).all():
        raise ValueError('need at least six finite gate probabilities')
    # Stable descending sort; ties go to the earliest original column.
    return np.sort(np.argsort(-probabilities, kind='stable')[:6])


def correlation_six(Z):
    """Start at least redundant column, then minimize summed squared correlation.

    Initial cost is sum of squared correlations with all other columns.
    Later cost is sum against already chosen columns. Original-index tie break.
    Return original-order columns so model initialization is comparable.
    """
    Z = np.asarray(Z, float)
    if Z.ndim != 2 or min(Z.shape) < 6 or not np.isfinite(Z).all():
        raise ValueError('need six rows and six varying columns')
    corr = np.corrcoef(Z, rowvar=False)
    if not np.isfinite(corr).all():
        raise ValueError('nonfinite correlation')
    cost = corr**2
    np.fill_diagonal(cost, 0.)
    selected = [int(np.argmin(cost.sum(axis=0)))]
    while len(selected) < 6:
        candidates = [i for i in range(len(corr)) if i not in selected]
        selected.append(min(candidates, key=lambda i: (float(cost[i, selected].sum()), i)))
    return np.sort(selected)


def fit_all(logprobs, chosen, entropy=None):
    X = representation_order(logprobs, chosen, 6)
    Z, keep, mean, scale = zscore_columns(X)
    z0, _, _, _ = zscore_columns(X[:, :6])
    if len(X) < 6 or z0.shape[1] < 3 or Z.shape[1] < 6:
        return {}, {m:'ValueError: insufficient rows or varying columns' for m in METHODS}, {m:0. for m in METHODS}
    anchor = z0.mean(axis=1)
    active = np.flatnonzero(keep)
    banks = {'all12':active, 'original6':active[active<6]}
    selection = {}; selection_seconds = {}; selection_failures = {}
    for bank in ('dufs6', 'correlation6'):
        started=time.perf_counter()
        try:
            if bank=='dufs6':
                _, d = adapted_dufs_soft_gates(Z.T, seeds=(0,1,2), epochs=120)
                local=top_six(d['raw_probabilities'])
                seed_sets=[set(top_six(p)) for p in d['per_seed_probabilities']]
                d['seed_top6_jaccard']=[len(seed_sets[i]&seed_sets[j])/len(seed_sets[i]|seed_sets[j]) for i,j in ((0,1),(0,2),(1,2))]
                selection[bank]={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in d.items()}
            else:
                local=correlation_six(Z)
            banks[bank]=active[local]
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            selection_failures[bank]=f'{type(error).__name__}: {error}'
        selection_seconds[bank]=time.perf_counter()-started
    fits, failures, seconds = {}, {}, {}
    for m in METHODS:
        bank, solver=m.split('__'); started=time.perf_counter()
        if bank not in banks:
            failures[m]=selection_failures[bank];seconds[m]=0.;continue
        try:
            cols=banks[bank]
            # Reuse exact historic z-score arrays for both reference banks.
            if bank=='original6':normalized=z0
            else:normalized=Z[:,np.searchsorted(active,cols)]
            p=normalized.shape[1]
            if p<3:raise ValueError('fewer than three varying features')
            if solver=='rbm':
                score,state,d=fit_rbm(normalized,maxiter=100)
            else:
                state=dict(a=np.zeros(p),w=np.full(p,2./p),b=np.asarray(0.))
                score=expit(normalized@state['w']);d=dict(trained=False)
            score,flipped,corr=_orient(score,anchor)
            if flipped:score=1+score
            if not np.isfinite(score).all():raise ValueError('nonfinite posterior')
            d.update(columns=cols.tolist(),feature_names=feature_names(6),active_columns=p,
                     normalization_mean=mean.tolist(),normalization_scale=scale.tolist(),
                     orientation=-1 if flipped else 1,anchor_correlation=float(corr) if corr is not None and np.isfinite(corr) else None,
                     score_sd=float(score.std()),collapsed=bool(score.std()<1e-3),
                     selection_seconds=selection_seconds.get(bank,0.),selection=selection.get(bank,{}))
            weights=np.zeros(12);weights[cols]=state['w']*d['orientation']
            fits[m]=dict(score=score,state=state,weights=weights,diagnostics=d)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError,AssertionError) as error:
            failures[m]=f'{type(error).__name__}: {error}'
        seconds[m]=time.perf_counter()-started
        # Charge selection once per bank, not once per solver.
        if solver=='rbm_initial':seconds[m]+=selection_seconds.get(bank,0.)
    return fits,failures,seconds
