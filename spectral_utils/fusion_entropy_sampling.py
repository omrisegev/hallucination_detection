"""Two entropy-coverage selectors serving the existing fixed-bank fusion."""
from copy import deepcopy
import numpy as np
from .fusion_prediction_quality import CORES, JOINT_CORES, fit_bank
from .fusion_token_gap import apply_readout, chosen_core
from .fusion_window_sampling import budget

SELECTORS=('entropy_tails','entropy_quantiles')
NEW_ARMS=tuple(f'sample_{s}__{c}' for s in SELECTORS for c in CORES)


def choose(entropy,selector):
    x=np.asarray(entropy,float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('INVALID_ENTROPY_SELECTION_INPUT')
    if selector not in SELECTORS:raise ValueError('UNKNOWN_SELECTOR')
    n=len(x);m=budget(n)
    if m==n:return np.arange(n,dtype=int)
    time=np.arange(n);ascending=np.lexsort((time,x))
    if selector=='entropy_tails':
        low=ascending[:m//2];descending=np.lexsort((time,-x))
        high=descending[~np.isin(descending,low)][:m-len(low)]
        selected=np.r_[low,high]
    else:
        ranks=((2*np.arange(m)+1)*n)//(2*m)
        selected=ascending[ranks]
    selected=np.sort(selected)
    if len(selected)!=m or np.any(np.diff(selected)<=0):raise ValueError('INVALID_SELECTION_BUDGET_OR_DUPLICATES')
    return selected


def score_selected(values,names,plan,ss,ee,selected,identity,reference):
    """Same fitting and dense readout as the registered high-only comparator."""
    selected=np.asarray(selected,int)
    if not len(selected) or np.any(np.diff(selected)<=0) or not np.isin(selected,plan.fit_indices).all():
        raise ValueError('INVALID_SELECTED_FITTING_ROWS')
    try:
        fitted,risks,details,shared=fit_bank(values,names,selected,identity)
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        fitted={};risks={};details={c:dict(valid=False,reason=str(exc)) for c in CORES}
        shared=dict(joint_valid=False,preparation_failure=str(exc),joint_failure=str(exc))
    out=dict(fitted);native={};methods={}
    for core in CORES:
        steps,d=apply_readout(risks.get(core),details[core],plan,ss,ee,reference)
        native[core]=d
        if d['valid']:
            out['native_'+core+'__window']=risks[core];out['native_'+core+'__risk']=steps
    for core in CORES:
        source=chosen_core(core,shared['joint_valid']);d=deepcopy(native[source])
        d.update(source_core=source,joint_fit_valid=shared['joint_valid'],fallback_to_sample_iu=source!=core,
                 route='fixed_original_bank')
        if source!=core:d['joint_failure']=shared.get('joint_failure')
        methods[core]=d
        if d['valid']:
            for suffix in ('window','risk'):out[core+'__'+suffix]=out['native_'+source+'__'+suffix].copy()
    return out,methods,dict(shared=shared,native=native,joint_valid=shared['joint_valid'])
