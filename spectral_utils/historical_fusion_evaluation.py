"""Metrics and nested calibration for the corrected historical comparator panel."""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata


def auc(y, scores):
    y=np.asarray(y,dtype=bool);scores=np.asarray(scores)
    p,n=int(y.sum()),int((~y).sum())
    if not p or not n:return None
    assert np.isfinite(scores).all()
    return float((rankdata(scores)[y].sum()-p*(p+1)/2)/(p*n))


def pb_metrics(target, prediction, valid, cells, weights=None):
    target=np.asarray(target);prediction=np.asarray(prediction);valid=np.asarray(valid,bool)
    cells=np.asarray(cells);weights=np.ones(len(target)) if weights is None else np.asarray(weights)
    results={}
    correct=valid & (prediction==target)
    for cell in sorted(set(cells)):
        if not cell.startswith('pb_'):continue
        mask=cells==cell;clean=mask & (target==-1);error=mask & (target>=0)
        nc,ne=float(weights[clean].sum()),float(weights[error].sum())
        ca=float(weights[clean & correct].sum()/nc) if nc else None
        ea=float(weights[error & correct].sum()/ne) if ne else None
        f1=None if ca is None or ea is None else (2*ca*ea/(ca+ea) if ca+ea else 0.)
        results[cell]=dict(answers=float(weights[mask].sum()),clean=nc,erroneous=ne,
            clean_accuracy=ca,error_exact_accuracy=ea,f1=f1,
            valid_decisions=float(weights[mask & valid].sum()))
    macros={}
    for panel in ('q4','q8','all'):
        values=[v['f1'] for cell,v in results.items() if panel=='all' or cell.endswith(panel)]
        macros[panel]=float(np.mean(values)) if values and None not in values else None
    return dict(cells=results,macros=macros)


def calibrate(detector, peak, valid, target, cells):
    """Labels allowed only from the current outer fold's inner held-out rows."""
    detector=np.asarray(detector);peak=np.asarray(peak);valid=np.asarray(valid,bool)
    eligible=valid & np.isfinite(detector)
    if not eligible.any():return dict(status='FAILED_NO_CALIBRATION_SCORES',threshold=None)
    grid=np.quantile(detector[eligible],np.linspace(.01,.99,99))
    objective=[]
    for threshold in grid:
        prediction=np.where(detector>=threshold,peak,-1)
        value=pb_metrics(target,prediction,eligible,cells)['macros']['all']
        objective.append(-np.inf if value is None else value)
    if not np.isfinite(objective).any():return dict(status='FAILED_CALIBRATION_CLASSES',threshold=None)
    selected=int(np.argmax(objective))
    return dict(status='CALIBRATED',threshold=float(grid[selected]),grid_index=selected,
                training_macro=float(objective[selected]),grid=grid.tolist(),
                calibration_rows=len(detector),valid_calibration_rows=int(eligible.sum()),
                labels_used=True)


def auc_plan(y,scores,groups):
    order=np.argsort(scores,kind='stable');s=scores[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    return np.asarray(y,bool)[order],np.asarray(groups)[order],starts


def weighted_auc(plan,weights):
    y,groups,starts=plan;w=weights[groups]
    pos=np.add.reduceat(w*y,starts);neg=np.add.reduceat(w*(~y),starts)
    p,n=pos.sum(),neg.sum()
    return float(np.dot(pos,np.cumsum(neg)-.5*neg)/(p*n)) if p and n else np.nan


def review_fixtures():
    from sklearn.metrics import roc_auc_score
    y=np.array([0,1,0,1,1,0]);s=np.array([.2,.2,.8,.6,.9,.1]);g=np.array([0,0,1,2,1,2])
    plan=auc_plan(y,s,g)
    for weights in (np.ones(3,int),np.array([0,1,2]),np.array([3,0,0]),np.array([2,1,0])):
        ix=np.repeat(np.arange(len(y)),weights[g])
        np.testing.assert_allclose(weighted_auc(plan,weights),roc_auc_score(y[ix],s[ix]),atol=1e-14)
    # Invalid clean prediction must stay a failure; step zero is a valid error.
    t=np.array([-1,-1,0,1]);p=np.array([-1,-1,0,-1]);v=np.array([1,0,1,1],bool)
    cells=np.array(['pb_fixture_q8']*4)
    metric=pb_metrics(t,p,v,cells)['cells']['pb_fixture_q8']
    assert metric['clean_accuracy']==.5 and metric['error_exact_accuracy']==.5 and metric['f1']==.5
    # Exact threshold search reviewed with an explicit small scalar loop.
    d=np.array([.1,.2,.8,.9]);loc=np.array([0,0,0,1]);v=np.ones(4,bool)
    result=calibrate(d,loc,v,t,cells)
    assert result['status']=='CALIBRATED' and result['training_macro']==1.
    assert .2 < result['threshold'] <= .8
    assert auc([0,0],[1,2]) is None and auc([0,1],[1,1])==.5
    return dict(status='PASS',checks=['weighted_AUC_vs_explicit_repeats','invalid_decision_accounting',
                                    'step_zero_error','threshold_search','ties_and_single_class'])
