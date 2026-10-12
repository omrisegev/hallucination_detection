"""Gate interventions on frozen fusion outputs; no fusion fitting."""
import numpy as np
from scipy.special import ndtr
from .historical_fusion_evaluation import calibrate, pb_metrics


def peak_statistics(risk, fitting_indices):
    risk=np.asarray(risk,float);indices=np.asarray(fitting_indices,int)
    if risk.ndim!=1 or not len(indices) or not np.isfinite(risk).all():
        raise ValueError('INVALID_FROZEN_WINDOW_CURVE')
    if np.any(indices<0) or np.any(indices>=len(risk)):raise ValueError('FIT_INDEX_OUTSIDE_CURVE')
    fit=risk[indices];scale=float(fit.std());maximum=float(risk.max())
    if not np.isfinite(scale) or scale<=1e-12:
        return maximum,np.nan,np.nan
    z=(maximum-float(fit.mean()))/scale
    return maximum,z,float(min(1.,len(indices)*ndtr(-z)))


def cross_gate(peaks,location_valid,donor_predictions,donor_valid):
    valid=np.asarray(location_valid,bool)&np.asarray(donor_valid,bool)
    predictions=np.where(np.asarray(donor_predictions)>=0,peaks,-1).astype(int)
    predictions[~valid]=-2
    return predictions,valid


def fit_calibrated_gate(detector,peaks,score_valid,target,cells,groups,outer):
    """Only the threshold uses labels; all fusion curves remain answer-only."""
    n=len(detector);prediction=np.full(n,-2,int);valid=np.zeros(n,bool);details={}
    pb=np.char.startswith(np.asarray(cells),'pb_');groups=np.asarray(groups)
    eligible=np.asarray(score_valid,bool)&np.isfinite(detector)
    for fold in range(5):
        train=pb&(outer!=fold);test=pb&(outer==fold)
        assert set(groups[train]).isdisjoint(groups[test])
        fitted=calibrate(detector[train],peaks[train],eligible[train],target[train],np.asarray(cells)[train])
        fitted.update(train_groups=sorted(set(groups[train])),test_groups=sorted(set(groups[test])),
                      training_rows=np.flatnonzero(train).tolist(),test_rows=np.flatnonzero(test).tolist())
        details[str(fold)]=fitted
        if fitted['status']!='CALIBRATED':continue
        valid[test]=eligible[test]
        prediction[test]=np.where(detector[test]>=fitted['threshold'],peaks[test],-1)
        prediction[test&~eligible]=-2
    return prediction,valid,details


def interval_summary(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    return dict(ci95=np.quantile(values,[.025,.975]).tolist() if len(values) else None,valid_draws=len(values))


def bootstrap_predictions(target,cells,groups,predictions,valid,draws=1000,seed=2026090707):
    cells=np.asarray(cells);group_names=sorted(set(groups));lookup={g:i for i,g in enumerate(group_names)}
    gi=np.array([lookup[g] for g in groups]);ng=len(group_names)
    weights=np.random.default_rng(seed).multinomial(ng,np.full(ng,1/ng),size=draws)
    cell_names=sorted(c for c in set(cells) if c.startswith('pb_'))
    denominators={}
    for cell in cell_names:
        for name,mask in [('clean',(cells==cell)&(target==-1)),('error',(cells==cell)&(target>=0))]:
            denominators[cell,name]=weights@np.bincount(gi[mask],minlength=ng)
    output={}
    for arm,pred in predictions.items():
        success=valid[arm]&(pred==target);cell_draws={}
        for cell in cell_names:
            fractions=[]
            for name,mask in [('clean',(cells==cell)&(target==-1)),('error',(cells==cell)&(target>=0))]:
                num=weights@np.bincount(gi[mask&success],minlength=ng);den=denominators[cell,name]
                fractions.append(np.divide(num,den,out=np.full(draws,np.nan),where=den!=0))
            c,e=fractions;cell_draws[cell]=np.divide(2*c*e,c+e,out=np.zeros(draws),where=c+e!=0)
        output[arm]={panel:np.mean([v for c,v in cell_draws.items() if panel=='all' or c.endswith(panel)],axis=0)
                     for panel in ('q4','q8','all')}
    return output
