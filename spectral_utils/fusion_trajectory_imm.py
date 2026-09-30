"""Chronological interpretation of existing feature-fusion trajectories.

Pointwise mean/GLS are weight-combination controls, not temporal learners.
The IMM uses the scalar sufficient statistic of a correlated observation model.
"""
from copy import deepcopy
import hashlib
import numpy as np
from .answer_localization_v2 import moment_plan
from .fused_trajectory_readouts import imm_filter, noise_variance
from .fusion_token_gap import apply_readout

SOURCES=dict(iu='dual__iu',joint_graph='dual__cond100_graph010',joint0='dual__cond100',
    joint_perm='dual__cond100_graph_perm',equal='dual__equal',equal_graph='dual__equal_graph010',equal_perm='dual__equal_graph_perm')
PAIRS=dict(iu_joint_graph=('iu','joint_graph'),iu_joint0=('iu','joint0'),iu_joint_perm=('iu','joint_perm'),
           equal_graph=('equal','equal_graph'),equal_perm=('equal','equal_perm'))
SINGLES=('iu','joint_graph','equal')
READOUTS=('mean','gls','hold','imm')
NEW_ARMS=tuple(f'traj_{family}__{readout}' for family in PAIRS for readout in READOUTS)+tuple(
    f'traj_{family}__{readout}' for family in SINGLES for readout in ('hold','imm'))+('traj_iu_joint_graph__imm_permuted',)


def normalize(values,indices):
    x=np.asarray(values,float);fit=x[indices];mean=float(fit.mean());sd=float(fit.std())
    if not np.isfinite(x).all() or not np.isfinite(sd) or sd<=1e-10:raise ValueError('DEGENERATE_TRAJECTORY')
    return (x-mean)/sd,dict(mean=mean,sd=sd)


def hold_curve(values,fit_indices,window_count):
    x=np.asarray(values,float);fi=np.asarray(fit_indices,int)
    if x.shape!=(len(fi),) or not np.isfinite(x).all() or not np.array_equal(fi,np.arange(len(fi))):
        raise ValueError('INVALID_CHRONOLOGICAL_GRID')
    if window_count not in (len(fi),len(fi)+1):raise ValueError('INVALID_TAIL_GRID')
    out=np.full(window_count,x[-1]);out[fi]=x;return out


def observation_model(values,fit_indices):
    y=np.asarray(values,float);fi=np.asarray(fit_indices,int)
    if y.ndim!=2 or y.shape[1] not in (1,2) or not np.isfinite(y).all() or len(fi)<2:raise ValueError('INVALID_SOURCE_TRAJECTORIES')
    fit=y[fi]
    if np.any(np.abs(fit.mean(0))>1e-8) or np.any(np.abs(fit.std(0)-1)>1e-8):raise ValueError('SOURCE_NOT_STANDARDIZED')
    retained=[0];correlation=None
    if y.shape[1]==2:
        correlation=float(np.corrcoef(fit,rowvar=False)[0,1])
        if correlation<=-1+1e-10:raise ValueError('CONFLICTING_SOURCE_ORIENTATION')
        if correlation<1-1e-10:retained.append(1)
    selected=y[:,retained];z=selected[fi];p=len(retained)
    diagonal=np.array([noise_variance(z[:,j]) for j in range(p)])
    corr=np.eye(p);constant_difference=False
    if p==2:
        delta=np.diff(z,axis=0)
        if np.any(delta.std(0)<=1e-12):rho=0.;constant_difference=True
        else:rho=float(np.clip(np.corrcoef(delta,rowvar=False)[0,1],-1,1))
        corr[0,1]=corr[1,0]=rho
    covariance=corr*np.sqrt(np.outer(diagonal,diagonal));lo,hi=np.linalg.eigvalsh(covariance)[[0,-1]]
    ridge=max(0.,float((hi-100*lo)/99),float(hi*1e-10));regularized=covariance+ridge*np.eye(p)
    ones=np.ones(p);inverse_ones=np.linalg.solve(regularized,ones);den=float(ones@inverse_ones)
    if not np.isfinite(den) or den<=0:raise ValueError('INVALID_EFFECTIVE_NOISE')
    weights=inverse_ones/den;raw_gls=selected@weights;gls,scale=normalize(raw_gls,fi)
    mean,mean_scale=normalize(selected.mean(1),fi);variance=1/den/scale['sd']**2
    detail=dict(retained=retained,source_correlation=correlation,noise_diagonal=diagonal,
        difference_correlation=corr,constant_difference=constant_difference,noise_covariance=covariance,
        regularized_noise=regularized,ridge=ridge,condition=float(np.linalg.cond(regularized)),weights=weights,
        gls_scale=scale,mean_scale=mean_scale,effective_variance=variance)
    return gls,mean,detail


def chronology(gls,variance,fit_indices,identity,permuted=False):
    x=np.asarray(gls)[fit_indices];order=np.arange(len(x))
    if permuted:
        seed=int(hashlib.sha256((identity+'/imm-time-permutation').encode()).hexdigest()[:8],16)
        order=np.random.default_rng(seed).permutation(len(x))
    result=imm_filter(x[order],variance,[.01*variance,variance])
    unpermuted={}
    for name,values in result.items():
        arr=np.empty_like(values);arr[order]=values;unpermuted[name]=arr
    normalized,scale=normalize(unpermuted['level'],np.arange(len(x)))
    return normalized,unpermuted,dict(level_scale=scale,permutation=order,process_variances=[.01*variance,variance])


def score_trajectories(token_count,ss,ee,source_arrays,source_meta,original_meta,identity):
    plan=moment_plan(token_count,8);fi=plan.fit_indices
    arrays=dict(window_starts=plan.starts,window_ends=plan.ends,fit_indices=fi,step_starts=np.asarray(ss,int),step_ends=np.asarray(ee,int))
    diagnostics=dict(families={},labels_used=False,original_routing=original_meta['routing'])
    methods={};reference=original_meta['methods']['moment__iu']
    families={**PAIRS,**{name:(name,) for name in SINGLES}}
    for family,components in families.items():
        expected=list(READOUTS if family in PAIRS else ('hold','imm'))
        if family=='iu_joint_graph':expected.append('imm_permuted')
        detail=dict(sources=[SOURCES[k] for k in components],inherited_parent_fallback=bool(
            original_meta['routing']['routes']['dual']=='moment_iu' and any(k.startswith('joint') for k in components)))
        try:
            if not all(source_meta['methods'][SOURCES[k]]['valid'] for k in components):raise ValueError('REQUIRED_SOURCE_UNAVAILABLE')
            y=np.column_stack([source_arrays[SOURCES[k]+'__window'] for k in components])
            if len(y)!=len(plan.starts):raise ValueError('SOURCE_GRID_MISMATCH')
            gls,mean,model=observation_model(y,fi);detail.update(model)
            arrays[family+'__components']=y;arrays[family+'__gls']=gls
            curves=dict(mean=mean,gls=gls,hold=hold_curve(gls[fi] if family in PAIRS else y[fi,0],fi,len(y)))
            for mode in ('imm','imm_permuted'):
                if mode not in expected:continue
                try:
                    level,state,info=chronology(gls,model['effective_variance'],fi,identity,mode=='imm_permuted')
                    detail[mode]=info
                    for name,values in state.items():arrays[family+'__'+mode+'_'+name]=values
                    curves[mode]=hold_curve(level,fi,len(y))
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:detail[mode+'_failure']=str(exc)
            detail['valid']=True
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            curves={};detail.update(valid=False,reason=str(exc))
        diagnostics['families'][family]=detail
        for readout in expected:
            arm=f'traj_{family}__{readout}'
            base=dict(valid=readout in curves,family=family,source_arm='+'.join(detail['sources']),readout=readout,
                      inherited_parent_fallback=detail['inherited_parent_fallback'])
            if not base['valid']:base['reason']=detail.get(readout+'_failure',detail.get('reason','READOUT_UNAVAILABLE'))
            step,info=apply_readout(curves.get(readout),base,plan,ss,ee,reference);methods[arm]=info
            if info['valid']:arrays[arm+'__window']=curves[readout];arrays[arm+'__risk']=step
    assert set(methods)==set(NEW_ARMS)
    return arrays,methods,diagnostics
