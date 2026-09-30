"""Reparameterize provided-token surprisal inside existing IU/Joint fusion."""
from copy import deepcopy
import numpy as np
from .answer_localization_v2 import moment_plan,moment_matrix,mixture_readout
from .fusion_context_bank import context_matrix
from .fusion_prediction_quality import fit_bank,fixed_bank,CORES,JOINT_CORES
from .window_localization import windows_to_tokens

NEW_ARMS=tuple('gap__'+core for core in CORES)+('gap_scalar','surprisal_scalar')
ANCHORS=dict(equal='dual__equal',iu='dual__iu',joint0='dual__cond100',graph010='dual__cond100_graph010',
    graph_perm='dual__cond100_graph_perm',equal_graph010='dual__equal_graph010',equal_graph_perm='dual__equal_graph_perm')


def provided_token_gap(raw):
    """Only for same-distribution unwarped teacher-forced raw29 inputs."""
    raw=np.asarray(raw,float)
    if raw.ndim!=2 or raw.shape[1]!=29 or not len(raw):raise ValueError('RAW29_REQUIRED')
    supplied,top1=raw[:,15],raw[:,23]
    if not np.isfinite(supplied).all() or not np.isfinite(top1).all():raise ValueError('NONFINITE_TOKEN_CONFIDENCE')
    if np.any(supplied<0) or np.any(top1>0):raise ValueError('INVALID_LOG_PROBABILITY_SIGN')
    gap=supplied+top1
    tolerance=1e-6*np.maximum(1.,np.maximum(supplied,np.abs(top1)))
    if np.any(gap < -tolerance):raise ValueError('TOKEN_GAP_NEGATIVE_DISTRIBUTION_MISMATCH')
    return np.maximum(gap,0.)


def gap_matrix(raw,bank):
    if bank not in ('moment','context'):raise ValueError('UNKNOWN_BANK')
    plan=moment_plan(len(raw),8);gap=provided_token_gap(raw);changed=np.array(raw,float,copy=True)
    changed[:,15]=gap
    builder=moment_matrix if bank=='moment' else context_matrix
    original,original_names=builder(raw,plan);values,names=builder(changed,plan)
    names=[n.replace('spilled_series__','provided_token_gap__') for n in names]
    assert original.shape==values.shape and values.shape[1]==27
    keep=[j for j in range(27) if j not in (3,4,5)]
    np.testing.assert_array_equal(values[:,keep],original[:,keep])
    return plan,values,names,original,original_names,gap


def chosen_core(core,joint_valid):
    """Model-fit eligibility alone controls the explicitly declared fallback."""
    return 'iu' if core in JOINT_CORES and not joint_valid else core


def apply_readout(risk,detail,plan,ss,ee,reference):
    out=deepcopy(detail);out.update(decision_valid=False,fixed_iu_valid=False,prediction=None,fixed_iu_prediction=None)
    if not out['valid']:return None,out
    token=windows_to_tokens(plan,risk);step=np.array([token[a:b].max() for a,b in zip(ss,ee)])
    if not np.isfinite(step).all():raise ValueError('NONFINITE_STEP_RISK')
    out['peak']=int(np.argmax(step))
    try:
        gate=mixture_readout(risk[plan.fit_indices],step)
        out.update(gate=gate,decision_valid=True,prediction=out['peak'] if gate['prediction']!=-1 else -1)
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:out['readout_error']=str(exc)
    out['fixed_iu_valid']=bool(reference['valid'] and reference['decision_valid'])
    if out['fixed_iu_valid']:out['fixed_iu_prediction']=out['peak'] if reference['prediction']!=-1 else -1
    return step,out


def score_gap(raw,ss,ee,original_arrays,original_meta,identity):
    bank=fixed_bank(original_meta['routing']);plan,values,names,base,base_names,gap=gap_matrix(raw,bank)
    np.testing.assert_array_equal(base,original_arrays[bank+'__features'])
    ss,ee=np.asarray(ss,int),np.asarray(ee,int)
    if ss.shape!=ee.shape or len(ss)==0 or np.any(ss<0) or np.any(ee>len(raw)) or np.any(ee<=ss):raise ValueError('INVALID_STEP_SPANS')
    out={'step_starts':ss,'step_ends':ee,'window_starts':plan.starts,'window_ends':plan.ends,'fit_indices':plan.fit_indices,
         'gap__features':values,'provided_token_gap':gap}
    reference=original_meta['methods']['moment__iu'];methods={}
    try:
        arrays,risks,details,shared=fit_bank(values,names,plan.fit_indices,identity)
        out.update({'gap__'+k:v for k,v in arrays.items()})
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        risks={};details={c:{'valid':False,'reason':str(exc)} for c in CORES}
        shared={'joint_valid':False,'preparation_failure':str(exc),'joint_failure':str(exc)}
    native={}
    for core in CORES:
        step,detail=apply_readout(risks.get(core),details[core],plan,ss,ee,reference);native[core]=detail
        if detail['valid']:
            out['native__'+core+'__window']=risks[core];out['native__'+core+'__risk']=step
    for core in CORES:
        source=chosen_core(core,shared['joint_valid']);arm='gap__'+core;detail=deepcopy(native[source])
        detail.update(source_arm='native__'+source,bank=bank,joint_fit_valid=shared['joint_valid'],
            fallback_to_gap_iu=source!=core,route='fixed_original_bank')
        if source!=core:detail['joint_failure']=shared.get('joint_failure')
        methods[arm]=detail
        if detail['valid']:
            out[arm+'__window']=out['native__'+source+'__window'].copy();out[arm+'__risk']=out['native__'+source+'__risk'].copy()
    for arm,stream in [('gap_scalar',gap),('surprisal_scalar',raw[:,15])]:
        means=np.array([stream[a:b].mean() for a,b in zip(plan.starts,plan.ends)]);fit=means[plan.fit_indices]
        sd=float(fit.std());mean=float(fit.mean())
        if not np.isfinite(sd) or sd<=1e-10:
            methods[arm]={'valid':False,'decision_valid':False,'fixed_iu_valid':False,'reason':'CONSTANT_SCALAR'};continue
        risk=(means-mean)/sd;step,detail=apply_readout(risk,{'valid':True,'fit_mean':mean,'fit_sd':sd,'orientation':'larger_declared_risk'},plan,ss,ee,reference)
        detail.update(source_arm=arm,bank='scalar',fallback_to_gap_iu=False);methods[arm]=detail
        out[arm+'__window']=risk;out[arm+'__risk']=step
    diagnostics={'bank':bank,'names':names,'original_names':base_names,'shared':shared,'labels_used':False,
        'gap_min':float(gap.min()),'gap_max':float(gap.max()),'near_zero_fraction':float(np.mean(gap<=1e-5))}
    assert set(methods)==set(NEW_ARMS)
    return out,methods,diagnostics
