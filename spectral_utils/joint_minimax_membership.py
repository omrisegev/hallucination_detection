"""Step411 staged membership with group rediscovery during refinement."""
import numpy as np
from .joint_sparse_membership import exact_aliases,alias_coordinates,source_grams
from .joint_staged_membership import learn_staged_structure,score_noise_aware_joint
from .joint_minimax_refinement import refine_minimax
from .lsml_gate_locator_research import _orient


def fit_minimax_joint(values,offsets,source_ids,inner_folds,*,anchor_index,seed=399170,notify=None):
    x=np.asarray(values,float);offsets=np.asarray(offsets,int);inner_folds=np.asarray(inner_folds)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('FINITE_MATRIX_REQUIRED')
    if offsets[0]!=0 or offsets[-1]!=len(x) or len(offsets)!=len(inner_folds)+1:
        raise ValueError('ANSWER_OFFSETS_MISMATCH')
    if len(source_ids)!=len(inner_folds) or np.any(np.diff(offsets)<1):raise ValueError('SOURCE_ROSTER_MISMATCH')
    if not 0<=anchor_index<x.shape[1]:raise ValueError('ANCHOR_OUT_OF_RANGE')
    aliases=exact_aliases(x);z=alias_coordinates(x,aliases)
    model=dict(valid=False,aliases=aliases,anchor_index=int(anchor_index),input_width=x.shape[1],seed=seed)
    try:
        gram,sizes,_=source_grams(z,offsets,source_ids,inner_folds)
        membership=learn_staged_structure(z,np.repeat(inner_folds,np.diff(offsets)),gram,int(sizes.sum()),
            seed=seed,calibration_seed=seed+5000,notify=notify)
        model['membership']=membership
        if not membership['valid']:
            model['failure']=membership['failure'];return model
        initial=np.asarray(membership['active']);part=np.asarray(membership['final_labels'])
        def trace(step,kept,retention):
            if notify and step%5==0:notify('REFINEMENT',removed=step,kept=kept,retention=retention)
        fit=refine_minimax(z[:,initial],part,np.repeat(inner_folds,np.diff(offsets)),
            seed=seed+200,group_seed=seed+100,notify=trace)
        w=np.zeros(z.shape[1]);w[initial]=fit['unoriented_weights']
        extended=np.column_stack((z,x[:,anchor_index]))
        w,orientation=_orient(extended,np.r_[w,0.],z.shape[1]);w=w[:-1]
        raw=np.zeros(x.shape[1])
        for j,ids in enumerate(aliases):raw[ids]=w[j]/len(ids)
        active=initial[np.asarray(fit['active'])]
        model.update(valid=True,refinement=fit,initial_active=initial,active=active,
            canonical_weights=w,expanded_weights=raw,orientation=orientation,
            active_original=[i for j in active for i in aliases[j]])
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        model['failure']=f'{type(exc).__name__}: {exc}'
    return model

