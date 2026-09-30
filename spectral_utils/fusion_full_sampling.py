"""Full-population bridge for all eight previously tested fitting-row selectors.

No target input. Same raw telemetry, fixed bank route, fitting recipes, dense
readout and original-window GMM support as the reviewed sampling experiments.
"""
from copy import deepcopy
import numpy as np
from .answer_localization_v2 import moment_plan
from .fusion_sampling_replication import score_sampling, selection_support
from .fusion_window_sampling import SELECTORS as OLD_SELECTORS, budget
from .fusion_entropy_sampling import SELECTORS as ENTROPY_SELECTORS, choose, score_selected
from .fusion_prediction_quality import CORES
from .fusion_full_shortlist import score_full_shortlist

SELECTORS=OLD_SELECTORS+ENTROPY_SELECTORS
ARMS=tuple(f'sample_{selector}__{core}' for selector in SELECTORS for core in CORES)


def score_full_sampling(raw,original_arrays,original_meta,identity,anchor_arrays=None,anchor_meta=None):
    token_count=len(raw)
    if token_count!=original_meta['tokens']:raise ValueError('TOKEN_COUNT_MISMATCH')
    if token_count<64:
        invalid=dict(valid=False,decision_valid=False,fixed_iu_valid=False,prediction=None,
            fixed_iu_prediction=None,reason='TOO_FEW_FIT_WINDOWS',joint_fit_valid=False)
        return dict(step_starts=original_arrays['step_starts'],step_ends=original_arrays['step_ends']),{
            arm:deepcopy(invalid) for arm in ARMS},dict(eligible=False,unsupported=True,labels_used=False)
    if anchor_arrays is None:
        additional,methods,detail=score_full_shortlist(original_arrays,original_meta,identity)
        anchor_arrays={**original_arrays,**additional}
        anchor_meta=dict(methods={**original_meta['methods'],**methods})
        anchor_mode='exact_pass2a_kernel_replay'
    else:
        if anchor_meta is None:raise ValueError('ANCHOR_METADATA_REQUIRED')
        anchor_mode='saved_pass2a_scores'
    ss,ee=original_arrays['step_starts'],original_arrays['step_ends']
    out,methods,diagnostics=score_sampling(raw,ss,ee,original_arrays,original_meta,anchor_arrays,anchor_meta,identity)
    plan=moment_plan(token_count,8);values=out['features'];names=diagnostics['names'] if 'names' in diagnostics else original_meta['diagnostics']['banks'][diagnostics['bank']]['names']
    # The older scorer stores its names through the chosen bank's feature schema.
    diagnostics['names']=list(names);diagnostics['anchor_mode']=anchor_mode
    reference=original_meta['methods']['moment__iu']
    for selector in ENTROPY_SELECTORS:
        selected=plan.fit_indices[choose(values[plan.fit_indices,0],selector)]
        out[selector+'__selected']=selected
        _,fraction=selection_support(token_count,plan.starts,plan.ends,selected,ss,ee)
        out[selector+'__step_support_fraction']=fraction
        assert len(selected)==budget(len(plan.fit_indices))
        if np.array_equal(selected,plan.fit_indices):
            diagnostics['fits'][selector]=dict(replay=True,joint_valid=diagnostics['original_joint_valid'])
            for core in CORES:
                arm=f'sample_{selector}__{core}';source=f'sample_full__{core}'
                methods[arm]=deepcopy(methods[source]);methods[arm].update(source_arm=source,anchor_replay=True)
                if methods[arm]['valid']:
                    for suffix in ('window','risk'):out[arm+'__'+suffix]=out[source+'__'+suffix].copy()
            continue
        arrays,native,detail=score_selected(values,names,plan,ss,ee,selected,identity,reference)
        out.update({selector+'__'+key:value for key,value in arrays.items()})
        diagnostics['fits'][selector]=dict(detail,replay=False)
        for core in CORES:
            arm=f'sample_{selector}__{core}';d=deepcopy(native[core])
            d.update(anchor_replay=False,source_arm=selector+'__native_'+d['source_core'],bank=diagnostics['bank'])
            methods[arm]=d
            if d['valid']:
                for suffix in ('window','risk'):out[arm+'__'+suffix]=arrays[core+'__'+suffix].copy()
    assert set(methods)==set(ARMS)
    return out,methods,diagnostics
