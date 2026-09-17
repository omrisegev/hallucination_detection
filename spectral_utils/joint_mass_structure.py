"""Staged sparse membership with mass-aware internal group discovery."""
import numpy as np
from .joint_sparse_membership import calibrate_penalty,sparse_membership
from .joint_staged_membership import staged_group_support
from .joint_mass_groups import discover_mass_groups
from .joint_lsml import covariance_matrix
from .joint_feature_selection import checked_fit
from .joint_group_reliability import group_reliability_weights

def learn_mass_structure(train,rowfold,grams,nrows,*,seed,calibration_seed,notify=None):
    def emit(status,**detail):
        if notify:notify(status,**detail)
    active=np.arange(train.shape[1]);rounds=[]
    for round_index in range(3):
        sub=train[:,active];cov=covariance_matrix(sub)
        discovery=discover_mass_groups(sub,rowfold,seed=seed+100)
        if discovery['status']!='SELECTED':raise ValueError('NO_ADMISSIBLE_SPARSE_STRUCTURE')
        penalty,calibration=calibrate_penalty(grams[:,active][:,:,active],nrows,cov,
            np.asarray(discovery['labels']),seed=calibration_seed+100*round_index)
        emit('SPARSE_FIT',round=round_index,features=len(active),penalty=penalty)
        local,audit=sparse_membership(cov,np.asarray(discovery['labels']),penalty=penalty,
            seed=seed+200+round_index)
        local,signal_audit=staged_group_support(local,np.asarray(discovery['labels']),audit)
        audit['signal_membership']=signal_audit
        selected=active[local]
        rounds.append(dict(round=round_index,active_before=active,active_after=selected,
            discovery=discovery,calibration=calibration,sparse=audit))
        emit('MEMBERSHIP',round=round_index,kept=len(selected),converged=audit['converged_starts'])
        if len(selected)<6:return dict(valid=False,rounds=rounds,active=selected,
            failure='TOO_FEW_ACTIVE_MEASUREMENTS')
        if np.array_equal(active,selected):break
        active=selected
    sub=train[:,active];cov=covariance_matrix(sub)
    discovery=discover_mass_groups(sub,rowfold,seed=seed+100)
    candidates=sorted([c for c in discovery['candidates'] if c['valid']],
        key=lambda c:(-c['median_stability'],-c['mean_stability'],-c['minimum_stability'],c['K']))
    attempts=[]
    for candidate in candidates:
        try:checked=checked_fit(cov,np.asarray(candidate['labels']),seed=seed+200)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            attempts.append(dict(K=candidate['K'],valid=False,reason=str(exc)));continue
        j=checked.joint;local,readout=group_reliability_weights(j.model_covariance,j.global_loading,candidate['labels'])
        w=np.zeros(train.shape[1]);w[active]=local;attempts.append(dict(K=candidate['K'],valid=True))
        return dict(valid=True,active=active,unoriented_weights=w,rounds=rounds,attempts=attempts,
            final_discovery=discovery,final_labels=candidate['labels'],readout=readout,
            observed_covariance=cov,model_covariance=j.model_covariance,global_loading=j.global_loading,
            group_loading=j.group_loading,diagonal_audit=j.diagonal_audit,
            converged_starts=j.converged_starts,multistart=j.multistart_audit,jacobian=j.jacobian_audit)
    return dict(valid=False,rounds=rounds,active=active,attempts=attempts,failure='NO_VALID_DEBIASED_STRUCTURE')

