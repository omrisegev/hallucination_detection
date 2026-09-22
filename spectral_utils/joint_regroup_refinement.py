"""Information refinement with training-only group rediscovery per deletion."""
import numpy as np
from .joint_feature_selection import checked_fit,factor_matrix,information
from .joint_block_balanced import discover_sourcefold_groups
from .joint_lsml import covariance_matrix,canonicalize_labels
from .joint_group_reliability import group_reliability_weights


def refine_with_regrouping(values,labels,row_folds,*,seed,group_seed,notify=None):
    x=np.asarray(values,float);row_folds=np.asarray(row_folds);part=canonicalize_labels(labels)
    if len(row_folds)!=len(x):raise ValueError('ROW_FOLDS_MISMATCH')
    cov=covariance_matrix(x);active=np.arange(x.shape[1]);initial=checked_fit(cov,part,seed)
    ref_a=factor_matrix(initial.joint.global_loading,initial.joint.group_loading,part)
    reference,_=information(cov,ref_a);relevant=reference>1e-8
    path=[];audit=[];current=initial;automatic_index=0;limit=max(0,x.shape[1]-8)
    for step in range(limit+1):
        subcov=cov[np.ix_(active,active)];j=current.joint
        retained,_=information(subcov,ref_a[active]);ratios=np.divide(retained,reference,out=np.ones_like(retained),where=relevant)
        valid_retention=bool(np.all(ratios[relevant]>=.95))
        if valid_retention:automatic_index=len(path)
        path.append(dict(active=active.copy(),labels=part.copy(),retention=ratios,
            minimum_retention=float(ratios[relevant].min()),group_sizes=[int(sum(part==g)) for g in np.unique(part)],
            converged_starts=j.converged_starts,misfit=j.relative_offdiag_misfit,
            global_loading=j.global_loading,group_loading=j.group_loading,
            multistart=j.multistart_audit,jacobian=j.jacobian_audit,diagonal_audit=j.diagonal_audit))
        if notify:notify(step,len(active),path[-1]['minimum_retention'])
        if not valid_retention or step==limit:break
        a=factor_matrix(j.global_loading,j.group_loading,part);info,loss=information(subcov,a)
        importance=np.mean(loss[:,info>1e-8]/info[info>1e-8],axis=1)
        candidates=[int(i) for i in np.argsort(importance,kind='stable') if np.sum(part==part[i])>2]
        if not candidates:break
        accepted=False;rejected=[]
        for index in candidates[:3]:
            proposal=np.delete(active,index)
            discovery=discover_sourcefold_groups(x[:,proposal],row_folds,seed=group_seed)
            attempt=dict(feature=int(active[index]),proposed_active=proposal,discovery=discovery)
            try:
                if discovery['status']!='SELECTED':raise ValueError('NO_ADMISSIBLE_REGROUPING')
                proposed_labels=np.asarray(discovery['labels'])
                fit=checked_fit(cov[np.ix_(proposal,proposal)],proposed_labels,seed+step+1)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                rejected.append(dict(attempt,reason=str(exc)));continue
            audit.append(dict(removed=int(active[index]),conditional_information_loss=float(importance[index]),
                proposed_active=proposal,discovery=discovery,rejected_candidates=rejected))
            active=proposal;part=proposed_labels;current=fit;accepted=True;break
        if not accepted:
            audit.append(dict(stopped='NO_VALID_DELETION_AMONG_THREE_BEST',rejected_candidates=rejected));break
    chosen=path[automatic_index];active=np.asarray(chosen['active']);part=np.asarray(chosen['labels'])
    final_cov=covariance_matrix(x[:,active]);final=checked_fit(final_cov,part,seed+automatic_index);j=final.joint
    np.testing.assert_allclose(j.relative_offdiag_misfit,chosen['misfit'],atol=1e-12)
    local,readout=group_reliability_weights(j.model_covariance,j.global_loading,part)
    w=np.zeros(x.shape[1]);w[active]=local
    selection=dict(path=path,automatic_index=automatic_index,automatic=chosen,deletion_audit=audit,
        initial_factor_information=reference,initial_factor_matrix=ref_a,retention_rule=.95,
        group_seed=group_seed,fit_seed=seed,reference='fixed initial factors; never reset after regrouping')
    return dict(active=active,unoriented_weights=w,selection=selection,labels=part,
        observed_covariance=final_cov,model_covariance=j.model_covariance,global_loading=j.global_loading,
        group_loading=j.group_loading,readout=readout,diagonal_audit=j.diagonal_audit,
        converged_starts=j.converged_starts,multistart=j.multistart_audit,jacobian=j.jacobian_audit)
