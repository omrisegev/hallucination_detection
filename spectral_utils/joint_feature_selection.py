"""Alternating Joint fit and backward feature inclusion, without error labels.

Selection estimates the conditional information about BOTH global and group
factors. It is a reconstruction surrogate, not a certificate of correctness.
The automatic stop preserves 95% of each initial factor's information.
"""
import numpy as np
from .joint_lsml import covariance_matrix, canonicalize_labels, hierarchical_joint_weights
from .joint_pair_jacobian import fit_joint_pairs_checked
from .lsml_gate_locator_research import _orient


def factor_matrix(v,u,labels):
    groups=np.unique(labels); a=np.zeros((len(v),1+len(groups)));a[:,0]=v
    for j,g in enumerate(groups):a[labels==g,j+1]=u[labels==g]
    return a


def information(cov,loadings):
    # Diagonal shrinkage only for the conditional-information calculation.
    cov=np.asarray(cov,float); diag=np.maximum(np.diag(cov),1e-12)
    regularized=(1-1e-4)*cov+1e-4*np.diag(diag)
    precision=np.linalg.inv(regularized); transformed=precision@loadings
    info=np.sum(loadings*transformed,axis=0)
    loss=transformed**2/np.diag(precision)[:,None]
    return info,loss


def checked_fit(cov,labels,seed):
    fit=fit_joint_pairs_checked(cov,labels,anchor_index=0,seed=seed,starts=5)
    j=fit.joint
    valid=bool(j.converged and j.multistart_audit['status']=='PASS' and
        j.jacobian_audit['full_global_rank'] and j.jacobian_audit['condition_number']<=1e8)
    if not valid:raise ValueError('INVALID_JOINT_FIT')
    return fit


def pruning_path(values,labels,*,anchor_index,seed,retention=.95,maximum_removals=43,notify=None,stop_after_automatic=False):
    """Refit Joint after each deletion. No protected feature/label-based choice.

    Groups stay fixed to isolate inclusion from group search. At least two
    members per group, checked pair identification, all original fit guards.
    The complete path is diagnostic; the automatic state is fixed by retention.
    """
    x=np.asarray(values,float); labels=canonicalize_labels(labels); cov=covariance_matrix(x)
    active=np.arange(x.shape[1]); initial=checked_fit(cov,labels,seed)
    ref_a=factor_matrix(initial.joint.global_loading,initial.joint.group_loading,labels)
    reference,_=information(cov,ref_a); relevant=reference>1e-8
    current=initial; path=[]; audit=[]; automatic_index=0; stop_crossed=False
    for step in range(maximum_removals+1):
        part=canonicalize_labels(labels[active]); subcov=cov[np.ix_(active,active)]
        j=current.joint
        _,local_w,readout=hierarchical_joint_weights(x[:,active],part,j.global_loading,anchor_index=0,small_m_guard=True)
        w=np.zeros(x.shape[1]);w[active]=local_w;w,orientation=_orient(x,w,anchor_index)
        retained,_=information(subcov,ref_a[active]);ratios=np.divide(retained,reference,out=np.ones_like(retained),where=relevant)
        pass_retention=bool(np.all(ratios[relevant]>=retention))
        if not stop_crossed and pass_retention:automatic_index=len(path)
        if not pass_retention:stop_crossed=True
        state=dict(active=active.copy(),weights=w,retention=ratios,minimum_retention=float(ratios[relevant].min()),
            converged_starts=j.converged_starts,misfit=j.relative_offdiag_misfit,
            group_sizes=[int(sum(part==g)) for g in np.unique(part)],readout=readout,orientation=orientation)
        path.append(state)
        if notify:notify(step,len(active),state['minimum_retention'])
        # When no deep diagnostic snapshots are requested, later states cannot
        # affect the predeclared FIRST retention crossing / automatic choice.
        if stop_after_automatic and stop_crossed:break
        if step==maximum_removals:break
        a=factor_matrix(j.global_loading,j.group_loading,part);info,loss=information(subcov,a)
        active_factor=info>1e-8
        importance=np.mean(loss[:,active_factor]/info[active_factor],axis=1)
        candidates=[int(i) for i in np.argsort(importance,kind='stable') if np.sum(part==part[i])>2]
        if not candidates:break
        accepted=False;failures=[]
        for index in candidates[:3]:
            proposal=np.delete(active,index); proposal_labels=canonicalize_labels(labels[proposal])
            try:
                fit=checked_fit(cov[np.ix_(proposal,proposal)],proposal_labels,seed+step+1)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                failures.append(dict(feature=int(active[index]),reason=str(exc)));continue
            audit.append(dict(removed=int(active[index]),conditional_information_loss=float(importance[index]),
                              rejected_candidates=failures))
            current=fit;active=proposal;accepted=True;break
        if not accepted:
            audit.append(dict(stopped='NO_VALID_DELETION_AMONG_THREE_BEST',rejected_candidates=failures));break
    return dict(path=path,automatic_index=automatic_index,automatic=path[automatic_index],
                deletion_audit=audit,initial_factor_information=reference,retention_rule=retention)
