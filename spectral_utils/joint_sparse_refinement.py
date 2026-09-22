"""Conditional-information refinement after Joint learned noise-only rows.

The95% rule is unchanged; it now refers to a model whose noise-only rows have
already been learned through the sparse Joint objective, so it need not protect
an incidental independent-noise group as a full-strength latent component.
"""
import numpy as np
from .joint_feature_selection import pruning_path,checked_fit
from .joint_lsml import covariance_matrix,canonicalize_labels
from .joint_group_reliability import group_reliability_weights


def refine_joint_membership(values,labels,*,seed,notify=None):
    x=np.asarray(values,float);labels=canonicalize_labels(labels)
    path=pruning_path(x,labels,anchor_index=0,seed=seed,retention=.95,
        maximum_removals=max(0,x.shape[1]-8),stop_after_automatic=True,notify=notify)
    active=np.asarray(path['automatic']['active']);part=canonicalize_labels(labels[active])
    cov=covariance_matrix(x[:,active])
    fit=checked_fit(cov,part,seed+path['automatic_index']);j=fit.joint
    np.testing.assert_allclose(j.relative_offdiag_misfit,path['automatic']['misfit'],atol=1e-12)
    local,readout=group_reliability_weights(j.model_covariance,j.global_loading,part)
    weight=np.zeros(x.shape[1]);weight[active]=local
    return dict(active=active,unoriented_weights=weight,selection=path,labels=part,
        observed_covariance=cov,model_covariance=j.model_covariance,global_loading=j.global_loading,
        group_loading=j.group_loading,readout=readout,diagonal_audit=j.diagonal_audit,
        converged_starts=j.converged_starts,multistart=j.multistart_audit,jacobian=j.jacobian_audit)
