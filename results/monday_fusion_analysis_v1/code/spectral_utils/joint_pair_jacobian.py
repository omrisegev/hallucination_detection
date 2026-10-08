"""Profile pair *products*, including zero, when checking global identification.

The factor coordinates u_i,u_j are singular at (0,0). Their first-order
Jacobian can omit a valid pair-residual nuisance direction. Parameterizing
each pair by its unrestricted product r avoids that false identification.
This is a review amendment to the frozen pair-fit prototype; fitted values
and covariance construction are unchanged.
"""
from dataclasses import replace
import numpy as np
from .joint_lsml import _profiled_jacobian_audit, canonicalize_labels
from .joint_pair_extension import fit_joint_pairs


def profiled_pair_jacobian(global_loading, group_loading, labels):
    v,u=np.asarray(global_loading,float),np.asarray(group_loading,float)
    labels=canonicalize_labels(labels);p=len(labels)
    if v.shape!=(p,) or u.shape!=(p,) or not np.isfinite(v).all() or not np.isfinite(u).all():
        raise ValueError('LOADING_PARTITION_MISMATCH')
    same=labels[:,None]==labels[None,:];mask=same.astype(float);np.fill_diagonal(mask,0.)
    groups=[np.flatnonzero(labels==g) for g in np.unique(labels)]
    if len(groups)<3 or min(map(len,groups))<2:
        raise ValueError('REQUIRES_K_THREE_AND_MINIMUM_GROUP_SIZE_TWO')
    pairs=[ids for ids in groups if len(ids)==2]
    if not pairs:return _profiled_jacobian_audit(v,u,mask)
    left,right=np.triu_indices(p,1);jv=np.zeros((len(left),p));nuisance=[]
    for row,(i,j) in enumerate(zip(left,right)):jv[row,i]=v[j];jv[row,j]=v[i]
    for ids in groups:
        if len(ids)==2:
            column=np.zeros(len(left));column[(left==ids[0])&(right==ids[1])]=1.;nuisance.append(column)
        else:
            for index in ids:
                column=np.zeros(len(left))
                for row,(i,j) in enumerate(zip(left,right)):
                    if same[i,j]:
                        if i==index:column[row]=u[j]
                        elif j==index:column[row]=u[i]
                if np.linalg.norm(column)>1e-12:nuisance.append(column)
    ju=np.column_stack(nuisance);basis,singular,_=np.linalg.svd(ju,full_matrices=False)
    tolerance=max(ju.shape)*np.finfo(float).eps*max(singular[0],1e-12)
    rank=int(np.sum(singular>tolerance));basis=basis[:,:rank]
    profiled=jv-basis@(basis.T@jv);norms=np.linalg.norm(profiled,axis=0);active=norms>1e-12
    normalized=profiled[:,active]/norms[active]
    values=np.linalg.svd(normalized,compute_uv=False) if normalized.size else np.array([])
    tolerance=max(normalized.shape)*np.finfo(float).eps*max(values[0],1e-12) if len(values) else np.inf
    global_rank=int(np.sum(values>tolerance))
    return {'full_global_rank':bool(active.sum()==p and global_rank==p),'active_global_columns':int(active.sum()),
        'rank':global_rank,'condition_number':float(values[0]/max(values[-1],1e-12)) if len(values) else float('inf'),
        'singular_values':values,'nuisance_columns':ju.shape[1],'nuisance_rank':rank,
        'pair_product_columns':len(pairs),'parameterization':'one_direct_residual_product_per_pair',
        'zero_pair_products':sum(float(u[ids[0]]*u[ids[1]])==0. for ids in pairs)}


def fit_joint_pairs_checked(covariance,labels,**kwargs):
    """Canonical entry point for future pair-enabled native-fusion trials."""
    result=fit_joint_pairs(covariance,labels,**kwargs)
    jacobian=profiled_pair_jacobian(result.joint.global_loading,result.joint.group_loading,labels)
    return replace(result,joint=replace(result.joint,jacobian_audit=jacobian))
