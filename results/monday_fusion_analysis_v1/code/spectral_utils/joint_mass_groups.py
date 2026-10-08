"""Training-only feature-mass-aware group discovery for Joint.

The feature measure solves a nonnegative kernel-energy problem on squared
correlations. Its objective depends on total mass at a repeated feature, not
the number of copies. Weighted complete-covariance rank-one projection and
weighted spectral clustering use that measure. This replaces group discovery;
the downstream sparse and checked Joint fits retain their existing objectives.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from .joint_lsml import covariance_matrix,canonicalize_labels,coassignment_from_labels


def feature_mass(covariance):
    c=np.asarray(covariance,float)
    scale=np.sqrt(np.maximum(np.diag(c),1e-12))
    corr=np.clip(c/scale[:,None]/scale[None,:],-1.,1.)
    kernel=corr*corr
    # min .5*m'Km - 1'm, m>=0. Normalize afterwards to a probability measure.
    def objective(m):
        km=kernel@m
        return float(.5*m@km-m.sum()),km-1.
    initial=1/np.maximum(kernel.sum(axis=1),1.)
    fit=minimize(objective,initial,jac=True,bounds=[(0.,None)]*len(c),
        method='L-BFGS-B',options=dict(ftol=1e-14,gtol=1e-10,maxiter=10000,maxls=50))
    raw=np.maximum(fit.x,0.);gradient=kernel@raw-1.
    projected=np.where(raw>0,gradient,np.minimum(gradient,0.))
    kkt=float(np.max(np.abs(projected)))
    if not np.isfinite(raw).all() or raw.sum()<=0 or kkt>1e-6:
        raise ValueError('FEATURE_MASS_KKT_FAILURE')
    return raw/raw.sum(),dict(raw_mass=raw,mass=raw/raw.sum(),kernel=kernel,
        objective=objective(raw)[0],projected_gradient=kkt,optimizer_success=bool(fit.success),
        iterations=int(fit.nit),rule='nonnegative squared-correlation kernel energy')


def mass_residual_affinity(covariance,mass):
    c=np.asarray(covariance,float);mass=np.asarray(mass,float);root=np.sqrt(mass)
    weighted=root[:,None]*c*root[None,:]
    eigenvalues,eigenvectors=np.linalg.eigh(weighted);value=float(eigenvalues[-1])
    if value<=1e-12:raise ValueError('NO_MASS_GLOBAL_COMPONENT')
    # This also defines the projection at zero-mass features without division.
    loading=c@(root*eigenvectors[:,-1])/np.sqrt(value)
    affinity=np.abs(c-np.outer(loading,loading))
    # Keep self affinity: repeated coordinates must not add an off-diagonal
    # private-noise term that the original coordinate's diagonal had excluded.
    return affinity,loading


def mass_spectral_cluster(affinity,mass,k):
    affinity=np.asarray(affinity,float);mass=np.asarray(mass,float)
    degree=affinity@mass
    if np.any(degree<=1e-14):raise ValueError('ZERO_MASS_GRAPH_DEGREE')
    root=np.sqrt(mass/degree)
    operator=root[:,None]*affinity*root[None,:]
    eigenvalues,eigenvectors=np.linalg.eigh(operator)
    order=np.argsort(eigenvalues)[-k:][::-1];values=eigenvalues[order];vectors=eigenvectors[:,order]
    if np.any(values<=1e-10):raise ValueError('TOO_FEW_POSITIVE_MASS_COMPONENTS')
    # Eigenfunction extension equals U/sqrt(mass) on positive-mass nodes,
    # while assigning zero-mass rows from their affinities to the support.
    embedding=(affinity@(root[:,None]*vectors))/np.sqrt(degree[:,None])/values
    norm=np.linalg.norm(embedding,axis=1)
    if np.any(norm<=1e-12):raise ValueError('ZERO_MASS_EMBEDDING')
    embedding/=norm[:,None]
    support=np.flatnonzero(mass>0)
    if len(support)<k:raise ValueError('TOO_FEW_POSITIVE_MASS_FEATURES')
    # Deterministic farthest-point initialization; splitting a point's mass
    # does not change its distance or create a distinct additional center.
    center=mass@embedding
    distance=np.sum((embedding-center)**2,axis=1)
    selected=[int(support[np.argmax(np.round(distance[support],12))])]
    for _ in range(1,k):
        distances=np.min(np.sum((embedding[:,None,:]-embedding[selected][None,:,:])**2,axis=2),axis=1)
        chosen=int(support[np.argmax(np.round(distances[support],12))])
        if chosen in selected or distances[chosen]<=1e-12:raise ValueError('MASS_EMBEDDING_CENTERS_COLLAPSE')
        selected.append(chosen)
    fitted=KMeans(n_clusters=k,init=embedding[selected],n_init=1,algorithm='lloyd',
        max_iter=300,tol=1e-10,random_state=0).fit(embedding,sample_weight=mass)
    return canonicalize_labels(fitted.labels_)


def mass_nmi(left,right,mass):
    left=np.asarray(left);right=np.asarray(right);mass=np.asarray(mass,float)
    _,li=np.unique(left,return_inverse=True);_,ri=np.unique(right,return_inverse=True)
    table=np.zeros((li.max()+1,ri.max()+1));np.add.at(table,(li,ri),mass/mass.sum())
    a=table.sum(axis=1);b=table.sum(axis=0);expected=a[:,None]*b[None,:]
    mask=table>0;mutual=float(np.sum(table[mask]*np.log(table[mask]/expected[mask])))
    ha=-float(np.sum(a[a>0]*np.log(a[a>0])));hb=-float(np.sum(b[b>0]*np.log(b[b>0])))
    return float(np.clip(mutual/np.sqrt(ha*hb),0,1)) if min(ha,hb)>0 else float(ha==hb)


def discover_mass_groups(values,row_folds,*,seed):
    # Retain the public seed contract; clustering here is deterministic.
    values=np.asarray(values,float);row_folds=np.asarray(row_folds)
    mass,mass_audit=feature_mass(covariance_matrix(values))
    fold_data=[]
    for f in np.unique(row_folds):
        cov=covariance_matrix(values[row_folds!=f]);weights,audit=feature_mass(cov)
        affinity,loading=mass_residual_affinity(cov,weights)
        fold_data.append(dict(mass=weights,audit=audit,affinity=affinity,loading=loading))
    candidates=[]
    for k in (3,4):
        try:
            parts=[mass_spectral_cluster(d['affinity'],d['mass'],k) for d in fold_data]
            affinity=np.mean([coassignment_from_labels(p) for p in parts],axis=0)
            labels=mass_spectral_cluster(affinity,mass,k)
            stability=[mass_nmi(labels,p,mass) for p in parts]
            sizes=[int(np.sum(labels==g)) for g in np.unique(labels)]
            valid=len(sizes)==k and min(sizes)>=2 and all(
                len(np.unique(p))==k and min(np.bincount(p))>=2 for p in parts)
            candidates.append(dict(K=k,labels=labels,parts=parts,group_sizes=sizes,stability=stability,
                valid=valid,median_stability=float(np.median(stability)),
                mean_stability=float(np.mean(stability)),minimum_stability=float(min(stability))))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            candidates.append(dict(K=k,valid=False,reason=str(exc)))
    valid=[c for c in candidates if c['valid']]
    # Kernels are reconstructed from the hashed training input by the auditor;
    # do not repeat five dense matrices at every deletion proposal checkpoint.
    compact=lambda audit:{key:value for key,value in audit.items() if key!='kernel'}
    details=dict(candidates=candidates,mass_audit=compact(mass_audit),fold_mass_audits=[compact(d['audit']) for d in fold_data],
        stability_measure='mass-weighted normalized mutual information',seed=seed)
    if not valid:return dict(status='NO_ADMISSIBLE_PARTITION',**details)
    best=sorted(valid,key=lambda c:(-c['median_stability'],-c['mean_stability'],-c['minimum_stability'],c['K']))[0]
    return dict(status='SELECTED',**best,**details)
