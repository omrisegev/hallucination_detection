import numpy as np
from spectral_utils.joint_mass_groups import feature_mass,mass_residual_affinity,mass_spectral_cluster,mass_nmi


def covariance_fixture():
    labels=np.repeat(np.arange(3),3)
    v=np.array([.3,.38,.34,.4,.31,.36,.29,.33,.39])
    u=np.array([.65,.6,.57,.62,.68,.58,.7,.59,.63])
    c=np.outer(v,v)+(labels[:,None]==labels[None,:])*np.outer(u,u)
    c+=np.diag(1-np.diag(c))
    return c,labels


def test_kernel_energy_and_mass_aggregate_under_exact_repetition():
    c,_=covariance_fixture();mass,audit=feature_mass(c)
    ids=np.r_[np.arange(9),0,0,4];copy=c[np.ix_(ids,ids)]
    extended,other=feature_mass(copy)
    aggregate=np.bincount(ids,weights=extended,minlength=9)
    np.testing.assert_allclose(aggregate,mass,atol=1e-6,rtol=1e-6)
    # Verify the nonnegative optimum directly, independently of the optimizer.
    for record in (audit,other):
        raw=np.asarray(record['raw_mass']);gradient=np.asarray(record['kernel'])@raw-1
        assert np.min(gradient)>=-1e-6
        assert np.max(np.abs(raw*gradient))<1e-6
    raw=np.asarray(other['raw_mass']);grouped=np.bincount(ids,weights=raw,minlength=9)
    direct=.5*grouped@(c*c)@grouped-grouped.sum()
    np.testing.assert_allclose(direct,other['objective'],atol=1e-12)


def test_weighted_projection_spectrum_partition_and_nmi_preserve_split_mass():
    c,labels=covariance_fixture();mass,_=feature_mass(c)
    ids=np.r_[np.arange(9),0,0,4];multiplicity=np.bincount(ids,minlength=9)
    extended=mass[ids]/multiplicity[ids]
    a,v=mass_residual_affinity(c,mass)
    ac,vc=mass_residual_affinity(c[np.ix_(ids,ids)],extended)
    np.testing.assert_allclose(np.outer(vc,vc),np.outer(v[ids],v[ids]),atol=1e-12)
    np.testing.assert_allclose(ac,a[np.ix_(ids,ids)],atol=1e-12)
    base=mass_spectral_cluster(a,mass,3);copy=mass_spectral_cluster(ac,extended,3)
    np.testing.assert_array_equal(copy[:,None]==copy[None,:],base[ids,None]==base[None,ids])
    perturbed=labels.copy();perturbed[1]=2
    np.testing.assert_allclose(mass_nmi(labels,perturbed,mass),mass_nmi(labels[ids],perturbed[ids],extended),atol=1e-12)
    # Removing diagonal affinity would break the underlying replicated matrix.
    assert np.all(np.diag(a)>0)


def test_zero_mass_features_have_a_finite_extension():
    c,_=covariance_fixture();mass=np.ones(9)/8;mass[1]=0.
    a,v=mass_residual_affinity(c,mass);labels=mass_spectral_cluster(a,mass,3)
    assert np.isfinite(v).all() and len(labels)==9 and len(np.unique(labels))==3
