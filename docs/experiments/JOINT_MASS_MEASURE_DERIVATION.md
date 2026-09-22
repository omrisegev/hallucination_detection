# Scope of the split-mass property

For nonconstant training columns, normalize their centered traces to unit
length, u_i. Then K_ij=(u_i'u_j)^2 is a Gram matrix of vec(u_i*u_i'), hence
positive semidefinite. The nonnegative quadratic objective is convex. The
implementation checks its projected KKT residual rather than assuming an
optimizer success flag proves a solution. No claim is made for an unbounded
zero-variance kernel coordinate: failure is explicit in this prototype.

If copies of coordinate i have identical correlations, replacing their masses
by the sum gives exactly the same .5*m'K*m - sum(m). Aggregate optimal mass is
therefore an optimum of the original problem. It need not be unique for a
singular kernel. Normalizing total mass preserves this equality. A perturbation
to a near copy changes K: exact equality does not automatically extend to it.

For a declared mass, the complete weighted covariance operator is a finite
representation of a covariance integral. Splitting mass at identical traces
preserves its nonzero eigenstructure. The covariance extension of the leading
component supplies loadings even at zero-mass coordinates. Applying absolute
residual affinity elementwise preserves the repeated-coordinate representation
only when diagonal affinities are retained. Zeroing the old diagonal while
retaining a new duplicate's matching off-diagonal entry would break it.

The mass-normalized affinity operator has the same property. Its eigenfunction
extension, row normalization and mass-weighted centroid objective operate on
the measure rather than the number of copies. Our synthetic fixedK test also
checks the chosen partition, but this is not a universal uniqueness theorem:
eigenvalue multiplicity, initialization ties and clustering local minima remain.
Mass-weighted NMI depends on aggregated contingency mass, so it also preserves
a split when both partitions agree on copy assignments.

These are properties of INTERNAL discovery operators. Minimum-two group
admissibility, choice between K3/K4, sparse thresholding, the downstream
unweighted Joint objective, refinement and the final argmax are additional
operations. The full wrapper first canonicalizes exact aliases as before.
Neither the derivation nor synthetic checks prove full-model near-copy
invariance, arbitrary surplus-feature tolerance, native coverage or localization
quality. The frozen full five-bank benchmark evaluates those separate claims.

This also differs from block-pair balancing: that historical method reweights
the final Joint loss after assigning groups. Here masses enter provisional
projection, spectral clustering and stability before the unchanged final fit.
