# Review amendment: zero pair products

2026-09-07. Added during the mathematical review of the frozen unlabeled
pair-group audit, after all 110 answer fits completed, before any new
localization score or benchmark label evaluation. Preserve all original
prototype sources and fit artifacts. This is not a retrospective launch
freeze for the amendment.

At u_i=u_j=0, the nuisance Jacobian of u_i*u_j is zero. Profiling those
two singular factor coordinates can overstate global identifiability.
Use one direct residual-product coordinate per pair, with derivative one
on its off-diagonal equation. The global factor must be identified after
profiling this coordinate even when the fitted pair residual is zero.
Groups of size at least three retain the legacy nuisance coordinates.

Add a synthetic counterexample where only two of three groups have nonzero
global loadings: the old zero-u Jacobian passes but reciprocal rescaling
of the two groups preserves all cross-group products, and pair residuals
absorb their internal differences. The product-profiled Jacobian must fail.
Verify generic nonzero-product equivalence and a genuine checked fit.

The amended canonical entry point for future experiments is
`spectral_utils.joint_pair_jacobian.fit_joint_pairs_checked`. It preserves
the fitted factors, covariance and native map, replacing only the pair
Jacobian diagnostic. Independently check this guard on all existing fitted
audit records, report any eligibility changes, and bind amendment source
and test hashes in the review. No label access or model-quality claim.
