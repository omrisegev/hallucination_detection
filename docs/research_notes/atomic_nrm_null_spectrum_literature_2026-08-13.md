# What an eigenvalue near one can—and cannot—identify

## Scope

This note audits the professional basis for the `lambda approximately 1`
heuristic in Neutral Residual Mode (NRM), especially after replacing six
provenance-family coordinates with atomic feature contributions.  The sources
below are primary papers.  The conclusion deliberately separates two claims:

1. **null geometry:** which covariance directions look like standardized,
   weakly dependent residual variation rather than a spike or redundancy;
2. **hallucination-target identification:** which of those directions should
   improve correctness discrimination, and with which sign.

The literature supports the first claim under stated models.  It does not
supply the second claim.

## Primary-source findings

### One is a population null center, not a sample eigenvalue selector

For standardized independent coordinates the population correlation is the
identity, so its population eigenvalues are one.  In finite or high-dimensional
samples, however, the eigenvalues spread into a bulk.  The original
[Marchenko–Pastur paper](https://doi.org/10.1070/SM1967v001n04ABEH001994)
establishes the limiting spectral distribution for relevant random-matrix
ensembles.  Thus, increasing the coordinate count makes `argmin |lambda-1|`
less—not more—distinguished: several sample modes can legitimately lie near
one under the null.

The analytic Marchenko–Pastur edges are not used literally in Atomic NRM v1.
Its residual matrices are standardized after projection on IU, then averaged
across cells with different sample sizes.  Those operations do not provide one
unambiguous `p/n_eff` satisfying the iid spherical model.  A direct
within-cell permutation null preserves the actual dimensions and sample sizes
without pretending that this effective aspect ratio is known.

### Parallel analysis justifies a null band, not semantic orientation

[Horn's original parallel-analysis paper](https://doi.org/10.1007/BF02289447)
argued that sample latent roots should be compared with roots generated from
random variables so sampling-error capitalization is removed.  A modern
primary analysis by
[Dobriban (2020)](https://doi.org/10.1214/19-AOS1907) proves consistency for
large components in certain high-dimensional factor models: feature-wise
permutations preserve noise while destroying suitable low-rank signal.  The
same paper is explicit that smaller components can be missed.

This supports Atomic NRM's use of independently permuted residual columns to
separate clearly non-null covariance structure from a neutral band.  It does
not imply that a vector retained inside that band predicts hallucination.  In
fact, parallel analysis is normally used to retain *departures* from noise;
Atomic NRM uses its complement as a nuisance-rejection device, which is an
application-specific inference rather than a theorem from parallel analysis.

### Spiked-covariance theory says bulk eigenvectors are not uniquely recoverable

[Baik, Ben Arous, and Peche](https://doi.org/10.1214/009117905000000233)
established a phase transition for extreme eigenvalues in a spiked sample
covariance model.  [Paul (2007)](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A17n418.pdf)
also finds a phase transition in sample eigenvectors: reliable population
alignment requires a sufficiently separated spike.  These results are about
specific asymptotic models, but their relevant warning is direct: a sample
eigenvector buried in a null bulk is not a stable, uniquely identified
population direction.

This rules out treating the single eigenvector numerically closest to one as a
general high-dimensional object.  At atomic resolution, the defensible object
is the **whole neutral subspace**, followed by an independently justified
anchor, not an arbitrary basis vector inside it.

### Perturbation theory favors a projector over a basis vector

[Davis and Kahan (1970)](https://doi.org/10.1137/0707001) bound rotations of
invariant subspaces using the gap separating an eigenvalue cluster from the
rest of the spectrum.  Inside a close or repeated cluster, individual
eigenvectors can rotate while the cluster's projector stays stable.  This is
the formal reason to retain all modes inside the permutation-null band and
project an anchor into their span.

The project result matches that prediction.  In the 17 common atomic
coordinates, the null band retained eigenvalues 0.960685 and 1.025557.
Single-mode selection changed sharply under dataset-family removal; the frozen
two-dimensional projector with an inverse-absolute-dependence anchor reached a
minimum leave-one-cell direction cosine of 0.975505.

### A noise subspace needs a separate observation model to identify a target

Noise-subspace methods do not infer a target from isotropy alone.  For example,
[Schmidt's MUSIC paper](https://doi.org/10.1109/TAP.1986.1143830) combines a
noise eigenspace with an explicit sensor-array steering model to locate
emitters.  The eigenspace rejects nuisance structure; the steering vector says
what physical target to look for.

NRM has no analogous theorem connecting its symmetric atomic anchor to
hallucination correctness.  The six provenance families can play that missing
role empirically: they encode which feature variants are repeated measurements
of the same measurement mechanism before residual geometry is estimated.

## Project-specific conclusion

The literature supports the following limited statement:

> After standardization, permutation-calibrated covariance eigenvalues can
> distinguish strong dependence/redundancy from a noise-like subspace, and a
> projector is more defensible than one eigenvector inside a close cluster.

It does **not** support:

> A residual mode near eigenvalue one is therefore the hallucination-correction
> direction.

The project's supervised ceiling makes the distinction observable.  Atomic
residuals contain more label-usable target information than family residuals:
at prior 0.3, the cross-fitted atomic head improves IU by +1.298pp versus
+0.721pp for the family head, a direct +0.577pp difference with interval
[+0.102,+0.910].  Yet the frozen label-free atomic neutral projector loses
0.667pp on original LOFO and loses on all retrospective transfer domains.
The target signal exists; null geometry does not identify it.

Therefore `lambda approximately 1` remains a valid **nuisance-rejection
heuristic** at the current evidence level, not a general target-identification
principle.  Any future group-free method needs an additional label-free source
of target orientation—an analogue of a steering model—not another refinement
of bulk eigenvalue selection alone.
