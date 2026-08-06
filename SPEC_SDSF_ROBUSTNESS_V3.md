# SDSF robustness v3: stability before flexibility

## Scientific question

The fixed feature-orientation contract removed the known sign(rho) failure, but
it did not make the dependency-weighted solve uniformly safe on the real
artifact: fixed-stable SDSF is 4.71 AUROC points below matched SU-PCR on average
(2 wins, 22 losses).  The next question is therefore not whether correlated
feature errors exist.  It is:

> Can dependency information outside U-PCR's leading two-dimensional signal
> subspace be admitted only when it is reproducible under label-free row
> resampling, while retaining SDSF's gain in worlds where that information is
> real?

This is a narrower and falsifiable claim.  It does not claim that synthetic
covariance worlds represent hallucination data.

## Why this direction follows from the literature

1. Tenzer et al.'s U-PCR model writes the expert covariance as a rank-at-most-two
   signal plus correlated-error covariance.  SU-PCR already handles sparse
   correlated errors, so merely adding low-rank-plus-sparse decomposition is a
   baseline, not our contribution.  Its exact uniqueness condition is also very
   restrictive: fewer than `(m-1)/2` corrupted pairs.
2. Candès et al.'s robust PCA and stable principal-component pursuit justify
   separating low-rank structure from sparse corruption and explicitly show why
   noisy recovery is approximate rather than exact.  They do not justify
   inverting every recovered direction in a downstream score.
3. Covariance and singular-value shrinkage work by Donoho, Gavish, Johnstone and
   related robust covariance estimators shows that noisy spectral directions
   should be attenuated according to estimation quality.  This motivates
   shrinkage, not a hard tail cutoff selected after seeing labels.
4. Lindenbaum, Salhov, Averbuch and Kluger's l0-CCA selects information that is
   reproducible across measurement channels using a label-free total-correlation
   criterion.  Their high-dimensional feature-selection regime differs from
   ours, but its reusable principle is *cross-view reproducibility*.
5. Lindenbaum and Averbuch's multi-view diffusion-map work similarly constructs
   consensus from structure shared between views.  For our current experiment,
   row-bootstrap consensus is the simplest version that does not invent a
   questionable synthetic channel split.  A real-data follow-up should test the
   repository's predeclared entropy-trace versus energy/log-probability channels.
6. DEEM models dependent binary classifiers with a deep energy model.  It is a
   useful nonlinear comparator, but it changes the data representation,
   objective, and optimizer simultaneously.  The current failure is already
   visible in a linear continuous solver, so a small spectral stabilization is
   the more diagnostic next experiment.

Primary sources:

- Tenzer et al., *Crowdsourcing Regression: A Spectral Approach*:
  https://proceedings.mlr.press/v151/tenzer22a.html
- Candès et al., *Robust Principal Component Analysis?*:
  https://arxiv.org/abs/0912.3599
- Zhou et al., *Stable Principal Component Pursuit*:
  https://arxiv.org/abs/1001.2363
- Donoho, Gavish and Johnstone, optimal covariance shrinkage:
  https://arxiv.org/abs/1311.0851
- Gavish and Donoho, optimal singular-value shrinkage:
  https://arxiv.org/abs/1405.7511
- Lindenbaum et al., l0-based sparse CCA: local digest
  `papers/digests/l0-based-sparse-canonical-correlation-analysis.md`
- Jaffe et al., dependent unsupervised ensembles:
  https://proceedings.mlr.press/v51/jaffe16.html
- Varma et al., inverse-covariance dependency discovery:
  https://proceedings.mlr.press/v97/varma19a.html
- DEEM: https://arxiv.org/abs/2601.20556

## Hypotheses

**H1 — mechanism.** In planted sparse-error worlds, current SDSF beats SU-PCR.

**H2 — innovation.** Bootstrap reliability shrinkage improves current SDSF on
the pooled primary worlds, with a paired 95% confidence interval excluding zero.

**H3 — safety.** The new candidate loses no more than 0.5 AUROC points on average
in the clean world and its primary-world 5th percentile is at least -2 points.

**H4 — boundary.** Small-sample and dense-dependence worlds are reported even
though they do not determine promotion.  A failure there identifies the next
assumption to revise; it must not be hidden by averaging.

## Candidate algorithm: stability-shrunk SDSF

1. Fit the existing SU-PCR/SDSF decomposition on the complete, confidence-
   oriented feature matrix.
2. Bootstrap rows and refit the same label-free reliability estimator.
3. Resolve only the unavoidable global sign against the full-data rho.
4. Express every bootstrap rho in the structured-covariance eigenbasis.
5. Preserve the two U-PCR signal coordinates.  For each tail coordinate use
   `kappa = SNR^2 / (SNR^2 + tau^2)`.
6. Optionally shrink off-diagonal covariance toward its diagonal, then solve the
   same condition-controlled system as SDSF.

No correctness labels, AUROC values, or per-cell choices enter these steps.

## Immutable research loop

The executable specification is
`scripts/sdsf_robustness_autoresearch.py`.  It freezes:

- six worlds: clean Gaussian, sparse small-sample, sparse Gaussian, sparse
  heavy-tailed t4, sparse 2% cell-contaminated, and dense block dependence;
- ten candidates and their order;
- 12 development repetitions and 24 disjoint sealed-validation repetitions;
- a robust development utility that prices the primary lower tail;
- promotion of exactly the top three development candidates;
- validation gates against both SU-PCR and current SDSF.

Unlike Karpathy's autoresearch loop, this process does not commit each trial,
reset git, or repeatedly optimize one exposed validation metric.  It borrows the
useful parts—small auditable changes, a machine-readable ledger, and persistent
iteration—while keeping development and validation evidence disjoint.

## Frozen interpretation rules

A promoted candidate advances only when all conditions hold on sealed seeds:

- pooled primary delta versus SU-PCR is positive and its paired CI lower bound
  is non-negative;
- pooled primary delta versus current SDSF is positive and its paired CI lower
  bound is non-negative;
- primary 5th percentile is at least -0.020 AUROC;
- clean-world mean delta versus SU-PCR is at least -0.005 AUROC.

Passing licenses one real-data experiment.  It is not evidence about LLM
hallucinations.  Failure requires a versioned hypothesis and disjoint seed
namespace; it does not authorize tuning on sealed results.

## Result of the frozen run

The top candidate was bootstrap tail shrinkage with `tau=0.5`.  On 72 sealed
primary repetitions it improved:

- SU-PCR by **+0.00802 AUROC** (95% CI **[+0.00727, +0.00874]**);
- current SDSF by **+0.00040 AUROC** (95% CI **[+0.00014, +0.00070]**).

Its primary 5th percentile versus SU-PCR was +0.00227.  It was essentially tied
in the clean world (+0.000001) and in dense-block stress (-0.000083), and improved
the small-sample sparse world by +0.00693.  This is a real but **incremental**
stabilization result; the dominant synthetic gain still comes from SDSF itself.
The correct next action is a matched real-data factorial, not a stronger claim.

Full evidence: `results/sdsf_robustness_v3/REPORT.md` and `summary.json`.

