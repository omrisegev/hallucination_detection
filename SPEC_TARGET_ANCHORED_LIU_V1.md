# Target-Anchored Laplacian IU-PCR — synthetic research specification

Status: **revised after independent plan review; synthetic data only**

## Why this cycle exists

The approved Laplacian IU-PCR v3 study established two facts:

1. A DUFS-derived graph can improve IU-PCR when its geometry agrees with the
   correctness target, beyond ordinary projected ridge and graph controls.
2. The same mechanism harms performance when a cleaner nuisance manifold is
   unrelated to correctness.

The consumed development seeds show that no existing label-free reliability
quantity fixes this failure. In the informative nuisance world, role means were:

| label-free quantity | signal | nuisance |
|---|---:|---:|
| adapted-DUFS gate probability | 0.900 | 0.999 |
| normalized positive IU-PCR rho | 0.644 | 0.961 |
| correlation with the IU-PCR score | 0.533 | 0.795 |
| leave-one-feature-out consensus correlation | 0.330 | 0.630 |

Multiplying DUFS by rho or consensus would therefore strengthen the wrong
manifold.

## Identifiability claim to test

Let `g` and `u` be identically distributed independent latent variables, and
let the feature matrix contain one block measuring `g` and another measuring
`u`. The same unlabeled feature distribution is compatible with correctness
being a function of either latent. Swapping the names of `g` and `u` changes the
target but not the observable feature distribution. Therefore no fully
unsupervised graph identifier can be guaranteed to select the correctness
manifold in both worlds.

This is not a weakness specific to DUFS. It is an information boundary. The
experiment will instantiate both targets on the **same feature matrices**.

## Literature basis

- Dror et al., *Unsupervised Ensemble Regression*: IU-PCR estimates expert
  reliability under an error model but does not identify an arbitrary external
  target among exchangeable latent structures.
- Tenzer et al., *Crowdsourcing Regression: A Spectral Approach*: explicitly
  proposes limited-label semi-supervised ensemble learning as future work.
- Lindenbaum et al., *Differentiable Unsupervised Feature Selection based on a
  Gated Laplacian*: provides the stochastic-gate/Laplacian mechanism.
- Shaham et al., *Deep Unsupervised Feature Selection by Discarding Nuisance
  and Correlated Features*: shows that nuisance dimensions can corrupt the
  Laplacian. We infer from this—not claim that the paper proves—that an
  unsupervised geometry criterion need not equal target relevance.
- Lee et al., *Self-Supervision Enhanced Feature Selection*: states directly
  that fully unsupervised selection can miss target-relevant features and uses
  labels to guide the final selection phase.
- Cohen et al., *Few-Sample Feature Selection via Feature Manifold Learning*:
  and Lee et al. motivate testing few-label target guidance; neither validates
  TA-LIU's univariate point-biserial gate.

TA-LIU is our construction. It is not a paper implementation and is not a new
dependent-error model.

## Proposed method: TA-LIU

Target-Anchored Laplacian IU-PCR uses a small calibration set only to identify
the graph coordinates. IU-PCR's ordinary covariance, rho estimator, U2
subspace, and final Laplacian solve remain unchanged.

For calibration indices `A` and oriented feature `i`, compute the point-biserial
correlation

```text
r_i = Corr(F_i[A], Y[A]).
```

The continuous graph coordinate gate is

```text
q_i = max(r_i, 0) / RMS(max(r, 0)).
```

If the calibration target has one class, or every positive correlation is
numerically zero, fall back to the ungated graph and record the reason. A zero
gate removes that coordinate from graph distances, but no feature is removed
from IU-PCR's ordinary rho estimate, U2, or final fusion. There is no tuned gate
temperature or sparsity coefficient. Feature orientation is fixed by the
upstream synthetic feature contract and may not use calibration labels.

This is precisely transductive: feature vectors from calibration and evaluation
samples enter ordinary IU-PCR covariance estimation and graph construction;
only calibration labels enter target gates or supervised controls. Performance
uses only the identical non-calibration complement for every method in a draw.
This is a semi-supervised protocol, not an unsupervised method.

## Controls

- Ordinary IU-PCR (`lambda=0`).
- Adapted-DUFS Laplacian IU-PCR at the already frozen `lambda=0.1`.
- Trace-matched projected ridge at `lambda=0.1`.
- Full-data continuous pseudo-anchor: graph gates are positive Pearson
  correlations with the ordinary IU-PCR score over all `n` feature vectors.
  Apply the identical positive clipping, RMS normalization, and all-zero
  ungated fallback as TA-LIU. This is the strongest label-free pseudo-target
  available to this construction.
- Same-budget **U2 logistic**: fixed-L2 (`C=1`) logistic regression on the two
  ordinary unlabeled PCR coordinates, trained only on the calibration indices.
- Same-budget **full logistic**: fixed-L2 (`C=1`) logistic regression on all 12
  oriented features, trained only on the identical calibration indices.
  One-class calibration draws return the empirical class-prior constant and
  record the fallback.
- Oracle-latent graph, synthetic ceiling only.
- TA-LIU label budgets `k in {4, 8, 16, 32, 64}`.

The primary candidate is frozen as **TA-LIU with k=16** before confirmation.
Other budgets form a sample-efficiency curve and cannot replace the primary
candidate after results are seen.

The supervised controls are necessary attribution tests. TA-LIU must be
noninferior to the strongest method receiving the same labels; otherwise the
result only shows that label injection helps, not that Laplacian fusion is a
label-efficient use of those labels.

The logistic pipeline is fully frozen. U2 coordinates are centered and scaled
to unit variance using all `n` transductive feature vectors without labels. Full
features already have zero mean and unit variance under the frozen synthetic
feature contract. Both controls use scikit-learn `LogisticRegression` with
`penalty="l2"`, `C=1.0`, `fit_intercept=True`, `class_weight=None`,
`solver="lbfgs"`, `max_iter=1000`, `tol=1e-8`, and `random_state=0`. A
one-class calibration prefix returns that class probability as a constant score
for every evaluation sample and records a fallback.

## Synthetic tasks

1. `smooth_signal`: positive graph-alignment control.
2. `nuisance_manifold`: the earlier equal-size, moderately separated blocks;
   it tests the already-observed broad nuisance failure.
3. `selective_target_signal`: six moderately noisy measurements of target `g`
   and six very clean measurements of nuisance `u`; DUFS selectively follows
   `u`.
4. `selective_target_nuisance`: the **same feature matrices** as task 3, but
   correctness is generated from `u`. This is the label-swap identifiability
   pair.
5. `correlated_errors`: dependency stress test.
6. `pure_noise`: calibration-overfitting negative control.

Tasks 3 and 4 are generated once per replicate: one `F`, one pair of latent
variables and noisy targets, then two target views. Store a cryptographic hash
of `F`. The hash, ordinary IU-PCR weights/scores, adapted-DUFS gates, DUFS graph,
DUFS-LIU weights/scores, projected-ridge weights/scores, and continuous
pseudo-anchor weights/scores must be exactly identical across the pair. These
observational/output invariants establish the identifiability counterexample;
opposite AUROC movement is only an illustration and is not required.

## Split and repetition protocol

- Development seed block: `40,000`; it is already consumed by exploratory
  diagnosis and may be used only for code/debug/frozen-rule checks.
- Confirmation seed block: `2,600,000`; it must not be generated before the
  source/config fingerprint is frozen.
- Eight dataset replicates per task in each split.
- Sixteen deterministic calibration permutations per dataset. Budgets are
  nested prefixes `{4, 8, 16, 32, 64}` of each permutation. The paired targets
  use the exact same index lists even if a prefix contains only one class; that
  case uses the declared fallback rather than changing queried samples.
- Calibration repeats are averaged within dataset. The dataset replicate, not
  the 16 draws, is the uncertainty unit. The label-swap pair shares only eight
  independent feature matrices and is analyzed as paired data.
- Development and confirmation are separate commands. Confirmation refuses a
  changed spec, config, new experiment script/module/test, dependency-version,
  or source hash.

## Primary metrics and gates

AUROC on non-calibration samples is primary; AUPRC is secondary. Report raw
per-replicate paired changes, mean changes, win counts, and Student-t one-sided
95% lower bounds across the eight dataset replicates. Also report leave-one-
replicate-out means as sensitivity diagnostics. Calibration-draw variability is
diagnostic only and never inflates the effective sample size.

TA-LIU k=16 earns a future real-data discussion only if all hold:

1. **Selective-nuisance rescue:** versus both IU-PCR and the continuous
   pseudo-anchor when `g` is the target: mean gain at least `+0.5` AUROC points,
   positive lower bound, and at least six of eight dataset wins.
2. **Label-swap consistency:** the same thresholds versus IU-PCR when `u` is the
   target on the paired identical feature matrices.
3. **Same-label attribution:** on each paired target, two separate preregistered
   contrasts must pass: the lower bound of TA-LIU minus U2 logistic is no worse
   than `-0.5` points, and the lower bound of TA-LIU minus full logistic is no
   worse than `-0.5` points. No comparator is selected by evaluation AUROC.
4. **Smooth preservation:** lower bound of TA-LIU minus adapted-DUFS LIU is no
   worse than `-0.5` points.
5. **Existing nuisance safety:** lower bound versus IU-PCR is no worse than
   `-0.5` points.
6. **Correlated-error safety:** lower bound versus IU-PCR is no worse than
   `-0.5` points. TA-LIU must not be described as solving dependent errors.
7. **Null safety:** the two-sided Student-t 95% interval for pure-noise AUROC is
   contained in `[0.45, 0.55]`, the interval for TA-LIU-minus-IU is contained in
   `[-0.5, +0.5]` AUROC points, and its one-sided lower bound is not positive.
8. **Identifiability invariants:** exact same-matrix hashes and bitwise equality
   of all declared label-free artifacts across paired targets.

Target-anchored gates must switch planted roles with the calibration target.
This is a mechanism diagnostic; the exact observational/output invariants—not
opposite AUROC signs—establish the identifiability result.

## Required visualizations

1. AUROC change versus calibration-label budget for every task.
2. Paired label-swap comparison on identical feature matrices.
3. Signal-versus-nuisance gate magnitude versus label budget.
4. Frozen k=16 candidate against all controls.

After the synthetic confirmation, an independent read-only agent must audit
the mathematics, identical-matrix pairing, calibration/evaluation separation,
effective sample size, source freeze, gates, raw summaries, and plots. Stop for
discussion regardless of pass or fail. Do not open real hallucination data.
