# Atomic Neutral-Residual Projector CS-IU — candidate v1 freeze

**Status:** frozen candidate for retrospective audit; not an approved detector.

**Freeze time:** 2026-08-13, before computing or inspecting any correctness
metric for this candidate or its controls. The development datasets' labels
had already been opened by earlier project experiments, so this is a
code/formula freeze—not a claim of fresh dataset blindness.

**Parent:** frozen NRM-CS-IU v1 remains untouched and is the incumbent control.

## Question and admissible information

This candidate tests whether the six hand-defined provenance families can be
removed from NRM.  It may use only the existing sign-oriented `mixed_v2`
features and the ordinary two-component IU-PCR fit.  Calibration and target
application receive no correctness labels, require no additional inference,
and remain affine in the input feature matrix.

Cross-cell coordinates are aligned by stable feature identity.  Feature
identity is not used semantically: any simultaneous permutation of feature
rows, names, weights, covariance coordinates, and direction coordinates must
leave scores unchanged.  No provenance-family name or registry is read by the
candidate implementation.

## Frozen source and eligibility

- Source: the already frozen 23-cell original roster used by NRM-CS-IU v1.
- Feature contract: `mixed_v2`.
- IU fit: `IU_FIT_DEFAULTS` from `spectral_utils/laplacian_upcr.py`.
- Eligibility: an atomic contribution must be present and nonconstant after
  IU residualization in all 23 source cells (`minimum_cell_fraction=1.0`).
- Target policy: all frozen eligible atoms must be present.  Missing atoms
  cause abstention/error; there is no target-time imputation or re-selection.

The 17 eligible atoms, in canonical lexical order, are:

`cusum_max`, `cusum_max_energy`, `cusum_max_spilled`, `epr`, `epr_energy`,
`epr_spilled`, `logprob_margin`, `mean_logprob_entropy`,
`mean_top1_logprob`, `min_energy`, `renyi_entropy_2`, `rpdi`, `sw_var_peak`,
`sw_var_peak_energy`, `sw_var_peak_spilled`, `topk_tail_mass`, `varentropy`.

The 13 frozen exclusions, solely because source coverage is incomplete, are:

`cusum_shift_idx`, `dominant_freq`, `high_band_power`, `hl_ratio`,
`hurst_exponent`, `low_band_power`, `min_spilled`, `pe_mean`,
`spectral_centroid`, `spectral_entropy`, `stft_max_high_power`,
`stft_spectral_entropy`, `trace_length`.

No atom was excluded for a small IU weight.  The minimum source-cell IU weight
relative to its cell maximum was 0.00531307, and no residual coordinate was
numerically inactive.

## Frozen operator

For source cell \(c\), atom \(i\), sign-oriented feature row \(F_{ci}\), and
ordinary IU-PCR weight \(w_{ci}\), define the atomic contribution

\[
h_{ci}=w_{ci}F_{ci}, \qquad b_c=\sum_i h_{ci}.
\]

Within each cell, standardize \(b_c\) and every \(h_{ci}\), regress each
standardized contribution on the standardized baseline, remove that component,
then standardize the residual.  This gives atomic residual matrix \(R_c\).
The source correlation is the equal-cell average

\[
C=\frac1{23}\sum_c R_c^\top R_c/n_c.
\]

The frozen spectrum is:

`0.001574950, 0.005777841, 0.025836910, 0.128885421, 0.221453750,`
`0.267540872, 0.332989533, 0.467459705, 0.553598826, 0.663183452,`
`0.827376018, 0.960684926, 1.025557331, 1.120627744, 1.272644071,`
`2.035597492, 7.089211159`.

### Neutral band

Independently permute every residual column within each source cell.  With
1,000 draws, seed 20260813, and two-sided alpha 0.05, use the 2.5th percentile
of the null minimum eigenvalue and 97.5th percentile of the null maximum
eigenvalue as a simultaneous interval.  The frozen interval is
[0.934489, 1.070026].  It retains both eigenvalues 0.960684926 and 1.025557331.

Let \(P_0\) be the projector onto all retained modes.  This projector, rather
than a single eigenvector closest to one, is mandatory because the latter can
switch within a near-degenerate neutral subspace.

### Symmetric anchor and direction

Define a nonsemantic redundancy-adjusted anchor

\[
a_i \propto \left(\sum_j |C_{ij}|\right)^{-1}, \qquad \|a\|_2=1,
\]

and the direction

\[
d=P_0a/\|P_0a\|_2,
\]

oriented so \(d^\top a>0\).  The retained anchor norm is 0.456515.  In the
eligible-feature order above, the frozen direction is:

`[-0.107854785, +0.097158436, +0.128394676, +0.025738803,`
` -0.209089851, +0.330406947, -0.159090866, +0.056423468,`
` +0.064316494, +0.205199301, +0.062264817, +0.187167762,`
` -0.227278683, +0.642956912, +0.437147716, +0.197509649,`
` +0.009468708]`.

Direction SHA-256:
`d7de9faeb68825ac540cbaa70868aeb52dcf548d6b7e11e480664f446e952edb`.

## Frozen target application and scale

On a target cell, refit ordinary IU-PCR and the contribution standardization
using target telemetry only.  Apply the frozen direction to the target atomic
residuals.  Normalize the raw correction on the same unlabeled target rows to
standard deviation

\[
1/\sqrt{p}=1/\sqrt{17}.
\]

This is the standard deviation of an equal atomic average under the retained
identity-covariance null.  It is fixed now and must not be tuned after labels.
The final score is the standardized IU score plus this orthogonal correction.
The implementation must also return the exactly equivalent feature-weight
vector and intercept.

## Frozen controls and decision rule

Retrospective development may compare:

1. ordinary IU;
2. untouched family NRM-CS-IU v1;
3. atomic single eigenvector closest to one;
4. the same neutral projector with an equal-atom anchor;
5. random partitions matched to the frozen family-size profile;
6. deterministic provenance-family refinements and coarsenings;
7. a grouping learned only from source residual dependence;
8. feature-order and family-name permutations.

The proposed formula is the inverse-absolute-dependence anchored projector
above.  The retrospective controls may falsify it, but they may not change its
direction, sign, scale, exclusions, source roster, null construction, or target
policy.  If it is not clearly at least as robust as family NRM, the bounded
conclusion is that provenance grouping remains an assumption required by the
observed evidence.  If it survives development, all score files and hashes
must be written before labels are opened on one genuinely untouched external
target.  There is no post-label pivot.

## Structural evidence and immutable references

- Neutral dimension: 2.
- Leave-one-source-cell direction absolute cosine: minimum 0.975505, median
  0.994124, maximum 0.999434.
- Source covariance SHA-256:
  `9ef9010eb9f7969831603db2ff0a484c77c1b05b9455901e0e83449b707ca3ed`.
- Input bundle SHA-256:
  `693a5b634f975ea32c7f840f3ab8366dd8ad638fe41cc76a60e24b1ac5a013e1`.
- Calibration artifact SHA-256:
  `586e9b33d2e6392c09598e7f0d4323e3a2341e8925e0fd053611d1205af5082a`.
- Structural JSON SHA-256:
  `0cd118e0d487f67af289898e1ce153044beb89247edf12bc93710541c54edb78`.
- Freeze base commit: `c476795d7c379966cb36e55357b98b928032cffa`.
- Candidate module SHA-256 at freeze:
  `6acf0810b01dd9053b62966f1c9dc555c36ec2928ad90138b3e3a551094ff4c4`.
- Structural script SHA-256 at freeze:
  `dcb17ed3aaf6819d83a7a17b8fbe619f3c9e887529333fd4cc147f45b0cd417f`.

The lambda-near-one construction is only a null-geometry rule: it excludes
strong shared dependence and near-deterministic redundancy.  Neither this
geometry, the permutation interval, nor the symmetric anchor proves that the
retained direction identifies hallucinations.  Target identification is an
empirical claim evaluated separately under the protocol above.

**Post-audit administrative clarification:** the freeze paragraph now states
explicitly that development labels were historically open. No operator,
direction, scale, exclusion, source, control, or decision rule changed. The
pre-metric spec SHA-256 recorded by the retrospective run is
`6051e8e133a43ad2dc1a03d627a8cb42a5fb519427433ed91c2cdb8fe1b17673`.
