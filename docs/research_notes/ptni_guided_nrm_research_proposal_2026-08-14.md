# PTNI-Guided Neutral Residual Mode

## Motivation, candidate designs, and a prospective experimental plan

**Date:** 2026-08-14
**Status:** prospective research note; not a preregistration and not an experimental result
**Scope:** a possible successor experiment after the frozen PTNI-IU/A6 program. This note does not modify A6, its estimator, its gates, or its execution order.
**Continuation handoff:** see `HANDOFF_A6_S0B_TO_CLAUDE_2026_08_15.md`; this proposal is explicitly included in the post-A6 alternative review requested there.

## 1. Research question

Can the intervention-derived target and nuisance information in PTNI-IU remove the manual-family and mode-selection weaknesses of Neutral Residual Mode (NRM), while preserving the useful spectral regularization that motivated NRM?

The intended hybrid is not a second detector stacked on top of PTNI or IU-PCR. It must still reduce to one affine head over the frozen mixed-v2 atomic coordinates, with exact IU-PCR as the zero-trust fallback.

## 2. Background

### 2.1 IU-PCR is the common anchor

Both NRM and PTNI-IU begin from the same target-local IU-PCR score. Let

\[
s_{IU}(z)=b_{IU}+u^Tz,
\]

where \(z\) is the frozen mixed-v2 representation of the present subset of the nominal 30 atomic features. Any proposed correction must add information not already represented by \(u\), preserve a path containing exact IU-PCR, and collapse to one effective weight vector and intercept at deployment.

### 2.2 What Family-NRM contributed

Family-NRM was motivated by the HARP-style principle that a useful target representation may become easier to identify after removing a known shared or semantic subspace. In our implementation, the available structural anchor was feature provenance rather than hidden-state or unembedding geometry.

Family-NRM therefore:

1. grouped engineered features into six provenance families;
2. formed family contributions to the IU score;
3. residualized those contributions against the shared IU score;
4. standardized the six residual coordinates;
5. estimated their unlabelled cross-environment covariance;
6. selected the eigenvector whose eigenvalue was closest to the unit-variance null; and
7. applied a small, fixed-scale IU-orthogonal correction.

This recovered all six signs of a supervised source teacher and produced positive frozen transfer results, including PRMBench. It established that the family residual space contains useful target-aligned information.

### 2.3 Why NRM is not yet a satisfactory group-free method

NRM has two linked weaknesses.

**Manual structural prior.** The six families are deterministic and label-independent, but they are still hand-defined provenance blocks. They are not statistically identified error-factor groups and they break feature-permutation invariance.

**Neutral-mode ambiguity.** For six families, the eigenvalue closest to one happened to be distinguished enough to define a useful direction. At atomic resolution, many sample eigenvalues lie in a noise-like bulk around one. Individual eigenvectors inside this bulk may rotate sharply while the subspace projector remains stable. The frozen Atomic-NRM study confirmed the problem: atomic residuals contained more supervised target information than family residuals, but null geometry alone did not identify the target direction and the label-free atomic candidate failed transfer.

The resulting diagnosis is:

> NRM can reject strong dependence and redundancy, but it lacks a defensible group-free steering signal that identifies which remaining direction is related to hallucination correctness.

### 2.4 What PTNI-IU contributes

PTNI-IU supplies the missing steering information through mechanically verified interventions rather than natural hallucination labels.

Its reciprocal 2x2 prompt-response crossover ensures that every prompt and every response appears equally often in valid and invalid combinations. Target contrasts measure the feature change when a fixed response changes from valid to invalid relative to the prompt. Nuisance contrasts measure changes caused by semantics-preserving rendering, formatting, or notation. Target-by-render interactions test whether the target effect survives nuisance variation.

PTNI therefore provides:

- a mechanically oriented target contrast;
- an empirical nuisance covariance;
- held-family and held-scorer tests of target transfer;
- a direct criterion for rejecting directions dominated by rendering nuisance; and
- an exact IU-anchored trust path.

This is not unsupervised in the strict covariance-only sense. It is intervention-supervised or mechanically supervised, without natural correctness labels during fitting or selection.

## 3. Why combine PTNI and NRM?

The methods address complementary parts of the problem:

| Component | Primary contribution | Primary limitation |
|---|---|---|
| Family-NRM | Spectral rejection of dependence and redundancy | Manual families and no group-free target orientation |
| Atomic NRM | Permutation-calibrated neutral subspace | Neutral geometry does not identify correctness |
| PTNI-IU | Mechanically identified target versus nuisance direction | May overfit intervention mechanics or retain unstable residual directions |

The proposed hybrid asks whether PTNI can act as the missing **steering model** inside an NRM-derived stable subspace. In that interpretation:

- NRM defines where strong redundancy and dependence should not dominate;
- PTNI defines what target-related direction to seek; and
- IU-PCR remains the deployment anchor and fallback.

This is scientifically preferable to claiming that an eigenvalue near one identifies hallucination by itself.

## 4. Candidate hybrid designs

### 4.1 Option A: intervention-derived soft groups

For every atomic feature, estimate a response fingerprint consisting of its target margin, nuisance responses, target-by-render interactions, scorer stability, and mutation-family stability. Cluster features using only these mechanically generated fingerprints. Freeze the learned groups on source scorers and then apply the original family-level NRM procedure.

**Potential benefit:** replaces handwritten provenance families with data-derived groups.
**Main risk:** clustering introduces a new model-selection problem and may simply repackage PTNI supervision as another unstable partition.

This option is useful as a diagnostic but is not the preferred primary method.

### 4.2 Option B: PTNI selection among NRM modes

Construct several NRM modes or a permutation-calibrated neutral subspace, then use nested PTNI intervention criteria to select a mode or a fixed combination of modes.

**Potential benefit:** directly removes the `argmin |lambda-1|` rule.
**Main risk:** selecting one basis vector remains unstable when eigenvalues are clustered. It may also create excessive researcher degrees of freedom unless the mode-combination rule is frozen before evaluation.

A whole-subspace projection is more defensible than single-mode selection.

### 4.3 Option C: PTNI-steered neutral residual subspace — recommended

The recommended design keeps atomic coordinates throughout and uses the NRM geometry only as a stable projector.

1. Fit the ordinary target-local IU-PCR anchor \(u\) without labels.
2. Form atomic IU-residual coordinates using only training-source data.
3. Estimate the residual covariance and its permutation-null spectrum.
4. Define a frozen neutral-subspace projector \(P_{N}\), retaining the complete noise-like eigenvalue cluster rather than one arbitrary eigenvector.
5. Estimate the PTNI target steering direction from reciprocal target contrasts while nuisance-whitening with the nuisance and interaction moments.
6. Project that steering direction into the neutral residual subspace:

\[
r_{hybrid,0}=P_N r_{PTNI}.
\]

7. Remove any remaining component parallel to target-local IU in the registered covariance metric, normalize at the registered IU trust scale, and define

\[
w_{hybrid}(\alpha)=u+\alpha r_{hybrid,\perp}.
\]

8. Keep \(\alpha=0\) as exact IU-PCR. If the projected target evidence is degenerate, unstable, or fails the nuisance gates, return exact IU rather than choosing a rescue mode.

An equivalent implementation may deflate the clearly non-neutral residual spikes before fitting the PTNI direction, provided the projector, order of operations, and normalization are frozen in advance. Both formulations must be compared numerically because projection and nuisance whitening need not commute.

## 5. Core hypotheses

### H1 — group-free identification

PTNI steering will recover a stable target-aligned correction in atomic residual space without using the six manual provenance families.

### H2 — incremental spectral regularization

The neutral residual projector will improve PTNI's held-family or held-scorer stability beyond PTNI alone, rather than merely shrinking the correction toward zero.

### H3 — nuisance suppression

The hybrid will preserve PTNI's target margin while reducing correction drift under nuisance renderings and target-by-render interactions.

### H4 — exact safe fallback

Under null, nuisance-only, missingness, or geometrically degenerate conditions, the selected hybrid will return exact IU-PCR at least as reliably as PTNI alone.

## 6. Experimental comparison

All candidates must use the same source groups, target-local transforms, IU anchors, outer/inner folds, alpha/ridge grids, bootstrap units, and held-scorer boundary. No natural correctness label may choose the neutral band, projector, steering direction, trust scale, or finalist.

### 6.1 Required primary arms

1. **IU-PCR:** exact `alpha=0` baseline.
2. **Frozen Family-NRM:** comparison only; never a selector or orienter.
3. **Frozen Atomic NRM:** negative structural reference showing what null geometry achieves without PTNI steering.
4. **PTNI-IU:** the frozen A6 method.
5. **PTNI-guided NRM:** the proposed hybrid.

### 6.2 Mechanism controls

6. PTNI direction projected into a cardinality-matched random subspace.
7. PTNI direction projected into a dependence-spike subspace rather than the neutral subspace.
8. Neutral projector with shuffled target polarity and full refitting.
9. Neutral projector with nuisance contrasts treated as target.
10. PTNI nuisance whitening without the NRM projector.
11. NRM projector with a symmetric or all-ones anchor instead of PTNI steering.

These controls distinguish genuine complementary structure from generic shrinkage, lower effective dimension, or additional fitting flexibility.

## 7. Evaluation sequence

This hybrid must not be inserted into the already frozen A6 run. It requires a new preregistered stage and an independent no-edit review before any result is opened.

### Stage 0 — algebra and implementation

- exact feature-name permutation equivariance;
- exact IU reconstruction at `alpha=0`;
- one affine transformed-coordinate weight and intercept;
- exact-duplicate and near-duplicate behavior;
- missingness and deletion behavior;
- deterministic projector and eigenspace tie handling;
- no natural-label object accepted by fit or selection APIs.

### Stage 1 — sealed simulation

Use the same target-only, nuisance-only, target-plus-nuisance, stronger-nuisance, family-specific, null, missingness, and near-redundancy worlds used to validate PTNI, with additional worlds in which:

- the target lies inside the neutral subspace;
- the target partly lies outside the neutral subspace;
- a nuisance direction lies inside the neutral subspace; and
- the neutral eigenspace has repeated or nearly repeated eigenvalues.

The hybrid must prefer target over nuisance, preserve exact fallback, and not pass solely by shrinking toward IU.

### Stage 2 — mechanical intervention premise

Run nested source-group evaluation on Qwen and one-way held-scorer evaluation on Llama. Report target ordering, correction margin, nuisance RMS, interaction RMS, leave-one-target-family transfer, leave-one-nuisance-family transfer, direction stability, retained PTNI norm after projection, and selected trust.

### Stage 3 — one-way natural-response veto

Freeze exactly one hybrid artifact before opening retrospective natural correctness. Labels may veto the candidate but may not choose the neutral band, alter the projector, change the trust scale, or select between projection orders.

### Stage 4 — independent confirmation

Only a candidate that passes every earlier stage may enter a separately sealed confirmation. It must be compared with the exact frozen PTNI and Family-NRM artifacts under identical examples and resampling units.

## 8. Minimum decision gates

Exact thresholds require a separate preregistration, but the following logical gates are mandatory.

1. **Nondegenerate retained signal:** the neutral projector must retain a predeclared minimum fraction of the PTNI correction norm and target margin. Otherwise the hybrid closes rather than reporting shrinkage as success.
2. **Material increment over IU-PCR:** the one frozen hybrid must improve over exact IU-PCR by a preregistered material margin on held natural hallucination detection, with a positive paired grouped-bootstrap lower bound and no registered domain-level material regression. A structural or mechanical-only gain is insufficient for promotion.
3. **Increment over PTNI:** a hybrid claim requires paired improvement over PTNI alone on the primary held mechanical statistic with a positive lower confidence bound and no registered cell-level material regression.
4. **Increment over Family-NRM:** a group-free replacement claim requires noninferiority to frozen Family-NRM on held natural evaluation and strict improvement on at least one preregistered transfer criterion.
5. **Nuisance benefit:** nuisance and interaction drift must improve or remain within a tight noninferiority margin relative to PTNI.
6. **Stability:** the neutral projector and final correction must remain stable under outer folds, leave-family-out fits, feature permutations, and close-eigenvalue rotations.
7. **Null behavior:** polarity-shuffle, nuisance-as-target, and matched-random-subspace controls must not produce a promotable correction; exact IU fallback must dominate when target evidence is absent.
8. **No capacity-only explanation:** the hybrid must beat cardinality-matched random projectors and simple norm-matched PTNI shrinkage.

## 9. Interpretation of possible outcomes

### Hybrid beats PTNI and Family-NRM

This would support the claim that PTNI supplies target identification while NRM supplies useful group-free spectral regularization. It would justify replacing the manual family partition with an intervention-steered atomic subspace.

### Hybrid matches PTNI but beats NRM

PTNI solved the identification problem, but NRM added no useful information. Prefer PTNI for parsimony.

### Hybrid beats Family-NRM but not PTNI

The manual groups are no longer necessary, but the neutral projector is also unnecessary. Prefer PTNI.

### Hybrid helps only after natural-label tuning

Reject the method as adaptively supervised. The hybrid may remain a supervised diagnostic but cannot be claimed as self-supervised or intervention-selected.

### Hybrid fails because the projector removes target information

This confirms the Atomic-NRM warning: noise-like spectral geometry is not a safe target filter. Retain PTNI or IU and close the hybrid route.

### PTNI fails its own A6 identification or transfer gates

Do not use NRM as a rescue layer. A spectral projector cannot repair the absence of a valid intervention-derived steering signal. The hybrid should open only if PTNI establishes the target premise but leaves a preregistered stability or nuisance-regularization question.

## 10. Main risks

- **Projection can remove the target.** The true hallucination direction may be a covariance spike rather than a neutral mode.
- **Mechanical-task shortcut transfer.** PTNI steering may identify prompt-response compatibility in constructed tasks but not natural hallucination.
- **Double use of source data.** Learning both the neutral projector and PTNI steering on the same groups may create optimistic stability unless every operation is nested inside source-group folds.
- **Hidden model-selection freedom.** Neutral-band definitions, projection order, ridge, trust, and subspace dimension must be frozen before held evaluation.
- **Loss of the original NRM claim.** Once PTNI supplies orientation, the hybrid is mechanically self-supervised, not covariance-only label-free NRM.
- **Redundant complexity.** If PTNI already nuisance-whitens effectively, the NRM projector may only add shrinkage and numerical instability.

## 11. Recommended next step

Do not modify or delay A6. First complete the frozen PTNI-IU program and preserve its outcome exactly.

If PTNI establishes a valid intervention target direction but shows a preregistered stability, redundancy, or nuisance-transfer limitation, write a separate protocol for **PTNI-steered neutral residual subspace**. Freeze one projector rule, one projection order, one trust path, and the comparison against PTNI alone before opening any corresponding result.

Beating IU-PCR materially on held natural hallucination detection is necessary but not sufficient. The additional hybrid-specific scientific test is whether NRM contributes reproducible value **beyond PTNI**, after target identification has already been supplied by interventions.
