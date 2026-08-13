# Automatic group-free IU research program v1

**Date:** 2026-08-13

**Status:** active research contract; A0 passed and is frozen, A1 is active

**Primary objective:** replace the hand-defined provenance quotient used by
NRM-CS-IU with an automatically identified correction that improves IU-PCR
for hallucination detection while preserving the deployment contract.

## 1. Scope and non-negotiable contract

The primary S1 method must:

1. use the frozen one-pass mixed-v2 telemetry available to IU-PCR;
2. fit without correctness labels;
3. contain no runtime or calibration dependency on `FEATURE_TO_VIEW`,
   `VIEW_ORDER`, family names, or a hand-authored equivalent;
4. remain a fusion rule: one affine score over the original feature matrix,
   with a numerical reconstruction check;
5. use no extra model pass at deployment;
6. allow cached cross-model material, feature-construction metadata, and
   environment identities during calibration, provided each is declared;
7. return exactly to IU-PCR when the auxiliary structural evidence is absent
   or fails its registered reliability gate.

Family NRM remains frozen as the strongest current label-free correction and
as a comparator. It may not be used to choose, orient, or tune the automatic
candidate. The supervised atomic head remains a diagnostic ceiling only.

The research priority is out-of-domain detection performance subject to this
contract. Interpretability, simplicity, and computational cost break ties;
they do not justify selecting a weaker method.

## 2. The problem is three problems

Every candidate must state separately how it solves:

1. **measurement structure:** repeated or dependent measurements and their
   effective multiplicity;
2. **target identification:** which latent component is hallucination-related
   rather than difficulty, length, style, model size, or another nuisance;
3. **orientation and trust:** the global sign and the correction magnitude.

Covariance structure alone is not accepted as target identification. In
particular, closeness of an eigenvalue to one, membership in a null band,
stability, non-Gaussianity, or a large spectral gap cannot by itself name a
component as hallucination-related.

## 3. Success ladder

### S1 — strict label-free success

- no correctness labels in fitting or selection;
- no manual feature groups;
- automatic structure may use environments, mechanically derived feature-DAG
  metadata, and paired cached model views;
- one-pass affine deployment;
- on untouched confirmation, paired grouped-bootstrap improvement over IU-PCR
  has a 95% interval with lower endpoint greater than zero;
- automatic minus frozen Family-NRM has a lower endpoint no worse than
  -0.002 AUROC (-0.2 percentage points);
- no registered domain-family macro loses more than 0.010 AUROC versus IU-PCR.

### S2 — self-supervised success

The S1 deployment and performance gates remain. Calibration may additionally
use automatically generated paired interventions, but no human correctness
labels. The claim must be named self-supervised, not unsupervised.

### S3 — minimal-label orientation success

The representation and candidate components are frozen without labels. At
most 32 pre-sampled labeled calibration responses may choose a component,
global sign, and trust scale; they may not train an unrestricted atomic
classifier or select features. Report the complete 4/8/16/32 label curve.

Passing a lower tier does not retroactively count as passing a higher tier.

## 4. Data and label boundary

### 4.1 Development surface

All original, ProcessBench, SemGrad, PRMBench, HLE, and RAGTruth labels already
opened during Steps 247--253 are retrospective development evidence. A method
may be frozen and diagnosed on this surface, but no new confirmation claim may
be made from it.

### 4.2 Unlabelled calibration surface

The 23-cell NRM source roster is the initial strict calibration population.
Phase A0 must record exact sample counts, active features, missingness,
environment identity, model identity, dataset identity, prompt/item identity,
and cross-model pairing coverage before a structural model is fitted.

### 4.3 Confirmation surface

Phase A0 must nominate and seal a naturally distributed response-level
benchmark or model family whose labels were not used in method development.
At most one frozen finalist per supervision tier may be opened on it. If one
surface cannot support all three tiers without adaptive reuse, reserve a
second confirmation surface before opening the first.

## 5. Common label-free structural gates

Before any development labels are opened for a candidate, save a score bundle,
configuration, source-data hashes, feature order, fitted parameters, and code
hash. The following tests are mandatory:

1. feature-order permutation equivariance;
2. exact-duplicate and near-duplicate stress tests;
3. missing-feature and sparse pair-coverage behavior;
4. leave-one-environment-out loading/subspace stability;
5. environment-label shuffle and item-pair shuffle negative controls where
   applicable;
6. affine score reconstruction error below `1e-10`;
7. deterministic repeatability under the declared seeds;
8. zero-evidence fallback equal to IU-PCR to numerical precision.

Structural hyperparameters are selected only by held-out-environment
reconstruction or likelihood, stability, identifiability diagnostics, and the
stress tests above. AUROC may not select rank, block count, regularization,
anchor, sign, or trust.

## 6. Experiment registry and execution order

Every route receives an implemented premise test and a recorded decision. A
failed premise closes the route without a broad parameter sweep, but does not
stop the program.

### A0 — identifiability, data, and simulator audit

**Question:** Is the auxiliary structure needed by the live routes present in
the available data?

Deliverables:

- a mechanically derived feature-DAG registry with channel, operator,
  reduction, source stream, parameters, and dependencies;
- a 23-cell environment/missingness/pair-coverage manifest;
- an exact audit of prompt/item overlap across Qwen and Llama caches;
- a simulator with target, shared difficulty, environment nuisance,
  operator/channel crossed effects, duplicates, missingness, and optional
  environment-specific target directions;
- a selected untouched confirmation surface and immutable label boundary.

Gate: do not start A1 with an undocumented feature mapping or an unknown
cross-model pairing boundary.

**Frozen execution result (Step 255): PASS.** The label-blind audit recovered
30 canonical features across 23 source environments, with 17 features present
in every environment and feature-pair coverage ranging from 8 to 23 cells. Six
cells contain fewer valid mixed-v2 bundle rows than manifest attempts (minimum
retention 19.8%); subsequent structural fitting must preserve the bundle
population and equal-environment weighting. Exact content-and-ID pairing was
verified for 3,400 fixed ProcessBench responses scored by Qwen3-4B, Qwen3-8B,
and Llama3.1-8B. The feature DAG is derived from extractor registries and
function signatures and has no dependency on `FEATURE_TO_VIEW`. The reserved
confirmation cell is `semgrad-triviaqa-qwen3-4b-confirmation-v1`; its labels
remain unopened and the cell still requires collection. Canonical artifacts:
`results/automatic_group_free_phase_a0_v1/`.

### A1 — factorial soft measurement model

**Hypothesis:** mixed-v2 features form an incomplete crossed design of
measurement channel by computational operator; modelling both axes explains
repetition more faithfully than marginal-correlation clustering.

Compare, without labels:

- mechanically supplied axes versus axes learned from anonymized features;
- additive channel/operator effects versus interaction factors;
- soft loadings versus hard clustering;
- held-out-feature and held-out-environment reconstruction;
- duplicate-balanced effective weights versus PCA and random partitions.

Premise gate: the learned representation must beat pooled PCA and
cardinality-matched random partitions in held-out reconstruction, remain
stable under environment deletion, and avoid giving an exact duplicate extra
total mass. Otherwise close the factorial route as a detector basis while
retaining the audit as evidence.

### A2 — multi-environment joint block diagonalization

**Hypothesis:** latent mechanisms have shared loadings but distinct variance
profiles across environments, permitting recovery from the collection of
environment covariance matrices.

Compare atomic JBD, factorial-coordinate JBD, approximate joint
diagonalization, and joint block diagonalization with block count chosen by
held-out-environment fit. Include pooled-covariance and shuffled-environment
controls.

Premise gate: aligned blocks must be reproducible under leave-one-environment-
out, improve held-out covariance reconstruction, and lose the claimed
advantage when environment identities are shuffled.

### A3 — primary strict hybrid: factorial quotient plus JBD

Use A1 to control measurement multiplicity and A2 to learn environment-stable
blocks. No candidate is evaluated as a detector until its target-component,
orientation, and trust rules are frozen. The primary selector must derive from
A4's cross-model decomposition when pairing is adequate; a selector based
only on the JBD spectrum is prohibited.

### A4 — paired cross-model multi-view identification

**Hypothesis:** item difficulty is primarily shared across model views while a
response/model-specific hallucination component varies within a paired item.

Fit a hierarchical shared/individual source model on exact item matches. Hold
out entire model families and environments, then repeat after shuffling item
pairs. Candidate selection, sign, and trust must be functions of the fitted
multi-view model and a declared confidence anchor, not labels.

Premise gate: the shared source must disappear or degrade under pair shuffle;
the individual candidate must transfer to a held-out model family and must
not reduce to model size, answer length, or dataset identity.

If exact pairing is inadequate, record the coverage failure and continue to
A5 rather than creating approximate semantic matches post hoc.

### A5 — continuous weak-supervision dependency model

**Hypothesis:** correctness and feature dependencies can be estimated jointly
more reliably than first clustering marginal correlations and then fusing.

Implement continuous latent correctness with sparse dependencies, then add
multi-environment reliability and the optional A1 factorial prior. Compare
against the already tested L-SML/tetrad route, independent IU-PCR, and a
dependency-only clustering control.

Premise gate: recover latent reliability and dependencies in misspecified
synthetics, remain duplicate-stable, and improve held-out-environment
likelihood over the independent model. A result that depends entirely on a
false majority-better-than-random assumption is rejected.

At the end of A5, freeze at most one S1 finalist using the registered
structural criteria plus retrospective development performance. Do not open
the untouched confirmation yet if an S2 finalist will be compared on the same
surface.

### A6 — self-supervised intervention route

This phase opens only if no S1 method passes the development promotion gate,
or after the S1 finalist is frozen as a pre-registered comparison.

Create paired calibration changes of two types:

- **target-changing:** evidence removal, evidence contradiction, verified
  entity/number substitution, or checkable reasoning-step corruption;
- **nuisance-changing:** meaning-preserving paraphrase, formatting, evidence
  reorder, and length/style controls that should not change correctness.

Keep the answer tokens fixed whenever the intervention permits. Learn an
affine atomic direction that ranks target-corrupted above target-supported
pairs while remaining invariant to nuisance pairs, anchored to IU-PCR. At
deployment, apply the frozen direction to the original one-pass features; no
intervention is run.

Required falsification: random pairs, style-only controls,
leave-one-intervention-family-out, and evaluation on natural hallucinations
not produced by the intervention generator. The existing RAG evidence-
ablation result supplies premise evidence but is not a new confirmation.

Freeze at most one S2 finalist.

### A7 — tiny-label orientation

This phase opens only after the S1 and S2 outcomes are recorded. Freeze the
unlabelled representation first. Pre-sample 4/8/16/32 labels and allow them to
choose only component, sign, and trust. Use nested, class-balanced fitting and
average ranking metrics within folds; never compute AUROC after concatenating
uncalibrated out-of-fold scores.

Freeze at most one S3 finalist.

### A8 — domain-conditional shrinkage

Apply a hierarchical deviation only to the strongest frozen global candidate:

`w_e = w_global + alpha_e * delta_e`.

`alpha_e` must depend only on unlabelled calibration evidence and equal zero
when that evidence is insufficient. Evaluate both known-domain calibration
and leave-one-domain-out deployment. This is reported as a separate batch-
calibration setting, not silently folded into the universal method.

### A9 — confirmation and completion audit

Run IU-PCR, frozen Family-NRM, the supervised atomic ceiling, and the frozen
S1/S2/S3 finalists under the same row and label boundary. Use grouped paired
bootstrap intervals and report cell-, dataset-family-, and overall macros.
Apply the success ladder exactly as written in Section 3.

The program is complete only when every A0--A9 route has either passed its
gate or has a reproducible closure artifact, and the strongest achieved tier
has an honest confirmation verdict.

## 7. Closed directions not to reopen without new evidence

- single eigenvector nearest eigenvalue one;
- atomic null-band/projector variants selected only from pooled covariance;
- marginal-correlation clustering or random partition search;
- corrected or uncorrected cubic orientation as a detector without a new
  transport mechanism;
- trust-scale tuning against the existing labelled cells;
- another DUFS, graph-kernel, whitening, or precision-weighting variant on the
  same static matrix;
- higher moments without an auxiliary target-identification assumption.

## 8. Documentation and commit discipline

Each phase produces: a preregistration/specification, code and tests, a
machine-readable run definition, frozen label-free scores and hashes, a result
report, an append-only `HISTORY.md` step, an updated `PROGRESS.md`, and a
focused commit. Large caches remain in Google Drive or the approved local
cache; source, manifests, compact scores, and reports are committed so Claude
and Codex can reproduce and continue the work.
