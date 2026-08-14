# Automatic group-free IU research program v1

**Date:** 2026-08-13

**Status:** active research contract; A0 passed, A1--A5 closed; A6 protocol and
S0a/S0b/S1 execution contract frozen before implementation

**Primary objective:** replace the hand-defined provenance quotient used by
NRM-CS-IU with an automatically identified correction that improves IU-PCR
for hallucination detection while preserving the deployment contract.

## 1. Scope and non-negotiable contract

The primary S1 method must:

1. use the frozen one-pass mixed-v2 telemetry available to IU-PCR;
2. fit without correctness labels;
3. contain no runtime or calibration dependency on `FEATURE_TO_VIEW`,
   `VIEW_ORDER`, family names, or a hand-authored equivalent;
4. remain a fusion rule: one affine score over the target's frozen
   present-roster subset of the nominal named mixed-v2 coordinate matrix
   (equivalently, a nominal vector with absent coefficients zero), with a
   numerical reconstruction check. Because mixed-v2 contains frozen nonlinear
   per-feature transforms, reconstruction is assessed after that transformer
   rather than against raw telemetry values;
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
- automatic structure may use environments, code-registered feature-DAG
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

- a code-registered feature-DAG registry with channel, operator,
  reduction, source stream, parameters, and dependencies;
- a 23-cell environment/missingness/pair-coverage manifest;
- an exact audit of prompt/item overlap across Qwen and Llama caches;
- a simulator with target, shared difficulty, environment nuisance,
  operator/channel crossed effects, duplicates, missingness, and optional
  environment-specific target directions;
- a selected untouched confirmation surface and immutable label boundary.

Gate: do not start A1 with an undocumented feature mapping or an unknown
cross-model pairing boundary.

**Frozen execution result (Steps 255/257): PASS.** The no-new-label audit recovered
30 canonical features across 23 source environments, with 17 features present
in every environment and feature-pair coverage ranging from 8 to 23 cells. Six
cells contain fewer valid mixed-v2 bundle rows than manifest attempts (minimum
retention 19.8%); subsequent structural fitting must preserve the bundle
population and equal-environment weighting. Exact content-and-ID pairing was
verified for 3,400 fixed ProcessBench responses scored by Qwen3-4B, Qwen3-8B,
and Llama3.1-8B. Source streams come from extractor registries; the operator
taxonomy is an explicit handwritten but label-blind mapping, while function
signatures record implementation provenance only. The input also inherits
mixed-v2 signs/transforms from earlier labelled development, so the precise
claim is no new labels beyond the frozen IU input contract. The stronger
reserved confirmation boundary is `popqa-gemma3-4b-it-confirmation-v1`:
PopQA is an unseen dataset family; the exact Gemma-3 checkpoint/generation is
unseen, though the broad Gemma family is not. A token-boundary alias rule,
official-substring secondary diagnostic, access smoke, and Qwen3-4B fallback
are sealed before collection. Canonical artifacts:
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

**Frozen execution result (Steps 256/257): CLOSE AS DETECTOR BASIS.** A hash-defined
16/7 structural-train/audit split was used without correctness labels. The
training-selected rank-6 interaction hybrid (25% mechanical factorial
projector, 75% anonymized PCA projector; ridge 0.1) improved equal-environment
audit MSE from 0.034704 for pooled PCA to 0.032009 and decisively beat cardinality-matched
random partitions, but the seven-environment grouped interval for its MSE
delta versus PCA crossed zero: -0.002695 [-0.005845, 0.000282]. It was stable
(minimum leave-one-training-environment projector overlap 0.9428), exactly
permutation-equivariant, deterministic, and exact-duplicate balanced. However,
an automatically appended rho=0.999 duplicate received 3.009 times the
original feature's combined soft-quotient mass, violating the frozen 1.10
gate. Hard channel/operator/factorial bases were materially worse. A1 is
therefore evidence that weak factorial metadata can regularize PCA, not an
admissible detector basis. A2 proceeds on raw atomic residual covariances.
Canonical artifacts: `results/automatic_group_free_phase_a1_v1/`.

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

**Frozen execution result (Step 257): CLOSE MISSING-AWARE JBD AS TARGET
BASIS.** The primary run retained all 30 atoms, completed missing covariance
entries from training folds only, and scored only genuinely observed held-out
pairs. JBD reached environment-macro MSE 0.028700 versus 0.032864 for pooled
PCA with identical recovered block sizes, mechanism count, and ridge. The
paired delta was -0.004164 with 95% interval [-0.012164, 0.000838], so the
capacity-matched gate failed. LOEO mechanism-rank ratio also failed at 0.618
versus the frozen 0.70 gate. Outer-fold structures ranged from a dominant
15--19 coordinate block to one fold with 30 singletons. The advantage vanished
under a train-only stationary PSD null preserving missingness and sample
counts. The 17-feature complete-core diagnostic independently failed its
matched interval/stability gates. A2 is retained as structural evidence, not
as a detector basis; this closes the implemented missing-aware route, not all
possible JBD algorithms.

Because A2 failed before a detector score, orientation, or trust rule was
constructed, the detector-only exact/near-duplicate, affine reconstruction,
and zero-evidence IU-PCR fallback gates were not run. They remain mandatory
before any future promotion and are not counted as passed.

### A3 — primary strict hybrid: factorial quotient plus JBD

Use A1 to control measurement multiplicity and A2 to learn environment-stable
blocks. No candidate is evaluated as a detector until its target-component,
orientation, and trust rules are frozen. The primary selector must derive from
A4's cross-model decomposition when pairing is adequate; a selector based
only on the JBD spectrum is prohibited.

**Execution result (Step 257): CLOSED BY PREMISE.** A1 failed duplicate
robustness and A2 failed capacity-matched improvement/stability, so their
hybrid is not constructed or evaluated as a detector.

### A4 — paired cross-model multi-view identification

**Hypothesis:** scorer-specific nuisance varies across scorer views while the
fixed response supplies a shared component. The 3,400 ProcessBench triples
are a nuisance-changing intervention only: response correctness cannot vary
across their scorer views. Invariance alone therefore cannot distinguish
hallucination from shared difficulty, length, or style.

Fit a hierarchical shared/individual source model on exact item matches. Hold
out entire model families and environments, then repeat after shuffling item
pairs. Candidate selection, sign, and trust must be functions of the fitted
multi-view model and a declared confidence anchor, not labels.

A4 may promote a target component only with an additional target-changing
contrast/anchor plus held-model and item-pair-shuffle falsification. The
pre-execution audit found no legal strict-S1 target-changing pair in the local
surface: the available evidence interventions belong to A6. Therefore this A4
execution is hard-closed in advance as `CLOSE_NO_TARGET_CONTRAST` for detector
promotion. It may separately pass or fail a shared/scorer-sensitive structure
premise and may pass only structural information forward to A5/A6.

Premise gate: the shared source must disappear or degrade under pair shuffle;
the individual candidate must transfer to a held-out model family and must
not reduce to model size, answer length, or dataset identity.

If exact pairing is inadequate, record the coverage failure and continue to
A5 rather than creating approximate semantic matches post hoc.

The frozen implementation protocol, including item-first outer splits,
feature-level text/length residualization, pairing-aware baselines, conditional
shuffle nulls, and non-vacuous gates, is
`docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A4_V1.md`.

**Execution result (Step 260): CLOSED.** CorrCA achieved 0.997881 repeatability
on the Qwen pair and 0.955465 correlation with held Llama consensus, but the
nested-selected `single:1 = trace_length` baseline reached 0.966908 on Llama.
The paired delta was -0.011444 [-0.016036,-0.009034], failing the registered
material gate. Independent post-held diagnosis found that CorrCA placed
0.997897--0.999279 loading on trace length across folds. The frozen nuisance
basis left a deterministic token-count residual, so the formal confound and
coarse conditional-null passes do not support a non-length mechanism claim.
Trace-only reproduced the strongest baseline; fixed-loading deletion of the
trace term lowered Llama correlation to 0.866653 without refitting. The phase
therefore closes as `CLOSE_SHARED_REPEATABLE_COMPONENT_PREMISE`, while the
pre-frozen detector result remains `CLOSE_NO_TARGET_CONTRAST`. No A4 component
is carried into A5 as target-identified evidence.

### A5 — continuous weak-supervision dependency model

**Hypothesis:** an item-level, equal-covariance latent mixture with sparse
within-component precision can recover an IU-orthogonal atomic correction more
reliably than marginal dependence models, while retaining an affine score.

A5 is restricted to one IU-anchored primary. Its 17-feature complete core is
automatically fixed by presence in all 23 A0 environments and excludes trace
length. The sparse graph/penalty is shared across environments; local mixture
parameters are adaptation-only. The mixture label switch is oriented solely by
positive covariance inner product with frozen IU-PCR. No majority assumption,
A1 factorial prior, A4 component, or retrospective label may select a model.

Unlabelled likelihood cannot identify correctness semantics. A required
observational-equivalence audit demonstrates this explicitly; A5's bounded
claim is conditional on inheriting IU-PCR as target anchor. A sealed synthetic
nuisance-dominance gate runs before large raw-cache transfer. Failure closes
the route without a broad sweep. Only if all synthetic gates pass is the
target-firewalled, group-aware 23-environment likelihood premise executed.

The one-way retrospective label gate may PASS or VETO exactly one already
frozen primary; it may not choose among arms or tune anything. The complete
executable protocol and gates are frozen in
`docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A5_V1.md`.

**Execution result (Step 263): CLOSED AT S1a.** The exact 100 registered
nuisance-dominant seeds ran against the committed boundary. Two repetitions
had registered numerical nonconvergence, so the frozen formal verdict is
`CLOSE_NUMERICAL_NONCONVERGENCE`. The 98 usable runs independently failed the
semantic premise: final and correction target preference were 62/98 and
25/98, and candidate-minus-IU AUROC averaged -0.038484 with 95% bootstrap
interval [-0.047495,-0.029659]. Alpha 1 was selected 46 times and harmed IU by
0.080974 AUROC on average. A5 therefore cannot be rescued as a numerical-only
issue: the likelihood selector repeatedly follows the stronger nuisance. S1b
and real-data A5 are forbidden; no real cache or retrospective label was
accessed. Canonical result:
`results/automatic_group_free_phase_a5_v1/REPORT.md`.

### A6 — self-supervised intervention route

**Status (Step 266): parent protocol and mechanically executable S0a/S0b/S1
contract preregistered and independently audited; unsealed construction/PTNI
development primitives implemented; no A6 telemetry, simulator result, natural
response, correctness sidecar, sealed seed, or target opened.** The target is one
mechanically parsed task-answer assertion,
not RAG faithfulness or contextual support. Evidence removal from a fixed
response therefore remains an ambiguous adjacent-task diagnostic rather than
a legal target-changing calibration pair.

The sole A6 candidate is reciprocal Paired Target/Nuisance Intervention IU
(`PTNI-IU`). Each source group contains two equally difficult, independently
verified task worlds and two deterministic answers. Scoring the complete 2x2
prompt-response crossover makes every prompt and response marginal exactly
50/50 correct/incorrect while response bytes stay fixed across the target
contrast. Canonical prompts plus three semantics-preserving render families
produce factorial target, nuisance, and target-by-render effects. The source
has exactly 900 accepted Qwen groups balanced over three semantic domains,
three AST mutation families, and short/certificate response grammars; Llama
uses a disjoint 900-group audit.

PTNI learns one nuisance-whitened atomic error direction, projects it exactly
orthogonal to target-local IU-PCR covariance, and selects a frozen trust on a
path containing exact `alpha=0` IU fallback. Deployment remains one affine pass
over the original mixed-v2 identities. Exact complete-block admission,
availability-aware controls, conditional sign permutation, two split-local
placebo families, an activated nuisance-as-target negative control, LO target
and nuisance families, duplicate/missingness/permutation gates, and a sealed
eight-world simulator are mandatory.

The stage order is irreversible: S0 mechanical construction/shortcut audit;
S1 sealed simulator; S2a nested Qwen quartets; S2b frozen Llama quartets using
only an unlabeled natural calibration matrix; S2c untouched greedy Llama errors
under a closed answer parser; S3 one-way retrospective answer-correctness veto;
and only then the sealed PopQA confirmation. Labels may never choose an arm,
sign, feature, transform, or trust. A failure returns IU or closes invalid,
then advances to A7 without a rescue variant.

Canonical frozen protocol:
`docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A6_V1.md`.

Canonical frozen S0a/S0b/S1 execution contract:
`docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A6_S0_S1_EXECUTION_V1.md`.
Its exact pre-freeze body SHA-256
`5c869db42633d04bf4c46110d95de83891c6ca6b10fdf381653b8a618a750615`
received the independent verdict `NO BLOCKERS`. This freezes implementation
choices; it does not open an execution boundary. The implemented source and
runtime inputs must receive a fresh no-edit review before S0a execution.

The Step-265 source is not an S0a/S1 boundary. Step 266 now specifies the
jointly disjoint natural/PopQA prompt manifests, derived-answer quotas, exact
pinned Qwen3/Llama tokenizer audit, target-manifest-bound duplicate preflight,
named feature-permutation canonicalization, complete nested controls/nulls/LO
selection, eight-world simulator, and append-only runner. Those items remain
to be implemented and independently reviewed before any sealed seed or
response telemetry may open.

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
