# Specification: HARP-inspired contribution-subspace IU-PCR v1

**Status:** supervised feasibility passed; label-free identifiability not yet
established.

**Date:** 2026-08-12

## 1. Decision

Do not continue the current DUFS strategy by searching for a better sample
graph and inserting it into the same projected LIU solve.  The next research
object is a correction in IU-PCR's own provenance-family contribution space.

The supervised proof of concept passes its within-cell held-out-sample gate.
It is therefore justified to ask a narrower next question:

> Can the cross-fitted supervised correction be predicted from cell-local,
> label-free structure, under leave-one-dataset-family-out evaluation?

That premise must pass before implementing or evaluating a new unsupervised
optimizer.  A supervised teacher, a meta-supervised predictor, and a truly
label-free final rule are three different claims and must remain separate.

## 2. Why DUFS/LIU failed

The evidence points to an objective/actuation problem, not merely a weak graph
construction heuristic.

1. **Smoothness is not correctness.**  DUFS rewards feature subsets that make
   the sample graph coherent.  A nuisance factor shared by several features
   can be smoother and more reproducible than truthfulness.
2. **The same observations define both sides of the objective.**  The feature
   matrix discovers the graph, and the graph then regularizes weights over that
   same matrix.  This self-confirming loop has no independent target anchor.
3. **Projection suppresses actuation.**  In LIU, the graph penalty acts inside
   ordinary IU-PCR's fixed two-dimensional covariance eigenspace.  Empirically,
   DUFS and IU weights were almost collinear, and stronger graph penalties were
   harmful.  An exploratory label-oracle graph preflight was also non-positive;
   it should be formalized before being cited as an audited result, but it is
   strong evidence against spending the next iteration on graph search alone.
4. **Deletion and deflation destroy mixed signal.**  Hard feature filtering,
   nuisance-subspace deflation, and coupled moment corrections have removed
   useful covariance together with nuisance.
5. **One global correction does not transfer.**  Fixed cross-family projectors,
   routers, donor-only LOFO corrections, and shared moment corrections have not
   generalized.  The useful correction appears cell-specific even when its
   coordinate system can be shared.
6. **Static unlabeled reliability proxies are insufficient.**  Stability,
   agreement, diffusion convergence, and related diagnostics often select
   coherent nuisance.  They are controls, not target identifiers.

Relevant audited evidence is summarized in:

- `docs/methods/dufs_liu.md`
- `docs/research_notes/atomic_operator_premise_audit_conclusion.md`
- `docs/research_notes/family_relevance_diagnostic_conclusion.md`
- `docs/research_notes/repeated_cross_view_diffusion_conclusion.md`
- `SPEC_TARGET_ANCHORED_LIU_V1.md`
- `SPEC_HIERARCHICAL_ACTIVE_SPECTRAL_V2.md`
- `RESEARCH_EVIDENCE_LEDGER.md`

## 3. What transfers from HARP

HARP's useful idea is architectural: expose a low-dimensional coordinate
system in which target-relevant variation can be separated from dominant but
irrelevant variation before making the final decision.

HARP itself is not an admissible method here.  It uses model hidden states, the
unembedding matrix, and supervised BCE training.  Those are white-box and
task-trained quantities.  The permissible analogue must live entirely inside
the existing one-pass fusion calculation.

The analogue proposed here is **IU contribution space**, not raw feature space
and not a hidden-state subspace.

## 4. Contribution-Subspace IU (CS-IU)

Let `F` be the existing feature-by-sample matrix and `w0` the ordinary IU-PCR
weight vector.  For each frozen provenance family `g`, define its score
contribution

```text
h_g(x) = sum_{i in g} w0_i F_i(x).
```

The contributions reconstruct the IU score exactly:

```text
b(x) = sum_g h_g(x) = w0^T F(x).
```

On a training partition, standardize `b` and the columns of `H`, then remove
from every contribution its linear projection onto `b`.  Call the resulting
family-disagreement matrix `R`.  The supervised teacher is

```text
s_delta(x) = b(x) + R(x) delta,
```

with class-balanced logistic loss and an L2 prior

```text
(lambda / 2) ||delta||^2.
```

The coefficient of `b` is fixed to one.  Therefore `delta = 0` is exactly the
IU-PCR ranking, rather than a separately re-fitted baseline.  The correction
changes only the weights assigned to already-computed family contributions.

This construction obeys the intended deployment boundary:

- no additional model inference;
- no new feature or signal;
- no hidden state, attention, gradient, or model weight;
- no hallucination-specific rule inside the fusion mechanism.

Correctness labels are used in the current teacher fit, so the current method
is supervised and is not a final solution.

## 5. Supervised PoC protocol and result

Implementation:

- `spectral_utils/contribution_subspace.py`
- `scripts/harp_contribution_subspace_poc.py`
- `scripts/test_contribution_subspace.py`

Artifacts:

- `results/harp_contribution_subspace_poc_v1/REPORT.md`
- `results/harp_contribution_subspace_poc_v1/summary.csv`
- `results/harp_contribution_subspace_poc_v1/teacher_targets.csv`
- `results/harp_contribution_subspace_poc_v1/RUN_DEFINITION.json`

Protocol:

- existing 24-cell mixed-v2 development bundle;
- one data-rule exclusion: fewer than 20 positives, leaving 23 cells;
- 30 stratified 60/40 train/evaluation splits per cell;
- primary teacher: all labels in the 60% training partition, `lambda=0.3`;
- frozen feature families and ordinary IU-PCR weights;
- full-feature and unrestricted family-space ridge controls;
- equal-family bootstrap in addition to cell-macro reporting.

Primary result:

| quantity | result |
|---|---:|
| IU-PCR cell-macro AUROC | 0.7698 |
| anchored CS-IU AUROC | 0.7778 |
| delta | **+0.800pp** |
| wins / losses | **21 / 2** |
| worst cell delta | **-0.238pp** |
| equal-family delta | +0.721pp |
| equal-family 95% bootstrap interval | **[+0.309, +1.108]pp** |
| median within-cell teacher-direction cosine | **0.967** |

The predeclared feasibility interpretation is positive: a small, anchored,
fusion-internal target correction exists and generalizes to held-out samples
within the same cell.  The result does not establish transfer to unseen cells
or identify a label-free rule.

The controls sharpen the conclusion.  Full-feature ridge has a larger mean
gain with all training labels (+1.388pp), so some gain is simply supervised
signal.  It also has a negative equal-family interval lower bound and a much
worse tail.  At 20--80 labels, unrestricted fits are substantially harmful;
strong anchoring is the only safe low-label behavior.  The contribution
coordinate system is therefore useful mainly as a low-dimensional, IU-centered
inductive bias rather than as proof that labels are unnecessary.

## 6. Next experiment: label-free identifiability before optimization

The cross-fitted primary coefficients in `teacher_targets.csv` define the
quantity that the next stage must explain.  For every training split, construct
an unlabeled fingerprint using only the training feature matrix and the
ordinary IU-PCR fit.  Candidate fingerprint blocks are:

1. family-contribution covariance and eigenspectrum;
2. family leave-one-out perturbations of `rho_hat`, `w0`, and sample scores;
3. additive-moment projection residuals aggregated by provenance family;
4. split-half stability of the contribution geometry;
5. masked-family predictability: predict each `h_g` from the other `h` columns
   using sample cross-fitting, and separate shared-predictable from
   family-private residual energy;
6. aggregate DUFS gates by family as a known-weak control.

These quantities are diagnostics, not assumed truth proxies.  In particular,
stability and masked reconstruction can still favor coherent nuisance.

### 6.1 Evaluation split

Use leave-one-dataset-family-out outer evaluation.  All rows from the held-out
family are unavailable while fitting any map from unlabeled fingerprints to
teacher corrections.  Within each development cell, teacher coefficients must
remain cross-fitted with respect to the samples on which their score benefit is
measured.

### 6.2 Controls

Compare against:

- zero correction, which is exactly IU-PCR;
- a single global mean correction;
- dataset-family mean correction;
- permuted teacher targets;
- norm-only and sign-only predictions;
- previously failed stability/agreement proxies;
- an unrestricted predictor with matched inputs to expose overfitting.

### 6.3 Premise gates

Do not build the final optimizer unless the predicted correction:

1. improves held-out-family AUROC over IU-PCR with an equal-family 95% interval
   above zero;
2. recovers at least 30% of the cross-fitted supervised teacher's equal-family
   gain;
3. wins in at least 16 of the 23 eligible cells;
4. has a worst-cell delta no lower than -1.0pp;
5. beats global-mean, family-mean, and permuted-target controls;
6. preserves exact IU identity when the predicted correction is zero.

Passing this stage establishes only **meta-supervised structural
predictability**.  It does not make the predictor unsupervised.

## 7. Route to a genuinely label-free rule

If one or more fingerprint blocks predict the teacher under LOFO, derive a
pre-specified cell-local objective from that block and remove the teacher from
the fitting loop.  The most plausible current candidate is a HARP-like
shared/private decomposition in contribution space:

1. cross-fit masked-family predictors on samples;
2. decompose each contribution into cross-family predictable and private
   residual components;
3. construct a generalized eigenproblem whose numerator rewards shared
   predictable structure and whose denominator penalizes private/nuisance
   energy;
4. orient and shrink the adjustment through IU-PCR's existing moment equation,
   with zero correction as the fixed prior;
5. freeze the objective and hyperparameters before evaluating final cells.

The orientation step is the critical unresolved issue.  Unlabeled covariance
alone cannot, in general, determine which of two sign-symmetric directions is
correctness-aligned.  HARP has an external structural anchor in the unembedding
operator; IU-PCR does not automatically have an equivalent.  If the moment
equation cannot supply a stable orientation, the final no-label goal is not
identified under the current input contract.  That is a valid negative result,
not a reason to relabel a supervised predictor as unsupervised.

## 8. Stop conditions

Stop this branch if any of the following occurs:

- the teacher direction is not reproducible within cells after stricter
  cross-fitting;
- unlabeled fingerprints predict teacher magnitude but not direction/sign;
- LOFO gains disappear after comparing with mean-correction controls;
- coherent synthetic nuisance passes the proposed shared/private criterion;
- performance requires using correctness labels, additional generations, new
  telemetry, or white-box internals at deployment;
- the proposed rule reduces to another graph smoothness penalty inside the
  unchanged LIU solve.

The immediate deliverable is therefore not “unsupervised CS-IU.”  It is the
LOFO identifiability audit that determines whether such an algorithm can exist
under the stated constraints.
