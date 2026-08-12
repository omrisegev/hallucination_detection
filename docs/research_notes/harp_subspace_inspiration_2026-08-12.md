# HARP as Future Inspiration for U-PCR

**Status:** origin note; followed by an implemented contribution-space candidate
**Date:** 2026-08-12
**Paper:** Hu et al., *HARP: Hallucination Detection via Reasoning Subspace Projection*, arXiv:2509.11536v2 (2025)
**Local source:** `papers/HARP Hallucination Detection via Reasoning Subspace Projection.pdf`

## Why this paper is relevant

Our experiments repeatedly show an identification problem. The strongest shared
covariance direction in the mixed-v2 features may describe answer length,
language style, semantic content, model confidence, or task difficulty rather
than correctness. An unsupervised fusion method cannot automatically know which
shared factor is the hallucination signal.

A useful working model is

\[
X = a y + Bz + \epsilon,
\]

where \(y\) is the desired correctness or hallucination factor, \(z\) contains
shared nuisance factors, and \(\epsilon\) is feature-specific noise.

HARP addresses a related problem before classification. It applies SVD to the
model's unembedding matrix, treats its dominant subspace as semantic, and uses
the small orthogonal residual subspace as a candidate reasoning representation.
Hidden states are projected into this reasoning subspace before a supervised
hallucination classifier is fitted. The important inspiration is therefore:

> Separate a known nuisance or semantic subspace before asking the fusion method
> to identify hallucination-related variation.

This is conceptually close to our earlier latent-factor and nuisance-deflation
questions, although HARP has information that standard U-PCR does not have.

## Important differences from our method

- HARP uses hidden states and the model's unembedding weights; mixed-v2 U-PCR
  currently uses token-probability telemetry.
- HARP trains its final detector with hallucination labels and binary
  cross-entropy. It is not an unsupervised competitor.
- HARP defines its semantic/reasoning split using model structure. U-PCR tries
  to infer useful shared directions from feature covariance alone.
- HARP's reported score is therefore a supervised white-box reference, not a
  like-for-like comparison with our single-pass label-free detector.

## Possible future uses

If algorithm development is reopened, HARP may motivate one of these additions:

1. **White-box auxiliary view.** Project token hidden states onto HARP-like
   residual directions, summarize the resulting trajectory, and add those
   summaries as a separate view beside mixed-v2.
2. **Nuisance residualization.** Define a semantic or known-nuisance subspace
   from model structure or registered metadata, remove only that component, and
   run IU-PCR or DUFS-LIU-PCR on the residual feature matrix.
3. **Multi-block fusion.** Keep probability, semantic-hidden-state, and
   reasoning-residual features in separate blocks so that one high-dimensional
   block cannot dominate only because it contains more coordinates.
4. **Unsupervised analogue.** Investigate whether an external, label-free
   structural criterion can identify which latent factor is semantic nuisance.
   Covariance size or smoothness alone is not enough, as our prior experiments
   showed.

These are hypotheses, not validated improvements. In particular, blindly
removing the leading covariance factor is unsafe: it may remove the true
hallucination signal.

## Evidence and evaluation boundary

The current HARP-aligned local cell is not strong comparative evidence. It has
256 TriviaQA responses and only about six positive examples. The local
DUFS-LIU-PCR AUROC of 0.938 and HARP's published value near 0.929 are not an
exact same-ID, same-split, same-model comparison. No superiority claim should
be made from this cell.

Any future test inspired by HARP should require:

- a larger and adequately balanced cohort;
- an exact model, dataset, split, grader, and metric match;
- a frozen mixed-v2 U-PCR/IU-PCR baseline;
- separate ablations for subspace projection, the added view, and the fusion;
- diagnostics measuring alignment with length, style, difficulty, and other
  registered nuisances;
- label-free choices frozen before final evaluation if the proposed method is
  claimed to be unsupervised;
- confirmation that the removed subspace is nuisance rather than target signal.

## Decision

The paper comparison alone still does not justify changing the detector.
However, its structural lesson has now produced a concrete IU-PCR-internal
candidate.  The follow-up did not transplant HARP's white-box projection.  It
used IU feature-provenance families as the available structural anchor.

## Follow-up outcome: contribution-space family balancing

The supervised proof-of-concept decomposed the ordinary IU score into exact
family contributions, residualized those contributions against the shared IU
score, and learned only a small anchored residual correction.  Across 23
eligible development cells it improved equal-family AUROC by **+0.721pp**
([+0.309, +1.108]pp), with 21 wins and 2 losses.  This established that a
correctness-aligned correction exists without discarding the IU target axis.

The first label-free version balanced each family's realized L1 IU leverage.
It improved equal-family AUROC by **+0.633pp** on the original cells and
transferred positively to Qwen3 ProcessBench.  But a frozen cardinality control
was much stronger on ProcessBench.  The evidence therefore identified a more
specific nuisance: the number of engineered coordinates assigned to a
measurement family.

The final current candidate is **Cardinality-Balanced Contribution-Subspace IU
(CB-CS-IU)**:

```text
d_g  = mean_h(log m_h) - log m_g
s_CB = standardized_IU + (1/G) * R d / std(R d)
```

Here `m_g` is the number of present features from provenance family `g`, and
`R` contains standardized family contributions after linear residualization
against standardized IU.  The correction is therefore orthogonal to the IU
score on the unlabeled fit batch, has fixed scale `1/G`, and maps exactly back
to one effective IU feature-weight vector plus an intercept.

Evidence accumulated after the pivot:

- original 23 cells: **+0.442pp** equal-family delta over IU, 17W/6L;
- frozen Qwen3 ProcessBench control: **+0.864pp** equal-subset delta, 6W/0L;
- post-freeze Llama-3.1-8B scorer-family confirmation: **+1.263pp** cell-macro
  delta, paired four-subset interval [+0.708, +1.692]pp, 4W/0L;
- on the Llama cells, CB also exceeded LB by **+0.298pp** and DUFS-LIU by
  **+1.114pp**, both with positive paired subset intervals.

The selection of cardinality over leverage was made after the Qwen3 report and
is disclosed as retrospective.  The Llama run confirms scorer-family transfer,
but uses the same underlying ProcessBench examples and labels.  A genuinely
new benchmark family remains the clean independent-example confirmation.

Canonical follow-up artifacts:

- `SPEC_HARP_CONTRIBUTION_SUBSPACE_IU_V1.md`
- `SPEC_CARDINALITY_BALANCED_CS_IU_V1.md`
- `SPEC_CARDINALITY_BALANCED_LLAMA_PROCESSBENCH_CONFIRMATION_V1.md`
- `spectral_utils/contribution_subspace.py`
- `results/cardinality_balanced_cs_iu_v1/REPORT.md`
- `results/cardinality_balanced_llama_processbench_v1/REPORT.md`

## Independent-example failure of cardinality

The missing confirmation was then run on SemGrad SciQ and TruthfulQA.  The
candidate was scored from telemetry-only payloads and hashes were verified
before its BEM labels were opened.  CB-CS-IU failed: its equal-dataset delta
was **-0.767pp**, with SciQ at +0.175pp and TruthfulQA at -1.708pp.  Reversing
the cardinality correction helped TruthfulQA.  Feature count was therefore a
useful ProcessBench correlate, not a general nuisance identifier.

This failure sharpened the HARP question.  The issue was no longer whether the
six-family contribution space contained a useful correction, but whether the
target direction could generalize across environments and then be identified
without labels.

## Global supervised teacher: target direction exists and transfers

One six-dimensional anchored correction was fitted on the original 23 cells,
with equal cell weight and class-balanced logistic loss.  It was evaluated by
leave-one-dataset-family-out and transferred unchanged to ProcessBench and
SemGrad.  External labels never entered the source fit.

The coefficient signs were identical in all eight LOFO folds:

```text
entropy_level          +
entropy_dynamics       -
sampled_token_energy   -
partition_energy       +
topk_distribution      +
structural             +
```

The teacher improved original LOFO by +0.410pp equal-group, Qwen ProcessBench
by +0.684pp, Llama ProcessBench by +1.191pp, and both SemGrad datasets by
+0.646pp equal-dataset.  This is the promised supervised proof: a reusable
target-aligned correction exists in contribution space.  It remains a research
instrument because its direction was fitted from correctness labels.

## Label-free result: neutral residual mode

The residual covariance supplied the missing structural rule.  Because every
family residual is standardized, independent residual variation has unit
covariance.  Modes with eigenvalues far above one represent shared dependence;
modes near zero represent near-deterministic redundancy.  NRM-CS-IU averages
the residual covariance across unlabelled source cells, selects the eigenvector
whose eigenvalue is closest to one, and orients its otherwise arbitrary sign
toward the equal-family anchor.  On a target batch it applies the same fixed
`1/G` residual trust scale used by the earlier contribution candidates.

The frozen source calibration selected eigenvalue `1.035378` and direction:

```text
[+0.093928, -0.113808, -0.673995,
 +0.714635, +0.112033, +0.026490]
```

This recovers the supervised teacher's six signs without receiving labels.
The operation remains affine in the original mixed-v2 matrix and exposes one
effective IU weight vector plus an intercept.

Evidence:

- original leave-one-family-out: **+0.277pp** equal-group,
  95% interval [+0.016,+0.533], 15W/8L;
- Qwen ProcessBench: **+0.557pp**, 7W/1L;
- Llama ProcessBench: **+1.580pp**, 4W/0L;
- SemGrad: **+1.310pp**, 2W/0L;
- frozen HLE/Qwen2.5-72B: **+0.345pp**, but the interval
  [-0.898,+1.628] crosses zero because only 68 answers were judged correct;
- frozen PRMBench/Qwen3-8B response confirmation: IU 0.720602 to NRM 0.725206,
  **+0.460pp**, source-grouped interval **[+0.068,+0.841]**, all five
  pre-registered gates passed.

PRMBench improves in six of nine error-class contrasts and regresses slightly
in three, so NRM is not a universal per-nuisance dominance claim.  The primary
pooled confirmation is nevertheless positive under source-group resampling.
It uses 6,966 responses, 758 correct controls, a new model, new examples, and a
new error taxonomy.  Exactly three readiness-identified alignment defects were
excluded before scoring.

NRM is the current label-free algorithmic addition motivated by HARP.  Its
important limitation is explicit: it is trans-environment unsupervised
calibration from multiple unlabelled source batches, not a per-cell-only
identifiability theorem.

Canonical NRM artifacts:

- `SPEC_NEUTRAL_RESIDUAL_MODE_CS_IU_V1.md`
- `SPEC_NEUTRAL_RESIDUAL_MODE_PRMBENCH_CONFIRMATION_V1.md`
- `spectral_utils/contribution_subspace.py`
- `results/neutral_residual_mode_cs_iu_v1/REPORT.md`
- `results/neutral_residual_mode_hle_v1/REPORT.md`
- `results/neutral_residual_mode_prmbench_v1/REPORT.md`
