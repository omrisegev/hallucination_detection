# RAG task adapters must be reconsidered after the benchmark review

**Date:** 2026-08-12

**Status:** Deferred research question. Do not change the algorithms until the
current benchmark-suite review is complete.

## Why this note exists

The current RAG answer-detection and RAG span-localization experiments do not
come from one unified mathematical model. They use different historical task
adapters. This makes the reports useful as baselines, but it does not establish
the best way to apply U-PCR or DUFS-LIU-PCR to either RAG task.

## The two tasks are different

### Answer-level detection

Input: one question, its retrieved evidence, and one fixed generated answer.

Output: one risk score for the complete answer.

The current evidence-contrast experiment rescored the fixed answer under full
context, no context, and leave-one-chunk-out conditions. It summarized changes
in token statistics into response-level evidence features and fused those
response-level features. Its strongest result suggests that evidence
interventions are informative, while the extra evidence-graph fusion gain is
not yet statistically confirmed.

### Token/span localization

Input: the same kind of question, evidence, and fixed answer.

Output: one risk score for every answer token or character span.

The current benchmark adapter uses only the full-context trace and five frozen
token-resolved views inherited from the historical GL-LIU localization work.
It does not use no-context or leave-one-chunk-out evidence changes. It also does
not test token-resolved counterparts of the complete mixed-v2 feature pool.

### Fixed-claim checking

The current RefChecker/KnowHalBench adapter receives claims that were already
extracted by the benchmark. It summarizes each claim with eight trace features
and six full-versus-no-context contrasts, then fits the three fusion solvers
separately in the accurate-, noisy-, and zero-context settings. This is useful
evidence that an isolated claim plus an evidence intervention contains a
detectable signal, but it is another task-specific feature contract rather than
the same method used at answer, sentence, and token resolution.

The current experiment evaluates claim checking only. It does not measure
claim extraction, does not use leave-one-chunk-out evidence attribution, and
collapses the human three-way labels into supported versus unsupported risk.

## The resulting interpretation problem

The answer experiment changes the evidence but compresses the token dimension.
The span experiment preserves the token dimension but discards the evidence
condition dimension. The GASP sentence adapter introduces seven different
evidence-sensitivity features, while the RefChecker claim adapter introduces a
separate fourteen-feature contract. Therefore their different performance
cannot be read as a clean comparison of task resolutions or as a general
verdict on U-PCR/DUFS-LIU.

In particular, the weak span result only establishes that full-context fusion
of the five historical core token views is weak. It does not establish that:

- the original mixed-v2 information is unsuitable for localization;
- evidence perturbation cannot improve localization;
- a shared RAG-specific spectral formulation would fail;
- DUFS cannot benefit when the graph contains meaningful evidence structure.

## Unified formulation to revisit

A more principled starting object is:

\[
X_{i,t,c,f},
\]

where:

- \(i\) is a response;
- \(t\) is an answer token or span;
- \(c\) is an evidence condition: full context, no context, or one
  leave-one-chunk-out condition;
- \(f\) is a token-resolved counterpart of an original feature.

The same object could support both tasks:

1. For localization, fuse evidence-conditioned changes at every token to
   produce \(s_{i,t}\).
2. For a supplied claim or sentence, aggregate the frozen token risks over its
   aligned token set. Claim extraction must be evaluated as a separate upstream
   component rather than silently assumed correct.
3. For answer detection, aggregate the frozen token risks \(s_{i,t}\) with a
   predeclared operator such as a mean, upper quantile, or top-k mean to produce
   \(s_i\).

This would let the two tasks share the same feature basis and fusion mechanism,
while differing only in the final aggregation. It would also make the role of
DUFS explicit: learn which feature/evidence views remain informative, while a
Laplacian can encode similarity across tokens, evidence chunks, or both.

## Questions that must be answered before implementation

1. Which original mixed-v2 features have exact token-resolved counterparts?
2. Which features require approximate rolling versions, and are those versions
   stable enough to include?
3. Should the primary contrast be full-minus-no-context, full-minus-LOO, or a
   distribution across removed chunks?
4. What graph should the Laplacian represent: token proximity, evidence-chunk
   relationships, feature similarity, or a structured product graph?
5. How can answer aggregation be fixed without choosing it on evaluation
   labels?
6. Can one label-free fit transfer across tasks, datasets, and evidence counts?
7. For claims, can the same local risk distinguish contradiction from missing
   support, or is a separate semantic decision layer required?
8. How much end-to-end performance is lost when claims are automatically
   extracted rather than supplied by the benchmark?

## Current decision

Keep the existing pages as transparent baseline results. Do not describe the
current five-view span adapter as the final RAG localization method. Return to
this design question after reviewing every benchmark domain and deciding which
application is scientifically worth developing further.
