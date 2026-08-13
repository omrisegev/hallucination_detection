# Evidence-Contrast U-PCR for RAG Hallucination Detection

**Date:** 2026-08-08; executed 2026-08-10
**Status:** Version 1 complete. Evidence-Contrast features succeeded; the
DUFS/Laplacian mechanism did not. See
`results/ragtruth_evidence_contrast_v1/REPORT.html`.

## Version 1 result

The registered experiment used 900 development responses and 2,700 test
responses. Scores were fitted without labels and hashed before evaluation. On
the 12,958-sentence QA/Data-to-Text LOO test cohort, EC-DUFS-LIU reached 0.7026
AUROC versus 0.6721 for approximate GASP-top50: +0.0305 with paired grouped 95%
interval [0.0237,0.0378].

This is not evidence for the proposed graph mechanism. EC-IU-PCR reached
0.7031; EC-DUFS-LIU minus EC-IU-PCR was -0.00048 with interval
[-0.00061,-0.00034]. Permuted and ungated graphs were also tied with IU-PCR.
The useful result is the Evidence-Contrast contract plus ordinary IU-PCR.

Do not tune another graph on the opened RAGTruth test set. The next experiment
should freeze EC-IU-PCR and test transfer, conflict hallucinations, and
response-level nuisance robustness. Full mathematics, audit boundaries and
limitations are in `results/ragtruth_evidence_contrast_v1/METHODS.md`.

The old 30-feature intrinsic mixed-v2 response detector was reconstructed only
as a separately hashed post-hoc audit. It scores 0.7629 pooled response AUROC,
but 0.7698 on QA and 0.4345 on Data-to-Text. This task reversal confirms that
its pooled value is not a robust RAG grounding result.

Current field and benchmark map:
`docs/research_notes/rag_localization_methods_and_benchmarks_2026.md`.

## Follow-up: keep the original 30 features

The next experiment corrected the scope of this direction. Instead of
replacing mixed-v2 with 8/14 EC features, it extracted all 30 original features
from every full, no-context and LOO trace. All 30 were available everywhere,
and the full-only score reproduced the previous implementation exactly.

Evidence removal materially improves cross-task transfer. Original-30 LOO
IU-PCR raises task-macro AUROC from 0.6002 for full-only IU-PCR to 0.7164,
with paired change +0.1163 [0.0795,0.1544]. It reaches 0.7178 on QA and 0.7150
on Data-to-Text. GASP-top50 is still slightly higher at 0.7225 task-macro.

DUFS adds only in the smaller no-context matrix (+0.0065 task-macro over
IU-PCR). It adds approximately zero to LOO and Hybrid. The next external
confirmation candidate is therefore Original-30 LOO IU-PCR, not a larger
DUFS-LIU variant. Full details:
`results/ragtruth_mixed_v2_evidence_aware_v1/REPORT.html`.

## Research question

Can the same U-PCR family used for reasoning-error detection also detect and localize unsupported claims in retrieval-augmented generation (RAG), if retrieval evidence is treated as a controlled intervention?

The main idea is to keep the generated answer fixed and measure how its token statistics change when evidence is removed or changed. These repeated measurements may reveal whether each part of the answer actually depends on the retrieved evidence.

This direction is not yet part of the leading method. It should remain a future direction until it passes the validation plan below.

## What we learned from the old RAG cache

A representative old Phase 10 cache (`llama8b/hotpotqa`) contains useful engineering data, but its labels are not strong enough for a scientific grounding claim:

- It contains 240 answers and a median of 223 retrieved documents per answer (minimum 59, maximum 382).
- Only the first 15 documents were inserted into the prompt.
- The gold-answer text appeared in those first 15 documents for only 25 of 240 examples (10.4%).
- Only 19.2% of answers included citations.
- The statement/citation subset contains 92 samples, of which 29 are positive.
- The fallback label treats a statement as grounded when its cited passage contains the gold-answer substring. This is not the same as checking whether the passage semantically supports the claim.

Therefore, the old V4 result (AUROC 0.756) should be treated as an exploratory engineering result, not as a publishable RAG hallucination benchmark. The cache can still be used for smoke tests because it contains answer text, entropy traces, token offsets, and retrieved documents. It does not contain all raw statistics needed for the proposed intervention experiment, such as top-k logits or log-sum-exp values under every evidence condition.

## Proposed method

Working name: **Evidence-Contrast U-PCR (EC-U-PCR)**.

For a fixed answer $y_i$ and retrieved chunks $C_i=\{c_{i1},\ldots,c_{im}\}$, rescore the same answer under several evidence conditions:

1. Full retrieved context.
2. No retrieved context.
3. Leave-one-chunk-out context for every chunk, or for a controlled subset of chunks.

For token $t$, the effect of removing chunk $j$ can be measured by

\[
\Delta \operatorname{NLL}_{ij,t}
=
\operatorname{NLL}_t(y_i\mid C_i\setminus c_{ij})
-
\operatorname{NLL}_t(y_i\mid C_i).
\]

The same contrast can be computed for entropy, probability margin, tail mass, Jensen-Shannon divergence, and temporal features such as CUSUM, sliding-window variance, and spectral entropy.

### Matrix construction

The evidence interventions are repeated views of the same answer. They must not be treated as independent samples.

For answer-level detection, construct a matrix $F_{\text{global}}$ whose columns are answers and whose rows summarize feature/intervention combinations, for example:

- Full context versus no context.
- Mean and maximum leave-one-out effect.
- Effect of removing cited or highly relevant chunks.
- Dispersion and stability of effects across chunks.

For localization, construct $F_{\text{local}}$ at token or sliding-window resolution using the same feature family. The global head asks whether the answer contains an unsupported claim; the local head identifies the most suspicious token or window.

The same solver family can be used at both resolutions. The preferred comparison is:

- EC-U-PCR using the deployed U-PCR solver.
- EC-DUFS-LIU using the dependency-aware Laplacian solver.

The algorithm is the same in both heads; only the unit represented by each column changes.

### Important distinction from noise removal

This resembles the repeated-measurement direction proposed for estimating feature noise, but evidence interventions have a different meaning. Variation caused by removing relevant evidence may be the desired causal signal, not nuisance noise. The method should therefore separate:

- Evidence-invariant confidence information.
- Evidence-sensitive contrast information.

It should not automatically subtract all within-answer intervention covariance.

## Relationship to existing work

The intervention itself is not sufficient novelty. GASP already holds a response fixed and rescores it under full-context, no-context, and leave-one-passage-out conditions. It reports approximately 0.73 response-level AUC and 0.67 span-level AUC on RAGTruth, transfers to TofuEval, and reports weak performance on short-answer RAGBench.

The possible contribution here is instead:

- Label-free spectral fusion of many dependent intervention traces.
- DUFS-LIU-style weighting of redundant or unreliable evidence views.
- One U-PCR solver family for both global detection and localization.
- No trained classifier or external natural-language-inference verifier in the main method.

The direct or category-level competitors should include:

- **GASP:** fixed-response evidence perturbation; the closest direct competitor.
- **LUMINA:** internal-state-based RAG hallucination detection.
- **ReDeEP:** retrieval-aware detection using model internals and evidence dependence.
- **HALT:** a supervised top-k token-probability ceiling, reported separately from label-free methods.
- **RT4CHART:** an external-verifier approach evaluated on RAGTruth++, reported in a separate access category.

The updated field audit adds two important benchmark decisions. TRIVIA+ is a
strong confirmation candidate because it targets long RAG contexts and
provides controlled noisy-label sets. L-CiteEval should be added only if the
method makes a citation-correctness or citation-completeness claim; it is not a
substitute for a span-labeled hallucination benchmark. The audit also separates
training-free perturbation, supervised token classifiers, mechanistic
white-box methods, and external verifiers so their numbers are not presented as
if they use the same information and compute.

## Future data-collection plan

### Primary benchmark: RAGTruth

RAGTruth provides prompts, retrieved contexts, fixed responses, response-level labels, and span annotations. It does not provide the token statistics required by our method, so conditional rescoring should be run on the cluster.

For the first experiment:

1. Use the exact published RAGTruth responses rather than generating new answers.
2. Begin with a scorer such as Qwen2.5-1.5B to make the protocol comparable to GASP where possible.
3. Run full-context, no-context, and leave-one-chunk-out scoring.
4. Save source and response identifiers, exact prompts and chunks, token IDs and offsets, target-token log-probabilities, top-20 raw log-probabilities, log-sum-exp, condition metadata, runtime, model revision, and tokenizer revision.
5. Build and freeze all scores without labels.
6. Open labels only for final evaluation.

RAGTruth contains several responses for the same source. Splits and bootstrap confidence intervals must therefore be grouped by `source_id`, not by response row, to avoid leakage.

### Confirmation datasets

If the RAGTruth experiment succeeds:

- Use RAGTruth++ to test robustness to improved labels and compare with RT4CHART.
- Use TofuEval as a positive transfer test.
- Use short-answer RAGBench as an explicit falsification test, because evidence perturbation may fail when the answer is easily recovered from model memory.
- Use TRIVIA+ as a long-context and label-noise confirmation test.
- Use L-CiteEval only after the output contract includes explicit citation
  correctness or completeness.

## Evaluation protocol

Explain and report each metric before showing results.

- **Response AUROC:** ranking quality for deciding whether an answer contains a hallucination.
- **Response AUPRC:** precision-recall performance, important when hallucinations are imbalanced.
- **Span AUROC:** token- or character-level ranking quality for localization.
- **Overlap F1 or intersection-over-union:** agreement between predicted and annotated hallucinated spans.

Use grouped bootstrap intervals and leave-source, leave-task, and leave-model-out analyses where the data permits them.

Required comparisons:

1. Full-context U-PCR without evidence interventions.
2. The published GASP score or a faithful reproduction.
3. EC-U-PCR with the deployed U-PCR solver.
4. EC-DUFS-LIU with dependency-aware fusion.
5. Supervised and external-verifier methods in clearly separate categories.

No feature orientation, hyperparameter, threshold, or method selection may use evaluation labels. The score definition and all unsupervised choices must be hashed and frozen before labels are opened.

## Critical failure tests

The method should be rejected or narrowed if it cannot handle the following cases:

- A fact is correct from model memory but unsupported by the supplied evidence.
- Several chunks contain redundant evidence, so removing one chunk has little effect.
- Removing irrelevant context changes the score because of prompt-length or attention effects.
- The result is mainly predicted by chunk count, chunk length, or answer length.
- The scoring model differs from the generator and measures its own beliefs instead of the generator's grounding.
- A short answer is stable without evidence even when the benchmark requires evidence grounding.
- Responses derived from the same source leak across training, selection, or evaluation splits.
- Retrieval failure and generation grounding failure are mixed into one label. These should be reported separately when possible.

## Decision rule

Do not implement this as a main research branch yet. The old cache is suitable only for engineering smoke tests.

Before implementation, pin the exact GASP protocol and code revision, create a versioned data manifest, and register the comparisons above. Continue only if evidence-contrast features add reproducible signal over full-context U-PCR and the gain transfers to held-out sources, models, or tasks with a non-zero grouped confidence interval.

## References

- RAGTruth: [ACL 2024 paper](https://aclanthology.org/2024.acl-long.585/) and [official repository](https://github.com/ParticleMedia/RAGTruth)
- GASP: [arXiv:2607.04223](https://arxiv.org/abs/2607.04223)
- LUMINA: [ICLR 2026 OpenReview](https://openreview.net/forum?id=oJgNNBNEJM)
- ReDeEP: [ICLR 2025 OpenReview](https://openreview.net/forum?id=ztzZDzgfrh)
- HALT: [arXiv:2602.02888](https://arxiv.org/abs/2602.02888)
- RT4CHART: [arXiv:2603.27752](https://arxiv.org/abs/2603.27752)
- RAGTruth out-of-distribution audit: [Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.952/)
- TRIVIA+: [arXiv:2605.11330](https://arxiv.org/abs/2605.11330)
- L-CiteEval: [arXiv:2410.02115](https://arxiv.org/abs/2410.02115)
