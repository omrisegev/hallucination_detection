# Fixed RAG and Reasoning Pipelines

## Purpose

This experiment freezes two application pipelines. It does not introduce a new covariance solver.
Both pipelines use the same token feature basis and full-pool, two-component IU-PCR.

## Shared feature contract

The contract is `shared-token-mixed-v2-applications-v1-2026-08-13`. It contains 29 token streams that
cover all 30 mixed-v2 response features. CUSUM magnitude and
CUSUM location share one positional stream. Trace length is a constant stream within one response;
it can change answer risk but cannot create a false local peak.

18 streams have an exact reduction back to a global feature.
11 are causal rolling counterparts of a whole-trace operation.
The rolling versions are disclosed as approximations; they are not presented as exact identities.

The same frozen mixed-v2 transformation rules are applied in both packages; each allowed training
population estimates its own label-free location and scale parameters. In particular,
permutation entropy is folded by squared distance and STFT spectral entropy is folded around its
label-free population mode. All columns are oriented so that larger means more confidence before
fusion. Final risk is the negative fused confidence.

## IU-PCR

Let `F` be the feature-by-sample matrix and `C = FF^T/n`. IU-PCR estimates the unobserved vector
`rho_f = Cov(f, Y)` from off-diagonal covariance equations, then solves inside the first two
eigenvectors `U` of `C`:

`w = U (U^T C U)^(-1) U^T rho_hat`.

The score is `w^T f`. No label enters this fit. This is based on unsupervised ensemble regression
(Dror et al., 2017) and the spectral crowdsourcing formulation (Tenzer et al., 2022).

## Fixed RAG pipeline

The observed object is `X[i,t,c,f]`: response `i`, answer token `t`, evidence condition `c`, and
shared feature `f`. Full context and no context are always required. If leave-one-chunk-out (LOO)
conditions exist, the evidence head also contains, for each base feature, maximum drop, mean of the
two largest drops, mean positive drop, and negative drop standard deviation.

The no-context head therefore has 2 × 29 columns. The LOO head has
6 × 29 columns. A response uses the LOO head when LOO traces exist and
the no-context head otherwise. This is a fixed availability rule, not benchmark-specific tuning.

One fused token-risk stream produces every output:

- native token risk for token/span evaluation;
- mean token risk inside a RAG sentence or supplied claim;
- maximum token risk inside a reasoning step;
- mean token risk for complete-answer RAG detection.

The supplied-claim adapter uses the same mean-inside-boundaries rule as the sentence adapter. It is
part of the fixed interface, but it is not separately validated in the present RAGTruth experiment.
RAG scoring requires `2 + J` teacher-forced passes for an answer with `J` evidence chunks: full
context, no context, and one pass for each LOO condition. The answer text is held fixed, so this is
not repeated stochastic generation.

## Fixed reasoning pipeline

The observed object is `X[i,t,f]` over the complete, uninterrupted reasoning trace. Feature
computation and IU-PCR happen before step boundaries are used. Each step receives the maximum token
risk inside its supplied span. The predicted error location is the step with maximum risk.

ProcessBench also requires a no-error decision. The frozen answer detector is
`0.75 z(global mixed-v2 IU risk) + 0.25 z(max local step risk)`. A calibration half chooses only
the operating threshold; it does not change features or fusion weights. PRMBench uses continuous
step risk and therefore needs no threshold.

Reasoning needs one teacher-forced scoring pass over the existing complete trace. It does not ask
the model to regenerate the reasoning path.

## Why DUFS/Laplacian is not in the final heads

DUFS-LIU-PCR remains an important control and implementation standard for the 24-cell detection
study. Across the completed RAG and reasoning experiments, however, its incremental gain over
IU-PCR was negligible or unstable. Keeping it in the fixed application pipeline would add
complexity without a demonstrated contribution. This report therefore uses IU-PCR and states that
decision explicitly.

## Evaluation and uncertainty

RAG AUROC and AUPRC intervals resample complete `source_id` groups. Sentence rows from one source
therefore remain together. ProcessBench compares methods on identical repeated calibration/evaluation
splits and reports the paired F1 difference. Published scores are shown as references only when the
model, sample IDs, or access level differs. RAGTruth is exploratory because its labels had already
been opened in earlier project work.

## References

- Dror et al., *Unsupervised Ensemble Regression* (2017).
- Tenzer et al., *Crowdsourcing Regression: A Spectral Approach* (AISTATS 2022).
- Lindenbaum et al., *Differentiable Unsupervised Feature Selection based on a Gated Laplacian*.
- Niu et al., *RAGTruth* (ACL 2024).
- *GASP: Look Beyond the Answer for RAG Grounding* (2026 preprint).
- Song et al., *PRMBench* (ACL 2025).
- Zheng et al., *ProcessBench* (2025).
