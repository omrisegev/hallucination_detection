# RAGTruth Evidence-Contrast Experiment: Methods

> **Audit notice.** The original blind-run score artifacts in this directory
> used the stored full-vocabulary entropy for two EC columns, although the
> approved contract specified entropy over the saved top 50 probabilities plus
> one tail category. The formula-faithful correction is in
> `../ragtruth_evidence_contrast_v1_top50_correction/`. It was run after the
> original test labels had been opened, so it confirms robustness of the
> conclusion but is not a second blinded test.

## 1. Question and claim boundary

This experiment asks whether a label-free spectral method can combine several
measurements of evidence dependence better than a simple fixed aggregation.

The answer text is fixed. We do not ask the language model to generate the
answer again. Instead, a scorer reads the same answer under these conditions:

1. all retrieved evidence is present (`full`);
2. all evidence is removed (`noctx`);
3. one evidence chunk is removed at a time (`loo_j`).

The scorer is `Qwen/Qwen2.5-1.5B-Instruct`. It gives the probability of every
token in the fixed answer. It also stores the 50 most likely tokens and their
log-probabilities.

This experiment contains two separate scientific claims:

- **feature claim:** evidence-removal measurements contain useful grounding
  information;
- **fusion claim:** DUFS and a graph Laplacian improve the fusion beyond
  ordinary IU-PCR.

Success on the first claim does not prove the second claim.

## 2. Data units and label boundary

RAGTruth contains several responses for one source. These responses are not
independent. Every bootstrap sample therefore resamples complete `source_id`
groups.

We use two cohorts:

| cohort | tasks | available conditions |
|---|---|---|
| `no-context` | QA, Summary, Data-to-Text | `full`, `noctx` |
| `LOO` | QA, Data-to-Text | `full`, `noctx`, every registered `loo_j` |

Summary has no independent leave-one-out records. We do not create or impute
them.

The adapter returns two objects. `RagDataset` contains only answer text,
metadata, exact token IDs and character offsets, and token measurements.
`RagLabelSet` contains response and sentence labels. Fitting functions accept
a `FeatureTable`, not a label object. Each score file is written and hashed
before the corresponding labels are opened.

A sentence is positive when it overlaps at least one RAGTruth hallucination
span. The primary localization result is therefore **sentence-level**, not a
token-level span boundary metric.

The cache preserves the index of each omitted chunk and the number of
supporting chunks, but it does not preserve the exact omitted-chunk text or its
metadata. We do not reconstruct that text from a different source. The example
cards therefore report only the index of the chunk with the largest likelihood
drop. This is an input-provenance limitation, not an imputation.

## 3. Token-level evidence measurements

Let \(y_{it}\) be answer token \(t\) in response \(i\). For condition \(c\),
let

\[
\ell_{it}^{(c)}=\log p(y_{it}\mid c)
\]

be the target-token log-probability. A large value means the scorer finds the
fixed token more likely.

For two token distributions \(P\) and \(Q\), Jensen-Shannon divergence is

\[
\operatorname{JSD}(P,Q)
=\tfrac12\operatorname{KL}(P\|M)
+\tfrac12\operatorname{KL}(Q\|M),
\qquad M=\tfrac12(P+Q).
\]

The cache stores only the top 50 probabilities. We form the union of the saved
token IDs and add one `OTHER` category for all remaining probability mass.
This value is bounded and symmetric, but it is an approximation of
full-vocabulary JSD.

## 4. Evidence-Contrast feature contracts

Every feature is oriented so that a large value means "more grounded". The
final hallucination score is the negative fused grounding score. No label-based
sign change is allowed.

### EC-noctx-v1

This eight-feature contract is available for all tasks.

| feature | formula or meaning |
|---|---|
| mean full target log-probability | \(\operatorname{mean}_t\ell_t^{full}\) |
| negative top-50-plus-tail entropy | negative entropy over the saved top 50 categories plus one aggregate tail category |
| top-1/top-2 margin | mean difference between the two largest saved probabilities |
| negative top-50 tail mass | negative probability outside the saved top 50 |
| mean context gap | \(\operatorname{mean}_t(\ell_t^{full}-\ell_t^{noctx})\) |
| 90th percentile context gap | upper-tail token sensitivity to removing all context |
| approximate no-context JSD | mean top-50-plus-tail JSD between `full` and `noctx` |
| entropy increase | mean top-50-plus-tail entropy in `noctx` minus the same entropy in `full` |

### EC-full-v1

The LOO cohort adds six features. For chunk \(j\), define

\[
d_j=\operatorname{mean}_t
(\ell_t^{full}-\ell_t^{loo_j}).
\]

The added features are the maximum \(d_j\), mean of the two largest \(d_j\),
mean positive \(d_j\), maximum chunk JSD, mean of the two largest chunk JSDs,
and the fraction of tokens for which at least one chunk has a positive drop.

Length, chunk count, context length, dispersion and concentration are saved as
diagnostics. They are not fusion inputs because their grounding direction is
ambiguous.

## 5. Direct baselines

### Full-context perplexity

The ranking score is negative mean full-context log-probability. It measures
intrinsic uncertainty and does not use evidence removal.

### Likelihood gap

The grounding measurement is the mean full-versus-no-context gap. We negate it
to obtain a hallucination score.

### GASP-LL

This uses two likelihood measurements: the full-versus-no-context gap and the
largest LOO likelihood drop. Each column is standardized over the unlabeled
evaluation cohort and the standardized values are summed. The negative sum is
the hallucination score.

### GASP-top50

This uses four standardized grounding measurements:

\[
s_{GASP50}=-\sum z(
\text{context gap},
\text{no-context JSD}_{50},
\text{max LOO drop},
\text{max LOO JSD}_{50}).
\]

GASP defines this construction with full-vocabulary JSD. Our saved data permits
only the top-50-plus-tail approximation. `GASP-top50` is therefore not a
faithful reproduction of the published GASP score.

### Intrinsic mixed-v2 DUFS-LIU audit

The raw telemetry can also reconstruct the project's existing global
`dufs-liu-mixed-v2-development-2026-08-07` contract. It extracts the registered
30 full-answer entropy, energy, time-series and top-k views, applies the four
frozen mixed-v2 transformations, and fits the existing DUFS-LIU response
detector.

This optional baseline was implemented only after the version-1 RAGTruth test
labels had already been opened. Its fit is label-free and its scores have a
separate hash, but it is a **post-hoc audit**. It cannot enter the registered
success decision and is not available at sentence level because the original
contract is a global answer detector.

## 6. U-PCR and IU-PCR

Let \(F\in\mathbb R^{m\times n}\) contain \(m\) standardized grounding
features for \(n\) samples. Its covariance is

\[
C=FF^\top/n.
\]

The U-PCR model follows Dror et al., *Unsupervised Ensemble Regression*. Each
feature is treated as a noisy regressor of one latent continuous target. Under
its additive error model, the off-diagonal covariance equations have the form

\[
C_{rs}\approx\rho_r+\rho_s-g^2,\qquad r\ne s,
\]

where \(\rho_r\) measures the relation between feature \(r\) and the latent
target, and \(g^2\) is a shared scale term. `EC-U-PCR` uses the repository's
frozen L2 moment solve, scale ratio 0.25, weak-regressor exclusion and simple
average fallback.

Tenzer et al., *Crowdsourcing Regression: A Spectral Approach*, develop the
continuous spectral regression setting used by our two-component full-pool
anchor. `EC-IU-PCR` keeps every EC feature and restricts the inverse solve to
the two leading covariance eigenvectors. It is also the exact \(\lambda=0\)
anchor for the Laplacian method.

These names describe our implementation. Neither paper evaluates this RAGTruth
feature contract.

## 7. DUFS-gated Laplacian IU-PCR

DUFS is based on Lindenbaum et al., *Differentiable Unsupervised Feature
Selection based on a Gated Laplacian*. For feature \(r\), it samples

\[
z_r=\min(1,\max(0,\mu_r+\epsilon_r)),
\qquad \epsilon_r\sim\mathcal N(0,\sigma^2),
\]

and learns the gate survival probability

\[
p_r=\Pr(z_r>0)=\Phi(\mu_r/\sigma).
\]

We use the paper's parameter-free loss form, three seeds and 80 epochs. We do
not delete features. The mean gate probabilities define distances between
samples. A self-tuning 7-nearest-neighbour graph \(W\) is built in that space.
Its normalized Laplacian is

\[
L=I-D^{-1/2}WD^{-1/2}.
\]

For fused grounding score \(s=F^\top w\), graph roughness is

\[
\frac1n s^\top Ls=w^\top Rw,
\qquad R=FLF^\top/n.
\]

Let \(U\) contain the same two IU-PCR eigenvectors. After matching the scale of
projected \(R\) and projected \(C\), our adaptation solves

\[
w_\lambda=
U\left[U^\top(C+\lambda\bar R)U\right]^{-1}U^\top\hat\rho.
\]

The frozen value is \(\lambda=0.1\). At \(\lambda=0\), the implementation must
equal EC-IU-PCR exactly.

This combination is our adaptation. The DUFS paper does not propose this
U-PCR weight equation, and the U-PCR papers do not use DUFS gates.

## 8. Mechanism controls and assumptions

The registered controls are:

- an ungated graph with the same Laplacian solve;
- a sample-permuted graph;
- \(\lambda\in\{0,0.1,0.3,1,3,10\}\), used only for diagnostics;
- exact equality with IU-PCR at \(\lambda=0\).

The fusion mechanism needs these assumptions:

1. DUFS must preserve an unlabeled geometry related to grounding.
2. Neighbours in this geometry should have similar grounding scores.
3. Graph roughness must add information inside the two-component IU subspace.
4. The result must not be explained mainly by answer length, context length or
   chunk count.

A stable DUFS loss is only an optimization check. It does not prove any of
these scientific assumptions.

## 9. Metrics and uncertainty

**AUROC** is the probability that a randomly selected positive item receives a
higher score than a randomly selected negative item. **AUPRC** summarizes the
precision-recall curve and is useful when positive labels are rare.

We report both at response and sentence level. Confidence intervals use 1,000
paired bootstrap samples. Each sample draws complete `source_id` groups, so
all responses and conditions from a selected source move together.

The primary comparison is

\[
\Delta=\operatorname{AUROC}(EC\text{-}DUFS\text{-}LIU)
-\operatorname{AUROC}(GASP\text{-}top50)
\]

on LOO sentences. Full success also requires a positive paired interval over
EC-IU-PCR. That rule separates new evidence features from a real DUFS/Laplacian
contribution.

## 10. Frozen settings

| setting | value |
|---|---:|
| DUFS seeds | 11, 23, 37 |
| DUFS epochs | 80 |
| graph neighbours | 7 |
| IU components | 2 |
| headline \(\lambda\) | 0.1 |
| sensitivity path | 0, 0.1, 0.3, 1, 3, 10 |
| bootstrap samples | 1,000 |
| grouping unit | `source_id` |

The implementation fits standardization, U-PCR moments, DUFS gates and graphs
on the unlabeled samples of the evaluated split. This is **label-free
transductive fitting**. It is not an inductive model trained on one population
and applied unchanged to another.

## 11. References

- Dror et al., [*Unsupervised Ensemble Regression*](https://arxiv.org/abs/1703.02965).
- Tenzer et al., [*Crowdsourcing Regression: A Spectral Approach*](https://proceedings.mlr.press/v151/tenzer22a.html), AISTATS 2022.
- Lindenbaum et al., [*Differentiable Unsupervised Feature Selection based on a Gated Laplacian*](https://proceedings.neurips.cc/paper/2021/hash/0bc10d8a74dbafbf242e30433e83aa56-Abstract.html), NeurIPS 2021.
- Niu et al., [*RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models*](https://aclanthology.org/2024.acl-long.585/), ACL 2024.
- [*GASP*](https://arxiv.org/abs/2607.04223), arXiv:2607.04223.
