# RAGTruth original-30 evidence-aware experiment: methods

## 1. Purpose and status

This experiment asks whether RAG evidence conditions help IU-PCR or DUFS-LIU
use the project's original 30 mixed-v2 features. It does not replace those
features with Evidence-Contrast features.

RAGTruth labels were already opened in an earlier experiment. This comparison
is therefore exploratory, not a blinded confirmation. Labels are still used
only by the evaluation command. They do not enter feature extraction,
standardization, DUFS, graph construction or fusion.

## 2. Data and evaluation unit

The scorer reads one fixed RAGTruth answer under several conditions:

- all retrieved evidence is present (`full`);
- all evidence is removed (`noctx`);
- one evidence chunk is removed (`loo_j`).

The answer token IDs are identical in all conditions. Only the evidence in the
scoring prompt changes. The experiment uses the response-level LOO cohort:
QA and Data-to-Text responses with all registered leave-one-out conditions.
Summary is not included because its cache has no independent LOO conditions.

The number of chunks differs between responses. LOO values are stored as a
ragged tensor with explicit response offsets and chunk indexes. No missing
chunk is invented. The cache does not preserve the exact omitted-chunk text,
so chunk diagnostics use indexes only.

## 3. The same 30 original features in every condition

For response \(i\), condition \(c\), and original feature \(f\), define

\[
R_{i,c,f}=\phi_f(H_{i,c},D_{i,c},Z_{i,c},P_{i,c}).
\]

Here:

- \(H\) is the token entropy trace;
- \(D=-\log p(y_t)\) is the target-token negative log-probability trace;
- \(Z\) is the saved token log-sum-exp trace;
- \(P\) contains saved top-50 token probabilities.

The functions \(\phi_f\) are exactly the original extractors:

| Source | Features |
|---|---|
| entropy trajectory | `epr`, `trace_length`, six FFT features, two STFT features, `rpdi`, `sw_var_peak`, `pe_mean`, `hurst_exponent`, `cusum_max`, `cusum_shift_idx` |
| target-token negative log-probability | `epr_spilled`, `sw_var_peak_spilled`, `cusum_max_spilled`, `min_spilled` |
| log-partition trace | `epr_energy`, `min_energy`, `sw_var_peak_energy`, `cusum_max_energy` |
| saved top-k distribution | `mean_top1_logprob`, `logprob_margin`, `mean_logprob_entropy`, `varentropy`, `renyi_entropy_2`, `topk_tail_mass` |

The input audit measures availability separately for `full`, `noctx` and LOO.
Scoring stops if any required original feature is missing. There is no median
imputation and no replacement feature.

## 4. Frozen mixed-v2 transformation

Every original feature has the project's frozen confidence direction. Four
features use the frozen mixed-v2 operations:

| feature | operation |
|---|---|
| `pe_mean` | negative squared standardized value |
| `stft_spectral_entropy` | negative distance from its unlabeled mode percentile |
| `cusum_shift_idx` | confidence-oriented raw value |
| `rpdi` | confidence-oriented raw value |

The transformation parameters are fitted on the unlabeled full-context rows
only. The same parameters are then applied to no-context and LOO rows:

\[
T_{i,c,:}=G_{\text{fit on full}}(R_{i,c,:}).
\]

Conditions are not standardized separately. Therefore their difference has a
shared meaning. The implementation checks that \(T_{\mathrm{full}}\) equals
the original mixed-v2 transformation to numerical precision.

## 5. Four fixed matrices

Define

\[
\Delta_i^0=T_{i,full,:}-T_{i,noctx,:},\qquad
\Delta_i^j=T_{i,full,:}-T_{i,loo_j,:}.
\]

A positive value means that evidence removal lowered the same
confidence-oriented original feature.

### Original-30 full

\[
B_{full}=[T_{full}]\in\mathbb R^{n\times30}.
\]

This is the original mixed-v2 feature matrix and uses no RAG contrast.

### Original-30 no-context

\[
B_{noctx}=[T_{full},\Delta^0]\in\mathbb R^{n\times60}.
\]

### Original-30 LOO evidence-aware

For each original feature, the observed chunk changes are summarized by their
maximum, mean of the largest two, mean of positive changes, and negative
standard deviation. The last value is high when the evidence effect is stable
across chunks.

\[
B_{LOO}=[T_{full},\Delta^0,\max_j\Delta^j,
\operatorname{top2mean}_j\Delta^j,
\operatorname{positiveMean}_j\Delta^j,
-\operatorname{std}_j\Delta^j]
\in\mathbb R^{n\times180}.
\]

The fixed answer length makes the trace-length contrast constant. Constant
derived columns are reported and removed before fusion. The original
full-context trace-length column remains.

### Hybrid

\[
B_{hybrid}=[B_{LOO},E_{EC-full-v1}]\in\mathbb R^{n\times194}.
\]

The final 14 columns are the existing EC contract. The original 30-feature
blocks remain present and are not replaced.

## 6. IU-PCR

Each \(B\) is standardized column by column without labels. Let
\(F=B^\top\), with features in rows and responses in columns, and

\[
C=FF^\top/n.
\]

U-PCR is based on Dror et al., *Unsupervised Ensemble Regression*. It treats
each row as a noisy regressor of one latent continuous target and estimates its
target covariance from off-diagonal feature covariance equations.

IU-PCR uses the two-component continuous spectral formulation associated with
Tenzer et al., *Crowdsourcing Regression: A Spectral Approach*. In this
repository it is the full-pool, two-component U-PCR solve. It is the
\(\lambda=0\) anchor for every matrix.

## 7. DUFS-LIU

DUFS is based on Lindenbaum et al., *Differentiable Unsupervised Feature
Selection based on a Gated Laplacian*. It learns one stochastic soft gate for
each input column from unlabeled sample geometry. Our implementation uses the
paper's parameter-free loss with fixed seeds 11, 23 and 37 for 80 epochs.

The source paper uses gates for feature selection. DUFS-LIU is our adaptation:
the continuous gates define a response-neighbour graph, and its normalized
Laplacian changes the IU-PCR projected solve:

\[
R=FLF^\top/n,
\]

\[
w_\lambda=
U[U^\top(C+\lambda\bar R)U]^{-1}U^\top\hat\rho.
\]

The graph uses \(k=7\) neighbours and the headline value is \(\lambda=0.1\).
At \(\lambda=0\), the weights must equal IU-PCR exactly.

## 8. Reference methods

- **GASP-top50** uses full/no-context and LOO likelihood changes plus an
  approximate top-50-plus-tail Jensen-Shannon divergence. Published GASP uses
  full-vocabulary divergence, so this is explicitly an approximation.
- **EC-IU-PCR** fuses the existing 14 Evidence-Contrast columns with IU-PCR.
- **EC-DUFS-LIU** applies the existing DUFS-LIU graph to those 14 columns.

The EC reference score files are hash-verified and response IDs are matched
exactly before reuse.

## 9. Mechanism controls

The **graph permutation** preserves the graph spectrum and edge weights but
assigns graph nodes to different responses.

The **condition-block permutation** independently permutes every evidence block
within QA or Data-to-Text. It preserves each block's marginal distribution but
breaks its pairing with the full response. The original full block stays fixed.
For Hybrid, the four intrinsic EC columns also stay fixed.

The report also compares DUFS gates learned from the 30-column full,
no-context and mean-LOO matrices. These diagnostics show whether evidence
conditions change what DUFS relies on without changing the base feature list.

## 10. Evaluation

**AUROC** measures ranking quality; 0.5 is random. **AUPRC** emphasizes the
positive hallucination class and depends on its frequency.

Every interval uses 1,000 source-grouped bootstrap samples. Complete
`source_id` groups are resampled, and every method uses the same draws. Results
are reported for all LOO responses and separately for QA and Data-to-Text.

Because QA and Data-to-Text have different label rates and score
distributions, pooled AUROC can reward task identification. We therefore also
report **task-macro AUROC**, the equal-weight mean of the two within-task
AUROCs, and a diagnostic pooled AUROC after label-free standardization of each
score inside each task.

The evaluation also reports score correlations with answer length, context
length and chunk count, and AUROC after linear residualization of those three
variables.

## 11. Claim boundary

This matrix is fixed before the run and is not tuned on test. However, the
benchmark labels were already known from earlier work. The result can diagnose
the mechanism and nominate a hypothesis. A final performance claim requires a
new benchmark, scorer, or untouched confirmation set.

## References

- Dror et al., [Unsupervised Ensemble Regression](https://arxiv.org/abs/1703.02965), 2017.
- Tenzer et al., [Crowdsourcing Regression: A Spectral Approach](https://proceedings.mlr.press/v151/tenzer22a.html), AISTATS 2022.
- Lindenbaum et al., [Differentiable Unsupervised Feature Selection based on a Gated Laplacian](https://proceedings.neurips.cc/paper/2021/hash/0bc10d8a74dbafbf242e30433e83aa56-Abstract.html), NeurIPS 2021.
- Niu et al., [RAGTruth](https://aclanthology.org/2024.acl-long.585/), ACL 2024.
- [GASP](https://arxiv.org/abs/2607.04223), 2026.
