# RAGTruth mixed-v2 evidence-aware experiment

**Date:** 2026-08-10  
**Status:** frozen exploratory design; RAGTruth labels were already opened in an
earlier experiment, so this is not a blinded confirmation.

## 1. Scientific question

Can the RAG conditions `full`, `noctx`, and `loo_j` help DUFS weight the same
30 features used by the original mixed-v2 detector? This experiment does not
replace those features with a new Evidence-Contrast feature set.

The experiment separates three questions:

1. Do the original 30 features carry useful RAG grounding information?
2. Does evidence removal improve those features?
3. Does DUFS-LIU add value beyond IU-PCR after it receives that structure?

All feature construction, transformations, gates, graphs and fusion weights
are label-free. Labels are used only by the evaluation command.

## 2. The raw tensor

For response \(i\), evidence condition \(c\), and original feature \(f\),
define

\[
R_{i,c,f}=\phi_f(H_{i,c},D_{i,c},Z_{i,c},P_{i,c}).
\]

Here \(H\) is token entropy, \(D\) is target-token negative log-probability,
\(Z\) is token log-sum-exp, and \(P\) contains the saved top-50 token
probabilities. The functions \(\phi_f\) are exactly the 30 extractors in the
frozen mixed-v2 contract: 16 entropy-trajectory features, four target-token
energy features, four log-partition features, and six top-k distribution
features.

The answer tokens are identical across evidence conditions. Therefore a
change in \(R_{i,c,f}\) is caused by the scoring context, not by a newly
generated answer.

The number of LOO conditions differs between responses. The implementation
stores a ragged tensor and an explicit condition index. It never invents or
imputes a missing chunk.

## 3. One shared mixed-v2 coordinate system

Let \(G\) be the frozen mixed-v2 orientation and transformation. Its location,
scale and mode parameters are fitted from the unlabeled `full` rows only:

\[
T_{i,c,:}=G_{\mathrm{fit\ on\ full}}(R_{i,c,:}).
\]

The same fitted transform is applied to `noctx` and every `loo_j`. Conditions
are not standardized separately. This is necessary for meaningful
subtraction. It also makes

\[
T_{i,\mathrm{full},:}
\]

exactly equal to the original full-context mixed-v2 matrix. An automated test
checks that equality.

The input audit must show which of the 30 raw features is finite in every
condition. If any required value is unavailable, scoring stops. The code does
not replace a missing original feature with a median or a different feature.

## 4. Four matrices made from the same 30 features

Define the no-context and chunk-removal changes

\[
\Delta_i^{0}=T_{i,\mathrm{full},:}-T_{i,\mathrm{noctx},:},
\qquad
\Delta_i^{j}=T_{i,\mathrm{full},:}-T_{i,\mathrm{loo}_j,:}.
\]

A positive change means that removing evidence reduced the confidence-oriented
value of the same original feature.

### Original mixed-v2, full-context only

\[
B_{\mathrm{full}}=[T_{\mathrm{full}}]\in\mathbb R^{n\times30}.
\]

### Original-30 No-Context Contrast

\[
B_{\mathrm{noctx}}=[T_{\mathrm{full}},\Delta^0]
\in\mathbb R^{n\times60}.
\]

### Original-30 LOO Evidence-Aware

For every original feature, summarize only the observed \(\Delta_i^j\):

- maximum drop;
- mean of the two largest drops;
- mean of positive drops, or zero when no drop is positive;
- negative standard deviation across chunks, so a larger value means more
  stable evidence sensitivity.

Then

\[
B_{\mathrm{LOO}}=[T_{\mathrm{full}},\Delta^0,
\max_j\Delta^j,\operatorname{top2mean}_j\Delta^j,
\operatorname{positiveMean}_j\Delta^j,-\operatorname{std}_j\Delta^j]
\in\mathbb R^{n\times180}.
\]

The trace-length contrast is expected to be constant because the answer is
fixed. Constant derived columns are reported and removed before fusion; the
original trace-length feature remains in the full block.

### Hybrid

\[
B_{\mathrm{hybrid}}=[B_{\mathrm{LOO}},E_{\mathrm{EC-full-v1}}]
\in\mathbb R^{n\times194},
\]

where \(E_{\mathrm{EC-full-v1}}\) is the already frozen 14-column
Evidence-Contrast contract. The 30-feature blocks remain present and
identifiable.

## 5. IU-PCR and DUFS-LIU input

Each matrix \(B\) is standardized column by column without labels. Constant
derived columns are removed and recorded. Write the resulting feature-by-item
matrix as \(F=B^\top\).

IU-PCR uses the two-component moment solution. DUFS learns a soft gate for
every column of \(B\), builds a gated \(k=7\) sample graph, and DUFS-LIU solves

\[
w_\lambda=U[U^\top(C+\lambda\bar R)U]^{-1}U^\top\hat\rho,
\qquad \lambda=0.1.
\]

Seeds are 11, 23 and 37, with 80 parameter-free DUFS epochs. At \(\lambda=0\),
the result must be exactly equal to ordinary IU-PCR.

## 6. Comparisons and controls

The response-level LOO cohort contains QA and Data-to-Text responses. Every
method uses the same response order and source-grouped bootstrap samples.

Reference methods:

- `GASP-top50`;
- `EC-IU-PCR`;
- `EC-DUFS-LIU`.

For each of the four matrices above, run IU-PCR and DUFS-LIU. Also run:

- a sample-graph permutation that preserves graph weights but breaks their
  assignment to responses;
- an evidence-block permutation within each task that preserves every block's
  marginal distribution but breaks its pairing with the full response;
- exact \(\lambda=0\) equality.

The evidence-block permutation leaves the original full block fixed. In the
Hybrid matrix it also leaves the four intrinsic EC columns fixed and permutes
only evidence-derived EC columns.

## 7. Mechanism diagnostics

The report must include:

- IU and DUFS-LIU fusion weights for every named column;
- DUFS gate probabilities for full, no-context and mean-LOO coordinates;
- gate stability across seeds and effective feature count;
- gate summaries by original feature and evidence block;
- task-specific and chunk-index-specific univariate diagnostics;
- answer-length, context-length and chunk-count correlations, plus
  residualized AUROC;
- condition-block and graph-permutation controls;
- QA and Data-to-Text results separately;
- DUFS training-loss curves and graph diagnostics.

## 8. Interpretation boundary

This experiment is exploratory because RAGTruth labels were already visible.
No variant may be tuned on test. A selected method can support a hypothesis,
but a final claim requires a new benchmark, scorer, or otherwise untouched
confirmation set.

