---
slug: self-supervision-enhanced-feature-selection
title: "Self-Supervision Enhanced Feature Selection with Correlated Gates"
authors: "Changhee Lee (Chung-Ang University, Korea)*, Fergus Imrie (UCLA)*, Mihaela van der Schaar (Cambridge / UCLA / Alan Turing Institute). *equal contribution"
arxiv_id: "not found in extract"
venue: "ICLR 2022 (conference paper, stated in the page header)"
year: 2022
source_pdf: papers/SELF-SUPERVISION ENHANCED FEATURE SELECTION.pdf
extracted_text: papers/extracted/self-supervision-enhanced-feature-selection.md
last_digested: 2026-08-05
---

## Summary

SEFS is a **semi-supervised** embedded feature-selection method for the small-labelled-sample,
highly-correlated-feature regime (clinical / omics). Two phases: a **Self-Supervision Phase**
pre-trains an encoder on unlabeled data via two pretext tasks — reconstruct the original
feature vector, and predict which features were masked — and a **Supervision Phase** that
jointly updates the encoder, a predictor and a per-feature selection probability `π` under a
supervised loss plus an ℓ0 penalty. Its headline novelty is **correlated gates**: the masking
vector is drawn from a multivariate Bernoulli built with a **Gaussian copula** carrying the
input features' correlation matrix `R`, so correlated features tend to be masked *together*.

## ⚠ THE FACT THAT MATTERS FOR US: SEFS IS NOT LABEL-FREE AT SELECTION

The per-feature selection probability `π` is **fixed and EQUAL across all features throughout
the Self-Supervision Phase** — verbatim, extract lines 277-280: *"each feature is selected
based on the multivariate Bernoulli distribution with an equal probability π, i.e., π_k = π
for k ∈ [p], which is a hyper-parameter fixed throughout the Self-Supervision Phase. The
adoption of an equal selection probability allows us to make no assumptions about the relative
feature importance, which is not known a priori."* Figure 1's caption says the same.

`π` only becomes feature-specific in the **Supervision Phase**, under Eq. (7), whose objective
contains `ℓY(y, f_φ∘f_θ(...))` — i.e. **labels**. The self-supervised half trains an *encoder*,
not a selection.

**Consequence**: there is no published label-free SEFS ranking. The correlated-gate sampler is
a *masking scheme for pretext tasks*; the selection itself is supervised. Any "label-free SEFS"
would be our own construction, not this paper's method.

## Method detail (for reimplementation)

- **Gate generation (§4.1, Eq. 4)**: Gaussian copula `C_R(U_1..U_p) = Φ_R(Φ⁻¹(U_1),…,Φ⁻¹(U_p))`.
  Sample `v = Lε`, `ε ~ N(0, I)`, `R = LLᵀ` (Cholesky); `u_k = Φ(v_k)`; `m_k = 1` iff `u_k ≤ π_k`.
- **Masked input (Eq. 2)**: `x̃ = m ⊙ x + (1 − m) ⊙ x̄`, where `x̄ = E[x]` — unselected features
  are replaced by their **mean**, not by zero.
- **Self-supervision loss (Eq. 5)**: `ℓX(x, x̂) + α·ℓM(m, m̂)` with `ℓX = ‖x − x̂‖²₂` and `ℓM` the
  per-feature binary cross-entropy on the mask.
- **Supervision loss (Eq. 7)**: `ℓY(y, f(m̃ ⊙ x + (1 − m̃) ⊙ x̄)) + β Σ_k π_k`, where the ℓ0 term
  simplifies because `E‖m‖₀ = Σ_k P(U_k ≤ π_k) = Σ_k π_k`.
- **Relaxation (Eq. 6)**: `m̃_k = σ( (log π_k − log(1−π_k) + log U_k − log(1−U_k)) / τ )`,
  differentiable in `π` (Wang & Yin 2020).

## Datasets & models used

- **Synthetic**: Block-Structured Noisy Two-Moons (TPR against ground-truth relevance).
- **Clinical**: UK Cystic Fibrosis registry (UKCF) — 6,754 adults 2008–2015, p = 245 clinical
  variables (11 static + 3×78 time-varying); label = respiratory failure (death or lung
  transplant) within 5 years. Run at `n_l = 32`, `n_u = 4754`, `|S| = 10`.
- **Proteomics**: CCLE (Cancer Cell Line Encyclopedia) — 899 cancer cell lines, response to 11
  drugs from proteomic measurements.
- Downstream evaluator is an **MLP** trained on the selected features (deliberately shared
  across all methods, to isolate the selection from the model class).

## Methods it compared itself against

7 baselines: **Lasso** (Tibshirani 1996), **Tree** / extremely randomized trees (Geurts 2006),
**L-Score** / Laplacian Score (He 2005), **BNNsel** (Liang 2018), **STG** (Yamada 2020), **DUFS**
(Lindenbaum 2020), and **STG (SS)** — their own semi-supervised extension of STG. Plus two
ablations: **SEFS (no SS)** (no self-supervision phase) and **SEFS (indep)** (independent rather
than correlated gates).

## Experiments — methodology & scores

Labeled samples split train/test; only `n_l` labeled samples train the selector, while the full
labeled set evaluates the discovered features. Metrics: AUROC/AUPRC (classification), MSE
(regression), TPR (synthetic, where ground truth is known).

| Setup | Metric | Score | Notes |
|---|---|---|---|
| UKCF, \|S\|=10, n_l=32, n_u=4754 | AUROC | **SEFS 0.846 ± 0.013** | best of 9 |
| " | AUROC | STG (SS) 0.810 ± 0.036 | best baseline |
| " | AUROC | Tree 0.807 ± 0.036 | |
| " | AUROC | **DUFS 0.799 ± 0.039** | best *fully unsupervised* |
| " | AUROC | SEFS (no SS) 0.785 ± 0.044 | ablation: −6.1pp |
| " | AUROC | STG 0.781 ± 0.048 | |
| " | AUROC | Lasso 0.767 ± 0.054 | |
| " | AUROC | L-Score 0.668 ± 0.010 | |
| " | AUROC | BNNsel 0.650 ± 0.051 | worst |
| " | AUPRC | SEFS **0.532 ± 0.027** vs STG (SS) 0.477 | |
| Two-Moons, n_l=20, n_u=1000 | AUROC (repr.) | SEFS **0.92 ± 0.02** | vs SEFS (indep) 0.88, SEFS (AE) 0.85 |

The `SEFS 0.92` vs `SEFS (indep) 0.88` row is the paper's own evidence that the **correlated**
gating (not just the self-supervision) is worth ~4pp.

## Connection to our pipeline

Relevant to the Step-224 feature-selection channel, but **not usable as a label-free keep rule**
(see the ⚠ section). Two honest options if we want the contribution:

1. **Copula-sampler ablation** — swap SEFS's Gaussian-copula correlated gates into an existing
   *label-free* gate objective (DUFS's Eq. 7 in `spectral_utils/selectors/a2_groupfs.py`), which
   isolates the paper's actual novelty in our regime. This is our construction, and must be
   named as such.
2. **Pseudo-labelled SEFS** — run the Supervision Phase against an L-SML/U-PCR consensus
   pseudo-label, exactly as `a6_pseudolabel_gates` does for DUFS (Step 194). Also ours.

Note the paper's own finding cuts against the fully-unsupervised route it would put us on:
DUFS and L-Score are its two weakest-to-middling baselines on UKCF precisely because *"fully
unsupervised feature selection methods struggle to discover relevant features … without the
guidance of label information"*. That is the same wall Steps 221-223 hit.

Also relevant: our regime is **N ≫ D** (n ≈ 100–600 rows, p ≈ 28 views); SEFS is built for
**N ≪ D** (p = 245 with n_l = 32). The multicollinearity motivation transfers; the sample-scarcity
motivation does not.

## Notes / open questions

- `arxiv_id` genuinely not present in the extracted text (ICLR camera-ready header only).
- The covariance matrix `R` construction is deferred to Appendix B, which was not read here —
  needed before any reimplementation, since `R` is what makes the gates correlated.
- Gemini's Step-224 research brief described SEFS as a solution to our *unsupervised* selection
  problem. That is wrong at the selection step; recorded here so the error is not re-inherited.
