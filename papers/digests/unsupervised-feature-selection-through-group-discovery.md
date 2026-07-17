---
slug: unsupervised-feature-selection-through-group-discovery
title: "Unsupervised Feature Selection Through Group Discovery (GroupFS)"
authors: "Shira Lifshitz, Ofir Lindenbaum, Gal Mishne, Ron Meir, Hadas Benisty"
arxiv_id: "arXiv:2511.09166"
venue: "AAAI 2026"
year: 2026
license: "CC BY 4.0"
source_pdf: "n/a (fetched HTML: https://arxiv.org/html/2511.09166v1, 2026-07-17)"
extracted_text: papers/extracted/unsupervised-feature-selection-through-group-discovery.md
last_digested: 2026-07-17
---

## Summary

GroupFS is an end-to-end differentiable, purely-unsupervised feature selector that
*jointly* (a) discovers latent groups of related features and (b) selects the most
informative groups. It builds two graphs — one over samples, one over features — and
enforces Laplacian smoothness on both, plus a group-sparsity regularizer. The signal that
distinguishes it from per-feature selectors (LS, DUFS, CAE): informative structure often
lives in feature *groups* (adjacent pixels, brain regions, correlated indicators), so it
gates whole groups on/off rather than individual features. Ofir Lindenbaum (one of Omri's
advisors' orbit) is a co-author; DUFS (Lindenbaum 2021) is the per-feature predecessor.

## Method (three-term objective)

Total loss `L = L_s + lambda_1 * L_f + lambda_2 * L_reg` over learnable params
`{pi (d×C logits of assignment M), mu (C group-gate means), Q (C×C projection)}`:

- **L_s — sample-wise smoothness.** Gumbel-Softmax assignment `M ∈ R^{d×C}` (feature->group
  soft membership); one STG gate `z_j` per group; feature weights `zhat_i = sum_j M_ij z_j`;
  mask `Xtilde = X_B ⊙ zhat`; build a self-tuning random-walk graph `P_Xtilde` on the masked
  batch each iteration; `L_s = -(1/(Bd)) tr(Xtilde^T P_Xtilde^t Xtilde)` (t=2 diffusion).
  Rewards feature groups that vary smoothly across the sample manifold.
- **L_f — feature-wise smoothness.** Embed `F = M Q ∈ R^{d×C}`; feature-graph Laplacian
  `L_feat`; `L_f = (1/(dC))[tr(F^T L_feat F) + beta ||F^T F - I||_F^2]`. Encourages similar
  features (neighbors on the feature graph) to share group assignments; orthogonality term
  keeps the C group-embeddings distinct. Columns of F centered + unit-l2 renormalized after
  each step.
- **L_reg — group sparsity.** `L_reg = (1/C) sum_j Phi(mu_j/sigma) (1/d) sum_i M_ij`. Penalizes
  active gates weighted by group size -> keeps few, small groups active.

Warm-start: spectral-cluster the feature graph -> logits `log pi_ij = Delta` on the assigned
cluster (`Delta = log(p_main/p_rest)`, `p_main = 0.7`), 0 else; `mu_j = 0.5`; `Q` random
orthonormal, rows scaled inversely to cluster sizes. Selection = sort groups by gate mean,
keep the top-ranked groups. Number of groups C via the Appendix-D self-tuning Procrustes
distortion heuristic (or a small grid).

## Hyperparameter defaults (App. B.1)

`K=7` (self-tuning kNN), `t=2` (diffusion), `sigma=0.5` (STG noise), Adam `lr=1e-3`,
temperature annealed `start_t=10 -> min_t=1e-2` via `temp(e)=max(min_t, start_t-(start_t-min_t)e/E)`,
`lambda_1 ∈ {0.1,1,10,100}` (so L_s,L_f comparable in epoch 1), `beta=1/lambda_1`, `lambda_2`
grid (all-closed -> all-open). Final model = lowest total loss. Synthetic: 500 epochs, BS 100.

## Results (paper)

Nine datasets (ALLAML, Lung500, METABRIC, images, biological). Clustering accuracy (k-means
on selected features): GroupFS best or tied-best on 7/9 vs LS/MCFS/CAE/DUFS/MGAGR/CompFS.
On the synthetic two-moons-with-nuisance benchmark it recovers the planted informative groups
(`RG_sim=1`, `TPR=1`, `FDR=0`) whenever `C <= 2+(d-10)`.

## Connection to our pipeline (why it is selector A2)

GroupFS is exactly a "feature selection that runs BEFORE fusion" method: from the unlabeled
`n×p` feature matrix (our `cell.V`, already z-scored + sign-oriented) it picks a subset of
views, and — uniquely among our A1/A3 candidates — it *also emits a group assignment* over
those views. That group assignment is a drop-in for L-SML's clustering-swap seam
(`lsml_continuous(..., groups=...)`), which is what motivates the `a2.groups@good5` variant:
the theorem-validation finding was that L-SML's own spectral clustering never achieved the
within/between-cluster contrast (Lemma-4 gap negative in 4/4 cells), so replacing *only the
clustering* with GroupFS's discovered groups (features fixed to GOOD_5) isolates whether the
clustering — not the selection — was the bottleneck. The SAME existing L-SML then fuses.

## Adaptations for our setting (implemented in spectral_utils/selectors/a2_groupfs.py)

- Input `cell.V` is already z-scored + sign-oriented -> skip the paper's per-feature scaling.
- Sample graph is O(n^2); subsample rows to <=1200 (via `rng`) and build the graph per
  minibatch (as the paper does), so cost is O(B^2), B<=256.
- Kept the paper's DENSE self-tuning kernel (Eq. 1, K=7-th NN bandwidth) rather than the
  generic median-bandwidth + kNN-sparsify default — the paper specifies it explicitly.
- `lambda_2` (group sparsity) chosen label-free by SELECTION STABILITY (mean pairwise Jaccard
  of the selected set over 5 seeds), among `{lambda0/4, lambda0, 4*lambda0}`, replacing the
  paper's lowest-loss grid search (we have no held-out clustering-accuracy signal per cell).
- `C` via the Appendix-D distortion heuristic, `C_max = min(p-1, 8)` (our p<=16 h16 / <=~40 c46).
- `lambda_1` snapped to the nearest of `{0.1,1,10,100}` by the epoch-1 `|L_s|/|L_f|` ratio.

## Notes / open questions

- No official code release: LindenbaumLab GitHub (checked 2026-07-17) has 9 repos, none is
  GroupFS; a GitHub search for `2511.09166`/`GroupFS` returned nothing. Reimplemented from the
  fetched HTML text. Building blocks reused conceptually (cited): jsvir/lscae (LSCAE — stochastic
  gates + Laplacian-score terms) and Ofirlin/DUFS (Gated-Laplacian, the 2021 predecessor).
- The paper reports one explicit equation number (Eq. 1); other display equations are
  unnumbered in the HTML source and are cited by section in the extraction.
- This is a feature-selection method paper, not a hallucination-detection paper: it enters our
  roadmap purely as the A2 selector, evaluated by `scripts/run_selector_bench.py`. Not judged by
  pass/fail gates — all per-cell numbers are reported and the researcher chooses.
