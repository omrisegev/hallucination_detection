# Feature-Subset Selection Landscape — Research Memo

**Status**: research memo only — no `spectral_utils` code changes, no pilot, no GPU this session.
**Trigger**: Ofir + Bracha meeting (Jul 2026) — the algorithm needs an added contribution; the chosen
candidate is a principled, label-free, **in-pipeline feature-subset selection step**, replacing the
manual grid search over macros (GOOD_5 / GOOD_6 / top_macro_5 / GOOD_5+logprob / STABLE_H9 / …) whose
per-cell winners visibly differ across (dataset, model, temperature) cells. Conformal calibration is
explicitly parked for later.
**Related docs**: `docs/research_notes/thesis_pivot_assessment.md` (Step 151 — aggregator-swap pilot),
`Spectral_LSML_Report.html`, `Research_Directions.md`.

---

## 0. TL;DR

- We have hard evidence of an **in-cell selection prize (+7.6pp macro AUROC)** that is **not reachable
  by a fixed lookup table** (LOCO transfer is flat) — so the fix has to be a **per-cell, label-free**
  mechanism, not a smarter macro.
- The pipeline **already contains** several pieces of label-free machinery that were built for other
  purposes but are structurally the right building blocks for a selection step: per-feature
  constant/saturated dropping (`subset_sweep.py:330-352`), the L-SML Eq-14/15 residual used for K
  selection, and U-PCR's eigen-projection residual + weak-expert dropping. FUSE (Candès group, 2026)
  is proof this exact move — turn a label-free assumption-violation statistic into a selection
  objective — works at frontier scale on a sibling algorithm.
- Four research threads turned up one **very concrete, previously-unidentified lead**: Ofir's "trace
  of a sub-matrix" criterion is almost certainly the **Gated Laplacian** trace objective
  `Tr[X̃ᵀ L_X̃ X̃]` (Lindenbaum et al., NeurIPS 2021) — a label-free, subset-level feature-selection
  score built for exactly this problem shape (small n, fixed feature pool).
- **U-PCR and continuous L-SML are not the same algorithm** — same lineage, different structural
  models of the off-diagonal covariance (multiplicative rank-1 `v⊗v` vs. additive `ρᵢ+ρⱼ−g²`), and
  we have direct empirical evidence (`results/subset_sweep/method_grid.csv`) that **which structural
  model fits better varies systematically by domain** — itself a candidate label-free diagnostic.
- Five candidate designs (D1–D5) are proposed at the end, ordered roughly by implementation cost.
  None has been piloted; this is a literature + design landscape, not a decision.

---

## 1. Problem statement + evidence

### 1.1 The pool has outgrown manual curation

`CANONICAL_POOL` (`spectral_utils/subset_sweep.py:88`) now has **46 registered fusion features**: the
20 core H(n) features (`FEAT_NAMES`), 10 repgrid energy/logprob views
(`spectral_utils/repgrid_scoring.py:32-120`), and 16 sweep-only temporal/anomaly views. Hand-picked
macros exist for convenience — `GOOD_5 = {epr, low_band_power, sw_var_peak, cusum_max,
spectral_entropy}`, `GOOD_6 = GOOD_5 + varentropy`, `STABLE_H9`, `top_macro_5`, `consensus_4`,
`GOOD_5+logprob/energy/spilled` — but none of them is the actual winner most of the time.

### 1.2 No fixed macro wins consistently

`results/repgrid/headline_X_vs_Y.csv` (40 rows, our-method-vs-published-baseline comparisons across
cells) records which subset the label-selected `best_subset` column picked per cell:

| winning subset | count / 40 |
|---|---|
| top_macro_5 | 12 |
| GOOD_6 | 11 |
| GOOD_5+logprob | 6 |
| GOOD_5 | 3 |
| GOOD_5+energy | 3 |
| consensus_4 | 3 |
| GOOD_5+spilled | 1 |
| STABLE_H9 | 1 |

**GOOD_5 — the "main configuration" documented for the thesis — wins only 3/40 per-cell picks.** No
single macro is dominant; the right subset is cell-dependent.

### 1.3 The prize is real, but only reachable in-cell

From the exhaustive subset sweep (`spectral_utils/subset_sweep.py`, Step 153; `results/subset_sweep/
sweep_summary.csv`, 51 cells, sizes 3–16, continuous L-SML, label-free epr-anchored orientation):

| | macro AUROC |
|---|---|
| in-cell oracle (best-of-up-to-65k subsets, **label-peeking**) | **0.747** |
| GOOD_5 (fixed) | 0.671 |
| **gap** | **+7.6pp** |

Per-domain, the gap is not uniform — it is concentrated exactly where the current thesis positioning
is weakest: **RAG +14.1pp** (16 cells), **GPQA +10.2pp** (5 cells), reasoning domains (MATH-500,
GSM8K) only +1–2pp (already near-saturated by GOOD_5).

But this oracle number is a **label-peeking ceiling**, not an achievable target. The honest,
label-free comparator is **Leave-One-Cell-Out (LOCO)** subset transfer (`results/subset_sweep/
loco.csv`, 48 cells): **LOCO macro 0.664 vs. GOOD_5 0.674 vs. oracle 0.741** — LOCO is *flat or
slightly negative* relative to the fixed macro. **A domain-conditioned lookup table cannot capture
the prize.** The selection signal has to come from something computable **inside the cell itself**,
without labels — which is exactly the FUSE-style move this memo investigates.

### 1.4 The pipeline already has label-free feature-viability diagnostics — an existing building block

`_build_cell_context` (`subset_sweep.py:319-380`) already drops features per-cell via three label-free
tests before any subset search runs: `missing`, `length-mismatch`, `all-nonfinite`, **`constant`**
(`std < 1e-8`), and **`saturated`** (`is_saturated`, `subset_sweep.py:146-149`: more than 40% of
values equal the median). `results/subset_sweep/overview.csv`'s `dropped` column shows this firing
with a clear **domain pattern**, not noise: `stft_max_high_power` / `stft_spectral_entropy` drop on
nearly every short-answer QA/RAG cell (14/16 of the pool survives), `trace_length` drops on
long-chain-of-thought reasoning cells (near-constant when generation is capped), `rpdi` drops on
several RAG cells. **This is already a working precedent for "detect, label-free, which features are
structurally unusable in this cell" — the natural next step is to generalize it from single-feature
viability to subset-level fitness.**

### 1.5 The ρ≥0.75 correlation filter is empirically the wrong diagnostic (for continuous encoding)

Step 153 found that subsets containing a pairwise-|ρ|≥0.75 violation score **higher**, not lower,
mean AUROC (0.600 vs. 0.556) for continuous L-SML — the clustering step absorbs dependence rather than
being hurt by it. `results/subset_sweep/rho_check.csv` (203 rows, 4 correlation bins × cell) shows the
same non-monotonic pattern cell-by-cell (e.g. GPQA/DeepSeek-R1-8B: mean AUROC is *not* monotonically
decreasing across `<0.25 → 0.25-0.5 → 0.5-0.75 → ≥0.75` bins). **The pairwise-correlation filter
inherited from the supervised `best_nadler_on` code path is not the right label-free selection
criterion for continuous L-SML** — this matters directly for any D1/D4 design below: whatever
violation statistic gets adopted must be validated against this exact null result, not assumed.

### 1.6 Which structural model fits a cell is itself informative — the U-PCR vs. L-SML comparison

`results/subset_sweep/method_grid.csv` (9,719 subset-fits, `lsml`/`flat`/`avg`/`upcr` columns) gives a
direct empirical answer to "are L-SML and U-PCR the same algorithm": **no.** Continuous L-SML wins
more often on average (mean AUROC 0.620 vs. U-PCR 0.597; L-SML beats U-PCR on 62% of subsets overall)
but the win rate is **strongly domain-dependent**: GSM8K 90% (L-SML dominant), repgrid 71%, QA 61%,
but GPQA and RAG are near coin-flips (53%, 53%) — meaning on a sizable fraction of subsets in those
domains, U-PCR's additive structural model fits the covariance better than L-SML's multiplicative
rank-1 model. §2.4 below unpacks why, and why this is itself a candidate label-free diagnostic
(feeds D1).

---

## 2. Assumptions audit

For each method, the assumption, the primary-source quote (verbatim, from `papers/extracted/`), how
the pipeline addresses it today, observed violation evidence from prior validation runs (not
speculation), a candidate label-free diagnostic, and a remedy hook.

### 2.1 SML (Jaffe–Nadler–Kluger, AISTATS 2015, arXiv:1407.7644 —
`papers/extracted/estimating-the-accuracies-of-multiple-classifiers-without-la.md`)

| assumption | source quote | pipeline today | observed violation evidence | label-free diagnostic | remedy hook |
|---|---|---|---|---|---|
| (i) instances iid | "The n instances xj are i.i.d. realizations from the marginal pX(x)." (§2) | Implicit — samples are per-(question,generation) rows | Not tested; likely fine within a cell (independent generations), possibly violated across K-sampled repeats of the same question (Phase-15 K-sweep cells) | Check for residual autocorrelation across same-question repeats | Exclude same-question repeats from a single fusion fit, or block-bootstrap by question |
| (ii) pairwise conditional independence given Y | "The m classifiers are conditionally independent... Pr(fi=ai,fj=aj\|Y=y) = Pr(fi=ai\|Y=y)Pr(fj=aj\|Y=y)." (§2, Eq. 1) | Not enforced directly; L-SML's latent-group extension (2.2) is the paper's own fix | §1.5: the ρ≥0.75 filter (a proxy for this assumption) is **empirically refuted** for continuous L-SML — violating subsets score *higher* | FUSE's Ŝ statistic (§2.3.4) generalizes this to triplets and is label-free | D1/D4 — use Ŝ or a pairwise analogue as the *actual* selection objective instead of a hard ρ threshold |
| (iii) majority better-than-random | "Most of the classifiers are better than random... for more than half of all classifiers, πi > 0.5... needed given an inherent ±1 sign ambiguity." (§2) | `ANCHOR_PRIORITY` (`subset_sweep.py:111`) picks a label-free sign anchor (epr first) | Step 153: label-free anchor picks the **wrong global sign on 3/29 cells** (~10%) — the "better than random" premise is not always safely satisfied at cell level; Step 184: anchor choice (epr vs. cusum_max) agrees exactly (Δ=0.0000) on 18/19 non-pilot cells, disagrees only on the n=30 CoQA pilot (small-n fragility) | Anchor-agreement across ≥2 independent anchors as a per-cell confidence signal | If anchors disagree, flag the cell as sign-ambiguous rather than silently picking one (D1 diagnostic gate) |

### 2.2 L-SML (Jaffe–Fetaya–Nadler–Jiang–Kluger, AISTATS 2016, arXiv:1510.05830 —
`papers/extracted/unsupervised-ensemble-learning-with-dependent-classifiers.md`)

| assumption | source quote | pipeline today | observed violation evidence | label-free diagnostic | remedy hook |
|---|---|---|---|---|---|
| latent-group model — K≤m latent variables, features conditionally independent given Y unless they share a group | "the unobserved αk are conditionally independent given the true label Y... Classifiers that depend on different latent variables are... conditionally independent given Y, whereas classifiers that depend on the same latent variable may have strongly correlated prediction errors." (§4.1) | `_score_matrix_lsml` (Eq. 15) + `_spectral_cluster_precomputed` (`fusion_utils.py:354,375`) | Step 136: dependence is genuinely block-structured (band-power block ρ 0.77–0.88, `results/feature_correlation_16.csv`), matching the model's premise — this is the one assumption with the *best* empirical fit | The existing Eq-14 residual already IS this diagnostic | None needed — this is the assumption our features satisfy best; a selection step should *preserve* this structure, not fight it |
| two-rank-one off-diagonal covariance (`von(von)ᵀ` within-group, `voff(voff)ᵀ` across-group) | "The population covariance matrix is therefore a combination of two rank-one matrices." (§4.2) | `sml_fuse_signed` cross-weights = leading eigenvector of cluster off-diag covariance (`fusion_utils.py:320-347`) | Step 136: cross-weights genuinely separate at K≥3 (0.34/0.33/0.30/0.02 pe_mean-suppression case) — structure holds; K=2 is structurally forced to 0.50/0.50 (not evidence either way) | — | — |
| K unknown, chosen by residual minimization | "Minimizing the residual of Eq. (14) for a general covariance matrix R̂ is NP-hard" (Lemma 3) — paper's own greedy/heuristic search is approximate | `_residual_lsml` (`fusion_utils.py:422`) exact grid over K∈{2..8}; small enough m that NP-hardness doesn't bite | 5-feature pool is K=2 in 16/28 cells (not "always", but a majority) — a genuinely weaker clustering test than the 16-feature pool (`[[feedback-lsml-5feat-degenerate]]`, verified 2026-07-01) | — | Feature-count matters for whether K-selection is even informative; a selection step that shrinks to 5 features partially defeats the K-selection machinery it relies on |

### 2.3 U-PCR (Dror–Nadler–Bilal–Kluger, arXiv:1703.02965 / Tenzer et al., AISTATS 2022 —
`papers/extracted/unsupervised-ensemble-regression.md`)

| assumption | source quote | pipeline today | observed violation evidence | label-free diagnostic | remedy hook |
|---|---|---|---|---|---|
| uncorrelated errors: `E[hᵢ(X)hⱼ(X)] = 0` | "we assume the m regressors make independent errors with respect to g(X), namely that... E[hi(X)hj(X)] = 0." (Eq. 13) — "Remark 1: assumption (13)... may be strongly violated at least for some pairs." (paper's own caveat) | `upcr_fuse` (`fusion_utils.py:1113`) — auto k∈{1,2} eigen-truncation + weak-expert dropping is the paper's built-in robustness mechanism for exactly this violation | §1.6: not directly tested per-pair, but the domain-dependent L-SML-vs-U-PCR win rate (90% GSM8K vs. 53% GPQA/RAG) is consistent with this assumption holding better on some domains than others | Residual of the additive-model fit (‖ρ̂ − projection‖/‖ρ̂‖, already computed inside `upcr_pipeline`) vs. the L-SML rank-1 residual — **the smaller residual tells you which structural model to trust, per cell, label-free** | D1 — use this residual comparison directly as a per-cell method-selection signal (see §2.4) |
| first two moments of Y known (`g² = Var(Y)`) | Requires `g2` (variance of the "consensus" `g(X)`) as an input to the linear system `Cᵢⱼ = ρᵢ + ρⱼ − g²` | **`var_y=0.25` hardcoded** (Bernoulli-max-variance assumption, i.e. assumes a 50/50 label split) | Not yet audited against actual per-cell positive-rate — `results/subset_sweep/overview.csv`'s `pos_rate` column ranges 0.242–0.705 across cells, i.e. **the true Bernoulli variance ranges ~0.18–0.25, not fixed at the hardcoded ceiling** | Replace the constant with the per-cell `pos_rate*(1-pos_rate)` — this uses only the *marginal* label rate, not per-sample labels, so it stays within the unsupervised-mean-known-a-priori spirit the paper allows, but is currently NOT done | Cheap, concrete audit item — recompute `g2_hat`'s grid search bounds from per-cell `pos_rate` instead of a fixed 0.25 |
| m≥3 experts (for the linear system to be solvable) | "if m ≥ 3 there are enough linearly independent equations to uniquely recover a" (Thm. 1) | Enforced implicitly by subset-size floor | Not violated in practice (min subset size ≥3 in the sweep) | — | — |

### 2.4 Continuous flat-SML vs. U-PCR — the two structural models, head to head

SML/L-SML model the off-diagonal covariance as **rank-1 multiplicative**: `Cᵢⱼ = vᵢvⱼ` (a single
shared "quality" vector `v`). U-PCR models it as **additive**: `Cᵢⱼ = ρᵢ + ρⱼ − g²` (each feature has
its own scalar reliability `ρᵢ`, combined additively, no cross term). These are genuinely different
generative assumptions about *how* features' errors correlate — not two names for the same fusion. §1.6's
empirical split (L-SML dominant on GSM8K/repgrid, near-coin-flip on GPQA/RAG) says **the right
structural model is itself domain- and possibly cell-dependent** — which is exactly the kind of
label-free, data-derived signal a selection step should exploit rather than commit to one fusion
family globally. Concretely: computing *both* residuals (L-SML's Eq-14 rank-1 residual and U-PCR's
projection residual) on a cell's covariance, label-free, and preferring whichever fits better is a
near-zero-cost diagnostic already implementable from code that exists (`fusion_utils.py:422`,
`:1242`) — this is D1's most direct, lowest-risk instantiation.

**This directly answers a standing question: is U-PCR the same algorithm as our continuous L-SML?
No — same Nadler-lineage family (both descend from the rank-1-covariance idea in Parisi et al. 2014),
but different structural models, different estimators (leading eigenvector vs. least-squares + a g²
grid search), different weight semantics (∝ balanced accuracy vs. MSE-optimal), and different
dependence handling (latent-group clustering vs. eigen-truncation + dropping). Our move to continuous
(z-scored) encoding pulls SML's discrete accuracy formalism toward U-PCR's native continuous-regression
setting, which is part of why the two now perform closer to each other than the discrete-vs-continuous
literature would suggest — but they remain distinct models, and §1.6 shows the choice between them
measurably matters and varies by domain.**

### 2.5 FUSE (Lee, Ma, Zhao, Nair, Spector, Cohen, Candès — arXiv:2604.18547 —
`papers/extracted/fuse-ensembling-verifiers-with-zero-labeled-data.md`)

| assumption | source quote | our analogue | remedy hook |
|---|---|---|---|
| triplet conditional independence (TCI) | "each triplet of verifiers produces conditionally independent scores given the ground-truth label" (Assumption 2.2) | Same rank-1/latent-tree moment structure as SML/L-SML — TCI is the natural generalization to check on our 46-feature pool | D1/D4 |
| Ŝ — label-free TCI-violation statistic | "Var(T_{j1,j2,j3}/Σ_{j1,j2}) ... = 0 [under TCI]... Proposition 2.4 provides a measure of violations of TCI that does not require ground-truth labels." (Eq. 4, §2.2) | No direct analogue computed today | **D1's primary candidate objective** — compute Ŝ over triplets of our features, minimize over subset choice or monotone per-feature transforms |
| binarization search | "we find τ⋆ which (approximately) minimizes Ŝ(τ) ... via coordinate descent" (§2.2) | No analogue — our features are continuous-encoded, not thresholded | D4 — adapt the transform-search idea to monotone feature reshaping instead of hard binarization, since §1.3/Step 134 already show continuous >> binary for us |
| dropping poor verifiers | "drop verifiers with estimated balanced accuracy less than ½... even prior works that focus on theoretical guarantees (e.g., Tenzer et al. 2022) use similar dropping heuristics" (App. D) | U-PCR's weak-expert dropping (`upcr_fuse`) already does this | Reuse directly as a pre-filter before any new selection step |

---

## 3. Web-research threads — annotated bibliographies

Grounding: every citation below was verified either by fetching the paper's abstract/landing page
directly (this session) or via the sub-agent's own fetch (marked). Items the agents could not verify
are explicitly tagged UNVERIFIED — treat those as leads to confirm with Ofir/primary sources, not facts.

### Thread A — Ofir Lindenbaum's feature-selection line, and the "trace of a sub-matrix" identification

**The lead is identified.** Ofir's "sub-matrix / trace" criterion is the objective of **Differentiable
Unsupervised Feature Selection based on a Gated Laplacian** (Lindenbaum, Shaham, Svirsky, Peterfreund,
Kluger — **NeurIPS 2021, arXiv:2007.04728** — spot-checked by direct fetch this session: title,
authors, and the Laplacian-score/gating description confirmed). The loss (paper's Eq. 6–7) is
`−Tr[X̃ᵀ L_X̃ X̃]/m + λ·Σᵢ ℙ(Zᵢ≥0)`, where `X̃` is the feature matrix gated by stochastic gates `Z` (a
literal *sub-matrix* of selected columns) and `L_X̃` is the graph Laplacian **recomputed on that
sub-matrix** — an exact match to "a sub-matrix where you look at its trace," and notably a
**subset-level**, not per-feature, score: it is fully unsupervised and built for small-`n`, fixed-`m`
tabular problems, which is our exact regime.

| paper | authors | venue/yr | ID | what it does | pipeline mapping | unsup.? | cost at n≈300, m=46 |
|---|---|---|---|---|---|---|---|
| Stochastic Gates (STG) | Yamada, Lindenbaum, Negahban, Kluger | ICML 2020 | arXiv:1810.04247 | Differentiable ℓ0-relaxation gating jointly trained with a predictor; foundational primitive for the whole line | selection objective (supervised) | No | Low code, but needs labels |
| **Gated Laplacian** (the trace-of-a-sub-matrix method) | Lindenbaum, Shaham, Svirsky, Peterfreund, Kluger | NeurIPS 2021 | arXiv:2007.04728 (fetched, confirmed) | Unsupervised subset selection: gate the feature matrix, score the gated sub-matrix by its Laplacian-score trace, optimize gates end-to-end | **selection objective — top D1 candidate** | **Yes** | Medium — small graph on 46 features is cheap; main tuning knob is the gate-sparsity λ per cell |
| LSCAE (correlated+nuisance FS) | Shaham, Lindenbaum, Svirsky, Kluger | Neural Networks 2022 | arXiv:2110.05306 | Concrete-autoencoder reconstruction + Laplacian-score term; drops both nuisance and redundant/correlated features; user sets subset size k | grouping/K selection | Yes | Medium — the "k features" knob maps directly onto our GOOD_5/GOOD_6 sizes |
| LSPIN (instance-wise gates) | Yang, Lindenbaum, Kluger | ICML 2022 | arXiv:2106.06468 | Per-sample (not per-dataset) sparse feature subset via a gating network | router/gating, per-sample | No | Likely underpowered at ≤500 samples/cell |
| L0-Sparse CCA | Lindenbaum, Salhov, Averbuch, Kluger | ICLR 2022 | arXiv:2010.05620 | Sparse-gated CCA between two feature views | selection objective (cross-view) | Partial | Only relevant if feature families are framed as CCA views |
| Multi-modal unsupervised FS | Yang, Lindenbaum, Kluger, Jaffe | UAI 2023 | arXiv:2303.09381 | Gated-Laplacian extended to a shared graph across modalities, no labels | selection objective, grouping | Yes | Medium — overkill unless feature families = modalities |
| Contextual FS (c-STG) | Sristi, Lindenbaum, Lavzin, Schiller, Mishne, Benisty | ICML 2024 | arXiv:2312.14254 | Gates conditioned on external context variables — subset varies with context | router/gating (context→subset) | No | Needs labels + many cells to train the conditioner; conceptually close to D5 |
| Spectral Self-supervised FS | Segal, Lindenbaum, Jaffe | TMLR 2024 | arXiv:2407.09061 | Graph-Laplacian-eigenvector pseudo-labels selected by model-stability, then features ranked against them | selection objective, diagnostic | Yes | Low-medium — label-free, robust to outliers, good fit at our scale |
| GroupFS | Lifshitz, Lindenbaum, Mishne, Meir, Benisty | AAAI 2026 | arXiv:2511.09166 | Differentiably *discovers* latent feature groups (no predefined partition) via Laplacian smoothness + group sparsity, no labels | grouping/K selection | Yes | Medium — newest, most directly "which features act together" |

Researchers/groups to watch: **Ofir Lindenbaum** (Bar-Ilan), **Yuval Kluger** (Yale, recurring senior
co-author), **Ariel Jaffe** (HUJI — also an L-SML co-author, a direct thesis-internal link), **Gal
Mishne** (UCSD) and **Hadas Benisty** (Technion, newest grouping work). "COPER" is real but is a
clustering paper, not FS (ICLR 2025, tangential). "DiCoLo" could not be verified as a Lindenbaum paper
— UNVERIFIED, drop it.

### Thread B — Boaz Nadler portfolio + SML-family follow-ups

Closes the Parisi-2014 citation gap and identifies K-selection prior art.

| paper | authors | venue/yr | ID | what it does | pipeline mapping | unsup.? |
|---|---|---|---|---|---|---|
| **Ranking and combining multiple predictors without labeled data** (lineage root) | Parisi, Strino, Nadler, Kluger | **PNAS 111(4):1253–1258, 2014** | **arXiv:1303.3257**, DOI 10.1073/pnas.1219097111 (fetched, confirmed — title/authors match exactly) | Rank-1 off-diagonal covariance under CI; leading eigenvector ∝ balanced accuracies. Founding result the whole SML/L-SML/FUSE family builds on. | background / lineage root | Yes |
| SML | Jaffe, Nadler, Kluger | AISTATS 2015 | arXiv:1407.7644 (cached, digested) | Our fusion's direct scalar-label ancestor | core (already used) | Yes |
| L-SML | Jaffe, Fetaya, Nadler, Jiang, Kluger | AISTATS 2016 | arXiv:1510.05830 (cached, digested) | Our project's core fusion algorithm | core (already used) | Yes |
| U-PCR | Dror, Nadler, Bilal, Kluger | arXiv 2017 / Tenzer et al., AISTATS 2022 | arXiv:1703.02965 (cached, digested) | Continuous-regression analogue | core (already used) | Yes |
| **Non-Parametric Detection of the Number of Signals** | Kritchman, Nadler | IEEE Trans. Signal Processing 57(10), 2009 | DOI 10.1109/TSP.2009.2022897 | Sequential Tracy-Widom hypothesis tests on sample-covariance eigenvalues to count how many exceed the RMT noise bulk | **K-selection for latent clustering** | Yes |
| Determining the number of components in a factor model | Kritchman, Nadler | Chemometrics & Intell. Lab. Systems 94(1), 2008 | DOI 10.1016/j.chemolab.2008.06.002 | Matrix-perturbation + RMT twin of the above, framed for few high-dim noisy samples | K-selection | Yes |
| A Deep Learning Approach to Unsupervised Ensemble Learning | Shaham, Cheng, Dror, Jaffe, Nadler, Chang, Kluger | ICML 2016 | arXiv:1602.02285 (cached) | RBM/DNN generalization for strongly-dependent classifiers; shows accuracy loss as dependence grows | alternative fusion / violation-vs-performance curve | Yes |
| SUMMA | Ahsen, Vogel, Stolovitzky | JMLR 20(166), 2019 | arXiv:1802.04684 | Non-Nadler spectral meta-learner for *ranked* predictions, no labels | alternative fusion structure | Yes |
| Estimating Accuracy from Unlabeled Data | Platanios, Blum, Mitchell (+ Poon, Horvitz for the 2017 logic extension) | UAI 2014 / arXiv:1705.07086 (2017) | — | Agreement-rate-based accuracy + dependency recovery, no labels | dependence/CI diagnostic | Yes |
| Spectral Methods Meet EM (spectral Dawid-Skene) | Zhang, Chen, Zhou, Jordan | NIPS 2014 / JMLR 17, 2016 | arXiv:1406.3824 | Method-of-moments spectral init + one EM step for discrete confusion matrices | background (discrete, less directly applicable) | Yes |
| Unsupervised Ensemble Learning via Deep Energy-based Models | Maymon, Buznah, Shaham | AISTATS 2026 | arXiv:2601.20556 | Newest lineage descendant — EBM meta-learner with CI-based guarantees | alternative fusion structure (watch) | Yes |

**Kritchman-Nadler applicability, with a caveat**: both papers solve exactly our K-selection
sub-problem (how many eigenvalues of a 46×46, or subset-sized, sample covariance are real signal vs.
random-matrix noise), but their Tracy-Widom/Marčenko-Pastur asymptotics assume both n and m large at a
stable ratio. At n≈100–500 and m up to 46 (ratio as low as ~2), the noise-edge threshold is only
approximate — treat `K̂` as an anchored prior, sanity-checked against the eigengap the pipeline already
logs, not a hard cutoff.

### Thread C — Tabular foundation-model frontier (2024–2026) and transferable concepts

Scope note: nearly every tabular *foundation model* below is fundamentally supervised (needs an
in-context labeled split); they're included for the architectural/meta-learning concepts, not as
drop-in tools. The genuinely unsupervised, small-n-adoptable rows are at the bottom.

| paper | authors | venue/yr | ID | transferable concept | unsup.-compatible? |
|---|---|---|---|---|---|
| **TabPFN v2** | Hollmann, Müller, Purucker, Krishnakumar, Körfer, Hoo, Schirrmeister, Hutter (+Bergman) | **Nature, Jan 2025**, DOI 10.1038/s41586-024-08328-6 (fetched, confirmed) | HF `Prior-Labs/TabPFN-v2-clf` | Amortized meta-learning across many small synthetic tabular tasks, adapt per-instance at inference with no per-task gradient training — the closest conceptual mirror of "learn what works across cells, adapt per cell" | Partial — ICL needs labels; the *prior-fitting/amortization paradigm* is label-agnostic in principle |
| TabICL | Qu, Holzmüller, Varoquaux, Le Morvan | ICML 2025 | arXiv:2502.05564 | Set-Transformer column embedder = permutation-invariant feature-set encoding; attention weights = interpretable per-cell soft feature-importance | Partial (head is supervised; encoder bias reusable) |
| CARTE | Kim, Grinsztajn, Varoquaux | NeurIPS 2024 | arXiv:2402.16785 | Cross-table transfer without matched schemas — analog for a selector that generalizes across heterogeneous cells | No (supervised) |
| FT-Transformer | Gorishniy, Rubachev, Khrulkov, Babenko | NeurIPS 2021 | arXiv:2106.11959 | Canonical "attention over features" architecture — attention weights ARE an implicit, differentiable, inspectable soft feature-selection map | No (supervised), but the primitive is generic |
| Survey on Deep Tabular Learning (incl. TabNet) | — (author list UNVERIFIED) | arXiv, Oct 2024 | arXiv:2410.12034 | Points to **TabNet's instance-wise learnable feature masks** — closest existing "select features adaptively per instance" mechanism | Partial |
| OpenFE | Zhang, Zhang, Fan, Luo, Liu, Liu, Cao, Li | ICML 2023 | arXiv:2211.12507 | Efficient two-stage prune-from-a-pool search pattern (label-driven scoring, but the search *pattern* transfers) | No |
| CAAFE | Hollmann, Müller, Hutter | NeurIPS 2023 | arXiv:2305.03403 | LLM-as-selector using dataset metadata — a cheap heuristic prior for the manual macro grid | No (labeled validation loop) |
| LassoNet | Lemhadri, Ruan, Abraham, Tibshirani | AISTATS 2021 / JMLR 22 | arXiv:1907.12207 | Full regularization path → nested feature subsets; swap the loss for a reconstruction/consistency objective to go label-free | Partial |
| **Concrete Autoencoders** | Balın, Abid, Zou | ICML 2019 | arXiv:1901.09346 | End-to-end differentiable, fully unsupervised, fixed-`m`, pick-`k`-of-`m` global feature selection via reconstruction | **Yes — most directly adoptable primitive at our scale** |
| Worse-than-Random (UFS baseline) | Rajabinasab, Houle, Chelly, Zimek | Pattern Recognition Letters, 2026 (preprint) | arXiv:2605.22973 | Governance finding: many SOTA unsupervised FS methods lose to *random* subset selection — mandates a random-subset floor for any claim | Yes (about UFS methodology) |

**Synthesis (3 most promising transferable concepts)**: (1) **Concrete Autoencoders** — the only
mechanism that is simultaneously unsupervised, fixed-pool, and comfortable at m=46/n≈300; a
Gumbel-Concrete gate picking k-of-46 to best reconstruct the held-out feature matrix per cell would
replace the manual macro grid with a label-free, differentiable choice. (2) **Attention-over-a-set as
implicit soft weighting** (FT-Transformer/TabICL/TabNet lineage) — needs an unsupervised objective
(reconstruction, cross-view consistency) grafted on since the originals are supervised, but the
inductive bias (order-invariant, per-instance, continuous weights instead of a hard subset) is
directly reusable. (3) **Amortized meta-learning across many small tasks, adapt per-instance**
(TabPFN v2 / XTab's shared-backbone-plus-per-cell-featurizer paradigm) — the strongest conceptual
match to the "learn across cells, deploy per cell" framing, feeding D5's cross-cell router flavor.
**Mandatory guardrail**: whatever gets built must clear the random-subset floor (Rajabinasab et al.
2026) — this should also be applied retroactively as a sanity check on the existing hand-picked
macros.

### Thread D — Assumption diagnostics + per-instance adaptive selection

| paper | authors | venue/yr | ID | maps to | unsup.? |
|---|---|---|---|---|---|
| **Vanishing tetrads** (Confirmatory Tetrad Analysis) | Bollen, Ting | Sociological Methods & Research, 1998 | DOI 10.1177/0049124198027001002 | **Direct label-free test of the rank-1 (SML `v⊗v`) off-diagonal structure** — a tetrad `σᵢⱼσₖₗ − σᵢₗσₖⱼ` vanishes in population under a single-factor model; bootstrap statistic is robust at small n | Yes |
| KCIT (kernel CI test) | Zhang, Peters, Janzing, Schölkopf | UAI 2011 | arXiv:1202.3775 | Nonparametric pairwise/triplet CI test conditioned on a proxy label | Yes |
| HSIC | Gretton, Fukumizu, Teo, Song, Schölkopf, Smola | NIPS 2007 | ACM 10.5555/2981562.2981636 | Cheap nonlinear pairwise-dependence screen — nonparametric cousin of the ρ-filter | Yes |
| Eigenvalue-ratio rank test | Ahn, Horenstein | Econometrica, 2013 | DOI 10.3982/ECTA8968 | Principled, threshold-free "is this covariance rank-1/rank-2?" test — sharper than an eigengap heuristic | Yes |
| Robust-PCA source-dependency recovery | Varma, Sala, He, Ratner, Ré | ICML 2019 | arXiv:1903.05844 | Inverse-covariance robust-PCA recovers weak-source dependency structure with no labels — a structurally-grounded nonparametric replacement for the ρ≥0.75 filter | Yes |
| Snorkel MeTaL | Ratner, Hancock, Dunnmon, Sala, Pandey, Ré | AAAI 2019 | arXiv:1810.02840 (not directly fetched — UNVERIFIED ID) | Matrix-completion on inverse covariance to recover source correlations, no labels | Yes |
| **FUSE** (already in §2.5) | Lee, Ma, Zhao, Nair, Spector, Cohen, Candès | arXiv, 2026 | arXiv:2604.18547 | Ŝ statistic — the single most directly reusable label-free violation objective | Yes |
| Algorithm Selection Problem (framing) | Rice | Advances in Computers 15, 1976 | — | Problem-space → feature-space → algorithm-space → performance; the vocabulary for a signature→config router | N/A |
| Automated Algorithm Selection survey | Kerschke, Hoos, Neumann, Trautmann | Evolutionary Computation 27(1), 2019 | — | Practical checklist: which meta-features work, how selectors are trained | Trained on labels |
| Auto-sklearn 2.0 | Feurer, Eggensperger, Falkner, Lindauer, Hutter | arXiv 2020 (orig. NeurIPS 2015) | arXiv:2007.04074 | Warm-starts config search via meta-feature matching to a database of past datasets — "labeled at train time, meta-feature lookup at deploy" | Meta-labeled train / unlabeled deploy |
| **MetaOD** | Zhao, Rossi, Akoglu | NeurIPS 2021 | arXiv:2009.10606 (fetched, confirmed) | **Closest published analog to D5**: selects the best *unsupervised* outlier-detection method for a new dataset from meta-features + historical performance, zero deploy-time labels | Meta-labeled train / unlabeled deploy |
| MoE gating survey | (author list UNVERIFIED) | arXiv, 2024 | arXiv:2407.06204 | Online, jointly-trained router alternative to a meta-feature lookup — flags that fixed/random routers can sometimes match learned ones (caution against over-engineering) | Trained end-to-end |

**Most FUSE-like reusable violation statistic**: FUSE's Ŝ (Prop. 2.4) is directly computable on our
feature triplets with no modification to the statistic itself — only the transform-search step (their
binarization) would need adapting to our continuous encoding (monotone reshaping instead of
thresholding, consistent with §1.3's finding that continuous beats binary for us). The **vanishing-tetrad
test** and the **Ahn-Horenstein eigenvalue-ratio test** are cheaper, more targeted pre-screens for the
specific rank-1 assumption SML/L-SML leans on; **Varma/Ratner's inverse-covariance robust-PCA** is the
nonparametric, structurally-grounded replacement candidate for the empirically-refuted ρ≥0.75 filter
(§1.5).

**Router meta-features + prior art**: a per-cell signature built from already-computable, label-free
quantities — leading-eigenvalue share / eigenvalue-ratio rank estimate, mean |Spearman ρ|, the L-SML
rank-1 residual vs. the U-PCR additive residual (§2.4), FUSE's Ŝ under best transform, `n`, trace-length
statistics, per-feature valid-rate from the existing constant/saturated dropping (§1.4) — is exactly
the meta-feature vector that **MetaOD** and **auto-sklearn**'s warm-starting use to route to a
configuration from historical (labeled-at-train-time) performance, applied label-free at deployment.
This is the concrete prior art for D5's flavor (ii).

---

## 4. Candidate pipeline-step designs

None piloted this session — design-only, with pros/cons and an A/B sketch against the 19-cell
replication grid for later.

**D1 — Assumption-violation-minimizing subset search.** Score each candidate subset (or a greedy/
gated search over subsets) by a label-free violation statistic: FUSE's Ŝ, the L-SML Eq-14 residual,
the U-PCR projection residual, or the vanishing-tetrad statistic — and pick the subset that minimizes
it. Lowest implementation risk (every ingredient already exists in the codebase or `papers/extracted/`
quotes above); direct heir of §1.4's constant/saturated dropping and §2.4's structural-model
comparison. Con: needs validation that "low violation" actually correlates with high AUROC — §1.5
shows one naive violation proxy (ρ≥0.75) does NOT correlate that way for continuous L-SML, so this
must be empirically checked, not assumed, before trusting it as a selection objective.

**D2 — Unsupervised gated feature selection as a pre-fusion step.** Run Concrete Autoencoders or the
Gated-Laplacian trace objective (Thread A/C's top picks) on the cell's feature matrix to pick or soft-
weight a subset, then fuse the result with L-SML/U-PCR as today. Con: Step 151's Track-A finding is a
sharp warning here — six *direction-free* anomaly/density scorers all lost 10-15pp to L-SML continuous
because they discard the oriented consensus direction that is the actual source of L-SML's signal; a
gating/reconstruction objective must be checked for the same failure mode (reconstruction-good ≠
label-relevant) before being trusted as a pre-filter.

**D3 — Rank/eigengap-guided grouping.** Use Kritchman-Nadler or Ahn-Horenstein rank tests to set K
(replacing/validating the current residual-based K selection) and to prune features that break the
two-rank-one structure. Lower priority than D1/D2 — mostly refines an assumption (§2.2) that already
fits our data reasonably well, per Step 136's block-structured correlation evidence.

**D4 — Transformation search à la FUSE.** Per-feature monotone transform (not binarization, given
§1.3) minimizing Ŝ or an analogous statistic, before fusion. Natural extension of D1 if the violation
statistic proves informative; more implementation cost than D1 alone.

**D5 — Omri's dual-use router.** A cell's unlabeled feature matrix produces a data signature
(correlation structure, K̂, clustering/structural-model residuals, trace-length stats, n) that
determines which subset/config to fuse — and that subset then produces the hallucination score. Same
data, two roles. Two flavors, with an explicit access-tier distinction that matters for the thesis's
unsupervised claim:
- **(i) In-cell, purely label-free** — pick the subset/config optimizing a label-free criterion
  computed on this cell alone (→ reduces to D1/D4, no extra machinery, fully unsupervised at both
  train and deploy time).
- **(ii) Cross-cell learned router** — train a signature→best-subset mapping offline on the ~51
  historical cells' sweep results (which *do* have labels, used only at training time), then apply it
  label-free to a new cell at deployment. This is structurally identical to **MetaOD** (Thread D) and
  **auto-sklearn's meta-feature warm-starting**: "labeled at train time, meta-feature lookup at
  deploy." §1.3's LOCO finding (flat/negative transfer for a naive domain lookup) is exactly why the
  *conditioning* variable matters — routing on a rich data signature instead of a coarse domain label
  is the part worth testing; a signature-based router could succeed where a domain-only lookup failed.
  **Must be stated explicitly in any writeup**: flavor (ii) uses historical labels at training time and
  is label-free only at deployment — a different access tier than flavor (i) or than L-SML/U-PCR
  themselves, which never see labels at all.

---

## 5. Recommendation + open questions for Ofir/Bracha

**Recommendation** (design landscape, not a decision): D1 is the lowest-risk, most code-adjacent
starting point — it reuses machinery the pipeline already has (Eq-14/U-PCR residuals, the
constant/saturated dropper) and has a direct, well-verified precedent (FUSE) for turning exactly this
kind of label-free violation statistic into a selection objective. D5-flavor-(i) is D1 restated as a
router; D5-flavor-(ii) is the more ambitious, MetaOD-shaped version worth scoping once D1 has a first
result to route on. D2/D3/D4 are worth keeping on the list but are higher-risk (D2, per the Step-151
direction-free-scorer warning) or lower-priority (D3, since the two-rank-one assumption already fits
reasonably well).

**Open questions for Ofir/Bracha**:
1. Does the Gated-Laplacian identification (§3, Thread A) match what Ofir meant by "sub-matrix /
   trace"? If not, GroupFS (AAAI 2026, newest) or the classical Nie et al. 2008 trace-ratio criterion
   are the next candidates to check.
2. Is there appetite for the D5-(ii) cross-cell router, given it's a different (train-time-labeled)
   access tier than the rest of the pipeline — does that complicate the thesis's unsupervised framing,
   or is "unsupervised at deployment" the right claim to make?
3. Priority order among D1–D4 for a follow-up pilot session, and whether the pilot should target the
   RAG/GPQA domains specifically (§1.3 — where the +7.6pp prize concentrates) or run across all
   domains for a cleaner macro comparison.
4. Should the hardcoded `var_y=0.25` in `upcr_fuse` (§2.3) be fixed opportunistically now (it's a
   one-line, low-risk change informed by already-computed `pos_rate`), independent of the larger
   selection-step decision?
