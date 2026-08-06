# SPEC — Solver mechanism, residual identifiability, and the DEEM soft collapse

**Status: SECONDARY / DIAGNOSTIC.** Nothing here is a registered arm, and nothing here may replace a
row in `results/dependency_fusion_study/`. This is an attribution analysis of an already-published
negative result, not a search for a configuration that wins.

**Preregistration discipline.** Every constant, threshold, null model, statistic, aggregation rule and
predicted reading below is fixed **before any AUROC or test statistic from these studies is read**.
This document is committed before the scripts run. Later sensitivity analyses get a new output
directory and are explicitly secondary, per `SPEC_DEPENDENCY_FUSION_EXPERIMENT.md` §9.

**Provenance.** Written in response to two rounds of independent review of commits `64f57cd` /
`4316531`. Round 1 identified seven defects in the first draft of this study; round 2 approved the
factorial and identified four more. Both rounds are incorporated. Where a design choice is the
reviewer's, it is marked; where it is ours, it is marked as such so it can be argued with.

---

## 0. Why this study exists

The Step-226 experiment reported `sdsf − su_pcr_reproduction = −5.65pp` (2W/22L, p = 6e-7) and
concluded that dependency-weighted fusion is harmful. That conclusion does not follow, because the
contrast changes two things at once. The shipped fixed-ρ 2×2
(`results/dependency_fusion_solver_matrix/`) separated them:

| effect, ρ held fixed at the SU estimate | mean Δ | median Δ | W/L | p |
|---|---:|---:|---|---:|
| solver PCR → ridge, observed `C` | **−3.74pp** | −2.21 | 2/22 | 6.0e-7 |
| matrix observed → structured, PCR solver | −0.37pp | −0.003 | 7/14 | 0.085 |
| matrix observed → structured, ridge solver | −1.90pp | −0.78 | 4/20 | 7.6e-5 |
| registered H2 (both at once) | −5.65pp | −2.85 | 2/22 | 6.0e-7 |

So **the solver carries the loss** and **the dependency-structured covariance is undemonstrated rather
than disproved**. Two questions remain open, and both are answerable without new inference:

1. *Why* does the ridge lose? The ridge does two separable things — it rescales the top-two
   coefficients, and it admits the low-eigenvalue tail. Nothing published separates them.
2. Is there residual dependency structure worth modelling at all, or is the leftover indistinguishable
   from what an independent-error model produces on this much data?

A third question is operational: `deem_deep_soft` is failing on 27 of 30 attempts and the runner
discards the evidence.

### 0.1 Synthetic admission addendum (post-run, not part of the preregistration above)

At Omri's request, real-data follow-up is now conditional on a known-truth synthetic admission
benchmark: `scripts/synthetic_dependency_fusion_validation.py`. This addendum records the observed
result after the run; it is not presented as text written before those results. The thresholds and
four worlds are embedded in the source that produced the SHA-256-bound summary.

The benchmark uses independent training and test draws from a joint Gaussian model whose
off-diagonal clean covariance exactly satisfies U-PCR's additive equation. It tests:

- a clean independent-error world;
- the same world with four planted sparse error-correlation edges at n=300 and n=3000;
- a dense-block stress world that intentionally violates the sparse-support assumption;
- random feature sign flips and the deployed two-pass sign(rho) orientation;
- population-oracle linear and two-component PCR references;
- IU-PCR, SU-PCR, SDSF, structured-matrix PCR, observed ridge, full-pool L-SML, and a small declared
  exact DUFS-PF + L-SML secondary sample.

Labels are generated with the test draw and reach only AUROC after training-only weights and anchor
orientation have been frozen. Forty repetitions per world and 6,000 independent test samples per
repetition separate estimator instability from test-metric noise.

**Observed decision: `STOP_AND_REVISE`.** In the primary sparse-large world:

- planted support recovery succeeded (mean recall 0.9375, precision 1.0000);
- the oracle full solve beat oracle PCR by +0.895pp, so the world contains the tail value required to
  test the solver claim;
- with planted orientation supplied as a diagnostic, SDSF beat SU-PCR on 40/40 repetitions by
  +0.786pp, 95% bootstrap CI [+0.704, +0.866];
- with deployable sign(rho), SDSF beat SU-PCR on 33/40 and had median +0.601pp, but seven catastrophic
  failures changed the mean to -1.616pp, CI [-3.617, +0.068];
- structured-matrix PCR stayed tied to SU-PCR (-0.017pp), while SDSF versus that same structured PCR
  reproduced the loss. The failure is the full/tail solver, not the structured covariance estimate;
- the label-free reliability-tail fraction correlated -0.908 with the SDSF effect. Half-sample
  polarity stability correlated only +0.063 because a wrong orientation can be stable.

A post-hoc rule that falls back to SU-PCR when `||(I-P2)rho||/||rho|| > 0.25` would have kept all 33
wins and removed all seven losses (+0.571pp mean), but that threshold was seen after v1 and is **not
evidence**. The next admissible experiment is a new, preregistered, disjoint-seed confirmation of that
unchanged guard (and alternatives fixed before the run). Until one passes, the orchestrator blocks
all new real-data Step-227 work without a bypass.

Artifacts: `results/synthetic_dependency_fusion/{REPORT.md,summary.json,replicates.csv,contrasts.csv,
method_summary.csv}`.

### 0.2 Fixed-orientation v2 result (post-run)

The feature-direction hypothesis was then isolated in a versioned run on a disjoint synthetic seed
namespace. `confidence-orientation-v1` emits every synthetic view in a frozen “higher = more likely
correct” direction; random sign flips remain only in the legacy `sign(rho)` control. The v1 numeric
thresholds were carried forward unchanged.

The correction removes the failure mechanism in the sparse worlds. In sparse-large, fixed SDSF
beats fixed SU-PCR by **+0.845pp**, 95% bootstrap CI **[+0.747, +0.943]pp**, with **39/40** wins. It
wins **77/80** repetitions across sparse-small and sparse-large. Support recovery remains accurate
(recall 0.956, precision 1.000), the clean-world no-harm gate passes, and SDSF captures 89.2% of the
available oracle-over-PCR gap.

The overall decision nevertheless remains **`STOP_AND_REVISE`**. The one failed preregistered gate
is fixed SU-PCR versus fixed IU-PCR: **-0.0004pp** against a required +0.25pp. Sparse covariance
cleaning does not improve the two-component PCR solution by itself. The supported, narrower claim
is that the full SDSF reliability/dependency-weighted solve recovers the planted tail value once
feature direction is fixed. The unsupported claim is that covariance cleaning alone is useful.
This narrows the mechanism attribution and does not retroactively convert the conjunctive gate to a
pass or license a new real-data run.

Artifacts: `results/synthetic_dependency_fusion_fixed_v2/{REPORT.md,summary.json,replicates.csv,
contrasts.csv,method_summary.csv}`. Feature contract and reproduction commands:
`FEATURE_ORIENTATION_CONTRACT.md`.

---

## 1. Hard constraints

1. **No real-data Step-227 computation starts until the registered sweep exits and synthetic
   admission passes.** The synthetic benchmark can run on the separate no-data computer and never
   reads the cache. The reviewer required the sweep-exit gate for the
   DEEM probe (a concurrent fit competes for CPU threads and memory bandwidth and can affect the
   registered run's timing and stability); Omri extended it to all three studies, because the residual
   study is ~48,000 decomposition refits. Gate: PID of the running sweep gone, no new `records.jsonl`
   line for ≥ 10 minutes, and a readable `summary.json` newer than the checkpoint. PID inspection is
   cross-platform (`os.kill(pid, 0)` plus `ps` on POSIX, `tasklist` on Windows); an inspection failure
   blocks execution rather than declaring the sweep dead.
2. **The four files hashed into `config_hash` are never edited** — `spectral_utils/upcr.py`,
   `spectral_utils/dependency_fusion.py`, `spectral_utils/deem_adapter.py`,
   `scripts/run_dependency_fusion_experiment.py`. `JsonlStore` discards records whose `config_hash`
   differs, so a single edit orphans the 24-cell checkpoint. All work here is in new files; existing
   routines are imported, never retyped.
3. **Label discipline.** Every weight vector, every orientation decision, every null draw, every
   configuration choice is computed from features alone. Labels enter only in `evaluate_score`, after
   the score is frozen.
4. **Reproduction gates before interpretation.** No number from these studies is read until the
   validity gate (`assert_good6`, GOOD_6 macro = 0.7733442) and the study's own wiring gates pass.

---

## 2. Data, cells, and the aggregation unit

24 in-scope cells (`scripts/inscope_cells.py`), m ∈ [19, 30] features, n ∈ [198, 8460] samples.

The cells are **not 24 independent replicates**. Grouped by source dataset via `dataset_family()`
(`scripts/run_dependency_fusion_experiment.py:144`) there are **8 families**:

| family | cells | domain |
|---|---:|---|
| gsm8k | 10 | math |
| math500 | 5 | math |
| triviaqa | 4 | QA |
| hotpotqa, sciq, nq_open, squad_v2, truthfulqa | 1 each | QA |

Consequences, fixed in advance:

- Any inferential claim about replication is made at the **family** level (§4.5), never by counting
  24 correlated cells.
- Every macro confidence interval in this document is an **equal-family bootstrap** — first average
  cells within each dataset family, then resample the eight family means. Merely concatenating all
  cells after resampling a family would still let GSM8K outweigh singleton families. (Reviewer,
  round 2.)
- The global statistic is reported with **leave-one-family-out** values so it cannot be driven by
  gsm8k or math500.

---

## 3. Study A — why does the ridge lose?

Script: `scripts/solver_mechanism_study.py`. Output: `results/solver_mechanism/`.
ρ is held fixed at the SU (sparse-corrected) estimate for every arm, so nothing here varies the
reliability estimate.

### 3.1 The factorial (round-1 item 1)

The first draft proposed `ridge_projected_rho = (PSD(C)+γI)⁻¹P₂ρ` and
`pcr_top2_ridge_filtered = Σ_{j≤2} ρ_j/(λ_j+γ)·v_j` as separate arms. With `P₂` built from the same
eigenbasis these are **the same vector**, and the reviewer was right to reject the pair. They collapse
into one arm, `h_ridge`, which now sits in a factorial that actually crosses the two mechanisms.

Let `(λ_j, v_j)` be the eigenpairs of the observed covariance `C = FFᵀ/n`, descending, and
`ρ_j = v_jᵀρ`:

```
h_PCR   = Σ_{j ≤ 2}  ρ_j /  λ_j      · v_j        top-two block, PCR scaling
h_ridge = Σ_{j ≤ 2}  ρ_j / (λ_j + γ) · v_j        top-two block, ridge scaling
t_ridge = Σ_{j > 2}  ρ_j / (λ_j + γ) · v_j        the low-eigenvalue tail
```

| | tail absent | + `t_ridge` |
|---|---|---|
| **PCR head scaling** | `h_PCR` ≡ committed `su_pcr_reproduction` | `h_PCR + t_ridge` |
| **ridge head scaling** | `h_ridge` | `h_ridge + t_ridge` ≡ committed `ridge_observed` |

Effects extracted, each reported with a family-blocked 95% CI and a paired Wilcoxon p:

- head rescaling, tail absent: `h_ridge − h_PCR`
- head rescaling, tail present: `(h_ridge+t_ridge) − (h_PCR+t_ridge)`
- **tail addition at the PCR head**: `(h_PCR+t_ridge) − h_PCR`
- tail addition at the ridge head: `(h_ridge+t_ridge) − h_ridge`
- interaction: the difference of the two tail effects (identically, of the two head effects)

**Why the observed covariance is the right stage.** `C = FFᵀ/n` is PSD by construction and measured
PSD: 0 clipped eigenvalues and `‖PSD(C)−C‖/‖C‖ ≤ 2.7e-15` on all 24 cells. So `PSD(C)` and `C` share
an eigensystem and the head/tail split is exact rather than approximate.

**Gates (both `SystemExit` on failure, asserted before any effect is read):**

- **wiring** — `h_PCR` and `h_ridge + t_ridge` reproduce the committed per-cell AUROC of
  `su_pcr_reproduction` and `ridge_observed` to **1e-9** on all 24 cells;
- **arithmetic identity** — `‖(h_ridge + t_ridge) − regularized_covariance_weights(C, ρ)[0]‖ / ‖·‖ ≤ 1e-10`
  per cell, i.e. the factorial's full corner *is* the registered ridge solution and not a lookalike.

### 3.2 The κ path and its predicted trend (round-2 item 2.7)

γ is chosen by the registered analytic rule (`regularized_covariance_weights`, `target_condition=100`).
The diagnostic path is **κ ∈ {3, 10, 30, 100, 300}**, reported in full for every γ-dependent arm on
every cell. **κ is never selected by label**; κ=100 remains the registered value.

**Predicted trend, fixed in advance**: if amplification of the low-eigenvalue tail is causal, the tail
effect `(h_PCR + t_ridge) − h_PCR` becomes **more negative as κ increases** (a larger κ means a
smaller γ, which admits the tail more aggressively). This is judged by a **family-weighted OLS slope
of the tail effect against log κ**, with a family-blocked bootstrap CI on the slope — not by looking
at the path. A slope whose CI contains zero falsifies the amplification account.

### 3.3 Held-out and sample-size test (round-1 item 3)

The factorial says which weight component hurts; it cannot say whether it hurts because of
finite-sample estimation variance or because the full-inverse model is structurally wrong.

- train fractions **{0.25, 0.50, 0.75}**, **R = 50** repetitions each;
- split index from `stable_hash(cell, fraction, rep)`, uniform at random, **never label-aware**;
- **everything label-free is fitted on train only**: the `sign(ρ̂)` polarity probe, `C`, the
  decomposition, `ρ`, every weight vector, **and the global anchor flip** — `anchor_orient` decides on
  the train score and that frozen sign is applied to the test score;
- AUROC is computed on held-out samples only.
- every repetition re-centers and re-scales each view using **training rows only**, then freezes that
  affine transform onto test rows. Training labels are never inspected; a fixed split with a
  single-class test set is retained with undefined AUROC rather than replaced.
- every repetition is written to `heldout_repetitions.csv`; `heldout.csv` contains paired summaries.

Reported per (cell, fraction): held-out AUROC per arm; the held-out ridge−PCR gap; the CV of
`‖t_ridge‖` across repetitions; median `|cos|` between `t_ridge` across repetitions; mean principal
angle of the top-2 subspace across repetitions; the same for the residual subspace; and `n_train / m`
with `n_train < 2m` flagged. At 25% of the smallest cell that is ~50 training samples against m up to
30 — a real regime, and it must be visible rather than averaged away.

**Pre-registered readings** (the reviewer's, verbatim):

- ridge gap shrinks with sample size ⇒ estimation variance is plausible;
- ridge gap persists despite stable tail estimates ⇒ structural model mismatch;
- tail changes substantially while the top subspace stays stable ⇒ low-subspace noise amplification.

### 3.4 Supporting descriptives — not causal (round-1 item 2)

The first draft offered a label-free leakage fraction as evidence for the mechanism. Correlation with
AUROC loss does not establish causality; the factorial in §3.1 is the causal test, and the following
are **supporting evidence only**, reported as such:

- leakage fraction `‖(I − P₂)ρ‖ / ‖ρ‖` (computable without running the ridge at all);
- ridge-tail weight norm relative to head norm, `‖t_ridge‖ / ‖h_ridge‖`;
- variance of the tail score, `Var(t_ridgeᵀF)`, and its share of `Var((h_ridge+t_ridge)ᵀF)`;
- head–tail score correlation, Pearson and Spearman;
- the ΔAUROC from adding the same tail to the PCR head — which is the factorial cell, and the only
  item in this list that is an intervention.

### 3.5 Matrix diagnostics (round-2 item 2.6)

"Signed condition number" was wrong wording; a condition number is not signed. Reported separately,
per matrix, in the eigensystem the arm actually used:

- **singular-value condition number** `σ_max/σ_min`;
- **minimum eigenvalue**, **maximum eigenvalue**;
- **number of negative eigenvalues** (inertia);
- and, for the ridge arms, the same three for `PSD(C) + γI`.

The `cond_raw_*` column name is retired: it reported `cond(PSD(C))`, not `cond(C)`.

### 3.6 The PSD attribution, corrected (round-1 item 6)

Three arms: `pcr_structured` on the raw indefinite `C_str` (committed), `pcr_structured_psd` on
`_nearest_psd(C_str)`, and `su_pcr_reproduction` on the observed `C`. The first draft's reading was
backwards. The correct mapping:

| observation | conclusion |
|---|---|
| raw ≈ PSD, both below observed | the **structured estimator** causes the loss |
| PSD ≈ observed, raw below | **indefiniteness of the raw matrix** causes the loss, and PSD repair fixes it |
| PSD below raw | the **projection itself** adds harm |

**Equivalence criterion** (round-2 item 2.8 — called an equivalence criterion, not TOST, since
requiring a 95% CI inside the interval is conservative rather than the standard two-one-sided-tests
procedure): `≈` means **|macro Δ| ≤ 0.25pp** *and* the paired **family-blocked** bootstrap 95% CI
lies inside **±0.50pp**. The margins are ours, chosen against the registered §8 thresholds (+1.0pp
meaningful gain, −0.5pp acceptable harm), not derived from these data.

Raw and PSD structured weights are **bit-identical on 18 of 24 cells** — only the 6 clipped cells can
differ — so the 24-cell macro dilutes the effect by 18 zeros. Both readings are preregistered: the
**6-cell subgroup is the informative one**, the 24-cell macro is the headline.

---

## 4. Study B — is the residual dependency identifiable at all?

Script: `scripts/residual_identifiability_study.py`. Output: `results/residual_identifiability/`.
Objects: `SparseDecomposition.residual`, `.sparse`, `.support`.

Purpose: before designing any SDSF v2, test whether there is structure to model. This study has a
preregistered **abandonment** condition, not only a success condition.

### 4.1 Null models (round-1 item 4)

The first draft permuted each feature's sample axis independently. That destroys the shared latent
rank-two signal as well as the error dependence, so it tests "no cross-feature relationship at all"
rather than "independent errors conditional on the latent variable" — and would make residual
structure look more significant than it is. Replaced by:

**Null (a) — fitted latent signal + independently permuted residuals. PRIMARY.**
`V₂` = the top-2 **magnitude** eigenvectors of the fitted low-rank component `L` (magnitude, matching
`_rank_projection_symmetric`, because `L` may be indefinite). Then
`F_lat = V₂V₂ᵀF`, `E = F − F_lat`, permute each **row** of `E` with an independent permutation, and
`F* = F_lat + E*`. Preserves the latent signal exactly and each error's marginal distribution;
destroys only cross-view error dependence.

**Null (b) — parametric bootstrap from the fitted independent-error latent model. SECONDARY.**
`C₀ = PSD(L) + D` with `D = diag(max(diag(C) − diag(PSD(L)), ε))`, ε = 1e-8·mean diagonal; draw n iid
Gaussian samples from `N(0, C₀)`.

**Null (c) — rank/copula-preserving bootstrap. DETERMINED NOT REQUIRED.**
The reviewer made this conditional on the pipeline being rank-based. It is not: `prepare_cell`
z-scores raw values (`spectral_utils/subset_sweep.py:383`, `fusion_utils.zscore`) and the runner forms
`C = FFᵀ/n`, a Pearson covariance on z-scored values. Per-view **excess kurtosis** is reported anyway,
and any cell whose median excess kurtosis exceeds 2 has its null-(b) p-value marked
**Gaussian-dependent**; null (a), which preserves the empirical marginals, is primary for that reason.

**Both nulls are put through the complete decomposition pipeline** (`sparse_upcr_fit` end to end) on
every draw, so the null distribution absorbs the decomposition's own bias and its tendency to fit
spurious support.

**B = 1000** draws per cell per null (round-1 item 5: at B=200 the smallest attainable empirical
p-value is 1/201 ≈ 0.00498, too coarse once corrected).

### 4.2 Identical preprocessing on every null sample (round-2 item 2.1)

Permuting residuals re-pairs them with the latent component and therefore changes each feature's
variance. Without re-standardization the observed and null covariances would not have gone through
identical preprocessing, and the comparison would be confounded by scale.

**Every null `F*` is re-standardized row-wise, with exactly `prepare_cell`'s convention** — mean 0 and
`std()` with ddof = 0, via `spectral_utils.fusion_utils.zscore` (`fusion_utils.py:41`) — **before**
`C* = F*F*ᵀ/n`. This applies to null (a) and null (b) alike.

**Verification gate**: `max |diag(C*) − 1| ≤ 1e-8` on **every** draw, and the same assertion on the
observed `C`. A violation is a hard error, not a warning — it would mean the null and the observation
are not comparable objects.

The same re-standardization and unit-diagonal gate are applied independently to both halves of every
split-half stability repetition.

### 4.3 Statistics

Magnitude as well as concentration — a tiny residual matrix can have a concentrated spectrum, so the
top-5 share alone cannot answer the question:

| statistic | role |
|---|---|
| `‖R‖₂ / ‖C^off‖₂` (operator norm, scale-free) | **PRIMARY** |
| `‖R‖_F / ‖C^off‖_F` (Frobenius) | secondary |
| top-5 \|eigenvalue\| share of `R` | secondary (the statistic already in the record) |
| split-half principal angles over `d_res` | stability |
| split-half support Jaccard, edge-sign agreement | stability |
| feature-family enrichment of the recovered support | **secondary only** |

Feature-family enrichment is over the four pre-specified, label-free families in the pool — base
spectral, `*_spilled`, `*_energy`, logprob. It is secondary and carries the reviewer's caveat
verbatim: spilled and energy views may already be related by construction, so enrichment there is not
automatically evidence of newly discovered error dependence.

### 4.4 Repeated split-half stability (round-2 item 2.3)

The first draft did not say how many split-halves were used; a single split could pass or fail the
gate arbitrarily. Fixed:

- **50 deterministic split-half repetitions per cell**, seeded by `stable_hash(cell, "splithalf", rep)`;
- the **cell statistic is the median over repetitions** of each of: principal angle over `d_res`,
  support Jaccard, edge-sign agreement;
- the 10th and 90th percentiles across repetitions are reported alongside every median.
- every repetition also applies the identical split-half procedure to a fresh primary-null sample;
  the observed median angle must be strictly below that null median as well as below 60°.

### 4.5 Inference: one global test, then eight family tests (round-2 item 2.2)

The first draft was self-contradictory — cell-wise tests were called "descriptive only" while success
required "cell-level significance in ≥5 families", which made them inferential. Resolved:

**Global primary endpoint `T` — one test, no multiplicity.**
Per cell, z-score the observed `‖R‖₂/‖C^off‖₂` against that cell's null-(a) draws. Average within
dataset family. Take the **unweighted mean over the 8 family means** → `T_obs`. The null distribution
of `T` is built by taking the b-th null draw of every cell, standardizing with leave-one-out moments
from the remaining draws, and aggregating identically → `T⁽ᵇ⁾`. One-sided (real structure ⇒ larger
residual than an independent-error model):

```
p = (1 + #{ T⁽ᵇ⁾ ≥ T_obs }) / (B + 1),   α = 0.05,   floor p ≈ 0.000999 at B = 1000
```

**Eight family-level empirical p-values.** Each family's cells are aggregated **exactly as the global
statistic aggregates them**, giving one empirical p-value per family from the same null draws.
**Benjamini–Hochberg FDR at q = 0.10 across the 8 family tests.** This is where the inferential
replication claim lives.

**Family stability** is the **family median** of the three cell statistics from §4.4, tested against
the thresholds in §4.6.

**The 24 cell-level p-values are descriptive**, reported for transparency, and carry no claim.

**Leave-one-family-out**: `T` and its p-value are recomputed with each family dropped in turn, so a
result cannot be driven by gsm8k (10 cells) or math500 (5).

### 4.6 Fixed numeric definitions

| quantity | value | why this value |
|---|---|---|
| residual subspace dimension | **`d_res = 5`** | the dimension already in the shipped record (top-5 share); `d ∈ {2,3}` reported descriptively only |
| support numerical-nonzero | **`\|S_ij\| > 1e-8 · max\|C^off\|`** | the code's exact-zero test is not numerically safe at this m — cf. the Step-205 finding that 1e-16 perturbations flip L-SML structure |
| subspace stability | median principal angle over `d_res` < **60°**, and strictly below the null median | 60° = mean cosine 0.5, i.e. better than half-random alignment |
| support stability | median Jaccard ≥ **0.50**, median edge-sign agreement ≥ **0.80** | ours |
| replication | ≥ **5 of 8** families passing BH-FDR **and** all three stability medians | Omri's call; the reviewer's 4-of-6 proportion mapped onto the true 8 families |

### 4.7 Decision

**Success** = global `p < 0.05` **and** ≥ 5 of 8 families pass **and** the stability medians hold
**and** leave-one-family-out shows no single family driving `T`.
⇒ The design worth building is the reviewer's: use dependency structure to **correct ρ** while keeping
U-PCR's low-dimensional spectral solver — a block/group error model, not another full inverse.

**Failure** = anything else. ⇒ **Stop building covariance decompositions**, and say so in the write-up.
This is a real abandonment condition and it is written before the test.

### 4.8 Cost control (operational, label-free)

24 cells × 2 nulls × 1000 draws = 48,000 full decomposition refits. The script times 3 cells first and
prints the projected wall clock; if that exceeds 8 h single-core it runs cells in parallel processes.
No labels are involved, so this choice cannot bias a result.

Exactly 1000 valid draws are required for each null and cell. A failed draw is a contextualized hard
error; it is never discarded in a way that changes Monte Carlo resolution. Parallel results are
restored to canonical `INSCOPE` order before aggregation.

---

## 5. Study C — the DEEM soft collapse

Script: `scripts/deem_soft_collapse_probe.py`. Output: `results/deem_probe/`.
**Runs only after the registered sweep exits** (§1.1).

### 5.1 What is known

From the checkpoint: `deem_irbm_hard` 35 ok, `deem_deep_hard` 31 ok, `deem_deep_soft` **3 ok / 27
failed**, and every one of the 27 failures is the identical
`ValueError: method returned a non-finite or constant score`. That is raised by `orient_score`
(`run_dependency_fusion_experiment.py:249`) **after** `fit_deem_score` has returned — so the fit
completes and `save_method_record`'s `except` branch (`:333-346`) discards the completed
`DeemRunResult`, including `model.history_`.

The probe constructs pinned `DEEM==0.2.0` with kwargs identical to the hashed adapter and injects a
trainer callback without editing either dependency. It retains scores, aligned probabilities, full
loss history, per-epoch output standard deviation, parameter and gradient norms, sparsemax zero/dead
unit fractions, and the last finite checkpoint for a collapsed or failed fit. The same path therefore
retains evidence if `model.fit()` raises partway through training.

### 5.2 Numeric definitions

| term | definition |
|---|---|
| **collapsed** | `sd(aligned class-1 score) < 1e-6`. Calibrated deliberately so that the 3 current `deem_deep_soft` "successes", which sit at σ ≈ 1e-8, are classified as **collapsed** rather than successful. The runner's own guard is the harder `std < 1e-12` |
| **degenerate-but-nonconstant** | `1e-6 ≤ sd < 1e-3`; reported separately, never counted as healthy |
| **healthy** | `sd ≥ 1e-3` **and** finite objective throughout training |
| **completion rate** | fraction of (cell, seed) fits that are healthy |
| **soft repair succeeded** | completion rate ≥ **90%** — the same 90% rule as registered §8 |
| **no meaningful advantage over IU** | mean paired `arm − iu_pcr` ≤ 0 **and** family-blocked 95% upper bound < **+1.0pp** |
| **primary comparison** | **ensemble AUROC**, because H3 is registered on `full.deem_deep_soft_ensemble` (`SPEC_DEPENDENCY_FUSION_EXPERIMENT.md` §6). Mean per-seed AUROC and between-seed spread are reported alongside and never substituted |

On wording (round-2 item 2.4): the criterion is "**shows no meaningful advantage over IU**", not
"stays below IU". A negative mean with an upper confidence bound below +1.0pp **rules out the
preregistered meaningful gain**; it does not prove inferiority, and the write-up must not say it does.

### 5.3 Separate stopping decisions (round-2 item 2.4)

The first draft's stop rule was not logically valid: it let poor **hard** DEEM performance veto a
successfully repaired **soft** DEEM. They are different methods. Replaced by three independent
decisions:

1. Soft repair unhealthy, **or** completion rate < 90% ⇒ **abandon soft DEEM**.
2. Hard DEEM shows no meaningful advantage over IU (§5.2) ⇒ **abandon hard DEEM**.
3. **If repaired soft DEEM is healthy, its predefined evaluation runs regardless of hard-DEEM
   performance.** Neither decision vetoes the other.

The selected configuration must first complete all 3 pilot cells × 5 seeds. If that gate passes, the
frozen configuration runs on all in-scope cells and all five seeds. The final evaluation requires
both seed-fit completion and ensemble-cell completion ≥90%; ensemble AUROC is compared with IU using
equal-family aggregation and a family bootstrap interval.

`_ensemble` requires all five seeds (`run_dependency_fusion_experiment.py:584`), so an empty H3
candidate set remains a legitimate finding — "the arm did not fit" — and is reported as one rather
than as a gap.

### 5.4 Configuration grid and tie-breaker

Grid, fixed in advance: `learning_rate ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}` × `epochs ∈ {100, 300, 1000}`
= 15 configurations, seed 0 only, on the **three failing cells with the smallest n** (ties broken
alphabetically — deterministic and label-free). The winner is then run on all 5 registered seeds.

**Deterministic label-free tie-breaker, applied in this order:**

1. **completion rate** — fraction of fits healthy with finite objective; higher wins;
2. **finite objective throughout training** — required, not a preference;
3. **cross-seed score stability** — median pairwise |Spearman| between seeds' scores; higher wins;
4. **smallest deviation from the registered default** — `|log10(lr/1e-3)| + |log10(epochs/100)|`;
   smaller wins.

**AUROC is never consulted at any step of the selection**, and the full choice trace is printed. The
pilot is secondary and never replaces a registered row.

---

## 6. Outputs

| directory | contents |
|---|---|
| `results/solver_mechanism/` | `per_cell.csv`, `factorial_effects.csv`, `kappa_path.csv`, `heldout.csv`, `heldout_repetitions.csv`, `summary.json` |
| `results/residual_identifiability/` | `per_cell.csv`, `family_tests.csv`, `null_draws_summary.csv`, `summary.json` |
| `results/deem_probe/` | `per_fit.csv`, `grid.csv`, `evaluation_seeds.csv`, `evaluation_per_cell.csv`, `artifacts/`, `summary.json` |

Plus an addendum to `results/dependency_fusion_raw/RAW_DATA_README.md`, and `§12 Solver mechanism` /
`§13 Residual identifiability` in `OPINION_DEPENDENCY_FUSION_RUN.md`. `HISTORY.md` and `PROGRESS.md`
are untouched until the conclusion is decided.

---

## 7. Verification checklist

- [ ] Sweep exited before any numerical step; `records.jsonl` only ever grew; the sweep was never signalled.
- [ ] `git status` shows no modification to the four hashed files; `run_config.json` `source_sha256` unchanged.
- [ ] `assert_good6` prints GOOD_6 macro = **0.7733442** before any arm runs.
- [ ] Study A wiring gate: both anchor corners within **1e-9** of committed per-cell AUROC, all 24 cells.
- [ ] Study A arithmetic gate: tail identity within **1e-10** relative, all 24 cells.
- [ ] κ path complete for all 5 κ on all 24 cells; trend reported as a family-weighted slope with CI.
- [ ] Held-out study: no weights or orientation ever fitted on test samples; `n_train/m` reported.
- [ ] Null preprocessing gate: `max |diag(C*) − 1| ≤ 1e-8` on all 48,000 draws.
- [ ] B = 1000 per cell per null, complete pipeline per draw, observed percentile reported.
- [ ] One global p-value; 8 family p-values under BH q = 0.10; 24 cell p-values labelled descriptive; LOFO reported.
- [ ] All macro intervals family-blocked, never a flat 24-cell resample.
- [ ] Condition numbers reported with inertia; no "signed condition number" anywhere.
- [ ] DEEM soft and hard stopping decisions reported separately; grid winner chosen by the four-step
      label-free tie-breaker with the choice trace printed.
