# PRM boosting as latent-class identification: a framing (draft, 2026-10-24)

**Status.** A working theory note for the thesis chapter on PRM boosting. It responds to the math peer review
(`../math_peer_review_2026-10-24/MATH_PEER_REVIEW_HE.md`) and to its history audit (`HISTORY_AUDIT.md`).
- Propositions P1–P3 and P4 are standard algebra.
- P5 is the new methodological claim.
- Every numerical statement names its source. The simulations are in-model checks, not evidence on real data.
- Nothing here changes a frozen result.

## 0. Summary

We study a **frozen process reward model (PRM)** that scores every step of an answer, plus **telemetry**:
token-probability channels read from the same teacher-forced forward pass. The goal is to rank and flag the
erroneous steps of each answer **without labels**.

We cast this as a latent-class factor model with three features that the classical spectral-meta-learner setting
(Parisi et al. 2014; Jaffe et al. 2015, 2016) does not have:
1. **Answer-level effects.** Every step of one answer shares a shift.
2. **Observed nuisance.** Step position and length move the views without being errors.
3. **Strongly heterogeneous view quality.** One view, the PRM, was trained with labels elsewhere; the others are
   weak and mutually dependent.

The framing explains three things:
- **Why the current method (CHAIN = L-SML over the PRM and the telemetry group scores) wins on PRMBench.**
  There the telemetry alone is at least as strong as the PRM, so almost any interior weighting helps.
- **Why it loses to strong PRMs on Socratic and Hard2Verify.** There the telemetry is weak, and the method gives
  the PRM about 1/(K+1) of the weight, far below what an improvement requires.
- **What is missing:** a label-free estimate of the PRM's quality relative to the telemetry.

We show that this quantity is **not identifiable from second moments** when the telemetry shares a nuisance
factor (P4). It **is identifiable by anchoring on the PRM** (P5), under an explicit assumption that can be
partly checked. The same estimate gives the Fisher-optimal weights and a p-free **gate** that predicts when
fusion beats the PRM.

## 1. Problem and estimands

**Data.**
- Answers j = 1..J; steps i = 1..n_j.
- View 0 is the PRM risk (logit(1 − reward)). Views 1..m are telemetry channels, each oriented so that higher means
  riskier, and grouped into K families.
- y_ij ∈ {0,1} marks an erroneous step and is never observed at fit time.
- z_ij are observed step covariates: relative position and log step length.

**Estimands.**
- **(E1) Ranking.** The within-answer AUC of a score s, averaged over answers that contain both classes. This is
  the primary target.
- **(E2) Decisions.** Per-step flags evaluated by PRMScore, the mean of the valid-class and error-class F1 on the
  official pool.

**What "boosting the PRM" means.**
- Find a label-free s = f(view 0, views 1..m) with AUC_within(s) ≥ AUC_within(view 0).
- Know, without labels, when no such improvement is expected. Then return the PRM: the **gate**.

**Access.** No labels in any fitted quantity. Labels are used only for evaluation and for diagnostics that are
marked as such.

## 2. Model

For view k of step i in answer j:

    x_ijk = a_jk + Δ_k · y_ij + β_k · z_ij + λ_k · η_ij + ε_ijk                                   (M)

| Term | Meaning | Assumption |
|---|---|---|
| a_jk | answer effect | arbitrary (fixed effects); no distributional assumption |
| Δ_k | class separation of view k | Δ_k ≥ 0 after orientation ("every view beats random", Jaffe et al. 2016, Lemma 4) |
| β_k z_ij | effect of observed covariates | linear in a flexible basis of z; Cov(y, z) arbitrary |
| λ_k η_ij | latent nuisance (e.g. a backbone "confidence state") | η ⊥ (y, z, ε); **anchor condition λ_0 = 0** (the nuisance does not load on the PRM) |
| ε_ijk | noise | independent of (y, z, η); block-diagonal across channel families; the PRM is its own block |

For AUC formulas only, we also take the class-conditional distributions to be Gaussian with a common covariance
(A-Gauss). The identification results (P1, P3–P5) do not need A-Gauss, except where stated.

Write C for the pooled covariance of the (centered, z-adjusted) views, u = √Var(y) · Δ for the signal loadings,
and Σ = C − uuᵀ for the noise-plus-nuisance covariance.

## 3. The current method, term by term

| Pipeline component | Model term it addresses | Status under (M) |
|---|---|---|
| Per-answer centering (part of `answer_z`) | removes a_j | justified (P1) |
| Per-answer SD scaling (rest of `answer_z`) | none | not justified in general; harmless when SDs factor as s_jk = s_k·r_j (P1) |
| Grouping by Eq.14/15 + spectral clustering (Jaffe et al. 2016) | block structure of ε | heuristic estimate of the blocks; stability to be reported |
| Within-group SML score | one virtual view per block | Jaffe et al. 2016, Lemma 1 analogue |
| Cross-stage SML over [PRM, group scores] | should estimate the relative quality u | **≈ equal weights in practice** (P3; PRM share × (K+1) has median 1.03, `HISTORY_AUDIT.md` §3) |
| Majority sign + anchor flip | Δ ≥ 0 | a negative estimate is a model violation, not an inverted view (Step 467) |
| Position/length residualization (CHAINRES) | β z | today a control; under (M), part of estimation (P5) |
| GAP flag count | decision layer | nonparametric split with a minority cap; not model-derived (P6) |

## 4. Propositions

### P1. Within-answer centering removes answer effects and keeps the one-factor structure

Let x̃_ij = x_ij − x̄_j. Then

    x̃_ij = Δ (y_ij − ȳ_j) + β (z_ij − z̄_j) + λ (η_ij − η̄_j) + (ε_ij − ε̄_j).

- The answer effects a_j are removed exactly, whatever their distribution.
- The latent becomes y − ȳ_j, still a single scalar multiplying Δ.
- Centering acts on each view separately with the same answer partition. Noise independence across views
  (and across blocks) is therefore preserved, and so are the off-diagonal identities used below (rank-one signal,
  P3; cumulant ratios, P5).

**Scaling.** Dividing view k in answer j by its within-answer SD s_jk keeps this structure only when
s_jk = s_k·r_j. For a strong view, s_jk grows with the answer's error fraction, so the scaling reweights strong
views answer by answer. This is a second-order effect, testable as answer-z vs centering-only (plan Stage 4).

*Numerical check (in-model).* 5,000 answers × 10 steps, with a shared answer factor that loads on all views:
- On raw data, the rank-one off-diagonal loading points at the answer factor: cos .994 with it, .639 with Δ.
- After centering: cos 1.000 with Δ.
- After answer-z: .999.

The review's claim that answer-z "creates dependence between views" (W6) holds for the scaling step only.

### P2. AUC of a linear fusion, and when fusion beats the PRM

Under A-Gauss, after centering and z-adjustment, the within-answer AUC of s = wᵀx is

    AUC(w) = Φ( wᵀΔ / √(2 wᵀΣw) ).

Reduce to two views: the PRM, with separation d_p (in within-class SD units), and a fixed telemetry composite T,
with separation d_T. Let ρ be their within-class correlation, and give the standardized views weights a and 1−a:

    D(a) = (a d_p + (1−a) d_T) / √(a² + (1−a)² + 2a(1−a)ρ),        AUC = Φ(D(a)/√2).

(i) **Improvement condition.**

    D(a) > d_p   ⇔   2a d_p (d_T − ρ d_p) > (1−a)(d_p² − d_T²).

(ii) **Strong telemetry.** If d_T ≥ d_p and ρ < 1, every a ∈ [0,1) improves on the PRM.

(iii) **Weak telemetry.** If ρ d_p < d_T < d_p, fusion helps only when the PRM's share satisfies a > a_min, where

    a_min = R / (1 + R),    R = (d_p² − d_T²) / (2 d_p (d_T − ρ d_p)).

(iv) **Redundant telemetry.** If d_T ≤ ρ d_p, no positive telemetry weight helps.

(v) **Fisher optimum.** D*² = (d_p² + d_T² − 2ρ d_p d_T)/(1−ρ²) ≥ d_p², with equality iff d_T = ρ d_p. Optimal
weights never hurt in the population.

**Reading.** (ii) is the PRMBench regime and (iii) is the Socratic regime.
- **PRMBench.** Telemetry alone beats the PRM on 5/5 PRMs (`STATUS_MAIN.csv`). D(a) has a broad, flat maximum,
  so near-equal weights lose almost nothing. This is why CHAIN ≈ ½PRM + ½mean on the three strong PRMs
  (.8377/.8373, .8428/.8424, .8156/.8128).
- **Socratic.** With the binormal conversion d = √2 Φ⁻¹(AUC), Qwen-72B has d_p ≈ 1.26 and d_T ≈ 0.50. Even with
  ρ = 0, a_min ≈ .52; with ρ = .3, a_min ≈ .82. The method gives the PRM .04 in that cell, and at most .25 in
  any Socratic cell (`results/sml_weight_estimator_v1/README.md` §4; collection `DIAG_*.json`).
- **Empirical corollary.** When the PRM's share is small, sign(CHAIN − PRM) ≈ sign(d_T − d_p). With labels,
  "telemetry-alone within-AUC ≥ PRM within-AUC" predicts the observed sign in 15/15 cells (`HISTORY_AUDIT.md` §3).

### P3. What the shipped cross stage computes

Take z-scored views that are conditionally independent (CI) given y. Then the off-diagonal correlation matrix is
R_off = uuᵀ − diag(u²). The shipped weights are its leading eigenvector v (`fusion_utils.sml_fuse_signed`).

From R_off v = λ v we get u_k (uᵀv) − u_k² v_k = λ v_k, hence

    v_k ∝ u_k / (λ₁ + u_k²),    where λ₁ is the largest root of   Σ_k u_k² / (λ₁ + u_k²) = 1.

Compare with the Fisher weight under the same model, w_k ∝ u_k / (1 − u_k²). The ratio
v_k / w_k ∝ (1 − u_k²)/(λ₁ + u_k²) decreases in u_k. Three consequences:
- **The strongest view is under-weighted systematically.**
- **Equal loadings give exactly equal weights.** This is why the PRM's share tracks 1/(K+1) when the group scores
  have similar loadings: median share × (K+1) is 1.03 across 24 cells.
- **Many weak views give v ∝ u ≈ Fisher.** This is the regime the SML was designed for (Parisi et al. 2014: the
  balanced-accuracy weights are the first-order expansion of the log-odds weights).

*Check.* In the review's CI toy (PRM d = 1.5, six views d = 0.3, p = .3): u_PRM = .567, u_T = .136, λ₁ ≈ .24. The
formula gives a PRM share of .243, which matches the numerical eigenvector. The AUC is .845, below the PRM alone
(.856).

### P4. With a telemetry-wide nuisance, second moments cannot see that the PRM is strong

Let the telemetry share a nuisance factor (λ_k ≠ 0) that does not touch the PRM. Then

    C_0k = u_0 u_k,        C_kl = u_k u_l + λ_k λ_l   (k ≠ l telemetry).

The triad estimate of the PRM's loading is biased down:

    û_0² = C_0a C_0b / C_ab = u_0² · u_a u_b / (u_a u_b + λ_a λ_b).

The leading eigenvector of the off-diagonal matrix rotates toward λ inside the telemetry block. This is the
mechanism recorded in `results/sml_weight_estimator_v1/README.md` §4, now as an identity.

If the PRM also loads on the nuisance with the opposite sign, as step position does on Socratic (within-class
correlation +.43 for the PRM, −.53 to −.58 for the telemetry), C_0k can turn negative. The majority-sign rule plus
the anchor flip then inverts the fusion (Step 467).

The review's corrected weight u/(1−u²) does not help here, because u itself is mis-estimated. In a Socratic-like
simulation, u/(1−u²) on 'complete'-scale rank-one loadings gives AUC .672 vs .856 for the PRM alone; the PRM's
loading is estimated at .12 instead of .47.

### P5. Identification by anchoring on the PRM

Assume the anchor condition (λ_0 = 0, after z-adjustment) and independent noise across the PRM and the telemetry
blocks.

**(a) Telemetry loadings.** u_k = C_0k / u_0 for every telemetry view k. One scalar, u_0, remains.

**(b) Second-order route.** M(t) = C_TT,off − t·ccᵀ (with c = C_T0) equals the off-diagonal of λλᵀ, which is rank
one, iff t = 1/u_0². This requires:
- K ≥ 4 telemetry blocks: K(K−1)/2 equations for K+1 unknowns;
- λ_T not parallel to u_T. Otherwise uuᵀ and λλᵀ cannot be separated and the profile in t is flat.

**(c) Third-order route.** If p ≠ ½ (so κ₃(y) ≠ 0), and y, η, ε_0, ε_k are mutually independent:

    κ(x_0, x_0, x_k) = Δ_0² Δ_k κ₃(y),     κ(x_0, x_k, x_k) = Δ_0 Δ_k² κ₃(y).

All other terms are joint cumulants of independent variables and vanish, even when the noise and η are skewed,
because λ_0 = 0. Hence

    Δ_0/Δ_k = κ(0,0,k)/κ(0,k,k),     u_0² = C_0k · κ(0,0,k)/κ(0,k,k).

This needs neither p nor conditional independence inside the telemetry. It is close in spirit to the third- vs
second-order ratios of Jaffe et al. (2015) and of FUSE's triplet-CI statistic (Prop. 2.4).

**(d) Weights.** With u known, the Fisher weights are w* ∝ Σ⁻¹u ∝ C⁻¹u (Sherman–Morrison, C = Σ + uuᵀ).

**(e) Gate.** Under A-Gauss the AUC is monotone in SNR(w) = (wᵀu)²/(wᵀΣw); the common factor Var(y) cancels.
For any fixed weights w, such as the shipped CHAIN weights, fusion beats the PRM iff

    G(w) = (wᵀu)² / (wᵀΣw) − u_0² / Σ_00  >  0,        Σ = C − uuᵀ.

**(f) Observed covariates.** Regress z out of every view, within answers and on a flexible basis, for estimation.
Every loading is then multiplied by the same √Var(ỹ), where ỹ is y with its z-explained part removed. The ratios
u_k/u_0 that determine w* and the sign of G are unchanged. Scoring can use raw or z-adjusted views; that is a
separate choice (plan Stage 3).

The earlier finding that removing z costs signal in scoring (LR-gap §9: the N56 oracle drops from .810 to .762)
does not apply to estimation.

**(g) Abstain conditions, all label-free.** Return the PRM when:
- K < 4 for route (b);
- an anchored covariance C_0k is ≤ 0 after z-adjustment (a violation of Δ ≥ 0 or of λ_0 = 0);
- a loading is out of range (u_k² ≥ C_kk);
- the bootstrap interval of G, resampled by answer, contains 0.

When both routes are available, disagreement between (b) and (c) is an over-identification check, as in
Jaffe et al. (2015).

**Interpretation.** The PRM plays the role of a *marker indicator* in confirmatory factor analysis:
- it fixes the identity of the error factor;
- its own loading is still estimated, so the PRM is not trusted blindly.

This uses what is special about PRM boosting: one view was trained, elsewhere, to target exactly the latent we
want.

*In-model checks* (pure Python; Gaussian unless stated; detailed in the plan file):

| Regime (N = 20k) | PRM | shipped SML | anchored Fisher | Bayes |
|---|---|---|---|---|
| strong PRM, telemetry nuisance (Socratic-like) | .856 | .703 | .880 | .880 |
| strong telemetry, mild nuisance (PRMBench-like) | .801 | .898 | .908 | .911 |
| strong PRM, no nuisance | .855 | .873 | .896 | .897 |

- **Third-order gate on random regimes.** Correct when deciding:
  - N = 1.6k: 15/19, abstaining 37%;
  - N = 6k: 24/30;
  - N = 30k: 15/16;
  - N = 94k: 8/8.
- **Known weaknesses:**
  - Heavy-tailed noise raises the variance of the third moments (6/12 at N = 30k; 8/12 with 1% winsorizing).
  - Noise shared by the PRM and the telemetry breaks the anchor (6/12 with strong skewed sharing).
  - When λ_0 ≠ 0 is not handled, the anchored covariances turn negative and the rule abstains, which is correct.
  - Per-view gaussianization destroys the third-order signal.

### P6. The decision layer

- **Bayes decisions need calibrated posteriors.** Per-step Bayes decisions for an F1-type metric need calibrated
  posteriors P(y=1|x). These require the prevalence p and the class-conditional densities. From second moments
  only u = √Var(y)·Δ is identified, so p is confounded with view quality. Third-moment prevalence estimates in
  this project were 1.5–3× too high: .22–.42 against a true .14–.16 (Steps 459, 464; `error_count_label_free_v1`).
- **GAP** sorts an answer's scores and flags the top k ≤ ⌈n/2⌉ at the largest gap. In one dimension this is the
  single-linkage two-cluster split, with the prior that errors are a minority of steps. It is label-free and
  answer-local, with no optimality claim. On PRMBench it equals the construction prior "about two errors per
  answer" (GAP − k=2 = −0.40 [−0.77, 0.00] on CHAIN).
- **A count rule with a guarantee needs labels for calibration.** Risk-controlled counting (Laufer-Goldshtein,
  `ADVISOR_TOOLS.md`) is such a rule and belongs to a separately labeled calibrated track.

## 5. Predictions and their status

| Prediction from the framing | Status |
|---|---|
| sign(CHAIN − PRM) follows d_T vs d_p when the PRM share is small | 15/15 with labels (observed) |
| CHAIN ≈ equal-weight fusion where d_T ≈ d_p (flat optimum) | observed on 3 strong PRMs, PRMBench |
| PRM share ≈ 1/(K+1) under similar group loadings | median 1.03 over 24 cells (observed) |
| u/(1−u²) on unanchored loadings fails under telemetry nuisance | in-model only; reference arm planned |
| Centering, not scaling, is what identifies the error factor | in-model only; data test planned (Stage 4) |
| A label-free anchored gate predicts the 15 signs, abstaining on small sets | **to be tested** (`GATE_PREREG.md`, Stage 2) |

## 6. Relation to the literature

**The moment-based unsupervised-ensemble line.**
- **Parisi et al. 2014.** The SML; its weights are first-order in classifier accuracy (P3).
- **Jaffe et al. 2015.** Third moments identify class imbalance and per-classifier accuracies. Their second- vs
  third-order consistency also gives our over-identification check.
- **Jaffe et al. 2016.** Dependent classifiers in latent groups; the L-SML grouping.

**Close neighbours.**
- **FUSE** (Lee et al., ICML 2026). Response-level ensembling of verifiers. It uses a triplet-CI violation statistic
  built from third- to second-order ratios, then Jaffe-2015 moments and a pseudo-label aggregator. It reports that
  the plain moment ensemble loses to the naive mean in 7 of 10 settings, as our telemetry-only banks do (Steps 467–468).
- **CARE** (Zhao et al., ICML 2026). Confounders as a latent low-rank part of the precision matrix. Here the main
  confounders are observed (z) and one view is a natural anchor.
- **Ising aggregation** (ICML 2026). The CI rule is suboptimal under class-dependent dependence. This is
  consistent with P2's warning that weights must track quality.
- **Dror et al. 2017.** The MSE-optimal ensemble w* = C⁻¹ρ, ρ_k = Cov(x_k, y). Our w* ∝ C⁻¹u is the
  classification analogue. Their g² non-identifiability parallels the p confounding in P6.

**Why equal weights often win.** The forecast-combination puzzle (Smith & Wallis 2009; Claeskens et al. 2016):
estimation error and misspecification favour equal weights unless one forecast clearly dominates. P2 says when
it dominates. P5 says how to detect that without labels.

**Identification by anchors / marker indicators.** Standard in confirmatory factor analysis. Third-order tensor
methods (Anandkumar et al. 2014) are the general route.

**Supervised or certified neighbours** use labels on our benchmarks and are compared under declared access:
CROP (conformal certificate on a supervised risk score), PRISM (retrained PRM).

**What this framing adds.**
1. Within-answer estimands with answer fixed effects.
2. Observed nuisance handled in estimation, not only as a control.
3. A heterogeneous-quality regime with one trained view used as an anchor.
4. A p-free gate that predicts when fusion beats the PRM.

The advisors' earlier positioning (Step 147: the contribution is the signal, not "spectral fusion") is kept:
1–4 are about identifying *this* signal correctly.

## 7. What this note does not claim

- The Gaussian AUC formula is an approximation for real within-answer AUC. Stage 1 checks its predictive accuracy
  with labels.
- The anchor condition can fail. The telemetry comes from the PRM's own fine-tuned backbone, so their noise may be
  shared. The standalone base-model telemetry (`results/prm_standalone_telemetry_v1`) is a natural check.
- In-model simulations are not evidence on real data. All real-data numbers here come from existing result files.
  PRMBench, Socratic and Hard2Verify are development data, inspected many times; nothing here is untouched
  confirmation.
