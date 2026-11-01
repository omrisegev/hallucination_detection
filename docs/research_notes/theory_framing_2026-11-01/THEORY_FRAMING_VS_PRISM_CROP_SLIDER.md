# Label-free PRM boosting: theoretical and algorithmic framing for comparison with PRISM, CROP and SLIDER (2026-11-01)

**Status.** Working note for the thesis chapter and for the advisor meeting. It extends
`docs/research_notes/math_rigor_2026-10-24/FRAMING_THEORY.md` (model (M), P1–P6). It adds three new statements
(T1–T3) and checks two of them on stored data. It also collects the evidence of Steps 475–487 and of the
raw-null diagnosis (D1–D3) in one place.

**Scope.** Four PRMs: Qwen-PRM-7B, Qwen-PRM-72B, PURE, RLHFlow. Math-Shepherd is not reported. In-scope arms:
AUTO, TELE16, SUB5 (CHAIN is a reference only). All numbers are development data. Labels are used only for
evaluation and for diagnostics that say so.

**New files in this directory.**

| File | Contents |
|---|---|
| `decompose_pooled_auc.py` | Checks T1 and T2 on the D1–D3 bundle (68 cells, branch `claude/raw-null-gate-2026-10-30` @ `f446884e`) |
| `DECOMP.csv` | One row per (cell, score): π_w, pooled / within / between AUC, answer-level AUC |
| `shape_only_power.py`, `SHAPE_ONLY_POWER.csv` | T3: power of a shape-only clean-answer test (Gaussian model) |

---

## 0. Summary

1. **A step-error detector has three layers:** the views (what we read), the aggregator (how we combine), and the
   decision rule (how many steps we flag, or where we cut). PRISM changes the view. CROP changes the aggregator
   and the decision rule, with labels. SLIDER makes a new view for one error type. We change the aggregator,
   without labels. A fair comparison holds the other layers fixed (Section 2).
2. **Our aggregator is invariant to per-answer affine changes (answer-z).** This removes answer effects (P1). It
   also removes all answer-level information. So our score can rank steps inside an answer. It cannot say whether
   an answer is clean, and it cannot feed a global threshold (T1).
3. **The metrics of CROP and the global thresholds of PRISM are between-answer metrics.** Pooled step AUROC is a
   weighted mean of within-answer and between-answer pair AUCs. The within-answer weight π_w is .0002–.0007 on
   all four benchmarks we checked. So pooled AUROC measures between-answer order, which answer-z discards (T2).
   This explains why fusion loses 2–4× in CROP's kept-prefix on CRV and PRM800K but wins on PRMBench under GAP.
4. **Every label-free clean-answer gate and count rule failed for one reason.** After answer-z, a gate can use
   only the shape of the answer. At the observed step separation (Cohen's d ≈ 0.8–1.15, D3), the best shape-only
   test has power .06–.07 at 5% false alarms. The gates worked only in synthetic tests with d = 2.3–3 (T3).
5. **Where fusion helps is predicted by P2.** Fusion beats a view when the other views are strong enough
   relative to it and not redundant with it. This held in 13/15 cells with labels (correlation .979).
6. **The algorithmic statement that follows:** label-free, invariant ranking (our contribution) plus a decision
   layer. The decision layer is either GAP (label-free, answer-local, prior "errors are a minority") or a
   calibrated rule (needs labels; about 50 labelled answers are enough on PRMBench, Step 487).
7. **Against each competitor, under matched access:**
   - **PRISM:** its gain (65.5 → 68.0) is an operating-point move. The untrained PRM at PRISM's TPR gives 68.26.
   - **CROP:** with Qwen-PRM-7B, label-free AUTO in CROP's own conformal layer keeps 21.6% of the prefix. This
     equals CROP's supervised Trace+PRM (21.8%). On CRV and PRM800K the PRM alone wins clearly (T1, T2).
   - **SLIDER:** on redundancy, telemetry reaches 76–79% accuracy against 61.9%. It uses one forward pass
     instead of about 50 LLM calls per problem. SLIDER's protocol is under-specified, so this is approximate.

---

## 1. What worked and what did not (Steps 475–487, D1–D3)

### 1.1 What worked

| # | Finding | Numbers | Source |
|---|---|---|---|
| W1 | Telemetry from one forward pass has real step-error signal | TELE16 standalone PRMBench PRMScore 67.6–68.3; within-AUC ProcessBench .60–.77, Socratic .62–.70 | `telemetry_only_vs_competitors_v1/TABLES.md`; Step 480 |
| W2 | Fusion improves weak PRMs on PRMBench, also at a matched operating point | matched TPR: RLHFlow +8.92 (CHAIN); PURE +1.02 | `STATUS_VS_COMPETITORS_2026-10-28_HE.md` §2 |
| W3 | Within-answer ranking gain on PRMBench beyond position and length | CHAIN > PRM within-AUC 5/5 after residualization | ablation controls |
| W4 | Redundancy and circular-logic errors: telemetry is the strong signal | PRISM T7 categories NR 73–74, NCL 69–72 vs NR 46–60 for every PRM; SLIDER protocol 76–79 vs 61.9 | `TABLES.md`; Step 481 |
| W5 | L-SML weights beat the plain mean, by a small but significant margin | standalone +0.42 to +1.13 PRMScore; backbone +0.61 to +1.58 | standalone review S6 |
| W6 | The GAP decision layer adds about +2 PRMScore over Q80 for every arm | Qwen-7B CHAIN 68.25 → 70.40 | `prm_boost_collection_v1` |
| W7 | Position residualization makes the Jaffe assumptions hold | all 16 channels above chance 8/8; Lemma-1 misfit ≤ .046 | Step 481 |
| W8 | P2 predicts the sign of (fusion − PRM) | 13/15 signs, r = .979 (with labels) | `prm_boost_math_rigor_v1/stage1` |
| W9 | Label-free AUTO is a good risk score inside CROP's conformal layer on PRMBench | kept 21.6 / 20.5 / 18.9 / 15.1 (7B / 72B / PURE / RLHFlow); CROP supervised Trace+PRM 21.8 | `TABLES.md` (AUTO) |
| W10 | Few labels are enough for a calibrated, supervised weighting on PRMBench | full LR beats AUTO from 50 / 50 / 100 / 50 labelled answers; +1.3 to +2.6 with the full pool | Step 487 |
| W11 | The ranking exists for clean answers too, but only in the raw (non-invariant) level | answer-level AUC of the raw answer max: ProcessBench .77, CRV-Arith .81, GSM8K .69 | D1; Step 484 |
| W12 | Count from a Markov DS posterior: partial gain in 3 cells (post hoc) | PRMBench PURE AUTO 69.70 (+0.64 over GAP); TELE16 69.38; Socratic TELE16 7B 65.07 | Step 486 |

### 1.2 What did not work

| # | Attempt | Result | Diagnosis (this note) |
|---|---|---|---|
| F1 | Fusion vs strong PRMs outside PRMBench (Socratic, ProcessBench, CRV, PRM800K) | loses; CROP kept PRM 97.4 / 68.3 / 79.8 vs CHAIN 24.0 / 35.0 / 32.5 | P2 regime (iii)–(iv); T1, T2 |
| F2 | "Above PRISM" on strong PRMs | at matched TPR CHAIN = PRM (7B −0.06, 72B −0.28) | T1 + operating-point effect (§4.7) |
| F3 | Label-free clean-answer gates: DS gate E1, E1b, raw-null G1–G3, answer mixture | 0/20 raw-null; 0/12 mixture; DS prevalence .85–1.0 | T3 |
| F4 | Label-free error count: DS thresholds, neighbor HMM, Markov DS count | HMM overcounts 2–3×; MDS E1 0/4; K2 beats MDS in all valid Socratic cells | T3, P6, §4.6 |
| F5 | PRM-anchored identification (P5) on real data | identified in 3/12 cells | anchor condition fails (\|corr\| .13–.35) |
| F6 | Bank, subset and SML-algorithm search | weighting ≤ 1 pt; bank 2–3 pts but tuned on PRMBench | P3: SML ≈ equal weights |
| F7 | Labels from another benchmark | Socratic → PRMBench: AUTO best for 4/4; PRM800K LR: Qwen 3–5% PRM weight | domain shift (Bracha LR, Step 487) |
| F8 | Socratic supervised win | mostly step position (LR on [PRM, position] matches the full LR) | benchmark design |

### 1.3 One open item from the other computer that needs a check

Step 483 (`claude/prm-boosting` @ `a76e0ba`) reports TELE16 + GAP Macro SLA 47.23% vs Mind the Gap 39.39% on
ProcessBench. The backbones differ (Qwen-PRM-7B backbone vs Mind the Gap's Qwen3-8B / 4B). The comparison is
not access-matched yet. Do not put it in the deck before a check.

---

## 2. The detector as three layers

**Definition.** A step-level error detector is a triple D = (Φ, f, δ):
- **Φ, views.** Per step i of answer j, a vector x_ij = Φ(question, steps) ∈ ℝ^m.
- **f, aggregator.** A score s_ij = f(x_j)_i ∈ ℝ, where x_j is the whole answer.
- **δ, decision rule.** Flags F_j = δ(s_j) ⊆ {1..n_j}, or a cut point (prefix length).

**Access vector.** For each layer, record (a) labels used, and where they come from (in-domain, other domain, none);
(b) extra compute (training, extra LLM calls).

| Method | Φ (views) | f (aggregator) | δ (decision) | Labels | Extra compute | Guarantee |
|---|---|---|---|---|---|---|
| **Ours (AUTO / TELE16 / SUB5)** | frozen PRM risk + 16 telemetry channels from the same pass (TELE16: base-model telemetry only) | answer-z per channel, L-SML (latent-group SML, label-free) | GAP: per-answer top-k, k from the largest gap, k ≤ ⌈n/2⌉ | none | none | none |
| **PRISM** (2606.09078) | re-trained Qwen-PRM-7B (Bradley–Terry pairs from PRM800K) | identity (one view) | global threshold (0.5, inferred from Fig. 5b) | PRM800K pairs (other domain) | fine-tuning | none |
| **CROP** (2605.30085) | frozen PRM + trace features from the generator | supervised LR on Y_t (60% of each dataset) | first exceedance of a global λ̂, set by conformal risk control on 20% | in-domain, 80% of the data | none | marginal P(kept prefix has a labelled error) ≤ α |
| **SLIDER** (2610.00571) | Step-RRI: PID of (prefix, step → answer) over ~50 GPT-4o-mini variants | identity | linear classifier threshold on a labelled held-out set | in-domain (size not given) | ~50 LLM calls + encoder + PID solver per problem | none |

**Reading.** Each paper changes a different layer. So a headline number mixes a layer effect with the other
layers. Section 5 gives the matched comparison per competitor.

---

## 3. Model

We keep model (M) of `FRAMING_THEORY.md` §2:

    x_ijk = a_jk + Δ_k · y_ij + β_k · z_ij + λ_k · η_ij + ε_ijk

- a_jk: answer effect (arbitrary, fixed effects);
- Δ_k ≥ 0: class separation of view k after orientation;
- z_ij: observed nuisance (relative position, length);
- η_ij: latent nuisance; ε: noise, block-diagonal across channel families.

Two estimands:
- **E1, ranking:** within-answer AUC.
- **E2, decisions:** flags scored by PRMScore (or, for CROP, kept prefix at risk α).

The new statements below are about what the invariance in our aggregator allows for E1 and E2.

---

## 4. Theoretical statements

### T1. Invariance: what answer-z keeps and what it removes (new)

Let G be the group of per-answer, per-channel positive affine maps: x_ijk → b_jk + c_jk x_ijk, c_jk > 0.

**Statement.**
- (a) Per-channel answer-z, z_ijk = (x_ijk − mean_j(x_·jk)) / sd_j(x_·jk), is a maximal invariant of G. Every
  G-invariant score is a function of z. Our fused score s_ij = Σ_k w_k z_ijk is G-invariant.
- (b) Under (M) with unrestricted a_jk, invariance to the location part (b_jk) is necessary for a result that
  holds for every a_jk. This is the justification for centering (P1). The scale part (c_jk) is an added design
  choice, harmless when the within-answer SDs factor (P1).
- (c) **Cost.** Every G-invariant score is unchanged when a constant is added to all steps of one answer, so the
  answer level is removed exactly. Our fused score is linear in z, so Σ_i s_ij = 0 in every answer. Hence
  max_i s_ij ≥ sd_j(s) / √(n_j − 1) > 0 for every answer with n_j ≥ 2, also for clean answers. (Equality case:
  n_j − 1 equal values and one low value.)
  - A global threshold on s cannot say "this answer is clean" from the level.
  - The answer level, which carries "this answer has an error", is removed exactly.

**Check on stored data** (`DECOMP.csv`, medians over cells):

| Benchmark (cells) | Share of answers with an error | Answer-level AUC, raw fused max | Answer-level AUC, answer-z fused max |
|---|---|---|---|
| ProcessBench (20) | .653 | .770 | .458 |
| CRV-Arith (20) | .083 | .813 | .364 |
| CRV-GSM8K (20) | .210 | .694 | .523 |
| PRMBench, standalone (8) | .972 | .557 | .530 |

- The raw level ranks answers by "has an error". The invariant score is at or below chance.
- This is one cause of F1 (CROP protocol, ProcessBench "no error" calls) and of F3.

**Relation to PRISM.** PRISM's loss L_SC = −log σ(r⁺ − r⁻) has ∂L/∂a = 0 for the pair center a
(PRISM App. 9.3). So PRISM's training target is also invariant to a shared shift, here per prefix. PRISM trains a
ranking objective but reports a global-threshold metric. §4.7 shows that its reported gain is an operating-point
gain. Both methods meet the same split between ranking and decision.

### T2. Pooled step AUROC is a between-answer statistic (new, exact identity)

Let E and C be the numbers of error and correct steps, and e_j, c_j the numbers in answer j.

**Identity.** Over all (error, correct) pairs,

    AUC_pooled = π_w · AUC_within + (1 − π_w) · AUC_between,      π_w = Σ_j e_j c_j / (E · C).

AUC_within is the pair-weighted mean of the per-answer AUCs. AUC_between uses pairs from different answers.
Proof: split the Mann–Whitney sum by pair type.

**Check** (`DECOMP.csv`; the identity holds to 1e-7 in all 136 rows):

| Benchmark | π_w (median) | Pooled AUC, raw fused | Pooled AUC, answer-z fused | Within-answer AUC (answer mean), raw / z |
|---|---|---|---|---|
| ProcessBench | .0002 | .814 | .722 | .773 / .730 |
| CRV-Arith | .0003 | .743 | .493 | .739 / .509 |
| CRV-GSM8K | .0007 | .691 | .536 | .625 / .562 |
| PRMBench (standalone) | .0002 | .736 | .733 | .770 / .771 |

**Consequences.**
- CROP's step AUROC, and any global threshold (PRISM, CROP's λ̂, the matched-TPR row), measure
  between-answer order. With between-answer order fixed, a gain in within-answer order (our E1) moves pooled
  AUROC by only π_w ≈ 10⁻⁴ times that gain.
- **Our metric and theirs are almost orthogonal.** This is why CHAIN loses 2–4× in CROP's kept-prefix on CRV and
  PRM800K but wins PRMBench under GAP (a per-answer rule).
- On PRMBench, between-answer AUC is not hurt by answer-z (.736 vs .733). 97% of answers have an error, so
  the answer level carries little information there.

### T3. Shape-only clean-answer detection is weak at the observed separation (new)

By T1, a label-free gate on the invariant score can use only the shape of the answer: one or two steps that stand
out. Take n correct steps ~ N(0, 1) and k error steps ~ N(d, 1). The invariant max test (Grubbs) is set to 5%
false alarms on clean answers (`SHAPE_ONLY_POWER.csv`, 200k draws):

| d | k | Power, shape-only (n = 12) | Power, known null (n = 12) |
|---|---|---|---|
| 0.8 | 1 | .056 | .079 |
| 1.0 | 1 | .063 | .095 |
| 1.15 | 1 | .069 | .112 |
| 1.15 | 2 | .069 | .170 |
| 2.3 | 1 | .219 | .399 |
| 3.0 | 1 | .403 | .660 |

- The observed step separation is d = 1.15 (ProcessBench, CRV-Arith), 0.79 (GSM8K), 0.91 (PRMBench) for the
  fused score, and 0.47–0.81 for single groups (D3).
- At these values, a shape-only gate is close to chance (.06–.07 against .05).
- The gates passed their synthetic tests only with d = 2.3–3 (raw-null selftest).
- **This explains F3 and F4 with one mechanism.** DS gates, the raw null, the answer mixture and Markov DS all
  work on scores that are invariant, or near-invariant, per answer.
- A **known null** (a calibrated level) doubles the power but needs a calibrated level. That level comes from
  labels (CROP) or from a view whose level is calibrated across answers (the PRM: raw PRM answer max AUROC
  .78–.96 on ProcessBench, Step 484).

**Limit.** T3 is a Gaussian, independent-step model. Real errors come in runs (P(error | previous error) .417 vs
.123, Step 485). Runs make the shape signal of one answer weaker than k independent shifted steps.

### T4. When fusion beats a view (P2, kept)

Two standardized views: PRM with separation d_p, telemetry composite with d_T, within-class correlation ρ, PRM
weight a:

    D(a) = (a d_p + (1 − a) d_T) / √(a² + (1 − a)² + 2a(1 − a)ρ),      AUC = Φ(D(a)/√2).

- Strong telemetry (d_T ≥ d_p): every a ∈ [0, 1) helps. This is PRMBench.
- Weak telemetry (ρ d_p < d_T < d_p): fusion helps only if a > a_min = R/(1 + R),
  R = (d_p² − d_T²) / (2 d_p (d_T − ρ d_p)). This is Socratic with strong PRMs.
- Redundant telemetry (d_T ≤ ρ d_p): no positive weight helps.
- Evidence: 13/15 signs, r = .979 between predicted and observed differences.

**Use for the comparison.** P2 is the formal version of "boosting helps weak PRMs and new error types". It gives
a condition, not a benchmark claim. No competitor has such a condition.

### T5. What the label-free weighting can and cannot estimate (P3, P4, kept)

- P3: the shipped SML weight is v_k ∝ u_k / (λ₁ + u_k²). It under-weights the strongest view. With similar group
  loadings it gives about 1/(K + 1) to each group (median share × (K + 1) = 1.03 over 24 cells).
- P4: a nuisance shared by the telemetry hides the PRM's strength in second moments.
- Consequence: the label-free weights are close to equal weights. This is good where P2 says the optimum is flat
  (PRMBench). It is bad where the PRM must dominate (Socratic, strong PRMs).
- With labels, Fisher weights on CHAIN's views gain only −.003 to +.022 within-AUC on PRMBench. The bottleneck
  is when to fuse, not how.

### T6. The decision layer (P6, extended with Steps 484–487)

- **Bayes flags need P(y = 1 | x).** This needs the prevalence p. From second moments only u = √Var(y)·Δ is
  identified, so p is confounded with view quality.
- **Agreement identifies voter quality, not error quantity** (Step 486). In the Markov DS count, the fitted
  per-voter δ tracks voter within-AUC (Spearman .6–1.0 in 5/6 valid cells). The count does not beat SHARE or K2.
- **GAP** is the single-linkage two-cluster split of one answer, with a minority cap. It is invariant (uses the
  shape only). On PRMBench it equals the construction prior of about two errors per answer.
- **Labels buy calibration cheaply.** About 50 in-domain labelled answers let a full LR beat AUTO on PRMBench.
  Below about 20 labels, AUTO is better (Step 487). Labels from another benchmark do not transfer (F7).

### T7. PRISM as an operating-point move (empirical, with the formal reason)

- PRMScore at a global threshold τ is a function of (TPR(τ), TNR(τ)). Moving τ along one ROC curve changes
  PRMScore without any change in ranking.
- PRISM reports one point: TPR 87.30, TNR 52.86, PRMScore 68.0. The untrained Qwen-PRM-7B with τ set to the
  same TPR gives 68.26. So PRISM's point is on or inside the baseline curve near that TPR.
- PRISM's own Theorem 9.1 assumes an additive score shift ±δ between the two models. A pure shift is a threshold
  change.
- **Fair statement:** PRISM's +2.5 is an operating-point gain. A ranking gain needs a ROC or AUC comparison.
  PRISM does not report one.

---

## 5. The algorithm, as we state it in the thesis

**Algorithm 1: invariant label-free boosting with a separate decision layer.**

Input: answers j = 1..J with steps; frozen PRM; base model (for telemetry).

1. **Views.** One teacher-forced pass. Read the PRM risk r_ij = logit(1 − reward) and 16 telemetry channels.
   (TELE16: base-model telemetry only, no PRM.)
2. **Nuisance.** Remove a cubic in relative position from each channel (estimation only, W7).
3. **Invariance.** Per-answer, per-channel answer-z (T1). Assumption: answer effects are arbitrary.
4. **Orientation.** Majority sign; a channel with a negative estimate is a model violation (Lemma 4 check).
5. **Latent groups.** Jaffe et al. 2016 grouping (Eq. 14/15). AUTO lets the PRM fall in any group. SML inside
   each group. DS-EM across groups. Assumption: conditional independence across groups.
   Label-free diagnostics: Lemma-1 misfit, eigengap, group stability.
6. **Ranking output.** s_ij = Σ_k w_k z_ijk. This is the contribution. It targets E1.
7. **Decision layer, one of:**
   - **D0, label-free:** GAP top-k per answer. No guarantee. Prior: errors are a minority of steps.
   - **D1, calibrated:** a labelled calibration set of m answers. Either CROP's conformal λ̂ (guarantee:
     marginal risk ≤ α), or a fitted count or weight (Step 487: m ≈ 50 on PRMBench).
8. **Answer level (proposal H, not yet tested).** For any decision that needs answer-level information (clean
   answer, prefix cut, ProcessBench "no error"), add a level term from a calibrated view:
   s^H_ij = ℓ_j + κ · s_ij, with ℓ_j = the raw PRM risk maximum of answer j.
   - By T1 and T2, ℓ_j carries the between-answer order and s_ij the within-answer order.
   - Prediction: H ≈ PRM on CRV / PRM800K (between-answer metrics), and H ≈ fusion on PRMBench under GAP.
   - Partial support only: on ProcessBench (PRISM T6 metric), the PRM gate with AUTO's order (NC) gives average
     F1 33–67, against 8.9–10.1 for the telemetry rule alone. The PRM alone gives 29–78, so NC is not yet better
     than the PRM.

**What is label-free and what is not.** Steps 1–7 (D0) use no labels. D1 uses in-domain labels only to set the
decision layer, not the ranking.

---

## 6. Matched-access comparison, per competitor

### 6.1 PRISM

| Question | Answer |
|---|---|
| Which layer does PRISM change? | Φ (re-trains the PRM) and, in effect, the operating point of δ |
| Matched comparison | Same PRM (Qwen-PRM-7B), same metric (official PRMScore), same TPR (87.3) |
| PRISM | 68.0 (re-training on PRM800K pairs) |
| Untrained PRM at PRISM's TPR | 68.26 |
| Ours at PRISM's TPR (global threshold, labels set τ) | AUTO 64.2–68.0; CHAIN 68.20 (7B) |
| Ours under GAP (label-free, per answer) | AUTO 69.9 / 70.2 / 69.1 / 65.9 (7B / 72B / PURE / RLHFlow) |
| Weak PRMs at matched TPR | RLHFlow +8.92 over its PRM (CHAIN) |
| Categories (PRISM T7) | Telemetry NR 73–74, NCL 69–72 vs PRISM 52–54 / 64–66 |
| Theory | T7: PRISM's gain is an operating-point gain. T1/T2: our gain is within-answer, so it shows under a per-answer rule, not under a global threshold |
| Claim we can make | "No training is needed to match PRISM's PRMBench number; a threshold move suffices. Label-free fusion adds a within-answer ranking gain that a per-answer decision rule uses, and a large gain for weak PRMs" |
| Claim we must not make | "Ours > PRISM" at the same global threshold on strong PRMs |
| Missing | PRISM checkpoint (not released). A ROC comparison would need it |

### 6.2 CROP

| Question | Answer |
|---|---|
| Which layers does CROP change? | f (supervised LR, in-domain) and δ (conformal global λ̂, first exceedance) |
| Matched comparison | Put our label-free score into CROP's δ with CROP's calibration split. Same metric (kept prefix at α = .05) |
| PRMBench, Qwen-PRM-7B | AUTO 21.6% vs CROP Trace+PRM 21.8% (supervised) vs Direct PRM 4.8% |
| PRMBench, standalone TELE16 | 16.4–18.2% (above CROP's token/format 16.2 and trace-feature 15.9) |
| CRV-Arith / PRM800K / CRV-GSM8K (Qwen-PRM-7B) | PRM 97.4 / 68.3 / 79.8 vs CHAIN 24.0 / 35.0 / 32.5 |
| Theory | T2: kept-prefix and pooled AUROC are between-answer quantities (π_w ≈ 10⁻⁴). T1: answer-z removes them. So CROP's protocol cannot see our gain and does see our loss |
| Claim we can make | "Label-free fusion matches CROP's supervised aggregator on PRMBench inside CROP's own guarantee. Our aggregator and CROP's decision layer compose" |
| Claim we must not make | Any CROP-protocol claim on CRV, GSM8K or PRM800K |
| Next | Proposal H inside CROP's δ on all six CROP datasets (CPU, stored scores) |

### 6.3 SLIDER

| Question | Answer |
|---|---|
| Which layer does SLIDER change? | Φ only: a new view for one error type (redundancy), plus a labelled threshold |
| Matched comparison | PRMBench redundancy class, LR threshold on a labelled split (SLIDER's own rule) |
| SLIDER | Step-RRI 61.91% accuracy; cosine 51.6; InfoGain 48.4 |
| Ours | TELE16 76–79% accuracy, 76–77% balanced; label-free rule 74–75% balanced; AUTO 69–76 |
| Cost | Ours: one forward pass. SLIDER: about 50+ GPT-4o-mini calls, an encoder and a PID solver per problem (~38 s per example) |
| Theory | SLIDER's Theorem 2 says a redundant step has Uni = Syn = 0 relative to its prefix. Our telemetry channels that measure surprise and change (q15_H1, chosen_surprisal, top15_turnover) are cheap proxies of "no new information". The view layer is where telemetry adds to a PRM (P2 regime (ii) for this class) |
| Claim we can make | "On redundancy, telemetry is a cheaper and stronger signal than Step-RRI, and PRMs are near chance there" |
| Caveat | SLIDER does not give η, split sizes or the metric definition. The comparison is approximate. SLIDER code is not released |

### 6.4 One table for the deck

| | PRISM | CROP | SLIDER | Ours |
|---|---|---|---|---|
| Layer changed | view (re-train) | aggregator + decision | view (new, one class) | aggregator |
| Labels | other domain (PRM800K pairs) | in-domain, 80% | in-domain threshold | none (D0) / ~50 answers (D1) |
| Extra compute | fine-tuning | none | ~50 LLM calls per problem | none |
| Guarantee | none | marginal risk ≤ α | none | none (D0); CROP's (D1) |
| Estimand | global threshold | prefix, between-answer | redundancy class | within-answer ranking |
| Formal result | BoN bound (FP caps accuracy) | conformal risk control | PID identity for redundancy | P2 condition, T1–T3 |
| Benchmarks × PRMs | 2 × 1 | 6 × 1 | 1 (class) × 0 | 8 × 4 |

---

## 7. Claims: defensible and not defensible

| Claim | Status | Basis |
|---|---|---|
| Label-free fusion improves within-answer ranking on PRMBench beyond position and length | **yes** | W3 |
| Label-free fusion improves weak PRMs at a matched operating point | **yes** (PRMBench) | W2 |
| Telemetry is the right view for redundancy-type errors | **yes** (approximate vs SLIDER) | W4 |
| PRISM's PRMBench gain is an operating-point gain | **yes**, with the caveat that a ROC check needs PRISM's model | T7 |
| Our score composes with CROP's conformal layer and matches its supervised aggregator on PRMBench | **yes**, Qwen-PRM-7B | W9 |
| P2 predicts where fusion helps | **yes, with labels**; label-free gate failed | T4, F5 |
| A label-free invariant method cannot decide "clean answer" at d ≈ 1 | **yes, as a model statement** (T3) plus 4 negative experiments | T1, T3, F3 |
| Fusion improves strong PRMs in general | **no** | F1 |
| Fusion helps under global-threshold or prefix metrics on CRV / PRM800K / GSM8K | **no** | F1, T2 |

---

## 8. Proposed next checks (CPU, stored scores; each needs a pre-registration)

1. **Proposal H in CROP's protocol** on the six CROP datasets and on ProcessBench (PRISM T6 metric). Pass rule
   to fix before the run: H ≥ max(PRM, fusion) − 1 pt on every dataset.
2. **T2 on the PRM-backbone PRMBench cells** (12 cells, on the PC only). Expected: π_w ≈ 10⁻⁴ again.
3. **Within / between split of every competitor row we report** (PRISM T2, CROP T6), so that each number in the
   deck states which order it measures.
4. **Mind the Gap (Step 483):** redo with a matched backbone before any claim.

---

## 9. Sources

- Theory base: `docs/research_notes/math_rigor_2026-10-24/FRAMING_THEORY.md`.
- Competitor readings: `docs/research_notes/competitor_explainers_he_2026-10-24/{01_CROP,04_PRISM,10_SLIDER}.md`.
- Competitor numbers: `results/telemetry_only_vs_competitors_v1/TABLES.md`
  (`claude/telemetry-only-vs-competitors-2026-10-28` @ `1ceb587`); `STATUS_VS_COMPETITORS_2026-10-28_HE.md`.
- Steps 484–487 and D1–D3: branch `claude/raw-null-gate-2026-10-30` @ `40ef9b7` (`HISTORY.md`,
  `results/raw_null_gate_diag_v1/`, `results/supervised_*_v1/`, `results/markov_ds_count_v1/`,
  `results/neighbor_readout_v1/`); Step 484 on `claude/answer-mixture-gate-2026-10-29` @ `acf8c92`.
- Step 483: `claude/prm-boosting` @ `a76e0ba`.
