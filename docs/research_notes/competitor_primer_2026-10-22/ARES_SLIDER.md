# ARES and SLIDER: how they work, and how they compare with PRM boosting (2026-10-22)

Sources read in full:
- ARES: arXiv 2507.12948v2 and the ACL Anthology PDF 2025.emnlp-main.382. Tables 1 and A7 and Algorithm 1 are identical in the two.
- SLIDER: arXiv 2610.00571v1 (HTML), including the Figure 7 image.

"[mine]" marks numbers I computed. "[unverified]" marks claims I could not check.

---

## 1. ARES (You et al., EMNLP 2025)

**Problem and idea.** An LLM judge that reads a whole chain is misled by earlier errors. Once step 4 is wrong, it may accept step 5 because step 5 follows from step 4 (Fig. 1).

ARES judges each step only against premises that were themselves judged sound. A step is "sound" if it is entailed by the base claims (the question) and the earlier sound steps (Def. 2.1).

The score of step k is its **entailment stability** τ_k. This is the expected entailment probability of the step, averaged over random subsets of the earlier claims. Each earlier claim is kept with its own soundness probability (Eq. 3, 6).

**What is guaranteed.** With N ≥ log(2m/δ)/(2ε²) sampled chains, every estimate τ̂_k is within ε of τ_k with probability at least 1−δ (Thm. 3.1; Hoeffding plus a union bound, App. A).

This bounds only the Monte Carlo error **relative to the chosen entailment model**, not whether a step is truly correct. The authors note that a miscalibrated model gives unreliable scores (Limitations). The abstract's phrase "certified … guarantees of its soundness" overstates what the theorem proves.

**At test time.**
- Inputs:
  - the question, split into sentences as base claims (`src/exp_helpers/datasets/prmbench.py`);
  - the steps;
  - an entailment model: GPT-4o-mini or Qwen3-4B answering on a 7-point Likert scale mapped to {1, .8, .6, .5, .4, .2, 0} at temperature 0 (C.1), or Qwen2.5-Math-PRM-7B (C.9);
  - ε = δ = 0.1 and base-claim prior p = 0.95 (C.2).
- Not needed: the final answer, variants or logits.
- Output: τ̂_k for each step.

**Algorithm (Alg. 1).**
1. N = log(2m/δ)/(2ε²). For m = 10 steps this gives 265 samples (Table 3 caption).
2. For each sample, draw a keep/drop flag for each base claim with probability p.
3. For k = 1..m: compute p_k = E(kept claims, step k), then keep step k with probability p_k.
4. τ̂_k = the mean of p_k over the samples.

The code removes duplicate premise subsets (`stability_deterministic.py`), so actual calls are 0.03x to 0.31x of the bound (Fig. 5). Before deduplication that is about 2,650 entailment calls for a 10-step chain [mine]. With the PRM backbone, each call is a PRM pass with the dropped steps removed (`qwen_prm_model.py`).

**Threshold (uses labels).** In 5-fold cross-validation, they sweep the observed score values and keep the one with the highest Macro-F1. One fold is used for tuning and four for testing (Sec. 4, C.3).

**Results.**
- Average Macro-F1 with GPT-4o-mini: 72.1, the mean of 0.640, 0.708, 0.903 and 0.633 (Table 1).
- **PRMBench Macro-F1 (Table 1):**
  - GPT-4o-mini: ARES 0.640 vs LLM-Judge 0.643.
  - Qwen3-4B: ARES 0.636 vs LLM-Judge 0.675.
- **PRM backbone (Table A7):**
  - PRM alone: **0.749**.
  - ARES on the PRM: 0.736, identical to Entail-Prev in every fold of `metrics/performance/prmbench.qwen2.5-math-prm-7b-vllm.no_cot2.json`.
  - ARES helps only on the synthetic ClaimTrees set (0.733 vs 0.594).
- PRMBench subset size: not stated. Categories: none reported; one pooled Macro-F1.

**Code.** github.com/fallcat/ares, MIT license. It has no end-to-end runner, so the subset cannot be recovered from the code [unverified].

**Similar to us.**
- Step scores, then a threshold.
- Two-class Macro-F1, the same formula as PRMScore.
- Same PRM, with a PRM-alone baseline.
- The score itself uses no labels.
- Several readings are averaged per step: one model in many contexts for ARES, many signals in one context for us.

**Different from us.**

| | ARES | PRM boosting |
|---|---|---|
| Model access | black-box, many calls per step | gray-box, one teacher-forced pass |
| Signal | verbalized entailment | logit telemetry |
| Weighting | equal | L-SML |
| Step dependencies | modeled explicitly | not modeled |
| Threshold | tuned on labels | GAP, label-free, set per answer |

**Is comparing PRMBench numbers fair? No.**
- The subset is unknown and the threshold is tuned on labels.
- Their scale is different. Their PRM scores 0.749, while the same PRM's official PRMScore is 65.49 in our parity run (`results/prmbench_paper_metric_parity_v1/README.md`, 6,211 answers).
- Only their internal ranking is usable: under their protocol, ARES does not improve the PRM on PRMBench.
- A fair test means running our arms through their 5-fold sweep, which is cheap on CPU.

**Limitations they state.** Depends on the entailment model's calibration; costly; assumes claims are already decomposed; only four datasets.

**Inconsistencies.**
- Eq. 4 exponent should be (1−p_i)^(1−α_i).
- Eq. 5 mislabels its left side and omits the drop branch.
- Thm. 3.1 cites Eq. 7 (the estimator) where it means Eq. 6.
- Sec. 4.1 claims the best PRMBench F1 with GPT-4o-mini, but it is a tie (0.640 vs 0.643).
- The Table A7 caption says "ARES performs the best", but the PRM alone wins on PRMBench and CaptainCook4D.
- "+8.2 points" cannot be reproduced from Table 1. The best baseline on each dataset gives +8.1; the best single baseline gives +15.0 [mine].
- The main runs use p = 0.95, although Table A11 favors p = 1.
- The threshold is described as fitted on "training" (Sec. 4) and on "validation" (C.3).
- Prompts were tuned "by manually examining the results" (C.2).

---

## 2. SLIDER / Step-RRI (Halder, Zhang, Dutta; arXiv 2610.00571v1, 30 Sep 2026)

The arXiv comments field says it was accepted at the ICLR 2026 Workshop on Logical Reasoning of LLMs.

**Problem and idea.** The target is **repetitive steps**, not errors in general.

Partial information decomposition (PID; Bertschinger et al. unique information, Def. 1) splits I(A; S_<i, S_i) into four non-negative parts:
- redundant information (Red), shared by the prefix and the step;
- information unique to the prefix;
- information unique to the step;
- synergy (Syn).

Here A is the final answer, S_<i the prefix and S_i the current step (Prop. 1).

Step-RRI_i = Red − η·max{Uni(A:S_i|S_<i), Syn}, with η ≥ 1 (Def. 2).

Why it detects redundancy: if the answer-relevant content of S_i is already in S_<i, the step's unique and synergistic parts vanish, so RRI > 0 (Thm. 2). This holds in one direction only. Their own Sec. 2 example (S2 = x/2·3/5) is a useful step that still carries redundancy.

**Where the randomness comes from.** One trace is one sample, and PID needs a distribution. GPT-4o-mini therefore writes up to 50 **numerical variants** of the question and solution, keeping the structure (App. C.1).

"Information" means co-variation across these variants: does the step's embedding cluster predict the answer's embedding cluster?

**At test time.**
- Inputs: the question, the tagged steps and **the final answer A**.
- Generator: up to 50 accepted variants plus rejected ones. Generation stops after 5 mismatches in a row. Each call rewrites the whole solution.
- Segmentation: one Qwen2.5-72B call for long traces.
- Encoder: MiniLM embeddings for every variant and step.
- Runtime: about 38 s per PRMBench problem (D.5).
- Output:
  - RRI_i for steps i ≥ 2;
  - Trajectory-RRI = #{i : RRI_i > 0} (Def. 3).

**Algorithm (Sec. 2, Fig. 4).**
1. Generate variants recursively. Accept a variant only if it has the same number of steps; on a mismatch, restart from the original.
2. Embed S_<i (the concatenated prefix text), S_i and A with all-MiniLM-L6-v2.
3. Cluster each with k-means, k = 10 (Table 2).
4. Build the empirical joint distribution of the three clusters.
5. Estimate PID with the CVX estimator of Liang et al. (2023).
6. Compute RRI.

The label-free rule is RRI > 0. On PRMBench, the threshold comes from "training a linear classifier" on a held-out set, which uses labels (Sec. 3).

**Results.**
- **PRMBench, redundancy category only.** Positive = annotated redundant step; about 750 problems (App. D). Metric: **accuracy**.
  - Step-RRI 61.91%.
  - InfoGain 48.38%.
  - Cosine similarity 51.60%.
  - (Sec. 3, Fig. 7.) The error bars are not defined.
- On PRMBench there is no PRMScore, no F1, no other category and no PRM baseline.
- Spearman between Trajectory-RRI and reasoning length (Table 1, on binned means):
  - GSM8K: R1-Distill 0.95, QwQ 0.91, GPT-4.1 0.73.
  - AIME2025: 0.89, 0.75, 0.82.
- Data selection: across 9 training sets, ρ = −0.97 with the judge's efficiency score; accuracy 87.4–89.5% (Fig. 9, Table 3).

**Code.** No link in the paper. A GitHub search on 2026-10-22 found none [unverified that none exists].

**Similar to us.**
- Label-free step score, fitted per problem.
- Compares each step with its prefix, as our turnover and innovation channels do.
- Their InfoGain baseline is a crude telemetry channel: the change in QwQ-32B top-10 next-token entropy (D.2).

**Different from us.**

| | SLIDER | PRM boosting |
|---|---|---|
| Target | one category (redundancy) | all error types |
| Extra inputs | final answer A plus about 50 generated variants | one pass, no answer |
| Scope | numeric problems only | any problem |
| Signal | text embeddings | model internals |
| PRM and fusion | none | PRM fused with telemetry |
| Threshold | labeled | GAP, label-free |

**Is comparing PRMBench numbers fair? No.**
- The class has 758 problems and 11,613 steps, 13.2% of them redundant (14.1% for steps ≥ 2) [mine, Hugging Face `hitsmy/PRMBench_Preview`]. Always answering "not redundant" scores 86.8%, so 61.91% cannot be plain accuracy at the base rate. It is probably balanced [unverified].
- The split, η, the handling of step 1 and the number of problems that survived variant generation are not stated.
- A local check of older arms (Q80 rule, pooled fitting) found label-free balanced accuracy of 71.8–73.3 (`results/prmbench_paper_metric_parity_v1`). It was **not** run for GAP.

**Limitations they state (App. A).** Information lost by discretization; sensitivity to the encoder and clustering; dependence on variant quality; math only.

**Inconsistencies and gaps.**
- Thm. 2 assumes exact containment, which no longer holds after embedding and clustering (C.1 itself notes that long prefixes lose information).
- At most 50 samples fill 1,000 cells, which biases plug-in estimates upward. Table 2 values rise with k, and no confidence intervals are given [my analysis].
- Trajectory-RRI is a count, so it grows with step count.
- Correlating bin means inflates ρ.
- InfoGain below 50% suggests it was oriented the wrong way.
- η is 1.25 in Fig. 6 and 1 in Fig. 11, and is not stated for PRMBench.
