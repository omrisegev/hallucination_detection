# Competitors of PRM boosting: what each does, how it is like us, how it differs (2026-10-22)

Omri asked for a primer on the seven papers named in the 2026-10-22 advisor email. The papers were read in full
from primary sources by three agents (`PRISM_RPRM.md`, `ARES_SLIDER.md`, `MTG_BR2.md` in this folder); CROP from
`papers/digests/crop-conformal-clean-prefix.md`. "Where we stand" numbers come from
`results/prm_boost_collection_v1/tables/` (COMP_*, CROP_C*), development data on PRMBench.

**Our method, for reference.** A frozen PRM scores each step. In the same teacher-forced pass we read 16
token-probability channels of its backbone. We standardize per answer and fuse PRM + channels with L-SML
(weights from covariance, no labels), then flag steps per answer with GAP. Metric: official PRMScore.

## 1. CROP (EMNLP 2026 Oral per arXiv): a conformal certificate on a prefix
- **What it does.** Takes any per-step risk score and returns the longest prefix of the answer it can certify.
  On a labelled calibration split it picks the largest threshold λ such that at most a fraction α of
  calibration answers keep an error inside the returned prefix. At test time it cuts at the first step whose risk
  exceeds λ. Guarantee: marginal P(returned prefix contains an error) ≤ α, under exchangeability.
- **Its best risk score, Trace+PRM,** is a supervised ℓ2 logistic regression on 60% of the answers. Its inputs are
  the PRM score, a second model's teacher-forced features (Llama-3.1-8B: likelihood, confidence, entropy, energy,
  hidden-state trajectory scalars) and surface/position cues.
- **Like us.** Fuses a frozen PRM with token-probability telemetry at step level; one teacher-forced pass.
- **Unlike us.** Supervised fusion and a labelled threshold; a second model; a global cut with a guarantee
  instead of a per-answer flag count; metric is prefix kept / risk and pooled AUROC, not PRMScore.
- **Where we stand (Qwen-7B, CROP's protocol and grid).** L-SML fitted on training answers keeps 22.3% of each
  trace vs 21.8% for CROP (23.2% on a finer grid). Our format/position + PRM regression gets 21.2%, close to theirs,
  which suggests their gain is mostly surface cues (an inference). CROP's pooled AUROC is higher (.834 vs .792).
  A supervised regression on our telemetry keeps 25.2%. Calibrated on PRMBench and tested on Socratic our score
  breaks the 5% target (6.7% Qwen-7B, 8.2-25.9% others); CROP's own cross-domain test also breaks (14.8%).

## 2. PRISM (arXiv, June 2026): retrain the PRM to stop over-crediting
- **What it does.** Diagnoses "overcredit": PRMs call wrong steps correct (Qwen-PRM-7B flags only 30.7% of
  erroneous PRMBench steps), because step labels are mostly "correct". It fine-tunes Qwen2.5-Math-PRM-7B with a
  pairwise loss −log σ(r_correct − r_wrong) on PRM800K pairs (about 26k, expanded to about 220k with "lookahead"
  negatives: a later valid step used as a premature step), trained easy to hard.
- **Like us.** Same base PRM, same benchmark and metric, same diagnosis (the native threshold flags too few errors).
- **Unlike us.** Retrains the weights on human labels; one global threshold; no telemetry.
- **Where we stand.** PRISM 68.0 (TPR 87.3, FPR 47.1). L-SML with GAP 70.4 (TPR 88.6, FPR 45.1): better on both
  rates. Our GAP rule alone on the unchanged PRM gives 68.01, the same as PRISM's retraining gain. PRISM's
  checkpoint and code are not released, so we cannot boost it. PRMBench is built from PRM800K solutions, which
  PRISM trains on.

## 3. Mind the Gap (ICML 2026): entropy drops as an error detector, no PRM
- **What it does.** Per token, "evidence" = negative entropy of the top-20 next-token distribution; smoothed with an
  EMA; a "drop" is a sharp fall. Answer risk = mean of the 5 steepest drops; the first big drop marks the first
  error. One teacher-forced pass of Qwen3 on ProcessBench.
- **Supervision.** The score uses no labels, but the accept/reject threshold and the step threshold are set on a
  labelled half of the test data.
- **Like us.** Exactly our telemetry access: one pass, top-k entropy, change features.
- **Unlike us.** One hand-designed channel, no PRM, no fusion; predicts the first error only; reports exact-match
  localization (SLA) on ProcessBench, not F1, and nothing on PRMBench. Its released code computes a different
  signal (top-1 log-prob) and has no step-level code.
- **Where we stand.** Not measured yet. Fair test: add its signal as a 17th channel, and run it alone on PRMBench
  with GAP, from the PRM backbone and from the standalone model.

## 4. BR2-PRM (ACL 2026, multimodal): reliability-weighted judge scores
- **What it does.** A multimodal judge gives each whole answer a score for each of nine error types. For each type,
  a Bayesian model learns how reliable the judge's score is, by counting scores against known labels on PRMBench-V
  (supervised naive Bayes, no EM). The answer score is the product of the per-type probabilities of "no error";
  it is used to pick the best of N answers (+0.7 to +4.8 points over plain averaging).
- **Like us.** Fuses several noisy signals from frozen models, with a reliability model.
- **Unlike us.** Fitted on labels; each type has its own label, so it never weights several views of one shared
  label as L-SML does; whole-answer selection, not step localization; multimodal, so our text PRMs cannot run on it.
- **Where we stand.** Not measured. A fair analogue on PRMBench: supervised naive Bayes, label-free Dawid-Skene EM
  and L-SML on the same binned channels, all flagged by GAP.

## 5. ARES (EMNLP 2025): entailment with sampled premises
- **What it does.** For each step, asks a judge (an LLM or the PRM) whether the step follows from the question and
  earlier steps. Earlier steps are kept at random with probability equal to their own score; the step's score is
  the average over many sampled chains (about 265 for a 10-step chain). The "guarantee" is only that this sampled
  estimate is within ε of its own target, not that a step is correct.
- **Supervision and cost.** Threshold tuned on labels (5-fold cross-validation, Macro-F1); up to thousands of judge
  calls per answer.
- **Like us.** Training-free step scoring on PRMBench.
- **Unlike us.** Many LLM calls; no telemetry; labelled threshold; an unstated PRMBench subset and its own metric.
- **Where we stand.** In ARES's own table the plain PRM (.749) beats ARES on the PRM (.736). Under ARES's protocol
  on our data: Qwen-7B PRM 68.17 vs L-SML 68.10 (tie); L-SML ahead on the weak PRMs (RLHFlow 64.13 vs 55.79,
  Math-Shepherd 64.34 vs 53.33). Approximate, since their subset is not stated.

## 6. R-PRM (EMNLP 2025): a generative PRM that writes critiques
- **What it does.** Fine-tunes Qwen2.5-Math-7B-Instruct to write an analysis of each step and end with "correct:
  Yes/No". Training: distilled critiques from Llama-3.3-70B, filtered by PRM800K labels (SFT), then DPO on correct vs
  wrong critiques. Step reward = mean P(Yes) over 10 sampled critiques.
- **Like us.** Same benchmark and metric; the score is an average of token-probability readouts.
- **Unlike us.** A different, trained model; about 10 generations per step; a global threshold (the paper says
  0.5, its released scorer uses 0.61).
- **Where we stand.** Leaderboard context only: 66.8 PRMScore (DPO variant) vs 65.5 for Qwen-PRM-7B. On ProcessBench
  the plain Qwen-PRM-7B (73.5) beats it (70.4).

## 7. SLIDER / Step-RRI (ICLR 2026 workshop per arXiv): redundant steps only
- **What it does.** Generates up to 50 number-changed variants of each problem with GPT-4o-mini, embeds prefix, step
  and final answer, and uses partial information decomposition to ask whether a step only repeats what the prefix
  already says about the answer. Positive score = redundant. Threshold from a labelled linear classifier.
- **Like us.** Label-free score at step level on PRMBench.
- **Unlike us.** Targets one error type; needs the final answer and many generated variants; about 38 s per problem.
- **Where we stand.** On PRMBench's redundancy class, L-SML with GAP reaches 71.2 balanced accuracy vs 54.7 for the
  PRM under the same rule; SLIDER reports 61.9 "accuracy", whose definition is unclear (always "not redundant"
  gives 86.8 plain accuracy). Approximate.

## Side by side

| Paper | Changes the PRM? | Labels used | Extra model calls | Signal | Fusion | Decision | Level |
|---|---|---|---|---|---|---|---|
| **Ours (L-SML)** | no (frozen) | none | none (same pass) | PRM + 16 telemetry channels | label-free spectral weights | per-answer GAP | step |
| CROP | no | step labels (LR) + calibration | a second model (Llama-8B) | PRM + Llama features + surface cues | supervised LR | global λ with a guarantee | prefix |
| PRISM | retrains it | PRM800K pairs | none | the PRM | none | global threshold | step |
| Mind the Gap | no PRM | threshold only | none | one entropy-drop channel | none | labelled threshold | first error |
| BR2-PRM | no | reliability fitted on labels | one judge generation per answer | 9 judge scores | supervised naive Bayes | best of N | answer |
| ARES | no | threshold only | hundreds to thousands per answer | entailment scores | sampling average | CV threshold | step |
| R-PRM | new trained model | distillation + labels | about 10 per step | generated critiques | average of samples | global threshold | step |
| SLIDER | no PRM | threshold only | up to 50 variants per problem | information decomposition | none | labelled classifier | step (redundancy) |

## Corrections to what the 2026-10-22 email said
- Mind the Gap is training-free, but its thresholds are set on labels.
- BR2-PRM is supervised naive Bayes ("Dawid-Skene style" only loosely); still fitted on labels, as the email says.
- R-PRM's 66.8 is the DPO variant; its released scorer uses a 0.61 threshold, not 0.5.
- In ARES's own table the plain PRM beats ARES.
