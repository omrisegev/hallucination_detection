# One SSL experiment: method (2026-11-18)

This file tells what each part of the analysis calculates, how it calculates it, and on which data.
The design rules are in `ONE_EXPERIMENT.md` and in `mac_handoff/ANSWERS_TO_MAC_2026-11-18.md` (answers 1-56).
The code is on branch `claude/ssl-one-experiment-2026-11-18`, folder `results/ssl_one_experiment_v1/`.

## 1. Purpose

The advisors asked for two things:

- A self-supervised calibration of the step score. We use no gold labels to fit it.
- An answer-level gate. The gate must stop GAP from marking steps in clean answers.

One GPU pass gives the data for both. We add answers with a known error. These answers act as labeled positives that cost nothing.

## 2. Data

| Item | Value |
|---|---|
| Model | Qwen2.5-Math-7B-Instruct, standalone (no PRM) |
| Benchmarks | ProcessBench (PB, 3,400 answers), PRMBench (6,969 answers) |
| GPU | AIRCC cycle4, B200, one job per benchmark and condition |
| Telemetry | 11 token channels and 5 step channels (TELE16) |

We run the model on three conditions of each answer:

| Condition | What it is | What we know |
|---|---|---|
| C (control) | The real answer | Nothing (gold labels are used only to score) |
| V2 (one foreign step) | Step k of the answer is replaced by one step of a different answer | Step k is wrong |
| V1 (permuted) | Step 1 stays; steps 2..n are put in a random new order | The answer is wrong |

V2 rules: k is random over the non-empty steps 2..n. The donor step comes from the same split side, a different question group, a similar relative position (±.2) and a similar length (±20%). A donor step is used fewer than 3 times. If no donor fits, the rules relax in 3 levels. If still no donor fits, we skip the answer.

## 3. Split

We split the question groups (not the answers) 50/50 into dev and test. A group contains all answers to the same source question, in both benchmarks. The split is stratified by the group strata of `FOLDS_V2.json` (seed 20261116). V1 and V2 donors come only from the same side.

- dev: we make all choices here (N, M, λ, keep or drop of each stage).
- test: we run it once. We freeze all choices from dev. Label-free parts (L-SML weights, π̂) are fitted again on the test C pool, because they need no labels.

## 4. Step table

For each step, the GPU job saves:

- The 48 largest token values of each token channel, sorted from high to low.
- The exact top10 mean, mean, median and p90 of each token channel over all tokens of the step.
- The 5 step channels and the digit-active mask.

Stage S needs only the 48 largest values, because the readout uses at most 40 tokens.

## 5. The method of record (the reference)

The method of record is "TELE16 R0 + L-SML + GAP". It has four parts:

1. **Readout.** For each token channel, the step value is the mean of the 10 largest token values in the step (top10).
2. **R0 transform.** Inside each answer, each channel is z-scored over the steps of that answer. For the 3 digit channels, the z-score uses only the steps where the channel is active. Inactive steps get 0.
3. **L-SML fusion.** L-SML gives one weight to each of the 16 channels, without labels. It first puts the channels into groups of dependent channels. Inside each group, it finds the weights from the leading eigenvector of the covariance (a rank-one fit). Then it combines the group scores in the same way. The sign of the result is set so that the fused score correlates positively with the mean of the channels. The fused score s is linear in the 16 inputs: s = X·W.
4. **GAP marking.** Inside each answer, GAP z-scores s and flags the k highest steps. k = min(ceil(m/2), m−1) comes from the largest gap in the sorted scores. GAP always flags at least one step in each answer. This is the reason the advisors asked for a gate.

**P0 parity check.** We run the method of record on the full C pool. The result must be within 1 point of the stored numbers (PB SLA 38.47, PRMBench PRMScore 68.26). We also compare our readouts with the stored banks (Spearman per channel).

## 6. Realism check (C2ST)

Before we use V2 or V1 as "known wrong", we check that they look like real wrong steps or answers.

- **Step level:** V2 step k against real wrong steps of dev C. **Answer level:** V2 answers and V1 answers against real wrong answers of dev C. (Gold labels select the real wrong items. This is a check, not a fit.)
- A logistic regression tries to tell the two sets apart (5-fold, folds by question group). We balance the sets by step length (step level) or by step count (answer level).
- We also compare the 16×16 channel correlation patterns of the two sets.
- **Pass:** classifier AUROC ≤ .65 and correlation-pattern agreement ≥ .5.
- If V2 fails at step level, stage S does not run. If both fail at answer level, stages A and G do not run.

## 7. Stage S: soft top-N step readout

**What it changes.** The top10 readout uses a fixed N = 10 tokens. Stage S lets each token channel c use its own N_c (1 to 40 tokens). A soft weight makes N continuous: token i (sorted from high to low) gets weight sigmoid((N_c + .5 − i)/1).

**Objective (on dev, V2 half A).**

- Term 1: the sum of squared correlations between the R0 channels at the V2 step k (the known wrong steps). Only pairs with at least one token channel count. Low values mean the channels are more independent inside the wrong class. L-SML assumes this independence.
- Term 2: −λ × the sum over channels of (paired AUC − .5). The paired AUC compares the V2 foreign step k with the parent's own step k. This term stops the optimizer from removing the signal.
- Optimizer: Adam, learning rate .05, 500 steps, central finite difference per channel. Start points: N = 3, 10 and 30 for all channels; we keep the best end point.

**Choice of λ (on dev).** λ ∈ {.1, .3, 1, 3}. For each λ, we score the real dev C pool. We keep the λ with the lowest real conditional |Spearman| (see section 12). Guard: the main metric must stay ≥ top10 − 1. Ties within .005 go to the higher main metric.

**Revert rule.** If a channel has AUC < .52 and lower than its top10 AUC, it goes back to top10.

**Keep rule.** We keep stage S only if the drop of the dev conditional |corr| has a bootstrap CI with lower bound > 0. Otherwise we use top10.

We also report (but do not use): an ILP that picks the best hard readout per channel among 8 candidates (top1/2/5/10/20, mean, median, p90), and the hard readout top-round(N).

## 8. Stage S2: clean proxy

The bottom 30% of steps by the Simes step score s_g on dev C act as a "probably clean" set. We add the same correlation term for this set to the objective. We keep S2 only if it lowers the conditional |corr| more than S alone (CI lower bound > 0).

## 9. Stage A: answer score

For each answer we make 17 features:

- For each of the 16 channels: we orient the channel so that "high = risk" (sign of its correlation with the fused score s on pool C). We change each step value to its ECDF value on pool C. Then we take the soft top-M mean over the steps of the answer (M_c from .5 to 15 steps).
- The Simes feature: −log10 of the Simes p-value of the answer's step scores s_g.

We standardize the 17 features on pool C and fuse them with an answer-level L-SML. The result is the answer score (high = more likely wrong).

M is fitted on the gate positive (V2 answers if V2 passes C2ST at answer level, else V1), with the same objective form as stage S. λ choice: the lowest real answer conditional |corr| on dev, with an AUROC guard of −.01 against M = 1. The keep rule is the same as in stage S.

## 10. Stage G: π̂ and the gate

**π̂ (share of wrong answers).** BBE (Garg et al. 2021) uses the answer scores of two sets: the known positives (synthetic wrong answers) and the unlabeled pool (real C answers). For each cut c, it divides the share of unlabeled answers above c by the share of positives above c. Above a high cut, almost all unlabeled answers are wrong, so this ratio estimates π. BBE takes the cut with the smallest ratio plus a confidence penalty (δ = .1, γ = .01). We cross-fit in 2 folds by question group, so no answer scores itself.

**P(wrong) per answer.** We fit an isotonic regression of "positive vs unlabeled" on the answer score (balanced weights). With this g, P = min(1, π̂·g/(1 − g)).

**Gate.** We open the π̂ share of answers with the highest answer score. GAP runs only inside open answers. Closed answers get no flag.

**PB change point.** For PB we also run a change-point rule. It looks for the first wrong step τ in each answer (or τ = none). The per-answer prior of "τ = none" is 1 − P. The step-score distributions before and after τ are fitted by free EM (20 bins, 30 iterations). The prediction is the most likely τ.

We also report π̂ with the other positive set (V1 if V2 is used, and the inverse) as a check.

## 11. Marking arms (all scored on the same pool)

| Arm | What it is |
|---|---|
| method of record (top10, no gate) | Section 5. The reference for all deltas. |
| final readout, no gate | Readout from stages S/S2 (or top10 if not kept), L-SML, GAP |
| final + gate (π̂) | The above, GAP only in the π̂ share of answers with the highest answer score |
| final + change point (prior 1 − P) | PB only, section 10 |
| final + label-perfect gate (oracle) | Opens exactly the truly wrong answers. Upper bound for any gate. Uses labels. |
| final + Simes gate α .05 | Opens answers with Simes p ≤ .05 (earlier gate) |
| final + Simes gate oracle α | Best α on the pool, chosen with labels. Upper bound for the Simes gate. |
| baselines (test only) | Equal mean of the 16 R0 channels, position only, length only, mean entropy, mean surprisal; each + GAP |

## 12. Metrics

- **PRMBench PRMScore** = 50 × (F1 of error steps + F1 of valid steps), from pooled counts. Rows of class "correct" are out (official rule).
- **PB F1** = harmonic mean of the accuracy on wrong answers (exact first error) and the accuracy on clean answers (no flag). We report the mean over the 4 subsets. The prediction is the first flagged step.
- **PB SLA** = the MtG protocol step-level accuracy, mean over the 4 subsets.
- **Conditional |corr|** = the mean absolute Spearman correlation over all pairs of the 16 R0 channels, calculated separately inside the real wrong steps and inside the real clean steps, then averaged. Low = more independent channels. This is the quantity that L-SML needs to be low.
- **CIs** = 95% clustered bootstrap over question groups (2,000 draws). Every arm has a delta against the method of record with its CI.

## 13. Where gold labels are used

| Use | Labels |
|---|---|
| Any fit (readout, L-SML, M, π̂, P, gate, change point) | Never |
| Choice of λ and keep / drop (dev only) | Yes, as named in sections 7-9 |
| C2ST real wrong sets (dev only) | Yes (a check, not a fit) |
| Oracle arms | Yes (marked "oracle") |
| Scoring and reports | Yes |
